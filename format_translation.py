"""Request/response format translation between Anthropic Messages API, OpenAI Chat Completions API, and OpenAI Responses API formats."""

import base64
import codecs
import json

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import effort_mapping
import util

from constants import (
    CLAUDE_DEFAULT_REASONING_EFFORT,
    FAKE_COMPACTION_PREFIX, FAKE_COMPACTION_SUMMARY_LABEL,
    COMPACTION_SUMMARY_PROMPT,
)

# Re-export header builders from their new home so existing callers
# that reference format_translation.build_*_headers_for_request continue to work.
from request_headers import (  # noqa: F401
    has_vision_input,
    build_copilot_headers,
    build_responses_headers_for_request,
    build_chat_headers_for_request,
    build_anthropic_headers_for_request,
)


# ─── Model resolution ────────────────────────────────────────────────────────

def normalize_upstream_model_name(model_name: str | None) -> str | None:
    if not isinstance(model_name, str):
        return model_name

    normalized = model_name.strip().lower()
    if normalized.startswith("anthropic/"):
        normalized = normalized.split("/", 1)[1]
    return normalized


def resolve_copilot_model_name(model_name: str | None) -> str | None:
    normalized = normalize_upstream_model_name(model_name)
    if not isinstance(normalized, str):
        return model_name

    if normalized in ("claude-opus-4.6", "claude-opus-4.7", "claude-sonnet-4.6", "claude-haiku-4.5"):
        return normalized

    if "opus" in normalized:
        return "claude-opus-4.7"
    if "sonnet" in normalized:
        return "claude-sonnet-4.6"
    if "haiku" in normalized:
        return "claude-haiku-4.5"
    return normalized


def _responses_model_disallows_temperature(model_name: str | None) -> bool:
    normalized = normalize_upstream_model_name(resolve_copilot_model_name(model_name))
    return normalized == "gpt-5.4-mini"


_COPILOT_UNSUPPORTED_RESPONSES_TOOL_TYPES = {"image_generation"}


# ─── Anthropic content helpers ────────────────────────────────────────────────

def _anthropic_effort_to_reasoning_effort(output_config) -> str | None:
    """Extract the selected effort level from Claude Code's output config."""
    if not isinstance(output_config, dict):
        return None
    effort = output_config.get("effort")
    if not isinstance(effort, str):
        return None
    normalized = effort.strip().lower()
    if normalized in ("low", "medium", "high", "max"):
        return normalized
    return None


def _anthropic_thinking_to_reasoning_effort(thinking) -> str | None:
    """Translate Anthropic ``thinking`` block into GHCP ``reasoning_effort``.

    Copilot's chat/completions endpoint expects a top-level ``reasoning_effort``
    string (``low``/``medium``/``high``/``max``).

    Claude Code's default Anthropic body often only carries
    ``thinking: {"type":"adaptive"}``, so we use the configured default
    effort unless the client explicitly enabled extended thinking with a
    positive ``budget_tokens``:

    - ``think``     → 4000  tokens → ``medium``
    - ``megathink`` → 10000 tokens → ``high``
    - ``ultrathink``→ 31999 tokens → ``max``
    """
    from constants import CLAUDE_DEFAULT_REASONING_EFFORT

    default_effort = CLAUDE_DEFAULT_REASONING_EFFORT or "medium"

    # No thinking object at all → configured default.
    if not isinstance(thinking, dict):
        return default_effort

    thinking_type = thinking.get("type")
    normalized_type = thinking_type.lower() if isinstance(thinking_type, str) else None

    # Explicit "enabled" with a positive budget → honor the budget threshold.
    if normalized_type == "enabled":
        budget_tokens = thinking.get("budget_tokens")
        if isinstance(budget_tokens, int) and budget_tokens > 0:
            if budget_tokens >= 24576:
                return "max"
            if budget_tokens >= 8192:
                return "high"
            return "medium"

    return default_effort


def _normalize_anthropic_cache_control(value):
    if not isinstance(value, dict):
        return None

    cache_type = value.get("type")
    if isinstance(cache_type, str) and cache_type:
        return {"type": cache_type}

    if "ephemeral" in value:
        return {"type": "ephemeral"}

    return dict(value)


def _attach_copilot_cache_control(target: dict, source: dict) -> dict:
    cache_control = _normalize_anthropic_cache_control(source.get("cache_control"))
    if cache_control is not None:
        target["copilot_cache_control"] = cache_control
    return target


def _anthropic_image_block_to_chat(item: dict) -> dict:
    source = item.get("source")
    if not isinstance(source, dict):
        raise ValueError("Anthropic image block is missing a valid source object")

    source_type = str(source.get("type", "")).lower()
    if source_type == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if not isinstance(media_type, str) or not isinstance(data, str):
            raise ValueError("Anthropic base64 image source must include media_type and data strings")
        return _attach_copilot_cache_control(
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}},
            item,
        )
    if source_type == "url":
        image_url = source.get("url")
        if not isinstance(image_url, str):
            raise ValueError("Anthropic URL image source must include a url string")
        return _attach_copilot_cache_control({"type": "image_url", "image_url": {"url": image_url}}, item)

    raise ValueError(f"Unsupported Anthropic image source type: {source_type}")


def _normalize_anthropic_content_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    raise ValueError("Anthropic message content must be a string or content block list")


def _anthropic_text_or_image_block_to_chat(item: dict) -> dict | None:
    item_type = str(item.get("type", "")).lower()
    if item_type == "text":
        text = item.get("text")
        if isinstance(text, str):
            return _attach_copilot_cache_control({"type": "text", "text": text}, item)
        return None
    if item_type == "image":
        return _anthropic_image_block_to_chat(item)
    return None


def _anthropic_block_type_is_ignorable(item_type: str) -> bool:
    """Return True for Anthropic block types that should not be replayed upstream.

    Claude Code may resend assistant ``thinking`` / ``redacted_thinking`` blocks
    from prior turns. GHCP's chat/responses schemas have no equivalent inbound
    field, and those blocks are not needed to preserve the actual visible/tool
    conversation state, so we drop them instead of rejecting the whole request.
    """
    return item_type in {"thinking", "redacted_thinking"}


def _anthropic_system_to_chat_content(system):
    if isinstance(system, str):
        return system
    if not isinstance(system, list):
        return ""

    converted = []
    has_copilot_cache_control = False
    for item in system:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).lower() != "text":
            raise ValueError("Anthropic system content currently supports text blocks only")
        text = item.get("text")
        if not isinstance(text, str):
            continue
        part = _attach_copilot_cache_control({"type": "text", "text": text}, item)
        if "copilot_cache_control" in part:
            has_copilot_cache_control = True
        converted.append(part)

    if not converted:
        return ""
    if not has_copilot_cache_control:
        return "".join(part.get("text", "") for part in converted)
    return converted


def _anthropic_blocks_to_chat_content(blocks: list[dict]):
    converted = []
    for item in blocks:
        content_item = _anthropic_text_or_image_block_to_chat(item)
        if content_item is not None:
            converted.append(content_item)
            continue
        item_type = str(item.get("type", "")).lower()
        if item_type in {"tool_use", "tool_result"}:
            raise ValueError(f"Anthropic block type {item_type} cannot be converted into chat message content directly")
        raise ValueError(f"Unsupported Anthropic content block type: {item_type}")

    if not converted:
        return ""
    if (
        len(converted) == 1
        and converted[0].get("type") == "text"
        and set(converted[0].keys()).issubset({"type", "text"})
    ):
        return converted[0].get("text", "")
    return converted


def _anthropic_tool_result_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            raise ValueError("Anthropic tool_result content currently supports text blocks only")
        return "".join(parts)
    raise ValueError("Anthropic tool_result content must be a string or list of text blocks")


def _anthropic_tool_use_to_chat_tool_call(item: dict) -> dict:
    tool_name = item.get("name")
    tool_id = item.get("id")
    tool_input = item.get("input")
    if not isinstance(tool_name, str) or not isinstance(tool_id, str):
        raise ValueError("Anthropic tool_use blocks must include string id and name")
    if tool_input is None:
        tool_input = {}
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(tool_input, separators=(",", ":")),
        },
    }


# ─── Anthropic → Chat translation ────────────────────────────────────────────

def anthropic_message_to_chat_messages(message: dict) -> list[dict]:
    role = str(message.get("role", "")).lower()
    if role not in {"user", "assistant"}:
        raise ValueError(f"Unsupported Anthropic role: {role}")

    blocks = _normalize_anthropic_content_blocks(message.get("content"))

    if role == "assistant":
        content_blocks = []
        tool_calls = []
        for item in blocks:
            item_type = str(item.get("type", "")).lower()
            if item_type == "tool_use":
                tool_calls.append(_anthropic_tool_use_to_chat_tool_call(item))
                continue
            if _anthropic_block_type_is_ignorable(item_type):
                continue
            content_item = _anthropic_text_or_image_block_to_chat(item)
            if content_item is None:
                raise ValueError(f"Unsupported Anthropic content block type: {item_type}")
            content_blocks.append(item)

        assistant_message = {"role": "assistant"}
        assistant_message["content"] = _anthropic_blocks_to_chat_content(content_blocks)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return [assistant_message]

    chat_messages = []
    buffered_user_blocks = []
    buffered_tool_messages = []
    for item in blocks:
        item_type = str(item.get("type", "")).lower()
        if item_type == "tool_result":
            tool_use_id = item.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                raise ValueError("Anthropic tool_result blocks must include tool_use_id")
            tool_text = _anthropic_tool_result_content_to_text(item.get("content", ""))
            if item.get("is_error") is True:
                tool_text = f"[tool_error]\n{tool_text}"
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": tool_text,
            }
            cache_control = _normalize_anthropic_cache_control(item.get("cache_control"))
            if cache_control is not None:
                tool_message["copilot_cache_control"] = cache_control
            buffered_tool_messages.append(tool_message)
            continue
        if _anthropic_block_type_is_ignorable(item_type):
            continue
        content_item = _anthropic_text_or_image_block_to_chat(item)
        if content_item is None:
            raise ValueError(f"Unsupported Anthropic content block type: {item_type}")
        buffered_user_blocks.append(item)

    chat_messages.extend(buffered_tool_messages)
    if buffered_user_blocks:
        chat_messages.append(
            {
                "role": "user",
                "content": _anthropic_blocks_to_chat_content(buffered_user_blocks),
            }
        )

    return chat_messages


def anthropic_tools_to_chat(tools) -> list[dict]:
    if not isinstance(tools, list):
        raise ValueError("Anthropic tools must be a list")

    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str):
            raise ValueError("Anthropic tools must include a string name")
        raw_desc = tool.get("description", "")
        desc = raw_desc if isinstance(raw_desc, str) and raw_desc else " "
        chat_tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object", "properties": {}},
            },
        }
        cache_control = _normalize_anthropic_cache_control(tool.get("cache_control"))
        if cache_control is not None:
            chat_tool["copilot_cache_control"] = cache_control
        converted.append(chat_tool)
    return converted


def anthropic_tool_choice_to_chat(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type", "")).lower()
        if choice_type == "auto":
            return "auto"
        if choice_type == "any":
            return "required"
        if choice_type == "tool":
            name = tool_choice.get("name")
            if not isinstance(name, str):
                raise ValueError("Anthropic tool_choice type=tool must include name")
            return {"type": "function", "function": {"name": name}}
        if choice_type == "none":
            return "none"
    if isinstance(tool_choice, str):
        normalized = tool_choice.lower()
        if normalized in {"auto", "none"}:
            return normalized
        if normalized == "any":
            return "required"
    raise ValueError("Unsupported Anthropic tool_choice value")


async def anthropic_request_to_chat(body: dict, api_base: str, api_key: str) -> dict:
    source_messages = body.get("messages")
    if not isinstance(source_messages, list):
        raise ValueError("Anthropic request must include a messages array")

    chat_messages = []
    system_content = _anthropic_system_to_chat_content(body.get("system"))
    if system_content:
        chat_messages.append({"role": "system", "content": system_content})

    for message in source_messages:
        if not isinstance(message, dict):
            continue
        chat_messages.extend(anthropic_message_to_chat_messages(message))

    payload = {
        "model": resolve_copilot_model_name(body.get("model")),
        "messages": chat_messages,
        "stream": bool(body.get("stream", False)),
    }

    if payload["stream"]:
        payload["stream_options"] = {"include_usage": True}

    for source_key, target_key in (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop_sequences", "stop"),
    ):
        value = body.get(source_key)
        if value is not None:
            payload[target_key] = value

    reasoning_effort = (
        _anthropic_effort_to_reasoning_effort(body.get("output_config"))
        or _anthropic_thinking_to_reasoning_effort(body.get("thinking"))
    )
    reasoning_effort = effort_mapping.map_effort_for_model(
        payload.get("model"), reasoning_effort
    )
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    if body.get("tools") is not None:
        payload["tools"] = anthropic_tools_to_chat(body.get("tools"))

    mapped_tool_choice = anthropic_tool_choice_to_chat(body.get("tool_choice"))
    if mapped_tool_choice is not None:
        payload["tool_choice"] = mapped_tool_choice

    return payload


# ─── Anthropic ↔ Responses translation ───────────────────────────────────────

def _response_content_text_type(item_type: str) -> bool:
    return item_type in {"text", "input_text", "output_text"}


def _response_content_item_to_chat(item: dict) -> dict | None:
    item_type = str(item.get("type", "")).lower()
    if _response_content_text_type(item_type):
        text = item.get("text") or item.get("input_text") or item.get("output_text")
        if isinstance(text, str):
            content_item = {"type": "text", "text": text}
            if isinstance(item.get("copilot_cache_control"), dict):
                content_item["copilot_cache_control"] = dict(item["copilot_cache_control"])
            return content_item
        return None

    if item_type == "input_image":
        image_url = item.get("image_url")
        if isinstance(image_url, str) and image_url:
            return {"type": "image_url", "image_url": {"url": image_url}}
        if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            return {"type": "image_url", "image_url": {"url": image_url["url"]}}
        image_base64 = item.get("image_base64")
        media_type = item.get("media_type")
        if isinstance(image_base64, str) and isinstance(media_type, str):
            return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}}
        raise ValueError("Responses input_image blocks must include image_url or image_base64/media_type")

    return None


def _response_message_content_to_chat(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    converted = []
    for item in content:
        if not isinstance(item, dict):
            continue
        content_item = _response_content_item_to_chat(item)
        if content_item is not None:
            converted.append(content_item)

    if not converted:
        return ""
    if (
        len(converted) == 1
        and converted[0].get("type") == "text"
        and set(converted[0].keys()).issubset({"type", "text"})
    ):
        return converted[0]["text"]
    return converted


def _merge_chat_system_prompt_parts(parts) -> str | list[dict]:
    merged = []
    for part in parts:
        if isinstance(part, str):
            if not part:
                continue
            part_items = [{"type": "text", "text": part}]
        elif isinstance(part, list):
            part_items = [dict(item) for item in part if isinstance(item, dict)]
            if not part_items:
                continue
        else:
            continue

        if merged:
            merged.append({"type": "text", "text": "\n\n"})
        merged.extend(part_items)

    if not merged:
        return ""
    # If every item is a plain {"type": "text", "text": "..."} with no extra
    # keys (e.g. copilot_cache_control), join into a single string.  The Chat
    # Completions API does not accept list-type content for system messages,
    # so we must flatten when possible.
    if all(
        item.get("type") == "text"
        and set(item.keys()).issubset({"type", "text"})
        for item in merged
    ):
        return "".join(item.get("text", "") for item in merged)
    return merged


def _responses_tool_to_chat(tool: dict) -> dict | None:
    if not isinstance(tool, dict):
        return None
    tool_type = str(tool.get("type", "")).lower()
    if tool_type != "function":
        return None

    if isinstance(tool.get("function"), dict):
        function = tool["function"]
        name = function.get("name")
        description = function.get("description", "")
        parameters = function.get("parameters")
    else:
        name = tool.get("name")
        description = tool.get("description", "")
        parameters = tool.get("parameters")

    if not isinstance(name, str):
        raise ValueError("Responses function tools must include a name")

    desc = description if isinstance(description, str) and description else " "
    chat_tool = {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": parameters if isinstance(parameters, dict) else {"type": "object", "properties": {}},
        },
    }
    # Forward cache_control from the source tool definition (Responses API
    # or flat format) so GHCP can set prompt-cache breakpoints on tools.
    for source in (tool, tool.get("function") or {}):
        cache_control = _normalize_anthropic_cache_control(source.get("cache_control"))
        if cache_control is not None:
            chat_tool["copilot_cache_control"] = cache_control
            break
    return chat_tool


def _responses_tool_choice_targets_removed_tool(tool_choice, removed_types: set[str], removed_names: set[str]) -> bool:
    if not isinstance(tool_choice, dict):
        return False

    choice_type = str(tool_choice.get("type", "")).strip().lower()
    if choice_type and choice_type in removed_types:
        return True

    function = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else tool_choice
    name = function.get("name")
    if isinstance(name, str) and name.strip().lower() in (removed_names | removed_types):
        return True

    return False


def sanitize_responses_tools_for_copilot(body: dict) -> dict:
    if not isinstance(body, dict):
        return body

    tools = body.get("tools")
    if not isinstance(tools, list):
        return body

    filtered_tools = []
    removed_types: set[str] = set()
    removed_names: set[str] = set()

    for tool in tools:
        if not isinstance(tool, dict):
            filtered_tools.append(tool)
            continue

        tool_type = str(tool.get("type", "")).strip().lower()
        if tool_type in _COPILOT_UNSUPPORTED_RESPONSES_TOOL_TYPES:
            removed_types.add(tool_type)
            name = tool.get("name")
            if isinstance(name, str) and name.strip():
                removed_names.add(name.strip().lower())
            continue

        filtered_tools.append(tool)

    if not removed_types:
        return body

    sanitized = dict(body)
    if filtered_tools:
        sanitized["tools"] = filtered_tools
    else:
        sanitized.pop("tools", None)
        sanitized.pop("parallel_tool_calls", None)

    if (
        not filtered_tools
        or _responses_tool_choice_targets_removed_tool(
            sanitized.get("tool_choice"), removed_types, removed_names
        )
    ):
        sanitized.pop("tool_choice", None)

    return sanitized


def responses_tool_choice_to_chat(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        normalized = tool_choice.lower()
        if normalized in {"auto", "none", "required"}:
            return normalized
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type", "")).lower()
        if choice_type in {"auto", "none", "required"}:
            return choice_type
        function = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else tool_choice
        name = function.get("name")
        if isinstance(name, str):
            return {"type": "function", "function": {"name": name}}
    raise ValueError("Unsupported Responses tool_choice value")


def _format_custom_tool_call_for_chat(item: dict) -> str | None:
    name = item.get("name")
    if not isinstance(name, str) or not name:
        return None

    tool_input = item.get("input")
    if isinstance(tool_input, str):
        input_text = tool_input.strip() or "[no input]"
    elif tool_input is None:
        input_text = "[no input]"
    else:
        input_text = json.dumps(tool_input, separators=(",", ":"))

    call_id = item.get("call_id") or item.get("id")
    call_suffix = f" ({call_id})" if isinstance(call_id, str) and call_id else ""
    return f"[Custom tool call{call_suffix}] {name}\n{input_text}"


def _format_custom_tool_output_for_chat(item: dict) -> str:
    call_id = item.get("call_id")
    label = f"[Custom tool result ({call_id})]" if isinstance(call_id, str) and call_id else "[Custom tool result]"

    output = item.get("output")
    if isinstance(output, list):
        output_text = "".join(util.extract_item_text(part) for part in output if isinstance(part, dict))
    elif isinstance(output, str):
        output_text = output
    elif output is None:
        output_text = ""
    else:
        output_text = json.dumps(output, separators=(",", ":"))

    output_text = output_text.strip()
    return f"{label}\n{output_text}" if output_text else label


def responses_request_to_chat(body: dict) -> dict:
    raw_input = body.get("input")
    chat_messages = []
    system_prompt_parts = []
    pending_tool_call_ids = set()
    deferred_messages = []

    def append_or_defer_message(message: dict):
        if pending_tool_call_ids and message.get("role") != "assistant":
            deferred_messages.append(message)
            return
        chat_messages.append(message)

    def flush_deferred_messages():
        if pending_tool_call_ids or not deferred_messages:
            return
        chat_messages.extend(deferred_messages)
        deferred_messages.clear()

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        system_prompt_parts.append(instructions)

    if isinstance(raw_input, str):
        chat_messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).lower()
            if item_type == "message":
                role = str(item.get("role", "user")).lower()
                if role not in {"system", "developer", "user", "assistant"}:
                    raise ValueError(f"Unsupported Responses message role: {role}")
                normalized_role = "system" if role == "developer" else role
                message_content = _response_message_content_to_chat(item.get("content"))
                if normalized_role == "system":
                    if message_content:
                        system_prompt_parts.append(message_content)
                    continue
                append_or_defer_message(
                    {
                        "role": normalized_role,
                        "content": message_content,
                    }
                )
                continue
            if item_type == "function_call":
                function_name = item.get("name")
                if not isinstance(function_name, str):
                    raise ValueError("Responses function_call items must include a name")
                call_id = item.get("call_id") or item.get("id") or f"call_{uuid4().hex}"
                chat_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": item.get("arguments")
                                    if isinstance(item.get("arguments"), str)
                                    else json.dumps(item.get("arguments", {}), separators=(",", ":")),
                                },
                            }
                        ],
                    }
                )
                pending_tool_call_ids.add(call_id)
                continue
            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str):
                    raise ValueError("Responses function_call_output items must include call_id")
                output_text = item.get("output")
                if isinstance(output_text, list):
                    output_text = "".join(util.extract_item_text(part) for part in output_text if isinstance(part, dict))
                elif not isinstance(output_text, str):
                    output_text = json.dumps(output_text, separators=(",", ":")) if output_text is not None else ""
                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output_text,
                    }
                )
                pending_tool_call_ids.discard(call_id)
                flush_deferred_messages()
                continue
            if item_type == "custom_tool_call":
                custom_tool_text = _format_custom_tool_call_for_chat(item)
                if custom_tool_text is None:
                    raise ValueError("Responses custom_tool_call items must include a name")
                chat_messages.append(
                    {
                        "role": "assistant",
                        "content": custom_tool_text,
                    }
                )
                continue
            if item_type == "custom_tool_call_output":
                chat_messages.append(
                    {
                        "role": "user",
                        "content": _format_custom_tool_output_for_chat(item),
                    }
                )
                continue
            if item_type in {"reasoning", "item_reference", "compaction", "web_search_call"}:
                # Native Responses web search history is represented by a paired
                # assistant message item containing the user-visible answer and
                # citations. Chat-targeted bridges cannot replay the built-in
                # tool call itself, so keep the answer message and skip the
                # internal tool-call record.
                continue
            raise ValueError(f"Unsupported Responses input item type: {item_type}")

    if deferred_messages:
        chat_messages.extend(deferred_messages)

    system_prompt = _merge_chat_system_prompt_parts(system_prompt_parts)
    if system_prompt:
        chat_messages.insert(0, {"role": "system", "content": system_prompt})

    payload = {
        "model": resolve_copilot_model_name(body.get("model")),
        "messages": chat_messages,
        "stream": bool(body.get("stream", False)),
    }
    if payload["stream"]:
        payload["stream_options"] = {"include_usage": True}

    for source_key, target_key in (
        ("max_output_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
    ):
        value = body.get(source_key)
        if value is not None:
            payload[target_key] = value

    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise ValueError("Responses tools must be a list")
        translated_tools = [tool for tool in (_responses_tool_to_chat(tool) for tool in tools) if tool is not None]
        if translated_tools:
            payload["tools"] = translated_tools

    mapped_tool_choice = responses_tool_choice_to_chat(body.get("tool_choice"))
    if mapped_tool_choice is not None and payload.get("tools"):
        payload["tool_choice"] = mapped_tool_choice

    incoming_reasoning = body.get("reasoning")
    incoming_effort = (
        incoming_reasoning.get("effort")
        if isinstance(incoming_reasoning, dict)
        else None
    )
    resolved_model = payload.get("model")
    # Codex omits `reasoning` (or sends `effort=null`) when its local catalog
    # doesn't advertise reasoning summary support; without an explicit effort
    # Copilot's Anthropic-fronted upstream silently disables extended thinking.
    if (
        incoming_effort is None
        and isinstance(resolved_model, str)
        and resolved_model.lower().startswith("claude-")
    ):
        incoming_effort = CLAUDE_DEFAULT_REASONING_EFFORT or "medium"
    mapped_effort = effort_mapping.map_effort_for_model(resolved_model, incoming_effort)
    if mapped_effort is not None:
        payload["reasoning_effort"] = mapped_effort

    return payload


def _chat_tool_to_responses(tool: dict) -> dict | None:
    if not isinstance(tool, dict):
        return None
    tool_type = str(tool.get("type", "")).lower()
    if tool_type != "function":
        return None
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    name = function.get("name")
    if not isinstance(name, str):
        raise ValueError("Chat function tools must include a name")
    return {
        "type": "function",
        "name": name,
        "description": function.get("description", "") if isinstance(function.get("description"), str) else "",
        "parameters": function.get("parameters") if isinstance(function.get("parameters"), dict) else {"type": "object", "properties": {}},
    }


def chat_tool_choice_to_responses(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        normalized = tool_choice.lower()
        if normalized in {"auto", "none", "required"}:
            return normalized
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type", "")).lower()
        if choice_type in {"auto", "none", "required"}:
            return choice_type
        function = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else tool_choice
        name = function.get("name")
        if isinstance(name, str):
            return {"type": "function", "name": name}
    raise ValueError("Unsupported chat tool_choice value")


def _chat_content_item_to_response_content(item: dict, *, role: str = "user") -> dict | None:
    item_type = str(item.get("type", "")).lower()
    if item_type == "text" and isinstance(item.get("text"), str):
        text_type = "output_text" if role == "assistant" else "input_text"
        content_item = {"type": text_type, "text": item["text"]}
        if isinstance(item.get("copilot_cache_control"), dict):
            content_item["copilot_cache_control"] = dict(item["copilot_cache_control"])
        return content_item
    if item_type == "image_url":
        image_url = item.get("image_url")
        if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            return {"type": "input_image", "image_url": image_url["url"]}
    return None


def anthropic_request_to_responses(body: dict) -> dict:
    source_messages = body.get("messages")
    if not isinstance(source_messages, list):
        raise ValueError("Anthropic request must include a messages array")

    response_input = []
    system_content = _anthropic_system_to_chat_content(body.get("system"))
    if isinstance(system_content, str) and system_content:
        response_input.append(
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": system_content}],
            }
        )
    elif isinstance(system_content, list) and system_content:
        response_input.append(
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": item.get("text", ""),
                        **({"copilot_cache_control": item["copilot_cache_control"]} if isinstance(item.get("copilot_cache_control"), dict) else {}),
                    }
                    for item in system_content
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                ],
            }
        )

    for message in source_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        blocks = _normalize_anthropic_content_blocks(message.get("content"))

        if role == "assistant":
            content = []
            for item in blocks:
                item_type = str(item.get("type", "")).lower()
                if item_type == "tool_use":
                    tool_name = item.get("name")
                    tool_id = item.get("id")
                    if not isinstance(tool_name, str) or not isinstance(tool_id, str):
                        raise ValueError("Anthropic tool_use blocks must include string id and name")
                    response_input.append(
                        {
                            "type": "function_call",
                            "call_id": tool_id,
                            "name": tool_name,
                            "arguments": json.dumps(item.get("input") or {}, separators=(",", ":")),
                        }
                    )
                    continue
                if _anthropic_block_type_is_ignorable(item_type):
                    continue
                content_item = _anthropic_text_or_image_block_to_chat(item)
                if content_item is None:
                    raise ValueError(f"Unsupported Anthropic content block type: {item_type}")
                response_content_item = _chat_content_item_to_response_content(content_item, role="assistant")
                if response_content_item is not None:
                    content.append(response_content_item)
            if content:
                response_input.append({"type": "message", "role": "assistant", "content": content})
            continue

        if role != "user":
            raise ValueError(f"Unsupported Anthropic role: {role}")

        buffered_user_content = []
        for item in blocks:
            item_type = str(item.get("type", "")).lower()
            if item_type == "tool_result":
                if buffered_user_content:
                    response_input.append({"type": "message", "role": "user", "content": buffered_user_content})
                    buffered_user_content = []
                tool_use_id = item.get("tool_use_id")
                if not isinstance(tool_use_id, str):
                    raise ValueError("Anthropic tool_result blocks must include tool_use_id")
                response_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_use_id,
                        "output": _anthropic_tool_result_content_to_text(item.get("content", "")),
                    }
                )
                continue
            if _anthropic_block_type_is_ignorable(item_type):
                continue
            content_item = _anthropic_text_or_image_block_to_chat(item)
            if content_item is None:
                raise ValueError(f"Unsupported Anthropic content block type: {item_type}")
            response_content_item = _chat_content_item_to_response_content(content_item)
            if response_content_item is not None:
                buffered_user_content.append(response_content_item)

        if buffered_user_content:
            response_input.append({"type": "message", "role": "user", "content": buffered_user_content})

    payload = {
        "model": resolve_copilot_model_name(body.get("model")),
        "input": response_input,
        "stream": bool(body.get("stream", False)),
    }

    for source_key, target_key in (
        ("max_tokens", "max_output_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("metadata", "metadata"),
    ):
        value = body.get(source_key)
        if value is not None:
            if source_key == "temperature" and _responses_model_disallows_temperature(payload.get("model")):
                continue
            payload[target_key] = value

    tools = body.get("tools")
    if tools is not None:
        payload["tools"] = [
            _chat_tool_to_responses(tool)
            for tool in anthropic_tools_to_chat(tools)
        ]

    mapped_tool_choice = anthropic_tool_choice_to_chat(body.get("tool_choice"))
    if mapped_tool_choice is not None:
        payload["tool_choice"] = chat_tool_choice_to_responses(mapped_tool_choice)

    reasoning_effort = (
        _anthropic_effort_to_reasoning_effort(body.get("output_config"))
        or _anthropic_thinking_to_reasoning_effort(body.get("thinking"))
    )
    reasoning_effort = effort_mapping.map_effort_for_model(
        payload.get("model"), reasoning_effort
    )
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}

    # copilot_cache_control is only valid for Chat Completions; strip for Responses API.
    payload["input"] = _strip_copilot_cache_control(payload["input"])

    return payload

# ─── Chat → Anthropic translation ────────────────────────────────────────────

def chat_usage_to_anthropic(usage) -> dict:
    if not isinstance(usage, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    cache_read_input_tokens = prompt_details.get("cached_tokens", 0) or 0

    return {
        "input_tokens": max(0, prompt_tokens - cache_read_input_tokens),
        "output_tokens": completion_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cache_read_input_tokens,
    }


def chat_stop_reason_to_anthropic(value) -> str | None:
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "stop_sequence",
        "tool_calls": "tool_use",
    }
    if not isinstance(value, str):
        return None
    return mapping.get(value, value)


def _extract_chat_message_text(message: dict) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _parse_tool_call_arguments(arguments) -> dict:
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"_raw": arguments}


def _chat_message_to_anthropic_content(message: dict) -> list[dict]:
    content = []
    text = _extract_chat_message_text(message)
    if text:
        content.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or f"toolu_{uuid4().hex}",
                    "name": function.get("name", ""),
                    "input": _parse_tool_call_arguments(function.get("arguments")),
                }
            )

    if not content:
        content.append({"type": "text", "text": ""})
    return content


def chat_completion_to_anthropic(payload: dict, fallback_model=None) -> dict:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    usage = chat_usage_to_anthropic(payload.get("usage") if isinstance(payload, dict) else {})

    return {
        "id": payload.get("id") if isinstance(payload, dict) else f"msg_{uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": fallback_model or (payload.get("model") if isinstance(payload, dict) else None),
        "content": _chat_message_to_anthropic_content(message if isinstance(message, dict) else {}),
        "stop_reason": chat_stop_reason_to_anthropic(first_choice.get("finish_reason") if isinstance(first_choice, dict) else None),
        "stop_sequence": None,
        "usage": usage,
    }


def response_usage_to_anthropic(usage) -> dict:
    normalized = util.normalize_usage_payload(usage if isinstance(usage, dict) else {})
    if not isinstance(normalized, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    return {
        "input_tokens": normalized.get("input_tokens", 0),
        "output_tokens": normalized.get("output_tokens", 0),
        "cache_creation_input_tokens": normalized.get("cache_creation_input_tokens", 0),
        "cache_read_input_tokens": normalized.get("cached_input_tokens", 0),
    }


def _response_output_items_to_anthropic_content(output) -> list[dict]:
    if not isinstance(output, list):
        return [{"type": "text", "text": ""}]

    content = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).lower()
        if item_type == "message" and str(item.get("role", "")).lower() == "assistant":
            for part in item.get("content", []):
                if not isinstance(part, dict):
                    continue
                text = part.get("text") or part.get("input_text") or part.get("output_text")
                if isinstance(text, str):
                    content.append({"type": "text", "text": text})
        if item_type == "function_call":
            content.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id") or item.get("id") or f"toolu_{uuid4().hex}",
                    "name": item.get("name") if isinstance(item.get("name"), str) else "",
                    "input": _parse_tool_call_arguments(item.get("arguments")),
                }
            )
    return content or [{"type": "text", "text": ""}]


def response_payload_to_anthropic(payload: dict, fallback_model=None) -> dict:
    output = payload.get("output") if isinstance(payload, dict) else None
    content = _response_output_items_to_anthropic_content(output)
    stop_reason = "tool_use" if any(item.get("type") == "tool_use" for item in content if isinstance(item, dict)) else "end_turn"

    return {
        "id": payload.get("id") if isinstance(payload, dict) else f"msg_{uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": fallback_model or (payload.get("model") if isinstance(payload, dict) else None),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": response_usage_to_anthropic(payload.get("usage") if isinstance(payload, dict) else {}),
    }


def chat_usage_to_response(usage) -> dict:
    if not isinstance(usage, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    completion_details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    reasoning_tokens = usage.get("reasoning_output_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0

    response_usage = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    cached_tokens = prompt_details.get("cached_tokens")
    if cached_tokens is not None:
        response_usage["input_tokens_details"] = {"cached_tokens": cached_tokens}
    if reasoning_tokens:
        response_usage["output_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return response_usage


def chat_completion_to_response(payload: dict, fallback_model=None) -> dict:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}

    output = []
    text = _extract_chat_message_text(message if isinstance(message, dict) else {})
    if text:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )

    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
            output.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.get("id") or f"call_{uuid4().hex}",
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments", "") if isinstance(function.get("arguments"), str) else json.dumps(function.get("arguments", {}), separators=(",", ":")),
                }
            )

    output_text_parts = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        text_content = util.extract_item_text(item).strip()
        if text_content:
            output_text_parts.append(text_content)

    return {
        "id": (payload.get("id") if isinstance(payload, dict) else None) or f"resp_{uuid4().hex}",
        "object": "response",
        "created_at": payload.get("created") if isinstance(payload, dict) else int(time.time()),
        "status": "completed",
        "model": fallback_model or (payload.get("model") if isinstance(payload, dict) else None),
        "output": output,
        "output_text": "\n\n".join(output_text_parts),
        "usage": chat_usage_to_response(payload.get("usage") if isinstance(payload, dict) else {}),
    }


# ─── Error translation ───────────────────────────────────────────────────────

def _anthropic_error_type_for_status(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code in {503, 529}:
        return "overloaded_error"
    return "api_error"


def anthropic_error_payload_from_openai(payload, status_code: int, fallback_message: str | None = None) -> dict:
    if isinstance(payload, dict):
        if payload.get("type") == "error" and isinstance(payload.get("error"), dict):
            error = payload["error"]
            error_type = error.get("type")
            message = error.get("message")
            if isinstance(error_type, str) and isinstance(message, str) and message:
                return payload

        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        message = error.get("message") if isinstance(error.get("message"), str) else None
        error_type = error.get("type") if isinstance(error.get("type"), str) else None
        detail = payload.get("detail") if isinstance(payload.get("detail"), str) else None

        return {
            "type": "error",
            "error": {
                "type": error_type or _anthropic_error_type_for_status(status_code),
                "message": message or detail or fallback_message or "Request failed",
            },
        }

    return {
        "type": "error",
        "error": {
            "type": _anthropic_error_type_for_status(status_code),
            "message": fallback_message or "Request failed",
        },
    }


def anthropic_error_response(status_code: int, message: str, error_type: str | None = None, headers: dict | None = None) -> JSONResponse:
    payload = {
        "type": "error",
        "error": {
            "type": error_type or _anthropic_error_type_for_status(status_code),
            "message": message,
        },
    }
    return JSONResponse(content=payload, status_code=status_code, headers=headers)


def anthropic_error_response_from_upstream(upstream: httpx.Response) -> JSONResponse:
    headers = {}
    retry_after = upstream.headers.get("retry-after")
    if retry_after:
        headers["retry-after"] = retry_after

    fallback_message = ""
    try:
        fallback_message = upstream.text.strip()
    except Exception:
        fallback_message = ""
    if not fallback_message:
        fallback_message = f"Upstream request failed with status {upstream.status_code}"

    try:
        payload = upstream.json()
    except json.JSONDecodeError:
        payload = None

    translated = anthropic_error_payload_from_openai(payload, upstream.status_code, fallback_message)
    return anthropic_error_response(
        upstream.status_code,
        translated["error"]["message"],
        translated["error"]["type"],
        headers=headers or None,
    )


def _openai_error_type_for_status(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request_error"
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 429:
        return "rate_limit_error"
    return "server_error"


def openai_error_response(status_code: int, message: str, error_type: str | None = None, code=None, param=None, headers: dict | None = None) -> JSONResponse:
    payload = {
        "error": {
            "message": message,
            "type": error_type or _openai_error_type_for_status(status_code),
            "param": param,
            "code": code,
        }
    }
    return JSONResponse(content=payload, status_code=status_code, headers=headers)


def upstream_request_error_status_and_message(exc: httpx.RequestError) -> tuple[int, str]:
    if isinstance(exc, httpx.TimeoutException):
        return 504, "Upstream request timed out"
    if isinstance(exc, httpx.ConnectError):
        return 502, "Upstream connection failed"
    return 502, "Upstream request failed"


def http_exception_detail_to_message(detail) -> str:
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str) and message:
            return message
    return "Request failed"


# ─── SSE helpers ──────────────────────────────────────────────────────────────

def sse_encode(event_name: str, payload: dict) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def parse_sse_block(raw_block: str) -> tuple[str | None, str | None]:
    event_name = None
    data_lines = []
    for line in raw_block.replace("\r\n", "\n").split("\n"):
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return event_name, None
    return event_name, "\n".join(data_lines)


async def iter_sse_messages(byte_iter):
    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")()
    async for chunk in byte_iter:
        if isinstance(chunk, bytes):
            buffer += decoder.decode(chunk)
        else:
            buffer += str(chunk)

        normalized = buffer.replace("\r\n", "\n")
        while "\n\n" in normalized:
            raw_block, normalized = normalized.split("\n\n", 1)
            event_name, data = parse_sse_block(raw_block)
            if data is not None:
                yield event_name, data
        buffer = normalized

    buffer += decoder.decode(b"", final=True)

    trailing = buffer.strip()
    if trailing:
        event_name, data = parse_sse_block(trailing)
        if data is not None:
            yield event_name, data


def extract_text_from_chat_delta(delta) -> str:
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
    return ""


def extract_reasoning_from_chat_delta(delta) -> str:
    """Extract incremental reasoning/thinking text from a chat completion delta.

    Different upstream providers expose Claude/Gemini-style chain-of-thought via
    different field names on the streaming ``delta`` object. The Copilot chat
    completions endpoint, mirroring what vscode-copilot-chat consumes, surfaces
    Anthropic thinking_delta events as ``delta.thinking`` (string). Some
    OpenAI-compatible providers (DeepSeek, Together, OpenRouter, Groq, etc.)
    use ``delta.reasoning_content`` or ``delta.reasoning``. Copilot's
    Anthropic-fronting endpoint emits ``delta.reasoning_text`` (singular
    string per chunk). Accept any of them
    so the upstream "in-progress thoughts" can be relayed downstream regardless
    of which provider Copilot is fronting.

    Returns the concatenated reasoning text fragment, or an empty string if
    the delta carries no reasoning payload.
    """
    if not isinstance(delta, dict):
        return ""

    parts: list[str] = []
    for key in ("thinking", "reasoning_content", "reasoning_text", "reasoning"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
        elif isinstance(value, dict):
            # Some providers wrap the chunk as {"text": "..."} or
            # {"summary": [{"text": "..."}]}.
            text = value.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
            summary = value.get("summary")
            if isinstance(summary, list):
                for item in summary:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
    return "".join(parts)


def extract_tool_call_deltas(delta) -> list[dict]:
    if not isinstance(delta, dict):
        return []
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [item for item in tool_calls if isinstance(item, dict)]


def _strip_anthropic_cache_control(value):
    if isinstance(value, list):
        return [_strip_anthropic_cache_control(item) for item in value]

    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key == "cache_control":
                continue
            sanitized[key] = _strip_anthropic_cache_control(item)
        return sanitized

    return value



def _strip_copilot_cache_control(value):
    """Recursively remove copilot_cache_control from a structure.

    copilot_cache_control is a Copilot Chat API extension. When bridging
    to the Responses API the field is not recognised by the upstream endpoint
    and must be stripped before the payload is sent.
    """
    if isinstance(value, list):
        return [_strip_copilot_cache_control(item) for item in value]

    if isinstance(value, dict):
        return {k: _strip_copilot_cache_control(v) for k, v in value.items() if k != "copilot_cache_control"}

    return value


def prepare_anthropic_outbound_body(body: dict, resolved_model: str | None) -> dict:
    allowed_keys = {
        "model",
        "messages",
        "system",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "metadata",
        "stop_sequences",
        "stream",
        "tools",
        "tool_choice",
        "service_tier",
        "thinking",
        "container",
        "mcp_servers",
        "betas",
        "output_config",
    }

    outbound = {}
    dropped = []
    for key, value in body.items():
        if key in allowed_keys:
            outbound[key] = value
        else:
            dropped.append(key)

    outbound["model"] = resolved_model

    if dropped:
        print(f"Anthropic proxy dropped unsupported fields: {', '.join(sorted(dropped))}", flush=True)

    return _strip_anthropic_cache_control(outbound)


# ─── Compaction ───────────────────────────────────────────────────────────────

def encode_fake_compaction(summary_text: str) -> str:
    encoded = base64.urlsafe_b64encode(summary_text.encode("utf-8")).decode("ascii")
    return f"{FAKE_COMPACTION_PREFIX}{encoded}"


def decode_fake_compaction(encrypted_content: str) -> str | None:
    if not isinstance(encrypted_content, str) or not encrypted_content.startswith(FAKE_COMPACTION_PREFIX):
        return None

    encoded = encrypted_content[len(FAKE_COMPACTION_PREFIX) :]
    try:
        decoded = base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
    except Exception:
        return None
    return decoded or None


def _summary_message_item(summary_text: str) -> dict:
    return {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": f"{FAKE_COMPACTION_SUMMARY_LABEL}\n{summary_text}",
            }
        ],
    }


def input_contains_compaction(input_items) -> bool:
    if not isinstance(input_items, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == "compaction" for item in input_items)


def _latest_compaction_window(input_items):
    if not isinstance(input_items, list):
        return input_items

    latest_compaction_index = None
    for index, item in enumerate(input_items):
        if isinstance(item, dict) and item.get("type") == "compaction":
            latest_compaction_index = index

    if latest_compaction_index is None:
        return input_items
    return input_items[latest_compaction_index:]


def _summarize_inline_data_image(image_url: str, *, detail=None) -> str | None:
    if not isinstance(image_url, str) or not image_url.startswith("data:"):
        return None

    media_type = "inline image"
    if ";" in image_url and image_url.startswith("data:"):
        media_type = image_url[5:].split(";", 1)[0] or media_type

    summary = f"[inline tool image omitted: {media_type}, {len(image_url)} chars]"
    if isinstance(detail, str) and detail.strip():
        summary = f"{summary[:-1]}, detail={detail.strip()}]"
    return summary


def _sanitize_function_call_output_item(item: dict) -> dict:
    if not isinstance(item, dict) or item.get("type") != "function_call_output":
        return item

    output = item.get("output")
    if not isinstance(output, list):
        return item

    changed = False
    sanitized_output = []
    for part in output:
        if not isinstance(part, dict):
            sanitized_output.append(part)
            continue

        if str(part.get("type", "")).lower() != "input_image":
            sanitized_output.append(part)
            continue

        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        summary_text = _summarize_inline_data_image(image_url, detail=part.get("detail"))
        if summary_text is None:
            sanitized_output.append(part)
            continue

        changed = True
        sanitized_output.append({"type": "input_text", "text": summary_text})

    if not changed:
        return item
    return {
        **item,
        "output": sanitized_output,
    }


def sanitize_input(input_items):
    """
    Preserve encrypted_content in reasoning items for multi-turn correctness.
    Treat the latest compaction item as the active handoff boundary so older
    pre-compaction prompt items are not replayed upstream.
    Expand locally synthesized compaction items into a readable summary message.
    Convert other compaction items into reasoning items for GHCP compatibility.
    Strip status=None which GHCP rejects.
    Pass everything else through unchanged.
    """
    if not isinstance(input_items, list):
        return input_items  # plain string — pass through untouched

    result = []
    for item in _latest_compaction_window(input_items):
        if not isinstance(item, dict):
            result.append(item)
            continue

        item_type = item.get("type")
        if item_type == "compaction":
            encrypted_content = item.get("encrypted_content")
            summary_text = decode_fake_compaction(encrypted_content)
            if summary_text is not None:
                result.append(_summary_message_item(summary_text))
                continue
            if isinstance(encrypted_content, str) and encrypted_content:
                result.append(
                    {
                        "type": "reasoning",
                        "encrypted_content": encrypted_content,
                    }
                )
                continue
            result.append(item)
            continue

        if item_type == "function_call_output":
            result.append(_sanitize_function_call_output_item(item))
            continue

        if item_type != "reasoning":
            result.append(item)
            continue

        filtered = {}
        for k, v in item.items():
            if k == "encrypted_content":
                if v is not None:
                    filtered[k] = v   # always preserve if present
                continue
            if k == "status" and v is None:
                continue              # strip status=None
            if v is not None:
                filtered[k] = v
        result.append(filtered)
    return result


def _compaction_message_item(role: str, text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None

    part_type = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": part_type, "text": text}],
    }


def _is_fake_compaction_summary_message(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    if str(item.get("type", "")).lower() != "message":
        return False
    text = util.extract_item_text(item)
    return text.startswith(f"{FAKE_COMPACTION_SUMMARY_LABEL}\n")


def _format_compaction_tool_call(item: dict) -> str | None:
    name = item.get("name")
    if not isinstance(name, str) or not name:
        return None

    arguments = item.get("arguments")
    if isinstance(arguments, str):
        arguments_text = arguments.strip() or "{}"
    elif arguments is None:
        arguments_text = "{}"
    else:
        arguments_text = json.dumps(arguments, separators=(",", ":"))

    call_id = item.get("call_id") or item.get("id")
    call_suffix = f" ({call_id})" if isinstance(call_id, str) and call_id else ""
    return f"[Tool call{call_suffix}] {name}\n{arguments_text}"


def _format_compaction_tool_output(item: dict) -> str | None:
    call_id = item.get("call_id")
    label = f"[Tool result ({call_id})]" if isinstance(call_id, str) and call_id else "[Tool result]"

    output = item.get("output")
    if isinstance(output, list):
        output_text = "".join(util.extract_item_text(part) for part in output if isinstance(part, dict))
    elif isinstance(output, str):
        output_text = output
    elif output is None:
        output_text = ""
    else:
        output_text = json.dumps(output, separators=(",", ":"))

    output_text = output_text.strip()
    return f"{label}\n{output_text}" if output_text else label


def _compaction_transcript_message_item(item: dict, *, force_user_role: bool) -> dict | None:
    role = str(item.get("role", "user")).lower()
    if role not in {"system", "developer", "user", "assistant"}:
        return None

    text = util.extract_item_text(item)
    if _is_fake_compaction_summary_message(item):
        return _compaction_message_item("user", text)
    if not force_user_role:
        return item
    if role != "user":
        text = f"[{role} message]\n{text}"
    return _compaction_message_item("user", text)


def _compaction_transcript_input_items(input_items, *, force_user_role: bool = False) -> list[dict]:
    transcript = []
    if not isinstance(input_items, list):
        return transcript

    for item in input_items:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type", "")).lower()
        if item_type == "message":
            transcript_item = _compaction_transcript_message_item(item, force_user_role=force_user_role)
            if transcript_item is not None:
                transcript.append(transcript_item)
            continue

        if item_type == "function_call":
            transcript_role = "user" if force_user_role else "assistant"
            transcript_item = _compaction_message_item(transcript_role, _format_compaction_tool_call(item))
            if transcript_item is not None:
                transcript.append(transcript_item)
            continue

        if item_type == "function_call_output":
            transcript_item = _compaction_message_item("user", _format_compaction_tool_output(item))
            if transcript_item is not None:
                transcript.append(transcript_item)
            continue

        if item_type == "compaction":
            encrypted_content = item.get("encrypted_content")
            summary_text = decode_fake_compaction(encrypted_content)
            if summary_text is not None:
                transcript_item = _compaction_message_item("user", f"{FAKE_COMPACTION_SUMMARY_LABEL}\n{summary_text}")
                if transcript_item is not None:
                    transcript.append(transcript_item)
            continue

        # reasoning and item_reference do not add useful user-visible
        # context for summarization and can trigger upstream validation
        # issues when bridged through tool-aware chat adapters.
        if item_type in {"reasoning", "item_reference"}:
            continue

    return transcript

def _compaction_requires_chat_transcript(model_name) -> bool:
    normalized = normalize_upstream_model_name(resolve_copilot_model_name(model_name))
    if not isinstance(normalized, str):
        return False
    return normalized.startswith("claude-") or normalized.startswith("anthropic/")


def _copy_compaction_passthrough_fields(source: dict, target: dict) -> None:
    if not isinstance(source, dict) or not isinstance(target, dict):
        return

    # Preserve the same cache/session affinity fields as ordinary Responses
    # traffic so the synthetic summary request can reuse GitHub's prompt state.
    for key in (
        "session_id",
        "sessionId",
        "prompt_cache_key",
        "promptCacheKey",
        "previous_response_id",
        "metadata",
        "user",
    ):
        if key in source:
            target[key] = source.get(key)


def _apply_compaction_request_config(source: dict, target: dict) -> None:
    if not isinstance(source, dict) or not isinstance(target, dict):
        return

    for key, value in source.items():
        if key == "input":
            continue
        target[key] = value


def _strip_chat_transcript_compaction_fields(target: dict) -> None:
    if not isinstance(target, dict):
        return

    # GHCP keeps tools in compact requests for cache affinity but sets
    # tool_choice to "none" so the model cannot invoke them during
    # summarization.  Match that behaviour here.
    if isinstance(target.get("tools"), list) and target["tools"]:
        target["tool_choice"] = "none"
    else:
        for key in ("tools", "tool_choice"):
            target.pop(key, None)
    target.pop("parallel_tool_calls", None)


def build_fake_compaction_request(body: dict, *, force_responses_safe_transcript: bool = False) -> dict:
    request_input = body.get("input")
    if isinstance(request_input, list):
        request_input = sanitize_input(request_input)

    # Match opencode's native compaction shape for Responses/GPT models so the
    # upstream prefix stays byte-stable and can reuse prompt cache / item state.
    # Flatten the transcript for chat-backed targets, or when a translated-model
    # compact will run through a GPT fallback that cannot safely replay the raw
    # assistant/tool-shaped Responses history.
    requires_transcript = force_responses_safe_transcript or _compaction_requires_chat_transcript(body.get("model"))
    if requires_transcript:
        if isinstance(request_input, list):
            input_items = _compaction_transcript_input_items(
                request_input,
                force_user_role=force_responses_safe_transcript,
            )
        elif isinstance(request_input, str):
            input_items = [_compaction_message_item("user", request_input)]
            input_items = [item for item in input_items if item is not None]
        else:
            input_items = []
    elif isinstance(request_input, list):
        input_items = list(request_input)
    elif isinstance(request_input, str):
        input_items = [_compaction_message_item("user", request_input)]
        input_items = [item for item in input_items if item is not None]
    else:
        input_items = []

    input_items.append({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": COMPACTION_SUMMARY_PROMPT}],
    })

    compact_request = {"input": input_items}
    _copy_compaction_passthrough_fields(body, compact_request)
    _apply_compaction_request_config(body, compact_request)
    if requires_transcript:
        _strip_chat_transcript_compaction_fields(compact_request)
    return compact_request


def extract_response_output_text(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = payload.get("output")
    if not isinstance(output, list):
        return None

    parts = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        if str(item.get("role", "")).lower() != "assistant":
            continue
        text = util.extract_item_text(item).strip()
        if text:
            parts.append(text)

    if not parts:
        return None
    return "\n\n".join(parts)
