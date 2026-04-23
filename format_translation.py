"""Request/response format translation between Anthropic Messages API, OpenAI Chat Completions API, and OpenAI Responses API formats."""

import base64
import codecs
import json
import time

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

# Top-level Responses-API fields that Codex may send but Copilot's upstream
# Responses endpoint rejects (e.g. priority/flex routing hints).
_COPILOT_UNSUPPORTED_RESPONSES_BODY_KEYS = {"service_tier"}


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
            "arguments": json.dumps(tool_input, separators=(",", ":"), ensure_ascii=False),
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

    if item_type in {"input_file", "file"}:
        # Responses-API input_file carries either a data: URL in file_data or
        # a file_url/file_id. Anthropic /v1/messages requires a `document`
        # block with a base64 source for PDFs; URL/file_id variants cannot
        # round-trip without an upstream fetch, so we drop them (matches TS
        # which only emits a document block for the data: URL case).
        file_data = item.get("file_data")
        filename = item.get("filename") or "document.pdf"
        if isinstance(file_data, str) and file_data.startswith("data:"):
            try:
                header, data = file_data.split(",", 1)
                media_type = header[5:].split(";", 1)[0] or "application/pdf"
            except ValueError:
                return None
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
                "title": filename,
            }
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


def sanitize_responses_body_for_copilot(body: dict) -> dict:
    """Strip top-level Responses-API fields that Copilot's upstream rejects.

    Codex sends fields such as ``service_tier`` (``priority``/``flex``) that
    Copilot's GitHub-fronted Responses endpoint does not accept. Drop them so
    the request is forwarded successfully; we still record the requested
    service tier separately for usage tracking via the Codex logs scraper.
    """
    if not isinstance(body, dict):
        return body
    removed = [k for k in _COPILOT_UNSUPPORTED_RESPONSES_BODY_KEYS if k in body]
    if not removed:
        return body
    sanitized = {k: v for k, v in body.items() if k not in _COPILOT_UNSUPPORTED_RESPONSES_BODY_KEYS}
    print(
        f"Responses proxy dropped Copilot-unsupported fields: {', '.join(sorted(removed))}",
        flush=True,
    )
    return sanitized


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
        input_text = json.dumps(tool_input, separators=(",", ":"), ensure_ascii=False)

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
        output_text = json.dumps(output, separators=(",", ":"), ensure_ascii=False)

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
                                    else json.dumps(item.get("arguments", {}), separators=(",", ":"), ensure_ascii=False),
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
                    output_text = json.dumps(output_text, separators=(",", ":"), ensure_ascii=False) if output_text is not None else ""
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
                            "arguments": json.dumps(item.get("input") or {}, separators=(",", ":"), ensure_ascii=False),
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
                    "arguments": function.get("arguments", "") if isinstance(function.get("arguments"), str) else json.dumps(function.get("arguments", {}), separators=(",", ":"), ensure_ascii=False),
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


def chat_completion_to_compaction_response(payload: dict, fallback_model=None) -> dict:
    """Wrap a chat-completions summary as a Responses compact payload.

    Codex's remote compaction client expects ``/responses/compact`` to return
    ``{"output": [ResponseItem...]}``, and later sends any compaction item back
    in normal Responses history.  Chat-backed models can produce the summary
    text, but they cannot produce a native compaction item, so encode the text
    in the proxy's local compaction format for the next request to expand.
    """
    choices = payload.get("choices") if isinstance(payload, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    summary_text = _extract_chat_message_text(message if isinstance(message, dict) else {}).strip()
    if not summary_text:
        summary_text = "(no summary available)"

    return {
        "id": (payload.get("id") if isinstance(payload, dict) else None) or f"resp_{uuid4().hex}",
        "object": "response",
        "created_at": payload.get("created") if isinstance(payload, dict) else int(time.time()),
        "status": "completed",
        "model": fallback_model or (payload.get("model") if isinstance(payload, dict) else None),
        "output": [
            {
                "type": "compaction",
                "encrypted_content": encode_fake_compaction(summary_text),
            }
        ],
        "output_text": summary_text,
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
    return f"event: {event_name}\ndata: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n".encode("utf-8")


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

    # GHCP rejects a stray ``content`` array on function_call_output items.
    if "content" in item:
        item = {k: v for k, v in item.items() if k != "content"}

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
            # GHCP's Responses API rejects a non-empty ``content`` array on
            # non-message items (e.g. function_call, function_call_output).
            # Codex sometimes echoes a stray empty/legacy ``content`` field on
            # these items; strip it defensively so upstream does not 400 with
            # "Invalid 'input[N].content': array too long".
            if (
                isinstance(item, dict)
                and "content" in item
                and item_type not in (None, "message")
            ):
                cleaned = {k: v for k, v in item.items() if k != "content"}
                result.append(cleaned)
            else:
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
            if k == "content":
                # GHCP's Responses API rejects ``content`` on reasoning items
                # ("array too long. Expected ... maximum length 0"). Reasoning
                # text belongs in ``summary`` / ``encrypted_content``; drop any
                # stray ``content`` payload here.
                continue
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
        arguments_text = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)

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
        output_text = json.dumps(output, separators=(",", ":"), ensure_ascii=False)

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

# ─── Responses → Anthropic Messages translation ──────────────────────────────

def _responses_input_text_to_anthropic_block(item: dict) -> dict | None:
    """Translate a Responses content part into an Anthropic content block.

    Returns ``None`` if the part is not representable (e.g. unknown type).
    """
    if not isinstance(item, dict):
        return None
    item_type = str(item.get("type", "")).lower()
    if item_type in {"input_text", "output_text", "text"}:
        text = item.get("text")
        if isinstance(text, str):
            return {"type": "text", "text": text}
        return None
    if item_type in {"input_file", "file"}:
        # Responses-API input_file carries either a data: URL in file_data or
        # a file_url/file_id. Anthropic /v1/messages requires a `document`
        # block with a base64 source for PDFs; URL/file_id variants cannot
        # round-trip without an upstream fetch, so we drop them (matches TS
        # which only emits a document block for the data: URL case).
        file_data = item.get("file_data")
        filename = item.get("filename") or "document.pdf"
        if isinstance(file_data, str) and file_data.startswith("data:"):
            try:
                header, data = file_data.split(",", 1)
                media_type = header[5:].split(";", 1)[0] or "application/pdf"
            except ValueError:
                return None
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
                "title": filename,
            }
        return None
    if item_type == "input_image":
        # Responses encodes images either as a bare image_url string or as a
        # nested {image_url:{url:...}} object. Anthropic wants either a base64
        # source (for data: URLs) or a url source.
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if not isinstance(image_url, str):
            return None
        if image_url.startswith("data:"):
            try:
                header, data = image_url.split(",", 1)
                # header looks like "data:image/png;base64"
                media_type = header[5:].split(";", 1)[0]
            except ValueError:
                return None
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type or "image/png",
                    "data": data,
                },
            }
        return {
            "type": "image",
            "source": {"type": "url", "url": image_url},
        }
    return None


def _split_reasoning_signature(signature: str) -> tuple[str, str]:
    """Split a reasoning signature of the form ``"<encrypted>@<id>"``.

    Mirrors copilot-api's ``parseReasoningSignature``: we split on the LAST
    ``@`` so encrypted payloads that happen to contain ``@`` still round-trip
    correctly. If the split is at the start/end or absent, fall back to
    treating the entire string as the encrypted payload with an empty id.
    """
    if not isinstance(signature, str) or not signature:
        return "", ""
    idx = signature.rfind("@")
    if idx <= 0 or idx == len(signature) - 1:
        return signature, ""
    return signature[:idx], signature[idx + 1:]


def _responses_reasoning_item_to_anthropic_block(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    summary = item.get("summary")
    text_parts: list[str] = []
    if isinstance(summary, list):
        for part in summary:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif isinstance(part, str):
                text_parts.append(part)
    elif isinstance(summary, str):
        text_parts.append(summary)
    thinking_text = "\n".join(text_parts)
    block: dict = {"type": "thinking", "thinking": thinking_text}
    encrypted = item.get("encrypted_content")
    if isinstance(encrypted, str) and encrypted:
        reasoning_id = item.get("id")
        if isinstance(reasoning_id, str) and reasoning_id:
            block["signature"] = f"{encrypted}@{reasoning_id}"
        else:
            block["signature"] = encrypted
    return block


def _responses_function_call_to_anthropic_block(item: dict) -> dict:
    name = item.get("name")
    if not isinstance(name, str):
        raise ValueError("Responses function_call items must include a name")
    call_id = item.get("call_id") or item.get("id")
    if not isinstance(call_id, str):
        raise ValueError("Responses function_call items must include call_id")
    raw_args = item.get("arguments")
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed = {"_raw": raw_args}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
    elif isinstance(raw_args, dict):
        parsed = raw_args
    else:
        parsed = {}
    return {
        "type": "tool_use",
        "id": call_id,
        "name": name,
        "input": parsed,
    }


def _responses_function_call_output_to_anthropic_block(item: dict) -> dict:
    call_id = item.get("call_id")
    if not isinstance(call_id, str):
        raise ValueError("Responses function_call_output items must include call_id")
    output = item.get("output")
    if isinstance(output, list):
        text = "".join(util.extract_item_text(part) for part in output if isinstance(part, dict))
    elif isinstance(output, str):
        text = output
    elif output is None:
        text = ""
    else:
        text = json.dumps(output, separators=(",", ":"), ensure_ascii=False)
    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": [{"type": "text", "text": text}],
    }


def _responses_tool_to_anthropic(tool: dict) -> dict | None:
    if not isinstance(tool, dict):
        return None
    tool_type = str(tool.get("type", "")).lower()
    if tool_type and tool_type != "function":
        return None
    name = tool.get("name") or (
        tool.get("function", {}).get("name") if isinstance(tool.get("function"), dict) else None
    )
    if not isinstance(name, str):
        raise ValueError("Responses tools must include a name")
    description = tool.get("description")
    if not isinstance(description, str):
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        description = function.get("description") if isinstance(function.get("description"), str) else ""
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {"type": "object", "properties": {}}
    return {
        "name": name,
        "description": description or "",
        "input_schema": parameters,
    }


def _responses_tool_choice_to_anthropic(tool_choice):
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        normalized = tool_choice.lower()
        if normalized == "auto":
            return {"type": "auto"}
        if normalized == "required":
            return {"type": "any"}
        if normalized == "none":
            return {"type": "none"}
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type", "")).lower()
        if choice_type in {"auto", "none"}:
            return {"type": choice_type}
        if choice_type == "required":
            return {"type": "any"}
        if choice_type == "function":
            name = tool_choice.get("name")
            if not isinstance(name, str):
                function = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
                name = function.get("name") if isinstance(function.get("name"), str) else None
            if not isinstance(name, str):
                raise ValueError("Responses tool_choice type=function must include name")
            return {"type": "tool", "name": name}
    raise ValueError("Unsupported Responses tool_choice value")


_ANTHROPIC_CACHEABLE_BLOCK_TYPES = {"text", "image", "document", "tool_use", "tool_result"}


def _responses_body_requests_prompt_cache(body: dict) -> bool:
    for key in ("prompt_cache_key", "promptCacheKey"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _add_anthropic_cache_control(block: dict) -> None:
    if not isinstance(block.get("cache_control"), dict):
        block["cache_control"] = {"type": "ephemeral"}


def _last_cacheable_anthropic_content_block(content) -> dict | None:
    if not isinstance(content, list):
        return None
    for block in reversed(content):
        if not isinstance(block, dict):
            continue
        if str(block.get("type", "")).lower() in _ANTHROPIC_CACHEABLE_BLOCK_TYPES:
            return block
    return None


def _apply_responses_prompt_cache_to_anthropic_messages(payload: dict) -> None:
    """Preserve Responses prompt-cache intent after translating to Messages.

    Copilot's Responses API accepts ``prompt_cache_key``. Anthropic Messages
    does not, so the equivalent cache request is explicit ``cache_control``
    breakpoints. Keep this to four markers, matching Anthropic's limit.
    """
    if not isinstance(payload, dict):
        return

    remaining = 4

    def mark(block: dict | None) -> None:
        nonlocal remaining
        if remaining <= 0 or not isinstance(block, dict):
            return
        _add_anthropic_cache_control(block)
        remaining -= 1

    tools = payload.get("tools")
    if isinstance(tools, list):
        for tool in reversed(tools):
            if isinstance(tool, dict):
                mark(tool)
                break

    system = payload.get("system")
    if isinstance(system, str) and system:
        block = {"type": "text", "text": system}
        mark(block)
        payload["system"] = [block]
    elif isinstance(system, list):
        mark(_last_cacheable_anthropic_content_block(system))

    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages[-2:]:
            if not isinstance(message, dict):
                continue
            mark(_last_cacheable_anthropic_content_block(message.get("content")))


def responses_request_to_anthropic_messages(body: dict) -> dict:
    """Translate an OpenAI Responses request body into Anthropic Messages shape.

    This is the inverse of :func:`anthropic_request_to_responses`. The result
    is a fresh dict ready to send to an Anthropic ``/v1/messages`` endpoint
    (apart from any host-specific adornments the caller may add).
    """
    if not isinstance(body, dict):
        raise ValueError("Responses body must be a dict")

    raw_input = body.get("input")
    messages: list[dict] = []
    system_extra: list[str] = []

    # Coalesce consecutive same-role messages.
    def push_blocks(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if messages and messages[-1].get("role") == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": list(blocks)})

    if isinstance(raw_input, str):
        if raw_input:
            push_blocks("user", [{"type": "text", "text": raw_input}])
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).lower()

            if item_type == "message":
                role = str(item.get("role", "user")).lower()
                content = item.get("content")
                if role in {"system", "developer"}:
                    # Route system/developer messages to top-level ``system``.
                    text_parts: list[str] = []
                    if isinstance(content, str):
                        text_parts.append(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                text_parts.append(part["text"])
                    text = "\n\n".join(p for p in text_parts if p)
                    if text:
                        system_extra.append(text)
                    continue
                if role not in {"user", "assistant"}:
                    raise ValueError(f"Unsupported Responses message role: {role}")
                blocks: list[dict] = []
                if isinstance(content, str):
                    blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for part in content:
                        block = _responses_input_text_to_anthropic_block(part)
                        if block is not None:
                            blocks.append(block)
                push_blocks(role, blocks)
                continue

            if item_type == "function_call":
                push_blocks("assistant", [_responses_function_call_to_anthropic_block(item)])
                continue

            if item_type == "function_call_output":
                push_blocks("user", [_responses_function_call_output_to_anthropic_block(item)])
                continue

            if item_type == "custom_tool_call":
                custom_tool_text = _format_custom_tool_call_for_chat(item)
                if custom_tool_text is None:
                    raise ValueError("Responses custom_tool_call items must include a name")
                push_blocks("assistant", [{"type": "text", "text": custom_tool_text}])
                continue

            if item_type == "custom_tool_call_output":
                push_blocks("user", [{"type": "text", "text": _format_custom_tool_output_for_chat(item)}])
                continue

            if item_type == "reasoning":
                block = _responses_reasoning_item_to_anthropic_block(item)
                if block is not None:
                    push_blocks("assistant", [block])
                continue

            if item_type in {"compaction", "item_reference", "web_search_call"}:
                # No native Anthropic equivalent; skip silently to mirror the
                # behaviour of responses_request_to_chat.
                continue

            raise ValueError(f"Unsupported Responses input item type: {item_type}")

    payload: dict = {
        "model": body.get("model"),
        "messages": messages,
    }

    # System / instructions
    system_parts: list[str] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        system_parts.append(instructions)
    system_parts.extend(system_extra)
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    # max_tokens is required by Anthropic.
    max_tokens = body.get("max_output_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = 4096
    payload["max_tokens"] = max_tokens

    for source_key, target_key in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("metadata", "metadata"),
    ):
        value = body.get(source_key)
        if value is not None:
            payload[target_key] = value

    # Anthropic requires temperature in [0, 1]; Codex commonly sends 1.5/2.0.
    temp = payload.get("temperature")
    if isinstance(temp, (int, float)):
        if temp < 0:
            payload["temperature"] = 0
        elif temp > 1:
            payload["temperature"] = 1

    # Stop sequences: Responses uses "stop_sequences" or "stop".
    stop = body.get("stop_sequences")
    if stop is None:
        stop = body.get("stop")
    if stop is not None:
        payload["stop_sequences"] = stop

    if body.get("stream") is not None:
        payload["stream"] = bool(body.get("stream"))

    # Tools
    tools = body.get("tools")
    if isinstance(tools, list):
        translated: list[dict] = []
        for tool in tools:
            converted = _responses_tool_to_anthropic(tool)
            if converted is not None:
                translated.append(converted)
        if translated:
            payload["tools"] = translated

    # tool_choice
    tool_choice = _responses_tool_choice_to_anthropic(body.get("tool_choice"))
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    # parallel_tool_calls (Responses) -> disable_parallel_tool_use (Anthropic)
    # Only emit when explicitly disabling parallel calls; some Claude models
    # reject the field, and the default is already to allow parallel.
    parallel = body.get("parallel_tool_calls")
    if parallel is False:
        payload["disable_parallel_tool_use"] = True

    # Reasoning effort -> thinking adaptive + output_config carry-through.
    # Anthropic models reject extended thinking when tool_choice forces a
    # specific tool (type=any/tool), so skip the thinking injection then.
    reasoning = body.get("reasoning")
    incoming_effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    mapped_effort = effort_mapping.map_effort_for_model(payload.get("model"), incoming_effort)
    forces_tool = isinstance(tool_choice, dict) and tool_choice.get("type") in ("any", "tool")
    if mapped_effort is not None and not forces_tool:
        payload["thinking"] = {"type": "adaptive", "display": "summarized"}
        payload["output_config"] = {"effort": mapped_effort}

    if _responses_body_requests_prompt_cache(body):
        _apply_responses_prompt_cache_to_anthropic_messages(payload)

    return payload


# ─── Anthropic Messages response → Responses response translation ────────────

def _anthropic_usage_to_responses_usage(usage) -> dict:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens")
    cache_create = usage.get("cache_creation_input_tokens")
    out: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    details: dict = {}
    if isinstance(cache_read, int) and cache_read:
        details["cached_tokens"] = cache_read
    if isinstance(cache_create, int) and cache_create:
        details["cache_creation_input_tokens"] = cache_create
    if details:
        out["input_tokens_details"] = details
    return out


_ANTHROPIC_STOP_REASON_TO_RESPONSES_STATUS = {
    "end_turn": ("completed", None),
    "tool_use": ("completed", None),
    "stop_sequence": ("completed", None),
    "pause_turn": ("completed", None),
    "max_tokens": ("incomplete", {"reason": "max_output_tokens"}),
    "refusal": ("incomplete", {"reason": "content_filter"}),
}


def _anthropic_content_blocks_to_responses_output(content) -> list[dict]:
    if not isinstance(content, list):
        return []

    output: list[dict] = []
    pending_message_parts: list[dict] = []

    def flush_message():
        nonlocal pending_message_parts
        if pending_message_parts:
            output.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": list(pending_message_parts),
                }
            )
            pending_message_parts = []

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "")).lower()

        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                pending_message_parts.append(
                    {"type": "output_text", "text": text, "annotations": []}
                )
            continue

        # Anything other than ``text`` becomes its own top-level item.
        flush_message()

        if block_type == "thinking":
            thinking_text = block.get("thinking")
            if not isinstance(thinking_text, str):
                thinking_text = ""
            raw_signature = block.get("signature")
            encrypted_content = None
            reasoning_id = None
            if isinstance(raw_signature, str) and raw_signature:
                encrypted_content, reasoning_id = _split_reasoning_signature(raw_signature)
            reasoning_item: dict = {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": thinking_text}],
            }
            if reasoning_id:
                reasoning_item["id"] = reasoning_id
            if encrypted_content:
                reasoning_item["encrypted_content"] = encrypted_content
            output.append(reasoning_item)
            continue

        if block_type == "redacted_thinking":
            # Preserve the opaque payload via encrypted_content if present.
            redacted = block.get("data") or block.get("signature")
            reasoning_item = {
                "type": "reasoning",
                "summary": [],
            }
            if isinstance(redacted, str) and redacted:
                reasoning_item["encrypted_content"] = redacted
            output.append(reasoning_item)
            continue

        if block_type == "tool_use":
            name = block.get("name")
            tool_id = block.get("id")
            if not isinstance(name, str) or not isinstance(tool_id, str):
                continue
            tool_input = block.get("input") if isinstance(block.get("input"), (dict, list)) else {}
            output.append(
                {
                    "type": "function_call",
                    "call_id": tool_id,
                    "name": name,
                    "arguments": json.dumps(tool_input, separators=(",", ":"), ensure_ascii=False),
                }
            )
            continue

        # Unknown block type — skip.

    flush_message()
    return output


def anthropic_response_to_responses(payload: dict, *, fallback_model: str | None = None) -> dict:
    """Translate a non-streaming Anthropic Messages response into a Responses payload.

    Inverse of :func:`response_payload_to_anthropic`.
    """
    if not isinstance(payload, dict):
        payload = {}

    content = payload.get("content")
    output_items = _anthropic_content_blocks_to_responses_output(content)

    stop_reason = payload.get("stop_reason")
    status_pair = _ANTHROPIC_STOP_REASON_TO_RESPONSES_STATUS.get(
        stop_reason if isinstance(stop_reason, str) else "",
        ("completed", None),
    )
    status, incomplete_details = status_pair

    response_id = payload.get("id")
    if not isinstance(response_id, str) or not response_id:
        response_id = "resp_anthropic"

    model = payload.get("model") or fallback_model

    # Aggregate output_text from any assistant message blocks for parity with
    # chat_completion_to_response.
    output_text_parts: list[str] = []
    for item in output_items:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                stripped = part["text"].strip()
                if stripped:
                    output_text_parts.append(stripped)

    created_at = payload.get("created") if isinstance(payload, dict) else None
    if not isinstance(created_at, int):
        created_at = int(time.time())

    result: dict = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "status": status,
        "output": output_items,
        "output_text": "\n\n".join(output_text_parts),
        "usage": _anthropic_usage_to_responses_usage(payload.get("usage")),
    }
    if incomplete_details is not None:
        result["incomplete_details"] = incomplete_details
    return result
