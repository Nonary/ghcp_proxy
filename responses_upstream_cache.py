"""Upstream-facing Responses request sanitization and cache-aware formatting."""

from __future__ import annotations

import json

import effort_mapping
import util


_COPILOT_UNSUPPORTED_RESPONSES_TOOL_TYPES = {"image_generation"}

# Native Responses fields the upstream Copilot endpoint actually consumes —
# either captured in CLI traffic or part of the documented Responses contract
# that contributes to the upstream prompt-prefix cache key. Anything else is
# treated as a Codex-only client field and stripped, since unknown keys
# perturb the upstream cache hash with no behavioral upside.
_COPILOT_RESPONSES_UPSTREAM_BODY_KEYS = {
    "include",
    "input",
    "instructions",
    "metadata",
    "model",
    "parallel_tool_calls",
    "previous_response_id",
    "prompt_cache_key",
    "promptCacheKey",
    "reasoning",
    "safety_identifier",
    "store",
    "stream",
    "text",
    "tools",
    "user",
}

_TOOL_SEARCH_TOOL_NAMES = {"tool_search", "tools.tool_search"}
_DANGEROUS_CODE_EXECUTION_TOOL_NAMES = {"mcp__ide__executecode"}


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


def _append_sanitizer_diagnostic(diagnostics: list[dict] | None, diagnostic: dict) -> None:
    if isinstance(diagnostics, list):
        diagnostics.append(diagnostic)


def sanitize_responses_body_for_copilot(body: dict, *, diagnostics: list[dict] | None = None) -> dict:
    """Keep native Responses upstream payloads aligned with Copilot CLI.

    Codex sends local/client fields such as ``client_metadata`` that aren't
    part of the upstream Responses contract and would split the upstream
    prompt cache hash for no behavioral gain. Drop those, but preserve the
    documented cache-bearing fields (``prompt_cache_key``,
    ``previous_response_id``, ``metadata``, ``user``, ``safety_identifier``)
    so upstream can keep its prefix cache stable across turns.
    """
    if not isinstance(body, dict):
        return body
    removed = sorted(k for k in body if k not in _COPILOT_RESPONSES_UPSTREAM_BODY_KEYS)
    if not removed:
        return body
    sanitized = {
        k: v
        for k, v in body.items()
        if k in _COPILOT_RESPONSES_UPSTREAM_BODY_KEYS
    }
    _append_sanitizer_diagnostic(
        diagnostics,
        {
            "kind": "responses_body",
            "action": "drop_non_copilot_cli_fields",
            "fields": removed,
        },
    )
    print(
        f"Responses proxy dropped non-Copilot-CLI fields: {', '.join(removed)}",
        flush=True,
    )
    return sanitized


def is_dangerous_code_execution_tool_name(name) -> bool:
    """Return True for request-provided tool names that expose local code execution."""
    return isinstance(name, str) and name.strip().lower() in _DANGEROUS_CODE_EXECUTION_TOOL_NAMES


def _responses_tool_name(tool: dict) -> str | None:
    name = tool.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    function = tool.get("function") if isinstance(tool.get("function"), dict) else None
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def _responses_tool_is_tool_search(tool: dict) -> bool:
    tool_type = str(tool.get("type", "")).strip().lower()
    if tool_type == "tool_search":
        return True
    if "tool_search" in tool:
        return True
    name = _responses_tool_name(tool)
    if not isinstance(name, str):
        return False
    normalized = name.strip().lower()
    return normalized in _TOOL_SEARCH_TOOL_NAMES or normalized.endswith("__tool_search")


def _drop_dangerous_responses_tools(node, *, path: str, removed: list[dict]):  # pragma: no mutate
    if isinstance(node, list):
        changed = False
        new_items = []
        for index, item in enumerate(node):
            item_path = f"{path}[{index}]"
            if isinstance(item, dict) and is_dangerous_code_execution_tool_name(_responses_tool_name(item)):
                removed.append(
                    {
                        "path": item_path,
                        "name": _responses_tool_name(item),
                        "type": item.get("type"),
                    }
                )
                changed = True
                continue
            new_item, item_changed = _drop_dangerous_responses_tools(
                item,
                path=item_path,
                removed=removed,
            )
            changed = changed or item_changed
            new_items.append(new_item)
        return (new_items if changed else node), changed

    if not isinstance(node, dict):
        return node, False

    if is_dangerous_code_execution_tool_name(_responses_tool_name(node)):
        removed.append(
            {
                "path": path,
                "name": _responses_tool_name(node),
                "type": node.get("type"),
            }
        )
        return None, True

    nested = node.get("tools")
    if not isinstance(nested, (list, dict)):
        return node, False

    new_nested, nested_changed = _drop_dangerous_responses_tools(
        nested,
        path=f"{path}.tools",
        removed=removed,
    )
    if not nested_changed:
        return node, False
    new_node = dict(node)
    if new_nested is None:
        new_node.pop("tools", None)
    else:
        new_node["tools"] = new_nested
    return new_node, True


def _strip_dangerous_responses_tools(body: dict, *, diagnostics: list[dict] | None = None) -> dict:  # pragma: no mutate
    tools = body.get("tools")
    if not isinstance(tools, (list, dict)):
        return body

    removed: list[dict] = []
    sanitized_tools, changed = _drop_dangerous_responses_tools(
        tools,
        path="tools",
        removed=removed,
    )
    if not changed:
        return body

    sanitized = dict(body)
    if sanitized_tools:
        sanitized["tools"] = sanitized_tools
    else:
        sanitized.pop("tools", None)
        sanitized.pop("parallel_tool_calls", None)

    removed_names = {
        str(item.get("name")).strip().lower()
        for item in removed
        if isinstance(item.get("name"), str) and str(item.get("name")).strip()
    }
    if (
        not sanitized.get("tools")
        or _responses_tool_choice_targets_removed_tool(sanitized.get("tool_choice"), set(), removed_names)
    ):
        sanitized.pop("tool_choice", None)

    _append_sanitizer_diagnostic(
        diagnostics,
        {
            "kind": "responses_tools",
            "action": "drop_dangerous_code_execution_tools",
            "tool_names": sorted(removed_names),
            "tools": removed[:20],
            "truncated": len(removed) > 20,
        },
    )
    return sanitized


def responses_tools_have_tool_search(tools) -> bool:
    """Return whether a Responses ``tools`` tree exposes a tool-search tool."""
    if isinstance(tools, list):
        return any(responses_tools_have_tool_search(tool) for tool in tools)
    if not isinstance(tools, dict):
        return False
    if _responses_tool_is_tool_search(tools):
        return True
    nested = tools.get("tools")
    if isinstance(nested, (list, dict)):
        return responses_tools_have_tool_search(nested)
    return False


def _strip_responses_defer_loading_fields(node, *, path: str, removed: list[dict]):
    if isinstance(node, list):
        changed = False
        new_items = []
        for index, item in enumerate(node):
            new_item, item_changed = _strip_responses_defer_loading_fields(
                item,
                path=f"{path}[{index}]",
                removed=removed,
            )
            changed = changed or item_changed
            new_items.append(new_item)
        return (new_items if changed else node), changed

    if not isinstance(node, dict):
        return node, False

    changed = False
    new_node = None
    if "defer_loading" in node:
        new_node = dict(node)
        removed_value = new_node.pop("defer_loading", None)
        removed.append(
            {
                "path": path,
                "name": _responses_tool_name(node),
                "type": node.get("type"),
                "value": removed_value,
            }
        )
        changed = True

    current = new_node if new_node is not None else node
    nested = current.get("tools")
    if isinstance(nested, (list, dict)):
        new_nested, nested_changed = _strip_responses_defer_loading_fields(
            nested,
            path=f"{path}.tools",
            removed=removed,
        )
        if nested_changed:
            if new_node is None:
                new_node = dict(node)
            new_node["tools"] = new_nested
            changed = True

    if changed:
        return (new_node if new_node is not None else node), True
    return node, False


def _strip_invalid_responses_defer_loading(body: dict, *, diagnostics: list[dict] | None = None) -> dict:  # pragma: no mutate
    tools = body.get("tools")
    if not isinstance(tools, (list, dict)):
        return body
    if responses_tools_have_tool_search(tools):
        return body

    removed: list[dict] = []
    sanitized_tools, changed = _strip_responses_defer_loading_fields(
        tools,
        path="tools",
        removed=removed,
    )
    if not changed:
        return body

    sanitized = dict(body)
    sanitized["tools"] = sanitized_tools
    tool_names = [
        str(item.get("name"))
        for item in removed
        if isinstance(item.get("name"), str) and item.get("name")
    ]
    _append_sanitizer_diagnostic(
        diagnostics,
        {
            "kind": "responses_tools",
            "action": "strip_defer_loading",
            "reason": "deferred_tools_require_tool_search",
            "count": len(removed),
            "tool_names": tool_names[:20],
            "tools": removed[:20],
            "truncated": len(removed) > 20,
        },
    )
    print(
        "Responses proxy stripped defer_loading from "
        f"{len(removed)} tool definition(s) because tool_search was absent",
        flush=True,
    )
    return sanitized


def sanitize_responses_tools_for_copilot(body: dict, *, diagnostics: list[dict] | None = None) -> dict:  # pragma: no mutate
    if not isinstance(body, dict):
        return body

    tools = body.get("tools")
    if not isinstance(tools, (list, dict)):
        return body

    body = _strip_dangerous_responses_tools(body, diagnostics=diagnostics)
    tools = body.get("tools")
    if not isinstance(tools, (list, dict)):
        return body

    body = _strip_invalid_responses_defer_loading(body, diagnostics=diagnostics)
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

    _append_sanitizer_diagnostic(
        diagnostics,
        {
            "kind": "responses_tools",
            "action": "drop_unsupported_tool_types",
            "tool_types": sorted(removed_types),
            "tool_names": sorted(removed_names),
        },
    )
    return sanitized


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


def _responses_input_text_to_anthropic_block(item: dict) -> dict | None:  # pragma: no mutate
    """Translate a Responses content part into an Anthropic content block."""
    if not isinstance(item, dict):
        return None
    item_type = str(item.get("type", "")).lower()
    if item_type in {"input_text", "output_text", "text"}:
        text = item.get("text")
        if isinstance(text, str):
            return {"type": "text", "text": text}
        return None
    if item_type in {"input_file", "file"}:
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
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if not isinstance(image_url, str):
            return None
        if image_url.startswith("data:"):
            try:
                header, data = image_url.split(",", 1)
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
    """Split a reasoning signature of the form ``"<encrypted>@<id>"``."""
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


def _responses_tool_to_anthropic(tool: dict) -> dict | None:  # pragma: no mutate
    if not isinstance(tool, dict):
        return None
    tool_type = str(tool.get("type", "")).lower()
    if tool_type and tool_type != "function":
        return None
    function = tool.get("function")
    name = tool.get("name") or (function.get("name") if isinstance(function, dict) else None)
    if not isinstance(name, str):
        raise ValueError("Responses tools must include a name")
    if is_dangerous_code_execution_tool_name(name):
        return None
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
            if is_dangerous_code_execution_tool_name(name):
                return None
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
    """Preserve Responses prompt-cache intent after translating to Messages."""
    if not isinstance(payload, dict):
        return

    def mark(block: dict | None) -> None:
        if not isinstance(block, dict):
            return
        _add_anthropic_cache_control(block)

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


def responses_request_to_anthropic_messages(body: dict) -> dict:  # pragma: no mutate
    """Translate an OpenAI Responses request body into Anthropic Messages shape."""
    if not isinstance(body, dict):
        raise ValueError("Responses body must be a dict")

    raw_input = body.get("input")
    messages: list[dict] = []
    system_extra: list[str] = []

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
                push_blocks("assistant", [_responses_reasoning_item_to_anthropic_block(item)])
                continue

            if item_type in {"compaction", "item_reference", "web_search_call"}:
                continue

            raise ValueError(f"Unsupported Responses input item type: {item_type}")

    payload: dict = {
        "model": body.get("model"),
        "messages": messages,
    }

    system_parts: list[str] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        system_parts.append(instructions)
    system_parts.extend(system_extra)
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

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

    temp = payload.get("temperature")
    if isinstance(temp, (int, float)):
        if temp < 0:
            payload["temperature"] = 0
        elif temp > 1:
            payload["temperature"] = 1

    stop = body.get("stop_sequences")
    if stop is None:
        stop = body.get("stop")
    if stop is not None:
        payload["stop_sequences"] = stop

    if body.get("stream") is not None:
        payload["stream"] = bool(body.get("stream"))

    tools = body.get("tools")
    if isinstance(tools, list):
        translated: list[dict] = []
        for tool in tools:
            converted = _responses_tool_to_anthropic(tool)
            if converted is not None:
                translated.append(converted)
        if translated:
            payload["tools"] = translated

    source_tool_choice = body.get("tool_choice")
    dangerous_tool_choice_removed = _responses_tool_choice_targets_removed_tool(
        source_tool_choice,
        set(),
        _DANGEROUS_CODE_EXECUTION_TOOL_NAMES,
    )
    tool_choice = None if dangerous_tool_choice_removed else _responses_tool_choice_to_anthropic(source_tool_choice)
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    parallel = body.get("parallel_tool_calls")
    if parallel is False and payload.get("tools") and not dangerous_tool_choice_removed:
        if isinstance(tool_choice, dict) and tool_choice.get("type") in ("auto", "any", "tool"):
            payload["tool_choice"] = {**tool_choice, "disable_parallel_tool_use": True}
        elif tool_choice is None:
            payload["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}

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


__all__ = [
    "_DANGEROUS_CODE_EXECUTION_TOOL_NAMES",
    "_add_anthropic_cache_control",
    "_apply_responses_prompt_cache_to_anthropic_messages",
    "_format_custom_tool_call_for_chat",
    "_format_custom_tool_output_for_chat",
    "_last_cacheable_anthropic_content_block",
    "_responses_body_requests_prompt_cache",
    "_responses_function_call_output_to_anthropic_block",
    "_responses_function_call_to_anthropic_block",
    "_responses_input_text_to_anthropic_block",
    "_responses_reasoning_item_to_anthropic_block",
    "_responses_tool_choice_targets_removed_tool",
    "_responses_tool_choice_to_anthropic",
    "_responses_tool_to_anthropic",
    "_split_reasoning_signature",
    "is_dangerous_code_execution_tool_name",
    "responses_request_to_anthropic_messages",
    "responses_tools_have_tool_search",
    "sanitize_responses_body_for_copilot",
    "sanitize_responses_tools_for_copilot",
]
