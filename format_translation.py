"""Request/response format translation between Anthropic Messages API, OpenAI Chat Completions API, and OpenAI Responses API formats."""

import base64
import json
import time
from uuid import uuid4

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from constants import (
    OPENCODE_VERSION, OPENCODE_HEADER_VERSION, OPENCODE_INTEGRATION_ID,
    FORWARDED_REQUEST_HEADERS, FORWARDED_SERVER_REQUEST_ID_HEADERS,
    FAKE_COMPACTION_PREFIX, FAKE_COMPACTION_SUMMARY_LABEL,
    COMPACTION_SUMMARY_PROMPT,
)
from util import _extract_item_text


# ─── Lazy imports to avoid circular dependencies ─────────────────────────────

def _get_initiator_policy():
    import proxy
    return proxy._initiator_policy


def _get_request_session_id():
    from usage_tracking import _request_session_id
    return _request_session_id


# ─── Model resolution ────────────────────────────────────────────────────────

def has_vision_input(value, depth=0, max_depth=10) -> bool:
    """Recursively find type='input_image' anywhere in the input tree."""
    if depth > max_depth or value is None:
        return False
    if isinstance(value, list):
        return any(has_vision_input(i, depth + 1, max_depth) for i in value)
    if not isinstance(value, dict):
        return False
    if str(value.get("type", "")).lower() == "input_image":
        return True
    content = value.get("content")
    if isinstance(content, list):
        return any(has_vision_input(i, depth + 1, max_depth) for i in content)
    return False


def model_requires_anthropic_beta(model_name) -> bool:
    if not isinstance(model_name, str):
        return False
    normalized = model_name.strip().lower()
    return "claude" in normalized or normalized.startswith("anthropic")


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

    if normalized in ("claude-opus-4.6", "claude-sonnet-4.6", "claude-haiku-4.5"):
        return normalized

    if "opus" in normalized:
        return "claude-opus-4.6"
    if "sonnet" in normalized:
        return "claude-sonnet-4.6"
    if "haiku" in normalized:
        return "claude-haiku-4.5"
    return normalized


# ─── Anthropic content helpers ────────────────────────────────────────────────

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
    for item in blocks:
        item_type = str(item.get("type", "")).lower()
        if item_type == "tool_result":
            if buffered_user_blocks:
                chat_messages.append(
                    {
                        "role": "user",
                        "content": _anthropic_blocks_to_chat_content(buffered_user_blocks),
                    }
                )
                buffered_user_blocks = []
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
            chat_messages.append(tool_message)
            continue
        content_item = _anthropic_text_or_image_block_to_chat(item)
        if content_item is None:
            raise ValueError(f"Unsupported Anthropic content block type: {item_type}")
        buffered_user_blocks.append(item)

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
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", "") if isinstance(tool.get("description"), str) else "",
                    "parameters": tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object", "properties": {}},
                },
            }
        )
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

    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        budget_tokens = thinking.get("budget_tokens")
        if isinstance(budget_tokens, int):
            payload["thinking_budget"] = budget_tokens

    if body.get("tools") is not None:
        payload["tools"] = anthropic_tools_to_chat(body.get("tools"))

    mapped_tool_choice = anthropic_tool_choice_to_chat(body.get("tool_choice"))
    if mapped_tool_choice is not None:
        payload["tool_choice"] = mapped_tool_choice

    return payload


# ─── Chat → Anthropic translation ────────────────────────────────────────────

def _chat_usage_to_anthropic(usage) -> dict:
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


def _chat_stop_reason_to_anthropic(value) -> str | None:
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
    usage = _chat_usage_to_anthropic(payload.get("usage") if isinstance(payload, dict) else {})

    return {
        "id": payload.get("id") if isinstance(payload, dict) else f"msg_{uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": (payload.get("model") if isinstance(payload, dict) else None) or fallback_model,
        "content": _chat_message_to_anthropic_content(message if isinstance(message, dict) else {}),
        "stop_reason": _chat_stop_reason_to_anthropic(first_choice.get("finish_reason") if isinstance(first_choice, dict) else None),
        "stop_sequence": None,
        "usage": usage,
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


def _upstream_request_error_status_and_message(exc: httpx.RequestError) -> tuple[int, str]:
    if isinstance(exc, httpx.TimeoutException):
        prefix = "Upstream request timed out"
        status_code = 504
    elif isinstance(exc, httpx.ConnectError):
        prefix = "Upstream connection failed"
        status_code = 502
    else:
        prefix = "Upstream request failed"
        status_code = 502

    detail = str(exc).strip()
    return status_code, f"{prefix}: {detail}" if detail else prefix


def _http_exception_detail_to_message(detail) -> str:
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str) and message:
            return message
    return "Request failed"


# ─── SSE helpers ──────────────────────────────────────────────────────────────

def _sse_encode(event_name: str, payload: dict) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def _parse_sse_block(raw_block: str) -> tuple[str | None, str | None]:
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


async def _iter_sse_messages(byte_iter):
    buffer = ""
    async for chunk in byte_iter:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8")
        else:
            buffer += str(chunk)

        normalized = buffer.replace("\r\n", "\n")
        while "\n\n" in normalized:
            raw_block, normalized = normalized.split("\n\n", 1)
            event_name, data = _parse_sse_block(raw_block)
            if data is not None:
                yield event_name, data
        buffer = normalized

    trailing = buffer.strip()
    if trailing:
        event_name, data = _parse_sse_block(trailing)
        if data is not None:
            yield event_name, data


def _extract_text_from_chat_delta(delta) -> str:
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


def _extract_tool_call_deltas(delta) -> list[dict]:
    if not isinstance(delta, dict):
        return []
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [item for item in tool_calls if isinstance(item, dict)]


# ─── Header building ─────────────────────────────────────────────────────────

def build_copilot_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "User-Agent": f"opencode/{OPENCODE_VERSION}",
        "Openai-Intent": "conversation-edits",
        "Editor-Version": OPENCODE_HEADER_VERSION,
        "Editor-Plugin-Version": OPENCODE_HEADER_VERSION,
        "Copilot-Integration-Id": OPENCODE_INTEGRATION_ID,
    }


def _apply_forwarded_request_headers(headers: dict, request: Request, request_body: dict | None = None):
    session_id = _get_request_session_id()(request, request_body)
    if session_id:
        headers["session_id"] = session_id

    for header_name in FORWARDED_REQUEST_HEADERS:
        header_value = request.headers.get(header_name)
        if header_value:
            headers[header_name] = header_value

    forwarded_server_request_id = None
    for header_name in FORWARDED_SERVER_REQUEST_ID_HEADERS:
        header_value = request.headers.get(header_name)
        if header_value:
            headers[header_name] = header_value
            if forwarded_server_request_id is None:
                forwarded_server_request_id = header_value

    if forwarded_server_request_id is not None:
        headers.setdefault("x-request-id", forwarded_server_request_id)
        headers.setdefault("x-github-request-id", forwarded_server_request_id)


def build_responses_headers_for_request(
    request: Request,
    body: dict,
    api_key: str,
    force_initiator: str | None = None,
    request_id: str | None = None,
) -> dict:
    headers = build_copilot_headers(api_key)
    _apply_forwarded_request_headers(headers, request, body)

    had_input = "input" in body
    effective_input, initiator = _get_initiator_policy().resolve_responses_input(
        body.get("input"),
        body.get("model"),
        force_initiator=force_initiator,
        request_id=request_id,
    )
    if had_input:
        body["input"] = effective_input
    headers["X-Initiator"] = initiator

    if has_vision_input(effective_input):
        headers["Copilot-Vision-Request"] = "true"

    if model_requires_anthropic_beta(body.get("model")):
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    return headers


def build_chat_headers_for_request(
    request: Request,
    messages,
    model_name: str,
    api_key: str,
    request_id: str | None = None,
) -> dict:
    headers = build_copilot_headers(api_key)
    _apply_forwarded_request_headers(headers, request)

    initiator = _get_initiator_policy().resolve_chat_messages(messages, model_name, request_id=request_id)
    headers["X-Initiator"] = initiator

    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and ("image_url" in item or item.get("type") == "image_url"):
                        headers["Copilot-Vision-Request"] = "true"
                        break
                if headers.get("Copilot-Vision-Request") == "true":
                    break

    if model_requires_anthropic_beta(model_name):
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    return headers


def _anthropic_messages_has_vision(messages) -> bool:
    if not isinstance(messages, list):
        return False
    for item in messages:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type", "")).lower()
            if part_type == "image":
                return True
            if part_type == "tool_result" and isinstance(part.get("content"), list):
                for nested in part["content"]:
                    if isinstance(nested, dict) and str(nested.get("type", "")).lower() == "image":
                        return True
    return False


def build_anthropic_headers_for_request(
    request: Request,
    body: dict,
    api_key: str,
    request_id: str | None = None,
) -> dict:
    headers = build_copilot_headers(api_key)
    _apply_forwarded_request_headers(headers, request, body)

    messages = body.get("messages")
    initiator = _get_initiator_policy().resolve_anthropic_messages(messages, body.get("model"), request_id=request_id)
    headers["X-Initiator"] = initiator

    if _anthropic_messages_has_vision(messages):
        headers["Copilot-Vision-Request"] = "true"

    if model_requires_anthropic_beta(body.get("model")):
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    return headers


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
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": f"{FAKE_COMPACTION_SUMMARY_LABEL}\n{summary_text}",
            }
        ],
    }


def input_contains_compaction(input_items) -> bool:
    if not isinstance(input_items, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == "compaction" for item in input_items)


def sanitize_input(input_items):
    """
    Preserve encrypted_content in reasoning items for multi-turn correctness.
    Expand locally synthesized compaction items into a readable summary message.
    Convert other compaction items into reasoning items for GHCP compatibility.
    Strip status=None which GHCP rejects.
    Pass everything else through unchanged.
    """
    if not isinstance(input_items, list):
        return input_items  # plain string — pass through untouched

    result = []
    for item in input_items:
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


def build_fake_compaction_request(body: dict) -> dict:
    request_input = body.get("input")
    if isinstance(request_input, list):
        request_input = sanitize_input(request_input)

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        request_input = [
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Original conversation instructions to preserve:\n{instructions}",
                    }
                ],
            },
            *(request_input if isinstance(request_input, list) else []),
        ]

    return {
        "model": body.get("model"),
        "instructions": COMPACTION_SUMMARY_PROMPT,
        "input": request_input if request_input is not None else [],
        "stream": False,
        "store": False,
        "tools": [],
        "parallel_tool_calls": False,
        "include": [],
        "reasoning": body.get("reasoning"),
        "text": body.get("text"),
    }


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
        text = _extract_item_text(item).strip()
        if text:
            parts.append(text)

    if not parts:
        return None
    return "\n\n".join(parts)


def build_fake_compaction_response(body: dict, summary_text: str, usage=None) -> dict:
    return {
        "id": f"resp_{uuid4().hex}",
        "object": "response.compaction",
        "created_at": int(time.time()),
        "status": "completed",
        "model": body.get("model"),
        "output": [
            {
                "type": "compaction",
                "encrypted_content": encode_fake_compaction(summary_text),
            }
        ],
        "usage": usage
        if isinstance(usage, dict)
        else {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
