"""Build upstream request headers for GitHub Copilot proxy requests."""

from fastapi import Request

from constants import (
    OPENCODE_VERSION, OPENCODE_HEADER_VERSION, OPENCODE_INTEGRATION_ID,
    FORWARDED_REQUEST_HEADERS, FORWARDED_SERVER_REQUEST_ID_HEADERS,
)


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


def _apply_forwarded_request_headers(headers: dict, request: Request, request_body: dict | None = None, *, session_id_resolver=None):
    session_id = session_id_resolver(request, request_body) if session_id_resolver else None
    if session_id:
        headers["session_id"] = session_id

    for header_name in FORWARDED_REQUEST_HEADERS:
        header_value = request.headers.get(header_name)
        if header_value:
            headers[header_name] = header_value

    if "x-client-request-id" not in headers and isinstance(session_id, str):
        normalized_session_id = session_id.strip()
        if normalized_session_id:
            headers["x-client-request-id"] = normalized_session_id

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

    return session_id


def _normalize_responses_prompt_cache_key(body: dict, session_id: str | None) -> None:
    if not isinstance(body, dict):
        return

    prompt_cache_key = None
    for key in ("prompt_cache_key", "promptCacheKey"):
        value = body.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                prompt_cache_key = normalized
                break

    if prompt_cache_key is None and isinstance(session_id, str):
        normalized_session_id = session_id.strip()
        if normalized_session_id:
            prompt_cache_key = normalized_session_id

    body.pop("promptCacheKey", None)
    if prompt_cache_key is not None:
        body["prompt_cache_key"] = prompt_cache_key


def build_responses_headers_for_request(
    request: Request,
    body: dict,
    api_key: str,
    force_initiator: str | None = None,
    request_id: str | None = None,
    *,
    initiator_policy=None,
    session_id_resolver=None,
) -> dict:
    headers = build_copilot_headers(api_key)
    session_id = _apply_forwarded_request_headers(headers, request, body, session_id_resolver=session_id_resolver)
    _normalize_responses_prompt_cache_key(body, session_id)

    had_input = "input" in body
    effective_input, initiator = initiator_policy.resolve_responses_input(
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
    *,
    initiator_policy=None,
    session_id_resolver=None,
) -> dict:
    headers = build_copilot_headers(api_key)
    _apply_forwarded_request_headers(headers, request, session_id_resolver=session_id_resolver)

    initiator = initiator_policy.resolve_chat_messages(messages, model_name, request_id=request_id)
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
    *,
    initiator_policy=None,
    session_id_resolver=None,
) -> dict:
    headers = build_copilot_headers(api_key)
    _apply_forwarded_request_headers(headers, request, body, session_id_resolver=session_id_resolver)

    messages = body.get("messages")
    initiator = initiator_policy.resolve_anthropic_messages(messages, body.get("model"), request_id=request_id)
    headers["X-Initiator"] = initiator

    if _anthropic_messages_has_vision(messages):
        headers["Copilot-Vision-Request"] = "true"

    if model_requires_anthropic_beta(body.get("model")):
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    return headers
