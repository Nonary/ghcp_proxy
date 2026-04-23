"""Build upstream request headers for GitHub Copilot proxy requests."""

import uuid

from fastapi import Request

from constants import (
    OPENCODE_VERSION, OPENCODE_INTEGRATION_ID, GITHUB_API_VERSION,
    FORWARDED_REQUEST_HEADERS, FORWARDED_SERVER_REQUEST_ID_HEADERS,
)

_CLIENT_SESSION_ID = str(uuid.uuid4())


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


def _interaction_type_for_initiator(initiator: str) -> str:
    if initiator == "user":
        return "conversation-user"
    return "conversation-agent"


def build_copilot_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "accept": "application/json",
        "User-Agent": f"opencode/{OPENCODE_VERSION}",
        "Openai-Intent": "conversation-agent",
        "Copilot-Integration-Id": OPENCODE_INTEGRATION_ID,
        "x-github-api-version": GITHUB_API_VERSION,
        "x-client-session-id": _CLIENT_SESSION_ID,
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
    verdict_sink: dict | None = None,
) -> dict:
    headers = build_copilot_headers(api_key)
    session_id = _apply_forwarded_request_headers(headers, request, body, session_id_resolver=session_id_resolver)
    _normalize_responses_prompt_cache_key(body, session_id)

    had_input = "input" in body
    effective_input, initiator = initiator_policy.resolve_responses_input(
        body.get("input"),
        body.get("model"),
        subagent=request.headers.get("x-openai-subagent"),
        force_initiator=force_initiator,
        request_id=request_id,
        verdict_sink=verdict_sink,
    )
    if had_input:
        body["input"] = effective_input
    headers["X-Initiator"] = initiator
    headers["x-interaction-type"] = _interaction_type_for_initiator(initiator)
    headers["x-interaction-id"] = str(uuid.uuid4())
    headers["x-agent-task-id"] = str(uuid.uuid4())

    if has_vision_input(effective_input):
        headers["Copilot-Vision-Request"] = "true"

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
    verdict_sink: dict | None = None,
) -> dict:
    headers = build_copilot_headers(api_key)
    _apply_forwarded_request_headers(headers, request, session_id_resolver=session_id_resolver)

    initiator = initiator_policy.resolve_chat_messages(
        messages,
        model_name,
        subagent=request.headers.get("x-openai-subagent"),
        request_id=request_id,
        verdict_sink=verdict_sink,
    )
    headers["X-Initiator"] = initiator
    headers["x-interaction-type"] = _interaction_type_for_initiator(initiator)
    headers["x-interaction-id"] = str(uuid.uuid4())
    headers["x-agent-task-id"] = str(uuid.uuid4())

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
    verdict_sink: dict | None = None,
) -> dict:
    headers = build_copilot_headers(api_key)
    _apply_forwarded_request_headers(headers, request, body, session_id_resolver=session_id_resolver)

    messages = body.get("messages")
    initiator = initiator_policy.resolve_anthropic_messages(
        messages,
        body.get("model"),
        system=body.get("system"),
        subagent=request.headers.get("x-openai-subagent"),
        request_id=request_id,
        verdict_sink=verdict_sink,
    )
    headers["X-Initiator"] = initiator
    headers["x-interaction-type"] = _interaction_type_for_initiator(initiator)
    headers["x-interaction-id"] = str(uuid.uuid4())
    headers["x-agent-task-id"] = str(uuid.uuid4())

    if _anthropic_messages_has_vision(messages):
        headers["Copilot-Vision-Request"] = "true"

    return headers


# ---------------------------------------------------------------------------
# Anthropic /v1/messages native passthrough helpers
# ---------------------------------------------------------------------------

ALLOWED_ANTHROPIC_BETAS = frozenset({
    "interleaved-thinking-2025-05-14",
    "context-management-2025-06-27",
    "advanced-tool-use-2025-11-20",
})

ADVANCED_TOOL_USE_MODELS_PREFIXES = (
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "claude-opus-4.5",
    "claude-opus-4.6",
)

CLAUDE_AGENT_USER_AGENT = (
    "vscode_claude_code/2.1.98 (external, sdk-ts, agent-sdk/0.2.98)"
)


def _normalize_model_for_betas(model: str | None) -> str:
    norm = (model or "").strip().lower()
    if norm.startswith("anthropic/"):
        norm = norm.split("/", 1)[1]
    return norm


def derive_anthropic_betas(
    *,
    client_betas: list[str] | None,
    body: dict,
    model: str,
) -> list[str]:
    """Filter inbound ``anthropic-beta`` values against the allowlist and
    auto-inject the betas this proxy knows how to use."""

    seen: set[str] = set()
    out: list[str] = []

    def _add(name: str) -> None:
        if name in ALLOWED_ANTHROPIC_BETAS and name not in seen:
            seen.add(name)
            out.append(name)

    if isinstance(client_betas, list):
        for entry in client_betas:
            if not isinstance(entry, str):
                continue
            for piece in entry.split(","):
                token = piece.strip()
                if token:
                    _add(token)

    # Auto-inject interleaved-thinking when explicit budget_tokens are used
    # (and the request is not already adaptive).
    thinking = body.get("thinking") if isinstance(body, dict) else None
    if isinstance(thinking, dict):
        t_type = thinking.get("type")
        budget = thinking.get("budget_tokens")
        if t_type == "enabled" and isinstance(budget, int) and budget > 0:
            _add("interleaved-thinking-2025-05-14")

    norm_model = _normalize_model_for_betas(model)
    if any(norm_model.startswith(p) for p in ADVANCED_TOOL_USE_MODELS_PREFIXES):
        _add("advanced-tool-use-2025-11-20")

    return out


def build_anthropic_messages_passthrough_headers(
    *,
    request_id: str,
    initiator: str,
    interaction_id: str | None,
    interaction_type: str | None,  # noqa: ARG001 - accepted for API symmetry
    anthropic_betas: list[str],
    base_headers: dict,
) -> dict:
    """Produce the upstream header set for a native Copilot Messages proxy
    request. Mirrors copilot-api's ``prepareMessageProxyHeaders``."""

    headers: dict = dict(base_headers) if isinstance(base_headers, dict) else {}

    # Drop any header (regardless of casing) that this function is about to
    # set itself. httpx merges duplicate-cased keys into a comma-joined value
    # which corrupts user-agent / openai-intent / interaction headers and
    # triggers Copilot validation errors.
    _drop_keys = (
        "copilot-integration-id",
        "user-agent",
        "openai-intent",
        "x-interaction-type",
        "x-interaction-id",
        "x-agent-task-id",
        "x-request-id",
        "x-initiator",
        "anthropic-version",
        "anthropic-beta",
    )
    for key in [k for k in list(headers.keys()) if k.lower() in _drop_keys]:
        headers.pop(key, None)

    headers["x-agent-task-id"] = request_id
    headers["x-request-id"] = request_id
    headers["x-interaction-type"] = "messages-proxy"
    headers["openai-intent"] = "messages-proxy"
    headers["user-agent"] = CLAUDE_AGENT_USER_AGENT
    headers["anthropic-version"] = "2023-06-01"

    if isinstance(anthropic_betas, list) and anthropic_betas:
        headers["anthropic-beta"] = ",".join(anthropic_betas)

    if isinstance(initiator, str) and initiator:
        headers["x-initiator"] = initiator

    if isinstance(interaction_id, str) and interaction_id:
        headers["x-interaction-id"] = interaction_id

    return headers
