"""Build upstream request headers for GitHub Copilot proxy requests."""

import hashlib
import json
import os
import uuid

from fastapi import Request

from constants import (
    OPENCODE_VERSION, OPENCODE_INTEGRATION_ID,
    COPILOT_CLI_INTEGRATION_ID, COPILOT_CLI_USER_AGENT,
    GITHUB_API_VERSION,
    FORWARDED_REQUEST_HEADERS,
)

_STABLE_ID_NAMESPACE = uuid.UUID("8fd22b32-4ce1-4af7-a3d6-7156a8f0ef9d")


def _stable_uuid(value: str) -> str:
    return str(uuid.uuid5(_STABLE_ID_NAMESPACE, value))


_CLIENT_SESSION_ID = os.environ.get("GHCP_COPILOT_CLIENT_SESSION_ID") or str(uuid.uuid4())
_COPILOT_CLI_CLIENT_MACHINE_ID = (
    os.environ.get("GHCP_COPILOT_CLIENT_MACHINE_ID")
    or "0d40238f-cbaa-4b91-a8d0-b46c0a95fdf6"
)
_COPILOT_CLI_EXP_ASSIGNMENT_CONTEXT = (
    os.environ.get("GHCP_COPILOT_CLIENT_EXP_ASSIGNMENT_CONTEXT")
    or (
        "cli_aa_c:1136621;h7c07110:1158940;5bb63a0f:1149772;"
        "e974i579:1132406;h0649438:1154840;be5i9337:1149385;"
        "no-gpt-default:1141892;916bj795:1153107;voting-aa-control:1146525;"
        "ibdi6602:1154657;19ji4148:1155452;voting-aa-treamtment-v2:1157379;"
        "voting-aa-control-v3:1153077;"
    )
)


def _responses_task_affinity_scope(affinity_value) -> str | None:
    """Return the parent task scope used for Responses cache affinity."""
    if not isinstance(affinity_value, str):
        return None
    normalized = affinity_value.strip()
    if not normalized:
        return None
    if len(normalized) >= 36 and normalized[8:9] == "-":
        return normalized[:8]
    return normalized


def _responses_subagent_task_id(parent_scope, subagent, affinity_value=None) -> str | None:
    if not isinstance(parent_scope, str) or not parent_scope.strip():
        return None
    if not isinstance(subagent, str) or not subagent.strip():
        return None
    if isinstance(affinity_value, str) and affinity_value.strip():
        child_scope = affinity_value.strip()
    else:
        child_scope = parent_scope.strip()
    return _copilot_uuid(
        f"responses-subagent-task:{parent_scope.strip()}:{subagent.strip()}:{child_scope}"
    )


_RESPONSES_CURRENT_PARENT_BY_CLIENT_SESSION: dict[str, tuple[str, str]] = {}


def _responses_current_parent(headers: dict) -> tuple[str, str] | None:
    client_session_id = headers.get("x-client-session-id")
    if not isinstance(client_session_id, str) or not client_session_id.strip():
        return None
    return _RESPONSES_CURRENT_PARENT_BY_CLIENT_SESSION.get(client_session_id.strip())


def _remember_responses_current_parent(headers: dict) -> None:
    client_session_id = headers.get("x-client-session-id")
    agent_task_id = headers.get("x-agent-task-id")
    interaction_id = headers.get("x-interaction-id")
    if not (
        isinstance(client_session_id, str)
        and client_session_id.strip()
        and isinstance(agent_task_id, str)
        and agent_task_id.strip()
        and isinstance(interaction_id, str)
        and interaction_id.strip()
    ):
        return
    _RESPONSES_CURRENT_PARENT_BY_CLIENT_SESSION[client_session_id.strip()] = (
        agent_task_id.strip(),
        interaction_id.strip(),
    )


def _apply_responses_current_parent(headers: dict) -> None:
    parent = _responses_current_parent(headers)
    if not parent:
        return
    parent_task_id, interaction_id = parent
    headers["x-agent-task-id"] = parent_task_id
    headers["x-interaction-id"] = interaction_id


def _apply_responses_current_subagent_parent(headers: dict, subagent: str | None) -> None:
    if not isinstance(subagent, str) or not subagent.strip():
        return
    parent = _responses_current_parent(headers)
    if not parent:
        return
    parent_task_id, interaction_id = parent
    current_task_id = headers.get("x-agent-task-id")
    headers["x-parent-agent-id"] = parent_task_id
    headers["x-interaction-id"] = interaction_id
    if not isinstance(current_task_id, str) or not current_task_id.strip() or current_task_id == parent_task_id:
        headers["x-agent-task-id"] = _copilot_uuid(
            f"responses-subagent-task:{parent_task_id}:{subagent.strip()}"
        )


_COPILOT_MACHINE_ID = hashlib.sha256(f"{uuid.getnode():012x}".encode("utf-8")).hexdigest()
_FORWARD_SESSION_HEADER_DEFAULT = True
_VISION_INPUT_INITIAL_DEPTH = 0
_VISION_INPUT_MAX_DEPTH = 10


def has_vision_input(
    value,
    depth: int = _VISION_INPUT_INITIAL_DEPTH,
    max_depth: int = _VISION_INPUT_MAX_DEPTH,
) -> bool:
    """Recursively find type='input_image' anywhere in the input tree."""
    if depth > max_depth or value is None:
        return False
    if isinstance(value, list):
        return any(has_vision_input(i, depth + 1, max_depth) for i in value)
    if not isinstance(value, dict):
        return False
    value_type = value.get("type")
    if isinstance(value_type, str) and value_type.lower() == "input_image":
        return True
    content = value.get("content")
    if isinstance(content, list):
        return any(has_vision_input(i, depth + 1, max_depth) for i in content)
    return False


def _interaction_type_for_initiator(initiator: str) -> str:
    if initiator == "user":
        return "conversation-user"
    return "conversation-agent"


def _interaction_id_for_session(session_id: str | None) -> str:
    if isinstance(session_id, str):
        normalized = session_id.strip()
        if normalized:
            return normalized
    return str(uuid.uuid4())


def _copilot_uuid(content: str) -> str:
    uuid_bytes = bytearray(hashlib.sha256(content.encode("utf-8")).digest()[:16])
    uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | 0x40
    uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(uuid_bytes)))


def _json_stringify_like(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _strip_cache_control(value):
    if not isinstance(value, dict):
        return value
    return {k: v for k, v in value.items() if k != "cache_control"}


def _find_last_user_content(messages) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            content_items = [
                _strip_cache_control(item)
                for item in content
                if not (isinstance(item, dict) and item.get("type") == "tool_result")
            ]
            if content_items:
                return _json_stringify_like(content_items)
    return None


def _responses_request_id_payload_messages(payload):
    if not isinstance(payload, dict):
        return None
    if "messages" in payload:
        return payload.get("messages")
    return payload.get("input")


def _generate_request_id_from_payload(payload, session_id: str | None = None) -> str:
    messages = _responses_request_id_payload_messages(payload)
    if isinstance(messages, str) and messages:
        last_user_content = messages
    else:
        last_user_content = _find_last_user_content(messages)

    if last_user_content:
        return _copilot_uuid(f"{session_id or ''}{_COPILOT_MACHINE_ID}{last_user_content}")

    return str(uuid.uuid4())


def _responses_affinity_value(payload, session_id: str | None = None) -> str | None:
    if isinstance(payload, dict):
        for key in ("prompt_cache_key", "promptCacheKey", "session_id", "sessionId"):
            value = payload.get(key)
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    return normalized
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in ("session_id", "sessionId"):
                value = metadata.get(key)
                if isinstance(value, str):
                    normalized = value.strip()
                    if normalized:
                        return normalized
    if isinstance(session_id, str):
        normalized = session_id.strip()
        if normalized:
            return normalized
    return None


def _responses_body_has_affinity_hint(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ("prompt_cache_key", "promptCacheKey", "session_id", "sessionId"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return True
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("session_id", "sessionId"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return True
    return False


def _responses_copilot_identity_headers(
    payload,
    session_id: str | None = None,
    request_id: str | None = None,
    *,
    stable_affinity: bool = False,
    subagent: str | None = None,
    client_session_id: str | None = None,
) -> dict[str, str]:
    del request_id
    if stable_affinity:
        affinity_value = _responses_affinity_value(payload, session_id)
        if affinity_value:
            parent_scope = _responses_task_affinity_scope(affinity_value) or affinity_value
            interaction_id = _copilot_uuid(f"responses-interaction:{parent_scope}")
            parent_task_id = _copilot_uuid(f"responses-task:{parent_scope}")
            headers = {
                "x-agent-task-id": parent_task_id,
                "x-interaction-id": interaction_id,
            }
            normalized_subagent = subagent.strip() if isinstance(subagent, str) and subagent.strip() else None
            if normalized_subagent:
                child_task_id = _responses_subagent_task_id(
                    parent_scope,
                    normalized_subagent,
                    affinity_value,
                )
                if child_task_id:
                    headers["x-agent-task-id"] = child_task_id
                    headers["x-parent-agent-id"] = parent_task_id
            return headers

    normalized_session_id = session_id.strip() if isinstance(session_id, str) and session_id.strip() else None
    normalized_client_session_id = (
        client_session_id.strip()
        if isinstance(client_session_id, str) and client_session_id.strip()
        else None
    )
    interaction_session_id = normalized_session_id or normalized_client_session_id
    is_anthropic_messages_payload = isinstance(payload, dict) and "messages" in payload
    root_session_id = None
    if is_anthropic_messages_payload and interaction_session_id:
        root_session_id = _copilot_uuid(interaction_session_id)

    agent_task_id = _generate_request_id_from_payload(
        payload,
        session_id=root_session_id if is_anthropic_messages_payload else None,
    )
    headers = {
        "x-agent-task-id": agent_task_id,
    }
    if is_anthropic_messages_payload:
        if root_session_id:
            headers["x-interaction-id"] = root_session_id
    else:
        if interaction_session_id:
            headers["x-interaction-id"] = _copilot_uuid(
                f"responses-interaction:{interaction_session_id}"
            )
        else:
            headers["x-interaction-id"] = _copilot_uuid(agent_task_id)
    return headers


def _messages_affinity_headers(initiator: str, interaction_id: str | None, request_id: str) -> dict[str, str]:
    """Return native Messages affinity headers for a request.

    Copilot's native Anthropic /v1/messages endpoint keys prompt-cache lineage
    off the body cache breakpoints and request affinity headers. Keep the
    conversation interaction stable when Claude Code supplies a real session
    id, but leave the agent task request-scoped. Making both values stable
    over-affinitizes separate internal requests and can make cache reads look
    like one giant task-wide prefix.
    """
    del initiator

    if isinstance(interaction_id, str):
        normalized = interaction_id.strip()
    else:
        normalized = ""

    if normalized:
        return {"x-interaction-id": normalized}

    return {"x-interaction-id": request_id}


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


def build_responses_copilot_headers(api_key: str) -> dict:
    headers = build_copilot_headers(api_key)
    headers["User-Agent"] = COPILOT_CLI_USER_AGENT
    headers["Copilot-Integration-Id"] = COPILOT_CLI_INTEGRATION_ID
    headers.update(
        {
            "x-stainless-retry-count": "0",
            "x-stainless-lang": "js",
            "x-stainless-package-version": "5.20.1",
            "x-stainless-os": "Windows",
            "x-stainless-arch": "x64",
            "x-stainless-runtime": "node",
            "x-stainless-runtime-version": "v25.6.0",
            "x-client-machine-id": _COPILOT_CLI_CLIENT_MACHINE_ID,
            "x-copilot-client-exp-assignment-context": _COPILOT_CLI_EXP_ASSIGNMENT_CONTEXT,
            "accept-language": "*",
            "sec-fetch-mode": "cors",
        }
    )
    return headers


def _apply_forwarded_request_headers(
    headers: dict,
    request: Request,
    request_body: dict | None = None,
    *,
    session_id_resolver=None,
    forward_session_header: bool = _FORWARD_SESSION_HEADER_DEFAULT,
    synthesize_client_request_id: bool = True,
):
    session_id = session_id_resolver(request, request_body) if session_id_resolver else None
    if session_id and forward_session_header:
        headers["session_id"] = session_id

    for header_name in FORWARDED_REQUEST_HEADERS:
        header_value = request.headers.get(header_name)
        if header_value:
            headers[header_name] = header_value

    if synthesize_client_request_id and "x-client-request-id" not in headers and isinstance(session_id, str):
        normalized_session_id = session_id.strip()
        if normalized_session_id:
            headers["x-client-request-id"] = normalized_session_id

    return session_id


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
    affinity_body: dict | None = None,
    stable_user_affinity: bool = False,
) -> dict:
    headers = build_responses_copilot_headers(api_key)
    identity_source = affinity_body if isinstance(affinity_body, dict) else body
    session_id = _apply_forwarded_request_headers(
        headers,
        request,
        identity_source,
        session_id_resolver=session_id_resolver,
        forward_session_header=False,
        synthesize_client_request_id=False,
    )
    headers.pop("x-client-request-id", None)
    headers.pop("x-request-id", None)
    headers.pop("x-github-request-id", None)
    inbound_subagent = request.headers.get("x-openai-subagent") if hasattr(request, "headers") else None
    inbound_subagent = (
        inbound_subagent.strip()
        if isinstance(inbound_subagent, str) and inbound_subagent.strip()
        else None
    )
    headers.pop("x-openai-subagent", None)

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
    if inbound_subagent:
        initiator = "agent"
    headers["X-Initiator"] = initiator
    headers["x-interaction-type"] = (
        "conversation-subagent"
        if inbound_subagent
        else _interaction_type_for_initiator(initiator)
    )
    stable_affinity = stable_user_affinity or _responses_body_has_affinity_hint(identity_source)
    headers.update(
        _responses_copilot_identity_headers(
            identity_source,
            session_id,
            request_id=request_id,
            stable_affinity=stable_affinity,
            subagent=inbound_subagent,
            client_session_id=headers.get("x-client-session-id"),
        )
    )
    if inbound_subagent:
        _apply_responses_current_subagent_parent(headers, inbound_subagent)
    else:
        _remember_responses_current_parent(headers)

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
    session_id = _apply_forwarded_request_headers(headers, request, session_id_resolver=session_id_resolver)

    initiator = initiator_policy.resolve_chat_messages(
        messages,
        model_name,
        subagent=request.headers.get("x-openai-subagent"),
        request_id=request_id,
        verdict_sink=verdict_sink,
    )
    headers["X-Initiator"] = initiator
    headers["x-interaction-type"] = _interaction_type_for_initiator(initiator)
    headers["x-interaction-id"] = _interaction_id_for_session(session_id)
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
    session_id = _apply_forwarded_request_headers(headers, request, body, session_id_resolver=session_id_resolver)

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
    headers["x-interaction-id"] = _interaction_id_for_session(session_id)
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
        del headers[key]

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

    headers.update(_messages_affinity_headers(initiator, interaction_id, request_id))

    return headers
