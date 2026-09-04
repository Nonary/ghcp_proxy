"""OpenAI Responses compatibility layer backed by the official Copilot SDK.

The SDK still runs an agent loop, but declaration-only tools deliberately
suspend that loop and expose ``external_tool.requested``.  Encoding the SDK
session/request identity in the OpenAI ``call_id`` lets Codex execute the tool
and resume the same Copilot turn in a later HTTP request.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import certifi

import auth
import excel_upstream
import format_translation
import util
from constants import TOKEN_DIR

try:
    from copilot import CopilotClient, Tool
    from copilot.rpc import ExternalToolTextResultForLlm, HandlePendingToolCallRequest
    from copilot.session import PermissionHandler
    from copilot.session_events import (
        AssistantIntentData,
        AssistantMessageData,
        AssistantMessageDeltaData,
        AssistantReasoningData,
        AssistantReasoningDeltaData,
        AssistantUsageData,
        ExternalToolRequestedData,
        SessionCompactionCompleteData,
        SessionCompactionStartData,
        SessionErrorData,
        SessionIdleData,
        SessionShutdownData,
        SubagentCompletedData,
        SubagentFailedData,
        SubagentSelectedData,
        SubagentStartedData,
    )
except ImportError as exc:  # pragma: no cover - exercised only on broken installs
    CopilotClient = None  # type: ignore[assignment,misc]
    Tool = None  # type: ignore[assignment,misc]
    ExternalToolTextResultForLlm = None  # type: ignore[assignment,misc]
    _SDK_IMPORT_ERROR: Exception | None = exc
    AssistantIntentData = AssistantMessageData = None  # type: ignore[assignment,misc]
    AssistantReasoningData = AssistantReasoningDeltaData = None  # type: ignore[assignment,misc]
    AssistantUsageData = ExternalToolRequestedData = None  # type: ignore[assignment,misc]
    SessionCompactionCompleteData = SessionCompactionStartData = None  # type: ignore[assignment,misc]
    SessionErrorData = SessionIdleData = SessionShutdownData = None  # type: ignore[assignment,misc]
    SubagentCompletedData = SubagentFailedData = None  # type: ignore[assignment,misc]
    SubagentSelectedData = SubagentStartedData = None  # type: ignore[assignment,misc]
else:
    _SDK_IMPORT_ERROR = None


RESPONSES_UPSTREAM_ENV = "GHCP_RESPONSES_UPSTREAM"
SDK_UPSTREAM = "sdk"
REST_UPSTREAM = "rest"
_CALL_ID_PREFIX = "ghcpsdk_"
_VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_TURN_TIMEOUT_SECONDS = float(os.environ.get("GHCP_UPSTREAM_TIMEOUT_SECONDS", "1800") or 1800)
_KEEPALIVE_INTERVAL_SECONDS = 15.0
_PARALLEL_TOOL_SETTLE_SECONDS = 0.5
_SDK_STATE_DIR = os.path.join(TOKEN_DIR, "copilot-sdk")
_SESSION_LEDGER_NAME = "proxy-sessions.json"
_SESSION_ALIASES_NAME = "proxy-session-aliases.json"
_SESSION_USAGE_NAME = "proxy-session-usage.json"
_SESSION_LEDGER_LOCK = threading.Lock()
_ABANDONED_SESSION_SECONDS = 24 * 60 * 60

_client: Any = None
_client_token: str | None = None
_client_lock: asyncio.Lock | None = None
_client_pruned = False


def _read_session_ledger_unlocked() -> dict[str, float]:
    try:
        with open(_state_path(_SESSION_LEDGER_NAME), encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        key: float(value)
        for key, value in payload.items()
        if isinstance(key, str) and isinstance(value, (int, float))
    }


def _write_session_ledger_unlocked(ledger: dict[str, float]) -> None:
    _write_json_state(_state_path(_SESSION_LEDGER_NAME), ledger, "proxy-sessions-")


def _remember_session(session_id: str) -> None:
    with _SESSION_LEDGER_LOCK:
        ledger = _read_session_ledger_unlocked()
        ledger[session_id] = time.time()
        _write_session_ledger_unlocked(ledger)


def _forget_session(session_id: str) -> None:
    with _SESSION_LEDGER_LOCK:
        ledger = _read_session_ledger_unlocked()
        if ledger.pop(session_id, None) is not None:
            _write_session_ledger_unlocked(ledger)


def _owns_session(session_id: str) -> bool:
    with _SESSION_LEDGER_LOCK:
        return session_id in _read_session_ledger_unlocked()


def _state_path(name: str) -> str:
    """Resolve a state file under the current state dir.

    Deliberately computed per call rather than at import: tests redirect
    _SDK_STATE_DIR, and a module-level constant would keep writing to the
    real one.
    """
    return os.path.join(_SDK_STATE_DIR, name)


def _read_json_state(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _write_json_state(path: str, payload: Any, prefix: str) -> None:
    os.makedirs(_SDK_STATE_DIR, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=_SDK_STATE_DIR)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _read_session_aliases_unlocked() -> dict[str, dict]:
    """Map each caller conversation id to its SDK session and consumed prefix.

    Older files stored a bare session id string; those load with an empty
    watermark, which simply forces one full replay before resuming kicks in.
    """
    payload = _read_json_state(_state_path(_SESSION_ALIASES_NAME))
    if not isinstance(payload, dict):
        return {}
    aliases: dict[str, dict] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            continue
        if isinstance(value, str) and value:
            aliases[key] = {"session_id": value, "segments": []}
            continue
        if not isinstance(value, dict):
            continue
        session_id = value.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            continue
        segments = value.get("segments")
        aliases[key] = {
            "session_id": session_id,
            "segments": [s for s in segments if isinstance(s, str)] if isinstance(segments, list) else [],
        }
    return aliases


def _write_session_aliases_unlocked(aliases: dict[str, dict]) -> None:
    _write_json_state(_state_path(_SESSION_ALIASES_NAME), aliases, "proxy-session-aliases-")


def _session_alias(body: dict) -> str | None:
    """Return the caller's stable conversation id, if one was supplied.

    This keys the SDK session a follow-up turn resumes, so it has to be
    per-thread rather than per-conversation: Codex hands the same
    ``session_id`` to a root thread and to every subagent it spawns, and
    resuming a subagent into its parent's session would splice the two
    histories together.  ``thread_id`` distinguishes them.
    """
    value = body.get("session_id") or body.get("sessionId")
    if isinstance(value, str) and value.strip():
        return value.strip()
    # Codex keeps this in client metadata.  Importing lazily avoids making the
    # SDK adapter depend on the rest of the request routing path at import time.
    try:
        import codex_agent_compat
        value = codex_agent_compat.codex_thread_id(body) or codex_agent_compat.codex_session_id(body)
    except Exception:
        value = None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _remember_session_alias(alias: str, session_id: str, fingerprints: list[str]) -> None:
    with _SESSION_LEDGER_LOCK:
        aliases = _read_session_aliases_unlocked()
        aliases[alias] = {"session_id": session_id, "segments": list(fingerprints)}
        _write_session_aliases_unlocked(aliases)


def _session_for_alias(alias: str) -> tuple[str, list[str]] | None:
    """Return the live SDK session for a caller conversation and its watermark."""
    with _SESSION_LEDGER_LOCK:
        entry = _read_session_aliases_unlocked().get(alias)
        if not entry:
            return None
        session_id = entry["session_id"]
        if session_id not in _read_session_ledger_unlocked():
            return None
        return session_id, entry["segments"]


def _forget_session_aliases(session_ids: set[str]) -> None:
    """Drop alias entries pointing at sessions that no longer exist."""
    if not session_ids:
        return
    with _SESSION_LEDGER_LOCK:
        aliases = _read_session_aliases_unlocked()
        remaining = {
            alias: entry
            for alias, entry in aliases.items()
            if entry["session_id"] not in session_ids
        }
        if len(remaining) != len(aliases):
            _write_session_aliases_unlocked(remaining)


# Alias watermarks are only durable once the turn they describe succeeded;
# until then they sit here keyed by SDK session id.
_pending_alias_watermark: dict[str, tuple[str, list[str]]] = {}


def _commit_alias_watermark(session_id: str, *, success: bool) -> None:
    pending = _pending_alias_watermark.pop(session_id, None)
    if pending is None or not success:
        return
    alias, fingerprints = pending
    try:
        _remember_session_alias(alias, session_id, fingerprints)
    except OSError:
        pass


async def _prune_abandoned_sessions(client: Any) -> None:
    cutoff = time.time() - _ABANDONED_SESSION_SECONDS
    with _SESSION_LEDGER_LOCK:
        ledger = _read_session_ledger_unlocked()
    pruned: set[str] = set()
    for session_id, updated_at in ledger.items():
        if updated_at >= cutoff:
            continue
        try:
            await client.delete_session(session_id)
        except Exception:
            pass
        _forget_session(session_id)
        pruned.add(session_id)
    _forget_session_aliases(pruned)
    _forget_shutdown_baselines(pruned)


async def _delete_owned_session(session_id: str) -> None:
    client = _client
    if client is None or not _owns_session(session_id):
        return
    try:
        await client.delete_session(session_id)
    except Exception:
        return
    _forget_session(session_id)
    _forget_session_aliases({session_id})
    _forget_shutdown_baselines({session_id})


def responses_upstream() -> str:
    """Return the selected Codex Responses upstream (SDK by default in v2)."""
    value = os.getenv(RESPONSES_UPSTREAM_ENV, SDK_UPSTREAM).strip().lower()
    return REST_UPSTREAM if value == REST_UPSTREAM else SDK_UPSTREAM


def enabled() -> bool:
    return responses_upstream() == SDK_UPSTREAM


def is_compaction_request(body: dict | None) -> bool:
    """Recognize Codex's in-band manual compaction turn.

    Recent Codex clients post this to /responses rather than
    /responses/compact and terminate the input with a compaction_trigger item.
    """
    if not isinstance(body, dict):
        return False
    input_items = body.get("input")
    return bool(
        isinstance(input_items, list)
        and input_items
        and isinstance(input_items[-1], dict)
        and input_items[-1].get("type") == "compaction_trigger"
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _encode_call_id(
    session_id: str,
    request_id: str,
    *,
    tool_name: str,
    tool_type: str,
) -> str:
    raw = json.dumps(
        {"s": session_id, "r": request_id, "n": tool_name, "t": tool_type},
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return _CALL_ID_PREFIX + encoded


def _decode_call_id(call_id: Any) -> dict[str, str] | None:
    if not isinstance(call_id, str) or not call_id.startswith(_CALL_ID_PREFIX):
        return None
    encoded = call_id[len(_CALL_ID_PREFIX) :]
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    # urlsafe_b64decode raises binascii.Error (not ValueError) for malformed
    # client-supplied call IDs.  A bad continuation is a 400, not an internal
    # error from the proxy.
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if not all(isinstance(value.get(key), str) and value[key] for key in ("s", "r", "n", "t")):
        return None
    if value["t"] not in {"function", "custom"}:
        return None
    return value


def _sanitize_tool_name(name: str, used: set[str]) -> str:
    candidate = name if _VALID_TOOL_NAME.fullmatch(name) else re.sub(r"[^A-Za-z0-9_-]", "_", name)
    candidate = candidate or "tool"
    base = candidate
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


@dataclass(frozen=True)
class ToolMetadata:
    original_name: str
    tool_type: str


@dataclass
class ToolRegistration:
    tools: list[Any] = field(default_factory=list)
    names: dict[str, ToolMetadata] = field(default_factory=dict)


def build_tool_registration(body: dict) -> ToolRegistration:
    registration = ToolRegistration()
    if body.get("tool_choice") == "none" or Tool is None:
        return registration
    used: set[str] = set()
    for spec in body.get("tools") or []:
        if not isinstance(spec, dict):
            continue
        tool_type = spec.get("type")
        if tool_type not in {"function", "custom"}:
            continue
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            continue
        safe_name = _sanitize_tool_name(name, used)
        if tool_type == "custom":
            # ``apply_patch`` (and potentially other names from the CLI's
            # built-in catalog) is a free-form runtime tool.  Registering a
            # client-owned custom tool under that name makes the SDK resume
            # it as a built-in custom call.  Gemini then rejects the pending
            # result because the SDK's continuation RPC has no tool-name
            # field.  A private runtime name keeps the call on the ordinary
            # external-function path; ``registration.names`` still maps it
            # back to the OpenAI custom-tool name for the caller.
            safe_name = _sanitize_tool_name(f"ghcp_custom_{safe_name}", used)
        description = spec.get("description")
        if not isinstance(description, str):
            description = ""
        if tool_type == "custom":
            description = (
                description.rstrip()
                + "\nReturn this custom tool's complete raw input in the JSON `input` field."
            ).strip()
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            }
        else:
            parameters = spec.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
        registration.names[safe_name] = ToolMetadata(name, tool_type)
        registration.tools.append(
            Tool(
                name=safe_name,
                description=description,
                parameters=parameters,
                overrides_built_in_tool=tool_type == "function",
                skip_permission=True,
                defer="never",
            )
        )
    return registration


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else json.dumps(content, ensure_ascii=False)
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
            elif part.get("type") in {"input_image", "image_url"}:
                parts.append("[image supplied by client]")
    return "\n".join(part for part in parts if part)


# Segment kinds.  ``_SEGMENT_USER`` marks content that originates with the
# caller; ``_SEGMENT_ECHO`` marks a transcript echo of work the SDK session
# performed itself, which a resumed session already holds and must not be
# re-sent; ``_SEGMENT_SUMMARY`` is the client-side compaction summary, which
# the session wrote itself; ``_SEGMENT_SYNTHETIC`` is text the proxy adds to
# a single turn (the post-compaction nudge).  The caller's next transcript
# never contains synthetic text, so it stays out of the resume watermark --
# fingerprinting it used to force a fresh session, and a replay of the whole
# post-compaction window, on the first real user message after a compaction.
_SEGMENT_USER = "user"
_SEGMENT_ECHO = "echo"
_SEGMENT_SUMMARY = "summary"
_SEGMENT_SYNTHETIC = "synthetic"

_CONTINUE_AFTER_COMPACTION_PROMPT = (
    "User: Please continue and complete your response to the user's request based on "
    "the work already completed in the summary above. Do not repeat investigations or "
    "tool calls already documented in the summary."
)


def _render_input_segments(value: Any) -> list[tuple[str, str]]:
    """Render a Responses transcript into ordered ``(kind, text)`` segments."""
    if isinstance(value, str):
        return [(_SEGMENT_USER, value)] if value else []
    if not isinstance(value, list):
        text = _text_from_content(value)
        return [(_SEGMENT_USER, text)] if text else []
    rendered: list[tuple[str, str]] = []
    # Truncate to the latest compaction window so pre-compaction history is not replayed.
    window_items = format_translation._latest_compaction_window(value)
    saw_compaction = False
    saw_user_after_compaction = False
    for item in window_items:
        if isinstance(item, str):
            rendered.append((_SEGMENT_USER, f"User: {item}"))
            if saw_compaction:
                saw_user_after_compaction = True
            continue
        if not isinstance(item, dict):
            continue
        if format_translation._is_subagent_notification_message(item):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            continue
        if item_type == "compaction":
            saw_compaction = True
            saw_user_after_compaction = False
            encrypted_content = item.get("encrypted_content")
            summary_text = None
            if isinstance(encrypted_content, str):
                summary_text = format_translation.decode_fake_compaction(encrypted_content)
            if not summary_text:
                summary_text = item.get("output_text") or item.get("summary") or item.get("text")
            if summary_text:
                rendered.append((
                    _SEGMENT_SUMMARY,
                    f"Assistant: {format_translation.FAKE_COMPACTION_SUMMARY_LABEL}\n{summary_text}",
                ))
            continue
        if item_type in {"function_call", "custom_tool_call"}:
            payload = item.get("arguments") if item_type == "function_call" else item.get("input")
            rendered.append((
                _SEGMENT_ECHO,
                f"Assistant tool call {item.get('name', '')}: {_text_from_content(payload)}",
            ))
            continue
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            rendered.append((_SEGMENT_ECHO, f"Tool result: {_text_from_content(item.get('output'))}"))
            continue
        role = item.get("role") or ("assistant" if item_type == "message" else "user")
        text = _text_from_content(item.get("content"))
        if text:
            if text.startswith(format_translation.FAKE_COMPACTION_SUMMARY_LABEL):
                saw_compaction = True
                saw_user_after_compaction = False
                rendered.append((
                    _SEGMENT_SUMMARY,
                    f"Assistant: {text}",
                ))
            else:
                is_assistant = str(role).lower() == "assistant"
                kind = _SEGMENT_ECHO if is_assistant else _SEGMENT_USER
                if not is_assistant and saw_compaction:
                    if not any(marker in text for marker in (
                        "<environment_context>",
                        "<permissions instructions>",
                        "<skills_instructions>",
                        "<instructions>",
                        "# AGENTS.md",
                    )):
                        saw_user_after_compaction = True
                rendered.append((kind, f"{str(role).capitalize()}: {text}"))

    if saw_compaction and not saw_user_after_compaction:
        rendered.append((_SEGMENT_SYNTHETIC, _CONTINUE_AFTER_COMPACTION_PROMPT))

    return rendered


def input_to_prompt(value: Any) -> str:
    """Render a full Responses transcript into one SDK user message."""
    return "\n\n".join(text for _, text in _render_input_segments(value))


def _durable_segments(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The segments the caller's later transcripts will replay verbatim."""
    return [segment for segment in segments if segment[0] != _SEGMENT_SYNTHETIC]


def _segment_fingerprint(kind: str, text: str) -> str:
    return hashlib.sha1(f"{kind}\x00{text}".encode("utf-8")).hexdigest()


def _segment_fingerprints(segments: list[tuple[str, str]]) -> list[str]:
    return [_segment_fingerprint(kind, text) for kind, text in _durable_segments(segments)]


# The last thing a session consumes before the caller compacts is the proxy's
# summary request, so this fingerprint at the end of a watermark identifies a
# session that wrote the compaction summary now sitting in the transcript.
_SUMMARY_REQUEST_FINGERPRINT = _segment_fingerprint(
    _SEGMENT_USER, f"User: {format_translation.COMPACTION_SUMMARY_PROMPT}"
)


def _resume_delta(
    segments: list[tuple[str, str]],
    fingerprints: list[str],
    seen: list[str],
) -> list[str]:
    """Return the caller-authored text a resumed session has not consumed yet.

    An empty result means the session cannot be resumed against this
    transcript -- either nothing new arrived, or the history diverged from
    what the session consumed -- and the caller should try
    ``_compaction_resume_delta`` before falling back to a fresh session
    carrying the full transcript.
    """
    if not seen or len(seen) >= len(fingerprints):
        return []
    if fingerprints[: len(seen)] != seen:
        return []
    durable = _durable_segments(segments)
    new_text = [text for kind, text in durable[len(seen):] if kind == _SEGMENT_USER]
    if not new_text:
        return []
    new_text.extend(text for kind, text in segments if kind == _SEGMENT_SYNTHETIC)
    return new_text


def _compaction_resume_delta(
    segments: list[tuple[str, str]],
    fingerprints: list[str],
    seen: list[str],
) -> list[str]:
    """Return what to send a session whose caller just compacted its transcript.

    A client-side compaction rewrites the transcript to preamble + summary, so
    the prefix check in ``_resume_delta`` fails even though the SDK session
    that wrote that summary is the right home for the thread: it still holds
    the working context (the SDK manages its own window), and the summary is
    its own last turn.  Replaying the summary into a fresh session instead
    hands the model a second-hand account of its own work, which it then
    re-verifies from scratch.  Recognize the handoff -- the session's last
    consumed segment was the proxy's summary request and the caller's
    preamble is unchanged -- and send only what follows the summary.
    """
    if not seen or seen[-1] != _SUMMARY_REQUEST_FINGERPRINT:
        return []
    durable = _durable_segments(segments)
    summary_index = next(
        (index for index, (kind, _) in enumerate(durable) if kind == _SEGMENT_SUMMARY),
        None,
    )
    if summary_index is None:
        return []
    if fingerprints[:summary_index] != seen[:summary_index]:
        return []
    new_text = [text for kind, text in durable[summary_index + 1:] if kind == _SEGMENT_USER]
    new_text.extend(text for kind, text in segments if kind == _SEGMENT_SYNTHETIC)
    return new_text


@dataclass(frozen=True)
class PendingToolResult:
    session_id: str
    request_id: str
    output: str
    tool_name: str = ""


def _is_caller_message(item: Any) -> bool:
    return isinstance(item, str) or (
        isinstance(item, dict)
        and item.get("type") in {None, "message"}
        and item.get("role") in {"user", "developer", "system"}
    )


def resolve_tool_continuation(value: Any) -> tuple[str, list[PendingToolResult]] | None:
    """Resolve the trailing tool results, allowing accompanying caller messages."""
    if not isinstance(value, list):
        return None
    trailing: list[PendingToolResult] = []
    for item in reversed(value):
        if _is_caller_message(item):
            continue
        if not isinstance(item, dict) or item.get("type") not in {
            "function_call_output",
            "custom_tool_call_output",
        }:
            break
        decoded = _decode_call_id(item.get("call_id"))
        if decoded is None:
            return None
        trailing.append(
            PendingToolResult(
                session_id=decoded["s"],
                request_id=decoded["r"],
                output=_text_from_content(item.get("output")),
                tool_name=decoded.get("n", ""),
            )
        )
    if not trailing:
        return None
    trailing.reverse()
    session_id = trailing[0].session_id
    if any(result.session_id != session_id for result in trailing):
        return None
    return session_id, trailing


async def _get_client():
    global _client, _client_lock, _client_pruned, _client_token
    if _SDK_IMPORT_ERROR is not None or CopilotClient is None:
        raise RuntimeError(
            "The official Copilot SDK is not installed. Run: pip install github-copilot-sdk"
        ) from _SDK_IMPORT_ERROR
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    token = auth.load_access_token()
    async with _client_lock:
        if _client is not None and token == _client_token:
            return _client
        if _client is not None:
            await _client.stop()
        # The SDK's first-run runtime downloader uses urllib rather than httpx.
        # Framework builds of Python on macOS commonly lack a usable system CA
        # chain, while certifi is already an httpx dependency.
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.makedirs(_SDK_STATE_DIR, exist_ok=True)
        _client = CopilotClient(
            github_token=token,
            use_logged_in_user=token is None,
            working_directory=os.getcwd(),
            base_directory=_SDK_STATE_DIR,
            log_level=os.getenv("GHCP_SDK_LOG_LEVEL", "error"),
            mode="empty",
        )
        await _client.start()
        _client_token = token
        if not _client_pruned:
            await _prune_abandoned_sessions(_client)
            _client_pruned = True
        return _client


def _reasoning_effort(body: dict) -> str | None:
    reasoning = body.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    return effort if effort in {"low", "medium", "high", "xhigh", "max"} else None


def _model_id_for_lookup(model: Any) -> str | None:
    if not isinstance(model, str) or not model.strip():
        return None
    value = model.strip().lower()
    # The Responses API may use provider-qualified ids while the SDK's model
    # registry stores the Copilot id without the provider prefix.
    if "/" in value:
        value = value.rsplit("/", 1)[1]
    return value


def _model_supports_reasoning_effort(
    model: Any,
    requested_effort: str,
    models: Any,
) -> bool | None:
    """Return the SDK registry's reasoning-effort capability for ``model``.

    ``None`` means that the registry could not answer (for example, an older
    SDK does not expose model metadata).  It is important to distinguish that
    from ``False``: the Copilot RPC rejects *any* reasoning-effort field for
    models such as Gemini Flash, rather than simply ignoring it.
    """
    requested_id = _model_id_for_lookup(model)
    if requested_id is None or not isinstance(models, (list, tuple)):
        return None

    for info in models:
        info_id = _model_id_for_lookup(getattr(info, "id", None))
        if info_id != requested_id:
            continue

        supported_efforts = getattr(info, "supported_reasoning_efforts", None)
        if isinstance(supported_efforts, (list, tuple)):
            return any(
                isinstance(effort, str) and effort.lower() == requested_effort.lower()
                for effort in supported_efforts
            )

        capabilities = getattr(info, "capabilities", None)
        supports = getattr(capabilities, "supports", None)
        supported = getattr(supports, "reasoning_effort", None)
        if isinstance(supported, bool):
            return supported
        return None
    return None


async def _reasoning_effort_for_client(body: dict, client: Any) -> str | None:
    """Filter the requested effort using the SDK's cached ``models.list``."""
    requested = _reasoning_effort(body)
    if requested is None:
        return None

    try:
        models = await client.list_models()
    except Exception:
        models = None

    # Keep the legacy behavior for model registries unavailable to this SDK,
    # except for Gemini/Grok where sending the field is known to be invalid.
    requested_model = body.get("model")
    supported = _model_supports_reasoning_effort(requested_model, requested, models)
    if supported is False:
        return None
    if supported is True:
        return requested
    model_id = _model_id_for_lookup(requested_model) or ""
    if model_id.startswith(("gemini-", "grok-")):
        return None
    return requested


def _reasoning_summary(body: dict) -> str:
    """Map Responses reasoning settings to the SDK's summary modes."""
    reasoning = body.get("reasoning")
    summary = reasoning.get("summary") if isinstance(reasoning, dict) else None
    if summary in {"none", "concise", "detailed"}:
        return summary
    return "detailed"


def _session_options(
    body: dict,
    registration: ToolRegistration,
    *,
    reasoning_effort: str | None | object = ...,
) -> dict[str, Any]:
    instructions = body.get("instructions")
    if reasoning_effort is ...:
        reasoning_effort = _reasoning_effort(body)
    options: dict[str, Any] = {
        "model": body.get("model") if isinstance(body.get("model"), str) else None,
        "reasoning_effort": reasoning_effort,
        # The SDK does not emit Copilot's reasoning/intent timeline events
        # unless a reasoning summary mode is selected.  Without this, Codex
        # receives only the final answer and tool calls.
        "reasoning_summary": _reasoning_summary(body),
        "streaming": bool(body.get("stream")),
        "tools": registration.tools,
        # Resume may retain previously registered tools when the new list is
        # empty. An explicit allowlist also enforces removals/tool_choice=none.
        "available_tools": [f"custom:{tool.name}" for tool in registration.tools],
        "include_sub_agent_streaming_events": True,
        "on_permission_request": PermissionHandler.approve_all,
        # Resumed SDK sessions own their conversation history, so they also
        # need the SDK's context-window management.  Disabling this makes a
        # long-lived Codex thread fail before the caller gets a chance to
        # consume the proxy's /responses/compact handoff.
        "infinite_sessions": {"enabled": True},
        "enable_managed_settings": False,
        "enable_config_discovery": False,
        "skip_custom_instructions": True,
        "enable_skills": False,
        "enable_file_hooks": False,
        "enable_host_git_operations": False,
    }
    if isinstance(instructions, str) and instructions:
        options["system_message"] = {"mode": "replace", "content": instructions}
    return {key: value for key, value in options.items() if value is not None}


# ---------------------------------------------------------------------------
# Live session pool
# ---------------------------------------------------------------------------
#
# Each Codex tool call used to end with ``session.disconnect`` (a
# ``session.destroy`` RPC) and the tool result arrived a moment later on a
# fresh ``resume_session``.  The SDK's own context management could not
# survive that cycle: its background compaction starts at the turn's first
# model call and takes ~25s, so every destroy killed it mid-flight, the
# resumed session rebuilt the full context from disk, and the next turn
# started the same compaction again -- 24 wasted summary calls in twelve
# minutes on one session, with the context never shrinking.  Keeping the
# session connected across the tool round-trip lets the turn run the way the
# SDK expects (result -> model -> ... -> turn end), so a compaction that
# starts during a turn can finish and take effect.

_LIVE_SESSION_IDLE_SECONDS = float(os.environ.get("GHCP_SDK_SESSION_IDLE_SECONDS", "300") or 300)
_LIVE_COMPACTION_WAIT_SECONDS = 600.0


@dataclass
class _LiveSession:
    session: Any
    options: dict[str, Any] | None = None
    input_fingerprints: list[str] = field(default_factory=list)
    in_use: bool = True
    pending_calls: bool = False
    compaction_in_flight: bool = False
    compaction_settled: asyncio.Event = field(default_factory=asyncio.Event)
    unsubscribe: Callable[[], None] | None = None
    reaper: asyncio.Task | None = None


_live_sessions: dict[str, _LiveSession] = {}


def _set_compaction_state(entry: _LiveSession, in_flight: bool) -> None:
    entry.compaction_in_flight = in_flight
    if in_flight:
        entry.compaction_settled.clear()
    else:
        entry.compaction_settled.set()


def _is_compaction_event(event: Any, *, started: bool) -> bool:
    data = getattr(event, "data", None)
    if started:
        return (
            (SessionCompactionStartData is not None and isinstance(data, SessionCompactionStartData))
            or _event_name(event) == "session.compaction_start"
        )
    return (
        (SessionCompactionCompleteData is not None and isinstance(data, SessionCompactionCompleteData))
        or _event_name(event) == "session.compaction_complete"
    )


async def _track_live_session(
    session: Any,
    *,
    options: dict[str, Any] | None = None,
    fingerprints: list[str] | None = None,
) -> _LiveSession:
    """Register a session object the current request is about to drive."""
    entry = _live_sessions.get(session.session_id)
    if entry is not None:
        if entry.session is session:
            if options is not None:
                entry.options = copy.deepcopy(options)
            if fingerprints is not None:
                entry.input_fingerprints = list(fingerprints)
            entry.in_use = True
            if entry.reaper is not None:
                entry.reaper.cancel()
                entry.reaper = None
            return entry
        # A different object for the same id: the old one was superseded by
        # a resume, so it no longer represents the runtime session.
        await _evict_live_session(session.session_id)

    entry = _LiveSession(
        session=session,
        options=copy.deepcopy(options),
        input_fingerprints=list(fingerprints or []),
    )
    loop = asyncio.get_running_loop()

    def handler(event: Any) -> None:
        if _is_compaction_event(event, started=True):
            loop.call_soon_threadsafe(_set_compaction_state, entry, True)
        elif _is_compaction_event(event, started=False):
            loop.call_soon_threadsafe(_set_compaction_state, entry, False)

    try:
        entry.unsubscribe = session.on(handler)
    except Exception:
        entry.unsubscribe = None
    _live_sessions[session.session_id] = entry
    return entry


async def _reuse_live_session(
    session_id: str, *, allow_pending: bool, options: dict[str, Any]
) -> Any | None:
    """Hand back a connected session for ``session_id`` if one is idle.

    A session parked on a pending tool call is only reusable by the request
    that delivers the result.  Any other request (a new user message while
    a tool was still running) needs ``continue_pending_work=False``, which
    is a resume-time option, so the live object is discarded first. Changed
    configuration also needs a resume: the SDK's live options API cannot
    replace tool declarations or the system message. Unchanged configurations
    stay connected so ordinary tool round-trips preserve background compaction.
    """
    entry = _live_sessions.get(session_id)
    if entry is None or entry.in_use:
        return None
    if (entry.pending_calls and not allow_pending) or entry.options != options:
        await _evict_live_session(session_id)
        return None
    entry.in_use = True
    if entry.reaper is not None:
        entry.reaper.cancel()
        entry.reaper = None
    return entry.session


def _continuation_prompt(
    body: dict,
    session_id: str,
    segments: list[tuple[str, str]],
    fingerprints: list[str],
) -> str:
    """Find new caller instructions without replaying the session's history."""
    entry = _live_sessions.get(session_id)
    seen = entry.input_fingerprints if entry is not None else []
    if not seen:
        alias = _session_alias(body)
        known = _session_for_alias(alias) if alias else None
        if known is not None and known[0] == session_id:
            seen = known[1]
    if seen and fingerprints[:len(seen)] == seen:
        return "\n\n".join(
            text for kind, text in _durable_segments(segments)[len(seen):]
            if kind == _SEGMENT_USER
        )

    # After a restart or with a partial transcript, the current tool-call /
    # assistant item bounds the new messages. Never resend earlier user turns.
    tail = []
    for item in reversed(body.get("input") or []):
        if _is_caller_message(item):
            tail.append(item)
        elif isinstance(item, dict) and item.get("type") in {
            "function_call_output", "custom_tool_call_output",
        }:
            continue
        else:
            break
    return "\n\n".join(
        text for kind, text in _render_input_segments(list(reversed(tail)))
        if kind == _SEGMENT_USER
    )


async def _evict_live_session(session_id: str) -> None:
    entry = _live_sessions.pop(session_id, None)
    if entry is None:
        return
    if entry.unsubscribe is not None:
        try:
            entry.unsubscribe()
        except Exception:
            pass
    if entry.reaper is not None and entry.reaper is not asyncio.current_task():
        entry.reaper.cancel()
    try:
        await entry.session.disconnect()
    except Exception:
        pass


async def _reap_live_session(session_id: str, entry: _LiveSession) -> None:
    try:
        if entry.compaction_in_flight:
            try:
                await asyncio.wait_for(entry.compaction_settled.wait(), _LIVE_COMPACTION_WAIT_SECONDS)
            except TimeoutError:
                pass
        if entry.pending_calls:
            await asyncio.sleep(_LIVE_SESSION_IDLE_SECONDS)
    except asyncio.CancelledError:
        return
    if entry.in_use or _live_sessions.get(session_id) is not entry:
        return
    await _evict_live_session(session_id)


async def _release_session(session: Any, outcome: "TurnOutcome | None", *, completed: bool) -> None:
    """Finish a request's use of ``session``.

    The session stays connected when the caller owes it a tool result or the
    SDK is compacting it in the background; a reaper disconnects it if
    neither resolves.  Everything else disconnects immediately, which is
    also what closes out per-session usage on the SDK side.
    """
    entry = _live_sessions.get(session.session_id)
    if entry is None or entry.session is not session:
        try:
            await session.disconnect()
        except Exception:
            pass
        return
    pending = bool(completed and outcome is not None and outcome.calls)
    if not completed or not (pending or entry.compaction_in_flight):
        await _evict_live_session(session.session_id)
        return
    entry.in_use = False
    entry.pending_calls = pending
    entry.reaper = asyncio.create_task(_reap_live_session(session.session_id, entry))


async def _evict_all_live_sessions() -> None:
    for session_id in list(_live_sessions):
        await _evict_live_session(session_id)


async def _open_session(body: dict, registration: ToolRegistration):
    client = await _get_client()
    continuation = resolve_tool_continuation(body.get("input"))
    reasoning_effort = await _reasoning_effort_for_client(body, client)
    options = _session_options(body, registration, reasoning_effort=reasoning_effort)
    segments = _render_input_segments(body.get("input"))
    fingerprints = _segment_fingerprints(segments)
    if continuation is None:
        alias = _session_alias(body)

        # Resuming lets the SDK keep owning the history.  Replaying the whole
        # transcript into a fresh session instead costs the caller its entire
        # context window before the agent does any work, which is why long
        # conversations used to stall out early.
        session = None
        prompt = None
        if alias:
            known = _session_for_alias(alias)
            if known is not None:
                known_session_id, seen = known
                new_text = _resume_delta(segments, fingerprints, seen)
                if not new_text:
                    new_text = _compaction_resume_delta(segments, fingerprints, seen)
                if new_text:
                    session = await _reuse_live_session(
                        known_session_id, allow_pending=False, options=options,
                    )
                    if session is None:
                        try:
                            session = await client.resume_session(
                                known_session_id,
                                continue_pending_work=False,
                                **options,
                            )
                        except Exception:
                            # The SDK discarded the session; fall through to a
                            # fresh one carrying the full transcript.
                            session = None
                    if session is not None:
                        prompt = "\n\n".join(new_text)
        if session is None:
            session = await client.create_session(**options)
            prompt = "\n\n".join(text for _, text in segments)

        await _track_live_session(session, options=options, fingerprints=fingerprints)
        _remember_session(session.session_id)
        if alias:
            _pending_alias_watermark[session.session_id] = (alias, fingerprints)

        async def dispatch() -> None:
            if not prompt:
                raise ValueError("input must contain at least one text message")
            await session.send(prompt)

        return session, dispatch

    session_id, results = continuation
    # Read the old watermark before reconfiguration can evict the live entry.
    steering_prompt = _continuation_prompt(body, session_id, segments, fingerprints)
    pending_work = True
    session = await _reuse_live_session(session_id, allow_pending=True, options=options)
    if session is None and _owns_session(session_id):
        try:
            session = await client.resume_session(
                session_id,
                continue_pending_work=True,
                **options,
            )
        except Exception:
            pending_work = False
            try:
                session = await client.resume_session(
                    session_id,
                    continue_pending_work=False,
                    **options,
                )
            except Exception:
                session = None

    if session is None:
        # If the SDK session could not be resumed (e.g. proxy was reset and session
        # was pruned or state file lost), fall back to creating a fresh session
        # carrying the transcript up to and including the tool output.
        session = await client.create_session(**options)
        prompt = "\n\n".join(text for _, text in segments)
        await _track_live_session(session, options=options, fingerprints=fingerprints)
        _remember_session(session.session_id)
        alias = _session_alias(body)
        if alias:
            _pending_alias_watermark[session.session_id] = (
                alias,
                fingerprints,
            )

        async def dispatch_fresh() -> None:
            if not prompt:
                raise ValueError("input must contain at least one text message")
            await session.send(prompt)

        return session, dispatch_fresh

    await _track_live_session(session, options=options, fingerprints=fingerprints)
    _remember_session(session.session_id)
    alias = _session_alias(body)
    if alias:
        _pending_alias_watermark[session.session_id] = (
            alias,
            fingerprints,
        )

    async def dispatch() -> None:
        if steering_prompt and pending_work:
            # Enqueue would wait for the agent to finish. Immediate steering
            # must precede the result that releases its next model call.
            await session.send(steering_prompt, mode="immediate")
        failed = not pending_work
        for result in results if pending_work else []:
            try:
                res = await session.rpc.tools.handle_pending_tool_call(
                    HandlePendingToolCallRequest(
                        request_id=result.request_id,
                        result=(
                            ExternalToolTextResultForLlm(
                                text_result_for_llm=result.output,
                                result_type="success",
                            )
                            if ExternalToolTextResultForLlm is not None
                            else result.output
                        ),
                    )
                )
                if res is not None and getattr(res, "success", None) is False:
                    failed = True
                    break
            except Exception:
                failed = True
                break

        if failed:
            # The pending tool call was lost (e.g. proxy or Copilot SDK daemon was reset
            # while the tool was running). The CLI discarded the in-memory pending
            # work and will never emit turn events on its own. Deliver the tool
            # results as a prompt so the model continues rather than leaving Codex
            # stuck on "reconnecting".
            tool_texts = []
            for result in results:
                name = getattr(result, "tool_name", "")
                prefix = f"Tool result for {name}: " if name else "Tool result: "
                tool_texts.append(f"{prefix}{result.output}")
            fallback_prompt = "\n\n".join(tool_texts) or "Tool execution completed."
            if steering_prompt and not pending_work:
                fallback_prompt = f"{steering_prompt}\n\n{fallback_prompt}"
            await session.send(fallback_prompt)

    return session, dispatch


@dataclass
class ToolCall:
    request_id: str
    name: str
    tool_type: str
    arguments: Any
    item_id: str = field(default_factory=lambda: _new_id("fc"))


@dataclass
class TurnOutcome:
    text: str = ""
    reasoning: str = ""
    reasoning_id: str = field(default_factory=lambda: _new_id("rs"))
    message_id: str = field(default_factory=lambda: _new_id("msg"))
    calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    # Two independent usage sources; ``_finalize_usage`` picks between them.
    shutdown_usage: dict[str, int] = field(default_factory=dict)
    event_usage: dict[str, int] = field(default_factory=dict)


def _event_name(event: Any) -> str:
    value = getattr(event, "type", "")
    return getattr(value, "value", value) or ""


def _usage_from_event(data: Any) -> dict[str, int]:
    # ``input_tokens`` from AssistantUsageData is the *total* input (fresh +
    # cached).  Subtract ``cache_read_tokens`` so that the flat dict we return
    # is consistent with the shape produced by ``_usage_delta``, where
    # ``input_tokens`` == total and ``fresh_input_tokens`` == uncached portion.
    input_tokens = int(getattr(data, "input_tokens", 0) or 0)
    output_tokens = int(getattr(data, "output_tokens", 0) or 0)
    cached_tokens = int(getattr(data, "cache_read_tokens", 0) or 0)
    cache_write_tokens = int(getattr(data, "cache_write_tokens", 0) or 0)
    reasoning_tokens = int(getattr(data, "reasoning_tokens", 0) or 0)
    # Cache creation is still fresh input.  Only a cache *read* was supplied
    # from prior context, so fresh input is total input minus cache reads.
    fresh = max(0, input_tokens - cached_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_input_tokens": cached_tokens,
        "cache_creation_input_tokens": cache_write_tokens,
        "fresh_input_tokens": fresh,
        "pricing_fresh_input_tokens": fresh,
        "pricing_cached_input_tokens": cached_tokens,
        "pricing_cache_creation_input_tokens": cache_write_tokens,
        "reasoning_output_tokens": reasoning_tokens,
    }


def _token_count(entry: Any) -> int:
    if entry is None:
        return 0
    value = getattr(entry, "token_count", None)
    if value is None and isinstance(entry, dict):
        value = entry.get("tokenCount")
    return int(value or 0)


def _extract_shutdown_usage(data: Any) -> dict[str, int]:
    """Extract cumulative token counts from SessionShutdownData or equivalent dict."""
    total_inp = 0
    cread = 0
    cwrite = 0
    out = 0
    reas = 0

    mm = getattr(data, "model_metrics", None) or (data.get("modelMetrics") if isinstance(data, dict) else None)
    if isinstance(mm, dict):
        for m_val in mm.values():
            u = getattr(m_val, "usage", None) or (m_val.get("usage") if isinstance(m_val, dict) else None)
            if u is None:
                continue
            inp_val = getattr(u, "input_tokens", None) if hasattr(u, "input_tokens") else (u.get("inputTokens") if isinstance(u, dict) else 0)
            read_val = getattr(u, "cache_read_tokens", None) if hasattr(u, "cache_read_tokens") else (u.get("cacheReadTokens") if isinstance(u, dict) else 0)
            write_val = getattr(u, "cache_write_tokens", None) if hasattr(u, "cache_write_tokens") else (u.get("cacheWriteTokens") if isinstance(u, dict) else 0)
            out_val = getattr(u, "output_tokens", None) if hasattr(u, "output_tokens") else (u.get("outputTokens") if isinstance(u, dict) else 0)
            reas_val = getattr(u, "reasoning_tokens", None) if hasattr(u, "reasoning_tokens") else (u.get("reasoningTokens") if isinstance(u, dict) else 0)
            total_inp += int(inp_val or 0)
            cread += int(read_val or 0)
            cwrite += int(write_val or 0)
            out += int(out_val or 0)
            reas += int(reas_val or 0)

    # ``tokenDetails`` is the session-wide superset: it also covers API calls
    # that never land in ``modelMetrics`` (observed ~11% higher on long
    # sessions), so it wins whenever it reports anything.  Reasoning tokens
    # only ever appear under ``modelMetrics``.
    td = getattr(data, "token_details", None) or (data.get("tokenDetails") if isinstance(data, dict) else None)
    if isinstance(td, dict):
        fresh_tok = _token_count(td.get("input"))
        read_tok = _token_count(td.get("cache_read"))
        write_tok = _token_count(td.get("cache_write"))
        out_tok = _token_count(td.get("output"))
        if fresh_tok or read_tok or write_tok or out_tok:
            # ``tokenDetails.input`` is the uncached remainder, so the total
            # input is the sum of all three buckets.  This matches
            # ``modelMetrics.inputTokens`` on the same event.
            total_inp = fresh_tok + read_tok + write_tok
            cread = read_tok
            cwrite = write_tok
            out = out_tok

    # ``cache_write`` is an independently billed subset of fresh input, not
    # previously cached context.  Do not subtract it from fresh input.
    fresh = max(0, total_inp - cread)

    return {
        "input_tokens": total_inp,
        "cached_input_tokens": cread,
        "cache_creation_input_tokens": cwrite,
        "fresh_input_tokens": fresh,
        "pricing_fresh_input_tokens": fresh,
        "pricing_cached_input_tokens": cread,
        "pricing_cache_creation_input_tokens": cwrite,
        "output_tokens": out,
        "reasoning_output_tokens": reas,
        "total_tokens": total_inp + out,
    }


def _usage_delta(current: dict[str, int], previous: dict[str, int] | None) -> dict[str, int]:
    """Compute per-turn token usage delta from cumulative session totals."""
    if previous is None:
        return dict(current)
    inp = max(0, current["input_tokens"] - previous.get("input_tokens", 0))
    out = max(0, current["output_tokens"] - previous.get("output_tokens", 0))
    cread = max(0, current["cached_input_tokens"] - previous.get("cached_input_tokens", 0))
    cwrite = max(0, current["cache_creation_input_tokens"] - previous.get("cache_creation_input_tokens", 0))
    reas = max(0, current["reasoning_output_tokens"] - previous.get("reasoning_output_tokens", 0))
    fresh = max(0, inp - cread)
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cached_input_tokens": cread,
        "cache_creation_input_tokens": cwrite,
        "fresh_input_tokens": fresh,
        "pricing_fresh_input_tokens": fresh,
        "pricing_cached_input_tokens": cread,
        "pricing_cache_creation_input_tokens": cwrite,
        "reasoning_output_tokens": reas,
        "total_tokens": inp + out,
    }


def _add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    """Accumulate usage from one SDK API call into the current turn.

    A Copilot turn can contain several model calls (for example, a tool call
    followed by the final answer).  ``assistant.usage`` is emitted per call,
    so replacing the previous record loses all but the last call.
    """
    for key in _USAGE_KEYS:
        total[key] = total.get(key, 0) + max(0, int(usage.get(key, 0) or 0))
    total["total_tokens"] = total.get("input_tokens", 0) + total.get("output_tokens", 0)


_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "fresh_input_tokens",
    "pricing_fresh_input_tokens",
    "pricing_cached_input_tokens",
    "pricing_cache_creation_input_tokens",
    "reasoning_output_tokens",
)


def _read_shutdown_baselines_unlocked() -> dict[str, dict[str, int]]:
    payload = _read_json_state(_state_path(_SESSION_USAGE_NAME))
    if not isinstance(payload, dict):
        return {}
    baselines: dict[str, dict[str, int]] = {}
    for session_id, usage in payload.items():
        if not isinstance(session_id, str) or not isinstance(usage, dict):
            continue
        baselines[session_id] = {
            key: int(usage.get(key) or 0) for key in _USAGE_KEYS + ("total_tokens",)
        }
    return baselines


def _shutdown_baseline(session_id: str) -> dict[str, int] | None:
    """Cumulative session usage as of the previous turn, or None if unseen.

    This has to survive a proxy restart: SDK sessions outlive the process
    now, and ``session.shutdown`` reports session-cumulative totals.  Losing
    the baseline would bill an entire multi-million-token session against
    whichever single request happened to come first after the restart.
    """
    with _SESSION_LEDGER_LOCK:
        return _read_shutdown_baselines_unlocked().get(session_id)


def _store_shutdown_baseline(session_id: str, usage: dict[str, int]) -> None:
    with _SESSION_LEDGER_LOCK:
        baselines = _read_shutdown_baselines_unlocked()
        baselines[session_id] = dict(usage)
        try:
            _write_json_state(_state_path(_SESSION_USAGE_NAME), baselines, "proxy-session-usage-")
        except OSError:
            pass


def _forget_shutdown_baselines(session_ids: set[str]) -> None:
    if not session_ids:
        return
    with _SESSION_LEDGER_LOCK:
        baselines = _read_shutdown_baselines_unlocked()
        remaining = {k: v for k, v in baselines.items() if k not in session_ids}
        if len(remaining) != len(baselines):
            try:
                _write_json_state(_state_path(_SESSION_USAGE_NAME), remaining, "proxy-session-usage-")
            except OSError:
                pass


def _record_shutdown_usage(outcome: "TurnOutcome", session_id: str, data: Any) -> None:
    current = _extract_shutdown_usage(data)
    outcome.shutdown_usage = _usage_delta(current, _shutdown_baseline(session_id))
    _store_shutdown_baseline(session_id, current)


def _format_client_usage(usage: dict[str, int] | None) -> dict[str, Any]:
    """Format token usage for OpenAI Responses API clients such as Codex.

    Responses API requires nested ``input_tokens_details.cached_tokens`` and
    ``output_tokens_details.reasoning_tokens`` so clients correctly recognize
    cached input and reasoning token breakdowns.
    """
    if not isinstance(usage, dict) or not usage:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cached_tokens = int(
        usage.get("cached_input_tokens")
        or (
            usage.get("input_tokens_details", {}).get("cached_tokens")
            if isinstance(usage.get("input_tokens_details"), dict)
            else 0
        )
        or 0
    )
    cache_creation_tokens = int(
        usage.get("cache_creation_input_tokens")
        or (
            usage.get("input_tokens_details", {}).get("cache_creation_input_tokens")
            if isinstance(usage.get("input_tokens_details"), dict)
            else 0
        )
        or 0
    )
    reasoning_tokens = int(
        usage.get("reasoning_output_tokens")
        or (
            usage.get("output_tokens_details", {}).get("reasoning_tokens")
            if isinstance(usage.get("output_tokens_details"), dict)
            else 0
        )
        or 0
    )
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))

    input_details: dict[str, int] = {"cached_tokens": cached_tokens}
    if cache_creation_tokens:
        input_details["cache_creation_input_tokens"] = cache_creation_tokens

    output_details: dict[str, int] = {"reasoning_tokens": reasoning_tokens}

    result = dict(usage)
    result.update({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_tokens_details": input_details,
        "output_tokens_details": output_details,
    })
    return result


def _finalize_usage(outcome: "TurnOutcome") -> None:
    """Pick the authoritative usage record for the turn.

    ``session.shutdown`` carries session-cumulative totals and lands at the
    end of a turn, while ``assistant.usage`` fires per model API call during
    it.  They describe the same tokens, so exactly one must win -- these used
    to be sibling ``elif`` branches assigning the same field, and whichever
    arrived last silently erased the other.
    """
    raw = outcome.shutdown_usage or outcome.event_usage
    outcome.usage = _format_client_usage(raw) if raw else {}



def _tool_call(data: Any, registration: ToolRegistration) -> ToolCall:
    safe_name = str(getattr(data, "tool_name", "tool"))
    metadata = registration.names.get(safe_name, ToolMetadata(safe_name, "function"))
    prefix = "ctc" if metadata.tool_type == "custom" else "fc"
    return ToolCall(
        request_id=str(getattr(data, "request_id")),
        name=metadata.original_name,
        tool_type=metadata.tool_type,
        arguments=getattr(data, "arguments", {}) or {},
        item_id=_new_id(prefix),
    )


def _event_queue(session: Any) -> tuple[asyncio.Queue, Callable[[], None]]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def handler(event: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    return queue, session.on(handler)


async def _wait_for_outcome(
    session: Any,
    dispatch: Callable[[], Awaitable[None]],
    registration: ToolRegistration,
) -> TurnOutcome:
    queue, unsubscribe = _event_queue(session)
    outcome = TurnOutcome()
    saw_delta = False
    saw_reasoning_delta = False
    dispatch_task = asyncio.create_task(dispatch())
    try:
        # ``send`` is asynchronous and may fail before the SDK emits any
        # event.  Give it one scheduling turn so invalid input/auth failures
        # are surfaced immediately instead of waiting for the full turn
        # timeout.
        await asyncio.sleep(0)
        if dispatch_task.done():
            error = dispatch_task.exception()
            if error is not None:
                raise error
        start_wait_time = time.time()
        while True:
            timeout = _PARALLEL_TOOL_SETTLE_SECONDS if outcome.calls else min(15.0, _TURN_TIMEOUT_SECONDS)
            try:
                event = await asyncio.wait_for(queue.get(), timeout=timeout)
                start_wait_time = time.time()
            except TimeoutError:
                if outcome.calls:
                    _finalize_usage(outcome)
                    return outcome
                if (time.time() - start_wait_time) >= _TURN_TIMEOUT_SECONDS:
                    raise TimeoutError(f"Timed out waiting for the Copilot SDK turn after {_TURN_TIMEOUT_SECONDS}s")
                continue
            data = getattr(event, "data", None)
            if isinstance(data, AssistantMessageDeltaData):
                if getattr(data, "parent_tool_call_id", None):
                    outcome.reasoning += data.delta_content
                    saw_reasoning_delta = True
                else:
                    outcome.text += data.delta_content
                    saw_delta = True
            elif isinstance(data, AssistantReasoningDeltaData):
                outcome.reasoning += data.delta_content
                saw_reasoning_delta = True
            elif isinstance(data, AssistantReasoningData):
                if not saw_reasoning_delta and data.content:
                    outcome.reasoning = data.content
            elif isinstance(data, AssistantIntentData):
                if not saw_reasoning_delta and not outcome.reasoning and data.intent:
                    outcome.reasoning = data.intent
            elif isinstance(data, AssistantMessageData):
                if getattr(data, "parent_tool_call_id", None):
                    if not saw_reasoning_delta and data.content:
                        outcome.reasoning = data.content
                else:
                    if not saw_delta:
                        outcome.text = data.content or outcome.text
            elif (
                (SessionShutdownData is not None and isinstance(data, SessionShutdownData))
                or _event_name(event) == "session.shutdown"
            ):
                _record_shutdown_usage(outcome, session.session_id, data)
                if outcome.calls:
                    _finalize_usage(outcome)
                    return outcome
            elif isinstance(data, AssistantUsageData):
                _add_usage(outcome.event_usage, _usage_from_event(data))
            elif isinstance(data, ExternalToolRequestedData):
                outcome.calls.append(_tool_call(data, registration))
            elif (SubagentStartedData is not None and isinstance(data, SubagentStartedData)) or _event_name(event) == "subagent.started":
                agent_name = getattr(data, "agent_display_name", None) or getattr(data, "agent_name", "subagent")
                outcome.reasoning += f"[Subagent '{agent_name}' started]\n"
                saw_reasoning_delta = True
            elif (SubagentCompletedData is not None and isinstance(data, SubagentCompletedData)) or _event_name(event) == "subagent.completed":
                agent_name = getattr(data, "agent_display_name", None) or getattr(data, "agent_name", "subagent")
                outcome.reasoning += f"[Subagent '{agent_name}' completed]\n"
                saw_reasoning_delta = True
            elif (SubagentFailedData is not None and isinstance(data, SubagentFailedData)) or _event_name(event) == "subagent.failed":
                agent_name = getattr(data, "agent_display_name", None) or getattr(data, "agent_name", "subagent")
                err_msg = getattr(data, "error", "error")
                outcome.reasoning += f"[Subagent '{agent_name}' failed: {err_msg}]\n"
                saw_reasoning_delta = True
            elif (
                (SessionCompactionCompleteData is not None and isinstance(data, SessionCompactionCompleteData))
                or _event_name(event) in {"session.compaction_start", "session.compaction_complete"}
            ):
                # Internal session maintenance, not assistant output.  The
                # model response to our compact prompt arrives through the
                # ordinary assistant message events.
                pass
            elif isinstance(data, SessionErrorData):
                raise RuntimeError(data.message)
            elif isinstance(data, SessionIdleData):
                _finalize_usage(outcome)
                return outcome
            elif _event_name(event) == "session.error":
                raise RuntimeError(str(getattr(data, "message", "Copilot session error")))
            if dispatch_task.done():
                error = dispatch_task.exception()
                if error is not None:
                    raise error
    finally:
        unsubscribe()
        if not dispatch_task.done():
            dispatch_task.cancel()


def _arguments_json(call: ToolCall) -> str:
    if call.tool_type == "custom":
        arguments = call.arguments
        # The SDK only exposes JSON-schema tools, so free-form Responses tools
        # are registered behind an {"input": "..."} shim.  Some models return
        # that shim as a JSON string rather than a decoded object.  Luna also
        # uses the semantically natural {"patch": "..."} spelling for
        # apply_patch.  Passing either wrapper through as the custom tool's raw
        # input makes Codex reject a valid patch, after which the model retries
        # the identical call indefinitely.
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError:
                return arguments
            if isinstance(decoded, dict):
                arguments = decoded
            else:
                return arguments
        if isinstance(arguments, dict):
            wrapped_input = arguments.get("input")
            if isinstance(wrapped_input, str):
                return wrapped_input
            if call.name == "apply_patch":
                wrapped_patch = arguments.get("patch")
                if isinstance(wrapped_patch, str):
                    return wrapped_patch
        return _text_from_content(arguments)
    if isinstance(call.arguments, str):
        try:
            json.loads(call.arguments)
            return call.arguments
        except json.JSONDecodeError:
            return json.dumps({"input": call.arguments}, ensure_ascii=False)
    return json.dumps(call.arguments or {}, ensure_ascii=False, separators=(",", ":"))


def _tool_item(session_id: str, call: ToolCall, *, completed: bool = True) -> dict:
    call_id = _encode_call_id(
        session_id,
        call.request_id,
        tool_name=call.name,
        tool_type=call.tool_type,
    )
    item = {
        "type": "custom_tool_call" if call.tool_type == "custom" else "function_call",
        "id": call.item_id,
        "call_id": call_id,
        "name": call.name,
        "status": "completed" if completed else "in_progress",
    }
    item["input" if call.tool_type == "custom" else "arguments"] = _arguments_json(call)
    return item


def _message_item(text: str, *, item_id: str | None = None, completed: bool = True) -> dict:
    return {
        "type": "message",
        "id": item_id or _new_id("msg"),
        "role": "assistant",
        "status": "completed" if completed else "in_progress",
        "content": ([{"type": "output_text", "text": text, "annotations": []}] if completed else []),
    }


def _reasoning_item(text: str, *, item_id: str | None = None, completed: bool = True) -> dict:
    formatted_text = format_translation.ensure_codex_reasoning_header(text) if (completed and text) else text
    return {
        "type": "reasoning",
        "id": item_id or _new_id("rs"),
        "status": "completed" if completed else "in_progress",
        "summary": ([{"type": "summary_text", "text": formatted_text}] if (completed and formatted_text) else []),
        "content": ([{"type": "reasoning_text", "text": formatted_text}] if (completed and formatted_text) else []),
        "encrypted_content": None,
    }


def _response_payload(body: dict, session_id: str, outcome: TurnOutcome, response_id: str) -> dict:
    output: list[dict] = []
    if outcome.reasoning:
        output.append(_reasoning_item(outcome.reasoning, item_id=outcome.reasoning_id, completed=True))
    if outcome.text:
        output.append(_message_item(outcome.text, item_id=outcome.message_id, completed=True))
    output.extend(_tool_item(session_id, call, completed=True) for call in outcome.calls)
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": body.get("model"),
        "output": output,
        "parallel_tool_calls": len(outcome.calls) > 1,
        "usage": _format_client_usage(outcome.usage),
    }


def to_compaction_payload(
    body: dict,
    session_id: str,
    outcome: TurnOutcome,
    response_id: str,
    *,
    fallback_model: str | None = None,
) -> dict:
    summary_text = outcome.text.strip() or "(no summary available)"
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": fallback_model or body.get("model"),
        "output": [
            {
                "type": "compaction",
                "encrypted_content": format_translation.encode_fake_compaction(summary_text),
            }
        ],
        "output_text": summary_text,
        "parallel_tool_calls": False,
        "usage": _format_client_usage(outcome.usage),
    }


def _sse(event_type: str, **payload: Any) -> bytes:
    data = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


async def _stream_turn(
    request: Request,
    body: dict,
    session: Any,
    dispatch: Callable[[], Awaitable[None]],
    registration: ToolRegistration,
    *,
    plan: Any = None,
    is_compact: bool = False,
    finish_usage_callback: Any = None,
    mark_first_output_callback: Any = None,
) -> AsyncIterator[bytes]:
    response_id = _new_id("resp")
    base = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "in_progress",
        "model": body.get("model"),
        "output": [],
    }
    yield _sse("response.created", response=base)
    yield _sse("response.in_progress", response=base)
    queue, unsubscribe = _event_queue(session)
    dispatch_task = asyncio.create_task(dispatch())
    outcome = TurnOutcome()
    output_index = 0
    final_payload: dict | None = None
    usage_finished = False

    reasoning_started = False
    reasoning_closed = False
    reasoning_output_index = 0
    saw_reasoning_delta = False

    message_started = False
    message_closed = False
    message_output_index = 0
    saw_delta = False

    first_output_marked = False

    def finish_usage() -> None:
        nonlocal usage_finished
        if usage_finished or finish_usage_callback is None or plan is None:
            return
        usage_finished = True
        status_code = 200 if final_payload is not None else 500
        try:
            finish_usage_callback(
                plan,
                status_code,
                response_payload=final_payload,
                response_text=outcome.text,
                reasoning_text=outcome.reasoning,
                usage=outcome.usage,
            )
        except Exception:
            pass

    def mark_first() -> None:
        nonlocal first_output_marked
        if not first_output_marked:
            first_output_marked = True
            if mark_first_output_callback is not None:
                try:
                    mark_first_output_callback()
                except Exception:
                    pass

    def emit_reasoning_start() -> list[bytes]:
        nonlocal reasoning_started, reasoning_output_index, output_index
        if reasoning_started:
            return []
        mark_first()
        reasoning_started = True
        reasoning_output_index = output_index
        output_index += 1
        return [
            _sse(
                "response.output_item.added",
                output_index=reasoning_output_index,
                item=_reasoning_item("", item_id=outcome.reasoning_id, completed=False),
            ),
            _sse(
                "response.reasoning_summary_part.added",
                item_id=outcome.reasoning_id,
                output_index=reasoning_output_index,
                summary_index=0,
                part={"type": "summary_text", "text": ""},
            ),
        ]

    def emit_reasoning_done() -> list[bytes]:
        nonlocal reasoning_closed
        if not reasoning_started or reasoning_closed:
            return []
        reasoning_closed = True
        return [
            _sse(
                "response.reasoning_summary_text.done",
                item_id=outcome.reasoning_id,
                output_index=reasoning_output_index,
                summary_index=0,
                text=outcome.reasoning,
            ),
            _sse(
                "response.reasoning_summary_part.done",
                item_id=outcome.reasoning_id,
                output_index=reasoning_output_index,
                summary_index=0,
                part={"type": "summary_text", "text": outcome.reasoning},
            ),
            _sse(
                "response.output_item.done",
                output_index=reasoning_output_index,
                item=_reasoning_item(outcome.reasoning, item_id=outcome.reasoning_id, completed=True),
            ),
        ]

    def emit_text_start() -> list[bytes]:
        nonlocal message_started, message_output_index, output_index
        if message_started:
            return []
        mark_first()
        chunks = list(emit_reasoning_done())
        message_started = True
        message_output_index = output_index
        output_index += 1
        chunks.extend([
            _sse(
                "response.output_item.added",
                output_index=message_output_index,
                item=_message_item("", item_id=outcome.message_id, completed=False),
            ),
            _sse(
                "response.content_part.added",
                item_id=outcome.message_id,
                output_index=message_output_index,
                content_index=0,
                part={"type": "output_text", "text": "", "annotations": []},
            ),
        ])
        return chunks

    def emit_text_done() -> list[bytes]:
        nonlocal message_closed
        if not message_started or message_closed:
            return []
        message_closed = True
        completed_message = _message_item(outcome.text, item_id=outcome.message_id, completed=True)
        return [
            _sse(
                "response.output_text.done",
                item_id=outcome.message_id,
                output_index=message_output_index,
                content_index=0,
                text=outcome.text,
            ),
            _sse(
                "response.content_part.done",
                item_id=outcome.message_id,
                output_index=message_output_index,
                content_index=0,
                part=completed_message["content"][0],
            ),
            _sse(
                "response.output_item.done",
                output_index=message_output_index,
                item=completed_message,
            ),
        ]

    try:
        await asyncio.sleep(0)
        if dispatch_task.done():
            error = dispatch_task.exception()
            if error is not None:
                raise error
        start_wait_time = time.time()
        while True:
            if await request.is_disconnected():
                await session.abort()
                return
            timeout = _PARALLEL_TOOL_SETTLE_SECONDS if outcome.calls else _KEEPALIVE_INTERVAL_SECONDS
            try:
                event = await asyncio.wait_for(queue.get(), timeout=timeout)
                start_wait_time = time.time()
            except TimeoutError:
                if outcome.calls:
                    break
                if (time.time() - start_wait_time) >= _TURN_TIMEOUT_SECONDS:
                    raise TimeoutError(f"Timed out waiting for the Copilot SDK turn after {_TURN_TIMEOUT_SECONDS}s")
                yield b": keep-alive\n\n"
                continue
            data = getattr(event, "data", None)
            # A compact response is a different Responses item type.  Do not
            # leak the SDK's ordinary assistant/tool items into that stream:
            # remote compaction v2 validates the streamed output and requires
            # exactly one compaction item.  Still collect assistant text so it
            # can be placed in the encrypted compaction payload below.
            if is_compact and isinstance(data, AssistantReasoningDeltaData):
                outcome.reasoning += data.delta_content
                saw_reasoning_delta = True
                continue
            if is_compact and isinstance(data, AssistantReasoningData):
                if not saw_reasoning_delta and data.content:
                    outcome.reasoning = data.content
                continue
            if is_compact and isinstance(data, AssistantMessageDeltaData):
                if not getattr(data, "parent_tool_call_id", None):
                    outcome.text += data.delta_content
                    saw_delta = True
                continue
            if is_compact and isinstance(data, AssistantMessageData):
                if not getattr(data, "parent_tool_call_id", None) and not saw_delta:
                    outcome.text = data.content or outcome.text
                continue
            if isinstance(data, AssistantReasoningDeltaData):
                for chunk in emit_reasoning_start():
                    yield chunk
                outcome.reasoning += data.delta_content
                saw_reasoning_delta = True
                yield _sse(
                    "response.reasoning_summary_text.delta",
                    item_id=outcome.reasoning_id,
                    output_index=reasoning_output_index,
                    summary_index=0,
                    delta=data.delta_content,
                )
            elif isinstance(data, AssistantReasoningData):
                if not saw_reasoning_delta and data.content:
                    for chunk in emit_reasoning_start():
                        yield chunk
                    outcome.reasoning = data.content
                    yield _sse(
                        "response.reasoning_summary_text.delta",
                        item_id=outcome.reasoning_id,
                        output_index=reasoning_output_index,
                        summary_index=0,
                        delta=data.content,
                    )
            elif isinstance(data, AssistantIntentData):
                if not saw_reasoning_delta and not outcome.reasoning and data.intent:
                    for chunk in emit_reasoning_start():
                        yield chunk
                    outcome.reasoning = data.intent
                    yield _sse(
                        "response.reasoning_summary_text.delta",
                        item_id=outcome.reasoning_id,
                        output_index=reasoning_output_index,
                        summary_index=0,
                        delta=data.intent,
                    )
            elif (SubagentStartedData is not None and isinstance(data, SubagentStartedData)) or _event_name(event) == "subagent.started":
                agent_name = getattr(data, "agent_display_name", None) or getattr(data, "agent_name", "subagent")
                notice = f"[Subagent '{agent_name}' started]\n"
                for chunk in emit_reasoning_start():
                    yield chunk
                outcome.reasoning += notice
                saw_reasoning_delta = True
                yield _sse(
                    "response.reasoning_summary_text.delta",
                    item_id=outcome.reasoning_id,
                    output_index=reasoning_output_index,
                    summary_index=0,
                    delta=notice,
                )
            elif (SubagentCompletedData is not None and isinstance(data, SubagentCompletedData)) or _event_name(event) == "subagent.completed":
                agent_name = getattr(data, "agent_display_name", None) or getattr(data, "agent_name", "subagent")
                notice = f"[Subagent '{agent_name}' completed]\n"
                for chunk in emit_reasoning_start():
                    yield chunk
                outcome.reasoning += notice
                saw_reasoning_delta = True
                yield _sse(
                    "response.reasoning_summary_text.delta",
                    item_id=outcome.reasoning_id,
                    output_index=reasoning_output_index,
                    summary_index=0,
                    delta=notice,
                )
            elif (SubagentFailedData is not None and isinstance(data, SubagentFailedData)) or _event_name(event) == "subagent.failed":
                agent_name = getattr(data, "agent_display_name", None) or getattr(data, "agent_name", "subagent")
                err_msg = getattr(data, "error", "error")
                notice = f"[Subagent '{agent_name}' failed: {err_msg}]\n"
                for chunk in emit_reasoning_start():
                    yield chunk
                outcome.reasoning += notice
                saw_reasoning_delta = True
                yield _sse(
                    "response.reasoning_summary_text.delta",
                    item_id=outcome.reasoning_id,
                    output_index=reasoning_output_index,
                    summary_index=0,
                    delta=notice,
                )
            elif (
                (SessionCompactionCompleteData is not None and isinstance(data, SessionCompactionCompleteData))
                or _event_name(event) in {"session.compaction_start", "session.compaction_complete"}
            ):
                # Internal bookkeeping, not downstream response text.
                pass
            elif isinstance(data, AssistantMessageDeltaData):
                if getattr(data, "parent_tool_call_id", None):
                    for chunk in emit_reasoning_start():
                        yield chunk
                    outcome.reasoning += data.delta_content
                    saw_reasoning_delta = True
                    yield _sse(
                        "response.reasoning_summary_text.delta",
                        item_id=outcome.reasoning_id,
                        output_index=reasoning_output_index,
                        summary_index=0,
                        delta=data.delta_content,
                    )
                else:
                    for chunk in emit_text_start():
                        yield chunk
                    outcome.text += data.delta_content
                    saw_delta = True
                    yield _sse(
                        "response.output_text.delta",
                        item_id=outcome.message_id,
                        output_index=message_output_index,
                        content_index=0,
                        delta=data.delta_content,
                    )
            elif isinstance(data, AssistantMessageData):
                if getattr(data, "parent_tool_call_id", None):
                    if not saw_reasoning_delta and data.content:
                        for chunk in emit_reasoning_start():
                            yield chunk
                        outcome.reasoning += data.content
                        yield _sse(
                            "response.reasoning_summary_text.delta",
                            item_id=outcome.reasoning_id,
                            output_index=reasoning_output_index,
                            summary_index=0,
                            delta=data.content,
                        )
                else:
                    if not saw_delta and data.content:
                        for chunk in emit_text_start():
                            yield chunk
                        outcome.text = data.content
                        yield _sse(
                            "response.output_text.delta",
                            item_id=outcome.message_id,
                            output_index=message_output_index,
                            content_index=0,
                            delta=data.content,
                        )
            elif (
                (SessionShutdownData is not None and isinstance(data, SessionShutdownData))
                or _event_name(event) == "session.shutdown"
            ):
                _record_shutdown_usage(outcome, session.session_id, data)
                if outcome.calls:
                    break
            elif isinstance(data, AssistantUsageData):
                _add_usage(outcome.event_usage, _usage_from_event(data))
            elif isinstance(data, ExternalToolRequestedData):
                outcome.calls.append(_tool_call(data, registration))
            elif isinstance(data, SessionErrorData):
                raise RuntimeError(data.message)
            elif isinstance(data, SessionIdleData):
                break
            if dispatch_task.done():
                error = dispatch_task.exception()
                if error is not None:
                    raise error

        if not is_compact:
            for chunk in emit_reasoning_done():
                yield chunk
            for chunk in emit_text_done():
                yield chunk

        for call in outcome.calls if not is_compact else []:
            call_output_index = output_index
            output_index += 1
            completed_item = _tool_item(session.session_id, call, completed=True)
            started_item = dict(completed_item)
            started_item["status"] = "in_progress"
            field = "input" if call.tool_type == "custom" else "arguments"
            value = completed_item[field]
            started_item[field] = ""
            delta_event = (
                "response.custom_tool_call_input.delta"
                if call.tool_type == "custom"
                else "response.function_call_arguments.delta"
            )
            done_event = (
                "response.custom_tool_call_input.done"
                if call.tool_type == "custom"
                else "response.function_call_arguments.done"
            )
            yield _sse("response.output_item.added", output_index=call_output_index, item=started_item)
            yield _sse(delta_event, item_id=completed_item["id"], output_index=call_output_index, delta=value)
            yield _sse(done_event, item_id=completed_item["id"], output_index=call_output_index, **{field: value})
            yield _sse("response.output_item.done", output_index=call_output_index, item=completed_item)

        _finalize_usage(outcome)
        if is_compact:
            final_payload = to_compaction_payload(body, session.session_id, outcome, response_id)
        else:
            final_payload = _response_payload(body, session.session_id, outcome, response_id)
        # Persist before yielding the terminal event.  A downstream can close
        # the connection as soon as it consumes response.completed, without
        # asking the async generator for another item.
        finish_usage()
        if is_compact:
            compact_item = final_payload["output"][0]
            compact_index = output_index
            yield _sse(
                "response.output_item.added",
                output_index=compact_index,
                item={"type": "compaction", "encrypted_content": None},
            )
            yield _sse(
                "response.output_item.done",
                output_index=compact_index,
                item=compact_item,
            )
        yield _sse(
            "response.completed",
            response=final_payload,
        )
    except asyncio.CancelledError:
        try:
            await asyncio.shield(session.abort())
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        raise
    except Exception as exc:
        yield _sse(
            "response.failed",
            response={
                **base,
                "status": "failed",
                "error": {"code": "sdk_upstream_error", "message": str(exc)},
            },
        )
    finally:
        unsubscribe()
        if not dispatch_task.done():
            dispatch_task.cancel()
        # Record the HTTP turn before any awaited cleanup.  Starlette cancels
        # streaming generators when the caller closes the response (normally
        # after receiving a tool call); cancellation used to interrupt
        # ``disconnect`` and skip this entire lifecycle update.  Those turns
        # consequently appeared in neither Requests nor Sessions.
        _remember_session(session.session_id)
        _commit_alias_watermark(session.session_id, success=final_payload is not None)
        finish_usage()
        try:
            await asyncio.shield(
                _release_session(session, outcome, completed=final_payload is not None)
            )
        except asyncio.CancelledError:
            # The shielded release continues independently; lifecycle
            # reporting above has already completed.
            pass
        except Exception:
            pass


async def handle_responses(
    request: Request,
    body: dict,
    *,
    plan: Any = None,
    is_compact: bool = False,
    finish_usage_callback: Any = None,
    mark_first_output_callback: Any = None,
) -> Response:
    if body.get("input") is None:
        return format_translation.openai_error_response(400, "input is required")
    registration = build_tool_registration(body)
    try:
        session, dispatch = await _open_session(body, registration)
    except Exception as exc:
        if finish_usage_callback is not None and plan is not None:
            try:
                finish_usage_callback(plan, 502, response_text=str(exc))
            except Exception:
                pass
        return format_translation.openai_error_response(502, f"Copilot SDK: {exc}")

    if plan is not None and getattr(plan, "usage_event", None) is not None:
        if not plan.usage_event.get("session_id"):
            plan.usage_event["session_id"] = session.session_id
            plan.usage_event["session_id_origin"] = "copilot_sdk"

    if bool(body.get("stream")):
        return StreamingResponse(
            _stream_turn(
                request,
                body,
                session,
                dispatch,
                registration,
                plan=plan,
                is_compact=is_compact,
                finish_usage_callback=finish_usage_callback,
                mark_first_output_callback=mark_first_output_callback,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    response_id = _new_id("resp")
    succeeded = False
    outcome = None
    try:
        outcome = await _wait_for_outcome(session, dispatch, registration)
        succeeded = True
        if is_compact:
            payload = to_compaction_payload(body, session.session_id, outcome, response_id)
        else:
            payload = _response_payload(body, session.session_id, outcome, response_id)
        if finish_usage_callback is not None and plan is not None:
            try:
                finish_usage_callback(
                    plan,
                    200,
                    response_payload=payload,
                    response_text=outcome.text,
                    reasoning_text=outcome.reasoning,
                    usage=outcome.usage,
                )
            except Exception:
                pass
        return JSONResponse(payload)
    except ValueError as exc:
        if finish_usage_callback is not None and plan is not None:
            try:
                finish_usage_callback(plan, 400, response_text=str(exc))
            except Exception:
                pass
        return format_translation.openai_error_response(400, str(exc))
    except Exception as exc:
        if finish_usage_callback is not None and plan is not None:
            try:
                finish_usage_callback(plan, 502, response_text=str(exc))
            except Exception:
                pass
        return format_translation.openai_error_response(502, f"Copilot SDK: {exc}")
    finally:
        await _release_session(session, outcome, completed=succeeded)
        _remember_session(session.session_id)
        _commit_alias_watermark(session.session_id, success=succeeded)


async def models_response() -> Response:
    try:
        client = await _get_client()
        models = await client.list_models()
    except Exception as exc:
        return format_translation.openai_error_response(502, f"Copilot SDK: {exc}")
    data = [
        {
            "id": model.id,
            "object": "model",
            "created": 0,
            "owned_by": "github-copilot",
        }
        for model in models
    ]
    return JSONResponse(
        excel_upstream.merge_local_models_payload({"object": "list", "data": data})
    )


async def shutdown() -> None:
    """Stop the managed Copilot runtime during application shutdown."""
    global _client, _client_pruned, _client_token
    client = _client
    _client = None
    _client_token = None
    _client_pruned = False
    await _evict_all_live_sessions()
    if client is not None:
        await client.stop()


# ---------------------------------------------------------------------------
# Session Ingestion & Discovery
# ---------------------------------------------------------------------------

_INGEST_CURSOR_FILE = os.path.join(_SDK_STATE_DIR, "session-cursor.json")
_SESSION_STATE_DIR = os.path.join(_SDK_STATE_DIR, "session-state")


def _read_cursor() -> dict:
    if os.path.isfile(_INGEST_CURSOR_FILE):
        try:
            with open(_INGEST_CURSOR_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def _write_cursor(cursor: dict) -> None:
    os.makedirs(_SDK_STATE_DIR, exist_ok=True)
    tmp = _INGEST_CURSOR_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cursor, f, indent=2)
    os.replace(tmp, _INGEST_CURSOR_FILE)


def _parse_session_state_file(session_id: str, events_path: str) -> list[dict]:
    """Parse session-state events.jsonl into individual per-turn usage events."""
    events: list[dict] = []
    try:
        with open(events_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [l for l in f if l.strip()]
    except OSError:
        return []

    model = "copilot-sdk"
    cwd = None
    current_turn = None
    prev_shutdown_usage = None
    prev_api_dur = 0
    turn_index = 0

    for line in lines:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        t = ev.get("type")
        d = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        ts = ev.get("timestamp")

        if t == "session.start":
            model = d.get("selectedModel") or model
            cwd = d.get("context", {}).get("cwd") or cwd
        elif t == "assistant.turn_start":
            current_turn = {
                "turn_index": turn_index,
                "turn_id": d.get("turnId"),
                "interaction_id": d.get("interactionId"),
                "started_at": ts,
                "model": model,
                "last_seen_ts": ts,
            }
            turn_index += 1
        elif t == "assistant.message":
            if d.get("model"):
                model = d.get("model")
            if current_turn is None:
                current_turn = {
                    "turn_index": turn_index,
                    "turn_id": d.get("turnId") or "0",
                    "interaction_id": d.get("interactionId"),
                    "started_at": ts,
                    "model": model,
                    "last_seen_ts": ts,
                }
                turn_index += 1
            else:
                current_turn["model"] = model
                if ts:
                    current_turn["last_seen_ts"] = ts
            if d.get("outputTokens") and "fallback_output_tokens" not in current_turn:
                current_turn["fallback_output_tokens"] = int(d["outputTokens"])
        elif t in {"tool.execution_start", "external_tool.requested"}:
            if current_turn is not None and ts:
                current_turn["last_seen_ts"] = ts
        elif t == "session.shutdown":
            if current_turn is None:
                current_turn = {
                    "turn_index": turn_index,
                    "turn_id": "0",
                    "interaction_id": None,
                    "started_at": ts,
                    "model": model,
                    "last_seen_ts": ts,
                }
                turn_index += 1
            if current_turn is not None:
                finish_ts = ts or current_turn["last_seen_ts"]
                api_dur = int(d.get("totalApiDurationMs", 0) or 0)
                dur = max(0, api_dur - prev_api_dur)
                prev_api_dur = api_dur
                shutdown_usage = _extract_shutdown_usage(d)
                usage_delta = _usage_delta(shutdown_usage, prev_shutdown_usage)
                prev_shutdown_usage = shutdown_usage

                interaction_id = current_turn.get("interaction_id") or f"turn_{current_turn['turn_index']}"
                req_id = f"copilot-sdk:{session_id}:{interaction_id}"
                turn_model = current_turn.get("model") or model
                turn_event = {
                    "request_id": req_id,
                    "started_at": current_turn["started_at"] or finish_ts,
                    "finished_at": finish_ts,
                    "path": "/v1/responses",
                    "method": "POST",
                    "requested_model": turn_model,
                    "resolved_model": turn_model,
                    "response_model": turn_model,
                    "initiator": "user",
                    "session_id": session_id,
                    "session_id_origin": "copilot_sdk",
                    "project_path": cwd,
                    "client_request_id": None,
                    "subagent": None,
                    "server_request_id": session_id,
                    "status_code": 200,
                    "success": True,
                    "duration_ms": dur,
                    "time_to_first_token_ms": None,
                    "usage": usage_delta,
                    "native_source": "copilot_sdk",
                    "native_source_event_key": req_id,
                }
                turn_event["cost_usd"] = util._usage_event_estimated_cost(turn_event, model_name=turn_model, usage=usage_delta)
                events.append(turn_event)
                current_turn = None

    if current_turn is not None and current_turn.get("started_at"):
        finish_ts = current_turn["last_seen_ts"] or current_turn["started_at"]
        dur = 0
        out_tokens = current_turn.get("fallback_output_tokens", 0)
        usage = {
            "input_tokens": 0,
            "output_tokens": out_tokens,
            "total_tokens": out_tokens,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "reasoning_output_tokens": 0,
        }
        interaction_id = current_turn.get("interaction_id") or f"turn_{current_turn['turn_index']}"
        req_id = f"copilot-sdk:{session_id}:{interaction_id}"
        turn_model = current_turn.get("model") or model
        turn_event = {
            "request_id": req_id,
            "started_at": current_turn["started_at"],
            "finished_at": finish_ts,
            "path": "/v1/responses",
            "method": "POST",
            "requested_model": turn_model,
            "resolved_model": turn_model,
            "response_model": turn_model,
            "initiator": "user",
            "session_id": session_id,
            "session_id_origin": "copilot_sdk",
            "project_path": cwd,
            "client_request_id": None,
            "subagent": None,
            "server_request_id": session_id,
            "status_code": 200,
            "success": True,
            "duration_ms": dur,
            "time_to_first_token_ms": None,
            "usage": usage,
            "native_source": "copilot_sdk",
            "native_source_event_key": req_id,
        }
        turn_event["cost_usd"] = util._usage_event_estimated_cost(turn_event, model_name=turn_model, usage=usage)
        events.append(turn_event)

    return events


def scan_session_state(record_callback: Callable[[dict], None]) -> int:
    """Scan session-state directory and emit usage events for completed SDK session turns."""
    if not os.path.isdir(_SESSION_STATE_DIR):
        return 0
    cursor = _read_cursor()
    ingested_count = 0
    dirty = False

    try:
        entries = sorted(os.listdir(_SESSION_STATE_DIR))
    except OSError:
        return 0

    for session_id in entries:
        session_dir = os.path.join(_SESSION_STATE_DIR, session_id)
        if not os.path.isdir(session_dir):
            continue
        # Requests handled by this proxy already report their usage through
        # ``finish_usage_callback``.  The SDK also persists the same
        # session.shutdown record, so ingesting an owned session here would
        # count every turn twice (once under the HTTP request id and once
        # under copilot-sdk:<session>:<interaction>).  The scanner is for SDK
        # sessions created outside the request lifecycle, such as sessions
        # left behind by a crashed/older proxy process.
        if _owns_session(session_id):
            continue
        events_path = os.path.join(session_dir, "events.jsonl")
        if not os.path.isfile(events_path):
            continue

        try:
            mtime = os.path.getmtime(events_path)
            size = os.path.getsize(events_path)
        except OSError:
            continue

        prior = cursor.get(session_id)
        prior_turns = 0
        if isinstance(prior, dict):
            if prior.get("size") == size and prior.get("mtime") == mtime:
                continue
            prior_turns = int(prior.get("ingested_turns", 0) or 0)

        turn_events = _parse_session_state_file(session_id, events_path)
        if not turn_events:
            continue

        new_turns = turn_events[prior_turns:]
        for event in new_turns:
            try:
                record_callback(event)
                ingested_count += 1
            except Exception as exc:
                print(f"copilot_sdk_upstream: failed to record turn {event.get('request_id')}: {exc}", flush=True)

        cursor[session_id] = {
            "size": size,
            "mtime": mtime,
            "ingested_turns": len(turn_events),
            "updated_at": time.time(),
        }
        dirty = True

    if dirty:
        try:
            _write_cursor(cursor)
        except OSError:
            pass

    return ingested_count


def start_background_scanner(
    record_callback: Callable[[dict], None],
    *,
    interval_seconds: float = 10.0,
) -> threading.Thread:
    """Start background thread scanning Copilot SDK session state."""
    def _run():
        while True:
            try:
                scan_session_state(record_callback)
            except Exception as exc:
                print(f"copilot_sdk_upstream: background scanner error: {exc}", flush=True)
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_run, name="CopilotSdkSessionScanner", daemon=True)
    thread.start()
    return thread


__all__ = [
    "REST_UPSTREAM",
    "SDK_UPSTREAM",
    "build_tool_registration",
    "enabled",
    "handle_responses",
    "input_to_prompt",
    "is_compaction_request",
    "models_response",
    "resolve_tool_continuation",
    "responses_upstream",
    "scan_session_state",
    "shutdown",
    "start_background_scanner",
    "to_compaction_payload",
]
