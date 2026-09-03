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
    from copilot.rpc import HandlePendingToolCallRequest
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
_PARALLEL_TOOL_SETTLE_SECONDS = 0.05
_SDK_STATE_DIR = os.path.join(TOKEN_DIR, "copilot-sdk")
_SESSION_LEDGER_FILE = os.path.join(_SDK_STATE_DIR, "proxy-sessions.json")
_SESSION_LEDGER_LOCK = threading.Lock()
_ABANDONED_SESSION_SECONDS = 24 * 60 * 60

_client: Any = None
_client_token: str | None = None
_client_lock: asyncio.Lock | None = None
_client_pruned = False


def _read_session_ledger_unlocked() -> dict[str, float]:
    try:
        with open(_SESSION_LEDGER_FILE, encoding="utf-8") as handle:
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
    os.makedirs(_SDK_STATE_DIR, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix="proxy-sessions-",
        suffix=".tmp",
        dir=_SDK_STATE_DIR,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(ledger, handle, separators=(",", ":"), sort_keys=True)
        os.replace(temporary_path, _SESSION_LEDGER_FILE)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


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


async def _prune_abandoned_sessions(client: Any) -> None:
    cutoff = time.time() - _ABANDONED_SESSION_SECONDS
    with _SESSION_LEDGER_LOCK:
        ledger = _read_session_ledger_unlocked()
    for session_id, updated_at in ledger.items():
        if updated_at >= cutoff:
            continue
        try:
            await client.delete_session(session_id)
        except Exception:
            pass
        _forget_session(session_id)


async def _delete_owned_session(session_id: str) -> None:
    client = _client
    if client is None or not _owns_session(session_id):
        return
    try:
        await client.delete_session(session_id)
    except Exception:
        return
    _forget_session(session_id)


def responses_upstream() -> str:
    """Return the selected Codex Responses upstream (SDK by default in v2)."""
    value = os.getenv(RESPONSES_UPSTREAM_ENV, SDK_UPSTREAM).strip().lower()
    return REST_UPSTREAM if value == REST_UPSTREAM else SDK_UPSTREAM


def enabled() -> bool:
    return responses_upstream() == SDK_UPSTREAM


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
                overrides_built_in_tool=True,
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


def input_to_prompt(value: Any) -> str:
    """Render a full Responses transcript into one SDK user message."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return _text_from_content(value)
    rendered: list[str] = []
    # Truncate to the latest compaction window so pre-compaction history is not replayed.
    window_items = format_translation._latest_compaction_window(value)
    for item in window_items:
        if isinstance(item, str):
            rendered.append(f"User: {item}")
            continue
        if not isinstance(item, dict):
            continue
        if format_translation._is_subagent_notification_message(item):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            continue
        if item_type == "compaction":
            encrypted_content = item.get("encrypted_content")
            summary_text = None
            if isinstance(encrypted_content, str):
                summary_text = format_translation.decode_fake_compaction(encrypted_content)
            if not summary_text:
                summary_text = item.get("output_text") or item.get("summary") or item.get("text")
            if summary_text:
                rendered.append(f"User: {format_translation.FAKE_COMPACTION_SUMMARY_LABEL}\n{summary_text}")
            continue
        if item_type in {"function_call", "custom_tool_call"}:
            payload = item.get("arguments") if item_type == "function_call" else item.get("input")
            rendered.append(f"Assistant tool call {item.get('name', '')}: {_text_from_content(payload)}")
            continue
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            rendered.append(f"Tool result: {_text_from_content(item.get('output'))}")
            continue
        role = item.get("role") or ("assistant" if item_type == "message" else "user")
        text = _text_from_content(item.get("content"))
        if text:
            rendered.append(f"{str(role).capitalize()}: {text}")
    return "\n\n".join(rendered)


@dataclass(frozen=True)
class PendingToolResult:
    session_id: str
    request_id: str
    output: str


def resolve_tool_continuation(value: Any) -> tuple[str, list[PendingToolResult]] | None:
    """Resolve only the trailing tool-result block, excluding older history."""
    if not isinstance(value, list):
        return None
    trailing: list[PendingToolResult] = []
    for index, item in enumerate(reversed(value)):
        if not isinstance(item, dict) or item.get("type") not in {
            "function_call_output",
            "custom_tool_call_output",
        }:
            if index == 0:
                return None
            break
        decoded = _decode_call_id(item.get("call_id"))
        if decoded is None:
            return None
        trailing.append(
            PendingToolResult(
                session_id=decoded["s"],
                request_id=decoded["r"],
                output=_text_from_content(item.get("output")),
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
    return effort if effort in {"low", "medium", "high", "xhigh"} else None


def _reasoning_summary(body: dict) -> str:
    """Map Responses reasoning settings to the SDK's summary modes."""
    reasoning = body.get("reasoning")
    summary = reasoning.get("summary") if isinstance(reasoning, dict) else None
    if summary in {"none", "concise", "detailed"}:
        return summary
    return "detailed"


def _session_options(body: dict, registration: ToolRegistration) -> dict[str, Any]:
    instructions = body.get("instructions")
    options: dict[str, Any] = {
        "model": body.get("model") if isinstance(body.get("model"), str) else None,
        "reasoning_effort": _reasoning_effort(body),
        # The SDK does not emit Copilot's reasoning/intent timeline events
        # unless a reasoning summary mode is selected.  Without this, Codex
        # receives only the final answer and tool calls.
        "reasoning_summary": _reasoning_summary(body),
        "streaming": bool(body.get("stream")),
        "tools": registration.tools,
        "available_tools": ["custom:*"],
        "include_sub_agent_streaming_events": True,
        "on_permission_request": PermissionHandler.approve_all,
        "infinite_sessions": {"enabled": False},
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


async def _open_session(body: dict, registration: ToolRegistration):
    client = await _get_client()
    continuation = resolve_tool_continuation(body.get("input"))
    options = _session_options(body, registration)
    if continuation is None:
        session = await client.create_session(**options)
        _remember_session(session.session_id)

        async def dispatch() -> None:
            prompt = input_to_prompt(body.get("input"))
            if not prompt:
                raise ValueError("input must contain at least one text message")
            await session.send(prompt)

        return session, dispatch

    session_id, results = continuation
    if not _owns_session(session_id):
        raise ValueError("Tool continuation does not belong to this proxy")
    _remember_session(session_id)
    session = await client.resume_session(
        session_id,
        continue_pending_work=True,
        **options,
    )

    async def dispatch() -> None:
        for result in results:
            await session.rpc.tools.handle_pending_tool_call(
                HandlePendingToolCallRequest(
                    request_id=result.request_id,
                    result=result.output,
                )
            )

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
    reasoning_tokens = int(getattr(data, "reasoning_tokens", 0) or 0)
    fresh = max(0, input_tokens - cached_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_input_tokens": cached_tokens,
        "cache_creation_input_tokens": 0,
        "fresh_input_tokens": fresh,
        "pricing_fresh_input_tokens": fresh,
        "pricing_cached_input_tokens": cached_tokens,
        "pricing_cache_creation_input_tokens": 0,
        "reasoning_output_tokens": reasoning_tokens,
    }


def _extract_shutdown_usage(data: Any) -> dict[str, int]:
    """Extract cumulative token counts from SessionShutdownData or equivalent dict."""
    total_inp = 0
    cread = 0
    cwrite = 0
    out = 0
    reas = 0
    found_mm = False

    mm = getattr(data, "model_metrics", None) or (data.get("modelMetrics") if isinstance(data, dict) else None)
    if isinstance(mm, dict):
        for m_val in mm.values():
            u = getattr(m_val, "usage", None) or (m_val.get("usage") if isinstance(m_val, dict) else None)
            if u is not None:
                found_mm = True
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

    td = getattr(data, "token_details", None) or (data.get("tokenDetails") if isinstance(data, dict) else None)
    if isinstance(td, dict):
        inp_tok = getattr(td.get("input"), "token_count", None) if hasattr(td.get("input"), "token_count") else (td.get("input", {}).get("tokenCount") if isinstance(td.get("input"), dict) else 0)
        read_tok = getattr(td.get("cache_read"), "token_count", None) if hasattr(td.get("cache_read"), "token_count") else (td.get("cache_read", {}).get("tokenCount") if isinstance(td.get("cache_read"), dict) else 0)
        write_tok = getattr(td.get("cache_write"), "token_count", None) if hasattr(td.get("cache_write"), "token_count") else (td.get("cache_write", {}).get("tokenCount") if isinstance(td.get("cache_write"), dict) else 0)
        out_tok = getattr(td.get("output"), "token_count", None) if hasattr(td.get("output"), "token_count") else (td.get("output", {}).get("tokenCount") if isinstance(td.get("output"), dict) else 0)

        i = int(inp_tok or 0)
        r = int(read_tok or 0)
        w = int(write_tok or 0)
        o = int(out_tok or 0)
        if total_inp == 0:
            total_inp = i + r + w
        if cread == 0:
            cread = r
        if cwrite == 0:
            cwrite = w
        if out == 0:
            out = o

    fresh = max(0, total_inp - cread - cwrite)

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
    fresh = max(0, inp - cread - cwrite)
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


_session_last_shutdown_usage: dict[str, dict[str, int]] = {}


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
                current_shutdown = _extract_shutdown_usage(data)
                prev_shutdown = _session_last_shutdown_usage.get(session.session_id)
                outcome.usage = _usage_delta(current_shutdown, prev_shutdown)
                _session_last_shutdown_usage[session.session_id] = current_shutdown
            elif isinstance(data, AssistantUsageData):
                outcome.usage = _usage_from_event(data)
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
            elif (SessionCompactionCompleteData is not None and isinstance(data, SessionCompactionCompleteData)) or _event_name(event) == "session.compaction_complete":
                summary = getattr(data, "summary_content", None)
                if summary and not outcome.text:
                    outcome.text = summary
            elif isinstance(data, SessionErrorData):
                raise RuntimeError(data.message)
            elif isinstance(data, SessionIdleData):
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
        if isinstance(call.arguments, dict) and isinstance(call.arguments.get("input"), str):
            return call.arguments["input"]
        return _text_from_content(call.arguments)
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
        "usage": outcome.usage or {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
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
        "usage": outcome.usage or {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
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

    reasoning_started = False
    reasoning_closed = False
    reasoning_output_index = 0
    saw_reasoning_delta = False

    message_started = False
    message_closed = False
    message_output_index = 0
    saw_delta = False

    first_output_marked = False

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
            elif (SessionCompactionCompleteData is not None and isinstance(data, SessionCompactionCompleteData)) or _event_name(event) == "session.compaction_complete":
                summary = getattr(data, "summary_content", None)
                if summary and not outcome.text:
                    outcome.text = summary
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
                current_shutdown = _extract_shutdown_usage(data)
                prev_shutdown = _session_last_shutdown_usage.get(session.session_id)
                outcome.usage = _usage_delta(current_shutdown, prev_shutdown)
                _session_last_shutdown_usage[session.session_id] = current_shutdown
            elif isinstance(data, AssistantUsageData):
                outcome.usage = _usage_from_event(data)
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

        for chunk in emit_reasoning_done():
            yield chunk
        for chunk in emit_text_done():
            yield chunk

        for call in outcome.calls:
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

        if is_compact:
            final_payload = to_compaction_payload(body, session.session_id, outcome, response_id)
        else:
            final_payload = _response_payload(body, session.session_id, outcome, response_id)
        yield _sse(
            "response.completed",
            response=final_payload,
        )
    except asyncio.CancelledError:
        await session.abort()
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
        await session.disconnect()
        _remember_session(session.session_id)
        if finish_usage_callback is not None and plan is not None:
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
    try:
        outcome = await _wait_for_outcome(session, dispatch, registration)
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
        await session.disconnect()
        _remember_session(session.session_id)


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
    "models_response",
    "resolve_tool_continuation",
    "responses_upstream",
    "scan_session_state",
    "shutdown",
    "start_background_scanner",
    "to_compaction_payload",
]
