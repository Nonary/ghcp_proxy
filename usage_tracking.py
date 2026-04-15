"""Usage event lifecycle, session/request tracking, persistence, archival, and SSE usage capture."""

import hashlib
import json
import os
import tempfile
import time
from collections import deque
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Callable
from uuid import uuid4

import httpx
from fastapi import Request

from constants import (
    TOKEN_DIR, USAGE_LOG_FILE, REQUEST_ERROR_LOG_FILE,
    DETAILED_REQUEST_HISTORY_LIMIT,
)
from util import (
    _json_default, _coerce_float, _coerce_int,
    utc_now, utc_now_iso,
    normalize_usage_payload, _normalize_model_name,
    _usage_event_model_name, _usage_event_source,
    _usage_event_cost, _premium_request_multiplier, _counted_premium_requests,
    _server_request_chain_key, _is_claude_request,
    extract_item_text, _parse_iso_datetime,
    _extract_payload_usage,
)
from event_bus import EventBus


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUEST_FINISHED_EVENT = "request_finished"
USAGE_EVENT_RECORDED_EVENT = "usage_event_recorded"


# ---------------------------------------------------------------------------
# State and archive store dataclasses
# ---------------------------------------------------------------------------

@dataclass
class UsageArchiveStore:
    init_storage: Callable[[], bool] = lambda: False
    lock: object = field(default_factory=Lock)
    connect: Callable[[], object] = lambda: None
    mark_unavailable: Callable[[str], None] = lambda error: None


@dataclass
class UsageTrackingState:
    usage_log_lock: object = field(default_factory=Lock)
    recent_usage_events: deque[dict] = field(default_factory=deque)
    archived_usage_events: list[dict] = field(default_factory=list)
    session_request_id_lock: object = field(default_factory=Lock)
    latest_server_request_ids_by_chain: dict[tuple[str, str], str] = field(default_factory=dict)
    active_server_request_ids_by_request: dict[str, dict[str, str | None]] = field(default_factory=dict)
    latest_claude_user_session_contexts: dict[tuple[str, str], dict[str, str | None]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure helper functions (module-level)
# ---------------------------------------------------------------------------

def request_session_id(request: Request, request_body: dict | None = None) -> str | None:
    for header_name in (
        "session_id",
        "session-id",
        "x-claude-code-session-id",
        "x-session-affinity",
        "x-opencode-session",
    ):
        header_value = request.headers.get(header_name)
        if isinstance(header_value, str):
            normalized = header_value.strip()
            if normalized:
                return normalized

    if isinstance(request_body, dict):
        for key in ("session_id", "sessionId", "prompt_cache_key", "promptCacheKey"):
            value = request_body.get(key)
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    return normalized
        metadata = request_body.get("metadata")
        if isinstance(metadata, dict):
            user_id = metadata.get("user_id")
            user_id_payload = None
            if isinstance(user_id, str):
                normalized = user_id.strip()
                if normalized:
                    try:
                        user_id_payload = json.loads(normalized)
                    except json.JSONDecodeError:
                        user_id_payload = None
            elif isinstance(user_id, dict):
                user_id_payload = user_id
            if isinstance(user_id_payload, dict):
                for key in ("session_id", "sessionId"):
                    value = user_id_payload.get(key)
                    if isinstance(value, str):
                        normalized = value.strip()
                        if normalized:
                            return normalized

    return None


def _claude_session_scope_key(
    client_request_id: str | None,
    subagent: str | None,
) -> tuple[str, str]:
    scope = client_request_id if isinstance(client_request_id, str) and client_request_id else "__global__"
    normalized_subagent = subagent if isinstance(subagent, str) and subagent else "__root__"
    return (scope, normalized_subagent)


def _apply_missing_claude_session_context(event: dict | None) -> dict | None:
    if not isinstance(event, dict):
        return event
    existing_session_id = event.get("session_id")
    if isinstance(existing_session_id, str) and existing_session_id:
        return event
    if _usage_event_source(event) != "claude":
        return event

    existing_session_id = event.get("session_id")
    if isinstance(existing_session_id, str) and existing_session_id:
        return event

    return event


def _normalize_recorded_usage_event(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return None

    normalized_event = dict(payload)
    normalized_usage = normalize_usage_payload(normalized_event.get("usage"))
    if isinstance(normalized_usage, dict):
        normalized_event["usage"] = normalized_usage
        if normalized_event.get("cost_usd") is None:
            normalized_event["cost_usd"] = _usage_event_cost(_usage_event_model_name(normalized_event), normalized_usage)
    _apply_missing_claude_session_context(normalized_event)
    return normalized_event


def _usage_event_archive_summary(event: dict) -> dict:
    summary = {
        "request_id": event.get("request_id"),
        "started_at": event.get("started_at"),
        "finished_at": event.get("finished_at"),
        "path": event.get("path"),
        "requested_model": event.get("requested_model"),
        "resolved_model": event.get("resolved_model"),
        "initiator": event.get("initiator"),
        "session_id": event.get("session_id"),
        "project_path": event.get("project_path"),
        "client_request_id": event.get("client_request_id"),
        "subagent": event.get("subagent"),
        "server_request_id": event.get("server_request_id"),
        "status_code": event.get("status_code"),
        "success": event.get("success"),
        "premium_requests": _counted_premium_requests(event),
        "cost_usd": round(_coerce_float(event.get("cost_usd")), 6),
    }

    normalized_usage = normalize_usage_payload(event.get("usage"))
    if isinstance(normalized_usage, dict):
        summary["usage"] = normalized_usage

    return summary


def _usage_event_archive_key(summary: dict) -> str:
    request_id = summary.get("request_id")
    if isinstance(request_id, str) and request_id:
        return f"request:{request_id}"
    serialized = json.dumps(summary, sort_keys=True, separators=(",", ":"), default=_json_default)
    return f"summary:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def _initiator_log_label(initiator: str | None) -> str:
    return "Agent" if initiator == "agent" else "User"


# ---------------------------------------------------------------------------
# SSE capture class
# ---------------------------------------------------------------------------

class SSEUsageCapture:
    def __init__(self, stream_type: str):
        self.stream_type = stream_type
        self.buffer = ""
        self.usage = None

    def _has_text(self, value) -> bool:
        if not isinstance(value, str):
            return False
        return bool(value.strip())

    def _consume_chat_payload(self, payload: dict) -> bool:
        if isinstance(payload.get("usage"), dict):
            self.usage = normalize_usage_payload(payload["usage"])

        choices = payload.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        delta = first_choice.get("delta") if isinstance(first_choice, dict) else {}
        from format_translation import extract_text_from_chat_delta
        return self._has_text(extract_text_from_chat_delta(delta))

    def _consume_responses_payload(self, payload: dict) -> bool:
        event_type = str(payload.get("type", "")).strip().lower()
        response = payload.get("response")
        if isinstance(response, dict):
            if isinstance(response.get("usage"), dict):
                self.usage = normalize_usage_payload(response["usage"])
        elif isinstance(payload.get("usage"), dict):
            self.usage = normalize_usage_payload(payload["usage"])

        has_output = False
        if event_type == "response.output_text.delta":
            has_output = self._has_text(payload.get("delta"))
        if event_type == "response.output_text.done":
            has_output = self._has_text(payload.get("text"))
        if event_type == "response.output_item.added":
            item = payload.get("item")
            if isinstance(item, dict):
                has_output = self._has_text(extract_item_text(item))
        if event_type == "response.content_part.added":
            part = payload.get("part")
            if isinstance(part, dict):
                has_output = self._has_text(
                    part.get("text") or part.get("input_text") or part.get("output_text")
                )
        return has_output

    def feed(self, chunk) -> bool:
        if isinstance(chunk, bytes):
            text = chunk.decode("utf-8", errors="replace")
        else:
            text = str(chunk)

        self.buffer += text
        normalized = self.buffer.replace("\r\n", "\n")
        saw_output = False

        while "\n\n" in normalized:
            raw_block, normalized = normalized.split("\n\n", 1)
            from format_translation import parse_sse_block
            _event_name, data = parse_sse_block(raw_block)
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if self.stream_type == "chat":
                saw_output = self._consume_chat_payload(payload) or saw_output
            else:
                saw_output = self._consume_responses_payload(payload) or saw_output

        self.buffer = normalized
        return saw_output


# ---------------------------------------------------------------------------
# UsageTracker — owns its state directly
# ---------------------------------------------------------------------------

class UsageTracker:
    """
    Self-contained usage tracker that owns its state, archive store, and
    event publishing.  All mutable state lives on ``self.state`` and
    ``self.archive_store``; there are no module-level globals.
    """

    def __init__(
        self,
        *,
        state: UsageTrackingState | None = None,
        archive_store: UsageArchiveStore | None = None,
        event_bus: EventBus | None = None,
        usage_log_file: str | None = None,
        error_log_file: str | None = None,
        on_request_finished: Callable | None = None,
        on_usage_event_recorded: Callable | None = None,
    ):
        self.state = state or UsageTrackingState()
        self.archive_store = archive_store or UsageArchiveStore()
        self.event_bus = event_bus
        self.usage_log_file = usage_log_file or USAGE_LOG_FILE
        self.error_log_file = error_log_file or REQUEST_ERROR_LOG_FILE
        self.on_request_finished = on_request_finished
        self.on_usage_event_recorded = on_usage_event_recorded

    # ------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------

    def _publish_event(self, event_name: str, *args, **kwargs):
        if self.event_bus is not None:
            self.event_bus.publish(event_name, *args, **kwargs)

    # ------------------------------------------------------------------
    # Delegating helpers (pure functions stay module-level)
    # ------------------------------------------------------------------

    def request_session_id(self, request: Request, request_body: dict | None = None) -> str | None:
        return request_session_id(request, request_body)

    def create_sse_capture(self, stream_type: str) -> SSEUsageCapture:
        return SSEUsageCapture(stream_type)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def clear_state(self):
        with self.state.usage_log_lock:
            self.state.recent_usage_events.clear()
            self.state.archived_usage_events.clear()
        with self.state.session_request_id_lock:
            self.state.latest_server_request_ids_by_chain.clear()
            self.state.active_server_request_ids_by_request.clear()
            self.state.latest_claude_user_session_contexts.clear()

    def replace_history(
        self,
        *,
        recent_events: list[dict] | None = None,
        archived_events: list[dict] | None = None,
    ):
        normalized_recent = [
            normalized
            for event in (recent_events or [])
            if (normalized := _normalize_recorded_usage_event(event)) is not None
        ]
        normalized_archived = [
            normalized
            for event in (archived_events or [])
            if (normalized := _normalize_recorded_usage_event(event)) is not None
        ]
        with self.state.usage_log_lock:
            self.state.recent_usage_events.clear()
            self.state.recent_usage_events.extend(normalized_recent)
            self.state.archived_usage_events.clear()
            self.state.archived_usage_events.extend(normalized_archived)

    def snapshot_archived_usage_events(self) -> list[dict]:
        with self.state.usage_log_lock:
            return list(self.state.archived_usage_events)

    def remember_latest_server_request_id(
        self,
        session_id: str | None,
        client_request_id: str | None,
        subagent: str | None,
        server_request_id: str | None,
    ):
        if not isinstance(server_request_id, str) or not server_request_id:
            return
        chain_key = _server_request_chain_key(session_id, client_request_id, subagent)
        with self.state.session_request_id_lock:
            self.state.latest_server_request_ids_by_chain[chain_key] = server_request_id

    # ------------------------------------------------------------------
    # Session / request context tracking (private methods)
    # ------------------------------------------------------------------

    def _remember_latest_claude_user_session_context(self, event: dict | None):
        if not isinstance(event, dict):
            return
        if _usage_event_source(event) != "claude":
            return
        if event.get("initiator") != "user":
            return
        session_id = event.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return
        scope_key = _claude_session_scope_key(
            event.get("client_request_id"),
            event.get("subagent"),
        )
        with self.state.session_request_id_lock:
            self.state.latest_claude_user_session_contexts[scope_key] = {
                "session_id": session_id,
                "session_id_origin": event.get("session_id_origin"),
                "client_request_id": event.get("client_request_id"),
                "subagent": event.get("subagent"),
                "server_request_id": event.get("server_request_id"),
            }

    def _get_latest_claude_user_session_context(
        self,
        client_request_id: str | None,
        subagent: str | None,
    ) -> dict[str, str | None] | None:
        scope_key = _claude_session_scope_key(client_request_id, subagent)
        with self.state.session_request_id_lock:
            context = self.state.latest_claude_user_session_contexts.get(scope_key)
            return dict(context) if isinstance(context, dict) else None

    def _remember_server_request_id(self, event: dict | None):
        if not isinstance(event, dict):
            return
        chain_key = _server_request_chain_key(
            event.get("session_id"),
            event.get("client_request_id"),
            event.get("subagent"),
        )
        server_request_id = event.get("server_request_id")
        if not isinstance(server_request_id, str) or not server_request_id:
            return
        with self.state.session_request_id_lock:
            self.state.latest_server_request_ids_by_chain[chain_key] = server_request_id

    def _remember_active_server_request_id(self, event: dict | None):
        if not isinstance(event, dict):
            return
        request_id = event.get("request_id")
        server_request_id = event.get("server_request_id")
        if not isinstance(request_id, str) or not request_id:
            return
        if not isinstance(server_request_id, str) or not server_request_id:
            return
        with self.state.session_request_id_lock:
            self.state.active_server_request_ids_by_request[request_id] = {
                "session_id": event.get("session_id"),
                "session_id_origin": event.get("session_id_origin"),
                "client_request_id": event.get("client_request_id"),
                "subagent": event.get("subagent"),
                "initiator": event.get("initiator"),
                "server_request_id": server_request_id,
            }

    def _forget_active_server_request_id(self, request_id: str | None):
        if not isinstance(request_id, str) or not request_id:
            return
        with self.state.session_request_id_lock:
            self.state.active_server_request_ids_by_request.pop(request_id, None)

    def _get_active_server_request_id(
        self,
        session_id: str | None,
        client_request_id: str | None,
        subagent: str | None,
        *,
        initiator: str | None = None,
    ) -> str | None:
        target_subagent = subagent if isinstance(subagent, str) and subagent else None
        with self.state.session_request_id_lock:
            for context in reversed(list(self.state.active_server_request_ids_by_request.values())):
                context_subagent = context.get("subagent")
                if isinstance(context_subagent, str):
                    context_subagent = context_subagent or None
                else:
                    context_subagent = None
                if context_subagent != target_subagent:
                    continue
                if initiator is not None and context.get("initiator") != initiator:
                    continue
                if isinstance(session_id, str) and session_id:
                    if context.get("session_id") != session_id:
                        continue
                elif isinstance(client_request_id, str) and client_request_id:
                    if context.get("client_request_id") != client_request_id:
                        continue
                elif target_subagent is not None:
                    continue
                server_request_id = context.get("server_request_id")
                if isinstance(server_request_id, str) and server_request_id:
                    return server_request_id
        return None

    def _get_active_request_context(
        self,
        session_id: str | None,
        client_request_id: str | None,
        subagent: str | None,
        *,
        initiator: str | None = None,
    ) -> dict[str, str | None] | None:
        target_subagent = subagent if isinstance(subagent, str) and subagent else None
        with self.state.session_request_id_lock:
            for context in reversed(list(self.state.active_server_request_ids_by_request.values())):
                context_subagent = context.get("subagent")
                if isinstance(context_subagent, str):
                    context_subagent = context_subagent or None
                else:
                    context_subagent = None
                if context_subagent != target_subagent:
                    continue
                if initiator is not None and context.get("initiator") != initiator:
                    continue
                if isinstance(session_id, str) and session_id:
                    if context.get("session_id") != session_id:
                        continue
                elif isinstance(client_request_id, str) and client_request_id:
                    if context.get("client_request_id") != client_request_id:
                        continue
                elif target_subagent is not None:
                    continue
                return dict(context)
        return None

    def _get_latest_server_request_id(
        self,
        session_id: str | None,
        client_request_id: str | None,
        subagent: str | None,
    ) -> str | None:
        chain_key = _server_request_chain_key(session_id, client_request_id, subagent)
        with self.state.session_request_id_lock:
            return self.state.latest_server_request_ids_by_chain.get(chain_key)

    def _resolve_server_request_id(
        self,
        request: Request,
        initiator: str | None,
        request_body: dict | None = None,
        *,
        session_id: str | None = None,
        client_request_id: str | None = None,
        subagent: str | None = None,
        allow_user_active_fallback: bool = False,
    ) -> tuple[str, str | None]:
        explicit_server_request_id = (
            request.headers.get("x-request-id")
            or request.headers.get("request-id")
            or request.headers.get("x-github-request-id")
        )
        if explicit_server_request_id:
            return explicit_server_request_id, explicit_server_request_id

        if session_id is None:
            session_id = request_session_id(request, request_body)
        if client_request_id is None:
            client_request_id = request.headers.get("x-client-request-id")
        if subagent is None:
            subagent = request.headers.get("x-openai-subagent")

        has_chain_context = any(
            isinstance(value, str) and value
            for value in (session_id, client_request_id, subagent)
        )

        if initiator == "agent":
            active_server_request_id = self._get_active_server_request_id(
                session_id,
                client_request_id,
                subagent,
                initiator="user",
            )
            if isinstance(active_server_request_id, str) and active_server_request_id:
                return active_server_request_id, active_server_request_id

        if initiator == "user" and allow_user_active_fallback and has_chain_context:
            active_server_request_id = self._get_active_server_request_id(
                session_id,
                client_request_id,
                subagent,
                initiator="user",
            )
            if isinstance(active_server_request_id, str) and active_server_request_id:
                return active_server_request_id, active_server_request_id

        if session_id is not None:
            prior_server_request_id = self._get_latest_server_request_id(
                session_id,
                client_request_id,
                subagent,
            )
            if isinstance(prior_server_request_id, str) and prior_server_request_id:
                return prior_server_request_id, prior_server_request_id

        generated_server_request_id = str(uuid4())
        return generated_server_request_id, explicit_server_request_id

    # ------------------------------------------------------------------
    # Persistence (private methods)
    # ------------------------------------------------------------------

    def _rewrite_usage_log(self, events: list[dict]):
        log_dir = os.path.dirname(self.usage_log_file) or TOKEN_DIR
        os.makedirs(log_dir, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(prefix="usage-log-", suffix=".jsonl", dir=log_dir)
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
                for event in events:
                    temp_file.write(json.dumps(event, separators=(",", ":"), default=_json_default))
                    temp_file.write("\n")
            os.replace(temp_path, self.usage_log_file)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _delete_archived_events(self, keys: list[str]):
        if not keys or not self.archive_store.init_storage():
            return
        try:
            with self.archive_store.lock:
                with closing(self.archive_store.connect()) as connection:
                    connection.executemany(
                        "DELETE FROM archived_usage_events WHERE archive_key = ?",
                        [(key,) for key in keys],
                    )
                    connection.commit()
        except Exception as exc:
            self.archive_store.mark_unavailable(str(exc))

    def _persist_event(self, event: dict):
        if not isinstance(event, dict):
            return

        os.makedirs(os.path.dirname(self.usage_log_file) or TOKEN_DIR, exist_ok=True)
        serialized = json.dumps(event, separators=(",", ":"), default=_json_default)
        with self.state.usage_log_lock:
            with open(self.usage_log_file, "a", encoding="utf-8") as f:
                f.write(serialized)
                f.write("\n")
            self.state.recent_usage_events.append(event)
        self._compact_if_needed()
        self._remember_server_request_id(event)
        self._remember_latest_claude_user_session_context(event)
        self._publish_event(USAGE_EVENT_RECORDED_EVENT, event)
        if self.on_usage_event_recorded is not None:
            self.on_usage_event_recorded(event)

    # ------------------------------------------------------------------
    # Archival
    # ------------------------------------------------------------------

    def load_archived_history(self):
        if not self.archive_store.init_storage():
            return

        try:
            with self.archive_store.lock:
                with closing(self.archive_store.connect()) as connection:
                    rows = connection.execute(
                        "SELECT payload_json FROM archived_usage_events ORDER BY recorded_at ASC"
                    ).fetchall()
        except Exception as exc:
            self.archive_store.mark_unavailable(str(exc))
            return

        self.state.archived_usage_events.clear()
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                continue
            normalized_event = _normalize_recorded_usage_event(payload)
            if normalized_event is not None:
                self.state.archived_usage_events.append(normalized_event)

    def _compact_if_needed(self):
        with self.state.usage_log_lock:
            overflow = len(self.state.recent_usage_events) - DETAILED_REQUEST_HISTORY_LIMIT
            if overflow <= 0:
                return
            if not self.archive_store.init_storage():
                return

            detailed_events = list(self.state.recent_usage_events)
            events_to_archive = detailed_events[:overflow]
            remaining_events = detailed_events[overflow:]
            archive_rows = []
            archived_summaries = []
            archive_keys = []
            for event in events_to_archive:
                summary = _usage_event_archive_summary(event)
                archive_key = _usage_event_archive_key(summary)
                recorded_at = summary.get("finished_at") or summary.get("started_at") or utc_now_iso()
                archive_rows.append(
                    (
                        archive_key,
                        recorded_at,
                        json.dumps(summary, separators=(",", ":"), default=_json_default),
                    )
                )
                archived_summaries.append(summary)
                archive_keys.append(archive_key)

            try:
                with self.archive_store.lock:
                    with closing(self.archive_store.connect()) as connection:
                        connection.executemany(
                            """
                            INSERT INTO archived_usage_events (archive_key, recorded_at, payload_json)
                            VALUES (?, ?, ?)
                            ON CONFLICT(archive_key) DO NOTHING
                            """,
                            archive_rows,
                        )
                        connection.commit()
                self._rewrite_usage_log(remaining_events)
            except Exception:
                self._delete_archived_events(archive_keys)
                return

            self.state.recent_usage_events.clear()
            self.state.recent_usage_events.extend(remaining_events)
            for summary in archived_summaries:
                normalized_event = _normalize_recorded_usage_event(summary)
                if normalized_event is not None:
                    self.state.archived_usage_events.append(normalized_event)

    def load_history(self):
        if not os.path.exists(self.usage_log_file):
            return

        try:
            with self.state.usage_log_lock:
                with open(self.usage_log_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        normalized_event = _normalize_recorded_usage_event(payload)
                        if normalized_event is None:
                            continue
                        self.state.recent_usage_events.append(normalized_event)
                        self._remember_server_request_id(normalized_event)
                        self._remember_latest_claude_user_session_context(normalized_event)
        except OSError:
            pass
        self._compact_if_needed()

    # ------------------------------------------------------------------
    # Usage event lifecycle
    # ------------------------------------------------------------------

    def start_event(
        self,
        request: Request,
        requested_model: str | None,
        resolved_model: str | None,
        initiator: str | None,
        request_id: str | None = None,
        request_body: dict | None = None,
        upstream_path: str | None = None,
        outbound_headers: dict | None = None,
    ) -> dict:
        # Log the proxy request (absorbed from log_proxy_request)
        parts = [
            "INFO:",
            f"Proxy request ({_initiator_log_label(initiator)}):",
            f"{request.method} {request.url.path}",
        ]
        if requested_model is not None:
            parts.append(f"requested_model={requested_model}")
        if resolved_model is not None and resolved_model != requested_model:
            parts.append(f"resolved_model={resolved_model}")
        print(" ".join(parts), flush=True)

        event_request_id = request_id or uuid4().hex
        client_request_id = request.headers.get("x-client-request-id")
        if not client_request_id and isinstance(outbound_headers, dict):
            outbound_client_request_id = outbound_headers.get("x-client-request-id")
            if isinstance(outbound_client_request_id, str):
                normalized_outbound_client_request_id = outbound_client_request_id.strip()
                if normalized_outbound_client_request_id:
                    client_request_id = normalized_outbound_client_request_id
        subagent = request.headers.get("x-openai-subagent")
        is_claude_request = _is_claude_request(request)
        has_chain_context = any(
            isinstance(value, str) and value
            for value in (client_request_id, subagent)
        )
        session_id = request_session_id(request, request_body)
        project_path = None
        session_id_origin = "request" if session_id else None

        if not session_id and is_claude_request:
            allow_active_session_attach = initiator == "agent" or has_chain_context
            if allow_active_session_attach:
                active_user_context = self._get_active_request_context(
                    session_id,
                    client_request_id,
                    subagent,
                    initiator="user",
                )
                if not isinstance(active_user_context, dict):
                    active_user_context = self._get_latest_claude_user_session_context(
                        client_request_id,
                        subagent,
                    )
                if isinstance(active_user_context, dict):
                    active_session_id = active_user_context.get("session_id")
                    if isinstance(active_session_id, str) and active_session_id:
                        session_id = active_session_id
                        session_id_origin = active_user_context.get("session_id_origin") or "request_id"

        if not session_id and initiator == "user" and is_claude_request:
            session_id = event_request_id
            session_id_origin = "request_id"

        server_request_id, prior_server_request_id = self._resolve_server_request_id(
            request,
            initiator,
            request_body,
            session_id=session_id,
            client_request_id=client_request_id,
            subagent=subagent,
            allow_user_active_fallback=is_claude_request and initiator == "user",
        )
        if isinstance(outbound_headers, dict):
            if session_id:
                outbound_headers["session_id"] = session_id
            outbound_headers["x-request-id"] = server_request_id
            outbound_headers["x-github-request-id"] = server_request_id
        started_at = utc_now_iso()
        event = {
            "request_id": event_request_id,
            "started_at": started_at,
            "path": request.url.path,
            "method": request.method,
            "upstream_path": upstream_path,
            "requested_model": requested_model,
            "resolved_model": resolved_model or requested_model,
            "initiator": initiator,
            "session_id": session_id,
            "session_id_origin": session_id_origin,
            "project_path": project_path,
            "client_request_id": client_request_id,
            "subagent": subagent,
            "server_request_id": server_request_id,
            "prior_server_request_id": prior_server_request_id,
            "_started_monotonic": time.perf_counter(),
        }
        self._remember_server_request_id(event)
        self._remember_active_server_request_id(event)
        return event

    def mark_first_output(self, event: dict | None):
        if not isinstance(event, dict):
            return
        if event.get("_first_output_monotonic") is None:
            event["_first_output_monotonic"] = time.perf_counter()

    def finish_event(
        self,
        event: dict | None,
        status_code: int,
        *,
        upstream: httpx.Response | None = None,
        response_payload: dict | None = None,
        response_text: str | None = None,
        usage: dict | None = None,
    ):
        if not isinstance(event, dict):
            return

        finished_at = utc_now()
        self._forget_active_server_request_id(event.get("request_id"))
        self._publish_event(
            REQUEST_FINISHED_EVENT,
            event.get("request_id"),
            finished_at=finished_at,
        )
        if self.on_request_finished is not None:
            self.on_request_finished(event.get("request_id"), finished_at=finished_at)
        finished_event = {
            **{key: value for key, value in event.items() if not str(key).startswith("_")},
            "finished_at": finished_at.isoformat(),
            "status_code": status_code,
            "success": status_code < 400,
        }

        started_monotonic = event.get("_started_monotonic")
        if isinstance(started_monotonic, (int, float)):
            finished_event["duration_ms"] = max(0, int(round((time.perf_counter() - started_monotonic) * 1000)))

        first_output_monotonic = event.get("_first_output_monotonic")
        if isinstance(started_monotonic, (int, float)) and isinstance(first_output_monotonic, (int, float)):
            finished_event["time_to_first_token_ms"] = max(0, int(round((first_output_monotonic - started_monotonic) * 1000)))

        if upstream is not None:
            for header_name in ("x-request-id", "request-id", "x-github-request-id"):
                header_value = upstream.headers.get(header_name)
                if header_value:
                    finished_event["upstream_request_id"] = header_value
                    break
            content_type = upstream.headers.get("content-type")
            if content_type:
                finished_event["response_content_type"] = content_type

        if isinstance(response_payload, dict):
            payload_response_id = response_payload.get("id")
            if isinstance(payload_response_id, str):
                finished_event["response_id"] = payload_response_id
            payload_model = response_payload.get("model")
            if isinstance(payload_model, str):
                finished_event["response_model"] = payload_model

        derived_usage = usage
        if isinstance(derived_usage, dict):
            derived_usage = normalize_usage_payload(derived_usage)
        if derived_usage is None and isinstance(response_payload, dict):
            derived_usage = _extract_payload_usage(response_payload)
        if isinstance(derived_usage, dict):
            finished_event["usage"] = derived_usage

        model_name = (
            finished_event.get("response_model")
            or finished_event.get("resolved_model")
            or finished_event.get("requested_model")
        )
        finished_event["cost_usd"] = _usage_event_cost(model_name, derived_usage)
        finished_event["premium_requests"] = _counted_premium_requests(
            {
                **finished_event,
                "premium_requests": _premium_request_multiplier(model_name) if status_code < 400 else 0.0,
            }
        )
        self._remember_server_request_id(finished_event)
        self._remember_latest_claude_user_session_context(finished_event)
        self._persist_event(finished_event)

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot_usage_events(self) -> list[dict]:
        with self.state.usage_log_lock:
            return list(self.state.recent_usage_events)

    def snapshot_all_usage_events(self) -> list[dict]:
        with self.state.usage_log_lock:
            return [*list(self.state.archived_usage_events), *list(self.state.recent_usage_events)]

    # ------------------------------------------------------------------
    # Error recording
    # ------------------------------------------------------------------

    def record_request_error(self, event: dict):
        if not isinstance(event, dict):
            return

        os.makedirs(os.path.dirname(self.error_log_file) or TOKEN_DIR, exist_ok=True)
        serialized = json.dumps(event, separators=(",", ":"), default=_json_default)
        with open(self.error_log_file, "a", encoding="utf-8") as f:
            f.write(serialized)
            f.write("\n")

    # ------------------------------------------------------------------
    # Backward-compatible public aliases
    # ------------------------------------------------------------------

    def compact_history_if_needed(self):
        return self._compact_if_needed()

    def record_usage_event(self, event: dict):
        return self._persist_event(event)

    def latest_server_request_id(
        self,
        session_id: str | None,
        client_request_id: str | None,
        subagent: str | None,
    ) -> str | None:
        return self._get_latest_server_request_id(session_id, client_request_id, subagent)


# ---------------------------------------------------------------------------
# Backward-compatible module-level function (kept for external callers)
# ---------------------------------------------------------------------------

def log_proxy_request(
    request: Request,
    requested_model: str | None,
    resolved_model: str | None,
    initiator: str | None,
):
    """Backward-compatible logging stub.

    In the refactored design this logging is absorbed into
    ``UsageTracker.start_event()``.  This function is kept so that
    existing mock patches in tests (``mock.patch.object(usage_tracking,
    "log_proxy_request")``) don't break.
    """
    pass
