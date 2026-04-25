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
from urllib.parse import parse_qs, unquote

import httpx
from fastapi import Request

from constants import (
    TOKEN_DIR, USAGE_LOG_FILE, REQUEST_ERROR_LOG_FILE,
    DETAILED_REQUEST_HISTORY_LIMIT,
    RESPONSE_REASONING_PREVIEW_MAX_CHARS,
)
from util import (
    _json_default, _coerce_float, _coerce_int,
    utc_now, utc_now_iso,
    normalize_usage_payload, _normalize_model_name,
    _usage_event_model_name, _usage_event_source,
    _usage_event_cost, _usage_event_estimated_cost,
    _premium_request_multiplier, _counted_premium_requests,
    _server_request_chain_key, _codex_native_session_id_from_request_id, _codex_logs_service_tiers, _is_claude_request,
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

def _header_value_to_float(val: str | None) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_urlencoded_header_fields(header_value: str | None) -> dict[str, str]:
    raw: dict[str, str] = {}
    if header_value is None:
        return raw
    for key, values in parse_qs(unquote(str(header_value)), keep_blank_values=True).items():
        if values:
            raw[key] = values[0]
    return raw


def extract_quota_snapshots_from_headers(headers) -> dict[str, dict[str, object]]:
    """Parse Copilot x-quota-snapshot-* headers into dashboard-ready payloads."""
    quota_snapshots: dict[str, dict[str, object]] = {}
    if headers is None:
        return quota_snapshots
    for header_name, header_value in headers.items():
        lowered = str(header_name).lower()
        if not lowered.startswith("x-quota-snapshot-"):
            continue
        bucket = lowered[len("x-quota-snapshot-"):]
        if not bucket:
            continue
        raw = _parse_urlencoded_header_fields(header_value)

        ent_value = _header_value_to_float(raw.get("ent"))
        rem_percent = _header_value_to_float(raw.get("rem"))
        overage = _header_value_to_float(raw.get("ov"))
        overage_permitted = (raw.get("ovPerm", "").lower() == "true") if "ovPerm" in raw else None
        reset_at = raw.get("rst")
        unlimited = ent_value is not None and ent_value < 0
        included: int | None = (
            int(ent_value) if (ent_value is not None and not unlimited) else None
        )

        percent_remaining: float | None = None
        percent_used: float | None = None
        if rem_percent is not None:
            percent_remaining = round(max(0.0, min(rem_percent, 100.0)), 2)
            percent_used = round(100.0 - percent_remaining, 2)

        absolute_remaining: float | None = None
        absolute_used: float | None = None
        if included is not None and percent_remaining is not None:
            absolute_remaining = round(included * percent_remaining / 100.0, 2)
            absolute_used = round(included - absolute_remaining, 2)

        quota_snapshots[bucket] = {
            "included": included,
            "unlimited": unlimited,
            "percent_remaining": percent_remaining,
            "percent_used": percent_used,
            "absolute_remaining": absolute_remaining,
            "absolute_used": absolute_used,
            "overage": overage,
            "overage_permitted": overage_permitted,
            "reset_at": reset_at,
            "raw": raw,
        }
    return quota_snapshots


def extract_usage_ratelimits_from_headers(headers) -> dict[str, dict[str, object]]:
    """Parse upstream x-usage-ratelimit-* headers.

    Copilot CLI's chat completions emit per-window usage gauges:
      x-usage-ratelimit-session: ent=0&ov=0.0&ovPerm=false&rem=93.2&rst=...
      x-usage-ratelimit-weekly:  ent=0&ov=0.0&ovPerm=false&rem=99.0&rst=...

    `rem` is a remaining percentage (0..100), not an absolute request count.
    """
    usage_ratelimits: dict[str, dict[str, object]] = {}
    if headers is None:
        return usage_ratelimits
    for header_name, header_value in headers.items():
        lowered = str(header_name).lower()
        if not lowered.startswith("x-usage-ratelimit-"):
            continue
        window = lowered[len("x-usage-ratelimit-"):]
        if not window:
            continue
        raw = _parse_urlencoded_header_fields(header_value)
        rem_percent = _header_value_to_float(raw.get("rem"))
        ent_value = _header_value_to_float(raw.get("ent"))
        overage = _header_value_to_float(raw.get("ov"))
        ov_perm_raw = raw.get("ovPerm")
        overage_permitted = (
            ov_perm_raw.lower() == "true" if isinstance(ov_perm_raw, str) and ov_perm_raw else None
        )
        reset_at = raw.get("rst")
        percent_remaining = (
            round(max(0.0, min(rem_percent, 100.0)), 2) if rem_percent is not None else None
        )
        percent_used = (
            round(100.0 - percent_remaining, 2) if percent_remaining is not None else None
        )
        usage_ratelimits[window] = {
            "percent_remaining": percent_remaining,
            "percent_used": percent_used,
            "entitlement": int(ent_value) if (ent_value is not None and ent_value > 0) else ent_value,
            "overage": overage,
            "overage_permitted": overage_permitted,
            "reset_at": reset_at,
            "raw": raw,
        }
    return usage_ratelimits

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
        for key in ("session_id", "sessionId"):
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


def _normalized_api_path(path: str | None) -> str | None:
    if not isinstance(path, str):
        return None
    normalized = path.strip().split("?", 1)[0].lower()
    if not normalized:
        return None
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    normalized = normalized.rstrip("/") or "/"
    if normalized.startswith("/v1/"):
        normalized = normalized[3:]
    return normalized


def _is_responses_api_path(path: str | None) -> bool:
    return _normalized_api_path(path) in {"/responses", "/responses/compact"}


def _drop_outbound_headers(headers: dict, header_names: tuple[str, ...]) -> None:
    names = {name.lower() for name in header_names}
    for key in list(headers.keys()):
        if isinstance(key, str) and key.lower() in names:
            headers.pop(key, None)


def _display_model_name(model_name: str | None) -> str | None:
    normalized = _normalize_model_name(model_name)
    if normalized is None:
        return model_name

    try:
        import format_translation

        resolved = format_translation.resolve_copilot_model_name(normalized)
    except Exception:
        resolved = normalized
    return _normalize_model_name(resolved) or normalized


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


def _normalize_recorded_usage_event(
    payload: dict | None,
    *,
    refresh_native_tiers: bool = True,
) -> dict | None:
    if not isinstance(payload, dict):
        return None

    normalized_event = dict(payload)

    native_session_id = _codex_native_session_id_from_request_id(normalized_event.get("request_id"))
    # Backfill native_source for events that were archived before the
    # marker was preserved through compaction. The codex_native ingestor
    # uses request_ids of the form "codex-native:<session>:<turn>" and the
    # synthetic path "/native/codex/responses", so either is a reliable
    # signal that this row originated from a Codex CLI rollout file.
    if not normalized_event.get("native_source"):
        request_id = normalized_event.get("request_id")
        path = normalized_event.get("path")
        if (
            (isinstance(request_id, str) and request_id.startswith("codex-native:"))
            or path == "/native/codex/responses"
        ):
            normalized_event["native_source"] = "codex_native"
    native_model_provider = normalized_event.get("native_model_provider")
    if (
        normalized_event.get("native_source") == "codex_native"
        and isinstance(native_model_provider, str)
        and native_model_provider.strip().lower() == "custom"
    ):
        return None
    if not normalized_event.get("session_id") and native_session_id:
        normalized_event["session_id"] = native_session_id
        normalized_event.setdefault("session_id_origin", "codex_native_request_id")
    if not normalized_event.get("server_request_id"):
        effective_native_session_id = normalized_event.get("session_id") or native_session_id
        if (
            normalized_event.get("native_source") == "codex_native"
            and isinstance(effective_native_session_id, str)
            and effective_native_session_id
        ):
            normalized_event["server_request_id"] = effective_native_session_id
    if normalized_event.get("native_source") == "codex_native":
        requested_source = normalized_event.get("native_requested_service_tier_source")
        effective_source = normalized_event.get("native_service_tier_source")
        should_refresh_native_tiers = refresh_native_tiers or str(
            os.environ.get("GHCP_REFRESH_CODEX_LOG_TIERS_ON_LOAD", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        native_service_tiers = (
            _codex_logs_service_tiers(
                normalized_event.get("session_id") or native_session_id,
                normalized_event.get("native_turn_id"),
                normalized_event.get("started_at"),
            )
            if should_refresh_native_tiers
            else {
                "requested": normalized_event.get("native_requested_service_tier"),
                "requested_source": requested_source,
                "effective": normalized_event.get("native_service_tier"),
                "effective_source": effective_source,
            }
        )
        requested_native_service_tier = native_service_tiers.get("requested")
        if isinstance(requested_native_service_tier, str) and requested_native_service_tier:
            normalized_event["native_requested_service_tier"] = requested_native_service_tier
            normalized_event["native_requested_service_tier_source"] = native_service_tiers.get("requested_source")
        elif should_refresh_native_tiers and normalized_event.get("native_requested_service_tier_source") != "codex_logs_request":
            normalized_event.pop("native_requested_service_tier", None)
            normalized_event.pop("native_requested_service_tier_source", None)

        exact_native_service_tier = native_service_tiers.get("effective")
        if isinstance(exact_native_service_tier, str) and exact_native_service_tier:
            normalized_event["native_service_tier"] = exact_native_service_tier
            normalized_event["native_service_tier_source"] = native_service_tiers.get("effective_source")
        elif should_refresh_native_tiers and not str(normalized_event.get("native_service_tier_source") or "").startswith("codex_logs_response"):
            normalized_event.pop("native_service_tier", None)
            normalized_event.pop("native_service_tier_source", None)

    normalized_usage = normalize_usage_payload(normalized_event.get("usage"))
    if isinstance(normalized_usage, dict):
        normalized_event["usage"] = normalized_usage
        # codex_native cost always reflects the current service-tier multiplier
        # (logs may have updated since archival); for non-native events, only
        # backfill when missing.
        if normalized_event.get("native_source") == "codex_native":
            normalized_event["cost_usd"] = _usage_event_estimated_cost(normalized_event, usage=normalized_usage)
        elif normalized_event.get("cost_usd") is None:
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

    # Preserve native-source markers across compaction so codex_native (and
    # any future ingested-source) traffic doesn't silently fall back to the
    # model-name heuristic in _usage_event_source after archival.
    for native_key in (
        "native_source",
        "native_origin",
        "native_cli_version",
        "native_model_provider",
        "native_plan_type",
        "native_requested_service_tier",
        "native_requested_service_tier_source",
        "native_service_tier",
        "native_service_tier_source",
        "native_reasoning_effort",
        "native_turn_id",
    ):
        value = event.get(native_key)
        if value is not None:
            summary[native_key] = value

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
        forwarded_server_request_id = (
            request.headers.get("x-request-id")
            or request.headers.get("request-id")
            or request.headers.get("x-github-request-id")
        )

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

        prior_server_request_id = forwarded_server_request_id or None

        if initiator == "agent":
            active_server_request_id = self._get_active_server_request_id(
                session_id,
                client_request_id,
                subagent,
                initiator="user",
            )
            if isinstance(active_server_request_id, str) and active_server_request_id:
                prior_server_request_id = active_server_request_id

        if prior_server_request_id is None and initiator == "user" and allow_user_active_fallback and has_chain_context:
            active_server_request_id = self._get_active_server_request_id(
                session_id,
                client_request_id,
                subagent,
                initiator="user",
            )
            if isinstance(active_server_request_id, str) and active_server_request_id:
                prior_server_request_id = active_server_request_id

        if prior_server_request_id is None and session_id is not None:
            latest_server_request_id = self._get_latest_server_request_id(
                session_id,
                client_request_id,
                subagent,
            )
            if isinstance(latest_server_request_id, str) and latest_server_request_id:
                prior_server_request_id = latest_server_request_id

        generated_server_request_id = str(uuid4())
        return generated_server_request_id, prior_server_request_id

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
            normalized_event = _normalize_recorded_usage_event(payload, refresh_native_tiers=False)
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
                        normalized_event = _normalize_recorded_usage_event(payload, refresh_native_tiers=False)
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
        prompt_preview: dict | None = None,
        initiator_verdict: dict | None = None,
    ) -> dict:
        # Log the proxy request (absorbed from log_proxy_request)
        display_requested_model = _display_model_name(requested_model)
        display_resolved_model = _display_model_name(resolved_model)
        parts = [
            "INFO:",
            f"Proxy request ({_initiator_log_label(initiator)}):",
            f"{request.method} {request.url.path}",
        ]
        if display_requested_model is not None:
            parts.append(f"requested_model={display_requested_model}")
        if (
            display_resolved_model is not None
            and display_resolved_model != display_requested_model
        ):
            parts.append(f"resolved_model={display_resolved_model}")
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
            outbound_path = upstream_path or getattr(request.url, "path", None)
            uses_responses_affinity_headers = _is_responses_api_path(outbound_path)
            if uses_responses_affinity_headers:
                _drop_outbound_headers(
                    outbound_headers,
                    ("x-request-id", "request-id", "x-github-request-id", "session_id", "session-id"),
                )
            else:
                if session_id:
                    outbound_headers["session_id"] = session_id
                    outbound_headers["x-interaction-id"] = session_id
                outbound_headers["x-request-id"] = server_request_id
                outbound_headers["x-github-request-id"] = server_request_id
                outbound_headers["x-agent-task-id"] = server_request_id
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
        if isinstance(prompt_preview, dict) and prompt_preview:
            event["request_prompt"] = prompt_preview
        if isinstance(initiator_verdict, dict) and initiator_verdict:
            safeguard_reason = initiator_verdict.get("safeguard_reason")
            candidate_initiator = initiator_verdict.get("candidate_initiator")
            resolved_initiator = initiator_verdict.get("resolved_initiator")
            verdict_snapshot = {
                key: value
                for key, value in initiator_verdict.items()
                if value is not None
            }
            if verdict_snapshot:
                event["initiator_verdict"] = verdict_snapshot
            if isinstance(safeguard_reason, str) and safeguard_reason:
                event["safeguard_triggered"] = True
                event["safeguard_reason"] = safeguard_reason
            if isinstance(candidate_initiator, str) and candidate_initiator:
                event["candidate_initiator"] = candidate_initiator
            if isinstance(resolved_initiator, str) and resolved_initiator:
                event["resolved_initiator"] = resolved_initiator
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
        reasoning_text: str | None = None,
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

            # Copilot upstream emits x-quota-snapshot-{chat,completions,premium_interactions}
            # with URL-encoded "ent=...&ov=...&ovPerm=...&rem=...&rst=..." values.
            # `rem` is a remaining percentage, not an absolute count. This is more
            # authoritative than the user-scoped /settings/billing endpoint because
            # it ships on every successful chat completion.
            quota_snapshots = extract_quota_snapshots_from_headers(upstream.headers)
            if quota_snapshots:
                finished_event["quota_snapshots"] = quota_snapshots

            # Copilot CLI's chat completions also emit per-window usage gauges.
            # We surface them so the dashboard and response reminders can show the
            # live session/weekly throttles GitHub uses to slow Copilot down before
            # the monthly premium cap bites.
            usage_ratelimits = extract_usage_ratelimits_from_headers(upstream.headers)
            if usage_ratelimits:
                finished_event["usage_ratelimits"] = usage_ratelimits

            # Also keep generic x-ratelimit-* if any upstream variant ever emits them.
            rate_limit_fields = {}
            for header_name in (
                "x-ratelimit-limit",
                "x-ratelimit-remaining",
                "x-ratelimit-reset",
                "x-ratelimit-used",
                "x-ratelimit-resource",
                "retry-after",
            ):
                header_value = upstream.headers.get(header_name)
                if header_value is None:
                    continue
                short_key = header_name.replace("x-ratelimit-", "").replace("-", "_")
                rate_limit_fields[short_key] = header_value
            if rate_limit_fields:
                finished_event["rate_limit"] = rate_limit_fields

        if isinstance(response_payload, dict):
            payload_response_id = response_payload.get("id")
            if isinstance(payload_response_id, str):
                finished_event["response_id"] = payload_response_id
            payload_model = response_payload.get("model")
            if isinstance(payload_model, str):
                finished_event["response_model"] = payload_model

        if isinstance(reasoning_text, str) and reasoning_text:
            # Mirror how response_text-style fields are surfaced: keep a bounded
            # excerpt so dashboards / trace viewers can show what the model was
            # actually thinking without retaining megabytes of reasoning.
            finished_event["reasoning_text"] = reasoning_text[:RESPONSE_REASONING_PREVIEW_MAX_CHARS]
            if len(reasoning_text) > RESPONSE_REASONING_PREVIEW_MAX_CHARS:
                finished_event["reasoning_text_truncated"] = True
                finished_event["reasoning_text_chars"] = len(reasoning_text)

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
        finished_event["cost_usd"] = _usage_event_estimated_cost(
            finished_event,
            model_name=model_name,
            usage=derived_usage,
        )
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




