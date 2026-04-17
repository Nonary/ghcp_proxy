"""Dashboard payload construction and SQLite cache.

Premium quota is sourced exclusively from the upstream `x-quota-snapshot-*`
response headers captured by `usage_tracking.UsageTrackingService`. There is
no separate REST billing fetch, no manual sync, and no billing token: every
successful chat completion already tells us what the dashboard needs.
"""

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from collections import deque
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock, Thread
from typing import Callable

from constants import (
    TOKEN_DIR, SQLITE_CACHE_FILE, DETAILED_REQUEST_HISTORY_LIMIT,
)
from util import (
    _json_default, _coerce_float, _coerce_int,
    utc_now, utc_now_iso, _parse_iso_datetime,
    normalize_usage_payload, _normalize_model_name,
    _usage_event_model_name, _usage_event_source,
    _pricing_entry_for_model, _usage_event_cost, _usage_event_cost_breakdown,
    _premium_request_multiplier, _counted_premium_requests,
    _month_key, month_key_for_source_row,
)


# ─── SQLite cache state ──────────────────────────────────────────────────────

_sqlite_cache_lock = Lock()
_sqlite_cache_enabled = True
_sqlite_cache_error = None


# ─── Dashboard SSE stream state ──────────────────────────────────────────────

_dashboard_stream_subscribers = set()
_dashboard_stream_lock = Lock()
_dashboard_stream_version = 0


# ─── Runtime dependencies ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class DashboardDependencies:
    load_access_token: Callable[[], str | None] = lambda: None
    load_api_key_payload: Callable[[], dict] = lambda: {}
    snapshot_all_usage_events: Callable[[], list[dict]] = lambda: []
    snapshot_usage_events: Callable[[], list[dict]] = lambda: []
    load_safeguard_trigger_stats: Callable[[datetime], dict] = lambda _now: {}


# ─── SQLite cache functions ──────────────────────────────────────────────────

class DashboardCacheStore:
    """Owns dashboard SQLite cache lifecycle and adapts it for runtime consumers."""

    @property
    def lock(self) -> Lock:
        return _sqlite_cache_lock

    def mark_unavailable(self, error: str):
        global _sqlite_cache_enabled, _sqlite_cache_error
        if _sqlite_cache_enabled:
            print(f"[sqlite] cache disabled: {error}", flush=True)
            _sqlite_cache_enabled = False
            _sqlite_cache_error = error

    def connect(self) -> sqlite3.Connection:
        if not _sqlite_cache_enabled:
            raise RuntimeError("sqlite cache disabled")
        cache_dir = os.path.dirname(SQLITE_CACHE_FILE) or TOKEN_DIR
        os.makedirs(cache_dir, exist_ok=True)
        connection = sqlite3.connect(SQLITE_CACHE_FILE, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def initialize(self) -> bool:
        if not _sqlite_cache_enabled:
            return False
        try:
            with self.lock:
                with closing(self.connect()) as connection:
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS cache_entries (
                            cache_key TEXT PRIMARY KEY,
                            payload_json TEXT NOT NULL,
                            updated_at TEXT NOT NULL
                        )
                        """
                    )
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS archived_usage_events (
                            archive_key TEXT PRIMARY KEY,
                            recorded_at TEXT NOT NULL,
                            payload_json TEXT NOT NULL
                        )
                        """
                    )
                    connection.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_archived_usage_events_recorded_at
                        ON archived_usage_events (recorded_at DESC)
                        """
                    )
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS safeguard_trigger_events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            triggered_at TEXT NOT NULL,
                            trigger_reason TEXT NOT NULL,
                            candidate_initiator TEXT,
                            resolved_initiator TEXT,
                            model_name TEXT,
                            request_id TEXT
                        )
                        """
                    )
                    connection.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_safeguard_trigger_events_triggered_at
                        ON safeguard_trigger_events (triggered_at DESC)
                        """
                    )
                    connection.commit()
            return True
        except Exception as exc:
            self.mark_unavailable(str(exc))
            return False

    def get(self, cache_key: str) -> dict | None:
        if not self.initialize():
            return None
        try:
            with self.lock:
                with closing(self.connect()) as connection:
                    row = connection.execute(
                        "SELECT payload_json, updated_at FROM cache_entries WHERE cache_key = ?",
                        (cache_key,),
                    ).fetchone()
        except Exception as exc:
            self.mark_unavailable(str(exc))
            return None
        if row is None:
            return None
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        payload.setdefault("loaded_at", row["updated_at"])
        return payload

    def get_latest(self, cache_key_prefix: str) -> dict | None:
        if not self.initialize():
            return None
        try:
            with self.lock:
                with closing(self.connect()) as connection:
                    row = connection.execute(
                        "SELECT payload_json, updated_at FROM cache_entries WHERE cache_key LIKE ? ORDER BY updated_at DESC LIMIT 1",
                        (f"{cache_key_prefix}%",),
                    ).fetchone()
        except Exception as exc:
            self.mark_unavailable(str(exc))
            return None
        if row is None:
            return None
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        payload.setdefault("loaded_at", row["updated_at"])
        return payload

    def put(self, cache_key: str, payload: dict):
        if not isinstance(payload, dict):
            return
        if not self.initialize():
            return
        updated_at = utc_now_iso()
        serialized = json.dumps(payload, separators=(",", ":"), default=_json_default)
        try:
            with self.lock:
                with closing(self.connect()) as connection:
                    connection.execute(
                        """
                        INSERT INTO cache_entries (cache_key, payload_json, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(cache_key) DO UPDATE SET
                            payload_json = excluded.payload_json,
                            updated_at = excluded.updated_at
                        """,
                        (cache_key, serialized, updated_at),
                    )
                    connection.commit()
        except Exception as exc:
            self.mark_unavailable(f"failed to write cache key '{cache_key}': {exc}")

    def usage_archive_store(self):
        from usage_tracking import UsageArchiveStore

        return UsageArchiveStore(
            init_storage=self.initialize,
            lock=self.lock,
            connect=self.connect,
            mark_unavailable=self.mark_unavailable,
        )

    def safeguard_event_store(self):
        return SafeguardEventStore(
            init_storage=self.initialize,
            lock=self.lock,
            connect=self.connect,
            mark_unavailable=self.mark_unavailable,
        )


class SafeguardEventStore:
    def __init__(self, *, init_storage, lock, connect, mark_unavailable):
        self.init_storage = init_storage
        self.lock = lock
        self.connect = connect
        self.mark_unavailable = mark_unavailable

    def clear(self):
        if not self.init_storage():
            return
        try:
            with self.lock:
                with closing(self.connect()) as connection:
                    connection.execute("DELETE FROM safeguard_trigger_events")
                    connection.commit()
        except Exception as exc:
            self.mark_unavailable(f"failed to clear safeguard trigger events: {exc}")

    def record_event(self, event: dict | None):
        if not isinstance(event, dict):
            return
        triggered_at = event.get("triggered_at") or utc_now_iso()
        trigger_reason = str(event.get("trigger_reason") or "").strip().lower() or "unknown"
        candidate_initiator = event.get("candidate_initiator")
        resolved_initiator = event.get("resolved_initiator")
        model_name = event.get("model_name")
        request_id = event.get("request_id")
        if not self.init_storage():
            return
        try:
            with self.lock:
                with closing(self.connect()) as connection:
                    connection.execute(
                        """
                        INSERT INTO safeguard_trigger_events (
                            triggered_at,
                            trigger_reason,
                            candidate_initiator,
                            resolved_initiator,
                            model_name,
                            request_id
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(triggered_at),
                            trigger_reason,
                            candidate_initiator,
                            resolved_initiator,
                            model_name,
                            request_id,
                        ),
                    )
                    connection.commit()
        except Exception as exc:
            self.mark_unavailable(f"failed to record safeguard trigger event: {exc}")

    def load_stats(self, now: datetime | None = None) -> dict:
        reference_now = now or utc_now()
        month_start, month_end = _current_billing_month_bounds(reference_now)
        day_start = datetime(reference_now.year, reference_now.month, reference_now.day, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        default_payload = {
            "today_count": 0,
            "current_month_count": 0,
            "all_time_count": 0,
            "latest_triggered_at": None,
        }
        if not self.init_storage():
            return default_payload
        try:
            with self.lock:
                with closing(self.connect()) as connection:
                    row = connection.execute(
                        """
                        SELECT
                            COUNT(*) AS all_time_count,
                            COALESCE(SUM(CASE WHEN triggered_at >= ? AND triggered_at < ? THEN 1 ELSE 0 END), 0) AS current_month_count,
                            COALESCE(SUM(CASE WHEN triggered_at >= ? AND triggered_at < ? THEN 1 ELSE 0 END), 0) AS today_count,
                            MAX(triggered_at) AS latest_triggered_at
                        FROM safeguard_trigger_events
                        """,
                        (
                            month_start.isoformat(),
                            month_end.isoformat(),
                            day_start.isoformat(),
                            day_end.isoformat(),
                        ),
                    ).fetchone()
        except Exception as exc:
            self.mark_unavailable(f"failed to load safeguard trigger stats: {exc}")
            return default_payload
        if row is None:
            return default_payload
        return {
            "today_count": int(row["today_count"] or 0),
            "current_month_count": int(row["current_month_count"] or 0),
            "all_time_count": int(row["all_time_count"] or 0),
            "latest_triggered_at": row["latest_triggered_at"],
        }


dashboard_cache_store = DashboardCacheStore()


def _set_sqlite_cache_unavailable(error: str):
    dashboard_cache_store.mark_unavailable(error)


def _sqlite_connect() -> sqlite3.Connection:
    return dashboard_cache_store.connect()


def _init_sqlite_cache():
    return dashboard_cache_store.initialize()


def _sqlite_cache_get(cache_key: str) -> dict | None:
    return dashboard_cache_store.get(cache_key)


def _sqlite_cache_get_latest(cache_key_prefix: str) -> dict | None:
    return dashboard_cache_store.get_latest(cache_key_prefix)


def _sqlite_cache_put(cache_key: str, payload: dict):
    dashboard_cache_store.put(cache_key, payload)


def create_safeguard_event_store():
    return dashboard_cache_store.safeguard_event_store()



def _current_billing_month_bounds(now: datetime | None = None) -> tuple[datetime, datetime]:
    current = now or utc_now()
    start = datetime(current.year, current.month, 1, tzinfo=timezone.utc)
    if current.month == 12:
        end = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)
    return start, end


# ─── Dashboard SSE stream ────────────────────────────────────────────────────

class DashboardStreamBroker:
    """Coordinates dashboard SSE subscriptions without exposing module internals."""

    def current_version(self) -> int:
        return _dashboard_stream_version

    def register_listener(self) -> asyncio.Queue[int]:
        queue: asyncio.Queue[int] = asyncio.Queue(maxsize=1)
        with _dashboard_stream_lock:
            _dashboard_stream_subscribers.add(queue)
        return queue

    def unregister_listener(self, queue: asyncio.Queue[int]):
        with _dashboard_stream_lock:
            _dashboard_stream_subscribers.discard(queue)

    def notify_listeners(self):
        global _dashboard_stream_version

        _dashboard_stream_version += 1
        with _dashboard_stream_lock:
            if not _dashboard_stream_subscribers:
                return
            subscribers = list(_dashboard_stream_subscribers)

        for queue in subscribers:
            try:
                queue.put_nowait(_dashboard_stream_version)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(_dashboard_stream_version)
                except asyncio.QueueEmpty:
                    pass
                except RuntimeError:
                    self.unregister_listener(queue)


dashboard_stream_broker = DashboardStreamBroker()


def _register_dashboard_stream_listener() -> asyncio.Queue[int]:
    return dashboard_stream_broker.register_listener()


def _unregister_dashboard_stream_listener(queue: asyncio.Queue[int]):
    dashboard_stream_broker.unregister_listener(queue)


def _notify_dashboard_stream_listeners():
    dashboard_stream_broker.notify_listeners()


# ─── Usage event session descriptors ─────────────────────────────────────────

def _usage_event_group_key(event: dict | None) -> tuple[str, str]:
    if not isinstance(event, dict):
        return ("codex", "unknown")

    source = _usage_event_source(event)
    group_id = None
    for key in ("session_id", "client_request_id", "request_id", "server_request_id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            group_id = value
            break

    if not isinstance(group_id, str) or not group_id:
        group_id = "unknown"
    return (source, group_id)


def _usage_event_session_descriptor(event: dict | None) -> dict[str, str]:
    source, group_id = _usage_event_group_key(event)

    actual_session_id = event.get("session_id") if isinstance(event, dict) else None
    if isinstance(actual_session_id, str) and actual_session_id:
        return {
            "source": source,
            "group_id": group_id,
            "session_id": actual_session_id,
            "session_kind": "session",
            "session_display_id": actual_session_id,
        }

    client_request_id = event.get("client_request_id") if isinstance(event, dict) else None
    if isinstance(client_request_id, str) and client_request_id:
        return {
            "source": source,
            "group_id": group_id,
            "session_id": "",
            "session_kind": "session",
            "session_display_id": client_request_id,
        }

    request_id = event.get("request_id") if isinstance(event, dict) else None
    if isinstance(request_id, str) and request_id:
        return {
            "source": source,
            "group_id": group_id,
            "session_id": "",
            "session_kind": "session",
            "session_display_id": request_id,
        }

    server_request_id = event.get("server_request_id") if isinstance(event, dict) else None
    if isinstance(server_request_id, str) and server_request_id:
        return {
            "source": source,
            "group_id": group_id,
            "session_id": "",
            "session_kind": "session",
            "session_display_id": server_request_id,
        }

    return {
        "source": source,
        "group_id": group_id,
        "session_id": "",
        "session_kind": "unknown",
        "session_display_id": "unknown",
    }


def _monotonic_loaded_at_from_payload(payload: dict | None) -> float:
    if not isinstance(payload, dict):
        return time.monotonic()

    loaded_at_value = payload.get("loaded_at")
    if isinstance(loaded_at_value, (int, float)):
        return _coerce_float(loaded_at_value, default=time.monotonic())

    if isinstance(loaded_at_value, str):
        parsed_at = _parse_iso_datetime(loaded_at_value)
        if parsed_at is not None:
            age_seconds = (utc_now() - parsed_at).total_seconds()
            if age_seconds >= 0:
                return max(0.0, time.monotonic() - age_seconds)

    return time.monotonic()


def seed_cached_payloads_from_sqlite():
    """No-op shim retained for backward compatibility.

    Premium quota is now sourced live from upstream `x-quota-snapshot-*` headers
    captured by UsageTrackingService. No SQLite-cached premium payload to seed.
    """
    return None


# ─── Usage aggregation and dashboard building ────────────────────────────────

def _new_usage_aggregate_bucket() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_output_tokens": 0,
        "cost_usd": 0.0,
        "cost_breakdown": {
            "input_fresh": 0.0,
            "cached_input": 0.0,
            "cache_creation": 0.0,
            "output": 0.0,
        },
        "_model_order": [],
        "_models": {},
        "_last_activity_dt": None,
        "project_path": None,
        "request_count": 0,
        "session_id": None,
        "session_kind": "unknown",
        "session_display_id": None,
    }


def _ingest_usage_event(bucket: dict, event: dict):
    if not isinstance(bucket, dict) or not isinstance(event, dict):
        return

    usage = normalize_usage_payload(event.get("usage")) or {}
    event_cost = event.get("cost_usd")
    if not isinstance(event_cost, (int, float)) or not event_cost:
        recomputed_cost = _usage_event_cost(_usage_event_model_name(event), usage)
        if recomputed_cost:
            event_cost = recomputed_cost
        elif not isinstance(event_cost, (int, float)):
            event_cost = 0.0

    event_time = _parse_iso_datetime(event.get("finished_at") or event.get("started_at"))
    if event_time is not None:
        current_last = bucket.get("_last_activity_dt")
        if not isinstance(current_last, datetime) or event_time > current_last:
            bucket["_last_activity_dt"] = event_time

    project_path = event.get("project_path")
    if bucket.get("project_path") is None and isinstance(project_path, str) and project_path:
        bucket["project_path"] = project_path

    input_tokens = _coerce_int(usage.get("input_tokens"))
    output_tokens = _coerce_int(usage.get("output_tokens"))
    total_tokens = _coerce_int(usage.get("total_tokens"))
    cached_input_tokens = _coerce_int(usage.get("cached_input_tokens"))
    cache_creation_tokens = _coerce_int(usage.get("cache_creation_input_tokens"))
    reasoning_output_tokens = _coerce_int(usage.get("reasoning_output_tokens"))
    model_name = _usage_event_model_name(event) or "unknown"
    cost_breakdown = _usage_event_cost_breakdown(model_name, usage)

    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["total_tokens"] += total_tokens
    bucket["cached_input_tokens"] += cached_input_tokens
    bucket["cache_creation_tokens"] += cache_creation_tokens
    bucket["reasoning_output_tokens"] += reasoning_output_tokens
    bucket["cost_usd"] += _coerce_float(event_cost)
    bucket_cost_breakdown = bucket.setdefault(
        "cost_breakdown",
        {"input_fresh": 0.0, "cached_input": 0.0, "cache_creation": 0.0, "output": 0.0},
    )
    for key, value in cost_breakdown.items():
        bucket_cost_breakdown[key] = bucket_cost_breakdown.get(key, 0.0) + _coerce_float(value)
    bucket["request_count"] += 1

    model_bucket = bucket["_models"].setdefault(model_name, {"inputTokens": 0})
    model_bucket["inputTokens"] += input_tokens
    if model_name not in bucket["_model_order"]:
        bucket["_model_order"].append(model_name)


def _finalize_usage_bucket(bucket: dict, source: str, *, session_id: str | None = None, month: str | None = None) -> dict:
    if not isinstance(bucket, dict):
        return {}

    last_activity_dt = bucket.get("_last_activity_dt")
    last_activity = last_activity_dt.isoformat() if isinstance(last_activity_dt, datetime) else None
    models = bucket.get("_models") if isinstance(bucket.get("_models"), dict) else {}
    model_order = bucket.get("_model_order") if isinstance(bucket.get("_model_order"), list) else []
    effective_session_id = session_id if session_id is not None else bucket.get("session_id")
    if not isinstance(effective_session_id, str):
        effective_session_id = None
    effective_display_id = bucket.get("session_display_id")
    if not isinstance(effective_display_id, str) or not effective_display_id:
        effective_display_id = effective_session_id
    cost_breakdown = bucket.get("cost_breakdown") if isinstance(bucket.get("cost_breakdown"), dict) else {}
    result = {
        "sessionId": effective_session_id,
        "sessionKind": bucket.get("session_kind") or "unknown",
        "sessionDisplayId": effective_display_id,
        "lastActivity": last_activity,
        "projectPath": bucket.get("project_path"),
        "inputTokens": bucket.get("input_tokens", 0),
        "outputTokens": bucket.get("output_tokens", 0),
        "totalTokens": bucket.get("total_tokens", 0),
        "requestCount": bucket.get("request_count", 0),
        "costBreakdown": {
            "input_fresh": round(_coerce_float(cost_breakdown.get("input_fresh")), 6),
            "cached_input": round(_coerce_float(cost_breakdown.get("cached_input")), 6),
            "cache_creation": round(_coerce_float(cost_breakdown.get("cache_creation")), 6),
            "output": round(_coerce_float(cost_breakdown.get("output")), 6),
        },
    }

    if month is not None:
        result["month"] = month

    if source == "claude":
        result["cacheReadTokens"] = bucket.get("cached_input_tokens", 0)
        result["cacheCreationTokens"] = bucket.get("cache_creation_tokens", 0)
        result["totalCost"] = bucket.get("cost_usd", 0.0)
        result["modelsUsed"] = list(model_order)
    else:
        result["cachedInputTokens"] = bucket.get("cached_input_tokens", 0)
        result["reasoningOutputTokens"] = bucket.get("reasoning_output_tokens", 0)
        result["costUSD"] = bucket.get("cost_usd", 0.0)
        result["models"] = {name: value for name, value in models.items()}

    return result


def _aggregate_usage_event_buckets(
    usage_events: list[dict] | None = None,
    *,
    snapshot_usage_events: Callable[[], list[dict]] | None = None,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, dict]]]:
    if usage_events is None:
        usage_events = snapshot_usage_events() if snapshot_usage_events is not None else []
    else:
        usage_events = list(usage_events)
    source_month_buckets: dict[str, dict[str, dict]] = {}
    source_session_buckets: dict[str, dict[str, dict]] = {}

    for event in usage_events:
        event_time = _parse_iso_datetime(event.get("finished_at") or event.get("started_at"))
        if event_time is None:
            continue

        descriptor = _usage_event_session_descriptor(event)
        source = _usage_event_source(event)
        month_key = _month_key(event_time)
        group_source = descriptor.get("source") or source
        group_id = descriptor.get("group_id")
        session_key = group_id
        if not isinstance(session_key, str) or not session_key:
            session_key = event.get("server_request_id") or event.get("request_id") or "unknown"
        if group_source != source:
            source = group_source

        source_month_bucket = source_month_buckets.setdefault(source, {})
        month_bucket = source_month_bucket.setdefault(month_key, _new_usage_aggregate_bucket())
        _ingest_usage_event(month_bucket, event)

        source_session_bucket = source_session_buckets.setdefault(source, {})
        session_bucket = source_session_bucket.setdefault(session_key, _new_usage_aggregate_bucket())
        if session_bucket.get("session_display_id") is None and descriptor.get("session_display_id"):
            session_bucket["session_display_id"] = descriptor.get("session_display_id")
        if not session_bucket.get("session_id") and descriptor.get("session_id"):
            session_bucket["session_id"] = descriptor.get("session_id")
        if session_bucket.get("session_kind") in {None, "", "unknown"} and descriptor.get("session_kind"):
            session_bucket["session_kind"] = descriptor.get("session_kind")
        _ingest_usage_event(session_bucket, event)

    return source_month_buckets, source_session_buckets


def collect_local_dashboard_usage(
    usage_events: list[dict] | None = None,
    *,
    snapshot_usage_events: Callable[[], list[dict]] | None = None,
) -> dict:
    source_month_buckets, source_session_buckets = _aggregate_usage_event_buckets(
        usage_events,
        snapshot_usage_events=snapshot_usage_events,
    )
    normalized_months = []
    normalized_sessions = []

    for source in sorted(set(source_month_buckets) | set(source_session_buckets)):
        month_buckets = source_month_buckets.get(source, {})
        session_buckets = source_session_buckets.get(source, {})
        for month_key, bucket in month_buckets.items():
            normalized_months.append(
                _normalize_month_row(source, _finalize_usage_bucket(bucket, source, month=month_key))
            )
        for session_id, bucket in session_buckets.items():
            normalized_sessions.append(
                normalize_session(source, _finalize_usage_bucket(bucket, source, session_id=bucket.get("session_id")))
            )

    normalized_sessions.sort(key=lambda item: item.get("last_activity") or "", reverse=True)
    return {
        "month_rows": normalized_months,
        "session_count": len(normalized_sessions),
        "recent_sessions": normalized_sessions[:20],
        "month_history": _combine_month_rows(normalized_months),
        "errors": [],
    }


def _normalize_usage_rollup(source: str, row: dict) -> dict:
    if source == "claude":
        models = row.get("modelsUsed") or list((row.get("models") or {}).keys())
        cost_usd = _coerce_float(row.get("totalCost"))
        if cost_usd == 0.0 and row.get("costUSD") is not None:
            cost_usd = _coerce_float(row.get("costUSD"))
        cached_tokens = _coerce_int(row.get("cacheReadTokens"))
        if cached_tokens == 0 and row.get("cachedInputTokens") is not None:
            cached_tokens = _coerce_int(row.get("cachedInputTokens"))
        cache_creation_tokens = _coerce_int(row.get("cacheCreationTokens"))
        if cache_creation_tokens == 0 and row.get("cacheCreationInputTokens") is not None:
            cache_creation_tokens = _coerce_int(row.get("cacheCreationInputTokens"))
        reasoning_tokens = 0
    else:
        models = list((row.get("models") or {}).keys())
        cost_usd = _coerce_float(row.get("costUSD"))
        cached_tokens = _coerce_int(row.get("cachedInputTokens"))
        cache_creation_tokens = 0
        reasoning_tokens = _coerce_int(row.get("reasoningOutputTokens"))
    raw_cost_breakdown = row.get("costBreakdown")
    cost_breakdown = {
        "input_fresh": 0.0,
        "cached_input": 0.0,
        "cache_creation": 0.0,
        "output": 0.0,
    }
    if isinstance(raw_cost_breakdown, dict):
        cost_breakdown["input_fresh"] = _coerce_float(raw_cost_breakdown.get("input_fresh"))
        cost_breakdown["cached_input"] = _coerce_float(raw_cost_breakdown.get("cached_input"))
        cost_breakdown["cache_creation"] = _coerce_float(raw_cost_breakdown.get("cache_creation"))
        cost_breakdown["output"] = _coerce_float(raw_cost_breakdown.get("output"))

    return {
        "source": source,
        "input_tokens": _coerce_int(row.get("inputTokens")),
        "output_tokens": _coerce_int(row.get("outputTokens")),
        "total_tokens": _coerce_int(row.get("totalTokens")),
        "cached_input_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "request_count": _coerce_int(row.get("requestCount")),
        "cost_usd": cost_usd,
        "cost_breakdown": cost_breakdown,
        "models": models,
    }


def normalize_session(source: str, session: dict) -> dict:
    return {
        **_normalize_usage_rollup(source, session),
        "session_id": session.get("sessionId"),
        "session_kind": session.get("sessionKind") or "unknown",
        "session_display_id": session.get("sessionDisplayId") or session.get("sessionId"),
        "last_activity": session.get("lastActivity"),
        "project_path": session.get("projectPath"),
    }


def _normalize_month_row(source: str, row: dict) -> dict:
    return {
        **_normalize_usage_rollup(source, row),
        "month_key": month_key_for_source_row(source, row),
        "month_label": row.get("month"),
    }


def _normalize_day_row(source: str, row: dict, day_key: str) -> dict:
    return {
        **_normalize_usage_rollup(source, row),
        "day_key": day_key,
        "day_label": day_key,
    }


def _combine_month_rows(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        month_key = row.get("month_key")
        if not month_key:
            continue
        current = grouped.setdefault(
            month_key,
            {
                "month_key": month_key,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "cache_creation_tokens": 0,
                "reasoning_output_tokens": 0,
                "request_count": 0,
                "cost_usd": 0.0,
                "cost_breakdown": {
                    "input_fresh": 0.0,
                    "cached_input": 0.0,
                    "cache_creation": 0.0,
                    "output": 0.0,
                },
                "sources": {},
            },
        )
        current["input_tokens"] += row.get("input_tokens", 0)
        current["output_tokens"] += row.get("output_tokens", 0)
        current["total_tokens"] += row.get("total_tokens", 0)
        current["cached_input_tokens"] += row.get("cached_input_tokens", 0)
        current["cache_creation_tokens"] += row.get("cache_creation_tokens", 0)
        current["reasoning_output_tokens"] += row.get("reasoning_output_tokens", 0)
        current["request_count"] += row.get("request_count", 0)
        current["cost_usd"] += row.get("cost_usd", 0.0)
        for key, value in (row.get("cost_breakdown") or {}).items():
            current["cost_breakdown"][key] = current["cost_breakdown"].get(key, 0.0) + _coerce_float(value)
        current["sources"][row["source"]] = row

    return [grouped[key] | {"cost_usd": round(grouped[key]["cost_usd"], 4)} for key in sorted(grouped.keys(), reverse=True)]


def _combine_day_rows(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        day_key = row.get("day_key")
        if not day_key:
            continue
        current = grouped.setdefault(
            day_key,
            {
                "day_key": day_key,
                "day_label": row.get("day_label") or day_key,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "cache_creation_tokens": 0,
                "reasoning_output_tokens": 0,
                "request_count": 0,
                "cost_usd": 0.0,
                "cost_breakdown": {
                    "input_fresh": 0.0,
                    "cached_input": 0.0,
                    "cache_creation": 0.0,
                    "output": 0.0,
                },
                "sources": {},
            },
        )
        current["input_tokens"] += row.get("input_tokens", 0)
        current["output_tokens"] += row.get("output_tokens", 0)
        current["total_tokens"] += row.get("total_tokens", 0)
        current["cached_input_tokens"] += row.get("cached_input_tokens", 0)
        current["cache_creation_tokens"] += row.get("cache_creation_tokens", 0)
        current["reasoning_output_tokens"] += row.get("reasoning_output_tokens", 0)
        current["request_count"] += row.get("request_count", 0)
        current["cost_usd"] += row.get("cost_usd", 0.0)
        for key, value in (row.get("cost_breakdown") or {}).items():
            current["cost_breakdown"][key] = current["cost_breakdown"].get(key, 0.0) + _coerce_float(value)
        current["sources"][row["source"]] = row

    return [grouped[key] | {"cost_usd": round(grouped[key]["cost_usd"], 4)} for key in sorted(grouped.keys())]


def _combine_usage_rows(rows: list[dict], *, month_key: str | None = None) -> dict:
    per_source = {}
    for row in rows:
        source = row.get("source")
        if not isinstance(source, str) or not source:
            continue
        current = per_source.setdefault(
            source,
            {
                "source": source,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "cache_creation_tokens": 0,
                "reasoning_output_tokens": 0,
                "request_count": 0,
                "cost_usd": 0.0,
                "cost_breakdown": {
                    "input_fresh": 0.0,
                    "cached_input": 0.0,
                    "cache_creation": 0.0,
                    "output": 0.0,
                },
                "models": [],
            },
        )
        current["input_tokens"] += row.get("input_tokens", 0)
        current["output_tokens"] += row.get("output_tokens", 0)
        current["total_tokens"] += row.get("total_tokens", 0)
        current["cached_input_tokens"] += row.get("cached_input_tokens", 0)
        current["cache_creation_tokens"] += row.get("cache_creation_tokens", 0)
        current["reasoning_output_tokens"] += row.get("reasoning_output_tokens", 0)
        current["request_count"] += row.get("request_count", 0)
        current["cost_usd"] += row.get("cost_usd", 0.0)
        for key, value in (row.get("cost_breakdown") or {}).items():
            current["cost_breakdown"][key] = current["cost_breakdown"].get(key, 0.0) + _coerce_float(value)
        current["models"] = sorted(set(current["models"]) | set(row.get("models") or []))

    combined = {
        "input_tokens": sum(item.get("input_tokens", 0) for item in per_source.values()),
        "output_tokens": sum(item.get("output_tokens", 0) for item in per_source.values()),
        "total_tokens": sum(item.get("total_tokens", 0) for item in per_source.values()),
        "cached_input_tokens": sum(item.get("cached_input_tokens", 0) for item in per_source.values()),
        "cache_creation_tokens": sum(item.get("cache_creation_tokens", 0) for item in per_source.values()),
        "reasoning_output_tokens": sum(item.get("reasoning_output_tokens", 0) for item in per_source.values()),
        "request_count": sum(item.get("request_count", 0) for item in per_source.values()),
        "cost_usd": round(sum(item.get("cost_usd", 0.0) for item in per_source.values()), 4),
        "cost_breakdown": {
            "input_fresh": round(sum(item.get("cost_breakdown", {}).get("input_fresh", 0.0) for item in per_source.values()), 6),
            "cached_input": round(sum(item.get("cost_breakdown", {}).get("cached_input", 0.0) for item in per_source.values()), 6),
            "cache_creation": round(sum(item.get("cost_breakdown", {}).get("cache_creation", 0.0) for item in per_source.values()), 6),
            "output": round(sum(item.get("cost_breakdown", {}).get("output", 0.0) for item in per_source.values()), 6),
        },
        "sources": {
            source: item
            | {
                "cost_usd": round(item.get("cost_usd", 0.0), 4),
                "cost_breakdown": {
                    "input_fresh": round(item.get("cost_breakdown", {}).get("input_fresh", 0.0), 6),
                    "cached_input": round(item.get("cost_breakdown", {}).get("cached_input", 0.0), 6),
                    "cache_creation": round(item.get("cost_breakdown", {}).get("cache_creation", 0.0), 6),
                    "output": round(item.get("cost_breakdown", {}).get("output", 0.0), 6),
                },
            }
            for source, item in per_source.items()
        },
    }
    if month_key is not None:
        combined["month_key"] = month_key
    return combined


def collect_daily_dashboard_usage(
    usage_events: list[dict] | None = None,
    *,
    snapshot_usage_events: Callable[[], list[dict]] | None = None,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> list[dict]:
    if usage_events is None:
        usage_events = snapshot_usage_events() if snapshot_usage_events is not None else []
    else:
        usage_events = list(usage_events)

    source_day_buckets: dict[str, dict[str, dict]] = {}
    for event in usage_events:
        event_time = _parse_iso_datetime(event.get("finished_at") or event.get("started_at"))
        if event_time is None:
            continue
        if start_at is not None and event_time < start_at:
            continue
        if end_at is not None and event_time >= end_at:
            continue

        source = _usage_event_source(event)
        day_key = event_time.astimezone(timezone.utc).strftime("%Y-%m-%d")
        source_bucket = source_day_buckets.setdefault(source, {})
        day_bucket = source_bucket.setdefault(day_key, _new_usage_aggregate_bucket())
        _ingest_usage_event(day_bucket, event)

    normalized_days = []
    for source in sorted(source_day_buckets):
        for day_key, bucket in source_day_buckets[source].items():
            normalized_days.append(
                _normalize_day_row(source, _finalize_usage_bucket(bucket, source), day_key)
            )
    return _combine_day_rows(normalized_days)


def _empty_day_history_row(day_key: str) -> dict:
    return {
        "day_key": day_key,
        "day_label": day_key,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_output_tokens": 0,
        "request_count": 0,
        "cost_usd": 0.0,
        "cost_breakdown": {
            "input_fresh": 0.0,
            "cached_input": 0.0,
            "cache_creation": 0.0,
            "output": 0.0,
        },
        "sources": {},
    }


def _latest_quota_snapshots(events: list[dict]) -> dict | None:
    """Return the most recent x-quota-snapshot-* payload captured on a successful event.

    The Copilot upstream emits per-bucket quota snapshots on every chat completion
    response. UsageTrackingService parses those into event["quota_snapshots"].
    Find the newest event that has them.
    """
    best_at = ""
    best: dict | None = None
    best_event_id: str | None = None
    for event in events:
        snapshots = event.get("quota_snapshots")
        if not isinstance(snapshots, dict) or not snapshots:
            continue
        recorded_at = event.get("finished_at") or event.get("started_at") or ""
        if not isinstance(recorded_at, str):
            continue
        if recorded_at <= best_at:
            continue
        best_at = recorded_at
        best = snapshots
        best_event_id = event.get("request_id") if isinstance(event.get("request_id"), str) else None
    if best is None:
        return None
    return {
        "captured_at": best_at,
        "request_id": best_event_id,
        "snapshots": best,
    }


def _build_premium_usage_summary(
    usage_events: list[dict],
    *,
    now: datetime,
) -> dict:
    """Build the dashboard's `premium` block from the most recent quota snapshot.

    Source of truth: the `quota_snapshots` field that UsageTrackingService writes
    on every successful chat completion (parsed from `x-quota-snapshot-*` headers).
    Returns an `awaiting-first-request` placeholder when no snapshot has been seen.
    """
    latest = _latest_quota_snapshots(usage_events)
    if latest is None:
        return {
            "configured": False,
            "source": "awaiting-first-request",
            "message": "Make a request through the proxy to populate the live quota.",
            "captured_at": None,
            "buckets": {},
            "included": None,
            "remaining": None,
            "used": None,
            "percent_remaining": None,
            "percent_used": None,
            "reset_at": None,
            "days_until_reset": None,
            "unlimited": None,
        }

    snapshots = latest["snapshots"]
    primary = snapshots.get("premium_interactions") if isinstance(snapshots, dict) else None
    primary = primary if isinstance(primary, dict) else {}

    reset_at = primary.get("reset_at")
    days_until_reset = _days_until(reset_at, now)

    return {
        "configured": True,
        "source": "upstream-quota-snapshot",
        "captured_at": latest.get("captured_at"),
        "request_id": latest.get("request_id"),
        "buckets": snapshots,
        "included": primary.get("included"),
        "remaining": primary.get("absolute_remaining"),
        "used": primary.get("absolute_used"),
        "percent_remaining": primary.get("percent_remaining"),
        "percent_used": primary.get("percent_used"),
        "overage": primary.get("overage"),
        "overage_permitted": primary.get("overage_permitted"),
        "reset_at": reset_at,
        "days_until_reset": days_until_reset,
        "unlimited": bool(primary.get("unlimited")),
    }


def _days_until(reset_at: str | None, now: datetime) -> int | None:
    parsed = _parse_iso_datetime(reset_at) if isinstance(reset_at, str) else None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    delta = parsed - now
    return max(0, int(delta.total_seconds() // 86400))


class DashboardService:
    """Owns dashboard aggregation with explicit auth/usage callbacks and runtime hooks."""

    def __init__(
        self,
        *,
        dependencies: DashboardDependencies | None = None,
        utc_now: Callable[[], datetime] = utc_now,
        utc_now_iso: Callable[[], str] = utc_now_iso,
        sqlite_cache_put: Callable[[str, dict], None] = dashboard_cache_store.put,
        notify_dashboard_stream_listeners: Callable[[], None] = dashboard_stream_broker.notify_listeners,
        stream_broker: "DashboardStreamBroker | None" = None,
        thread_class: Callable[[], type] | type = Thread,
    ):
        self.dependencies = dependencies or DashboardDependencies()
        self.utc_now = utc_now
        self.utc_now_iso = utc_now_iso
        self.sqlite_cache_put = sqlite_cache_put
        self.notify_dashboard_stream_listeners = notify_dashboard_stream_listeners
        self._stream_broker = stream_broker or dashboard_stream_broker
        self.thread_class = thread_class

    def _resolved_thread_class(self):
        resolved_thread_class = self.thread_class
        if callable(resolved_thread_class) and not isinstance(resolved_thread_class, type):
            resolved_thread_class = resolved_thread_class()
        return resolved_thread_class

    def build_payload(self, force_refresh: bool = False) -> dict:
        now = self.utc_now()
        month_start, month_end = _current_billing_month_bounds(now)
        current_month_key = _month_key(now)
        current_day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        usage_events = self.dependencies.snapshot_all_usage_events()
        detailed_usage_events = self.dependencies.snapshot_usage_events()
        current_month_events = []
        for event in usage_events:
            recorded_at = _parse_iso_datetime(event.get("finished_at") or event.get("started_at"))
            if recorded_at is None:
                continue
            if month_start <= recorded_at < month_end:
                current_month_events.append(event)

        premium_summary = _build_premium_usage_summary(usage_events, now=now)

        local_usage = collect_local_dashboard_usage(usage_events)
        month_rows = list(local_usage.get("month_rows") or [])
        current_month_usage = _combine_usage_rows(
            [row for row in month_rows if row.get("month_key") == current_month_key],
            month_key=current_month_key,
        )
        all_time_usage = _combine_usage_rows(month_rows)
        all_time_usage["months_tracked"] = len(local_usage.get("month_history") or [])
        daily_history = collect_daily_dashboard_usage(
            usage_events,
            start_at=month_start,
            end_at=month_end,
        )
        safeguard_stats = self.dependencies.load_safeguard_trigger_stats(now)
        if not isinstance(safeguard_stats, dict):
            safeguard_stats = {}
        daily_history_by_key = {row["day_key"]: row for row in daily_history if isinstance(row.get("day_key"), str)}
        filled_daily_history = []
        day_cursor = month_start
        while day_cursor <= current_day_start:
            day_key = day_cursor.strftime("%Y-%m-%d")
            filled_daily_history.append(daily_history_by_key.get(day_key) or _empty_day_history_row(day_key))
            day_cursor += timedelta(days=1)

        recent_requests = sorted(
            detailed_usage_events,
            key=lambda item: item.get("finished_at") or item.get("started_at") or "",
            reverse=True,
        )[:DETAILED_REQUEST_HISTORY_LIMIT]

        return {
            "generated_at": now.isoformat(),
            "premium": premium_summary,
            "safeguard": {
                "today_count": int(safeguard_stats.get("today_count") or 0),
                "current_month_count": int(safeguard_stats.get("current_month_count") or 0),
                "all_time_count": int(safeguard_stats.get("all_time_count") or 0),
                "latest_triggered_at": safeguard_stats.get("latest_triggered_at"),
            },
            "current_month": {
                "label": current_month_key,
                "start_at": month_start.isoformat(),
                "end_at": month_end.isoformat(),
                "proxy_requests": len(current_month_events),
                "sessions": local_usage.get("session_count", 0),
                "usage": current_month_usage,
                "daily_history": filled_daily_history,
            },
            "all_time": {
                "proxy_requests": len(usage_events),
                "archived_requests": max(len(usage_events) - len(detailed_usage_events), 0),
                "detailed_requests": len(detailed_usage_events),
                "sessions": local_usage.get("session_count", 0),
                "usage": all_time_usage,
            },
            "recent_sessions": local_usage.get("recent_sessions") or [],
            "recent_requests": recent_requests,
            "month_history": (local_usage.get("month_history") or [])[:12],
        }

    # ─── Stream broker delegation ─────────────────────────────────────────────

    def register_stream_listener(self) -> asyncio.Queue:
        return self._stream_broker.register_listener()

    def unregister_stream_listener(self, queue: asyncio.Queue):
        self._stream_broker.unregister_listener(queue)

    def current_stream_version(self) -> int:
        return self._stream_broker.current_version()


# ─── Public factory API ───────────────────────────────────────────────────────


def create_dashboard_service(
    dependencies: DashboardDependencies,
    **kwargs,
) -> DashboardService:
    """Create a fully-wired DashboardService. Encapsulates cache/broker setup."""
    return DashboardService(
        dependencies=dependencies,
        sqlite_cache_put=dashboard_cache_store.put,
        notify_dashboard_stream_listeners=dashboard_stream_broker.notify_listeners,
        stream_broker=dashboard_stream_broker,
        **kwargs,
    )


def create_usage_archive_store():
    """Create a usage archive store backed by the dashboard cache."""
    return dashboard_cache_store.usage_archive_store()


def initialize():
    """Seed cached payloads from SQLite. Call once at startup."""
    seed_cached_payloads_from_sqlite()
