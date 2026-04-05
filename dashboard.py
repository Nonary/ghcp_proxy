"""Dashboard payload construction, SQLite cache, GitHub billing API integration, and premium quota management."""

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from collections import deque
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Callable

import httpx

from constants import (
    TOKEN_DIR, SQLITE_CACHE_FILE, PREMIUM_CACHE_TTL_SECONDS,
    SKU_PREMIUM_ALLOWANCES, DETAILED_REQUEST_HISTORY_LIMIT,
)
from util import (
    _json_default, _coerce_float, _coerce_int,
    utc_now, utc_now_iso, _parse_iso_datetime,
    normalize_usage_payload, _normalize_model_name,
    _usage_event_model_name, _usage_event_source,
    _pricing_entry_for_model, _usage_event_cost,
    _premium_request_multiplier, _month_key, month_key_for_source_row,
)


# ─── SQLite cache state ──────────────────────────────────────────────────────

_sqlite_cache_lock = Lock()
_sqlite_cache_enabled = True
_sqlite_cache_error = None


# ─── Dashboard SSE stream state ──────────────────────────────────────────────

_dashboard_stream_subscribers = set()
_dashboard_stream_lock = Lock()
_dashboard_stream_version = 0


# ─── Premium cache state ─────────────────────────────────────────────────────

_premium_cache_lock = Lock()
_premium_cache = {
    "loaded_at": 0.0,
    "payload": None,
    "refreshing": False,
    "last_error": None,
    "last_started_at": None,
}


# ─── Runtime dependencies ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class DashboardDependencies:
    load_billing_token: Callable[[], str | None] = lambda: None
    load_access_token: Callable[[], str | None] = lambda: None
    load_api_key_payload: Callable[[], dict] = lambda: {}
    snapshot_all_usage_events: Callable[[], list[dict]] = lambda: []
    snapshot_usage_events: Callable[[], list[dict]] = lambda: []


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


# ─── GitHub billing API ──────────────────────────────────────────────────────

def _infer_premium_allowance(api_key_payload: dict) -> int | None:
    override = os.environ.get("GHCP_PREMIUM_REQUESTS_INCLUDED")
    if override:
        return _coerce_int(override, default=None)

    sku = str(api_key_payload.get("sku") or "").strip().lower()
    if not sku:
        return None

    if "plus" in sku:
        return SKU_PREMIUM_ALLOWANCES["plus"]
    if "enterprise" in sku:
        return SKU_PREMIUM_ALLOWANCES["enterprise"]
    if "business" in sku:
        return SKU_PREMIUM_ALLOWANCES["business"]
    if "pro" in sku:
        return SKU_PREMIUM_ALLOWANCES["pro"]
    if "free" in sku:
        return SKU_PREMIUM_ALLOWANCES["free"]
    return None


def _extract_quota_summary(api_key_payload: dict) -> dict:
    included = _infer_premium_allowance(api_key_payload)
    sku = api_key_payload.get("sku")
    reset_date = api_key_payload.get("limited_user_reset_date")
    quotas = api_key_payload.get("limited_user_quotas")
    remaining = None
    if isinstance(quotas, dict):
        for key in ("remaining", "premium_requests_remaining", "available", "left"):
            if key in quotas:
                remaining = _coerce_float(quotas.get(key), default=None)
                break

    return {
        "sku": sku,
        "included": included,
        "official_remaining": remaining,
        "reset_date": reset_date,
    }


def _github_rest_headers(access_token: str, scheme: str = "Bearer") -> dict:
    authorization = (
        f"Bearer {access_token}"
        if scheme.lower() == "bearer"
        else f"token {access_token}"
    )
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": authorization,
        "X-GitHub-Api-Version": "2026-03-10",
        "User-Agent": "ghcp-proxy-dashboard",
    }


def _github_rest_get_json(access_token: str, url: str, params: dict | None = None) -> tuple[dict, str]:
    if not access_token:
        raise RuntimeError("No GitHub token provided for GitHub REST call")
    last_error = None
    for scheme in ("Bearer", "token"):
        headers = _github_rest_headers(access_token, scheme=scheme)
        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(url, headers=headers, params=params)
                status = response.status_code
                if status == 401:
                    last_error = f"{scheme} auth failed: 401 unauthorized"
                    continue
                if status == 403:
                    body = (response.text or "").strip()
                    last_error = (
                        f"{scheme} auth got 403 forbidden: {body[:240] if body else 'forbidden'}"
                    )
                    if scheme == "Bearer":
                        continue
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"GitHub REST request to {url} timed out: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"GitHub REST response from {url} was not JSON: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            if scheme == "Bearer" and exc.response.status_code in {401, 403}:
                continue
            raise RuntimeError(f"GitHub REST request to {url} failed: {exc.response.status_code} {exc.response.text}") from exc
        except Exception as exc:
            raise RuntimeError(f"GitHub REST request to {url} failed ({scheme}): {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"GitHub REST response from {url} had non-dict payload")
        return payload, scheme
    raise RuntimeError(last_error or f"GitHub REST request to {url} failed with unsupported authentication scheme")


def _load_cached_github_identity() -> dict | None:
    return _sqlite_cache_get("github_identity")


def _fetch_github_identity(access_token: str) -> dict:
    cached = _load_cached_github_identity()
    if isinstance(cached, dict) and isinstance(cached.get("login"), str) and cached.get("login"):
        return cached

    payload, _ = _github_rest_get_json(access_token, "https://api.github.com/user")
    if not isinstance(payload, dict) or not isinstance(payload.get("login"), str):
        raise RuntimeError("GitHub user API did not return a valid login")

    identity = {"login": payload["login"]}
    _sqlite_cache_put("github_identity", identity)
    return identity


def _load_billing_org_candidates(access_token: str) -> list[str]:
    try:
        payload, _ = _github_rest_get_json(access_token, "https://api.github.com/user/orgs")
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    candidates = []
    for org in payload:
        if not isinstance(org, dict):
            continue
        login = org.get("login")
        if isinstance(login, str) and login:
            candidates.append(login)
    return candidates


def _billing_target_from_env_or_identity(identity: dict) -> tuple[str, str]:
    scope = str(os.environ.get("GHCP_GITHUB_BILLING_SCOPE") or "user").strip().lower()
    target = str(os.environ.get("GHCP_GITHUB_BILLING_TARGET") or "").strip()
    if scope == "user":
        return "user", target or identity["login"]
    if scope in {"organization", "org"}:
        if not target:
            raise RuntimeError("GHCP_GITHUB_BILLING_TARGET is required when GHCP_GITHUB_BILLING_SCOPE=org")
        return "org", target
    if scope == "enterprise":
        if not target:
            raise RuntimeError("GHCP_GITHUB_BILLING_TARGET is required when GHCP_GITHUB_BILLING_SCOPE=enterprise")
        return "enterprise", target
    raise RuntimeError(f"Unsupported GHCP_GITHUB_BILLING_SCOPE: {scope}")


def _official_premium_cache_key(scope: str, target: str, year: int, month: int) -> str:
    return f"premium_usage:{scope}:{target}:{year:04d}:{month:02d}"


def _premium_usage_endpoint(scope: str, target: str) -> str:
    if scope == "user":
        return f"https://api.github.com/users/{target}/settings/billing/premium_request/usage"
    if scope == "org":
        return f"https://api.github.com/organizations/{target}/settings/billing/premium_request/usage"
    if scope == "enterprise":
        return f"https://api.github.com/enterprises/{target}/settings/billing/premium_request/usage"
    raise RuntimeError(f"Unsupported billing scope: {scope}")


def _extract_explicit_remaining_from_billing_payload(payload: dict) -> float | None:
    if not isinstance(payload, dict):
        return None
    candidate_paths = (
        ("remainingQuota",),
        ("remaining",),
        ("quota", "remaining"),
        ("includedUsage", "remaining"),
        ("entitlement", "remaining"),
    )
    for path in candidate_paths:
        current = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return _coerce_float(current, default=None)
    return None


def _extract_included_from_billing_payload(payload: dict) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("included", "entitlement", "allowed", "quota"):
        if key in payload:
            value = _coerce_int(payload.get(key), default=None)
            if value is not None:
                return value
    usage_summary = payload.get("summary")
    if isinstance(usage_summary, dict):
        for key in ("included", "allowed", "quota"):
            if key in usage_summary:
                value = _coerce_int(usage_summary.get(key), default=None)
                if value is not None:
                    return value
    return None


def _infer_remaining_from_billing_payload(payload: dict, included: int | None) -> tuple[float | None, float]:
    usage_items = payload.get("usageItems") if isinstance(payload, dict) else None
    if not isinstance(usage_items, list):
        return None, 0.0

    included_used = 0.0
    total_used = 0.0
    for item in usage_items:
        if not isinstance(item, dict):
            continue
        total_used += _coerce_float(item.get("grossQuantity"))
        discount_quantity = item.get("discountQuantity")
        net_quantity = item.get("netQuantity")
        if discount_quantity is not None:
            included_used += _coerce_float(discount_quantity)
        elif net_quantity is not None:
            included_used += max(_coerce_float(item.get("grossQuantity")) - _coerce_float(net_quantity), 0.0)
    if included is None:
        return None, total_used
    return max(included - included_used, 0.0), total_used


def _empty_official_premium_payload() -> dict:
    return {
        "available": False,
        "loaded_at": None,
        "scope": None,
        "target": None,
        "remaining": None,
        "used": 0.0,
        "included": None,
        "reset_date": None,
        "source": "github-rest-billing-api",
        "raw": None,
        "error": None,
        "inference": None,
    }


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
    premium_payload = dashboard_cache_store.get_latest("premium_usage:")
    if isinstance(premium_payload, dict):
        with _premium_cache_lock:
            _premium_cache["payload"] = premium_payload
            _premium_cache["loaded_at"] = _monotonic_loaded_at_from_payload(premium_payload)
            _premium_cache["last_error"] = premium_payload.get("error")
            _premium_cache["refreshing"] = False
            _premium_cache["last_started_at"] = None


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
    if not isinstance(event_cost, (int, float)):
        event_cost = _usage_event_cost(_usage_event_model_name(event), usage)

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

    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["total_tokens"] += total_tokens
    bucket["cached_input_tokens"] += cached_input_tokens
    bucket["cache_creation_tokens"] += cache_creation_tokens
    bucket["reasoning_output_tokens"] += reasoning_output_tokens
    bucket["cost_usd"] += _coerce_float(event_cost)
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


def normalize_session(source: str, session: dict) -> dict:
    if source == "claude":
        models = session.get("modelsUsed") or list((session.get("models") or {}).keys())
        cost_usd = _coerce_float(session.get("totalCost"))
        if cost_usd == 0.0 and session.get("costUSD") is not None:
            cost_usd = _coerce_float(session.get("costUSD"))
        cached_tokens = _coerce_int(session.get("cacheReadTokens"))
        if cached_tokens == 0 and session.get("cachedInputTokens") is not None:
            cached_tokens = _coerce_int(session.get("cachedInputTokens"))
        cache_creation_tokens = _coerce_int(session.get("cacheCreationTokens"))
        if cache_creation_tokens == 0 and session.get("cacheCreationInputTokens") is not None:
            cache_creation_tokens = _coerce_int(session.get("cacheCreationInputTokens"))
        reasoning_tokens = 0
    else:
        models = list((session.get("models") or {}).keys())
        cost_usd = _coerce_float(session.get("costUSD"))
        cached_tokens = _coerce_int(session.get("cachedInputTokens"))
        cache_creation_tokens = 0
        reasoning_tokens = _coerce_int(session.get("reasoningOutputTokens"))

    return {
        "source": source,
        "session_id": session.get("sessionId"),
        "session_kind": session.get("sessionKind") or "unknown",
        "session_display_id": session.get("sessionDisplayId") or session.get("sessionId"),
        "last_activity": session.get("lastActivity"),
        "project_path": session.get("projectPath"),
        "input_tokens": _coerce_int(session.get("inputTokens")),
        "output_tokens": _coerce_int(session.get("outputTokens")),
        "total_tokens": _coerce_int(session.get("totalTokens")),
        "cached_input_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "cost_usd": cost_usd,
        "models": models,
    }


def _normalize_month_row(source: str, row: dict) -> dict:
    if source == "claude":
        models = row.get("modelsUsed") or []
        cost_usd = _coerce_float(row.get("totalCost"))
        cached_tokens = _coerce_int(row.get("cacheReadTokens"))
        cache_creation_tokens = _coerce_int(row.get("cacheCreationTokens"))
        reasoning_tokens = 0
    else:
        models = list((row.get("models") or {}).keys())
        cost_usd = _coerce_float(row.get("costUSD"))
        cached_tokens = _coerce_int(row.get("cachedInputTokens"))
        cache_creation_tokens = 0
        reasoning_tokens = _coerce_int(row.get("reasoningOutputTokens"))

    return {
        "source": source,
        "month_key": month_key_for_source_row(source, row),
        "month_label": row.get("month"),
        "input_tokens": _coerce_int(row.get("inputTokens")),
        "output_tokens": _coerce_int(row.get("outputTokens")),
        "total_tokens": _coerce_int(row.get("totalTokens")),
        "cached_input_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "cost_usd": cost_usd,
        "models": models,
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
                "cost_usd": 0.0,
                "sources": {},
            },
        )
        current["input_tokens"] += row.get("input_tokens", 0)
        current["output_tokens"] += row.get("output_tokens", 0)
        current["total_tokens"] += row.get("total_tokens", 0)
        current["cached_input_tokens"] += row.get("cached_input_tokens", 0)
        current["cache_creation_tokens"] += row.get("cache_creation_tokens", 0)
        current["reasoning_output_tokens"] += row.get("reasoning_output_tokens", 0)
        current["cost_usd"] += row.get("cost_usd", 0.0)
        current["sources"][row["source"]] = row

    return [grouped[key] | {"cost_usd": round(grouped[key]["cost_usd"], 4)} for key in sorted(grouped.keys(), reverse=True)]


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
                "cost_usd": 0.0,
                "models": [],
            },
        )
        current["input_tokens"] += row.get("input_tokens", 0)
        current["output_tokens"] += row.get("output_tokens", 0)
        current["total_tokens"] += row.get("total_tokens", 0)
        current["cached_input_tokens"] += row.get("cached_input_tokens", 0)
        current["cache_creation_tokens"] += row.get("cache_creation_tokens", 0)
        current["reasoning_output_tokens"] += row.get("reasoning_output_tokens", 0)
        current["cost_usd"] += row.get("cost_usd", 0.0)
        current["models"] = sorted(set(current["models"]) | set(row.get("models") or []))

    combined = {
        "input_tokens": sum(item.get("input_tokens", 0) for item in per_source.values()),
        "output_tokens": sum(item.get("output_tokens", 0) for item in per_source.values()),
        "total_tokens": sum(item.get("total_tokens", 0) for item in per_source.values()),
        "cached_input_tokens": sum(item.get("cached_input_tokens", 0) for item in per_source.values()),
        "cache_creation_tokens": sum(item.get("cache_creation_tokens", 0) for item in per_source.values()),
        "reasoning_output_tokens": sum(item.get("reasoning_output_tokens", 0) for item in per_source.values()),
        "cost_usd": round(sum(item.get("cost_usd", 0.0) for item in per_source.values()), 4),
        "sources": {
            source: item | {"cost_usd": round(item.get("cost_usd", 0.0), 4)}
            for source, item in per_source.items()
        },
    }
    if month_key is not None:
        combined["month_key"] = month_key
    return combined


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
        thread_class: Callable[[], type] | type = Thread,
    ):
        self.dependencies = dependencies or DashboardDependencies()
        self.utc_now = utc_now
        self.utc_now_iso = utc_now_iso
        self.sqlite_cache_put = sqlite_cache_put
        self.notify_dashboard_stream_listeners = notify_dashboard_stream_listeners
        self.thread_class = thread_class

    def _resolved_thread_class(self):
        resolved_thread_class = self.thread_class
        if callable(resolved_thread_class) and not isinstance(resolved_thread_class, type):
            resolved_thread_class = resolved_thread_class()
        return resolved_thread_class

    def reset_official_premium_cache(self):
        with _premium_cache_lock:
            _premium_cache.update(
                {
                    "loaded_at": 0.0,
                    "payload": None,
                    "refreshing": False,
                    "last_error": None,
                    "last_started_at": None,
                }
            )

    def official_premium_cache_state(self) -> dict:
        with _premium_cache_lock:
            return dict(_premium_cache)

    def collect_official_premium_payload(
        self,
        now: datetime | None = None,
        *,
        skip_cache: bool = False,
    ) -> dict:
        current = now or self.utc_now()
        access_token = self.dependencies.load_billing_token() or self.dependencies.load_access_token()
        if not access_token:
            raise RuntimeError(
                "No GitHub OAuth token is available for billing API requests. "
                "Set GHCP_GITHUB_BILLING_TOKEN, set a billing token via /api/config/billing-token (UI), or run proxy auth."
            )

        identity = _fetch_github_identity(access_token)
        scope, target = _billing_target_from_env_or_identity(identity)
        api_key_payload = self.dependencies.load_api_key_payload()
        if not isinstance(api_key_payload, dict):
            api_key_payload = {}
        quota_summary = _extract_quota_summary(api_key_payload)
        included = quota_summary.get("included")
        explicit_scope = os.environ.get("GHCP_GITHUB_BILLING_SCOPE")
        candidates: list[tuple[str, str]] = [(scope, target)]
        seen_candidates = {f"{scope}:{target}"}
        if not explicit_scope:
            for org_login in _load_billing_org_candidates(access_token):
                candidate_key = f"org:{org_login}"
                if candidate_key in seen_candidates:
                    continue
                seen_candidates.add(candidate_key)
                candidates.append(("org", org_login))

        payload = None
        cache_key = None
        last_error = None
        for attempt_scope, attempt_target in candidates:
            cache_key = _official_premium_cache_key(attempt_scope, attempt_target, current.year, current.month)
            if not skip_cache:
                cached = _sqlite_cache_get(cache_key)
                if isinstance(cached, dict):
                    return cached

            endpoint = _premium_usage_endpoint(attempt_scope, attempt_target)
            fetch_attempts = (
                ({"year": current.year, "month": current.month}, "with month filter"),
                ({}, "without month filter"),
            )
            for attempt_params, attempt_label in fetch_attempts:
                try:
                    payload, _ = _github_rest_get_json(access_token, endpoint, params=attempt_params)
                    scope, target = attempt_scope, attempt_target
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if "without month filter" in attempt_label:
                        print(
                            f"Billing API fallback attempt failed for {attempt_scope}:{attempt_target} "
                            f"({attempt_label}): {exc}",
                            flush=True,
                        )
                        continue
                    print(f"Billing API month-filtered call failed ({attempt_label}): {exc}", flush=True)
            if payload is not None:
                break

        if payload is None or cache_key is None:
            raise RuntimeError(f"GitHub billing API failed after probing identities: {last_error}")

        endpoint_included = _extract_included_from_billing_payload(payload if isinstance(payload, dict) else {})
        effective_included = _coerce_int(endpoint_included, default=None)
        if effective_included is None:
            effective_included = _coerce_int(included, default=None)

        explicit_remaining = _extract_explicit_remaining_from_billing_payload(payload if isinstance(payload, dict) else {})
        inferred_remaining, total_used = _infer_remaining_from_billing_payload(
            payload if isinstance(payload, dict) else {},
            effective_included,
        )
        reset_date = quota_summary.get("reset_date")
        if isinstance(payload, dict):
            reset_date = payload.get("resetDate") or payload.get("reset_date") or reset_date
        result = {
            "available": True,
            "loaded_at": self.utc_now_iso(),
            "scope": scope,
            "target": target,
            "remaining": explicit_remaining if explicit_remaining is not None else inferred_remaining,
            "used": total_used,
            "included": effective_included,
            "reset_date": reset_date,
            "source": "github-rest-billing-api",
            "raw": payload if isinstance(payload, dict) else None,
            "error": None,
            "inference": "explicit" if explicit_remaining is not None else "usageItems",
        }
        self.sqlite_cache_put(cache_key, result)
        self.sqlite_cache_put("premium_usage:latest", result)
        return result

    def refresh_official_premium_cache_sync(self):
        with _premium_cache_lock:
            _premium_cache["refreshing"] = True
            _premium_cache["last_started_at"] = self.utc_now_iso()

        try:
            result = self.collect_official_premium_payload(skip_cache=True)
            self.sqlite_cache_put("premium_usage:latest", result)
            with _premium_cache_lock:
                _premium_cache["loaded_at"] = time.monotonic()
                _premium_cache["payload"] = result
                _premium_cache["last_error"] = None
            self.notify_dashboard_stream_listeners()
        except Exception as exc:
            with _premium_cache_lock:
                _premium_cache["last_error"] = str(exc)
        finally:
            with _premium_cache_lock:
                _premium_cache["refreshing"] = False

    def trigger_official_premium_refresh(self, force: bool = False):
        with _premium_cache_lock:
            payload = _premium_cache.get("payload")
            loaded_at = _premium_cache.get("loaded_at", 0.0)
            refreshing = _premium_cache.get("refreshing", False)
            is_stale = payload is None or (time.monotonic() - loaded_at) >= PREMIUM_CACHE_TTL_SECONDS
            should_refresh = force or is_stale
            if refreshing or not should_refresh:
                return
            _premium_cache["refreshing"] = True
            _premium_cache["last_started_at"] = self.utc_now_iso()

        collector = self.collect_official_premium_payload
        sqlite_cache_put = self.sqlite_cache_put
        notify_listeners = self.notify_dashboard_stream_listeners

        def _runner():
            try:
                result = collector(skip_cache=force)
                sqlite_cache_put("premium_usage:latest", result)
                with _premium_cache_lock:
                    _premium_cache["loaded_at"] = time.monotonic()
                    _premium_cache["payload"] = result
                    _premium_cache["last_error"] = None
                notify_listeners()
            except Exception as exc:
                with _premium_cache_lock:
                    _premium_cache["last_error"] = str(exc)
            finally:
                with _premium_cache_lock:
                    _premium_cache["refreshing"] = False

        self._resolved_thread_class()(target=_runner, daemon=True).start()

    def get_official_premium_payload(self, force_refresh: bool = False) -> dict:
        if force_refresh:
            self.refresh_official_premium_cache_sync()
        else:
            self.trigger_official_premium_refresh(force=False)
        with _premium_cache_lock:
            payload = _premium_cache.get("payload") or _empty_official_premium_payload()
            age_seconds = None
            loaded_at = _premium_cache.get("loaded_at", 0.0)
            if loaded_at:
                age_seconds = max(0.0, time.monotonic() - loaded_at)
            return {
                **payload,
                "refreshing": bool(_premium_cache.get("refreshing")),
                "age_seconds": age_seconds,
                "last_error": _premium_cache.get("last_error"),
                "last_started_at": _premium_cache.get("last_started_at"),
            }

    def build_payload(self, force_refresh: bool = False) -> dict:
        now = self.utc_now()
        month_start, month_end = _current_billing_month_bounds(now)
        current_month_key = _month_key(now)
        usage_events = self.dependencies.snapshot_all_usage_events()
        detailed_usage_events = self.dependencies.snapshot_usage_events()
        current_month_events = []
        for event in usage_events:
            recorded_at = _parse_iso_datetime(event.get("finished_at") or event.get("started_at"))
            if recorded_at is None:
                continue
            if month_start <= recorded_at < month_end:
                current_month_events.append(event)

        premium_used = round(sum(_coerce_float(event.get("premium_requests")) for event in current_month_events), 2)
        api_key_payload = self.dependencies.load_api_key_payload()
        if not isinstance(api_key_payload, dict):
            api_key_payload = {}
        quota_summary = _extract_quota_summary(api_key_payload)
        included = quota_summary.get("included")
        has_tracked_premium_data = bool(current_month_events)
        tracked_remaining = None
        if included is not None and has_tracked_premium_data:
            tracked_remaining = round(max(included - premium_used, 0.0), 2)
        official_premium = self.get_official_premium_payload(force_refresh=force_refresh)
        official_included = official_premium.get("included")
        official_used = official_premium.get("used")
        official_remaining = official_premium.get("remaining")
        official_reset_date = official_premium.get("reset_date")
        official_available = bool(official_premium.get("available"))

        local_usage = collect_local_dashboard_usage(usage_events)
        month_rows = list(local_usage.get("month_rows") or [])
        current_month_usage = _combine_usage_rows(
            [row for row in month_rows if row.get("month_key") == current_month_key],
            month_key=current_month_key,
        )
        all_time_usage = _combine_usage_rows(month_rows)
        all_time_usage["months_tracked"] = len(local_usage.get("month_history") or [])

        recent_requests = sorted(
            detailed_usage_events,
            key=lambda item: item.get("finished_at") or item.get("started_at") or "",
            reverse=True,
        )[:DETAILED_REQUEST_HISTORY_LIMIT]

        return {
            "generated_at": now.isoformat(),
            "premium": {
                "sku": quota_summary.get("sku") or official_premium.get("raw", {}).get("sku"),
                "included": official_included if official_included is not None else included,
                "used": official_used if official_available and official_used is not None else premium_used,
                "tracked_remaining": tracked_remaining,
                "has_tracked_data": has_tracked_premium_data,
                "official_remaining": official_remaining if official_available else quota_summary.get("official_remaining"),
                "reset_date": official_reset_date if official_reset_date is not None else quota_summary.get("reset_date"),
                "source": official_premium.get("source") if official_available else "proxy-request-log",
                "official": official_premium,
            },
            "current_month": {
                "label": current_month_key,
                "start_at": month_start.isoformat(),
                "end_at": month_end.isoformat(),
                "proxy_requests": len(current_month_events),
                "sessions": local_usage.get("session_count", 0),
                "usage": current_month_usage,
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
