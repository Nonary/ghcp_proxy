"""Ingest native Codex CLI traffic into the proxy's usage log.

Codex stores per-turn usage in `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`
files, with `token_count` events that include the cumulative and per-turn
input/cached/output/reasoning token counts plus rate-limit metadata. This
module scans those files, converts each turn into a usage event compatible
with `UsageTracker._persist_event`, and tags it with `native_source =
"codex_native"` so the dashboard can break it out separately from proxied
("Copilot-backed") Codex traffic.

The ingestor is deliberately read-only against the Codex files. State for
which turns have already been ingested is kept in a small JSON cursor file
under `~/.config/ghcp_proxy/codex-native-cursor.json`, so re-running is
idempotent and incremental.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from constants import TOKEN_DIR
from util import _codex_logs_service_tiers, _usage_event_estimated_cost, _normalize_model_name


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CODEX_HOME = os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
CODEX_SESSIONS_DIR = os.path.join(CODEX_HOME, "sessions")
CURSOR_FILE = os.path.join(TOKEN_DIR, "codex-native-cursor.json")

# Source label written into emitted events (see util._usage_event_source).
NATIVE_SOURCE_LABEL = "codex_native"


# ---------------------------------------------------------------------------
# Cursor (per-rollout-file scan position)
# ---------------------------------------------------------------------------

_cursor_lock = threading.Lock()


def _load_cursor() -> dict:
    try:
        with open(CURSOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"files": {}}
    if not isinstance(data, dict):
        return {"files": {}}
    files = data.get("files")
    if not isinstance(files, dict):
        data["files"] = {}
    return data


def _save_cursor(cursor: dict) -> None:
    os.makedirs(os.path.dirname(CURSOR_FILE) or TOKEN_DIR, exist_ok=True)
    tmp = CURSOR_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cursor, f, separators=(",", ":"))
    os.replace(tmp, CURSOR_FILE)


# ---------------------------------------------------------------------------
# Rollout discovery
# ---------------------------------------------------------------------------

def _iter_rollout_files() -> Iterable[str]:
    pattern = os.path.join(CODEX_SESSIONS_DIR, "*", "*", "*", "rollout-*.jsonl")
    return sorted(glob.glob(pattern))


# In-process cache of parsed header state per rollout file. Lets a partial
# (incremental) read pick up at `start_offset` without having to re-scan the
# entire file just to recover the session_id / current model / turn counter.
@dataclass
class _FileState:
    meta: "_SessionMeta"
    current_model: str | None
    current_effort: str | None
    current_turn_id: str | None
    turn_index: int
    last_offset: int


_file_state_cache: dict[str, _FileState] = {}
_file_state_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Per-file parser
# ---------------------------------------------------------------------------

@dataclass
class _SessionMeta:
    session_id: str | None = None
    cwd: str | None = None
    originator: str | None = None
    cli_version: str | None = None
    model_provider: str | None = None
    model: str | None = None  # falls back to turn_context.model
    started_at: str | None = None  # ISO8601 from session_meta.payload.timestamp


def _normalize_token_count_payload(payload: dict | None) -> dict | None:
    """Translate Codex `token_count.info.last_token_usage` to our usage shape.

    Codex/OpenAI report `input_tokens` as the *total* input including cached
    tokens, with `cached_input_tokens` being a subset of that total. The
    proxy's cost math (`util._usage_event_cost_breakdown`) bills
    `input_tokens` at the fresh rate AND `cached_input_tokens` at the cached
    rate as separate buckets, so we must subtract the cached portion here to
    avoid double-billing the cached tokens.
    """
    if not isinstance(payload, dict):
        return None
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    last = info.get("last_token_usage")
    if not isinstance(last, dict):
        return None
    raw_input = int(last.get("input_tokens", 0) or 0)
    cached = int(last.get("cached_input_tokens", 0) or 0)
    # Guard: cached should never exceed total input, but clamp defensively.
    if cached > raw_input:
        cached = raw_input
    fresh_input = raw_input - cached
    return {
        "input_tokens": fresh_input,
        "output_tokens": int(last.get("output_tokens", 0) or 0),
        "total_tokens": int(last.get("total_tokens", 0) or 0),
        "cached_input_tokens": cached,
        "cache_creation_input_tokens": 0,
        "reasoning_output_tokens": int(last.get("reasoning_output_tokens", 0) or 0),
    }


def _parse_rollout(
    path: str,
    start_offset: int,
) -> tuple[list[dict], int]:
    """Return (new_events, new_byte_offset). Reads incrementally from `start_offset`.

    Uses an in-process cache so we don't have to re-scan the whole file from
    byte 0 just to recover session_meta / current_model state. Cache is keyed
    by `path` and invalidated on truncation or when start_offset jumps backward.
    """
    events: list[dict] = []

    try:
        size = os.path.getsize(path)
    except OSError:
        return events, start_offset

    if start_offset > size:
        # File rotated/truncated — drop cache and restart.
        with _file_state_lock:
            _file_state_cache.pop(path, None)
        start_offset = 0

    with _file_state_lock:
        cached = _file_state_cache.get(path)

    if cached is not None and cached.last_offset == start_offset:
        # Header state still good; resume from where we left off.
        meta = cached.meta
        current_model = cached.current_model
        current_effort = cached.current_effort
        current_turn_id = cached.current_turn_id
        turn_index = cached.turn_index
        seek_to = start_offset
        replay_header_only = False
    else:
        # No cache (cold start or file changed in unexpected way) — replay
        # from byte 0 to recover header context, but only emit events whose
        # source byte offset is >= start_offset.
        meta = _SessionMeta()
        current_model = None
        current_effort = None
        current_turn_id = None
        turn_index = 0
        seek_to = 0
        replay_header_only = True

    try:
        with open(path, "rb") as f:
            f.seek(seek_to)
            byte_pos = seek_to
            while True:
                line = f.readline()
                if not line:
                    break
                line_start = byte_pos
                byte_pos += len(line)
                try:
                    decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
                except Exception:
                    continue
                if not decoded:
                    continue
                try:
                    record = json.loads(decoded)
                except json.JSONDecodeError:
                    continue
                rtype = record.get("type")
                payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}

                if rtype == "session_meta":
                    meta.session_id = payload.get("id") or meta.session_id
                    meta.cwd = payload.get("cwd") or meta.cwd
                    meta.originator = payload.get("originator") or meta.originator
                    meta.cli_version = payload.get("cli_version") or meta.cli_version
                    meta.model_provider = payload.get("model_provider") or meta.model_provider
                    meta.started_at = payload.get("timestamp") or meta.started_at
                    continue

                if rtype == "turn_context":
                    model_value = payload.get("model")
                    if isinstance(model_value, str) and model_value:
                        current_model = model_value
                        if not meta.model:
                            meta.model = model_value
                    effort_value = payload.get("effort")
                    if isinstance(effort_value, str) and effort_value:
                        current_effort = effort_value
                    continue

                if rtype != "event_msg":
                    continue
                inner = payload
                if inner.get("type") == "task_started":
                    turn_id_value = inner.get("turn_id")
                    if isinstance(turn_id_value, str) and turn_id_value:
                        current_turn_id = turn_id_value
                    continue
                if inner.get("type") != "token_count":
                    continue
                # If this session used a proxy, it will be ingested from the proxy logs.
                # Skip native ingest to avoid double-counting.
                if meta.model_provider == "custom":
                    continue
                # When we're replaying the header from byte 0 because the
                # cache was cold, skip emitting events we've already persisted.
                if replay_header_only and line_start < start_offset:
                    turn_index += 1
                    continue

                usage = _normalize_token_count_payload(inner)
                if usage is None:
                    continue
                if usage["total_tokens"] == 0:
                    continue

                model_name = current_model or meta.model or "gpt-5"
                normalized_model = _normalize_model_name(model_name) or model_name

                rate_limits = inner.get("rate_limits") if isinstance(inner.get("rate_limits"), dict) else None
                plan_type_raw = rate_limits.get("plan_type") if isinstance(rate_limits, dict) else None
                plan_type = plan_type_raw.strip() if isinstance(plan_type_raw, str) and plan_type_raw.strip() else None

                ts = record.get("timestamp")
                request_id = f"codex-native:{meta.session_id or os.path.basename(path)}:{turn_index}"
                turn_index += 1
                native_service_tiers = _codex_logs_service_tiers(meta.session_id, current_turn_id, ts)
                native_requested_service_tier = native_service_tiers.get("requested")
                native_service_tier = native_service_tiers.get("effective")

                event = {
                    "request_id": request_id,
                    "started_at": ts,
                    "finished_at": ts,
                    "path": "/native/codex/responses",
                    "method": "POST",
                    "requested_model": normalized_model,
                    "resolved_model": normalized_model,
                    "response_model": normalized_model,
                    "initiator": "user",
                    "session_id": meta.session_id,
                    "session_id_origin": "codex_rollout",
                    "project_path": meta.cwd,
                    "client_request_id": None,
                    "subagent": None,
                    "server_request_id": meta.session_id,
                    "status_code": 200,
                    "success": True,
                    "duration_ms": None,
                    "time_to_first_token_ms": None,
                    "usage": usage,
                    "premium_requests": 0.0,
                    "quota_snapshots": None,
                    "rate_limit": rate_limits,
                    "native_source": NATIVE_SOURCE_LABEL,
                    "native_origin": meta.originator,
                    "native_cli_version": meta.cli_version,
                    "native_model_provider": meta.model_provider,
                    "native_plan_type": plan_type,
                    "native_requested_service_tier": native_requested_service_tier,
                    "native_requested_service_tier_source": native_service_tiers.get("requested_source"),
                    "native_service_tier": native_service_tier,
                    "native_service_tier_source": native_service_tiers.get("effective_source"),
                    "native_reasoning_effort": current_effort,
                    "native_turn_id": current_turn_id,
                    "native_rollout_path": path,
                }
                event["cost_usd"] = round(
                    _usage_event_estimated_cost(event, model_name=normalized_model, usage=usage),
                    6,
                )
                events.append(event)
            new_offset = byte_pos
    except OSError:
        return events, start_offset

    # Cache the parsed header state so the next scan can seek directly.
    with _file_state_lock:
        _file_state_cache[path] = _FileState(
            meta=meta,
            current_model=current_model,
            current_effort=current_effort,
            current_turn_id=current_turn_id,
            turn_index=turn_index,
            last_offset=new_offset,
        )

    return events, new_offset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_once(
    record: Callable[[dict], None],
    *,
    cursor: dict | None = None,
) -> dict:
    """Scan all rollout files once, emitting unseen `token_count` events.

    `record` is invoked synchronously for each new event (typically
    `usage_tracker.record_usage_event`). Returns the updated cursor (also
    persisted to disk).
    """
    if cursor is None:
        with _cursor_lock:
            cursor = _load_cursor()
    files_state: dict = cursor.setdefault("files", {})

    if not os.path.isdir(CODEX_SESSIONS_DIR):
        return cursor

    dirty = False
    for path in _iter_rollout_files():
        prior = files_state.get(path) or {}
        prior_offset = int(prior.get("offset", 0)) if isinstance(prior, dict) else 0
        prior_mtime = float(prior.get("mtime", 0.0)) if isinstance(prior, dict) else 0.0

        # Skip files that haven't been modified since the last scan AND
        # whose offset already covers the full file.
        try:
            current_mtime = os.path.getmtime(path)
            current_size = os.path.getsize(path)
        except OSError:
            continue
        if (
            prior_mtime
            and current_mtime <= prior_mtime
            and prior_offset >= current_size
        ):
            continue

        new_events, new_offset = _parse_rollout(path, prior_offset)
        for event in new_events:
            try:
                record(event)
            except Exception as exc:
                print(
                    f"codex_native_ingest: failed to record event from {path}: {exc}",
                    flush=True,
                )
        files_state[path] = {
            "offset": new_offset,
            "mtime": current_mtime,
            "updated_at": time.time(),
        }
        dirty = True

    if dirty:
        cursor["last_scan_at"] = time.time()
        with _cursor_lock:
            try:
                _save_cursor(cursor)
            except OSError as exc:
                print(f"codex_native_ingest: failed to persist cursor: {exc}", flush=True)
    return cursor


def start_background_scanner(
    record: Callable[[dict], None],
    *,
    interval_seconds: float = 5.0,
) -> threading.Thread:
    """Run `scan_once` immediately, then every `interval_seconds` in a daemon thread.

    Default 5s — most rollouts are small (write rate: a few token_count
    events per turn) and the per-scan cost is dominated by file mtime checks
    plus tail-reads from the cached offset, so polling is cheap.
    """

    def _run():
        # Initial scan, then loop.
        try:
            scan_once(record)
        except Exception as exc:
            print(f"codex_native_ingest: initial scan failed: {exc}", flush=True)
        while True:
            time.sleep(max(1.0, interval_seconds))
            try:
                scan_once(record)
            except Exception as exc:
                print(f"codex_native_ingest: periodic scan failed: {exc}", flush=True)

    thread = threading.Thread(target=_run, name="codex-native-ingest", daemon=True)
    thread.start()
    return thread

