"""Pure stateless utility functions for ghcp_proxy."""

import glob
import gzip
import json
import os
import re
import sqlite3
import time
import zlib

from datetime import datetime, timezone
from fastapi import HTTPException, Request

try:
    import compression.zstd as _stdlib_zstd
except ImportError:
    _stdlib_zstd = None

try:
    import zstandard as _zstandard
except ImportError:
    _zstandard = None

try:
    import brotli
except ImportError:
    brotli = None

from constants import MODEL_PRICING_ALIASES, MODEL_PRICING, PREMIUM_REQUEST_MULTIPLIERS


# ---------------------------------------------------------------------------
# JSON / coercion helpers
# ---------------------------------------------------------------------------


def zstd_available() -> bool:
    return _stdlib_zstd is not None or _zstandard is not None


def zstd_compress(data: bytes) -> bytes:
    if _stdlib_zstd is not None:
        return _stdlib_zstd.compress(data)
    if _zstandard is not None:
        return _zstandard.ZstdCompressor().compress(data)
    raise RuntimeError("zstd support requires Python 3.14+ or the zstandard package")


def zstd_decompress(data: bytes) -> bytes:
    if _stdlib_zstd is not None:
        return _stdlib_zstd.decompress(data)
    if _zstandard is not None:
        return _zstandard.ZstdDecompressor().decompress(data)
    raise RuntimeError("zstd support requires Python 3.14+ or the zstandard package")


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _coerce_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _month_key(value: datetime) -> str:
    return value.strftime("%Y-%m")


def month_key_for_source_row(source: str, row: dict) -> str | None:
    raw_value = row.get("month")
    if not isinstance(raw_value, str):
        return None
    for fmt in ("%Y-%m", "%b %Y"):
        try:
            return datetime.strptime(raw_value, fmt).strftime("%Y-%m")
        except ValueError:
            continue
    return raw_value if source == "claude" else None


# ---------------------------------------------------------------------------
# Usage / payload helpers
# ---------------------------------------------------------------------------

def _extract_payload_usage(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return None

    usage = payload.get("usage")
    return normalize_usage_payload(usage)


def normalize_usage_payload(usage: dict | None) -> dict | None:
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")

    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")

    cached_tokens = usage.get("cache_read_input_tokens")
    cached_tokens_from_raw_details = False
    if cached_tokens is None:
        cached_tokens = usage.get("cached_input_tokens")
    if cached_tokens is None:
        for details_key in ("input_tokens_details", "prompt_tokens_details"):
            details = usage.get(details_key)
            if isinstance(details, dict):
                cached_tokens = details.get("cached_tokens")
                if cached_tokens is not None:
                    cached_tokens_from_raw_details = True
                    break

    cache_creation_tokens = usage.get("cache_creation_input_tokens")
    reasoning_tokens = usage.get("reasoning_output_tokens")
    if reasoning_tokens is None:
        for details_key in ("output_tokens_details", "completion_tokens_details"):
            details = usage.get(details_key)
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens")
                if reasoning_tokens is not None:
                    break
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = usage.get("totalTokens")

    normalized_input_tokens = _coerce_int(input_tokens, default=0)
    normalized_output_tokens = _coerce_int(output_tokens, default=0)
    normalized_cached_tokens = _coerce_int(cached_tokens, default=0)
    normalized_cache_creation_tokens = _coerce_int(cache_creation_tokens, default=0)
    normalized_reasoning_tokens = _coerce_int(reasoning_tokens, default=0)
    if cached_tokens_from_raw_details and normalized_cached_tokens > 0:
        normalized_input_tokens = max(0, normalized_input_tokens - normalized_cached_tokens)

    normalized_total_tokens = _coerce_int(total_tokens, default=None)
    if cached_tokens_from_raw_details:
        normalized_total_tokens = normalized_input_tokens + normalized_output_tokens
    elif normalized_total_tokens is None:
        normalized_total_tokens = normalized_input_tokens + normalized_output_tokens

    return {
        "input_tokens": normalized_input_tokens,
        "output_tokens": normalized_output_tokens,
        "total_tokens": normalized_total_tokens,
        "cached_input_tokens": normalized_cached_tokens,
        "cache_creation_input_tokens": normalized_cache_creation_tokens,
        "reasoning_output_tokens": normalized_reasoning_tokens,
    }


# ---------------------------------------------------------------------------
# Request key / classification helpers
# ---------------------------------------------------------------------------

def _server_request_chain_key(
    session_id: str | None,
    client_request_id: str | None,
    subagent: str | None,
) -> tuple[str, str]:
    scope = None
    if isinstance(session_id, str) and session_id:
        scope = f"session:{session_id}"
    elif isinstance(client_request_id, str) and client_request_id:
        scope = f"client:{client_request_id}"
    else:
        scope = "global"
    normalized_subagent = subagent if isinstance(subagent, str) and subagent else "__root__"
    return (scope, normalized_subagent)


def _codex_native_session_id_from_request_id(request_id: str | None) -> str | None:
    if not isinstance(request_id, str):
        return None
    prefix = "codex-native:"
    if not request_id.startswith(prefix):
        return None
    remainder = request_id[len(prefix):]
    if not remainder:
        return None
    session_id, separator, turn_id = remainder.rpartition(":")
    if not separator or not session_id or not turn_id:
        return None
    return session_id


FAST_SERVICE_TIERS = frozenset({"fast", "priority"})


_CODEX_LOGS_SERVICE_TIER_CACHE: dict[str, object] = {
    "path": None,
    "mtime": None,
    "size": None,
    "values": {},
}
_CODEX_LOGS_SERVICE_TIER_RE = re.compile(r'"service_tier"\s*:\s*"([^"]+)"')


def _codex_logs_db_path(codex_home: str | None = None) -> str | None:
    home = codex_home or os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    if not isinstance(home, str) or not home:
        return None

    candidates: list[str] = []
    for pattern in ("logs_*.sqlite", "logs.sqlite"):
        candidates.extend(glob.glob(os.path.join(home, pattern)))
    if not candidates:
        return None

    def _sort_key(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    return max(candidates, key=_sort_key)


def _extract_codex_log_service_tier(feedback_log_body: str | None) -> str | None:
    if not isinstance(feedback_log_body, str) or not feedback_log_body:
        return None
    match = _CODEX_LOGS_SERVICE_TIER_RE.search(feedback_log_body)
    if not match:
        return None
    tier = match.group(1).strip().lower()
    return tier or None


def _codex_logs_service_tiers(
    session_id: str | None,
    turn_id: str | None = None,
    started_at: str | None = None,
    logs_db_path: str | None = None,
) -> dict[str, str | None]:
    empty = {
        "requested": None,
        "requested_source": None,
        "effective": None,
        "effective_source": None,
    }
    if not isinstance(session_id, str) or not session_id:
        return dict(empty)

    db_path = logs_db_path or _codex_logs_db_path()
    if not isinstance(db_path, str) or not db_path:
        return dict(empty)

    try:
        mtime = os.path.getmtime(db_path)
        size = os.path.getsize(db_path)
    except OSError:
        return dict(empty)

    cache_values = _CODEX_LOGS_SERVICE_TIER_CACHE.get("values")
    if not isinstance(cache_values, dict):
        cache_values = {}
    if (
        _CODEX_LOGS_SERVICE_TIER_CACHE.get("path") != db_path
        or _CODEX_LOGS_SERVICE_TIER_CACHE.get("mtime") != mtime
        or _CODEX_LOGS_SERVICE_TIER_CACHE.get("size") != size
    ):
        cache_values = {}
        _CODEX_LOGS_SERVICE_TIER_CACHE["path"] = db_path
        _CODEX_LOGS_SERVICE_TIER_CACHE["mtime"] = mtime
        _CODEX_LOGS_SERVICE_TIER_CACHE["size"] = size
        _CODEX_LOGS_SERVICE_TIER_CACHE["values"] = cache_values

    event_dt = _parse_iso_datetime(started_at)
    event_ts = int(event_dt.timestamp()) if event_dt else None
    cache_key = (session_id, turn_id or "", event_ts)
    if cache_key in cache_values:
        cached = cache_values.get(cache_key)
        return dict(cached) if isinstance(cached, dict) else dict(empty)

    def _lookup(query_turn_id: str | None) -> dict[str, str | None]:
        params: list[object] = [session_id]
        where_parts = [
            "thread_id = ?",
            "target = 'codex_api::endpoint::responses_websocket'",
            "instr(feedback_log_body, '\"service_tier\"') > 0",
            "("
            "instr(feedback_log_body, 'stream_request:model_client.stream_responses_websocket') > 0 "
            "or instr(feedback_log_body, 'websocket event: {\"type\":\"response.created\"') > 0 "
            "or instr(feedback_log_body, 'websocket event: {\"type\":\"response.in_progress\"') > 0 "
            "or instr(feedback_log_body, 'websocket event: {\"type\":\"response.completed\"') > 0"
            ")",
        ]
        if isinstance(query_turn_id, str) and query_turn_id:
            where_parts.append("feedback_log_body like ?")
            params.append(f"%turn.id={query_turn_id}%")

        order_clause = "ts ASC, ts_nanos ASC, id ASC"
        if query_turn_id is None and event_ts is not None:
            order_clause = "abs(ts - ?) ASC, ts ASC, ts_nanos ASC, id ASC"
            params.append(event_ts)

        sql = (
            "SELECT feedback_log_body FROM logs "
            f"WHERE {' AND '.join(where_parts)} "
            f"ORDER BY {order_clause} LIMIT 50"
        )
        resolved = dict(empty)
        try:
            with sqlite3.connect(db_path, timeout=1.0) as conn:
                conn.execute("PRAGMA query_only = ON")
                for (feedback_log_body,) in conn.execute(sql, params):
                    tier = _extract_codex_log_service_tier(feedback_log_body)
                    if not tier:
                        continue
                    body = feedback_log_body or ""
                    if (
                        '"stream":true' in body
                        and "stream_request:model_client.stream_responses_websocket" in body
                    ):
                        resolved["requested"] = tier
                        resolved["requested_source"] = "codex_logs_request"
                        continue
                    if 'websocket event: {"type":"response.completed"' in body:
                        resolved["effective"] = tier
                        resolved["effective_source"] = "codex_logs_response_completed"
                        continue
                    if (
                        resolved["effective"] is None
                        and (
                            'websocket event: {"type":"response.in_progress"' in body
                            or 'websocket event: {"type":"response.created"' in body
                        )
                    ):
                        resolved["effective"] = tier
                        resolved["effective_source"] = "codex_logs_response_progress"
        except sqlite3.Error:
            return dict(empty)
        return resolved

    resolved_tiers = _lookup(turn_id)
    if (
        resolved_tiers["requested"] is None
        and resolved_tiers["effective"] is None
        and turn_id
    ):
        resolved_tiers = _lookup(None)

    cache_values[cache_key] = dict(resolved_tiers)
    return dict(resolved_tiers)


def _is_claude_request(request: Request | None) -> bool:
    path = str(request.url.path if request is not None and hasattr(request, "url") else "")
    return path == "/v1/messages"


# ---------------------------------------------------------------------------
# Model name helpers
# ---------------------------------------------------------------------------

def _normalize_model_name(model_name: str | None) -> str | None:
    if not isinstance(model_name, str):
        return None
    normalized = model_name.strip().lower().replace("_", "-")
    if normalized.startswith("anthropic/"):
        normalized = normalized.split("/", 1)[1]
    if normalized.startswith("openai/"):
        normalized = normalized.split("/", 1)[1]
    normalized = MODEL_PRICING_ALIASES.get(normalized, normalized)
    if normalized not in MODEL_PRICING:
        undated = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", normalized)
        if undated != normalized:
            candidate = MODEL_PRICING_ALIASES.get(undated, undated)
            if candidate in MODEL_PRICING:
                normalized = candidate
    return normalized


def _usage_event_model_name(event: dict | None) -> str | None:
    if not isinstance(event, dict):
        return None

    for key in ("response_model", "resolved_model", "requested_model"):
        model_name = event.get(key)
        normalized = _normalize_model_name(model_name)
        if normalized:
            return normalized
    return None


def _usage_event_source(event: dict | None) -> str:
    if not isinstance(event, dict):
        return "codex"

    # Explicit native_source wins over heuristic detection. Used by
    # ingestors (e.g. codex_native_ingest) to mark traffic that did not flow
    # through this proxy but should still be counted toward expense.
    native_source = event.get("native_source")
    if isinstance(native_source, str) and native_source.strip():
        return native_source.strip()

    # Derive source from the originally requested model so that remapped
    # requests (e.g. Codex -> Claude) keep their original source identity.
    requested = _normalize_model_name(event.get("requested_model"))
    if isinstance(requested, str):
        if requested.startswith("claude-"):
            return "claude"
        if requested.startswith("gpt-"):
            return "codex"

    # Fall back to the resolved/response model when no requested_model exists.
    model_name = _usage_event_model_name(event)
    if isinstance(model_name, str):
        if model_name.startswith("claude-"):
            return "claude"
        if model_name.startswith("gpt-"):
            return "codex"

    path = str(event.get("path") or "")
    if path.endswith("/messages"):
        return "claude"
    if path.endswith("/responses") or path.endswith("/responses/compact") or path.endswith("/chat/completions"):
        return "codex"
    return "codex"

# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------

def _pricing_entry_for_model(model_name: str | None) -> dict | None:
    normalized = _normalize_model_name(model_name)
    if not normalized:
        return None
    return MODEL_PRICING.get(normalized)


def _anthropic_cache_creation_rate_per_million(entry: dict) -> float:
    input_rate = _coerce_float(entry.get("input_per_million"))
    cache_write_rate = _coerce_float(entry.get("cache_write_5m_per_million"))
    if cache_write_rate > 0:
        return cache_write_rate
    # The proxy does not record cache TTL separately, so default to the 5m write rate.
    return round(input_rate * 1.25, 6)


def _usage_event_cost_breakdown(model_name: str | None, usage: dict | None) -> dict[str, float]:
    breakdown = {
        "input_fresh": 0.0,
        "cached_input": 0.0,
        "cache_creation": 0.0,
        "output": 0.0,
    }
    if not isinstance(usage, dict):
        return breakdown

    entry = _pricing_entry_for_model(model_name)
    if not isinstance(entry, dict):
        return breakdown

    input_tokens = _coerce_int(usage.get("input_tokens"))
    output_tokens = _coerce_int(usage.get("output_tokens"))
    cached_input_tokens = _coerce_int(usage.get("cached_input_tokens"))
    if cached_input_tokens == 0 and usage.get("cache_read_input_tokens") is not None:
        cached_input_tokens = _coerce_int(usage.get("cache_read_input_tokens"))
    cache_creation_input_tokens = _coerce_int(usage.get("cache_creation_input_tokens"))
    reasoning_output_tokens = _coerce_int(usage.get("reasoning_output_tokens"))

    input_rate = _coerce_float(entry.get("input_per_million"))
    output_rate = _coerce_float(entry.get("output_per_million"))
    cached_rate = entry.get("cached_input_per_million")
    if cached_rate is None and str(entry.get("provider") or "").lower() == "anthropic":
        cached_rate = round(input_rate * 0.1, 6)
    cached_rate = _coerce_float(cached_rate, default=input_rate)

    cache_creation_rate = input_rate
    if str(entry.get("provider") or "").lower() == "anthropic":
        cache_creation_rate = _anthropic_cache_creation_rate_per_million(entry)

    billable_output_tokens = output_tokens + reasoning_output_tokens
    breakdown["input_fresh"] = (input_tokens * input_rate) / 1_000_000.0
    breakdown["cached_input"] = (cached_input_tokens * cached_rate) / 1_000_000.0
    breakdown["cache_creation"] = (cache_creation_input_tokens * cache_creation_rate) / 1_000_000.0
    breakdown["output"] = (billable_output_tokens * output_rate) / 1_000_000.0
    return breakdown


def _usage_event_cost(model_name: str | None, usage: dict | None) -> float:
    return sum(_usage_event_cost_breakdown(model_name, usage).values())


def _fast_service_tier_cost_multiplier(event: dict | None) -> float:
    if _usage_event_model_name(event) == "gpt-5.5":
        return 7.5
    return 2.0


def _usage_event_cost_multiplier(event: dict | None) -> float:
    if not isinstance(event, dict):
        return 1.0
    requested_service_tier = event.get("native_requested_service_tier")
    if isinstance(requested_service_tier, str) and requested_service_tier.strip().lower() in FAST_SERVICE_TIERS:
        return _fast_service_tier_cost_multiplier(event)
    service_tier = event.get("native_service_tier")
    if isinstance(service_tier, str) and service_tier.strip().lower() in FAST_SERVICE_TIERS:
        return _fast_service_tier_cost_multiplier(event)
    return 1.0


def _usage_event_estimated_cost(
    event: dict | None,
    *,
    model_name: str | None = None,
    usage: dict | None = None,
) -> float:
    resolved_model_name = model_name
    if resolved_model_name is None and isinstance(event, dict):
        resolved_model_name = _usage_event_model_name(event)
    resolved_usage = usage
    if resolved_usage is None and isinstance(event, dict):
        resolved_usage = event.get("usage")
    return _usage_event_cost(resolved_model_name, resolved_usage) * _usage_event_cost_multiplier(event)


def _premium_request_multiplier(model_name: str | None) -> float:
    normalized = _normalize_model_name(model_name)
    if not normalized:
        return 1.0
    return PREMIUM_REQUEST_MULTIPLIERS.get(normalized, 1.0)


def _counted_premium_requests(event: dict | None) -> float:
    if not isinstance(event, dict):
        return 0.0
    initiator = event.get("initiator")
    if isinstance(initiator, str) and initiator.strip().lower() == "agent":
        return 0.0
    return _coerce_float(event.get("premium_requests"))


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------

def extract_item_text(item) -> str:
    if not isinstance(item, dict):
        return ""

    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for entry in content:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("text"), str):
                parts.append(entry["text"])
            elif isinstance(entry.get("input_text"), str):
                parts.append(entry["input_text"])
        return "".join(parts)

    if isinstance(item.get("text"), str):
        return item["text"]
    if isinstance(item.get("input_text"), str):
        return item["input_text"]
    return ""


def _extract_text_content(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Request body parsing
# ---------------------------------------------------------------------------

async def parse_json_request(request: Request, error_callback=None) -> dict:
    raw_body = await request.body()
    try:
        if not raw_body:
            return {}
        content_encoding = str(request.headers.get("content-encoding", "")).strip().lower()

        if content_encoding == "gzip":
            raw_body = gzip.decompress(raw_body)
        elif content_encoding == "deflate":
            raw_body = zlib.decompress(raw_body)
        elif content_encoding == "zstd":
            raw_body = zstd_decompress(raw_body)
        elif content_encoding == "br":
            if brotli is None:
                raise HTTPException(status_code=400, detail="Invalid JSON body: unsupported brotli request encoding")
            raw_body = brotli.decompress(raw_body)
        elif raw_body.startswith(b"\x1f\x8b"):
            raw_body = gzip.decompress(raw_body)
        elif raw_body.startswith(b"\x28\xb5\x2f\xfd"):
            raw_body = zstd_decompress(raw_body)

        return json.loads(raw_body)
    except HTTPException:
        raise
    except Exception:
        path = getattr(getattr(request, "url", None), "path", "?")
        content_type = str(request.headers.get("content-type", "")).strip()
        content_encoding = str(request.headers.get("content-encoding", "")).strip().lower()
        preview_hex = raw_body[:24].hex()
        preview_text = raw_body[:160].decode("utf-8", errors="replace")
        if error_callback is not None:
            error_callback(
                {
                    "at": utc_now_iso(),
                    "path": path,
                    "content_type": content_type,
                    "content_encoding": content_encoding,
                    "body_len": len(raw_body),
                    "preview_hex": preview_hex,
                    "preview_text": preview_text,
                }
            )
        print(
            f"WARN: Invalid JSON body path={path} content_type={content_type!r} "
            f"content_encoding={content_encoding!r} body_len={len(raw_body)} preview_hex={preview_hex}",
            flush=True,
        )
        raise HTTPException(status_code=400, detail="Invalid JSON body")
