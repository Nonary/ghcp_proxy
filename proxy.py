"""
Lightweight GitHub Copilot reverse proxy — Responses API path.
Designed for Codex / codex-mini / gpt-5.1-codex and any model that
requires the Responses API instead of Chat Completions.

Usage:
  python ghcp_proxy.py
  → If no token exists, prompts you to authorize via GitHub device flow
  → Then starts serving on http://localhost:8000

Configure Codex:
  export OPENAI_BASE_URL=http://localhost:8000/v1
  export OPENAI_API_KEY=anything
"""

import asyncio
import auth
import dashboard as dashboard_module
import format_translation
import json
import os
import sqlite3
import sys
import tempfile
import time
import premium_plan_config
import safeguard_config as safeguard_config_module
import usage_tracking
import util
from dataclasses import dataclass
from threading import Lock, Thread
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from anthropic_stream import AnthropicStreamTranslator
from bridge_streams import ChatToResponsesStreamTranslator, ResponsesToAnthropicStreamTranslator
from initiator_policy import InitiatorPolicy
from event_bus import EventBus
from model_routing_config import ModelRoutingConfig, ModelRoutingConfigService
from protocol_bridge import BridgeExecutionPlan, ProtocolBridgePlanner
from proxy_client_config import (
    ProxyClientConfig,
    ProxyClientConfigService,
    normalize_proxy_targets,
)

# ─── Import from new modules ─────────────────────────────────────────────────

from constants import (
    DASHBOARD_FILE,
    DETAILED_REQUEST_HISTORY_LIMIT,
    CODEX_CONFIG_FILE,
    CODEX_PROXY_CONFIG,
    CLAUDE_SETTINGS_FILE,
    CLAUDE_PROXY_SETTINGS,
    CLAUDE_MAX_CONTEXT_TOKENS,
    CLAUDE_MAX_OUTPUT_TOKENS,
    DEFAULT_UPSTREAM_TIMEOUT_SECONDS,
    REQUEST_TRACE_LOG_FILE,
    REQUEST_TRACE_HISTORY_LIMIT,
    REQUEST_TRACE_RETENTION_SLACK,
    REQUEST_TRACE_BODY_MAX_BYTES,
    TOKEN_DIR,
)

from rate_limiting import (
    throttle_upstream_request,
    throttled_client_post,
    throttled_client_send,
)


# ─── App & Global State ──────────────────────────────────────────────────────

app = FastAPI()
_REQUEST_TRACE_LOCK = Lock()
_TRACE_HEADER_ALLOWLIST = {
    "content-type",
    "user-agent",
    "openai-intent",
    "editor-version",
    "editor-plugin-version",
    "copilot-integration-id",
    "x-initiator",
    "copilot-vision-request",
    "anthropic-beta",
    "session_id",
    "x-client-request-id",
    "x-request-id",
    "x-github-request-id",
}
AUTH_FAILURE_MESSAGE = "GHCP auth failed"
INVALID_BRIDGE_REQUEST_MESSAGE = "Invalid request"

safeguard_event_store = dashboard_module.create_safeguard_event_store()


def _record_safeguard_trigger(event: dict):
    safeguard_event_store.record_event(event)
    try:
        dashboard_service.notify_dashboard_stream_listeners()
    except NameError:
        pass

_initiator_policy = InitiatorPolicy(on_safeguard_triggered=_record_safeguard_trigger)
safeguard_config_service = safeguard_config_module.SafeguardConfigService(
    safeguard_config_module.SafeguardConfig()
)


def _apply_safeguard_settings(settings: dict):
    cooldown = settings.get("cooldown_seconds") if isinstance(settings, dict) else None
    if isinstance(cooldown, (int, float)):
        _initiator_policy.request_finish_guard_seconds = float(cooldown)


_apply_safeguard_settings(safeguard_config_service.load_settings())
usage_event_bus = EventBus()


class GracefulStreamingResponse(StreamingResponse):
    """Suppress shutdown/disconnect cancellation noise for long-lived streams."""

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        except asyncio.CancelledError:
            return


def set_initiator_policy(policy: InitiatorPolicy):
    global _initiator_policy
    policy.on_safeguard_triggered = _record_safeguard_trigger
    _initiator_policy = policy
    _apply_safeguard_settings(safeguard_config_service.load_settings())
    usage_tracker.on_request_finished = policy.note_request_finished


usage_tracker = usage_tracking.UsageTracker(
    state=usage_tracking.UsageTrackingState(),
    archive_store=dashboard_module.create_usage_archive_store(),
    event_bus=usage_event_bus,
    on_request_finished=_initiator_policy.note_request_finished,
    on_usage_event_recorded=lambda _event: dashboard_service.notify_dashboard_stream_listeners(),
)

client_proxy_config_service = ProxyClientConfigService(
    ProxyClientConfig(
        codex_config_file=CODEX_CONFIG_FILE,
        codex_proxy_config=CODEX_PROXY_CONFIG,
        claude_settings_file=CLAUDE_SETTINGS_FILE,
        claude_proxy_settings=CLAUDE_PROXY_SETTINGS,
        claude_max_context_tokens=CLAUDE_MAX_CONTEXT_TOKENS,
        claude_max_output_tokens=CLAUDE_MAX_OUTPUT_TOKENS,
    )
)
model_routing_config_service = ModelRoutingConfigService(ModelRoutingConfig())
premium_plan_config_service = premium_plan_config.PremiumPlanConfigService(
    premium_plan_config.PremiumPlanConfig()
)
bridge_planner = ProtocolBridgePlanner(model_routing_config_service)

dashboard_service = dashboard_module.create_dashboard_service(
    dependencies=dashboard_module.DashboardDependencies(
        load_api_key_payload=auth.load_api_key_payload,
        load_premium_plan_config=premium_plan_config_service.load_settings,
        snapshot_all_usage_events=usage_tracker.snapshot_all_usage_events,
        snapshot_usage_events=usage_tracker.snapshot_usage_events,
        load_safeguard_trigger_stats=safeguard_event_store.load_stats,
    ),
    utc_now=util.utc_now,
    utc_now_iso=util.utc_now_iso,
    thread_class=Thread,
)




# ─── Module-level parse_json_request wrapper ──────────────────────────────────
# Wraps the util implementation to inject the error callback for recording
# request parsing errors.

async def parse_json_request(request: Request) -> dict:
    return await util.parse_json_request(request, error_callback=usage_tracker.record_request_error)



def configured_upstream_timeout_seconds() -> int:
    raw = str(os.environ.get("GHCP_UPSTREAM_TIMEOUT_SECONDS", "")).strip()
    if not raw:
        return DEFAULT_UPSTREAM_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        print(
            f"Warning: ignoring invalid GHCP_UPSTREAM_TIMEOUT_SECONDS={raw!r}; using {DEFAULT_UPSTREAM_TIMEOUT_SECONDS}",
            file=sys.stderr,
            flush=True,
        )
        return DEFAULT_UPSTREAM_TIMEOUT_SECONDS
    if value <= 0:
        print(
            f"Warning: GHCP_UPSTREAM_TIMEOUT_SECONDS must be > 0; using {DEFAULT_UPSTREAM_TIMEOUT_SECONDS}",
            file=sys.stderr,
            flush=True,
        )
        return DEFAULT_UPSTREAM_TIMEOUT_SECONDS
    return value


# ─── Initialization ──────────────────────────────────────────────────────────

usage_tracker.load_archived_history()
usage_tracker.load_history()
_initiator_policy.seed_from_usage_events(usage_tracker.snapshot_usage_events())
dashboard_module.initialize()


# ─── Upstream response helpers ────────────────────────────────────────────────

def _extract_upstream_json_payload(upstream: httpx.Response) -> dict | None:
    content_type = upstream.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        return None
    try:
        payload = upstream.json()
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_upstream_text(upstream: httpx.Response) -> str | None:
    try:
        text = upstream.text
    except Exception:
        return None
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    return text[:4096]


@dataclass
class UpstreamRequestPlan:
    request_id: str
    upstream_url: str
    headers: dict
    body: dict
    usage_event: dict | None
    requested_model: str | None
    resolved_model: str | None
    source_body: dict | None = None
    trace_context: dict | None = None


def _env_flag(name: str) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_flag_default(name: str, *, default: bool) -> bool:
    """``_env_flag`` variant that defaults to True unless explicitly disabled.

    Accepts 0/false/no/off (case-insensitive) as opt-out when ``default`` is
    True. Any other value — including unset — keeps the default.
    """
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return default


def request_tracing_enabled() -> bool:
    # Default-on: request tracing is always a rolling window bounded by
    # REQUEST_TRACE_HISTORY_LIMIT, so it's cheap to leave on. Users can still
    # opt out with GHCP_TRACE_REQUESTS=0 (accepts 0/false/no/off).
    return _env_flag_default("GHCP_TRACE_REQUESTS", default=True)


def request_trace_log_path() -> str:
    configured = str(os.environ.get("GHCP_TRACE_LOG_FILE", "")).strip()
    return os.path.expanduser(configured or REQUEST_TRACE_LOG_FILE)


def _header_trace_subset(headers: dict | None) -> dict:
    if not isinstance(headers, dict):
        return {}
    subset = {}
    for key, value in headers.items():
        normalized_key = str(key).strip()
        if not normalized_key or normalized_key.lower() not in _TRACE_HEADER_ALLOWLIST:
            continue
        subset[normalized_key] = value
    return subset


def _sorted_counts(values: dict[str, int]) -> dict[str, int]:
    return {key: values[key] for key in sorted(values)}


def _count_trace_items(items) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(items, list):
        return counts
    for item in items:
        if isinstance(item, dict):
            item_type = str(item.get("type", "dict")).strip() or "dict"
        else:
            item_type = type(item).__name__
        counts[item_type] = counts.get(item_type, 0) + 1
    return _sorted_counts(counts)


def _count_trace_roles(items) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(items, list):
        return counts
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if not role:
            continue
        counts[role] = counts.get(role, 0) + 1
    return _sorted_counts(counts)


def _trace_messages_summary(messages) -> dict:
    if isinstance(messages, str):
        return {"kind": "string", "chars": len(messages)}
    if not isinstance(messages, list):
        return {"kind": type(messages).__name__}

    part_counts: dict[str, int] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    part_type = str(part.get("type", "dict")).strip() or "dict"
                else:
                    part_type = type(part).__name__
                part_counts[part_type] = part_counts.get(part_type, 0) + 1
        elif isinstance(content, str) and content:
            part_counts["text"] = part_counts.get("text", 0) + 1

    return {
        "kind": "list",
        "count": len(messages),
        "roles": _count_trace_roles(messages),
        "content_part_types": _sorted_counts(part_counts),
    }


def _trace_input_summary(input_value) -> dict:
    if isinstance(input_value, str):
        return {"kind": "string", "chars": len(input_value)}
    if not isinstance(input_value, list):
        return {"kind": type(input_value).__name__}

    encrypted_reasoning_items = 0
    for item in input_value:
        if isinstance(item, dict) and item.get("type") == "reasoning" and isinstance(item.get("encrypted_content"), str):
            encrypted_reasoning_items += 1

    return {
        "kind": "list",
        "count": len(input_value),
        "item_types": _count_trace_items(input_value),
        "roles": _count_trace_roles(input_value),
        "has_compaction": format_translation.input_contains_compaction(input_value),
        "encrypted_reasoning_items": encrypted_reasoning_items,
    }


def _trace_body_summary(body: dict | None) -> dict | None:
    if not isinstance(body, dict):
        return None

    summary = {
        "keys": sorted(body.keys()),
        "model": body.get("model"),
        "stream": body.get("stream"),
    }

    for source_key, target_key in (
        ("session_id", "session_id"),
        ("sessionId", "session_id"),
        ("prompt_cache_key", "prompt_cache_key"),
        ("promptCacheKey", "prompt_cache_key"),
        ("previous_response_id", "previous_response_id"),
    ):
        value = body.get(source_key)
        if isinstance(value, str) and value.strip():
            summary[target_key] = value.strip()

    tools = body.get("tools")
    if isinstance(tools, list):
        summary["tool_count"] = len(tools)

    if "input" in body:
        summary["input"] = _trace_input_summary(body.get("input"))
    if "messages" in body:
        summary["messages"] = _trace_messages_summary(body.get("messages"))

    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        summary["metadata_keys"] = sorted(metadata.keys())

    return summary


def _trace_response_summary(
    upstream: httpx.Response | None = None,
    response_payload: dict | None = None,
    usage: dict | None = None,
) -> dict:
    summary: dict = {}
    if upstream is not None:
        summary["status_code"] = upstream.status_code
        content_type = upstream.headers.get("content-type")
        if content_type:
            summary["content_type"] = content_type
        for header_name in ("x-request-id", "request-id", "x-github-request-id"):
            header_value = upstream.headers.get(header_name)
            if header_value:
                summary["upstream_request_id"] = header_value
                break

    if isinstance(response_payload, dict):
        for key in ("id", "object", "model"):
            value = response_payload.get(key)
            if isinstance(value, str) and value:
                summary[key] = value
        output = response_payload.get("output")
        if isinstance(output, list):
            summary["output_item_types"] = _count_trace_items(output)

    normalized_usage = util.normalize_usage_payload(usage)
    if normalized_usage is None and isinstance(response_payload, dict):
        normalized_usage = util.normalize_usage_payload(response_payload.get("usage"))
    if isinstance(normalized_usage, dict):
        summary["usage"] = normalized_usage

    error_payload = None
    if isinstance(response_payload, dict):
        maybe_error = response_payload.get("error")
        if isinstance(maybe_error, dict):
            error_payload = maybe_error
    if isinstance(error_payload, dict):
        error_summary = {}
        for key in ("type", "code", "param"):
            value = error_payload.get(key)
            if value is not None:
                error_summary[key] = value
        if error_summary:
            summary["error"] = error_summary

    return summary


def _append_request_trace(payload: dict, *, force: bool = False) -> None:
    if not force and not request_tracing_enabled():
        return
    trace_path = request_trace_log_path()
    try:
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        with _REQUEST_TRACE_LOCK:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, separators=(",", ":"), default=util._json_default))
                f.write("\n")
            _enforce_trace_retention_locked(trace_path)
    except OSError as exc:
        print(f"Warning: failed to write request trace log: {exc}", file=sys.stderr, flush=True)


def _enforce_trace_retention_locked(trace_path: str) -> None:
    """Keep the trace log bounded at ``REQUEST_TRACE_HISTORY_LIMIT`` rows.

    Only rewrites the file once it has drifted ``REQUEST_TRACE_RETENTION_SLACK``
    entries past the limit, so the rewrite cost is amortized across many
    appends. Caller must hold ``_REQUEST_TRACE_LOCK``.
    """
    limit = REQUEST_TRACE_HISTORY_LIMIT
    if limit <= 0:
        return
    threshold = limit + max(REQUEST_TRACE_RETENTION_SLACK, 0)
    try:
        # Fast pre-check: only count lines when the file is large enough that
        # it *might* be over-limit. Short bodies are ~200-500B; long ones are
        # multi-KB. Assume worst-case 256B/line to decide whether to scan.
        size = os.path.getsize(trace_path)
    except OSError:
        return
    if size < threshold * 256:
        return

    try:
        with open(trace_path, "rb") as f:
            line_count = sum(1 for _ in f)
    except OSError as exc:
        print(f"Warning: trace retention scan failed: {exc}", file=sys.stderr, flush=True)
        return
    if line_count <= threshold:
        return

    keep = limit
    drop = line_count - keep
    log_dir = os.path.dirname(trace_path) or TOKEN_DIR
    try:
        fd, temp_path = tempfile.mkstemp(prefix="request-trace-", suffix=".jsonl", dir=log_dir)
        try:
            with os.fdopen(fd, "wb") as out_f, open(trace_path, "rb") as in_f:
                for idx, line in enumerate(in_f):
                    if idx < drop:
                        continue
                    out_f.write(line)
            os.replace(temp_path, trace_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    except OSError as exc:
        print(f"Warning: trace retention rewrite failed: {exc}", file=sys.stderr, flush=True)


def _trim_trace_field(value, *, max_bytes: int = REQUEST_TRACE_BODY_MAX_BYTES):
    """Cap a body-ish trace field so retained rows stay bounded in size.

    If the serialized form is under ``max_bytes`` the value is returned
    unchanged. Otherwise we return a wrapper dict that preserves type info
    plus a truncated prefix, so the row remains self-describing.
    """
    if value is None or max_bytes <= 0:
        return value
    try:
        serialized = json.dumps(value, separators=(",", ":"), default=util._json_default)
    except (TypeError, ValueError):
        return value
    encoded = serialized.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return value
    truncated = encoded[:max_bytes].decode("utf-8", errors="replace")
    return {
        "_truncated": True,
        "original_bytes": len(encoded),
        "preview": truncated,
        "original_type": type(value).__name__,
    }


def _trim_trace_text(value, *, max_chars: int = REQUEST_TRACE_BODY_MAX_BYTES):
    if not isinstance(value, str) or max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n…[truncated; original {len(value)} chars]"


def _emit_request_trace_start(
    *,
    request_id: str,
    request: Request,
    upstream_url: str,
    upstream_path: str | None,
    requested_model: str | None,
    resolved_model: str | None,
    request_body: dict | None,
    upstream_body: dict | None,
    outbound_headers: dict | None,
    trace_metadata: dict | None = None,
) -> dict:
    parsed_upstream = urlsplit(upstream_url)
    context = {
        "request_id": request_id,
        "client_path": request.url.path,
        "upstream_host": parsed_upstream.netloc,
        "upstream_path": upstream_path or parsed_upstream.path,
    }
    # Snapshot the initiator verdict at emit time — the caller may still be
    # mutating the shared sink (it's populated during header_builder and again
    # on subsequent requests using the same dict reference in tests).
    initiator_verdict = None
    if isinstance(trace_metadata, dict):
        raw_verdict = trace_metadata.get("initiator_verdict")
        if isinstance(raw_verdict, dict):
            initiator_verdict = dict(raw_verdict)
    payload = {
        "event": "request_started",
        "time": util.utc_now_iso(),
        **context,
        "method": request.method,
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "request_body": _trace_body_summary(request_body),
        "upstream_body": _trace_body_summary(upstream_body),
        "outbound_headers": _header_trace_subset(outbound_headers),
        "trace": trace_metadata or {},
    }
    if initiator_verdict is not None:
        payload["initiator_verdict"] = initiator_verdict
        context["initiator_verdict"] = initiator_verdict
    _append_request_trace(payload)
    return context


def _should_force_failure_trace(plan: UpstreamRequestPlan | None, status_code: int) -> bool:
    if not isinstance(plan, UpstreamRequestPlan) or status_code < 400:
        return False
    trace = plan.trace_context if isinstance(plan.trace_context, dict) else {}
    return trace.get("bridge") is True


def _finish_usage_and_trace(
    plan: UpstreamRequestPlan | None,
    status_code: int,
    *,
    upstream: httpx.Response | None = None,
    response_payload: dict | None = None,
    response_text: str | None = None,
    usage: dict | None = None,
) -> None:
    if isinstance(plan, UpstreamRequestPlan):
        usage_tracker.finish_event(
            plan.usage_event,
            status_code,
            upstream=upstream,
            response_payload=response_payload,
            response_text=response_text,
            usage=usage,
        )
        force_trace = _should_force_failure_trace(plan, status_code)
        if request_tracing_enabled() or force_trace:
            trace_payload = {
                "event": "request_finished",
                "time": util.utc_now_iso(),
                **(plan.trace_context or {"request_id": plan.request_id}),
                "requested_model": plan.requested_model,
                "resolved_model": plan.resolved_model,
                "response": _trace_response_summary(upstream=upstream, response_payload=response_payload, usage=usage),
                "response_text_present": isinstance(response_text, str) and bool(response_text),
            }
            if status_code >= 400:
                trace_payload["source_body"] = _trim_trace_field(
                    plan.source_body if isinstance(plan.source_body, dict) else plan.body
                )
                trace_payload["upstream_body"] = _trim_trace_field(plan.body)
                trace_payload["outbound_headers"] = _header_trace_subset(plan.headers)
                if isinstance(response_payload, dict):
                    trace_payload["response_payload"] = _trim_trace_field(response_payload)
                if isinstance(response_text, str) and response_text:
                    trace_payload["response_text"] = _trim_trace_text(response_text)
            _append_request_trace(trace_payload, force=force_trace)
        return

    usage_tracker.finish_event(
        None,
        status_code,
        upstream=upstream,
        response_payload=response_payload,
        response_text=response_text,
        usage=usage,
    )


def _prepare_upstream_request(
    request: Request,
    *,
    body: dict,
    requested_model: str | None,
    resolved_model: str | None,
    upstream_path: str,
    upstream_url: str,
    header_builder,
    error_response,
    api_key: str | None = None,
    source_body: dict | None = None,
    trace_metadata: dict | None = None,
) -> tuple[UpstreamRequestPlan | None, Response | None]:
    request_id = uuid4().hex

    effective_api_key = api_key
    if effective_api_key is None:
        try:
            effective_api_key = auth.get_api_key()
        except Exception:
            return None, error_response(401, AUTH_FAILURE_MESSAGE)

    headers = header_builder(effective_api_key, request_id)
    initiator_verdict = None
    if isinstance(trace_metadata, dict):
        initiator_verdict = trace_metadata.get("initiator_verdict")
    usage_event = usage_tracker.start_event(
        request,
        requested_model,
        resolved_model,
        headers.get("X-Initiator"),
        request_id=request_id,
        request_body=body,
        upstream_path=upstream_path,
        outbound_headers=headers,
    )
    trace_context = {
        "request_id": request_id,
        "client_path": request.url.path,
        "upstream_path": upstream_path,
        **(trace_metadata or {}),
    }
    if initiator_verdict is not None:
        trace_context["initiator_verdict"] = initiator_verdict
    if request_tracing_enabled():
        trace_context = _emit_request_trace_start(
            request_id=request_id,
            request=request,
            upstream_url=upstream_url,
            upstream_path=upstream_path,
            requested_model=requested_model,
            resolved_model=resolved_model,
            request_body=source_body if isinstance(source_body, dict) else body,
            upstream_body=body,
            outbound_headers=headers,
            trace_metadata=trace_metadata,
        )
    return (
        UpstreamRequestPlan(
            request_id=request_id,
            upstream_url=upstream_url,
            headers=headers,
            body=body,
            usage_event=usage_event,
            requested_model=requested_model,
            resolved_model=resolved_model,
            source_body=source_body if isinstance(source_body, dict) else body,
            trace_context=trace_context,
        ),
        None,
    )


def _bridge_error_response(plan: BridgeExecutionPlan):
    if plan.caller_protocol == "anthropic":
        return format_translation.anthropic_error_response
    return format_translation.openai_error_response


def _build_bridge_headers(
    request: Request,
    original_body: dict,
    bridge_plan: BridgeExecutionPlan,
    api_key: str,
    request_id: str | None = None,
    *,
    force_initiator: str | None = None,
    verdict_sink: dict | None = None,
) -> dict:
    if bridge_plan.header_kind == "responses":
        return format_translation.build_responses_headers_for_request(
            request,
            bridge_plan.upstream_body,
            api_key,
            force_initiator=force_initiator,
            request_id=request_id,
            initiator_policy=_initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            verdict_sink=verdict_sink,
        )
    if bridge_plan.header_kind == "chat":
        return format_translation.build_chat_headers_for_request(
            request,
            bridge_plan.upstream_body.get("messages", []),
            bridge_plan.upstream_body.get("model"),
            api_key,
            request_id=request_id,
            initiator_policy=_initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            verdict_sink=verdict_sink,
        )
    if bridge_plan.header_kind == "anthropic":
        return format_translation.build_anthropic_headers_for_request(
            request,
            original_body,
            api_key,
            request_id=request_id,
            initiator_policy=_initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            verdict_sink=verdict_sink,
        )
    raise ValueError(f"Unsupported bridge header kind: {bridge_plan.header_kind}")


def _prepare_bridge_request(
    request: Request,
    *,
    original_body: dict,
    bridge_plan: BridgeExecutionPlan,
    api_base: str,
    api_key: str | None = None,
    force_initiator: str | None = None,
) -> tuple[UpstreamRequestPlan | None, Response | None]:
    upstream_url = f"{api_base.rstrip('/')}{bridge_plan.upstream_path}"
    verdict_sink: dict = {}
    return _prepare_upstream_request(
        request,
        body=bridge_plan.upstream_body,
        requested_model=bridge_plan.requested_model,
        resolved_model=bridge_plan.resolved_model,
        upstream_path=bridge_plan.upstream_path,
        upstream_url=upstream_url,
        header_builder=lambda resolved_api_key, request_id: _build_bridge_headers(
            request,
            original_body,
            bridge_plan,
            resolved_api_key,
            request_id=request_id,
            force_initiator=force_initiator,
            verdict_sink=verdict_sink,
        ),
        error_response=_bridge_error_response(bridge_plan),
        api_key=api_key,
        source_body=original_body,
        trace_metadata={
            "bridge": True,
            "strategy_name": bridge_plan.strategy_name,
            "caller_protocol": bridge_plan.caller_protocol,
            "upstream_protocol": bridge_plan.upstream_protocol,
            "header_kind": bridge_plan.header_kind,
            "initiator_verdict": verdict_sink,
        },
    )


def _translate_bridge_success_payload(bridge_plan: BridgeExecutionPlan, payload: dict) -> dict:
    if bridge_plan.caller_protocol == "responses" and bridge_plan.upstream_protocol == "chat":
        return format_translation.chat_completion_to_response(payload, fallback_model=bridge_plan.resolved_model)
    if bridge_plan.caller_protocol == "anthropic" and bridge_plan.upstream_protocol == "responses":
        return format_translation.response_payload_to_anthropic(payload, fallback_model=bridge_plan.resolved_model)
    if bridge_plan.caller_protocol == "anthropic" and bridge_plan.upstream_protocol == "chat":
        return format_translation.chat_completion_to_anthropic(payload, fallback_model=bridge_plan.resolved_model)
    return payload


def _bridge_error_response_from_upstream(bridge_plan: BridgeExecutionPlan, upstream: httpx.Response) -> Response:
    if bridge_plan.caller_protocol == "anthropic":
        return format_translation.anthropic_error_response_from_upstream(upstream)
    return proxy_non_streaming_response(upstream)


async def _post_non_streaming_request(plan: UpstreamRequestPlan, *, error_response) -> Response:
    try:
        async with httpx.AsyncClient(timeout=configured_upstream_timeout_seconds()) as client:
            upstream = await throttled_client_post(
                client,
                plan.upstream_url,
                headers=plan.headers,
                json=plan.body,
            )
        _finish_usage_and_trace(
            plan,
            upstream.status_code,
            upstream=upstream,
            response_payload=_extract_upstream_json_payload(upstream),
            response_text=_extract_upstream_text(upstream),
        )
        return proxy_non_streaming_response(upstream)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(plan, status_code, response_text=message)
        return error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(plan, 599)
        raise


async def _post_bridge_non_streaming_request(plan: UpstreamRequestPlan, bridge_plan: BridgeExecutionPlan) -> Response:
    error_response = _bridge_error_response(bridge_plan)
    try:
        async with httpx.AsyncClient(timeout=configured_upstream_timeout_seconds()) as client:
            upstream = await throttled_client_post(
                client,
                plan.upstream_url,
                headers=plan.headers,
                json=plan.body,
            )
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(plan, status_code, response_text=message)
        return error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(plan, 599)
        raise

    if upstream.status_code >= 400:
        response_payload = _extract_upstream_json_payload(upstream)
        response_text = _extract_upstream_text(upstream)
        if bridge_plan.caller_protocol == "anthropic":
            fallback_message = response_text or f"Upstream request failed with status {upstream.status_code}"
            response_payload = format_translation.anthropic_error_payload_from_openai(
                response_payload,
                upstream.status_code,
                fallback_message,
            )
            response_text = response_payload.get("error", {}).get("message")
        _finish_usage_and_trace(
            plan,
            upstream.status_code,
            upstream=upstream,
            response_payload=response_payload,
            response_text=response_text,
        )
        return _bridge_error_response_from_upstream(bridge_plan, upstream)

    upstream_payload = _extract_upstream_json_payload(upstream)
    if not isinstance(upstream_payload, dict):
        message = "Upstream response did not include a JSON object payload"
        _finish_usage_and_trace(plan, 502, upstream=upstream, response_text=message)
        return error_response(502, message)

    translated_payload = _translate_bridge_success_payload(bridge_plan, upstream_payload)
    _finish_usage_and_trace(
        plan,
        upstream.status_code,
        upstream=upstream,
        response_payload=translated_payload,
        response_text=(
            format_translation.extract_response_output_text(translated_payload)
            if bridge_plan.caller_protocol == "responses"
            else util.extract_item_text(translated_payload.get("content", [{}])[0])
            if isinstance(translated_payload.get("content"), list)
            else None
        ),
    )
    return JSONResponse(content=translated_payload, status_code=upstream.status_code)


def proxy_non_streaming_response(upstream: httpx.Response) -> Response:
    """
    Preserve the upstream status code and body shape.

    Most endpoints return JSON, but compaction can return non-JSON payloads
    such as SSE-style frames. When JSON parsing fails, fall back to relaying
    the raw body with the upstream content type instead of crashing.
    """
    headers = {}
    for name in ("content-type", "cache-control", "retry-after"):
        value = upstream.headers.get(name)
        if value:
            headers[name] = value

    content_type = upstream.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            return JSONResponse(
                content=upstream.json(),
                status_code=upstream.status_code,
                headers=headers,
            )
        except json.JSONDecodeError:
            pass

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers,
    )


async def proxy_streaming_response(
    upstream_url: str,
    headers: dict,
    body: dict,
    timeout: int = 300,
    usage_event: dict | None = None,
    stream_type: str = "responses",
    trace_plan: UpstreamRequestPlan | None = None,
) -> Response:
    """
    Relay an upstream SSE response while preserving upstream error statuses.

    If the upstream request fails before the stream starts, return the upstream
    error body as a normal HTTP response instead of masking it as 200 SSE.
    """
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(trace_plan, status_code, response_text=message)
        await client.aclose()
        return format_translation.openai_error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(trace_plan, 599)
        await client.aclose()
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=_extract_upstream_json_payload(upstream),
                response_text=_extract_upstream_text(upstream),
            )
            return proxy_non_streaming_response(upstream)
        finally:
            await upstream.aclose()
            await client.aclose()

    response_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    content_type = upstream.headers.get("content-type")
    if content_type:
        response_headers["content-type"] = content_type

    async def stream_upstream():
        capture = usage_tracker.create_sse_capture(stream_type)
        try:
            async for chunk in upstream.aiter_bytes():
                if capture.feed(chunk):
                    usage_tracker.mark_first_output(usage_event)
                yield chunk
        finally:
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                usage=capture.usage if isinstance(capture.usage, dict) else None,
            )
            await upstream.aclose()
            await client.aclose()

    return GracefulStreamingResponse(
        stream_upstream(),
        status_code=upstream.status_code,
        headers=response_headers,
    )


async def proxy_anthropic_streaming_response(
    upstream_url: str,
    headers: dict,
    body: dict,
    fallback_model: str,
    timeout: int = 300,
    usage_event: dict | None = None,
    trace_plan: UpstreamRequestPlan | None = None,
) -> Response:
    """
    Translate upstream chat-completions SSE into Anthropic Messages SSE.
    """
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(trace_plan, status_code, response_text=message)
        await client.aclose()
        return format_translation.anthropic_error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(trace_plan, 599)
        await client.aclose()
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            fallback_message = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
            error_payload = format_translation.anthropic_error_payload_from_openai(
                _extract_upstream_json_payload(upstream),
                upstream.status_code,
                fallback_message,
            )
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=error_payload,
                response_text=error_payload.get("error", {}).get("message"),
            )
            return format_translation.anthropic_error_response_from_upstream(upstream)
        finally:
            await upstream.aclose()
            await client.aclose()

    response_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "content-type": "text/event-stream; charset=utf-8",
    }

    async def stream_translated():
        translator = AnthropicStreamTranslator(
            fallback_model,
            mark_first_output=lambda: usage_tracker.mark_first_output(usage_event),
        )
        try:
            async for event in translator.translate(upstream.aiter_bytes()):
                yield event
        finally:
            response_payload = translator.build_response_payload()
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text=translator.response_text,
                usage=response_payload["usage"],
            )
            await upstream.aclose()
            await client.aclose()

    return GracefulStreamingResponse(
        stream_translated(),
        status_code=upstream.status_code,
        headers=response_headers,
    )


async def proxy_responses_from_chat_streaming_response(
    upstream_url: str,
    headers: dict,
    body: dict,
    fallback_model: str,
    timeout: int = 300,
    usage_event: dict | None = None,
    trace_plan: UpstreamRequestPlan | None = None,
) -> Response:
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(trace_plan, status_code, response_text=message)
        await client.aclose()
        return format_translation.openai_error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(trace_plan, 599)
        await client.aclose()
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=_extract_upstream_json_payload(upstream),
                response_text=_extract_upstream_text(upstream),
            )
            return proxy_non_streaming_response(upstream)
        finally:
            await upstream.aclose()
            await client.aclose()

    response_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "content-type": "text/event-stream; charset=utf-8",
    }

    async def stream_translated():
        translator = ChatToResponsesStreamTranslator(
            fallback_model,
            mark_first_output=lambda: usage_tracker.mark_first_output(usage_event),
        )
        try:
            async for event in translator.translate(upstream.aiter_bytes()):
                yield event
        finally:
            response_payload = translator.build_response_payload()
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text=translator.response_text,
                usage=response_payload["usage"],
            )
            await upstream.aclose()
            await client.aclose()

    return GracefulStreamingResponse(
        stream_translated(),
        status_code=upstream.status_code,
        headers=response_headers,
    )


async def proxy_anthropic_from_responses_streaming_response(
    upstream_url: str,
    headers: dict,
    body: dict,
    fallback_model: str,
    timeout: int = 300,
    usage_event: dict | None = None,
    trace_plan: UpstreamRequestPlan | None = None,
) -> Response:
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(trace_plan, status_code, response_text=message)
        await client.aclose()
        return format_translation.anthropic_error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(trace_plan, 599)
        await client.aclose()
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            fallback_message = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
            error_payload = format_translation.anthropic_error_payload_from_openai(
                _extract_upstream_json_payload(upstream),
                upstream.status_code,
                fallback_message,
            )
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=error_payload,
                response_text=error_payload.get("error", {}).get("message"),
            )
            return format_translation.anthropic_error_response_from_upstream(upstream)
        finally:
            await upstream.aclose()
            await client.aclose()

    response_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "content-type": "text/event-stream; charset=utf-8",
    }

    async def stream_translated():
        translator = ResponsesToAnthropicStreamTranslator(
            fallback_model,
            mark_first_output=lambda: usage_tracker.mark_first_output(usage_event),
        )
        try:
            async for event in translator.translate(upstream.aiter_bytes()):
                yield event
        finally:
            response_payload = translator.build_response_payload()
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text=translator.response_text,
                usage=response_payload["usage"],
            )
            await upstream.aclose()
            await client.aclose()

    return GracefulStreamingResponse(
        stream_translated(),
        status_code=upstream.status_code,
        headers=response_headers,
    )


async def _proxy_bridge_streaming_response(plan: UpstreamRequestPlan, bridge_plan: BridgeExecutionPlan) -> Response:
    if bridge_plan.caller_protocol == "responses" and bridge_plan.upstream_protocol == "responses":
        return await proxy_streaming_response(
            plan.upstream_url,
            plan.headers,
            plan.body,
            timeout=300,
            usage_event=plan.usage_event,
            stream_type="responses",
            trace_plan=plan,
        )
    if bridge_plan.caller_protocol == "responses" and bridge_plan.upstream_protocol == "chat":
        return await proxy_responses_from_chat_streaming_response(
            plan.upstream_url,
            plan.headers,
            plan.body,
            bridge_plan.resolved_model,
            timeout=300,
            usage_event=plan.usage_event,
            trace_plan=plan,
        )
    if bridge_plan.caller_protocol == "anthropic" and bridge_plan.upstream_protocol == "chat":
        return await proxy_anthropic_streaming_response(
            plan.upstream_url,
            plan.headers,
            plan.body,
            bridge_plan.resolved_model,
            timeout=300,
            usage_event=plan.usage_event,
            trace_plan=plan,
        )
    return await proxy_anthropic_from_responses_streaming_response(
        plan.upstream_url,
        plan.headers,
        plan.body,
        bridge_plan.resolved_model,
        timeout=300,
        usage_event=plan.usage_event,
        trace_plan=plan,
    )


# ─── Dashboard routes ─────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def dashboard_root():
    return RedirectResponse(url="/ui", status_code=307)


@app.get("/ui", response_class=HTMLResponse)
async def dashboard():
    return FileResponse(DASHBOARD_FILE, media_type="text/html")


@app.get("/api/dashboard")
async def dashboard_api(request: Request):
    refresh = request.query_params.get("refresh", "").lower() in {"1", "true", "yes"}
    payload = await asyncio.to_thread(dashboard_service.build_payload, refresh)
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


@app.get("/api/dashboard/stream")
async def dashboard_stream(request: Request):
    heartbeat_seconds = 20
    poll_seconds = 1.0
    queue = dashboard_service.register_stream_listener()
    last_version = dashboard_service.current_stream_version()

    async def stream():
        nonlocal last_version
        last_heartbeat = time.monotonic()
        try:
            initial_payload = await asyncio.to_thread(dashboard_service.build_payload, False)
            yield format_translation.sse_encode("dashboard", initial_payload)
            while True:
                if await request.is_disconnected():
                    break

                try:
                    version = await asyncio.wait_for(queue.get(), timeout=poll_seconds)
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if now - last_heartbeat >= heartbeat_seconds:
                        last_heartbeat = now
                        yield format_translation.sse_encode("heartbeat", {"at": util.utc_now_iso()})
                    continue

                if version == last_version:
                    continue
                last_version = version
                payload = await asyncio.to_thread(dashboard_service.build_payload, False)
                yield format_translation.sse_encode("dashboard", payload)
        finally:
            dashboard_service.unregister_stream_listener(queue)

    return GracefulStreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Config API routes ────────────────────────────────────────────────────────

@app.get("/api/config/premium-plan")
async def premium_plan_status_api():
    return JSONResponse(content=premium_plan_config_service.config_payload())


@app.get("/api/config/safeguard")
async def safeguard_status_api():
    return JSONResponse(content=safeguard_config_service.config_payload())


@app.post("/api/config/safeguard")
async def safeguard_config_api(request: Request):
    payload = await parse_json_request(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    if bool(payload.get("reset")):
        result = safeguard_config_service.save_settings(
            {"cooldown_seconds": safeguard_config_service.default_settings()["cooldown_seconds"]}
        )
    else:
        result = safeguard_config_service.save_settings(payload)
    _apply_safeguard_settings(result)
    return JSONResponse(content=result)


@app.post("/api/config/premium-plan")
async def premium_plan_config_api(request: Request):
    payload = await parse_json_request(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    if bool(payload.get("clear")):
        result = premium_plan_config_service.clear_settings()
    else:
        result = premium_plan_config_service.save_settings(payload)
    return JSONResponse(content=result)


@app.get("/api/config/client-proxy")
async def client_proxy_status_api():
    return JSONResponse(content=client_proxy_config_service.proxy_client_status_payload())


@app.get("/api/config/model-remapping")
@app.get("/api/config/model-routing")
async def model_routing_status_api():
    return JSONResponse(content=model_routing_config_service.config_payload())


@app.post("/api/config/model-remapping")
@app.post("/api/config/model-routing")
async def model_routing_config_api(request: Request):
    payload = await parse_json_request(request)
    return JSONResponse(content=model_routing_config_service.save_settings(payload))


@app.post("/api/config/client-proxy")
async def client_proxy_install_api(request: Request):
    payload = await parse_json_request(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")
    targets = normalize_proxy_targets(payload)
    action = payload.get("action", "enable")
    if not isinstance(action, str):
        raise HTTPException(status_code=400, detail='Action must be "enable" or "disable".')

    action = action.strip().lower()
    if action == "install":
        action = "enable"
    if action not in {"enable", "disable"}:
        raise HTTPException(status_code=400, detail='Unsupported action. Use "enable" or "disable".')

    clients = {}

    for target in targets:
        try:
            if action == "disable":
                clients[target] = client_proxy_config_service.disable_target(target)
            else:
                clients[target] = client_proxy_config_service.enable_target(target)
        except Exception as exc:
            clients[target] = client_proxy_config_service.empty_proxy_status(target)
            clients[target]["error"] = str(exc)
            clients[target]["status_message"] = "failed to write config"

    return JSONResponse(
        content={
            "clients": clients,
            "message": (
                "Proxy enabled for: "
                if action == "enable"
                else "Proxy disabled for: "
            )
            + (
                ", ".join(
                    target
                    for target, payload in sorted(clients.items())
                    if not payload.get("error")
                )
                or "none"
            ),
        }
    )


# ─── Route: /v1/responses  (Codex / Responses API) ───────────────────────────

@app.post("/v1/responses")
async def responses(request: Request):
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return format_translation.openai_error_response(
            exc.status_code,
            format_translation.http_exception_detail_to_message(exc.detail),
        )

    # Sanitize input (multi-turn encrypted_content passthrough)
    raw_input = body.get("input")
    has_compaction_input = format_translation.input_contains_compaction(raw_input)
    if raw_input is not None:
        body["input"] = format_translation.sanitize_input(raw_input)

    try:
        api_key = auth.get_api_key()
    except Exception:
        return format_translation.openai_error_response(401, AUTH_FAILURE_MESSAGE)

    api_base = auth.get_api_base()
    try:
        bridge_plan = await bridge_planner.plan(
            "responses",
            body,
            api_base=api_base,
            api_key=api_key,
            subagent=request.headers.get("x-openai-subagent"),
        )
    except ValueError:
        return format_translation.openai_error_response(400, INVALID_BRIDGE_REQUEST_MESSAGE)

    plan, error_response = _prepare_bridge_request(
        request,
        original_body=body,
        bridge_plan=bridge_plan,
        api_base=api_base,
        api_key=api_key,
        force_initiator="agent" if has_compaction_input else None,
    )
    if error_response is not None:
        return error_response

    if bridge_plan.stream:
        return await _proxy_bridge_streaming_response(plan, bridge_plan)
    return await _post_bridge_non_streaming_request(plan, bridge_plan)


@app.post("/v1/responses/compact")
async def responses_compact(request: Request):
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return format_translation.openai_error_response(
            exc.status_code,
            format_translation.http_exception_detail_to_message(exc.detail),
        )

    summary_request = format_translation.build_fake_compaction_request(body)
    try:
        api_key = auth.get_api_key()
    except Exception:
        return format_translation.openai_error_response(401, AUTH_FAILURE_MESSAGE)

    api_base = auth.get_api_base()
    try:
        bridge_plan = await bridge_planner.plan(
            "responses",
            summary_request,
            api_base=api_base,
            api_key=api_key,
            subagent=request.headers.get("x-openai-subagent"),
            is_compact=True,
        )
    except ValueError:
        return format_translation.openai_error_response(400, INVALID_BRIDGE_REQUEST_MESSAGE)

    plan, error_response = _prepare_bridge_request(
        request,
        original_body=summary_request,
        bridge_plan=bridge_plan,
        api_base=api_base,
        api_key=api_key,
        force_initiator="agent",
    )
    if error_response is not None:
        return error_response

    if bridge_plan.stream:
        return await _proxy_bridge_streaming_response(plan, bridge_plan)
    return await _post_bridge_non_streaming_request(plan, bridge_plan)


# ─── Route: /v1/chat/completions  (non-Codex models) ─────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    For models that still use the Chat API.
    Codex does NOT use this endpoint — it uses /v1/responses above.
    """
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return format_translation.openai_error_response(
            exc.status_code,
            format_translation.http_exception_detail_to_message(exc.detail),
        )

    messages = body.get("messages", [])

    upstream_url = f"{auth.get_api_base().rstrip('/')}/chat/completions"
    verdict_sink: dict = {}
    plan, error_response = _prepare_upstream_request(
        request,
        body=body,
        requested_model=body.get("model"),
        resolved_model=body.get("model"),
        upstream_path="/chat/completions",
        upstream_url=upstream_url,
        header_builder=lambda api_key, request_id: format_translation.build_chat_headers_for_request(
            request,
            messages,
            body.get("model"),
            api_key,
            request_id=request_id,
            initiator_policy=_initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            verdict_sink=verdict_sink,
        ),
        error_response=format_translation.openai_error_response,
        trace_metadata={"initiator_verdict": verdict_sink},
    )
    if error_response is not None:
        return error_response

    if body.get("stream", False):
        return await proxy_streaming_response(
            plan.upstream_url,
            plan.headers,
            plan.body,
            timeout=300,
            usage_event=plan.usage_event,
            stream_type="chat",
            trace_plan=plan,
        )
    return await _post_non_streaming_request(plan, error_response=format_translation.openai_error_response)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """
    Anthropic-compatible route.
    Translate Anthropic Messages payloads onto GHCP's OpenAI-compatible chat
    endpoint so Claude Code can reuse Copilot's cache semantics.
    """
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return format_translation.anthropic_error_response(
            exc.status_code,
            format_translation.http_exception_detail_to_message(exc.detail),
        )

    try:
        api_key = auth.get_api_key()
    except Exception:
        return format_translation.anthropic_error_response(401, AUTH_FAILURE_MESSAGE)

    api_base = auth.get_api_base()
    try:
        bridge_plan = await bridge_planner.plan(
            "messages",
            body,
            api_base=api_base,
            api_key=api_key,
            subagent=request.headers.get("x-openai-subagent"),
        )
    except ValueError:
        return format_translation.anthropic_error_response(400, INVALID_BRIDGE_REQUEST_MESSAGE)

    plan, error_response = _prepare_bridge_request(
        request,
        original_body=body,
        bridge_plan=bridge_plan,
        api_base=api_base,
        api_key=api_key,
    )
    if error_response is not None:
        return error_response

    if bridge_plan.stream:
        return await _proxy_bridge_streaming_response(plan, bridge_plan)
    return await _post_bridge_non_streaming_request(plan, bridge_plan)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Run auth interactively in the terminal BEFORE the server starts.
    # If no token exists this will print the device flow prompt and block until
    # the user authorizes (or the script exits on failure).
    auth.ensure_authenticated()

    # Step 2: Start the server in the foreground on this terminal.
    print("Starting GHCP proxy on http://localhost:8000 (loopback only)", flush=True)
    print("  Responses API : POST /v1/responses", flush=True)
    print("  Compaction    : POST /v1/responses/compact", flush=True)
    print("  Chat API      : POST /v1/chat/completions", flush=True)
    print("", flush=True)
    print("  Set in your shell:", flush=True)
    print("    export OPENAI_BASE_URL=http://localhost:8000/v1", flush=True)
    print("    export OPENAI_API_KEY=anything", flush=True)
    print("", flush=True)

    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False, timeout_graceful_shutdown=2)
