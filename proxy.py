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

import os
import sys


def _prepare_standalone_process_file_descriptors():
    """Avoid inheriting a descriptor table that is already near the soft limit."""
    try:
        import resource
    except ImportError:
        return

    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return

    target_limit = 4096
    if hard_limit != resource.RLIM_INFINITY:
        target_limit = min(target_limit, hard_limit)
    if soft_limit < target_limit:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard_limit))
            soft_limit = target_limit
        except (OSError, ValueError):
            pass

    try:
        close_until = int(soft_limit)
    except (OverflowError, ValueError):
        close_until = 4096
    os.closerange(3, max(3, close_until))


if __name__ == "__main__":
    _prepare_standalone_process_file_descriptors()


import asyncio
import auth
import atexit
import auto_update
import background_proxy
import dashboard as dashboard_module
import format_translation
import hashlib
import messages_preprocess
import migrate_runtime_paths
import json
import sqlite3
import tempfile
import time
import threading
import safeguard_config as safeguard_config_module
import update_notice
import usage_reminder
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
from bridge_streams import (
    AnthropicToResponsesStreamTranslator,
    ChatToResponsesStreamTranslator,
    ResponsesToAnthropicStreamTranslator,
)
from initiator_policy import InitiatorPolicy
from event_bus import EventBus
from model_routing_config import ModelRoutingConfig, ModelRoutingConfigService, model_provider_family
from protocol_bridge import BridgeExecutionPlan, ProtocolBridgePlanner
from proxy_client_config import (
    ProxyClientConfig,
    ProxyClientConfigService,
    normalize_proxy_targets,
)

# ─── Import from new modules ─────────────────────────────────────────────────

from constants import (
    CLIENT_PROXY_SETTINGS_FILE,
    DASHBOARD_FILE,
    DETAILED_REQUEST_HISTORY_LIMIT,
    CODEX_PRIMARY_CONFIG_FILE,
    CODEX_MANAGED_CONFIG_FILE,
    CODEX_PROXY_MODEL_CATALOG_FILE,
    CODEX_PROXY_CONFIG,
    CODEX_PROXY_MODEL_CONTEXT_WINDOW,
    CODEX_PROXY_MODEL_AUTO_COMPACT_TOKEN_LIMIT,
    CLAUDE_SETTINGS_FILE,
    CLAUDE_PROXY_SETTINGS,
    CLAUDE_MAX_CONTEXT_TOKENS,
    CLAUDE_MAX_OUTPUT_TOKENS,
    DEFAULT_UPSTREAM_TIMEOUT_SECONDS,
    LEGACY_BILLING_TOKEN_FILE,
    LEGACY_PREMIUM_PLAN_CONFIG_FILE,
    PROXY_PID_FILE,
    REQUEST_TRACE_LOG_FILE,
    REQUEST_TRACE_HISTORY_LIMIT,
    REQUEST_TRACE_RETENTION_SLACK,
    REQUEST_TRACE_BODY_MAX_BYTES,
    REQUEST_PROMPT_PREVIEW_MAX_CHARS,
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
_CLIENT_PROXY_STARTUP_RESTORE_LOCK = Lock()
_CLIENT_PROXY_STARTUP_RESTORE_COMPLETE = False
_CLIENT_PROXY_SHUTDOWN_REVERT_LOCK = Lock()
_CLIENT_PROXY_SHUTDOWN_REVERT_COMPLETE = False
_RESPONSES_REASONING_TRIM_INPUT_ITEM_THRESHOLD = 48
_RESPONSES_REASONING_ENCRYPTED_KEEP_LAST = 6
migrated_runtime_files = migrate_runtime_paths.migrate_legacy_runtime_files()
if migrated_runtime_files:
    print(f"runtime migration: copied {len(migrated_runtime_files)} legacy file(s)", flush=True)
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
    "x-openai-subagent",
    "x-interaction-id",
    "x-interaction-type",
    "x-agent-task-id",
    "x-client-session-id",
    "x-request-id",
    "x-github-request-id",
}
AUTH_FAILURE_MESSAGE = "GitHub Copilot authorization required. Open /ui to sign in."
INVALID_BRIDGE_REQUEST_MESSAGE = "Invalid request"

safeguard_event_store = dashboard_module.create_safeguard_event_store()


def _stream_with_update_notice(byte_iter, protocol: str, upstream_headers=None):
    notices: list[str] = []
    update_text = auto_update_runtime_controller.update_notice_text_if_due()
    if update_text:
        notices.append(update_text)
    usage_windows = usage_tracking.extract_usage_ratelimits_from_headers(upstream_headers)
    usage_text = usage_reminder_controller.usage_notice_text_if_due(usage_windows)
    if usage_text:
        notices.append(usage_text)
    if not notices:
        return byte_iter
    notice_text = "\n\n".join(notices)
    return update_notice.inject_text_notice(byte_iter, protocol, notice_text)


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
usage_reminder_controller = usage_reminder.UsageReminderController(
    usage_tracker.snapshot_all_usage_events,
)

model_routing_config_service = ModelRoutingConfigService(ModelRoutingConfig())
client_proxy_config_service = ProxyClientConfigService(
    ProxyClientConfig(
        codex_primary_config_file=CODEX_PRIMARY_CONFIG_FILE,
        codex_managed_config_file=CODEX_MANAGED_CONFIG_FILE,
        codex_model_catalog_file=CODEX_PROXY_MODEL_CATALOG_FILE,
        codex_proxy_config=CODEX_PROXY_CONFIG,
        codex_model_context_window=CODEX_PROXY_MODEL_CONTEXT_WINDOW,
        codex_model_auto_compact_token_limit=CODEX_PROXY_MODEL_AUTO_COMPACT_TOKEN_LIMIT,
        claude_settings_file=CLAUDE_SETTINGS_FILE,
        claude_proxy_settings=CLAUDE_PROXY_SETTINGS,
        claude_max_context_tokens=CLAUDE_MAX_CONTEXT_TOKENS,
        claude_max_output_tokens=CLAUDE_MAX_OUTPUT_TOKENS,
        client_proxy_settings_file=CLIENT_PROXY_SETTINGS_FILE,
    ),
    model_capabilities_provider=lambda: fetch_copilot_model_capabilities(),
    model_routing_settings_provider=lambda: model_routing_config_service.load_settings(),
)
background_proxy_manager = background_proxy.BackgroundProxyManager()
auto_update_manager = auto_update.AutoUpdateManager()
auto_update_runtime_controller = auto_update.AutoUpdateRuntimeController(auto_update_manager)
bridge_planner = ProtocolBridgePlanner(
    model_routing_config_service,
    capability_resolver=lambda model: model_supports_native_messages(model) if model else False,
)

dashboard_service = dashboard_module.create_dashboard_service(
    dependencies=dashboard_module.DashboardDependencies(
        load_api_key_payload=auth.load_api_key_payload,
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


def _write_proxy_pid_file() -> None:
    try:
        os.makedirs(os.path.dirname(PROXY_PID_FILE), exist_ok=True)
        with open(PROXY_PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
            f.write("\n")
    except OSError as exc:
        print(f"Warning: failed to write proxy pid file: {exc}", file=sys.stderr, flush=True)


def _remove_proxy_pid_file() -> None:
    try:
        with open(PROXY_PID_FILE, encoding="utf-8") as f:
            recorded_pid = f.read().strip()
    except OSError:
        return
    if recorded_pid != str(os.getpid()):
        return
    try:
        os.remove(PROXY_PID_FILE)
    except OSError:
        pass


# ─── Initialization ──────────────────────────────────────────────────────────

usage_tracker.load_archived_history()
usage_tracker.load_history()
_initiator_policy.seed_from_usage_events(usage_tracker.snapshot_usage_events())
dashboard_module.initialize()

# Ingest native Codex CLI traffic (sessions/*/rollout-*.jsonl) so it shows
# up in the dashboard alongside proxied traffic, tagged as `codex_native`.
try:
    import codex_native_ingest
    _codex_native_interval = float(os.environ.get("GHCP_CODEX_NATIVE_INGEST_INTERVAL", "5") or 5)
    if _codex_native_interval > 0:
        codex_native_ingest.start_background_scanner(
            usage_tracker.record_usage_event,
            interval_seconds=_codex_native_interval,
        )
except Exception as _codex_ingest_exc:  # pragma: no cover - best effort
    print(f"codex_native_ingest: disabled ({_codex_ingest_exc})", flush=True)


@app.on_event("startup")
async def _app_startup_restore_client_proxy_configs():
    restore_client_proxy_configs_on_startup()
    auto_update_runtime_controller.start_periodic_checks()


@app.on_event("shutdown")
async def _app_shutdown_revert_client_proxy_configs():
    await auto_update_runtime_controller.stop_periodic_checks()
    revert_client_proxy_configs_on_shutdown()


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
    auto_update_request_tracked: bool = False


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


def restore_client_proxy_configs_on_startup() -> dict[str, object]:
    global _CLIENT_PROXY_STARTUP_RESTORE_COMPLETE
    with _CLIENT_PROXY_STARTUP_RESTORE_LOCK:
        if _CLIENT_PROXY_STARTUP_RESTORE_COMPLETE:
            return {
                "attempted": False,
                "restored": False,
                "reason": "already-ran",
                "clients": {},
            }
        _CLIENT_PROXY_STARTUP_RESTORE_COMPLETE = True

    try:
        result = client_proxy_config_service.restore_proxy_configs_on_startup()
    except Exception as exc:  # pragma: no cover - best effort
        print(f"client proxy startup restore failed: {exc}", flush=True)
        return {
            "attempted": True,
            "restored": False,
            "reason": "error",
            "error": str(exc),
            "clients": {},
        }

    if result.get("attempted"):
        print(f"Client proxy startup restore: {json.dumps(result, default=str)}", flush=True)
    return result


def revert_client_proxy_configs_on_shutdown() -> dict[str, object]:
    global _CLIENT_PROXY_SHUTDOWN_REVERT_COMPLETE
    with _CLIENT_PROXY_SHUTDOWN_REVERT_LOCK:
        if _CLIENT_PROXY_SHUTDOWN_REVERT_COMPLETE:
            return {
                "attempted": False,
                "reverted": False,
                "reason": "already-ran",
                "clients": {},
            }
        _CLIENT_PROXY_SHUTDOWN_REVERT_COMPLETE = True

    try:
        result = client_proxy_config_service.revert_proxy_configs_on_shutdown()
    except Exception as exc:  # pragma: no cover - best effort
        print(f"client proxy shutdown revert failed: {exc}", flush=True)
        return {
            "attempted": True,
            "reverted": False,
            "reason": "error",
            "error": str(exc),
            "clients": {},
        }

    if result.get("attempted"):
        print(f"Client proxy shutdown revert: {json.dumps(result, default=str)}", flush=True)
    return result


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
        "sequence": _trace_input_sequence(input_value),
    }


def _trace_hash(value) -> str | None:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()[:16]


def _trace_text_chars(value) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_trace_text_chars(item) for item in value)
    if isinstance(value, dict):
        total = 0
        for key in ("text", "input_text", "output_text"):
            text = value.get(key)
            if isinstance(text, str):
                total += len(text)
        for key in ("content", "output"):
            nested = value.get(key)
            if isinstance(nested, (list, dict, str)):
                total += _trace_text_chars(nested)
        return total
    return 0


def _trace_input_sequence(input_value: list) -> list[dict]:
    sequence = []
    for index, item in enumerate(input_value):
        if not isinstance(item, dict):
            sequence.append({"index": index, "type": type(item).__name__})
            continue
        entry = {
            "index": index,
            "type": item.get("type"),
        }
        for key in ("role", "name", "status"):
            value = item.get(key)
            if isinstance(value, str) and value:
                entry[key] = value
        for key in ("id", "call_id"):
            value = item.get(key)
            if isinstance(value, str) and value:
                entry[f"{key}_hash"] = _trace_hash(value)
        if "content" in item:
            entry["content_chars"] = _trace_text_chars(item.get("content"))
        if "output" in item:
            entry["output_chars"] = _trace_text_chars(item.get("output"))
        encrypted = item.get("encrypted_content")
        if isinstance(encrypted, str) and encrypted:
            entry["encrypted_content_chars"] = len(encrypted)
            entry["encrypted_content_hash"] = _trace_hash(encrypted)
        sequence.append(entry)
    return sequence


def _trace_tools_deferred_count(tools) -> int:
    if isinstance(tools, list):
        return sum(_trace_tools_deferred_count(tool) for tool in tools)
    if not isinstance(tools, dict):
        return 0
    count = 1 if "defer_loading" in tools else 0
    nested = tools.get("tools")
    if isinstance(nested, (list, dict)):
        count += _trace_tools_deferred_count(nested)
    return count


def _trace_body_summary(body: dict | None) -> dict | None:
    if not isinstance(body, dict):
        return None

    summary = {
        "keys": sorted(body.keys()),
        "model": body.get("model"),
        "stream": body.get("stream"),
    }

    reasoning_effort = body.get("reasoning_effort")
    if isinstance(reasoning_effort, str) and reasoning_effort:
        summary["reasoning_effort"] = reasoning_effort
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort:
            summary["reasoning_effort"] = effort
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        snapshot: dict = {}
        t_type = thinking.get("type")
        if isinstance(t_type, str):
            snapshot["type"] = t_type
        budget = thinking.get("budget_tokens")
        if isinstance(budget, int):
            snapshot["budget_tokens"] = budget
        if snapshot:
            summary["thinking"] = snapshot

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
        deferred_tool_count = _trace_tools_deferred_count(tools)
        if deferred_tool_count:
            summary["deferred_tool_count"] = deferred_tool_count
        if format_translation.responses_tools_have_tool_search(tools):
            summary["tool_search_present"] = True
    elif isinstance(tools, dict):
        deferred_tool_count = _trace_tools_deferred_count(tools)
        if deferred_tool_count:
            summary["deferred_tool_count"] = deferred_tool_count
        if format_translation.responses_tools_have_tool_search(tools):
            summary["tool_search_present"] = True

    if "input" in body:
        summary["input"] = _trace_input_summary(body.get("input"))
    if "messages" in body:
        summary["messages"] = _trace_messages_summary(body.get("messages"))
    body_fingerprint = _trace_hash(body)
    if body_fingerprint:
        summary["body_fingerprint"] = body_fingerprint

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


def _extract_prompt_preview(
    body: dict | None,
    *,
    truncate: bool = True,
    max_chars: int = REQUEST_PROMPT_PREVIEW_MAX_CHARS,
) -> dict | None:
    """Pull a human-readable prompt preview out of a request body.

    Returns a dict with ``system`` (concatenated system/developer prompts),
    ``user`` (most recent user turn text) and ``truncated`` flags. ``None``
    is returned when the body carries no recognizable prompt material.
    Works for both OpenAI ``messages``/``input`` shapes and Anthropic
    ``system`` + ``messages`` bodies.
    """
    if not isinstance(body, dict) or max_chars <= 0:
        return None

    system_parts: list[str] = []
    user_parts: list[str] = []

    raw_system = body.get("system")
    if isinstance(raw_system, str) and raw_system.strip():
        system_parts.append(raw_system)
    elif isinstance(raw_system, list):
        for entry in raw_system:
            text = util.extract_item_text(entry) if isinstance(entry, dict) else ""
            if not text and isinstance(entry, dict) and isinstance(entry.get("text"), str):
                text = entry["text"]
            if isinstance(text, str) and text.strip():
                system_parts.append(text)

    def _collect(items):
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or item.get("type") or "").strip().lower()
            text = util.extract_item_text(item)
            if not isinstance(text, str) or not text.strip():
                continue
            if role in ("system", "developer"):
                system_parts.append(text)
            elif role in ("user", "human", "message", ""):
                user_parts.append(text)

    _collect(body.get("messages"))
    input_value = body.get("input")
    if isinstance(input_value, str) and input_value.strip():
        user_parts.append(input_value)
    else:
        _collect(input_value)

    if not system_parts and not user_parts:
        return None

    def _finalize(parts: list[str]) -> tuple[str, bool]:
        combined = "\n\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
        if not combined:
            return "", False
        # Keep the most recent context for user prompts (tail) and the
        # leading context for system prompts (head) since the head carries
        # the instructions.
        if not truncate or max_chars <= 0 or len(combined) <= max_chars:
            return combined, False
        return combined[:max_chars] + f"\n…[truncated; original {len(combined)} chars]", True

    system_text, system_truncated = _finalize(system_parts)
    # For user prompts, prefer the latest turn when truncating.
    user_combined = "\n\n".join(part.strip() for part in user_parts if isinstance(part, str) and part.strip())
    user_truncated = False
    if truncate and max_chars > 0 and len(user_combined) > max_chars:
        user_combined = "…[truncated; original " + str(len(user_combined)) + " chars]\n" + user_combined[-max_chars:]
        user_truncated = True

    preview: dict = {}
    if system_text:
        preview["system"] = system_text
        if system_truncated:
            preview["system_truncated"] = True
    if user_combined:
        preview["user"] = user_combined
        if user_truncated:
            preview["user_truncated"] = True
    return preview or None


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
    prompt_preview: dict | None = None,
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
    if prompt_preview is None:
        prompt_preview = _extract_prompt_preview(
            request_body if isinstance(request_body, dict) else upstream_body
        )
    if prompt_preview:
        payload["request_prompt"] = prompt_preview
        context["request_prompt"] = prompt_preview
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
    reasoning_text: str | None = None,
    usage: dict | None = None,
) -> None:
    if isinstance(plan, UpstreamRequestPlan):
        try:
            usage_tracker.finish_event(
                plan.usage_event,
                status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text=response_text,
                reasoning_text=reasoning_text,
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
                    "reasoning_text_present": isinstance(reasoning_text, str) and bool(reasoning_text),
                }
                if isinstance(reasoning_text, str) and reasoning_text:
                    trace_payload["reasoning_text"] = _trim_trace_text(reasoning_text)
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
        finally:
            if plan.auto_update_request_tracked:
                auto_update_runtime_controller.note_request_finished(plan.request_id)
        return

    usage_tracker.finish_event(
        None,
        status_code,
        upstream=upstream,
        response_payload=response_payload,
        response_text=response_text,
        reasoning_text=reasoning_text,
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

    def header_value(name: str):
        if not isinstance(headers, dict):
            return None
        value = headers.get(name)
        if value is not None:
            return value
        target = name.lower()
        for key, candidate in headers.items():
            if isinstance(key, str) and key.lower() == target:
                return candidate
        return None

    headers = header_builder(effective_api_key, request_id)
    initiator_header = header_value("X-Initiator")
    initiator = str(initiator_header or "").strip().lower()
    initiator_verdict = None
    if isinstance(trace_metadata, dict):
        initiator_verdict = trace_metadata.get("initiator_verdict")
    prompt_preview = _extract_prompt_preview(
        source_body if isinstance(source_body, dict) else body,
        truncate=initiator != "user",
    )
    usage_event = usage_tracker.start_event(
        request,
        requested_model,
        resolved_model,
        initiator_header,
        request_id=request_id,
        request_body=body,
        upstream_path=upstream_path,
        outbound_headers=headers,
        prompt_preview=prompt_preview,
        initiator_verdict=initiator_verdict if isinstance(initiator_verdict, dict) else None,
    )
    auto_update_runtime_controller.note_request_started(request_id)
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
            prompt_preview=prompt_preview,
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
            auto_update_request_tracked=True,
        ),
        None,
    )



def _bridge_error_response(plan: BridgeExecutionPlan):
    if plan.caller_protocol == "anthropic":
        return format_translation.anthropic_error_response
    return format_translation.openai_error_response


import request_headers as _request_headers_module


def _build_anthropic_messages_passthrough_headers(
    request: Request,
    *,
    original_body: dict,
    bridge_plan: BridgeExecutionPlan,
    api_key: str,
    request_id: str | None = None,
    verdict_sink: dict | None = None,
) -> dict:
    """Headers for native Anthropic Messages passthrough."""
    base_headers = format_translation.build_copilot_headers(api_key)

    # Resolve initiator from the original Anthropic-shaped request (when the
    # caller is Claude Code) or from translated messages (when called via
    # Codex /v1/responses bridge).
    if bridge_plan.caller_protocol == "anthropic":
        body_for_initiator = original_body if isinstance(original_body, dict) else bridge_plan.upstream_body
        messages = body_for_initiator.get("messages") if isinstance(body_for_initiator, dict) else None
        system = body_for_initiator.get("system") if isinstance(body_for_initiator, dict) else None
        model_for_initiator = (
            body_for_initiator.get("model")
            if isinstance(body_for_initiator, dict)
            else bridge_plan.resolved_model
        )
        initiator = _initiator_policy.resolve_anthropic_messages(
            messages,
            model_for_initiator,
            system=system,
            subagent=request.headers.get("x-openai-subagent"),
            request_id=request_id,
            verdict_sink=verdict_sink,
        )
    else:
        body_for_initiator = bridge_plan.upstream_body if isinstance(bridge_plan.upstream_body, dict) else {}
        messages = body_for_initiator.get("messages")
        system = body_for_initiator.get("system")
        initiator = _initiator_policy.resolve_anthropic_messages(
            messages,
            bridge_plan.resolved_model,
            system=system,
            subagent=request.headers.get("x-openai-subagent"),
            request_id=request_id,
            verdict_sink=verdict_sink,
        )

    # Forward standard request id / session headers via the existing helper
    # so we behave like other bridge paths.
    _request_headers_module._apply_forwarded_request_headers(
        base_headers,
        request,
        original_body if isinstance(original_body, dict) else None,
        session_id_resolver=usage_tracking.request_session_id,
    )

    # Parse incoming anthropic-beta header (comma separated) for derive_anthropic_betas.
    incoming_beta = None
    for header_name in ("anthropic-beta", "Anthropic-Beta"):
        value = request.headers.get(header_name) if hasattr(request, "headers") else None
        if value:
            incoming_beta = value
            break
    incoming_betas = (
        [piece.strip() for piece in incoming_beta.split(",") if piece.strip()]
        if isinstance(incoming_beta, str)
        else []
    )

    body_for_betas = bridge_plan.upstream_body if isinstance(bridge_plan.upstream_body, dict) else {}
    anthropic_betas = _request_headers_module.derive_anthropic_betas(
        client_betas=incoming_betas,
        body=body_for_betas,
        model=bridge_plan.resolved_model or "",
    )

    interaction_id = usage_tracking.request_session_id(
        request,
        original_body if isinstance(original_body, dict) else None,
    )
    headers = _request_headers_module.build_anthropic_messages_passthrough_headers(
        request_id=request_id or "",
        initiator=initiator,
        interaction_id=interaction_id if isinstance(interaction_id, str) and interaction_id else None,
        interaction_type=None,
        anthropic_betas=anthropic_betas,
        base_headers=base_headers,
    )
    return headers


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
            affinity_body=original_body,
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
    if bridge_plan.header_kind == "messages":
        return _build_anthropic_messages_passthrough_headers(
            request,
            original_body=original_body,
            bridge_plan=bridge_plan,
            api_key=api_key,
            request_id=request_id,
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
    trace_metadata_extra: dict | None = None,
) -> tuple[UpstreamRequestPlan | None, Response | None]:
    upstream_url = f"{api_base.rstrip('/')}{bridge_plan.upstream_path}"
    verdict_sink: dict = {}
    trace_metadata = {
        "bridge": True,
        "strategy_name": bridge_plan.strategy_name,
        "caller_protocol": bridge_plan.caller_protocol,
        "upstream_protocol": bridge_plan.upstream_protocol,
        "header_kind": bridge_plan.header_kind,
        "initiator_verdict": verdict_sink,
    }
    if bridge_plan.diagnostics:
        trace_metadata["sanitizer_diagnostics"] = list(bridge_plan.diagnostics)
    if isinstance(trace_metadata_extra, dict):
        trace_metadata.update(trace_metadata_extra)
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
        trace_metadata=trace_metadata,
    )


def _translate_bridge_success_payload(bridge_plan: BridgeExecutionPlan, payload: dict) -> dict:
    if bridge_plan.caller_protocol == "responses" and bridge_plan.upstream_protocol == "chat":
        if bridge_plan.is_compact:
            return format_translation.chat_completion_to_compaction_response(
                payload,
                fallback_model=bridge_plan.resolved_model,
            )
        return format_translation.chat_completion_to_response(payload, fallback_model=bridge_plan.resolved_model)
    if bridge_plan.caller_protocol == "anthropic" and bridge_plan.upstream_protocol == "responses":
        return format_translation.response_payload_to_anthropic(payload, fallback_model=bridge_plan.resolved_model)
    if bridge_plan.caller_protocol == "anthropic" and bridge_plan.upstream_protocol == "chat":
        return format_translation.chat_completion_to_anthropic(payload, fallback_model=bridge_plan.resolved_model)
    if bridge_plan.caller_protocol == "anthropic" and bridge_plan.upstream_protocol == "messages":
        # Native Anthropic passthrough: strip any internal-only fields
        # (none defined yet) and return upstream payload as-is.
        return payload
    if bridge_plan.caller_protocol == "responses" and bridge_plan.upstream_protocol == "messages":
        return format_translation.anthropic_response_to_responses(
            payload, fallback_model=bridge_plan.resolved_model,
        )
    return payload


def _bridge_error_response_from_upstream(bridge_plan: BridgeExecutionPlan, upstream: httpx.Response) -> Response:
    if bridge_plan.caller_protocol == "anthropic":
        return format_translation.anthropic_error_response_from_upstream(upstream)
    if bridge_plan.caller_protocol == "responses" and bridge_plan.upstream_protocol == "messages":
        payload = _extract_upstream_json_payload(upstream)
        message = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                message = error["message"]
        return format_translation.openai_error_response(upstream.status_code, message)
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
        source_iter = _stream_with_update_notice(upstream.aiter_bytes(), stream_type, getattr(upstream, "headers", None))
        try:
            async for chunk in source_iter:
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
            async for event in translator.translate(_stream_with_update_notice(upstream.aiter_bytes(), "chat", getattr(upstream, "headers", None))):
                yield event
        finally:
            response_payload = translator.build_response_payload()
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text=translator.response_text,
                reasoning_text=translator.thinking_text,
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
        upstream_iter = _stream_with_update_notice(upstream.aiter_bytes(), "chat", getattr(upstream, "headers", None))
        try:
            async for event in translator.translate(upstream_iter):
                yield event
        finally:
            response_payload = translator.build_response_payload()
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text=translator.response_text,
                reasoning_text=translator.reasoning_text or None,
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
            async for event in translator.translate(_stream_with_update_notice(upstream.aiter_bytes(), "responses", getattr(upstream, "headers", None))):
                yield event
        finally:
            response_payload = translator.build_response_payload()
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text=translator.response_text,
                reasoning_text=translator.thinking_text,
                usage=response_payload["usage"],
            )
            await upstream.aclose()
            await client.aclose()

    return GracefulStreamingResponse(
        stream_translated(),
        status_code=upstream.status_code,
        headers=response_headers,
    )



# ─── Anthropic Messages passthrough streaming ────────────────────────────────


async def proxy_anthropic_passthrough_streaming_response(
    upstream_url: str,
    headers: dict,
    body: dict,
    fallback_model: str,
    timeout: int = 300,
    usage_event: dict | None = None,
    trace_plan: UpstreamRequestPlan | None = None,
) -> Response:
    """Re-emit upstream Anthropic /v1/messages SSE bytes verbatim to the
    client, while parsing message_start / message_delta usage so we can still
    report token counts to the dashboard."""
    del fallback_model  # unused; kept for signature symmetry with siblings.

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
    # Propagate quota snapshot headers if present.
    for key, value in upstream.headers.items():
        if key.lower().startswith("x-quota-snapshot"):
            response_headers[key] = value

    def merge_anthropic_usage(usage_state: dict, usage: dict) -> None:
        if not isinstance(usage, dict):
            return

        def read_int(key: str):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
            return None

        input_tokens = read_int("input_tokens")
        if input_tokens is not None:
            usage_state["input_tokens"] = input_tokens

        output_tokens = read_int("output_tokens")
        if output_tokens is not None:
            usage_state["output_tokens"] = output_tokens

        cache_read = read_int("cache_read_input_tokens")
        if cache_read is None:
            cache_read = read_int("cached_input_tokens")
        if cache_read is not None:
            usage_state["cached_input_tokens"] = cache_read

        cache_creation = read_int("cache_creation_input_tokens")
        if cache_creation is not None:
            usage_state["cache_creation_input_tokens"] = cache_creation

        total_tokens = read_int("total_tokens")
        if total_tokens is not None:
            usage_state["total_tokens"] = total_tokens
        else:
            usage_state["total_tokens"] = (
                int(usage_state.get("input_tokens", 0) or 0)
                + int(usage_state.get("output_tokens", 0) or 0)
            )

    async def stream_passthrough():
        usage_state: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        first_output_marked = False
        buffer = ""
        try:
            async for chunk in _stream_with_update_notice(upstream.aiter_bytes(), "anthropic", getattr(upstream, "headers", None)):
                if chunk:
                    yield chunk
                    try:
                        text = chunk.decode("utf-8", errors="replace")
                    except Exception:
                        text = ""
                    buffer += text
                    normalized = buffer.replace("\r\n", "\n")
                    while "\n\n" in normalized:
                        raw_block, normalized = normalized.split("\n\n", 1)
                        event_name, data = format_translation.parse_sse_block(raw_block)
                        if not data:
                            continue
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        evt = (event_name or payload.get("type") if isinstance(payload, dict) else event_name) or ""
                        evt = str(evt).lower()
                        if evt == "message_start" and isinstance(payload, dict):
                            message = payload.get("message")
                            if isinstance(message, dict) and isinstance(message.get("usage"), dict):
                                merge_anthropic_usage(usage_state, message["usage"])
                        elif evt == "message_delta" and isinstance(payload, dict):
                            u = payload.get("usage")
                            if isinstance(u, dict):
                                merge_anthropic_usage(usage_state, u)
                        elif evt in ("content_block_delta", "content_block_start"):
                            if not first_output_marked:
                                first_output_marked = True
                                usage_tracker.mark_first_output(usage_event)
                    buffer = normalized
        finally:
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                usage=usage_state,
            )
            await upstream.aclose()
            await client.aclose()

    return GracefulStreamingResponse(
        stream_passthrough(),
        status_code=upstream.status_code,
        headers=response_headers,
    )


async def proxy_responses_from_anthropic_streaming_response(
    upstream_url: str,
    headers: dict,
    body: dict,
    fallback_model: str,
    timeout: int = 300,
    usage_event: dict | None = None,
    trace_plan: UpstreamRequestPlan | None = None,
) -> Response:
    """Translate upstream Anthropic Messages SSE into Responses SSE."""
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
            upstream_payload = _extract_upstream_json_payload(upstream)
            upstream_text = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
            # Upstream is Anthropic-shaped; reshape to a Responses-shaped error
            # so Codex callers receive the format they expect.
            err_message = upstream_text
            if isinstance(upstream_payload, dict):
                err_obj = upstream_payload.get("error")
                if isinstance(err_obj, dict) and isinstance(err_obj.get("message"), str):
                    err_message = err_obj["message"]
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=upstream_payload,
                response_text=err_message,
            )
            return format_translation.openai_error_response(upstream.status_code, err_message)
        finally:
            await upstream.aclose()
            await client.aclose()

    response_headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "content-type": "text/event-stream; charset=utf-8",
    }
    for key, value in upstream.headers.items():
        if key.lower().startswith("x-quota-snapshot"):
            response_headers[key] = value

    async def stream_translated():
        translator = AnthropicToResponsesStreamTranslator(
            model=fallback_model,
            mark_first_output=lambda: usage_tracker.mark_first_output(usage_event),
        )
        try:
            async for evbytes in translator.translate(_stream_with_update_notice(upstream.aiter_bytes(), "anthropic", getattr(upstream, "headers", None))):
                yield evbytes
            yield b"data: [DONE]\n\n"
        finally:
            response_payload = translator.build_response_payload()
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text=translator.response_text,
                reasoning_text=translator.reasoning_text,
                usage=response_payload.get("usage"),
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
    if bridge_plan.caller_protocol == "anthropic" and bridge_plan.upstream_protocol == "messages":
        return await proxy_anthropic_passthrough_streaming_response(
            plan.upstream_url,
            plan.headers,
            plan.body,
            bridge_plan.resolved_model,
            timeout=300,
            usage_event=plan.usage_event,
            trace_plan=plan,
        )
    if bridge_plan.caller_protocol == "responses" and bridge_plan.upstream_protocol == "messages":
        return await proxy_responses_from_anthropic_streaming_response(
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


_COPILOT_MODEL_CAPS_CACHE: dict[str, object] = {"key": None, "ts": 0.0, "data": {}}
_COPILOT_MODEL_CAPS_LOCK = threading.Lock()
_COPILOT_MODEL_CAPS_TTL_SECONDS = 300.0
# Capability fetch is best-effort and used during client-config writes (codex
# `enable_target`). Use a tight timeout so a slow/unreachable upstream does not
# stall proxy initialization — defaults still work without it.
_COPILOT_MODEL_CAPS_FETCH_TIMEOUT_SECONDS = 5.0


def fetch_copilot_model_capabilities() -> dict[str, dict]:
    """Best-effort fetch of upstream Copilot /models capabilities.

    Returns a mapping ``{model_id: capabilities_dict}`` enriched into the
    shape consumed by ``ProxyClientConfigService``. Any failure (missing
    auth, network error, parse error) returns an empty dict so the caller
    falls back to defaults.
    """
    try:
        api_base = auth.get_api_base().rstrip("/")
    except Exception:
        return {}

    now = time.monotonic()
    with _COPILOT_MODEL_CAPS_LOCK:
        cache = _COPILOT_MODEL_CAPS_CACHE
        if (
            cache.get("key") == api_base
            and isinstance(cache.get("data"), dict)
            and cache["data"]
            and (now - float(cache.get("ts", 0.0))) < _COPILOT_MODEL_CAPS_TTL_SECONDS
        ):
            return dict(cache["data"])  # type: ignore[arg-type]

    try:
        api_key = auth.get_api_key()
    except Exception:
        return {}

    headers = format_translation.build_copilot_headers(api_key)
    url = f"{api_base}/models"
    try:
        with httpx.Client(timeout=_COPILOT_MODEL_CAPS_FETCH_TIMEOUT_SECONDS) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return {}

    raw_entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(raw_entries, list):
        return {}

    result: dict[str, dict] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        caps = entry.get("capabilities") if isinstance(entry.get("capabilities"), dict) else {}
        limits = caps.get("limits") if isinstance(caps.get("limits"), dict) else {}
        supports = caps.get("supports") if isinstance(caps.get("supports"), dict) else {}

        max_context_window = limits.get("max_context_window_tokens")
        max_prompt_tokens = limits.get("max_prompt_tokens")
        reasoning_efforts = supports.get("reasoning_effort") if isinstance(supports.get("reasoning_effort"), list) else None
        vision_flag = bool(supports.get("vision"))
        parallel = supports.get("parallel_tool_calls")
        input_modalities = ["text", "image"] if vision_flag else ["text"]

        enriched: dict[str, object] = {
            "input_modalities": input_modalities,
            "vision": vision_flag,
        }
        if isinstance(max_prompt_tokens, int) and max_prompt_tokens > 0:
            enriched["context_window"] = max_prompt_tokens
        elif isinstance(max_context_window, int) and max_context_window > 0:
            enriched["context_window"] = max_context_window
        if isinstance(max_context_window, int) and max_context_window > 0:
            enriched["max_context_window"] = max_context_window
        elif isinstance(max_prompt_tokens, int) and max_prompt_tokens > 0:
            enriched["max_context_window"] = max_prompt_tokens
        if reasoning_efforts is not None:
            enriched["reasoning_efforts"] = reasoning_efforts
        if isinstance(parallel, bool):
            enriched["parallel_tool_calls"] = parallel
        supported_endpoints = entry.get("supported_endpoints") or []
        if not isinstance(supported_endpoints, list):
            supported_endpoints = []
        enriched["supported_endpoints"] = list(supported_endpoints)
        enriched["messages_endpoint_supported"] = "/v1/messages" in supported_endpoints
        result[model_id] = enriched

    with _COPILOT_MODEL_CAPS_LOCK:
        _COPILOT_MODEL_CAPS_CACHE["key"] = api_base
        _COPILOT_MODEL_CAPS_CACHE["ts"] = now
        _COPILOT_MODEL_CAPS_CACHE["data"] = result
    return dict(result)


# Models known to natively support Anthropic /v1/messages upstream. This is a
# safety net for empty/stale capability caches and for cache records that lag
# newly exposed Claude endpoints. Case-insensitive prefix match.
_NATIVE_MESSAGES_FALLBACK_ALLOWLIST = frozenset({
    "claude-sonnet-4.5", "claude-sonnet-4.6",
    "claude-opus-4.5", "claude-opus-4.6", "claude-opus-4.7",
    "claude-haiku-4.5",
})


def model_supports_native_messages(model: str) -> bool:
    """Returns True when `model` advertises `/v1/messages` in the Copilot
    `/models` capability cache, or matches the known-native fallback allowlist
    by case-insensitive prefix.

    The allowlist is intentionally consulted even when the capability cache has
    a stale/negative record. Routing known Claude models through the chat bridge
    loses native Messages thinking/tool semantics and can leave Codex with a
    reasoning-only turn followed by an invisible/stalled tool call.
    """
    if not isinstance(model, str) or not model:
        return False

    candidate = model.lower()
    if candidate.startswith("anthropic/"):
        candidate = candidate.split("/", 1)[1]

    with _COPILOT_MODEL_CAPS_LOCK:
        cache = _COPILOT_MODEL_CAPS_CACHE
        data = cache.get("data") if isinstance(cache, dict) else None
        record = data.get(model) if isinstance(data, dict) and data else None
        if isinstance(record, dict):
            if bool(record.get("messages_endpoint_supported")):
                return True

    for entry in _NATIVE_MESSAGES_FALLBACK_ALLOWLIST:
        if candidate.startswith(entry.lower()):
            return True
    return False


async def _proxy_models_request() -> Response:
    try:
        api_key = auth.get_api_key()
    except Exception:
        return format_translation.openai_error_response(401, AUTH_FAILURE_MESSAGE)

    upstream_url = f"{auth.get_api_base().rstrip('/')}/models"
    headers = format_translation.build_copilot_headers(api_key)

    try:
        async with httpx.AsyncClient(timeout=configured_upstream_timeout_seconds()) as client:
            request = client.build_request("GET", upstream_url, headers=headers)
            upstream = await throttled_client_send(client, request)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        return format_translation.openai_error_response(status_code, message)

    return proxy_non_streaming_response(upstream)


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


# ─── Auth routes ──────────────────────────────────────────────────────────────


@app.get("/api/auth/status")
async def auth_status_api():
    return JSONResponse(
        content=auth.auth_status(),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/auth/device")
async def auth_device_api():
    return JSONResponse(
        content=auth.begin_device_flow(),
        headers={"Cache-Control": "no-store"},
    )


# ─── Config API routes ────────────────────────────────────────────────────────

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


@app.get("/api/config/client-proxy")
async def client_proxy_status_api():
    return JSONResponse(content=client_proxy_config_service.proxy_client_status_payload())


@app.post("/api/config/client-proxy/settings")
async def client_proxy_settings_api(request: Request):
    payload = await parse_json_request(request)
    result = client_proxy_config_service.save_client_proxy_settings(payload)
    return JSONResponse(content=result)


@app.get("/api/config/model-remapping")
@app.get("/api/config/model-routing")
async def model_routing_status_api():
    return JSONResponse(content=model_routing_config_service.config_payload())


@app.post("/api/config/model-remapping")
@app.post("/api/config/model-routing")
async def model_routing_config_api(request: Request):
    payload = await parse_json_request(request)
    result = model_routing_config_service.save_settings(payload)
    client_proxy_config_service.refresh_client_model_metadata()
    return JSONResponse(content=result)


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




def _responses_message_role(item) -> str | None:
    if not isinstance(item, dict):
        return None
    if str(item.get("type", "")).lower() != "message":
        return None
    role = item.get("role")
    if not isinstance(role, str):
        return None
    normalized = role.strip().lower()
    return normalized or None


def _responses_input_developer_message_count(input_value) -> int:
    if not isinstance(input_value, list):
        return 0
    return sum(1 for item in input_value if _responses_message_role(item) == "developer")


def _responses_input_has_tool_history(input_value) -> bool:
    if not isinstance(input_value, list):
        return False
    tool_item_types = {
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
    }
    for item in input_value:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).lower() in tool_item_types:
            return True
    return False


def _encrypted_reasoning_strip_reason_for_responses_context(request: Request, input_value) -> str | None:
    """Classify contexts where replayed encrypted reasoning is not worth forwarding.

    Codex subagent/fork starts can replay the parent transcript, including
    encrypted reasoning blobs, under a fresh prompt-cache key. Copilot/OpenAI
    validates those blobs against the active lineage and rejects foreign ones
    with ``invalid_request_body``. Keep normal same-thread replay intact, but
    strip ciphertext when we have an explicit subagent marker or the input has
    the fork shape Codex emits today (multiple developer messages). Summaries
    remain available, so the visible transcript is preserved.
    """
    subagent = request.headers.get("x-openai-subagent") if hasattr(request, "headers") else None
    if isinstance(subagent, str) and subagent.strip():
        return "subagent_header"
    if _responses_input_developer_message_count(input_value) > 1:
        return "multiple_developer_messages"
    if _responses_input_has_tool_history(input_value):
        return "tool_history"
    return None


def _should_drop_reasoning_items_for_responses_context(strip_reason: str | None) -> bool:
    return strip_reason is not None


def _max_encrypted_reasoning_items_for_responses_input(input_value) -> int | None:
    if not isinstance(input_value, list):
        return None
    if len(input_value) < _RESPONSES_REASONING_TRIM_INPUT_ITEM_THRESHOLD:
        return None
    return _RESPONSES_REASONING_ENCRYPTED_KEEP_LAST


def _responses_input_encrypted_content_count(input_value) -> int:
    if not isinstance(input_value, list):
        return 0
    count = 0
    for item in input_value:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"reasoning", "compaction"}:
            continue
        encrypted_content = item.get("encrypted_content")
        if isinstance(encrypted_content, str) and encrypted_content:
            count += 1
    return count


def _responses_input_sanitization_trace(
    raw_input,
    sanitized_input,
    *,
    max_encrypted_reasoning_items: int | None,
    encrypted_reasoning_strip_reason: str | None,
    dropped_reasoning_items: bool,
) -> dict | None:
    if not isinstance(raw_input, list):
        return None
    before_encrypted = _responses_input_encrypted_content_count(raw_input)
    after_encrypted = _responses_input_encrypted_content_count(sanitized_input)
    if (
        not before_encrypted
        and max_encrypted_reasoning_items is None
        and encrypted_reasoning_strip_reason is None
        and not dropped_reasoning_items
    ):
        return None

    if encrypted_reasoning_strip_reason is not None:
        preservation = "disabled"
    elif max_encrypted_reasoning_items is not None:
        preservation = "limited"
    else:
        preservation = "preserved"

    return {
        "input_items_before": len(raw_input),
        "input_items_after": len(sanitized_input) if isinstance(sanitized_input, list) else None,
        "encrypted_content_items_before": before_encrypted,
        "encrypted_content_items_after": after_encrypted,
        "encrypted_content_items_dropped": max(0, before_encrypted - after_encrypted),
        "encrypted_content_preservation": preservation,
        "encrypted_content_strip_reason": encrypted_reasoning_strip_reason,
        "encrypted_keep_last": max_encrypted_reasoning_items,
        "reasoning_items_dropped": dropped_reasoning_items,
        "trim_input_item_threshold": _RESPONSES_REASONING_TRIM_INPUT_ITEM_THRESHOLD,
    }

# ─── Route: /v1/responses  (Codex / Responses API) ───────────────────────────

@app.get("/api/config/background-proxy")
async def background_proxy_status_api():
    return JSONResponse(content=background_proxy_manager.status_payload())


@app.get("/api/config/auto-update")
async def auto_update_status_api():
    return JSONResponse(content=auto_update_runtime_controller.status_payload())


@app.post("/api/config/auto-update")
async def auto_update_config_api(request: Request):
    payload = await parse_json_request(request)
    action = str(payload.get("action") or "").strip().lower()
    if action == "set_mode":
        mode = str(payload.get("mode") or "").strip().lower()
        try:
            settings = auto_update_manager.set_mode(mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse(content={**auto_update_runtime_controller.status_payload(), "settings": settings})
    if action == "check_now":
        result = await auto_update_runtime_controller.run_due_check(force=True)
        return JSONResponse(content={**auto_update_runtime_controller.status_payload(), "result": result})
    if action == "pull_update":
        result = await auto_update_runtime_controller.apply_update()
        return JSONResponse(content={**auto_update_runtime_controller.status_payload(), "result": result})
    if action == "upgrade_anyway":
        result = await auto_update_runtime_controller.apply_update(
            override_local_changes=True,
        )
        return JSONResponse(content={**auto_update_runtime_controller.status_payload(), "result": result})
    if action == "restart_proxy":
        scheduled = auto_update_runtime_controller.restart_when_idle("dashboard")
        return JSONResponse(content={**auto_update_runtime_controller.status_payload(), "scheduled": scheduled})
    raise HTTPException(status_code=400, detail="unsupported auto-update action")


@app.post("/api/config/background-proxy")
async def background_proxy_config_api(request: Request):
    payload = await parse_json_request(request)
    action = payload.get("action")
    try:
        if action == "enable_startup":
            result = background_proxy_manager.enable_startup()
            message = "Background startup enabled."
        elif action == "disable_startup":
            result = background_proxy_manager.disable_startup()
            message = "Background startup disabled."
        elif action == "install_shell_commands":
            result = background_proxy_manager.install_shell_commands()
            message = "Shell commands installed."
        elif action == "uninstall_shell_commands":
            result = background_proxy_manager.uninstall_shell_commands()
            message = "Shell commands removed."
        else:
            raise HTTPException(status_code=400, detail="Unsupported background proxy action.")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update background proxy setup: {exc}") from exc
    return JSONResponse(content={**result, "message": message})


@app.post("/responses")
@app.post("/v1/responses")
async def responses(request: Request):
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return format_translation.openai_error_response(
            exc.status_code,
            format_translation.http_exception_detail_to_message(exc.detail),
        )

    # Sanitize input (multi-turn encrypted_content passthrough). Forked
    # subagent contexts replay parent reasoning under a new lineage, so do not
    # forward unverifiable encrypted reasoning blobs in that shape.
    raw_input = body.get("input")
    has_compaction_input = format_translation.input_contains_compaction(raw_input)
    encrypted_reasoning_strip_reason = _encrypted_reasoning_strip_reason_for_responses_context(
        request, raw_input
    )
    drop_reasoning_items = _should_drop_reasoning_items_for_responses_context(
        encrypted_reasoning_strip_reason
    )
    max_encrypted_reasoning_items = _max_encrypted_reasoning_items_for_responses_input(raw_input)
    input_sanitization_trace = None
    if raw_input is not None:
        body["input"] = format_translation.sanitize_input(
            raw_input,
            preserve_encrypted_content=encrypted_reasoning_strip_reason is None,
            max_encrypted_reasoning_items=max_encrypted_reasoning_items,
            drop_reasoning_items=drop_reasoning_items,
        )
        input_sanitization_trace = _responses_input_sanitization_trace(
            raw_input,
            body.get("input"),
            max_encrypted_reasoning_items=max_encrypted_reasoning_items,
            encrypted_reasoning_strip_reason=encrypted_reasoning_strip_reason,
            dropped_reasoning_items=drop_reasoning_items,
        )

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
        trace_metadata_extra=(
            {"responses_input_sanitization": input_sanitization_trace}
            if input_sanitization_trace is not None
            else None
        ),
    )
    if error_response is not None:
        return error_response

    if bridge_plan.stream:
        return await _proxy_bridge_streaming_response(plan, bridge_plan)
    return await _post_bridge_non_streaming_request(plan, bridge_plan)


@app.post("/responses/compact")
@app.post("/v1/responses/compact")
async def responses_compact(request: Request):
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return format_translation.openai_error_response(
            exc.status_code,
            format_translation.http_exception_detail_to_message(exc.detail),
        )

    resolved_target = model_routing_config_service.resolve_target_model(body.get("model"))
    force_responses_safe_transcript = (
        isinstance(resolved_target, str)
        and model_provider_family(resolved_target) not in (None, "codex")
    )
    summary_request = format_translation.build_fake_compaction_request(
        body,
        force_responses_safe_transcript=force_responses_safe_transcript,
    )
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

@app.post("/chat/completions")
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


@app.get("/models")
@app.get("/v1/models")
async def models():
    return await _proxy_models_request()


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
    update_result = auto_update_manager.startup_check_for_update()
    if update_result.get("attempted"):
        print(f"Auto-update check: {json.dumps(update_result, default=str)}", flush=True)

    # Best-effort cleanup of legacy on-disk state from the pre-header-driven
    # quota implementation. Safe to remove unconditionally; if the user never
    # configured them, the unlinks are no-ops.
    for legacy_path in (
        LEGACY_PREMIUM_PLAN_CONFIG_FILE,
        LEGACY_BILLING_TOKEN_FILE,
    ):
        try:
            os.remove(legacy_path)
        except OSError:
            pass

    # Start the server immediately so first-run setup can complete from the
    # browser dashboard instead of blocking on a terminal prompt.
    print("Starting GHCP proxy on http://localhost:8000 (loopback only)", flush=True)
    print("  Responses API : POST /v1/responses", flush=True)
    print("  Compaction    : POST /v1/responses/compact", flush=True)
    print("  Chat API      : POST /v1/chat/completions", flush=True)
    print("  Dashboard     : GET  /ui", flush=True)
    print("", flush=True)
    print("  If this is a fresh setup, open /ui and complete GitHub sign-in there.", flush=True)
    print("", flush=True)
    print("  Set in your shell:", flush=True)
    print("    export OPENAI_BASE_URL=http://localhost:8000/v1", flush=True)
    print("    export OPENAI_API_KEY=anything", flush=True)
    print("", flush=True)

    _write_proxy_pid_file()
    atexit.register(_remove_proxy_pid_file)
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False, timeout_graceful_shutdown=2)
    finally:
        revert_client_proxy_configs_on_shutdown()
        _remove_proxy_pid_file()
