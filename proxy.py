"""
Lightweight GitHub Copilot reverse proxy — Responses API path.
Designed for Codex / codex-mini / gpt-5.1-codex and any model that
requires the Responses API instead of Chat Completions.

Usage:
  python proxy.py
  → If no token exists, prompts you to authorize via GitHub device flow
  → Then starts serving on http://127.0.0.1:8000

Configure Codex:
  export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
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
import gzip
import hashlib
import messages_preprocess
import migrate_runtime_paths
import json
import sqlite3
import tempfile
import time
import threading
import safeguard_config as safeguard_config_module
import protocol_replies
import upstream_errors
import update_notice
import usage_reminder
import usage_tracking
import util
from collections import OrderedDict, deque
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
    ResponsesStreamIdSyncer,
    ResponsesToAnthropicStreamTranslator,
)
from initiator_policy import InitiatorPolicy, is_approval_agent_request
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
    REQUEST_PROMPT_ARCHIVE_DIR,
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
_REQUEST_PROMPT_LOCK = Lock()
_REQUEST_PROMPT_ACTIVE_IDS: set[str] = set()
_REQUEST_PROMPT_FILE_PREFIX = "request-prompt-"
_CLIENT_PROXY_STARTUP_RESTORE_LOCK = Lock()
_CLIENT_PROXY_STARTUP_RESTORE_COMPLETE = False
_CLIENT_PROXY_SHUTDOWN_REVERT_LOCK = Lock()
_CLIENT_PROXY_SHUTDOWN_REVERT_COMPLETE = False
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
    "x-parent-agent-id",
    "x-client-session-id",
    "x-client-machine-id",
    "x-copilot-client-exp-assignment-context",
    "x-github-api-version",
    "x-stainless-retry-count",
    "x-stainless-lang",
    "x-stainless-package-version",
    "x-stainless-os",
    "x-stainless-arch",
    "x-stainless-runtime",
    "x-stainless-runtime-version",
    "accept-language",
    "sec-fetch-mode",
    "x-request-id",
    "x-github-request-id",
    "accept",
    "accept-encoding",
    "host",
    "connection",
    "content-length",
}
AUTH_FAILURE_MESSAGE = "GitHub Copilot authorization required. Open /ui to sign in."
INVALID_BRIDGE_REQUEST_MESSAGE = "Invalid request"
DEBUG_DETAIL_CONTEXT_REQUESTS = 10

safeguard_event_store = dashboard_module.create_safeguard_event_store()
_DEBUG_DETAIL_CAPTURE_LOCK = threading.Lock()
DEBUG_DETAIL_SESSION_BUFFER_LIMIT = 64
_DEBUG_DETAIL_REQUEST_SNAPSHOT_INDEX_MAXLEN = DEBUG_DETAIL_SESSION_BUFFER_LIMIT
DEBUG_DETAIL_SESSION_DETAIL_LIMIT = DEBUG_DETAIL_CONTEXT_REQUESTS
_DEBUG_DETAIL_SESSION_RECENT_REQUESTS: OrderedDict[str, deque[dict]] = OrderedDict()
_DEBUG_DETAIL_REQUEST_SNAPSHOTS_BY_ID: OrderedDict[str, dict] = OrderedDict()
_DEBUG_DETAIL_SESSION_CAPTURED_REQUEST_IDS: OrderedDict[str, set[str]] = OrderedDict()
_DEBUG_DETAIL_SNAPSHOT_SEQUENCE = 0

# Upstream prompt-cache settle. This is deliberately a cross-lineage family
# guard, not a same-lineage throttle: back-to-back turns in one append-only
# conversation are exactly what prompt caching is meant to handle.  The only
# case we delay is a different lineage entering the same Copilot parent-task
# family immediately after a sibling/parent finished.
CACHE_SETTLE_DELAY_SECONDS = 0.0
_PROMPT_CACHE_SETTLE_LOCK = threading.Lock()
_PROMPT_CACHE_LAST_FINISH_BY_FAMILY: dict[tuple[str, str], tuple[str, float]] = {}

def _reset_debug_detail_capture_state() -> None:
    global _DEBUG_DETAIL_SNAPSHOT_SEQUENCE
    with _DEBUG_DETAIL_CAPTURE_LOCK:
        _DEBUG_DETAIL_SESSION_RECENT_REQUESTS.clear()
        _DEBUG_DETAIL_REQUEST_SNAPSHOTS_BY_ID.clear()
        _DEBUG_DETAIL_SESSION_CAPTURED_REQUEST_IDS.clear()
        _DEBUG_DETAIL_SNAPSHOT_SEQUENCE = 0


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


def request_prompt_archive_dir() -> str:
    configured = str(os.environ.get("GHCP_REQUEST_PROMPT_ARCHIVE_DIR", "")).strip()
    return os.path.expanduser(configured or REQUEST_PROMPT_ARCHIVE_DIR)


def _request_prompt_file_name(request_id: str | None) -> str | None:
    if not isinstance(request_id, str):
        return None
    normalized_request_id = request_id.strip()
    if not normalized_request_id:
        return None
    safe_request_id = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in normalized_request_id
    )
    if not safe_request_id:
        return None
    return f"{_REQUEST_PROMPT_FILE_PREFIX}{safe_request_id}.json"


def _request_prompt_file_path(request_id: str | None) -> str | None:
    filename = _request_prompt_file_name(request_id)
    if filename is None:
        return None
    return os.path.join(request_prompt_archive_dir(), filename)


def _save_request_prompt_record(
    request_id: str | None,
    request_path: str | None,
    request_body: dict | None,
) -> None:
    archive_path = _request_prompt_file_path(request_id)
    if archive_path is None or not isinstance(request_body, dict):
        return

    prompt_text = util.extract_request_prompt_text(request_body)
    if not prompt_text:
        return

    record = {
        "request_id": request_id,
        "path": request_path,
        "stored_at": util.utc_now_iso(),
        "char_count": len(prompt_text),
        "prompt_text": prompt_text,
    }
    archive_dir = os.path.dirname(archive_path)
    temp_path = None
    try:
        os.makedirs(archive_dir, exist_ok=True)
        with _REQUEST_PROMPT_LOCK:
            _REQUEST_PROMPT_ACTIVE_IDS.add(str(request_id))
            fd, temp_path = tempfile.mkstemp(prefix="request-prompt-", suffix=".tmp", dir=archive_dir)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":"), default=util._json_default))
            os.replace(temp_path, archive_path)
    except OSError as exc:
        with _REQUEST_PROMPT_LOCK:
            _REQUEST_PROMPT_ACTIVE_IDS.discard(str(request_id))
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        print(f"Warning: failed to write request prompt archive: {exc}", file=sys.stderr, flush=True)


def _load_request_prompt_record(request_id: str | None) -> dict | None:
    archive_path = _request_prompt_file_path(request_id)
    if archive_path is None or not os.path.exists(archive_path):
        return None
    try:
        with open(archive_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    prompt_text = payload.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return None

    char_count = payload.get("char_count")
    if not isinstance(char_count, int):
        char_count = len(prompt_text)
    return {
        "request_id": payload.get("request_id") if isinstance(payload.get("request_id"), str) else request_id,
        "path": payload.get("path") if isinstance(payload.get("path"), str) else None,
        "stored_at": payload.get("stored_at") if isinstance(payload.get("stored_at"), str) else None,
        "char_count": char_count,
        "prompt_text": prompt_text,
    }


def _recent_request_prompt_ids() -> set[str]:
    keep_ids = {
        request_id
        for event in usage_tracker.snapshot_usage_events()
        if isinstance(event, dict)
        for request_id in [event.get("request_id")]
        if isinstance(request_id, str) and request_id
    }
    with _REQUEST_PROMPT_LOCK:
        keep_ids.update(_REQUEST_PROMPT_ACTIVE_IDS)
    return keep_ids


def _prune_request_prompt_archive(request_ids: set[str] | None = None) -> None:
    archive_dir = request_prompt_archive_dir()
    if not os.path.isdir(archive_dir):
        return

    keep_ids = set(request_ids or _recent_request_prompt_ids())
    keep_files = {
        filename
        for request_id in keep_ids
        if (filename := _request_prompt_file_name(request_id)) is not None
    }
    try:
        with _REQUEST_PROMPT_LOCK:
            for entry in os.listdir(archive_dir):
                if not entry.startswith(_REQUEST_PROMPT_FILE_PREFIX) or not entry.endswith(".json"):
                    continue
                if entry in keep_files:
                    continue
                try:
                    os.unlink(os.path.join(archive_dir, entry))
                except OSError:
                    continue
    except OSError as exc:
        print(f"Warning: failed to prune request prompt archive: {exc}", file=sys.stderr, flush=True)


def _handle_usage_event_recorded(event: dict | None) -> None:
    request_id = event.get("request_id") if isinstance(event, dict) else None
    if isinstance(request_id, str) and request_id:
        with _REQUEST_PROMPT_LOCK:
            _REQUEST_PROMPT_ACTIVE_IDS.discard(request_id)
    _prune_request_prompt_archive()
    dashboard_service.notify_dashboard_stream_listeners()


usage_tracker = usage_tracking.UsageTracker(
    state=usage_tracking.UsageTrackingState(),
    archive_store=dashboard_module.create_usage_archive_store(),
    event_bus=usage_event_bus,
    on_request_finished=_initiator_policy.note_request_finished,
    on_usage_event_recorded=_handle_usage_event_recorded,
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


def _debug_prompt_logging_settings() -> dict[str, object]:
    try:
        settings = client_proxy_config_service.load_client_proxy_settings()
    except Exception:
        return {}
    return settings if isinstance(settings, dict) else {}


def _debug_prompt_logging_enabled() -> bool:
    return bool(_debug_prompt_logging_settings().get("debug_prompt_logging_enabled", False))


def _prompt_logging_permitted() -> bool:
    return _debug_prompt_logging_enabled()


def _prompt_trace_value(value):
    return value


def _prompt_payload_for_dashboard(value):
    return value


def _client_proxy_settings_with_trace_status(payload: dict[str, object]) -> dict[str, object]:
    return dict(payload)


def _save_client_proxy_settings(payload: dict) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")
    result = client_proxy_config_service.save_client_proxy_settings({
        "revert_on_shutdown": bool(payload.get("revert_on_shutdown", True)),
        "debug_prompt_logging_enabled": bool(payload.get("debug_prompt_logging_enabled", False)),
    })
    try:
        dashboard_service.notify_dashboard_stream_listeners()
    except NameError:
        pass
    return _client_proxy_settings_with_trace_status(result)


dashboard_service = dashboard_module.create_dashboard_service(
    dependencies=dashboard_module.DashboardDependencies(
        load_api_key_payload=auth.load_api_key_payload,
        snapshot_all_usage_events=usage_tracker.snapshot_all_usage_events,
        snapshot_usage_events=usage_tracker.snapshot_usage_events,
        load_safeguard_trigger_stats=safeguard_event_store.load_stats,
        prompt_payload=_prompt_payload_for_dashboard,
    ),
    utc_now=util.utc_now,
    utc_now_iso=util.utc_now_iso,
    thread_class=Thread,
)


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


_UPSTREAM_CLIENT: "httpx.AsyncClient | None" = None
_UPSTREAM_CLIENT_LOCK = threading.Lock()


def _get_upstream_client() -> "httpx.AsyncClient":
    global _UPSTREAM_CLIENT
    if _UPSTREAM_CLIENT is not None:
        return _UPSTREAM_CLIENT
    with _UPSTREAM_CLIENT_LOCK:
        if _UPSTREAM_CLIENT is not None:
            return _UPSTREAM_CLIENT
        timeout = httpx.Timeout(configured_upstream_timeout_seconds())
        limits = httpx.Limits(
            max_connections=1,
            max_keepalive_connections=1,
            keepalive_expiry=300.0,
        )
        try:
            _UPSTREAM_CLIENT = httpx.AsyncClient(http2=True, timeout=timeout, limits=limits)
        except (ImportError, RuntimeError):
            _UPSTREAM_CLIENT = httpx.AsyncClient(timeout=timeout, limits=limits)
        atexit.register(_shutdown_upstream_client)
    return _UPSTREAM_CLIENT


def _shutdown_upstream_client() -> None:
    global _UPSTREAM_CLIENT
    client = _UPSTREAM_CLIENT
    _UPSTREAM_CLIENT = None
    if client is None:
        return
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(client.aclose())
        finally:
            loop.close()
    except Exception:
        pass


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


usage_tracker.load_archived_history()
usage_tracker.load_history()
_initiator_policy.seed_from_usage_events(usage_tracker.snapshot_usage_events())
dashboard_module.initialize()

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


def _friendly_limit_message_from_upstream(upstream: httpx.Response) -> str | None:
    return upstream_errors.friendly_limit_message_from_upstream(upstream)


def _empty_openai_usage() -> dict:
    return protocol_replies.empty_openai_usage()


def _empty_anthropic_usage() -> dict:
    return protocol_replies.empty_anthropic_usage()


def _synthetic_reply_for_message(message: str) -> upstream_errors.SyntheticReply:
    return upstream_errors.SyntheticReply(
        status_for_trace=200,
        client_status=200,
        message=message,
        reason="compat",
        usage_shape="zero",
    )


def _friendly_limit_chat_payload(message: str, model: str | None = None) -> dict:
    return protocol_replies.chat_payload(message, model)


def _friendly_limit_responses_payload(message: str, model: str | None = None) -> dict:
    return protocol_replies.responses_payload(message, model)


def _friendly_limit_anthropic_payload(message: str, model: str | None = None) -> dict:
    return protocol_replies.anthropic_payload(message, model)


def _friendly_limit_payload_for_bridge(bridge_plan: BridgeExecutionPlan, message: str) -> dict:
    return protocol_replies.build_synthetic_payload(
        _synthetic_reply_for_message(message),
        protocol=bridge_plan.caller_protocol,
        model=bridge_plan.resolved_model or bridge_plan.requested_model,
        is_compact=bridge_plan.is_compact,
    )


def _friendly_limit_non_streaming_response(
    message: str,
    *,
    caller_protocol: str,
    model: str | None = None,
    is_compact: bool = False,
) -> JSONResponse:
    return protocol_replies.render_synthetic_reply(
        _synthetic_reply_for_message(message),
        protocol=caller_protocol,
        stream=False,
        model=model,
        is_compact=is_compact,
    )


async def _friendly_limit_responses_stream(message: str, model: str | None):
    async for chunk in protocol_replies._responses_stream(_synthetic_reply_for_message(message), model):
        yield chunk


async def _friendly_limit_chat_stream(message: str, model: str | None):
    async for chunk in protocol_replies._chat_stream(_synthetic_reply_for_message(message), model):
        yield chunk


async def _friendly_limit_anthropic_stream(message: str, model: str | None):
    async for chunk in protocol_replies._anthropic_stream(_synthetic_reply_for_message(message), model):
        yield chunk


def _friendly_limit_streaming_response(message: str, *, protocol: str, model: str | None = None) -> Response:
    return protocol_replies.render_synthetic_reply(
        _synthetic_reply_for_message(message),
        protocol=protocol,
        stream=True,
        model=model,
        streaming_response_class=GracefulStreamingResponse,
    )


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
    replay_subagent: str | None = None
    trace_context: dict | None = None
    debug_detail_session_key: str | None = None
    auto_update_request_tracked: bool = False


def _prompt_cache_settle_delay_seconds() -> float:
    raw_value = os.environ.get("GHCP_PROXY_RESPONSES_CACHE_SETTLE_DELAY_SECONDS")
    if raw_value is None:
        return CACHE_SETTLE_DELAY_SECONDS
    try:
        return max(0.0, float(str(raw_value).strip()))
    except (TypeError, ValueError):
        return CACHE_SETTLE_DELAY_SECONDS


def _responses_plan_header_value(
    plan: "UpstreamRequestPlan | None",
    header_name: str,
) -> str | None:
    if not isinstance(plan, UpstreamRequestPlan):
        return None
    headers = plan.headers if isinstance(plan.headers, dict) else None
    if not headers:
        return None
    wanted = header_name.lower()
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == wanted:
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _responses_plan_lineage(plan: "UpstreamRequestPlan | None") -> str | None:
    if not isinstance(plan, UpstreamRequestPlan):
        return None
    agent_task_id = _responses_plan_header_value(plan, "x-agent-task-id")
    if agent_task_id:
        return agent_task_id
    body = plan.body if isinstance(plan.body, dict) else None
    if isinstance(body, dict):
        pck = body.get("prompt_cache_key") or body.get("promptCacheKey")
        if isinstance(pck, str):
            normalized = pck.strip()
            if len(normalized) >= 36 and normalized[8:9] == "-":
                return normalized
    return None


def _responses_cache_settle_identity(
    plan: "UpstreamRequestPlan | None",
) -> tuple[str, str, str] | None:
    if not isinstance(plan, UpstreamRequestPlan) or not isinstance(plan.body, dict):
        return None
    model = str(
        plan.resolved_model or plan.requested_model or plan.body.get("model") or ""
    ).strip().lower()
    if model != "gpt-5.5":
        return None
    lineage = _responses_plan_lineage(plan)
    if not lineage:
        return None
    parent_task = _responses_plan_header_value(plan, "x-parent-agent-id")
    family_root = parent_task or lineage
    return model, lineage, f"family:{family_root}"


async def _wait_for_responses_cache_settle(plan: "UpstreamRequestPlan | None") -> None:
    identity = _responses_cache_settle_identity(plan)
    if identity is None:
        return
    model, lineage, family = identity
    delay_seconds = _prompt_cache_settle_delay_seconds()
    if delay_seconds <= 0:
        return
    with _PROMPT_CACHE_SETTLE_LOCK:
        last = _PROMPT_CACHE_LAST_FINISH_BY_FAMILY.get((model, family))
    if not last:
        return
    last_lineage, last_finished_at = last
    if last_lineage == lineage:
        return
    wait_seconds = delay_seconds - (time.monotonic() - float(last_finished_at))
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)


def _remember_responses_cache_settle_finish(
    plan: "UpstreamRequestPlan | None",
    status_code: int,
) -> None:
    if status_code >= 400:
        return
    identity = _responses_cache_settle_identity(plan)
    if identity is None:
        return
    model, lineage, family = identity
    now = time.monotonic()
    with _PROMPT_CACHE_SETTLE_LOCK:
        _PROMPT_CACHE_LAST_FINISH_BY_FAMILY[(model, family)] = (lineage, now)


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


def request_body_dump_enabled() -> bool:
    # Body dumps are still gated by debug_prompt_logging_enabled; this flag only
    # controls whether approved full-detail captures are written.
    return _env_flag_default("GHCP_DUMP_REQUEST_BODIES", default=True)


def request_body_dump_dir() -> str:
    configured = str(os.environ.get("GHCP_REQUEST_BODY_DUMP_DIR", "")).strip()
    if configured:
        return os.path.expanduser(configured)
    return os.path.join(os.path.dirname(request_trace_log_path()), "request-bodies")


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


def _trace_canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


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
            sequence.append({"index": index, "type": type(item).__name__, "item_hash": _trace_hash(item)})
            continue
        entry = {
            "index": index,
            "type": item.get("type"),
            "item_hash": _trace_hash(item),
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
            entry["content_hash"] = _trace_hash(item.get("content"))
        if "output" in item:
            entry["output_chars"] = _trace_text_chars(item.get("output"))
            entry["output_hash"] = _trace_hash(item.get("output"))
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
    for key in sorted(body.keys()):
        if key in ("input", "messages"):
            continue
        fingerprint = _trace_hash(body.get(key))
        if fingerprint:
            summary[f"{key}_fingerprint"] = fingerprint

    return summary


def _debug_detail_normalized_string(value) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _debug_detail_body_session_id(body: dict | None) -> str | None:
    return usage_tracking.request_body_session_id(body)


def _debug_detail_header_value(headers: dict | None, header_name: str) -> str | None:
    return _debug_detail_normalized_string(_header_value_case_insensitive(headers, header_name))


def _debug_detail_session_key(
    *,
    request: Request | None = None,
    request_body: dict | None = None,
    upstream_body: dict | None = None,
    resolved_model: str | None = None,
    outbound_headers: dict | None = None,
) -> tuple[str, str] | None:
    """Resolve the session bucket used for debug prompt logging."""

    if request is not None:
        session_id = usage_tracking.request_session_id(
            request,
            request_body if isinstance(request_body, dict) else upstream_body,
        )
        if session_id:
            return f"session:{session_id}", "request_session_id"

    for body in (request_body, upstream_body):
        session_id = _debug_detail_body_session_id(body)
        if session_id:
            return f"session:{session_id}", "body_session_id"

    for header_name, source in (
        ("session_id", "header_session_id"),
        ("session-id", "header_session_id"),
        ("x-claude-code-session-id", "header_session_id"),
        ("x-session-affinity", "header_session_id"),
        ("x-opencode-session", "header_session_id"),
        ("x-client-request-id", "header_client_request_id"),
        ("x-client-session-id", "header_client_session_id"),
        ("x-interaction-id", "header_interaction_id"),
        ("x-parent-agent-id", "header_parent_agent_id"),
        ("x-agent-task-id", "header_agent_task_id"),
    ):
        value = _debug_detail_header_value(outbound_headers, header_name)
        if value:
            return f"{source}:{value}", source

    return None


def _debug_detail_session_buffer_locked(session_key: str) -> deque[dict]:
    buffer = _DEBUG_DETAIL_SESSION_RECENT_REQUESTS.get(session_key)
    if buffer is None:
        buffer = deque(maxlen=DEBUG_DETAIL_CONTEXT_REQUESTS)
        _DEBUG_DETAIL_SESSION_RECENT_REQUESTS[session_key] = buffer
    else:
        _DEBUG_DETAIL_SESSION_RECENT_REQUESTS.move_to_end(session_key)
    return buffer


def _evict_debug_detail_sessions_locked() -> None:
    while len(_DEBUG_DETAIL_SESSION_RECENT_REQUESTS) > DEBUG_DETAIL_SESSION_BUFFER_LIMIT:
        evicted_session_key, _ = _DEBUG_DETAIL_SESSION_RECENT_REQUESTS.popitem(last=False)
        _DEBUG_DETAIL_SESSION_CAPTURED_REQUEST_IDS.pop(evicted_session_key, None)
        for request_id, snapshot in list(_DEBUG_DETAIL_REQUEST_SNAPSHOTS_BY_ID.items()):
            if snapshot.get("_session_key") == evicted_session_key:
                _DEBUG_DETAIL_REQUEST_SNAPSHOTS_BY_ID.pop(request_id, None)


def _remember_debug_detail_snapshot_locked(snapshot: dict) -> None:
    request_id = _debug_detail_normalized_string(snapshot.get("request_id"))
    if not request_id:
        return
    _DEBUG_DETAIL_REQUEST_SNAPSHOTS_BY_ID.pop(request_id, None)
    _DEBUG_DETAIL_REQUEST_SNAPSHOTS_BY_ID[request_id] = snapshot
    while len(_DEBUG_DETAIL_REQUEST_SNAPSHOTS_BY_ID) > _DEBUG_DETAIL_REQUEST_SNAPSHOT_INDEX_MAXLEN:
        _DEBUG_DETAIL_REQUEST_SNAPSHOTS_BY_ID.popitem(last=False)


def _debug_detail_after_snapshots_locked(session_key: str, buster_snapshot: dict) -> list[dict]:
    buster_sequence = buster_snapshot.get("_debug_detail_sequence")
    if not isinstance(buster_sequence, int):
        return []
    after_snapshots = [
        snapshot
        for snapshot in _DEBUG_DETAIL_REQUEST_SNAPSHOTS_BY_ID.values()
        if snapshot.get("_session_key") == session_key
        and isinstance(snapshot.get("_debug_detail_sequence"), int)
        and snapshot.get("_debug_detail_sequence") > buster_sequence
    ]
    after_snapshots.sort(key=lambda snapshot: snapshot.get("_debug_detail_sequence", 0))
    return after_snapshots[:DEBUG_DETAIL_CONTEXT_REQUESTS]


def _debug_detail_session_captured_ids_locked(session_key: str) -> set[str]:
    captured = _DEBUG_DETAIL_SESSION_CAPTURED_REQUEST_IDS.get(session_key)
    if captured is None:
        captured = set()
        _DEBUG_DETAIL_SESSION_CAPTURED_REQUEST_IDS[session_key] = captured
    else:
        _DEBUG_DETAIL_SESSION_CAPTURED_REQUEST_IDS.move_to_end(session_key)
    return captured


def _debug_detail_capture_slots_remaining_locked(session_key: str) -> int:
    captured = _debug_detail_session_captured_ids_locked(session_key)
    return max(0, DEBUG_DETAIL_SESSION_DETAIL_LIMIT - len(captured))


def _claim_debug_detail_capture_locked(session_key: str, snapshot: dict) -> bool:
    request_id = _debug_detail_normalized_string(snapshot.get("request_id"))
    if not request_id:
        return False
    captured = _debug_detail_session_captured_ids_locked(session_key)
    if request_id in captured:
        return False
    if len(captured) >= DEBUG_DETAIL_SESSION_DETAIL_LIMIT:
        return False
    captured.add(request_id)
    return True


def _header_value_case_insensitive(headers: dict | None, name: str) -> str | None:
    if not isinstance(headers, dict):
        return None
    target = str(name).lower()
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == target and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _trace_metadata_verdict(trace_metadata: dict | None) -> dict:
    if not isinstance(trace_metadata, dict):
        return {}
    verdict = trace_metadata.get("initiator_verdict")
    return dict(verdict) if isinstance(verdict, dict) else {}


def _debug_detail_always_capture_reasons(
    outbound_headers: dict | None,
    trace_metadata: dict | None,
) -> list[str]:
    if not _prompt_logging_permitted():
        return []
    reasons: list[str] = []
    initiator = str(_header_value_case_insensitive(outbound_headers, "x-initiator") or "").strip().lower()
    verdict = _trace_metadata_verdict(trace_metadata)
    resolved_initiator = str(verdict.get("resolved_initiator") or "").strip().lower()
    if initiator == "user" or resolved_initiator == "user":
        reasons.append("user_initiated")
    safeguard_reason = verdict.get("safeguard_reason")
    if isinstance(safeguard_reason, str) and safeguard_reason.strip():
        reasons.append("safeguarded")
    return reasons


def _debug_detail_capture_info(
    *,
    reasons: list[str],
    phase: str | None = None,
    incident_id: str | None = None,
    context_window: int = DEBUG_DETAIL_CONTEXT_REQUESTS,
) -> dict:
    info = {
        "enabled": True,
        "reasons": list(dict.fromkeys(reason for reason in reasons if reason)),
        "context_window": context_window,
    }
    if phase:
        info["phase"] = phase
    if incident_id:
        info["incident_id"] = incident_id
    return info


def _with_debug_detail_capture_info(
    snapshot: dict,
    *,
    reasons: list[str],
    phase: str | None = None,
    incident_id: str | None = None,
) -> dict:
    event = dict(snapshot)
    for key in list(event.keys()):
        if isinstance(key, str) and key.startswith("_"):
            event.pop(key, None)
    event["debug_detail_capture"] = _debug_detail_capture_info(
        reasons=reasons,
        phase=phase,
        incident_id=incident_id,
    )
    return event


def _outbound_json_wire_bytes(body: dict | None) -> bytes | None:
    if body is None:
        return None
    try:
        return json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
            default=util._json_default,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None


def _build_debug_detail_snapshot(
    *,
    request_id: str,
    context: dict,
    request: Request,
    requested_model: str | None,
    resolved_model: str | None,
    request_body: dict | None,
    upstream_body: dict | None,
    outbound_headers: dict | None,
) -> dict:
    full_prompt_preview = _extract_prompt_preview(
        request_body if isinstance(request_body, dict) else upstream_body,
        truncate=False,
    )
    session_key_pair = _debug_detail_session_key(
        request=request,
        request_body=request_body,
        upstream_body=upstream_body,
        resolved_model=resolved_model,
        outbound_headers=outbound_headers,
    )
    snapshot = {
        "event": "request_debug_detail",
        "time": util.utc_now_iso(),
        "request_id": request_id,
        "client_path": context.get("client_path") or getattr(getattr(request, "url", None), "path", None),
        "upstream_host": context.get("upstream_host"),
        "upstream_path": context.get("upstream_path"),
        "method": getattr(request, "method", None),
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "request_body_summary": _trace_body_summary(request_body),
        "upstream_body_summary": _trace_body_summary(upstream_body),
        "outbound_headers": _header_trace_subset(outbound_headers),
    }
    if session_key_pair is not None:
        snapshot["_session_key"], snapshot["_session_key_source"] = session_key_pair
    if full_prompt_preview:
        snapshot["request_prompt"] = _prompt_trace_value(full_prompt_preview)
    if isinstance(request_body, dict):
        snapshot["source_body"] = _prompt_trace_value(request_body)
    if isinstance(upstream_body, dict):
        snapshot["upstream_body"] = _prompt_trace_value(upstream_body)
        upstream_wire_bytes = _outbound_json_wire_bytes(upstream_body)
        if upstream_wire_bytes is not None:
            snapshot["upstream_body_wire"] = _prompt_trace_value(
                upstream_wire_bytes.decode("utf-8", errors="replace")
            )
            snapshot["upstream_body_wire_size"] = len(upstream_wire_bytes)
            snapshot["upstream_body_wire_sha256"] = hashlib.sha256(upstream_wire_bytes).hexdigest()
    return snapshot


def _register_debug_detail_snapshot(snapshot: dict) -> tuple[dict | None, list[dict]]:
    """Persist full prompt/body detail for every request when debug prompt logging is enabled."""
    if not _debug_prompt_logging_enabled():
        return None, []
    return _debug_detail_capture_info(reasons=["debug_prompt_logging"], phase="current"), []


def _trace_context_allows_full_debug_detail(trace_context: dict | None) -> bool:
    if not isinstance(trace_context, dict):
        return False
    capture = trace_context.get("debug_detail_capture")
    return isinstance(capture, dict) and capture.get("enabled") is True


def _plan_allows_full_debug_detail(plan: "UpstreamRequestPlan | None") -> bool:
    if not isinstance(plan, UpstreamRequestPlan):
        return False
    if not _prompt_logging_permitted():
        return False
    if _trace_context_allows_full_debug_detail(plan.trace_context):
        return True
    if "user_initiated" in _debug_detail_always_capture_reasons(plan.headers, plan.trace_context):
        return True
    if "safeguarded" in _debug_detail_always_capture_reasons(plan.headers, plan.trace_context):
        return True
    return False


def _effective_trace_usage(response_payload: dict | None = None, usage: dict | None = None) -> dict | None:
    normalized_usage = util.normalize_usage_payload(usage)
    if isinstance(normalized_usage, dict):
        return normalized_usage
    if isinstance(response_payload, dict):
        normalized_usage = util.normalize_usage_payload(response_payload.get("usage"))
        if isinstance(normalized_usage, dict):
            return normalized_usage
    return None


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

    normalized_usage = _effective_trace_usage(response_payload=response_payload, usage=usage)
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
    if not force and not (request_tracing_enabled() or _debug_prompt_logging_enabled()):
        return
    trace_path = request_trace_log_path()
    try:
        line = json.dumps(payload, separators=(",", ":"), default=util._json_default) + "\n"
        executor = _get_request_trace_executor()
        executor.submit(_write_request_trace_line, trace_path, line)
    except Exception as exc:
        print(f"Warning: failed to schedule request trace log write: {exc}", file=sys.stderr, flush=True)


def _write_request_trace_line(trace_path: str, line: str) -> None:
    try:
        log_dir = os.path.dirname(trace_path) or TOKEN_DIR
        os.makedirs(log_dir, exist_ok=True)
        with _REQUEST_TRACE_LOCK:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(line)
            _enforce_trace_retention_locked(trace_path)
    except OSError as exc:
        print(f"Warning: failed to write request trace log: {exc}", file=sys.stderr, flush=True)


def _trim_trace_field(value, *, max_bytes: int = REQUEST_TRACE_BODY_MAX_BYTES):
    """Cap body-ish trace fields so retained rows stay bounded in size."""
    if value is None or max_bytes <= 0:
        return value
    try:
        serialized = json.dumps(value, separators=(",", ":"), default=util._json_default)
    except (TypeError, ValueError):
        return value
    encoded = serialized.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return value
    return {
        "_truncated": True,
        "original_bytes": len(encoded),
        "preview": encoded[:max_bytes].decode("utf-8", errors="replace"),
        "original_type": type(value).__name__,
    }


def _trim_trace_text(value, *, max_chars: int = REQUEST_TRACE_BODY_MAX_BYTES):
    if not isinstance(value, str) or max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n...[truncated; original {len(value)} chars]"


def _is_response_completed_event(event_name: str | None, data: str | None) -> bool:
    if str(event_name or "").strip().lower() == "response.completed":
        return True
    if not data or data == "[DONE]":
        return False
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and str(payload.get("type") or "").strip().lower() == "response.completed"


def _enforce_body_dump_retention_locked(dump_dir: str) -> None:
    """Cap body-dump directory at REQUEST_TRACE_HISTORY_LIMIT files."""
    limit = REQUEST_TRACE_HISTORY_LIMIT
    if limit <= 0:
        return
    try:
        entries = os.listdir(dump_dir)
    except OSError:
        return
    if len(entries) <= limit + max(REQUEST_TRACE_RETENTION_SLACK, 0):
        return
    paths = []
    for name in entries:
        full = os.path.join(dump_dir, name)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            continue
        paths.append((mtime, full))
    paths.sort()
    for _, path in paths[: max(0, len(paths) - limit)]:
        try:
            os.unlink(path)
        except OSError:
            pass


def _enforce_trace_retention_locked(trace_path: str) -> None:
    """Keep the trace log bounded at REQUEST_TRACE_HISTORY_LIMIT rows."""
    limit = REQUEST_TRACE_HISTORY_LIMIT
    if limit <= 0:
        return
    threshold = limit + max(REQUEST_TRACE_RETENTION_SLACK, 0)
    try:
        size = os.path.getsize(trace_path)
    except OSError:
        return
    if size < threshold * 256:
        return
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    if len(lines) <= threshold:
        return
    try:
        with open(trace_path, "w", encoding="utf-8") as f:
            f.writelines(lines[-limit:])
    except OSError as exc:
        print(f"Warning: trace retention rewrite failed: {exc}", file=sys.stderr, flush=True)


def _anthropic_messages_usage_for_tracking(usage: dict | None) -> dict | None:
    if not isinstance(usage, dict):
        return None
    raw_input = util._coerce_int(usage.get("input_tokens"), default=0)
    cache_read = util._coerce_int(usage.get("cache_read_input_tokens"), default=None)
    if cache_read is None:
        cache_read = util._coerce_int(usage.get("cached_input_tokens"), default=0)
    cache_creation = util._coerce_int(usage.get("cache_creation_input_tokens"), default=0)
    non_cache_read_input = raw_input + cache_creation
    tracked = dict(usage)
    tracked["input_tokens"] = non_cache_read_input
    tracked["cached_input_tokens"] = cache_read
    tracked["cache_read_input_tokens"] = cache_read
    tracked["cache_creation_input_tokens"] = cache_creation
    tracked.pop("fresh_input_tokens", None)
    tracked["pricing_fresh_input_tokens"] = raw_input
    tracked["pricing_cached_input_tokens"] = cache_read
    tracked["pricing_cache_creation_input_tokens"] = cache_creation
    cache_creation_detail = tracked.get("cache_creation")
    if isinstance(cache_creation_detail, dict):
        tracked["cache_creation"] = dict(cache_creation_detail)
    output_tokens = util._coerce_int(usage.get("output_tokens"), default=None)
    if output_tokens is not None:
        tracked["total_tokens"] = non_cache_read_input + output_tokens
    return tracked


def _anthropic_messages_usage_for_client(usage: dict | None) -> tuple[dict | None, bool]:
    if not isinstance(usage, dict):
        return None, False
    raw_input = util._coerce_int(usage.get("input_tokens"), default=0)
    cache_read = util._coerce_int(usage.get("cache_read_input_tokens"), default=None)
    if cache_read is None:
        cache_read = util._coerce_int(usage.get("cached_input_tokens"), default=0)
    cache_creation = util._coerce_int(usage.get("cache_creation_input_tokens"), default=0)
    non_cache_read_input = raw_input + cache_creation
    if non_cache_read_input == raw_input and cache_read == 0:
        return usage, False
    client_usage = dict(usage)
    client_usage["input_tokens"] = non_cache_read_input
    client_usage["cache_read_input_tokens"] = cache_read
    client_usage["cache_creation_input_tokens"] = cache_creation
    client_usage["cached_input_tokens"] = cache_read
    output_tokens = util._coerce_int(usage.get("output_tokens"), default=None)
    if output_tokens is not None:
        client_usage["total_tokens"] = client_usage["input_tokens"] + output_tokens
    return client_usage, True


def _anthropic_messages_payload_for_client(payload: dict | None) -> tuple[dict | None, bool]:
    if not isinstance(payload, dict):
        return payload, False
    client_usage, changed = _anthropic_messages_usage_for_client(payload.get("usage"))
    if not changed:
        return payload, False
    client_payload = dict(payload)
    client_payload["usage"] = client_usage
    return client_payload, True


_REQUEST_BODY_DUMP_LOCK = threading.Lock()
_REQUEST_BODY_DUMP_EXECUTOR: "concurrent.futures.ThreadPoolExecutor | None" = None
_REQUEST_BODY_DUMP_EXECUTOR_LOCK = threading.Lock()
_REQUEST_TRACE_EXECUTOR: "concurrent.futures.ThreadPoolExecutor | None" = None
_REQUEST_TRACE_EXECUTOR_LOCK = threading.Lock()


def _get_request_trace_executor() -> "concurrent.futures.ThreadPoolExecutor":
    global _REQUEST_TRACE_EXECUTOR
    if _REQUEST_TRACE_EXECUTOR is not None:
        return _REQUEST_TRACE_EXECUTOR
    with _REQUEST_TRACE_EXECUTOR_LOCK:
        if _REQUEST_TRACE_EXECUTOR is None:
            import concurrent.futures
            _REQUEST_TRACE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ghcp-trace"
            )
    return _REQUEST_TRACE_EXECUTOR


def _get_request_body_dump_executor() -> "concurrent.futures.ThreadPoolExecutor":
    global _REQUEST_BODY_DUMP_EXECUTOR
    if _REQUEST_BODY_DUMP_EXECUTOR is not None:
        return _REQUEST_BODY_DUMP_EXECUTOR
    with _REQUEST_BODY_DUMP_EXECUTOR_LOCK:
        if _REQUEST_BODY_DUMP_EXECUTOR is None:
            import concurrent.futures
            _REQUEST_BODY_DUMP_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="ghcp-body-dump"
            )
    return _REQUEST_BODY_DUMP_EXECUTOR


def _dump_outbound_request_body(
    *,
    request_id: str,
    context: dict,
    request: Request,
    requested_model: str | None,
    resolved_model: str | None,
    request_body: dict | None,
    upstream_body: dict | None,
    outbound_headers: dict | None,
) -> None:
    """Persist the exact outbound body and full headers for an approved capture."""
    if not request_body_dump_enabled():
        return
    try:
        dump_dir = request_body_dump_dir()
        # Build a snapshot on the caller's thread so we don't race the request
        # handler mutating the body after we hand off; serialization happens
        # in the background thread.
        upstream_wire_bytes = _outbound_json_wire_bytes(upstream_body)
        snapshot = {
            "request_id": request_id,
            "time": util.utc_now_iso(),
            "method": request.method,
            "client_path": context.get("client_path"),
            "upstream_host": context.get("upstream_host"),
            "upstream_path": context.get("upstream_path"),
            "requested_model": requested_model,
            "resolved_model": resolved_model,
            "outbound_headers": dict(outbound_headers) if isinstance(outbound_headers, dict) else None,
            "request_body": _prompt_trace_value(request_body),
            "upstream_body": _prompt_trace_value(upstream_body),
        }
        if upstream_wire_bytes is not None:
            snapshot["upstream_body_wire"] = _prompt_trace_value(
                upstream_wire_bytes.decode("utf-8", errors="replace")
            )
            snapshot["upstream_body_wire_size"] = len(upstream_wire_bytes)
            snapshot["upstream_body_wire_sha256"] = hashlib.sha256(upstream_wire_bytes).hexdigest()
        safe_rid = "".join(ch for ch in str(request_id) if ch.isalnum() or ch in ("-", "_")) or "request"
        out_path = os.path.join(dump_dir, f"{safe_rid}.json")
        executor = _get_request_body_dump_executor()
        executor.submit(_write_request_body_dump, out_path, dump_dir, snapshot)
    except Exception as exc:  # pragma: no cover - never let dump errors fail upstream
        print(f"Warning: failed to schedule request body dump: {exc}", file=sys.stderr, flush=True)


def _write_request_body_dump(out_path: str, dump_dir: str, snapshot: dict) -> None:
    """Background worker: serialize the snapshot and persist it.

    Runs on the body-dump executor so the event loop is never blocked by
    disk I/O. Catches every exception so a malformed payload cannot leak
    out of the worker.
    """
    try:
        with _REQUEST_BODY_DUMP_LOCK:
            os.makedirs(dump_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, default=util._json_default)
            _enforce_body_dump_retention_locked(dump_dir)
    except Exception as exc:  # pragma: no cover - dump must never raise
        print(f"Warning: failed to write request body dump: {exc}", file=sys.stderr, flush=True)


def _protect_plan_prompt_trace_state(plan: UpstreamRequestPlan) -> None:
    return None


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
    trace_details = dict(trace_metadata) if isinstance(trace_metadata, dict) else {}
    debug_detail_snapshot = _build_debug_detail_snapshot(
        request_id=request_id,
        context=context,
        request=request,
        requested_model=requested_model,
        resolved_model=resolved_model,
        request_body=request_body,
        upstream_body=upstream_body,
        outbound_headers=outbound_headers,
    )
    debug_detail_session_key = _debug_detail_normalized_string(debug_detail_snapshot.get("_session_key"))
    debug_detail_capture, debug_detail_events = _register_debug_detail_snapshot(debug_detail_snapshot)
    if debug_detail_capture is not None:
        context["debug_detail_capture"] = debug_detail_capture
        if "request_prompt" in debug_detail_snapshot:
            context["request_prompt"] = debug_detail_snapshot["request_prompt"]
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
        "trace": trace_details,
    }
    if debug_detail_capture is not None and prompt_preview is None:
        prompt_preview = _extract_prompt_preview(
            request_body if isinstance(request_body, dict) else upstream_body,
            truncate=False,
        )
    if debug_detail_capture is not None and prompt_preview:
        protected_prompt_preview = _prompt_trace_value(prompt_preview)
        payload["request_prompt"] = protected_prompt_preview
        context["request_prompt"] = protected_prompt_preview
    if debug_detail_capture is not None:
        if isinstance(request_body, dict):
            payload["source_body"] = _prompt_trace_value(_trim_trace_field(request_body))
        if isinstance(upstream_body, dict):
            payload["upstream_body"] = _prompt_trace_value(_trim_trace_field(upstream_body))
    if initiator_verdict is not None:
        payload["initiator_verdict"] = initiator_verdict
        context["initiator_verdict"] = initiator_verdict
    _append_request_trace(payload)
    for debug_detail_event in debug_detail_events:
        _append_request_trace(debug_detail_event)
    if debug_detail_capture is not None:
        _dump_outbound_request_body(
            request_id=request_id,
            context=context,
            request=request,
            requested_model=requested_model,
            resolved_model=resolved_model,
            request_body=request_body,
            upstream_body=upstream_body,
            outbound_headers=outbound_headers,
        )
    if debug_detail_session_key:
        context["_debug_detail_session_key"] = debug_detail_session_key
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
            _protect_plan_prompt_trace_state(plan)
            usage_tracker.finish_event(
                plan.usage_event,
                status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text=response_text,
                reasoning_text=reasoning_text,
                usage=usage,
            )
            effective_usage = _effective_trace_usage(response_payload=response_payload, usage=usage)
            force_trace = _should_force_failure_trace(plan, status_code)
            if request_tracing_enabled() or _debug_prompt_logging_enabled() or force_trace:
                trace_context = dict(plan.trace_context or {"request_id": plan.request_id})
                trace_context.pop("_debug_detail_session_key", None)
                trace_payload = {
                    "event": "request_finished",
                    "time": util.utc_now_iso(),
                    **trace_context,
                    "requested_model": plan.requested_model,
                    "resolved_model": plan.resolved_model,
                    "response": _trace_response_summary(
                        upstream=upstream,
                        response_payload=response_payload,
                        usage=effective_usage,
                    ),
                    "response_text_present": isinstance(response_text, str) and bool(response_text),
                    "reasoning_text_present": isinstance(reasoning_text, str) and bool(reasoning_text),
                }
                if isinstance(reasoning_text, str) and reasoning_text:
                    trace_payload["reasoning_text"] = _trim_trace_text(reasoning_text)
                if status_code >= 400:
                    if _plan_allows_full_debug_detail(plan):
                        trace_payload["source_body"] = _prompt_trace_value(
                            _trim_trace_field(plan.source_body if isinstance(plan.source_body, dict) else plan.body)
                        )
                        trace_payload["upstream_body"] = _prompt_trace_value(_trim_trace_field(plan.body))
                    else:
                        trace_payload["source_body"] = _trace_body_summary(
                            plan.source_body if isinstance(plan.source_body, dict) else plan.body
                        )
                        trace_payload["upstream_body"] = _trace_body_summary(plan.body)
                    trace_payload["outbound_headers"] = _header_trace_subset(plan.headers)
                    if isinstance(response_payload, dict):
                        trace_payload["response_payload"] = _trim_trace_field(response_payload)
                    if isinstance(response_text, str) and response_text:
                        trace_payload["response_text"] = _trim_trace_text(response_text)
                _append_request_trace(trace_payload, force=force_trace)
        finally:
            _remember_responses_cache_settle_finish(plan, status_code)
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
    replay_subagent: str | None = None,
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
    always_capture_reasons = _debug_detail_always_capture_reasons(headers, trace_metadata)
    prompt_preview = None
    stored_prompt_preview = None
    if always_capture_reasons:
        prompt_preview = _extract_prompt_preview(
            source_body if isinstance(source_body, dict) else body,
            truncate=False,
        )
        stored_prompt_preview = (
            _prompt_trace_value(prompt_preview)
            if prompt_preview
            else None
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
        prompt_preview=stored_prompt_preview,
        initiator_verdict=initiator_verdict if isinstance(initiator_verdict, dict) else None,
    )
    _save_request_prompt_record(
        request_id,
        request.url.path,
        source_body if isinstance(source_body, dict) else body,
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
    debug_detail_session_key = None
    if request_tracing_enabled() or _debug_prompt_logging_enabled():
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
            prompt_preview=stored_prompt_preview,
        )
        if isinstance(trace_context, dict):
            debug_detail_session_key = _debug_detail_normalized_string(
                trace_context.pop("_debug_detail_session_key", None)
            )
        if (
            isinstance(usage_event, dict)
            and "request_prompt" not in usage_event
            and isinstance(trace_context, dict)
            and "request_prompt" in trace_context
        ):
            usage_event["request_prompt"] = trace_context["request_prompt"]
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
            replay_subagent=(
                replay_subagent.strip()
                if isinstance(replay_subagent, str) and replay_subagent.strip()
                else None
            ),
            trace_context=trace_context,
            debug_detail_session_key=debug_detail_session_key,
            auto_update_request_tracked=True,
        ),
        None,
    )



def _bridge_error_response(plan: BridgeExecutionPlan):
    if plan.caller_protocol == "anthropic":
        return format_translation.anthropic_error_response
    return format_translation.openai_error_response


def _responses_effective_subagent(
    request: Request,
    body: dict | None,
    *,
    approval_agent: bool | None = None,
) -> str | None:
    """Return the inbound or synthesized worker identity for Responses state.

    The header builder intentionally removes ``x-openai-subagent`` before the
    Copilot request is sent. Keep its normalized value with the request plan so
    replay-ID observation and repair stay in the same worker namespace.  An
    approval prompt without the inbound header is synthesized as ``guardian``
    by the bridge, so replay state needs that same identity too.
    """
    inbound_subagent = (
        request.headers.get("x-openai-subagent")
        if hasattr(request, "headers")
        else None
    )
    if isinstance(inbound_subagent, str) and inbound_subagent.strip():
        return inbound_subagent.strip()
    if approval_agent is None:
        approval_agent = is_approval_agent_request(
            inbound_protocol="responses",
            body=body if isinstance(body, dict) else None,
        )
    return "guardian" if approval_agent else None


import request_headers as _request_headers_module
import responses_replay_ids


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

    interaction_id = usage_tracking.request_body_session_id(
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
        # Stable affinity when the caller supplies a durable session hint.
        stable_affinity_hint = isinstance(original_body, dict) and any(
            isinstance(original_body.get(k), str) and original_body.get(k).strip()
            for k in ("sessionId", "session_id")
        )
        headers = format_translation.build_responses_headers_for_request(
            request,
            bridge_plan.upstream_body,
            api_key,
            force_initiator=force_initiator,
            request_id=request_id,
            initiator_policy=_initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            verdict_sink=verdict_sink,
            affinity_body=original_body,
            stable_user_affinity=(
                bridge_plan.caller_protocol == "anthropic" or stable_affinity_hint
            ),
            synthetic_subagent="guardian" if bridge_plan.approval_agent else None,
        )
        return headers
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
            affinity_body=original_body,
            synthetic_subagent="guardian" if bridge_plan.approval_agent else None,
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
    replay_subagent = _responses_effective_subagent(
        request,
        original_body,
        approval_agent=bridge_plan.approval_agent,
    )
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
        replay_subagent=replay_subagent,
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
        # Native Anthropic passthrough: return upstream payload as-is.
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
        await _wait_for_responses_cache_settle(plan)
        client = _get_upstream_client()
        upstream = await throttled_client_post(
            client,
            plan.upstream_url,
            headers=plan.headers,
            json=plan.body,
        )
        if upstream.status_code >= 400:
            return _handle_upstream_error(
                upstream,
                trace_plan=plan,
                caller_protocol="chat",
                stream=False,
                model=plan.resolved_model or plan.requested_model,
                fallback_error_response=proxy_non_streaming_response,
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
        await _wait_for_responses_cache_settle(plan)
        client = _get_upstream_client()
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
        return _handle_upstream_error(
            upstream,
            trace_plan=plan,
            caller_protocol=bridge_plan.caller_protocol,
            stream=False,
            model=bridge_plan.resolved_model or bridge_plan.requested_model,
            is_compact=bridge_plan.is_compact,
            fallback_trace=(
                _anthropic_upstream_error_trace
                if bridge_plan.caller_protocol == "anthropic"
                else _default_upstream_error_trace
            ),
            fallback_error_response=lambda upstream_response: _bridge_error_response_from_upstream(
                bridge_plan,
                upstream_response,
            ),
        )

    upstream_payload = _extract_upstream_json_payload(upstream)
    if not isinstance(upstream_payload, dict):
        message = "Upstream response did not include a JSON object payload"
        _finish_usage_and_trace(plan, 502, upstream=upstream, response_text=message)
        return error_response(502, message)

    if bridge_plan.caller_protocol == "responses" and bridge_plan.upstream_protocol == "responses":
        _, replay_id_state = responses_replay_ids.state_for_body(
            plan.source_body,
            headers=plan.headers,
            subagent=plan.replay_subagent,
        )
        if replay_id_state is not None:
            replay_id_state.observe_response_payload(upstream_payload)

    translated_payload = _translate_bridge_success_payload(bridge_plan, upstream_payload)
    if bridge_plan.caller_protocol == "anthropic" and bridge_plan.upstream_protocol == "messages":
        translated_payload, _ = _anthropic_messages_payload_for_client(translated_payload)
    tracking_usage = (
        _anthropic_messages_usage_for_tracking(upstream_payload.get("usage"))
        if bridge_plan.upstream_protocol == "messages"
        and isinstance(upstream_payload.get("usage"), dict)
        else None
    )
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
        # When upstream is Anthropic Messages, derive tracking-shape usage from
        # the raw upstream usage regardless of caller protocol. The translated
        # payload's usage (Responses-shape for responses callers) loses cache
        # creation tokens and miscomputes ``fresh_input_tokens`` because the
        # Responses-shape input_tokens is gross prompt input with cache reads
        # moved into ``input_tokens_details.cached_tokens`` for the client.
        # Cache writes remain part of gross input so Codex does not undercount
        # a turn that created or refreshed a prompt-cache segment.
        usage=tracking_usage,
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


def _publish_synthetic_reply_event(
    reply: upstream_errors.SyntheticReply,
    trace_plan: UpstreamRequestPlan | None,
) -> None:
    if not reply.event_name:
        return
    payload = dict(reply.event_payload or {})
    if isinstance(trace_plan, UpstreamRequestPlan):
        payload.setdefault("request_id", trace_plan.request_id)
    try:
        usage_event_bus.publish(reply.event_name, payload)
    except Exception as exc:  # pragma: no cover - observers must not affect proxying
        print(f"Warning: synthetic upstream error event failed: {exc}", file=sys.stderr, flush=True)


def _default_upstream_error_trace(upstream: httpx.Response) -> tuple[dict | None, str | None]:
    return _extract_upstream_json_payload(upstream), _extract_upstream_text(upstream)


def _anthropic_upstream_error_trace(upstream: httpx.Response) -> tuple[dict | None, str | None]:
    fallback_message = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
    error_payload = format_translation.anthropic_error_payload_from_openai(
        _extract_upstream_json_payload(upstream),
        upstream.status_code,
        fallback_message,
    )
    return error_payload, error_payload.get("error", {}).get("message")


def _responses_error_trace_from_anthropic(upstream: httpx.Response) -> tuple[dict | None, str | None]:
    upstream_payload = _extract_upstream_json_payload(upstream)
    upstream_text = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
    err_message = upstream_text
    if isinstance(upstream_payload, dict):
        err_obj = upstream_payload.get("error")
        if isinstance(err_obj, dict) and isinstance(err_obj.get("message"), str):
            err_message = err_obj["message"]
    return upstream_payload, err_message


def _openai_error_response_from_anthropic(upstream: httpx.Response) -> Response:
    _, err_message = _responses_error_trace_from_anthropic(upstream)
    return format_translation.openai_error_response(
        upstream.status_code,
        err_message or f"Upstream request failed with status {upstream.status_code}",
    )


def _handle_upstream_error(
    upstream: httpx.Response,
    *,
    trace_plan: UpstreamRequestPlan | None,
    caller_protocol: str,
    stream: bool,
    model: str | None,
    fallback_error_response,
    fallback_trace=None,
    is_compact: bool = False,
) -> Response:
    synthetic = upstream_errors.translate(upstream)
    if synthetic is not None:
        response_payload = protocol_replies.build_synthetic_payload(
            synthetic,
            protocol=caller_protocol,
            model=model,
            is_compact=is_compact,
        )
        usage = (
            protocol_replies.empty_usage_for_protocol(caller_protocol)
            if synthetic.usage_shape == "zero"
            else None
        )
        _finish_usage_and_trace(
            trace_plan,
            synthetic.status_for_trace,
            upstream=upstream,
            response_payload=response_payload,
            response_text=synthetic.message,
            usage=usage,
        )
        _publish_synthetic_reply_event(synthetic, trace_plan)
        return protocol_replies.render_synthetic_reply(
            synthetic,
            protocol=caller_protocol,
            stream=stream,
            model=model,
            is_compact=is_compact,
            streaming_response_class=GracefulStreamingResponse,
        )

    trace_builder = fallback_trace or _default_upstream_error_trace
    response_payload, response_text = trace_builder(upstream)
    _finish_usage_and_trace(
        trace_plan,
        upstream.status_code,
        upstream=upstream,
        response_payload=response_payload,
        response_text=response_text,
    )
    return fallback_error_response(upstream)


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
    client = _get_upstream_client()
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        await _wait_for_responses_cache_settle(trace_plan)
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(trace_plan, status_code, response_text=message)
        return format_translation.openai_error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(trace_plan, 599)
        raise

    if upstream.status_code >= 400:
        await upstream.aread()
        if upstream.status_code >= 400:
            try:
                await upstream.aread()
                return _handle_upstream_error(
                    upstream,
                    trace_plan=trace_plan,
                    caller_protocol=stream_type,
                    stream=True,
                    model=trace_plan.resolved_model if isinstance(trace_plan, UpstreamRequestPlan) else None,
                    fallback_error_response=proxy_non_streaming_response,
                )
            finally:
                await upstream.aclose()

    response_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    content_type = upstream.headers.get("content-type")
    if content_type:
        response_headers["content-type"] = content_type

    async def stream_upstream():
        capture = usage_tracker.create_sse_capture(stream_type)
        replay_id_state = None
        if stream_type == "responses":
            replay_source_body = (
                trace_plan.source_body
                if isinstance(trace_plan, UpstreamRequestPlan)
                else body
            )
            replay_headers = (
                trace_plan.headers
                if isinstance(trace_plan, UpstreamRequestPlan)
                else headers
            )
            _, replay_id_state = responses_replay_ids.state_for_body(
                replay_source_body,
                headers=replay_headers,
                subagent=(
                    trace_plan.replay_subagent
                    if isinstance(trace_plan, UpstreamRequestPlan)
                    else None
                ),
            )
        source_iter = _stream_with_update_notice(upstream.aiter_bytes(), stream_type, getattr(upstream, "headers", None))
        if stream_type == "responses":
            source_iter = ResponsesStreamIdSyncer(replay_id_state).sync(source_iter)
        response_payload: dict | None = None
        finish_called = False
        try:
            if stream_type == "responses":
                text_buffer = ""
                terminal_blocks: list[bytes] = []
                holding_terminal = False
                output_index = 0
                content_index = 0

                async for chunk in source_iter:
                    if capture.feed(chunk):
                        usage_tracker.mark_first_output(usage_event)
                    if isinstance(chunk, bytes):
                        text_buffer += chunk.decode("utf-8", errors="replace")
                    else:
                        text_buffer += str(chunk)
                    normalized = text_buffer.replace("\r\n", "\n")
                    while "\n\n" in normalized:
                        raw_block, normalized = normalized.split("\n\n", 1)
                        if not raw_block.strip():
                            continue
                        event_name, data = format_translation.parse_sse_block(raw_block)
                        payload = None
                        if data and data != "[DONE]":
                            try:
                                payload = json.loads(data)
                            except json.JSONDecodeError:
                                payload = None
                        if isinstance(payload, dict):
                            event_type = str(payload.get("type") or event_name or "").strip().lower()
                            if event_type == "response.output_text.delta":
                                if isinstance(payload.get("output_index"), int):
                                    output_index = payload["output_index"]
                                if isinstance(payload.get("content_index"), int):
                                    content_index = payload["content_index"]
                        block = f"{raw_block}\n\n".encode("utf-8")
                        if holding_terminal or data == "[DONE]" or _is_response_completed_event(event_name, data):
                            holding_terminal = True
                            terminal_blocks.append(block)
                        else:
                            yield block
                    text_buffer = normalized

                if text_buffer:
                    trailing = text_buffer.encode("utf-8")
                    if holding_terminal:
                        terminal_blocks.append(trailing)
                    else:
                        yield trailing

                for block in terminal_blocks:
                    yield block
            else:
                async for chunk in source_iter:
                    if capture.feed(chunk):
                        usage_tracker.mark_first_output(usage_event)
                    yield chunk

            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                usage=capture.usage if isinstance(capture.usage, dict) else None,
            )
            finish_called = True
        finally:
            if not finish_called:
                _finish_usage_and_trace(
                    trace_plan,
                    upstream.status_code,
                    upstream=upstream,
                    usage=capture.usage if isinstance(capture.usage, dict) else None,
                )
            await upstream.aclose()

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
    client = _get_upstream_client()
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(trace_plan, status_code, response_text=message)
        return format_translation.anthropic_error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(trace_plan, 599)
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            return _handle_upstream_error(
                upstream,
                trace_plan=trace_plan,
                caller_protocol="anthropic",
                stream=True,
                model=fallback_model,
                fallback_trace=_anthropic_upstream_error_trace,
                fallback_error_response=format_translation.anthropic_error_response_from_upstream,
            )
        finally:
            await upstream.aclose()

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
    client = _get_upstream_client()
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(trace_plan, status_code, response_text=message)
        return format_translation.openai_error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(trace_plan, 599)
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            return _handle_upstream_error(
                upstream,
                trace_plan=trace_plan,
                caller_protocol="responses",
                stream=True,
                model=trace_plan.resolved_model if isinstance(trace_plan, UpstreamRequestPlan) else None,
                fallback_error_response=proxy_non_streaming_response,
            )
        finally:
            await upstream.aclose()

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
    client = _get_upstream_client()
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(trace_plan, status_code, response_text=message)
        return format_translation.anthropic_error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(trace_plan, 599)
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            return _handle_upstream_error(
                upstream,
                trace_plan=trace_plan,
                caller_protocol="anthropic",
                stream=True,
                model=fallback_model,
                fallback_trace=_anthropic_upstream_error_trace,
                fallback_error_response=format_translation.anthropic_error_response_from_upstream,
            )
        finally:
            await upstream.aclose()

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
    """Re-emit upstream Anthropic /v1/messages SSE to the client.

    The event payloads are otherwise passed through, but usage is normalized so
    cache writes are reflected in ``input_tokens`` for Claude Code's aggregate
    usage display. We still parse message_start / message_delta usage for the
    local dashboard.
    """
    client = _get_upstream_client()
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(trace_plan, status_code, response_text=message)
        return format_translation.anthropic_error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(trace_plan, 599)
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            return _handle_upstream_error(
                upstream,
                trace_plan=trace_plan,
                caller_protocol="anthropic",
                stream=True,
                model=fallback_model,
                fallback_trace=_anthropic_upstream_error_trace,
                fallback_error_response=format_translation.anthropic_error_response_from_upstream,
            )
        finally:
            await upstream.aclose()

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

        output_tokens = read_int("output_tokens")
        if output_tokens is not None:
            usage_state["output_tokens"] = output_tokens

        cache_read = read_int("cache_read_input_tokens")
        if cache_read is None:
            cache_read = read_int("cached_input_tokens")
        if cache_read is not None:
            usage_state["pricing_cached_input_tokens"] = cache_read
            usage_state["cached_input_tokens"] = cache_read
            usage_state["cache_read_input_tokens"] = cache_read

        cache_creation = read_int("cache_creation_input_tokens")
        if cache_creation is not None:
            usage_state["pricing_cache_creation_input_tokens"] = cache_creation
            usage_state["cache_creation_input_tokens"] = cache_creation

        input_tokens = read_int("input_tokens")
        if input_tokens is not None:
            usage_state["pricing_fresh_input_tokens"] = input_tokens

        pricing_fresh = int(usage_state.get("pricing_fresh_input_tokens", 0) or 0)
        pricing_cache_creation = int(usage_state.get("pricing_cache_creation_input_tokens", 0) or 0)
        usage_state["input_tokens"] = pricing_fresh + pricing_cache_creation
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
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "pricing_fresh_input_tokens": 0,
            "pricing_cached_input_tokens": 0,
            "pricing_cache_creation_input_tokens": 0,
        }
        first_output_marked = False
        buffer = ""
        try:
            async for chunk in _stream_with_update_notice(upstream.aiter_bytes(), "anthropic", getattr(upstream, "headers", None)):
                if not chunk:
                    continue
                try:
                    text = chunk.decode("utf-8", errors="replace")
                except Exception:
                    text = ""
                buffer += text
                normalized = buffer.replace("\r\n", "\n")
                while "\n\n" in normalized:
                    raw_block, normalized = normalized.split("\n\n", 1)
                    event_name, data = format_translation.parse_sse_block(raw_block)
                    emit_block = (raw_block + "\n\n").encode("utf-8")
                    if data:
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            payload = None
                        evt = (event_name or payload.get("type") if isinstance(payload, dict) else event_name) or ""
                        evt = str(evt).lower()
                        client_payload = payload
                        client_usage_changed = False
                        if evt == "message_start" and isinstance(payload, dict):
                            message = payload.get("message")
                            if isinstance(message, dict) and isinstance(message.get("usage"), dict):
                                raw_usage = message["usage"]
                                merge_anthropic_usage(usage_state, raw_usage)
                                client_usage, client_usage_changed = _anthropic_messages_usage_for_client(raw_usage)
                                if client_usage_changed:
                                    client_payload = dict(payload)
                                    client_message = dict(message)
                                    client_message["usage"] = client_usage
                                    client_payload["message"] = client_message
                        elif evt == "message_delta" and isinstance(payload, dict):
                            u = payload.get("usage")
                            if isinstance(u, dict):
                                merge_anthropic_usage(usage_state, u)
                                client_usage, client_usage_changed = _anthropic_messages_usage_for_client(u)
                                if client_usage_changed:
                                    client_payload = dict(payload)
                                    client_payload["usage"] = client_usage
                        elif evt in ("content_block_delta", "content_block_start"):
                            if not first_output_marked:
                                first_output_marked = True
                                usage_tracker.mark_first_output(usage_event)
                        if client_usage_changed and isinstance(client_payload, dict):
                            emit_block = format_translation.sse_encode(event_name or evt, client_payload)
                    yield emit_block
                buffer = normalized
            if buffer:
                yield buffer.encode("utf-8")
        finally:
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                usage=usage_state,
            )
            await upstream.aclose()

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
    client = _get_upstream_client()
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        _finish_usage_and_trace(trace_plan, status_code, response_text=message)
        return format_translation.openai_error_response(status_code, message)
    except Exception:
        _finish_usage_and_trace(trace_plan, 599)
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            return _handle_upstream_error(
                upstream,
                trace_plan=trace_plan,
                caller_protocol="responses",
                stream=True,
                model=trace_plan.resolved_model if isinstance(trace_plan, UpstreamRequestPlan) else fallback_model,
                fallback_trace=_responses_error_trace_from_anthropic,
                fallback_error_response=_openai_error_response_from_anthropic,
            )
        finally:
            await upstream.aclose()

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
            raw_anthropic_usage = translator.anthropic_raw_usage
            tracking_usage = (
                _anthropic_messages_usage_for_tracking(raw_anthropic_usage)
                if raw_anthropic_usage
                else response_payload.get("usage")
            )
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text=translator.response_text,
                reasoning_text=translator.reasoning_text,
                usage=tracking_usage,
            )
            await upstream.aclose()

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
        picker_enabled = entry.get("model_picker_enabled")
        if isinstance(picker_enabled, bool):
            enriched["model_picker_enabled"] = picker_enabled
        display_name = entry.get("name")
        if isinstance(display_name, str) and display_name.strip():
            enriched["display_name"] = display_name.strip()
        vendor = entry.get("vendor")
        if isinstance(vendor, str) and vendor.strip():
            enriched["provider"] = vendor.strip()
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
        client = _get_upstream_client()
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


# Pre-load and pre-compress the dashboard HTML at import time. The file is
# ~400KB raw and compresses to ~50KB; serving the precompressed bytes saves
# both the network transfer and the per-request gzip cost.
_DASHBOARD_HTML_LOCK = threading.Lock()
_DASHBOARD_HTML_RAW: bytes | None = None
_DASHBOARD_HTML_GZIPPED: bytes | None = None
_DASHBOARD_HTML_MTIME: float = 0.0
_DASHBOARD_HTML_ETAG: str = ""


def _load_dashboard_html_bytes() -> tuple[bytes, bytes, str]:
    global _DASHBOARD_HTML_RAW, _DASHBOARD_HTML_GZIPPED, _DASHBOARD_HTML_MTIME, _DASHBOARD_HTML_ETAG
    with _DASHBOARD_HTML_LOCK:
        try:
            stat = os.stat(DASHBOARD_FILE)
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError:
            mtime = 0.0
            size = 0
        if _DASHBOARD_HTML_RAW is None or mtime != _DASHBOARD_HTML_MTIME:
            with open(DASHBOARD_FILE, "rb") as f:
                raw = f.read()
            _DASHBOARD_HTML_RAW = raw
            _DASHBOARD_HTML_GZIPPED = gzip.compress(raw, compresslevel=9)
            _DASHBOARD_HTML_MTIME = mtime
            # Strong-ish ETag from size + mtime + content hash; cheap to
            # compute once at load time and stable for the file's lifetime.
            digest = hashlib.sha256(raw).hexdigest()[:16]
            _DASHBOARD_HTML_ETAG = f'"dash-{size}-{int(mtime)}-{digest}"'
        return _DASHBOARD_HTML_RAW, _DASHBOARD_HTML_GZIPPED, _DASHBOARD_HTML_ETAG


@app.get("/ui", response_class=HTMLResponse)
async def dashboard(request: Request):
    raw, gzipped, etag = _load_dashboard_html_bytes()
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
    accept_encoding = request.headers.get("accept-encoding", "")
    if "gzip" in accept_encoding.lower():
        return Response(
            content=gzipped,
            media_type="text/html; charset=utf-8",
            headers={
                "Content-Encoding": "gzip",
                "Vary": "Accept-Encoding",
                "Cache-Control": "no-cache",
                "ETag": etag,
            },
        )
    return Response(
        content=raw,
        media_type="text/html; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "ETag": etag,
        },
    )


def _build_dashboard_response_body(refresh: bool, gzip_body: bool = False) -> tuple[bytes, bool]:
    payload = dashboard_service.build_payload(refresh)
    body = json.dumps(payload, separators=(",", ":"), default=util._json_default).encode("utf-8")
    if gzip_body and len(body) >= 1024:
        return gzip.compress(body, compresslevel=6), True
    return body, False


def _build_dashboard_sse_event(event_name: str) -> bytes:
    payload = dashboard_service.build_payload(False)
    return format_translation.sse_encode(event_name, payload)


@app.get("/api/dashboard")
async def dashboard_api(request: Request):
    refresh = request.query_params.get("refresh", "").lower() in {"1", "true", "yes"}
    accept_encoding = request.headers.get("accept-encoding", "")
    accepts_gzip = "gzip" in accept_encoding.lower()
    body, encoded = await asyncio.to_thread(
        _build_dashboard_response_body, refresh, accepts_gzip
    )
    headers = {"Cache-Control": "no-store"}
    if encoded:
        headers["Content-Encoding"] = "gzip"
        headers["Vary"] = "Accept-Encoding"
    return Response(
        content=body,
        media_type="application/json",
        headers=headers,
    )


@app.get("/api/dashboard/stream")
async def dashboard_stream(request: Request):
    heartbeat_seconds = 20
    poll_seconds = 1.0
    queue = dashboard_service.register_stream_listener()
    last_version = dashboard_service.current_stream_version()

    async def stream():
        nonlocal last_version
        # Emit an initial heartbeat so EventSource clients see the stream is
        # live immediately. The page concurrently fetches /api/dashboard, so
        # we deliberately skip a redundant initial dashboard build here and
        # only stream payloads when the version changes.
        yield format_translation.sse_encode("heartbeat", {"at": util.utc_now_iso()})
        last_heartbeat = time.monotonic()
        try:
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
                chunk = await asyncio.to_thread(_build_dashboard_sse_event, "dashboard")
                yield chunk
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


def _load_request_prompt_payload(request_id: str) -> dict:
    if not isinstance(request_id, str) or not request_id:
        return {"available": False}
    _prune_request_prompt_archive()
    target = None
    for event in reversed(usage_tracker.snapshot_usage_events()):
        if isinstance(event, dict) and event.get("request_id") == request_id:
            target = event
            break
    if target is not None:
        raw_prompt = target.get("request_prompt")
        if isinstance(raw_prompt, dict):
            prompt = _prompt_payload_for_dashboard(raw_prompt)
            if isinstance(prompt, dict):
                return {"available": True, "request_prompt": prompt}
            return {"available": True, "locked": True}

    archived_prompt = _load_request_prompt_record(request_id)
    if isinstance(archived_prompt, dict):
        prompt_text = archived_prompt.get("prompt_text")
        if isinstance(prompt_text, str) and prompt_text.strip():
            return {
                "available": True,
                "request_prompt": {"user": prompt_text},
                "prompt_text": prompt_text,
                "path": archived_prompt.get("path"),
                "stored_at": archived_prompt.get("stored_at"),
                "char_count": archived_prompt.get("char_count"),
            }

    if target is None:
        return {"available": False, "not_found": True}
    return {"available": False}


@app.get("/api/request-prompt/{request_id}")
async def request_prompt_api(request_id: str):
    payload = await asyncio.to_thread(_load_request_prompt_payload, request_id)
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


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
    payload = client_proxy_config_service.proxy_client_status_payload()
    settings = payload.get("settings")
    if isinstance(settings, dict):
        payload["settings"] = _client_proxy_settings_with_trace_status(settings)
    return JSONResponse(content=payload)


@app.post("/api/config/client-proxy/settings")
async def client_proxy_settings_api(request: Request):
    payload = await parse_json_request(request)
    result = _save_client_proxy_settings(payload)
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




def _responses_route_uses_native_responses_passthrough(body: dict) -> bool:
    requested_model = body.get("model") if isinstance(body, dict) else None
    resolved_target = model_routing_config_service.resolve_target_model(requested_model)
    return model_provider_family(resolved_target or requested_model) == "codex"


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


def _responses_body_has_cache_lineage(body: dict | None) -> bool:
    if not isinstance(body, dict):
        return False
    for key in ("prompt_cache_key", "promptCacheKey", "previous_response_id"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _encrypted_reasoning_strip_reason_for_responses_context(
    request: Request,
    input_value,
    request_body: dict | None = None,
) -> str | None:
    """Drop replayed reasoning ciphertext only when Codex is starting a new lineage."""

    if _responses_body_has_cache_lineage(request_body):
        return None
    if _responses_input_developer_message_count(input_value) > 1:
        return "multiple_developer_messages_without_cache_lineage"
    return None


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
    encrypted_reasoning_strip_reason: str | None,
    dropped_reasoning_items: bool,
) -> dict | None:
    if not isinstance(raw_input, list):
        return None
    before_encrypted = _responses_input_encrypted_content_count(raw_input)
    after_encrypted = _responses_input_encrypted_content_count(sanitized_input)
    if (
        not before_encrypted
        and encrypted_reasoning_strip_reason is None
        and not dropped_reasoning_items
    ):
        return None

    preservation = "disabled" if encrypted_reasoning_strip_reason is not None else "preserved"
    return {
        "input_items_before": len(raw_input),
        "input_items_after": len(sanitized_input) if isinstance(sanitized_input, list) else None,
        "encrypted_content_items_before": before_encrypted,
        "encrypted_content_items_after": after_encrypted,
        "encrypted_content_items_dropped": max(0, before_encrypted - after_encrypted),
        "encrypted_content_preservation": preservation,
        "encrypted_content_strip_reason": encrypted_reasoning_strip_reason,
        "encrypted_keep_last": None,
        "reasoning_items_dropped": dropped_reasoning_items,
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

    raw_input = body.get("input")
    has_compaction_input = format_translation.input_contains_compaction(raw_input)
    native_responses_passthrough = _responses_route_uses_native_responses_passthrough(body)
    encrypted_reasoning_strip_reason = _encrypted_reasoning_strip_reason_for_responses_context(
        request, raw_input, body
    )
    drop_reasoning_items = encrypted_reasoning_strip_reason is not None
    input_sanitization_trace = None
    if raw_input is not None:
        body["input"] = format_translation.sanitize_input(
            raw_input,
            preserve_encrypted_content=encrypted_reasoning_strip_reason is None,
            drop_reasoning_items=drop_reasoning_items,
            native_responses_passthrough=native_responses_passthrough,
        )
        input_sanitization_trace = _responses_input_sanitization_trace(
            raw_input,
            body.get("input"),
            encrypted_reasoning_strip_reason=encrypted_reasoning_strip_reason,
            dropped_reasoning_items=drop_reasoning_items,
        )

    replay_id_repair_trace = None
    if native_responses_passthrough:
        replay_subagent = _responses_effective_subagent(request, body)
        body, replay_id_repair_trace = responses_replay_ids.repair_missing_replay_ids(
            body,
            headers=request.headers,
            subagent=replay_subagent,
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

    trace_metadata_extra = {}
    if input_sanitization_trace is not None:
        trace_metadata_extra["responses_input_sanitization"] = input_sanitization_trace
    if replay_id_repair_trace is not None:
        trace_metadata_extra["responses_replay_id_repair"] = replay_id_repair_trace

    plan, error_response = _prepare_bridge_request(
        request,
        original_body=body,
        bridge_plan=bridge_plan,
        api_base=api_base,
        api_key=api_key,
        force_initiator="agent" if has_compaction_input else None,
        trace_metadata_extra=trace_metadata_extra or None,
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
            affinity_body=body,
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
    print("Starting GHCP proxy on http://127.0.0.1:8000 (loopback only)", flush=True)
    print("  Responses API : POST /v1/responses", flush=True)
    print("  Compaction    : POST /v1/responses/compact", flush=True)
    print("  Chat API      : POST /v1/chat/completions", flush=True)
    print("  Dashboard     : GET  /ui", flush=True)
    print("", flush=True)
    print("  If this is a fresh setup, open /ui and complete GitHub sign-in there.", flush=True)
    print("", flush=True)
    print("  Set in your shell:", flush=True)
    print("    export OPENAI_BASE_URL=http://127.0.0.1:8000/v1", flush=True)
    print("    export OPENAI_API_KEY=anything", flush=True)
    print("", flush=True)

    _write_proxy_pid_file()
    atexit.register(_remove_proxy_pid_file)
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False, timeout_graceful_shutdown=2)
    finally:
        revert_client_proxy_configs_on_shutdown()
        _remove_proxy_pid_file()
