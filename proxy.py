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
import protocol_replies
import upstream_errors
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
    ResponsesStreamIdSyncer,
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
CACHE_TRIPWIRE_INPUT_TOKEN_THRESHOLD = 30_000
CACHE_TRIPWIRE_CACHED_INPUT_TOKEN_THRESHOLD = 30_000
CACHE_TRIPWIRE_FRESH_CACHED_GAP_THRESHOLD = 50_000
CACHE_TRIPWIRE_CONSECUTIVE_HIT_THRESHOLD = 3
CACHE_TRIPWIRE_REASON = "token_tripwire"
CACHE_SETTLE_DELAY_SECONDS = 5.0

safeguard_event_store = dashboard_module.create_safeguard_event_store()
_CACHE_TRIPWIRE_LOCK = threading.Lock()
_CACHE_TRIPWIRE_CONSECUTIVE_HITS = 0
_PROMPT_CACHE_SETTLE_LOCK = threading.Lock()
_PROMPT_CACHE_LAST_FINISH_BY_LINEAGE: dict[tuple[str, str], float] = {}
_PROMPT_CACHE_TRACE_LOCK = threading.Lock()
_PROMPT_CACHE_LAST_INPUT_BY_LINEAGE: dict[tuple[str, str], dict] = {}
_PROMPT_CACHE_AFFINITY_TRACE_LOCK = threading.Lock()
_PROMPT_CACHE_LAST_AFFINITY_BY_LINEAGE: dict[tuple[str, str], dict] = {}


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


# ---------------------------------------------------------------------------
# Shared upstream HTTP client.
#
# Copilot's /responses backend pins its KV (prompt) cache to the connection:
# each new TCP connection lands on a randomly-chosen pod whose cache may be
# cold for the prefix, while a long-lived HTTP/2 connection multiplexes every
# request onto the same pod and lets the cache grow monotonically — the same
# behavior the official Copilot CLI gets out of the box. Constructing an
# httpx.AsyncClient per request fragments the cache and produces wildly
# oscillating cached_input_tokens values for byte-identical payloads. Use a
# single process-wide client for all upstream Copilot calls instead.
# ---------------------------------------------------------------------------
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
            max_connections=8,
            max_keepalive_connections=4,
            keepalive_expiry=300.0,
        )
        try:
            _UPSTREAM_CLIENT = httpx.AsyncClient(
                http2=True,
                timeout=timeout,
                limits=limits,
            )
        except (ImportError, RuntimeError):
            # h2 package not installed; fall back to HTTP/1.1 with persistent
            # keep-alive. Still better than per-request clients.
            _UPSTREAM_CLIENT = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
            )
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


def _reset_cache_tripwire_consecutive_hits() -> None:
    global _CACHE_TRIPWIRE_CONSECUTIVE_HITS
    with _CACHE_TRIPWIRE_LOCK:
        _CACHE_TRIPWIRE_CONSECUTIVE_HITS = 0


def _cache_tripwire_candidate_usage(usage: dict | None) -> dict | None:
    normalized = util.normalize_usage_payload(usage)
    if not isinstance(normalized, dict):
        return None
    input_tokens = util._coerce_int(normalized.get("input_tokens"), default=0)
    cached_tokens = util._coerce_int(normalized.get("cached_input_tokens"), default=0)
    fresh_tokens = max(0, input_tokens - cached_tokens)
    fresh_cached_gap = max(0, fresh_tokens - cached_tokens)
    if (
        input_tokens > CACHE_TRIPWIRE_INPUT_TOKEN_THRESHOLD
        and cached_tokens < CACHE_TRIPWIRE_CACHED_INPUT_TOKEN_THRESHOLD
        and fresh_cached_gap >= CACHE_TRIPWIRE_FRESH_CACHED_GAP_THRESHOLD
    ):
        return normalized
    return None


def _cache_tripwire_usage(usage: dict | None) -> dict | None:
    global _CACHE_TRIPWIRE_CONSECUTIVE_HITS
    if not _cache_tripwire_enabled():
        _reset_cache_tripwire_consecutive_hits()
        return None
    normalized = _cache_tripwire_candidate_usage(usage)
    with _CACHE_TRIPWIRE_LOCK:
        if not isinstance(normalized, dict):
            _CACHE_TRIPWIRE_CONSECUTIVE_HITS = 0
            return None
        _CACHE_TRIPWIRE_CONSECUTIVE_HITS += 1
        if _CACHE_TRIPWIRE_CONSECUTIVE_HITS < CACHE_TRIPWIRE_CONSECUTIVE_HIT_THRESHOLD:
            return None
        return normalized


def _cache_tripwire_enabled() -> bool:
    try:
        settings = client_proxy_config_service.load_client_proxy_settings()
    except Exception:
        return True
    if not isinstance(settings, dict):
        return True
    return bool(settings.get("token_tripwire_enabled", True))


def _prompt_cache_settle_delay_seconds() -> float:
    raw_value = os.environ.get("GHCP_PROXY_RESPONSES_CACHE_SETTLE_DELAY_SECONDS")
    if raw_value is None:
        return CACHE_SETTLE_DELAY_SECONDS
    try:
        return max(0.0, float(str(raw_value).strip()))
    except (TypeError, ValueError):
        return CACHE_SETTLE_DELAY_SECONDS


def _responses_cache_settle_lineage(plan: "UpstreamRequestPlan | None") -> tuple[str, str] | None:
    """Per-(agent-task) settle key. Kept for callers that only need the own-lineage key."""
    keys = _responses_cache_settle_keys(plan)
    return keys[0] if keys else None


def _responses_cache_settle_keys(plan: "UpstreamRequestPlan | None") -> list[tuple[str, str]]:
    """All settle keys a plan should wait/remember under.

    Always includes a per-agent-task key. Also includes a per-family key
    derived from x-parent-agent-id (subagents) or x-agent-task-id (parents)
    so that subagent + parent activity in the same parent-task family
    settle each other. Empirically the upstream prompt cache evicts a
    cached prefix when a sibling (parent or sister subagent) writes to the
    same family namespace, so back-to-back cross-family turns regress to
    near-zero cache hits even though our own outbound prefix is byte-stable.
    """
    if not isinstance(plan, UpstreamRequestPlan) or not isinstance(plan.body, dict):
        return []
    model = str(plan.resolved_model or plan.requested_model or plan.body.get("model") or "").strip().lower()
    if model != "gpt-5.5":
        return []
    keys: list[tuple[str, str]] = []
    own_task = _responses_plan_header_value(plan, "x-agent-task-id") or _responses_plan_task_prefix(plan)
    if own_task:
        keys.append((model, own_task))
    parent_task = _responses_plan_header_value(plan, "x-parent-agent-id")
    family_root = parent_task or own_task
    if family_root:
        family_key = (model, f"family:{family_root}")
        if family_key not in keys:
            keys.append(family_key)
    return keys


async def _wait_for_responses_cache_settle(plan: "UpstreamRequestPlan | None") -> None:
    keys = _responses_cache_settle_keys(plan)
    if not keys:
        return
    delay_seconds = _prompt_cache_settle_delay_seconds()
    if delay_seconds <= 0:
        return
    with _PROMPT_CACHE_SETTLE_LOCK:
        last_finished_at = max(
            (
                t
                for t in (_PROMPT_CACHE_LAST_FINISH_BY_LINEAGE.get(k) for k in keys)
                if isinstance(t, (int, float))
            ),
            default=None,
        )
    if last_finished_at is None:
        return
    wait_seconds = delay_seconds - (time.monotonic() - float(last_finished_at))
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)


def _remember_responses_cache_settle_finish(plan: "UpstreamRequestPlan | None", status_code: int) -> None:
    if status_code >= 400:
        return
    keys = _responses_cache_settle_keys(plan)
    if not keys:
        return
    now = time.monotonic()
    with _PROMPT_CACHE_SETTLE_LOCK:
        for key in keys:
            _PROMPT_CACHE_LAST_FINISH_BY_LINEAGE[key] = now


# ─── Parent prompt-cache LRU keepalive ───────────────────────────────────────
# Subagent activity on the same Codex task evicts the parent's idle prefix from
# upstream cache after ~10s of cumulative writes. Snapshot the parent's last
# successful upstream body per task prefix; when a subagent finishes for the
# same task and the parent has been idle past a small threshold, fire a small
# warmer POST on the parent's lineage so upstream's LRU keeps the prefix hot.
_PARENT_KEEPALIVE_LOCK = threading.Lock()
_PARENT_KEEPALIVE_SNAPSHOTS: dict[str, dict] = {}
_PARENT_KEEPALIVE_MAX_SNAPSHOTS = 64


def _parent_keepalive_enabled() -> bool:
    raw = str(os.environ.get("GHCP_PROXY_PARENT_KEEPALIVE_ENABLED", "0")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _parent_keepalive_min_idle_seconds() -> float:
    raw = os.environ.get("GHCP_PROXY_PARENT_KEEPALIVE_MIN_IDLE_SECONDS")
    if raw is None:
        return 6.0
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return 6.0


def _parent_keepalive_min_interval_seconds() -> float:
    raw = os.environ.get("GHCP_PROXY_PARENT_KEEPALIVE_MIN_INTERVAL_SECONDS")
    if raw is None:
        return 5.0
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return 5.0


def _parent_keepalive_max_output_tokens() -> int:
    raw = os.environ.get("GHCP_PROXY_PARENT_KEEPALIVE_MAX_OUTPUT_TOKENS")
    if raw is None:
        return 16
    try:
        return max(1, int(str(raw).strip()))
    except (TypeError, ValueError):
        return 16


def _parent_keepalive_snapshot_ttl_seconds() -> float:
    raw = os.environ.get("GHCP_PROXY_PARENT_KEEPALIVE_SNAPSHOT_TTL_SECONDS")
    if raw is None:
        return 300.0
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return 300.0


def _responses_plan_task_prefix(plan: "UpstreamRequestPlan | None") -> str | None:
    """Copilot parent task key, with prompt-cache-key fallback for old plans."""
    if not isinstance(plan, UpstreamRequestPlan):
        return None
    parent_agent_id = _responses_plan_header_value(plan, "x-parent-agent-id")
    if parent_agent_id:
        return parent_agent_id
    agent_task_id = _responses_plan_header_value(plan, "x-agent-task-id")
    if agent_task_id:
        return agent_task_id
    body = plan.body if isinstance(plan.body, dict) else None
    if not isinstance(body, dict):
        return None
    pck = body.get("prompt_cache_key") or body.get("promptCacheKey")
    if not isinstance(pck, str):
        return None
    normalized = pck.strip()
    if len(normalized) >= 36 and normalized[8:9] == "-":
        return normalized[:8]
    return None


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


def _responses_plan_role(plan: "UpstreamRequestPlan | None") -> str | None:
    """Classify a /responses gpt-5.5 plan as ``parent`` or ``subagent``.

    Copilot keeps parent and subagents on one client session and separates
    subagent lineage with ``conversation-subagent`` plus ``x-parent-agent-id``.
    """
    if not isinstance(plan, UpstreamRequestPlan):
        return None
    if "/responses" not in (plan.upstream_url or ""):
        return None
    model = str(plan.resolved_model or plan.requested_model or "").strip().lower()
    if model != "gpt-5.5":
        return None
    interaction_type = _responses_plan_header_value(plan, "x-interaction-type")
    parent_agent_id = _responses_plan_header_value(plan, "x-parent-agent-id")
    if parent_agent_id or interaction_type == "conversation-subagent":
        return "subagent"
    cs_id = _responses_plan_header_value(plan, "x-client-session-id")
    if not cs_id:
        return None
    return "parent"


def _evict_oldest_parent_keepalive_snapshot_locked() -> None:
    """Caller must hold ``_PARENT_KEEPALIVE_LOCK``."""
    if len(_PARENT_KEEPALIVE_SNAPSHOTS) <= _PARENT_KEEPALIVE_MAX_SNAPSHOTS:
        return
    oldest_key = None
    oldest_at = float("inf")
    for key, snap in _PARENT_KEEPALIVE_SNAPSHOTS.items():
        ts = snap.get("finished_at", 0.0)
        if ts < oldest_at:
            oldest_at = ts
            oldest_key = key
    if oldest_key is not None:
        _PARENT_KEEPALIVE_SNAPSHOTS.pop(oldest_key, None)


def _remember_parent_for_keepalive(
    plan: "UpstreamRequestPlan | None",
    status_code: int,
) -> None:
    if status_code >= 400:
        return
    if not _parent_keepalive_enabled():
        return
    if _responses_plan_role(plan) != "parent":
        return
    task_prefix = _responses_plan_task_prefix(plan)
    if not task_prefix:
        return
    if not isinstance(plan.body, dict) or not plan.upstream_url:
        return
    snapshot_body = dict(plan.body)
    for key in (
        "prompt_cache_key",
        "promptCacheKey",
        "prompt_cache_retention",
        "previous_response_id",
        "session_id",
        "sessionId",
    ):
        snapshot_body.pop(key, None)
    snapshot = {
        "upstream_url": plan.upstream_url,
        "headers": dict(plan.headers) if isinstance(plan.headers, dict) else {},
        "body": snapshot_body,
        "finished_at": time.monotonic(),
    }
    with _PARENT_KEEPALIVE_LOCK:
        existing = _PARENT_KEEPALIVE_SNAPSHOTS.get(task_prefix)
        if existing is not None:
            snapshot["last_warmer_at"] = existing.get("last_warmer_at", 0.0)
            snapshot["warmer_in_flight"] = existing.get("warmer_in_flight", False)
        else:
            snapshot["last_warmer_at"] = 0.0
            snapshot["warmer_in_flight"] = False
        _PARENT_KEEPALIVE_SNAPSHOTS[task_prefix] = snapshot
        _evict_oldest_parent_keepalive_snapshot_locked()


def _maybe_fire_parent_keepalive(
    plan: "UpstreamRequestPlan | None",
    status_code: int,
) -> None:
    if status_code >= 400:
        return
    if not _parent_keepalive_enabled():
        return
    if _responses_plan_role(plan) != "subagent":
        return
    task_prefix = _responses_plan_task_prefix(plan)
    if not task_prefix:
        return
    now = time.monotonic()
    fire = False
    with _PARENT_KEEPALIVE_LOCK:
        snap = _PARENT_KEEPALIVE_SNAPSHOTS.get(task_prefix)
        if snap is None:
            return
        ttl = _parent_keepalive_snapshot_ttl_seconds()
        if ttl > 0 and now - snap.get("finished_at", 0.0) > ttl:
            _PARENT_KEEPALIVE_SNAPSHOTS.pop(task_prefix, None)
            return
        if snap.get("warmer_in_flight"):
            return
        if now - snap.get("finished_at", 0.0) < _parent_keepalive_min_idle_seconds():
            return
        if now - snap.get("last_warmer_at", 0.0) < _parent_keepalive_min_interval_seconds():
            return
        snap["warmer_in_flight"] = True
        snap["last_warmer_at"] = now
        fire = True
    if not fire:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        with _PARENT_KEEPALIVE_LOCK:
            s = _PARENT_KEEPALIVE_SNAPSHOTS.get(task_prefix)
            if s:
                s["warmer_in_flight"] = False
        return
    loop.create_task(_fire_parent_keepalive(task_prefix))


async def _fire_parent_keepalive(task_prefix: str) -> None:
    with _PARENT_KEEPALIVE_LOCK:
        snap = _PARENT_KEEPALIVE_SNAPSHOTS.get(task_prefix)
        if snap is None:
            return
        upstream_url = snap.get("upstream_url")
        headers = dict(snap.get("headers") or {})
        body = dict(snap.get("body") or {})
    if not isinstance(upstream_url, str) or not upstream_url:
        with _PARENT_KEEPALIVE_LOCK:
            s = _PARENT_KEEPALIVE_SNAPSHOTS.get(task_prefix)
            if s:
                s["warmer_in_flight"] = False
        return
    body["max_output_tokens"] = _parent_keepalive_max_output_tokens()
    body["stream"] = False
    for key in (
        "prompt_cache_key",
        "promptCacheKey",
        "prompt_cache_retention",
        "previous_response_id",
        "session_id",
        "sessionId",
    ):
        body.pop(key, None)
    keepalive_id = uuid4().hex
    started_at = util.utc_now_iso()
    if request_tracing_enabled():
        _append_request_trace(
            {
                "event": "parent_keepalive_started",
                "time": started_at,
                "request_id": keepalive_id,
                "task_prefix": task_prefix,
                "upstream_url": upstream_url,
            }
        )
    status_code: int | None = None
    error_message: str | None = None
    try:
        client = _get_upstream_client()
        response = await client.post(upstream_url, headers=headers, json=body, timeout=30)
        status_code = response.status_code
    except httpx.RequestError as exc:
        error_message = format_translation.upstream_request_error_status_and_message(exc)[1]
    except Exception as exc:  # noqa: BLE001 - keepalive must never raise
        error_message = str(exc)
    finally:
        with _PARENT_KEEPALIVE_LOCK:
            s = _PARENT_KEEPALIVE_SNAPSHOTS.get(task_prefix)
            if s:
                s["warmer_in_flight"] = False
        if request_tracing_enabled():
            payload = {
                "event": "parent_keepalive_finished",
                "time": util.utc_now_iso(),
                "request_id": keepalive_id,
                "task_prefix": task_prefix,
                "status_code": status_code,
            }
            if error_message:
                payload["error"] = error_message
            _append_request_trace(payload)


def _cache_tripwire_reply(usage: dict) -> upstream_errors.SyntheticReply:
    input_tokens = util._coerce_int(usage.get("input_tokens"), default=0)
    cached_tokens = util._coerce_int(usage.get("cached_input_tokens"), default=0)
    fresh_tokens = max(0, input_tokens - cached_tokens)
    fresh_cached_gap = max(0, fresh_tokens - cached_tokens)
    message = (
        "Safety tripwire activated. The proxy stopped the request chain after this response "
        "because it looked like the prompt cache was not being reused correctly, which "
        "could burn through your Copilot session limit; this warning was appended so the "
        "upstream message is still preserved. "
        f"This request had {input_tokens} input tokens, {cached_tokens} cached "
        f"tokens, and {fresh_tokens} fresh input tokens. "
        f"The safety floor is {CACHE_TRIPWIRE_CACHED_INPUT_TOKEN_THRESHOLD} "
        "cached input tokens for large requests, with a "
        f"{CACHE_TRIPWIRE_FRESH_CACHED_GAP_THRESHOLD} token fresh/cache gap limit "
        f"(this request's gap was {fresh_cached_gap}). "
        "If you believe this is an error, you can disable the tripwire in Settings. "
        "Before continuing, debug the cache lineage: compare prompt_cache_key, "
        "promptCacheKey, previous_response_id, x-request-id, x-agent-task-id, "
        "x-interaction-id, encrypted_content preservation, and any request-body "
        "omissions or reordering."
    )
    return upstream_errors.SyntheticReply(
        status_for_trace=200,
        client_status=200,
        message=message,
        reason=CACHE_TRIPWIRE_REASON,
        usage_shape="preserve",
        event_name="cache_tripwire_activated",
        event_payload={
            "reason": CACHE_TRIPWIRE_REASON,
            "threshold_input_tokens": CACHE_TRIPWIRE_INPUT_TOKEN_THRESHOLD,
            "threshold_cached_input_tokens": CACHE_TRIPWIRE_CACHED_INPUT_TOKEN_THRESHOLD,
            "threshold_fresh_cached_gap_tokens": CACHE_TRIPWIRE_FRESH_CACHED_GAP_THRESHOLD,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "fresh_input_tokens": fresh_tokens,
            "fresh_cached_gap_tokens": fresh_cached_gap,
        },
    )


def _cache_tripwire_response_payload(
    reply: upstream_errors.SyntheticReply,
    *,
    protocol: str,
    model: str | None,
    is_compact: bool = False,
) -> dict:
    return protocol_replies.build_synthetic_payload(
        reply,
        protocol=protocol,
        model=model,
        is_compact=is_compact,
    )


def _cache_tripwire_appended_message(reply: upstream_errors.SyntheticReply) -> str:
    return f"\n\n{reply.message}"


def _append_to_responses_payload(payload: dict, suffix: str) -> bool:
    appended = False
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        payload["output_text"] = f"{output_text}{suffix}"
        appended = True

    output = payload.get("output")
    if not isinstance(output, list):
        return appended
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message" or str(item.get("role", "")).lower() != "assistant":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                part["text"] = f"{part['text']}{suffix}"
                return True
    return appended


def _append_to_chat_payload(payload: dict, suffix: str) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    message = first.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        return False
    message["content"] = f"{message['content']}{suffix}"
    return True


def _append_to_anthropic_payload(payload: dict, suffix: str) -> bool:
    content = payload.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            block["text"] = f"{block['text']}{suffix}"
            return True
    return False


def _append_cache_tripwire_reply_to_payload(
    payload: dict,
    reply: upstream_errors.SyntheticReply,
    *,
    protocol: str,
    model: str | None,
    is_compact: bool = False,
) -> dict:
    suffix = _cache_tripwire_appended_message(reply)
    normalized = protocol_replies._normalize_protocol(protocol)
    appended = False
    if normalized == "anthropic":
        appended = _append_to_anthropic_payload(payload, suffix)
    elif normalized == "chat":
        appended = _append_to_chat_payload(payload, suffix)
    elif is_compact:
        appended = _append_to_chat_payload(payload, suffix)
    else:
        appended = _append_to_responses_payload(payload, suffix)
    if appended:
        return payload
    return _cache_tripwire_response_payload(reply, protocol=protocol, model=model, is_compact=is_compact)


def _cache_tripwire_warning_delta(
    reply: upstream_errors.SyntheticReply,
    *,
    output_index: int,
    content_index: int,
) -> bytes:
    return format_translation.sse_encode(
        "response.output_text.delta",
        {
            "type": "response.output_text.delta",
            "output_index": output_index,
            "content_index": content_index,
            "delta": _cache_tripwire_appended_message(reply),
        },
    )


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


def _cache_tripwire_appended_stream_chunks(
    buffered_chunks: list[bytes],
    reply: upstream_errors.SyntheticReply,
) -> list[bytes]:
    text = b"".join(buffered_chunks).decode("utf-8", errors="replace").replace("\r\n", "\n")
    raw_blocks = text.split("\n\n")
    output: list[bytes] = []
    output_index = 0
    content_index = 0
    inserted = False

    for raw_block in raw_blocks:
        if not raw_block.strip():
            continue
        event_name, data = format_translation.parse_sse_block(raw_block)
        if data == "[DONE]":
            continue
        if data:
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

        if not inserted and _is_response_completed_event(event_name, data):
            output.append(
                _cache_tripwire_warning_delta(
                    reply,
                    output_index=output_index,
                    content_index=content_index,
                )
            )
            inserted = True

        output.append(f"{raw_block}\n\n".encode("utf-8"))

    if not inserted:
        output.append(
            _cache_tripwire_warning_delta(
                reply,
                output_index=output_index,
                content_index=content_index,
            )
        )
    output.append(b"data: [DONE]\n\n")
    return output


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


def request_body_dump_enabled() -> bool:
    # Default-on so cache-bust diagnosis has full outbound bytes for byte-diffing
    # paired requests. Opt out with GHCP_DUMP_REQUEST_BODIES=0.
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


def _trace_input_item_hashes(input_value) -> list[str | None] | None:
    if not isinstance(input_value, list):
        return None
    return [_trace_hash(item) for item in input_value]


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
    for key in sorted(body.keys()):
        if key in ("input", "messages"):
            continue
        fingerprint = _trace_hash(body.get(key))
        if fingerprint:
            summary[f"{key}_fingerprint"] = fingerprint

    return summary


def _prompt_cache_trace_lineage(
    body: dict | None,
    model: str | None,
    outbound_headers: dict | None = None,
) -> tuple[str, str] | None:
    if not isinstance(body, dict):
        return None
    model_name = str(model or body.get("model") or "").strip().lower()
    if not model_name:
        return None
    prompt_cache_key = body.get("prompt_cache_key") or body.get("promptCacheKey")
    if isinstance(prompt_cache_key, str) and prompt_cache_key.strip():
        return model_name, prompt_cache_key.strip()
    if isinstance(outbound_headers, dict):
        for header_name in ("x-agent-task-id", "x-parent-agent-id"):
            header_value = outbound_headers.get(header_name)
            if isinstance(header_value, str) and header_value.strip():
                return model_name, f"{header_name}:{header_value.strip()}"
    return None


def _prompt_cache_prefix_diagnostics(
    *,
    request_id: str,
    upstream_body: dict | None,
    resolved_model: str | None,
    outbound_headers: dict | None,
) -> dict | None:
    lineage = _prompt_cache_trace_lineage(upstream_body, resolved_model, outbound_headers)
    if lineage is None or not isinstance(upstream_body, dict):
        return None
    item_hashes = _trace_input_item_hashes(upstream_body.get("input"))
    if item_hashes is None:
        return None

    snapshot = {
        "request_id": request_id,
        "item_hashes": item_hashes,
        "item_count": len(item_hashes),
        "body_fingerprint": _trace_hash(upstream_body),
    }
    with _PROMPT_CACHE_TRACE_LOCK:
        previous = _PROMPT_CACHE_LAST_INPUT_BY_LINEAGE.get(lineage)
        _PROMPT_CACHE_LAST_INPUT_BY_LINEAGE[lineage] = snapshot

    if not isinstance(previous, dict):
        return {
            "lineage_model": lineage[0],
            "previous_request_id": None,
            "current_item_count": len(item_hashes),
        }

    previous_hashes = previous.get("item_hashes")
    if not isinstance(previous_hashes, list):
        return None
    common_prefix_items = 0
    for previous_hash, current_hash in zip(previous_hashes, item_hashes):
        if previous_hash != current_hash:
            break
        common_prefix_items += 1
    previous_is_prefix = common_prefix_items == len(previous_hashes)
    diagnostics = {
        "lineage_model": lineage[0],
        "previous_request_id": previous.get("request_id"),
        "previous_item_count": previous.get("item_count"),
        "current_item_count": len(item_hashes),
        "common_prefix_items": common_prefix_items,
        "previous_is_prefix": previous_is_prefix,
        "current_extends_previous": previous_is_prefix and len(item_hashes) >= len(previous_hashes),
        "previous_body_fingerprint": previous.get("body_fingerprint"),
        "current_body_fingerprint": snapshot["body_fingerprint"],
    }
    if not previous_is_prefix:
        diagnostics["first_mismatch_index"] = common_prefix_items
        diagnostics["previous_item_hash"] = (
            previous_hashes[common_prefix_items]
            if common_prefix_items < len(previous_hashes)
            else None
        )
        diagnostics["current_item_hash"] = (
            item_hashes[common_prefix_items]
            if common_prefix_items < len(item_hashes)
            else None
        )
    return diagnostics


def _prompt_cache_affinity_diagnostics(
    *,
    request_id: str,
    upstream_body: dict | None,
    resolved_model: str | None,
    outbound_headers: dict | None,
) -> dict | None:
    lineage = _prompt_cache_trace_lineage(upstream_body, resolved_model, outbound_headers)
    if lineage is None or not isinstance(outbound_headers, dict):
        return None

    def header_value(name: str) -> str | None:
        value = outbound_headers.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        target = name.lower()
        for key, candidate in outbound_headers.items():
            if isinstance(key, str) and key.lower() == target and isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    snapshot = {
        "request_id": request_id,
        "x_agent_task_id": header_value("x-agent-task-id"),
        "x_interaction_id": header_value("x-interaction-id"),
        "x_client_session_id": header_value("x-client-session-id"),
    }
    with _PROMPT_CACHE_AFFINITY_TRACE_LOCK:
        previous = _PROMPT_CACHE_LAST_AFFINITY_BY_LINEAGE.get(lineage)
        _PROMPT_CACHE_LAST_AFFINITY_BY_LINEAGE[lineage] = snapshot

    if not isinstance(previous, dict):
        return None

    changed_fields = []
    for field in ("x_agent_task_id", "x_interaction_id", "x_client_session_id"):
        if previous.get(field) != snapshot.get(field):
            changed_fields.append(field)

    if not changed_fields:
        return None

    return {
        "lineage_model": lineage[0],
        "previous_request_id": previous.get("request_id"),
        "changed_fields": changed_fields,
        "previous": {field: previous.get(field) for field in changed_fields},
        "current": {field: snapshot.get(field) for field in changed_fields},
    }


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


_REQUEST_BODY_DUMP_LOCK = threading.Lock()
_REQUEST_BODY_DUMP_EXECUTOR: "concurrent.futures.ThreadPoolExecutor | None" = None
_REQUEST_BODY_DUMP_EXECUTOR_LOCK = threading.Lock()


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
    """Persist the exact outbound body and full headers for a request.

    Default-on so paired hit/bust analysis can byte-diff what we sent upstream
    — which fingerprints in the trace cannot expose. Opt out with
    GHCP_DUMP_REQUEST_BODIES=0. The actual file write runs on a background
    thread so it never blocks the event loop, and any failure is swallowed
    so a serialization issue cannot fail the upstream request itself. Files
    are bounded by REQUEST_TRACE_HISTORY_LIMIT.
    """
    if not request_body_dump_enabled():
        return
    try:
        dump_dir = request_body_dump_dir()
        # Build a snapshot on the caller's thread so we don't race the request
        # handler mutating the body after we hand off; serialization happens
        # in the background thread.
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
            "request_body": request_body,
            "upstream_body": upstream_body,
        }
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
        upstream_body = snapshot.get("upstream_body")
        try:
            wire_bytes = json.dumps(upstream_body, default=util._json_default).encode("utf-8")
        except (TypeError, ValueError):
            wire_bytes = None
        if wire_bytes is not None:
            snapshot["upstream_body_wire"] = wire_bytes.decode("utf-8", errors="replace")
            snapshot["upstream_body_wire_size"] = len(wire_bytes)
            snapshot["upstream_body_wire_sha256"] = hashlib.sha256(wire_bytes).hexdigest()
        with _REQUEST_BODY_DUMP_LOCK:
            os.makedirs(dump_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, default=util._json_default)
            _enforce_body_dump_retention_locked(dump_dir)
    except Exception as exc:  # pragma: no cover - dump must never raise
        print(f"Warning: failed to write request body dump: {exc}", file=sys.stderr, flush=True)


def _enforce_body_dump_retention_locked(dump_dir: str) -> None:
    """Cap body-dump directory at REQUEST_TRACE_HISTORY_LIMIT files.

    Caller must hold ``_REQUEST_BODY_DUMP_LOCK``. Bodies can be 10-100KB
    each; without retention the directory grows unbounded.
    """
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
    drop = len(paths) - limit
    for _, path in paths[:drop]:
        try:
            os.unlink(path)
        except OSError:
            pass


def _anthropic_messages_usage_for_tracking(usage: dict | None) -> dict | None:
    """Internal accounting shape for Anthropic Messages prompt-cache usage.

    Anthropic reports ``input_tokens`` as fresh non-cache-read input, then
    reports cache reads/writes separately. Persist non-cache-read input while
    preserving visible cache buckets, and keep pricing-only counters for cost
    calculation so the display shape cannot double-charge cache usage.
    """
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
    """Return client-visible Anthropic usage with cache reads kept separate.

    Anthropic's raw Messages usage reports cache writes separately from
    ``input_tokens`` and cache reads under ``cache_read_input_tokens``. Claude
    Code's aggregate usage display treats cache fields as separate buckets, so
    ``input_tokens`` should exclude cache reads while still including cache
    writes. Internal tracking keeps separate ``pricing_*`` counters for cost.
    """
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
    usage = payload.get("usage")
    client_usage, changed = _anthropic_messages_usage_for_client(usage)
    if not changed:
        return payload, False
    client_payload = dict(payload)
    client_payload["usage"] = client_usage
    return client_payload, True


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
    trace_details = trace_metadata if isinstance(trace_metadata, dict) else {}
    cache_prefix_diagnostics = _prompt_cache_prefix_diagnostics(
        request_id=request_id,
        upstream_body=upstream_body,
        resolved_model=resolved_model,
        outbound_headers=outbound_headers,
    )
    if cache_prefix_diagnostics is not None:
        trace_details = dict(trace_details)
        trace_details["prompt_cache_prefix"] = cache_prefix_diagnostics
    cache_affinity_diagnostics = _prompt_cache_affinity_diagnostics(
        request_id=request_id,
        upstream_body=upstream_body,
        resolved_model=resolved_model,
        outbound_headers=outbound_headers,
    )
    if cache_affinity_diagnostics is not None:
        trace_details = dict(trace_details)
        trace_details["prompt_cache_affinity_drift"] = cache_affinity_diagnostics
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
            _remember_responses_cache_settle_finish(plan, status_code)
            _remember_parent_for_keepalive(plan, status_code)
            _maybe_fire_parent_keepalive(plan, status_code)
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
        # Stable affinity when the caller supplies a durable cache/session
        # lineage hint. Without that hint we keep rotate-on-user behavior so
        # unrelated one-off requests do not silently merge into one bucket.
        stable_affinity_hint = isinstance(original_body, dict) and any(
            isinstance(original_body.get(k), str) and original_body.get(k).strip()
            for k in ("prompt_cache_key", "promptCacheKey", "sessionId", "session_id")
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
        if _should_retry_without_prompt_cache_retention(upstream, plan):
            _drop_prompt_cache_retention_for_retry(plan)
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
        if _should_retry_without_prompt_cache_retention(upstream, plan):
            _drop_prompt_cache_retention_for_retry(plan)
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

    translated_payload = _translate_bridge_success_payload(bridge_plan, upstream_payload)
    if bridge_plan.caller_protocol == "anthropic" and bridge_plan.upstream_protocol == "messages":
        translated_payload, _ = _anthropic_messages_payload_for_client(translated_payload)
    tracking_usage = (
        _anthropic_messages_usage_for_tracking(upstream_payload.get("usage"))
        if bridge_plan.upstream_protocol == "messages"
        and isinstance(upstream_payload.get("usage"), dict)
        else None
    )
    tripwire_usage = (
        _cache_tripwire_usage(
            translated_payload.get("usage") if isinstance(translated_payload, dict) else None
        )
        if bridge_plan.upstream_protocol == "responses"
        else None
    )
    if isinstance(tripwire_usage, dict):
        reply = _cache_tripwire_reply(tripwire_usage)
        response_payload = _append_cache_tripwire_reply_to_payload(
            translated_payload,
            reply,
            protocol=bridge_plan.caller_protocol,
            model=bridge_plan.resolved_model or bridge_plan.requested_model,
            is_compact=bridge_plan.is_compact,
        )
        _finish_usage_and_trace(
            plan,
            reply.status_for_trace,
            upstream=upstream,
            response_payload=response_payload,
            response_text=(
                format_translation.extract_response_output_text(response_payload)
                if bridge_plan.caller_protocol == "responses"
                else util.extract_item_text(response_payload.get("content", [{}])[0])
                if isinstance(response_payload.get("content"), list)
                else None
            )
            or reply.message,
            usage=tracking_usage if isinstance(tracking_usage, dict) else tripwire_usage,
        )
        _publish_synthetic_reply_event(reply, plan)
        return JSONResponse(content=response_payload, status_code=reply.client_status)
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
        # Responses-shape input_tokens already excludes cache reads.
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


def _plan_uses_prompt_cache_retention(plan: UpstreamRequestPlan | None) -> bool:
    return (
        isinstance(plan, UpstreamRequestPlan)
        and isinstance(plan.body, dict)
        and "prompt_cache_retention" in plan.body
    )


def _upstream_rejected_prompt_cache_retention(upstream: httpx.Response) -> bool:
    if upstream.status_code not in {400, 422}:
        return False
    payload = _extract_upstream_json_payload(upstream)
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            param = error.get("param")
            if isinstance(param, str) and param == "prompt_cache_retention":
                return True
            message = error.get("message")
            if isinstance(message, str) and "prompt_cache_retention" in message:
                return True
        message = payload.get("message")
        if isinstance(message, str) and "prompt_cache_retention" in message:
            return True
    text = _extract_upstream_text(upstream)
    return isinstance(text, str) and "prompt_cache_retention" in text


def _drop_prompt_cache_retention_for_retry(plan: UpstreamRequestPlan) -> None:
    if not isinstance(plan.body, dict):
        return
    if "prompt_cache_retention" not in plan.body:
        return
    plan.body = {key: value for key, value in plan.body.items() if key != "prompt_cache_retention"}
    if isinstance(plan.trace_context, dict):
        plan.trace_context["prompt_cache_retention_retry"] = {
            "action": "drop_unsupported_field",
            "field": "prompt_cache_retention",
        }


def _should_retry_without_prompt_cache_retention(
    upstream: httpx.Response,
    plan: UpstreamRequestPlan | None,
) -> bool:
    return _plan_uses_prompt_cache_retention(plan) and _upstream_rejected_prompt_cache_retention(upstream)


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
        if _should_retry_without_prompt_cache_retention(upstream, trace_plan):
            await upstream.aclose()
            _drop_prompt_cache_retention_for_retry(trace_plan)
            client = _get_upstream_client()
            retry_body = trace_plan.body if isinstance(trace_plan, UpstreamRequestPlan) else body
            request = client.build_request("POST", upstream_url, headers=headers, json=retry_body)
            try:
                upstream = await throttled_client_send(client, request, stream=True)
            except httpx.RequestError as exc:
                status_code, message = format_translation.upstream_request_error_status_and_message(exc)
                _finish_usage_and_trace(trace_plan, status_code, response_text=message)
                return format_translation.openai_error_response(status_code, message)
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
        source_iter = _stream_with_update_notice(upstream.aiter_bytes(), stream_type, getattr(upstream, "headers", None))
        if stream_type == "responses":
            source_iter = ResponsesStreamIdSyncer().sync(source_iter)
        buffered_chunks: list[bytes] = []
        finish_called = False
        try:
            async for chunk in source_iter:
                if capture.feed(chunk):
                    usage_tracker.mark_first_output(usage_event)
                buffered_chunks.append(chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8"))
            tripwire_usage = (
                _cache_tripwire_usage(capture.usage if isinstance(capture.usage, dict) else None)
                if stream_type == "responses"
                else None
            )
            if isinstance(tripwire_usage, dict):
                reply = _cache_tripwire_reply(tripwire_usage)
                model = trace_plan.resolved_model if isinstance(trace_plan, UpstreamRequestPlan) else None
                response_payload = _cache_tripwire_response_payload(
                    reply,
                    protocol=stream_type,
                    model=model,
                )
                _finish_usage_and_trace(
                    trace_plan,
                    reply.status_for_trace,
                    upstream=upstream,
                    response_payload=response_payload,
                    response_text=reply.message,
                    usage=tripwire_usage,
                )
                finish_called = True
                _publish_synthetic_reply_event(reply, trace_plan)
                usage_tracker.mark_first_output(usage_event)
                for appended_chunk in _cache_tripwire_appended_stream_chunks(buffered_chunks, reply):
                    yield appended_chunk
                return
            _finish_usage_and_trace(
                trace_plan,
                upstream.status_code,
                upstream=upstream,
                usage=capture.usage if isinstance(capture.usage, dict) else None,
            )
            finish_called = True
            for chunk in buffered_chunks:
                yield chunk
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
    """Classify contexts where replayed encrypted reasoning is not worth forwarding.

    Codex subagent/fork starts can replay the parent transcript, including
    encrypted reasoning blobs, under a fresh prompt-cache key. Copilot/OpenAI
    validates those blobs against the active lineage and rejects foreign ones
    with ``invalid_request_body``. Copilot CLI also replays tool history with
    encrypted reasoning and no body cache-lineage fields, so tool history alone
    is not a fork signal. Strip ciphertext only when we have an explicit
    subagent marker or the input has a fork shape Codex has emitted before.
    Summaries remain available, so the visible transcript is preserved.
    """
    if _responses_body_has_cache_lineage(request_body):
        return None
    subagent = request.headers.get("x-openai-subagent") if hasattr(request, "headers") else None
    if isinstance(subagent, str) and subagent.strip():
        return "subagent_header"
    if _responses_input_developer_message_count(input_value) > 1:
        return "multiple_developer_messages_without_cache_lineage"
    return None


def _should_drop_reasoning_items_for_responses_context(strip_reason: str | None) -> bool:
    return strip_reason is not None


def _responses_route_uses_native_responses_passthrough(body: dict) -> bool:
    requested_model = body.get("model") if isinstance(body, dict) else None
    resolved_target = model_routing_config_service.resolve_target_model(requested_model)
    return model_provider_family(resolved_target or requested_model) == "codex"


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

    if encrypted_reasoning_strip_reason is not None:
        preservation = "disabled"
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

    # Sanitize input (multi-turn encrypted_content passthrough). Forked
    # subagent contexts replay parent reasoning under a new lineage, so do not
    # forward unverifiable encrypted reasoning blobs in that shape.
    raw_input = body.get("input")
    has_compaction_input = format_translation.input_contains_compaction(raw_input)
    native_responses_passthrough = _responses_route_uses_native_responses_passthrough(body)
    encrypted_reasoning_strip_reason = _encrypted_reasoning_strip_reason_for_responses_context(
        request, raw_input, body
    )
    drop_reasoning_items = _should_drop_reasoning_items_for_responses_context(
        encrypted_reasoning_strip_reason
    )
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
