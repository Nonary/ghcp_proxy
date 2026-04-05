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
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from threading import Thread
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from anthropic_stream import AnthropicStreamTranslator
from initiator_policy import InitiatorPolicy
from event_bus import EventBus
from proxy_runtime import (
    AuthRuntimeBindings,
    ProxyRuntime,
    ProxySettings,
    UsageTrackingRuntimeBindings,
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
)

from util import (
    _json_default,
    _utc_now,
    _utc_now_iso,
    _normalize_usage_payload,
    _extract_payload_usage,
    _usage_event_cost,
    _premium_request_multiplier,
    _month_key_for_source_row,
    parse_json_request as _parse_json_request_impl,
)

from rate_limiting import (
    throttle_upstream_request,
    throttled_client_post,
    throttled_client_send,
)

from format_translation import (
    has_vision_input,
    model_requires_anthropic_beta,
    normalize_upstream_model_name,
    resolve_copilot_model_name,
    _normalize_anthropic_cache_control,
    _attach_copilot_cache_control,
    _anthropic_image_block_to_chat,
    _normalize_anthropic_content_blocks,
    _anthropic_text_or_image_block_to_chat,
    _anthropic_system_to_chat_content,
    _anthropic_blocks_to_chat_content,
    _anthropic_tool_result_content_to_text,
    _anthropic_tool_use_to_chat_tool_call,
    anthropic_message_to_chat_messages,
    anthropic_tools_to_chat,
    anthropic_tool_choice_to_chat,
    anthropic_request_to_chat,
    _chat_usage_to_anthropic,
    _chat_stop_reason_to_anthropic,
    _extract_chat_message_text,
    _parse_tool_call_arguments,
    _chat_message_to_anthropic_content,
    chat_completion_to_anthropic,
    _anthropic_error_type_for_status,
    anthropic_error_payload_from_openai,
    anthropic_error_response,
    anthropic_error_response_from_upstream,
    _openai_error_type_for_status,
    openai_error_response,
    _upstream_request_error_status_and_message,
    _http_exception_detail_to_message,
    _sse_encode,
    _parse_sse_block,
    _iter_sse_messages,
    _extract_text_from_chat_delta,
    _extract_tool_call_deltas,
    build_copilot_headers,
    _apply_forwarded_request_headers,
    build_responses_headers_for_request,
    build_chat_headers_for_request,
    build_anthropic_headers_for_request,
    _anthropic_messages_has_vision,
    _strip_anthropic_cache_control,
    prepare_anthropic_outbound_body,
    encode_fake_compaction,
    decode_fake_compaction,
    _summary_message_item,
    input_contains_compaction,
    sanitize_input,
    build_fake_compaction_request,
    extract_response_output_text,
    build_fake_compaction_response,
)

from auth import (
    _gh_headers,
    _load_access_token,
    _save_access_token,
    _load_billing_token,
    _save_billing_token,
    _clear_billing_token,
    _billing_token_status,
    _backup_config_file,
    _latest_backup_path,
    _parse_toml_values,
    _empty_proxy_status,
    _disable_client_proxy_config,
    _normalize_proxy_targets,
    _load_api_key,
    _load_api_key_payload,
    _get_api_base,
    _device_flow,
    _refresh_api_key,
    get_api_key,
    ensure_authenticated,
)

from usage_tracking import (
    REQUEST_FINISHED_EVENT,
    USAGE_EVENT_RECORDED_EVENT,
    UsageArchiveStore,
    UsageTracker,
    UsageTrackingState,
    configure_usage_tracking,
    _usage_log_lock,
    _recent_usage_events,
    _archived_usage_events,
    _session_request_id_lock,
    _latest_server_request_ids_by_chain,
    _active_server_request_ids_by_request,
    _latest_claude_user_session_contexts,
    _request_session_id,
    _remember_server_request_id,
    _remember_active_server_request_id,
    _forget_active_server_request_id,
    _get_active_server_request_id_for_request,
    _get_active_request_context_for_request,
    _get_latest_server_request_id_for_request,
    _resolve_server_request_id,
    _apply_missing_claude_session_context,
    _normalize_recorded_usage_event,
    _start_usage_event,
    _mark_usage_event_first_output,
    _snapshot_usage_events,
    _snapshot_all_usage_events,
    _record_usage_event,
    _record_request_error,
    _rewrite_usage_log_locked,
    _usage_event_archive_summary,
    _usage_event_archive_key,
    _SSEUsageCapture,
    _initiator_log_label,
    log_proxy_request,
)

from dashboard import (
    DashboardDependencies,
    DashboardService,
    seed_cached_payloads_from_sqlite,
    _init_sqlite_cache,
    _sqlite_cache_lock,
    _sqlite_connect,
    _sqlite_cache_put,
    _set_sqlite_cache_unavailable,
    _infer_premium_allowance,
    _extract_quota_summary,
    _github_rest_headers,
    _github_rest_get_json,
    _fetch_github_identity,
    _load_cached_github_identity,
    _load_billing_org_candidates,
    _empty_official_premium_payload,
    _current_billing_month_bounds,
    _register_dashboard_stream_listener,
    _unregister_dashboard_stream_listener,
    _notify_dashboard_stream_listeners,
    _premium_cache_lock,
    _premium_cache,
    _new_usage_aggregate_bucket,
    _ingest_usage_event,
    _finalize_usage_bucket,
    _aggregate_usage_event_buckets,
    _normalize_month_row,
    _combine_month_rows,
    _combine_usage_rows,
    _usage_event_group_key,
    _usage_event_session_descriptor,
)


# ─── App & Global State ──────────────────────────────────────────────────────

app = FastAPI()

_initiator_policy = InitiatorPolicy()
usage_event_bus = EventBus()


def _current_usage_tracking_state() -> UsageTrackingState:
    return UsageTrackingState(
        usage_log_lock=_usage_log_lock,
        recent_usage_events=_recent_usage_events,
        archived_usage_events=_archived_usage_events,
        session_request_id_lock=_session_request_id_lock,
        latest_server_request_ids_by_chain=_latest_server_request_ids_by_chain,
        active_server_request_ids_by_request=_active_server_request_ids_by_request,
        latest_claude_user_session_contexts=_latest_claude_user_session_contexts,
    )


usage_event_bus.subscribe(
    REQUEST_FINISHED_EVENT,
    lambda request_id, finished_at=None: _initiator_policy.note_request_finished(
        request_id,
        finished_at=finished_at,
    ),
)
usage_event_bus.subscribe(
    USAGE_EVENT_RECORDED_EVENT,
    lambda _event: _notify_dashboard_stream_listeners(),
)

usage_tracker = UsageTracker(
    state=_current_usage_tracking_state(),
    archive_store=UsageArchiveStore(
        init_storage=lambda: _init_sqlite_cache(),
        lock=_sqlite_cache_lock,
        connect=lambda: _sqlite_connect(),
        mark_unavailable=lambda error: _set_sqlite_cache_unavailable(error),
    ),
    event_bus=usage_event_bus,
)
configure_usage_tracking(
    state=usage_tracker.state,
    archive_store=usage_tracker.archive_store,
    event_bus=usage_event_bus,
)


def _current_proxy_settings() -> ProxySettings:
    return ProxySettings(
        codex_config_file=CODEX_CONFIG_FILE,
        codex_proxy_config=CODEX_PROXY_CONFIG,
        claude_settings_file=CLAUDE_SETTINGS_FILE,
        claude_proxy_settings=CLAUDE_PROXY_SETTINGS,
        claude_max_context_tokens=CLAUDE_MAX_CONTEXT_TOKENS,
        claude_max_output_tokens=CLAUDE_MAX_OUTPUT_TOKENS,
    )


_runtime = ProxyRuntime(
    settings_provider=_current_proxy_settings,
    auth_runtime=AuthRuntimeBindings(
        load_api_key_payload=lambda: _load_api_key_payload(),
    ),
    usage_tracking_runtime=UsageTrackingRuntimeBindings(
        state_provider=_current_usage_tracking_state,
    ),
)

dashboard_service = DashboardService(
    dependencies=DashboardDependencies(
        load_billing_token=lambda: _load_billing_token(),
        load_access_token=lambda: _load_access_token(),
        load_api_key_payload=lambda: _load_api_key_payload(),
        snapshot_all_usage_events=lambda: _snapshot_all_usage_events(),
        snapshot_usage_events=lambda: _snapshot_usage_events(),
    ),
    utc_now=lambda: _utc_now(),
    utc_now_iso=lambda: _utc_now_iso(),
    sqlite_cache_put=lambda cache_key, payload: _sqlite_cache_put(cache_key, payload),
    notify_dashboard_stream_listeners=lambda: _notify_dashboard_stream_listeners(),
    thread_class=lambda: Thread,
)


# ─── Module-level parse_json_request wrapper ──────────────────────────────────
# Wraps the util implementation to inject the error callback for recording
# request parsing errors.

async def parse_json_request(request: Request) -> dict:
    return await _parse_json_request_impl(request, error_callback=_record_request_error)


# Runtime-backed compatibility surface for tests and route handlers.
_codex_proxy_status = _runtime.codex_proxy_status
_claude_proxy_status = _runtime.claude_proxy_status
_write_codex_proxy_config = _runtime.write_codex_proxy_config
_write_claude_proxy_settings = _runtime.write_claude_proxy_settings
_disable_codex_proxy_config = _runtime.disable_codex_proxy_config
_disable_claude_proxy_settings = _runtime.disable_claude_proxy_settings
_proxy_client_status_payload = _runtime.proxy_client_status_payload
_load_archived_usage_history = _runtime.load_archived_usage_history
_load_usage_history = _runtime.load_usage_history
_compact_usage_history_if_needed = _runtime.compact_usage_history_if_needed
_finish_usage_event = _runtime.finish_usage_event


# ─── Initialization ──────────────────────────────────────────────────────────

_runtime.initialize(_initiator_policy)
seed_cached_payloads_from_sqlite()


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
    upstream_url: str
    headers: dict
    body: dict
    usage_event: dict | None
    requested_model: str | None
    resolved_model: str | None


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
) -> tuple[UpstreamRequestPlan | None, Response | None]:
    request_id = uuid4().hex

    effective_api_key = api_key
    if effective_api_key is None:
        try:
            effective_api_key = get_api_key()
        except Exception as exc:
            return None, error_response(401, f"GHCP auth failed: {exc}")

    headers = header_builder(effective_api_key, request_id)
    log_proxy_request(request, requested_model, resolved_model, headers.get("X-Initiator"))
    usage_event = _start_usage_event(
        request,
        requested_model,
        resolved_model,
        headers.get("X-Initiator"),
        request_id=request_id,
        request_body=body,
        upstream_path=upstream_path,
        outbound_headers=headers,
    )
    return (
        UpstreamRequestPlan(
            upstream_url=upstream_url,
            headers=headers,
            body=body,
            usage_event=usage_event,
            requested_model=requested_model,
            resolved_model=resolved_model,
        ),
        None,
    )


async def _post_non_streaming_request(plan: UpstreamRequestPlan, *, error_response) -> Response:
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            upstream = await throttled_client_post(
                client,
                plan.upstream_url,
                headers=plan.headers,
                json=plan.body,
            )
        _finish_usage_event(
            plan.usage_event,
            upstream.status_code,
            upstream=upstream,
            response_payload=_extract_upstream_json_payload(upstream),
            response_text=_extract_upstream_text(upstream),
        )
        return proxy_non_streaming_response(upstream)
    except httpx.RequestError as exc:
        status_code, message = _upstream_request_error_status_and_message(exc)
        _finish_usage_event(plan.usage_event, status_code, response_text=message)
        return error_response(status_code, message)
    except Exception:
        _finish_usage_event(plan.usage_event, 599)
        raise


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
        status_code, message = _upstream_request_error_status_and_message(exc)
        _finish_usage_event(usage_event, status_code, response_text=message)
        await client.aclose()
        return openai_error_response(status_code, message)
    except Exception:
        _finish_usage_event(usage_event, 599)
        await client.aclose()
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            _finish_usage_event(
                usage_event,
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
        capture = _SSEUsageCapture(stream_type)
        try:
            async for chunk in upstream.aiter_bytes():
                if capture.feed(chunk):
                    _mark_usage_event_first_output(usage_event)
                yield chunk
        finally:
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                usage=capture.usage if isinstance(capture.usage, dict) else None,
            )
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
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
) -> Response:
    """
    Translate upstream chat-completions SSE into Anthropic Messages SSE.
    """
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request("POST", upstream_url, headers=headers, json=body)
    try:
        upstream = await throttled_client_send(client, request, stream=True)
    except httpx.RequestError as exc:
        status_code, message = _upstream_request_error_status_and_message(exc)
        _finish_usage_event(usage_event, status_code, response_text=message)
        await client.aclose()
        return anthropic_error_response(status_code, message)
    except Exception:
        _finish_usage_event(usage_event, 599)
        await client.aclose()
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            fallback_message = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
            error_payload = anthropic_error_payload_from_openai(_extract_upstream_json_payload(upstream), upstream.status_code, fallback_message)
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=error_payload,
                response_text=error_payload.get("error", {}).get("message"),
            )
            return anthropic_error_response_from_upstream(upstream)
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
            mark_first_output=lambda: _mark_usage_event_first_output(usage_event),
        )
        try:
            async for event in translator.translate(upstream.aiter_bytes()):
                yield event
        finally:
            response_payload = translator.build_response_payload()
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text=translator.response_text,
                usage=response_payload["usage"],
            )
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_translated(),
        status_code=upstream.status_code,
        headers=response_headers,
    )


dashboard_service.trigger_official_premium_refresh()


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
    queue = _register_dashboard_stream_listener()
    last_version = _runtime.dashboard_stream_version

    async def stream():
        nonlocal last_version
        last_heartbeat = time.monotonic()
        try:
            initial_payload = await asyncio.to_thread(dashboard_service.build_payload, False)
            yield _sse_encode("dashboard", initial_payload)
            while True:
                if await request.is_disconnected():
                    break

                try:
                    version = await asyncio.wait_for(queue.get(), timeout=poll_seconds)
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if now - last_heartbeat >= heartbeat_seconds:
                        last_heartbeat = now
                        yield _sse_encode("heartbeat", {"at": _utc_now_iso()})
                    continue

                if version == last_version:
                    continue
                last_version = version
                payload = await asyncio.to_thread(dashboard_service.build_payload, False)
                yield _sse_encode("dashboard", payload)
        finally:
            _unregister_dashboard_stream_listener(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Config API routes ────────────────────────────────────────────────────────

@app.get("/api/config/billing-token")
async def billing_token_status_api():
    return JSONResponse(content=_billing_token_status())


@app.post("/api/config/billing-token")
async def billing_token_config_api(request: Request):
    payload = await parse_json_request(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    if bool(payload.get("clear")):
        _clear_billing_token()
        return JSONResponse(content=_billing_token_status())

    token = payload.get("token")
    if token is None:
        raise HTTPException(status_code=400, detail="Missing token field")
    if not isinstance(token, str):
        raise HTTPException(status_code=400, detail="Token must be a string")
    token = token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token must not be empty")

    current_status = _billing_token_status()
    if current_status.get("readonly"):
        raise HTTPException(
            status_code=409,
            detail="Billing token is configured via GHCP_GITHUB_BILLING_TOKEN and cannot be changed via UI.",
        )

    _save_billing_token(token)
    return JSONResponse(content=_billing_token_status())


@app.get("/api/config/client-proxy")
async def client_proxy_status_api():
    return JSONResponse(content=_proxy_client_status_payload())


@app.post("/api/config/client-proxy")
async def client_proxy_install_api(request: Request):
    payload = await parse_json_request(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")
    targets = _normalize_proxy_targets(payload)
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
                if target == "codex":
                    clients[target] = _disable_codex_proxy_config()
                else:
                    clients[target] = _disable_claude_proxy_settings()
            elif target == "codex":
                clients[target] = _write_codex_proxy_config()
            else:
                clients[target] = _write_claude_proxy_settings()
        except Exception as exc:
            clients[target] = _empty_proxy_status(
                target,
                CODEX_CONFIG_FILE if target == "codex" else CLAUDE_SETTINGS_FILE,
            )
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
        return openai_error_response(exc.status_code, _http_exception_detail_to_message(exc.detail))

    # Sanitize input (multi-turn encrypted_content passthrough)
    raw_input = body.get("input")
    has_compaction_input = input_contains_compaction(raw_input)
    if raw_input is not None:
        body["input"] = sanitize_input(raw_input)

    upstream_url = f"{_get_api_base().rstrip('/')}/responses"
    plan, error_response = _prepare_upstream_request(
        request,
        body=body,
        requested_model=body.get("model"),
        resolved_model=body.get("model"),
        upstream_path="/responses",
        upstream_url=upstream_url,
        header_builder=lambda api_key, request_id: build_responses_headers_for_request(
            request,
            body,
            api_key,
            force_initiator="agent" if has_compaction_input else None,
            request_id=request_id,
        ),
        error_response=openai_error_response,
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
            stream_type="responses",
        )
    return await _post_non_streaming_request(plan, error_response=openai_error_response)


@app.post("/v1/responses/compact")
async def responses_compact(request: Request):
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return openai_error_response(exc.status_code, _http_exception_detail_to_message(exc.detail))

    summary_request = build_fake_compaction_request(body)
    upstream_url = f"{_get_api_base().rstrip('/')}/responses"
    plan, error_response = _prepare_upstream_request(
        request,
        body=summary_request,
        requested_model=body.get("model"),
        resolved_model=summary_request.get("model"),
        upstream_path="/responses",
        upstream_url=upstream_url,
        header_builder=lambda api_key, request_id: build_responses_headers_for_request(
            request,
            summary_request,
            api_key,
            force_initiator="agent",
            request_id=request_id,
        ),
        error_response=openai_error_response,
    )
    if error_response is not None:
        return error_response

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            upstream = await throttled_client_post(
                client,
                plan.upstream_url,
                headers=plan.headers,
                json=plan.body,
            )
    except httpx.RequestError as exc:
        status_code, message = _upstream_request_error_status_and_message(exc)
        _finish_usage_event(plan.usage_event, status_code, response_text=message)
        return openai_error_response(status_code, message)
    except Exception:
        _finish_usage_event(plan.usage_event, 599)
        raise

    if upstream.status_code >= 400:
        _finish_usage_event(
            plan.usage_event,
            upstream.status_code,
            upstream=upstream,
            response_payload=_extract_upstream_json_payload(upstream),
            response_text=_extract_upstream_text(upstream),
        )
        return proxy_non_streaming_response(upstream)

    try:
        upstream_payload = upstream.json()
    except json.JSONDecodeError as e:
        _finish_usage_event(
            plan.usage_event,
            502,
            upstream=upstream,
            response_text=f"Invalid JSON from upstream summarization response: {e}",
        )
        return openai_error_response(502, f"Invalid JSON from upstream summarization response: {e}")

    summary_text = extract_response_output_text(upstream_payload)
    if not summary_text:
        _finish_usage_event(
            plan.usage_event,
            502,
            upstream=upstream,
            response_payload=upstream_payload,
            response_text="Upstream summarization response did not include assistant text output",
        )
        return openai_error_response(502, "Upstream summarization response did not include assistant text output")

    compacted_response = build_fake_compaction_response(body, summary_text, upstream_payload.get("usage"))
    _finish_usage_event(
        plan.usage_event,
        upstream.status_code,
        upstream=upstream,
        response_payload=compacted_response,
    )
    return JSONResponse(content=compacted_response, status_code=200)


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
        return openai_error_response(exc.status_code, _http_exception_detail_to_message(exc.detail))

    messages = body.get("messages", [])

    upstream_url = f"{_get_api_base().rstrip('/')}/chat/completions"
    plan, error_response = _prepare_upstream_request(
        request,
        body=body,
        requested_model=body.get("model"),
        resolved_model=body.get("model"),
        upstream_path="/chat/completions",
        upstream_url=upstream_url,
        header_builder=lambda api_key, request_id: build_chat_headers_for_request(
            request,
            messages,
            body.get("model"),
            api_key,
            request_id=request_id,
        ),
        error_response=openai_error_response,
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
        )
    return await _post_non_streaming_request(plan, error_response=openai_error_response)


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
        return anthropic_error_response(exc.status_code, _http_exception_detail_to_message(exc.detail))

    try:
        api_key = get_api_key()
    except Exception as exc:
        return anthropic_error_response(401, f"GHCP auth failed: {exc}")

    api_base = _get_api_base()
    try:
        outbound_body = await anthropic_request_to_chat(body, api_base, api_key)
    except ValueError as exc:
        return anthropic_error_response(400, str(exc))

    upstream_url = f"{api_base.rstrip('/')}/chat/completions"
    plan, error_response = _prepare_upstream_request(
        request,
        body=outbound_body,
        requested_model=body.get("model"),
        resolved_model=outbound_body.get("model"),
        upstream_path="/chat/completions",
        upstream_url=upstream_url,
        header_builder=lambda resolved_api_key, request_id: build_anthropic_headers_for_request(
            request,
            body,
            resolved_api_key,
            request_id=request_id,
        ),
        error_response=anthropic_error_response,
        api_key=api_key,
    )
    if error_response is not None:
        return error_response

    if outbound_body.get("stream"):
        return await proxy_anthropic_streaming_response(
            plan.upstream_url,
            plan.headers,
            plan.body,
            outbound_body.get("model"),
            timeout=300,
            usage_event=plan.usage_event,
        )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            upstream = await throttled_client_post(
                client,
                plan.upstream_url,
                headers=plan.headers,
                json=plan.body,
            )
        if upstream.status_code >= 400:
            fallback_message = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
            error_payload = anthropic_error_payload_from_openai(_extract_upstream_json_payload(upstream), upstream.status_code, fallback_message)
            _finish_usage_event(
                plan.usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=error_payload,
                response_text=error_payload.get("error", {}).get("message"),
            )
            return anthropic_error_response(
                upstream.status_code,
                error_payload["error"]["message"],
                error_payload["error"]["type"],
                {"retry-after": upstream.headers.get("retry-after")} if upstream.headers.get("retry-after") else None,
            )
        translated = chat_completion_to_anthropic(upstream.json(), fallback_model=outbound_body.get("model"))
        _finish_usage_event(
            plan.usage_event,
            upstream.status_code,
            upstream=upstream,
            response_payload=translated,
        )
        return JSONResponse(content=translated, status_code=upstream.status_code)
    except httpx.RequestError as exc:
        status_code, message = _upstream_request_error_status_and_message(exc)
        _finish_usage_event(plan.usage_event, status_code, response_text=message)
        return anthropic_error_response(status_code, message)
    except Exception:
        _finish_usage_event(plan.usage_event, 599)
        raise


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Run auth interactively in the terminal BEFORE the server starts.
    # If no token exists this will print the device flow prompt and block until
    # the user authorizes (or the script exits on failure).
    ensure_authenticated()

    # Step 2: Start the server in the foreground on this terminal.
    print("Starting GHCP proxy on http://localhost:8000", flush=True)
    print("  Responses API : POST /v1/responses", flush=True)
    print("  Compaction    : POST /v1/responses/compact", flush=True)
    print("  Chat API      : POST /v1/chat/completions", flush=True)
    print("", flush=True)
    print("  Set in your shell:", flush=True)
    print("    export OPENAI_BASE_URL=http://localhost:8000/v1", flush=True)
    print("    export OPENAI_API_KEY=anything", flush=True)
    print("", flush=True)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False, timeout_graceful_shutdown=2)
