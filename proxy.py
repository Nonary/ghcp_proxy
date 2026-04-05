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
import time
import usage_tracking
import util
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
    DEFAULT_UPSTREAM_TIMEOUT_SECONDS,
)

from rate_limiting import (
    throttle_upstream_request,
    throttled_client_post,
    throttled_client_send,
)


# ─── App & Global State ──────────────────────────────────────────────────────

app = FastAPI()

_initiator_policy = InitiatorPolicy()
usage_event_bus = EventBus()


class GracefulStreamingResponse(StreamingResponse):
    """Suppress shutdown/disconnect cancellation noise for long-lived streams."""

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        except asyncio.CancelledError:
            return


def get_initiator_policy() -> InitiatorPolicy:
    return _initiator_policy


def set_initiator_policy(policy: InitiatorPolicy):
    global _initiator_policy
    _initiator_policy = policy


def _current_usage_tracking_state() -> usage_tracking.UsageTrackingState:
    return usage_tracking.current_usage_tracking_state()


usage_event_bus.subscribe(
    usage_tracking.REQUEST_FINISHED_EVENT,
    lambda request_id, finished_at=None: _initiator_policy.note_request_finished(
        request_id,
        finished_at=finished_at,
    ),
)

usage_tracker = usage_tracking.UsageTracker(
    state=_current_usage_tracking_state(),
    archive_store=usage_tracking.UsageArchiveStore(
        init_storage=lambda: dashboard_module._init_sqlite_cache(),
        lock=dashboard_module._sqlite_cache_lock,
        connect=lambda: dashboard_module._sqlite_connect(),
        mark_unavailable=lambda error: dashboard_module._set_sqlite_cache_unavailable(error),
    ),
    event_bus=usage_event_bus,
)
usage_tracking.configure_usage_tracking(
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
        load_api_key_payload=lambda: auth.load_api_key_payload(),
    ),
    usage_tracking_runtime=UsageTrackingRuntimeBindings(
        state_provider=_current_usage_tracking_state,
    ),
)

dashboard_service = dashboard_module.DashboardService(
    dependencies=dashboard_module.DashboardDependencies(
        load_billing_token=lambda: auth.load_billing_token(),
        load_access_token=lambda: auth.load_access_token(),
        load_api_key_payload=lambda: auth.load_api_key_payload(),
        snapshot_all_usage_events=usage_tracker.snapshot_all_usage_events,
        snapshot_usage_events=usage_tracker.snapshot_usage_events,
    ),
    utc_now=util.utc_now,
    utc_now_iso=util.utc_now_iso,
    sqlite_cache_put=lambda cache_key, payload: dashboard_module._sqlite_cache_put(cache_key, payload),
    notify_dashboard_stream_listeners=lambda: dashboard_module._notify_dashboard_stream_listeners(),
    thread_class=lambda: Thread,
)

usage_event_bus.subscribe(
    usage_tracking.USAGE_EVENT_RECORDED_EVENT,
    lambda _event: dashboard_service.notify_dashboard_stream_listeners(),
)


# ─── Module-level parse_json_request wrapper ──────────────────────────────────
# Wraps the util implementation to inject the error callback for recording
# request parsing errors.

async def parse_json_request(request: Request) -> dict:
    return await util.parse_json_request(request, error_callback=usage_tracker.record_request_error)


codex_proxy_status = _runtime.codex_proxy_status
claude_proxy_status = _runtime.claude_proxy_status
write_codex_proxy_config = _runtime.write_codex_proxy_config
write_claude_proxy_settings = _runtime.write_claude_proxy_settings
disable_codex_proxy_config = _runtime.disable_codex_proxy_config
disable_claude_proxy_settings = _runtime.disable_claude_proxy_settings
proxy_client_status_payload = _runtime.proxy_client_status_payload


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

_runtime.initialize(_initiator_policy)
dashboard_module.seed_cached_payloads_from_sqlite()


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
            effective_api_key = auth.get_api_key()
        except Exception as exc:
            return None, error_response(401, f"GHCP auth failed: {exc}")

    headers = header_builder(effective_api_key, request_id)
    usage_tracking.log_proxy_request(request, requested_model, resolved_model, headers.get("X-Initiator"))
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
        async with httpx.AsyncClient(timeout=configured_upstream_timeout_seconds()) as client:
            upstream = await throttled_client_post(
                client,
                plan.upstream_url,
                headers=plan.headers,
                json=plan.body,
            )
        usage_tracker.finish_event(
            plan.usage_event,
            upstream.status_code,
            upstream=upstream,
            response_payload=_extract_upstream_json_payload(upstream),
            response_text=_extract_upstream_text(upstream),
        )
        return proxy_non_streaming_response(upstream)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        usage_tracker.finish_event(plan.usage_event, status_code, response_text=message)
        return error_response(status_code, message)
    except Exception:
        usage_tracker.finish_event(plan.usage_event, 599)
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
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        usage_tracker.finish_event(usage_event, status_code, response_text=message)
        await client.aclose()
        return format_translation.openai_error_response(status_code, message)
    except Exception:
        usage_tracker.finish_event(usage_event, 599)
        await client.aclose()
        raise

    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            usage_tracker.finish_event(
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
        capture = usage_tracker.create_sse_capture(stream_type)
        try:
            async for chunk in upstream.aiter_bytes():
                if capture.feed(chunk):
                    usage_tracker.mark_first_output(usage_event)
                yield chunk
        finally:
            usage_tracker.finish_event(
                usage_event,
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
        usage_tracker.finish_event(usage_event, status_code, response_text=message)
        await client.aclose()
        return format_translation.anthropic_error_response(status_code, message)
    except Exception:
        usage_tracker.finish_event(usage_event, 599)
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
            usage_tracker.finish_event(
                usage_event,
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
            usage_tracker.finish_event(
                usage_event,
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
    queue = dashboard_module._register_dashboard_stream_listener()
    last_version = _runtime.dashboard_stream_version

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
            dashboard_module._unregister_dashboard_stream_listener(queue)

    return GracefulStreamingResponse(
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
    return JSONResponse(content=auth.billing_token_status())


@app.post("/api/config/billing-token")
async def billing_token_config_api(request: Request):
    payload = await parse_json_request(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    if bool(payload.get("clear")):
        auth.clear_billing_token()
        return JSONResponse(content=auth.billing_token_status())

    token = payload.get("token")
    if token is None:
        raise HTTPException(status_code=400, detail="Missing token field")
    if not isinstance(token, str):
        raise HTTPException(status_code=400, detail="Token must be a string")
    token = token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token must not be empty")

    current_status = auth.billing_token_status()
    if current_status.get("readonly"):
        raise HTTPException(
            status_code=409,
            detail="Billing token is configured via GHCP_GITHUB_BILLING_TOKEN and cannot be changed via UI.",
        )

    auth.save_billing_token(token)
    return JSONResponse(content=auth.billing_token_status())


@app.get("/api/config/client-proxy")
async def client_proxy_status_api():
    return JSONResponse(content=proxy_client_status_payload())


@app.post("/api/config/client-proxy")
async def client_proxy_install_api(request: Request):
    payload = await parse_json_request(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")
    targets = auth.normalize_proxy_targets(payload)
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
                    clients[target] = disable_codex_proxy_config()
                else:
                    clients[target] = disable_claude_proxy_settings()
            elif target == "codex":
                clients[target] = write_codex_proxy_config()
            else:
                clients[target] = write_claude_proxy_settings()
        except Exception as exc:
            clients[target] = auth.empty_proxy_status(
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
        return format_translation.openai_error_response(
            exc.status_code,
            format_translation.http_exception_detail_to_message(exc.detail),
        )

    # Sanitize input (multi-turn encrypted_content passthrough)
    raw_input = body.get("input")
    has_compaction_input = format_translation.input_contains_compaction(raw_input)
    if raw_input is not None:
        body["input"] = format_translation.sanitize_input(raw_input)

    upstream_url = f"{auth.get_api_base().rstrip('/')}/responses"
    plan, error_response = _prepare_upstream_request(
        request,
        body=body,
        requested_model=body.get("model"),
        resolved_model=body.get("model"),
        upstream_path="/responses",
        upstream_url=upstream_url,
        header_builder=lambda api_key, request_id: format_translation.build_responses_headers_for_request(
            request,
            body,
            api_key,
            force_initiator="agent" if has_compaction_input else None,
            request_id=request_id,
        ),
        error_response=format_translation.openai_error_response,
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
    return await _post_non_streaming_request(plan, error_response=format_translation.openai_error_response)


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
    upstream_url = f"{auth.get_api_base().rstrip('/')}/responses"
    plan, error_response = _prepare_upstream_request(
        request,
        body=summary_request,
        requested_model=body.get("model"),
        resolved_model=summary_request.get("model"),
        upstream_path="/responses",
        upstream_url=upstream_url,
        header_builder=lambda api_key, request_id: format_translation.build_responses_headers_for_request(
            request,
            summary_request,
            api_key,
            force_initiator="agent",
            request_id=request_id,
        ),
        error_response=format_translation.openai_error_response,
    )
    if error_response is not None:
        return error_response

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
        usage_tracker.finish_event(plan.usage_event, status_code, response_text=message)
        return format_translation.openai_error_response(status_code, message)
    except Exception:
        usage_tracker.finish_event(plan.usage_event, 599)
        raise

    if upstream.status_code >= 400:
        usage_tracker.finish_event(
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
        usage_tracker.finish_event(
            plan.usage_event,
            502,
            upstream=upstream,
            response_text=f"Invalid JSON from upstream summarization response: {e}",
        )
        return format_translation.openai_error_response(502, f"Invalid JSON from upstream summarization response: {e}")

    summary_text = format_translation.extract_response_output_text(upstream_payload)
    if not summary_text:
        usage_tracker.finish_event(
            plan.usage_event,
            502,
            upstream=upstream,
            response_payload=upstream_payload,
            response_text="Upstream summarization response did not include assistant text output",
        )
        return format_translation.openai_error_response(502, "Upstream summarization response did not include assistant text output")

    compacted_response = format_translation.build_fake_compaction_response(body, summary_text, upstream_payload.get("usage"))
    usage_tracker.finish_event(
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
        return format_translation.openai_error_response(
            exc.status_code,
            format_translation.http_exception_detail_to_message(exc.detail),
        )

    messages = body.get("messages", [])

    upstream_url = f"{auth.get_api_base().rstrip('/')}/chat/completions"
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
        ),
        error_response=format_translation.openai_error_response,
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
    except Exception as exc:
        return format_translation.anthropic_error_response(401, f"GHCP auth failed: {exc}")

    api_base = auth.get_api_base()
    try:
        outbound_body = await format_translation.anthropic_request_to_chat(body, api_base, api_key)
    except ValueError as exc:
        return format_translation.anthropic_error_response(400, str(exc))

    upstream_url = f"{api_base.rstrip('/')}/chat/completions"
    plan, error_response = _prepare_upstream_request(
        request,
        body=outbound_body,
        requested_model=body.get("model"),
        resolved_model=outbound_body.get("model"),
        upstream_path="/chat/completions",
        upstream_url=upstream_url,
        header_builder=lambda resolved_api_key, request_id: format_translation.build_anthropic_headers_for_request(
            request,
            body,
            resolved_api_key,
            request_id=request_id,
        ),
        error_response=format_translation.anthropic_error_response,
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
        async with httpx.AsyncClient(timeout=configured_upstream_timeout_seconds()) as client:
            upstream = await throttled_client_post(
                client,
                plan.upstream_url,
                headers=plan.headers,
                json=plan.body,
            )
        if upstream.status_code >= 400:
            fallback_message = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
            error_payload = format_translation.anthropic_error_payload_from_openai(
                _extract_upstream_json_payload(upstream),
                upstream.status_code,
                fallback_message,
            )
            usage_tracker.finish_event(
                plan.usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=error_payload,
                response_text=error_payload.get("error", {}).get("message"),
            )
            return format_translation.anthropic_error_response(
                upstream.status_code,
                error_payload["error"]["message"],
                error_payload["error"]["type"],
                {"retry-after": upstream.headers.get("retry-after")} if upstream.headers.get("retry-after") else None,
            )
        translated = format_translation.chat_completion_to_anthropic(
            upstream.json(),
            fallback_model=outbound_body.get("model"),
        )
        usage_tracker.finish_event(
            plan.usage_event,
            upstream.status_code,
            upstream=upstream,
            response_payload=translated,
        )
        return JSONResponse(content=translated, status_code=upstream.status_code)
    except httpx.RequestError as exc:
        status_code, message = format_translation.upstream_request_error_status_and_message(exc)
        usage_tracker.finish_event(plan.usage_event, status_code, response_text=message)
        return format_translation.anthropic_error_response(status_code, message)
    except Exception:
        usage_tracker.finish_event(plan.usage_event, 599)
        raise


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Run auth interactively in the terminal BEFORE the server starts.
    # If no token exists this will print the device flow prompt and block until
    # the user authorizes (or the script exits on failure).
    auth.ensure_authenticated()

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

    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False, timeout_graceful_shutdown=2)
