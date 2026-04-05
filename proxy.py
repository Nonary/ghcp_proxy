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
from threading import Thread
from uuid import uuid4

import auth as _auth_module
import dashboard as _dashboard_module
import usage_tracking as _usage_tracking_module
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from initiator_policy import InitiatorPolicy

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
    _coerce_float,
    _coerce_int,
    _utc_now,
    _utc_now_iso,
    _normalize_usage_payload,
    _normalize_model_name,
    _usage_event_model_name,
    _usage_event_source,
    _extract_payload_usage,
    _pricing_entry_for_model,
    _usage_event_cost,
    _premium_request_multiplier,
    _server_request_chain_key,
    _is_claude_request,
    _month_key,
    _month_key_for_source_row,
    _extract_item_text,
    _extract_text_content,
    _parse_iso_datetime,
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
    _codex_proxy_status,
    _empty_proxy_status,
    _claude_proxy_status,
    _write_codex_proxy_config,
    _write_claude_proxy_settings,
    _disable_client_proxy_config,
    _disable_codex_proxy_config,
    _disable_claude_proxy_settings,
    _proxy_client_status_payload,
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
    _finish_usage_event,
    _snapshot_usage_events,
    _snapshot_all_usage_events,
    _record_usage_event,
    _record_request_error,
    _rewrite_usage_log_locked,
    _load_usage_history,
    _compact_usage_history_if_needed,
    _usage_event_archive_summary,
    _usage_event_archive_key,
    _load_archived_usage_history,
    _delete_archived_usage_events,
    _SSEUsageCapture,
    _initiator_log_label,
    log_proxy_request,
)

from dashboard import (
    _sqlite_cache_lock,
    _sqlite_cache_enabled,
    _init_sqlite_cache,
    _sqlite_connect,
    _sqlite_cache_get,
    _sqlite_cache_get_latest,
    _sqlite_cache_put,
    _infer_premium_allowance,
    _extract_quota_summary,
    _github_rest_headers,
    _github_rest_get_json,
    _fetch_github_identity,
    _load_cached_github_identity,
    _load_billing_org_candidates,
    _collect_official_premium_payload,
    _empty_official_premium_payload,
    _current_billing_month_bounds,
    _premium_cache_lock,
    _premium_cache,
    _refresh_official_premium_cache_sync,
    _trigger_official_premium_refresh,
    _get_official_premium_payload,
    _seed_cached_payloads_from_sqlite,
    _register_dashboard_stream_listener,
    _unregister_dashboard_stream_listener,
    _notify_dashboard_stream_listeners,
    _dashboard_stream_subscribers,
    _dashboard_stream_lock,
    _dashboard_stream_version,
    _new_usage_aggregate_bucket,
    _ingest_usage_event,
    _finalize_usage_bucket,
    _aggregate_usage_event_buckets,
    _collect_local_dashboard_usage,
    _normalize_session,
    _normalize_month_row,
    _combine_month_rows,
    _combine_usage_rows,
    _build_dashboard_payload,
    _usage_event_group_key,
    _usage_event_session_descriptor,
)


# ─── App & Global State ──────────────────────────────────────────────────────

app = FastAPI()

_initiator_policy = InitiatorPolicy()
_sqlite_cache_error = getattr(_dashboard_module, "_sqlite_cache_error", None)


# ─── Module-level parse_json_request wrapper ──────────────────────────────────
# Wraps the util implementation to inject the error callback for recording
# request parsing errors.

async def parse_json_request(request: Request) -> dict:
    return await _parse_json_request_impl(request, error_callback=_record_request_error)


# ─── Compatibility wrappers for extracted modules ────────────────────────────

def _sync_auth_module():
    _auth_module.CODEX_CONFIG_FILE = CODEX_CONFIG_FILE
    _auth_module.CODEX_CONFIG_DIR = os.path.dirname(CODEX_CONFIG_FILE) or _auth_module.CODEX_CONFIG_DIR
    _auth_module.CODEX_PROXY_CONFIG = CODEX_PROXY_CONFIG
    _auth_module.CLAUDE_SETTINGS_FILE = CLAUDE_SETTINGS_FILE
    _auth_module.CLAUDE_CONFIG_DIR = os.path.dirname(CLAUDE_SETTINGS_FILE) or _auth_module.CLAUDE_CONFIG_DIR
    _auth_module.CLAUDE_PROXY_SETTINGS = CLAUDE_PROXY_SETTINGS
    _auth_module.CLAUDE_MAX_CONTEXT_TOKENS = CLAUDE_MAX_CONTEXT_TOKENS
    _auth_module.CLAUDE_MAX_OUTPUT_TOKENS = CLAUDE_MAX_OUTPUT_TOKENS


def _sync_dashboard_module():
    _dashboard_module._sqlite_cache_lock = _sqlite_cache_lock
    _dashboard_module._sqlite_cache_enabled = _sqlite_cache_enabled
    _dashboard_module._sqlite_cache_error = _sqlite_cache_error
    _dashboard_module._sqlite_connect = _sqlite_connect
    _dashboard_module._sqlite_cache_get = _sqlite_cache_get
    _dashboard_module._sqlite_cache_get_latest = _sqlite_cache_get_latest
    _dashboard_module._sqlite_cache_put = _sqlite_cache_put
    _dashboard_module._premium_cache_lock = _premium_cache_lock
    _dashboard_module._premium_cache = _premium_cache
    _dashboard_module._collect_official_premium_payload = _collect_official_premium_payload
    _dashboard_module._get_official_premium_payload = _get_official_premium_payload
    _dashboard_module._notify_dashboard_stream_listeners = _notify_dashboard_stream_listeners
    _dashboard_module._utc_now = _utc_now
    _dashboard_module._utc_now_iso = _utc_now_iso
    _dashboard_module.Thread = Thread


def _sync_usage_tracking_module():
    _usage_tracking_module._usage_log_lock = _usage_log_lock
    _usage_tracking_module._recent_usage_events = _recent_usage_events
    _usage_tracking_module._archived_usage_events = _archived_usage_events
    _usage_tracking_module._session_request_id_lock = _session_request_id_lock
    _usage_tracking_module._latest_server_request_ids_by_chain = _latest_server_request_ids_by_chain
    _usage_tracking_module._active_server_request_ids_by_request = _active_server_request_ids_by_request
    _usage_tracking_module._latest_claude_user_session_contexts = _latest_claude_user_session_contexts
    _usage_tracking_module._record_usage_event = _record_usage_event
    _usage_tracking_module._rewrite_usage_log_locked = _rewrite_usage_log_locked
    _usage_tracking_module._snapshot_usage_events = _snapshot_usage_events
    _usage_tracking_module._snapshot_all_usage_events = _snapshot_all_usage_events
    _usage_tracking_module._utc_now = _utc_now
    _usage_tracking_module._utc_now_iso = _utc_now_iso
    _usage_tracking_module._extract_payload_usage = _extract_payload_usage
    _usage_tracking_module._normalize_usage_payload = _normalize_usage_payload
    _usage_tracking_module._usage_event_cost = _usage_event_cost
    _usage_tracking_module._premium_request_multiplier = _premium_request_multiplier
    _usage_tracking_module.DETAILED_REQUEST_HISTORY_LIMIT = DETAILED_REQUEST_HISTORY_LIMIT


def _refresh_dashboard_aliases():
    global _sqlite_cache_enabled, _sqlite_cache_error
    _sqlite_cache_enabled = _dashboard_module._sqlite_cache_enabled
    _sqlite_cache_error = getattr(_dashboard_module, "_sqlite_cache_error", None)


def _codex_proxy_status():
    _sync_auth_module()
    return _auth_module._codex_proxy_status()


def _claude_proxy_status():
    _sync_auth_module()
    return _auth_module._claude_proxy_status()


def _write_codex_proxy_config():
    _sync_auth_module()
    return _auth_module._write_codex_proxy_config()


def _write_claude_proxy_settings():
    _sync_auth_module()
    return _auth_module._write_claude_proxy_settings()


def _disable_codex_proxy_config():
    _sync_auth_module()
    return _auth_module._disable_codex_proxy_config()


def _disable_claude_proxy_settings():
    _sync_auth_module()
    return _auth_module._disable_claude_proxy_settings()


def _proxy_client_status_payload():
    _sync_auth_module()
    return _auth_module._proxy_client_status_payload()


def _load_archived_usage_history():
    _sync_dashboard_module()
    _sync_usage_tracking_module()
    result = _usage_tracking_module._load_archived_usage_history()
    _refresh_dashboard_aliases()
    return result


def _load_usage_history():
    _sync_dashboard_module()
    _sync_usage_tracking_module()
    result = _usage_tracking_module._load_usage_history()
    _refresh_dashboard_aliases()
    return result


def _compact_usage_history_if_needed():
    _sync_dashboard_module()
    _sync_usage_tracking_module()
    result = _usage_tracking_module._compact_usage_history_if_needed()
    _refresh_dashboard_aliases()
    return result


def _finish_usage_event(*args, **kwargs):
    _sync_usage_tracking_module()
    result = _usage_tracking_module._finish_usage_event(*args, **kwargs)
    _refresh_dashboard_aliases()
    return result


def _seed_cached_payloads_from_sqlite():
    _sync_dashboard_module()
    result = _dashboard_module._seed_cached_payloads_from_sqlite()
    _refresh_dashboard_aliases()
    return result


def _trigger_official_premium_refresh(force: bool = False):
    _sync_dashboard_module()
    result = _dashboard_module._trigger_official_premium_refresh(force=force)
    _refresh_dashboard_aliases()
    return result


def _build_dashboard_payload(force_refresh: bool = False) -> dict:
    _sync_auth_module()
    _sync_dashboard_module()
    _sync_usage_tracking_module()
    result = _dashboard_module._build_dashboard_payload(force_refresh=force_refresh)
    _refresh_dashboard_aliases()
    return result


# ─── Initialization ──────────────────────────────────────────────────────────

_load_archived_usage_history()
_load_usage_history()
_initiator_policy.seed_from_usage_events(_snapshot_usage_events())
_seed_cached_payloads_from_sqlite()


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
        message_id = f"msg_{uuid4().hex}"
        model_name = fallback_model
        input_tokens = 0
        output_tokens = 0
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0
        stop_reason = None
        message_started = False
        last_message_start_usage = None
        next_block_index = 0
        text_block = None
        tool_blocks = {}
        active_block = None
        stream_closed = False
        response_text_parts: list[str] = []

        async def emit_block_stop(block):
            if not isinstance(block, dict) or block.get("closed") is True:
                return
            block["closed"] = True
            yield _sse_encode(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": block["anthropic_index"],
                },
            )

        async def ensure_message_started():
            nonlocal message_started, last_message_start_usage
            if message_started:
                return
            usage_payload = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
            }
            yield _sse_encode(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model_name,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage_payload,
                    },
                },
            )
            last_message_start_usage = usage_payload
            message_started = True

        async def refresh_message_started_usage():
            nonlocal last_message_start_usage
            if not message_started:
                return

            usage_payload = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
            }
            if usage_payload == last_message_start_usage:
                return

            yield _sse_encode(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model_name,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage_payload,
                    },
                },
            )
            last_message_start_usage = usage_payload

        async def ensure_text_block():
            nonlocal next_block_index, text_block, active_block

            if text_block is not None and text_block.get("closed") is not True:
                active_block = ("text", text_block["anthropic_index"])
                return

            if active_block is not None:
                block_type, block_key = active_block
                active = text_block if block_type == "text" else tool_blocks.get(block_key)
                if active is not None:
                    async for event in emit_block_stop(active):
                        yield event
                active_block = None

            text_block = {
                "anthropic_index": next_block_index,
                "closed": False,
            }
            next_block_index += 1
            active_block = ("text", text_block["anthropic_index"])
            yield _sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": text_block["anthropic_index"],
                    "content_block": {"type": "text", "text": ""},
                },
            )

        async def ensure_tool_block(tool_delta: dict):
            nonlocal next_block_index, active_block

            openai_index = tool_delta.get("index", 0)
            state = tool_blocks.get(openai_index)
            if state is None:
                state = {
                    "anthropic_index": next_block_index,
                    "id": tool_delta.get("id") if isinstance(tool_delta.get("id"), str) else f"toolu_{uuid4().hex}",
                    "name": "",
                    "arguments": "",
                    "closed": False,
                    "started": False,
                }
                tool_blocks[openai_index] = state
                next_block_index += 1

            function = tool_delta.get("function") if isinstance(tool_delta.get("function"), dict) else {}
            if isinstance(tool_delta.get("id"), str):
                state["id"] = tool_delta["id"]
            if isinstance(function.get("name"), str) and function.get("name"):
                if not state["name"]:
                    state["name"] = function["name"]
                elif function["name"] not in state["name"]:
                    state["name"] += function["name"]
            if isinstance(function.get("arguments"), str) and function.get("arguments"):
                state["arguments"] += function["arguments"]

            if state["started"] and state["closed"] is not True:
                active_block = ("tool", openai_index)
                return

            if active_block is not None:
                block_type, block_key = active_block
                active = text_block if block_type == "text" else tool_blocks.get(block_key)
                if active is not None:
                    async for event in emit_block_stop(active):
                        yield event
                active_block = None

            state["started"] = True
            state["closed"] = False
            active_block = ("tool", openai_index)
            yield _sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": state["anthropic_index"],
                    "content_block": {
                        "type": "tool_use",
                        "id": state["id"],
                        "name": state["name"],
                        "input": {},
                    },
                },
            )

        try:
            async for _event_name, data in _iter_sse_messages(upstream.aiter_bytes()):
                if data == "[DONE]":
                    break

                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if isinstance(payload.get("id"), str):
                    message_id = payload["id"]
                if isinstance(payload.get("model"), str):
                    model_name = payload["model"]

                usage = payload.get("usage")
                if isinstance(usage, dict):
                    anthropic_usage = _chat_usage_to_anthropic(usage)
                    input_tokens = anthropic_usage.get("input_tokens", input_tokens) or input_tokens
                    output_tokens = anthropic_usage.get("output_tokens", output_tokens) or output_tokens
                    cache_creation_input_tokens = (
                        anthropic_usage.get("cache_creation_input_tokens", cache_creation_input_tokens)
                        or cache_creation_input_tokens
                    )
                    cache_read_input_tokens = (
                        anthropic_usage.get("cache_read_input_tokens", cache_read_input_tokens) or cache_read_input_tokens
                    )

                async for event in ensure_message_started():
                    yield event
                async for event in refresh_message_started_usage():
                    yield event

                choices = payload.get("choices")
                first_choice = choices[0] if isinstance(choices, list) and choices else {}
                delta = first_choice.get("delta") if isinstance(first_choice, dict) else {}
                text_delta = _extract_text_from_chat_delta(delta)
                if text_delta:
                    response_text_parts.append(text_delta)
                    _mark_usage_event_first_output(usage_event)
                    async for event in ensure_text_block():
                        yield event
                    yield _sse_encode(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": text_block["anthropic_index"],
                            "delta": {"type": "text_delta", "text": text_delta},
                        },
                    )

                for tool_delta in _extract_tool_call_deltas(delta):
                    async for event in ensure_tool_block(tool_delta):
                        yield event
                    tool_state = tool_blocks.get(tool_delta.get("index", 0))
                    function = tool_delta.get("function") if isinstance(tool_delta.get("function"), dict) else {}
                    arguments_chunk = function.get("arguments")
                    if isinstance(arguments_chunk, str) and arguments_chunk:
                        yield _sse_encode(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": tool_state["anthropic_index"],
                                "delta": {"type": "input_json_delta", "partial_json": arguments_chunk},
                            },
                        )

                finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
                mapped_stop_reason = _chat_stop_reason_to_anthropic(finish_reason)
                if mapped_stop_reason is not None:
                    stop_reason = mapped_stop_reason

            async for event in ensure_message_started():
                yield event
            async for event in refresh_message_started_usage():
                yield event

            if active_block is not None:
                block_type, block_key = active_block
                active = text_block if block_type == "text" else tool_blocks.get(block_key)
                if active is not None:
                    async for event in emit_block_stop(active):
                        yield event
                active_block = None

            if text_block is None and not tool_blocks:
                empty_text_block = {
                    "anthropic_index": next_block_index,
                    "closed": False,
                }
                yield _sse_encode(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": empty_text_block["anthropic_index"],
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                async for event in emit_block_stop(empty_text_block):
                    yield event

            yield _sse_encode(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason or "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_creation_input_tokens": cache_creation_input_tokens,
                        "cache_read_input_tokens": cache_read_input_tokens,
                    },
                },
            )
            yield _sse_encode("message_stop", {"type": "message_stop"})
            stream_closed = True
        finally:
            response_payload = {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "stop_reason": stop_reason or "end_turn",
                "content": [{"type": "text", "text": "".join(response_text_parts)}] if response_text_parts else [],
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": cache_creation_input_tokens,
                    "cache_read_input_tokens": cache_read_input_tokens,
                },
            }
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=response_payload,
                response_text="".join(response_text_parts) if response_text_parts else None,
                usage=response_payload["usage"],
            )
            await upstream.aclose()
            await client.aclose()

        if not stream_closed:
            yield _sse_encode("message_stop", {"type": "message_stop"})

    return StreamingResponse(
        stream_translated(),
        status_code=upstream.status_code,
        headers=response_headers,
    )


_trigger_official_premium_refresh()


# ─── Dashboard routes ─────────────────────────────────────────────────────────

def _render_dashboard_html() -> str:
    with open(DASHBOARD_FILE, encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
async def dashboard_root():
    return RedirectResponse(url="/ui", status_code=307)


@app.get("/ui", response_class=HTMLResponse)
async def dashboard():
    return FileResponse(DASHBOARD_FILE, media_type="text/html")


@app.get("/api/dashboard")
async def dashboard_api(request: Request):
    refresh = request.query_params.get("refresh", "").lower() in {"1", "true", "yes"}
    payload = await asyncio.to_thread(_build_dashboard_payload, refresh)
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


@app.get("/api/dashboard/stream")
async def dashboard_stream(request: Request):
    heartbeat_seconds = 20
    poll_seconds = 1.0
    queue = _register_dashboard_stream_listener()
    last_version = _dashboard_stream_version

    async def stream():
        nonlocal last_version
        last_heartbeat = time.monotonic()
        try:
            initial_payload = await asyncio.to_thread(_build_dashboard_payload, False)
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
                payload = await asyncio.to_thread(_build_dashboard_payload, False)
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
    request_id = uuid4().hex

    # Sanitize input (multi-turn encrypted_content passthrough)
    raw_input = body.get("input")
    has_compaction_input = input_contains_compaction(raw_input)
    if raw_input is not None:
        body["input"] = sanitize_input(raw_input)

    try:
        api_key = get_api_key()
    except Exception as e:
        return openai_error_response(401, f"GHCP auth failed: {e}")

    headers = build_responses_headers_for_request(
        request,
        body,
        api_key,
        force_initiator="agent" if has_compaction_input else None,
        request_id=request_id,
    )
    upstream_url = f"{_get_api_base().rstrip('/')}/responses"
    is_streaming = body.get("stream", False)
    log_proxy_request(request, body.get("model"), body.get("model"), headers.get("X-Initiator"))
    usage_event = _start_usage_event(
        request,
        body.get("model"),
        body.get("model"),
        headers.get("X-Initiator"),
        request_id=request_id,
        request_body=body,
        upstream_path="/responses",
        outbound_headers=headers,
    )

    if is_streaming:
        return await proxy_streaming_response(
            upstream_url,
            headers,
            body,
            timeout=300,
            usage_event=usage_event,
            stream_type="responses",
        )
    else:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                upstream = await throttled_client_post(client, upstream_url, headers=headers, json=body)
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=_extract_upstream_json_payload(upstream),
                response_text=_extract_upstream_text(upstream),
            )
            return proxy_non_streaming_response(upstream)
        except httpx.RequestError as exc:
            status_code, message = _upstream_request_error_status_and_message(exc)
            _finish_usage_event(usage_event, status_code, response_text=message)
            return openai_error_response(status_code, message)
        except Exception:
            _finish_usage_event(usage_event, 599)
            raise


@app.post("/v1/responses/compact")
async def responses_compact(request: Request):
    try:
        body = await parse_json_request(request)
    except HTTPException as exc:
        return openai_error_response(exc.status_code, _http_exception_detail_to_message(exc.detail))
    request_id = uuid4().hex

    try:
        api_key = get_api_key()
    except Exception as e:
        return openai_error_response(401, f"GHCP auth failed: {e}")

    summary_request = build_fake_compaction_request(body)
    headers = build_responses_headers_for_request(
        request,
        summary_request,
        api_key,
        force_initiator="agent",
        request_id=request_id,
    )
    upstream_url = f"{_get_api_base().rstrip('/')}/responses"
    log_proxy_request(request, body.get("model"), summary_request.get("model"), headers.get("X-Initiator"))
    usage_event = _start_usage_event(
        request,
        body.get("model"),
        summary_request.get("model"),
        headers.get("X-Initiator"),
        request_id=request_id,
        request_body=summary_request,
        upstream_path="/responses",
        outbound_headers=headers,
    )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            upstream = await throttled_client_post(client, upstream_url, headers=headers, json=summary_request)
    except httpx.RequestError as exc:
        status_code, message = _upstream_request_error_status_and_message(exc)
        _finish_usage_event(usage_event, status_code, response_text=message)
        return openai_error_response(status_code, message)
    except Exception:
        _finish_usage_event(usage_event, 599)
        raise

    if upstream.status_code >= 400:
        _finish_usage_event(
            usage_event,
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
            usage_event,
            502,
            upstream=upstream,
            response_text=f"Invalid JSON from upstream summarization response: {e}",
        )
        return openai_error_response(502, f"Invalid JSON from upstream summarization response: {e}")

    summary_text = extract_response_output_text(upstream_payload)
    if not summary_text:
        _finish_usage_event(
            usage_event,
            502,
            upstream=upstream,
            response_payload=upstream_payload,
            response_text="Upstream summarization response did not include assistant text output",
        )
        return openai_error_response(502, "Upstream summarization response did not include assistant text output")

    compacted_response = build_fake_compaction_response(body, summary_text, upstream_payload.get("usage"))
    _finish_usage_event(
        usage_event,
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
    request_id = uuid4().hex

    messages = body.get("messages", [])

    try:
        api_key = get_api_key()
    except Exception as e:
        return openai_error_response(401, f"GHCP auth failed: {e}")

    headers = build_chat_headers_for_request(request, messages, body.get("model"), api_key, request_id=request_id)

    upstream_url = f"{_get_api_base().rstrip('/')}/chat/completions"
    is_streaming = body.get("stream", False)
    log_proxy_request(request, body.get("model"), body.get("model"), headers.get("X-Initiator"))
    usage_event = _start_usage_event(
        request,
        body.get("model"),
        body.get("model"),
        headers.get("X-Initiator"),
        request_id=request_id,
        request_body=body,
        upstream_path="/chat/completions",
        outbound_headers=headers,
    )

    if is_streaming:
        return await proxy_streaming_response(
            upstream_url,
            headers,
            body,
            timeout=300,
            usage_event=usage_event,
            stream_type="chat",
        )
    else:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                upstream = await throttled_client_post(client, upstream_url, headers=headers, json=body)
            _finish_usage_event(
                usage_event,
                upstream.status_code,
                upstream=upstream,
                response_payload=_extract_upstream_json_payload(upstream),
                response_text=_extract_upstream_text(upstream),
            )
            return proxy_non_streaming_response(upstream)
        except httpx.RequestError as exc:
            status_code, message = _upstream_request_error_status_and_message(exc)
            _finish_usage_event(usage_event, status_code, response_text=message)
            return openai_error_response(status_code, message)
        except Exception:
            _finish_usage_event(usage_event, 599)
            raise


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
    request_id = uuid4().hex

    try:
        api_key = get_api_key()
    except Exception as e:
        return anthropic_error_response(401, f"GHCP auth failed: {e}")

    api_base = _get_api_base()
    try:
        outbound_body = await anthropic_request_to_chat(body, api_base, api_key)
    except ValueError as e:
        return anthropic_error_response(400, str(e))

    headers = build_anthropic_headers_for_request(request, body, api_key, request_id=request_id)
    upstream_url = f"{api_base.rstrip('/')}/chat/completions"
    log_proxy_request(request, body.get("model"), outbound_body.get("model"), headers.get("X-Initiator"))
    usage_event = _start_usage_event(
        request,
        body.get("model"),
        outbound_body.get("model"),
        headers.get("X-Initiator"),
        request_id=request_id,
        request_body=outbound_body,
        upstream_path="/chat/completions",
        outbound_headers=headers,
    )

    if outbound_body.get("stream"):
        return await proxy_anthropic_streaming_response(
            upstream_url,
            headers,
            outbound_body,
            outbound_body.get("model"),
            timeout=300,
            usage_event=usage_event,
        )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            upstream = await throttled_client_post(client, upstream_url, headers=headers, json=outbound_body)
        if upstream.status_code >= 400:
            fallback_message = _extract_upstream_text(upstream) or f"Upstream request failed with status {upstream.status_code}"
            error_payload = anthropic_error_payload_from_openai(_extract_upstream_json_payload(upstream), upstream.status_code, fallback_message)
            _finish_usage_event(
                usage_event,
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
            usage_event,
            upstream.status_code,
            upstream=upstream,
            response_payload=translated,
        )
        return JSONResponse(content=translated, status_code=upstream.status_code)
    except httpx.RequestError as exc:
        status_code, message = _upstream_request_error_status_and_message(exc)
        _finish_usage_event(usage_event, status_code, response_text=message)
        return anthropic_error_response(status_code, message)
    except Exception:
        _finish_usage_event(usage_event, 599)
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
