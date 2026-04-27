"""Render normalized synthetic replies into client protocol response shapes."""

from __future__ import annotations

import time
from typing import Type
from uuid import uuid4

from fastapi.responses import JSONResponse, Response, StreamingResponse

import format_translation
from upstream_errors import SyntheticReply


def empty_openai_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def empty_anthropic_usage() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def empty_usage_for_protocol(protocol: str | None) -> dict:
    return empty_anthropic_usage() if _normalize_protocol(protocol) == "anthropic" else empty_openai_usage()


def _normalize_protocol(protocol: str | None) -> str:
    normalized = str(protocol or "").strip().lower()
    if normalized in {"anthropic", "chat"}:
        return normalized
    return "responses"


def chat_payload(message: str, model: str | None = None) -> dict:
    return {
        "id": f"chatcmpl_{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": message},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def responses_payload(message: str, model: str | None = None) -> dict:
    return {
        "id": f"resp_{uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": message}],
            }
        ],
        "output_text": message,
        "usage": empty_openai_usage(),
    }


def anthropic_payload(message: str, model: str | None = None) -> dict:
    return {
        "id": f"msg_{uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": message}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": empty_anthropic_usage(),
    }


def build_synthetic_payload(
    reply: SyntheticReply,
    *,
    protocol: str,
    model: str | None = None,
    is_compact: bool = False,
) -> dict:
    normalized = _normalize_protocol(protocol)
    if normalized == "anthropic":
        return anthropic_payload(reply.message, model)
    if normalized == "chat":
        return chat_payload(reply.message, model)
    if is_compact:
        return format_translation.chat_completion_to_compaction_response(
            chat_payload(reply.message, model),
            fallback_model=model,
        )
    return responses_payload(reply.message, model)


async def _responses_stream(reply: SyntheticReply, model: str | None):
    response_payload = responses_payload(reply.message, model)
    response_started = {
        "id": response_payload["id"],
        "object": "response",
        "created_at": response_payload["created_at"],
        "status": "in_progress",
        "model": response_payload.get("model"),
        "output": [],
    }
    item = response_payload["output"][0]
    yield format_translation.sse_encode(
        "response.created",
        {"type": "response.created", "response": response_started},
    )
    yield format_translation.sse_encode(
        "response.output_item.added",
        {"type": "response.output_item.added", "output_index": 0, "item": {**item, "content": []}},
    )
    yield format_translation.sse_encode(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        },
    )
    yield format_translation.sse_encode(
        "response.output_text.delta",
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": reply.message},
    )
    yield format_translation.sse_encode(
        "response.output_text.done",
        {"type": "response.output_text.done", "output_index": 0, "content_index": 0, "text": reply.message},
    )
    yield format_translation.sse_encode(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": reply.message},
        },
    )
    yield format_translation.sse_encode(
        "response.output_item.done",
        {"type": "response.output_item.done", "output_index": 0, "item": item},
    )
    yield format_translation.sse_encode(
        "response.completed",
        {"type": "response.completed", "response": response_payload},
    )
    yield b"data: [DONE]\n\n"


async def _chat_stream(reply: SyntheticReply, model: str | None):
    payload = {
        "id": f"chatcmpl_{uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": reply.message}, "finish_reason": None}],
    }
    yield format_translation.sse_encode("message", payload)
    done_payload = {
        **payload,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    yield format_translation.sse_encode("message", done_payload)
    yield b"data: [DONE]\n\n"


async def _anthropic_stream(reply: SyntheticReply, model: str | None):
    message_id = f"msg_{uuid4().hex}"
    usage = empty_anthropic_usage()
    yield format_translation.sse_encode(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
            },
        },
    )
    yield format_translation.sse_encode(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    )
    yield format_translation.sse_encode(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": reply.message}},
    )
    yield format_translation.sse_encode("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield format_translation.sse_encode(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": usage,
        },
    )
    yield format_translation.sse_encode("message_stop", {"type": "message_stop"})


def render_synthetic_reply(
    reply: SyntheticReply,
    *,
    protocol: str,
    stream: bool,
    model: str | None = None,
    is_compact: bool = False,
    streaming_response_class: Type[StreamingResponse] = StreamingResponse,
) -> Response:
    if not stream:
        payload = build_synthetic_payload(reply, protocol=protocol, model=model, is_compact=is_compact)
        return JSONResponse(content=payload, status_code=reply.client_status)

    normalized = _normalize_protocol(protocol)
    if normalized == "anthropic":
        iterator = _anthropic_stream(reply, model)
    elif normalized == "chat":
        iterator = _chat_stream(reply, model)
    else:
        iterator = _responses_stream(reply, model)
    return streaming_response_class(
        iterator,
        status_code=reply.client_status,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "content-type": "text/event-stream; charset=utf-8",
        },
    )
