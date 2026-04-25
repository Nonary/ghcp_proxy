"""Protocol-aware SSE notice injection for update prompts."""

from __future__ import annotations

import json
import time
from typing import AsyncIterator
from uuid import uuid4

import format_translation


def update_notice_text() -> str:
    return (
        "By the way, an update is available for GHCP Proxy. "
        "Visit http://localhost:8000 to update the proxy from the dashboard."
    )


async def inject_text_notice(byte_iter, protocol: str, notice_text: str) -> AsyncIterator[bytes]:
    text = str(notice_text or "").strip()
    if not text:
        async for chunk in byte_iter:
            yield chunk
        return

    normalized = str(protocol or "").strip().lower()
    if normalized == "responses":
        async for chunk in _inject_responses_notice(byte_iter, text):
            yield chunk
        return
    if normalized == "chat":
        async for chunk in _inject_chat_notice(byte_iter, text):
            yield chunk
        return
    if normalized == "anthropic":
        async for chunk in _inject_anthropic_notice(byte_iter, text):
            yield chunk
        return

    async for chunk in byte_iter:
        yield chunk


def _responses_notice_events(notice_text: str, *, output_index: int) -> list[bytes]:
    text = notice_text + "\n\n"
    item_id = f"msg_{uuid4().hex}"
    return [
        format_translation.sse_encode(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "type": "message",
                    "id": item_id,
                    "role": "assistant",
                    "content": [],
                },
            },
        ),
        format_translation.sse_encode(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            },
        ),
        format_translation.sse_encode(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            },
        ),
        format_translation.sse_encode(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            },
        ),
        format_translation.sse_encode(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text},
            },
        ),
        format_translation.sse_encode(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {
                    "type": "message",
                    "id": item_id,
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                },
            },
        ),
    ]


async def _inject_responses_notice(byte_iter, notice_text: str):
    inserted = False
    next_output_index = 0
    async for event_name, data in format_translation.iter_sse_messages(byte_iter):
        if data == "[DONE]":
            if not inserted:
                for event in _responses_notice_events(notice_text, output_index=next_output_index):
                    yield event
                inserted = True
            yield b"data: [DONE]\n\n"
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue

        output_index = payload.get("output_index")
        if isinstance(output_index, int):
            next_output_index = max(next_output_index, output_index + 1)

        event_type = str(payload.get("type") or event_name or "").lower()
        if event_type == "response.completed" and not inserted:
            for event in _responses_notice_events(notice_text, output_index=next_output_index):
                yield event
            inserted = True
            response = payload.get("response") if isinstance(payload.get("response"), dict) else None
            if isinstance(response, dict):
                response = dict(response)
                output = response.get("output")
                notice_output_text = notice_text + "\n\n"
                notice_item = {
                    "type": "message",
                    "id": f"msg_{uuid4().hex}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": notice_output_text}],
                }
                response["output"] = [*(output if isinstance(output, list) else []), notice_item]
                if isinstance(response.get("output_text"), str):
                    response["output_text"] = response["output_text"] + notice_output_text
                payload = dict(payload)
                payload["response"] = response

        yield format_translation.sse_encode(event_name or event_type, payload)

    if not inserted:
        for event in _responses_notice_events(notice_text, output_index=next_output_index):
            yield event


async def _inject_chat_notice(byte_iter, notice_text: str):
    buffered_terminal = None
    template = {}
    inserted = False
    async for event_name, data in format_translation.iter_sse_messages(byte_iter):
        if data == "[DONE]":
            if not inserted:
                yield _chat_notice_event(template, notice_text, event_name="message")
                inserted = True
            if buffered_terminal is not None:
                yield buffered_terminal
                buffered_terminal = None
            yield b"data: [DONE]\n\n"
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict):
            template = payload
        if _chat_payload_is_terminal(payload):
            if buffered_terminal is not None:
                if not inserted:
                    yield _chat_notice_event(template, notice_text, event_name="message")
                    inserted = True
                yield buffered_terminal
            buffered_terminal = format_translation.sse_encode(event_name or "message", payload)
            continue

        if buffered_terminal is not None:
            if not inserted:
                yield _chat_notice_event(template, notice_text, event_name="message")
                inserted = True
            yield buffered_terminal
            buffered_terminal = None
        yield format_translation.sse_encode(event_name or "message", payload)

    if not inserted:
        yield _chat_notice_event(template, notice_text, event_name="message")
    if buffered_terminal is not None:
        yield buffered_terminal


def _chat_payload_is_terminal(payload: dict) -> bool:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    if not isinstance(first_choice, dict):
        return False
    return first_choice.get("finish_reason") is not None


def _chat_notice_event(template: dict, notice_text: str, *, event_name: str) -> bytes:
    payload = {
        "id": template.get("id") if isinstance(template.get("id"), str) else f"chatcmpl_{uuid4().hex}",
        "object": template.get("object") or "chat.completion.chunk",
        "created": template.get("created") or int(time.time()),
        "model": template.get("model"),
        "choices": [
            {
                "index": 0,
                "delta": {"content": notice_text + "\n\n"},
                "finish_reason": None,
            }
        ],
    }
    return format_translation.sse_encode(event_name or "message", payload)


async def _inject_anthropic_notice(byte_iter, notice_text: str):
    inserted = False
    next_index = 0
    async for event_name, data in format_translation.iter_sse_messages(byte_iter):
        if data == "[DONE]":
            if not inserted:
                for event in _anthropic_notice_events(notice_text, index=next_index):
                    yield event
                inserted = True
            yield b"data: [DONE]\n\n"
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue

        evt = str(event_name or payload.get("type") or "").lower()
        index = payload.get("index") if isinstance(payload, dict) else None
        if isinstance(index, int):
            next_index = max(next_index, index + 1)

        if evt in {"message_delta", "message_stop"} and not inserted:
            for event in _anthropic_notice_events(notice_text, index=next_index):
                yield event
            inserted = True

        yield format_translation.sse_encode(event_name or evt, payload)

    if not inserted:
        for event in _anthropic_notice_events(notice_text, index=next_index):
            yield event


def _anthropic_notice_events(notice_text: str, *, index: int) -> list[bytes]:
    text = notice_text + "\n\n"
    return [
        format_translation.sse_encode(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_translation.sse_encode(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        format_translation.sse_encode(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        ),
    ]
