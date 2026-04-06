"""Streaming translators for cross-protocol bridge strategies."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable
from uuid import uuid4

import format_translation


@dataclass
class _TextItemState:
    output_index: int
    content_index: int = 0
    started: bool = False
    closed: bool = False


@dataclass
class _ToolItemState:
    output_index: int
    call_id: str
    name: str = ""
    arguments: str = ""
    started: bool = False
    closed: bool = False


class ChatToResponsesStreamTranslator:
    """Translate upstream chat-completions SSE into Responses SSE."""

    def __init__(
        self,
        fallback_model: str | None,
        *,
        mark_first_output: Callable[[], None] | None = None,
    ):
        self._mark_first_output = mark_first_output
        self._fallback_model = fallback_model
        self._response_id = f"resp_{uuid4().hex}"
        self._model_name = fallback_model
        self._created_at = int(time.time())
        self._usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self._response_started = False
        self._text_item: _TextItemState | None = None
        self._tool_items: dict[int, _ToolItemState] = {}
        self._text_parts: list[str] = []

    @property
    def response_text(self) -> str:
        return "".join(self._text_parts)

    def build_response_payload(self) -> dict:
        output = []
        if self.response_text:
            output.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": self.response_text}],
                }
            )
        for openai_index in sorted(self._tool_items):
            tool = self._tool_items[openai_index]
            output.append(
                {
                    "type": "function_call",
                    "call_id": tool.call_id,
                    "name": tool.name,
                    "arguments": tool.arguments,
                }
            )
        return {
            "id": self._response_id,
            "object": "response",
            "created_at": self._created_at,
            "status": "completed",
            "model": self._model_name,
            "output": output,
            "output_text": self.response_text,
            "usage": self._usage,
        }

    def _update_usage(self, payload: dict):
        if isinstance(payload.get("id"), str) and not self._response_started:
            self._response_id = payload["id"].replace("chatcmpl_", "resp_", 1)
        if self._fallback_model is None and isinstance(payload.get("model"), str):
            self._model_name = payload["model"]
        if isinstance(payload.get("usage"), dict):
            self._usage = format_translation.chat_usage_to_response(payload["usage"])

    def _response_created_event(self) -> bytes:
        return format_translation.sse_encode(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": self._response_id,
                    "object": "response",
                    "created_at": self._created_at,
                    "status": "in_progress",
                    "model": self._model_name,
                    "output": [],
                },
            },
        )

    def _ensure_response_started(self) -> list[bytes]:
        if self._response_started:
            return []
        self._response_started = True
        return [self._response_created_event()]

    def _ensure_text_item(self) -> list[bytes]:
        if self._text_item is not None and self._text_item.started and not self._text_item.closed:
            return []
        self._text_item = _TextItemState(output_index=0, started=True)
        return [
            format_translation.sse_encode(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                    },
                },
            ),
            format_translation.sse_encode(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                    },
                },
            ),
        ]

    def _ensure_tool_item(self, tool_delta: dict) -> list[bytes]:
        openai_index = tool_delta.get("index") if isinstance(tool_delta.get("index"), int) else 0
        state = self._tool_items.get(openai_index)
        if state is None:
            state = _ToolItemState(
                output_index=openai_index + 1,
                call_id=tool_delta.get("id") if isinstance(tool_delta.get("id"), str) else f"call_{uuid4().hex}",
            )
            self._tool_items[openai_index] = state

        function = tool_delta.get("function") if isinstance(tool_delta.get("function"), dict) else {}
        if isinstance(tool_delta.get("id"), str):
            state.call_id = tool_delta["id"]
        if isinstance(function.get("name"), str) and function.get("name"):
            state.name += function["name"]
        if isinstance(function.get("arguments"), str) and function.get("arguments"):
            state.arguments += function["arguments"]

        if state.started and not state.closed:
            return []

        state.started = True
        return [
            format_translation.sse_encode(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": state.output_index,
                    "item": {
                        "type": "function_call",
                        "call_id": state.call_id,
                        "name": state.name,
                        "arguments": "",
                    },
                },
            )
        ]

    async def translate(self, byte_iter) -> AsyncIterator[bytes]:
        async for _event_name, data in format_translation.iter_sse_messages(byte_iter):
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            self._update_usage(payload)
            for event in self._ensure_response_started():
                yield event

            choices = payload.get("choices")
            first_choice = choices[0] if isinstance(choices, list) and choices else {}
            delta = first_choice.get("delta") if isinstance(first_choice, dict) else {}

            text_delta = format_translation.extract_text_from_chat_delta(delta)
            if text_delta:
                self._text_parts.append(text_delta)
                if self._mark_first_output is not None:
                    self._mark_first_output()
                for event in self._ensure_text_item():
                    yield event
                yield format_translation.sse_encode(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text_delta,
                    },
                )

            for tool_delta in format_translation.extract_tool_call_deltas(delta):
                if self._mark_first_output is not None:
                    self._mark_first_output()
                for event in self._ensure_tool_item(tool_delta):
                    yield event
                openai_index = tool_delta.get("index") if isinstance(tool_delta.get("index"), int) else 0
                tool_state = self._tool_items.get(openai_index)
                function = tool_delta.get("function") if isinstance(tool_delta.get("function"), dict) else {}
                arguments_chunk = function.get("arguments")
                if isinstance(arguments_chunk, str) and arguments_chunk and tool_state is not None:
                    yield format_translation.sse_encode(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "output_index": tool_state.output_index,
                            "delta": arguments_chunk,
                        },
                    )

        for event in self._ensure_response_started():
            yield event

        if self._text_item is not None and not self._text_item.closed:
            self._text_item.closed = True
            yield format_translation.sse_encode(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "text": self.response_text,
                },
            )
            yield format_translation.sse_encode(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": self.response_text},
                },
            )
            yield format_translation.sse_encode(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": self.response_text}],
                    },
                },
            )

        for openai_index in sorted(self._tool_items):
            tool_state = self._tool_items[openai_index]
            if tool_state.closed:
                continue
            tool_state.closed = True
            yield format_translation.sse_encode(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": tool_state.output_index,
                    "arguments": tool_state.arguments,
                },
            )
            yield format_translation.sse_encode(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": tool_state.output_index,
                    "item": {
                        "type": "function_call",
                        "call_id": tool_state.call_id,
                        "name": tool_state.name,
                        "arguments": tool_state.arguments,
                    },
                },
            )

        yield format_translation.sse_encode(
            "response.completed",
            {
                "type": "response.completed",
                "response": self.build_response_payload(),
            },
        )


class ResponsesToAnthropicStreamTranslator:
    """Translate upstream Responses SSE into Anthropic Messages SSE."""

    def __init__(
        self,
        fallback_model: str | None,
        *,
        mark_first_output: Callable[[], None] | None = None,
    ):
        self._mark_first_output = mark_first_output
        self._fallback_model = fallback_model
        self._message_id = f"msg_{uuid4().hex}"
        self._model_name = fallback_model
        self._usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        self._message_started = False
        self._last_message_start_usage: dict | None = None
        self._next_block_index = 0
        self._text_block_index: int | None = None
        self._text_block_closed = False
        self._tool_blocks: dict[int, _ToolItemState] = {}
        self._text_parts: list[str] = []

    @property
    def response_text(self) -> str | None:
        if not self._text_parts:
            return None
        return "".join(self._text_parts)

    def build_response_payload(self) -> dict:
        content = []
        if self.response_text:
            content.append({"type": "text", "text": self.response_text})
        for output_index in sorted(self._tool_blocks):
            tool = self._tool_blocks[output_index]
            content.append(
                {
                    "type": "tool_use",
                    "id": tool.call_id,
                    "name": tool.name,
                    "input": format_translation._parse_tool_call_arguments(tool.arguments),
                }
            )
        return {
            "id": self._message_id,
            "type": "message",
            "role": "assistant",
            "model": self._model_name,
            "stop_reason": "tool_use" if self._tool_blocks else "end_turn",
            "content": content or [{"type": "text", "text": ""}],
            "usage": self._usage,
        }

    def _message_start_event(self, usage_payload: dict) -> bytes:
        return format_translation.sse_encode(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self._message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self._model_name,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage_payload,
                },
            },
        )

    def _ensure_message_started(self) -> list[bytes]:
        if self._message_started:
            return []
        self._message_started = True
        self._last_message_start_usage = dict(self._usage)
        return [self._message_start_event(self._usage)]

    def _refresh_message_started_usage(self) -> list[bytes]:
        if not self._message_started or self._usage == self._last_message_start_usage:
            return []
        self._last_message_start_usage = dict(self._usage)
        return [self._message_start_event(self._usage)]

    def _ensure_text_block(self) -> list[bytes]:
        if self._text_block_index is not None and not self._text_block_closed:
            return []
        self._text_block_index = self._next_block_index
        self._next_block_index += 1
        self._text_block_closed = False
        return [
            format_translation.sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._text_block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        ]

    def _ensure_tool_block(self, output_index: int, item: dict) -> list[bytes]:
        state = self._tool_blocks.get(output_index)
        if state is None:
            state = _ToolItemState(
                output_index=self._next_block_index,
                call_id=item.get("call_id") if isinstance(item.get("call_id"), str) else f"toolu_{uuid4().hex}",
            )
            self._tool_blocks[output_index] = state
            self._next_block_index += 1
        if isinstance(item.get("call_id"), str):
            state.call_id = item["call_id"]
        if isinstance(item.get("name"), str):
            state.name = item["name"]
        if isinstance(item.get("arguments"), str):
            state.arguments = item["arguments"]

        if state.started and not state.closed:
            return []

        state.started = True
        return [
            format_translation.sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": state.output_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": state.call_id,
                        "name": state.name,
                        "input": {},
                    },
                },
            )
        ]

    def _apply_response_usage(self, usage):
        anthropic_usage = format_translation.response_usage_to_anthropic(usage)
        self._usage = anthropic_usage

    async def translate(self, byte_iter) -> AsyncIterator[bytes]:
        async for _event_name, data in format_translation.iter_sse_messages(byte_iter):
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = str(payload.get("type", "")).lower()
            response = payload.get("response") if isinstance(payload.get("response"), dict) else None
            if isinstance(response, dict):
                if isinstance(response.get("id"), str):
                    self._message_id = response["id"]
                if self._fallback_model is None and isinstance(response.get("model"), str):
                    self._model_name = response["model"]
                if isinstance(response.get("usage"), dict):
                    self._apply_response_usage(response["usage"])

            if isinstance(payload.get("usage"), dict):
                self._apply_response_usage(payload["usage"])

            for event in self._ensure_message_started():
                yield event
            for event in self._refresh_message_started_usage():
                yield event

            if event_type == "response.output_item.added":
                item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
                output_index = payload.get("output_index") if isinstance(payload.get("output_index"), int) else 0
                if str(item.get("type", "")).lower() == "function_call":
                    if self._mark_first_output is not None:
                        self._mark_first_output()
                    for event in self._ensure_tool_block(output_index, item):
                        yield event
                continue

            if event_type == "response.function_call_arguments.delta":
                output_index = payload.get("output_index") if isinstance(payload.get("output_index"), int) else 0
                delta = payload.get("delta")
                state = self._tool_blocks.get(output_index)
                if isinstance(delta, str) and state is not None:
                    state.arguments += delta
                    yield format_translation.sse_encode(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": state.output_index,
                            "delta": {"type": "input_json_delta", "partial_json": delta},
                        },
                    )
                continue

            text_delta = None
            if event_type == "response.output_text.delta":
                text_delta = payload.get("delta")
            elif event_type == "response.output_text.done" and not self._text_parts:
                text_delta = payload.get("text")
            if isinstance(text_delta, str) and text_delta:
                self._text_parts.append(text_delta)
                if self._mark_first_output is not None:
                    self._mark_first_output()
                for event in self._ensure_text_block():
                    yield event
                yield format_translation.sse_encode(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._text_block_index,
                        "delta": {"type": "text_delta", "text": text_delta},
                    },
                )

        for event in self._ensure_message_started():
            yield event
        for event in self._refresh_message_started_usage():
            yield event

        if self._text_block_index is not None and not self._text_block_closed:
            self._text_block_closed = True
            yield format_translation.sse_encode(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": self._text_block_index,
                },
            )

        for output_index in sorted(self._tool_blocks):
            state = self._tool_blocks[output_index]
            if state.closed:
                continue
            state.closed = True
            yield format_translation.sse_encode(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": state.output_index,
                },
            )

        yield format_translation.sse_encode(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use" if self._tool_blocks else "end_turn",
                    "stop_sequence": None,
                },
                "usage": self._usage,
            },
        )
        yield format_translation.sse_encode("message_stop", {"type": "message_stop"})
