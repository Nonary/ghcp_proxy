"""Anthropic Messages SSE translation for upstream chat-completions streams."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator, Callable
from uuid import uuid4

import format_translation


@dataclass
class TextBlockState:
    anthropic_index: int
    closed: bool = False


@dataclass
class ThinkingBlockState:
    anthropic_index: int
    closed: bool = False


@dataclass
class ToolBlockState:
    anthropic_index: int
    id: str
    name: str = ""
    arguments: str = ""
    started: bool = False
    closed: bool = False


class AnthropicStreamTranslator:
    """Translates upstream chat SSE chunks into Anthropic Messages SSE."""

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
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_creation_input_tokens = 0
        self._cache_read_input_tokens = 0
        self._stop_reason: str | None = None
        self._message_started = False
        self._last_message_start_usage: dict | None = None
        self._next_block_index = 0
        self._text_block: TextBlockState | None = None
        self._thinking_block: ThinkingBlockState | None = None
        self._tool_blocks: dict[int, ToolBlockState] = {}
        self._active_block: tuple[str, int] | None = None
        self._response_text_parts: list[str] = []
        self._thinking_text_parts: list[str] = []

    @property
    def response_text(self) -> str | None:
        if not self._response_text_parts:
            return None
        return "".join(self._response_text_parts)

    @property
    def thinking_text(self) -> str | None:
        if not self._thinking_text_parts:
            return None
        return "".join(self._thinking_text_parts)

    def build_response_payload(self) -> dict:
        return {
            "id": self._message_id,
            "type": "message",
            "role": "assistant",
            "model": self._model_name,
            "stop_reason": self._stop_reason or "end_turn",
            "content": (
                [{"type": "text", "text": self.response_text}]
                if self.response_text
                else []
            ),
            "usage": self._usage_payload(),
        }

    def _usage_payload(self) -> dict:
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cache_creation_input_tokens": self._cache_creation_input_tokens,
            "cache_read_input_tokens": self._cache_read_input_tokens,
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

    def _emit_block_stop(self, block: TextBlockState | ThinkingBlockState | ToolBlockState | None) -> list[bytes]:
        if block is None or block.closed:
            return []
        block.closed = True
        return [
            format_translation.sse_encode(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": block.anthropic_index,
                },
            )
        ]

    def _close_active_block(self) -> list[bytes]:
        if self._active_block is None:
            return []

        block_type, block_key = self._active_block
        self._active_block = None
        if block_type == "text":
            return self._emit_block_stop(self._text_block)
        if block_type == "thinking":
            return self._emit_block_stop(self._thinking_block)
        return self._emit_block_stop(self._tool_blocks.get(block_key))

    def _ensure_message_started(self) -> list[bytes]:
        if self._message_started:
            return []
        usage_payload = self._usage_payload()
        self._last_message_start_usage = usage_payload
        self._message_started = True
        return [self._message_start_event(usage_payload)]

    def _refresh_message_started_usage(self) -> list[bytes]:
        if not self._message_started:
            return []

        usage_payload = self._usage_payload()
        if usage_payload == self._last_message_start_usage:
            return []

        self._last_message_start_usage = usage_payload
        return [self._message_start_event(usage_payload)]

    def _ensure_text_block(self) -> list[bytes]:
        if self._text_block is not None and not self._text_block.closed:
            self._active_block = ("text", self._text_block.anthropic_index)
            return []

        events = self._close_active_block()
        self._text_block = TextBlockState(anthropic_index=self._next_block_index)
        self._next_block_index += 1
        self._active_block = ("text", self._text_block.anthropic_index)
        events.append(
            format_translation.sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._text_block.anthropic_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        return events

    def _ensure_thinking_block(self) -> list[bytes]:
        if self._thinking_block is not None and not self._thinking_block.closed:
            self._active_block = ("thinking", self._thinking_block.anthropic_index)
            return []

        events = self._close_active_block()
        self._thinking_block = ThinkingBlockState(anthropic_index=self._next_block_index)
        self._next_block_index += 1
        self._active_block = ("thinking", self._thinking_block.anthropic_index)
        events.append(
            format_translation.sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._thinking_block.anthropic_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
        )
        return events

    def _ensure_tool_block(self, tool_delta: dict) -> list[bytes]:
        openai_index = tool_delta.get("index")
        if not isinstance(openai_index, int):
            openai_index = 0

        state = self._tool_blocks.get(openai_index)
        if state is None:
            state = ToolBlockState(
                anthropic_index=self._next_block_index,
                id=tool_delta.get("id")
                if isinstance(tool_delta.get("id"), str)
                else f"toolu_{uuid4().hex}",
            )
            self._tool_blocks[openai_index] = state
            self._next_block_index += 1

        function = tool_delta.get("function") if isinstance(tool_delta.get("function"), dict) else {}
        if isinstance(tool_delta.get("id"), str):
            state.id = tool_delta["id"]
        if isinstance(function.get("name"), str) and function.get("name"):
            function_name = function["name"]
            if not state.name:
                state.name = function_name
            elif function_name not in state.name:
                state.name += function_name
        if isinstance(function.get("arguments"), str) and function.get("arguments"):
            state.arguments += function["arguments"]

        if state.started and not state.closed:
            self._active_block = ("tool", openai_index)
            return []

        events = self._close_active_block()
        state.started = True
        state.closed = False
        self._active_block = ("tool", openai_index)
        events.append(
            format_translation.sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": state.anthropic_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": state.id,
                        "name": state.name,
                        "input": {},
                    },
                },
            )
        )
        return events

    def _finish_content_blocks(self) -> list[bytes]:
        events = self._close_active_block()
        if self._text_block is None and not self._tool_blocks:
            empty_text_block = TextBlockState(anthropic_index=self._next_block_index)
            events.append(
                format_translation.sse_encode(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": empty_text_block.anthropic_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            events.extend(self._emit_block_stop(empty_text_block))
        return events

    def _update_message_metadata(self, payload: dict):
        if isinstance(payload.get("id"), str):
            self._message_id = payload["id"]
        if self._fallback_model is None and isinstance(payload.get("model"), str):
            self._model_name = payload["model"]

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return

        anthropic_usage = format_translation.chat_usage_to_anthropic(usage)
        self._input_tokens = anthropic_usage.get("input_tokens", self._input_tokens) or self._input_tokens
        self._output_tokens = anthropic_usage.get("output_tokens", self._output_tokens) or self._output_tokens
        self._cache_creation_input_tokens = (
            anthropic_usage.get(
                "cache_creation_input_tokens",
                self._cache_creation_input_tokens,
            )
            or self._cache_creation_input_tokens
        )
        self._cache_read_input_tokens = (
            anthropic_usage.get("cache_read_input_tokens", self._cache_read_input_tokens)
            or self._cache_read_input_tokens
        )

    async def translate(self, byte_iter) -> AsyncIterator[bytes]:
        async for _event_name, data in format_translation.iter_sse_messages(byte_iter):
            if data == "[DONE]":
                break

            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            self._update_message_metadata(payload)

            for event in self._ensure_message_started():
                yield event
            for event in self._refresh_message_started_usage():
                yield event

            choices = payload.get("choices")
            first_choice = choices[0] if isinstance(choices, list) and choices else {}
            delta = first_choice.get("delta") if isinstance(first_choice, dict) else {}

            thinking_delta = format_translation.extract_reasoning_from_chat_delta(delta)
            if thinking_delta:
                self._thinking_text_parts.append(thinking_delta)
                if self._mark_first_output is not None:
                    self._mark_first_output()
                for event in self._ensure_thinking_block():
                    yield event
                yield format_translation.sse_encode(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._thinking_block.anthropic_index,
                        "delta": {"type": "thinking_delta", "thinking": thinking_delta},
                    },
                )

            text_delta = format_translation.extract_text_from_chat_delta(delta)
            if text_delta:
                self._response_text_parts.append(text_delta)
                if self._mark_first_output is not None:
                    self._mark_first_output()
                for event in self._ensure_text_block():
                    yield event
                yield format_translation.sse_encode(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._text_block.anthropic_index,
                        "delta": {"type": "text_delta", "text": text_delta},
                    },
                )

            for tool_delta in format_translation.extract_tool_call_deltas(delta):
                for event in self._ensure_tool_block(tool_delta):
                    yield event
                tool_state = self._tool_blocks.get(
                    tool_delta.get("index") if isinstance(tool_delta.get("index"), int) else 0
                )
                function = tool_delta.get("function") if isinstance(tool_delta.get("function"), dict) else {}
                arguments_chunk = function.get("arguments")
                if isinstance(arguments_chunk, str) and arguments_chunk and tool_state is not None:
                    yield format_translation.sse_encode(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tool_state.anthropic_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": arguments_chunk,
                            },
                        },
                    )

            finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
            mapped_stop_reason = format_translation.chat_stop_reason_to_anthropic(finish_reason)
            if mapped_stop_reason is not None:
                self._stop_reason = mapped_stop_reason

        for event in self._ensure_message_started():
            yield event
        for event in self._refresh_message_started_usage():
            yield event
        for event in self._finish_content_blocks():
            yield event

        yield format_translation.sse_encode(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": self._stop_reason or "end_turn",
                    "stop_sequence": None,
                },
                "usage": self._usage_payload(),
            },
        )
        yield format_translation.sse_encode("message_stop", {"type": "message_stop"})
