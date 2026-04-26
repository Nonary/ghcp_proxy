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


@dataclass
class _ReasoningItemState:
    """Tracks a Responses-style reasoning output item being streamed."""

    output_index: int
    item_id: str = ""
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
        self._reasoning_item: _ReasoningItemState | None = None
        self._reasoning_parts: list[str] = []
        self._next_output_index = 0

    @property
    def response_text(self) -> str:
        return "".join(self._text_parts)

    @property
    def reasoning_text(self) -> str:
        return "".join(self._reasoning_parts)

    def build_response_payload(self) -> dict:
        output = []
        if self.reasoning_text:
            reasoning_id = (
                self._reasoning_item.item_id
                if self._reasoning_item is not None and self._reasoning_item.item_id
                else f"rs_{uuid4().hex}"
            )
            output.append(
                {
                    "type": "reasoning",
                    "id": reasoning_id,
                    "summary": [{"type": "summary_text", "text": self.reasoning_text}],
                    "content": [{"type": "reasoning_text", "text": self.reasoning_text}],
                    "encrypted_content": None,
                }
            )
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
        output_index = self._next_output_index
        self._next_output_index += 1
        self._text_item = _TextItemState(output_index=output_index, started=True)
        return [
            format_translation.sse_encode(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
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
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                    },
                },
            ),
        ]

    def _ensure_reasoning_item(self) -> list[bytes]:
        if self._reasoning_item is not None and self._reasoning_item.started and not self._reasoning_item.closed:
            return []
        output_index = self._next_output_index
        self._next_output_index += 1
        item_id = f"rs_{uuid4().hex}"
        self._reasoning_item = _ReasoningItemState(
            output_index=output_index, item_id=item_id, started=True
        )
        return [
            format_translation.sse_encode(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "type": "reasoning",
                        "id": item_id,
                        "summary": [],
                        "content": [],
                        "encrypted_content": None,
                    },
                },
            ),
            format_translation.sse_encode(
                "response.reasoning_summary_part.added",
                {
                    "type": "response.reasoning_summary_part.added",
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                },
            ),
        ]

    def _ensure_tool_item(self, tool_delta: dict) -> list[bytes]:
        openai_index = tool_delta.get("index") if isinstance(tool_delta.get("index"), int) else 0
        state = self._tool_items.get(openai_index)
        if state is None:
            state = _ToolItemState(
                output_index=self._next_output_index,
                call_id=tool_delta.get("id") if isinstance(tool_delta.get("id"), str) else f"call_{uuid4().hex}",
            )
            self._tool_items[openai_index] = state
            self._next_output_index += 1

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

            reasoning_delta = format_translation.extract_reasoning_from_chat_delta(delta)
            if reasoning_delta:
                if self._mark_first_output is not None:
                    self._mark_first_output()
                # Codex's `new_reasoning_summary_block` drops cells whose text
                # has no `**bold**` header; Claude's reasoning_text doesn't
                # include one, so prepend a synthetic "**Thinking**" header.
                first_chunk_needs_header = not self._reasoning_parts
                for event in self._ensure_reasoning_item():
                    yield event
                if first_chunk_needs_header:
                    header = "**Thinking**\n\n"
                    self._reasoning_parts.append(header)
                    yield format_translation.sse_encode(
                        "response.reasoning_summary_text.delta",
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": self._reasoning_item.item_id,
                            "output_index": self._reasoning_item.output_index,
                            "summary_index": 0,
                            "delta": header,
                        },
                    )
                self._reasoning_parts.append(reasoning_delta)
                yield format_translation.sse_encode(
                    "response.reasoning_summary_text.delta",
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": self._reasoning_item.item_id,
                        "output_index": self._reasoning_item.output_index,
                        "summary_index": 0,
                        "delta": reasoning_delta,
                    },
                )

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
                        "output_index": self._text_item.output_index,
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

        if self._reasoning_item is not None and not self._reasoning_item.closed:
            self._reasoning_item.closed = True
            reasoning_text = self.reasoning_text
            yield format_translation.sse_encode(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": self._reasoning_item.item_id,
                    "output_index": self._reasoning_item.output_index,
                    "summary_index": 0,
                    "text": reasoning_text,
                },
            )
            yield format_translation.sse_encode(
                "response.reasoning_summary_part.done",
                {
                    "type": "response.reasoning_summary_part.done",
                    "item_id": self._reasoning_item.item_id,
                    "output_index": self._reasoning_item.output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": reasoning_text},
                },
            )
            yield format_translation.sse_encode(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": self._reasoning_item.output_index,
                    "item": {
                        "type": "reasoning",
                        "id": self._reasoning_item.item_id,
                        "summary": [{"type": "summary_text", "text": reasoning_text}],
                        "content": [{"type": "reasoning_text", "text": reasoning_text}],
                        "encrypted_content": None,
                    },
                },
            )

        if self._text_item is not None and not self._text_item.closed:
            self._text_item.closed = True
            text_output_index = self._text_item.output_index
            yield format_translation.sse_encode(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "output_index": text_output_index,
                    "content_index": 0,
                    "text": self.response_text,
                },
            )
            yield format_translation.sse_encode(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "output_index": text_output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": self.response_text},
                },
            )
            yield format_translation.sse_encode(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": text_output_index,
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
        self._thinking_block_index: int | None = None
        self._thinking_block_closed = False
        self._tool_blocks: dict[int, _ToolItemState] = {}
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []

    @property
    def response_text(self) -> str | None:
        if not self._text_parts:
            return None
        return "".join(self._text_parts)

    @property
    def thinking_text(self) -> str | None:
        if not self._thinking_parts:
            return None
        return "".join(self._thinking_parts)

    def build_response_payload(self) -> dict:
        content = []
        if self.thinking_text:
            content.append({"type": "thinking", "thinking": self.thinking_text})
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
        events = self._close_thinking_block()
        self._text_block_index = self._next_block_index
        self._next_block_index += 1
        self._text_block_closed = False
        events.append(
            format_translation.sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._text_block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        return events

    def _ensure_thinking_block(self) -> list[bytes]:
        if self._thinking_block_index is not None and not self._thinking_block_closed:
            return []
        self._thinking_block_index = self._next_block_index
        self._next_block_index += 1
        self._thinking_block_closed = False
        return [
            format_translation.sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._thinking_block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
        ]

    def _close_thinking_block(self) -> list[bytes]:
        if self._thinking_block_index is None or self._thinking_block_closed:
            return []
        self._thinking_block_closed = True
        return [
            format_translation.sse_encode(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": self._thinking_block_index,
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
                elif str(item.get("type", "")).lower() == "reasoning":
                    # Some providers emit a reasoning item with the full summary
                    # already populated. Forward any inline summary text now.
                    summary = item.get("summary")
                    if isinstance(summary, list):
                        for entry in summary:
                            text = entry.get("text") if isinstance(entry, dict) else None
                            if isinstance(text, str) and text:
                                self._thinking_parts.append(text)
                                if self._mark_first_output is not None:
                                    self._mark_first_output()
                                for event in self._ensure_thinking_block():
                                    yield event
                                yield format_translation.sse_encode(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": self._thinking_block_index,
                                        "delta": {"type": "thinking_delta", "thinking": text},
                                    },
                                )
                continue

            if event_type in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning.delta",
            ):
                reasoning_delta = payload.get("delta")
                if isinstance(reasoning_delta, dict):
                    reasoning_delta = reasoning_delta.get("text")
                if isinstance(reasoning_delta, str) and reasoning_delta:
                    self._thinking_parts.append(reasoning_delta)
                    if self._mark_first_output is not None:
                        self._mark_first_output()
                    for event in self._ensure_thinking_block():
                        yield event
                    yield format_translation.sse_encode(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._thinking_block_index,
                            "delta": {"type": "thinking_delta", "thinking": reasoning_delta},
                        },
                    )
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

        for event in self._close_thinking_block():
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

# ---------------------------------------------------------------------------
# AnthropicToResponsesStreamTranslator
# ---------------------------------------------------------------------------


def _anthropic_usage_to_responses_usage(usage) -> dict:
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    in_t = usage.get("input_tokens")
    out_t = usage.get("output_tokens")
    in_t = int(in_t) if isinstance(in_t, (int, float)) else 0
    out_t = int(out_t) if isinstance(out_t, (int, float)) else 0
    cached = usage.get("cache_read_input_tokens")
    cached = int(cached) if isinstance(cached, (int, float)) else 0
    cache_create = usage.get("cache_creation_input_tokens")
    cache_create = int(cache_create) if isinstance(cache_create, (int, float)) else 0
    out: dict = {
        "input_tokens": in_t,
        "output_tokens": out_t,
        "total_tokens": in_t + out_t,
    }
    details: dict = {}
    if cached:
        details["cached_tokens"] = cached
    if cache_create:
        details["cache_creation_input_tokens"] = cache_create
    if details:
        out["input_tokens_details"] = details
    return out


def _anthropic_stop_reason_to_responses_status(stop_reason) -> str:
    """Map Anthropic stop_reason to a Responses ``status`` value."""
    norm = (str(stop_reason).strip().lower() if stop_reason is not None else "")
    if norm in ("end_turn", "stop_sequence", "tool_use", "pause_turn", ""):
        return "completed"
    if norm in ("max_tokens", "refusal"):
        return "incomplete"
    return "completed"


def _anthropic_stop_reason_to_incomplete_details(stop_reason):
    """Return the matching Responses ``incomplete_details`` block, or None."""
    norm = (str(stop_reason).strip().lower() if stop_reason is not None else "")
    if norm == "max_tokens":
        return {"reason": "max_output_tokens"}
    if norm == "refusal":
        return {"reason": "content_filter"}
    return None


@dataclass
class _AnthropicTextBlockState:
    output_index: int
    item_id: str
    text_parts: list
    closed: bool = False


@dataclass
class _AnthropicReasoningBlockState:
    output_index: int
    item_id: str
    summary_parts: list
    encrypted_parts: list
    closed: bool = False


@dataclass
class _AnthropicToolBlockState:
    output_index: int
    item_id: str
    call_id: str
    name: str
    arguments_parts: list
    streamed_args: bool = False
    closed: bool = False


class AnthropicToResponsesStreamTranslator:
    """Translate upstream Anthropic Messages SSE into Responses SSE."""

    def __init__(
        self,
        *,
        request_id: str | None = None,
        model: str | None = None,
        created_at: int | None = None,
        mark_first_output: Callable[[], None] | None = None,
    ):
        self._mark_first_output = mark_first_output
        self._response_id = request_id or f"resp_{uuid4().hex}"
        self._model_name = model
        self._created_at = created_at if isinstance(created_at, int) else int(time.time())
        self._usage: dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        # Raw upstream Anthropic-shape usage retained so the proxy can derive
        # the internal tracking shape (with pricing_* counters) without
        # round-tripping through the Responses-shape ``self._usage`` (which
        # collapses cache writes into details and would lose them when the
        # downstream tracker re-normalizes).
        self._anthropic_raw_usage: dict = {}
        self._stop_reason: str | None = None
        self._response_started = False
        self._completed = False
        self._failed = False
        # Block index (anthropic) -> state
        self._blocks: dict = {}
        # Order of finalized output items, by anthropic block index
        self._output_order: list = []
        self._next_output_index = 0

    # ------------------------------------------------------------------
    # Public properties used by the proxy for usage/trace summaries
    # ------------------------------------------------------------------

    @property
    def response_text(self) -> str | None:
        parts = []
        for idx in self._output_order:
            state = self._blocks.get(idx)
            if isinstance(state, _AnthropicTextBlockState):
                parts.append("".join(state.text_parts))
        if not parts:
            return None
        return "".join(parts)

    @property
    def reasoning_text(self) -> str | None:
        parts = []
        for idx in self._output_order:
            state = self._blocks.get(idx)
            if isinstance(state, _AnthropicReasoningBlockState):
                parts.append("".join(state.summary_parts))
        if not parts:
            return None
        return "".join(parts)

    @property
    def anthropic_raw_usage(self) -> dict:
        """Latest raw Anthropic-shape usage seen on message_start/message_delta.

        Empty dict if no usage was reported. Caller should treat empty as
        "no usage available" rather than as zero counters.
        """
        return dict(self._anthropic_raw_usage)

    def build_response_payload(self) -> dict:
        output: list = []
        for idx in self._output_order:
            state = self._blocks.get(idx)
            if isinstance(state, _AnthropicTextBlockState):
                text = "".join(state.text_parts)
                output.append(
                    {
                        "type": "message",
                        "id": state.item_id,
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            elif isinstance(state, _AnthropicReasoningBlockState):
                summary = "".join(state.summary_parts)
                raw_signature = "".join(state.encrypted_parts)
                enc_part, parsed_id = format_translation._split_reasoning_signature(raw_signature)
                encrypted = enc_part or None
                reasoning_id = parsed_id or state.item_id
                output.append(
                    {
                        "type": "reasoning",
                        "id": reasoning_id,
                        "summary": [{"type": "summary_text", "text": summary}],
                        "content": [{"type": "reasoning_text", "text": summary}],
                        "encrypted_content": encrypted,
                    }
                )
            elif isinstance(state, _AnthropicToolBlockState):
                output.append(
                    {
                        "type": "function_call",
                        "id": state.item_id,
                        "call_id": state.call_id,
                        "name": state.name,
                        "arguments": "".join(state.arguments_parts),
                    }
                )
        status = _anthropic_stop_reason_to_responses_status(self._stop_reason)
        incomplete_details = _anthropic_stop_reason_to_incomplete_details(self._stop_reason)
        # Aggregate output_text from any assistant message blocks (parity with
        # chat_completion_to_response and the non-streaming Anthropic translator).
        output_text_parts: list = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    stripped = part["text"].strip()
                    if stripped:
                        output_text_parts.append(stripped)
        result = {
            "id": self._response_id,
            "object": "response",
            "created_at": self._created_at,
            "status": status,
            "model": self._model_name,
            "output": output,
            "output_text": "\n\n".join(output_text_parts),
            "usage": self._usage,
        }
        if incomplete_details is not None:
            result["incomplete_details"] = incomplete_details
        return result

    # ------------------------------------------------------------------
    # Internal emission helpers
    # ------------------------------------------------------------------

    def _ensure_response_started(self) -> list:
        if self._response_started:
            return []
        self._response_started = True
        snapshot = {
            "id": self._response_id,
            "object": "response",
            "created_at": self._created_at,
            "status": "in_progress",
            "model": self._model_name,
            "output": [],
        }
        return [
            format_translation.sse_encode(
                "response.created",
                {"type": "response.created", "response": snapshot},
            ),
            format_translation.sse_encode(
                "response.in_progress",
                {"type": "response.in_progress", "response": snapshot},
            ),
        ]

    def _open_text_block(self, anth_index: int, content_block: dict) -> list:
        output_index = self._next_output_index
        self._next_output_index += 1
        item_id = f"msg_{uuid4().hex}"
        state = _AnthropicTextBlockState(
            output_index=output_index,
            item_id=item_id,
            text_parts=[],
        )
        self._blocks[anth_index] = state
        self._output_order.append(anth_index)
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
        ]

    def _open_reasoning_block(self, anth_index: int, content_block: dict) -> list:
        output_index = self._next_output_index
        self._next_output_index += 1
        item_id = f"rs_{uuid4().hex}"
        state = _AnthropicReasoningBlockState(
            output_index=output_index,
            item_id=item_id,
            summary_parts=[],
            encrypted_parts=[],
        )
        self._blocks[anth_index] = state
        self._output_order.append(anth_index)
        return [
            format_translation.sse_encode(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "type": "reasoning",
                        "id": item_id,
                        "summary": [],
                        "content": [],
                        "encrypted_content": None,
                    },
                },
            ),
            format_translation.sse_encode(
                "response.reasoning_summary_part.added",
                {
                    "type": "response.reasoning_summary_part.added",
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                },
            ),
        ]

    def _open_tool_block(self, anth_index: int, content_block: dict) -> list:
        output_index = self._next_output_index
        self._next_output_index += 1
        call_id = content_block.get("id") if isinstance(content_block.get("id"), str) else f"call_{uuid4().hex}"
        name = content_block.get("name") if isinstance(content_block.get("name"), str) else ""
        item_id = f"fc_{uuid4().hex}"
        # Seed arguments from content_block.input when present so streams that
        # never emit input_json_delta still produce valid JSON arguments.
        seeded_input = content_block.get("input")
        seeded_parts: list = []
        if isinstance(seeded_input, (dict, list)):
            try:
                serialized = json.dumps(seeded_input, separators=(",", ":"), ensure_ascii=False)
            except (TypeError, ValueError):
                serialized = ""
            if serialized and serialized != "{}":
                seeded_parts.append(serialized)
        state = _AnthropicToolBlockState(
            output_index=output_index,
            item_id=item_id,
            call_id=call_id,
            name=name,
            arguments_parts=seeded_parts,
        )
        self._blocks[anth_index] = state
        self._output_order.append(anth_index)
        return [
            format_translation.sse_encode(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": name,
                        "arguments": "",
                    },
                },
            ),
        ]

    def _close_block(self, anth_index: int) -> list:
        state = self._blocks.get(anth_index)
        if state is None or getattr(state, "closed", False):
            return []
        state.closed = True
        events: list = []
        if isinstance(state, _AnthropicTextBlockState):
            text = "".join(state.text_parts)
            events.append(
                format_translation.sse_encode(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "output_index": state.output_index,
                        "content_index": 0,
                        "text": text,
                    },
                )
            )
            events.append(
                format_translation.sse_encode(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "output_index": state.output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": text},
                    },
                )
            )
            events.append(
                format_translation.sse_encode(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": state.output_index,
                        "item": {
                            "type": "message",
                            "id": state.item_id,
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        },
                    },
                )
            )
        elif isinstance(state, _AnthropicReasoningBlockState):
            summary = "".join(state.summary_parts)
            raw_signature = "".join(state.encrypted_parts)
            enc_part, parsed_id = format_translation._split_reasoning_signature(raw_signature)
            encrypted = enc_part or None
            reasoning_id = parsed_id or state.item_id
            events.append(
                format_translation.sse_encode(
                    "response.reasoning_summary_text.done",
                    {
                        "type": "response.reasoning_summary_text.done",
                        "item_id": reasoning_id,
                        "output_index": state.output_index,
                        "summary_index": 0,
                        "text": summary,
                    },
                )
            )
            events.append(
                format_translation.sse_encode(
                    "response.reasoning_summary_part.done",
                    {
                        "type": "response.reasoning_summary_part.done",
                        "item_id": reasoning_id,
                        "output_index": state.output_index,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": summary},
                    },
                )
            )
            events.append(
                format_translation.sse_encode(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": state.output_index,
                        "item": {
                            "type": "reasoning",
                            "id": reasoning_id,
                            "summary": [{"type": "summary_text", "text": summary}],
                            "content": [{"type": "reasoning_text", "text": summary}],
                            "encrypted_content": encrypted,
                        },
                    },
                )
            )
        elif isinstance(state, _AnthropicToolBlockState):
            arguments = "".join(state.arguments_parts)
            if not arguments:
                arguments = "{}"
            if state.arguments_parts and not state.streamed_args:
                events.append(
                    format_translation.sse_encode(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "output_index": state.output_index,
                            "item_id": state.item_id,
                            "delta": arguments,
                        },
                    )
                )
            events.append(
                format_translation.sse_encode(
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "output_index": state.output_index,
                        "item_id": state.item_id,
                        "arguments": arguments,
                    },
                )
            )
            events.append(
                format_translation.sse_encode(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": state.output_index,
                        "item": {
                            "type": "function_call",
                            "id": state.item_id,
                            "call_id": state.call_id,
                            "name": state.name,
                            "arguments": arguments,
                        },
                    },
                )
            )
        return events

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def feed_event(self, event_name, data):
        """Yield translated Responses SSE byte events for one upstream event."""
        if not isinstance(data, dict):
            data = {}
        evt = (event_name or data.get("type") or "").strip().lower()

        if evt == "ping":
            return
        if evt == "error":
            # Surface upstream errors as a response.failed event
            for evbytes in self._ensure_response_started():
                yield evbytes
            err = data.get("error") if isinstance(data.get("error"), dict) else {}
            self._failed = True
            self._completed = True  # suppress later response.completed
            yield format_translation.sse_encode(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        "id": self._response_id,
                        "object": "response",
                        "status": "failed",
                        "error": err,
                    },
                },
            )
            return

        if evt == "message_start":
            message = data.get("message") if isinstance(data.get("message"), dict) else {}
            if isinstance(message.get("id"), str) and message["id"]:
                # Use upstream id as a stable response id
                self._response_id = message["id"].replace("msg_", "resp_", 1)
            if isinstance(message.get("model"), str) and message["model"]:
                if not self._model_name:
                    self._model_name = message["model"]
            usage = message.get("usage")
            if isinstance(usage, dict):
                self._usage = _anthropic_usage_to_responses_usage(usage)
                # Anthropic Messages reports input_tokens (fresh non-cache),
                # cache_read_input_tokens, and cache_creation_input_tokens
                # separately. Persist the raw shape so the proxy can derive
                # the internal tracking shape with pricing_* counters.
                self._anthropic_raw_usage = {
                    key: usage[key]
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                        "cached_input_tokens",
                    )
                    if key in usage
                }
            for evbytes in self._ensure_response_started():
                yield evbytes
            return

        # All subsequent events expect response started
        for evbytes in self._ensure_response_started():
            yield evbytes

        if evt == "content_block_start":
            anth_index = data.get("index") if isinstance(data.get("index"), int) else 0
            content_block = data.get("content_block") if isinstance(data.get("content_block"), dict) else {}
            block_type = str(content_block.get("type", "")).lower()
            if block_type == "text":
                for evbytes in self._open_text_block(anth_index, content_block):
                    yield evbytes
            elif block_type == "thinking":
                for evbytes in self._open_reasoning_block(anth_index, content_block):
                    yield evbytes
            elif block_type == "tool_use":
                if self._mark_first_output is not None:
                    self._mark_first_output()
                for evbytes in self._open_tool_block(anth_index, content_block):
                    yield evbytes
            return

        if evt == "content_block_delta":
            anth_index = data.get("index") if isinstance(data.get("index"), int) else 0
            delta = data.get("delta") if isinstance(data.get("delta"), dict) else {}
            delta_type = str(delta.get("type", "")).lower()
            state = self._blocks.get(anth_index)
            if state is None:
                return
            if delta_type == "text_delta" and isinstance(state, _AnthropicTextBlockState):
                text = delta.get("text")
                if isinstance(text, str) and text:
                    state.text_parts.append(text)
                    if self._mark_first_output is not None:
                        self._mark_first_output()
                    yield format_translation.sse_encode(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "output_index": state.output_index,
                            "content_index": 0,
                            "delta": text,
                        },
                    )
            elif delta_type == "thinking_delta" and isinstance(state, _AnthropicReasoningBlockState):
                text = delta.get("thinking")
                if isinstance(text, str) and text:
                    state.summary_parts.append(text)
                    if self._mark_first_output is not None:
                        self._mark_first_output()
                    yield format_translation.sse_encode(
                        "response.reasoning_summary_text.delta",
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": state.item_id,
                            "output_index": state.output_index,
                            "summary_index": 0,
                            "delta": text,
                        },
                    )
            elif delta_type == "signature_delta" and isinstance(state, _AnthropicReasoningBlockState):
                signature = delta.get("signature")
                if isinstance(signature, str) and signature:
                    state.encrypted_parts.append(signature)
            elif delta_type == "input_json_delta" and isinstance(state, _AnthropicToolBlockState):
                partial = delta.get("partial_json")
                if isinstance(partial, str) and partial:
                    state.arguments_parts.append(partial)
                    state.streamed_args = True
                    yield format_translation.sse_encode(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "output_index": state.output_index,
                            "item_id": state.item_id,
                            "delta": partial,
                        },
                    )
            return

        if evt == "content_block_stop":
            anth_index = data.get("index") if isinstance(data.get("index"), int) else 0
            for evbytes in self._close_block(anth_index):
                yield evbytes
            return

        if evt == "message_delta":
            delta = data.get("delta") if isinstance(data.get("delta"), dict) else {}
            stop = delta.get("stop_reason")
            if isinstance(stop, str):
                self._stop_reason = stop
            usage = data.get("usage")
            if isinstance(usage, dict):
                merged = dict(self._usage)
                in_t = usage.get("input_tokens")
                out_t = usage.get("output_tokens")
                if isinstance(in_t, (int, float)):
                    merged["input_tokens"] = int(in_t)
                if isinstance(out_t, (int, float)):
                    merged["output_tokens"] = int(out_t)
                merged["total_tokens"] = int(merged.get("input_tokens", 0)) + int(merged.get("output_tokens", 0))
                details = dict(merged.get("input_tokens_details") or {})
                cached = usage.get("cache_read_input_tokens")
                if isinstance(cached, (int, float)) and cached:
                    details["cached_tokens"] = int(cached)
                cache_creation = usage.get("cache_creation_input_tokens")
                if isinstance(cache_creation, (int, float)) and cache_creation:
                    details["cache_creation_input_tokens"] = int(cache_creation)
                if details:
                    merged["input_tokens_details"] = details
                self._usage = merged
                # Mirror the same upstream-shape fields into the raw cache so
                # tracking-shape derivation reflects the latest counters.
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                    "cached_input_tokens",
                ):
                    if key in usage:
                        self._anthropic_raw_usage[key] = usage[key]
            return

        if evt == "message_stop":
            # Don't emit response.completed here; finalize() handles it so
            # callers can still close any unterminated blocks deterministically.
            return

    def finalize(self):
        """Yield response.completed (and any forgotten block-close events)."""
        if self._completed:
            return
        for evbytes in self._ensure_response_started():
            yield evbytes
        # Close any blocks still open (defensive against truncated streams).
        for anth_index in list(self._output_order):
            state = self._blocks.get(anth_index)
            if state is not None and not getattr(state, "closed", False):
                for evbytes in self._close_block(anth_index):
                    yield evbytes
        self._completed = True
        yield format_translation.sse_encode(
            "response.completed",
            {
                "type": "response.completed",
                "response": self.build_response_payload(),
            },
        )

    async def translate(self, byte_iter):
        """Convenience: translate a raw httpx byte stream end-to-end."""
        async for event_name, data in format_translation.iter_sse_messages(byte_iter):
            if data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            for evbytes in self.feed_event(event_name, payload):
                yield evbytes
        for evbytes in self.finalize():
            yield evbytes
