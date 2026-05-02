import unittest
from types import SimpleNamespace
from unittest import mock

import gzip
import json

import format_translation
import proxy
import usage_tracking
import util

import httpx


class FormatTranslationTests(unittest.TestCase):
    def test_parse_json_request_accepts_gzip_encoded_body(self):
        raw = gzip.compress(b'{"model":"gpt-5","input":"hello"}')
        request = SimpleNamespace(
            headers={"content-encoding": "gzip"},
            body=mock.AsyncMock(return_value=raw),
        )

        parsed = proxy.asyncio.run(proxy.parse_json_request(request))

        self.assertEqual(parsed["model"], "gpt-5")
        self.assertEqual(parsed["input"], "hello")

    def test_parse_json_request_accepts_zstd_encoded_body(self):
        raw = util.zstd_compress(b'{"model":"gpt-5","input":"hello"}')
        request = SimpleNamespace(
            headers={"content-encoding": "zstd"},
            body=mock.AsyncMock(return_value=raw),
        )

        parsed = proxy.asyncio.run(proxy.parse_json_request(request))

        self.assertEqual(parsed["model"], "gpt-5")
        self.assertEqual(parsed["input"], "hello")

    def test_chat_usage_to_response_exact_contract(self):
        self.assertEqual(
            format_translation.chat_usage_to_response(None),
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        self.assertEqual(
            format_translation.chat_usage_to_response(
                {
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 6},
                    "completion_tokens_details": {"reasoning_tokens": 3},
                }
            ),
            {
                "input_tokens": 20,
                "output_tokens": 4,
                "total_tokens": 24,
                "input_tokens_details": {"cached_tokens": 6},
                "output_tokens_details": {"reasoning_tokens": 3},
            },
        )

    def test_chat_completion_to_response_exact_contract(self):
        self.assertEqual(
            format_translation.chat_completion_to_response(
                {"id": "chat-empty", "created": 123, "model": "gpt-5.4", "choices": []}
            ),
            {
                "id": "chat-empty",
                "object": "response",
                "created_at": 123,
                "status": "completed",
                "model": "gpt-5.4",
                "output": [],
                "output_text": "",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        )

        translated = format_translation.chat_completion_to_response(
            {
                "id": "chat-1",
                "created": 456,
                "model": "gpt-5.4",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "Read", "arguments": '{"file":"main.py"}'},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
            fallback_model="resolved-model",
        )

        self.assertEqual(translated["id"], "chat-1")
        self.assertEqual(translated["object"], "response")
        self.assertEqual(translated["created_at"], 456)
        self.assertEqual(translated["status"], "completed")
        self.assertEqual(translated["model"], "resolved-model")
        self.assertEqual(
            translated["output"],
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file":"main.py"}',
                },
            ],
        )
        self.assertEqual(translated["output_text"], "hello")
        self.assertEqual(translated["usage"], {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})

    def test_tool_choice_helpers_exact_contracts(self):
        self.assertIsNone(format_translation.responses_tool_choice_to_chat(None))
        self.assertEqual(format_translation.responses_tool_choice_to_chat("required"), "required")
        self.assertEqual(format_translation.responses_tool_choice_to_chat({"type": "required"}), "required")
        self.assertEqual(
            format_translation.responses_tool_choice_to_chat({"type": "function", "name": "Read"}),
            {"type": "function", "function": {"name": "Read"}},
        )
        self.assertEqual(
            format_translation.responses_tool_choice_to_chat(
                {"type": "function", "function": {"name": "Read"}}
            ),
            {"type": "function", "function": {"name": "Read"}},
        )
        self.assertIsNone(
            format_translation.responses_tool_choice_to_chat(
                {"type": "function", "name": "mcp__ide__executeCode"}
            )
        )
        with self.assertRaisesRegex(ValueError, "Unsupported Responses tool_choice value"):
            format_translation.responses_tool_choice_to_chat({})

        self.assertIsNone(format_translation.chat_tool_choice_to_responses(None))
        self.assertEqual(format_translation.chat_tool_choice_to_responses("auto"), "auto")
        self.assertEqual(format_translation.chat_tool_choice_to_responses({"type": "required"}), "required")
        self.assertEqual(
            format_translation.chat_tool_choice_to_responses({"type": "function", "function": {"name": "Read"}}),
            {"type": "function", "name": "Read"},
        )
        self.assertIsNone(
            format_translation.chat_tool_choice_to_responses(
                {"type": "function", "function": {"name": "mcp__ide__executeCode"}}
            )
        )
        with self.assertRaisesRegex(ValueError, "Unsupported chat tool_choice value"):
            format_translation.chat_tool_choice_to_responses({})

        self.assertIsNone(format_translation.anthropic_tool_choice_to_chat(None))
        self.assertEqual(format_translation.anthropic_tool_choice_to_chat("auto"), "auto")
        self.assertEqual(format_translation.anthropic_tool_choice_to_chat({"type": "any"}), "required")
        self.assertEqual(
            format_translation.anthropic_tool_choice_to_chat({"type": "tool", "name": "Read"}),
            {"type": "function", "function": {"name": "Read"}},
        )
        self.assertIsNone(
            format_translation.anthropic_tool_choice_to_chat(
                {"type": "tool", "name": "mcp__ide__executeCode"}
            )
        )
        with self.assertRaisesRegex(ValueError, "Unsupported Anthropic tool_choice value"):
            format_translation.anthropic_tool_choice_to_chat({})

    def test_anthropic_request_to_chat_translates_cache_control_to_copilot_cache_control(self):
        body = {
            "model": "claude-sonnet-4.6",
            "system": [
                {"type": "text", "text": "first"},
                {
                    "type": "text",
                    "text": "cached",
                    "cache_control": {"ephemeral": {"scope": "conversation"}},
                },
            ],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        self.assertEqual(outbound["model"], "claude-sonnet-4.6")
        self.assertEqual(outbound["reasoning_effort"], "medium")
        self.assertNotIn("thinking_budget", outbound)
        self.assertNotIn("thinking", outbound)
        self.assertEqual(outbound["messages"][0]["role"], "system")
        self.assertIsInstance(outbound["messages"][0]["content"], list)
        self.assertNotIn("stream_options", outbound)
        self.assertEqual(
            outbound["messages"][0]["content"][1]["copilot_cache_control"],
            {"type": "ephemeral"},
        )
        self.assertEqual(
            outbound["messages"][1]["content"][0]["copilot_cache_control"],
            {"type": "ephemeral"},
        )

    def test_anthropic_thinking_to_reasoning_effort_thresholds(self):
        from constants import CLAUDE_DEFAULT_REASONING_EFFORT as _default
        cases = [
            # Absent / disabled / adaptive → configured default.
            ({"type": "enabled", "budget_tokens": 0}, _default),
            ({"type": "disabled", "budget_tokens": 31999}, _default),
            ({"type": "disabled"}, _default),
            ({"type": "adaptive"}, _default),
            (None, _default),
            ({}, _default),
            # Explicit enabled with budget respects thresholds.
            ({"type": "enabled", "budget_tokens": 1024}, "medium"),
            ({"type": "enabled", "budget_tokens": 4000}, "medium"),
            ({"type": "enabled", "budget_tokens": 8191}, "medium"),
            ({"type": "enabled", "budget_tokens": 8192}, "high"),
            ({"type": "enabled", "budget_tokens": 10000}, "high"),
            ({"type": "enabled", "budget_tokens": 24575}, "high"),
            ({"type": "enabled", "budget_tokens": 24576}, "max"),
            ({"type": "enabled", "budget_tokens": 31999}, "max"),
        ]
        for thinking, expected in cases:
            with self.subTest(thinking=thinking):
                self.assertEqual(
                    format_translation._anthropic_thinking_to_reasoning_effort(thinking),
                    expected,
                )

    def test_anthropic_request_to_chat_emits_reasoning_effort_high(self):
        body = {
            "model": "claude-sonnet-4.6",
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "messages": [{"role": "user", "content": "hi"}],
        }
        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))
        self.assertEqual(outbound["reasoning_effort"], "high")
        self.assertNotIn("thinking_budget", outbound)
        self.assertNotIn("reasoning", outbound)

    def test_anthropic_request_to_chat_emits_reasoning_effort_max(self):
        body = {
            "model": "claude-sonnet-4.6",
            "thinking": {"type": "enabled", "budget_tokens": 31999},
            "messages": [{"role": "user", "content": "hi"}],
        }
        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))
        self.assertEqual(outbound["reasoning_effort"], "max")

    def test_anthropic_request_to_chat_omits_reasoning_effort_when_no_thinking(self):
        from constants import CLAUDE_DEFAULT_REASONING_EFFORT as _default
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "hi"}],
        }
        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))
        self.assertEqual(outbound["reasoning_effort"], _default)
        self.assertNotIn("thinking_budget", outbound)

    def test_anthropic_request_to_chat_adaptive_thinking_defaults_medium(self):
        from constants import CLAUDE_DEFAULT_REASONING_EFFORT as _default
        body = {
            "model": "claude-sonnet-4.6",
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))
        self.assertEqual(outbound["reasoning_effort"], _default)

    def test_anthropic_request_to_chat_ignores_assistant_thinking_blocks(self):
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "internal"},
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "visible answer"},
                    ],
                },
            ],
        }

        outbound = proxy.asyncio.run(
            format_translation.anthropic_request_to_chat(
                body, "https://example.invalid", "test-key"
            )
        )

        self.assertEqual(
            outbound["messages"],
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "visible answer"},
            ],
        )

    def test_anthropic_request_to_responses_ignores_assistant_thinking_blocks(self):
        body = {
            "model": "gpt-5.4",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "internal"},
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "visible answer"},
                    ],
                },
            ],
        }

        outbound = format_translation.anthropic_request_to_responses(body)

        self.assertEqual(
            outbound["input"][0],
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
        )
        self.assertEqual(outbound["input"][1]["type"], "message")
        self.assertEqual(outbound["input"][1]["role"], "assistant")
        self.assertEqual(
            outbound["input"][1]["content"],
            [{"type": "output_text", "text": "visible answer", "annotations": []}],
        )
        self.assertNotIn("status", outbound["input"][1])
        self.assertNotIn("id", outbound["input"][1])
        self.assertEqual(outbound["text"], {"format": {"type": "text"}, "verbosity": "low"})

    def test_anthropic_effort_to_reasoning_effort_helper(self):
        self.assertEqual(
            format_translation._anthropic_effort_to_reasoning_effort({"effort": "low"}),
            "low",
        )
        self.assertEqual(
            format_translation._anthropic_effort_to_reasoning_effort({"effort": "HIGH"}),
            "high",
        )
        self.assertEqual(
            format_translation._anthropic_effort_to_reasoning_effort({"effort": "max"}),
            "max",
        )
        self.assertIsNone(
            format_translation._anthropic_effort_to_reasoning_effort({"effort": "bogus"}),
        )
        self.assertIsNone(format_translation._anthropic_effort_to_reasoning_effort(None))
        self.assertIsNone(format_translation._anthropic_effort_to_reasoning_effort({}))

    def test_output_config_effort_overrides_thinking_in_chat(self):
        body = {
            "model": "claude-opus-4.6",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))
        self.assertEqual(outbound["reasoning_effort"], "low")

    def test_output_config_effort_overrides_thinking_in_responses(self):
        body = {
            "model": "gpt-5.4",
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        outbound = format_translation.anthropic_request_to_responses(body)
        self.assertEqual(outbound["reasoning"], {"effort": "high"})

    def test_opus_47_clamps_effort_to_medium_in_chat(self):
        body = {
            "model": "claude-opus-4.7",
            "output_config": {"effort": "high"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))
        self.assertEqual(outbound["reasoning_effort"], "medium")

    def test_opus_47_clamps_effort_to_medium_in_responses(self):
        body = {
            "model": "claude-opus-4.7",
            "output_config": {"effort": "max"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        outbound = format_translation.anthropic_request_to_responses(body)
        self.assertEqual(outbound["reasoning"], {"effort": "medium"})

    def test_haiku_45_omits_reasoning_effort_in_chat(self):
        body = {
            "model": "claude-haiku-4.5",
            "output_config": {"effort": "medium"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        outbound = proxy.asyncio.run(
            format_translation.anthropic_request_to_chat(
                body, "https://example.invalid", "test-key"
            )
        )
        self.assertNotIn("reasoning_effort", outbound)

    def test_haiku_45_omits_reasoning_effort_in_responses(self):
        body = {
            "model": "claude-haiku-4.5",
            "output_config": {"effort": "medium"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        outbound = format_translation.anthropic_request_to_responses(body)
        self.assertNotIn("reasoning", outbound)

    def test_responses_request_to_chat_preserves_xhigh_for_gpt_models(self):
        body = {
            "model": "openai/gpt-5.4",
            "input": "hi",
            "reasoning": {"effort": "xhigh"},
        }

        translated = format_translation.responses_request_to_chat(body)

        self.assertEqual(translated["reasoning_effort"], "xhigh")

    def test_responses_request_to_chat_maps_max_to_xhigh_for_gpt_models(self):
        body = {
            "model": "openai/gpt-5.4",
            "input": "hi",
            "reasoning": {"effort": "max"},
        }

        translated = format_translation.responses_request_to_chat(body)

        self.assertEqual(translated["reasoning_effort"], "xhigh")

    def test_responses_request_to_chat_maps_xhigh_to_max_for_claude_models(self):
        body = {
            "model": "anthropic/claude-sonnet-4.6",
            "input": "hi",
            "reasoning": {"effort": "xhigh"},
        }

        translated = format_translation.responses_request_to_chat(body)

        self.assertEqual(translated["reasoning_effort"], "max")

    def test_responses_request_to_chat_defaults_claude_reasoning_when_missing(self):
        # Codex omits the `reasoning` field entirely when its local model
        # catalog does not advertise reasoning summary support. The proxy must
        # still enable extended thinking for Claude models so Opus/Sonnet
        # actually return `delta.thinking` chunks.
        from constants import CLAUDE_DEFAULT_REASONING_EFFORT as _default

        for body in (
            {"model": "anthropic/claude-opus-4.7", "input": "hi"},
            {"model": "claude-opus-4.7", "input": "hi", "reasoning": None},
            {"model": "claude-sonnet-4.6", "input": "hi", "reasoning": {"effort": None}},
        ):
            translated = format_translation.responses_request_to_chat(body)
            expected = "medium" if translated["model"] == "claude-opus-4.7" else _default
            self.assertEqual(translated.get("reasoning_effort"), expected, body)

    def test_responses_request_to_chat_default_omits_reasoning_for_haiku(self):
        body = {"model": "claude-haiku-4.5", "input": "hi"}
        translated = format_translation.responses_request_to_chat(body)
        self.assertNotIn("reasoning_effort", translated)

    def test_responses_request_to_chat_does_not_default_for_non_claude(self):
        body = {"model": "openai/gpt-5.4", "input": "hi"}
        translated = format_translation.responses_request_to_chat(body)
        self.assertNotIn("reasoning_effort", translated)

    def test_model_resolution_helpers_are_exact_for_upstream_cache_keys(self):
        cases = [
            (None, None, None),
            (123, 123, 123),
            (" anthropic/Claude-Sonnet-Latest ", "claude-sonnet-latest", "claude-sonnet-4.6"),
            ("anthropic/claude-opus-4.6", "claude-opus-4.6", "claude-opus-4.6"),
            ("claude-opus-experimental", "claude-opus-experimental", "claude-opus-4.7"),
            ("claude-haiku-next", "claude-haiku-next", "claude-haiku-4.5"),
            ("openai/gpt-5.4", "gpt-5.4", "gpt-5.4"),
        ]

        for source, normalized, resolved in cases:
            with self.subTest(source=source):
                self.assertEqual(format_translation.normalize_upstream_model_name(source), normalized)
                self.assertEqual(format_translation.resolve_copilot_model_name(source), resolved)

    def test_anthropic_to_chat_full_upstream_payload_shape_is_stable(self):
        body = {
            "model": "anthropic/claude-sonnet-latest",
            "system": [
                {"type": "text", "text": "sys-a"},
                {"type": "text", "text": "sys-b", "cache_control": {"ephemeral": {"scope": "conversation"}}},
            ],
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look", "cache_control": {"type": "ephemeral"}},
                        {"type": "image", "source": {"type": "url", "url": "https://example.invalid/a.png"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calling"},
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"path": "a.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": [{"type": "text", "text": "file"}],
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "done"},
                    ],
                },
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read files",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                    "cache_control": {"type": "ephemeral"},
                },
                {"name": "mcp__ide__executeCode", "description": "Execute code", "input_schema": {"type": "object"}},
            ],
            "tool_choice": {"type": "tool", "name": "Read"},
            "stream": True,
            "max_tokens": 123,
            "temperature": 0.2,
            "top_p": 0.8,
            "stop_sequences": ["STOP"],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        self.assertEqual(
            outbound,
            {
                "model": "claude-sonnet-4.6",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "sys-a"},
                            {"type": "text", "text": "sys-b", "copilot_cache_control": {"type": "ephemeral"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "look", "copilot_cache_control": {"type": "ephemeral"}},
                            {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "calling",
                        "tool_calls": [
                            {
                                "id": "tool_1",
                                "type": "function",
                                "function": {"name": "Read", "arguments": '{"path":"a.py"}'},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "tool_1",
                        "content": "file",
                        "copilot_cache_control": {"type": "ephemeral"},
                    },
                    {"role": "user", "content": "done"},
                ],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 123,
                "temperature": 0.2,
                "top_p": 0.8,
                "stop": ["STOP"],
                "reasoning_effort": "high",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Read files",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        },
                        "copilot_cache_control": {"type": "ephemeral"},
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "Read"}},
            },
        )

    def test_responses_to_chat_full_upstream_payload_shape_is_stable(self):
        body = {
            "model": "anthropic/claude-sonnet-4.6",
            "instructions": "global",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "dev", "copilot_cache_control": {"type": "ephemeral"}}],
                },
                {"type": "message", "role": "system", "content": "sys"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello", "copilot_cache_control": {"type": "ephemeral"}},
                        {"type": "input_file", "filename": "doc.pdf", "file_data": "data:application/pdf;base64,PDF"},
                        {"type": "input_image", "image_url": {"url": "https://example.invalid/i.png"}},
                    ],
                },
                {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": {"path": "a.py"}},
                {"type": "message", "role": "user", "content": "defer until tool done"},
                {"type": "function_call_output", "call_id": "call_1", "output": [{"type": "output_text", "text": "contents"}]},
                {"type": "custom_tool_call", "call_id": "ct_1", "name": "apply_patch", "input": {"patch": "x"}},
                {"type": "custom_tool_call_output", "call_id": "ct_1", "output": {"ok": True}},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read files",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "function", "name": "mcp__ide__executeCode"},
            ],
            "tool_choice": {"type": "function", "name": "Read"},
            "stream": True,
            "max_output_tokens": 55,
            "temperature": 0.3,
            "top_p": 0.7,
            "reasoning": {"effort": "max"},
        }

        translated = format_translation.responses_request_to_chat(body)

        self.assertEqual(
            translated,
            {
                "model": "claude-sonnet-4.6",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "global"},
                            {"type": "text", "text": "\n\n"},
                            {"type": "text", "text": "dev", "copilot_cache_control": {"type": "ephemeral"}},
                            {"type": "text", "text": "\n\n"},
                            {"type": "text", "text": "sys"},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello", "copilot_cache_control": {"type": "ephemeral"}},
                            {
                                "type": "document",
                                "source": {"type": "base64", "media_type": "application/pdf", "data": "PDF"},
                                "title": "doc.pdf",
                            },
                            {"type": "image_url", "image_url": {"url": "https://example.invalid/i.png"}},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "Read", "arguments": '{"path":"a.py"}'},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
                    {"role": "user", "content": "defer until tool done"},
                    {"role": "assistant", "content": '[Custom tool call (ct_1)] apply_patch\n{"patch":"x"}'},
                    {"role": "user", "content": '[Custom tool result (ct_1)]\n{"ok":true}'},
                ],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 55,
                "temperature": 0.3,
                "top_p": 0.7,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Read files",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        },
                        "copilot_cache_control": {"type": "ephemeral"},
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "Read"}},
                "reasoning_effort": "max",
            },
        )

    def test_anthropic_to_responses_full_upstream_payload_shape_is_stable(self):
        body = {
            "model": "anthropic/claude-sonnet-4.6",
            "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}},
                        {"type": "image", "source": {"type": "url", "url": "https://example.invalid/i.png"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "answer"},
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"path": "a.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tool_1", "content": "contents"},
                        {"type": "text", "text": "continue"},
                    ],
                },
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read files",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tool_choice": {"type": "tool", "name": "Read"},
            "stream": True,
            "max_tokens": 99,
            "temperature": 0.4,
            "top_p": 0.9,
            "metadata": {"m": "v"},
            "output_config": {"effort": "high"},
        }

        translated = format_translation.anthropic_request_to_responses(body)

        self.assertEqual(
            translated,
            {
                "model": "claude-sonnet-4.6",
                "input": [
                    {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "sys"}]},
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "hello"},
                            {"type": "input_image", "image_url": "https://example.invalid/i.png"},
                        ],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                    },
                    {"type": "function_call", "call_id": "tool_1", "name": "Read", "arguments": '{"path":"a.py"}'},
                    {"type": "function_call_output", "call_id": "tool_1", "output": "contents"},
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "continue"}]},
                ],
                "stream": True,
                "store": False,
                "parallel_tool_calls": True,
                "include": ["reasoning.encrypted_content"],
                "text": {"format": {"type": "text"}, "verbosity": "low"},
                "max_output_tokens": 99,
                "temperature": 0.4,
                "top_p": 0.9,
                "tools": [
                    {
                        "type": "function",
                        "name": "Read",
                        "description": "Read files",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    }
                ],
                "tool_choice": {"type": "function", "name": "Read"},
                "reasoning": {"effort": "high"},
            },
        )

    def test_anthropic_image_and_cache_helpers_are_exact(self):
        self.assertEqual(
            format_translation._normalize_anthropic_cache_control(
                {"ephemeral": {"scope": "conversation"}}
            ),
            {"type": "ephemeral"},
        )
        self.assertEqual(
            format_translation._normalize_anthropic_cache_control(
                {"ttl": "5m", "scope": "conversation"}
            ),
            {"ttl": "5m", "scope": "conversation"},
        )
        self.assertIsNone(format_translation._normalize_anthropic_cache_control("bad"))

        self.assertEqual(
            format_translation._anthropic_image_block_to_chat(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AAAA",
                    },
                    "cache_control": {"type": "ephemeral"},
                }
            ),
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
                "copilot_cache_control": {"type": "ephemeral"},
            },
        )
        with self.assertRaisesRegex(ValueError, "missing a valid source object"):
            format_translation._anthropic_image_block_to_chat({"type": "image"})
        with self.assertRaisesRegex(ValueError, "must include media_type and data strings"):
            format_translation._anthropic_image_block_to_chat(
                {"type": "image", "source": {"type": "base64", "data": "AAAA"}}
            )
        with self.assertRaisesRegex(ValueError, "must include a url string"):
            format_translation._anthropic_image_block_to_chat(
                {"type": "image", "source": {"type": "url", "url": 123}}
            )
        with self.assertRaisesRegex(ValueError, "Unsupported Anthropic image source type: file"):
            format_translation._anthropic_image_block_to_chat(
                {"type": "image", "source": {"type": "file"}}
            )

    def test_anthropic_request_to_responses_keeps_metadata_session_out_of_body(self):
        translated = format_translation.anthropic_request_to_responses(
            {
                "model": "claude-opus-4.6",
                "metadata": {"user_id": '{"device_id":"device-1","session_id":"session-1"}'},
                "messages": [{"role": "user", "content": "hello"}],
            }
        )

        self.assertNotIn("prompt_cache_key", translated)

    def test_anthropic_request_to_responses_ignores_plain_metadata_user_id_for_prompt_cache(self):
        translated = format_translation.anthropic_request_to_responses(
            {
                "model": "claude-opus-4.6",
                "metadata": {"user_id": "plain-session"},
                "messages": [{"role": "user", "content": "hello"}],
            }
        )

        self.assertNotIn("prompt_cache_key", translated)

    def test_anthropic_system_content_and_tools_helpers_are_exact(self):
        self.assertEqual(
            format_translation._normalize_anthropic_cache_control(
                {"type": "", "ephemeral": {"scope": "conversation"}}
            ),
            {"type": "ephemeral"},
        )
        self.assertEqual(
            format_translation._anthropic_system_to_chat_content(
                [
                    "skip this",
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": 123},
                    {"type": "text", "text": "b"},
                ]
            ),
            "ab",
        )
        self.assertEqual(format_translation._anthropic_system_to_chat_content([]), "")
        self.assertEqual(format_translation._anthropic_system_to_chat_content(123), "")
        with self.assertRaisesRegex(ValueError, "supports text blocks only"):
            format_translation._anthropic_system_to_chat_content([{"type": "image"}])

        self.assertEqual(format_translation._anthropic_blocks_to_chat_content([]), "")
        self.assertEqual(
            format_translation._anthropic_blocks_to_chat_content(
                [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
                ]
            ),
            [
                {"type": "text", "text": "a"},
                {
                    "type": "text",
                    "text": "b",
                    "copilot_cache_control": {"type": "ephemeral"},
                },
            ],
        )
        with self.assertRaisesRegex(ValueError, "tool_use cannot be converted"):
            format_translation._anthropic_blocks_to_chat_content(
                [{"type": "tool_use", "id": "tool_1", "name": "Read"}]
            )

        with self.assertRaisesRegex(ValueError, "Anthropic tools must be a list"):
            format_translation.anthropic_tools_to_chat({"name": "Read"})
        with self.assertRaisesRegex(ValueError, "Anthropic tools must include a string name"):
            format_translation.anthropic_tools_to_chat([{"description": "missing"}])
        self.assertEqual(
            format_translation.anthropic_tools_to_chat(
                [
                    "skip",
                    {
                        "name": "NoDescription",
                        "description": "",
                        "input_schema": "bad",
                    },
                    {
                        "name": "Cached",
                        "description": "cached tool",
                        "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                        "cache_control": {"type": "ephemeral"},
                    },
                ]
            ),
            [
                {
                    "type": "function",
                    "function": {
                        "name": "NoDescription",
                        "description": " ",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Cached",
                        "description": "cached tool",
                        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                    },
                    "copilot_cache_control": {"type": "ephemeral"},
                },
            ],
        )

    def test_tool_choice_and_responses_chat_helpers_are_exact(self):
        self.assertIsNone(format_translation.anthropic_tool_choice_to_chat(None))
        self.assertEqual(format_translation.anthropic_tool_choice_to_chat({"type": "auto"}), "auto")
        self.assertEqual(format_translation.anthropic_tool_choice_to_chat({"type": "any"}), "required")
        self.assertEqual(format_translation.anthropic_tool_choice_to_chat({"type": "none"}), "none")
        self.assertEqual(format_translation.anthropic_tool_choice_to_chat("any"), "required")
        self.assertIsNone(
            format_translation.anthropic_tool_choice_to_chat(
                {"type": "tool", "name": "mcp__ide__executeCode"}
            )
        )
        with self.assertRaisesRegex(ValueError, "type=tool must include name"):
            format_translation.anthropic_tool_choice_to_chat({"type": "tool"})
        with self.assertRaisesRegex(ValueError, "Unsupported Anthropic tool_choice value"):
            format_translation.anthropic_tool_choice_to_chat({"type": "bogus"})
        with self.assertRaisesRegex(ValueError, "Unsupported Anthropic tool_choice value"):
            format_translation.anthropic_tool_choice_to_chat("required")

        self.assertEqual(
            format_translation._response_content_item_to_chat(
                {"type": "input_text", "input_text": "input fallback"}
            ),
            {"type": "text", "text": "input fallback"},
        )
        self.assertEqual(
            format_translation._response_content_item_to_chat(
                {"type": "output_text", "output_text": "output fallback", "copilot_cache_control": {"type": "ephemeral"}}
            ),
            {
                "type": "text",
                "text": "output fallback",
                "copilot_cache_control": {"type": "ephemeral"},
            },
        )
        self.assertIsNone(format_translation._response_content_item_to_chat({"type": "text", "text": 123}))
        self.assertIsNone(format_translation._response_content_item_to_chat({"type": "input_file", "file_data": "data:bad"}))
        with self.assertRaisesRegex(ValueError, "must include image_url or image_base64/media_type"):
            format_translation._response_content_item_to_chat({"type": "input_image"})
        self.assertEqual(
            format_translation._response_content_item_to_chat(
                {"type": "input_image", "image_base64": "AAAA", "media_type": "image/png"}
            ),
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        )
        self.assertEqual(format_translation._response_message_content_to_chat("plain"), "plain")
        self.assertEqual(format_translation._response_message_content_to_chat(123), "")
        self.assertEqual(
            format_translation._response_message_content_to_chat(
                [{"type": "output_text", "output_text": "only text"}]
            ),
            "only text",
        )

        with self.assertRaisesRegex(ValueError, "Responses function tools must include a name"):
            format_translation._responses_tool_to_chat({"type": "function"})
        self.assertEqual(
            format_translation._responses_tool_to_chat(
                {
                    "type": "function",
                    "function": {
                        "name": "Nested",
                        "description": "nested desc",
                        "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
                        "cache_control": {"type": "ephemeral"},
                    },
                }
            ),
            {
                "type": "function",
                "function": {
                    "name": "Nested",
                    "description": "nested desc",
                    "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
                },
                "copilot_cache_control": {"type": "ephemeral"},
            },
        )
        self.assertEqual(
            format_translation._responses_tool_to_chat(
                {"type": "function", "name": "Flat", "description": "", "parameters": "bad"}
            ),
            {
                "type": "function",
                "function": {
                    "name": "Flat",
                    "description": " ",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        )

    def test_anthropic_request_to_chat_preserves_cache_control_on_tool_results(self):
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"file": "test.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file contents",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        tool_msg = [m for m in outbound["messages"] if m["role"] == "tool"][0]
        self.assertEqual(
            tool_msg["copilot_cache_control"],
            {"type": "ephemeral"},
        )

    def test_anthropic_request_to_chat_no_cache_control_on_tool_result_without_it(self):
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"file": "test.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file contents",
                        }
                    ],
                },
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        tool_msg = [m for m in outbound["messages"] if m["role"] == "tool"][0]
        self.assertNotIn("copilot_cache_control", tool_msg)

    def test_anthropic_request_to_chat_emits_tool_results_before_user_text(self):
        body = {
            "model": "claude-opus-4.7",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"file": "test.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is the output."},
                        {"type": "tool_result", "tool_use_id": "tool_1", "content": "file contents"},
                    ],
                },
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        self.assertEqual([message["role"] for message in outbound["messages"]], ["assistant", "tool", "user"])
        self.assertEqual(outbound["messages"][1]["tool_call_id"], "tool_1")
        self.assertEqual(outbound["messages"][2]["content"], "Here is the output.")

    def test_build_fake_compaction_request_preserves_openai_xhigh_reasoning(self):
        body = {
            "model": "openai/gpt-5.4",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            "reasoning": {"effort": "xhigh", "summary": "auto"},
        }

        compact_request = format_translation.build_fake_compaction_request(body)

        self.assertEqual(compact_request["reasoning"], {"effort": "xhigh", "summary": "auto"})
        self.assertEqual(body["reasoning"], {"effort": "xhigh", "summary": "auto"})

    def test_build_fake_compaction_request_preserves_anthropic_max_reasoning(self):
        body = {
            "model": "anthropic/claude-sonnet-4.6",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            "reasoning": {"effort": "max"},
        }

        compact_request = format_translation.build_fake_compaction_request(body)

        self.assertEqual(compact_request["reasoning"], {"effort": "max"})
    def test_build_fake_compaction_request_preserves_cache_affinity_fields(self):
        body = {
            "model": "gpt-5.4",
            "input": "hello",
            "sessionId": "session-123",
            "promptCacheKey": "cache-123",
            "previous_response_id": "resp_prev",
            "metadata": {"user_id": '{"session_id":"metadata-session"}'},
            "user": "user-123",
        }

        compact_request = format_translation.build_fake_compaction_request(body)

        self.assertEqual(compact_request["sessionId"], "session-123")
        self.assertEqual(compact_request["promptCacheKey"], "cache-123")
        self.assertEqual(compact_request["previous_response_id"], "resp_prev")
        self.assertEqual(compact_request["metadata"], body["metadata"])
        self.assertEqual(compact_request["user"], "user-123")

    def test_build_fake_compaction_request_preserves_request_config_for_codex_models(self):
        body = {
            "model": "gpt-5.4",
            "input": "hello",
            "tools": [
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "stream": True,
            "store": False,
        }

        compact_request = format_translation.build_fake_compaction_request(body)

        self.assertEqual(compact_request["tools"], body["tools"])
        self.assertEqual(compact_request["include"], body["include"])
        self.assertTrue(compact_request["parallel_tool_calls"])
        self.assertEqual(compact_request["tool_choice"], "auto")
        self.assertTrue(compact_request["stream"])
        self.assertFalse(compact_request["store"])

    def test_sanitize_responses_tools_preserves_defer_loading_without_tool_search(self):
        diagnostics = []
        body = {
            "model": "gpt-5.5",
            "input": "hello",
            "tools": [
                {
                    "type": "namespace",
                    "name": "codex_app",
                    "tools": [
                        {
                            "type": "function",
                            "name": "automation_update",
                            "defer_loading": True,
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                }
            ],
        }

        sanitized = format_translation.sanitize_responses_tools_for_copilot(
            body,
            diagnostics=diagnostics,
        )

        self.assertIs(sanitized, body)
        nested_tool = sanitized["tools"][0]["tools"][0]
        self.assertTrue(nested_tool["defer_loading"])
        self.assertEqual(nested_tool["name"], "automation_update")
        self.assertEqual(diagnostics, [])

    def test_sanitize_responses_tools_keeps_defer_loading_with_tool_search(self):
        body = {
            "model": "gpt-5.5",
            "input": "hello",
            "tools": [
                {"type": "function", "name": "tool_search", "parameters": {"type": "object", "properties": {}}},
                {
                    "type": "namespace",
                    "name": "codex_app",
                    "tools": [
                        {
                            "type": "function",
                            "name": "automation_update",
                            "defer_loading": True,
                        }
                    ],
                },
            ],
        }

        sanitized = format_translation.sanitize_responses_tools_for_copilot(body)

        self.assertIs(sanitized, body)
        self.assertTrue(sanitized["tools"][1]["tools"][0]["defer_loading"])

    def test_sanitize_responses_tools_drops_dangerous_execute_code_even_when_deferred(self):
        diagnostics = []
        body = {
            "model": "gpt-5.5",
            "input": "hello",
            "tools": [
                {"type": "function", "name": "read", "parameters": {"type": "object", "properties": {}}},
                {
                    "type": "function",
                    "name": "mcp__ide__executeCode",
                    "defer_loading": True,
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
            "tool_choice": {"type": "function", "name": "mcp__ide__executeCode"},
        }

        sanitized = format_translation.sanitize_responses_tools_for_copilot(
            body,
            diagnostics=diagnostics,
        )

        self.assertEqual([tool["name"] for tool in sanitized["tools"]], ["read"])
        self.assertNotIn("tool_choice", sanitized)
        self.assertEqual(diagnostics[0]["action"], "drop_dangerous_code_execution_tools")
        self.assertEqual(diagnostics[0]["tool_names"], ["mcp__ide__executecode"])

    def test_sanitize_responses_body_drops_only_known_unsupported_fields(self):
        diagnostics = []
        body = {
            "model": "gpt-5.5",
            "input": "hello",
            "client_metadata": {"session_id": "local-only"},
            "previous_response_id": "resp_prev",
            "prompt_cache_key": "cache-local",
            "service_tier": "priority",
            "tool_choice": "auto",
            "context_management": [{"type": "compaction", "compact_threshold": 100000}],
        }

        sanitized = format_translation.sanitize_responses_body_for_copilot(
            body,
            diagnostics=diagnostics,
        )

        self.assertNotIn("service_tier", sanitized)
        self.assertEqual(sanitized["client_metadata"], {"session_id": "local-only"})
        self.assertEqual(sanitized["tool_choice"], "auto")
        self.assertEqual(sanitized["context_management"], [{"type": "compaction", "compact_threshold": 100000}])
        self.assertNotIn("previous_response_id", sanitized)
        self.assertNotIn("prompt_cache_key", sanitized)
        self.assertEqual(diagnostics[0]["action"], "drop_unsupported_copilot_fields")
        self.assertEqual(
            diagnostics[0]["fields"],
            ["previous_response_id", "prompt_cache_key", "service_tier"],
        )

    def test_sanitize_responses_body_preserves_explicit_tool_choice(self):
        choice = {"type": "function", "name": "read"}
        body = {
            "model": "gpt-5.5",
            "input": "hello",
            "tool_choice": choice,
        }

        sanitized = format_translation.sanitize_responses_body_for_copilot(body)

        self.assertIs(sanitized, body)
        self.assertEqual(sanitized["tool_choice"], choice)

    def test_cache_keyed_responses_input_preserves_instruction_message_positions(self):
        body = {
            "model": "gpt-5.5",
            "prompt_cache_key": "cache-123",
            "instructions": "base instructions",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "developer guidance"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "first user"}],
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "mid-turn guidance"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "second user"}],
                },
            ],
        }
        diagnostics = []

        normalized = format_translation.normalize_responses_instructions_for_copilot(
            body,
            diagnostics=diagnostics,
        )
        normalized = format_translation.normalize_responses_input_for_copilot(
            normalized,
            diagnostics=diagnostics,
        )

        self.assertIs(normalized, body)
        self.assertEqual([item["role"] for item in normalized["input"]], ["developer", "user", "developer", "user"])
        self.assertEqual(normalized["instructions"], "base instructions")
        self.assertEqual(diagnostics, [])

    def test_responses_request_to_chat_drops_dangerous_execute_code_tool_and_choice(self):
        body = {
            "model": "claude-opus-4.6",
            "input": "hello",
            "tools": [
                {
                    "type": "function",
                    "name": "mcp__ide__executeCode",
                    "description": "Execute code",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "mcp__ide__executeCode"},
        }

        translated = format_translation.responses_request_to_chat(body)

        self.assertNotIn("tools", translated)
        self.assertNotIn("tool_choice", translated)

    def test_anthropic_request_to_responses_drops_dangerous_execute_code_tool_and_choice(self):
        body = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "mcp__ide__executeCode",
                    "description": "Execute code",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "tool", "name": "mcp__ide__executeCode"},
        }

        translated = format_translation.anthropic_request_to_responses(body)

        self.assertNotIn("tools", translated)
        self.assertNotIn("tool_choice", translated)

    def test_build_fake_compaction_request_keeps_tools_with_none_choice_for_claude_models(self):
        body = {
            "model": "claude-opus-4.6",
            "input": "hello",
            "tools": [
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "stream": True,
            "store": False,
        }

        compact_request = format_translation.build_fake_compaction_request(body)

        self.assertEqual(compact_request["input"][0]["content"][0]["text"], "hello")
        self.assertEqual(compact_request["input"][-1]["content"][0]["text"], format_translation.COMPACTION_SUMMARY_PROMPT)
        self.assertEqual(compact_request["tools"], body["tools"])
        self.assertEqual(compact_request["tool_choice"], "none")
        self.assertNotIn("parallel_tool_calls", compact_request)
        self.assertEqual(compact_request["include"], body["include"])
        self.assertTrue(compact_request["stream"])
        self.assertFalse(compact_request["store"])

    def test_compaction_summary_prompt_matches_captured_copilot_prompt(self):
        self.assertTrue(
            format_translation.COMPACTION_SUMMARY_PROMPT.startswith(
                "Please create a detailed summary of the conversation so far."
            )
        )
        self.assertIn(
            "7. <checkpoint_title> - 2-6 word description of the main work done",
            format_translation.COMPACTION_SUMMARY_PROMPT,
        )
        self.assertIn(
            "Please write the summary now, following the structure and guidelines above.",
            format_translation.COMPACTION_SUMMARY_PROMPT,
        )

    def test_build_fake_compaction_request_transcriptizes_tool_history_for_bridging(self):
        body = {
            "model": "claude-opus-4.6",
            "input": [
                {
                    "type": "compaction",
                    "encrypted_content": format_translation.encode_fake_compaction("carry this forward"),
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file":"main.py"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file contents",
                },
                {
                    "type": "function_call_output",
                    "call_id": 123,
                    "output": "numeric id should not appear in label",
                },
                {
                    "type": "function_call_output",
                    "call_id": "",
                    "output": None,
                },
            ],
        }

        compact_request = format_translation.build_fake_compaction_request(body)

        self.assertEqual(
            compact_request["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "[Compacted conversation summary]\ncarry this forward",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '[Tool call (call_1)] Read\n{"file":"main.py"}',
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "[Tool result (call_1)]\nfile contents",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "[Tool result]\nnumeric id should not appear in label",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "[Tool result]",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": format_translation.COMPACTION_SUMMARY_PROMPT}],
                },
            ],
        )

    def test_build_fake_compaction_request_chat_translation_keeps_tools_with_none_choice(self):
        body = {
            "model": "claude-opus-4.6",
            "input": [
                {
                    "type": "compaction",
                    "encrypted_content": format_translation.encode_fake_compaction("carry this forward"),
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file contents",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
            "tool_choice": "auto",
        }

        compact_request = format_translation.build_fake_compaction_request(body)
        translated = format_translation.responses_request_to_chat(compact_request)

        self.assertEqual([message["role"] for message in translated["messages"]], ["user", "user", "user"])
        self.assertTrue(
            translated["messages"][-1]["content"].startswith(
                "Please create a detailed summary of the conversation so far."
            )
        )
        self.assertNotIn("tool_calls", translated["messages"][0])
        self.assertEqual(len(translated["tools"]), 1)
        self.assertEqual(translated["tool_choice"], "none")

    def test_build_fake_compaction_request_can_force_responses_safe_transcript(self):
        body = {
            "model": "gpt-5.2",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "please inspect this"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I found the issue."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file":"main.py"}',
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
        }

        compact_request = format_translation.build_fake_compaction_request(
            body,
            force_responses_safe_transcript=True,
        )

        self.assertEqual([item["role"] for item in compact_request["input"]], ["user", "user", "user", "user"])
        self.assertEqual(compact_request["input"][1]["content"][0]["text"], "[assistant message]\nI found the issue.")
        self.assertEqual(compact_request["input"][2]["content"][0]["text"], '[Tool call (call_1)] Read\n{"file":"main.py"}')
        self.assertEqual(compact_request["input"][-1]["content"][0]["text"], format_translation.COMPACTION_SUMMARY_PROMPT)
        self.assertEqual(compact_request["tool_choice"], "none")
        self.assertNotIn("parallel_tool_calls", compact_request)

    def test_responses_request_to_chat_transcriptizes_custom_tool_history(self):
        body = {
            "model": "claude-opus-4.6",
            "input": [
                {
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "name": "apply_patch",
                    "status": "completed",
                    "input": "*** Begin Patch\n*** Update File: a.txt\n+hello\n*** End Patch",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "Exit code: 0\nOutput:\nSuccess.",
                },
            ],
        }

        translated = format_translation.responses_request_to_chat(body)

        self.assertEqual(
            translated["messages"],
            [
                {
                    "role": "assistant",
                    "content": "[Custom tool call (call_1)] apply_patch\n*** Begin Patch\n*** Update File: a.txt\n+hello\n*** End Patch",
                },
                {
                    "role": "user",
                    "content": "[Custom tool result (call_1)]\nExit code: 0\nOutput:\nSuccess.",
                },
            ],
        )

    def test_build_fake_compaction_request_preserves_native_responses_items_for_codex_models(self):
        body = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "compaction",
                    "encrypted_content": format_translation.encode_fake_compaction("carry this forward"),
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file":"main.py"}',
                    "id": "fc_1",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file contents",
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "opaque-reasoning",
                    "summary": [{"type": "summary_text", "text": "thinking"}],
                },
                {
                    "type": "item_reference",
                    "id": "rs_1",
                },
            ],
        }

        compact_request = format_translation.build_fake_compaction_request(body)

        self.assertEqual(
            compact_request["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "[Compacted conversation summary]\ncarry this forward",
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file":"main.py"}',
                    "id": "fc_1",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file contents",
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "opaque-reasoning",
                    "summary": [{"type": "summary_text", "text": "thinking"}],
                },
                {
                    "type": "item_reference",
                    "id": "rs_1",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": format_translation.COMPACTION_SUMMARY_PROMPT}],
                },
            ],
        )

    def test_sanitize_input_uses_latest_local_compaction_as_boundary(self):
        sanitized = format_translation.sanitize_input(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "old context"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "old reply"}],
                },
                {
                    "type": "compaction",
                    "encrypted_content": format_translation.encode_fake_compaction("carry this forward"),
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue from here"}],
                },
            ]
        )

        self.assertEqual(len(sanitized), 2)
        self.assertEqual(sanitized[0]["type"], "message")
        self.assertEqual(sanitized[0]["role"], "user")
        self.assertIn("carry this forward", sanitized[0]["content"][0]["text"])
        self.assertEqual(
            sanitized[1],
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "continue from here"}],
            },
        )

    def test_sanitize_input_uses_latest_opaque_compaction_as_boundary(self):
        sanitized = format_translation.sanitize_input(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "old context"}],
                },
                {
                    "type": "compaction",
                    "encrypted_content": "opaque-upstream-token",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue from here"}],
                },
            ]
        )

        self.assertEqual(
            sanitized,
            [
                {
                    "type": "reasoning",
                    "encrypted_content": "opaque-upstream-token",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue from here"}],
                },
            ],
        )

    def test_sanitize_input_replaces_inline_tool_images_with_text_summary(self):
        sanitized = format_translation.sanitize_input(
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AAAA",
                            "detail": "original",
                        }
                    ],
                }
            ]
        )

        self.assertEqual(
            sanitized,
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": [
                        {
                            "type": "input_text",
                            "text": "[inline tool image omitted: image/png, 26 chars, detail=original]",
                        }
                    ],
                }
            ],
        )

    def test_sanitize_input_strips_content_from_non_message_items(self):
        # GHCP's Responses API rejects a non-empty ``content`` array on
        # non-message items: it emits "Invalid 'input[N].content': array too
        # long. Expected an array with maximum length 0".
        sanitized = format_translation.sanitize_input(
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "shell",
                    "arguments": "{}",
                    "content": [{"type": "input_text", "text": "leaked"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                    "content": [{"type": "input_text", "text": "leaked"}],
                },
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "input_text", "text": "leaked"}],
                },
            ]
        )

        for item in sanitized:
            self.assertNotIn("content", item, item)

    def test_sanitize_input_can_strip_unverifiable_reasoning_ciphertext(self):
        sanitized = format_translation.sanitize_input(
            [
                {
                    "type": "reasoning",
                    "id": "rs_keep",
                    "summary": [{"type": "summary_text", "text": "visible reasoning summary"}],
                    "encrypted_content": "foreign-ciphertext",
                    "content": [{"type": "reasoning_text", "text": "hidden"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_drop",
                    "encrypted_content": "ciphertext-only",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue"}],
                },
            ],
            preserve_encrypted_content=False,
        )

        self.assertEqual(
            sanitized,
            [
                {
                    "type": "reasoning",
                    "id": "rs_keep",
                    "summary": [{"type": "summary_text", "text": "visible reasoning summary"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue"}],
                },
            ],
        )

    def test_sanitize_input_can_drop_reasoning_items_for_tool_history_replay(self):
        sanitized = format_translation.sanitize_input(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "start"}],
                },
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "volatile"}],
                    "encrypted_content": "ciphertext",
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "spawn_agent",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "agent",
                },
            ],
            preserve_encrypted_content=False,
            drop_reasoning_items=True,
        )

        self.assertEqual(
            sanitized,
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "start"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "spawn_agent",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "agent",
                },
            ],
        )

    def test_anthropic_request_to_chat_stream_requests_usage_chunks(self):
        body = {
            "model": "claude-sonnet-4.6",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        self.assertEqual(outbound["stream_options"], {"include_usage": True})

    def test_anthropic_request_to_responses_translates_tool_use_and_tool_result(self):
        body = {
            "model": "gpt-5.4",
            "system": "Follow the spec",
            "metadata": {"user_id": "local-only"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"file": "main.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tool_1", "content": "file contents"},
                    ],
                },
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {"file": {"type": "string"}}},
                }
            ],
        }

        translated = format_translation.anthropic_request_to_responses(body)

        self.assertEqual(translated["model"], "gpt-5.4")
        self.assertEqual(translated["input"][0]["role"], "developer")
        self.assertEqual(translated["input"][0]["content"][0]["text"], "Follow the spec")
        self.assertEqual(translated["input"][1]["role"], "user")
        self.assertEqual(translated["input"][1]["content"][0]["text"], "hello")
        self.assertEqual(translated["input"][2]["type"], "function_call")
        self.assertEqual(translated["input"][2]["call_id"], "tool_1")
        self.assertNotIn("id", translated["input"][2])
        self.assertNotIn("status", translated["input"][2])
        self.assertEqual(translated["input"][3]["type"], "function_call_output")
        self.assertEqual(translated["input"][3]["output"], "file contents")
        self.assertNotIn("status", translated["input"][3])
        self.assertEqual(translated["tools"][0]["name"], "Read")
        self.assertNotIn("metadata", translated)
        self.assertNotIn("instructions", translated)
        self.assertEqual(translated["text"], {"format": {"type": "text"}, "verbosity": "low"})

    def test_anthropic_request_to_responses_strips_temperature_for_gpt_5_4_mini(self):
        body = {
            "model": "gpt-5.4-mini",
            "max_tokens": 64,
            "temperature": 0,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "approve or block"}],
                }
            ],
        }

        translated = format_translation.anthropic_request_to_responses(body)

        self.assertEqual(translated["model"], "gpt-5.4-mini")
        self.assertEqual(translated["max_output_tokens"], 64)
        self.assertEqual(translated["reasoning"], {"effort": "high"})
        self.assertNotIn("temperature", translated)

    def test_responses_request_to_chat_translates_function_calls_and_outputs(self):
        body = {
            "model": "claude-opus-4.6",
            "instructions": "Be helpful",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file":"main.py"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file contents",
                },
            ],
            "stream": True,
        }

        translated = format_translation.responses_request_to_chat(body)

        self.assertEqual(translated["model"], "claude-opus-4.6")
        self.assertEqual(translated["messages"][0]["role"], "system")
        self.assertEqual(translated["messages"][1]["role"], "user")
        self.assertEqual(translated["messages"][1]["content"], "hello")
        self.assertEqual(translated["messages"][2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(translated["messages"][3]["role"], "tool")
        self.assertEqual(translated["messages"][3]["content"], "file contents")
        self.assertEqual(translated["stream_options"], {"include_usage": True})

    def test_responses_request_to_chat_defers_user_messages_until_tool_output(self):
        body = {
            "model": "claude-opus-4.7",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file":"main.py"}',
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Use that for the answer."}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file contents",
                },
            ],
        }

        translated = format_translation.responses_request_to_chat(body)

        self.assertEqual([message["role"] for message in translated["messages"]], ["user", "assistant", "tool", "user"])
        self.assertEqual(translated["messages"][2]["tool_call_id"], "call_1")
        self.assertEqual(translated["messages"][3]["content"], "Use that for the answer.")

    def test_responses_request_to_chat_skips_web_search_call_items(self):
        body = {
            "model": "claude-opus-4.6",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "latest launch news"}],
                },
                {
                    "type": "web_search_call",
                    "id": "ws_123",
                    "status": "completed",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Here is the summary."}],
                },
            ],
        }

        translated = format_translation.responses_request_to_chat(body)

        self.assertEqual(
            translated["messages"],
            [
                {"role": "user", "content": "latest launch news"},
                {"role": "assistant", "content": "Here is the summary."},
            ],
        )

    def test_responses_request_to_chat_omits_empty_tool_config(self):
        body = {
            "model": "claude-opus-4.6",
            "input": "hello",
            "tools": [],
            "tool_choice": "auto",
        }

        translated = format_translation.responses_request_to_chat(body)

        self.assertNotIn("tools", translated)
        self.assertNotIn("tool_choice", translated)
        self.assertEqual(translated["messages"][0]["content"], "hello")

    def test_responses_request_to_chat_merges_instructions_and_developer_messages(self):
        body = {
            "model": "claude-sonnet-4.6",
            "instructions": "Base instructions",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Extra developer guidance"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            ],
        }

        translated = format_translation.responses_request_to_chat(body)

        self.assertEqual([message["role"] for message in translated["messages"]], ["system", "user"])
        system_content = translated["messages"][0]["content"]
        if isinstance(system_content, str):
            system_text = system_content
        else:
            system_text = "".join(
                item.get("text", "")
                for item in system_content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        self.assertIn("Base instructions", system_text)
        self.assertIn("Extra developer guidance", system_text)
        self.assertEqual(translated["messages"][1]["content"], "hello")

    def test_response_payload_to_anthropic_maps_function_call_and_usage(self):
        payload = {
            "id": "resp_123",
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": '{"file":"main.py"}',
                },
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 3,
                "input_tokens_details": {"cached_tokens": 5},
            },
        }

        translated = format_translation.response_payload_to_anthropic(payload, fallback_model="gpt-5.4")

        self.assertEqual(translated["id"], "resp_123")
        self.assertEqual(translated["content"][0], {"type": "text", "text": "hello"})
        self.assertEqual(translated["content"][1]["type"], "tool_use")
        self.assertEqual(translated["content"][1]["id"], "call_1")
        self.assertEqual(translated["usage"]["input_tokens"], 7)
        self.assertEqual(translated["usage"]["cache_read_input_tokens"], 5)
        self.assertEqual(translated["stop_reason"], "tool_use")

    def test_chat_completion_to_response_maps_tool_calls_and_usage(self):
        payload = {
            "id": "chatcmpl_123",
            "model": "claude-opus-4.6",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "Read", "arguments": '{"file":"main.py"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 6},
            },
        }

        translated = format_translation.chat_completion_to_response(payload)

        self.assertEqual(translated["id"], "chatcmpl_123")
        self.assertEqual(translated["output"][0]["type"], "message")
        self.assertEqual(translated["output"][1]["type"], "function_call")
        self.assertEqual(translated["output"][1]["call_id"], "call_1")
        self.assertEqual(translated["output_text"], "hello")
        self.assertEqual(translated["usage"]["input_tokens"], 20)
        self.assertEqual(translated["usage"]["input_tokens_details"], {"cached_tokens": 6})

    def test_chat_completion_to_response_prefers_fallback_model_when_present(self):
        payload = {
            "id": "chatcmpl_123",
            "model": "gpt-5.4",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        translated = format_translation.chat_completion_to_response(payload, fallback_model="claude-sonnet-4.6")

        self.assertEqual(translated["model"], "claude-sonnet-4.6")

    def test_chat_completion_to_compaction_response_encodes_summary_item(self):
        payload = {
            "id": "chatcmpl_123",
            "model": "claude-opus-4.7",
            "created": 123,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "summary text",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 6},
            },
        }

        translated = format_translation.chat_completion_to_compaction_response(payload)

        self.assertEqual(translated["id"], "chatcmpl_123")
        self.assertEqual(translated["model"], "claude-opus-4.7")
        self.assertEqual(translated["output_text"], "summary text")
        self.assertEqual(translated["usage"]["input_tokens"], 20)
        self.assertEqual(translated["output"][0]["type"], "compaction")
        self.assertEqual(
            format_translation.decode_fake_compaction(translated["output"][0]["encrypted_content"]),
            "summary text",
        )

    def test_response_payload_to_anthropic_prefers_fallback_model_when_present(self):
        payload = {
            "id": "resp_123",
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
        }

        translated = format_translation.response_payload_to_anthropic(payload, fallback_model="claude-sonnet-4.6")

        self.assertEqual(translated["model"], "claude-sonnet-4.6")

    def test_chat_completion_to_anthropic_maps_cached_usage_tokens(self):
        payload = {
            "id": "chatcmpl_123",
            "model": "claude-sonnet-4.6",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "cache_creation_input_tokens": 20,
                },
            },
        }

        translated = format_translation.chat_completion_to_anthropic(payload)

        self.assertEqual(translated["usage"]["input_tokens"], 20)
        self.assertEqual(translated["usage"]["output_tokens"], 25)
        self.assertEqual(translated["usage"]["cache_read_input_tokens"], 80)
        self.assertEqual(translated["usage"]["cache_creation_input_tokens"], 0)

    def test_chat_completion_to_anthropic_subtracts_cache_reads_from_input_tokens(self):
        payload = {
            "id": "chatcmpl_456",
            "model": "claude-sonnet-4.6",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 135200,
                "completion_tokens": 690,
                "prompt_tokens_details": {
                    "cached_tokens": 125000,
                },
            },
        }

        translated = format_translation.chat_completion_to_anthropic(payload)

        self.assertEqual(translated["usage"]["input_tokens"], 10200)
        self.assertEqual(translated["usage"]["cache_read_input_tokens"], 125000)
        self.assertEqual(translated["usage"]["cache_creation_input_tokens"], 0)

    def test_anthropic_stream_refreshes_message_start_usage_when_openai_usage_arrives_late(self):
        self.assertEqual(
            format_translation.extract_reasoning_from_chat_delta({"thinking": {"text": "direct text"}}),
            "direct text",
        )

        chunks = [
            (
                'event: message\n'
                'data: {"id":"chatcmpl_123","model":"claude-haiku-4.5","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            ).encode("utf-8"),
            (
                'event: message\n'
                'data: {"id":"chatcmpl_123","model":"claude-haiku-4.5","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":30,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":10}}}\n\n'
            ).encode("utf-8"),
            b"data: [DONE]\n\n",
        ]

        class FakeUpstream:
            status_code = 200

            async def aiter_bytes(self):
                for chunk in chunks:
                    yield chunk

            async def aread(self):
                return b""

            async def aclose(self):
                return None

        class FakeClient:
            def build_request(self, *args, **kwargs):
                return object()

            async def aclose(self):
                return None

        async def collect_stream_body():
            response = await proxy.proxy_anthropic_streaming_response(
                "https://example.invalid/chat/completions",
                {"Authorization": "Bearer test"},
                {"stream": True},
                "claude-haiku-4.5",
            )
            body = b""
            async for chunk in response.body_iterator:
                body += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
            return body.decode("utf-8")

        with (
            mock.patch.object(proxy.httpx, "AsyncClient", return_value=FakeClient()),
            mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(return_value=FakeUpstream())),
        ):
            body = proxy.asyncio.run(collect_stream_body())

        self.assertEqual(body.count("event: message_start"), 2)
        self.assertIn(
            '"usage":{"input_tokens":20,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":10}',
            body,
        )

    def test_upstream_request_error_status_and_message_maps_timeout_to_504(self):
        request = httpx.Request("POST", "https://example.invalid/responses")
        timeout_error = httpx.ReadTimeout("Timed out", request=request)

        self.assertEqual(
            format_translation.upstream_request_error_status_and_message(timeout_error),
            (504, "Upstream request timed out"),
        )

    def test_anthropic_error_payload_from_openai_uses_anthropic_shape(self):
        payload = {
            "error": {
                "message": "bad request",
                "type": "invalid_request_error",
            }
        }

        translated = format_translation.anthropic_error_payload_from_openai(payload, 400)

        self.assertEqual(
            translated,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "bad request",
                },
            },
        )

    def test_anthropic_error_response_from_upstream_translates_openai_error(self):
        upstream = httpx.Response(
            429,
            json={
                "error": {
                    "message": "rate limited",
                }
            },
            headers={"retry-after": "12"},
        )

        response = format_translation.anthropic_error_response_from_upstream(upstream)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "12")
        self.assertEqual(
            response.body,
            b'{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}',
        )

    def test_resolve_copilot_model_name_maps_dated_haiku_to_canonical_form(self):
        self.assertEqual(
            format_translation.resolve_copilot_model_name("claude-haiku-4-5-20251001"),
            "claude-haiku-4.5",
        )

    def test_resolve_copilot_model_name_preserves_supported_opus_4_7(self):
        self.assertEqual(
            format_translation.resolve_copilot_model_name("claude-opus-4.7"),
            "claude-opus-4.7",
        )

    def test_resolve_copilot_model_name_maps_generic_opus_to_latest_supported_version(self):
        self.assertEqual(
            format_translation.resolve_copilot_model_name("opus"),
            "claude-opus-4.7",
        )


    def test_anthropic_request_to_responses_uses_output_text_for_assistant(self):
        """Assistant content must use output_text, not input_text, in Responses API payloads."""
        body = {
            "model": "gpt-5.4",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi there"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "thanks"}],
                },
            ],
        }

        translated = format_translation.anthropic_request_to_responses(body)

        # Find the assistant message in input
        assistant_msgs = [item for item in translated["input"]
                          if isinstance(item, dict) and item.get("role") == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)
        ast_content = assistant_msgs[0]["content"]
        self.assertEqual(ast_content[0]["type"], "output_text")
        self.assertEqual(ast_content[0]["text"], "hi there")

        # User messages should still use input_text
        user_msgs = [item for item in translated["input"]
                     if isinstance(item, dict) and item.get("role") == "user"]
        for user_msg in user_msgs:
            for part in user_msg.get("content", []):
                self.assertEqual(part["type"], "input_text")

    def test_anthropic_request_to_responses_replays_responses_item_metadata(self):
        body = {
            "model": "gpt-5.4",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "response_item_id": "msg_prev",
                            "response_item_status": "completed",
                            "response_annotations": [],
                        },
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Read",
                            "input": {"file": "main.py"},
                            "response_item_id": "fc_prev",
                            "response_item_status": "completed",
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "contents",
                        }
                    ],
                },
            ],
        }

        translated = format_translation.anthropic_request_to_responses(body)

        assistant_msg = translated["input"][1]
        self.assertEqual(assistant_msg["id"], "msg_prev")
        self.assertEqual(assistant_msg["status"], "completed")
        self.assertEqual(
            assistant_msg["content"],
            [{"type": "output_text", "text": "hi", "annotations": []}],
        )
        function_call = translated["input"][2]
        self.assertEqual(function_call["id"], "fc_prev")
        self.assertEqual(function_call["call_id"], "call_1")
        self.assertEqual(function_call["status"], "completed")
        tool_output = translated["input"][3]
        self.assertNotIn("status", tool_output)

    def test_anthropic_request_to_responses_does_not_synthesize_replay_item_ids(self):
        body = {
            "model": "gpt-5.4",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "Read",
                            "input": {"file": "main.py"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "contents"}],
                },
            ],
        }

        first = format_translation.anthropic_request_to_responses(body)
        second = format_translation.anthropic_request_to_responses(body)

        self.assertNotIn("id", first["input"][1])
        self.assertNotIn("id", first["input"][2])
        self.assertEqual(first, second)

    def test_anthropic_request_to_responses_strips_copilot_cache_control(self):
        """copilot_cache_control is a Chat API extension and must not leak into Responses API payloads."""
        body = {
            "model": "gpt-5.4",
            "system": [
                {"type": "text", "text": "first"},
                {
                    "type": "text",
                    "text": "cached system",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
        }

        translated = format_translation.anthropic_request_to_responses(body)

        # The payload must not contain copilot_cache_control anywhere.
        import json
        raw = json.dumps(translated)
        self.assertNotIn("cache_control", raw)
        self.assertNotIn("copilot_cache_control", raw)

        # Content should still be present (only the cache hint is stripped).
        self.assertIn("cached system", translated["input"][0]["content"][1]["text"])
        user_msg = translated["input"][1]
        self.assertEqual(user_msg["content"][0]["text"], "hello")

    def test_anthropic_request_to_responses_strips_volatile_billing_system_block(self):
        body = {
            "model": "gpt-5.4",
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=2.1.98.134; cc_entrypoint=cli; cch=a38f0;",
                },
                {
                    "type": "text",
                    "text": "stable guidance",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        }

        first = format_translation.anthropic_request_to_responses(body)
        body["system"][0]["text"] = (
            "x-anthropic-billing-header: cc_version=2.1.98.134; cc_entrypoint=cli; cch=9b349;"
        )
        second = format_translation.anthropic_request_to_responses(body)

        self.assertEqual(first["input"][0], second["input"][0])
        self.assertEqual(first["input"][0]["content"], [{"type": "input_text", "text": "stable guidance"}])
        self.assertNotIn("x-anthropic-billing-header", str(first))

    def test_anthropic_request_to_responses_strips_leading_billing_line_from_system_string(self):
        body = {
            "model": "gpt-5.4",
            "system": (
                "x-anthropic-billing-header: cc_version=2.1.98.134; cc_entrypoint=cli; cch=a38f0;\n\n"
                "stable guidance"
            ),
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        }

        translated = format_translation.anthropic_request_to_responses(body)

        self.assertEqual(translated["input"][0]["content"][0]["text"], "stable guidance")

    def test_anthropic_request_to_chat_keeps_copilot_cache_control(self):
        """copilot_cache_control must survive in the Chat API path so GHCP caching works."""
        body = {
            "model": "claude-sonnet-4.6",
            "system": [
                {
                    "type": "text",
                    "text": "cached system",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "k"))

        import json
        raw = json.dumps(outbound)
        self.assertIn("copilot_cache_control", raw)
        # System message content list should carry the cache hint
        sys_content = outbound["messages"][0]["content"]
        self.assertIsInstance(sys_content, list)
        self.assertEqual(sys_content[0]["copilot_cache_control"], {"type": "ephemeral"})
        # User message content should carry the cache hint
        user_content = outbound["messages"][1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0]["copilot_cache_control"], {"type": "ephemeral"})

    def test_iter_sse_messages_handles_split_utf8_code_point_across_chunks(self):
        payload = (
            'event: message\n'
            'data: {"choices":[{"delta":{"content":"hello – world"}}]}\n\n'
        ).encode("utf-8")
        split_at = payload.index("–".encode("utf-8")) + 1
        chunks = [payload[:split_at], payload[split_at:]]

        async def byte_iter():
            for chunk in chunks:
                yield chunk

        async def collect():
            return [message async for message in format_translation.iter_sse_messages(byte_iter())]

        messages = proxy.asyncio.run(collect())

        self.assertEqual(
            messages,
            [("message", '{"choices":[{"delta":{"content":"hello – world"}}]}')],
        )




class ResponsesToAnthropicMessagesTests(unittest.TestCase):
    def test_instructions_become_system_and_input_message_text(self):
        body = {
            "model": "claude-sonnet-4.6",
            "instructions": "You are helpful.",
            "metadata": {"user_id": "local-only"},
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi"}],
                }
            ],
            "max_output_tokens": 1024,
        }
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertEqual(out["system"], "You are helpful.")
        self.assertEqual(out["max_tokens"], 1024)
        self.assertEqual(out["metadata"], {"user_id": "local-only"})
        self.assertEqual(
            out["messages"],
            [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        )

    def test_default_max_tokens_when_unset(self):
        body = {"model": "claude-sonnet-4.6", "input": []}
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertEqual(out["max_tokens"], 64000)
        self.assertEqual(out["messages"], [])
        self.assertNotIn("system", out)

    def test_function_call_and_output_become_tool_use_and_tool_result(self):
        body = {
            "model": "claude-sonnet-4.6",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "do it"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "do_thing",
                    "arguments": '{"x": 1}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "result text",
                },
            ],
        }
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertEqual(len(out["messages"]), 3)
        self.assertEqual(out["messages"][0]["role"], "user")
        self.assertEqual(out["messages"][1]["role"], "assistant")
        self.assertEqual(
            out["messages"][1]["content"],
            [{"type": "tool_use", "id": "call_1", "name": "do_thing", "input": {"x": 1}}],
        )
        self.assertEqual(out["messages"][2]["role"], "user")
        self.assertEqual(
            out["messages"][2]["content"],
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [{"type": "text", "text": "result text"}],
                }
            ],
        )

    def test_sanitize_input_preserves_all_encrypted_reasoning_ciphertext(self):
        sanitized = format_translation.sanitize_input(
            [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "summary 1"}],
                    "encrypted_content": "ciphertext-1",
                },
                {
                    "type": "reasoning",
                    "id": "rs_2",
                    "encrypted_content": "ciphertext-2",
                },
                {
                    "type": "reasoning",
                    "id": "rs_3",
                    "summary": [{"type": "summary_text", "text": "summary 3"}],
                    "encrypted_content": "ciphertext-3",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue"}],
                },
            ],
        )

        self.assertEqual(
            sanitized,
            [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "summary 1"}],
                    "encrypted_content": "ciphertext-1",
                },
                {
                    "type": "reasoning",
                    "id": "rs_2",
                    "encrypted_content": "ciphertext-2",
                },
                {
                    "type": "reasoning",
                    "id": "rs_3",
                    "summary": [{"type": "summary_text", "text": "summary 3"}],
                    "encrypted_content": "ciphertext-3",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "continue"}],
                },
            ],
        )

    def test_custom_tool_history_becomes_text_transcript(self):
        body = {
            "model": "claude-sonnet-4.6",
            "input": [
                {
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "name": "apply_patch",
                    "input": "*** Begin Patch\n*** End Patch",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_1",
                    "output": "Exit code: 0\nSuccess.",
                },
            ],
        }
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertEqual(len(out["messages"]), 2)
        self.assertEqual(out["messages"][0]["role"], "assistant")
        self.assertIn(
            "[Custom tool call (call_1)] apply_patch",
            out["messages"][0]["content"][0]["text"],
        )
        self.assertEqual(out["messages"][1]["role"], "user")
        self.assertIn(
            "[Custom tool result (call_1)]",
            out["messages"][1]["content"][0]["text"],
        )

    def test_consecutive_function_calls_coalesce_into_single_assistant_turn(self):
        body = {
            "model": "claude-sonnet-4.6",
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}"},
            ],
        }
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertEqual(len(out["messages"]), 1)
        self.assertEqual(out["messages"][0]["role"], "assistant")
        self.assertEqual(len(out["messages"][0]["content"]), 2)

    def test_consecutive_function_call_outputs_coalesce(self):
        body = {
            "model": "claude-sonnet-4.6",
            "input": [
                {"type": "function_call_output", "call_id": "c1", "output": "a"},
                {"type": "function_call_output", "call_id": "c2", "output": "b"},
            ],
        }
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertEqual(len(out["messages"]), 1)
        self.assertEqual(out["messages"][0]["role"], "user")
        self.assertEqual(len(out["messages"][0]["content"]), 2)

    def test_reasoning_item_becomes_thinking_block_with_signature(self):
        body = {
            "model": "claude-sonnet-4.6",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "thought"}],
                    "encrypted_content": "OPAQUE-SIG",
                }
            ],
        }
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertEqual(out["messages"][0]["role"], "assistant")
        self.assertEqual(
            out["messages"][0]["content"],
            [{"type": "thinking", "thinking": "thought", "signature": "OPAQUE-SIG"}],
        )

    def test_tool_choice_mapping_table(self):
        cases = [
            ("auto", {"type": "auto"}),
            ("required", {"type": "any"}),
            ("none", {"type": "none"}),
            ({"type": "function", "name": "foo"}, {"type": "tool", "name": "foo"}),
            ({"type": "auto"}, {"type": "auto"}),
        ]
        for src, expected in cases:
            body = {"model": "claude-sonnet-4.6", "input": [], "tool_choice": src}
            out = format_translation.responses_request_to_anthropic_messages(body)
            self.assertEqual(out["tool_choice"], expected, f"for {src!r}")

    def test_tools_translated_with_input_schema(self):
        body = {
            "model": "claude-sonnet-4.6",
            "input": [],
            "tools": [
                {
                    "type": "function",
                    "name": "do",
                    "description": "desc",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                }
            ],
        }
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertEqual(
            out["tools"],
            [
                {
                    "name": "do",
                    "description": "desc",
                    "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                }
            ],
        )

    def test_drops_dangerous_execute_code_tool_and_choice(self):
        body = {
            "model": "claude-sonnet-4.6",
            "input": [],
            "tools": [
                {
                    "type": "function",
                    "name": "mcp__ide__executeCode",
                    "description": "Execute code",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "mcp__ide__executeCode"},
            "parallel_tool_calls": False,
        }

        out = format_translation.responses_request_to_anthropic_messages(body)

        self.assertNotIn("tools", out)
        self.assertNotIn("tool_choice", out)

    def test_dangerous_tool_choice_does_not_become_auto_when_safe_tools_remain(self):
        body = {
            "model": "claude-sonnet-4.6",
            "input": [],
            "tools": [
                {
                    "type": "function",
                    "name": "read",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "function",
                    "name": "mcp__ide__executeCode",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
            "tool_choice": {"type": "function", "name": "mcp__ide__executeCode"},
            "parallel_tool_calls": False,
        }

        out = format_translation.responses_request_to_anthropic_messages(body)

        self.assertEqual([tool["name"] for tool in out["tools"]], ["read"])
        self.assertNotIn("tool_choice", out)

    def test_responses_prompt_cache_key_adds_messages_cache_breakpoints(self):
        body = {
            "model": "claude-sonnet-4.6",
            "prompt_cache_key": "session-123",
            "instructions": "system prompt",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "first"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "second"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "third"}],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }

        out = format_translation.responses_request_to_anthropic_messages(body)

        self.assertIsInstance(out["system"], list)
        self.assertEqual(out["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(out["tools"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", out["messages"][0]["content"][0])
        self.assertEqual(out["messages"][1]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(out["messages"][2]["content"][0]["cache_control"], {"type": "ephemeral"})

        def count_cache_controls(value):
            if isinstance(value, dict):
                return int(isinstance(value.get("cache_control"), dict)) + sum(
                    count_cache_controls(v) for v in value.values()
                )
            if isinstance(value, list):
                return sum(count_cache_controls(v) for v in value)
            return 0

        self.assertEqual(count_cache_controls(out), 4)

    def test_reasoning_effort_becomes_adaptive_thinking(self):
        body = {
            "model": "claude-sonnet-4.6",
            "input": [],
            "reasoning": {"effort": "high"},
        }
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertEqual(out["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(out["output_config"], {"effort": "high"})

    def test_parallel_tool_calls_inverts_to_disable_parallel(self):
        body = {
            "model": "claude-sonnet-4.6",
            "input": [],
            "tools": [
                {
                    "type": "function",
                    "name": "do",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "parallel_tool_calls": False,
        }
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertNotIn("disable_parallel_tool_use", out)
        self.assertEqual(
            out["tool_choice"],
            {"type": "auto", "disable_parallel_tool_use": True},
        )

    def test_parallel_tool_calls_disables_existing_tool_choice(self):
        body = {
            "model": "claude-sonnet-4.6",
            "input": [],
            "tools": [
                {
                    "type": "function",
                    "name": "do",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "function", "name": "do"},
            "parallel_tool_calls": False,
        }
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertNotIn("disable_parallel_tool_use", out)
        self.assertEqual(
            out["tool_choice"],
            {"type": "tool", "name": "do", "disable_parallel_tool_use": True},
        )

    def test_developer_message_skipped_from_messages(self):
        body = {
            "model": "claude-sonnet-4.6",
            "instructions": "sys",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "ignored"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            ],
        }
        out = format_translation.responses_request_to_anthropic_messages(body)
        self.assertEqual(len(out["messages"]), 1)
        self.assertEqual(out["messages"][0]["role"], "user")

    def test_round_trip_anthropic_to_responses_back_to_anthropic(self):
        original = {
            "model": "claude-sonnet-4.6",
            "system": "sys prompt",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "ok"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "search",
                            "input": {"q": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "results"}],
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ],
            "tool_choice": {"type": "auto"},
        }
        as_responses = format_translation.anthropic_request_to_responses(original)
        back = format_translation.responses_request_to_anthropic_messages(as_responses)

        self.assertEqual(back["system"], "sys prompt")
        self.assertEqual(back["max_tokens"], 2048)
        # Tool round-trip
        self.assertEqual(back["tools"][0]["name"], "search")
        self.assertEqual(back["tools"][0]["input_schema"], original["tools"][0]["input_schema"])
        self.assertEqual(back["tool_choice"], {"type": "auto"})

        # Message structure preserved (3 turns: user, assistant w/ text+tool_use, user w/ tool_result)
        self.assertEqual([m["role"] for m in back["messages"]], ["user", "assistant", "user"])
        # Assistant turn has both text and tool_use blocks
        assistant_block_types = [b["type"] for b in back["messages"][1]["content"]]
        self.assertIn("text", assistant_block_types)
        self.assertIn("tool_use", assistant_block_types)
        # Tool use id preserved
        tool_use = next(b for b in back["messages"][1]["content"] if b["type"] == "tool_use")
        self.assertEqual(tool_use["id"], "toolu_1")
        self.assertEqual(tool_use["input"], {"q": "x"})
        # Tool result preserved
        self.assertEqual(back["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(back["messages"][2]["content"][0]["tool_use_id"], "toolu_1")


class AnthropicResponseToResponsesTests(unittest.TestCase):
    def test_text_only_response(self):
        anth = {
            "id": "msg_abc",
            "model": "claude-sonnet-4.6",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        out = format_translation.anthropic_response_to_responses(anth)
        self.assertEqual(out["id"], "msg_abc")
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["model"], "claude-sonnet-4.6")
        self.assertEqual(len(out["output"]), 1)
        msg = out["output"][0]
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], [{"type": "output_text", "text": "hello", "annotations": []}])
        self.assertEqual(out["usage"], {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7})

    def test_tool_use_response(self):
        anth = {
            "id": "msg_t",
            "model": "claude-sonnet-4.6",
            "content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "toolu_9", "name": "search", "input": {"q": "y"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }
        out = format_translation.anthropic_response_to_responses(anth)
        self.assertEqual(out["status"], "completed")
        types = [item["type"] for item in out["output"]]
        self.assertEqual(types, ["message", "function_call"])
        fc = out["output"][1]
        self.assertEqual(fc["call_id"], "toolu_9")
        self.assertEqual(fc["name"], "search")
        self.assertEqual(json.loads(fc["arguments"]), {"q": "y"})

    def test_response_payload_to_anthropic_preserves_replay_metadata(self):
        payload = {
            "id": "resp_1",
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "message",
                    "id": "msg_prev",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                },
                {
                    "type": "function_call",
                    "id": "fc_prev",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": "{\"file\":\"main.py\"}",
                    "status": "completed",
                },
            ],
        }

        translated = format_translation.response_payload_to_anthropic(payload)

        text_block = translated["content"][0]
        self.assertEqual(text_block["response_item_id"], "msg_prev")
        self.assertEqual(text_block["response_item_status"], "completed")
        self.assertEqual(text_block["response_annotations"], [])
        tool_block = translated["content"][1]
        self.assertEqual(tool_block["id"], "call_1")
        self.assertEqual(tool_block["response_item_id"], "fc_prev")
        self.assertEqual(tool_block["response_item_status"], "completed")

    def test_thinking_plus_text_with_signature(self):
        anth = {
            "id": "msg_th",
            "model": "claude-sonnet-4.6",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "SIG"},
                {"type": "text", "text": "answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        out = format_translation.anthropic_response_to_responses(anth)
        types = [item["type"] for item in out["output"]]
        self.assertEqual(types, ["reasoning", "message"])
        self.assertEqual(out["output"][0]["encrypted_content"], "SIG")
        self.assertEqual(
            out["output"][0]["summary"], [{"type": "summary_text", "text": "hmm"}]
        )

    def test_thinking_signature_with_id_is_split_for_responses(self):
        anth = {
            "id": "msg_th",
            "model": "claude-sonnet-4.6",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "ENC@reason_1"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        out = format_translation.anthropic_response_to_responses(anth)
        self.assertEqual(out["output"][0]["id"], "reason_1")
        self.assertEqual(out["output"][0]["encrypted_content"], "ENC")

    def test_max_tokens_stop_marks_incomplete(self):
        anth = {
            "id": "x",
            "model": "claude-sonnet-4.6",
            "content": [{"type": "text", "text": "partial"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        out = format_translation.anthropic_response_to_responses(anth)
        self.assertEqual(out["status"], "incomplete")
        self.assertEqual(out["incomplete_details"], {"reason": "max_output_tokens"})

    def test_usage_carries_cache_details(self):
        anth = {
            "id": "x",
            "model": "claude-sonnet-4.6",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 7,
                "output_tokens": 3,
                "cache_read_input_tokens": 4,
                "cache_creation_input_tokens": 2,
            },
        }
        out = format_translation.anthropic_response_to_responses(anth)
        # Responses usage is gross input + cached details; Anthropic Messages
        # usage is fresh input + separate cache read/write counters.
        self.assertEqual(out["usage"]["input_tokens"], 13)
        self.assertEqual(out["usage"]["output_tokens"], 3)
        self.assertEqual(out["usage"]["total_tokens"], 16)
        self.assertEqual(
            out["usage"]["input_tokens_details"],
            {"cached_tokens": 4, "cache_creation_input_tokens": 2},
        )


class ReasoningSignatureRoundTripTests(unittest.TestCase):
    def test_encrypted_content_signature_round_trips(self):
        responses_body = {
            "model": "claude-sonnet-4.6",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "think"}],
                    "encrypted_content": "ABC",
                }
            ],
        }
        anth = format_translation.responses_request_to_anthropic_messages(responses_body)
        thinking_block = anth["messages"][0]["content"][0]
        self.assertEqual(thinking_block["signature"], "ABC")

        # And back via the response translator (since the response shape mirrors
        # an assistant turn carrying the same thinking block).
        fake_response = {
            "id": "x",
            "model": "claude-sonnet-4.6",
            "content": [thinking_block],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        round_tripped = format_translation.anthropic_response_to_responses(fake_response)
        self.assertEqual(round_tripped["output"][0]["encrypted_content"], "ABC")

    def test_responses_reasoning_id_is_carried_in_anthropic_signature(self):
        responses_body = {
            "model": "claude-sonnet-4.6",
            "input": [
                {
                    "id": "reason_123",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "think"}],
                    "encrypted_content": "ABC",
                }
            ],
        }

        anth = format_translation.responses_request_to_anthropic_messages(responses_body)
        thinking_block = anth["messages"][0]["content"][0]
        self.assertEqual(thinking_block["signature"], "ABC@reason_123")

        fake_response = {
            "id": "x",
            "model": "claude-sonnet-4.6",
            "content": [thinking_block],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        round_tripped = format_translation.anthropic_response_to_responses(fake_response)
        self.assertEqual(round_tripped["output"][0]["id"], "reason_123")
        self.assertEqual(round_tripped["output"][0]["encrypted_content"], "ABC")


import json  # noqa: E402  (re-imported here to keep the new tests self-contained)


if __name__ == "__main__":
    unittest.main()
