import unittest
from types import SimpleNamespace
from unittest import mock

import gzip

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
        self.assertEqual(translated["input"][1]["role"], "user")
        self.assertEqual(translated["input"][1]["content"][0]["text"], "hello")
        self.assertEqual(translated["input"][2]["type"], "function_call")
        self.assertEqual(translated["input"][2]["call_id"], "tool_1")
        self.assertEqual(translated["input"][3]["type"], "function_call_output")
        self.assertEqual(translated["input"][3]["output"], "file contents")
        self.assertEqual(translated["tools"][0]["name"], "Read")

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
        self.assertNotIn("copilot_cache_control", raw)

        # Content should still be present (only the cache hint is stripped).
        dev_msg = translated["input"][0]
        self.assertEqual(dev_msg["role"], "developer")
        self.assertTrue(any("cached system" in item.get("text", "") for item in dev_msg["content"]))
        user_msg = translated["input"][1]
        self.assertEqual(user_msg["content"][0]["text"], "hello")

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


if __name__ == "__main__":
    unittest.main()
