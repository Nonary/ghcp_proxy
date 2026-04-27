import asyncio
import json
import unittest

import httpx

import format_translation


class FormatTranslationErrorsStreamCompactionMutantsTests(unittest.TestCase):
    def test_chat_stop_reason_to_anthropic_exact_mapping_and_passthrough(self):
        cases = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "stop_sequence",
            "tool_calls": "tool_use",
            "custom_stop": "custom_stop",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(format_translation.chat_stop_reason_to_anthropic(source), expected)

        for source in (None, 0, {"stop": "end_turn"}, ["stop"]):
            with self.subTest(source=source):
                self.assertIsNone(format_translation.chat_stop_reason_to_anthropic(source))

    def test_anthropic_error_payload_from_openai_exact_fallbacks(self):
        anthropic_payload = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "already anthropic"},
        }
        self.assertIs(
            format_translation.anthropic_error_payload_from_openai(anthropic_payload, 429),
            anthropic_payload,
        )

        self.assertEqual(
            format_translation.anthropic_error_payload_from_openai(
                {"error": {"type": "invalid_request_error", "message": "bad input"}},
                400,
                "fallback",
            ),
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad input"},
            },
        )
        self.assertEqual(
            format_translation.anthropic_error_payload_from_openai(
                {"error": {"type": "quota_error", "message": ""}, "detail": "quota detail"},
                429,
                "fallback",
            ),
            {
                "type": "error",
                "error": {"type": "quota_error", "message": "quota detail"},
            },
        )
        self.assertEqual(
            format_translation.anthropic_error_payload_from_openai({"detail": "plain detail"}, 404),
            {
                "type": "error",
                "error": {"type": "not_found_error", "message": "plain detail"},
            },
        )
        self.assertEqual(
            format_translation.anthropic_error_payload_from_openai({}, 503, "upstream unavailable"),
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "upstream unavailable"},
            },
        )
        self.assertEqual(
            format_translation.anthropic_error_payload_from_openai(None, 418),
            {
                "type": "error",
                "error": {"type": "api_error", "message": "Request failed"},
            },
        )

    def test_anthropic_error_response_from_upstream_preserves_retry_after_and_messages(self):
        upstream = httpx.Response(
            429,
            json={"error": {"type": "rate_limit_error", "message": "slow down"}},
            headers={"retry-after": "9"},
        )

        response = format_translation.anthropic_error_response_from_upstream(upstream)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "9")
        self.assertEqual(
            json.loads(response.body),
            {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        )

        text_upstream = httpx.Response(503, content=b"  backend offline  ")
        text_response = format_translation.anthropic_error_response_from_upstream(text_upstream)
        self.assertEqual(text_response.status_code, 503)
        self.assertNotIn("retry-after", text_response.headers)
        self.assertEqual(
            json.loads(text_response.body),
            {"type": "error", "error": {"type": "overloaded_error", "message": "backend offline"}},
        )

    def test_sse_encode_parse_and_iter_messages_exact_boundaries(self):
        self.assertEqual(
            format_translation.sse_encode("delta", {"text": "cafe \u00e9", "n": 1}),
            'event: delta\ndata: {"text":"cafe \u00e9","n":1}\n\n'.encode("utf-8"),
        )
        self.assertEqual(
            format_translation.parse_sse_block(
                ": ignored\r\nevent: update\r\ndata:  hello\r\ndata:{\"x\":1}"
            ),
            ("update", 'hello\n{"x":1}'),
        )
        self.assertEqual(format_translation.parse_sse_block("event: ping\n: no data"), ("ping", None))

        async def chunks():
            for chunk in (
                b"event: first\r\ndata: {\"a\":1}\r\n\r\n",
                b"data: caf\xc3",
                b"\xa9\n\n",
                "event: tail\ndata: done",
            ):
                yield chunk

        async def collect():
            return [item async for item in format_translation.iter_sse_messages(chunks())]

        self.assertEqual(
            asyncio.run(collect()),
            [("first", '{"a":1}'), (None, "caf\u00e9"), ("tail", "done")],
        )

    def test_extract_reasoning_from_chat_delta_accepts_all_supported_shapes(self):
        self.assertEqual(format_translation.extract_reasoning_from_chat_delta(None), "")
        self.assertEqual(format_translation.extract_reasoning_from_chat_delta({"thinking": "thought"}), "thought")
        self.assertEqual(
            format_translation.extract_reasoning_from_chat_delta(
                {
                    "thinking": "a",
                    "reasoning_content": "b",
                    "reasoning_text": "c",
                    "reasoning": {
                        "text": "d",
                        "summary": [{"text": "e"}, {"text": ""}, {"ignored": "x"}],
                    },
                }
            ),
            "abcde",
        )
        self.assertEqual(
            format_translation.extract_reasoning_from_chat_delta(
                {"reasoning": ["f", {"text": "g"}, {"text": ""}, 1]}
            ),
            "fg",
        )
        self.assertEqual(
            format_translation.extract_reasoning_from_chat_delta({"thinking": {"text": "h"}}),
            "h",
        )

    def test_merge_chat_system_prompt_parts_flattens_only_plain_text_parts(self):
        self.assertEqual(
            format_translation._merge_chat_system_prompt_parts(
                ["alpha", [{"type": "text", "text": "beta"}], "", [{"type": "text", "text": "gamma"}]]
            ),
            "alpha\n\nbeta\n\ngamma",
        )
        self.assertEqual(
            format_translation._merge_chat_system_prompt_parts([None, {"ignored": True}, "after"]),
            "after",
        )
        self.assertEqual(format_translation._merge_chat_system_prompt_parts([[], "after"]), "after")
        self.assertEqual(format_translation._merge_chat_system_prompt_parts([[{"type": "text"}]]), "")

        cached_part = {
            "type": "text",
            "text": "cached",
            "copilot_cache_control": {"type": "ephemeral"},
        }
        self.assertEqual(
            format_translation._merge_chat_system_prompt_parts([[cached_part], "tail"]),
            [
                cached_part,
                {"type": "text", "text": "\n\n"},
                {"type": "text", "text": "tail"},
            ],
        )
        self.assertEqual(format_translation._merge_chat_system_prompt_parts([None, [], ""]), "")

    def test_fake_compaction_summary_detection_requires_message_type_and_label_newline(self):
        label = format_translation.FAKE_COMPACTION_SUMMARY_LABEL
        fake_item = {
            "type": "Message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": f"{label}\nsummary"}],
        }

        self.assertTrue(format_translation._is_fake_compaction_summary_message(fake_item))
        self.assertFalse(format_translation._is_fake_compaction_summary_message(None))
        self.assertFalse(
            format_translation._is_fake_compaction_summary_message(
                {"type": "function_call_output", "output": f"{label}\nsummary"}
            )
        )
        self.assertFalse(
            format_translation._is_fake_compaction_summary_message(
                {"type": "message", "content": [{"type": "input_text", "text": f"{label} summary"}]}
            )
        )

    def test_compaction_transcript_message_item_roles_and_fake_summary(self):
        missing_role = {"type": "message", "content": [{"type": "input_text", "text": "hello"}]}
        self.assertIs(
            format_translation._compaction_transcript_message_item(missing_role, force_user_role=False),
            missing_role,
        )

        self.assertEqual(
            format_translation._compaction_transcript_message_item(
                {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "sys"}]},
                force_user_role=True,
            ),
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "[system message]\nsys"}],
            },
        )
        self.assertEqual(
            format_translation._compaction_transcript_message_item(
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev"}]},
                force_user_role=True,
            )["content"][0]["text"],
            "[developer message]\ndev",
        )
        self.assertEqual(
            format_translation._compaction_transcript_message_item(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
                force_user_role=True,
            ),
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "[assistant message]\nanswer"}],
            },
        )

        fake_summary = {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": f"{format_translation.FAKE_COMPACTION_SUMMARY_LABEL}\nold summary",
                }
            ],
        }
        self.assertEqual(
            format_translation._compaction_transcript_message_item(fake_summary, force_user_role=False),
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"{format_translation.FAKE_COMPACTION_SUMMARY_LABEL}\nold summary",
                    }
                ],
            },
        )

    def test_build_fake_compaction_request_native_and_transcript_shapes(self):
        native_body = {
            "model": "gpt-5",
            "input": "compact this",
            "session_id": "session-a",
            "prompt_cache_key": "cache-a",
            "tools": [{"type": "function", "name": "Read"}],
            "parallel_tool_calls": True,
        }

        native = format_translation.build_fake_compaction_request(native_body)

        self.assertEqual(native["model"], "gpt-5")
        self.assertEqual(native["session_id"], "session-a")
        self.assertEqual(native["prompt_cache_key"], "cache-a")
        self.assertEqual(native["tools"], [{"type": "function", "name": "Read"}])
        self.assertTrue(native["parallel_tool_calls"])
        self.assertEqual(native["input"][0]["role"], "user")
        self.assertEqual(native["input"][0]["content"], [{"type": "input_text", "text": "compact this"}])
        self.assertEqual(
            native["input"][-1],
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": format_translation.COMPACTION_SUMMARY_PROMPT}],
            },
        )

        transcript_body = {
            "model": "claude-sonnet-4.6",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "assistant text"}],
                },
                {"type": "function_call", "call_id": "call-1", "name": "Read", "arguments": {"path": "a.py"}},
                {"type": "function_call_output", "call_id": "call-1", "output": "file text"},
                {"type": "reasoning", "encrypted_content": "opaque"},
                {"type": "item_reference", "id": "item-1"},
            ],
            "tools": [{"type": "function", "name": "Read"}],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

        transcript = format_translation.build_fake_compaction_request(transcript_body)

        self.assertEqual(transcript["model"], "claude-sonnet-4.6")
        self.assertEqual(transcript["tools"], [{"type": "function", "name": "Read"}])
        self.assertEqual(transcript["tool_choice"], "none")
        self.assertNotIn("parallel_tool_calls", transcript)
        self.assertEqual(
            [item["content"][0]["text"] for item in transcript["input"][:-1]],
            [
                "assistant text",
                '[Tool call (call-1)] Read\n{"path":"a.py"}',
                "[Tool result (call-1)]\nfile text",
            ],
        )
        self.assertEqual([item["role"] for item in transcript["input"][:-1]], ["assistant", "assistant", "user"])
        self.assertEqual(
            transcript["input"][-1],
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": format_translation.COMPACTION_SUMMARY_PROMPT}],
            },
        )

    def test_chat_completion_to_compaction_response_exact_contract(self):
        payload = {
            "id": "chatcmpl-1",
            "created": 123,
            "model": "claude-sonnet-4.6",
            "choices": [{"message": {"content": [{"text": "one"}, {"text": " two"}]}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }

        translated = format_translation.chat_completion_to_compaction_response(payload, fallback_model="resolved")

        self.assertEqual(translated["id"], "chatcmpl-1")
        self.assertEqual(translated["object"], "response")
        self.assertEqual(translated["created_at"], 123)
        self.assertEqual(translated["status"], "completed")
        self.assertEqual(translated["model"], "resolved")
        self.assertEqual(translated["output_text"], "one two")
        self.assertEqual(len(translated["output"]), 1)
        self.assertEqual(translated["output"][0]["type"], "compaction")
        self.assertEqual(
            format_translation.decode_fake_compaction(translated["output"][0]["encrypted_content"]),
            "one two",
        )
        self.assertEqual(
            translated["usage"],
            {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 2},
            },
        )

        empty = format_translation.chat_completion_to_compaction_response(
            {"id": "chatcmpl-empty", "choices": [{"message": {"content": ""}}]}
        )
        self.assertEqual(empty["output_text"], "(no summary available)")
        self.assertEqual(
            format_translation.decode_fake_compaction(empty["output"][0]["encrypted_content"]),
            "(no summary available)",
        )


if __name__ == "__main__":
    unittest.main()
