import json
import unittest
from unittest import mock

import format_translation


class AnthropicResponseMutationCoverageTests(unittest.TestCase):
    def test_chat_usage_to_anthropic_exact_defaults_and_cache_math(self):
        zero_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        self.assertEqual(format_translation.chat_usage_to_anthropic(None), zero_usage)
        self.assertEqual(format_translation.chat_usage_to_anthropic(["not", "a", "dict"]), zero_usage)
        self.assertEqual(set(format_translation.chat_usage_to_anthropic(None)), set(zero_usage))

        self.assertEqual(
            format_translation.chat_usage_to_anthropic(
                {
                    "prompt_tokens": 9,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 2},
                }
            ),
            {
                "input_tokens": 7,
                "output_tokens": 4,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 2,
            },
        )
        self.assertEqual(
            format_translation.chat_usage_to_anthropic(
                {
                    "prompt_tokens": 3,
                    "completion_tokens": 8,
                    "prompt_tokens_details": {"cached_tokens": 5},
                }
            ),
            {
                "input_tokens": 0,
                "output_tokens": 8,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 5,
            },
        )
        self.assertEqual(
            format_translation.chat_usage_to_anthropic({"completion_tokens": 6}),
            {
                "input_tokens": 0,
                "output_tokens": 6,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )

    def test_chat_message_to_anthropic_content_exact_text_and_tool_contract(self):
        self.assertEqual(
            format_translation._chat_message_to_anthropic_content(
                {
                    "content": [
                        "ignored scalar",
                        {"type": "text", "text": "alpha"},
                        {"type": "output_text", "output_text": "ignored non-chat text field"},
                        {"type": "text", "text": " beta"},
                    ],
                    "tool_calls": [
                        "ignored tool",
                        {
                            "id": "call_direct",
                            "type": "function",
                            "function": {
                                "name": "Direct",
                                "arguments": '{"n":1}',
                            },
                        },
                    ],
                }
            ),
            [
                {"type": "text", "text": "alpha beta"},
                {
                    "type": "tool_use",
                    "id": "call_direct",
                    "name": "Direct",
                    "input": {"n": 1},
                },
            ],
        )

    def test_response_usage_to_anthropic_exact_defaults_and_normalized_details(self):
        zero_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        self.assertEqual(format_translation.response_usage_to_anthropic(None), zero_usage)
        with mock.patch.object(format_translation.util, "normalize_usage_payload", return_value=None):
            self.assertEqual(format_translation.response_usage_to_anthropic({"input_tokens": 99}), zero_usage)
        with mock.patch.object(
            format_translation.util,
            "normalize_usage_payload",
            return_value={"fresh_input_tokens": 5, "output_tokens": 2},
        ):
            self.assertEqual(
                format_translation.response_usage_to_anthropic({"input_tokens": 99}),
                {
                    "input_tokens": 5,
                    "output_tokens": 2,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            )
        with mock.patch.object(
            format_translation.util,
            "normalize_usage_payload",
            return_value={"fresh_input_tokens": 5},
        ):
            self.assertEqual(
                format_translation.response_usage_to_anthropic({"input_tokens": 99}),
                {
                    "input_tokens": 5,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            )

        self.assertEqual(
            format_translation.response_usage_to_anthropic({"input_tokens": 5, "output_tokens": 2}),
            {
                "input_tokens": 5,
                "output_tokens": 2,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )

        self.assertEqual(
            format_translation.response_usage_to_anthropic(
                {
                    "input_tokens": 20,
                    "output_tokens": 6,
                    "input_tokens_details": {"cached_tokens": 7},
                    "cache_creation_input_tokens": 3,
                }
            ),
            {
                "input_tokens": 13,
                "output_tokens": 6,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 7,
            },
        )
        self.assertEqual(
            format_translation.response_usage_to_anthropic(
                {
                    "input_tokens": 20,
                    "fresh_input_tokens": 4,
                    "output_tokens": 6,
                    "cached_input_tokens": 9,
                    "cache_creation_input_tokens": 3,
                }
            ),
            {
                "input_tokens": 4,
                "output_tokens": 6,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 9,
            },
        )

    def test_anthropic_usage_to_responses_usage_exact_contract(self):
        self.assertEqual(
            format_translation._anthropic_usage_to_responses_usage(None),
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        self.assertEqual(
            format_translation._anthropic_usage_to_responses_usage(
                {
                    "input_tokens": 11,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 7,
                    "cache_creation_input_tokens": 2,
                }
            ),
            {
                "input_tokens": 18,
                "output_tokens": 5,
                "total_tokens": 23,
                "input_tokens_details": {
                    "cached_tokens": 7,
                    "cache_creation_input_tokens": 2,
                },
            },
        )
        self.assertEqual(
            format_translation._anthropic_usage_to_responses_usage(
                {
                    "input_tokens": 11,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            ),
            {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
        )

    def test_chat_completion_to_anthropic_exact_full_payload_with_tool_call(self):
        payload = {
            "id": "chatcmpl_exact",
            "model": "upstream-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "first "},
                            {"type": "text", "text": "second"},
                        ],
                        "tool_calls": [
                            {
                                "id": "call_read",
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"path":"app.py"}',
                                },
                            },
                            {
                                "id": "call_list",
                                "type": "function",
                                "function": {
                                    "name": "List",
                                    "arguments": "[1,2]",
                                },
                            },
                            "ignored",
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
                {
                    "message": {"role": "assistant", "content": "ignored choice"},
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        }

        self.assertEqual(
            format_translation.chat_completion_to_anthropic(payload, fallback_model="resolved-model"),
            {
                "id": "chatcmpl_exact",
                "type": "message",
                "role": "assistant",
                "model": "resolved-model",
                "content": [
                    {"type": "text", "text": "first second"},
                    {
                        "type": "tool_use",
                        "id": "call_read",
                        "name": "Read",
                        "input": {"path": "app.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "call_list",
                        "name": "List",
                        "input": {"value": [1, 2]},
                    },
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 4,
                },
            },
        )

    def test_chat_completion_to_anthropic_exact_fallbacks_and_stop_reasons(self):
        base = {
            "id": "chatcmpl_stop",
            "model": "payload-model",
            "choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}],
        }
        self.assertEqual(
            format_translation.chat_completion_to_anthropic(base),
            {
                "id": "chatcmpl_stop",
                "type": "message",
                "role": "assistant",
                "model": "payload-model",
                "content": [{"type": "text", "text": ""}],
                "stop_reason": "stop_sequence",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        )
        self.assertEqual(
            format_translation.chat_completion_to_anthropic(
                {"id": "chatcmpl_length", "model": "payload-model", "choices": [{"finish_reason": "length"}]},
                fallback_model="fallback-model",
            )["stop_reason"],
            "max_tokens",
        )

    def test_response_output_items_to_anthropic_content_exact_messages_and_tools(self):
        self.assertEqual(
            format_translation._response_output_items_to_anthropic_content(None),
            [{"type": "text", "text": ""}],
        )
        self.assertEqual(
            format_translation._response_output_items_to_anthropic_content([]),
            [{"type": "text", "text": ""}],
        )
        self.assertEqual(
            format_translation._response_output_items_to_anthropic_content(
                [
                    "ignored",
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "output_text", "text": "ignore user"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "answer"},
                            "ignored part before later text",
                            {"type": "input_text", "input_text": "fallback input"},
                            {"type": "text", "text": "plain"},
                        ],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_fallback",
                        "name": 7,
                        "arguments": "{bad json",
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_search",
                        "name": "Search",
                        "arguments": '{"query":"mutants"}',
                    },
                ]
            ),
            [
                {"type": "text", "text": "answer"},
                {"type": "text", "text": "fallback input"},
                {"type": "text", "text": "plain"},
                {
                    "type": "tool_use",
                    "id": "fc_fallback",
                    "name": "",
                    "input": {"_raw": "{bad json"},
                },
                {
                    "type": "tool_use",
                    "id": "call_search",
                    "name": "Search",
                    "input": {"query": "mutants"},
                },
            ],
        )

    def test_response_payload_to_anthropic_exact_full_payloads(self):
        payload = {
            "id": "resp_exact",
            "model": "responses-model",
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
                "input_tokens": 15,
                "output_tokens": 3,
                "input_tokens_details": {"cached_tokens": 5},
            },
        }
        self.assertEqual(
            format_translation.response_payload_to_anthropic(payload, fallback_model="fallback-model"),
            {
                "id": "resp_exact",
                "type": "message",
                "role": "assistant",
                "model": "fallback-model",
                "content": [
                    {"type": "text", "text": "hello"},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Read",
                        "input": {"file": "main.py"},
                    },
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 5,
                },
            },
        )
        self.assertEqual(
            format_translation.response_payload_to_anthropic(
                {
                    "id": "resp_text",
                    "model": "responses-model",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "only text"}],
                        }
                    ],
                }
            ),
            {
                "id": "resp_text",
                "type": "message",
                "role": "assistant",
                "model": "responses-model",
                "content": [{"type": "text", "text": "only text"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        )

    def test_anthropic_response_to_responses_exact_content_status_and_usage(self):
        payload = {
            "id": "msg_exact",
            "model": "claude-sonnet-4.6",
            "created": 123456,
            "content": [
                {"type": "text", "text": " alpha "},
                {"type": "text", "text": "beta"},
                {"type": "thinking", "thinking": "chain", "signature": "ENC@rs_1"},
                {"type": "redacted_thinking", "data": "REDACTED"},
                {"type": "tool_use", "id": "toolu_1", "name": "Search", "input": {"q": "x"}},
                {"type": "unknown", "ignored": True},
                {"not": "a block"},
                {"type": "text", "text": " gamma "},
            ],
            "stop_reason": "max_tokens",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 5,
                "cache_read_input_tokens": 7,
                "cache_creation_input_tokens": 2,
            },
        }
        self.assertEqual(
            format_translation.anthropic_response_to_responses(payload, fallback_model="fallback-model"),
            {
                "id": "msg_exact",
                "object": "response",
                "created_at": 123456,
                "model": "claude-sonnet-4.6",
                "status": "incomplete",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": " alpha ", "annotations": []},
                            {"type": "output_text", "text": "beta", "annotations": []},
                        ],
                    },
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "chain"}],
                        "id": "rs_1",
                        "encrypted_content": "ENC",
                    },
                    {
                        "type": "reasoning",
                        "summary": [],
                        "encrypted_content": "REDACTED",
                    },
                    {
                        "type": "function_call",
                        "call_id": "toolu_1",
                        "name": "Search",
                        "arguments": '{"q":"x"}',
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": " gamma ", "annotations": []}],
                    },
                ],
                "output_text": "alpha\n\nbeta\n\ngamma",
                "usage": {
                    "input_tokens": 18,
                    "output_tokens": 5,
                    "total_tokens": 23,
                    "input_tokens_details": {
                        "cached_tokens": 7,
                        "cache_creation_input_tokens": 2,
                    },
                },
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        )

    def test_anthropic_response_to_responses_exact_defaults_and_refusal(self):
        with mock.patch.object(format_translation.time, "time", return_value=2468.9):
            self.assertEqual(
                format_translation.anthropic_response_to_responses(
                    {
                        "id": "",
                        "content": [{"type": "text", "text": "blocked"}],
                        "stop_reason": "refusal",
                    },
                    fallback_model="fallback-model",
                ),
                {
                    "id": "resp_anthropic",
                    "object": "response",
                    "created_at": 2468,
                    "model": "fallback-model",
                    "status": "incomplete",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "blocked", "annotations": []}],
                        }
                    ],
                    "output_text": "blocked",
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "incomplete_details": {"reason": "content_filter"},
                },
            )

        self.assertEqual(
            format_translation.anthropic_response_to_responses(
                {"id": "msg_done", "model": "model-a", "content": [], "stop_reason": "pause_turn"}
            ),
            {
                "id": "msg_done",
                "object": "response",
                "created_at": mock.ANY,
                "model": "model-a",
                "status": "completed",
                "output": [],
                "output_text": "",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        )

        self.assertEqual(
            format_translation.anthropic_response_to_responses(
                {
                    "id": "msg_malformed_parts",
                    "model": "model-a",
                    "created": 1357,
                    "content": [
                        "ignored string part",
                        {"type": "text", "text": 123},
                        {"type": "text", "text": " kept "},
                    ],
                }
            ),
            {
                "id": "msg_malformed_parts",
                "object": "response",
                "created_at": 1357,
                "model": "model-a",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": " kept ", "annotations": []}],
                    }
                ],
                "output_text": "kept",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        )

        malformed_response_output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    "ignored string part",
                    {"type": "output_text", "text": 123},
                    {"type": "output_text", "text": " kept "},
                ],
            }
        ]
        with mock.patch.object(
            format_translation,
            "_anthropic_content_blocks_to_responses_output",
            return_value=malformed_response_output,
        ):
            self.assertEqual(
                format_translation.anthropic_response_to_responses(
                    {"id": "msg_patched_output", "model": "model-a", "created": 9753, "content": []}
                ),
                {
                    "id": "msg_patched_output",
                    "object": "response",
                    "created_at": 9753,
                    "model": "model-a",
                    "status": "completed",
                    "output": malformed_response_output,
                    "output_text": "kept",
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                },
            )


if __name__ == "__main__":
    unittest.main()
