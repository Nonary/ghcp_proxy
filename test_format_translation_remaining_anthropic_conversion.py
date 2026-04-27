import unittest
from unittest import mock

import format_translation


class RemainingAnthropicConversionMutationTests(unittest.TestCase):
    def test_blocks_to_chat_content_errors_are_exact_for_missing_type_and_tool_result(self):
        with self.assertRaises(ValueError) as missing_type:
            format_translation._anthropic_blocks_to_chat_content([{"text": "missing type"}])
        self.assertEqual(
            str(missing_type.exception),
            "Unsupported Anthropic content block type: ",
        )

        with self.assertRaises(ValueError) as tool_result:
            format_translation._anthropic_blocks_to_chat_content(
                [{"type": "tool_result", "tool_use_id": "tool_1", "content": "ok"}]
            )
        self.assertEqual(
            str(tool_result.exception),
            "Anthropic block type tool_result cannot be converted into chat message content directly",
        )

    def test_anthropic_tools_to_chat_errors_are_exact(self):
        with self.assertRaises(ValueError) as not_list:
            format_translation.anthropic_tools_to_chat({"name": "Read"})
        self.assertEqual(str(not_list.exception), "Anthropic tools must be a list")

        with self.assertRaises(ValueError) as missing_name:
            format_translation.anthropic_tools_to_chat([{"description": "missing"}])
        self.assertEqual(
            str(missing_name.exception),
            "Anthropic tools must include a string name",
        )

    def test_anthropic_tools_to_chat_preserves_schema_and_cache_control(self):
        self.assertEqual(
            format_translation.anthropic_tools_to_chat(
                [
                    {
                        "name": "Search",
                        "description": "",
                        "input_schema": {"type": "object", "required": ["query"]},
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            [
                {
                    "type": "function",
                    "function": {
                        "name": "Search",
                        "description": " ",
                        "parameters": {"type": "object", "required": ["query"]},
                    },
                    "copilot_cache_control": {"type": "ephemeral"},
                }
            ],
        )

    def test_anthropic_request_to_responses_user_errors_and_thinking_are_exact(self):
        with self.assertRaises(ValueError) as missing_tool_id:
            format_translation.anthropic_request_to_responses(
                {
                    "model": "claude-haiku-4.5",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "content": "ok"}],
                        }
                    ],
                }
            )
        self.assertEqual(
            str(missing_tool_id.exception),
            "Anthropic tool_result blocks must include tool_use_id",
        )

        with self.assertRaises(ValueError) as missing_type:
            format_translation.anthropic_request_to_responses(
                {
                    "model": "claude-haiku-4.5",
                    "messages": [{"role": "user", "content": [{"text": "missing type"}]}],
                }
            )
        self.assertEqual(
            str(missing_type.exception),
            "Unsupported Anthropic content block type: ",
        )

        self.assertEqual(
            format_translation.anthropic_request_to_responses(
                {
                    "model": "claude-haiku-4.5",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "thinking", "thinking": "hidden"},
                                {"type": "text", "text": "visible"},
                            ],
                        }
                    ],
                }
            )["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "visible"}],
                }
            ],
        )

    def test_anthropic_request_to_responses_tool_use_arguments_are_compact_utf8_json(self):
        translated = format_translation.anthropic_request_to_responses(
            {
                "model": "claude-haiku-4.5",
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "Lookup",
                                "input": {"query": "café", "items": [1, 2]},
                            }
                        ],
                    }
                ],
            }
        )

        self.assertEqual(
            translated["input"],
            [
                {
                    "type": "function_call",
                    "call_id": "toolu_1",
                    "name": "Lookup",
                    "arguments": '{"query":"café","items":[1,2]}',
                }
            ],
        )

    def test_anthropic_request_to_responses_strips_cache_control_from_responses_input(self):
        translated = format_translation.anthropic_request_to_responses(
            {
                "model": "claude-haiku-4.5",
                "system": [
                    {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}},
                        ],
                    }
                ],
            }
        )

        self.assertEqual(
            translated["input"],
            [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "sys"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            ],
        )

    def test_extract_chat_message_text_joins_only_dict_text_items(self):
        self.assertEqual(format_translation._extract_chat_message_text("not a message"), "")
        self.assertEqual(
            format_translation._extract_chat_message_text(
                {
                    "content": [
                        "skip",
                        {"type": "input_text", "text": "alpha"},
                        {"type": "output_text", "text": "beta"},
                        {"type": "input_text", "text": 123},
                    ]
                }
            ),
            "alphabeta",
        )

    def test_content_blocks_to_responses_output_non_string_reasoning_fields(self):
        self.assertEqual(
            format_translation._anthropic_content_blocks_to_responses_output(
                [
                    {"type": "thinking", "thinking": 123, "signature": 456},
                    {"type": "redacted_thinking", "data": {"opaque": True}},
                    {"type": "text", "text": "after"},
                ]
            ),
            [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": ""}],
                },
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "after", "annotations": []}],
                },
            ],
        )

    def test_content_blocks_to_responses_output_function_call_arguments_are_exact(self):
        self.assertEqual(
            format_translation._anthropic_content_blocks_to_responses_output(
                [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "Search",
                        "input": {"query": "cafe", "city": "Montréal"},
                    }
                ]
            ),
            [
                {
                    "type": "function_call",
                    "call_id": "tool_1",
                    "name": "Search",
                    "arguments": '{"query":"cafe","city":"Montréal"}',
                }
            ],
        )

    def test_anthropic_response_to_responses_skips_non_message_output_items(self):
        output_items = [
            "ignored top-level string",
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hidden"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": " visible "}],
            },
        ]
        with mock.patch.object(
            format_translation,
            "_anthropic_content_blocks_to_responses_output",
            return_value=output_items,
        ):
            self.assertEqual(
                format_translation.anthropic_response_to_responses(
                    {
                        "id": "msg_output_items",
                        "model": "model-a",
                        "created": 2468,
                        "content": [],
                        "stop_reason": 123,
                    }
                ),
                {
                    "id": "msg_output_items",
                    "object": "response",
                    "created_at": 2468,
                    "model": "model-a",
                    "status": "completed",
                    "output": output_items,
                    "output_text": "visible",
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                },
            )

    def test_anthropic_response_to_responses_non_dict_payload_uses_defaults(self):
        translated = format_translation.anthropic_response_to_responses(
            None,
            fallback_model="fallback-model",
        )

        self.assertEqual(translated["id"], "resp_anthropic")
        self.assertEqual(translated["object"], "response")
        self.assertIsInstance(translated["created_at"], int)
        self.assertEqual(translated["model"], "fallback-model")
        self.assertEqual(translated["status"], "completed")
        self.assertEqual(translated["output"], [])
        self.assertEqual(translated["output_text"], "")
        self.assertEqual(
            translated["usage"],
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )


if __name__ == "__main__":
    unittest.main()
