import unittest
from unittest import mock

import format_translation


class ResponsesChatMutationContractTests(unittest.TestCase):
    def test_responses_string_input_is_exact_user_message(self):
        translated = format_translation.responses_request_to_chat(
            {"model": "openai/gpt-5.4", "input": "hello"}
        )

        self.assertEqual(
            translated,
            {
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    def test_responses_pending_tool_call_keeps_assistant_messages_before_tool_output(self):
        translated = format_translation.responses_request_to_chat(
            {
                "model": "openai/gpt-5.4",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "Read",
                        "arguments": {"path": "a.py"},
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "still thinking"}],
                    },
                    {"type": "message", "role": "user", "content": "defer me"},
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": [{"type": "output_text", "text": "contents"}],
                    },
                ],
            }
        )

        self.assertEqual(
            translated["messages"],
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": '{"path":"a.py"}',
                            },
                        }
                    ],
                },
                {"role": "assistant", "content": "still thinking"},
                {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
                {"role": "user", "content": "defer me"},
            ],
        )

    def test_responses_function_call_uses_id_fallback_exactly(self):
        translated = format_translation.responses_request_to_chat(
            {
                "model": "openai/gpt-5.4",
                "input": [
                    {
                        "type": "function_call",
                        "id": "fc_item_1",
                        "name": "Lookup",
                        "arguments": {"q": "mutants"},
                    }
                ],
            }
        )

        self.assertEqual(translated["messages"][0]["tool_calls"][0]["id"], "fc_item_1")
        self.assertEqual(
            translated["messages"][0]["tool_calls"][0]["function"],
            {"name": "Lookup", "arguments": '{"q":"mutants"}'},
        )

    def test_responses_custom_tool_output_does_not_stop_processing_following_items(self):
        translated = format_translation.responses_request_to_chat(
            {
                "model": "openai/gpt-5.4",
                "input": [
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "custom_1",
                        "output": {"ok": True},
                    },
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "after custom"}],
                    },
                ],
            }
        )

        self.assertEqual(
            translated["messages"],
            [
                {"role": "user", "content": '[Custom tool result (custom_1)]\n{"ok":true}'},
                {"role": "user", "content": "after custom"},
            ],
        )

    def test_responses_request_rejects_exact_invalid_boundaries(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Responses message role: bad"):
            format_translation.responses_request_to_chat(
                {
                    "model": "openai/gpt-5.4",
                    "input": [{"type": "message", "role": "bad", "content": "x"}],
                }
            )

        with self.assertRaises(ValueError) as tools_error:
            format_translation.responses_request_to_chat(
                {"model": "openai/gpt-5.4", "input": "x", "tools": {"type": "function"}}
            )
        self.assertEqual(str(tools_error.exception), "Responses tools must be a list")

        with self.assertRaisesRegex(ValueError, "Unsupported Responses input item type: file_search_call"):
            format_translation.responses_request_to_chat(
                {
                    "model": "openai/gpt-5.4",
                    "input": [{"type": "file_search_call"}],
                }
            )

    def test_response_content_item_to_chat_exact_boundaries(self):
        self.assertEqual(
            format_translation._response_content_item_to_chat(
                {"type": "TEXT", "text": "visible", "input_text": "fallback"}
            ),
            {"type": "text", "text": "visible"},
        )
        self.assertEqual(
            format_translation._response_content_item_to_chat(
                {
                    "type": "input_file",
                    "file_data": "data:;base64,PDF",
                }
            ),
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "PDF",
                },
                "title": "document.pdf",
            },
        )
        self.assertEqual(
            format_translation._response_content_item_to_chat(
                {
                    "type": "file",
                    "filename": "report.pdf",
                    "file_data": "data:application/pdf;base64,ABC",
                }
            ),
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "ABC",
                },
                "title": "report.pdf",
            },
        )
        self.assertEqual(
            format_translation._response_content_item_to_chat(
                {"type": "input_image", "image_url": "https://example.invalid/a.png"}
            ),
            {"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}},
        )
        self.assertEqual(
            format_translation._response_content_item_to_chat(
                {"type": "input_image", "image_url": {"url": "https://example.invalid/b.png"}}
            ),
            {"type": "image_url", "image_url": {"url": "https://example.invalid/b.png"}},
        )
        self.assertIsNone(format_translation._response_content_item_to_chat({"type": "file", "file_id": "file_1"}))
        self.assertIsNone(format_translation._response_content_item_to_chat({"type": "unknown", "text": "x"}))
        with self.assertRaisesRegex(ValueError, "must include image_url or image_base64/media_type"):
            format_translation._response_content_item_to_chat({"type": "input_image", "image_url": ""})

    def test_responses_tool_to_chat_exact_boundaries(self):
        self.assertIsNone(format_translation._responses_tool_to_chat(None))
        self.assertIsNone(format_translation._responses_tool_to_chat({"type": "web_search_preview"}))
        self.assertIsNone(
            format_translation._responses_tool_to_chat(
                {"type": "function", "name": "mcp__ide__executeCode"}
            )
        )
        self.assertEqual(
            format_translation._responses_tool_to_chat(
                {
                    "type": "FUNCTION",
                    "name": "FlatCached",
                    "description": 123,
                    "parameters": None,
                    "cache_control": {"type": "ephemeral"},
                }
            ),
            {
                "type": "function",
                "function": {
                    "name": "FlatCached",
                    "description": " ",
                    "parameters": {"type": "object", "properties": {}},
                },
                "copilot_cache_control": {"type": "ephemeral"},
            },
        )
        with self.assertRaisesRegex(ValueError, "Responses function tools must include a name"):
            format_translation._responses_tool_to_chat(
                {"type": "function", "function": {"name": 123}}
            )

    def test_tool_choice_helpers_case_and_dangerous_boundaries(self):
        self.assertEqual(format_translation.responses_tool_choice_to_chat("AUTO"), "auto")
        self.assertEqual(format_translation.responses_tool_choice_to_chat({"type": "NONE"}), "none")
        self.assertEqual(
            format_translation.responses_tool_choice_to_chat({"name": "Read"}),
            {"type": "function", "function": {"name": "Read"}},
        )
        self.assertIsNone(
            format_translation.responses_tool_choice_to_chat(
                {"function": {"name": "mcp__ide__executeCode"}}
            )
        )
        with self.assertRaisesRegex(ValueError, "Unsupported Responses tool_choice value"):
            format_translation.responses_tool_choice_to_chat({"function": {"name": 123}})

        self.assertEqual(format_translation.chat_tool_choice_to_responses("REQUIRED"), "required")
        self.assertEqual(format_translation.chat_tool_choice_to_responses({"type": "AUTO"}), "auto")
        self.assertEqual(format_translation.chat_tool_choice_to_responses({"type": "none"}), "none")
        self.assertEqual(
            format_translation.chat_tool_choice_to_responses({"function": {"name": "Read"}}),
            {"type": "function", "name": "Read"},
        )
        self.assertIsNone(
            format_translation.chat_tool_choice_to_responses(
                {"name": "mcp__ide__executeCode"}
            )
        )
        with self.assertRaises(ValueError) as chat_choice_error:
            format_translation.chat_tool_choice_to_responses({"function": {"name": 123}})
        self.assertEqual(str(chat_choice_error.exception), "Unsupported chat tool_choice value")

    def test_chat_usage_to_response_exact_reasoning_and_zero_boundaries(self):
        self.assertEqual(
            format_translation.chat_usage_to_response(
                {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 7},
                    "reasoning_output_tokens": 5,
                }
            ),
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        )
        self.assertEqual(
            format_translation.chat_usage_to_response(
                {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "prompt_tokens_details": "bad",
                    "completion_tokens_details": "bad",
                }
            ),
            {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        )

    def test_chat_usage_to_response_defaults_missing_completion_and_direct_reasoning(self):
        self.assertEqual(
            format_translation.chat_usage_to_response(
                {
                    "prompt_tokens": 8,
                    "prompt_tokens_details": {"cached_tokens": 4},
                    "reasoning_output_tokens": 6,
                }
            ),
            {
                "input_tokens": 8,
                "output_tokens": 0,
                "total_tokens": 8,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens_details": {"reasoning_tokens": 6},
            },
        )

    def test_chat_completion_to_response_uses_first_choice_and_exact_output_contract(self):
        translated = format_translation.chat_completion_to_response(
            {
                "id": "chat_1",
                "created": 1000,
                "model": "chat-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": " a "}, {"type": "text", "text": "b"}],
                            "tool_calls": [
                                "skip",
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Lookup",
                                        "arguments": {"q": "mutants"},
                                    },
                                },
                            ],
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": "must be ignored",
                        }
                    },
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6},
            },
            fallback_model="fallback-model",
        )

        self.assertEqual(
            translated,
            {
                "id": "chat_1",
                "object": "response",
                "created_at": 1000,
                "status": "completed",
                "model": "fallback-model",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": " a b"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "Lookup",
                        "arguments": '{"q":"mutants"}',
                    },
                ],
                "output_text": "a b",
                "usage": {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
            },
        )

    def test_chat_completion_to_response_tool_only_contract(self):
        translated = format_translation.chat_completion_to_response(
            {
                "id": "chat_tool_only",
                "created": 1001,
                "model": "chat-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"arguments": ""},
                                }
                            ],
                        }
                    }
                ],
            }
        )

        self.assertEqual(
            translated["output"],
            [
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "",
                    "arguments": "",
                }
            ],
        )
        self.assertEqual(translated["output_text"], "")

    def test_chat_completion_to_response_preserves_tool_function_and_model_key(self):
        translated = format_translation.chat_completion_to_response(
            {
                "id": "chat_tool_contract",
                "created": 1002,
                "model": "source-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_contract",
                                    "type": "function",
                                    "function": {
                                        "name": "Lookup",
                                        "arguments": '{"needle":"value"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        )

        self.assertIn("model", translated)
        self.assertNotIn("XXmodelXX", translated)
        self.assertEqual(translated["model"], "source-model")
        self.assertEqual(
            translated["output"],
            [
                {
                    "type": "function_call",
                    "call_id": "call_contract",
                    "name": "Lookup",
                    "arguments": '{"needle":"value"}',
                }
            ],
        )
        self.assertEqual(translated["output_text"], "")

    def test_chat_completion_to_response_non_string_arguments_and_output_text_join(self):
        translated = format_translation.chat_completion_to_response(
            {
                "id": "chat_multitext",
                "created": 1003,
                "model": "chat-model",
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": " first "},
                                {"type": "text", "text": "second"},
                            ],
                            "tool_calls": [
                                {
                                    "id": "call_non_ascii",
                                    "type": "function",
                                    "function": {
                                        "name": "Lookup",
                                        "arguments": {"city": "Montréal", "n": 2},
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        )

        self.assertEqual(
            translated["output"],
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": " first second"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_non_ascii",
                    "name": "Lookup",
                    "arguments": '{"city":"Montréal","n":2}',
                },
            ],
        )
        self.assertEqual(translated["output_text"], "first second")

    def test_chat_completion_to_response_output_text_skips_tool_calls_and_joins_messages(self):
        original_extract = format_translation.util.extract_item_text
        with mock.patch.object(
            format_translation,
            "_extract_chat_message_text",
            return_value="first",
        ), mock.patch.object(
            format_translation,
            "util",
            wraps=format_translation.util,
        ) as wrapped_util:
            def extract_item_text(item):
                if isinstance(item, dict) and item.get("type") == "function_call":
                    raise AssertionError("function calls must not contribute output_text")
                text = original_extract(item)
                if isinstance(item, dict) and item.get("type") == "message":
                    if text == "first":
                        return " first "
                    return " second "
                return text

            wrapped_util.extract_item_text.side_effect = extract_item_text
            translated = format_translation.chat_completion_to_response(
                {
                    "id": "chat_output_text",
                    "created": 1004,
                    "model": "chat-model",
                    "choices": [
                        {
                            "message": {
                                "content": "ignored by mock",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "Lookup", "arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ],
                }
            )

        self.assertEqual(translated["output_text"], "first")
        self.assertEqual(translated["output"][1]["type"], "function_call")


if __name__ == "__main__":
    unittest.main()
