import asyncio
import unittest

import format_translation


def run_chat(body):
    return asyncio.run(
        format_translation.anthropic_request_to_chat(
            body,
            "https://example.invalid",
            "test-key",
        )
    )


class AnthropicRequestMutationTests(unittest.TestCase):
    def test_system_content_helper_exact_shapes_and_errors(self):
        self.assertEqual(format_translation._anthropic_system_to_chat_content("plain system"), "plain system")
        self.assertEqual(
            format_translation._anthropic_system_to_chat_content(
                [
                    {"type": "text", "text": "alpha"},
                    {"type": "text"},
                    {"type": "text", "text": "omega"},
                ]
            ),
            "alphaomega",
        )
        self.assertEqual(
            format_translation._anthropic_system_to_chat_content(
                [
                    {"type": "text", "text": "alpha"},
                    {"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}},
                ]
            ),
            [
                {"type": "text", "text": "alpha"},
                {
                    "type": "text",
                    "text": "cached",
                    "copilot_cache_control": {"type": "ephemeral"},
                },
            ],
        )
        with self.assertRaises(ValueError) as ctx:
            format_translation._anthropic_system_to_chat_content([{"type": "image"}])
        self.assertEqual(
            str(ctx.exception),
            "Anthropic system content currently supports text blocks only",
        )

    def test_image_block_helper_exact_outputs_and_error_messages(self):
        self.assertEqual(
            format_translation._anthropic_image_block_to_chat(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                    "cache_control": {"type": "ephemeral"},
                }
            ),
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
                "copilot_cache_control": {"type": "ephemeral"},
            },
        )
        self.assertEqual(
            format_translation._anthropic_image_block_to_chat(
                {"type": "image", "source": {"type": "url", "url": "https://example.invalid/i.png"}}
            ),
            {"type": "image_url", "image_url": {"url": "https://example.invalid/i.png"}},
        )
        for block, message in (
            (
                {"type": "image"},
                "Anthropic image block is missing a valid source object",
            ),
            (
                {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
                "Anthropic base64 image source must include media_type and data strings",
            ),
            (
                {"type": "image", "source": {"type": "url", "url": 123}},
                "Anthropic URL image source must include a url string",
            ),
            (
                {"type": "image", "source": {}},
                "Unsupported Anthropic image source type: ",
            ),
        ):
            with self.subTest(block=block):
                with self.assertRaises(ValueError) as ctx:
                    format_translation._anthropic_image_block_to_chat(block)
                self.assertEqual(str(ctx.exception), message)

    def test_message_to_chat_assistant_tool_use_then_text_exact_order(self):
        self.assertEqual(
            format_translation.anthropic_message_to_chat_messages(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"path": "é.py"}},
                        {"type": "text", "text": "visible"},
                    ],
                }
            ),
            [
                {
                    "role": "assistant",
                    "content": "visible",
                    "tool_calls": [
                        {
                            "id": "tool_1",
                            "type": "function",
                            "function": {"name": "Read", "arguments": '{"path":"é.py"}'},
                        }
                    ],
                }
            ],
        )

    def test_message_to_chat_user_tool_result_edge_cases_exact(self):
        self.assertEqual(
            format_translation.anthropic_message_to_chat_messages(
                {
                    "role": "user",
                    "content": [
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "tool_result", "tool_use_id": "tool_1", "is_error": True},
                        {"type": "text", "text": "follow up"},
                    ],
                }
            ),
            [
                {"role": "tool", "tool_call_id": "tool_1", "content": "[tool_error]\n"},
                {"role": "user", "content": "follow up"},
            ],
        )
        self.assertEqual(
            format_translation.anthropic_message_to_chat_messages(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tool_2", "content": "ok", "is_error": False},
                    ],
                }
            ),
            [{"role": "tool", "tool_call_id": "tool_2", "content": "ok"}],
        )

    def test_message_to_chat_missing_role_and_type_errors_are_exact(self):
        with self.assertRaises(ValueError) as ctx:
            format_translation.anthropic_message_to_chat_messages({"content": "hello"})
        self.assertEqual(str(ctx.exception), "Unsupported Anthropic role: ")

        for role in ("assistant", "user"):
            with self.subTest(role=role):
                with self.assertRaises(ValueError) as ctx:
                    format_translation.anthropic_message_to_chat_messages(
                        {"role": role, "content": [{"text": "missing type"}]}
                    )
                self.assertEqual(str(ctx.exception), "Unsupported Anthropic content block type: ")

    def test_content_blocks_to_responses_output_exact_mixed_sequence(self):
        self.assertEqual(format_translation._anthropic_content_blocks_to_responses_output("text"), [])
        self.assertEqual(
            format_translation._anthropic_content_blocks_to_responses_output(
                [
                    {"type": "text", "text": "alpha"},
                    {"type": "thinking", "thinking": "chain", "signature": "cipher@rs_1"},
                    {"type": "text", "text": "beta"},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"path": "é.py"}},
                    {"type": "tool_use", "id": "tool_bad"},
                    {"type": "text", "text": "gamma"},
                ]
            ),
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "alpha", "annotations": []}],
                },
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "chain"}],
                    "id": "rs_1",
                    "encrypted_content": "cipher",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "beta", "annotations": []}],
                },
                {"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
                {"type": "function_call", "call_id": "tool_1", "name": "Read", "arguments": '{"path":"é.py"}'},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "gamma", "annotations": []}],
                },
            ],
        )

    def test_content_blocks_to_responses_output_missing_type_and_empty_signature(self):
        self.assertEqual(
            format_translation._anthropic_content_blocks_to_responses_output(
                [
                    {"type": "thinking", "thinking": "no signature"},
                    {"type": "redacted_thinking", "signature": "redacted-sig"},
                    {"text": "ignored"},
                    {"type": "text", "text": "after"},
                ]
            ),
            [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "no signature"}],
                },
                {"type": "reasoning", "summary": [], "encrypted_content": "redacted-sig"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "after", "annotations": []}],
                },
            ],
        )

    def test_anthropic_request_to_chat_exact_payload_defaults_and_options(self):
        body = {
            "model": "claude-haiku-4.5",
            "system": "system",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 12,
            "temperature": 0.2,
            "top_p": 0.8,
            "stop_sequences": ["stop"],
        }

        self.assertEqual(
            run_chat(body),
            {
                "model": "claude-haiku-4.5",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello"},
                ],
                "stream": False,
                "max_tokens": 12,
                "temperature": 0.2,
                "top_p": 0.8,
                "stop": ["stop"],
            },
        )

    def test_anthropic_request_to_chat_tool_choice_gating(self):
        base = {"model": "claude-haiku-4.5", "messages": [{"role": "user", "content": "hello"}]}
        self.assertEqual(run_chat({**base, "tool_choice": {"type": "auto"}})["tool_choice"], "auto")
        self.assertEqual(run_chat({**base, "tool_choice": {"type": "none"}})["tool_choice"], "none")
        self.assertNotIn("tool_choice", run_chat({**base, "tool_choice": {"type": "any"}}))
        self.assertNotIn("tool_choice", run_chat({**base, "tool_choice": {"type": "tool", "name": "Read"}}))
        self.assertEqual(
            run_chat(
                {
                    **base,
                    "tools": [{"name": "Read", "description": "Read", "input_schema": {"type": "object"}}],
                    "tool_choice": {"type": "any"},
                }
            )["tool_choice"],
            "required",
        )

    def test_anthropic_request_to_responses_exact_system_user_assistant_tool_payload(self):
        body = {
            "model": "claude-haiku-4.5",
            "system": [
                "skip non-dict system item",
                {"type": "text", "text": "sys"},
                {"type": "text"},
                {"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}},
                        {"type": "image", "source": {"type": "url", "url": "https://example.invalid/i.png"}},
                        {"type": "tool_result", "tool_use_id": "tool_0"},
                        {"type": "text", "text": "after tool"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"path": "é.py"}},
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "text", "text": "visible"},
                    ],
                },
            ],
            "stream": False,
            "max_tokens": 9,
            "top_p": 0.7,
            "metadata": {"trace": "t1"},
        }

        self.assertEqual(
            format_translation.anthropic_request_to_responses(body),
            {
                "model": "claude-haiku-4.5",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [
                            {"type": "input_text", "text": "sys"},
                            {"type": "input_text", "text": "cached"},
                        ],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "hello"},
                            {"type": "input_image", "image_url": "https://example.invalid/i.png"},
                        ],
                    },
                    {"type": "function_call_output", "call_id": "tool_0", "output": ""},
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "after tool"}],
                    },
                    {"type": "function_call", "call_id": "tool_1", "name": "Read", "arguments": '{"path":"é.py"}'},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "visible", "annotations": []}],
                    },
                ],
                "stream": False,
                "store": False,
                "parallel_tool_calls": True,
                "include": ["reasoning.encrypted_content"],
                "text": {"format": {"type": "text"}, "verbosity": "low"},
                "max_output_tokens": 9,
                "top_p": 0.7,
            },
        )

    def test_anthropic_request_to_responses_string_system_exact_developer_message(self):
        self.assertEqual(
            format_translation.anthropic_request_to_responses(
                {
                    "model": "claude-haiku-4.5",
                    "system": "plain system",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            ),
            {
                "model": "claude-haiku-4.5",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "plain system"}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    },
                ],
                "stream": False,
                "store": False,
                "parallel_tool_calls": True,
                "include": ["reasoning.encrypted_content"],
                "text": {"format": {"type": "text"}, "verbosity": "low"},
            },
        )

    def test_anthropic_request_to_responses_errors_and_temperature_continue(self):
        with self.assertRaises(ValueError) as ctx:
            format_translation.anthropic_request_to_responses({"model": "gpt-5.4", "messages": None})
        self.assertEqual(str(ctx.exception), "Anthropic request must include a messages array")

        with self.assertRaises(ValueError) as ctx:
            format_translation.anthropic_request_to_responses({"model": "gpt-5.4", "messages": [{"content": "hello"}]})
        self.assertEqual(str(ctx.exception), "Unsupported Anthropic role: ")

        with self.assertRaises(ValueError) as ctx:
            format_translation.anthropic_request_to_responses(
                {"model": "gpt-5.4", "messages": [{"role": "assistant", "content": [{"text": "missing"}]}]}
            )
        self.assertEqual(str(ctx.exception), "Unsupported Anthropic content block type: ")

        with self.assertRaises(ValueError) as ctx:
            format_translation.anthropic_request_to_responses(
                {
                    "model": "gpt-5.4",
                    "messages": [
                        {"role": "assistant", "content": [{"type": "tool_use", "id": "tool_1"}]},
                    ],
                }
            )
        self.assertEqual(str(ctx.exception), "Anthropic tool_use blocks must include string id and name")

        translated = format_translation.anthropic_request_to_responses(
            {
                "model": "gpt-5.4-mini",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.4,
                "top_p": 0.6,
                "metadata": {"kept": True},
            }
        )
        self.assertNotIn("temperature", translated)
        self.assertEqual(translated["top_p"], 0.6)
        self.assertNotIn("metadata", translated)

    def test_anthropic_request_to_responses_tool_choice_gating(self):
        base = {"model": "claude-haiku-4.5", "messages": [{"role": "user", "content": "hello"}]}
        self.assertEqual(format_translation.anthropic_tool_choice_to_chat("NONE"), "none")
        self.assertEqual(format_translation.anthropic_request_to_responses({**base, "tool_choice": {"type": "auto"}})["tool_choice"], "auto")
        self.assertEqual(format_translation.anthropic_request_to_responses({**base, "tool_choice": {"type": "none"}})["tool_choice"], "none")
        self.assertNotIn(
            "tool_choice",
            format_translation.anthropic_request_to_responses({**base, "tool_choice": {"type": "any"}}),
        )
        self.assertNotIn(
            "tool_choice",
            format_translation.anthropic_request_to_responses(
                {**base, "tool_choice": {"type": "tool", "name": "Read"}}
            ),
        )
        translated = format_translation.anthropic_request_to_responses(
            {
                **base,
                "tools": [{"name": "Read", "description": "Read", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "any"},
            }
        )
        self.assertEqual(translated["tool_choice"], "required")
        self.assertEqual(
            translated["tools"],
            [{"type": "function", "name": "Read", "description": "Read", "parameters": {"type": "object"}}],
        )


if __name__ == "__main__":
    unittest.main()
