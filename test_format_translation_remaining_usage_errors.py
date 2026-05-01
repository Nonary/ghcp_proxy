import unittest

import format_translation


class FormatTranslationRemainingUsageErrorsTests(unittest.TestCase):
    def test_anthropic_error_type_for_status_exact_contract(self):
        cases = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            429: "rate_limit_error",
            503: "overloaded_error",
            529: "overloaded_error",
            418: "api_error",
            500: "api_error",
        }
        for status_code, expected_type in cases.items():
            with self.subTest(status_code=status_code):
                self.assertEqual(
                    format_translation._anthropic_error_type_for_status(status_code),
                    expected_type,
                )

    def test_anthropic_error_payload_preserves_only_complete_anthropic_errors(self):
        payload = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "already anthropic"},
        }

        self.assertIs(
            format_translation.anthropic_error_payload_from_openai(payload, 429),
            payload,
        )

        self.assertEqual(
            format_translation.anthropic_error_payload_from_openai(
                {
                    "type": "error",
                    "error": {"type": 429, "message": "numeric type"},
                    "detail": "detail fallback",
                },
                429,
                "explicit fallback",
            ),
            {
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "numeric type"},
            },
        )

        for error in (
            {"type": "rate_limit_error", "message": ""},
            {"type": "rate_limit_error", "message": None},
        ):
            with self.subTest(error=error):
                self.assertEqual(
                    format_translation.anthropic_error_payload_from_openai(
                        {"type": "error", "error": error, "detail": "detail fallback"},
                        429,
                        "explicit fallback",
                    ),
                    {
                        "type": "error",
                        "error": {
                            "type": "rate_limit_error",
                            "message": "detail fallback",
                        },
                    },
                )

    def test_anthropic_error_payload_fallback_shape_and_message_precedence(self):
        self.assertEqual(
            format_translation.anthropic_error_payload_from_openai({}, 500),
            {
                "type": "error",
                "error": {"type": "api_error", "message": "Request failed"},
            },
        )
        self.assertEqual(
            format_translation.anthropic_error_payload_from_openai(
                {"error": {"type": "custom_error", "message": "upstream message"}},
                400,
                "explicit fallback",
            ),
            {
                "type": "error",
                "error": {"type": "custom_error", "message": "upstream message"},
            },
        )
        self.assertEqual(
            format_translation.anthropic_error_payload_from_openai(
                {"detail": "detail message"},
                404,
                "explicit fallback",
            ),
            {
                "type": "error",
                "error": {"type": "not_found_error", "message": "detail message"},
            },
        )
        self.assertEqual(
            format_translation.anthropic_error_payload_from_openai(
                ["not", "a", "dict"],
                503,
                "explicit fallback",
            ),
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "explicit fallback"},
            },
        )

    def test_anthropic_usage_to_responses_usage_defaults_and_cache_contract(self):
        self.assertEqual(
            format_translation._anthropic_usage_to_responses_usage(None),
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        self.assertEqual(
            format_translation._anthropic_usage_to_responses_usage({}),
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        self.assertEqual(
            format_translation._anthropic_usage_to_responses_usage(
                {"cache_read_input_tokens": "7", "cache_creation_input_tokens": "3"}
            ),
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        self.assertEqual(
            format_translation._anthropic_usage_to_responses_usage(
                {
                    "input_tokens": 13,
                    "output_tokens": 8,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 2,
                }
            ),
            {
                "input_tokens": 20,
                "output_tokens": 8,
                "total_tokens": 28,
                "input_tokens_details": {
                    "cached_tokens": 5,
                    "cache_creation_input_tokens": 2,
                },
            },
        )
        self.assertEqual(
            format_translation._anthropic_usage_to_responses_usage(
                {
                    "input_tokens": 13,
                    "output_tokens": 8,
                    "cached_input_tokens": 5,
                }
            ),
            {
                "input_tokens": 18,
                "output_tokens": 8,
                "total_tokens": 26,
                "input_tokens_details": {"cached_tokens": 5},
            },
        )

    def test_parse_sse_block_crlf_boundaries_are_normalized_exactly(self):
        self.assertEqual(
            format_translation.parse_sse_block(
                "event: content_block_delta\r\n"
                "data:  first line\r\n"
                ": keepalive ignored\r\n"
                "data: second line\r\n"
            ),
            ("content_block_delta", "first line\nsecond line"),
        )
        self.assertEqual(
            format_translation.parse_sse_block("event: delta\ndata:payload-without-space"),
            ("delta", "payload-without-space"),
        )

    def test_extract_tool_call_deltas_filters_dict_items_from_delta_tool_calls(self):
        tool_call = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
        self.assertEqual(format_translation.extract_tool_call_deltas(None), [])
        self.assertEqual(format_translation.extract_tool_call_deltas({"tool_calls": tool_call}), [])
        self.assertEqual(
            format_translation.extract_tool_call_deltas(
                {"tool_calls": [tool_call, "not a dict", {"index": 1}]}
            ),
            [tool_call, {"index": 1}],
        )

    def test_summarize_inline_data_image_exact_contract(self):
        self.assertIsNone(format_translation._summarize_inline_data_image(None))
        self.assertIsNone(format_translation._summarize_inline_data_image("https://example.invalid/x.png"))
        self.assertEqual(
            format_translation._summarize_inline_data_image(
                "data:image/png;name=foo;base64,AAAA",
                detail="  original  ",
            ),
            "[inline tool image omitted: image/png, 35 chars, detail=original]",
        )
        self.assertEqual(
            format_translation._summarize_inline_data_image("data:,raw-payload"),
            "[inline tool image omitted: inline image, 17 chars]",
        )
        self.assertEqual(
            format_translation._summarize_inline_data_image("data:;base64,AAAA", detail="   "),
            "[inline tool image omitted: inline image, 17 chars]",
        )
        self.assertEqual(
            format_translation._summarize_inline_data_image(
                "data:application/json,{\"ok\":true}",
                detail=None,
            ),
            "[inline tool image omitted: inline image, 33 chars]",
        )
