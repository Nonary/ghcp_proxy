import unittest

import format_translation


class RemainingReasoningCompactionTests(unittest.TestCase):
    def test_anthropic_tool_result_content_text_join_and_validation(self):
        self.assertEqual(
            format_translation._anthropic_tool_result_content_to_text(
                [
                    "ignored scalar",
                    {"type": "text", "text": "alpha"},
                    {"type": "text", "text": "beta"},
                ]
            ),
            "alphabeta",
        )

        with self.assertRaisesRegex(
            ValueError,
            "^Anthropic tool_result content currently supports text blocks only$",
        ):
            format_translation._anthropic_tool_result_content_to_text(
                [{"type": "image", "text": "not valid tool output text"}]
            )

        with self.assertRaisesRegex(
            ValueError,
            "^Anthropic tool_result content must be a string or list of text blocks$",
        ):
            format_translation._anthropic_tool_result_content_to_text({"type": "text", "text": "nope"})

    def test_chat_completion_to_response_handles_non_dict_payload_shape(self):
        translated = format_translation.chat_completion_to_response(None, fallback_model="fallback")

        self.assertEqual(translated["object"], "response")
        self.assertEqual(translated["status"], "completed")
        self.assertEqual(translated["model"], "fallback")
        self.assertEqual(translated["output"], [])
        self.assertEqual(translated["output_text"], "")
        self.assertEqual(translated["usage"], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        self.assertIsInstance(translated["created_at"], int)
        self.assertTrue(translated["id"].startswith("resp_"))

    def test_chat_completion_to_compaction_response_fallback_summary_shape(self):
        translated = format_translation.chat_completion_to_compaction_response(
            {"id": "chat-empty", "created": 321, "model": "chat-model", "choices": {"unexpected": "shape"}}
        )

        self.assertEqual(
            {
                "id": translated["id"],
                "object": translated["object"],
                "created_at": translated["created_at"],
                "status": translated["status"],
                "model": translated["model"],
                "output_text": translated["output_text"],
                "usage": translated["usage"],
            },
            {
                "id": "chat-empty",
                "object": "response",
                "created_at": 321,
                "status": "completed",
                "model": "chat-model",
                "output_text": "(no summary available)",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            },
        )
        self.assertEqual(
            translated["output"],
            [
                {
                    "type": "compaction",
                    "encrypted_content": translated["output"][0]["encrypted_content"],
                }
            ],
        )
        self.assertEqual(
            format_translation.decode_fake_compaction(translated["output"][0]["encrypted_content"]),
            "(no summary available)",
        )
        self.assertIsNone(format_translation.decode_fake_compaction(None))
        self.assertIsNone(format_translation.decode_fake_compaction("not-a-local-compaction"))

    def test_extract_reasoning_from_chat_delta_reads_nested_summary_text(self):
        self.assertEqual(format_translation.extract_reasoning_from_chat_delta(None), "")
        self.assertEqual(
            format_translation.extract_reasoning_from_chat_delta({"thinking": {"text": "direct text"}}),
            "direct text",
        )
        self.assertEqual(
            format_translation.extract_reasoning_from_chat_delta(
                {"reasoning": {"summary": [{"text": "first"}, {"ignored": "x"}, {"text": "second"}]}}
            ),
            "firstsecond",
        )
        self.assertEqual(
            format_translation.extract_reasoning_from_chat_delta(
                {
                    "thinking": "a",
                    "reasoning_content": {"summary": [{"text": "b"}]},
                    "reasoning_text": "c",
                    "reasoning": [{"text": "d"}, {"text": ""}, {"ignored": "x"}],
                }
            ),
            "abcd",
        )
        self.assertEqual(
            format_translation.extract_reasoning_from_chat_delta(
                {"reasoning": ["first", {"ignored": "x"}, {"text": "second"}]}
            ),
            "firstsecond",
        )

    def test_extract_text_from_chat_delta_joins_text_parts_and_ignores_non_text(self):
        self.assertEqual(format_translation.extract_text_from_chat_delta(None), "")
        self.assertEqual(format_translation.extract_text_from_chat_delta({"content": "direct text"}), "direct text")
        self.assertEqual(
            format_translation.extract_text_from_chat_delta(
                {
                    "content": [
                        {"text": "alpha"},
                        "ignored scalar",
                        {"text": 123},
                        {"text": "beta"},
                    ]
                }
            ),
            "alphabeta",
        )

    def test_extract_reasoning_from_chat_delta_non_dict_and_list_text_contracts(self):
        self.assertEqual(format_translation.extract_reasoning_from_chat_delta(None), "")
        self.assertEqual(
            format_translation.extract_reasoning_from_chat_delta(
                {
                    "reasoning": [
                        {"text": "alpha"},
                        "beta",
                        {"ignored": "x"},
                        {"text": "gamma"},
                    ]
                }
            ),
            "alphabetagamma",
        )

    def test_sanitize_function_call_output_omits_content_and_summarizes_dict_data_images(self):
        self.assertIsNone(format_translation._sanitize_function_call_output_item(None))

        sanitized = format_translation._sanitize_function_call_output_item(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "content": [{"type": "input_text", "text": "must be omitted"}],
                "output": [
                    "raw scalar output",
                    {
                        "type": "input_image",
                        "image_url": {"url": "https://example.invalid/image.png"},
                    },
                    {
                        "type": "input_image",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                        "detail": "high",
                    },
                    {"type": "input_text", "text": "preserved"},
                ],
            }
        )

        self.assertEqual(
            sanitized,
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    "raw scalar output",
                    {
                        "type": "input_image",
                        "image_url": {"url": "https://example.invalid/image.png"},
                    },
                    {
                        "type": "input_text",
                        "text": "[inline tool image omitted: image/png, 26 chars, detail=high]",
                    },
                    {"type": "input_text", "text": "preserved"},
                ],
            },
        )

    def test_summarize_inline_data_image_requires_data_url_and_handles_media_details(self):
        self.assertIsNone(format_translation._summarize_inline_data_image(None))
        self.assertIsNone(format_translation._summarize_inline_data_image("https://example.invalid/image.png"))
        self.assertEqual(
            format_translation._summarize_inline_data_image("data:,AAAA", detail=123),
            "[inline tool image omitted: inline image, 10 chars]",
        )
        self.assertEqual(
            format_translation._summarize_inline_data_image(
                "data:image/svg+xml;charset=utf-8;base64,AAAA",
                detail=" low ",
            ),
            "[inline tool image omitted: image/svg+xml, 44 chars, detail=low]",
        )

    def test_reasoning_replay_value_requires_nonempty_encrypted_or_summary_text(self):
        self.assertFalse(format_translation._reasoning_summary_has_text([{"ignored": "missing text"}]))
        self.assertFalse(format_translation._reasoning_summary_has_text([{"text": "   "}]))
        self.assertTrue(format_translation._reasoning_summary_has_text([{"text": " direct summary "}]))
        self.assertFalse(format_translation._reasoning_item_has_replay_value({"encrypted_content": ""}))
        self.assertFalse(
            format_translation._reasoning_item_has_replay_value(
                {"encrypted_content": "", "summary": [{"text": "   "}]}
            )
        )
        self.assertTrue(
            format_translation._reasoning_item_has_replay_value(
                {"encrypted_content": "", "summary": [{"text": "keep me"}]}
            )
        )

    def test_reasoning_summary_has_text_requires_nonblank_string_text(self):
        self.assertFalse(format_translation._reasoning_summary_has_text("   "))
        self.assertTrue(format_translation._reasoning_summary_has_text(" visible string summary "))
        self.assertFalse(format_translation._reasoning_summary_has_text([{"text": "   "}]))
        self.assertFalse(format_translation._reasoning_summary_has_text([{"text": 123}]))
        self.assertTrue(format_translation._reasoning_summary_has_text([{"text": "visible"}]))

    def test_sanitize_input_preserves_non_null_reasoning_status_and_strips_nulls(self):
        sanitized = format_translation.sanitize_input(
            [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": "kept"}],
                    "content": [],
                    "note": None,
                }
            ]
        )

        self.assertEqual(
            sanitized,
            [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": "kept"}],
                }
            ],
        )

    def test_sanitize_input_preserves_all_reasoning_ciphertext(self):
        self.assertEqual(
            format_translation.sanitize_input(
                [
                    {"type": "reasoning", "encrypted_content": ""},
                    "ignored scalar",
                    {"type": "message", "encrypted_content": "not reasoning"},
                    {"type": "reasoning", "encrypted_content": "ciphertext-1"},
                    {"type": "reasoning", "encrypted_content": "ciphertext-2"},
                ]
            ),
            [
                {"type": "reasoning", "encrypted_content": ""},
                "ignored scalar",
                {"type": "message", "encrypted_content": "not reasoning"},
                {"type": "reasoning", "encrypted_content": "ciphertext-1"},
                {"type": "reasoning", "encrypted_content": "ciphertext-2"},
            ],
        )

    def test_format_compaction_tool_call_uses_empty_json_for_blank_string_arguments(self):
        self.assertIsNone(
            format_translation._format_compaction_tool_call(
                {"type": "function_call", "call_id": "call_empty", "name": "", "arguments": {}}
            )
        )
        self.assertEqual(
            format_translation._format_compaction_tool_call(
                {"type": "function_call", "id": "item_1", "name": "ListFiles", "arguments": None}
            ),
            "[Tool call (item_1)] ListFiles\n{}",
        )
        self.assertEqual(
            format_translation._format_compaction_tool_call(
                {"type": "function_call", "call_id": "call_1", "name": "Lookup", "arguments": "   "}
            ),
            "[Tool call (call_1)] Lookup\n{}",
        )

    def test_fake_compaction_summary_detection_requires_message_with_label_prefix(self):
        summary = {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"{format_translation.FAKE_COMPACTION_SUMMARY_LABEL}\nprevious summary",
                }
            ],
        }

        self.assertFalse(format_translation._is_fake_compaction_summary_message(None))
        self.assertFalse(
            format_translation._is_fake_compaction_summary_message(
                {"type": "compaction", "content": summary["content"]}
            )
        )
        self.assertFalse(
            format_translation._is_fake_compaction_summary_message(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "[Compacted conversation summary]"}],
                }
            )
        )
        self.assertTrue(format_translation._is_fake_compaction_summary_message(summary))

    def test_summary_message_item_exact_contract(self):
        self.assertEqual(
            format_translation._summary_message_item("carry this forward"),
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"{format_translation.FAKE_COMPACTION_SUMMARY_LABEL}\ncarry this forward",
                    }
                ],
            },
        )

    def test_latest_compaction_window_uses_last_compaction_item(self):
        raw_input = "plain transcript"
        self.assertIs(format_translation._latest_compaction_window(raw_input), raw_input)

        no_compaction = [{"type": "message", "id": "m1"}, "raw scalar"]
        self.assertIs(format_translation._latest_compaction_window(no_compaction), no_compaction)

        older = {"type": "compaction", "encrypted_content": "older"}
        latest = {"type": "compaction", "encrypted_content": "latest"}
        after = {"type": "message", "id": "after"}
        self.assertEqual(
            format_translation._latest_compaction_window(
                [
                    {"type": "message", "id": "before"},
                    older,
                    "between",
                    {"type": "message", "id": "middle"},
                    latest,
                    after,
                ]
            ),
            [latest, after],
        )

    def test_sanitize_input_expands_latest_local_compaction_summary(self):
        latest_summary = "latest compacted state"
        sanitized = format_translation.sanitize_input(
            [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old"}]},
                {"type": "compaction", "encrypted_content": format_translation.encode_fake_compaction("old summary")},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "middle"}]},
                {
                    "type": "compaction",
                    "encrypted_content": format_translation.encode_fake_compaction(latest_summary),
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "after"}],
                },
            ]
        )

        self.assertEqual(
            sanitized,
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{format_translation.FAKE_COMPACTION_SUMMARY_LABEL}\n{latest_summary}",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "after"}],
                },
            ],
        )

    def test_sanitize_input_strips_non_message_content_and_continues(self):
        message = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "message content stays"}],
        }

        self.assertEqual(
            format_translation.sanitize_input(
                [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "Lookup",
                        "arguments": "{}",
                        "content": [{"type": "input_text", "text": "legacy content"}],
                    },
                    message,
                ]
            ),
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Lookup",
                    "arguments": "{}",
                },
                message,
            ],
        )

    def test_sanitize_input_preserves_scalar_passthrough_items(self):
        raw_items = [
            "raw scalar transcript item",
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "message content stays"}],
            },
        ]

        self.assertEqual(format_translation.sanitize_input(raw_items), raw_items)

    def test_sanitize_input_omits_reasoning_ciphertext_when_replay_disabled(self):
        self.assertEqual(
            format_translation.sanitize_input(
                [
                    {
                        "type": "reasoning",
                        "encrypted_content": "ciphertext",
                        "summary": [{"text": "visible reasoning summary"}],
                        "content": [{"type": "input_text", "text": "invalid reasoning content"}],
                        "status": None,
                        "metadata": {"keep": True},
                        "nullable": None,
                    }
                ],
                preserve_encrypted_content=False,
            ),
            [
                {
                    "type": "reasoning",
                    "summary": [{"text": "visible reasoning summary"}],
                    "metadata": {"keep": True},
                }
            ],
        )

    def test_sanitize_input_preserves_compaction_without_replayable_content(self):
        empty_compaction = {"type": "compaction", "encrypted_content": ""}
        missing_compaction = {"type": "compaction", "metadata": {"keep": True}}

        self.assertEqual(format_translation.sanitize_input([empty_compaction]), [empty_compaction])
        self.assertEqual(format_translation.sanitize_input([missing_compaction]), [missing_compaction])

    def test_format_compaction_tool_output_contract_for_labels_and_output_shapes(self):
        self.assertEqual(
            format_translation._format_compaction_tool_output(
                {"type": "function_call_output", "call_id": "call_1", "output": " file contents "}
            ),
            "[Tool result (call_1)]\nfile contents",
        )
        self.assertEqual(
            format_translation._format_compaction_tool_output(
                {
                    "type": "function_call_output",
                    "call_id": "",
                    "output": [
                        "ignored scalar",
                        {"type": "output_text", "text": "alpha"},
                        {"type": "input_text", "text": "beta"},
                    ],
                }
            ),
            "[Tool result]\nalphabeta",
        )
        self.assertEqual(
            format_translation._format_compaction_tool_output(
                {"type": "function_call_output", "call_id": "call_empty", "output": "   "}
            ),
            "[Tool result (call_empty)]",
        )
        self.assertEqual(
            format_translation._format_compaction_tool_output(
                {"type": "function_call_output", "output": {"ok": True, "count": 2}}
            ),
            '[Tool result]\n{"ok":true,"count":2}',
        )

    def test_copy_compaction_passthrough_fields_preserves_user_key(self):
        target = {"existing": True}
        format_translation._copy_compaction_passthrough_fields(
            {
                "session_id": "session",
                "sessionId": "session-camel",
                "prompt_cache_key": "cache",
                "promptCacheKey": "cache-camel",
                "previous_response_id": "resp_prev",
                "metadata": {"trace": "1"},
                "user": "user-123",
                "input": "not copied here",
            },
            target,
        )

        self.assertEqual(
            target,
            {
                "existing": True,
                "session_id": "session",
                "sessionId": "session-camel",
                "prompt_cache_key": "cache",
                "promptCacheKey": "cache-camel",
                "previous_response_id": "resp_prev",
                "metadata": {"trace": "1"},
                "user": "user-123",
            },
        )

        list_target = []
        self.assertIsNone(
            format_translation._copy_compaction_passthrough_fields({"user": "user-123"}, list_target)
        )
        self.assertEqual(list_target, [])

    def test_compaction_transcript_message_item_accepts_system_and_developer_roles(self):
        for role in ("system", "developer"):
            with self.subTest(role=role):
                item = {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": f"{role} instructions"}],
                }

                self.assertIs(format_translation._compaction_transcript_message_item(item, force_user_role=False), item)

                forced = format_translation._compaction_transcript_message_item(item, force_user_role=True)
                self.assertEqual(
                    forced,
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"[{role} message]\n{role} instructions",
                            }
                        ],
                    },
                )

    def test_compaction_transcript_message_item_role_contract_edges(self):
        missing_role = {
            "type": "message",
            "content": [{"type": "input_text", "text": "plain user text"}],
        }
        self.assertIs(
            format_translation._compaction_transcript_message_item(missing_role, force_user_role=False),
            missing_role,
        )
        self.assertEqual(
            format_translation._compaction_transcript_message_item(missing_role, force_user_role=True),
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "plain user text"}],
            },
        )

        assistant_item = {
            "type": "message",
            "role": "Assistant",
            "content": [{"type": "output_text", "text": "mixed case role"}],
        }
        self.assertIs(
            format_translation._compaction_transcript_message_item(assistant_item, force_user_role=False),
            assistant_item,
        )
        self.assertEqual(
            format_translation._compaction_transcript_message_item(assistant_item, force_user_role=True),
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "[assistant message]\nmixed case role"}],
            },
        )

        self.assertIsNone(
            format_translation._compaction_transcript_message_item(
                {
                    "type": "message",
                    "role": "tool",
                    "content": [{"type": "input_text", "text": "not a transcript role"}],
                },
                force_user_role=True,
            )
        )

        fake_summary = {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": f"{format_translation.FAKE_COMPACTION_SUMMARY_LABEL}\nprior summary",
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
                        "text": f"{format_translation.FAKE_COMPACTION_SUMMARY_LABEL}\nprior summary",
                    }
                ],
            },
        )

    def test_compaction_transcript_omits_reasoning_and_item_references(self):
        transcript = format_translation._compaction_transcript_input_items(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "visible"}],
                },
                {"type": "reasoning", "summary": [{"text": "hidden thought"}]},
                {"type": "item_reference", "id": "rs_1"},
            ]
        )

        self.assertEqual(
            transcript,
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "visible"}],
                }
            ],
        )

    def test_compaction_transcript_input_items_decodes_fake_compaction_summary(self):
        transcript = format_translation._compaction_transcript_input_items(
            [
                {"type": "compaction", "encrypted_content": "opaque-upstream-token"},
                {
                    "type": "compaction",
                    "encrypted_content": format_translation.encode_fake_compaction("carry this forward"),
                },
            ],
            force_user_role=True,
        )

        self.assertEqual(
            transcript,
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{format_translation.FAKE_COMPACTION_SUMMARY_LABEL}\ncarry this forward",
                        }
                    ],
                }
            ],
        )

    def test_compaction_transcript_input_items_preserves_items_after_tool_calls(self):
        transcript = format_translation._compaction_transcript_input_items(
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Lookup",
                    "arguments": {"query": "alpha"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "lookup result",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "final answer"}],
                },
            ],
        )

        self.assertEqual(
            transcript,
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "[Tool call (call_1)] Lookup\n{\"query\":\"alpha\"}",
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "[Tool result (call_1)]\nlookup result"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "final answer"}],
                },
            ],
        )

    def test_build_fake_compaction_request_defaults_to_native_responses_transcript_for_gpt(self):
        compact_request = format_translation.build_fake_compaction_request(
            {
                "model": "gpt-5.4",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "native assistant"}],
                    }
                ],
            }
        )

        self.assertEqual(compact_request["input"][0]["role"], "assistant")
        self.assertEqual(compact_request["input"][0]["content"][0]["type"], "output_text")
        self.assertEqual(compact_request["input"][0]["content"][0]["text"], "native assistant")
        self.assertEqual(compact_request["input"][1]["content"][0]["text"], format_translation.COMPACTION_SUMMARY_PROMPT)

    def test_build_fake_compaction_request_sanitizes_list_input_before_appending_prompt(self):
        compact_request = format_translation.build_fake_compaction_request(
            {
                "model": "gpt-5.4",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "old prompt"}],
                    },
                    {
                        "type": "compaction",
                        "encrypted_content": format_translation.encode_fake_compaction("carry forward"),
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "Lookup",
                        "arguments": "{}",
                        "content": [{"type": "input_text", "text": "strip me"}],
                    },
                ],
            }
        )

        self.assertEqual(
            compact_request["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{format_translation.FAKE_COMPACTION_SUMMARY_LABEL}\ncarry forward",
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Lookup",
                    "arguments": "{}",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": format_translation.COMPACTION_SUMMARY_PROMPT}],
                },
            ],
        )

    def test_build_fake_compaction_request_transcript_string_input_uses_user_role(self):
        compact_request = format_translation.build_fake_compaction_request(
            {
                "model": "claude-opus-4.6",
                "input": "summarize this",
            }
        )

        self.assertEqual(compact_request["input"][0]["role"], "user")
        self.assertEqual(compact_request["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(compact_request["input"][0]["content"][0]["text"], "summarize this")
        self.assertEqual(compact_request["input"][1]["content"][0]["text"], format_translation.COMPACTION_SUMMARY_PROMPT)

    def test_build_fake_compaction_request_preserves_passthrough_config_and_cache_fields(self):
        compact_request = format_translation.build_fake_compaction_request(
            {
                "model": "gpt-5.4",
                "input": "summarize this",
                "reasoning": {"effort": "high"},
                "store": False,
                "metadata": {"session": "abc"},
                "include": ["reasoning.encrypted_content"],
            }
        )

        self.assertEqual(compact_request["reasoning"], {"effort": "high"})
        self.assertIs(compact_request["store"], False)
        self.assertEqual(compact_request["metadata"], {"session": "abc"})
        self.assertEqual(compact_request["include"], ["reasoning.encrypted_content"])


if __name__ == "__main__":
    unittest.main()
