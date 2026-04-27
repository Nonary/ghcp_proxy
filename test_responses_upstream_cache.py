import unittest
from unittest import mock

import responses_upstream_cache as ruc


class ResponsesUpstreamCacheSanitizerTests(unittest.TestCase):
    def test_body_sanitizer_handles_non_dict_clean_body_and_diagnostics_opt_in(self):
        self.assertEqual(ruc.sanitize_responses_body_for_copilot("bad"), "bad")
        clean = {"model": "gpt-5.4", "input": "hi", "stream": False}
        self.assertIs(ruc.sanitize_responses_body_for_copilot(clean), clean)

        diagnostics = object()
        sanitized = ruc.sanitize_responses_body_for_copilot(
            {"model": "gpt-5.4", "input": "hi", "local": True},
            diagnostics=diagnostics,
        )
        self.assertEqual(sanitized, {"model": "gpt-5.4", "input": "hi"})

    def test_body_sanitizer_exact_upstream_contract_and_diagnostic(self):
        diagnostics = []
        body = {
            "model": "gpt-5.4",
            "input": "hello",
            "instructions": "sys",
            "stream": True,
            "include": ["output_text"],
            "parallel_tool_calls": False,
            "reasoning": {"effort": "high"},
            "store": False,
            "text": {"format": {"type": "text"}},
            "tools": [{"type": "function", "name": "Read"}],
            "client_metadata": {"local": True},
            "prompt_cache_key": "cache-key",
            "previous_response_id": "resp_prev",
            "service_tier": "priority",
        }
        with mock.patch("builtins.print") as print_mock:
            sanitized = ruc.sanitize_responses_body_for_copilot(body, diagnostics=diagnostics)

        self.assertEqual(
            sanitized,
            {
                "model": "gpt-5.4",
                "input": "hello",
                "instructions": "sys",
                "stream": True,
                "include": ["output_text"],
                "parallel_tool_calls": False,
                "reasoning": {"effort": "high"},
                "store": False,
                "text": {"format": {"type": "text"}},
                "tools": [{"type": "function", "name": "Read"}],
            },
        )
        self.assertEqual(
            diagnostics,
            [
                {
                    "kind": "responses_body",
                    "action": "drop_non_copilot_cli_fields",
                    "fields": [
                        "client_metadata",
                        "previous_response_id",
                        "prompt_cache_key",
                        "service_tier",
                    ],
                }
            ],
        )
        print_mock.assert_called_once_with(
            "Responses proxy dropped non-Copilot-CLI fields: "
            "client_metadata, previous_response_id, prompt_cache_key, service_tier",
            flush=True,
        )

    def test_tool_choice_removed_detection_covers_type_and_function_names(self):
        self.assertFalse(ruc._responses_tool_choice_targets_removed_tool("auto", {"image_generation"}, set()))
        self.assertTrue(
            ruc._responses_tool_choice_targets_removed_tool(
                {"type": "image_generation"},
                {"image_generation"},
                set(),
            )
        )
        self.assertTrue(
            ruc._responses_tool_choice_targets_removed_tool(
                {"type": "function", "function": {"name": "Draw"}},
                set(),
                {"draw"},
            )
        )
        self.assertFalse(
            ruc._responses_tool_choice_targets_removed_tool(
                {"type": "function", "function": {"name": "Read"}},
                {"image_generation"},
                {"draw"},
            )
        )

    def test_tool_search_detection_covers_flat_nested_suffix_and_non_tools(self):
        self.assertFalse(ruc.responses_tools_have_tool_search(None))
        self.assertFalse(ruc.responses_tools_have_tool_search({"type": "function", "name": "read"}))
        self.assertFalse(ruc.responses_tools_have_tool_search({"type": "function", "function": {"name": 3}}))
        self.assertTrue(ruc.responses_tools_have_tool_search({"type": "tool_search"}))
        self.assertTrue(ruc.responses_tools_have_tool_search({"tool_search": {"query": "x"}}))
        self.assertTrue(ruc.responses_tools_have_tool_search({"type": "function", "name": "tools.tool_search"}))
        self.assertTrue(ruc.responses_tools_have_tool_search({"type": "function", "name": "mcp__tools__tool_search"}))
        self.assertTrue(
            ruc.responses_tools_have_tool_search(
                [{"type": "namespace", "tools": [{"type": "function", "function": {"name": "tool_search"}}]}]
            )
        )

    def test_dangerous_tool_stripping_covers_lists_dicts_and_empty_tool_sets(self):
        no_tools = {"model": "gpt-5.4"}
        self.assertIs(ruc.sanitize_responses_tools_for_copilot(no_tools), no_tools)
        self.assertIs(ruc.sanitize_responses_tools_for_copilot("bad"), "bad")

        diagnostics = []
        all_dangerous = {
            "tools": [{"type": "function", "name": "mcp__ide__executeCode"}],
            "parallel_tool_calls": True,
            "tool_choice": {"type": "function", "name": "mcp__ide__executeCode"},
        }
        sanitized = ruc.sanitize_responses_tools_for_copilot(all_dangerous, diagnostics=diagnostics)
        self.assertNotIn("tools", sanitized)
        self.assertNotIn("parallel_tool_calls", sanitized)
        self.assertNotIn("tool_choice", sanitized)
        self.assertEqual(
            diagnostics[0],
            {
                "kind": "responses_tools",
                "action": "drop_dangerous_code_execution_tools",
                "tool_names": ["mcp__ide__executecode"],
                "tools": [
                    {
                        "path": "tools[0]",
                        "name": "mcp__ide__executeCode",
                        "type": "function",
                    }
                ],
                "truncated": False,
            },
        )

        nested_dict = {
            "tools": {
                "type": "namespace",
                "name": "ide",
                "tools": {"type": "function", "name": "mcp__ide__executeCode"},
            }
        }
        sanitized = ruc.sanitize_responses_tools_for_copilot(nested_dict)
        self.assertEqual(sanitized["tools"], {"type": "namespace", "name": "ide"})

        nested_list = {
            "tools": {
                "type": "namespace",
                "name": "ide",
                "tools": [
                    {"type": "function", "name": "read"},
                    {"type": "function", "name": "mcp__ide__executeCode"},
                ],
            }
        }
        sanitized = ruc.sanitize_responses_tools_for_copilot(nested_list)
        self.assertEqual(sanitized["tools"]["tools"], [{"type": "function", "name": "read"}])

        self.assertEqual(
            ruc._strip_dangerous_responses_tools({"tools": "bad"}),
            {"tools": "bad"},
        )
        removed_for_mock = [
            {"path": "tools[0]", "name": "mcp__ide__executeCode", "type": "function"},
            {"path": "tools[1]", "name": None, "type": "function"},
            {"path": "tools[2]", "name": "", "type": "function"},
        ]

        def fake_drop(tools, *, path, removed):
            self.assertEqual(path, "tools")
            removed.extend(removed_for_mock)
            return tools, True

        with mock.patch.object(ruc, "_drop_dangerous_responses_tools", side_effect=fake_drop):
            diagnostics = []
            sanitized = ruc._strip_dangerous_responses_tools(
                {"tools": [{"type": "function", "name": "Read"}]},
                diagnostics=diagnostics,
            )

        self.assertEqual(sanitized["tools"], [{"type": "function", "name": "Read"}])
        self.assertEqual(diagnostics[0]["tool_names"], ["mcp__ide__executecode"])
        removed = []
        self.assertEqual(
            ruc._drop_dangerous_responses_tools(
                {"type": "function", "name": "mcp__ide__executeCode"},
                path="tools",
                removed=removed,
            ),
            (None, True),
        )
        self.assertEqual(
            removed,
            [{"path": "tools", "name": "mcp__ide__executeCode", "type": "function"}],
        )

    def test_dangerous_tool_stripping_exact_paths_noop_and_truncation(self):
        safe_tools = [
            {"type": "function", "name": "Read"},
            {"type": "namespace", "name": "safe", "tools": [{"type": "function", "name": "Write"}]},
        ]
        self.assertEqual(
            ruc._drop_dangerous_responses_tools(safe_tools, path="tools", removed=[]),
            (safe_tools, False),
        )

        removed = []
        sanitized_tools, changed = ruc._drop_dangerous_responses_tools(
            [
                {"type": "function", "name": "mcp__ide__executeCode"},
                {
                    "type": "namespace",
                    "name": "ide",
                    "tools": [
                        {"type": "function", "name": "read"},
                        {"type": "function", "function": {"name": "mcp__ide__executeCode"}},
                    ],
                },
            ],
            path="tools",
            removed=removed,
        )
        self.assertTrue(changed)
        self.assertEqual(
            sanitized_tools,
            [
                {
                    "type": "namespace",
                    "name": "ide",
                    "tools": [{"type": "function", "name": "read"}],
                }
            ],
        )
        self.assertEqual(
            removed,
            [
                {"path": "tools[0]", "name": "mcp__ide__executeCode", "type": "function"},
                {"path": "tools[1].tools[1]", "name": "mcp__ide__executeCode", "type": "function"},
            ],
        )

        diagnostics = []
        body = {
            "tools": [
                {"type": "function", "name": "mcp__ide__executeCode"}
                for _ in range(22)
            ],
            "tool_choice": {"type": "function", "name": "mcp__ide__executeCode"},
        }
        sanitized = ruc.sanitize_responses_tools_for_copilot(body, diagnostics=diagnostics)
        self.assertEqual(sanitized, {})
        self.assertEqual(diagnostics[0]["kind"], "responses_tools")
        self.assertEqual(diagnostics[0]["action"], "drop_dangerous_code_execution_tools")
        self.assertEqual(diagnostics[0]["tool_names"], ["mcp__ide__executecode"])
        self.assertEqual(len(diagnostics[0]["tools"]), 20)
        self.assertTrue(diagnostics[0]["truncated"])

    def test_defer_loading_sanitizer_handles_absent_search_noops_and_nested_changes(self):
        with_search = {
            "tools": [
                {"type": "function", "name": "tool_search"},
                {"type": "function", "name": "read", "defer_loading": True},
            ]
        }
        self.assertIs(ruc.sanitize_responses_tools_for_copilot(with_search), with_search)

        no_defer = {"tools": [{"type": "function", "name": "read"}]}
        self.assertIs(ruc.sanitize_responses_tools_for_copilot(no_defer), no_defer)

        nested = {
            "tools": {
                "type": "namespace",
                "name": "app",
                "tools": [{"type": "function", "name": "deferred", "defer_loading": False}],
            }
        }
        diagnostics = []
        sanitized = ruc.sanitize_responses_tools_for_copilot(nested, diagnostics=diagnostics)
        self.assertNotIn("defer_loading", sanitized["tools"]["tools"][0])
        self.assertEqual(diagnostics[0]["action"], "strip_defer_loading")

        self.assertEqual(ruc._strip_invalid_responses_defer_loading({"tools": "bad"}), {"tools": "bad"})
        direct_defer = ruc._strip_responses_defer_loading_fields(
            {"type": "namespace", "tools": [{"type": "function", "name": "read"}]},
            path="tools",
            removed=[],
        )
        self.assertEqual(direct_defer, ({"type": "namespace", "tools": [{"type": "function", "name": "read"}]}, False))
        direct_defer = ruc._strip_responses_defer_loading_fields(
            {"type": "namespace", "defer_loading": True, "tools": [{"type": "function", "name": "read"}]},
            path="tools",
            removed=[],
        )
        self.assertFalse(direct_defer[0]["defer_loading"] if "defer_loading" in direct_defer[0] else False)
        direct_defer = ruc._strip_responses_defer_loading_fields(
            {
                "type": "namespace",
                "defer_loading": True,
                "tools": [{"type": "function", "name": "nested", "defer_loading": True}],
            },
            path="tools",
            removed=[],
        )
        self.assertNotIn("defer_loading", direct_defer[0])
        self.assertNotIn("defer_loading", direct_defer[0]["tools"][0])

    def test_defer_loading_sanitizer_exact_diagnostic_and_truncation(self):
        diagnostics = []
        body = {
            "tools": [
                {"type": "function", "name": f"tool_{index}", "defer_loading": index % 2 == 0}
                for index in range(22)
            ]
        }
        sanitized = ruc.sanitize_responses_tools_for_copilot(body, diagnostics=diagnostics)

        self.assertEqual(
            sanitized,
            {
                "tools": [
                    {"type": "function", "name": f"tool_{index}"}
                    for index in range(22)
                ]
            },
        )
        self.assertEqual(diagnostics[0]["kind"], "responses_tools")
        self.assertEqual(diagnostics[0]["action"], "strip_defer_loading")
        self.assertEqual(diagnostics[0]["reason"], "deferred_tools_require_tool_search")
        self.assertEqual(diagnostics[0]["count"], 22)
        self.assertEqual(diagnostics[0]["tool_names"], [f"tool_{index}" for index in range(20)])
        self.assertEqual(diagnostics[0]["tools"][0], {"path": "tools[0]", "name": "tool_0", "type": "function", "value": True})
        self.assertEqual(len(diagnostics[0]["tools"]), 20)
        self.assertTrue(diagnostics[0]["truncated"])

        diagnostics = []
        body = {
            "tools": [
                {"type": "function", "name": f"tool_{index}", "defer_loading": True}
                for index in range(20)
            ]
        }
        with mock.patch("builtins.print") as print_mock:
            ruc.sanitize_responses_tools_for_copilot(body, diagnostics=diagnostics)

        self.assertFalse(diagnostics[0]["truncated"])
        print_mock.assert_called_once_with(
            "Responses proxy stripped defer_loading from 20 tool definition(s) because tool_search was absent",
            flush=True,
        )

        diagnostics = []
        body = {
            "tools": [
                {"type": "function", "name": f"tool_{index}", "defer_loading": True}
                for index in range(21)
            ]
        }
        ruc.sanitize_responses_tools_for_copilot(body, diagnostics=diagnostics)
        self.assertTrue(diagnostics[0]["truncated"])

    def test_unsupported_responses_tools_filter_tool_choices_and_keep_non_dict_items(self):
        body = {
            "tools": [
                "raw",
                {"type": "image_generation", "name": "Draw"},
                {"type": "function", "name": "Read"},
            ],
            "tool_choice": {"type": "function", "function": {"name": "Draw"}},
        }
        diagnostics = []
        sanitized = ruc.sanitize_responses_tools_for_copilot(body, diagnostics=diagnostics)
        self.assertEqual(sanitized["tools"], ["raw", {"type": "function", "name": "Read"}])
        self.assertNotIn("tool_choice", sanitized)
        self.assertEqual(
            diagnostics[-1],
            {
                "kind": "responses_tools",
                "action": "drop_unsupported_tool_types",
                "tool_types": ["image_generation"],
                "tool_names": ["draw"],
            },
        )

        untargeted_choice = {
            "tools": [
                {"type": "image_generation", "name": "Draw"},
                {"type": "function", "name": "Read"},
            ],
            "tool_choice": {"type": "function", "name": "Read"},
        }
        sanitized = ruc.sanitize_responses_tools_for_copilot(untargeted_choice)
        self.assertEqual(sanitized["tool_choice"], {"type": "function", "name": "Read"})

        only_unsupported = {
            "tools": [{"type": "image_generation"}],
            "parallel_tool_calls": True,
            "tool_choice": {"type": "image_generation"},
        }
        sanitized = ruc.sanitize_responses_tools_for_copilot(only_unsupported)
        self.assertNotIn("tools", sanitized)
        self.assertNotIn("parallel_tool_calls", sanitized)
        self.assertNotIn("tool_choice", sanitized)

        only_unsupported_without_parallel = {
            "tools": [{"type": "image_generation"}],
            "tool_choice": {"type": "image_generation"},
        }
        sanitized = ruc.sanitize_responses_tools_for_copilot(only_unsupported_without_parallel)
        self.assertEqual(sanitized, {})


class ResponsesUpstreamCacheFormattingHelpersTests(unittest.TestCase):
    def test_custom_tool_call_and_output_formatting_covers_all_input_shapes(self):
        self.assertIsNone(ruc._format_custom_tool_call_for_chat({"input": "x"}))
        self.assertEqual(
            ruc._format_custom_tool_call_for_chat({"name": "shell", "call_id": "ct_1", "input": "  "}),
            "[Custom tool call (ct_1)] shell\n[no input]",
        )
        self.assertEqual(
            ruc._format_custom_tool_call_for_chat({"name": "shell", "id": "ct_2", "input": {"cmd": "ls"}}),
            '[Custom tool call (ct_2)] shell\n{"cmd":"ls"}',
        )
        self.assertEqual(
            ruc._format_custom_tool_call_for_chat({"name": "shell", "id": "ct_3", "input": {"text": "café", "values": [1, 2]}}),
            '[Custom tool call (ct_3)] shell\n{"text":"café","values":[1,2]}',
        )
        with mock.patch.object(ruc.json, "dumps", return_value="encoded") as dumps_mock:
            self.assertEqual(
                ruc._format_custom_tool_call_for_chat({"name": "shell", "input": {"text": "café"}}),
                "[Custom tool call] shell\nencoded",
            )
        dumps_mock.assert_called_once_with(
            {"text": "café"},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self.assertEqual(
            ruc._format_custom_tool_call_for_chat({"name": "shell", "call_id": 7, "input": "ls"}),
            "[Custom tool call] shell\nls",
        )
        self.assertEqual(
            ruc._format_custom_tool_call_for_chat({"name": "shell"}),
            "[Custom tool call] shell\n[no input]",
        )

        self.assertEqual(ruc._format_custom_tool_output_for_chat({"call_id": "ct_1", "output": " ok "}), "[Custom tool result (ct_1)]\nok")
        self.assertEqual(ruc._format_custom_tool_output_for_chat({"output": None}), "[Custom tool result]")
        self.assertEqual(ruc._format_custom_tool_output_for_chat({"output": {"ok": True}}), '[Custom tool result]\n{"ok":true}')
        self.assertEqual(
            ruc._format_custom_tool_output_for_chat({"output": {"text": "café", "values": [1, 2]}}),
            '[Custom tool result]\n{"text":"café","values":[1,2]}',
        )
        with mock.patch.object(ruc.json, "dumps", return_value="encoded") as dumps_mock:
            self.assertEqual(
                ruc._format_custom_tool_output_for_chat({"output": {"text": "café"}}),
                "[Custom tool result]\nencoded",
            )
        dumps_mock.assert_called_once_with(
            {"text": "café"},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self.assertEqual(
            ruc._format_custom_tool_output_for_chat({"output": [{"type": "output_text", "text": "a"}, "skip"]}),
            "[Custom tool result]\na",
        )
        self.assertEqual(
            ruc._format_custom_tool_output_for_chat(
                {"output": [{"type": "output_text", "text": "a"}, {"type": "output_text", "text": "b"}]}
            ),
            "[Custom tool result]\nab",
        )

    def test_content_block_conversion_covers_text_files_images_and_invalid_shapes(self):
        self.assertIsNone(ruc._responses_input_text_to_anthropic_block("bad"))
        self.assertEqual(ruc._responses_input_text_to_anthropic_block({"type": "text", "text": "hi"}), {"type": "text", "text": "hi"})
        self.assertEqual(ruc._responses_input_text_to_anthropic_block({"type": "output_text", "text": "hi"})["text"], "hi")
        self.assertIsNone(ruc._responses_input_text_to_anthropic_block({"type": "input_text", "text": 1}))
        self.assertIsNone(ruc._responses_input_text_to_anthropic_block({"type": "file", "file_data": 1}))
        self.assertEqual(
            ruc._responses_input_text_to_anthropic_block({"type": "file", "file_data": "data:;base64,PDF"}),
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "PDF"},
                "title": "document.pdf",
            },
        )
        self.assertEqual(
            ruc._responses_input_text_to_anthropic_block(
                {"type": "input_file", "filename": "notes.pdf", "file_data": "data:application/pdf;base64,PDF"}
            )["title"],
            "notes.pdf",
        )
        self.assertIsNone(ruc._responses_input_text_to_anthropic_block({"type": "file", "file_data": "data:bad"}))
        self.assertEqual(
            ruc._responses_input_text_to_anthropic_block({"type": "input_image", "image_url": "data:;base64,IMG"}),
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "IMG"}},
        )
        self.assertIsNone(ruc._responses_input_text_to_anthropic_block({"type": "input_image", "image_url": "data:bad"}))
        self.assertEqual(
            ruc._responses_input_text_to_anthropic_block({"type": "input_image", "image_url": "https://x/y.png"})["source"],
            {"type": "url", "url": "https://x/y.png"},
        )

    def test_reasoning_signature_and_block_conversion_cover_edges(self):
        self.assertEqual(ruc._split_reasoning_signature(None), ("", ""))
        self.assertEqual(ruc._split_reasoning_signature(123), ("", ""))
        self.assertEqual(ruc._split_reasoning_signature("cipher"), ("cipher", ""))
        self.assertEqual(ruc._split_reasoning_signature(""), ("", ""))
        self.assertEqual(ruc._split_reasoning_signature("@id"), ("@id", ""))
        self.assertEqual(ruc._split_reasoning_signature("cipher@"), ("cipher@", ""))
        self.assertEqual(ruc._split_reasoning_signature("cipher@with@id"), ("cipher@with", "id"))

        self.assertIsNone(ruc._responses_reasoning_item_to_anthropic_block("bad"))
        self.assertEqual(
            ruc._responses_reasoning_item_to_anthropic_block({"summary": "one"})["thinking"],
            "one",
        )
        self.assertEqual(
            ruc._responses_reasoning_item_to_anthropic_block({"summary": ["a", {"text": "b"}, {"text": 2}]})["thinking"],
            "a\nb",
        )
        self.assertEqual(
            ruc._responses_reasoning_item_to_anthropic_block({"summary": [{"text": "a"}, 3]})["thinking"],
            "a",
        )
        self.assertEqual(
            ruc._responses_reasoning_item_to_anthropic_block({"encrypted_content": "cipher"})["signature"],
            "cipher",
        )

    def test_function_call_output_and_tool_conversion_cover_fallbacks(self):
        self.assertEqual(
            ruc._responses_function_call_to_anthropic_block({"name": "Read", "call_id": "c", "arguments": ""})["input"],
            {},
        )
        self.assertEqual(
            ruc._responses_function_call_to_anthropic_block({"name": "Read", "call_id": "c", "arguments": {"path": "a"}})["input"],
            {"path": "a"},
        )
        self.assertEqual(
            ruc._responses_function_call_output_to_anthropic_block({"call_id": "c", "output": [{"type": "output_text", "text": "ok"}]})["content"],
            [{"type": "text", "text": "ok"}],
        )
        self.assertEqual(
            ruc._responses_function_call_output_to_anthropic_block({"call_id": "c", "output": {"text": "café", "values": [1, 2]}})["content"],
            [{"type": "text", "text": '{"text":"café","values":[1,2]}'}],
        )
        with mock.patch.object(ruc.json, "dumps", return_value="encoded") as dumps_mock:
            self.assertEqual(
                ruc._responses_function_call_output_to_anthropic_block({"call_id": "c", "output": {"text": "café"}})["content"],
                [{"type": "text", "text": "encoded"}],
            )
        dumps_mock.assert_called_once_with(
            {"text": "café"},
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self.assertEqual(
            ruc._responses_function_call_output_to_anthropic_block(
                {"call_id": "c", "output": [{"type": "output_text", "text": "a"}, {"type": "output_text", "text": "b"}]}
            )["content"],
            [{"type": "text", "text": "ab"}],
        )
        with self.assertRaisesRegex(ValueError, "Responses function_call items must include a name"):
            ruc._responses_function_call_to_anthropic_block({"call_id": "c"})
        with self.assertRaisesRegex(ValueError, "Responses function_call items must include call_id"):
            ruc._responses_function_call_to_anthropic_block({"name": "Read"})
        with self.assertRaisesRegex(ValueError, "Responses function_call_output items must include call_id"):
            ruc._responses_function_call_output_to_anthropic_block({"output": "ok"})
        with self.assertRaises(ValueError) as context:
            ruc._responses_function_call_output_to_anthropic_block({"output": "ok"})
        self.assertEqual(str(context.exception), "Responses function_call_output items must include call_id")

        self.assertIsNone(ruc._responses_tool_to_anthropic({"type": "web_search"}))
        self.assertEqual(
            ruc._responses_tool_to_anthropic(
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read files",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ),
            {
                "name": "Read",
                "description": "Read files",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        )
        self.assertEqual(
            ruc._responses_tool_to_anthropic({"function": {"name": "Read"}}),
            {"name": "Read", "description": "", "input_schema": {"type": "object", "properties": {}}},
        )
        with self.assertRaisesRegex(ValueError, "Responses tools must include a name"):
            ruc._responses_tool_to_anthropic({"type": "function"})
        with self.assertRaises(ValueError) as context:
            ruc._responses_tool_to_anthropic({"type": "function"})
        self.assertEqual(str(context.exception), "Responses tools must include a name")

    def test_tool_choice_and_cache_helpers_cover_noops_and_existing_markers(self):
        self.assertIsNone(ruc._responses_tool_choice_to_anthropic(None))
        self.assertEqual(ruc._responses_tool_choice_to_anthropic("auto"), {"type": "auto"})
        self.assertEqual(ruc._responses_tool_choice_to_anthropic("required"), {"type": "any"})
        self.assertEqual(ruc._responses_tool_choice_to_anthropic("none"), {"type": "none"})
        self.assertEqual(ruc._responses_tool_choice_to_anthropic({"type": "auto"}), {"type": "auto"})
        self.assertEqual(ruc._responses_tool_choice_to_anthropic({"type": "none"}), {"type": "none"})
        self.assertEqual(ruc._responses_tool_choice_to_anthropic({"type": "required"}), {"type": "any"})
        self.assertEqual(ruc._responses_tool_choice_to_anthropic({"type": "function", "name": "Read"}), {"type": "tool", "name": "Read"})
        self.assertEqual(
            ruc._responses_tool_choice_to_anthropic({"type": "function", "function": {"name": "Read"}}),
            {"type": "tool", "name": "Read"},
        )
        self.assertIsNone(ruc._responses_tool_choice_to_anthropic({"type": "function", "name": "mcp__ide__executeCode"}))
        with self.assertRaisesRegex(ValueError, "Responses tool_choice type=function must include name"):
            ruc._responses_tool_choice_to_anthropic({"type": "function"})
        with self.assertRaisesRegex(ValueError, "Unsupported Responses tool_choice"):
            ruc._responses_tool_choice_to_anthropic({"type": "custom"})
        with self.assertRaises(ValueError) as context:
            ruc._responses_tool_choice_to_anthropic({"type": "function"})
        self.assertEqual(str(context.exception), "Responses tool_choice type=function must include name")
        with self.assertRaises(ValueError) as context:
            ruc._responses_tool_choice_to_anthropic({"type": "custom"})
        self.assertEqual(str(context.exception), "Unsupported Responses tool_choice value")

        self.assertFalse(ruc._responses_body_requests_prompt_cache({}))
        self.assertFalse(ruc._responses_body_requests_prompt_cache({"prompt_cache_key": 1, "promptCacheKey": " "}))
        self.assertTrue(ruc._responses_body_requests_prompt_cache({"promptCacheKey": "cache"}))

        block = {"type": "text", "cache_control": {"type": "persist"}}
        ruc._add_anthropic_cache_control(block)
        self.assertEqual(block["cache_control"], {"type": "persist"})
        self.assertIsNone(ruc._last_cacheable_anthropic_content_block(["bad", {"type": "unknown"}]))

        ruc._apply_responses_prompt_cache_to_anthropic_messages("bad")
        payload = {"tools": ["bad"], "system": [{"type": "text", "text": "sys"}], "messages": "bad"}
        ruc._apply_responses_prompt_cache_to_anthropic_messages(payload)
        self.assertEqual(payload["system"][0]["cache_control"], {"type": "ephemeral"})
        payload = {"system": [{"type": "unknown"}], "messages": [{"content": []}]}
        ruc._apply_responses_prompt_cache_to_anthropic_messages(payload)
        self.assertEqual(payload, {"system": [{"type": "unknown"}], "messages": [{"content": []}]})

        payload = {
            "tools": [{"name": "Read"}],
            "system": "sys",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "oldest"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "call", "name": "Read", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call", "content": []}]},
            ],
        }
        ruc._apply_responses_prompt_cache_to_anthropic_messages(payload)
        self.assertEqual(payload["tools"], [{"name": "Read", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(payload["system"], [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(payload["messages"][0]["content"], [{"type": "text", "text": "oldest"}])
        self.assertEqual(payload["messages"][1]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(payload["messages"][2]["content"][0]["cache_control"], {"type": "ephemeral"})


class ResponsesToAnthropicMessagesRequestTests(unittest.TestCase):
    def test_request_translation_rejects_non_dict_body_with_exact_message(self):
        with self.assertRaisesRegex(ValueError, "Responses body must be a dict"):
            ruc.responses_request_to_anthropic_messages("bad")

    def test_request_translation_covers_empty_inputs_skips_and_system_content_edges(self):
        out = ruc.responses_request_to_anthropic_messages(
            {
                "model": "claude-sonnet-4.6",
                "input": [
                    "skip",
                    {"type": "message", "role": "developer", "content": 3},
                    {"type": "message", "role": "developer", "content": [{"bad": "shape"}]},
                    {"type": "message", "role": "developer", "content": [{"text": ""}, {"text": "dev"}]},
                    {"type": "message", "role": "system", "content": ""},
                    {"type": "message", "role": "user", "content": ""},
                    {"type": "message", "role": "user", "content": None},
                    {"type": "message", "role": "assistant", "content": [{"type": "unknown"}]},
                    {"type": "compaction"},
                    {"type": "item_reference"},
                    {"type": "web_search_call"},
                ],
                "max_output_tokens": 12,
                "temperature": 0.5,
                "stop_sequences": ["STOP"],
                "stream": False,
            }
        )
        self.assertEqual(out["system"], "dev")
        self.assertEqual(out["messages"], [{"role": "user", "content": [{"type": "text", "text": ""}]}])
        self.assertEqual(out["max_tokens"], 12)
        self.assertEqual(out["temperature"], 0.5)
        self.assertEqual(out["stop_sequences"], ["STOP"])
        self.assertFalse(out["stream"])

        empty_string = ruc.responses_request_to_anthropic_messages({"model": "claude-sonnet-4.6", "input": ""})
        self.assertEqual(empty_string["messages"], [])
        no_input = ruc.responses_request_to_anthropic_messages({"model": "claude-sonnet-4.6"})
        self.assertEqual(no_input["messages"], [])
        non_string_instructions = ruc.responses_request_to_anthropic_messages(
            {"model": "claude-sonnet-4.6", "input": "hi", "instructions": 7}
        )
        self.assertNotIn("system", non_string_instructions)

    def test_request_translation_handles_non_list_tools_no_tool_choice_and_no_reasoning(self):
        out = ruc.responses_request_to_anthropic_messages(
            {
                "model": "claude-haiku-4.5",
                "input": "hi",
                "tools": {"type": "function", "name": "Read"},
                "tool_choice": {"type": "function", "name": "mcp__ide__executeCode"},
                "reasoning": "bad",
            }
        )
        self.assertEqual(out["messages"], [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        self.assertNotIn("tools", out)
        self.assertNotIn("tool_choice", out)
        self.assertNotIn("thinking", out)

        out = ruc.responses_request_to_anthropic_messages(
            {
                "model": "claude-sonnet-4.6",
                "input": "hi",
                "tools": [{"type": "image_generation", "name": "Draw"}],
                "parallel_tool_calls": False,
            }
        )
        self.assertNotIn("tools", out)
        self.assertNotIn("tool_choice", out)

        out = ruc.responses_request_to_anthropic_messages(
            {
                "model": "claude-sonnet-4.6",
                "input": "hi",
                "tools": [{"type": "function", "name": "Read"}],
                "tool_choice": "none",
                "parallel_tool_calls": False,
            }
        )
        self.assertEqual(out["tool_choice"], {"type": "none"})

        out = ruc.responses_request_to_anthropic_messages(
            {
                "model": "claude-sonnet-4.6",
                "input": "hi",
                "tools": [{"type": "function", "name": "Read"}],
                "tool_choice": {"type": "function", "name": "mcp__ide__executeCode"},
                "parallel_tool_calls": False,
            }
        )
        self.assertEqual(out["tools"], [{"name": "Read", "description": "", "input_schema": {"type": "object", "properties": {}}}])
        self.assertNotIn("tool_choice", out)

    def test_request_translation_raises_for_missing_custom_tool_name(self):
        with self.assertRaisesRegex(ValueError, "custom_tool_call items must include a name"):
            ruc.responses_request_to_anthropic_messages(
                {"model": "claude-sonnet-4.6", "input": [{"type": "custom_tool_call", "input": "x"}]}
            )
        with self.assertRaises(ValueError) as context:
            ruc.responses_request_to_anthropic_messages(
                {"model": "claude-sonnet-4.6", "input": [{"type": "custom_tool_call", "input": "x"}]}
            )
        self.assertEqual(str(context.exception), "Responses custom_tool_call items must include a name")


if __name__ == "__main__":
    unittest.main()
