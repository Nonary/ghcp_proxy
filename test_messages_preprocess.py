"""Tests for messages_preprocess.py."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

import messages_preprocess as mp
from messages_preprocess import (
    apply_prompt_cache_breakpoints,
    COMPACT_AUTO_CONTINUE,
    COMPACT_REQUEST,
    apply_adaptive_thinking,
    budget_tokens_to_effort,
    detect_compact_type,
    filter_assistant_thinking_placeholders,
    merge_tool_result_with_reminder,
    prepare_messages_passthrough_payload,
    sanitize_ide_tools,
    strip_cache_control_scope,
    strip_tool_reference_turn_boundary,
)


class StripCacheControlScopeTests(unittest.TestCase):
    def test_strips_scope_from_system_messages_and_tools(self):
        body = {
            "system": [
                {"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "scope": "session"}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_1",
                            "cache_control": {"type": "ephemeral", "scope": "turn"},
                            "content": [
                                {"type": "text", "text": "ok",
                                 "cache_control": {"type": "ephemeral", "scope": "session"}},
                            ],
                        }
                    ],
                }
            ],
            "tools": [
                {"name": "t", "cache_control": {"type": "ephemeral", "scope": "session"}}
            ],
        }
        strip_cache_control_scope(body)
        self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})
        tr = body["messages"][0]["content"][0]
        self.assertEqual(tr["cache_control"], {"type": "ephemeral"})
        self.assertEqual(tr["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(body["tools"][0]["cache_control"], {"type": "ephemeral"})

    def test_no_op_when_no_cache_control(self):
        body = {"system": "hello", "messages": [{"role": "user", "content": "hi"}]}
        snap = copy.deepcopy(body)
        strip_cache_control_scope(body)
        self.assertEqual(body, snap)

    def test_handles_empty_body(self):
        self.assertEqual(strip_cache_control_scope({}), {})

    def test_handles_non_dict_messages_and_non_dict_blocks(self):
        body = {
            "system": [{"type": "text", "text": "x"}, "not-a-block"],
            "messages": [
                "not-a-message",
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "ok", "cache_control": {"type": "ephemeral", "scope": "turn"}},
                        "not-a-block",
                    ],
                },
            ],
            "tools": ["not-a-tool"],
        }

        strip_cache_control_scope(body)

        self.assertEqual(body["messages"][1]["content"][0]["text"], "ok")
        self.assertEqual(body["messages"][1]["content"][0]["cache_control"], {"type": "ephemeral"})


class ApplyPromptCacheBreakpointsTests(unittest.TestCase):
    def _count_cache_controls(self, value):
        if isinstance(value, dict):
            return int(isinstance(value.get("cache_control"), dict)) + sum(
                self._count_cache_controls(v) for v in value.values()
            )
        if isinstance(value, list):
            return sum(self._count_cache_controls(v) for v in value)
        return 0

    def test_expands_existing_cache_intent_to_stable_breakpoints(self):
        body = {
            "system": "sys",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "first"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "third",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
            "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        }

        apply_prompt_cache_breakpoints(body)

        self.assertIsInstance(body["system"], list)
        self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(body["tools"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", body["messages"][0]["content"][0])
        self.assertEqual(body["messages"][1]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(body["messages"][2]["content"][0]["cache_control"], {"type": "ephemeral"})

    def test_replaces_excess_existing_cache_markers_with_at_most_four_breakpoints(self):
        body = {
            "system": [
                {"type": "text", "text": "sys-1", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "sys-2", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first", "cache_control": {"type": "ephemeral"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "second", "cache_control": {"type": "ephemeral"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "cache_control": {"type": "ephemeral"},
                            "content": [
                                {
                                    "type": "text",
                                    "text": "nested",
                                    "cache_control": {"type": "ephemeral"},
                                },
                            ],
                        },
                    ],
                },
            ],
            "tools": [
                {"name": "Read", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}},
            ],
        }

        apply_prompt_cache_breakpoints(body)

        self.assertEqual(self._count_cache_controls(body), 4)
        self.assertNotIn("cache_control", body["system"][0])
        self.assertEqual(body["system"][1]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(body["tools"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", body["messages"][0]["content"][0])
        self.assertEqual(body["messages"][1]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(body["messages"][2]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", body["messages"][2]["content"][0]["content"][0])

    def test_nested_cache_marker_is_enough_to_trigger_stable_breakpoints(self):
        body = {
            "system": [{"type": "text", "text": "sys"}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "nested",
                                    "cache_control": {"type": "ephemeral"},
                                },
                            ],
                        },
                    ],
                },
            ],
        }

        apply_prompt_cache_breakpoints(body)

        self.assertEqual(self._count_cache_controls(body), 2)
        self.assertEqual(body["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(body["messages"][0]["content"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", body["messages"][0]["content"][0]["content"][0])

    def test_noop_without_cache_intent(self):
        body = {"system": "sys", "messages": [{"role": "user", "content": "hi"}]}
        snap = copy.deepcopy(body)
        apply_prompt_cache_breakpoints(body)
        self.assertEqual(body, snap)


class FilterAssistantThinkingTests(unittest.TestCase):
    def test_drops_thinking_placeholder_text(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Thinking...", "signature": "sig"},
                        {"type": "thinking", "thinking": "real", "signature": "@bad"},
                        {"type": "thinking", "thinking": "real", "signature": "good-sig"},
                        {"type": "text", "text": "answer"},
                    ],
                }
            ]
        }
        filter_assistant_thinking_placeholders(body)
        kinds = [b["type"] for b in body["messages"][0]["content"]]
        self.assertEqual(kinds, ["thinking", "text"])
        kept = body["messages"][0]["content"][0]
        self.assertEqual(kept["signature"], "good-sig")

    def test_user_messages_untouched(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "thinking", "thinking": "Thinking..."}]}
            ]
        }
        filter_assistant_thinking_placeholders(body)
        self.assertEqual(len(body["messages"][0]["content"]), 1)

    def test_noops_for_malformed_bodies_and_content(self):
        self.assertEqual(filter_assistant_thinking_placeholders("bad"), "bad")
        body = {"messages": ["bad", {"role": "assistant", "content": "not-a-list"}]}

        filter_assistant_thinking_placeholders(body)

        self.assertEqual(body["messages"][1]["content"], "not-a-list")

    def test_placeholder_predicate_negative_and_missing_fields(self):
        self.assertFalse(mp._is_placeholder_thinking({"type": "text", "text": "not thinking"}))
        self.assertTrue(mp._is_placeholder_thinking({"type": "thinking", "thinking": "", "signature": "sig"}))
        self.assertTrue(mp._is_placeholder_thinking({"type": "thinking", "thinking": "real"}))
        self.assertTrue(mp._is_placeholder_thinking({"type": "thinking", "thinking": "real", "signature": 123}))

    def test_filter_continues_after_malformed_or_skipped_messages(self):
        body = {
            "messages": [
                "bad",
                {"role": "user", "content": [{"type": "thinking", "thinking": "Thinking..."}]},
                {"role": "assistant", "content": "not-a-list"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Thinking...", "signature": "sig"},
                        {"type": "text", "text": "answer"},
                    ],
                },
            ]
        }

        filter_assistant_thinking_placeholders(body)

        self.assertEqual(body["messages"][3]["content"], [{"type": "text", "text": "answer"}])


class MergeToolResultWithReminderTests(unittest.TestCase):
    def test_folds_trailing_reminder_into_last_tool_result(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "a",
                         "content": [{"type": "text", "text": "out-a"}]},
                        {"type": "tool_result", "tool_use_id": "b",
                         "content": [{"type": "text", "text": "out-b"}]},
                        {"type": "text", "text": "<system-reminder>be nice</system-reminder>"},
                    ],
                }
            ]
        }
        merge_tool_result_with_reminder(body)
        content = body["messages"][0]["content"]
        self.assertEqual(len(content), 2)  # reminder folded in
        last_tr = content[-1]
        self.assertEqual(last_tr["tool_use_id"], "b")
        texts = [c["text"] for c in last_tr["content"] if c.get("type") == "text"]
        self.assertIn("<system-reminder>be nice</system-reminder>", texts)

    def test_string_tool_result_content_concatenated(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "a", "content": "hello"},
                        {"type": "text", "text": "<system-reminder>R</system-reminder>"},
                    ],
                }
            ]
        }
        merge_tool_result_with_reminder(body)
        tr = body["messages"][0]["content"][0]
        self.assertEqual(tr["content"], "hello\n\n<system-reminder>R</system-reminder>")
        self.assertEqual(len(body["messages"][0]["content"]), 1)

    def test_preserves_cache_control_from_merged_text_on_tool_result(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "a", "content": "hello"},
                        {
                            "type": "text",
                            "text": "<system-reminder>R</system-reminder>",
                            "cache_control": {"type": "ephemeral", "scope": "x"},
                        },
                    ],
                }
            ]
        }
        cleaned = prepare_messages_passthrough_payload(body, model_supports_adaptive=False)

        tr = cleaned["messages"][0]["content"][0]
        self.assertEqual(tr["cache_control"], {"type": "ephemeral"})
        self.assertEqual(len(cleaned["messages"][0]["content"]), 1)

    def test_no_tool_result_no_change(self):
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "<system-reminder>R</system-reminder>"}]}
            ]
        }
        snap = copy.deepcopy(body)
        merge_tool_result_with_reminder(body)
        self.assertEqual(body, snap)

    def test_direct_merge_helpers_cover_reference_and_attachment_shapes(self):
        self.assertFalse(mp._has_tool_reference({"content": "text"}))
        self.assertTrue(mp._is_text_block({"type": "text", "text": "x"}))
        self.assertFalse(mp._is_text_block("bad"))
        self.assertFalse(mp._is_text_block({"type": "image", "text": "x"}))
        self.assertFalse(mp._is_text_block({"type": "text", "text": 3}))
        self.assertFalse(mp._is_attachment_block("bad"))
        self.assertTrue(mp._is_attachment_block({"type": "image"}))
        self.assertTrue(mp._is_attachment_block({"type": "document"}))

        with_tool_ref = {
            "content": [
                {"type": "text", "text": "existing"},
                {"type": "tool_reference", "id": "ref"},
            ]
        }
        mp._merge_content_with_texts(with_tool_ref, [{"type": "text", "text": "new"}])
        self.assertEqual(len(with_tool_ref["content"]), 2)

        no_content = {}
        mp._merge_content_with_texts(no_content, [{"type": "text", "text": "new"}])
        self.assertEqual(no_content["content"], [{"type": "text", "text": "new"}])

        string_texts = {"content": "base"}
        mp._merge_content_with_texts(string_texts, [{"type": "text", "text": "one"}, {"type": "text"}])
        self.assertEqual(string_texts["content"], "base\n\none\n\n")

        list_texts = {"content": []}
        mp._merge_content_with_texts(list_texts, [{"type": "text"}])
        self.assertEqual(list_texts["content"], [{"type": "text", "text": ""}])

        no_text_field = {}
        mp._merge_content_with_texts(no_text_field, [{"type": "text"}])
        self.assertEqual(no_text_field["content"], [{"type": "text", "text": ""}])

        string_attachment = {"content": "plain"}
        mp._merge_content_with_attachments(string_attachment, [{"type": "image", "source": {}}])
        self.assertEqual(string_attachment["content"][0], {"type": "text", "text": "plain"})

        list_attachment = {"content": [{"type": "text", "text": "plain"}]}
        mp._merge_content_with_attachments(list_attachment, [{"type": "document", "source": {}}])
        self.assertEqual(list_attachment["content"][-1]["type"], "document")

        empty_attachment = {}
        mp._merge_content_with_attachments(empty_attachment, [{"type": "image", "source": {}}])
        self.assertEqual(empty_attachment["content"][0]["type"], "image")

    def test_mergeable_tool_result_indices_and_pdf_detection(self):
        tool_results = [
            "bad",
            {"is_error": True},
            {"content": [{"type": "tool_reference"}]},
            {"content": "PDF file read: report.pdf"},
            {"content": [{"type": "text", "text": "PDF file read: report.pdf"}]},
            {"content": [{"type": "document"}]},
            {"content": [{"type": "image"}]},
            {"content": [{"type": "text", "text": "plain"}]},
        ]

        self.assertEqual(mp._get_mergeable_tool_result_indices(tool_results), [3, 4, 5, 6, 7])
        self.assertTrue(mp._starts_with_pdf_file_read(tool_results[3]))
        self.assertTrue(mp._starts_with_pdf_file_read(tool_results[4]))
        self.assertFalse(mp._starts_with_pdf_file_read(tool_results[5]))
        self.assertFalse(mp._starts_with_pdf_file_read(tool_results[6]))
        self.assertFalse(mp._starts_with_pdf_file_read({}))
        self.assertFalse(
            mp._starts_with_pdf_file_read(
                {
                    "content": [
                        {"type": "text", "text": "PDF file read: report.pdf"},
                        {"type": "document", "source": {}},
                    ]
                }
            )
        )
        self.assertFalse(mp._starts_with_pdf_file_read({"content": ["not-a-block"]}))
        self.assertFalse(mp._starts_with_pdf_file_read({"content": [{"type": "text", "text": 123}]}))

    def test_assign_and_merge_attachment_helpers_cover_pairing_and_fallbacks(self):
        target = {}
        mp._assign_attachments_to_tool_results(target, [], tool_result_indices=[0])
        self.assertEqual(target, {})

        mp._assign_attachments_to_tool_results(
            target,
            [(0, {"type": "image", "id": "a"}), (1, {"type": "image", "id": "b"})],
            tool_result_indices=[0, 1],
        )
        self.assertEqual(target[0][0]["id"], "a")
        self.assertEqual(target[1][0]["id"], "b")

        mp._assign_attachments_to_tool_results(
            target,
            [(2, {"type": "image", "id": "c"})],
            tool_result_indices=[],
            fallback_indices=[],
        )
        self.assertNotIn(2, target)

        mp._assign_attachments_to_tool_results(
            target,
            [(3, {"type": "document", "id": "d"}), (4, {"type": "image", "id": "e"})],
            tool_result_indices=[0],
        )
        self.assertEqual([a["id"] for a in target[0][-2:]], ["d", "e"])

        fallback_target = {}
        mp._assign_attachments_to_tool_results(
            fallback_target,
            [(5, {"type": "image", "id": "fresh"})],
            tool_result_indices=[0, 1],
            fallback_indices=[7],
        )
        self.assertEqual(fallback_target, {7: [{"type": "image", "id": "fresh"}]})

        tool_results = [{"content": []}, {"content": []}]
        mp._merge_attachments_into_tool_results(tool_results, {})
        self.assertEqual(tool_results, [{"content": []}, {"content": []}])

        mp._merge_attachments_into_tool_results(tool_results, {1: [{"type": "image", "id": "img"}]})
        self.assertEqual(tool_results[1]["content"], [{"type": "image", "id": "img"}])

    def test_merge_user_message_content_unknown_shapes_and_pdf_document_pairing(self):
        self.assertIsNone(mp._merge_user_message_content(["bad"]))
        self.assertIsNone(mp._merge_user_message_content([{"type": "unknown"}]))
        self.assertIsNone(mp._merge_user_message_content([{"type": "tool_result", "content": "only"}]))

        merged = mp._merge_user_message_content(
            [
                {"type": "tool_result", "tool_use_id": "a", "content": "A"},
                {"type": "tool_result", "tool_use_id": "b", "content": "B"},
                {"type": "text", "text": "text-a"},
                {"type": "text", "text": "text-b"},
            ]
        )
        self.assertEqual(merged[0]["content"], "A\n\ntext-a")
        self.assertEqual(merged[1]["content"], "B\n\ntext-b")

        merged = mp._merge_user_message_content(
            [
                {"type": "tool_result", "tool_use_id": "pdf", "content": "PDF file read: report.pdf"},
                {"type": "tool_result", "tool_use_id": "other", "content": []},
                {"type": "document", "source": {"type": "base64", "data": "doc"}},
                {"type": "image", "source": {"type": "base64", "data": "img"}},
            ]
        )
        self.assertEqual(merged[0]["content"][1]["type"], "document")
        self.assertEqual(merged[1]["content"][0]["type"], "image")

        merged = mp._merge_user_message_content(
            [
                {"type": "tool_result", "tool_use_id": "plain", "content": []},
                {"type": "image", "source": {"type": "base64", "data": "img"}},
            ]
        )
        self.assertEqual(merged[0]["content"][0]["type"], "image")

    def test_pdf_document_pairing_keeps_remaining_attachments_on_pdf_result(self):
        merged = mp._merge_user_message_content(
            [
                {"type": "tool_result", "tool_use_id": "pdf", "content": "PDF file read: report.pdf"},
                {"type": "document", "source": {"type": "base64", "data": "doc"}},
                {"type": "image", "source": {"type": "base64", "data": "img"}},
            ]
        )

        self.assertEqual([b["type"] for b in merged[0]["content"]], ["text", "document", "image"])

    def test_pdf_document_pairing_targets_pdf_result_not_first_result(self):
        merged = mp._merge_user_message_content(
            [
                {"type": "tool_result", "tool_use_id": "plain", "content": []},
                {"type": "tool_result", "tool_use_id": "pdf", "content": "PDF file read: report.pdf"},
                {"type": "document", "source": {"type": "base64", "data": "doc"}},
                {"type": "image", "source": {"type": "base64", "data": "img"}},
            ]
        )

        self.assertEqual(merged[0]["tool_use_id"], "plain")
        self.assertEqual([b["type"] for b in merged[0]["content"]], ["image"])
        self.assertEqual(merged[1]["tool_use_id"], "pdf")
        self.assertEqual([b["type"] for b in merged[1]["content"]], ["text", "document"])

    def test_equal_count_attachments_pair_with_each_tool_result(self):
        merged = mp._merge_user_message_content(
            [
                {"type": "tool_result", "tool_use_id": "a", "content": []},
                {"type": "tool_result", "tool_use_id": "b", "content": []},
                {"type": "image", "source": {"type": "base64", "data": "img-a"}},
                {"type": "image", "source": {"type": "base64", "data": "img-b"}},
            ]
        )

        self.assertEqual(merged[0]["content"], [{"type": "image", "source": {"type": "base64", "data": "img-a"}}])
        self.assertEqual(merged[1]["content"], [{"type": "image", "source": {"type": "base64", "data": "img-b"}}])

    def test_extra_pdf_documents_fall_back_to_remaining_tool_results(self):
        merged = mp._merge_user_message_content(
            [
                {"type": "tool_result", "tool_use_id": "plain", "content": []},
                {"type": "tool_result", "tool_use_id": "pdf", "content": "PDF file read: report.pdf"},
                {"type": "document", "source": {"type": "base64", "data": "doc-a"}},
                {"type": "document", "source": {"type": "base64", "data": "doc-b"}},
            ]
        )

        self.assertEqual([b["source"]["data"] for b in merged[0]["content"]], ["doc-b"])
        self.assertEqual([b["type"] for b in merged[1]["content"]], ["text", "document"])
        self.assertEqual(merged[1]["content"][1]["source"]["data"], "doc-a")

    def test_non_list_messages_noop(self):
        body = {"messages": "not-a-list"}

        merge_tool_result_with_reminder(body)

        self.assertEqual(body, {"messages": "not-a-list"})

    def test_merge_tool_result_with_reminder_skips_malformed_and_compact_last_message(self):
        self.assertEqual(merge_tool_result_with_reminder("bad"), "bad")
        body = {
            "messages": [
                "bad",
                {"role": "assistant", "content": [{"type": "text", "text": "skip"}]},
                {"role": "user", "content": "skip"},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "a", "content": "A"},
                        {"type": "text", "text": "compact prompt"},
                    ],
                },
            ]
        }

        merge_tool_result_with_reminder(body, skip_last_message=True)

        self.assertEqual(len(body["messages"][-1]["content"]), 2)


class StripToolReferenceTurnBoundaryTests(unittest.TestCase):
    def test_strips_only_boundary_when_tool_reference_is_present(self):
        body = {
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "skip"}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": [{"type": "tool_reference", "id": "ref"}],
                        },
                        {"type": "text", "text": " Tool loaded. "},
                        {"type": "text", "text": "keep"},
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Tool loaded."}]},
            ]
        }

        strip_tool_reference_turn_boundary(body)

        self.assertEqual([b.get("text") for b in body["messages"][1]["content"] if b.get("type") == "text"], ["keep"])
        self.assertEqual(body["messages"][2]["content"], [{"type": "text", "text": "Tool loaded."}])

    def test_noops_for_malformed_bodies(self):
        self.assertEqual(strip_tool_reference_turn_boundary("bad"), "bad")
        body = {"messages": ["bad", {"role": "user", "content": "not-a-list"}]}

        strip_tool_reference_turn_boundary(body)

        self.assertEqual(body["messages"][1]["content"], "not-a-list")

    def test_continues_after_non_list_content_before_tool_reference_turn(self):
        body = {
            "messages": [
                {"role": "user", "content": "not-a-list"},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": [{"type": "tool_reference", "id": "ref"}]},
                        {"type": "text", "text": "Tool loaded."},
                        {"type": "text", "text": "keep"},
                    ],
                },
            ]
        }

        strip_tool_reference_turn_boundary(body)

        self.assertEqual([b.get("text") for b in body["messages"][1]["content"] if b.get("type") == "text"], ["keep"])

    def test_does_not_strip_boundary_for_non_tool_result_reference_carrier(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "content": [{"type": "tool_reference", "id": "ref"}]},
                        {"type": "text", "text": "Tool loaded."},
                    ],
                }
            ]
        }

        strip_tool_reference_turn_boundary(body)

        texts = [b.get("text") for b in body["messages"][0]["content"] if b.get("type") == "text"]
        self.assertEqual(texts, [None, "Tool loaded."])


class SanitizeIdeToolsTests(unittest.TestCase):
    def test_drops_execute_code_and_rewrites_diagnostics(self):
        body = {
            "tools": [
                {"name": "mcp__ide__executeCode", "description": "old"},
                {"name": "mcp__ide__getDiagnostics", "description": "old"},
                {"name": "Bash", "description": "shell"},
            ]
        }
        sanitize_ide_tools(body)
        names = [t["name"] for t in body["tools"]]
        self.assertEqual(names, ["mcp__ide__getDiagnostics", "Bash"])
        diag = next(t for t in body["tools"] if t["name"] == "mcp__ide__getDiagnostics")
        self.assertIn("diagnostics from VS Code", diag["description"])

    def test_drops_execute_code_even_when_defer_loading(self):
        body = {"tools": [{"name": "mcp__ide__executeCode", "defer_loading": True}]}
        sanitize_ide_tools(body)
        self.assertEqual(body["tools"], [])

    def test_no_tools_noop(self):
        self.assertEqual(sanitize_ide_tools({}), {})
        body = {"tools": "not-a-list"}
        self.assertEqual(sanitize_ide_tools(body), {"tools": "not-a-list"})

    def test_keeps_non_dict_tools_and_handles_non_dict_body(self):
        self.assertEqual(sanitize_ide_tools("bad"), "bad")
        body = {"tools": ["not-a-tool", {"name": "mcp__IDE__executeCode"}, {"name": 3}]}

        sanitize_ide_tools(body)

        self.assertEqual(body["tools"], ["not-a-tool", {"name": 3}])


class ApplyAdaptiveThinkingTests(unittest.TestCase):
    def test_converts_enabled_to_adaptive(self):
        body = {
            "model": "claude-sonnet-4.5",
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        }
        apply_adaptive_thinking(body, supports_adaptive=True)
        # When the client provided ``thinking``, do not add ``display`` (TS parity).
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertEqual(body["output_config"], {"effort": "medium"})

    def test_disabled_when_not_supported(self):
        body = {
            "model": "claude-sonnet-4.5",
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        }
        snap = copy.deepcopy(body)
        apply_adaptive_thinking(body, supports_adaptive=False)
        self.assertEqual(body, snap)

    def test_disabled_when_tool_choice_forces_tool(self):
        body = {
            "model": "claude-sonnet-4.5",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tool_choice": {"type": "any"},
        }
        snap = copy.deepcopy(body)
        apply_adaptive_thinking(body, supports_adaptive=True)
        self.assertEqual(body, snap)

    def test_disabled_when_tool_choice_targets_specific_tool(self):
        body = {
            "model": "claude-sonnet-4.5",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "tool", "name": "Bash"},
        }
        snap = copy.deepcopy(body)

        apply_adaptive_thinking(body, supports_adaptive=True)

        self.assertEqual(body, snap)

    def test_defaults_to_high_with_summarized_display_and_output_config(self):
        body = {"model": "claude-sonnet-4.5", "messages": [{"role": "user", "content": "hi"}]}

        apply_adaptive_thinking(body, supports_adaptive=True)

        self.assertEqual(body["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(body["output_config"], {"effort": "high"})

    def test_messages_only_payload_is_not_treated_as_empty(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}

        apply_adaptive_thinking(body, supports_adaptive=True)

        self.assertEqual(body["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(body["output_config"], {"effort": "high"})

    def test_preserves_and_maps_incoming_output_config_effort(self):
        body = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "max"},
        }

        apply_adaptive_thinking(body, supports_adaptive=True)

        self.assertEqual(body["output_config"]["effort"], "xhigh")

    def test_minimal_effort_is_low_and_whitelist_clamps_to_last_allowed(self):
        body = {
            "model": "claude-sonnet-4.5",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "minimal"},
        }

        apply_adaptive_thinking(body, supports_adaptive=True, reasoning_efforts=["low", "", 3, "medium"])

        self.assertEqual(body["output_config"]["effort"], "low")

        body = {
            "model": "claude-sonnet-4.5",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "ultra"},
        }

        apply_adaptive_thinking(body, supports_adaptive=True, reasoning_efforts=["low", "medium"])

        self.assertEqual(body["output_config"]["effort"], "medium")

        body = {
            "model": "claude-sonnet-4.5",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "ultra"},
        }

        apply_adaptive_thinking(body, supports_adaptive=True, reasoning_efforts=["low", "medium", "high"])

        self.assertEqual(body["output_config"]["effort"], "high")

    def test_opus_47_and_empty_payload_guards(self):
        empty = {}
        apply_adaptive_thinking(empty, supports_adaptive=True)
        self.assertEqual(empty, {})

        body = {"model": "anthropic/claude-opus-4.7", "thinking": {"type": "enabled"}}
        apply_adaptive_thinking(body, supports_adaptive=True)
        self.assertEqual(body["thinking"], {"type": "adaptive", "display": "summarized"})

    def test_empty_or_non_string_output_config_effort_uses_default(self):
        for effort in ("", 3):
            with self.subTest(effort=effort):
                body = {
                    "model": "claude-sonnet-4.5",
                    "messages": [{"role": "user", "content": "hi"}],
                    "output_config": {"effort": effort},
                }

                apply_adaptive_thinking(body, supports_adaptive=True)

                self.assertEqual(body["output_config"]["effort"], "high")

    def test_mapped_minimal_effort_is_normalized_to_low(self):
        body = {"model": "claude-sonnet-4.5", "messages": [{"role": "user", "content": "hi"}]}

        with mock.patch("messages_preprocess.map_effort_for_model", return_value="minimal"):
            apply_adaptive_thinking(body, supports_adaptive=True)

        self.assertEqual(body["output_config"]["effort"], "low")

    def test_mapped_none_effort_is_normalized_to_low(self):
        body = {"model": "claude-sonnet-4.5", "messages": [{"role": "user", "content": "hi"}]}

        with mock.patch("messages_preprocess.map_effort_for_model", return_value="none"):
            apply_adaptive_thinking(body, supports_adaptive=True)

        self.assertEqual(body["output_config"]["effort"], "low")


class DetectCompactTypeTests(unittest.TestCase):
    def test_detects_summary_auto_continue_and_system_forms(self):
        summary_text = "\n".join(
            [
                "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.",
                "Your task is to create a detailed summary of the conversation so far",
                "Pending Tasks:",
            ]
        )
        self.assertEqual(
            detect_compact_type({"messages": [{"role": "user", "content": [{"type": "text", "text": summary_text}]}]}),
            COMPACT_REQUEST,
        )
        self.assertEqual(
            detect_compact_type(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.",
                        }
                    ]
                }
            ),
            COMPACT_AUTO_CONTINUE,
        )
        self.assertEqual(
            detect_compact_type(
                {"system": "You are a helpful AI assistant tasked with summarizing conversations now"}
            ),
            COMPACT_REQUEST,
        )
        self.assertEqual(
            detect_compact_type(
                {"system": [{"type": "text", "text": "You are a helpful AI assistant tasked with summarizing conversations now"}]}
            ),
            COMPACT_REQUEST,
        )

    def test_non_compact_and_candidate_text_filters(self):
        self.assertEqual(detect_compact_type("bad"), 0)
        self.assertEqual(detect_compact_type({"system": "ordinary"}), 0)
        self.assertEqual(detect_compact_type({"system": ["bad", {"type": "text", "text": 3}]}), 0)
        self.assertEqual(detect_compact_type({"messages": [{"role": "assistant", "content": "not user"}]}), 0)
        self.assertEqual(detect_compact_type({"messages": [{"role": "user", "content": 3}]}), 0)
        self.assertEqual(
            mp._compact_candidate_text(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<system-reminder>skip</system-reminder>"},
                        "bad",
                        {"type": "image"},
                        {"type": "text", "text": 3},
                        {"type": "text", "text": "keep"},
                    ],
                }
            ),
            "keep",
        )
        self.assertEqual(
            mp._compact_candidate_text(
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
            ),
            "a\n\nb",
        )
        self.assertEqual(mp._compact_candidate_text({"role": "user", "content": 3}), "")
        self.assertEqual(mp._compact_candidate_text("bad"), "")
        self.assertEqual(
            detect_compact_type(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Pending Tasks:\nCurrent Work:",
                        }
                    ]
                }
            ),
            0,
        )
        self.assertEqual(
            detect_compact_type(
                {
                    "system": [
                        "bad",
                        {"type": "text", "text": "You are a helpful AI assistant tasked with summarizing conversations now"},
                    ]
                }
            ),
            COMPACT_REQUEST,
        )
        self.assertEqual(
            detect_compact_type(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "\n".join(
                                [
                                    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.",
                                    "Your task is to create a detailed summary of the conversation so far",
                                ]
                            ),
                        }
                    ]
                }
            ),
            0,
        )


class BudgetTokensToEffortTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(budget_tokens_to_effort(0), "low")
        self.assertEqual(budget_tokens_to_effort(500), "low")
        self.assertEqual(budget_tokens_to_effort(1024), "low")
        self.assertEqual(budget_tokens_to_effort(1025), "medium")
        self.assertEqual(budget_tokens_to_effort(2048), "medium")
        self.assertEqual(budget_tokens_to_effort(8192), "medium")
        self.assertEqual(budget_tokens_to_effort(8193), "high")
        self.assertEqual(budget_tokens_to_effort(16000), "high")
        self.assertEqual(budget_tokens_to_effort(32000), "high")
        self.assertEqual(budget_tokens_to_effort(None), "low")


class PrepareMessagesPassthroughPayloadTests(unittest.TestCase):
    def test_returns_deep_copy_and_runs_pipeline(self):
        body = {
            "model": "claude-sonnet-4.5",
            "thinking": {"type": "enabled", "budget_tokens": 9000},
            "system": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "scope": "x"}},
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "Thinking..."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "a",
                         "content": [{"type": "text", "text": "out"}]},
                        {"type": "text", "text": "<system-reminder>R</system-reminder>"},
                    ],
                },
            ],
            "tools": [{"name": "mcp__ide__executeCode"}],
        }
        snap = copy.deepcopy(body)
        cleaned = prepare_messages_passthrough_payload(body, model_supports_adaptive=True)
        # Original untouched
        self.assertEqual(body, snap)
        # cleaned variants
        self.assertEqual(cleaned["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(cleaned["messages"][0]["content"], [])
        # reminder folded into the tool_result
        last_user = cleaned["messages"][1]["content"]
        self.assertEqual(len(last_user), 1)
        self.assertEqual(last_user[0]["type"], "tool_result")
        # ide executeCode dropped
        self.assertEqual(cleaned["tools"], [])
        # adaptive thinking applied (client supplied ``thinking`` so no ``display``)
        self.assertEqual(cleaned["thinking"], {"type": "adaptive"})
        self.assertEqual(cleaned["output_config"], {"effort": "high"})

    def test_empty_body(self):
        self.assertEqual(
            prepare_messages_passthrough_payload({}, model_supports_adaptive=True),
            {},
        )

    def test_compact_payload_skips_only_last_message_merge(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "a", "content": "A"},
                        {"type": "text", "text": "fold"},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "b", "content": "B"},
                        {"type": "text", "text": "compact prompt"},
                    ],
                },
            ]
        }

        cleaned = prepare_messages_passthrough_payload(
            body,
            model_supports_adaptive=False,
            is_compact=True,
        )

        self.assertEqual(cleaned["messages"][0]["content"], [{"type": "tool_result", "tool_use_id": "a", "content": "A\n\nfold"}])
        self.assertEqual(len(cleaned["messages"][1]["content"]), 2)

    def test_passes_reasoning_efforts_into_adaptive_thinking(self):
        cleaned = prepare_messages_passthrough_payload(
            {
                "model": "claude-sonnet-4.5",
                "messages": [{"role": "user", "content": "hi"}],
                "output_config": {"effort": "ultra"},
            },
            model_supports_adaptive=True,
            reasoning_efforts=["low", "medium", "high"],
        )

        self.assertEqual(cleaned["output_config"]["effort"], "high")

    def test_pipeline_passes_cleaned_payload_to_tool_reference_strip(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": [{"type": "tool_reference", "id": "ref"}]},
                        {"type": "text", "text": "Tool loaded."},
                    ],
                }
            ]
        }

        with mock.patch(
            "messages_preprocess.strip_tool_reference_turn_boundary",
            wraps=strip_tool_reference_turn_boundary,
        ) as strip_mock:
            cleaned = prepare_messages_passthrough_payload(body, model_supports_adaptive=False)

        self.assertIs(strip_mock.call_args.args[0], cleaned)
        self.assertIsNot(strip_mock.call_args.args[0], body)

    def test_non_dict_and_temperature_clamp_paths(self):
        self.assertEqual(prepare_messages_passthrough_payload("bad", model_supports_adaptive=True), "bad")

        high = {"temperature": 2, "messages": []}
        low = {"temperature": -1, "messages": []}
        ok = {"temperature": 0.5, "messages": []}

        prepare_messages_passthrough_payload(high, model_supports_adaptive=False)
        prepare_messages_passthrough_payload(low, model_supports_adaptive=False)
        prepare_messages_passthrough_payload(ok, model_supports_adaptive=False)
        mp.clamp_temperature_for_claude("bad")

        self.assertEqual(high["temperature"], 2)
        self.assertEqual(low["temperature"], -1)
        self.assertEqual(ok["temperature"], 0.5)
        self.assertEqual(prepare_messages_passthrough_payload(high, model_supports_adaptive=False)["temperature"], 1)
        self.assertEqual(prepare_messages_passthrough_payload(low, model_supports_adaptive=False)["temperature"], 0)

        direct = {"temperature": 0.5}
        mp.clamp_temperature_for_claude(direct)
        self.assertEqual(direct["temperature"], 0.5)


if __name__ == "__main__":
    unittest.main()
