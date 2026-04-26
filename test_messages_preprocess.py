"""Tests for messages_preprocess.py."""

from __future__ import annotations

import copy
import unittest

from messages_preprocess import (
    apply_prompt_cache_breakpoints,
    apply_adaptive_thinking,
    budget_tokens_to_effort,
    filter_assistant_thinking_placeholders,
    merge_tool_result_with_reminder,
    prepare_messages_passthrough_payload,
    sanitize_ide_tools,
    strip_cache_control_scope,
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


class BudgetTokensToEffortTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(budget_tokens_to_effort(500), "low")
        self.assertEqual(budget_tokens_to_effort(1024), "low")
        self.assertEqual(budget_tokens_to_effort(2048), "medium")
        self.assertEqual(budget_tokens_to_effort(8192), "medium")
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


if __name__ == "__main__":
    unittest.main()
