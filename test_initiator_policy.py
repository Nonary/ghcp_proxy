import unittest
from datetime import datetime, timezone
from unittest import mock

import initiator_policy
import proxy


class InitiatorPolicyTests(unittest.TestCase):
    def setUp(self):
        proxy.set_initiator_policy(initiator_policy.InitiatorPolicy())

    def test_safeguard_trigger_callback_records_reason(self):
        recorded = []
        policy = initiator_policy.InitiatorPolicy(on_safeguard_triggered=recorded.append)
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=start)
        policy.note_request_finished("req-1", finished_at=start.replace(second=4))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=5)):
            initiator = policy.resolve_initiator("user", "gpt-5.4", request_id="req-2")

        self.assertEqual(initiator, "agent")
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["trigger_reason"], "cooldown")
        self.assertEqual(recorded[0]["resolved_initiator"], "agent")
        self.assertEqual(recorded[0]["request_id"], "req-2")

    def test_forced_agent_responses_requests_stay_agent(self):
        from types import SimpleNamespace
        import format_translation
        import usage_tracking

        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "hello",
        }

        headers = format_translation.build_responses_headers_for_request(
            request,
            body,
            "test-key",
            force_initiator="agent",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(body["input"], "hello")

    def test_haiku_requests_are_always_agent(self):
        from types import SimpleNamespace
        import format_translation
        import usage_tracking

        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "claude-haiku-4.5",
            "input": "hello",
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_codex_bootstrap_mini_does_not_activate_safeguard(self):
        policy = initiator_policy.InitiatorPolicy()
        bootstrap_input = [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "developer instructions"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "<environment_context>\n  <cwd>D:\\sources\\ghcp_proxy</cwd>\n</environment_context>",
                    }
                ],
            },
        ]
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start):
            _normalized, initiator = policy.resolve_responses_input(
                bootstrap_input,
                "gpt-5.4-mini",
                request_id="bootstrap-1",
            )

        self.assertEqual(initiator, "agent")

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=1)):
            self.assertEqual(
                policy.resolve_initiator("user", "gpt-5.4", request_id="user-1"),
                "user",
            )

    def test_responses_environment_context_only_stays_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "developer instructions"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "# AGENTS.md instructions\n\n"
                            "<INSTRUCTIONS>\nUse PowerShell.\n</INSTRUCTIONS>\n"
                            "<environment_context>\n"
                            "  <cwd>D:\\sources\\ghcp_proxy</cwd>\n"
                            "  <shell>powershell</shell>\n"
                            "</environment_context>"
                        ),
                    }
                ],
            },
        ]

        normalized_input, initiator = policy.resolve_responses_input(input_items, "gpt-5.5")

        self.assertIs(normalized_input, input_items)
        self.assertEqual(initiator, "agent")

    def test_responses_environment_context_with_trailing_prompt_is_user(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "developer instructions"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "# AGENTS.md instructions\n\n"
                            "<INSTRUCTIONS>\nUse PowerShell.\n</INSTRUCTIONS>\n"
                            "<environment_context>\n"
                            "  <cwd>D:\\sources\\sunshine</cwd>\n"
                            "  <shell>powershell</shell>\n"
                            "</environment_context>\n\n"
                            "It appears our simple web server submodule is not pushed to remote."
                        ),
                    }
                ],
            },
        ]
        verdict = {}

        normalized_input, initiator = policy.resolve_responses_input(
            input_items,
            "gpt-5.5",
            verdict_sink=verdict,
        )

        self.assertIs(normalized_input, input_items)
        self.assertEqual(initiator, "user")
        self.assertEqual(verdict["candidate_initiator"], "user")

    def test_responses_environment_context_with_trailing_agent_marker_stays_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {
                "type": "message",
                "role": "user",
                "content": (
                    "<environment_context>\n"
                    "  <cwd>D:\\sources\\ghcp_proxy</cwd>\n"
                    "</environment_context>\n\n"
                    "_ summarize the current branch"
                ),
            },
        ]

        normalized_input, initiator = policy.resolve_responses_input(input_items, "gpt-5.5")

        self.assertIs(normalized_input, input_items)
        self.assertEqual(initiator, "agent")
        self.assertTrue(input_items[-1]["content"].endswith("summarize the current branch"))
        self.assertNotIn("\n_ ", input_items[-1]["content"])

    def test_subagent_requests_do_not_activate_safeguard(self):
        recorded = []
        policy = initiator_policy.InitiatorPolicy(on_safeguard_triggered=recorded.append)
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=start)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=5)):
            initiator = policy.resolve_initiator(
                "user",
                "gpt-5.4",
                subagent="guardian",
                request_id="req-2",
            )

        self.assertEqual(initiator, "agent")
        self.assertEqual(recorded, [])

    def test_responses_latest_user_message_wins_over_prior_assistant_history(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {"type": "message", "role": "assistant", "content": "previous answer"},
            {"type": "message", "role": "user", "content": "new prompt"},
        ]

        normalized_input, initiator = policy.resolve_responses_input(input_items, "gpt-5.4")

        self.assertIs(normalized_input, input_items)
        self.assertEqual(initiator, "user")

    def test_responses_latest_user_message_wins_over_prior_tool_history(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {"type": "message", "role": "assistant", "content": "calling tool"},
            {"type": "function_call", "call_id": "call-1", "name": "search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-1", "output": "result"},
            {"type": "message", "role": "user", "content": "continue with this"},
        ]

        normalized_input, initiator = policy.resolve_responses_input(input_items, "gpt-5.4")

        self.assertIs(normalized_input, input_items)
        self.assertEqual(initiator, "user")

    def test_codex_wrapped_prompt_after_tool_history_is_agent_continuation(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {
                "type": "message",
                "role": "user",
                "content": (
                    "<environment_context>\n"
                    "  <cwd>D:\\sources\\ghcp_proxy</cwd>\n"
                    "</environment_context>\n\n"
                    "why was 501ad942fbd14a1c832101cd713dca26 not flagged?"
                ),
            },
            {"type": "function_call", "call_id": "call-1", "name": "shell_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-1", "output": "fatal: unknown revision"},
            {
                "type": "message",
                "role": "user",
                "content": (
                    "<environment_context>\n"
                    "  <cwd>D:\\sources\\ghcp_proxy</cwd>\n"
                    "</environment_context>\n\n"
                    "why was 501ad942fbd14a1c832101cd713dca26 not flagged?\n\n"
                    "thats a request id not a git commit"
                ),
            },
        ]
        verdict = {}

        normalized_input, initiator = policy.resolve_responses_input(
            input_items,
            "gpt-5.5",
            verdict_sink=verdict,
        )

        self.assertIs(normalized_input, input_items)
        self.assertEqual(initiator, "agent")
        self.assertEqual(verdict["candidate_initiator"], "agent")

    def test_codex_title_generation_mini_request_is_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "developer instructions"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "# AGENTS.md instructions\n<environment_context>\n  <cwd>/tmp/worktree</cwd>\n</environment_context>",
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You are a helpful assistant. You will be presented with a user prompt, and your job is to provide a short title for a task that will be created from that prompt.\nGenerate a concise UI title (18-36 characters) for this task.\nUser prompt:\nFix the stale patch dashboard",
                    }
                ],
            },
        ]

        normalized_input, initiator = policy.resolve_responses_input(input_items, "gpt-5.4-mini")

        self.assertIs(normalized_input, input_items)
        self.assertEqual(initiator, "agent")

    def test_responses_function_call_output_tail_stays_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {"type": "message", "role": "user", "content": "find it"},
            {"type": "function_call", "call_id": "call-1", "name": "search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-1", "output": "result"},
        ]

        _normalized_input, initiator = policy.resolve_responses_input(input_items, "gpt-5.4")

        self.assertEqual(initiator, "agent")

    def test_plus_prefix_forces_user_for_plain_responses_input(self):
        policy = initiator_policy.InitiatorPolicy()

        normalized_input, initiator = policy.resolve_responses_input("+ hello", "gpt-5.4")

        self.assertEqual(normalized_input, "hello")
        self.assertEqual(initiator, "user")

    def test_plus_prefix_forces_user_for_latest_responses_user_message(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {"type": "message", "role": "assistant", "content": "previous answer"},
            {"type": "message", "role": "user", "content": "+ new prompt"},
        ]

        normalized_input, initiator = policy.resolve_responses_input(input_items, "gpt-5.4")

        self.assertIs(normalized_input, input_items)
        self.assertEqual(input_items[-1]["content"], "new prompt")
        self.assertEqual(initiator, "user")

    def test_plus_prefix_does_not_bypass_safeguard_on_replayed_codex_followup(self):
        # Regression: Codex replays the entire input list on every follow-up
        # Responses API turn. The original "+"-prefixed user message is still
        # the most-recent role:"user" item, but it is now followed by tool
        # calls and tool outputs. The explicit-user bypass must NOT fire on
        # those continuation turns — otherwise every follow-up gets billed
        # as a user premium request even though the harness, not the human,
        # initiated it.
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {"type": "message", "role": "user", "content": "+ investigate the bug"},
            {"type": "function_call", "name": "shell", "arguments": "{}", "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "reasoning", "summary": []},
        ]

        # First turn: pretend the user just sent it. "+" must force user.
        # We exercise the on-the-wire shape directly via the candidate helper.
        normalized_input, candidate = initiator_policy._determine_responses_candidate(
            [{"type": "message", "role": "user", "content": "+ investigate the bug"}]
        )
        self.assertEqual(candidate, initiator_policy._EXPLICIT_USER_INITIATOR)

        # Follow-up: same user message replayed with tool calls trailing it.
        normalized_input, candidate = initiator_policy._determine_responses_candidate(input_items)
        # Candidate should fall back to plain agent — the trailing function_call /
        # function_call_output items prove this is an agent-initiated continuation,
        # so the safeguard can clamp it.
        self.assertEqual(candidate, "agent")

    def test_plus_prefix_does_not_bypass_safeguard_with_trailing_assistant(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {"type": "message", "role": "user", "content": "+ kick this off"},
            {"type": "message", "role": "assistant", "content": "working on it"},
        ]

        normalized_input, candidate = initiator_policy._determine_responses_candidate(input_items)
        self.assertEqual(candidate, "agent")

    def test_plus_prefix_does_not_bypass_safeguard_on_replayed_anthropic_followup(self):
        # Anthropic equivalent of the Codex-replay bug. On follow-up turns the
        # harness appends an assistant message containing tool_use blocks and
        # then a new user message containing tool_result blocks. The original
        # "+"-prefixed user turn is no longer fresh and must NOT trigger the
        # explicit-user safeguard bypass.
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "+ investigate the bug"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "ok"},
                ],
            },
        ]

        candidate = initiator_policy._determine_anthropic_candidate(messages)
        # Should fall back to agent — the trailing assistant tool_use + user
        # tool_result prove this is an agent-initiated continuation. The "+"
        # bypass must not fire on the original (now-stale) user turn.
        self.assertEqual(candidate, "agent")

    def test_plus_prefix_forces_user_for_chat_messages(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "+ new prompt"},
        ]

        initiator = policy.resolve_chat_messages(messages, "gpt-5.4")

        self.assertEqual(messages[-1]["content"], "new prompt")
        self.assertEqual(initiator, "user")

    def test_plus_prefixed_skill_invocation_stays_user_for_responses_input(self):
        policy = initiator_policy.InitiatorPolicy()

        normalized_input, initiator = policy.resolve_responses_input("+ $superthinker debug this", "gpt-5.4")

        self.assertEqual(normalized_input, "$superthinker debug this")
        self.assertEqual(initiator, "user")

    def test_plus_prefixed_skill_invocation_stays_user_for_chat_messages(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "+ $superthinker debug this"},
        ]

        initiator = policy.resolve_chat_messages(messages, "gpt-5.4")

        self.assertEqual(messages[-1]["content"], "$superthinker debug this")
        self.assertEqual(initiator, "user")

    def test_plus_prefix_forces_user_for_anthropic_messages(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "+ new prompt"}]},
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(messages[-1]["content"][-1]["text"], "new prompt")
        self.assertEqual(initiator, "user")

    def test_plus_prefixed_skill_invocation_stays_user_for_anthropic_messages(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "+ $superthinker debug this"}]},
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(messages[-1]["content"][-1]["text"], "$superthinker debug this")
        self.assertEqual(initiator, "user")

    def test_plus_prefixed_anthropic_user_message_wins_over_trailing_task_notification(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "+ new prompt"}]},
            {"role": "user", "content": [{"type": "text", "text": "<task-notification>queued</task-notification>"}]},
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(messages[1]["content"][-1]["text"], "new prompt")
        self.assertEqual(initiator, "user")

    def test_trailing_underscore_marker_with_prompt_after_reminders_forces_agent(self):
        # Regression: Claude Code appends <system-reminder> blocks AFTER the
        # user's typed prompt in the same content string, so the marker ends
        # up at the START of the LAST line — followed by the actual prompt.
        policy = initiator_policy.InitiatorPolicy()
        appended_text = (
            "<system-reminder>\nAuto Mode Active\n</system-reminder>\n"
            "<system-reminder>\nclaudeMd contents\n</system-reminder>\n"
            "_ what do you mean entire last line?"
        )
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
            {"role": "user", "content": [{"type": "text", "text": appended_text}]},
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(initiator, "agent")
        final_text = messages[-1]["content"][-1]["text"]
        # Marker stripped; user's prompt survives on the same final line.
        self.assertTrue(final_text.endswith("what do you mean entire last line?"))
        self.assertNotIn("\n_ ", final_text)

    def test_trailing_underscore_marker_no_space_also_strips(self):
        # User types "_whats..." (no space after the marker).
        policy = initiator_policy.InitiatorPolicy()
        appended_text = (
            "<system-reminder>\nstuff\n</system-reminder>\n"
            "_whats the simplest fix?"
        )
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "prev"}]},
            {"role": "user", "content": [{"type": "text", "text": appended_text}]},
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(initiator, "agent")
        final_text = messages[-1]["content"][-1]["text"]
        self.assertTrue(final_text.endswith("whats the simplest fix?"))
        self.assertNotIn("_whats", final_text)

    def test_trailing_lone_underscore_marker_strips(self):
        # Marker on its own final line, no following text.
        policy = initiator_policy.InitiatorPolicy()
        appended_text = (
            "<system-reminder>\nstuff\n</system-reminder>\n"
            "_"
        )
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "prev"}]},
            {"role": "user", "content": [{"type": "text", "text": appended_text}]},
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(initiator, "agent")
        self.assertFalse(messages[-1]["content"][-1]["text"].rstrip().endswith("_"))

    def test_trailing_underscore_in_single_line_message_does_not_match(self):
        # No newline ⇒ "last line" is the whole text. The leading-prefix
        # check already handled that, so a trailing "_" inside an identifier
        # at the end of a single-line message must NOT be treated as a marker.
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "prev"}]},
            {"role": "user", "content": [{"type": "text", "text": "rename foo_bar to baz_"}]},
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(initiator, "user")
        self.assertEqual(messages[-1]["content"][-1]["text"], "rename foo_bar to baz_")

    def test_anthropic_tool_result_with_empty_text_tail_is_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "done", "is_error": False},
                    {"type": "text", "text": "   "},
                ],
            },
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(initiator, "agent")

    def test_anthropic_compaction_summary_is_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
            {
                "role": "user",
                "content": (
                    "This session is being continued from a previous conversation that ran out of context. "
                    "The summary below covers the earlier portion of the conversation.\n\nSummary:\n1. Example"
                ),
            },
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-opus-4.6")

        self.assertEqual(initiator, "agent")

    def test_anthropic_native_compaction_prompt_is_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
            {
                "role": "user",
                "content": (
                    "1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail\n"
                    "6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.\n"
                    "8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request.\n"
                    "REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis> block followed by a <summary> block."
                ),
            },
        ]

        initiator = policy.resolve_anthropic_messages(
            messages,
            "claude-opus-4.7",
            system="You are a helpful AI assistant tasked with summarizing conversations.",
        )

        self.assertEqual(initiator, "agent")

    def test_anthropic_local_command_wrapper_is_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
            {
                "role": "user",
                "content": (
                    "<command-name>/cost</command-name>\n"
                    "<command-message>cost</command-message>\n"
                    "<command-args></command-args>"
                ),
            },
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(initiator, "agent")

    def test_anthropic_tool_result_with_real_user_text_stays_user(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "done", "is_error": False},
                    {"type": "text", "text": "Now compare that with upstream."},
                ],
            },
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(initiator, "user")

    def test_anthropic_tool_result_with_system_reminder_only_is_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "done", "is_error": False},
                    {"type": "text", "text": "<system-reminder>\nThe task tools haven't been used recently.\n</system-reminder>"},
                ],
            },
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-opus-4.6")

        self.assertEqual(initiator, "agent")

    def test_responses_security_monitor_system_prompt_is_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        verdict = {}
        input_items = [
            {
                "type": "message",
                "role": "system",
                "content": "You are a security monitor for autonomous AI coding agents.",
            },
            {
                "type": "message",
                "role": "user",
                "content": "review this action",
            },
        ]

        _normalized, initiator = policy.resolve_responses_input(
            input_items,
            "gpt-5.4",
            verdict_sink=verdict,
        )

        self.assertEqual(initiator, "agent")
        self.assertEqual(verdict["candidate_initiator"], "agent")

    def test_chat_security_monitor_system_prompt_is_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        verdict = {}
        messages = [
            {
                "role": "system",
                "content": "You are a security monitor for autonomous AI coding agents.",
            },
            {
                "role": "user",
                "content": "review this action",
            },
        ]

        initiator = policy.resolve_chat_messages(
            messages,
            "gpt-5.4",
            verdict_sink=verdict,
        )

        self.assertEqual(initiator, "agent")
        self.assertEqual(verdict["candidate_initiator"], "agent")

    def test_anthropic_security_monitor_system_prompt_is_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        verdict = {}
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "review this action"}],
            },
        ]

        initiator = policy.resolve_anthropic_messages(
            messages,
            "claude-opus-4.6",
            system="You are a security monitor for autonomous AI coding agents.",
            verdict_sink=verdict,
        )

        self.assertEqual(initiator, "agent")
        self.assertEqual(verdict["candidate_initiator"], "agent")

    def test_anthropic_security_monitor_system_prompt_with_header_blocks_is_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        verdict = {}
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "review this action"}],
            },
        ]

        initiator = policy.resolve_anthropic_messages(
            messages,
            "claude-opus-4.6",
            system=[
                {"type": "text", "text": "Approval rubric"},
                {"type": "text", "text": "You are a security monitor for autonomous AI coding agents."},
            ],
            verdict_sink=verdict,
        )

        self.assertEqual(initiator, "agent")
        self.assertEqual(verdict["candidate_initiator"], "agent")

    def test_anthropic_user_message_with_security_monitor_phrase_stays_user(self):
        policy = initiator_policy.InitiatorPolicy()
        verdict = {}
        messages = [
            {
                "role": "user",
                "content": (
                    'The system prompt says "You are a security monitor for autonomous AI coding agents." '
                    "Can you explain what that means?"
                ),
            },
        ]

        initiator = policy.resolve_anthropic_messages(
            messages,
            "claude-opus-4.6",
            verdict_sink=verdict,
        )

        self.assertEqual(initiator, "user")
        self.assertEqual(verdict["candidate_initiator"], "user")

    def test_anthropic_system_reminder_with_real_user_text_stays_user(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "done", "is_error": False},
                    {"type": "text", "text": "<system-reminder>\nTask tools reminder.\n</system-reminder>"},
                    {"type": "text", "text": "Can you look at the proxy code?"},
                ],
            },
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-opus-4.6")

        self.assertEqual(initiator, "user")

    def test_anthropic_tool_result_with_step_away_recap_only_is_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {}}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "done", "is_error": False},
                    {
                        "type": "text",
                        "text": (
                            "The user stepped away and is coming back. Recap in under 40 words, "
                            "1-2 plain sentences, no markdown. Lead with the overall goal and "
                            "current task, then the one next action."
                        ),
                    },
                ],
            },
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-opus-4.6")

        self.assertEqual(initiator, "agent")

    def test_plus_prefix_does_not_override_haiku(self):
        policy = initiator_policy.InitiatorPolicy()

        normalized_input, initiator = policy.resolve_responses_input("+ hello", "claude-haiku-4.5")

        self.assertEqual(normalized_input, "hello")
        self.assertEqual(initiator, "agent")

    def test_plus_prefix_does_not_override_codex_bootstrap_mini(self):
        policy = initiator_policy.InitiatorPolicy()
        bootstrap_input = [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "developer instructions"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "+ <environment_context>\n  <cwd>D:\\sources\\ghcp_proxy</cwd>\n</environment_context>",
                    }
                ],
            },
        ]

        normalized_input, initiator = policy.resolve_responses_input(bootstrap_input, "gpt-5.4-mini")

        self.assertIs(normalized_input, bootstrap_input)
        self.assertEqual(
            normalized_input[-1]["content"][0]["text"],
            "<environment_context>\n  <cwd>D:\\sources\\ghcp_proxy</cwd>\n</environment_context>",
        )
        self.assertEqual(initiator, "agent")

    def test_plus_prefix_bypasses_active_request_safeguard(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=start)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=5)):
            normalized_input, initiator = policy.resolve_responses_input(
                "+ hello",
                "gpt-5.4",
            )

        self.assertEqual(normalized_input, "hello")
        self.assertEqual(initiator, "user")

    def test_plus_prefix_active_request_does_not_trigger_safeguard_event(self):
        recorded = []
        policy = initiator_policy.InitiatorPolicy(on_safeguard_triggered=recorded.append)
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=start)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=5)):
            _normalized_input, initiator = policy.resolve_responses_input(
                "+ hello",
                "gpt-5.4",
                request_id="req-2",
            )

        self.assertEqual(initiator, "user")
        self.assertEqual(recorded, [])

    def test_plus_prefix_does_not_override_haiku_agent_rule(self):
        policy = initiator_policy.InitiatorPolicy()
        _normalized, initiator = policy.resolve_responses_input(
            "+ hello", "claude-haiku-4.5"
        )
        self.assertEqual(initiator, "agent")

    def test_plus_prefix_does_not_override_subagent_agent_rule(self):
        policy = initiator_policy.InitiatorPolicy()
        _normalized, initiator = policy.resolve_responses_input(
            "+ hello", "gpt-5.4", subagent="general-purpose"
        )
        self.assertEqual(initiator, "agent")

    def test_plus_prefix_does_not_override_codex_mini_bootstrap_rule(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {"type": "message", "role": "developer", "content": "+ dev preamble"},
            {
                "type": "message",
                "role": "user",
                "content": "<environment_context>\n  <cwd>/tmp</cwd>\n</environment_context>",
            },
        ]
        _normalized, initiator = policy.resolve_responses_input(input_items, "gpt-5.4-mini")
        self.assertEqual(initiator, "agent")

    def test_trailing_skill_only_responses_user_message_forces_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {"type": "message", "role": "developer", "content": "developer instructions"},
            {"type": "message", "role": "user", "content": "The stubborn problem is a premium charge mismatch."},
            {
                "type": "message",
                "role": "user",
                "content": "<skill>\nApply the superthinker skill.\n</skill>",
            },
        ]

        normalized_input, initiator = policy.resolve_responses_input(input_items, "claude-opus-4.7")

        self.assertIs(normalized_input, input_items)
        self.assertEqual(initiator, "agent")

    def test_trailing_skill_only_anthropic_user_message_forces_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Investigate the premium charge mismatch."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "<skill>\nApply the superthinker skill.\n</skill>"}],
            },
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-opus-4.7")

        self.assertEqual(initiator, "agent")

    def test_skill_phrase_invocation_forces_agent_for_responses_input(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {"type": "message", "role": "assistant", "content": "previous answer"},
            {"type": "message", "role": "user", "content": "Use the superthinker skill for this billing bug."},
        ]

        normalized_input, initiator = policy.resolve_responses_input(input_items, "claude-opus-4.7")

        self.assertIs(normalized_input, input_items)
        self.assertEqual(initiator, "agent")

    def test_generic_skill_catalog_description_does_not_force_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<system-reminder>\n"
                            "The following skills are available for use with the Skill tool:\n\n"
                            "- update-config: Use this skill to configure the Claude Code harness via settings.json.\n"
                            "</system-reminder>"
                        ),
                    },
                    {
                        "type": "text",
                        "text": "Please fix the model routing source attribution bug.",
                    },
                ],
            },
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-opus-4.6")

        self.assertEqual(initiator, "user")

    def test_plus_prefixed_prompt_overrides_trailing_skill_block_for_responses_input(self):
        policy = initiator_policy.InitiatorPolicy()
        input_items = [
            {"type": "message", "role": "developer", "content": "developer instructions"},
            {"type": "message", "role": "user", "content": "+ Investigate the billing bug."},
            {
                "type": "message",
                "role": "user",
                "content": "<skill>\nApply the superthinker skill.\n</skill>",
            },
        ]

        normalized_input, initiator = policy.resolve_responses_input(input_items, "claude-opus-4.7")

        self.assertIs(normalized_input, input_items)
        self.assertEqual(input_items[1]["content"], "Investigate the billing bug.")
        self.assertEqual(initiator, "user")

    def test_plus_prefixed_prompt_overrides_trailing_skill_block_for_anthropic_messages(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "+ Investigate the billing bug."}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": "<skill>\nApply the superthinker skill.\n</skill>"}],
            },
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-opus-4.7")

        self.assertEqual(messages[1]["content"][-1]["text"], "Investigate the billing bug.")
        self.assertEqual(initiator, "user")

    def test_plus_prefix_bypasses_cooldown_for_plain_responses_input(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=start)
        policy.note_request_finished("req-1", finished_at=start.replace(second=5))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=10)):
            normalized_input, initiator = policy.resolve_responses_input("+ hello", "gpt-5.4")

        self.assertEqual(normalized_input, "hello")
        self.assertEqual(initiator, "user")

    def test_plus_prefixed_responses_message_bypasses_cooldown_through_trailing_turn_aborted_wrapper(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        input_items = [
            {"type": "message", "role": "assistant", "content": "previous answer"},
            {"type": "message", "role": "user", "content": "+ new prompt"},
            {
                "type": "message",
                "role": "user",
                "content": "<turn_aborted>interrupted</turn_aborted>",
            },
        ]

        policy.note_request_started("req-1", "user", started_at=start)
        policy.note_request_finished("req-1", finished_at=start.replace(second=5))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=10)):
            normalized_input, initiator = policy.resolve_responses_input(input_items, "gpt-5.4")

        self.assertIs(normalized_input, input_items)
        self.assertEqual(input_items[1]["content"], "new prompt")
        self.assertEqual(initiator, "user")

    def test_plus_prefix_bypasses_cooldown_for_chat_messages(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        messages = [
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "+ new prompt"},
        ]

        policy.note_request_started("req-1", "user", started_at=start)
        policy.note_request_finished("req-1", finished_at=start.replace(second=5))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=10)):
            initiator = policy.resolve_chat_messages(messages, "gpt-5.4")

        self.assertEqual(messages[-1]["content"], "new prompt")
        self.assertEqual(initiator, "user")

    def test_plus_prefix_bypasses_cooldown_for_anthropic_messages(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "previous answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "+ new prompt"}]},
        ]

        policy.note_request_started("req-1", "user", started_at=start)
        policy.note_request_finished("req-1", finished_at=start.replace(second=5))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=10)):
            initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(messages[-1]["content"][-1]["text"], "new prompt")
        self.assertEqual(initiator, "user")

    def test_parallel_user_candidate_requests_both_resolve_to_user(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=start)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=5)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

    def test_request_resolution_with_request_id_marks_parallel_user_request_active(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 5, tzinfo=timezone.utc)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5", request_id="req-1"), "user")

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=1)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5", request_id="req-2"), "user")

        self.assertEqual(set(policy._active_requests), {"req-1", "req-2"})

    def test_recent_finished_request_forces_following_user_looking_request_to_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        finished_at = datetime(2026, 4, 4, 18, 10, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=finished_at.replace(second=0))
        policy.note_request_finished("req-1", finished_at=finished_at)

        with mock.patch.object(initiator_policy, "utc_now", return_value=finished_at.replace(second=10)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

    def test_guard_expires_15_seconds_after_request_finishes(self):
        policy = initiator_policy.InitiatorPolicy()
        base = datetime(2026, 4, 4, 18, 20, tzinfo=timezone.utc)

        policy.note_request_started("req-0", "user", started_at=base)
        policy.note_request_finished("req-0", finished_at=base.replace(second=5))

        policy.note_request_started("req-1", "agent", started_at=base.replace(second=10))
        policy.note_request_finished("req-1", finished_at=base.replace(second=12))

        with mock.patch.object(initiator_policy, "utc_now", return_value=base.replace(second=25)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

        with mock.patch.object(initiator_policy, "utc_now", return_value=base.replace(second=28)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

    def test_any_request_activity_refreshes_the_15_second_timer(self):
        policy = initiator_policy.InitiatorPolicy()
        first_finish = datetime(2026, 4, 4, 18, 40, 5, tzinfo=timezone.utc)
        second_start = datetime(2026, 4, 4, 18, 40, 14, tzinfo=timezone.utc)
        second_finish = datetime(2026, 4, 4, 18, 40, 18, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=first_finish.replace(second=0))
        policy.note_request_finished("req-1", finished_at=first_finish)

        policy.note_request_started("req-2", "agent", started_at=second_start)
        policy.note_request_finished("req-2", finished_at=second_finish)

        with mock.patch.object(initiator_policy, "utc_now", return_value=datetime(2026, 4, 4, 18, 40, 30, tzinfo=timezone.utc)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

        with mock.patch.object(initiator_policy, "utc_now", return_value=datetime(2026, 4, 4, 18, 40, 34, tzinfo=timezone.utc)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

    def test_stream_like_request_does_not_block_parallel_user_request(self):
        policy = initiator_policy.InitiatorPolicy()
        started_at = datetime(2026, 4, 4, 18, 30, tzinfo=timezone.utc)
        finished_at = datetime(2026, 4, 4, 18, 31, tzinfo=timezone.utc)

        policy.note_request_started("stream-1", "user", started_at=started_at)

        with mock.patch.object(initiator_policy, "utc_now", return_value=finished_at):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

        policy.note_request_finished("stream-1", finished_at=finished_at)

        with mock.patch.object(initiator_policy, "utc_now", return_value=finished_at.replace(second=10)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

        with mock.patch.object(initiator_policy, "utc_now", return_value=finished_at.replace(second=16)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

    def test_safeguard_inactive_until_first_user_request(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "agent", started_at=start)
        policy.note_request_finished("req-1", finished_at=start.replace(second=10))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=12)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

    def test_active_stream_does_not_block_user_before_first_user_request(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("stream-1", "agent", started_at=start)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=5)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "user")

    def test_haiku_then_opus_user_prompt_is_user(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-haiku-4.5", request_id="haiku-1"),
                "agent",
            )

        policy.note_request_finished("haiku-1", finished_at=start.replace(second=2))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=3)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-sonnet-4.6", request_id="opus-1"),
                "user",
            )

    def test_haiku_streaming_does_not_block_first_user_opus(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-haiku-4.5", request_id="haiku-1"),
                "agent",
            )

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=1)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-sonnet-4.6", request_id="opus-1"),
                "user",
            )

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=2)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-sonnet-4.6"),
                "user",
            )

    def test_safeguard_activates_after_first_user_request(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "agent", started_at=start)
        policy.note_request_finished("req-1", finished_at=start.replace(second=5))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=6)):
            self.assertEqual(
                policy.resolve_initiator("user", "gpt-5", request_id="req-2"),
                "user",
            )

        policy.note_request_finished("req-2", finished_at=start.replace(second=6))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=7)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

    def test_any_copilot_activity_reactivates_safeguard_after_first_user_request(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start):
            policy.resolve_initiator("user", "gpt-5", request_id="req-1")

        policy.note_request_finished("req-1", finished_at=start.replace(second=5))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=10)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=25)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-haiku-4.5", request_id="haiku-1"),
                "agent",
            )

        policy.note_request_finished("haiku-1", finished_at=start.replace(second=26))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=27)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-opus-4.6", request_id="opus-1"),
                "agent",
            )

    def test_seeded_user_history_does_not_pre_activate_safeguard(self):
        policy = initiator_policy.InitiatorPolicy()
        old_time = datetime(2026, 4, 4, 17, 0, tzinfo=timezone.utc)
        now = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.seed_from_usage_events([
            {"finished_at": old_time.isoformat(), "initiator": "user"},
            {"finished_at": old_time.replace(second=5).isoformat(), "initiator": "agent"},
        ])

        with mock.patch.object(initiator_policy, "utc_now", return_value=now):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-haiku-4.5", request_id="haiku-1"),
                "agent",
            )

        with mock.patch.object(initiator_policy, "utc_now", return_value=now.replace(second=1)):
            self.assertEqual(
                policy.resolve_initiator("user", "claude-opus-4.6", request_id="opus-1"),
                "user",
            )


if __name__ == "__main__":
    unittest.main()
