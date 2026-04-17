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

    def test_plus_prefix_forces_user_for_chat_messages(self):
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": "+ new prompt"},
        ]

        initiator = policy.resolve_chat_messages(messages, "gpt-5.4")

        self.assertEqual(messages[-1]["content"], "new prompt")
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

    def test_anthropic_transcript_container_is_agent(self):
        """Claude Code approval agents wrap the conversation in a <transcript> block.

        Even with a genuine-looking user preamble (e.g. CLAUDE.md configuration)
        earlier in the messages list, the trailing transcript container is
        agent-generated and must be classified as agent. Regression test for
        a classifier gap uncovered by real /v1/messages captures.
        """
        policy = initiator_policy.InitiatorPolicy()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The following is the user's CLAUDE.md configuration."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<transcript>\n"},
                    {"type": "text", "text": "User: something the user said earlier\n"},
                    {"type": "text", "text": "Assistant: earlier reply\n"},
                    {"type": "text", "text": "</transcript>\n"},
                    {"type": "text", "text": "Err on the side of blocking."},
                ],
            },
        ]

        initiator = policy.resolve_anthropic_messages(messages, "claude-sonnet-4.6")

        self.assertEqual(initiator, "agent")

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
