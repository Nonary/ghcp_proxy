import unittest
from datetime import datetime, timezone
from unittest import mock

import initiator_policy
import proxy


class InitiatorPolicyTests(unittest.TestCase):
    def setUp(self):
        proxy.set_initiator_policy(initiator_policy.InitiatorPolicy())

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

    def test_active_request_forces_following_user_request_to_agent(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        policy.note_request_started("req-1", "user", started_at=start)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=5)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

    def test_request_resolution_with_request_id_marks_activity_for_other_requests(self):
        policy = initiator_policy.InitiatorPolicy()
        start = datetime(2026, 4, 4, 18, 5, tzinfo=timezone.utc)

        with mock.patch.object(initiator_policy, "utc_now", return_value=start):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5", request_id="req-1"), "user")

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=1)):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5", request_id="req-2"), "agent")

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

    def test_stream_like_request_stays_active_until_finished(self):
        policy = initiator_policy.InitiatorPolicy()
        started_at = datetime(2026, 4, 4, 18, 30, tzinfo=timezone.utc)
        finished_at = datetime(2026, 4, 4, 18, 31, tzinfo=timezone.utc)

        policy.note_request_started("stream-1", "user", started_at=started_at)

        with mock.patch.object(initiator_policy, "utc_now", return_value=finished_at):
            self.assertEqual(policy.resolve_initiator("user", "gpt-5"), "agent")

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
                "agent",
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
