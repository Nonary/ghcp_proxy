import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import format_translation
import initiator_policy
import proxy
import usage_tracking


class RequestHeadersTests(unittest.TestCase):
    def setUp(self):
        proxy.set_initiator_policy(initiator_policy.InitiatorPolicy())
        proxy.usage_tracker.clear_state()

    def test_responses_requests_default_to_user(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "hello",
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["input"], "hello")

    def test_gpt_5_4_mini_environment_bootstrap_is_agent(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5.4-mini",
            "input": [
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
            ],
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_gpt_5_4_mini_real_user_prompt_stays_user(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5.4-mini",
            "input": [
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
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Fix the safeguard bug"}],
                },
            ],
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "user")

    def test_underscore_prefixed_responses_string_is_agent_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "_hello",
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(body["input"], "hello")

    def test_plus_prefixed_responses_string_is_user_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "+hello",
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["input"], "hello")

    def test_plus_prefixed_responses_string_does_not_override_forced_agent_and_is_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "+hello",
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

    def test_plus_prefixed_responses_string_bypasses_cooldown_and_is_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "+hello",
        }
        start = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)

        proxy._initiator_policy.note_request_started("req-1", "user", started_at=start)
        proxy._initiator_policy.note_request_finished("req-1", finished_at=start.replace(second=5))

        with mock.patch.object(initiator_policy, "utc_now", return_value=start.replace(second=10)):
            headers = format_translation.build_responses_headers_for_request(
                request, body, "test-key",
                initiator_policy=proxy._initiator_policy,
                session_id_resolver=usage_tracking.request_session_id,
            )

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["input"], "hello")

    def test_responses_history_with_assistant_item_and_trailing_user_is_user(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": [
                {"role": "user", "content": "_old request"},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "new request"},
            ],
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["input"][0]["content"], "old request")
        self.assertEqual(body["input"][-1]["content"], "new request")

    def test_responses_underscore_prefixed_user_before_tool_history_is_agent(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "developer instructions"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "_open the codeql csv for me"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I’m checking the file now."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "Read",
                    "arguments": "{\"file\":\"alerts.csv\"}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "csv contents",
                },
            ],
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(
            body["input"][1]["content"][0]["text"],
            "open the codeql csv for me",
        )

    def test_codex_title_generation_mini_request_headers_are_agent(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5.4-mini",
            "input": [
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
            ],
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_chat_underscore_prefixed_user_message_is_agent_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"), headers={})
        messages = [
            {"role": "assistant", "content": "prior work"},
            {"role": "user", "content": "_finish the task"},
        ]

        headers = format_translation.build_chat_headers_for_request(
            request, messages, "gpt-4.1", "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(messages[-1]["content"], "finish the task")

    def test_chat_plus_prefixed_user_message_is_user_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"), headers={})
        messages = [
            {"role": "assistant", "content": "prior work"},
            {"role": "user", "content": "+finish the task"},
        ]

        headers = format_translation.build_chat_headers_for_request(
            request, messages, "gpt-4.1", "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(messages[-1]["content"], "finish the task")

    def test_responses_function_call_output_follow_up_stays_agent(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read main.py"}]},
                {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "{\"file\":\"main.py\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "file contents"},
            ],
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_responses_mcp_approval_response_follow_up_stays_agent(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run the command"}]},
                {"type": "mcp_approval_response", "approval_request_id": "apr_1", "approve": True},
            ],
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_responses_subagent_request_stays_agent(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            headers={"x-openai-subagent": "guardian"},
        )
        body = {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Review and improve the layout in dashboard.html"}],
                },
            ],
        }

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_chat_tool_message_follow_up_stays_agent(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"), headers={})
        messages = [
            {"role": "user", "content": "read main.py"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": "{\"file\":\"main.py\"}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
        ]

        headers = format_translation.build_chat_headers_for_request(
            request, messages, "gpt-4.1", "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_anthropic_underscore_prefixed_user_message_is_agent_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/messages"), headers={})
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "_hello"}],
                }
            ],
        }

        headers = format_translation.build_anthropic_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(body["messages"][0]["content"][0]["text"], "hello")

    def test_anthropic_plus_prefixed_user_message_is_user_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/messages"), headers={})
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "+hello"}],
                }
            ],
        }

        headers = format_translation.build_anthropic_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["messages"][0]["content"][0]["text"], "hello")

    def test_anthropic_plus_prefixed_user_message_wins_over_trailing_task_notification(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/messages"), headers={})
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "previous answer"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "+hello"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "<task-notification>queued</task-notification>"}],
                },
            ],
        }

        headers = format_translation.build_anthropic_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["messages"][1]["content"][0]["text"], "hello")

    def test_anthropic_security_monitor_system_prompt_sets_agent_initiator(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/messages"), headers={})
        body = {
            "model": "claude-sonnet-4.6",
            "system": "You are a security monitor for autonomous AI coding agents.",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "review this action"}],
                }
            ],
        }

        headers = format_translation.build_anthropic_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_anthropic_security_monitor_system_prompt_with_header_blocks_sets_agent_initiator(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/messages"), headers={})
        body = {
            "model": "claude-sonnet-4.6",
            "system": [
                {"type": "text", "text": "Approval rubric"},
                {"type": "text", "text": "You are a security monitor for autonomous AI coding agents."},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "review this action"}],
                }
            ],
        }

        headers = format_translation.build_anthropic_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_anthropic_tool_result_follow_up_stays_agent(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/messages"), headers={})
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "read main.py"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file": "main.py"}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file contents"}],
                },
            ],
        }

        headers = format_translation.build_anthropic_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_anthropic_user_text_after_tool_result_is_user(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/messages"), headers={})
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file": "main.py"}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file contents"},
                        {"type": "text", "text": "now summarize it"},
                    ],
                },
            ],
        }

        headers = format_translation.build_anthropic_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "user")

    def test_build_anthropic_headers_for_request_uses_body_session_id(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/messages"), headers={})
        body = {
            "model": "claude-sonnet-4.6",
            "sessionId": "claude-session",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        }

        headers = format_translation.build_anthropic_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["session_id"], "claude-session")

    def test_build_responses_headers_for_request_forwards_latest_server_request_id(self):
        proxy.usage_tracker.remember_latest_server_request_id("session-123", None, None, "server-prev")

        request = SimpleNamespace(
            headers={"session_id": "session-123"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        body = {"model": "gpt-5.4", "input": "hello"}

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertNotIn("x-request-id", headers)
        self.assertNotIn("x-github-request-id", headers)

    def test_build_responses_headers_for_request_uses_hyphenated_session_header(self):
        request = SimpleNamespace(
            headers={"session-id": "session-123"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        body = {"model": "gpt-5.4", "input": "hello"}

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["session_id"], "session-123")

    def test_build_responses_headers_for_request_uses_body_session_id(self):
        request = SimpleNamespace(
            headers={},
            url=SimpleNamespace(path="/v1/responses"),
        )
        body = {"model": "gpt-5.4", "input": "hello", "sessionId": "session-123"}

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["session_id"], "session-123")
        self.assertEqual(headers["x-client-request-id"], "session-123")
        self.assertEqual(body["prompt_cache_key"], "session-123")

    def test_build_responses_headers_for_request_uses_conversation_agent_intent(self):
        request = SimpleNamespace(
            headers={},
            url=SimpleNamespace(path="/v1/responses"),
        )
        body = {"model": "gpt-5.4", "input": "hello"}

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["Openai-Intent"], "conversation-agent")

    def test_build_responses_headers_for_request_normalizes_prompt_cache_key_alias(self):
        request = SimpleNamespace(
            headers={},
            url=SimpleNamespace(path="/v1/responses"),
        )
        body = {"model": "gpt-5.4", "input": "hello", "promptCacheKey": "cache-123"}

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["session_id"], "cache-123")
        self.assertEqual(headers["x-client-request-id"], "cache-123")
        self.assertEqual(body["prompt_cache_key"], "cache-123")
        self.assertNotIn("promptCacheKey", body)

    def test_build_responses_headers_for_request_preserves_incoming_client_request_id(self):
        request = SimpleNamespace(
            headers={"x-client-request-id": "client-123"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        body = {"model": "gpt-5.4", "input": "hello", "sessionId": "session-123"}

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["session_id"], "session-123")
        self.assertEqual(headers["x-client-request-id"], "client-123")

    def test_build_responses_headers_for_request_preserves_incoming_server_request_id(self):
        request = SimpleNamespace(
            headers={"x-request-id": "server-prev"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        body = {"model": "gpt-5.4", "input": "hello"}

        headers = format_translation.build_responses_headers_for_request(
            request, body, "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["x-request-id"], "server-prev")
        self.assertEqual(headers["x-github-request-id"], "server-prev")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Anthropic /v1/messages passthrough header helpers
# ---------------------------------------------------------------------------

import request_headers as _rh


class DeriveAnthropicBetasTests(unittest.TestCase):
    def test_filters_to_allowlist(self):
        out = _rh.derive_anthropic_betas(
            client_betas=[
                "interleaved-thinking-2025-05-14",
                "prompt-caching-2024-07-31",  # not in allowlist
                "context-management-2025-06-27",
            ],
            body={},
            model="claude-haiku-4.5",
        )
        self.assertIn("interleaved-thinking-2025-05-14", out)
        self.assertIn("context-management-2025-06-27", out)
        self.assertNotIn("prompt-caching-2024-07-31", out)

    def test_auto_injects_interleaved_thinking_when_budget_set(self):
        out = _rh.derive_anthropic_betas(
            client_betas=None,
            body={"thinking": {"type": "enabled", "budget_tokens": 4096}},
            model="claude-haiku-4.5",
        )
        self.assertIn("interleaved-thinking-2025-05-14", out)

    def test_does_not_inject_thinking_for_adaptive(self):
        out = _rh.derive_anthropic_betas(
            client_betas=None,
            body={"thinking": {"type": "adaptive"}},
            model="claude-haiku-4.5",
        )
        self.assertNotIn("interleaved-thinking-2025-05-14", out)

    def test_auto_injects_advanced_tool_use_for_sonnet_46(self):
        out = _rh.derive_anthropic_betas(
            client_betas=None,
            body={},
            model="claude-sonnet-4.6",
        )
        self.assertIn("advanced-tool-use-2025-11-20", out)

    def test_no_advanced_tool_use_for_sonnet_4(self):
        out = _rh.derive_anthropic_betas(
            client_betas=None,
            body={},
            model="claude-sonnet-4.0",
        )
        self.assertNotIn("advanced-tool-use-2025-11-20", out)

    def test_dedupes_and_handles_comma_separated(self):
        out = _rh.derive_anthropic_betas(
            client_betas=[
                "interleaved-thinking-2025-05-14, context-management-2025-06-27",
                "interleaved-thinking-2025-05-14",
            ],
            body={},
            model="claude-haiku-4.5",
        )
        self.assertEqual(
            sorted(out),
            sorted({"interleaved-thinking-2025-05-14", "context-management-2025-06-27"}),
        )


class BuildAnthropicMessagesPassthroughHeadersTests(unittest.TestCase):
    def test_sets_required_headers(self):
        base = {
            "Authorization": "Bearer x",
            "Copilot-Integration-Id": "vscode-chat",
            "content-type": "application/json",
        }
        headers = _rh.build_anthropic_messages_passthrough_headers(
            request_id="req-123",
            initiator="agent",
            interaction_id="int-9",
            interaction_type=None,
            anthropic_betas=[
                "interleaved-thinking-2025-05-14",
                "advanced-tool-use-2025-11-20",
            ],
            base_headers=base,
        )
        self.assertEqual(headers["x-agent-task-id"], "req-123")
        self.assertEqual(headers["x-request-id"], "req-123")
        self.assertEqual(headers["x-interaction-type"], "messages-proxy")
        self.assertEqual(headers["openai-intent"], "messages-proxy")
        self.assertEqual(headers["x-initiator"], "agent")
        self.assertEqual(headers["x-interaction-id"], "int-9")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(
            headers["anthropic-beta"],
            "interleaved-thinking-2025-05-14,advanced-tool-use-2025-11-20",
        )
        self.assertIn("vscode_claude_code/2.1.98", headers["user-agent"])
        # Copilot-Integration-Id removed (case insensitive)
        self.assertNotIn("Copilot-Integration-Id", headers)
        self.assertNotIn("copilot-integration-id", headers)
        # Authorization passed through
        self.assertEqual(headers["Authorization"], "Bearer x")
        # Input dict not mutated
        self.assertIn("Copilot-Integration-Id", base)

    def test_omits_anthropic_beta_when_empty_and_interaction_id_optional(self):
        headers = _rh.build_anthropic_messages_passthrough_headers(
            request_id="r",
            initiator="user",
            interaction_id=None,
            interaction_type=None,
            anthropic_betas=[],
            base_headers={"copilot-integration-id": "vscode-chat"},
        )
        self.assertNotIn("anthropic-beta", headers)
        self.assertNotIn("x-interaction-id", headers)
        self.assertNotIn("copilot-integration-id", headers)
        self.assertEqual(headers["x-initiator"], "user")
