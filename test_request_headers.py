import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import format_translation
import initiator_policy
import proxy
import request_headers
import usage_tracking


class RequestHeadersTests(unittest.TestCase):
    def setUp(self):
        proxy.set_initiator_policy(initiator_policy.InitiatorPolicy())
        proxy.usage_tracker.clear_state()

    def test_base_copilot_header_contract_is_exact(self):
        self.assertEqual(
            request_headers.build_copilot_headers("test-key"),
            {
                "Authorization": "Bearer test-key",
                "content-type": "application/json",
                "accept": "application/json",
                "User-Agent": "opencode/1.3.13",
                "Openai-Intent": "conversation-agent",
                "Copilot-Integration-Id": "vscode-chat",
                "x-github-api-version": "2026-01-09",
                "x-client-session-id": request_headers._CLIENT_SESSION_ID,
            },
        )

    def test_has_vision_input_depth_and_type_contract(self):
        self.assertFalse(request_headers.has_vision_input(None))
        self.assertFalse(request_headers.has_vision_input({"type": "text"}))
        self.assertFalse(request_headers.has_vision_input({"type": None, "content": []}))
        self.assertTrue(request_headers.has_vision_input({"type": "input_image"}, max_depth=0))
        self.assertFalse(request_headers.has_vision_input([{"type": "input_image"}], max_depth=0))
        self.assertTrue(request_headers.has_vision_input([{"type": "INPUT_IMAGE"}], max_depth=1))

        depth_10 = {"type": "input_image"}
        for _ in range(10):
            depth_10 = {"content": [depth_10]}
        self.assertTrue(request_headers.has_vision_input(depth_10))

        depth_11 = {"type": "input_image"}
        for _ in range(11):
            depth_11 = {"content": [depth_11]}
        self.assertFalse(request_headers.has_vision_input(depth_11))
        self.assertFalse(request_headers.has_vision_input({"content": [{"content": [{"type": "input_image"}]}]}, max_depth=1))

    def test_generate_request_id_from_payload_uses_copilot_last_user_content(self):
        first = request_headers._generate_request_id_from_payload(
            {
                "messages": [
                    {"role": "user", "content": "old"},
                    {"role": "assistant", "content": "ok"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "new", "cache_control": {"type": "ephemeral"}},
                            {"type": "tool_result", "content": "ignored"},
                        ],
                    },
                ]
            },
            session_id="session-a",
        )
        second = request_headers._generate_request_id_from_payload(
            {
                "messages": [
                    {"role": "user", "content": "old"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": [{"type": "text", "text": "new"}]},
                ]
            },
            session_id="session-a",
        )
        different_session = request_headers._generate_request_id_from_payload(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "new"}]}]},
            session_id="session-b",
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, different_session)

    def test_chat_vision_scan_stops_after_first_image_message(self):
        class RecordingPolicy:
            def resolve_chat_messages(self, *args, **kwargs):
                return "agent"

        class ExplodingDict(dict):
            def __contains__(self, key):
                raise AssertionError("vision scan should stop after image is found")

        request = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"), headers={})
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}]},
            {"role": "user", "content": [ExplodingDict()]},
        ]

        with mock.patch.object(request_headers.uuid, "uuid4", return_value="agent-task-id"):
            headers = request_headers.build_chat_headers_for_request(
                request,
                messages,
                "gpt-4.1",
                "test-key",
                initiator_policy=RecordingPolicy(),
                session_id_resolver=lambda req, body_arg=None: "session-id",
            )

        self.assertEqual(headers["Copilot-Vision-Request"], "true")
        self.assertEqual(
            request_headers.build_responses_copilot_headers("test-key"),
            {
                "Authorization": "Bearer test-key",
                "content-type": "application/json",
                "accept": "application/json",
                "User-Agent": "copilot/1.0.36 (client/github/cli win32 v25.6.0) term/unknown",
                "Openai-Intent": "conversation-agent",
                "Copilot-Integration-Id": "copilot-developer-cli",
                "x-github-api-version": "2026-01-09",
                "x-client-session-id": request_headers._CLIENT_SESSION_ID,
            },
        )

    def test_responses_header_contract_is_exact_and_uses_cache_affinity_locally(self):
        class RecordingPolicy:
            def __init__(self):
                self.calls = []

            def resolve_responses_input(
                self,
                input_value,
                model_name,
                *,
                subagent=None,
                force_initiator=None,
                request_id=None,
                verdict_sink=None,
            ):
                self.calls.append(
                    {
                        "input_value": input_value,
                        "model_name": model_name,
                        "subagent": subagent,
                        "force_initiator": force_initiator,
                        "request_id": request_id,
                        "verdict_sink": verdict_sink,
                    }
                )
                return input_value, "user"

        policy = RecordingPolicy()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            headers={
                "x-client-request-id": "client-request",
                "x-openai-subagent": "worker",
            },
        )
        body = {
            "model": "gpt-5.4-mini",
            "prompt_cache_key": " cache-key ",
            "promptCacheKey": "other-key",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
                }
            ],
        }
        verdict_sink = {}

        headers = request_headers.build_responses_headers_for_request(
            request,
            body,
            "test-key",
            force_initiator="user",
            request_id="req-1",
            initiator_policy=policy,
            session_id_resolver=lambda req, body_arg: "session-id",
            verdict_sink=verdict_sink,
        )

        self.assertEqual(
            policy.calls,
            [
                {
                    "input_value": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
                        }
                    ],
                    "model_name": "gpt-5.4-mini",
                    "subagent": "worker",
                    "force_initiator": "user",
                    "request_id": "req-1",
                    "verdict_sink": verdict_sink,
                }
            ],
        )
        request_id_header = headers.pop("x-request-id")
        agent_task_id = headers.pop("x-agent-task-id")
        interaction_id = headers.pop("x-interaction-id")
        self.assertEqual(request_id_header, agent_task_id)
        self.assertEqual(interaction_id, request_headers._copilot_uuid(agent_task_id))
        self.assertEqual(
            headers,
            {
                "Authorization": "Bearer test-key",
                "content-type": "application/json",
                "accept": "application/json",
                "User-Agent": "copilot/1.0.36 (client/github/cli win32 v25.6.0) term/unknown",
                "Openai-Intent": "conversation-agent",
                "Copilot-Integration-Id": "copilot-developer-cli",
                "x-github-api-version": "2026-01-09",
                "x-client-session-id": request_headers._CLIENT_SESSION_ID,
                "X-Initiator": "user",
                "x-interaction-type": "conversation-user",
                "Copilot-Vision-Request": "true",
            },
        )
        self.assertEqual(body.get("prompt_cache_key"), " cache-key ")
        self.assertEqual(body.get("promptCacheKey"), "other-key")

    def test_chat_header_contract_preserves_forwarded_ids_and_detects_image_url_key(self):
        class RecordingPolicy:
            def __init__(self):
                self.calls = []

            def resolve_chat_messages(
                self,
                messages,
                model_name,
                *,
                subagent=None,
                request_id=None,
                verdict_sink=None,
            ):
                self.calls.append(
                    {
                        "messages": messages,
                        "model_name": model_name,
                        "subagent": subagent,
                        "request_id": request_id,
                        "verdict_sink": verdict_sink,
                    }
                )
                return "agent"

        policy = RecordingPolicy()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/chat/completions"),
            headers={
                "x-client-request-id": "client-request",
                "x-openai-subagent": "worker",
            },
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"image_url": {"url": "https://example.invalid/a.png"}},
                ],
            }
        ]
        verdict_sink = {}

        with mock.patch.object(request_headers.uuid, "uuid4", return_value="agent-task-id"):
            headers = request_headers.build_chat_headers_for_request(
                request,
                messages,
                "gpt-4.1",
                "test-key",
                request_id="req-1",
                initiator_policy=policy,
                session_id_resolver=lambda req, body_arg=None: "session-id",
                verdict_sink=verdict_sink,
            )

        self.assertEqual(
            policy.calls,
            [
                {
                    "messages": messages,
                    "model_name": "gpt-4.1",
                    "subagent": "worker",
                    "request_id": "req-1",
                    "verdict_sink": verdict_sink,
                }
            ],
        )
        self.assertEqual(
            headers,
            {
                "Authorization": "Bearer test-key",
                "content-type": "application/json",
                "accept": "application/json",
                "User-Agent": "opencode/1.3.13",
                "Openai-Intent": "conversation-agent",
                "Copilot-Integration-Id": "vscode-chat",
                "x-github-api-version": "2026-01-09",
                "x-client-session-id": request_headers._CLIENT_SESSION_ID,
                "session_id": "session-id",
                "x-client-request-id": "client-request",
                "x-openai-subagent": "worker",
                "X-Initiator": "agent",
                "x-interaction-type": "conversation-agent",
                "x-interaction-id": "session-id",
                "x-agent-task-id": "agent-task-id",
                "Copilot-Vision-Request": "true",
            },
        )

    def test_anthropic_header_contract_preserves_forwarded_ids_and_detects_nested_vision(self):
        class RecordingPolicy:
            def __init__(self):
                self.calls = []

            def resolve_anthropic_messages(
                self,
                messages,
                model_name,
                *,
                system=None,
                subagent=None,
                request_id=None,
                verdict_sink=None,
            ):
                self.calls.append(
                    {
                        "messages": messages,
                        "model_name": model_name,
                        "system": system,
                        "subagent": subagent,
                        "request_id": request_id,
                        "verdict_sink": verdict_sink,
                    }
                )
                return "user"

        policy = RecordingPolicy()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            headers={
                "x-client-request-id": "client-request",
                "x-openai-subagent": "worker",
            },
        )
        body = {
            "model": "claude-sonnet-4.6",
            "system": "sys",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "image", "source": {"type": "url", "url": "https://example.invalid/a.png"}}],
                        }
                    ],
                }
            ],
        }
        verdict_sink = {}

        with mock.patch.object(request_headers.uuid, "uuid4", return_value="agent-task-id"):
            headers = request_headers.build_anthropic_headers_for_request(
                request,
                body,
                "test-key",
                request_id="req-1",
                initiator_policy=policy,
                session_id_resolver=lambda req, body_arg=None: "session-id",
                verdict_sink=verdict_sink,
            )

        self.assertEqual(
            policy.calls,
            [
                {
                    "messages": body["messages"],
                    "model_name": "claude-sonnet-4.6",
                    "system": "sys",
                    "subagent": "worker",
                    "request_id": "req-1",
                    "verdict_sink": verdict_sink,
                }
            ],
        )
        self.assertEqual(
            headers,
            {
                "Authorization": "Bearer test-key",
                "content-type": "application/json",
                "accept": "application/json",
                "User-Agent": "opencode/1.3.13",
                "Openai-Intent": "conversation-agent",
                "Copilot-Integration-Id": "vscode-chat",
                "x-github-api-version": "2026-01-09",
                "x-client-session-id": request_headers._CLIENT_SESSION_ID,
                "session_id": "session-id",
                "x-client-request-id": "client-request",
                "x-openai-subagent": "worker",
                "X-Initiator": "user",
                "x-interaction-type": "conversation-user",
                "x-interaction-id": "session-id",
                "x-agent-task-id": "agent-task-id",
                "Copilot-Vision-Request": "true",
            },
        )

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
        self.assertEqual(headers["Copilot-Integration-Id"], "copilot-developer-cli")
        self.assertIn("copilot/1.0.36", headers["User-Agent"])
        self.assertIn("x-interaction-id", headers)
        self.assertIn("x-agent-task-id", headers)
        self.assertIn("x-client-session-id", headers)
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
        self.assertNotIn("x-openai-subagent", headers)

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
        self.assertNotIn("x-openai-subagent", headers)

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

    def test_has_vision_input_handles_none_scalars_depth_and_nested_images(self):
        self.assertFalse(request_headers.has_vision_input(None))
        self.assertFalse(request_headers.has_vision_input("text"))
        self.assertFalse(request_headers.has_vision_input({"content": []}, depth=11, max_depth=10))
        self.assertTrue(
            request_headers.has_vision_input(
                {
                    "type": "message",
                    "content": [
                        {"type": "input_text", "text": "look"},
                        {"type": "input_image", "image_url": "https://example.invalid/a.png"},
                    ],
                }
            )
        )

    def test_responses_identity_headers_match_native_copilot_responses_shape(self):
        payload = {
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
                {"type": "function_call", "call_id": "call-1", "name": "Read", "arguments": "{}"},
            ]
        }

        headers = request_headers._responses_copilot_identity_headers(payload)

        self.assertEqual(headers["x-request-id"], headers["x-agent-task-id"])
        self.assertEqual(headers["x-interaction-id"], request_headers._copilot_uuid(headers["x-agent-task-id"]))

    def test_responses_identity_headers_match_anthropic_bridge_session_shape(self):
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        first = request_headers._responses_copilot_identity_headers(payload, " session-a ")
        second = request_headers._responses_copilot_identity_headers(payload, "session-a")
        later_turn = request_headers._responses_copilot_identity_headers(
            {"messages": [{"role": "user", "content": "follow up"}]},
            "session-a",
        )

        self.assertEqual(first, second)
        self.assertEqual(first["x-request-id"], first["x-agent-task-id"])
        self.assertEqual(first["x-interaction-id"], request_headers._copilot_uuid("session-a"))
        self.assertEqual(later_turn["x-interaction-id"], first["x-interaction-id"])
        self.assertNotEqual(later_turn["x-agent-task-id"], first["x-agent-task-id"])

    def test_interaction_id_for_blank_session_generates_new_id(self):
        generated = request_headers._interaction_id_for_session(" ")

        self.assertIsInstance(generated, str)
        self.assertNotEqual(generated, "")

    def test_apply_forwarded_request_headers_synthesizes_client_request_id_from_session(self):
        request = SimpleNamespace(
            headers={"x-openai-subagent": "worker"},
            url=SimpleNamespace(path="/v1/messages"),
        )
        headers = {}

        session_id = request_headers._apply_forwarded_request_headers(
            headers,
            request,
            {"sessionId": "session-123"},
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(session_id, "session-123")
        self.assertEqual(headers["session_id"], "session-123")
        self.assertEqual(headers["x-client-request-id"], "session-123")
        self.assertEqual(headers["x-openai-subagent"], "worker")

    def test_apply_forwarded_request_headers_preserves_forwarded_client_request_id(self):
        request = SimpleNamespace(
            headers={"x-client-request-id": "client-request"},
            url=SimpleNamespace(path="/v1/messages"),
        )
        headers = {}

        session_id = request_headers._apply_forwarded_request_headers(
            headers,
            request,
            {"sessionId": "session-123"},
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(session_id, "session-123")
        self.assertEqual(headers["session_id"], "session-123")
        self.assertEqual(headers["x-client-request-id"], "client-request")

    def test_apply_forwarded_request_headers_ignores_blank_session_for_client_request_id(self):
        request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/messages"))
        headers = {}

        session_id = request_headers._apply_forwarded_request_headers(
            headers,
            request,
            {"sessionId": " "},
            session_id_resolver=lambda _request, _body: " ",
        )

        self.assertEqual(session_id, " ")
        self.assertEqual(headers["session_id"], " ")
        self.assertNotIn("x-client-request-id", headers)

    def test_build_chat_headers_detects_image_url_content_and_skips_non_dict_items(self):
        request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/chat/completions"))
        messages = [
            "not-a-message",
            {"role": "user", "content": ["not-a-dict", {"type": "image_url", "image_url": {"url": "x"}}]},
            {"role": "user", "content": [{"type": "text", "text": "after image"}]},
        ]

        headers = format_translation.build_chat_headers_for_request(
            request,
            messages,
            "gpt-4.1",
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["Copilot-Vision-Request"], "true")

    def test_build_chat_headers_without_list_or_image_content_omits_vision_header(self):
        request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/chat/completions"))

        for messages in (
            None,
            [{"role": "user", "content": "plain text"}],
            [{"role": "user", "content": [{"type": "text"}]}],
        ):
            with self.subTest(messages=messages):
                headers = format_translation.build_chat_headers_for_request(
                    request,
                    messages,
                    "gpt-4.1",
                    "test-key",
                    initiator_policy=proxy._initiator_policy,
                    session_id_resolver=usage_tracking.request_session_id,
                )
                self.assertNotIn("Copilot-Vision-Request", headers)

    def test_anthropic_headers_detect_direct_and_nested_image_blocks(self):
        request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/messages"))
        direct_body = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "x"}}]}],
        }
        nested_body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "image", "source": {"type": "url", "url": "x"}}],
                        }
                    ],
                }
            ],
        }

        for body in (direct_body, nested_body):
            with self.subTest(body=body):
                headers = format_translation.build_anthropic_headers_for_request(
                    request,
                    body,
                    "test-key",
                    initiator_policy=proxy._initiator_policy,
                    session_id_resolver=usage_tracking.request_session_id,
                )
                self.assertEqual(headers["Copilot-Vision-Request"], "true")

    def test_anthropic_vision_detector_false_for_malformed_or_text_only_messages(self):
        self.assertFalse(request_headers._anthropic_messages_has_vision(None))
        self.assertFalse(request_headers._anthropic_messages_has_vision(["not-a-message"]))
        self.assertFalse(request_headers._anthropic_messages_has_vision([{"content": "text"}]))
        self.assertFalse(request_headers._anthropic_messages_has_vision([{"content": ["not-a-part"]}]))
        self.assertFalse(
            request_headers._anthropic_messages_has_vision(
                [{"content": [{"type": "tool_result", "content": [{"type": "text", "text": "no image"}]}]}]
            )
        )

    def test_build_responses_headers_for_request_without_input_leaves_body_without_input(self):
        request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/responses"))
        body = {"model": "gpt-5.4"}

        headers = format_translation.build_responses_headers_for_request(
            request,
            body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertNotIn("input", body)

    def test_build_responses_headers_marks_vision_input(self):
        request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/responses"))
        body = {
            "model": "gpt-5.4",
            "input": [{"type": "message", "content": [{"type": "input_image", "image_url": "x"}]}],
        }

        headers = format_translation.build_responses_headers_for_request(
            request,
            body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(headers["Copilot-Vision-Request"], "true")

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

    def test_build_responses_headers_for_request_does_not_forward_latest_server_request_id(self):
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

        self.assertIn("x-request-id", headers)
        self.assertEqual(headers["x-request-id"], headers["x-agent-task-id"])
        self.assertNotEqual(headers["x-request-id"], "server-prev")
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

        self.assertNotIn("session_id", headers)
        self.assertIn("x-client-session-id", headers)
        self.assertIn("x-interaction-id", headers)
        self.assertIn("x-agent-task-id", headers)

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

        self.assertNotIn("session_id", headers)
        self.assertNotIn("x-client-request-id", headers)
        self.assertIn("x-client-session-id", headers)
        self.assertIn("x-interaction-id", headers)
        self.assertIn("x-agent-task-id", headers)
        self.assertEqual(body["sessionId"], "session-123")
        self.assertNotIn("prompt_cache_key", body)

    def test_build_responses_headers_for_request_disables_session_forwarding(self):
        class RecordingPolicy:
            def resolve_responses_input(self, input_value, model_name, **kwargs):
                del model_name, kwargs
                return input_value, "agent"

        request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/responses"))
        body = {"model": "gpt-5.4", "input": "hello", "sessionId": "session-123"}

        with mock.patch.object(
            request_headers,
            "_apply_forwarded_request_headers",
            wraps=request_headers._apply_forwarded_request_headers,
        ) as apply_forwarded:
            headers = request_headers.build_responses_headers_for_request(
                request,
                body,
                "test-key",
                initiator_policy=RecordingPolicy(),
                session_id_resolver=usage_tracking.request_session_id,
            )

        self.assertNotIn("session_id", headers)
        self.assertNotIn("x-client-request-id", headers)
        apply_forwarded.assert_called_once_with(
            mock.ANY,
            request,
            body,
            session_id_resolver=usage_tracking.request_session_id,
            forward_session_header=False,
            synthesize_client_request_id=False,
        )

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

    def test_build_responses_headers_for_request_uses_prompt_cache_key_for_affinity_only(self):
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

        self.assertNotIn("session_id", headers)
        self.assertNotIn("x-client-request-id", headers)
        self.assertIn("x-client-session-id", headers)
        self.assertIn("x-interaction-id", headers)
        self.assertIn("x-agent-task-id", headers)
        self.assertEqual(body.get("promptCacheKey"), "cache-123")

    def test_build_responses_headers_for_request_uses_incoming_client_request_id_for_affinity_only(self):
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

        self.assertNotIn("session_id", headers)
        self.assertNotIn("x-client-request-id", headers)
        self.assertIn("x-interaction-id", headers)
        self.assertIn("x-agent-task-id", headers)

    def test_build_responses_headers_for_request_uses_payload_identity_not_prompt_cache_key(self):
        first_request = SimpleNamespace(headers={"x-openai-subagent": "worker"}, url=SimpleNamespace(path="/v1/responses"))
        first_body = {"model": "gpt-5.4", "input": "hello", "prompt_cache_key": "cache-a"}
        first_headers = format_translation.build_responses_headers_for_request(
            first_request,
            first_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        second_request = SimpleNamespace(headers={"x-openai-subagent": "worker"}, url=SimpleNamespace(path="/v1/responses"))
        second_body = {"model": "gpt-5.4", "input": "hello", "prompt_cache_key": "cache-b"}
        second_headers = format_translation.build_responses_headers_for_request(
            second_request,
            second_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        third_request = SimpleNamespace(headers={"x-openai-subagent": "worker"}, url=SimpleNamespace(path="/v1/responses"))
        third_body = {"model": "gpt-5.4", "input": "continue", "prompt_cache_key": "cache-a"}
        third_headers = format_translation.build_responses_headers_for_request(
            third_request,
            third_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(first_headers["x-interaction-id"], second_headers["x-interaction-id"])
        self.assertEqual(first_headers["x-agent-task-id"], second_headers["x-agent-task-id"])
        self.assertNotEqual(first_headers["x-interaction-id"], third_headers["x-interaction-id"])
        self.assertNotEqual(first_headers["x-agent-task-id"], third_headers["x-agent-task-id"])
        self.assertEqual(first_body.get("prompt_cache_key"), "cache-a")
        self.assertEqual(second_body.get("prompt_cache_key"), "cache-b")
        self.assertEqual(third_body.get("prompt_cache_key"), "cache-a")

    def test_build_responses_headers_for_request_rotates_explicit_affinity_on_native_user_turns(self):
        first_request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/responses"))
        first_body = {"model": "gpt-5.4", "input": "first", "prompt_cache_key": "cache-user-turns"}
        first_headers = format_translation.build_responses_headers_for_request(
            first_request,
            first_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        second_request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/responses"))
        second_body = {"model": "gpt-5.4", "input": "follow up", "prompt_cache_key": "cache-user-turns"}
        second_headers = format_translation.build_responses_headers_for_request(
            second_request,
            second_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(first_headers["X-Initiator"], "user")
        self.assertEqual(second_headers["X-Initiator"], "user")
        self.assertNotEqual(first_headers["x-interaction-id"], second_headers["x-interaction-id"])
        self.assertNotEqual(first_headers["x-agent-task-id"], second_headers["x-agent-task-id"])
        self.assertEqual(first_body.get("prompt_cache_key"), "cache-user-turns")
        self.assertEqual(second_body.get("prompt_cache_key"), "cache-user-turns")

    def test_build_responses_headers_for_request_uses_claude_bridge_session_as_interaction(self):
        first_request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/messages"))
        first_body = {"model": "gpt-5.4", "input": "first", "prompt_cache_key": "cache-user-turns"}
        first_original_body = {
            "model": "claude-opus-4.6",
            "metadata": {"user_id": '{"session_id":"claude-cache-session"}'},
            "messages": [{"role": "user", "content": "first"}],
        }
        first_headers = format_translation.build_responses_headers_for_request(
            first_request,
            first_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            affinity_body=first_original_body,
            stable_user_affinity=True,
        )

        second_request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/messages"))
        second_body = {"model": "gpt-5.4", "input": "follow up", "prompt_cache_key": "cache-user-turns"}
        second_original_body = {
            "model": "claude-opus-4.6",
            "metadata": {"user_id": '{"session_id":"claude-cache-session"}'},
            "messages": [{"role": "user", "content": "follow up"}],
        }
        second_headers = format_translation.build_responses_headers_for_request(
            second_request,
            second_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            affinity_body=second_original_body,
            stable_user_affinity=True,
        )

        self.assertEqual(first_headers["X-Initiator"], "user")
        self.assertEqual(second_headers["X-Initiator"], "user")
        self.assertEqual(first_headers["x-interaction-id"], second_headers["x-interaction-id"])
        self.assertNotEqual(first_headers["x-agent-task-id"], second_headers["x-agent-task-id"])
        self.assertEqual(first_body.get("prompt_cache_key"), "cache-user-turns")
        self.assertEqual(second_body.get("prompt_cache_key"), "cache-user-turns")

    def test_build_responses_headers_for_request_keeps_agent_task_stable_within_user_turn(self):
        request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/messages"))
        first_body = {
            "model": "gpt-5.4",
            "prompt_cache_key": "cache-user-turns",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix it"}]},
                {"type": "function_call", "name": "Read", "call_id": "call_1", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
            ],
        }
        second_body = {
            "model": "gpt-5.4",
            "prompt_cache_key": "cache-user-turns",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix it"}]},
                {"type": "function_call", "name": "Read", "call_id": "call_1", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
                {"type": "reasoning", "encrypted_content": "ciphertext"},
                {"type": "function_call", "name": "Edit", "call_id": "call_2", "arguments": "{}"},
            ],
        }

        first_headers = format_translation.build_responses_headers_for_request(
            request,
            first_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            stable_user_affinity=True,
        )
        second_headers = format_translation.build_responses_headers_for_request(
            request,
            second_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            stable_user_affinity=True,
        )

        self.assertEqual(first_headers["x-interaction-id"], second_headers["x-interaction-id"])
        self.assertEqual(first_headers["x-agent-task-id"], second_headers["x-agent-task-id"])

    def test_build_responses_headers_for_request_can_use_unsent_affinity_body(self):
        request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/messages"))
        upstream_body = {"model": "gpt-5.4", "input": "hello"}
        original_body = {
            "model": "claude-opus-4.6",
            "metadata": {"user_id": '{"session_id":"claude-cache-session"}'},
            "messages": [{"role": "user", "content": "hello"}],
        }

        first_headers = format_translation.build_responses_headers_for_request(
            request,
            upstream_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            affinity_body=original_body,
            stable_user_affinity=True,
        )
        second_headers = format_translation.build_responses_headers_for_request(
            request,
            {"model": "gpt-5.4", "input": "follow up"},
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            affinity_body=original_body,
            stable_user_affinity=True,
        )

        self.assertEqual(first_headers["X-Initiator"], "user")
        self.assertEqual(first_headers["x-interaction-id"], second_headers["x-interaction-id"])
        self.assertEqual(first_headers["x-agent-task-id"], second_headers["x-agent-task-id"])
        self.assertNotIn("metadata", upstream_body)

    def test_build_responses_headers_for_request_rotates_default_affinity_without_session(self):
        request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/messages"))

        first_headers = format_translation.build_responses_headers_for_request(
            request,
            {"model": "gpt-5.4", "input": "first"},
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )
        second_headers = format_translation.build_responses_headers_for_request(
            request,
            {"model": "gpt-5.4", "input": "follow up"},
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertEqual(first_headers["X-Initiator"], "user")
        self.assertEqual(second_headers["X-Initiator"], "user")
        self.assertNotEqual(first_headers["x-interaction-id"], second_headers["x-interaction-id"])
        self.assertNotEqual(first_headers["x-agent-task-id"], second_headers["x-agent-task-id"])

    def test_build_responses_headers_for_request_ignores_plain_metadata_user_id_as_affinity(self):
        request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/messages"))
        first_body = {
            "model": "gpt-5.4",
            "input": "first",
        }
        original_body = {
            "model": "claude-opus-4.6",
            "metadata": {"user_id": "plain-claude-session"},
            "messages": [{"role": "user", "content": "hello"}],
        }

        first_headers = format_translation.build_responses_headers_for_request(
            request,
            first_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            affinity_body=original_body,
            stable_user_affinity=True,
        )
        second_headers = format_translation.build_responses_headers_for_request(
            request,
            {"model": "gpt-5.4", "input": "follow up"},
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
            affinity_body=original_body,
            stable_user_affinity=True,
        )

        self.assertNotEqual(first_headers["x-agent-task-id"], "plain-claude-session")
        self.assertNotIn("x-interaction-id", first_headers)
        self.assertNotIn("x-interaction-id", second_headers)
        self.assertEqual(first_headers["x-agent-task-id"], second_headers["x-agent-task-id"])

    def test_build_responses_headers_for_compact_preserves_cache_affinity_fields(self):
        user_request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/responses"))
        user_body = {"model": "gpt-5.4", "input": "hello"}
        user_headers = format_translation.build_responses_headers_for_request(
            user_request,
            user_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        compact_request = SimpleNamespace(headers={}, url=SimpleNamespace(path="/v1/responses/compact"))
        compact_body = {
            "model": "gpt-5.4",
            "input": "hello",
            "sessionId": "session-123",
            "promptCacheKey": "cache-123",
            "previous_response_id": "resp_prev",
        }
        compact_headers = format_translation.build_responses_headers_for_request(
            compact_request,
            compact_body,
            "test-key",
            initiator_policy=proxy._initiator_policy,
            session_id_resolver=usage_tracking.request_session_id,
        )

        self.assertNotIn("session_id", compact_headers)
        self.assertNotIn("x-client-request-id", compact_headers)
        self.assertEqual(compact_body["sessionId"], "session-123")
        self.assertEqual(compact_body.get("promptCacheKey"), "cache-123")
        self.assertEqual(compact_body["previous_response_id"], "resp_prev")
        self.assertEqual(compact_headers["x-interaction-id"], user_headers["x-interaction-id"])
        self.assertEqual(compact_headers["x-agent-task-id"], user_headers["x-agent-task-id"])

    def test_build_responses_headers_for_request_does_not_forward_incoming_server_request_id(self):
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

        self.assertIn("x-request-id", headers)
        self.assertEqual(headers["x-request-id"], headers["x-agent-task-id"])
        self.assertNotEqual(headers["x-request-id"], "server-prev")
        self.assertNotIn("x-github-request-id", headers)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Anthropic /v1/messages passthrough header helpers
# ---------------------------------------------------------------------------

import request_headers as _rh


class DeriveAnthropicBetasTests(unittest.TestCase):
    def test_filters_to_allowlist(self):
        self.assertEqual(_rh._normalize_model_for_betas(None), "")
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

    def test_interleaved_thinking_requires_enabled_positive_integer_budget(self):
        for thinking in (
            {"type": "enabled", "budget_tokens": 0},
            {"type": "enabled", "budget_tokens": -1},
            {"type": "adaptive", "budget_tokens": 4096},
            {"type": "enabled", "budget_tokens": "4096"},
        ):
            with self.subTest(thinking=thinking):
                out = _rh.derive_anthropic_betas(
                    client_betas=None,
                    body={"thinking": thinking},
                    model="claude-haiku-4.5",
                )
                self.assertNotIn("interleaved-thinking-2025-05-14", out)

        out = _rh.derive_anthropic_betas(
            client_betas=None,
            body={"thinking": {"type": "enabled", "budget_tokens": 1}},
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

    def test_auto_injects_advanced_tool_use_for_anthropic_prefixed_opus_46(self):
        out = _rh.derive_anthropic_betas(
            client_betas=None,
            body={},
            model="anthropic/claude-opus-4.6",
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

    def test_client_beta_values_are_split_only_on_commas(self):
        out = _rh.derive_anthropic_betas(
            client_betas=[
                "interleaved-thinking-2025-05-14 context-management-2025-06-27",
            ],
            body={},
            model="claude-haiku-4.5",
        )

        self.assertEqual(out, [])

    def test_ignores_non_string_client_beta_entries(self):
        out = _rh.derive_anthropic_betas(
            client_betas=[123, None, " , context-management-2025-06-27 , "],
            body={},
            model="claude-haiku-4.5",
        )

        self.assertEqual(out, ["context-management-2025-06-27"])


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

    def test_reuses_messages_affinity_for_turns_in_same_session(self):
        first = _rh.build_anthropic_messages_passthrough_headers(
            request_id="req-1",
            initiator="user",
            interaction_id="session-1",
            interaction_type=None,
            anthropic_betas=[],
            base_headers={},
        )
        second = _rh.build_anthropic_messages_passthrough_headers(
            request_id="req-2",
            initiator="agent",
            interaction_id="session-1",
            interaction_type=None,
            anthropic_betas=[],
            base_headers={},
        )
        third = _rh.build_anthropic_messages_passthrough_headers(
            request_id="req-3",
            initiator="user",
            interaction_id="session-1",
            interaction_type=None,
            anthropic_betas=[],
            base_headers={},
        )

        self.assertEqual(first["x-interaction-id"], "session-1")
        self.assertEqual(second["x-interaction-id"], "session-1")
        self.assertEqual(third["x-interaction-id"], "session-1")
        self.assertEqual(first["x-agent-task-id"], "req-1")
        self.assertEqual(second["x-agent-task-id"], "req-2")
        self.assertEqual(third["x-agent-task-id"], "req-3")
        self.assertEqual(first["x-request-id"], "req-1")
        self.assertEqual(second["x-request-id"], "req-2")
        self.assertEqual(third["x-request-id"], "req-3")

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
        self.assertIn("x-interaction-id", headers)
        self.assertNotIn("copilot-integration-id", headers)
        self.assertEqual(headers["x-initiator"], "user")
        self.assertEqual(headers["x-interaction-id"], "r")
        self.assertEqual(headers["x-agent-task-id"], "r")

    def test_without_messages_interaction_id_does_not_reuse_default_affinity(self):
        first = _rh.build_anthropic_messages_passthrough_headers(
            request_id="req-a",
            initiator="user",
            interaction_id=None,
            interaction_type=None,
            anthropic_betas=[],
            base_headers={},
        )
        second = _rh.build_anthropic_messages_passthrough_headers(
            request_id="req-b",
            initiator="user",
            interaction_id=None,
            interaction_type=None,
            anthropic_betas=[],
            base_headers={},
        )

        self.assertEqual(first["x-interaction-id"], "req-a")
        self.assertEqual(first["x-agent-task-id"], "req-a")
        self.assertEqual(second["x-interaction-id"], "req-b")
        self.assertEqual(second["x-agent-task-id"], "req-b")

    def test_non_dict_base_headers_empty_initiator_and_non_list_betas_are_omitted(self):
        headers = _rh.build_anthropic_messages_passthrough_headers(
            request_id="r",
            initiator="",
            interaction_id="",
            interaction_type="ignored",
            anthropic_betas=None,
            base_headers=None,
        )

        self.assertEqual(headers["x-agent-task-id"], "r")
        self.assertEqual(headers["x-request-id"], "r")
        self.assertEqual(headers["x-interaction-type"], "messages-proxy")
        self.assertEqual(headers["openai-intent"], "messages-proxy")
        self.assertNotIn("x-initiator", headers)
        self.assertEqual(headers["x-interaction-id"], "r")
        self.assertNotIn("anthropic-beta", headers)

    def test_managed_headers_are_removed_case_insensitively_before_replacement(self):
        headers = _rh.build_anthropic_messages_passthrough_headers(
            request_id="new-request",
            initiator="agent",
            interaction_id="new-interaction",
            interaction_type=None,
            anthropic_betas=["context-management-2025-06-27"],
            base_headers={
                "User-Agent": "stale",
                "Openai-Intent": "stale",
                "X-Interaction-Type": "stale",
                "X-Interaction-Id": "stale",
                "X-Agent-Task-Id": "stale",
                "X-Request-Id": "stale",
                "X-Initiator": "stale",
                "Anthropic-Version": "stale",
                "Anthropic-Beta": "stale",
                "Authorization": "Bearer keep",
            },
        )

        self.assertNotIn("User-Agent", headers)
        self.assertNotIn("Openai-Intent", headers)
        self.assertNotIn("X-Interaction-Type", headers)
        self.assertNotIn("X-Interaction-Id", headers)
        self.assertNotIn("X-Agent-Task-Id", headers)
        self.assertNotIn("X-Request-Id", headers)
        self.assertNotIn("X-Initiator", headers)
        self.assertNotIn("Anthropic-Version", headers)
        self.assertNotIn("Anthropic-Beta", headers)
        self.assertEqual(headers["Authorization"], "Bearer keep")
        self.assertEqual(headers["user-agent"], _rh.CLAUDE_AGENT_USER_AGENT)
        self.assertEqual(headers["openai-intent"], "messages-proxy")
        self.assertEqual(headers["x-interaction-type"], "messages-proxy")
        self.assertEqual(headers["x-interaction-id"], "new-interaction")
        self.assertEqual(headers["x-agent-task-id"], "new-request")
        self.assertEqual(headers["x-request-id"], "new-request")
        self.assertEqual(headers["x-initiator"], "agent")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(headers["anthropic-beta"], "context-management-2025-06-27")
