import asyncio
import compression.zstd as pyzstd
import auth
import dashboard
import format_translation
import gzip
import os
import unittest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import httpx
import initiator_policy
import proxy
import util
import usage_tracking


class ProxyInitiatorTests(unittest.TestCase):
    def setUp(self):
        proxy.set_initiator_policy(initiator_policy.InitiatorPolicy())
        state = proxy.usage_tracker.state
        with state.usage_log_lock:
            state.recent_usage_events.clear()
            state.archived_usage_events.clear()
        with state.session_request_id_lock:
            state.latest_server_request_ids_by_chain.clear()
            state.active_server_request_ids_by_request.clear()
            state.latest_claude_user_session_contexts.clear()
        proxy.dashboard_service.reset_official_premium_cache()

    def test_configured_upstream_timeout_seconds_defaults_to_300(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GHCP_UPSTREAM_TIMEOUT_SECONDS", None)
            self.assertEqual(proxy.configured_upstream_timeout_seconds(), 300)

    def test_configured_upstream_timeout_seconds_uses_env_override(self):
        with mock.patch.dict(os.environ, {"GHCP_UPSTREAM_TIMEOUT_SECONDS": "480"}, clear=False):
            self.assertEqual(proxy.configured_upstream_timeout_seconds(), 480)

    def _make_usage_tracker(
        self,
        *,
        archive_store: usage_tracking.UsageArchiveStore | None = None,
        event_bus=None,
    ) -> usage_tracking.UsageTracker:
        return usage_tracking.UsageTracker(
            state=usage_tracking.UsageTrackingState(),
            archive_store=archive_store,
            event_bus=event_bus,
        )

    def _make_usage_log_path(self, prefix: str = "usage-log-") -> Path:
        path = Path.cwd() / f"{prefix}{uuid4().hex}.jsonl"
        def _cleanup():
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                pass
        self.addCleanup(_cleanup)
        return path

    def test_parse_json_request_accepts_gzip_encoded_body(self):
        raw = gzip.compress(b'{"model":"gpt-5","input":"hello"}')
        request = SimpleNamespace(
            headers={"content-encoding": "gzip"},
            body=mock.AsyncMock(return_value=raw),
        )

        parsed = proxy.asyncio.run(proxy.parse_json_request(request))

        self.assertEqual(parsed["model"], "gpt-5")
        self.assertEqual(parsed["input"], "hello")

    def test_parse_json_request_accepts_zstd_encoded_body(self):
        raw = pyzstd.compress(b'{"model":"gpt-5","input":"hello"}')
        request = SimpleNamespace(
            headers={"content-encoding": "zstd"},
            body=mock.AsyncMock(return_value=raw),
        )

        parsed = proxy.asyncio.run(proxy.parse_json_request(request))

        self.assertEqual(parsed["model"], "gpt-5")
        self.assertEqual(parsed["input"], "hello")

    def test_responses_requests_default_to_user(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "hello",
        }

        headers = format_translation.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["input"], "hello")

    def test_underscore_prefixed_responses_string_is_agent_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": "_hello",
        }

        headers = format_translation.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(body["input"], "hello")

    def test_only_latest_responses_user_item_controls_agent_prefix(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "gpt-5",
            "input": [
                {"role": "user", "content": "_old request"},
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "new request"},
            ],
        }

        headers = format_translation.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "user")
        self.assertEqual(body["input"][-1]["content"], "new request")

    def test_chat_underscore_prefixed_user_message_is_agent_and_stripped(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"), headers={})
        messages = [
            {"role": "assistant", "content": "prior work"},
            {"role": "user", "content": "_finish the task"},
        ]

        headers = format_translation.build_chat_headers_for_request(request, messages, "gpt-4.1", "test-key")

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(messages[-1]["content"], "finish the task")

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

        headers = format_translation.build_anthropic_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(body["messages"][0]["content"][0]["text"], "hello")

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

        headers = format_translation.build_anthropic_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["session_id"], "claude-session")

    def test_anthropic_request_to_chat_translates_cache_control_to_copilot_cache_control(self):
        body = {
            "model": "claude-sonnet-4.6",
            "system": [
                {"type": "text", "text": "first"},
                {
                    "type": "text",
                    "text": "cached",
                    "cache_control": {"ephemeral": {"scope": "conversation"}},
                },
            ],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        self.assertEqual(outbound["model"], "claude-sonnet-4.6")
        self.assertEqual(outbound["thinking_budget"], 4096)
        self.assertEqual(outbound["messages"][0]["role"], "system")
        self.assertIsInstance(outbound["messages"][0]["content"], list)
        self.assertNotIn("stream_options", outbound)
        self.assertEqual(
            outbound["messages"][0]["content"][1]["copilot_cache_control"],
            {"type": "ephemeral"},
        )
        self.assertEqual(
            outbound["messages"][1]["content"][0]["copilot_cache_control"],
            {"type": "ephemeral"},
        )

    def test_anthropic_request_to_chat_preserves_cache_control_on_tool_results(self):
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"file": "test.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file contents",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        tool_msg = [m for m in outbound["messages"] if m["role"] == "tool"][0]
        self.assertEqual(
            tool_msg["copilot_cache_control"],
            {"type": "ephemeral"},
        )

    def test_anthropic_request_to_chat_no_cache_control_on_tool_result_without_it(self):
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tool_1", "name": "Read", "input": {"file": "test.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "file contents",
                        }
                    ],
                },
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        tool_msg = [m for m in outbound["messages"] if m["role"] == "tool"][0]
        self.assertNotIn("copilot_cache_control", tool_msg)

    def test_build_fake_compaction_request_preserves_openai_xhigh_reasoning(self):
        body = {
            "model": "openai/gpt-5.4",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            "reasoning": {"effort": "xhigh", "summary": "auto"},
        }

        compact_request = format_translation.build_fake_compaction_request(body)

        self.assertEqual(compact_request["reasoning"], {"effort": "xhigh", "summary": "auto"})
        self.assertEqual(body["reasoning"], {"effort": "xhigh", "summary": "auto"})

    def test_build_fake_compaction_request_preserves_anthropic_max_reasoning(self):
        body = {
            "model": "anthropic/claude-sonnet-4.6",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
            "reasoning": {"effort": "max"},
        }

        compact_request = format_translation.build_fake_compaction_request(body)

        self.assertEqual(compact_request["reasoning"], {"effort": "max"})

    def test_anthropic_request_to_chat_stream_requests_usage_chunks(self):
        body = {
            "model": "claude-sonnet-4.6",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        }

        outbound = proxy.asyncio.run(format_translation.anthropic_request_to_chat(body, "https://example.invalid", "test-key"))

        self.assertEqual(outbound["stream_options"], {"include_usage": True})

    def test_start_usage_event_tracks_metadata_only(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={
                "session_id": "session-123",
                "x-client-request-id": "client-456",
            },
        )
        outbound_body = {
            "model": "claude-haiku-4.5",
            "messages": [
                {"role": "system", "content": "stay concise"},
                {"role": "user", "content": [{"type": "text", "text": "inspect file.py"}]},
            ],
            "max_tokens": 1024,
            "requestType": "ChatMessages",
            "otherOptions": {"intent": "debug"},
        }

        event = tracker.start_event(
            request,
            requested_model="claude-haiku-4-5-20251001",
            resolved_model="claude-haiku-4.5",
            initiator="agent",
            request_id="req-123",
            request_body=outbound_body,
            upstream_path="/chat/completions",
        )

        self.assertEqual(event["request_id"], "req-123")
        self.assertEqual(event["upstream_path"], "/chat/completions")
        self.assertEqual(event["session_id"], "session-123")
        self.assertEqual(event["client_request_id"], "client-456")
        self.assertIsInstance(event["server_request_id"], str)
        self.assertNotIn("request_text", event)
        self.assertNotIn("request_options", event)
        self.assertNotIn("request_payload", event)

    def test_start_usage_event_uses_hyphenated_session_header(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"session-id": "session-123"},
        )
        outbound_headers = {}

        event = tracker.start_event(
            request,
            requested_model="claude-haiku-4.5",
            resolved_model="claude-haiku-4.5",
            initiator="agent",
            outbound_headers=outbound_headers,
        )

        self.assertEqual(event["session_id"], "session-123")
        self.assertEqual(outbound_headers["session_id"], "session-123")

    def test_start_usage_event_uses_request_body_session_id(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        outbound_headers = {}

        event = tracker.start_event(
            request,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            initiator="agent",
            request_body={"sessionId": "session-123"},
            outbound_headers=outbound_headers,
        )

        self.assertEqual(event["session_id"], "session-123")
        self.assertEqual(outbound_headers["session_id"], "session-123")

    def test_start_usage_event_uses_claude_code_session_header(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"x-claude-code-session-id": "claude-session"},
        )
        outbound_headers = {}

        event = tracker.start_event(
            request,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            initiator="user",
            outbound_headers=outbound_headers,
        )

        self.assertEqual(event["session_id"], "claude-session")
        self.assertEqual(event["session_id_origin"], "request")
        self.assertEqual(outbound_headers["session_id"], "claude-session")

    def test_start_usage_event_uses_opencode_session_affinity_header(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"x-session-affinity": "opencode-session"},
        )
        outbound_headers = {}

        event = tracker.start_event(
            request,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            initiator="user",
            outbound_headers=outbound_headers,
        )

        self.assertEqual(event["session_id"], "opencode-session")
        self.assertEqual(event["session_id_origin"], "request")
        self.assertEqual(outbound_headers["session_id"], "opencode-session")

    def test_start_usage_event_uses_claude_metadata_user_id_session_id(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        outbound_headers = {}

        event = tracker.start_event(
            request,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            initiator="user",
            request_body={
                "metadata": {
                    "user_id": '{"device_id":"device-123","session_id":"claude-session"}'
                }
            },
            outbound_headers=outbound_headers,
        )

        self.assertEqual(event["session_id"], "claude-session")
        self.assertEqual(event["session_id_origin"], "request")
        self.assertEqual(outbound_headers["session_id"], "claude-session")

    def test_start_usage_event_user_uses_request_id_session_when_session_missing(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        outbound_headers = {}
        request_id = "user-request-123"

        event = tracker.start_event(
            request,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            initiator="user",
            request_id=request_id,
            outbound_headers=outbound_headers,
        )

        self.assertEqual(event["request_id"], request_id)
        self.assertEqual(event["session_id"], request_id)
        self.assertEqual(event["session_id_origin"], "request_id")
        self.assertEqual(outbound_headers["session_id"], request_id)
        self.assertEqual(outbound_headers["x-request-id"], event["server_request_id"])
        self.assertEqual(outbound_headers["x-github-request-id"], event["server_request_id"])

    def test_start_usage_event_user_follow_up_attaches_to_active_claude_session(self):
        tracker = self._make_usage_tracker()
        first_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"x-client-request-id": "client-123"},
        )
        first_event = tracker.start_event(
            first_request,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            initiator="user",
        )

        follow_up_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"x-client-request-id": "client-123"},
        )
        follow_up_event = tracker.start_event(
            follow_up_request,
            requested_model="claude-haiku-4.5",
            resolved_model="claude-haiku-4.5",
            initiator="user",
        )

        self.assertEqual(follow_up_event["session_id"], first_event["session_id"])
        self.assertEqual(follow_up_event["session_id_origin"], "request_id")
        self.assertEqual(follow_up_event["prior_server_request_id"], first_event["server_request_id"])
        self.assertEqual(follow_up_event["server_request_id"], first_event["server_request_id"])

    def test_start_usage_event_agent_follow_up_attaches_to_active_claude_session(self):
        tracker = self._make_usage_tracker()
        user_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        user_event = tracker.start_event(
            user_request,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            initiator="user",
        )

        agent_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        agent_event = tracker.start_event(
            agent_request,
            requested_model="claude-haiku-4.5",
            resolved_model="claude-haiku-4.5",
            initiator="agent",
        )

        self.assertEqual(agent_event["session_id"], user_event["session_id"])
        self.assertEqual(agent_event["session_id_origin"], "request_id")
        self.assertEqual(agent_event["prior_server_request_id"], user_event["server_request_id"])
        self.assertEqual(agent_event["server_request_id"], user_event["server_request_id"])

    def test_start_usage_event_agent_follow_up_after_user_finish_attaches_to_latest_claude_session(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        user_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        user_event = tracker.start_event(
            user_request,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            initiator="user",
        )

        with mock.patch.object(usage_tracking, "USAGE_LOG_FILE", str(log_path)):
            tracker.finish_event(user_event, 200)

        agent_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        agent_event = tracker.start_event(
            agent_request,
            requested_model="claude-haiku-4.5",
            resolved_model="claude-haiku-4.5",
            initiator="agent",
        )

        self.assertEqual(agent_event["session_id"], user_event["session_id"])
        self.assertEqual(agent_event["session_id_origin"], "request_id")
        self.assertEqual(agent_event["prior_server_request_id"], user_event["server_request_id"])
        self.assertEqual(agent_event["server_request_id"], user_event["server_request_id"])

    def test_start_usage_event_user_non_claude_request_does_not_generate_implicit_session(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )

        event = tracker.start_event(
            request,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            initiator="user",
            outbound_headers={},
        )

        self.assertIsNone(event["session_id"])
        self.assertIsNone(event["session_id_origin"])
        self.assertIsNone(event["prior_server_request_id"])

    def test_start_usage_event_user_without_chain_context_does_not_reuse_unrelated_active_request_id(self):
        tracker = self._make_usage_tracker()
        first_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        first_event = tracker.start_event(
            first_request,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            initiator="user",
            request_id="first-user-request",
        )

        second_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        second_event = tracker.start_event(
            second_request,
            requested_model="claude-haiku-4.5",
            resolved_model="claude-haiku-4.5",
            initiator="user",
            request_id="second-user-request",
        )

        self.assertEqual(first_event["session_id"], "first-user-request")
        self.assertEqual(second_event["session_id"], "second-user-request")
        self.assertNotEqual(second_event["server_request_id"], first_event["server_request_id"])
        self.assertIsNone(second_event["prior_server_request_id"])

    def test_start_usage_event_agent_inherits_latest_session_server_request_id(self):
        tracker = self._make_usage_tracker()
        with tracker.state.session_request_id_lock:
            tracker.state.latest_server_request_ids_by_chain[("session:session-123", "__root__")] = "server-prev"

        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={"session_id": "session-123"},
        )

        event = tracker.start_event(
            request,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            initiator="agent",
        )

        self.assertEqual(event["prior_server_request_id"], "server-prev")
        self.assertEqual(event["server_request_id"], "server-prev")

    def test_start_usage_event_agent_inherits_latest_body_session_server_request_id(self):
        tracker = self._make_usage_tracker()
        with tracker.state.session_request_id_lock:
            tracker.state.latest_server_request_ids_by_chain[("session:session-123", "__root__")] = "server-prev"

        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )

        event = tracker.start_event(
            request,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            initiator="agent",
            request_body={"sessionId": "session-123"},
        )

        self.assertEqual(event["prior_server_request_id"], "server-prev")
        self.assertEqual(event["server_request_id"], "server-prev")

    def test_start_usage_event_user_request_starts_new_server_request_id(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )

        event = tracker.start_event(
            request,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            initiator="user",
        )

        self.assertIsInstance(event["server_request_id"], str)
        self.assertIsNone(event["prior_server_request_id"])

    def test_start_usage_event_agent_inherits_active_user_server_request_id(self):
        tracker = self._make_usage_tracker()
        user_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        user_event = tracker.start_event(
            user_request,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            initiator="user",
        )

        agent_request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        agent_event = tracker.start_event(
            agent_request,
            requested_model="claude-haiku-4.5",
            resolved_model="claude-haiku-4.5",
            initiator="agent",
        )

        self.assertEqual(agent_event["prior_server_request_id"], user_event["server_request_id"])
        self.assertEqual(agent_event["server_request_id"], user_event["server_request_id"])

    def test_start_usage_event_subagent_request_starts_its_own_server_request_id(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"x-openai-subagent": "worker-1"},
        )

        event = tracker.start_event(
            request,
            requested_model="claude-haiku-4.5",
            resolved_model="claude-haiku-4.5",
            initiator="agent",
        )

        self.assertIsInstance(event["server_request_id"], str)
        self.assertIsNone(event["prior_server_request_id"])

    def test_start_usage_event_applies_server_request_id_to_outbound_headers(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        outbound_headers = {}

        event = tracker.start_event(
            request,
            requested_model="claude-haiku-4.5",
            resolved_model="claude-haiku-4.5",
            initiator="user",
            outbound_headers=outbound_headers,
        )

        self.assertEqual(outbound_headers["x-request-id"], event["server_request_id"])
        self.assertEqual(outbound_headers["x-github-request-id"], event["server_request_id"])

    def test_finish_usage_event_tracks_usage_and_timing(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        event = {
            "request_id": "req-999",
            "resolved_model": "gpt-5.4",
            "_started_monotonic": 10.0,
            "_first_output_monotonic": 10.45,
        }
        response_payload = {
            "id": "resp_123",
            "model": "gpt-5.4",
            "output_text": "done",
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }
        upstream = SimpleNamespace(headers={"x-request-id": "server-abc", "content-type": "application/json"})

        with (
            mock.patch.object(usage_tracking.time, "perf_counter", return_value=12.0),
            mock.patch.object(usage_tracking, "USAGE_LOG_FILE", str(log_path)),
        ):
            tracker.finish_event(
                event,
                200,
                upstream=upstream,
                response_payload=response_payload,
            )

        finished = tracker.state.recent_usage_events[0]
        self.assertEqual(finished["response_id"], "resp_123")
        self.assertEqual(finished["upstream_request_id"], "server-abc")
        self.assertEqual(finished["usage"]["input_tokens"], 11)
        self.assertEqual(finished["duration_ms"], 2000)
        self.assertEqual(finished["time_to_first_token_ms"], 450)
        self.assertEqual(finished["premium_requests"], 1.0)
        self.assertTrue(finished["success"])

    def test_finish_usage_event_remembers_session_server_request_id(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        event = {
            "request_id": "req-999",
            "resolved_model": "gpt-5.4",
            "session_id": "session-123",
            "server_request_id": "proxy-chain-123",
        }
        upstream = SimpleNamespace(headers={"x-request-id": "server-abc"})

        with mock.patch.object(usage_tracking, "USAGE_LOG_FILE", str(log_path)):
            tracker.finish_event(event, 200, upstream=upstream)

        self.assertEqual(
            tracker.latest_server_request_id("session-123", None, None),
            "proxy-chain-123",
        )

    def test_finish_usage_event_records_cost_from_model_pricing(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        event = {
            "request_id": "req-999",
            "resolved_model": "gpt-5.4",
            "_started_monotonic": 10.0,
        }
        usage = {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "cached_input_tokens": 1_000_000,
        }

        with mock.patch.object(usage_tracking, "USAGE_LOG_FILE", str(log_path)):
            tracker.finish_event(event, 200, usage=usage)

        finished = tracker.state.recent_usage_events[0]
        self.assertAlmostEqual(finished["cost_usd"], 17.75)

    def test_sse_usage_capture_extracts_token_usage(self):
        tracker = self._make_usage_tracker()
        capture = tracker.create_sse_capture("responses")

        saw_output = capture.feed(
            b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
        )
        capture.feed(
            b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.4","usage":{"input_tokens":5,"output_tokens":2,"input_tokens_details":{"cached_tokens":3},"output_tokens_details":{"reasoning_tokens":1}},"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}]}}\n\n'
        )

        self.assertTrue(saw_output)
        self.assertEqual(capture.usage["input_tokens"], 2)
        self.assertEqual(capture.usage["output_tokens"], 2)
        self.assertEqual(capture.usage["total_tokens"], 4)
        self.assertEqual(capture.usage["cached_input_tokens"], 3)
        self.assertEqual(capture.usage["reasoning_output_tokens"], 1)

    def test_sse_usage_capture_normalizes_cached_prompt_tokens(self):
        tracker = self._make_usage_tracker()
        capture = tracker.create_sse_capture("chat")
        capture.feed(
            b'event: message\ndata: {"id":"chatcmpl_1","choices":[{"delta":{"content":"Hi"}}],"usage":{"prompt_tokens":20,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":7}}}\n\n'
        )

        self.assertEqual(capture.usage["input_tokens"], 13)
        self.assertEqual(capture.usage["output_tokens"], 4)
        self.assertEqual(capture.usage["total_tokens"], 17)
        self.assertEqual(capture.usage["cached_input_tokens"], 7)

    def test_build_responses_headers_for_request_forwards_latest_server_request_id(self):
        with proxy.usage_tracker.state.session_request_id_lock:
            proxy.usage_tracker.state.latest_server_request_ids_by_chain[("session:session-123", "__root__")] = "server-prev"

        request = SimpleNamespace(
            headers={"session_id": "session-123"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        body = {"model": "gpt-5.4", "input": "hello"}

        headers = format_translation.build_responses_headers_for_request(request, body, "test-key")

        self.assertNotIn("x-request-id", headers)
        self.assertNotIn("x-github-request-id", headers)

    def test_build_responses_headers_for_request_uses_hyphenated_session_header(self):
        request = SimpleNamespace(
            headers={"session-id": "session-123"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        body = {"model": "gpt-5.4", "input": "hello"}

        headers = format_translation.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["session_id"], "session-123")

    def test_build_responses_headers_for_request_uses_body_session_id(self):
        request = SimpleNamespace(
            headers={},
            url=SimpleNamespace(path="/v1/responses"),
        )
        body = {"model": "gpt-5.4", "input": "hello", "sessionId": "session-123"}

        headers = format_translation.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["session_id"], "session-123")

    def test_build_responses_headers_for_request_preserves_incoming_server_request_id(self):
        request = SimpleNamespace(
            headers={"x-request-id": "server-prev"},
            url=SimpleNamespace(path="/v1/responses"),
        )
        body = {"model": "gpt-5.4", "input": "hello"}

        headers = format_translation.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["x-request-id"], "server-prev")
        self.assertEqual(headers["x-github-request-id"], "server-prev")

    def test_load_usage_history_normalizes_cached_tokens(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        log_path.write_text(
            '{"session_id":"session-123","server_request_id":"server-abc","usage":{"prompt_tokens":20,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":7}}}\n',
            encoding="utf-8",
        )

        with mock.patch.object(usage_tracking, "USAGE_LOG_FILE", str(log_path)):
            tracker.load_history()
            events = list(tracker.state.recent_usage_events)

        self.assertEqual(events[0]["usage"]["input_tokens"], 13)
        self.assertEqual(events[0]["usage"]["total_tokens"], 17)
        self.assertEqual(events[0]["usage"]["cached_input_tokens"], 7)
        self.assertEqual(
            tracker.latest_server_request_id("session-123", None, None),
            "server-abc",
        )

    def test_load_usage_history_normalizes_responses_usage_details(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        log_path.write_text(
            '{"session_id":"session-123","server_request_id":"server-def","usage":{"input_tokens":20,"output_tokens":4,"input_tokens_details":{"cached_tokens":7},"output_tokens_details":{"reasoning_tokens":2}}}\n',
            encoding="utf-8",
        )

        with mock.patch.object(usage_tracking, "USAGE_LOG_FILE", str(log_path)):
            tracker.load_history()
            events = list(tracker.state.recent_usage_events)

        self.assertEqual(events[0]["usage"]["input_tokens"], 13)
        self.assertEqual(events[0]["usage"]["total_tokens"], 17)
        self.assertEqual(events[0]["usage"]["cached_input_tokens"], 7)
        self.assertEqual(events[0]["usage"]["reasoning_output_tokens"], 2)
        self.assertEqual(
            tracker.latest_server_request_id("session-123", None, None),
            "server-def",
        )

    def test_normalize_usage_payload_preserves_explicit_cached_input_shape(self):
        normalized = util.normalize_usage_payload(
            {
                "input_tokens": 20,
                "output_tokens": 4,
                "total_tokens": 24,
                "cached_input_tokens": 7,
            }
        )

        self.assertEqual(normalized["input_tokens"], 20)
        self.assertEqual(normalized["output_tokens"], 4)
        self.assertEqual(normalized["total_tokens"], 24)
        self.assertEqual(normalized["cached_input_tokens"], 7)

    def test_chat_completion_to_anthropic_maps_cached_usage_tokens(self):
        payload = {
            "id": "chatcmpl_123",
            "model": "claude-sonnet-4.6",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "cache_creation_input_tokens": 20,
                },
            },
        }

        translated = format_translation.chat_completion_to_anthropic(payload)

        self.assertEqual(translated["usage"]["input_tokens"], 20)
        self.assertEqual(translated["usage"]["output_tokens"], 25)
        self.assertEqual(translated["usage"]["cache_read_input_tokens"], 80)
        self.assertEqual(translated["usage"]["cache_creation_input_tokens"], 0)

    def test_chat_completion_to_anthropic_subtracts_cache_reads_from_input_tokens(self):
        payload = {
            "id": "chatcmpl_456",
            "model": "claude-sonnet-4.6",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 135200,
                "completion_tokens": 690,
                "prompt_tokens_details": {
                    "cached_tokens": 125000,
                },
            },
        }

        translated = format_translation.chat_completion_to_anthropic(payload)

        self.assertEqual(translated["usage"]["input_tokens"], 10200)
        self.assertEqual(translated["usage"]["cache_read_input_tokens"], 125000)
        self.assertEqual(translated["usage"]["cache_creation_input_tokens"], 0)

    def test_anthropic_stream_refreshes_message_start_usage_when_openai_usage_arrives_late(self):
        chunks = [
            (
                'event: message\n'
                'data: {"id":"chatcmpl_123","model":"claude-haiku-4.5","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            ).encode("utf-8"),
            (
                'event: message\n'
                'data: {"id":"chatcmpl_123","model":"claude-haiku-4.5","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":30,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":10}}}\n\n'
            ).encode("utf-8"),
            b"data: [DONE]\n\n",
        ]

        class FakeUpstream:
            status_code = 200

            async def aiter_bytes(self):
                for chunk in chunks:
                    yield chunk

            async def aread(self):
                return b""

            async def aclose(self):
                return None

        class FakeClient:
            def build_request(self, *args, **kwargs):
                return object()

            async def aclose(self):
                return None

        async def collect_stream_body():
            response = await proxy.proxy_anthropic_streaming_response(
                "https://example.invalid/chat/completions",
                {"Authorization": "Bearer test"},
                {"stream": True},
                "claude-haiku-4.5",
            )
            body = b""
            async for chunk in response.body_iterator:
                body += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
            return body.decode("utf-8")

        with (
            mock.patch.object(proxy.httpx, "AsyncClient", return_value=FakeClient()),
            mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(return_value=FakeUpstream())),
        ):
            body = proxy.asyncio.run(collect_stream_body())

        self.assertEqual(body.count("event: message_start"), 2)
        self.assertIn(
            '"usage":{"input_tokens":20,"output_tokens":5,"cache_creation_input_tokens":0,"cache_read_input_tokens":10}',
            body,
        )

    def test_proxy_streaming_response_connect_error_returns_openai_error(self):
        request = httpx.Request("POST", "https://example.invalid/responses")

        class FakeClient:
            def __init__(self):
                self.aclose = mock.AsyncMock()

            def build_request(self, *args, **kwargs):
                return request

        fake_client = FakeClient()
        usage_event = {"request_id": "req-123"}
        connect_error = httpx.ConnectError("All connection attempts failed", request=request)

        with (
            mock.patch.object(proxy.httpx, "AsyncClient", return_value=fake_client),
            mock.patch.object(proxy, "throttled_client_send", mock.AsyncMock(side_effect=connect_error)),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
        ):
            response = proxy.asyncio.run(
                proxy.proxy_streaming_response(
                    "https://example.invalid/responses",
                    {"Authorization": "Bearer test"},
                    {"model": "gpt-5.4-mini", "stream": True},
                    usage_event=usage_event,
                )
            )

        finish_usage.assert_called_once_with(
            usage_event,
            502,
            response_text="Upstream connection failed: All connection attempts failed",
        )
        fake_client.aclose.assert_awaited_once()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.body,
            b'{"error":{"message":"Upstream connection failed: All connection attempts failed","type":"server_error","param":null,"code":null}}',
        )

    def test_graceful_streaming_response_swallows_cancelled_error(self):
        response = proxy.GracefulStreamingResponse(iter(()))
        receive = mock.AsyncMock()
        send = mock.AsyncMock()

        with mock.patch.object(
            proxy.StreamingResponse,
            "__call__",
            mock.AsyncMock(side_effect=asyncio.CancelledError()),
        ) as parent_call:
            proxy.asyncio.run(response({}, receive, send))

        parent_call.assert_awaited_once_with({}, receive, send)

    def test_upstream_request_error_status_and_message_maps_timeout_to_504(self):
        request = httpx.Request("POST", "https://example.invalid/responses")
        timeout_error = httpx.ReadTimeout("Timed out", request=request)

        self.assertEqual(
            format_translation.upstream_request_error_status_and_message(timeout_error),
            (504, "Upstream request timed out: Timed out"),
        )

    def test_anthropic_error_payload_from_openai_uses_anthropic_shape(self):
        payload = {
            "error": {
                "message": "bad request",
                "type": "invalid_request_error",
            }
        }

        translated = format_translation.anthropic_error_payload_from_openai(payload, 400)

        self.assertEqual(
            translated,
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "bad request",
                },
            },
        )

    def test_anthropic_error_response_from_upstream_translates_openai_error(self):
        upstream = httpx.Response(
            429,
            json={
                "error": {
                    "message": "rate limited",
                }
            },
            headers={"retry-after": "12"},
        )

        response = format_translation.anthropic_error_response_from_upstream(upstream)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "12")
        self.assertEqual(
            response.body,
            b'{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}',
        )

    def test_anthropic_messages_route_uses_anthropic_headers_and_error_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"x-client-request-id": "req-123"},
        )
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        }
        outbound = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        upstream = httpx.Response(
            400,
            json={
                "error": {
                    "message": "unsupported field",
                    "type": "invalid_request_error",
                }
            },
            headers={"content-type": "application/json"},
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(format_translation, "anthropic_request_to_chat", mock.AsyncMock(return_value=outbound)),
            mock.patch.object(usage_tracking, "log_proxy_request"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event"),
            mock.patch.object(format_translation, "build_anthropic_headers_for_request", return_value={"X-Initiator": "user"}) as build_headers,
            mock.patch.object(format_translation, "build_chat_headers_for_request", side_effect=AssertionError("unexpected chat headers")),
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(return_value=upstream)) as post,
        ):
            response = proxy.asyncio.run(proxy.anthropic_messages(request))

        build_headers.assert_called_once_with(request, body, "test-key", request_id=mock.ANY)
        self.assertEqual(post.await_args.args[1], "https://example.invalid/chat/completions")
        self.assertEqual(post.await_args.kwargs["headers"], {"X-Initiator": "user"})
        self.assertEqual(post.await_args.kwargs["json"], outbound)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.body,
            b'{"type":"error","error":{"type":"invalid_request_error","message":"unsupported field"}}',
        )

    def test_responses_route_invalid_json_returns_openai_error_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )

        with mock.patch.object(
            proxy,
            "parse_json_request",
            mock.AsyncMock(side_effect=proxy.HTTPException(status_code=400, detail="Invalid JSON body")),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.body,
            b'{"error":{"message":"Invalid JSON body","type":"invalid_request_error","param":null,"code":null}}',
        )

    def test_responses_route_upstream_connect_error_returns_openai_error_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        body = {
            "model": "gpt-5.4-mini",
            "input": "hello",
            "stream": False,
        }
        connect_error = httpx.ConnectError(
            "All connection attempts failed",
            request=httpx.Request("POST", "https://example.invalid/responses"),
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(format_translation, "build_responses_headers_for_request", return_value={"X-Initiator": "agent"}),
            mock.patch.object(usage_tracking, "log_proxy_request"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(side_effect=connect_error)),
        ):
            response = proxy.asyncio.run(proxy.responses(request))

        finish_usage.assert_called_once_with(
            None,
            502,
            response_text="Upstream connection failed: All connection attempts failed",
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.body,
            b'{"error":{"message":"Upstream connection failed: All connection attempts failed","type":"server_error","param":null,"code":null}}',
        )

    def test_anthropic_messages_invalid_json_returns_anthropic_error_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )

        with mock.patch.object(
            proxy,
            "parse_json_request",
            mock.AsyncMock(side_effect=proxy.HTTPException(status_code=400, detail="Invalid JSON body")),
        ):
            response = proxy.asyncio.run(proxy.anthropic_messages(request))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.body,
            b'{"type":"error","error":{"type":"invalid_request_error","message":"Invalid JSON body"}}',
        )

    def test_anthropic_messages_upstream_connect_error_returns_anthropic_error_shape(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )
        body = {
            "model": "claude-sonnet-4.6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ],
        }
        outbound = {
            "model": "claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        connect_error = httpx.ConnectError(
            "All connection attempts failed",
            request=httpx.Request("POST", "https://example.invalid/chat/completions"),
        )

        with (
            mock.patch.object(proxy, "parse_json_request", mock.AsyncMock(return_value=body)),
            mock.patch.object(auth, "get_api_key", return_value="test-key"),
            mock.patch.object(auth, "get_api_base", return_value="https://example.invalid"),
            mock.patch.object(format_translation, "anthropic_request_to_chat", mock.AsyncMock(return_value=outbound)),
            mock.patch.object(format_translation, "build_anthropic_headers_for_request", return_value={"X-Initiator": "agent"}),
            mock.patch.object(usage_tracking, "log_proxy_request"),
            mock.patch.object(proxy.usage_tracker, "start_event", return_value=None),
            mock.patch.object(proxy.usage_tracker, "finish_event") as finish_usage,
            mock.patch.object(proxy, "throttled_client_post", mock.AsyncMock(side_effect=connect_error)),
        ):
            response = proxy.asyncio.run(proxy.anthropic_messages(request))

        finish_usage.assert_called_once_with(
            None,
            502,
            response_text="Upstream connection failed: All connection attempts failed",
        )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.body,
            b'{"type":"error","error":{"type":"api_error","message":"Upstream connection failed: All connection attempts failed"}}',
        )

    def test_forced_agent_responses_requests_stay_agent(self):
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
        )

        self.assertEqual(headers["X-Initiator"], "agent")
        self.assertEqual(body["input"], "hello")

    def test_haiku_requests_are_always_agent(self):
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/responses"), headers={})
        body = {
            "model": "claude-haiku-4.5",
            "input": "hello",
        }

        headers = format_translation.build_responses_headers_for_request(request, body, "test-key")

        self.assertEqual(headers["X-Initiator"], "agent")

    def test_resolve_copilot_model_name_maps_dated_haiku_to_canonical_form(self):
        self.assertEqual(
            format_translation.resolve_copilot_model_name("claude-haiku-4-5-20251001"),
            "claude-haiku-4.5",
        )

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

    def test_enabling_codex_proxy_is_idempotent_when_already_enabled(self):
        fd, raw_path = tempfile.mkstemp(prefix="codex-config-", suffix=".toml", dir=str(Path.cwd()))
        os.close(fd)
        config_path = Path(raw_path)
        try:
            config_path.write_text(proxy.CODEX_PROXY_CONFIG + "\n", encoding="utf-8")

            with mock.patch.object(proxy, "CODEX_CONFIG_FILE", str(config_path)):
                status = proxy.write_codex_proxy_config()

            backups = list(config_path.parent.glob(f"{config_path.name}.ghcp-proxy.bak.*"))
            self.assertTrue(status["configured"])
            self.assertEqual(status["status_message"], "proxy already enabled")
            self.assertEqual(backups, [])
        finally:
            config_path.unlink(missing_ok=True)
            for backup_path in config_path.parent.glob(f"{config_path.name}.ghcp-proxy.bak.*"):
                backup_path.unlink(missing_ok=True)

    def test_disabling_codex_proxy_is_idempotent_when_already_disabled(self):
        fd, raw_path = tempfile.mkstemp(prefix="codex-config-", suffix=".toml", dir=str(Path.cwd()))
        os.close(fd)
        config_path = Path(raw_path)
        config_path.unlink(missing_ok=True)
        try:

            with mock.patch.object(proxy, "CODEX_CONFIG_FILE", str(config_path)):
                status = proxy.disable_codex_proxy_config()

            self.assertFalse(status["configured"])
            self.assertEqual(status["status_message"], "proxy already disabled")
            self.assertIsNone(status["backup_path"])
        finally:
            config_path.unlink(missing_ok=True)

    def test_disabling_codex_proxy_restores_latest_backup(self):
        fd, raw_path = tempfile.mkstemp(prefix="codex-config-", suffix=".toml", dir=str(Path.cwd()))
        os.close(fd)
        config_path = Path(raw_path)
        try:
            backup_path = Path(f"{config_path}.ghcp-proxy.bak.20260404_180000")
            config_path.write_text(proxy.CODEX_PROXY_CONFIG + "\n", encoding="utf-8")
            backup_contents = 'model_provider = "openai"\n'
            backup_path.write_text(backup_contents, encoding="utf-8")

            with mock.patch.object(proxy, "CODEX_CONFIG_FILE", str(config_path)):
                status = proxy.disable_codex_proxy_config()

            self.assertFalse(status["configured"])
            self.assertTrue(status["restored_from_backup"])
            self.assertFalse(backup_path.exists())
            self.assertEqual(config_path.read_text(encoding="utf-8"), backup_contents)
        finally:
            config_path.unlink(missing_ok=True)
            Path(f"{config_path}.ghcp-proxy.bak.20260404_180000").unlink(missing_ok=True)

    def test_claude_proxy_status_requires_token_caps(self):
        fd, raw_path = tempfile.mkstemp(prefix="claude-settings-", suffix=".json", dir=str(Path.cwd()))
        os.close(fd)
        settings_path = Path(raw_path)
        try:
            settings_path.write_text(
                (
                    '{\n'
                    '  "env": {\n'
                    '    "ANTHROPIC_BASE_URL": "http://localhost:8000",\n'
                    '    "ANTHROPIC_AUTH_TOKEN": "sk-dummy",\n'
                    '    "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1"\n'
                    '  }\n'
                    '}\n'
                ),
                encoding="utf-8",
            )

            with mock.patch.object(proxy, "CLAUDE_SETTINGS_FILE", str(settings_path)):
                status = proxy.claude_proxy_status()

            self.assertFalse(status["configured"])
            self.assertEqual(status["status_message"], "proxy configured, missing context cap and output cap")
        finally:
            settings_path.unlink(missing_ok=True)
            for backup_path in settings_path.parent.glob(f"{settings_path.name}.ghcp-proxy.bak.*"):
                backup_path.unlink(missing_ok=True)

    def test_write_claude_proxy_settings_preserves_existing_keys_and_adds_context_cap(self):
        fd, raw_path = tempfile.mkstemp(prefix="claude-settings-", suffix=".json", dir=str(Path.cwd()))
        os.close(fd)
        settings_path = Path(raw_path)
        try:
            settings_path.write_text(
                (
                    '{\n'
                    '  "env": {\n'
                    '    "ANTHROPIC_BASE_URL": "http://localhost:8000",\n'
                    '    "ANTHROPIC_AUTH_TOKEN": "sk-dummy",\n'
                    '    "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1"\n'
                    '  },\n'
                    '  "skipDangerousModePermissionPrompt": true,\n'
                    '  "model": "opus"\n'
                    '}\n'
                ),
                encoding="utf-8",
            )

            with mock.patch.object(proxy, "CLAUDE_SETTINGS_FILE", str(settings_path)):
                status = proxy.write_claude_proxy_settings()

            written = proxy.json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertTrue(status["configured"])
            self.assertEqual(
                written["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"],
                proxy.CLAUDE_MAX_CONTEXT_TOKENS,
            )
            self.assertEqual(
                written["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"],
                proxy.CLAUDE_MAX_OUTPUT_TOKENS,
            )
            self.assertTrue(written["skipDangerousModePermissionPrompt"])
            self.assertEqual(written["model"], "opus")
            self.assertEqual(written["effortLevel"], "medium")
        finally:
            settings_path.unlink(missing_ok=True)
            for backup_path in settings_path.parent.glob(f"{settings_path.name}.ghcp-proxy.bak.*"):
                backup_path.unlink(missing_ok=True)

    def test_month_key_for_source_rows(self):
        self.assertEqual(util.month_key_for_source_row("claude", {"month": "2026-04"}), "2026-04")
        self.assertEqual(util.month_key_for_source_row("codex", {"month": "Apr 2026"}), "2026-04")


    def test_build_dashboard_payload_combines_claude_and_codex(self):
        fixed_now = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        usage_events = [
            {
                "request_id": "claude-req",
                "session_id": "claude-session",
                "server_request_id": "claude-chain",
                "started_at": "2026-04-04T17:50:00+00:00",
                "finished_at": "2026-04-04T17:51:00+00:00",
                "resolved_model": "claude-sonnet-4.6",
                "premium_requests": 1.0,
                "path": "/v1/responses",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cached_input_tokens": 30,
                    "cache_creation_input_tokens": 10,
                    "total_tokens": 160,
                },
                "cost_usd": 1.25,
            },
            {
                "request_id": "codex-req",
                "session_id": "codex-session",
                "server_request_id": "codex-chain",
                "started_at": "2026-04-04T17:55:00+00:00",
                "finished_at": "2026-04-04T17:56:00+00:00",
                "resolved_model": "gpt-5.4",
                "premium_requests": 0.33,
                "path": "/v1/responses",
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 40,
                    "cached_input_tokens": 50,
                    "reasoning_output_tokens": 12,
                    "total_tokens": 240,
                },
                "cost_usd": 2.75,
            },
        ]

        with proxy.usage_tracker.state.usage_log_lock:
            proxy.usage_tracker.state.archived_usage_events.clear()
            proxy.usage_tracker.state.recent_usage_events.clear()
            proxy.usage_tracker.state.recent_usage_events.extend(usage_events)

        with (
            mock.patch.object(proxy.dashboard_service, "utc_now", return_value=fixed_now),
            mock.patch.object(auth, "load_api_key_payload", return_value={"sku": "plus_monthly_subscriber_quota"}),
            mock.patch.object(
                proxy.dashboard_service,
                "get_official_premium_payload",
                return_value={
                    "available": True,
                    "remaining": 1420,
                    "used": 80,
                    "included": 1500,
                    "reset_date": None,
                    "source": "github-rest-billing-api",
                    "raw": {},
                    "refreshing": False,
                    "error": None,
                },
            ),
        ):
            payload = proxy.dashboard_service.build_payload()

        self.assertEqual(payload["premium"]["included"], 1500)
        self.assertEqual(payload["premium"]["used"], 80)
        self.assertEqual(payload["premium"]["official_remaining"], 1420)
        self.assertEqual(payload["current_month"]["usage"]["cost_usd"], 4.0)
        self.assertEqual(payload["current_month"]["usage"]["total_tokens"], 400)
        self.assertEqual(payload["recent_sessions"][0]["source"], "codex")
        self.assertEqual(payload["recent_sessions"][1]["source"], "claude")

    def test_build_dashboard_payload_keeps_archived_totals_and_recent_requests_separate(self):
        fixed_now = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        archived_event = {
            "request_id": "archived-req",
            "session_id": "session-older",
            "server_request_id": "chain-older",
            "started_at": "2026-03-30T18:00:00+00:00",
            "finished_at": "2026-03-30T18:01:00+00:00",
            "resolved_model": "gpt-5.4",
            "path": "/v1/responses",
            "premium_requests": 0.33,
            "usage": {
                "input_tokens": 300,
                "output_tokens": 75,
                "cached_input_tokens": 25,
                "total_tokens": 375,
            },
            "cost_usd": 3.5,
        }
        recent_event = {
            "request_id": "recent-req",
            "session_id": "session-newer",
            "server_request_id": "chain-newer",
            "started_at": "2026-04-04T17:55:00+00:00",
            "finished_at": "2026-04-04T17:56:00+00:00",
            "resolved_model": "gpt-5.4",
            "path": "/v1/responses",
            "premium_requests": 0.33,
            "usage": {
                "input_tokens": 200,
                "output_tokens": 40,
                "cached_input_tokens": 50,
                "total_tokens": 240,
            },
            "cost_usd": 2.75,
        }

        with proxy.usage_tracker.state.usage_log_lock:
            proxy.usage_tracker.state.archived_usage_events.clear()
            proxy.usage_tracker.state.recent_usage_events.clear()
            proxy.usage_tracker.state.archived_usage_events.append(archived_event)
            proxy.usage_tracker.state.recent_usage_events.append(recent_event)

        with (
            mock.patch.object(proxy.dashboard_service, "utc_now", return_value=fixed_now),
            mock.patch.object(auth, "load_api_key_payload", return_value={"sku": "plus_monthly_subscriber_quota"}),
            mock.patch.object(
                proxy.dashboard_service,
                "get_official_premium_payload",
                return_value={
                    "available": False,
                    "remaining": None,
                    "used": None,
                    "included": None,
                    "reset_date": None,
                    "source": "github-rest-billing-api",
                    "raw": {},
                    "refreshing": False,
                    "error": None,
                },
            ),
        ):
            payload = proxy.dashboard_service.build_payload()

        self.assertEqual(payload["all_time"]["proxy_requests"], 2)
        self.assertEqual(payload["all_time"]["archived_requests"], 1)
        self.assertEqual(payload["all_time"]["detailed_requests"], 1)
        self.assertEqual(payload["all_time"]["usage"]["cost_usd"], 6.25)
        self.assertEqual(payload["all_time"]["usage"]["total_tokens"], 615)
        self.assertEqual(payload["current_month"]["usage"]["cost_usd"], 2.75)
        self.assertEqual(len(payload["recent_requests"]), 1)
        self.assertEqual(payload["recent_requests"][0]["request_id"], "recent-req")
        self.assertEqual(payload["month_history"][0]["month_key"], "2026-04")
        self.assertEqual(payload["month_history"][1]["month_key"], "2026-03")

    def test_load_usage_history_compacts_old_requests_into_archive(self):
        events = []
        for idx in range(proxy.DETAILED_REQUEST_HISTORY_LIMIT + 2):
            minute = idx % 60
            events.append({
                "request_id": f"req-{idx:04d}",
                "session_id": f"session-{idx:04d}",
                "server_request_id": f"chain-{idx:04d}",
                "started_at": f"2026-04-04T18:{minute:02d}:00+00:00",
                "finished_at": f"2026-04-04T18:{minute:02d}:30+00:00",
                "resolved_model": "gpt-5.4",
                "path": "/v1/responses",
                "premium_requests": 0.33,
                "usage": {
                    "input_tokens": 100 + idx,
                    "output_tokens": 25,
                    "cached_input_tokens": 10,
                    "total_tokens": 125 + idx,
                },
                "cost_usd": round(0.01 * (idx + 1), 4),
            })

        log_path = self._make_usage_log_path(prefix="usage-history-")
        log_path.write_text(
            "".join(
                f"{usage_tracking.json.dumps(event, separators=(',', ':'), default=usage_tracking._json_default)}\n"
                for event in events
            ),
            encoding="utf-8",
        )
        db_uri = "file:usage-archive-test?mode=memory&cache=shared"
        keeper = proxy.sqlite3.connect(db_uri, uri=True)
        keeper.row_factory = proxy.sqlite3.Row
        keeper.execute(
            """
            CREATE TABLE archived_usage_events (
                archive_key TEXT PRIMARY KEY,
                recorded_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        keeper.commit()

        def connect_shared():
            connection = proxy.sqlite3.connect(db_uri, uri=True)
            connection.row_factory = proxy.sqlite3.Row
            return connection

        tracker = self._make_usage_tracker(
            archive_store=usage_tracking.UsageArchiveStore(
                init_storage=lambda: True,
                lock=Lock(),
                connect=lambda: connect_shared(),
                mark_unavailable=lambda error: None,
            ),
        )

        with (
            mock.patch.object(usage_tracking, "USAGE_LOG_FILE", str(log_path)),
            mock.patch.object(usage_tracking, "rewrite_usage_log_locked") as rewrite_usage_log,
        ):
            tracker.load_history()

        self.assertEqual(len(tracker.state.recent_usage_events), proxy.DETAILED_REQUEST_HISTORY_LIMIT)
        self.assertEqual(len(tracker.state.archived_usage_events), 2)
        self.assertEqual(tracker.state.archived_usage_events[0]["request_id"], "req-0000")
        self.assertEqual(tracker.state.archived_usage_events[1]["request_id"], "req-0001")
        self.assertEqual(tracker.state.recent_usage_events[0]["request_id"], "req-0002")
        rewrite_usage_log.assert_called_once()
        self.assertEqual(len(rewrite_usage_log.call_args.args[0]), proxy.DETAILED_REQUEST_HISTORY_LIMIT)
        archived_count = keeper.execute("SELECT COUNT(*) FROM archived_usage_events").fetchone()[0]
        self.assertEqual(archived_count, 2)
        keeper.close()

    def test_dashboard_api_refresh_param_forces_refresh_and_disables_http_caching(self):
        request = SimpleNamespace(query_params={"refresh": "1"})
        mocked_to_thread = mock.AsyncMock(return_value={"ok": True})

        with mock.patch.object(proxy.asyncio, "to_thread", mocked_to_thread):
            response = proxy.asyncio.run(proxy.dashboard_api(request))

        mocked_to_thread.assert_awaited_once_with(proxy.dashboard_service.build_payload, True)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_trigger_official_premium_refresh_notifies_dashboard_stream_listeners(self):
        class ImmediateThread:
            def __init__(self, target=None, daemon=None):
                self._target = target
                self.daemon = daemon

            def start(self):
                if self._target is not None:
                    self._target()

        premium_payload = {
            "available": True,
            "loaded_at": "2026-04-04T18:00:00+00:00",
            "remaining": 1420,
            "used": 80,
            "included": 1500,
            "reset_date": None,
            "source": "github-rest-billing-api",
            "raw": {},
            "error": None,
        }

        with (
            mock.patch.object(
                proxy.dashboard_service,
                "collect_official_premium_payload",
                return_value=premium_payload,
            ),
            mock.patch.object(proxy.dashboard_service, "sqlite_cache_put"),
            mock.patch.object(proxy.dashboard_service, "notify_dashboard_stream_listeners") as notify,
            mock.patch.object(proxy.dashboard_service, "thread_class", ImmediateThread),
        ):
            proxy.dashboard_service.trigger_official_premium_refresh()

        notify.assert_called_once_with()
        cache_state = proxy.dashboard_service.official_premium_cache_state()
        self.assertEqual(cache_state["payload"], premium_payload)
        self.assertFalse(cache_state["refreshing"])
        self.assertIsNone(cache_state["last_error"])
        self.assertGreater(cache_state["loaded_at"], 0.0)

    def test_normalize_session_claude_accepts_cached_input_tokens_shape(self):
        normalized = dashboard.normalize_session(
            "claude",
            {
                "sessionId": "claude-session",
                "sessionKind": "session",
                "sessionDisplayId": "claude-session",
                "lastActivity": "2026-04-05T01:32:10Z",
                "inputTokens": 27700,
                "outputTokens": 212,
                "cachedInputTokens": 102300,
                "totalTokens": 27912,
                "costUSD": 1.25,
                "models": {"claude-sonnet-4-6": {"inputTokens": 27700}},
            },
        )

        self.assertEqual(normalized["cached_input_tokens"], 102300)
        self.assertEqual(normalized["cost_usd"], 1.25)
        self.assertEqual(normalized["models"], ["claude-sonnet-4-6"])

    def test_collect_local_dashboard_usage_prefers_request_id_rows_over_chain_rows(self):
        usage = dashboard.collect_local_dashboard_usage(
            [
                {
                    "request_id": "claude-req",
                    "server_request_id": "chain-123",
                    "started_at": "2026-04-05T01:30:00+00:00",
                    "finished_at": "2026-04-05T01:31:00+00:00",
                    "resolved_model": "claude-sonnet-4.6",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cached_input_tokens": 30,
                        "total_tokens": 120,
                    },
                }
            ]
        )

        row = usage["recent_sessions"][0]
        self.assertIsNone(row["session_id"])
        self.assertEqual(row["session_kind"], "session")
        self.assertEqual(row["session_display_id"], "claude-req")


if __name__ == "__main__":
    unittest.main()
