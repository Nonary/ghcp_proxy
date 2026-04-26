import json
import sqlite3
import unittest
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import proxy
import usage_tracking


class UsageTrackingTests(unittest.TestCase):
    def setUp(self):
        proxy.usage_tracker.clear_state()

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
        self.assertEqual(outbound_headers["x-interaction-id"], "session-123")
        self.assertEqual(outbound_headers["x-request-id"], event["server_request_id"])
        self.assertEqual(outbound_headers["x-github-request-id"], event["server_request_id"])
        self.assertEqual(outbound_headers["x-agent-task-id"], event["server_request_id"])

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
        self.assertNotIn("session_id", outbound_headers)
        self.assertNotIn("x-interaction-id", outbound_headers)
        self.assertNotIn("x-agent-task-id", outbound_headers)
        self.assertNotIn("x-request-id", outbound_headers)
        self.assertNotIn("x-github-request-id", outbound_headers)

    def test_normalize_recorded_usage_event_backfills_codex_native_session_and_chain(self):
        normalized = usage_tracking._normalize_recorded_usage_event(
            {
                "request_id": "codex-native:session-native:49",
                "path": "/native/codex/responses",
                "requested_model": "gpt-5.4",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cached_input_tokens": 20,
                    "total_tokens": 130,
                },
            }
        )

        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["native_source"], "codex_native")
        self.assertEqual(normalized["session_id"], "session-native")
        self.assertEqual(normalized["session_id_origin"], "codex_native_request_id")
        self.assertEqual(normalized["server_request_id"], "session-native")

    def test_normalize_recorded_usage_event_backfills_requested_fast_tier_and_doubles_gpt54_cost(self):
        with mock.patch(
            "usage_tracking._codex_logs_service_tiers",
            return_value={
                "requested": "priority",
                "requested_source": "codex_logs_request",
                "effective": "default",
                "effective_source": "codex_logs_response_completed",
            },
        ):
            normalized = usage_tracking._normalize_recorded_usage_event(
                {
                    "request_id": "codex-native:session-fast:1",
                    "path": "/native/codex/responses",
                    "requested_model": "gpt-5.4",
                    "native_source": "codex_native",
                    "native_turn_id": "turn-fast-1",
                    "native_reasoning_effort": "high",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "cached_input_tokens": 20,
                        "total_tokens": 130,
                    },
                    "cost_usd": 0.0,
                }
            )

        baseline = usage_tracking._usage_event_cost(
            "gpt-5.4",
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "cached_input_tokens": 20,
                "total_tokens": 130,
            },
        )
        self.assertEqual(normalized["native_requested_service_tier"], "priority")
        self.assertEqual(normalized["native_requested_service_tier_source"], "codex_logs_request")
        self.assertEqual(normalized["native_service_tier"], "default")
        self.assertEqual(normalized["native_service_tier_source"], "codex_logs_response_completed")
        self.assertAlmostEqual(normalized["cost_usd"], baseline * 2, places=8)

    def test_normalize_recorded_usage_event_backfills_requested_fast_tier_and_multiplies_gpt55_cost(self):
        with mock.patch(
            "usage_tracking._codex_logs_service_tiers",
            return_value={
                "requested": "priority",
                "requested_source": "codex_logs_request",
                "effective": "default",
                "effective_source": "codex_logs_response_completed",
            },
        ):
            normalized = usage_tracking._normalize_recorded_usage_event(
                {
                    "request_id": "codex-native:session-fast-55:1",
                    "path": "/native/codex/responses",
                    "requested_model": "gpt-5.5",
                    "native_source": "codex_native",
                    "native_turn_id": "turn-fast-55-1",
                    "native_reasoning_effort": "high",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "cached_input_tokens": 20,
                        "total_tokens": 130,
                    },
                    "cost_usd": 0.0,
                }
            )

        baseline = usage_tracking._usage_event_cost(
            "gpt-5.5",
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "cached_input_tokens": 20,
                "total_tokens": 130,
            },
        )
        self.assertEqual(normalized["native_requested_service_tier"], "priority")
        self.assertEqual(normalized["native_requested_service_tier_source"], "codex_logs_request")
        self.assertEqual(normalized["native_service_tier"], "default")
        self.assertEqual(normalized["native_service_tier_source"], "codex_logs_response_completed")
        self.assertAlmostEqual(normalized["cost_usd"], baseline * 2.5, places=8)

    def test_finish_usage_event_counts_gpt55_as_seven_and_half_premium_requests(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        event = {
            "request_id": "req-gpt55-premium",
            "initiator": "user",
            "resolved_model": "gpt-5.5",
        }

        tracker.usage_log_file = str(log_path)
        tracker.finish_event(event, 200)

        finished = tracker.snapshot_usage_events()[0]
        self.assertEqual(finished["premium_requests"], 7.5)

    def test_normalize_recorded_usage_event_drops_proxied_codex_native_rows(self):
        normalized = usage_tracking._normalize_recorded_usage_event(
            {
                "request_id": "codex-native:session-custom:1",
                "path": "/native/codex/responses",
                "native_source": "codex_native",
                "native_model_provider": "custom",
                "requested_model": "gpt-5.4",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "cached_input_tokens": 20,
                    "total_tokens": 110,
                },
            }
        )

        self.assertIsNone(normalized)

    def test_normalize_recorded_usage_event_overrides_stale_native_tier_with_codex_logs(self):
        with mock.patch(
            "usage_tracking._codex_logs_service_tiers",
            return_value={
                "effective": "default",
                "effective_source": "codex_logs_response_completed",
            },
        ):
            normalized = usage_tracking._normalize_recorded_usage_event(
                {
                    "request_id": "codex-native:session-default:1",
                    "path": "/native/codex/responses",
                    "requested_model": "gpt-5.4",
                    "native_source": "codex_native",
                    "native_turn_id": "turn-default-1",
                    "native_service_tier": "fast",
                    "native_reasoning_effort": "high",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "cached_input_tokens": 20,
                        "total_tokens": 130,
                    },
                    "cost_usd": 999.0,
                }
            )

        baseline = usage_tracking._usage_event_cost(
            "gpt-5.4",
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "cached_input_tokens": 20,
                "total_tokens": 130,
            },
        )
        self.assertEqual(normalized["native_service_tier"], "default")
        self.assertEqual(normalized["native_service_tier_source"], "codex_logs_response_completed")
        self.assertAlmostEqual(normalized["cost_usd"], baseline, places=8)

    def test_normalize_recorded_usage_event_clears_stale_native_tiers_when_logs_are_missing(self):
        with mock.patch("usage_tracking._codex_logs_service_tiers", return_value={}):
            normalized = usage_tracking._normalize_recorded_usage_event(
                {
                    "request_id": "codex-native:session-missing:1",
                    "path": "/native/codex/responses",
                    "requested_model": "gpt-5.4",
                    "native_source": "codex_native",
                    "native_turn_id": "turn-missing-1",
                    "native_requested_service_tier": "priority",
                    "native_requested_service_tier_source": "config",
                    "native_service_tier": "fast",
                    "native_reasoning_effort": "high",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "cached_input_tokens": 20,
                        "total_tokens": 130,
                    },
                    "cost_usd": 999.0,
                }
            )

        self.assertNotIn("native_requested_service_tier", normalized)
        self.assertNotIn("native_requested_service_tier_source", normalized)
        self.assertNotIn("native_service_tier", normalized)
        self.assertNotIn("native_service_tier_source", normalized)

    def test_start_usage_event_uses_outbound_client_request_id_when_present(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        outbound_headers = {"x-client-request-id": "session-123"}

        event = tracker.start_event(
            request,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            initiator="agent",
            request_body={"sessionId": "session-123"},
            outbound_headers=outbound_headers,
        )

        self.assertEqual(event["session_id"], "session-123")
        self.assertEqual(event["client_request_id"], "session-123")

    def test_start_usage_event_logs_canonical_requested_model_without_duplicate_resolved_model(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )

        with mock.patch("builtins.print") as print_mock:
            tracker.start_event(
                request,
                requested_model="claude-opus-4-7",
                resolved_model="claude-opus-4.7",
                initiator="agent",
            )

        print_mock.assert_called_once()
        logged_line = print_mock.call_args.args[0]
        self.assertIn("requested_model=claude-opus-4.7", logged_line)
        self.assertNotIn("resolved_model=", logged_line)

    def test_start_usage_event_logs_resolved_model_when_real_remap_occurs(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={},
        )

        with mock.patch("builtins.print") as print_mock:
            tracker.start_event(
                request,
                requested_model="gpt-5.4",
                resolved_model="claude-opus-4.7",
                initiator="agent",
            )

        print_mock.assert_called_once()
        logged_line = print_mock.call_args.args[0]
        self.assertIn("requested_model=gpt-5.4", logged_line)
        self.assertIn("resolved_model=claude-opus-4.7", logged_line)

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

    def test_start_usage_event_ignores_plain_claude_metadata_user_id_as_session_id(self):
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
            resolved_model="gpt-5.4",
            initiator="user",
            request_body={"metadata": {"user_id": "plain-claude-session"}},
            outbound_headers=outbound_headers,
        )

        self.assertEqual(event["session_id"], event["request_id"])
        self.assertEqual(event["session_id_origin"], "request_id")
        self.assertEqual(outbound_headers["session_id"], event["request_id"])

    def test_start_usage_event_uses_metadata_session_id(self):
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
            resolved_model="gpt-5.4",
            initiator="user",
            request_body={"metadata": {"session_id": "metadata-session"}},
            outbound_headers=outbound_headers,
        )

        self.assertEqual(event["session_id"], "metadata-session")
        self.assertEqual(event["session_id_origin"], "request")
        self.assertEqual(outbound_headers["session_id"], "metadata-session")

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
        self.assertNotEqual(follow_up_event["server_request_id"], first_event["server_request_id"])

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
        self.assertNotEqual(agent_event["server_request_id"], user_event["server_request_id"])

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

        tracker.usage_log_file = str(log_path)
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
        self.assertNotEqual(agent_event["server_request_id"], user_event["server_request_id"])

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

    def test_start_usage_event_agent_links_latest_session_server_request_id(self):
        tracker = self._make_usage_tracker()
        tracker.remember_latest_server_request_id("session-123", None, None, "server-prev")

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
            outbound_headers={},
        )

        self.assertEqual(event["prior_server_request_id"], "server-prev")
        self.assertNotEqual(event["server_request_id"], "server-prev")

    def test_start_usage_event_responses_uses_stable_affinity_when_linking_latest_session(self):
        tracker = self._make_usage_tracker()
        tracker.remember_latest_server_request_id("session-123", None, None, "server-prev")
        outbound_headers = {}

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
            outbound_headers=outbound_headers,
        )

        self.assertEqual(event["prior_server_request_id"], "server-prev")
        self.assertNotIn("session_id", outbound_headers)
        self.assertNotIn("x-interaction-id", outbound_headers)
        self.assertNotIn("x-agent-task-id", outbound_headers)
        self.assertNotIn("x-request-id", outbound_headers)
        self.assertNotIn("x-github-request-id", outbound_headers)

    def test_start_usage_event_uses_forwarded_server_request_id_as_prior_only(self):
        tracker = self._make_usage_tracker()
        outbound_headers = {"x-request-id": "server-prev"}
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={"session_id": "session-123", "x-request-id": "server-prev"},
        )

        event = tracker.start_event(
            request,
            requested_model="gpt-5.4",
            resolved_model="gpt-5.4",
            initiator="agent",
            outbound_headers=outbound_headers,
        )

        self.assertEqual(event["prior_server_request_id"], "server-prev")
        self.assertNotEqual(event["server_request_id"], "server-prev")
        self.assertNotIn("session_id", outbound_headers)
        self.assertNotIn("x-interaction-id", outbound_headers)
        self.assertNotIn("x-agent-task-id", outbound_headers)
        self.assertNotIn("x-request-id", outbound_headers)
        self.assertNotIn("x-github-request-id", outbound_headers)

    def test_start_usage_event_agent_links_latest_body_session_server_request_id(self):
        tracker = self._make_usage_tracker()
        tracker.remember_latest_server_request_id("session-123", None, None, "server-prev")

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
        self.assertNotEqual(event["server_request_id"], "server-prev")

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
        self.assertNotEqual(agent_event["server_request_id"], user_event["server_request_id"])

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

    def test_start_usage_event_keeps_request_id_for_responses_to_messages_upstream(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/responses"),
            method="POST",
            headers={},
        )
        outbound_headers = {}

        event = tracker.start_event(
            request,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            initiator="user",
            upstream_path="/v1/messages",
            outbound_headers=outbound_headers,
        )

        self.assertEqual(outbound_headers["x-request-id"], event["server_request_id"])
        self.assertEqual(outbound_headers["x-github-request-id"], event["server_request_id"])
        self.assertEqual(outbound_headers["x-agent-task-id"], event["server_request_id"])

    def test_start_usage_event_preserves_native_messages_agent_task_affinity(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"x-claude-code-session-id": "claude-session"},
        )
        outbound_headers = {"x-agent-task-id": "stable-task"}

        event = tracker.start_event(
            request,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            initiator="agent",
            upstream_path="/v1/messages",
            outbound_headers=outbound_headers,
        )

        self.assertEqual(outbound_headers["x-agent-task-id"], event["server_request_id"])
        self.assertEqual(outbound_headers["x-request-id"], event["server_request_id"])
        self.assertEqual(outbound_headers["x-github-request-id"], event["server_request_id"])
        self.assertEqual(outbound_headers["x-interaction-id"], "claude-session")

    def test_start_usage_event_preserves_native_messages_interaction_affinity(self):
        tracker = self._make_usage_tracker()
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/messages"),
            method="POST",
            headers={"x-claude-code-session-id": "broad-claude-session"},
        )
        outbound_headers = {
            "x-interaction-id": "request-scoped-interaction",
            "x-agent-task-id": "request-scoped-task",
        }

        event = tracker.start_event(
            request,
            requested_model="claude-sonnet-4.6",
            resolved_model="claude-sonnet-4.6",
            initiator="user",
            upstream_path="/v1/messages",
            outbound_headers=outbound_headers,
        )

        self.assertEqual(outbound_headers["session_id"], "broad-claude-session")
        self.assertEqual(outbound_headers["x-interaction-id"], "request-scoped-interaction")
        self.assertEqual(outbound_headers["x-agent-task-id"], event["server_request_id"])
        self.assertEqual(outbound_headers["x-request-id"], event["server_request_id"])

    def test_finish_usage_event_tracks_usage_and_timing(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        event = {
            "request_id": "req-999",
            "initiator": "user",
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

        tracker.usage_log_file = str(log_path)
        with mock.patch.object(usage_tracking.time, "perf_counter", return_value=12.0):
            tracker.finish_event(
                event,
                200,
                upstream=upstream,
                response_payload=response_payload,
            )

        finished = tracker.snapshot_usage_events()[0]
        self.assertEqual(finished["response_id"], "resp_123")
        self.assertEqual(finished["upstream_request_id"], "server-abc")
        self.assertEqual(finished["usage"]["input_tokens"], 11)
        self.assertEqual(finished["duration_ms"], 2000)
        self.assertEqual(finished["time_to_first_token_ms"], 450)
        self.assertEqual(finished["premium_requests"], 1.0)
        self.assertTrue(finished["success"])

    def test_finish_usage_event_parses_x_usage_ratelimit_headers(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        event = {
            "request_id": "req-rl-1",
            "initiator": "user",
            "resolved_model": "gpt-5-mini",
        }
        upstream = SimpleNamespace(
            headers={
                "content-type": "application/json",
                # URL-encoded values exactly as Copilot CLI's `/responses` upstream emits them.
                "x-usage-ratelimit-session": "ent=0&ov=0.0&ovPerm=false&rem=93.2&rst=2026-04-21T06%3A22%3A37Z",
                "x-usage-ratelimit-weekly": "ent=0&ov=0.0&ovPerm=true&rem=99.0&rst=2026-04-27T00%3A00%3A00Z",
            }
        )

        tracker.usage_log_file = str(log_path)
        tracker.finish_event(event, 200, upstream=upstream, response_payload={"id": "resp_rl"})

        finished = tracker.snapshot_usage_events()[0]
        windows = finished["usage_ratelimits"]
        self.assertIn("session", windows)
        self.assertIn("weekly", windows)
        self.assertEqual(windows["session"]["percent_remaining"], 93.2)
        self.assertEqual(windows["session"]["percent_used"], 6.8)
        self.assertEqual(windows["session"]["overage"], 0.0)
        self.assertFalse(windows["session"]["overage_permitted"])
        self.assertEqual(windows["session"]["reset_at"], "2026-04-21T06:22:37Z")
        self.assertEqual(windows["weekly"]["percent_remaining"], 99.0)
        self.assertTrue(windows["weekly"]["overage_permitted"])
        self.assertEqual(windows["weekly"]["reset_at"], "2026-04-27T00:00:00Z")

    def test_finish_usage_event_does_not_count_agent_request_toward_premium_quota(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        event = {
            "request_id": "req-999",
            "initiator": "agent",
            "resolved_model": "gpt-5.4",
        }

        tracker.usage_log_file = str(log_path)
        tracker.finish_event(event, 200)

        finished = tracker.snapshot_usage_events()[0]
        self.assertEqual(finished["premium_requests"], 0.0)
        self.assertEqual(finished["cost_usd"], 0.0)

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

        tracker.usage_log_file = str(log_path)
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

        tracker.usage_log_file = str(log_path)
        tracker.finish_event(event, 200, usage=usage)

        finished = tracker.snapshot_usage_events()[0]
        self.assertAlmostEqual(finished["cost_usd"], 17.75)

    def test_finish_usage_event_cost_uses_pricing_cache_fields(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        event = {
            "request_id": "req-999",
            "resolved_model": "claude-sonnet-4.6",
        }
        usage = {
            "input_tokens": 2_000_000,
            "output_tokens": 1_000_000,
            "cached_input_tokens": 1_000_000,
            "cache_creation_input_tokens": 1_000_000,
            "pricing_fresh_input_tokens": 1_000_000,
            "pricing_cached_input_tokens": 1_000_000,
            "pricing_cache_creation_input_tokens": 1_000_000,
        }

        tracker.usage_log_file = str(log_path)
        tracker.finish_event(event, 200, usage=usage)

        finished = tracker.snapshot_usage_events()[0]
        self.assertEqual(finished["usage"]["input_tokens"], 2_000_000)
        self.assertEqual(finished["usage"]["cached_input_tokens"], 1_000_000)
        self.assertEqual(finished["usage"]["cache_creation_input_tokens"], 1_000_000)
        self.assertEqual(finished["usage"]["pricing_fresh_input_tokens"], 1_000_000)
        self.assertEqual(finished["usage"]["pricing_cached_input_tokens"], 1_000_000)
        self.assertEqual(finished["usage"]["pricing_cache_creation_input_tokens"], 1_000_000)
        self.assertAlmostEqual(finished["cost_usd"], 22.05)

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
        self.assertEqual(capture.usage["input_tokens"], 5)
        self.assertEqual(capture.usage["output_tokens"], 2)
        self.assertEqual(capture.usage["total_tokens"], 7)
        self.assertEqual(capture.usage["cached_input_tokens"], 3)
        self.assertEqual(capture.usage["fresh_input_tokens"], 2)
        self.assertEqual(capture.usage["reasoning_output_tokens"], 1)

    def test_sse_usage_capture_normalizes_cached_prompt_tokens(self):
        tracker = self._make_usage_tracker()
        capture = tracker.create_sse_capture("chat")
        capture.feed(
            b'event: message\ndata: {"id":"chatcmpl_1","choices":[{"delta":{"content":"Hi"}}],"usage":{"prompt_tokens":20,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":7}}}\n\n'
        )

        self.assertEqual(capture.usage["input_tokens"], 20)
        self.assertEqual(capture.usage["output_tokens"], 4)
        self.assertEqual(capture.usage["total_tokens"], 24)
        self.assertEqual(capture.usage["cached_input_tokens"], 7)
        self.assertEqual(capture.usage["fresh_input_tokens"], 13)

    def test_load_usage_history_normalizes_cached_tokens(self):
        tracker = self._make_usage_tracker()
        log_path = self._make_usage_log_path()
        log_path.write_text(
            '{"session_id":"session-123","server_request_id":"server-abc","usage":{"prompt_tokens":20,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":7}}}\n',
            encoding="utf-8",
        )

        tracker.usage_log_file = str(log_path)
        tracker.load_history()
        events = tracker.snapshot_usage_events()

        self.assertEqual(events[0]["usage"]["input_tokens"], 20)
        self.assertEqual(events[0]["usage"]["total_tokens"], 24)
        self.assertEqual(events[0]["usage"]["cached_input_tokens"], 7)
        self.assertEqual(events[0]["usage"]["fresh_input_tokens"], 13)
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

        tracker.usage_log_file = str(log_path)
        tracker.load_history()
        events = tracker.snapshot_usage_events()

        self.assertEqual(events[0]["usage"]["input_tokens"], 20)
        self.assertEqual(events[0]["usage"]["total_tokens"], 24)
        self.assertEqual(events[0]["usage"]["cached_input_tokens"], 7)
        self.assertEqual(events[0]["usage"]["fresh_input_tokens"], 13)
        self.assertEqual(events[0]["usage"]["reasoning_output_tokens"], 2)
        self.assertEqual(
            tracker.latest_server_request_id("session-123", None, None),
            "server-def",
        )

    def test_normalize_usage_payload_preserves_explicit_cached_input_shape(self):
        import util
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

    def test_normalize_usage_payload_preserves_copilot_responses_totals(self):
        import util
        normalized = util.normalize_usage_payload(
            {
                "input_tokens": 20344,
                "output_tokens": 269,
                "total_tokens": 20613,
                "input_tokens_details": {"cached_tokens": 11776},
            }
        )

        self.assertEqual(normalized["input_tokens"], 20344)
        self.assertEqual(normalized["output_tokens"], 269)
        self.assertEqual(normalized["total_tokens"], 20613)
        self.assertEqual(normalized["cached_input_tokens"], 11776)
        self.assertEqual(normalized["fresh_input_tokens"], 8568)

    def test_load_usage_history_compacts_old_requests_into_archive(self):
        from constants import DETAILED_REQUEST_HISTORY_LIMIT

        events = []
        for idx in range(DETAILED_REQUEST_HISTORY_LIMIT + 2):
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
                f"{json.dumps(event, separators=(',', ':'))}\n"
                for event in events
            ),
            encoding="utf-8",
        )
        db_uri = "file:usage-archive-test?mode=memory&cache=shared"
        keeper = sqlite3.connect(db_uri, uri=True)
        keeper.row_factory = sqlite3.Row
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
            connection = sqlite3.connect(db_uri, uri=True)
            connection.row_factory = sqlite3.Row
            return connection

        tracker = self._make_usage_tracker(
            archive_store=usage_tracking.UsageArchiveStore(
                init_storage=lambda: True,
                lock=Lock(),
                connect=lambda: connect_shared(),
                mark_unavailable=lambda error: None,
            ),
        )

        tracker.usage_log_file = str(log_path)
        with mock.patch.object(tracker, "_rewrite_usage_log") as rewrite_usage_log:
            tracker.load_history()

        recent_events = tracker.snapshot_usage_events()
        archived_events = tracker.snapshot_archived_usage_events()
        self.assertEqual(len(recent_events), DETAILED_REQUEST_HISTORY_LIMIT)
        self.assertEqual(len(archived_events), 2)
        self.assertEqual(archived_events[0]["request_id"], "req-0000")
        self.assertEqual(archived_events[1]["request_id"], "req-0001")
        self.assertEqual(recent_events[0]["request_id"], "req-0002")
        rewrite_usage_log.assert_called_once()
        self.assertEqual(len(rewrite_usage_log.call_args.args[0]), DETAILED_REQUEST_HISTORY_LIMIT)
        archived_count = keeper.execute("SELECT COUNT(*) FROM archived_usage_events").fetchone()[0]
        self.assertEqual(archived_count, 2)
        keeper.close()

    def test_load_history_reuses_stored_codex_native_service_tiers(self):
        event = {
            "request_id": "codex-native:session-1:0",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:00+00:00",
            "path": "/native/codex/responses",
            "requested_model": "gpt-5",
            "resolved_model": "gpt-5",
            "native_source": "codex_native",
            "session_id": "session-1",
            "native_turn_id": "turn-1",
            "native_requested_service_tier": "priority",
            "native_requested_service_tier_source": "codex_logs_request",
            "native_service_tier": "priority",
            "native_service_tier_source": "codex_logs_response_completed",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        }

        log_path = self._make_usage_log_path(prefix="usage-history-native-")
        log_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
        tracker = self._make_usage_tracker()
        tracker.usage_log_file = str(log_path)

        with mock.patch.object(
            usage_tracking,
            "_codex_logs_service_tiers",
            side_effect=AssertionError("stored tiers should be reused"),
        ):
            tracker.load_history()

        [loaded] = tracker.snapshot_usage_events()
        self.assertEqual(loaded["native_requested_service_tier"], "priority")
        self.assertEqual(loaded["native_service_tier"], "priority")


if __name__ == "__main__":
    unittest.main()
