import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import auth
import dashboard
import initiator_policy
import proxy
import usage_tracking
import util


class DashboardTests(unittest.TestCase):
    def setUp(self):
        proxy.set_initiator_policy(initiator_policy.InitiatorPolicy())
        proxy.usage_tracker.clear_state()
        proxy.dashboard_service.reset_official_premium_cache()

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

        proxy.usage_tracker.replace_history(recent_events=usage_events)

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

        proxy.usage_tracker.replace_history(
            recent_events=[recent_event],
            archived_events=[archived_event],
        )

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
