import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import dashboard
import initiator_policy
import proxy
import usage_tracking
import util


class DashboardTests(unittest.TestCase):
    def setUp(self):
        proxy.set_initiator_policy(initiator_policy.InitiatorPolicy())
        proxy.usage_tracker.clear_state()

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

        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                load_premium_plan_config=lambda: {
                    "configured": True,
                    "plan": "pro_plus",
                    "plan_label": "Pro+",
                    "included": 1500,
                    "synced_percent": 0.0,
                    "synced_used": 0.0,
                    "synced_at": "2026-04-01T00:00:00+00:00",
                    "synced_month": "2026-04",
                },
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()

        self.assertEqual(payload["premium"]["included"], 1500)
        self.assertEqual(payload["premium"]["used"], 1.33)
        self.assertEqual(payload["premium"]["remaining"], 1498.67)
        self.assertEqual(payload["premium"]["percent_used"], 0.09)
        self.assertEqual(payload["premium"]["tracked_since_sync"], 1.33)
        self.assertEqual(payload["current_month"]["usage"]["cost_usd"], 4.0)
        self.assertEqual(payload["current_month"]["usage"]["total_tokens"], 400)
        self.assertEqual(payload["current_month"]["usage"]["request_count"], 2)
        self.assertEqual(payload["all_time"]["usage"]["request_count"], 2)
        self.assertEqual(len(payload["current_month"]["daily_history"]), 4)
        self.assertEqual(payload["current_month"]["daily_history"][-1]["day_key"], "2026-04-04")
        self.assertEqual(payload["current_month"]["daily_history"][-1]["cost_usd"], 4.0)
        self.assertEqual(payload["recent_sessions"][0]["source"], "codex")
        self.assertEqual(payload["recent_sessions"][1]["source"], "claude")

    def test_build_dashboard_payload_includes_exact_claude_cost_breakdown(self):
        fixed_now = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        usage_events = [
            {
                "request_id": "claude-req",
                "session_id": "claude-session",
                "server_request_id": "claude-chain",
                "started_at": "2026-04-04T17:50:00+00:00",
                "finished_at": "2026-04-04T17:51:00+00:00",
                "requested_model": "claude-opus-4.7",
                "resolved_model": "claude-opus-4.7",
                "path": "/v1/messages",
                "usage": {
                    "input_tokens": 1_000,
                    "output_tokens": 1_000,
                    "cached_input_tokens": 1_000,
                    "total_tokens": 3_000,
                },
            },
        ]

        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                load_premium_plan_config=lambda: {},
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()
        usage = payload["current_month"]["usage"]
        claude_usage = usage["sources"]["claude"]

        self.assertAlmostEqual(usage["cost_usd"], 0.0305)
        self.assertAlmostEqual(usage["cost_breakdown"]["input_fresh"], 0.005)
        self.assertAlmostEqual(usage["cost_breakdown"]["cached_input"], 0.0005)
        self.assertAlmostEqual(usage["cost_breakdown"]["cache_creation"], 0.0)
        self.assertAlmostEqual(usage["cost_breakdown"]["output"], 0.025)
        self.assertAlmostEqual(claude_usage["cost_breakdown"]["input_fresh"], 0.005)
        self.assertAlmostEqual(claude_usage["cost_breakdown"]["cached_input"], 0.0005)
        self.assertAlmostEqual(claude_usage["cost_breakdown"]["output"], 0.025)

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

        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                load_premium_plan_config=lambda: {},
                snapshot_all_usage_events=lambda: [archived_event, recent_event],
                snapshot_usage_events=lambda: [recent_event],
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()

        self.assertEqual(payload["all_time"]["proxy_requests"], 2)
        self.assertEqual(payload["all_time"]["archived_requests"], 1)
        self.assertEqual(payload["all_time"]["detailed_requests"], 1)
        self.assertEqual(payload["all_time"]["usage"]["cost_usd"], 6.25)
        self.assertEqual(payload["all_time"]["usage"]["total_tokens"], 615)
        self.assertEqual(payload["all_time"]["usage"]["request_count"], 2)
        self.assertEqual(payload["current_month"]["usage"]["cost_usd"], 2.75)
        self.assertEqual(payload["current_month"]["daily_history"][-1]["day_key"], "2026-04-04")
        self.assertEqual(payload["current_month"]["daily_history"][-1]["request_count"], 1)
        self.assertEqual(len(payload["recent_requests"]), 1)
        self.assertEqual(payload["recent_requests"][0]["request_id"], "recent-req")
        self.assertEqual(payload["month_history"][0]["month_key"], "2026-04")
        self.assertEqual(payload["month_history"][1]["month_key"], "2026-03")
        self.assertIsNone(payload["premium"]["included"])
        self.assertEqual(payload["premium"]["used"], 0.33)
        self.assertIsNone(payload["premium"]["percent_used"])

    def test_build_dashboard_payload_zero_fills_daily_history_for_current_month(self):
        fixed_now = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        usage_events = [
            {
                "request_id": "day-two",
                "started_at": "2026-04-02T10:00:00+00:00",
                "finished_at": "2026-04-02T10:01:00+00:00",
                "resolved_model": "gpt-5.4",
                "path": "/v1/responses",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
                "cost_usd": 1.5,
            }
        ]

        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                load_premium_plan_config=lambda: {},
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()

        self.assertEqual(
            [row["day_key"] for row in payload["current_month"]["daily_history"]],
            ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04"],
        )
        self.assertEqual(payload["current_month"]["daily_history"][0]["cost_usd"], 0.0)
        self.assertEqual(payload["current_month"]["daily_history"][1]["cost_usd"], 1.5)
        self.assertEqual(payload["current_month"]["daily_history"][2]["request_count"], 0)

    def test_build_dashboard_payload_includes_safeguard_trigger_counts(self):
        fixed_now = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                load_api_key_payload=lambda: {},
                snapshot_all_usage_events=lambda: [],
                snapshot_usage_events=lambda: [],
                load_safeguard_trigger_stats=lambda _now: {
                    "today_count": 1,
                    "current_month_count": 2,
                    "all_time_count": 3,
                    "latest_triggered_at": "2026-04-04T17:30:00+00:00",
                },
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()

        self.assertEqual(payload["safeguard"]["today_count"], 1)
        self.assertEqual(payload["safeguard"]["current_month_count"], 2)
        self.assertEqual(payload["safeguard"]["all_time_count"], 3)
        self.assertEqual(payload["safeguard"]["latest_triggered_at"], "2026-04-04T17:30:00+00:00")

    def test_dashboard_api_refresh_param_forces_refresh_and_disables_http_caching(self):
        request = SimpleNamespace(query_params={"refresh": "1"})
        mocked_to_thread = mock.AsyncMock(return_value={"ok": True})

        with mock.patch.object(proxy.asyncio, "to_thread", mocked_to_thread):
            response = proxy.asyncio.run(proxy.dashboard_api(request))

        mocked_to_thread.assert_awaited_once_with(proxy.dashboard_service.build_payload, True)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_build_dashboard_payload_adds_tracked_requests_on_top_of_synced_percent(self):
        fixed_now = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        usage_events = [
            {
                "request_id": "before-sync",
                "initiator": "user",
                "started_at": "2026-04-04T17:40:00+00:00",
                "finished_at": "2026-04-04T17:41:00+00:00",
                "resolved_model": "gpt-5.4",
                "path": "/v1/responses",
                "premium_requests": 1.0,
            },
            {
                "request_id": "after-sync",
                "initiator": "user",
                "started_at": "2026-04-04T17:55:00+00:00",
                "finished_at": "2026-04-04T17:56:00+00:00",
                "resolved_model": "gpt-5.4-mini",
                "path": "/v1/responses",
                "premium_requests": 0.33,
            },
        ]
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                load_premium_plan_config=lambda: {
                    "configured": True,
                    "plan": "pro_plus",
                    "plan_label": "Pro+",
                    "included": 1500,
                    "synced_percent": 10.0,
                    "synced_used": 150.0,
                    "synced_at": "2026-04-04T17:50:00+00:00",
                    "synced_month": "2026-04",
                },
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()

        self.assertEqual(payload["premium"]["used"], 150.33)
        self.assertEqual(payload["premium"]["tracked_this_month"], 1.33)
        self.assertEqual(payload["premium"]["tracked_since_sync"], 0.33)
        self.assertEqual(payload["premium"]["remaining"], 1349.67)
        self.assertEqual(payload["premium"]["percent_used"], 10.02)
        self.assertTrue(payload["premium"]["sync_current_month"])

    def test_build_dashboard_payload_excludes_agent_requests_from_premium_usage(self):
        fixed_now = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        usage_events = [
            {
                "request_id": "user-req",
                "initiator": "user",
                "started_at": "2026-04-04T17:40:00+00:00",
                "finished_at": "2026-04-04T17:41:00+00:00",
                "resolved_model": "gpt-5.4",
                "path": "/v1/responses",
                "premium_requests": 1.0,
            },
            {
                "request_id": "agent-req",
                "initiator": "agent",
                "started_at": "2026-04-04T17:55:00+00:00",
                "finished_at": "2026-04-04T17:56:00+00:00",
                "resolved_model": "gpt-5.4-mini",
                "path": "/v1/responses",
                "premium_requests": 0.33,
            },
        ]
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                load_premium_plan_config=lambda: {
                    "configured": True,
                    "plan": "pro_plus",
                    "plan_label": "Pro+",
                    "included": 1500,
                    "synced_percent": 10.0,
                    "synced_used": 150.0,
                    "synced_at": "2026-04-04T17:30:00+00:00",
                    "synced_month": "2026-04",
                },
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()

        self.assertEqual(payload["premium"]["tracked_this_month"], 1.0)
        self.assertEqual(payload["premium"]["tracked_since_sync"], 1.0)
        self.assertEqual(payload["premium"]["used"], 151.0)
        self.assertEqual(payload["premium"]["remaining"], 1349.0)
        self.assertEqual(payload["premium"]["percent_used"], 10.07)

    def test_build_dashboard_payload_clamps_manual_plan_usage_at_100_percent(self):
        fixed_now = datetime(2026, 4, 4, 18, 0, tzinfo=timezone.utc)
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                load_premium_plan_config=lambda: {
                    "configured": True,
                    "plan": "pro",
                    "plan_label": "Pro",
                    "included": 300,
                    "synced_percent": 99.8,
                    "synced_used": 299.4,
                    "synced_at": "2026-04-04T17:00:00+00:00",
                    "synced_month": "2026-04",
                },
                snapshot_all_usage_events=lambda: [
                    {
                        "request_id": "overflow-req",
                        "started_at": "2026-04-04T17:55:00+00:00",
                        "finished_at": "2026-04-04T17:56:00+00:00",
                        "resolved_model": "claude-sonnet-4.6",
                        "path": "/v1/responses",
                        "premium_requests": 1.0,
                    }
                ],
                snapshot_usage_events=lambda: [],
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()

        self.assertEqual(payload["premium"]["used"], 300.0)
        self.assertEqual(payload["premium"]["remaining"], 0.0)
        self.assertEqual(payload["premium"]["percent_used"], 100.0)

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
