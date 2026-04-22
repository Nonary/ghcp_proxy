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
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()

        self.assertEqual(payload["current_month"]["usage"]["cost_usd"], 4.0)
        self.assertEqual(payload["current_month"]["usage"]["total_tokens"], 400)
        self.assertEqual(payload["current_month"]["usage"]["request_count"], 2)
        self.assertEqual(payload["current_month"]["usage"]["premium_requests"], 1.33)
        self.assertEqual(payload["all_time"]["usage"]["request_count"], 2)
        self.assertEqual(payload["all_time"]["usage"]["premium_requests"], 1.33)
        self.assertEqual(len(payload["current_month"]["daily_history"]), 4)
        self.assertEqual(payload["current_month"]["daily_history"][-1]["day_key"], "2026-04-04")
        self.assertEqual(payload["current_month"]["daily_history"][-1]["cost_usd"], 4.0)
        self.assertEqual(payload["current_month"]["daily_history"][-1]["premium_requests"], 1.33)
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
        self.assertEqual(payload["premium"]["source"], "awaiting-first-request")

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

    def test_premium_summary_derived_from_quota_snapshot_headers(self):
        fixed_now = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
        usage_events = [
            {
                "request_id": "req-1",
                "started_at": "2026-04-17T11:30:00+00:00",
                "finished_at": "2026-04-17T11:30:01+00:00",
                "quota_snapshots": {
                    "premium_interactions": {
                        "included": 1000,
                        "unlimited": False,
                        "percent_remaining": 36.0,
                        "percent_used": 64.0,
                        "absolute_remaining": 360.0,
                        "absolute_used": 640.0,
                        "overage": 0.0,
                        "overage_permitted": True,
                        "reset_at": "2026-05-01T00:00:00Z",
                        "raw": {"ent": "1000", "rem": "36.0"},
                    },
                    "chat": {
                        "included": None,
                        "unlimited": True,
                        "percent_remaining": 100.0,
                        "percent_used": 0.0,
                        "absolute_remaining": None,
                        "absolute_used": None,
                        "overage": 0.0,
                        "overage_permitted": False,
                        "reset_at": "2026-05-01T00:00:00Z",
                        "raw": {"ent": "-1", "rem": "100.0"},
                    },
                },
            },
        ]
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()
        premium = payload["premium"]

        self.assertEqual(premium["source"], "upstream-quota-snapshot")
        self.assertEqual(premium["included"], 1000)
        self.assertEqual(premium["remaining"], 360.0)
        self.assertEqual(premium["used"], 640.0)
        self.assertEqual(premium["percent_remaining"], 36.0)
        self.assertEqual(premium["percent_used"], 64.0)
        self.assertEqual(premium["reset_at"], "2026-05-01T00:00:00Z")
        self.assertEqual(premium["days_until_reset"], 13)
        self.assertFalse(premium["unlimited"])
        # The chat bucket was unlimited; surfaced through the buckets dict.
        self.assertTrue(premium["buckets"]["chat"]["unlimited"])

    def test_premium_summary_uses_latest_snapshot(self):
        fixed_now = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)

        def _snapshot(percent_remaining, when):
            return {
                "premium_interactions": {
                    "included": 1000,
                    "unlimited": False,
                    "percent_remaining": percent_remaining,
                    "percent_used": 100.0 - percent_remaining,
                    "absolute_remaining": 1000 * percent_remaining / 100.0,
                    "absolute_used": 1000 - 1000 * percent_remaining / 100.0,
                    "overage": 0.0,
                    "overage_permitted": True,
                    "reset_at": "2026-05-01T00:00:00Z",
                    "raw": {},
                },
            }

        usage_events = [
            {"request_id": "old", "started_at": "2026-04-17T10:00:00+00:00",
             "finished_at": "2026-04-17T10:00:01+00:00",
             "quota_snapshots": _snapshot(50.0, "10:00:01")},
            {"request_id": "newer", "started_at": "2026-04-17T11:30:00+00:00",
             "finished_at": "2026-04-17T11:30:01+00:00",
             "quota_snapshots": _snapshot(36.0, "11:30:01")},
        ]
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()
        self.assertEqual(payload["premium"]["percent_remaining"], 36.0)
        self.assertEqual(payload["premium"]["request_id"], "newer")

    def test_premium_summary_awaiting_first_request_when_no_snapshots(self):
        fixed_now = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                snapshot_all_usage_events=lambda: [],
                snapshot_usage_events=lambda: [],
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )
        payload = service.build_payload()
        self.assertFalse(payload["premium"]["configured"])
        self.assertEqual(payload["premium"]["source"], "awaiting-first-request")
        self.assertIsNone(payload["premium"]["percent_remaining"])
        self.assertIsNone(payload["premium"]["reset_at"])
        self.assertIsNone(payload["premium"]["days_until_reset"])

    def test_premium_summary_clamps_days_until_reset_at_zero(self):
        fixed_now = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
        usage_events = [
            {
                "request_id": "req-1",
                "started_at": "2026-04-17T11:30:00+00:00",
                "finished_at": "2026-04-17T11:30:01+00:00",
                "quota_snapshots": {
                    "premium_interactions": {
                        "included": 1000,
                        "unlimited": False,
                        "percent_remaining": 36.0,
                        "percent_used": 64.0,
                        "absolute_remaining": 360.0,
                        "absolute_used": 640.0,
                        "overage": 0.0,
                        "overage_permitted": True,
                        "reset_at": "2026-05-01T00:00:00Z",
                        "raw": {},
                    },
                },
            },
        ]
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )
        payload = service.build_payload()
        # Reset already passed; days_until_reset should be clamped at 0, not negative.
        self.assertEqual(payload["premium"]["days_until_reset"], 0)

    def test_premium_consumption_summary_rolls_up_today_week_month_and_all_time(self):
        fixed_now = datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc)
        usage_events = [
            {
                "request_id": "today",
                "started_at": "2026-04-17T11:30:00+00:00",
                "finished_at": "2026-04-17T11:30:01+00:00",
                "resolved_model": "claude-sonnet-4.6",
                "premium_requests": 1.0,
            },
            {
                "request_id": "within-week",
                "started_at": "2026-04-12T09:00:00+00:00",
                "finished_at": "2026-04-12T09:00:01+00:00",
                "resolved_model": "gpt-5.4",
                "premium_requests": 0.33,
            },
            {
                "request_id": "month-only",
                "started_at": "2026-04-02T09:00:00+00:00",
                "finished_at": "2026-04-02T09:00:01+00:00",
                "resolved_model": "gpt-5.4-mini",
                "premium_requests": 0.1,
            },
            {
                "request_id": "older",
                "started_at": "2026-03-29T09:00:00+00:00",
                "finished_at": "2026-03-29T09:00:01+00:00",
                "resolved_model": "claude-sonnet-4.6",
                "premium_requests": 1.0,
            },
            {
                "request_id": "not-premium",
                "started_at": "2026-04-17T10:00:00+00:00",
                "finished_at": "2026-04-17T10:00:01+00:00",
                "resolved_model": "gpt-4.1",
                "premium_requests": 0.0,
            },
        ]
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        payload = service.build_payload()
        consumed = payload["premium"]["consumed"]

        self.assertEqual(consumed["today"], 1.0)
        self.assertEqual(consumed["last_7_days"], 1.33)
        self.assertEqual(consumed["current_month"], 1.43)
        self.assertEqual(consumed["all_time"], 2.43)





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

    def test_collect_local_dashboard_usage_groups_codex_native_turns_by_session(self):
        usage = dashboard.collect_local_dashboard_usage(
            [
                {
                    "request_id": "codex-native:native-session-123:49",
                    "server_request_id": "native-session-123",
                    "started_at": "2026-04-05T01:30:00+00:00",
                    "finished_at": "2026-04-05T01:31:00+00:00",
                    "requested_model": "gpt-5.4",
                    "native_source": "codex_native",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cached_input_tokens": 30,
                        "total_tokens": 120,
                    },
                },
                {
                    "request_id": "codex-native:native-session-123:50",
                    "server_request_id": "native-session-123",
                    "started_at": "2026-04-05T01:32:00+00:00",
                    "finished_at": "2026-04-05T01:33:00+00:00",
                    "requested_model": "gpt-5.4",
                    "native_source": "codex_native",
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 40,
                        "cached_input_tokens": 10,
                        "total_tokens": 240,
                    },
                },
            ]
        )

        row = usage["recent_sessions"][0]
        self.assertEqual(row["source"], "codex_native")
        self.assertEqual(row["session_id"], "native-session-123")
        self.assertEqual(row["session_display_id"], "native-session-123")
        self.assertEqual(row["request_count"], 2)
        self.assertEqual(row["total_tokens"], 360)


    def test_usage_ratelimits_summary_derived_from_headers(self):
        fixed_now = datetime(2026, 4, 21, 4, 22, 37, tzinfo=timezone.utc)
        usage_events = [
            {
                "request_id": "req-rl-1",
                "started_at": "2026-04-21T01:47:53+00:00",
                "finished_at": "2026-04-21T01:47:54+00:00",
                "usage_ratelimits": {
                    "session": {
                        "percent_remaining": 93.2,
                        "percent_used": 6.8,
                        "entitlement": 0,
                        "overage": 0.0,
                        "overage_permitted": False,
                        "reset_at": "2026-04-21T06:22:37Z",
                        "raw": {"rem": "93.2"},
                    },
                    "weekly": {
                        "percent_remaining": 99.0,
                        "percent_used": 1.0,
                        "entitlement": 0,
                        "overage": 0.0,
                        "overage_permitted": False,
                        "reset_at": "2026-04-27T00:00:00Z",
                        "raw": {"rem": "99.0"},
                    },
                },
            },
        ]
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )

        block = service.build_payload()["usage_ratelimits"]
        self.assertTrue(block["configured"])
        self.assertEqual(block["source"], "upstream-usage-ratelimit-headers")
        self.assertEqual(block["request_id"], "req-rl-1")
        self.assertEqual(block["windows"]["session"]["percent_remaining"], 93.2)
        # 2026-04-21T06:22:37Z is exactly two hours after fixed_now -> 7200s.
        self.assertEqual(block["windows"]["session"]["seconds_until_reset"], 7200)
        self.assertEqual(block["windows"]["weekly"]["percent_remaining"], 99.0)
        self.assertGreater(block["windows"]["weekly"]["seconds_until_reset"], 0)

    def test_usage_ratelimits_summary_uses_latest_event(self):
        fixed_now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)

        def _windows(rem):
            return {
                "session": {
                    "percent_remaining": rem,
                    "percent_used": round(100.0 - rem, 2),
                    "entitlement": 0,
                    "overage": 0.0,
                    "overage_permitted": False,
                    "reset_at": "2026-04-21T18:00:00Z",
                    "raw": {},
                },
            }

        usage_events = [
            {"request_id": "older", "finished_at": "2026-04-21T10:00:00+00:00",
             "usage_ratelimits": _windows(80.0)},
            {"request_id": "newer", "finished_at": "2026-04-21T11:30:00+00:00",
             "usage_ratelimits": _windows(40.0)},
        ]
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                snapshot_all_usage_events=lambda: usage_events,
                snapshot_usage_events=lambda: usage_events,
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )
        block = service.build_payload()["usage_ratelimits"]
        self.assertEqual(block["request_id"], "newer")
        self.assertEqual(block["windows"]["session"]["percent_remaining"], 40.0)

    def test_usage_ratelimits_summary_awaiting_first_request(self):
        fixed_now = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
        service = dashboard.DashboardService(
            dependencies=dashboard.DashboardDependencies(
                snapshot_all_usage_events=lambda: [],
                snapshot_usage_events=lambda: [],
                load_safeguard_trigger_stats=lambda _now: {},
            ),
            utc_now=lambda: fixed_now,
        )
        block = service.build_payload()["usage_ratelimits"]
        self.assertFalse(block["configured"])
        self.assertEqual(block["source"], "awaiting-first-request")
        self.assertEqual(block["windows"], {})

    def _burn_event(self, ts, model, percent_used, reset_at, *, input_tokens=0, cached_input_tokens=0, output_tokens=0, reasoning=0, window="session"):
        return {
            "request_id": ts,
            "finished_at": ts,
            "response_model": model,
            "usage": {
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "output_tokens": output_tokens,
                "reasoning_output_tokens": reasoning,
            },
            "usage_ratelimits": {
                window: {
                    "percent_remaining": round(100.0 - percent_used, 4),
                    "percent_used": percent_used,
                    "entitlement": 0,
                    "overage": 0.0,
                    "overage_permitted": False,
                    "reset_at": reset_at,
                    "raw": {},
                },
            },
        }

    def test_burn_rate_attributes_delta_to_later_event_model(self):
        reset = "2026-04-21T18:00:00Z"
        events = [
            self._burn_event("2026-04-21T10:00:00+00:00", "gpt-5-mini", 5.0, reset, input_tokens=1000, output_tokens=500),
            self._burn_event("2026-04-21T10:01:00+00:00", "claude-opus-4.7", 9.0, reset, input_tokens=2000, output_tokens=2000),
        ]
        burn = dashboard._build_burn_rate_summary(events)
        self.assertIn("session", burn)
        models = {row["model"]: row for row in burn["session"]["models"]}
        # Only the later event gets attribution (one delta = 4.0%).
        self.assertEqual(burn["session"]["samples"], 1)
        self.assertIn("claude-opus-4.7", models)
        self.assertNotIn("gpt-5-mini", models)
        opus = models["claude-opus-4.7"]
        self.assertEqual(opus["sample_target_tokens"], 100_000)
        self.assertAlmostEqual(opus["window_delta_percent"], 2.0, places=4)
        self.assertAlmostEqual(opus["limit_percent_per_target_tokens"], 100.0, places=4)
        self.assertAlmostEqual(opus["burn_rates"]["input"]["limit_percent_per_target_tokens"], 100.0, places=4)
        self.assertAlmostEqual(opus["burn_rates"]["output"]["limit_percent_per_target_tokens"], 100.0, places=4)
        self.assertEqual(opus["tokens"]["input_tokens"], 2000.0)
        self.assertEqual(opus["tokens"]["output_tokens"], 2000.0)
        self.assertEqual(opus["tokens"]["total_tokens"], 4000.0)
        self.assertEqual(opus["responsibility_percent"], {"input": 50.0, "cached": 0.0, "output": 50.0})

    def test_burn_rate_handles_anthropic_token_convention(self):
        reset = "2026-04-21T18:00:00Z"
        events = [
            self._burn_event("2026-04-21T10:00:00+00:00", "claude-opus-4.7", 1.0, reset),
            # Anthropic shape: cached > input means input is already fresh-only.
            self._burn_event(
                "2026-04-21T10:01:00+00:00",
                "claude-opus-4.7",
                3.0,
                reset,
                input_tokens=1000,
                cached_input_tokens=20000,
                output_tokens=500,
            ),
        ]
        burn = dashboard._build_burn_rate_summary(events)
        opus = burn["session"]["models"][0]
        # Fresh input == input_tokens (not subtracted) under Anthropic shape.
        self.assertEqual(opus["tokens"]["input_tokens"], 1000.0)
        self.assertEqual(opus["tokens"]["cached_input_tokens"], 0.0)
        self.assertEqual(opus["tokens"]["output_tokens"], 500.0)
        self.assertAlmostEqual(opus["burn_rates"]["input"]["limit_percent_per_target_tokens"], 133.3333, places=4)
        self.assertAlmostEqual(opus["burn_rates"]["output"]["limit_percent_per_target_tokens"], 133.3333, places=4)
        self.assertEqual(opus["responsibility_percent"], {"input": 66.67, "cached": 0.0, "output": 33.33})

    def test_burn_rate_uses_last_100k_input_or_output_tokens_and_splits_boundary_request(self):
        reset = "2026-04-21T18:00:00Z"
        events = [
            self._burn_event("2026-04-21T10:00:00+00:00", "gpt-5-mini", 0.0, reset),
            self._burn_event("2026-04-21T10:01:00+00:00", "gpt-5-mini", 1.0, reset, input_tokens=600_000),
            self._burn_event("2026-04-21T10:02:00+00:00", "gpt-5-mini", 2.0, reset, output_tokens=500_000),
        ]
        burn = dashboard._build_burn_rate_summary(events)
        mini = burn["session"]["models"][0]
        self.assertEqual(mini["samples"], 2)
        self.assertEqual(mini["sampled_tokens"], 200_000.0)
        self.assertEqual(mini["sampled_requests"], 1)
        self.assertAlmostEqual(mini["window_delta_percent"], 0.2, places=4)
        self.assertAlmostEqual(mini["limit_percent_per_target_tokens"], 0.2, places=4)
        self.assertAlmostEqual(mini["burn_rates"]["input"]["limit_percent_per_target_tokens"], 0.1667, places=4)
        self.assertAlmostEqual(mini["burn_rates"]["output"]["limit_percent_per_target_tokens"], 0.2, places=4)
        self.assertEqual(mini["responsibility_percent"], {"input": 50.0, "cached": 0.0, "output": 50.0})

    def test_burn_rate_reports_latest_reset_while_sampling_recent_historical_tokens(self):
        reset = "2026-04-21T18:00:00Z"
        events = [
            self._burn_event("2026-04-21T09:00:00+00:00", "gpt-5-mini", 0.0, "2026-04-21T10:00:00Z"),
            self._burn_event("2026-04-21T09:01:00+00:00", "gpt-5-mini", 10.0, "2026-04-21T10:00:00Z", input_tokens=1_000_000),
            self._burn_event("2026-04-21T10:00:00+00:00", "gpt-5-mini", 0.0, reset),
            self._burn_event("2026-04-21T10:01:00+00:00", "gpt-5-mini", 1.0, reset, input_tokens=1_000_000),
        ]
        burn = dashboard._build_burn_rate_summary(events)
        mini = burn["session"]["models"][0]
        self.assertEqual(burn["session"]["reset_at"], reset)
        self.assertAlmostEqual(mini["window_delta_percent"], 0.1, places=4)
        self.assertAlmostEqual(mini["limit_percent_per_target_tokens"], 0.1, places=4)

    def test_burn_rate_time_buckets_are_historical_averages_across_resets(self):
        old_reset = "2026-04-21T10:00:00Z"
        new_reset = "2026-04-21T18:00:00Z"
        events = [
            self._burn_event("2026-04-21T09:00:00+00:00", "gpt-5-mini", 0.0, old_reset),
            self._burn_event("2026-04-21T09:05:00+00:00", "gpt-5-mini", 2.0, old_reset, input_tokens=1_000),
            self._burn_event("2026-04-21T10:00:00+00:00", "claude-opus-4.7", 0.0, new_reset),
            self._burn_event("2026-04-21T10:35:00+00:00", "claude-opus-4.7", 3.0, new_reset, input_tokens=3_000),
        ]

        burn = dashboard._build_burn_rate_summary(events)
        bucket = next(b for b in burn["session"]["time_buckets"]["buckets"] if b["minutes"] == 30)
        models = {row["model"]: row for row in bucket["models"]}

        self.assertEqual(bucket["periods_observed"], 2)
        self.assertEqual(bucket["segments_observed"], 2)
        self.assertAlmostEqual(bucket["totals"]["delta_percent"], 2.5, places=4)
        self.assertAlmostEqual(bucket["avg_delta_percent_per_minute"], 0.083333, places=6)
        self.assertAlmostEqual(bucket["request_count"], 1.0, places=2)
        self.assertAlmostEqual(bucket["totals"]["billable_tokens"], 2_000.0, places=2)
        self.assertAlmostEqual(models["gpt-5-mini"]["delta_percent"], 1.0, places=4)
        self.assertAlmostEqual(models["gpt-5-mini"]["requests"], 0.5, places=2)
        self.assertEqual(models["gpt-5-mini"]["active_periods"], 1)
        self.assertAlmostEqual(models["claude-opus-4.7"]["delta_percent"], 1.5, places=4)
        self.assertAlmostEqual(models["claude-opus-4.7"]["requests"], 0.5, places=2)

    def test_burn_rate_larger_buckets_normalize_to_same_30m_equivalent(self):
        reset = "2026-04-21T18:00:00Z"
        events = [self._burn_event("2026-04-21T05:30:00+00:00", "gpt-5-mini", 0.0, reset)]
        percent_used = 0.0
        for idx in range(12):
            percent_used += 2.0
            hour = 6 + (idx // 2)
            minute = 0 if idx % 2 == 0 else 30
            ts = f"2026-04-21T{hour:02d}:{minute:02d}:00+00:00"
            events.append(self._burn_event(ts, "gpt-5-mini", percent_used, reset, input_tokens=1_000))

        burn = dashboard._build_burn_rate_summary(events)
        bucket_30m = next(b for b in burn["session"]["time_buckets"]["buckets"] if b["minutes"] == 30)
        bucket_6h = next(b for b in burn["session"]["time_buckets"]["buckets"] if b["minutes"] == 360)

        self.assertEqual(bucket_30m["reporting_minutes"], 30)
        self.assertEqual(bucket_6h["reporting_minutes"], 30)
        self.assertAlmostEqual(bucket_30m["totals"]["delta_percent"], 2.0, places=4)
        self.assertAlmostEqual(bucket_6h["totals"]["delta_percent"], 2.0, places=4)
        self.assertAlmostEqual(bucket_30m["avg_delta_percent_per_minute"], 0.066667, places=6)
        self.assertAlmostEqual(bucket_6h["avg_delta_percent_per_minute"], 0.066667, places=6)
        self.assertAlmostEqual(bucket_30m["request_count"], 1.0, places=2)
        self.assertAlmostEqual(bucket_6h["request_count"], 1.0, places=2)
        self.assertAlmostEqual(bucket_30m["totals"]["billable_tokens"], 1_000.0, places=2)
        self.assertAlmostEqual(bucket_6h["totals"]["billable_tokens"], 1_000.0, places=2)

    def test_burn_rate_treats_reasoning_as_output_share(self):
        reset = "2026-04-21T18:00:00Z"
        events = [
            self._burn_event("2026-04-21T10:00:00+00:00", "gpt-5.4", 0.0, reset),
            self._burn_event(
                "2026-04-21T10:01:00+00:00",
                "gpt-5.4",
                1.0,
                reset,
                input_tokens=500_000,
                output_tokens=250_000,
                reasoning=250_000,
            ),
        ]
        burn = dashboard._build_burn_rate_summary(events)
        gpt = burn["session"]["models"][0]
        self.assertEqual(gpt["responsibility_percent"], {"input": 50.0, "cached": 0.0, "output": 50.0})
        self.assertAlmostEqual(gpt["burn_rates"]["output"]["limit_percent_per_target_tokens"], 0.1, places=4)

    def test_burn_rate_apportions_mixed_request_delta_before_scaling_output(self):
        reset = "2026-04-21T18:00:00Z"
        events = [
            self._burn_event("2026-04-21T10:00:00+00:00", "claude-opus-4.7", 0.0, reset),
            self._burn_event(
                "2026-04-21T10:01:00+00:00",
                "claude-opus-4.7",
                1.0,
                reset,
                input_tokens=50_000,
                output_tokens=5_000,
            ),
        ]
        burn = dashboard._build_burn_rate_summary(events)
        opus = burn["session"]["models"][0]
        self.assertAlmostEqual(opus["burn_rates"]["input"]["limit_percent_per_target_tokens"], 1.8182, places=4)
        # Previously this charged the full 1% delta to only 5K output tokens and
        # inflated the estimate to 20% per 100K output.
        self.assertAlmostEqual(opus["burn_rates"]["output"]["limit_percent_per_target_tokens"], 1.8182, places=4)

    def test_burn_rate_excludes_cached_only_events(self):
        reset = "2026-04-21T18:00:00Z"
        events = [
            self._burn_event("2026-04-21T10:00:00+00:00", "gpt-5-mini", 0.0, reset),
            self._burn_event("2026-04-21T10:01:00+00:00", "gpt-5-mini", 1.0, reset, input_tokens=100_000, cached_input_tokens=100_000),
        ]
        burn = dashboard._build_burn_rate_summary(events)
        self.assertEqual(burn["session"]["samples"], 0)
        self.assertEqual(burn["session"]["models"], [])
        self.assertEqual(burn["session"]["time_buckets"]["timeline"], [])

    def test_burn_rate_skips_delta_across_reset_boundary(self):
        events = [
            self._burn_event("2026-04-21T10:00:00+00:00", "gpt-5-mini", 90.0, "2026-04-21T11:00:00Z", input_tokens=500, output_tokens=500),
            # Window reset; new reset_at -> previous sample discarded, no attribution.
            self._burn_event("2026-04-21T11:05:00+00:00", "gpt-5-mini", 5.0, "2026-04-21T18:00:00Z", input_tokens=500, output_tokens=500),
        ]
        burn = dashboard._build_burn_rate_summary(events)
        self.assertEqual(burn["session"]["samples"], 0)
        self.assertEqual(burn["session"]["models"], [])

    def test_burn_rate_sorts_models_by_limit_impact(self):
        reset = "2026-04-21T18:00:00Z"
        events = [
            self._burn_event("2026-04-21T10:00:00+00:00", "gpt-5-mini", 0.0, reset),
            self._burn_event("2026-04-21T10:01:00+00:00", "gpt-5-mini", 1.0, reset, input_tokens=1_000_000),
            self._burn_event("2026-04-21T10:02:00+00:00", "claude-opus-4.7", 2.0, reset, input_tokens=500_000),
        ]
        burn = dashboard._build_burn_rate_summary(events)
        # claude-opus-4.7: 1.0% over 500k => 0.2% / 100K, so it sorts ahead of gpt-5-mini.
        self.assertEqual(burn["session"]["models"][0]["model"], "claude-opus-4.7")

    def test_burn_rate_normalizes_dated_model_revisions(self):
        reset = "2026-04-21T18:00:00Z"
        events = [
            self._burn_event("2026-04-21T10:00:00+00:00", "gpt-5.4", 0.0, reset),
            self._burn_event("2026-04-21T10:01:00+00:00", "gpt-5.4-2026-03-05", 1.0, reset, input_tokens=500_000),
            self._burn_event("2026-04-21T10:02:00+00:00", "gpt-5.4", 2.0, reset, input_tokens=500_000),
        ]

        burn = dashboard._build_burn_rate_summary(events)
        models = {row["model"]: row for row in burn["session"]["models"]}

        self.assertIn("gpt-5.4", models)
        self.assertNotIn("gpt-5.4-2026-03-05", models)
        self.assertEqual(models["gpt-5.4"]["samples"], 2)
        self.assertAlmostEqual(models["gpt-5.4"]["limit_percent_per_target_tokens"], 0.2, places=4)



if __name__ == "__main__":
    unittest.main()
