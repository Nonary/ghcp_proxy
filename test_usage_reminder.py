from datetime import datetime, timedelta, timezone
import unittest
from unittest import mock

import usage_reminder


class UsageReminderTests(unittest.TestCase):
    def _event(self, when: datetime, percent_used: float, reset_at: str):
        return {
            "finished_at": when.isoformat().replace("+00:00", "Z"),
            "usage_ratelimits": {
                "session": {
                    "percent_used": percent_used,
                    "percent_remaining": 100.0 - percent_used,
                    "reset_at": reset_at,
                }
            },
        }

    def test_controller_injects_once_per_session_ten_percent_threshold(self):
        now = datetime(2026, 4, 25, 18, 0, tzinfo=timezone.utc)
        reset_at = (now + timedelta(hours=4)).isoformat().replace("+00:00", "Z")
        events = [self._event(now - timedelta(hours=2), 5.0, reset_at)]
        controller = usage_reminder.UsageReminderController(
            lambda: events,
            now_func=lambda: now,
        )
        current = {
            "session": {
                "percent_remaining": 88.0,
                "percent_used": 12.0,
                "reset_at": reset_at,
            },
            "weekly": {
                "percent_remaining": 97.5,
                "percent_used": 2.5,
                "reset_at": "2026-04-27T00:00:00Z",
            },
        }

        notice = controller.usage_notice_text_if_due(current)
        self.assertIn("88% remaining", notice)
        self.assertIn("10% 5h-usage reminder", notice)
        self.assertIn("burn-down", notice)
        self.assertIn("not expected to run out before the 5h limit resets", notice)
        self.assertIn("Weekly limit: 97.5% remaining", notice)

        self.assertEqual(controller.usage_notice_text_if_due(current), "")

        current["session"]["percent_remaining"] = 82.0
        current["session"]["percent_used"] = 18.0
        self.assertEqual(controller.usage_notice_text_if_due(current), "")

        current["session"]["percent_remaining"] = 78.0
        current["session"]["percent_used"] = 22.0
        notice = controller.usage_notice_text_if_due(current)
        self.assertIn("20% 5h-usage reminder", notice)

    def test_history_threshold_prevents_duplicate_after_restart(self):
        now = datetime(2026, 4, 25, 18, 0, tzinfo=timezone.utc)
        reset_at = (now + timedelta(hours=4)).isoformat().replace("+00:00", "Z")
        events = [self._event(now - timedelta(minutes=5), 21.0, reset_at)]
        controller = usage_reminder.UsageReminderController(
            lambda: events,
            now_func=lambda: now,
        )

        notice = controller.usage_notice_text_if_due({
            "session": {
                "percent_remaining": 78.0,
                "percent_used": 22.0,
                "reset_at": reset_at,
            }
        })

        self.assertEqual(notice, "")

    def test_projection_warns_when_expected_to_run_out_before_reset(self):
        now = datetime(2026, 4, 25, 18, 0, tzinfo=timezone.utc)
        reset_at = (now + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
        text = usage_reminder.build_usage_reminder_text(
            {
                "session": {
                    "percent_remaining": 30.0,
                    "percent_used": 70.0,
                    "reset_at": reset_at,
                },
                "weekly": {
                    "percent_remaining": 90.0,
                    "percent_used": 10.0,
                    "reset_at": "2026-04-27T00:00:00Z",
                },
            },
            [self._event(now - timedelta(hours=1), 20.0, reset_at)],
            now=now,
            threshold_percent=70,
        )

        self.assertIn("70% 5h-usage reminder", text)
        self.assertIn("expected to run out", text)
        self.assertIn("before the 5h limit resets", text)

    def test_friendly_limit_message_reports_session_and_weekly_reset_durations(self):
        import httpx
        import proxy

        now = datetime(2026, 4, 25, 18, 0, tzinfo=timezone.utc)
        upstream = httpx.Response(
            429,
            json={"error": {"message": "rate limited"}},
            headers={
                "x-usage-ratelimit-session": "rem=0&rst=2026-04-25T20%3A30%3A00Z",
                "x-usage-ratelimit-weekly": "rem=0&rst=2026-04-27T00%3A00%3A00Z",
            },
        )

        with mock.patch.object(proxy.util, "utc_now", return_value=now):
            message = proxy._friendly_limit_message_from_upstream(upstream)

        self.assertIn("5h and weekly usage limits", message)
        self.assertIn("The 5h limit resets in about 2 hours 30 minutes", message)
        self.assertIn("The weekly limit resets in about 1 day 6 hours", message)

    def test_friendly_limit_message_ignores_unexhausted_429(self):
        import httpx
        import proxy

        upstream = httpx.Response(
            429,
            json={"error": {"message": "rate limited"}},
            headers={"x-usage-ratelimit-session": "rem=12.5&rst=2026-04-25T20%3A30%3A00Z"},
        )

        self.assertIsNone(proxy._friendly_limit_message_from_upstream(upstream))


if __name__ == "__main__":
    unittest.main()
