import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from premium_plan_config import PremiumPlanConfig, PremiumPlanConfigService


class PremiumPlanConfigTests(unittest.TestCase):
    def _make_service(self) -> PremiumPlanConfigService:
        path = Path.cwd() / f"premium-plan-{uuid4().hex}.json"

        def _cleanup():
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                pass

        self.addCleanup(_cleanup)
        return PremiumPlanConfigService(PremiumPlanConfig(config_file=str(path)))

    def test_save_settings_normalizes_pro_plus_and_persists_sync_state(self):
        service = self._make_service()
        now = datetime(2026, 4, 14, 12, 30, tzinfo=timezone.utc)

        payload = service.save_settings({"plan": "pro+", "current_percent": 72.5}, now=now)

        self.assertTrue(payload["configured"])
        self.assertEqual(payload["plan"], "pro_plus")
        self.assertEqual(payload["plan_label"], "Pro+")
        self.assertEqual(payload["included"], 1500)
        self.assertEqual(payload["synced_percent"], 72.5)
        self.assertEqual(payload["synced_used"], 1087.5)
        self.assertEqual(payload["synced_month"], "2026-04")
        self.assertEqual(payload["synced_at"], now.isoformat())

    def test_save_settings_clamps_percent_into_valid_range(self):
        service = self._make_service()
        now = datetime(2026, 4, 14, 12, 30, tzinfo=timezone.utc)

        payload = service.save_settings({"plan": "pro", "current_percent": 180}, now=now)

        self.assertEqual(payload["synced_percent"], 100.0)
        self.assertEqual(payload["synced_used"], 300.0)

    def test_clear_settings_returns_defaults(self):
        service = self._make_service()
        service.save_settings({"plan": "business", "current_percent": 12.5}, now=datetime(2026, 4, 14, tzinfo=timezone.utc))

        payload = service.clear_settings()

        self.assertFalse(payload["configured"])
        self.assertEqual(payload["plan"], "")
        self.assertEqual(payload["available_plans"][0]["id"], "pro")

    def test_save_settings_accepts_enterprise_plan(self):
        service = self._make_service()
        now = datetime(2026, 4, 14, 12, 30, tzinfo=timezone.utc)

        payload = service.save_settings({"plan": "enterprise", "current_percent": 20}, now=now)

        self.assertTrue(payload["configured"])
        self.assertEqual(payload["plan"], "enterprise")
        self.assertEqual(payload["plan_label"], "Enterprise")
        self.assertEqual(payload["included"], 1000)
        self.assertEqual(payload["synced_used"], 200.0)

    def test_save_settings_requires_valid_plan(self):
        service = self._make_service()

        with self.assertRaises(HTTPException) as ctx:
            service.save_settings({"plan": "team", "current_percent": 20})

        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
