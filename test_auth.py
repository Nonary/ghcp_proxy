import unittest
from unittest import mock

import auth


class AuthTests(unittest.TestCase):
    def setUp(self):
        auth._AUTH_FLOW_STATE.update(
            {
                "state": "idle",
                "flow_id": None,
                "started_at": None,
                "expires_at": None,
                "poll_interval_seconds": None,
                "verification_uri": None,
                "verification_uri_complete": None,
                "user_code": None,
                "error": None,
                "warning": "",
                "message": "",
            }
        )

    def test_get_api_key_requires_cached_auth_when_noninteractive(self):
        with (
            mock.patch.object(auth, "load_api_key", return_value=None),
            mock.patch.object(auth, "load_access_token", return_value=None),
            mock.patch.object(auth, "_device_flow", side_effect=AssertionError("interactive prompt should not start")),
        ):
            with self.assertRaisesRegex(RuntimeError, "authorization required"):
                auth.get_api_key()

    def test_begin_device_flow_returns_pending_snapshot_and_starts_background_poller(self):
        thread = mock.Mock()
        thread.start = mock.Mock()

        with (
            mock.patch.object(auth, "load_api_key", return_value=None),
            mock.patch.object(auth, "load_access_token", return_value=None),
            mock.patch.object(
                auth,
                "_device_flow_info",
                return_value={
                    "device_code": "device-code-123",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 5,
                    "expires_in": 900,
                },
            ),
            mock.patch.object(auth, "Thread", return_value=thread) as thread_cls,
        ):
            payload = auth.begin_device_flow()

        self.assertFalse(payload["authenticated"])
        self.assertEqual(payload["state"], "pending")
        self.assertEqual(payload["user_code"], "ABCD-EFGH")
        self.assertEqual(payload["verification_uri"], "https://github.com/login/device")
        self.assertEqual(payload["poll_interval_seconds"], 5)
        thread_cls.assert_called_once()
        self.assertTrue(thread_cls.call_args.kwargs["daemon"])
        self.assertEqual(thread_cls.call_args.kwargs["kwargs"]["device_code"], "device-code-123")
        thread.start.assert_called_once_with()

    def test_auth_status_reports_authenticated_from_cached_access_token(self):
        with (
            mock.patch.object(auth, "load_api_key", return_value=None),
            mock.patch.object(auth, "load_access_token", return_value="cached-access-token"),
        ):
            payload = auth.auth_status()

        self.assertTrue(payload["authenticated"])
        self.assertEqual(payload["state"], "authenticated")
        self.assertTrue(payload["has_access_token"])

    def test_get_api_key_force_refresh_bypasses_cached_api_key(self):
        with (
            mock.patch.object(auth, "load_api_key", return_value="cached-api-key") as load_api_key,
            mock.patch.object(auth, "load_access_token", return_value="access-token"),
            mock.patch.object(auth, "_refresh_api_key", return_value="fresh-api-key") as refresh,
        ):
            key = auth.get_api_key(force_refresh=True)

        self.assertEqual(key, "fresh-api-key")
        load_api_key.assert_not_called()
        refresh.assert_called_once_with("access-token")
