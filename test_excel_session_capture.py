from __future__ import annotations

import base64
import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import excel_upstream
from excel_session_capture import (
    ExcelSessionCaptureManager,
    load_macos_excel_session,
    refresh_macos_excel_session,
)


ROOT = Path(__file__).resolve().parent
MACOS_SCRIPT_PATH = ROOT / "tools" / "prime-excel-session-macos.py"


def _jwt_with_exp(expiration: float) -> str:
    def encode(value: object) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({'exp': expiration})}."


class MacLocalStorageCaptureTests(unittest.TestCase):
    def _write_database(
        self,
        root: Path,
        *,
        expiration: float,
        account_id: str = "account-id",
    ) -> Path:
        database = (
            root
            / "Default"
            / "profile"
            / "origin"
            / "LocalStorage"
            / "localstorage.sqlite3"
        )
        database.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "authMode": "chatgpt",
            "sessionInfo": {
                "access_token": _jwt_with_exp(expiration),
                "expires_at": expiration,
            },
            "userInfo": {
                "chatgpt_account_id": account_id,
                "chatgpt_account_user_id": "account-user-id",
            },
        }
        connection = sqlite3.connect(database)
        try:
            connection.execute(
                "CREATE TABLE ItemTable "
                "(key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB NOT NULL)"
            )
            connection.execute(
                "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
                ("bps_auth_tokens", json.dumps(payload).encode("utf-16-le")),
            )
            connection.commit()
        finally:
            connection.close()
        return database

    def test_load_session_reads_webkit_utf16_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_database(root, expiration=time.time() + 600)
            headers = load_macos_excel_session(root)
        self.assertTrue(headers["authorization"].startswith("Bearer "))
        self.assertEqual(headers["chatgpt-account-id"], "account-id")
        self.assertEqual(headers["x-openai-account-id"], "account-id")
        self.assertEqual(headers["x-openai-account-user-id"], "account-user-id")
        self.assertEqual(headers["x-basispoints-auth-mode"], "chatgpt")

    def test_load_session_reports_missing_sign_in(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError,
                "No signed-in ChatGPT Excel session",
            ):
                load_macos_excel_session(Path(directory))

    def test_automatic_refresh_populates_session_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_database(root, expiration=time.time() + 600)
            store = excel_upstream.ExcelSessionStore()
            with mock.patch("excel_session_capture.sys.platform", "darwin"):
                status = refresh_macos_excel_session(
                    store,
                    force=True,
                    website_data=root,
                )
        self.assertTrue(status["configured"])
        self.assertFalse(status["expired"])
        self.assertEqual(
            store.request_headers(stream=False)["chatgpt-account-id"],
            "account-id",
        )

    def test_automatic_refresh_preserves_expired_status_and_message(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_database(root, expiration=time.time() - 60)
            store = excel_upstream.ExcelSessionStore()
            with mock.patch("excel_session_capture.sys.platform", "darwin"):
                status = refresh_macos_excel_session(
                    store,
                    force=True,
                    website_data=root,
                )
            self.assertTrue(status["configured"])
            self.assertTrue(status["expired"])
            with mock.patch("excel_upstream.sys.platform", "darwin"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Refresh the ChatGPT Excel task pane",
                ):
                    store.request_headers(stream=False)

    def test_forced_refresh_picks_up_newer_excel_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = self._write_database(
                root,
                expiration=time.time() + 600,
                account_id="account-one",
            )
            store = excel_upstream.ExcelSessionStore()
            with mock.patch("excel_session_capture.sys.platform", "darwin"):
                refresh_macos_excel_session(store, force=True, website_data=root)
            database.unlink()
            self._write_database(
                root,
                expiration=time.time() + 1200,
                account_id="account-two",
            )
            with mock.patch("excel_session_capture.sys.platform", "darwin"):
                refresh_macos_excel_session(store, force=True, website_data=root)
        self.assertEqual(
            store.request_headers(stream=False)["chatgpt-account-id"],
            "account-two",
        )


class ExcelSessionCaptureManagerTests(unittest.TestCase):
    def _manager(self) -> ExcelSessionCaptureManager:
        return ExcelSessionCaptureManager(
            script_path=str(ROOT / "tools" / "prime-excel-session.js"),
            macos_script_path=str(MACOS_SCRIPT_PATH),
            session_status_provider=lambda: {
                "configured": False,
                "configured_at": None,
            },
        )

    def test_macos_status_requires_no_proxy_or_certificate(self):
        manager = self._manager()
        with mock.patch("excel_session_capture.sys.platform", "darwin"):
            status = manager.status()
        self.assertEqual(status["method"], "webkit-localstorage-sqlite")
        self.assertTrue(status["available"])
        self.assertFalse(status["proxy_active"])
        self.assertFalse(status["setup_required"])
        self.assertEqual(status["workflow_version"], 3)
        self.assertEqual(status["ca_install_command"], "")
        self.assertEqual(status["install_command"], "")


if __name__ == "__main__":
    unittest.main()
