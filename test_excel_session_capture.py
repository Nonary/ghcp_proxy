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
    _decompress_snappy,
    load_macos_excel_session,
    load_windows_excel_session,
    refresh_macos_excel_session,
)


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


class WindowsLocalStorageCacheTests(unittest.TestCase):
    def test_snappy_literal_decompression(self):
        self.assertEqual(_decompress_snappy(b"\x03\x08abc"), b"abc")

    def test_cache_reader_loads_webview2_localstorage_entry(self):
        payload = {
            "authMode": "chatgpt",
            "sessionInfo": {"access_token": _jwt_with_exp(time.time() + 600)},
            "userInfo": {
                "chatgpt_account_id": "account-id",
                "chatgpt_account_user_id": "account-user-id",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = (
                root
                / "profile"
                / "EBWebView"
                / "Default"
                / "Local Storage"
                / "leveldb"
                / "000001.ldb"
            )
            table.parent.mkdir(parents=True)
            table.touch()
            with mock.patch(
                "excel_session_capture._leveldb_table_entries",
                return_value=[
                    (
                        b"https://bps.openai.com\x00bps_auth_tokens",
                        json.dumps(payload).encode("utf-8"),
                    )
                ],
            ):
                headers = load_windows_excel_session(root)
        self.assertTrue(headers["authorization"].startswith("Bearer "))
        self.assertEqual(headers["chatgpt-account-id"], "account-id")
        self.assertEqual(headers["x-openai-account-user-id"], "account-user-id")


if __name__ == "__main__":
    unittest.main()
