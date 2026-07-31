"""Background capture process for the ChatGPT Excel WebView session.

Windows Excel exposes WebView2 over Chrome DevTools Protocol. Excel for macOS
stores the signed-in add-in session in WebKit LocalStorage, so macOS reads that
SQLite database directly without installing a certificate or changing proxies.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable


_MACOS_STORAGE_KEY = "bps_auth_tokens"
_MACOS_WEBSITE_DATA = Path(
    os.environ.get(
        "GHCP_EXCEL_WEBKIT_WEBSITE_DATA_DIR",
        str(
            Path.home()
            / "Library/Containers/com.microsoft.Excel/Data/Library/WebKit/WebsiteData"
        ),
    )
).expanduser()
_MACOS_REFRESH_LOCK = threading.Lock()


def _decode_macos_storage_value(value: object) -> dict[str, object]:
    if isinstance(value, bytes):
        text = value.decode("utf-16-le")
    elif isinstance(value, str):
        text = value
    else:
        raise ValueError("the WebKit LocalStorage value has an unsupported type")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("the WebKit LocalStorage token payload is not an object")
    return payload


def _jwt_payload(token: str) -> dict[str, object]:
    try:
        encoded = token.split(".", 2)[1]
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        payload = json.loads(decoded)
    except (IndexError, ValueError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _macos_session_headers(payload: dict[str, object]) -> tuple[dict[str, str], float]:
    session_info = payload.get("sessionInfo")
    user_info = payload.get("userInfo")
    if not isinstance(session_info, dict) or not isinstance(user_info, dict):
        raise ValueError("the stored ChatGPT session is missing session or user data")
    access_token = session_info.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ValueError("the stored ChatGPT session has no access token")
    claims = _jwt_payload(access_token)
    auth_claims = claims.get("https://api.openai.com/auth")
    if not isinstance(auth_claims, dict):
        auth_claims = {}
    account_id = user_info.get("chatgpt_account_id") or auth_claims.get(
        "chatgpt_account_id"
    )
    account_user_id = user_info.get("chatgpt_account_user_id") or auth_claims.get(
        "chatgpt_account_user_id"
    )
    if not isinstance(account_id, str) or not account_id:
        raise ValueError("the stored ChatGPT session has no account ID")
    headers = {
        "authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "x-openai-account-id": account_id,
    }
    if isinstance(account_user_id, str) and account_user_id:
        headers["x-openai-account-user-id"] = account_user_id
    auth_mode = payload.get("authMode")
    if isinstance(auth_mode, str) and auth_mode:
        headers["x-basispoints-auth-mode"] = auth_mode
    expires_at = session_info.get("expires_at")
    if not isinstance(expires_at, (int, float)):
        expires_at = claims.get("exp")
    return headers, float(expires_at) if isinstance(expires_at, (int, float)) else 0.0


def _macos_database_paths(website_data: Path) -> list[Path]:
    return sorted(
        website_data.glob("Default/*/*/LocalStorage/localstorage.sqlite3"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )


def load_macos_excel_session(website_data: Path | None = None) -> dict[str, str]:
    root = website_data or _MACOS_WEBSITE_DATA
    candidates: list[tuple[float, dict[str, str]]] = []
    errors: list[str] = []
    for database in _macos_database_paths(root):
        try:
            uri = f"file:{urllib.parse.quote(str(database))}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=1)
            try:
                row = connection.execute(
                    "SELECT value FROM ItemTable WHERE key = ?",
                    (_MACOS_STORAGE_KEY,),
                ).fetchone()
            finally:
                connection.close()
            if row is None:
                continue
            headers, expires_at = _macos_session_headers(
                _decode_macos_storage_value(row[0])
            )
            candidates.append((expires_at, headers))
        except (OSError, sqlite3.Error, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{database}: {exc}")
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    if errors:
        raise RuntimeError(errors[0])
    raise RuntimeError(
        "No signed-in ChatGPT Excel session was found. Open the ChatGPT task pane "
        "in Excel and sign in."
    )


def refresh_macos_excel_session(
    session_store,
    *,
    force: bool = False,
    website_data: Path | None = None,
) -> dict[str, object]:
    if sys.platform != "darwin":
        return session_store.status()
    status = session_store.status()
    if not force and status.get("configured") and not status.get("expired"):
        return status
    with _MACOS_REFRESH_LOCK:
        status = session_store.status()
        if not force and status.get("configured") and not status.get("expired"):
            return status
        try:
            headers = load_macos_excel_session(website_data)
            return session_store.configure(
                headers,
                persist=False,
                allow_expired=True,
            )
        except (OSError, RuntimeError, ValueError, UnicodeError, json.JSONDecodeError):
            return session_store.status()


class ExcelSessionCaptureManager:
    def __init__(
        self,
        *,
        script_path: str,
        macos_script_path: str | None = None,
        macos_conf_dir: str | None = None,
        session_status_provider: Callable[[], dict[str, object]],
        proxy_url: str = "http://127.0.0.1:8000/api/config/excel-session",
    ):
        self._script_path = script_path
        self._macos_script_path = macos_script_path or os.path.join(
            os.path.dirname(script_path),
            "prime-excel-session-macos.py",
        )
        # Retained for caller compatibility with older versions.
        self._macos_conf_dir = macos_conf_dir
        self._session_status_provider = session_status_provider
        self._proxy_url = proxy_url
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._timeout_seconds: int | None = None
        self._baseline_configured_at: object = None
        self._captured = False
        self._stopping = False
        self._error = ""
        self._message = ""

    def _node_path(self) -> str:
        return shutil.which("node") or ""

    def _capture_method(self) -> str:
        if sys.platform == "win32":
            return "webview2-devtools"
        if sys.platform == "darwin":
            return "webkit-localstorage-sqlite"
        return "unavailable"

    def _macos_command(self, *, timeout_seconds: int) -> list[str]:
        del timeout_seconds
        return [
            sys.executable,
            self._macos_script_path,
            "--session-url",
            self._proxy_url,
        ]

    def available(self) -> bool:
        if sys.platform == "win32":
            return bool(self._node_path() and os.path.isfile(self._script_path))
        if sys.platform == "darwin":
            return os.path.isfile(self._macos_script_path)
        return False

    def _unavailable_message(self) -> str:
        if sys.platform == "darwin":
            return "The macOS WebKit LocalStorage session reader is missing."
        if sys.platform == "win32":
            return (
                "Excel session capture requires Windows, Node.js, and "
                "tools/prime-excel-session.js"
            )
        return "Excel session capture is supported on Windows and macOS."

    def start(self, *, devtools_port: int = 9222, timeout_seconds: int = 300) -> dict[str, object]:
        if not self.available():
            raise RuntimeError(self._unavailable_message())
        if not isinstance(devtools_port, int) or not 1 <= devtools_port <= 65_535:
            raise ValueError("DevTools port must be between 1 and 65535")
        if not isinstance(timeout_seconds, int) or not 10 <= timeout_seconds <= 900:
            raise ValueError("capture timeout must be between 10 and 900 seconds")

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self.status()
            command = (
                self._macos_command(timeout_seconds=timeout_seconds)
                if sys.platform == "darwin"
                else [
                    self._node_path(),
                    self._script_path,
                    "--devtools-port",
                    str(devtools_port),
                    "--proxy-url",
                    self._proxy_url,
                    "--timeout-ms",
                    str(timeout_seconds * 1000),
                ]
            )
            creation_flags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if sys.platform == "win32"
                else 0
            )
            self._baseline_configured_at = self._session_status_provider().get(
                "configured_at"
            )
            try:
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creation_flags,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"Could not start the Excel session capture helper: {exc}"
                ) from exc
            self._started_at = time.time()
            self._finished_at = None
            self._timeout_seconds = timeout_seconds
            self._captured = False
            self._stopping = False
            self._error = ""
            self._message = self._waiting_message()
            process = self._process
        threading.Thread(
            target=self._watch_capture_progress,
            args=(process,),
            name="ghcp-excel-session-capture-progress",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._watch_process,
            args=(process,),
            name="ghcp-excel-session-capture",
            daemon=True,
        ).start()
        return self.status()

    def _waiting_message(self) -> str:
        if sys.platform == "darwin":
            return "Reading the signed-in ChatGPT Excel session from WebKit LocalStorage."
        return (
            "Waiting for the next ChatGPT Excel prompt. Send one message in the "
            "Excel add-in."
        )

    def _capture_success_message(self) -> str:
        session_status = self._session_status_provider()
        if session_status.get("persisted"):
            return "GPT Excel session captured and saved securely."
        if sys.platform == "darwin":
            return "GPT Excel session loaded from WebKit LocalStorage."
        return "GPT Excel session captured."

    def _new_session_was_captured(self) -> bool:
        status = self._session_status_provider()
        return bool(
            status.get("configured")
            and status.get("configured_at") != self._baseline_configured_at
        )

    def _watch_capture_progress(self, process: subprocess.Popen[str]) -> None:
        while process.poll() is None:
            if self._new_session_was_captured():
                with self._lock:
                    if process is self._process:
                        self._captured = True
                        self._error = ""
                        self._message = self._capture_success_message()
                return
            time.sleep(0.05)

    def _watch_process(self, process: subprocess.Popen[str]) -> None:
        stdout, stderr = process.communicate()
        with self._lock:
            if process is not self._process:
                return
            self._finished_at = time.time()
            captured = self._captured or self._new_session_was_captured()
            if captured:
                self._captured = True
                self._error = ""
                self._message = self._capture_success_message()
            elif self._stopping:
                self._error = ""
                self._message = ""
            else:
                detail = (stderr or stdout or "Excel session capture failed").strip()
                self._error = detail[-600:]
                self._message = ""

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._stopping = True
        if process is not None and process.poll() is None:
            process.terminate()

    def status(self) -> dict[str, object]:
        with self._lock:
            process = self._process
            process_running = process is not None and process.poll() is None
            capturing = process_running and not self._captured and not self._stopping
            phase = ""
            if capturing:
                phase = "reading_local_storage" if sys.platform == "darwin" else "waiting_for_excel"
            elif process_running and self._captured:
                phase = "finishing"
            elif not process_running and self._captured:
                phase = "complete"
            result: dict[str, object] = {
                "available": self.available(),
                "capturing": capturing,
                "process_running": process_running,
                "method": self._capture_method(),
                "platform": sys.platform,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "timeout_seconds": self._timeout_seconds,
                "deadline_at": (
                    self._started_at + self._timeout_seconds
                    if self._started_at is not None and self._timeout_seconds is not None
                    else None
                ),
                "phase": phase,
                "proxy_active": False,
                "workflow_version": 3 if sys.platform == "darwin" else 1,
                "cancel_supported": True,
                "proxy_port": None,
                "excel_helper_restarted": False,
                "error": self._error,
                "message": self._message if not capturing else self._waiting_message(),
                "prerequisite_error": "" if self.available() else self._unavailable_message(),
                "devtools_default_port": 9222,
            }
            if sys.platform == "darwin":
                result.update(
                    {
                        "ca_certificate_path": "",
                        "ca_certificate_exists": False,
                        "ca_trusted": False,
                        "setup_required": False,
                        "network_service": "",
                        "ca_install_command": "",
                        "install_command": "",
                        "install_url": "",
                        "intercepted_host": "",
                    }
                )
            return result
