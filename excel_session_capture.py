"""Background capture process for the ChatGPT Excel WebView session."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable


class ExcelSessionCaptureManager:
    def __init__(
        self,
        *,
        script_path: str,
        session_status_provider: Callable[[], dict[str, object]],
        proxy_url: str = "http://127.0.0.1:8000/api/config/excel-session",
    ):
        self._script_path = script_path
        self._session_status_provider = session_status_provider
        self._proxy_url = proxy_url
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._error = ""
        self._message = ""

    def _node_path(self) -> str:
        return shutil.which("node") or ""

    def available(self) -> bool:
        return bool(
            sys.platform == "win32"
            and self._node_path()
            and os.path.isfile(self._script_path)
        )

    def start(self, *, devtools_port: int = 9222, timeout_seconds: int = 300) -> dict[str, object]:
        if not self.available():
            raise RuntimeError(
                "Excel session capture requires Windows, Node.js, and tools/prime-excel-session.js"
            )
        if not isinstance(devtools_port, int) or not 1 <= devtools_port <= 65_535:
            raise ValueError("DevTools port must be between 1 and 65535")
        if not isinstance(timeout_seconds, int) or not 10 <= timeout_seconds <= 900:
            raise ValueError("capture timeout must be between 10 and 900 seconds")

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self.status()
            command = [
                self._node_path(),
                self._script_path,
                "--devtools-port",
                str(devtools_port),
                "--proxy-url",
                self._proxy_url,
                "--timeout-ms",
                str(timeout_seconds * 1000),
            ]
            creation_flags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if sys.platform == "win32"
                else 0
            )
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
            self._started_at = time.time()
            self._finished_at = None
            self._error = ""
            self._message = (
                "Waiting for the next ChatGPT Excel prompt. Send one message in the Excel add-in."
            )
            process = self._process
        threading.Thread(
            target=self._watch_process,
            args=(process,),
            name="ghcp-excel-session-capture",
            daemon=True,
        ).start()
        return self.status()

    def _watch_process(self, process: subprocess.Popen[str]) -> None:
        stdout, stderr = process.communicate()
        with self._lock:
            if process is not self._process:
                return
            self._finished_at = time.time()
            if process.returncode == 0 and bool(
                self._session_status_provider().get("configured")
            ):
                self._error = ""
                self._message = "GPT Excel session captured and encrypted with Windows DPAPI."
            else:
                detail = (stderr or stdout or "Excel session capture failed").strip()
                self._error = detail[-600:]
                self._message = ""

    def stop(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def status(self) -> dict[str, object]:
        with self._lock:
            process = self._process
            capturing = process is not None and process.poll() is None
            return {
                "available": self.available(),
                "capturing": capturing,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "error": self._error,
                "message": self._message,
                "devtools_default_port": 9222,
            }
