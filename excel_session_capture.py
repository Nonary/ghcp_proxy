"""Background capture process for the ChatGPT Excel WebView session.

Windows Excel exposes its WebView2 instance over Chrome DevTools Protocol.
Excel for macOS uses Safari's WKWebView instead, so the Mac capture path uses
a temporary mitmproxy Secure Web Proxy restricted to bps.openai.com.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import ssl
import subprocess
import sys
import threading
import time
from collections.abc import Callable

from app_paths import user_state_dir


_EXCEL_CAPTURE_HOST = "bps.openai.com"
_MACOS_MITMPROXY_DIR_NAME = "excel-capture-mitmproxy"
_MACOS_CA_CERTIFICATE_NAME = "mitmproxy-ca-cert.pem"
_MACOS_CAPTURE_STATUS_NAME = "capture-status.json"


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
            "capture-excel-session-mitm.py",
        )
        self._macos_proxy_wrapper_path = os.path.join(
            os.path.dirname(self._macos_script_path),
            "capture-excel-session-macos.py",
        )
        self._macos_conf_dir = macos_conf_dir or os.path.join(
            user_state_dir(),
            _MACOS_MITMPROXY_DIR_NAME,
        )
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

    def _mitmdump_path(self) -> str:
        discovered = shutil.which("mitmdump")
        if discovered:
            return discovered
        for candidate in (
            os.path.expanduser("~/.local/bin/mitmdump"),
            "/opt/homebrew/bin/mitmdump",
            "/usr/local/bin/mitmdump",
            os.path.expanduser(
                "~/Applications/mitmproxy.app/Contents/MacOS/mitmdump"
            ),
            "/Applications/mitmproxy.app/Contents/MacOS/mitmdump",
        ):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return ""

    def _brew_path(self) -> str:
        discovered = shutil.which("brew")
        if discovered:
            return discovered
        for candidate in (
            os.path.expanduser("~/.local/bin/brew"),
            "/opt/homebrew/bin/brew",
            "/usr/local/bin/brew",
        ):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return ""

    def _capture_method(self) -> str:
        if sys.platform == "win32":
            return "webview2-devtools"
        if sys.platform == "darwin":
            return "mitmproxy-secure-web-proxy"
        return "unavailable"

    def _macos_ca_certificate_path(self) -> str:
        return os.path.join(self._macos_conf_dir, _MACOS_CA_CERTIFICATE_NAME)

    def _macos_keychain_path(self) -> str:
        return os.path.expanduser("~/Library/Keychains/login.keychain-db")

    def _macos_capture_status_path(self) -> str:
        return os.path.join(
            self._macos_conf_dir,
            _MACOS_CAPTURE_STATUS_NAME,
        )

    def _macos_capture_runtime_status(
        self,
        process: subprocess.Popen[str] | None,
    ) -> dict[str, object]:
        if process is None:
            return {}
        try:
            with open(
                self._macos_capture_status_path(),
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("wrapper_pid") != process.pid:
            return {}
        return payload

    def _macos_ca_trusted(self) -> bool:
        certificate_path = self._macos_ca_certificate_path()
        if sys.platform != "darwin" or not os.path.isfile(certificate_path):
            return False
        try:
            with open(certificate_path, encoding="ascii") as handle:
                certificate_pem = handle.read()
            certificate_der = ssl.PEM_cert_to_DER_cert(certificate_pem)
            fingerprint = hashlib.sha256(certificate_der).hexdigest().upper()
            result = subprocess.run(
                [
                    "/usr/bin/security",
                    "find-certificate",
                    "-a",
                    "-Z",
                    self._macos_keychain_path(),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3,
                check=False,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return False
        normalized_output = result.stdout.replace(" ", "").upper()
        return result.returncode == 0 and fingerprint in normalized_output

    def _macos_ca_install_command(self) -> str:
        certificate_path = self._macos_ca_certificate_path()
        return shlex.join(
            [
                "/usr/bin/security",
                "add-trusted-cert",
                "-r",
                "trustRoot",
                "-p",
                "ssl",
                "-s",
                "bps.openai.com",
                "-k",
                self._macos_keychain_path(),
                certificate_path,
            ]
        )

    def _macos_command(self, *, timeout_seconds: int) -> list[str]:
        return [
            sys.executable,
            self._macos_proxy_wrapper_path,
            "--mitmdump",
            self._mitmdump_path(),
            "--addon",
            self._macos_script_path,
            "--confdir",
            self._macos_conf_dir,
            "--session-url",
            self._proxy_url,
            "--timeout-seconds",
            str(timeout_seconds),
            "--status-file",
            self._macos_capture_status_path(),
        ]

    def available(self) -> bool:
        if sys.platform == "win32":
            return bool(self._node_path() and os.path.isfile(self._script_path))
        if sys.platform == "darwin":
            return bool(
                self._mitmdump_path()
                and os.path.isfile(self._macos_script_path)
                and os.path.isfile(self._macos_proxy_wrapper_path)
            )
        return False

    def _unavailable_message(self) -> str:
        if sys.platform == "darwin":
            if not self._mitmdump_path():
                return (
                    "Mac Excel capture requires mitmproxy. Install its macOS "
                    "package (or use Homebrew), then refresh this page."
                )
            return "Mac Excel capture helper tools are missing."
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
            if sys.platform == "darwin":
                os.makedirs(self._macos_conf_dir, mode=0o700, exist_ok=True)
                try:
                    os.chmod(self._macos_conf_dir, 0o700)
                except OSError:
                    pass
                try:
                    os.unlink(self._macos_capture_status_path())
                except FileNotFoundError:
                    pass
                command = self._macos_command(timeout_seconds=timeout_seconds)
            else:
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

    def _waiting_message(
        self,
        *,
        certificate_exists: bool | None = None,
        certificate_trusted: bool | None = None,
    ) -> str:
        if sys.platform == "darwin":
            if certificate_exists is None:
                certificate_exists = os.path.isfile(
                    self._macos_ca_certificate_path()
                )
            if not certificate_exists:
                return (
                    "Starting the restricted Mac capture proxy and creating its "
                    "dedicated certificate."
                )
            if certificate_trusted is None:
                certificate_trusted = self._macos_ca_trusted()
            if not certificate_trusted:
                return (
                    "Trust the dedicated bps.openai.com capture certificate, "
                    "then send one message in the ChatGPT Excel add-in."
                )
        return (
            "Waiting for the next ChatGPT Excel prompt. Send one message in the "
            "Excel add-in."
        )

    def _capture_success_message(self) -> str:
        session_status = self._session_status_provider()
        if session_status.get("persisted"):
            return "GPT Excel session captured and saved securely."
        if sys.platform == "darwin":
            return (
                "GPT Excel session captured in memory. Recapture it after the "
                "proxy restarts."
            )
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
            time.sleep(0.25)

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
            capturing = (
                process_running
                and not self._captured
                and not self._stopping
            )
            certificate_path = ""
            certificate_exists = False
            certificate_trusted = False
            macos_runtime: dict[str, object] = {}
            if sys.platform == "darwin":
                certificate_path = self._macos_ca_certificate_path()
                certificate_exists = os.path.isfile(certificate_path)
                certificate_trusted = (
                    self._macos_ca_trusted() if certificate_exists else False
                )
                macos_runtime = self._macos_capture_runtime_status(process)
            capture_phase = str(macos_runtime.get("phase") or "")
            proxy_active = bool(macos_runtime.get("proxy_active"))
            if process_running and self._captured:
                capture_phase = "finishing"
            elif not process_running and self._captured:
                capture_phase = "complete"
            elif self._stopping and process_running:
                capture_phase = "restoring"
            capture_message = self._message
            if capturing:
                if sys.platform == "darwin" and not certificate_trusted:
                    capture_message = self._waiting_message(
                        certificate_exists=certificate_exists,
                        certificate_trusted=certificate_trusted,
                    )
                elif (
                    sys.platform == "darwin"
                    and proxy_active
                    and capture_phase == "waiting_for_excel"
                ):
                    capture_message = (
                        "Ready. Send one message in the ChatGPT Excel task "
                        "pane; capture and proxy cleanup are automatic."
                    )
                elif sys.platform == "darwin" and capture_phase in {
                    "",
                    "starting",
                    "proxy_ready",
                }:
                    capture_message = (
                        "Starting the protected Mac capture and reconnecting "
                        "Excel…"
                    )
                else:
                    capture_message = self._waiting_message(
                        certificate_exists=certificate_exists,
                        certificate_trusted=certificate_trusted,
                    )
            elif self._stopping and process_running:
                capture_message = (
                    "Stopping capture and restoring the previous Mac proxy "
                    "settings…"
                )
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
                    if self._started_at is not None
                    and self._timeout_seconds is not None
                    else None
                ),
                "phase": capture_phase,
                "proxy_active": proxy_active,
                "workflow_version": 2,
                "cancel_supported": True,
                "proxy_port": (
                    int(macos_runtime.get("proxy_port"))
                    if isinstance(macos_runtime.get("proxy_port"), int)
                    else None
                ),
                "excel_helper_restarted": bool(
                    macos_runtime.get("excel_helper_restarted")
                ),
                "error": self._error,
                "message": capture_message,
                "prerequisite_error": (
                    "" if self.available() else self._unavailable_message()
                ),
                "devtools_default_port": 9222,
            }
            if sys.platform == "darwin":
                result.update(
                    {
                        "ca_certificate_path": certificate_path,
                        "ca_certificate_exists": certificate_exists,
                        "ca_trusted": certificate_trusted,
                        "setup_required": not certificate_trusted,
                        "network_service": str(
                            macos_runtime.get("network_service") or ""
                        ),
                        "ca_install_command": (
                            self._macos_ca_install_command()
                            if certificate_exists
                            else ""
                        ),
                        "install_command": (
                            (
                                f"{shlex.quote(self._brew_path())} install "
                                "--cask mitmproxy"
                            )
                            if not self._mitmdump_path() and self._brew_path()
                            else ""
                        ),
                        "install_url": (
                            "https://mitmproxy.org/"
                            if not self._mitmdump_path()
                            else ""
                        ),
                        "intercepted_host": (
                            f"{_EXCEL_CAPTURE_HOST}:443"
                        ),
                    }
                )
            return result
