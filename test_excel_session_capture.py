from __future__ import annotations

import importlib.util
import json
import os
import signal
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from excel_session_capture import ExcelSessionCaptureManager


ROOT = Path(__file__).resolve().parent
WRAPPER_PATH = ROOT / "tools" / "capture-excel-session-macos.py"
WRAPPER_SPEC = importlib.util.spec_from_file_location(
    "capture_excel_session_macos",
    WRAPPER_PATH,
)
assert WRAPPER_SPEC is not None and WRAPPER_SPEC.loader is not None
macos_capture = importlib.util.module_from_spec(WRAPPER_SPEC)
sys.modules[WRAPPER_SPEC.name] = macos_capture
WRAPPER_SPEC.loader.exec_module(macos_capture)


class MacCaptureWrapperTests(unittest.TestCase):
    def test_network_service_matches_primary_interface(self):
        output = """\
An asterisk (*) denotes that a network service is disabled.
(1) Thunderbolt Bridge
(Hardware Port: Thunderbolt Bridge, Device: bridge0)

(2) Wi-Fi
(Hardware Port: Wi-Fi, Device: en0)
"""
        with mock.patch.object(macos_capture, "_run", return_value=output):
            self.assertEqual(macos_capture._network_service("en0"), "Wi-Fi")

    def test_secure_proxy_state_is_parsed(self):
        output = """\
Enabled: Yes
Server: proxy.example
Port: 8443
Authenticated Proxy Enabled: 1
"""
        with mock.patch.object(macos_capture, "_run", return_value=output):
            self.assertEqual(
                macos_capture._secure_proxy_state("Wi-Fi"),
                macos_capture.SecureProxyState(
                    enabled=True,
                    server="proxy.example",
                    port=8443,
                    authenticated=True,
                ),
            )

    def test_disabled_proxy_configuration_is_restored_exactly(self):
        state = macos_capture.SecureProxyState(
            enabled=False,
            server="",
            port=0,
            authenticated=False,
        )
        with mock.patch.object(macos_capture, "_run") as run:
            macos_capture._restore_proxy("Wi-Fi", state)
        self.assertEqual(
            run.call_args_list,
            [
                mock.call(
                    [
                        macos_capture._NETWORKSETUP,
                        "-setsecurewebproxy",
                        "Wi-Fi",
                        "",
                        "0",
                    ]
                ),
                mock.call(
                    [
                        macos_capture._NETWORKSETUP,
                        "-setsecurewebproxystate",
                        "Wi-Fi",
                        "off",
                    ]
                ),
            ],
        )

    def test_excel_networking_helper_is_restarted(self):
        with (
            mock.patch.object(
                macos_capture,
                "_excel_networking_pids",
                side_effect=[[101], [202]],
            ),
            mock.patch.object(macos_capture.os, "kill") as kill,
        ):
            restarted, replacement_pid = (
                macos_capture._restart_excel_networking_helper()
            )
        self.assertTrue(restarted)
        self.assertEqual(replacement_pid, 202)
        kill.assert_called_once_with(101, signal.SIGKILL)

    def test_missing_excel_helper_is_not_an_error(self):
        with mock.patch.object(
            macos_capture,
            "_excel_networking_pids",
            return_value=[],
        ):
            self.assertEqual(
                macos_capture._restart_excel_networking_helper(),
                (False, None),
            )

    def test_status_file_is_private_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "capture-status.json")
            macos_capture._write_status(
                path,
                phase="waiting_for_excel",
                service="Wi-Fi",
                proxy_active=True,
                excel_helper_restarted=True,
                excel_helper_pid=202,
            )
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["phase"], "waiting_for_excel")
            self.assertEqual(payload["network_service"], "Wi-Fi")
            self.assertTrue(payload["proxy_active"])
            self.assertEqual(payload["excel_helper_pid"], 202)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)


class ExcelSessionCaptureManagerTests(unittest.TestCase):
    def _manager(self, directory: str) -> ExcelSessionCaptureManager:
        return ExcelSessionCaptureManager(
            script_path=str(ROOT / "tools" / "prime-excel-session.js"),
            macos_script_path=str(
                ROOT / "tools" / "capture-excel-session-mitm.py"
            ),
            macos_conf_dir=directory,
            session_status_provider=lambda: {
                "configured": False,
                "configured_at": None,
            },
        )

    def test_macos_command_uses_guarded_wrapper_and_status_file(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self._manager(directory)
            with mock.patch.object(
                manager,
                "_mitmdump_path",
                return_value="/usr/local/bin/mitmdump",
            ):
                command = manager._macos_command(timeout_seconds=600)
            self.assertEqual(command[1], str(WRAPPER_PATH))
            self.assertIn("--status-file", command)
            self.assertEqual(
                command[command.index("--status-file") + 1],
                os.path.join(directory, "capture-status.json"),
            )
            self.assertEqual(
                command[command.index("--timeout-seconds") + 1],
                "600",
            )

    def test_runtime_status_rejects_a_stale_wrapper_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self._manager(directory)
            path = manager._macos_capture_status_path()
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "wrapper_pid": 10,
                        "phase": "waiting_for_excel",
                        "proxy_active": True,
                    },
                    handle,
                )
            self.assertEqual(
                manager._macos_capture_runtime_status(
                    SimpleNamespace(pid=11)
                ),
                {},
            )


if __name__ == "__main__":
    unittest.main()
