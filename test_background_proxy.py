import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import background_proxy


class BackgroundProxyTests(unittest.TestCase):
    def test_windows_startup_and_profile_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env = {
                "APPDATA": str(root / "Roaming"),
                "USERPROFILE": str(root / "User"),
            }
            manager = background_proxy.BackgroundProxyManager(
                repo_dir=str(root / "repo"),
                python_executable=str(root / "Python" / "python.exe"),
                platform="win32",
            )

            with mock.patch.dict(os.environ, env, clear=True):
                status = manager.enable_startup()
                startup_path = Path(status["startup_path"])
                self.assertTrue(startup_path.exists())
                startup_body = startup_path.read_text(encoding="utf-8")
                self.assertIn("Start-Process", startup_body)
                self.assertIn("proxy.py", startup_body)

                status = manager.install_shell_commands()
                profile_path = Path(status["shell_profile_path"])
                profile_body = profile_path.read_text(encoding="utf-8")
                self.assertIn("function Start-GHProxy", profile_body)
                self.assertIn("function Stop-GHProxy", profile_body)
                self.assertTrue(status["shell_commands_installed"])

                status = manager.disable_startup()
                self.assertFalse(Path(status["startup_path"]).exists())

                status = manager.uninstall_shell_commands()
                self.assertFalse(status["shell_commands_installed"])
                self.assertNotIn("Start-GHProxy", profile_path.read_text(encoding="utf-8"))

    def test_macos_launch_agent_and_zsh_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            home = root / "home"
            home.mkdir()
            manager = background_proxy.BackgroundProxyManager(
                repo_dir=str(root / "repo"),
                python_executable="/usr/bin/python3",
                platform="darwin",
            )

            with mock.patch.dict(os.environ, {"HOME": str(home)}, clear=True):
                status = manager.enable_startup()
                launch_agent = Path(status["startup_path"])
                self.assertTrue(launch_agent.exists())
                plist = launch_agent.read_text(encoding="utf-8")
                self.assertIn("com.ghcp-proxy", plist)
                self.assertIn("RunAtLoad", plist)

                status = manager.install_shell_commands()
                zshrc = Path(status["shell_profile_path"])
                body = zshrc.read_text(encoding="utf-8")
                self.assertIn("start-ghproxy()", body)
                self.assertIn("stop-ghproxy()", body)
                self.assertTrue(status["shell_commands_installed"])

    def test_linux_reports_unsupported(self):
        manager = background_proxy.BackgroundProxyManager(platform="linux")
        self.assertFalse(manager.startup_supported())
        self.assertFalse(manager.shell_commands_supported())
        self.assertEqual(manager.command_names(), {})


if __name__ == "__main__":
    unittest.main()
