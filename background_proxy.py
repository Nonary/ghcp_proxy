"""Install background startup and shell commands for ghcp_proxy."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from constants import (
    PROXY_PID_FILE,
    PROXY_STDERR_LOG_FILE,
    PROXY_STDOUT_LOG_FILE,
    TOKEN_DIR,
)

START_MARKER = "# >>> ghcp_proxy commands >>>"
END_MARKER = "# <<< ghcp_proxy commands <<<"


def _quote_ps(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _profile_has_block(path: str) -> bool:
    return START_MARKER in _read_text(path)


def _expand_user_path(path: str) -> str:
    if path.startswith("~/"):
        home = os.environ.get("HOME")
        if home:
            return os.path.join(home, path[2:])
    return os.path.expanduser(path)


def _replace_profile_block(path: str, block: str | None) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            current = f.read()
    except OSError:
        current = ""

    start = current.find(START_MARKER)
    end = current.find(END_MARKER)
    if start != -1 and end != -1 and end >= start:
        end += len(END_MARKER)
        current = current[:start].rstrip() + "\n\n" + current[end:].lstrip()

    if block:
        current = current.rstrip() + "\n\n" + block.rstrip() + "\n"
    elif not current.strip():
        current = ""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(current)


@dataclass(frozen=True)
class BackgroundProxyManager:
    repo_dir: str = os.path.dirname(__file__)
    python_executable: str = sys.executable
    platform: str = sys.platform

    @property
    def proxy_script(self) -> str:
        return os.path.join(self.repo_dir, "proxy.py")

    def status_payload(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "startup_supported": self.startup_supported(),
            "startup_installed": self.startup_installed(),
            "startup_current": self.startup_current(),
            "startup_enabled": self.startup_enabled(),
            "startup_path": self.startup_path(),
            "shell_commands_supported": self.shell_commands_supported(),
            "shell_commands_current": self.shell_commands_current(),
            "shell_commands_installed": self.shell_commands_installed(),
            "shell_profile_path": self.shell_profile_path(),
            "commands": self.command_names(),
            "pid_file": PROXY_PID_FILE,
            "stdout_log": PROXY_STDOUT_LOG_FILE,
            "stderr_log": PROXY_STDERR_LOG_FILE,
        }

    def startup_supported(self) -> bool:
        return self.platform in {"win32", "darwin"}

    def shell_commands_supported(self) -> bool:
        return self.platform in {"win32", "darwin"}

    def command_names(self) -> dict[str, str]:
        if self.platform == "win32":
            return {"start": "Start-GHProxy", "stop": "Stop-GHProxy"}
        if self.platform == "darwin":
            return {"start": "start-ghproxy", "stop": "stop-ghproxy"}
        return {}

    def startup_path(self) -> str:
        if self.platform == "win32":
            appdata = os.environ.get("APPDATA") or os.path.expanduser("~\\AppData\\Roaming")
            return os.path.join(
                appdata,
                "Microsoft",
                "Windows",
                "Start Menu",
                "Programs",
                "Startup",
                "Start-GHProxy.cmd",
            )
        if self.platform == "darwin":
            return _expand_user_path("~/Library/LaunchAgents/com.ghcp-proxy.plist")
        return ""

    def shell_profile_path(self) -> str:
        if self.platform == "win32":
            user_profile = os.environ.get("USERPROFILE") or os.path.expanduser("~")
            return os.path.join(user_profile, "Documents", "PowerShell", "Microsoft.PowerShell_profile.ps1")
        if self.platform == "darwin":
            return _expand_user_path("~/.zshrc")
        return ""

    def startup_installed(self) -> bool:
        path = self.startup_path()
        return bool(path and os.path.exists(path))

    def startup_current(self) -> bool:
        path = self.startup_path()
        if not path or not os.path.exists(path):
            return False
        return self._text_targets_current_proxy(_read_text(path))

    def startup_enabled(self) -> bool:
        return self.startup_current()

    def shell_commands_current(self) -> bool:
        path = self.shell_profile_path()
        if not path or not _profile_has_block(path):
            return False
        return self._text_targets_current_proxy(_read_text(path))

    def shell_commands_installed(self) -> bool:
        return self.shell_commands_current()

    def _text_targets_current_proxy(self, text: str) -> bool:
        if not text:
            return False
        proxy_script = os.path.abspath(self.proxy_script)
        candidates = {
            proxy_script,
            proxy_script.replace("\\", "/"),
            _quote_xml(proxy_script),
            _quote_xml(proxy_script.replace("\\", "/")),
        }
        return any(candidate and candidate in text for candidate in candidates)

    def enable_startup(self) -> dict[str, object]:
        if not self.startup_supported():
            raise RuntimeError("background startup is only supported on Windows and macOS")
        path = self.startup_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        content = self._windows_startup_cmd() if self.platform == "win32" else self._macos_launch_agent()
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return self.status_payload()

    def disable_startup(self) -> dict[str, object]:
        path = self.startup_path()
        if path and os.path.exists(path):
            os.remove(path)
        return self.status_payload()

    def install_shell_commands(self) -> dict[str, object]:
        if not self.shell_commands_supported():
            raise RuntimeError("shell commands are only supported on PowerShell for Windows and zsh on macOS")
        block = self._powershell_profile_block() if self.platform == "win32" else self._zsh_profile_block()
        _replace_profile_block(self.shell_profile_path(), block)
        return self.status_payload()

    def uninstall_shell_commands(self) -> dict[str, object]:
        path = self.shell_profile_path()
        if path:
            _replace_profile_block(path, None)
        return self.status_payload()

    def _windows_startup_cmd(self) -> str:
        command = (
            f"Start-Process -WindowStyle Hidden -FilePath {_quote_ps(self.python_executable)} "
            f"-ArgumentList @({_quote_ps(self.proxy_script)}) -WorkingDirectory {_quote_ps(self.repo_dir)} "
            f"-RedirectStandardOutput {_quote_ps(PROXY_STDOUT_LOG_FILE)} -RedirectStandardError {_quote_ps(PROXY_STDERR_LOG_FILE)}"
        )
        return f"@echo off\npowershell -NoProfile -WindowStyle Hidden -Command \"{command}\"\n"

    def _macos_launch_agent(self) -> str:
        os.makedirs(TOKEN_DIR, exist_ok=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.ghcp-proxy</string>
  <key>ProgramArguments</key>
  <array>
    <string>{_quote_xml(self.python_executable)}</string>
    <string>{_quote_xml(self.proxy_script)}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>{_quote_xml(self.repo_dir)}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>{_quote_xml(PROXY_STDOUT_LOG_FILE)}</string>
  <key>StandardErrorPath</key>
  <string>{_quote_xml(PROXY_STDERR_LOG_FILE)}</string>
</dict>
</plist>
"""

    def _powershell_profile_block(self) -> str:
        python = _quote_ps(self.python_executable)
        script = _quote_ps(self.proxy_script)
        repo = _quote_ps(self.repo_dir)
        pid_file = _quote_ps(PROXY_PID_FILE)
        stdout = _quote_ps(PROXY_STDOUT_LOG_FILE)
        stderr = _quote_ps(PROXY_STDERR_LOG_FILE)
        return f"""{START_MARKER}
function Start-GHProxy {{
    $client = New-Object System.Net.Sockets.TcpClient
    try {{
        $client.Connect('127.0.0.1', 8000)
        Write-Host 'GHCP Proxy is already listening on http://localhost:8000'
        return
    }} catch {{ }} finally {{
        $client.Dispose()
    }}
    New-Item -ItemType Directory -Force -Path (Split-Path {pid_file}) | Out-Null
    Start-Process -WindowStyle Hidden -FilePath {python} -ArgumentList @({script}) -WorkingDirectory {repo} -RedirectStandardOutput {stdout} -RedirectStandardError {stderr}
    Write-Host 'GHCP Proxy started in the background at http://localhost:8000'
}}

function Stop-GHProxy {{
    $pidPath = {pid_file}
    if (Test-Path $pidPath) {{
        $raw = Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1
        $proxyPid = 0
        if ([int]::TryParse([string]$raw, [ref]$proxyPid)) {{
            $proc = Get-Process -Id $proxyPid -ErrorAction SilentlyContinue
            if ($proc) {{
                Stop-Process -Id $proxyPid
                Write-Host 'GHCP Proxy stopped.'
                return
            }}
        }}
    }}
    Write-Host 'No GHCP Proxy pid file/process was found.'
}}
{END_MARKER}"""

    def _zsh_profile_block(self) -> str:
        python = shlex_quote(self.python_executable)
        script = shlex_quote(self.proxy_script)
        repo = shlex_quote(self.repo_dir)
        pid_file = shlex_quote(PROXY_PID_FILE)
        stdout = shlex_quote(PROXY_STDOUT_LOG_FILE)
        stderr = shlex_quote(PROXY_STDERR_LOG_FILE)
        return f"""{START_MARKER}
start-ghproxy() {{
  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "GHCP Proxy is already listening on http://localhost:8000"
    return 0
  fi
  mkdir -p "$(dirname {pid_file})"
  (cd {repo} && nohup {python} {script} >> {stdout} 2>> {stderr} &)
  echo "GHCP Proxy started in the background at http://localhost:8000"
}}

stop-ghproxy() {{
  if [[ -f {pid_file} ]]; then
    local proxy_pid
    proxy_pid="$(cat {pid_file} 2>/dev/null)"
    if [[ -n "$proxy_pid" ]] && kill -0 "$proxy_pid" 2>/dev/null; then
      kill "$proxy_pid"
      echo "GHCP Proxy stopped."
      return 0
    fi
  fi
  echo "No GHCP Proxy pid file/process was found."
}}
{END_MARKER}"""


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)
