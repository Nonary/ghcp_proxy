#!/usr/bin/env python3
"""Run the Mac Excel capture through a temporary Secure Web Proxy.

Safari/WKWebView can send Basispoints requests over QUIC, which may bypass
mitmproxy's local-capture host filtering. An explicit HTTPS proxy forces the
WebView to use an HTTP CONNECT tunnel instead. The previous macOS proxy state
is restored whenever this wrapper exits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass


_NETWORKSETUP = "/usr/sbin/networksetup"
_ROUTE = "/sbin/route"
_IFCONFIG = "/sbin/ifconfig"
_LSAPPINFO = "/usr/bin/lsappinfo"
_PGREP = "/usr/bin/pgrep"
_LISTEN_HOST = "127.0.0.1"
_EXCEL_HOST_PATTERN = r"^bps\.openai\.com:443$"
_EXCEL_NETWORK_DISPLAY_NAME = "Microsoft Excel Networking"


@dataclass(frozen=True)
class SecureProxyState:
    enabled: bool
    server: str
    port: int
    authenticated: bool


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        raise RuntimeError(output or f"{command[0]} exited {result.returncode}")
    return output


def _primary_interface() -> str:
    output = _run([_ROUTE, "-n", "get", "default"])
    match = re.search(r"^\s*interface:\s*(\S+)", output, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not determine the active Mac network interface")
    return match.group(1)


def _network_service(interface: str) -> str:
    output = _run([_NETWORKSETUP, "-listnetworkserviceorder"])
    lines = output.splitlines()
    active_services: list[tuple[str, str]] = []
    for index, line in enumerate(lines[:-1]):
        match = re.match(r"^\(\d+\)\s+(\*?)(.+)$", line.strip())
        if not match:
            continue
        device_match = re.search(r"\bDevice:\s*([^,)]+)", lines[index + 1])
        if not device_match:
            continue
        service = match.group(2).strip()
        device = device_match.group(1).strip()
        if device == interface:
            return service
        if not match.group(1):
            active_services.append((service, device))

    # A full-tunnel VPN can make a transient utun device the default route.
    # Those devices are not necessarily represented as networksetup services,
    # but Secure Web Proxy settings still belong to the underlying active
    # Wi-Fi/Ethernet service. Prefer service order when more than one physical
    # interface is active.
    if interface.startswith("utun"):
        for service, device in active_services:
            try:
                device_state = _run([_IFCONFIG, device])
            except RuntimeError:
                continue
            if re.search(r"^\s*status:\s*active\s*$", device_state, re.MULTILINE):
                return service
    raise RuntimeError(
        f"Could not find the Mac network service for interface {interface}"
    )


def _secure_proxy_state(service: str) -> SecureProxyState:
    output = _run([_NETWORKSETUP, "-getsecurewebproxy", service])
    values: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    try:
        return SecureProxyState(
            enabled=values.get("Enabled", "").lower() == "yes",
            server=values.get("Server", ""),
            port=int(values.get("Port") or 0),
            authenticated=(
                values.get("Authenticated Proxy Enabled", "") == "1"
            ),
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse the existing Secure Web Proxy state: {output}"
        ) from exc


def _available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((_LISTEN_HOST, 0))
        return int(listener.getsockname()[1])


def _enable_capture_proxy(service: str, listen_port: int) -> None:
    _run(
        [
            _NETWORKSETUP,
            "-setsecurewebproxy",
            service,
            _LISTEN_HOST,
            str(listen_port),
        ]
    )
    _run([_NETWORKSETUP, "-setsecurewebproxystate", service, "on"])


def _restore_proxy(service: str, state: SecureProxyState) -> None:
    _run(
        [
            _NETWORKSETUP,
            "-setsecurewebproxy",
            service,
            state.server,
            str(state.port),
        ]
    )
    _run(
        [
            _NETWORKSETUP,
            "-setsecurewebproxystate",
            service,
            "on" if state.enabled else "off",
        ]
    )


def _wait_until_listening(
    process: subprocess.Popen[bytes],
    listen_port: int,
) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"mitmdump exited before opening port {listen_port}"
            )
        try:
            with socket.create_connection(
                (_LISTEN_HOST, listen_port),
                timeout=0.2,
            ):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(
        f"mitmdump did not open {_LISTEN_HOST}:{listen_port}"
    )


def _excel_networking_pids() -> list[int]:
    result = subprocess.run(
        [_PGREP, "-x", "com.apple.WebKit.Networking"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("Could not inspect Excel's WebKit networking helper")
    matching: list[int] = []
    for value in result.stdout.split():
        try:
            process_id = int(value)
        except ValueError:
            continue
        try:
            display_name = _run(
                [_LSAPPINFO, "info", "-only", "name", str(process_id)]
            )
        except RuntimeError:
            continue
        if _EXCEL_NETWORK_DISPLAY_NAME in display_name:
            matching.append(process_id)
    return matching


def _restart_excel_networking_helper() -> tuple[bool, int | None]:
    previous_process_ids = set(_excel_networking_pids())
    for process_id in previous_process_ids:
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeError(
                "Could not reconnect the ChatGPT Excel task pane"
            ) from exc
    if not previous_process_ids:
        return False, None

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current_process_ids = set(_excel_networking_pids())
        replacements = current_process_ids - previous_process_ids
        if replacements:
            return True, min(replacements)
        time.sleep(0.1)
    # WebKit may defer relaunching the helper until the task pane next performs
    # network activity. The old QUIC connection is still gone, which is enough.
    return True, None


def _write_status(
    path: str,
    *,
    phase: str,
    service: str = "",
    proxy_active: bool = False,
    proxy_port: int | None = None,
    excel_helper_restarted: bool = False,
    excel_helper_pid: int | None = None,
) -> None:
    payload = {
        "wrapper_pid": os.getpid(),
        "phase": phase,
        "network_service": service,
        "proxy_active": proxy_active,
        "proxy_host": _LISTEN_HOST if proxy_active else "",
        "proxy_port": proxy_port if proxy_active else None,
        "excel_helper_restarted": excel_helper_restarted,
        "excel_helper_pid": excel_helper_pid,
        "updated_at": time.time(),
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, mode=0o700, exist_ok=True)
    temporary_path = f"{path}.{os.getpid()}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.chmod(temporary_path, 0o600)
    os.replace(temporary_path, path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitmdump", required=True)
    parser.add_argument("--addon", required=True)
    parser.add_argument("--confdir", required=True)
    parser.add_argument("--session-url", required=True)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--status-file", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    service = _network_service(_primary_interface())
    _write_status(arguments.status_file, phase="starting", service=service)
    previous_state = _secure_proxy_state(service)
    if previous_state.enabled:
        raise RuntimeError(
            "Mac Excel capture cannot temporarily replace an existing Secure "
            "Web Proxy. Disable it first, then try capture again"
        )
    if previous_state.authenticated:
        raise RuntimeError(
            "Mac Excel capture cannot temporarily replace an authenticated "
            "Secure Web Proxy because macOS does not expose its saved password"
        )
    listen_port = _available_loopback_port()
    command = [
        arguments.mitmdump,
        "--mode",
        "regular",
        "--listen-host",
        _LISTEN_HOST,
        "--listen-port",
        str(listen_port),
        "--allow-hosts",
        _EXCEL_HOST_PATTERN,
        "-s",
        arguments.addon,
        "--set",
        f"confdir={arguments.confdir}",
        "--set",
        f"ghcp_excel_session_url={arguments.session_url}",
        "--set",
        f"ghcp_excel_capture_timeout={arguments.timeout_seconds}",
        "--set",
        "flow_detail=0",
        "--set",
        "termlog_verbosity=error",
        "--set",
        "show_ignored_hosts=false",
    ]
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    received_signal = 0

    def stop_child(signum: int, _frame) -> None:
        nonlocal received_signal
        received_signal = signum
        if child.poll() is None:
            child.terminate()

    previous_handlers = {
        signum: signal.signal(signum, stop_child)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    proxy_configuration_started = False
    try:
        _wait_until_listening(child, listen_port)
        proxy_configuration_started = True
        _enable_capture_proxy(service, listen_port)
        _write_status(
            arguments.status_file,
            phase="proxy_ready",
            service=service,
            proxy_active=True,
            proxy_port=listen_port,
        )
        helper_restarted, helper_pid = _restart_excel_networking_helper()
        _write_status(
            arguments.status_file,
            phase="waiting_for_excel",
            service=service,
            proxy_active=True,
            proxy_port=listen_port,
            excel_helper_restarted=helper_restarted,
            excel_helper_pid=helper_pid,
        )
        print(
            (
                f"Temporary Secure Web Proxy enabled for {service}; "
                "Excel WebKit connection refreshed"
                if helper_restarted
                else (
                    f"Temporary Secure Web Proxy enabled for {service}; "
                    "open the ChatGPT Excel task pane"
                )
            ),
            file=sys.stderr,
            flush=True,
        )
        return_code = child.wait()
        return 128 + received_signal if received_signal else return_code
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if proxy_configuration_started:
            try:
                _write_status(
                    arguments.status_file,
                    phase="restoring",
                    service=service,
                    proxy_active=True,
                    proxy_port=listen_port,
                )
                _restore_proxy(service, previous_state)
                _write_status(
                    arguments.status_file,
                    phase="restored",
                    service=service,
                )
                print(
                    f"Secure Web Proxy restored for {service}",
                    file=sys.stderr,
                    flush=True,
                )
            except RuntimeError as exc:
                print(
                    f"WARNING: Could not restore Secure Web Proxy: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print(f"Mac Excel capture failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
