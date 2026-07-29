"""Capture an Excel Basispoints session from mitmproxy.

This add-on deliberately inspects only the exact Responses endpoint. The
mitmdump process is started with allow_hosts restricted to bps.openai.com, so
traffic for every other host passes through without TLS interception.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

from mitmproxy import ctx, http


_RESPONSES_HOST = "bps.openai.com"
_RESPONSES_PATH = "/basispoints/api/responses"
_TOOLS_VERSION_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
)
_ALLOWED_HEADERS = frozenset(
    {
        "authorization",
        "chatgpt-account-id",
        "user-agent",
        "x-basispoints-auth-mode",
        "x-openai-account-id",
        "x-openai-account-user-id",
        "x-openai-internal-basispoints-browser-name",
        "x-openai-internal-basispoints-browser-ua-brands",
        "x-openai-internal-basispoints-browser-ua-mobile",
        "x-openai-internal-basispoints-browser-ua-platform",
        "x-openai-internal-basispoints-client-agent-profile",
        "x-openai-internal-basispoints-client-editor",
        "x-openai-internal-basispoints-client-host",
        "x-openai-internal-basispoints-client-platform",
        "x-openai-internal-basispoints-client-platform-class",
        "x-openai-internal-basispoints-client-product",
        "x-openai-internal-basispoints-client-runtime",
        "x-openai-internal-basispoints-office-host",
        "x-openai-internal-basispoints-office-platform",
        "x-stainless-arch",
        "x-stainless-lang",
        "x-stainless-os",
        "x-stainless-package-version",
        "x-stainless-retry-count",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
    }
)


def _valid_tools_version_id(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and 1 <= len(value) <= 160
        and all(character in _TOOLS_VERSION_CHARACTERS for character in value)
    )


def _tools_version_id(flow: http.HTTPFlow) -> str | None:
    try:
        body = json.loads(flow.request.content)
    except (TypeError, UnicodeError, json.JSONDecodeError):
        return None
    value = (
        body.get("metadata", {}).get("bps_tools_version_id")
        if isinstance(body, dict) and isinstance(body.get("metadata"), dict)
        else None
    )
    return value if _valid_tools_version_id(value) else None


def _selected_headers(flow: http.HTTPFlow) -> dict[str, str]:
    return {
        name: value
        for name in _ALLOWED_HEADERS
        if isinstance((value := flow.request.headers.get(name)), str) and value
    }


class ExcelSessionCapture:
    def __init__(self) -> None:
        self._captured = False
        self._captured_flow_id = ""
        self._timeout_task: asyncio.Task | None = None
        self._shutdown_task: asyncio.Task | None = None
        self._last_error = ""

    def load(self, loader) -> None:
        loader.add_option(
            "ghcp_excel_session_url",
            str,
            "http://127.0.0.1:8000/api/config/excel-session",
            "Loopback GHCP Proxy Excel session endpoint.",
        )
        loader.add_option(
            "ghcp_excel_capture_timeout",
            int,
            300,
            "Seconds to wait for a matching Excel request.",
        )

    def running(self) -> None:
        self._timeout_task = asyncio.create_task(self._stop_after_timeout())

    async def _stop_after_timeout(self) -> None:
        await asyncio.sleep(ctx.options.ghcp_excel_capture_timeout)
        if not self._captured:
            detail = (
                f": {self._last_error}" if self._last_error else ""
            )
            ctx.log.error(
                "Timed out waiting for a ChatGPT Excel Responses request" + detail
            )
        ctx.master.shutdown()

    async def _stop_after_capture_grace(self) -> None:
        # response() normally shuts down after Excel receives its complete
        # upstream response. This fallback prevents a long-lived SSE connection
        # from leaving the capture process running forever.
        await asyncio.sleep(60)
        ctx.master.shutdown()

    def request(self, flow: http.HTTPFlow) -> None:
        if self._captured:
            return
        request = flow.request
        if (
            request.method.upper() != "POST"
            or request.pretty_host.lower() != _RESPONSES_HOST
            or request.path.split("?", 1)[0] != _RESPONSES_PATH
        ):
            return

        payload = {
            "headers": _selected_headers(flow),
            "tools_version_id": _tools_version_id(flow),
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        capture_request = urllib.request.Request(
            ctx.options.ghcp_excel_session_url,
            data=encoded,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(capture_request, timeout=5) as response:
                response_payload = json.loads(response.read())
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError(
                        f"GHCP Proxy returned HTTP {response.status}"
                    )
                if not isinstance(response_payload, dict) or not response_payload.get(
                    "configured"
                ):
                    raise RuntimeError("GHCP Proxy did not accept the captured session")
        except urllib.error.HTTPError as exc:
            try:
                error_payload = json.loads(exc.read())
                detail = error_payload.get("detail")
            except (UnicodeError, json.JSONDecodeError):
                detail = None
            self._last_error = (
                str(detail)
                if isinstance(detail, str) and detail
                else f"GHCP Proxy returned HTTP {exc.code}"
            )
            ctx.log.error(self._last_error)
            return
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            self._last_error = str(exc)
            ctx.log.error(f"Could not save the Excel session: {exc}")
            return

        self._captured = True
        self._captured_flow_id = flow.id
        flow.metadata["ghcp_excel_session_captured"] = True
        print("GHCP_EXCEL_SESSION_CAPTURED", flush=True)
        if self._timeout_task is not None:
            self._timeout_task.cancel()
        self._shutdown_task = asyncio.create_task(self._stop_after_capture_grace())

    def response(self, flow: http.HTTPFlow) -> None:
        if flow.id == self._captured_flow_id:
            ctx.master.shutdown()

    def error(self, flow: http.HTTPFlow) -> None:
        if flow.id == self._captured_flow_id:
            ctx.master.shutdown()

    def done(self) -> None:
        for task in (self._timeout_task, self._shutdown_task):
            if task is not None:
                task.cancel()


addons = [ExcelSessionCapture()]
