#!/usr/bin/env python3
"""Manually refresh GHCP Proxy from macOS Excel WebKit LocalStorage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from excel_session_capture import _MACOS_WEBSITE_DATA, load_macos_excel_session


def submit_session(session_url: str, headers: dict[str, str]) -> None:
    encoded = json.dumps({"headers": headers}, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        session_url,
        data=encoded,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail")
        except (UnicodeError, json.JSONDecodeError, AttributeError):
            detail = None
        raise RuntimeError(
            detail if isinstance(detail, str) and detail else f"GHCP Proxy returned HTTP {exc.code}"
        ) from exc
    if not isinstance(payload, dict) or not payload.get("configured"):
        raise RuntimeError("GHCP Proxy did not accept the stored Excel session")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-url",
        default="http://127.0.0.1:8000/api/config/excel-session",
    )
    parser.add_argument("--website-data", default=str(_MACOS_WEBSITE_DATA))
    args = parser.parse_args()
    try:
        submit_session(
            args.session_url,
            load_macos_excel_session(Path(args.website_data).expanduser()),
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not refresh the ChatGPT Excel session: {exc}", file=sys.stderr)
        return 1
    print("GPT Excel session refreshed from macOS WebKit LocalStorage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
