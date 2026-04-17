"""GitHub OAuth device flow, token management, and API key handling."""

import json
import os
import sys
import time
from datetime import datetime, timezone

import httpx

from constants import (
    GITHUB_CLIENT_ID, GITHUB_DEVICE_CODE_URL, GITHUB_ACCESS_TOKEN_URL,
    GITHUB_API_KEY_URL,
    TOKEN_DIR, ACCESS_TOKEN_FILE, API_KEY_FILE,
    GITHUB_COPILOT_API_BASE,
)


def _gh_headers(access_token: str = None) -> dict:
    h = {
        "accept": "application/json",
        "editor-version": "vscode/1.85.1",
        "editor-plugin-version": "copilot/1.155.0",
        "user-agent": "GithubCopilot/1.155.0",
        "accept-encoding": "gzip,deflate,br",
        "content-type": "application/json",
    }
    if access_token:
        h["authorization"] = f"token {access_token}"
    return h


def load_access_token() -> str | None:
    try:
        with open(ACCESS_TOKEN_FILE, encoding="utf-8") as f:
            tok = f.read().strip()
        return tok or None
    except OSError:
        return None


def _save_access_token(token: str):
    os.makedirs(TOKEN_DIR, exist_ok=True)
    with open(ACCESS_TOKEN_FILE, "w") as f:
        f.write(token)


def load_api_key() -> str | None:
    try:
        with open(API_KEY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if data["expires_at"] > datetime.now().timestamp():
            return data["token"]
    except Exception:
        pass
    return None


def load_api_key_payload() -> dict:
    try:
        with open(API_KEY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_api_base() -> str:
    """Use the endpoint embedded in api-key.json if present, else default."""
    try:
        with open(API_KEY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("endpoints", {}).get("api") or GITHUB_COPILOT_API_BASE
    except Exception:
        return GITHUB_COPILOT_API_BASE


def _device_flow() -> str:
    """
    Interactive GitHub OAuth device flow.
    Prints the verification URL and code to the terminal, then polls until
    the user authorizes (or times out after ~60 seconds).
    Returns the GitHub OAuth access token.
    """
    with httpx.Client() as c:
        r = c.post(
            GITHUB_DEVICE_CODE_URL,
            headers=_gh_headers(),
            json={"client_id": GITHUB_CLIENT_ID, "scope": "read:user read:org"},
        )
        r.raise_for_status()
        info = r.json()

    print("", flush=True)
    print("─" * 60, flush=True)
    print("  GitHub Copilot — Authorization Required", flush=True)
    print("─" * 60, flush=True)
    print(f"  1. Open:  {info['verification_uri']}", flush=True)
    print(f"  2. Enter: {info['user_code']}", flush=True)
    print("─" * 60, flush=True)
    print("  Waiting for authorization...", flush=True)

    interval = info.get("interval", 5)
    max_attempts = max(12, info.get("expires_in", 60) // interval)

    with httpx.Client() as c:
        for attempt in range(max_attempts):
            time.sleep(interval)
            r = c.post(
                GITHUB_ACCESS_TOKEN_URL,
                headers=_gh_headers(),
                json={
                    "client_id": GITHUB_CLIENT_ID,
                    "device_code": info["device_code"],
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            d = r.json()

            if "access_token" in d:
                print("  Authorized successfully.", flush=True)
                print("-" * 60, flush=True)
                print("", flush=True)
                _save_access_token(d["access_token"])
                return d["access_token"]

            error = d.get("error", "")
            if error == "authorization_pending":
                dots = "." * ((attempt % 3) + 1)
                print(f"  Waiting{dots}", end="\r", flush=True)
                continue
            elif error == "slow_down":
                interval += 5
                continue
            elif error in ("expired_token", "access_denied"):
                print(f"\n  Authorization failed: {error}", flush=True)
                break
            else:
                print(f"\n  Unexpected response: {d}", flush=True)
                break

    raise RuntimeError("Device flow failed — could not obtain access token.")


def _refresh_api_key(access_token: str) -> str:
    """Exchange OAuth access token for a short-lived GHCP API key (~30 min TTL)."""
    with httpx.Client() as c:
        r = c.get(GITHUB_API_KEY_URL, headers=_gh_headers(access_token))
        r.raise_for_status()
        data = r.json()
    os.makedirs(TOKEN_DIR, exist_ok=True)
    with open(API_KEY_FILE, "w") as f:
        json.dump(data, f)
    return data["token"]


def get_api_key() -> str:
    """Returns a valid GHCP API key, refreshing transparently when expired."""
    key = load_api_key()
    if key:
        return key
    access_token = load_access_token() or _device_flow()
    return _refresh_api_key(access_token)


def ensure_authenticated():
    """
    Called at startup — before the server accepts any requests.
    Runs the full auth flow interactively in the terminal if needed.
    """
    print("Checking GitHub Copilot authentication...", flush=True)
    try:
        key = get_api_key()
        print("Authenticated. GHCP API key valid.", flush=True)
        return key
    except Exception as e:
        print(f"\nAuthentication failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
