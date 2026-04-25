"""GitHub OAuth device flow, token management, and API key handling."""

import json
import os
import sys
import time
from datetime import datetime, timezone
from threading import Lock, Thread

import httpx

from constants import (
    GITHUB_CLIENT_ID, GITHUB_DEVICE_CODE_URL, GITHUB_ACCESS_TOKEN_URL,
    GITHUB_API_KEY_URL,
    TOKEN_DIR, ACCESS_TOKEN_FILE, API_KEY_FILE,
    GITHUB_COPILOT_API_BASE,
)


_AUTH_FLOW_LOCK = Lock()
_AUTH_FLOW_STATE: dict[str, object] = {
    "state": "idle",
    "flow_id": None,
    "started_at": None,
    "expires_at": None,
    "poll_interval_seconds": None,
    "verification_uri": None,
    "verification_uri_complete": None,
    "user_code": None,
    "error": None,
    "warning": "",
    "message": "",
}


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


def clear_api_key_cache() -> bool:
    """Remove the cached short-lived Copilot API key, if present."""
    try:
        os.remove(API_KEY_FILE)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def _utc_timestamp() -> float:
    return datetime.now(timezone.utc).timestamp()


def _iso_timestamp(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _device_flow_info() -> dict:
    with httpx.Client() as c:
        r = c.post(
            GITHUB_DEVICE_CODE_URL,
            headers=_gh_headers(),
            json={"client_id": GITHUB_CLIENT_ID, "scope": "read:user read:org"},
        )
        r.raise_for_status()
        info = r.json()
    if not isinstance(info, dict):
        raise RuntimeError("Device flow failed — invalid GitHub device code response.")
    return info


def _poll_for_access_token(
    device_code: str,
    *,
    interval: int,
    expires_in: int,
    interactive: bool = False,
) -> str:
    if not isinstance(device_code, str) or not device_code:
        raise RuntimeError("Device flow failed — missing device code.")

    max_attempts = max(12, max(1, expires_in) // max(1, interval))

    with httpx.Client() as c:
        for attempt in range(max_attempts):
            time.sleep(interval)
            r = c.post(
                GITHUB_ACCESS_TOKEN_URL,
                headers=_gh_headers(),
                json={
                    "client_id": GITHUB_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            d = r.json()

            if "access_token" in d:
                _save_access_token(d["access_token"])
                return d["access_token"]

            error = d.get("error", "")
            if error == "authorization_pending":
                if interactive:
                    dots = "." * ((attempt % 3) + 1)
                    print(f"  Waiting{dots}", end="\r", flush=True)
                continue
            elif error == "slow_down":
                interval += 5
                continue
            elif error in ("expired_token", "access_denied"):
                if interactive:
                    print(f"\n  Authorization failed: {error}", flush=True)
                break
            else:
                if interactive:
                    print(f"\n  Unexpected response: {d}", flush=True)
                break

    raise RuntimeError("Device flow failed — could not obtain access token.")


def _device_flow() -> str:
    """
    Interactive GitHub OAuth device flow.
    Prints the verification URL and code to the terminal, then polls until
    the user authorizes (or times out after ~60 seconds).
    Returns the GitHub OAuth access token.
    """
    info = _device_flow_info()

    print("", flush=True)
    print("─" * 60, flush=True)
    print("  GitHub Copilot — Authorization Required", flush=True)
    print("─" * 60, flush=True)
    print(f"  1. Open:  {info['verification_uri']}", flush=True)
    print(f"  2. Enter: {info['user_code']}", flush=True)
    print("─" * 60, flush=True)
    print("  Waiting for authorization...", flush=True)

    access_token = _poll_for_access_token(
        info["device_code"],
        interval=int(info.get("interval", 5) or 5),
        expires_in=int(info.get("expires_in", 60) or 60),
        interactive=True,
    )
    print("  Authorized successfully.", flush=True)
    print("-" * 60, flush=True)
    print("", flush=True)
    return access_token


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


def refresh_api_key(*, interactive: bool = False) -> str:
    """Force-refresh the short-lived Copilot API key using the cached OAuth token."""
    access_token = load_access_token()
    if not access_token:
        if not interactive:
            raise RuntimeError("GitHub Copilot authorization required.")
        access_token = _device_flow()
    return _refresh_api_key(access_token)


def _authenticated_snapshot(*, message: str | None = None, warning: str | None = None) -> dict:
    api_key = load_api_key()
    access_token = load_access_token()
    result = {
        "authenticated": bool(api_key or access_token),
        "state": "authenticated",
        "message": message or "GitHub Copilot is authenticated.",
        "error": "",
        "warning": warning if warning is not None else str(_AUTH_FLOW_STATE.get("warning") or ""),
        "has_access_token": bool(access_token),
        "has_api_key": bool(api_key),
        "verification_uri": None,
        "verification_uri_complete": None,
        "user_code": None,
        "started_at": None,
        "expires_at": None,
        "poll_interval_seconds": None,
    }
    return result


def _flow_snapshot_unlocked() -> dict:
    if load_api_key() or load_access_token():
        return _authenticated_snapshot()

    expires_at = _AUTH_FLOW_STATE.get("expires_at")
    state = str(_AUTH_FLOW_STATE.get("state") or "idle")
    error = str(_AUTH_FLOW_STATE.get("error") or "")

    if state in {"starting", "pending"} and isinstance(expires_at, (int, float)) and expires_at <= _utc_timestamp():
        state = "error"
        error = error or "Authorization timed out before completion."
        _AUTH_FLOW_STATE.update(
            {
                "state": state,
                "error": error,
                "warning": "",
                "message": "Start sign-in again to request a new GitHub device code.",
                "verification_uri": None,
                "verification_uri_complete": None,
                "user_code": None,
                "started_at": None,
                "expires_at": None,
                "poll_interval_seconds": None,
                "flow_id": None,
            }
        )

    state = str(_AUTH_FLOW_STATE.get("state") or "idle")
    if state == "starting":
        message = str(_AUTH_FLOW_STATE.get("message") or "Requesting GitHub device code...")
    elif state == "pending":
        message = str(_AUTH_FLOW_STATE.get("message") or "Authorize the device code in GitHub to finish setup.")
    elif state == "error":
        message = str(_AUTH_FLOW_STATE.get("message") or "GitHub sign-in is not complete.")
    else:
        message = "GitHub Copilot is not authenticated yet."

    return {
        "authenticated": False,
        "state": state if state in {"starting", "pending", "error"} else "unauthenticated",
        "message": message,
        "error": str(_AUTH_FLOW_STATE.get("error") or ""),
        "warning": str(_AUTH_FLOW_STATE.get("warning") or ""),
        "has_access_token": False,
        "has_api_key": False,
        "verification_uri": _AUTH_FLOW_STATE.get("verification_uri"),
        "verification_uri_complete": _AUTH_FLOW_STATE.get("verification_uri_complete"),
        "user_code": _AUTH_FLOW_STATE.get("user_code"),
        "started_at": _iso_timestamp(_AUTH_FLOW_STATE.get("started_at")),
        "expires_at": _iso_timestamp(_AUTH_FLOW_STATE.get("expires_at")),
        "poll_interval_seconds": _AUTH_FLOW_STATE.get("poll_interval_seconds"),
    }


def auth_status() -> dict:
    with _AUTH_FLOW_LOCK:
        return _flow_snapshot_unlocked()


def _complete_browser_auth_flow(
    flow_id: str,
    *,
    device_code: str,
    interval: int,
    expires_in: int,
):
    try:
        access_token = _poll_for_access_token(
            device_code,
            interval=interval,
            expires_in=expires_in,
        )
        warning = ""
        try:
            _refresh_api_key(access_token)
        except Exception as exc:
            warning = f"Authorized, but the API key refresh failed: {exc}"

        with _AUTH_FLOW_LOCK:
            if _AUTH_FLOW_STATE.get("flow_id") != flow_id:
                return
            _AUTH_FLOW_STATE.update(
                {
                    "state": "authenticated",
                    "flow_id": None,
                    "started_at": None,
                    "expires_at": None,
                    "poll_interval_seconds": None,
                    "verification_uri": None,
                    "verification_uri_complete": None,
                    "user_code": None,
                    "error": "",
                    "warning": warning,
                    "message": "GitHub authorization completed.",
                }
            )
    except Exception as exc:
        with _AUTH_FLOW_LOCK:
            if _AUTH_FLOW_STATE.get("flow_id") != flow_id:
                return
            _AUTH_FLOW_STATE.update(
                {
                    "state": "error",
                    "flow_id": None,
                    "verification_uri": None,
                    "verification_uri_complete": None,
                    "user_code": None,
                    "started_at": None,
                    "expires_at": None,
                    "poll_interval_seconds": None,
                    "error": str(exc),
                    "warning": "",
                    "message": "GitHub authorization did not complete.",
                }
            )


def begin_device_flow() -> dict:
    with _AUTH_FLOW_LOCK:
        existing = _flow_snapshot_unlocked()
        if existing["authenticated"] or existing["state"] in {"starting", "pending"}:
            return existing
        _AUTH_FLOW_STATE.update(
            {
                "state": "starting",
                "flow_id": None,
                "started_at": _utc_timestamp(),
                "expires_at": None,
                "poll_interval_seconds": None,
                "verification_uri": None,
                "verification_uri_complete": None,
                "user_code": None,
                "error": "",
                "warning": "",
                "message": "Requesting GitHub device code...",
            }
        )

    try:
        info = _device_flow_info()
        flow_id = f"flow-{int(time.time() * 1000)}"
        started_at = _utc_timestamp()
        interval = int(info.get("interval", 5) or 5)
        expires_in = int(info.get("expires_in", 60) or 60)
        expires_at = started_at + max(1, expires_in)
        verification_uri = info.get("verification_uri")
        verification_uri_complete = info.get("verification_uri_complete")
        user_code = info.get("user_code")
        device_code = info.get("device_code")
        if not isinstance(verification_uri, str) or not verification_uri:
            raise RuntimeError("GitHub device flow did not return a verification URL.")
        if not isinstance(user_code, str) or not user_code:
            raise RuntimeError("GitHub device flow did not return a user code.")
        if not isinstance(device_code, str) or not device_code:
            raise RuntimeError("GitHub device flow did not return a device code.")
    except Exception as exc:
        with _AUTH_FLOW_LOCK:
            _AUTH_FLOW_STATE.update(
                {
                    "state": "error",
                    "flow_id": None,
                    "started_at": None,
                    "expires_at": None,
                    "poll_interval_seconds": None,
                    "verification_uri": None,
                    "verification_uri_complete": None,
                    "user_code": None,
                    "error": str(exc),
                    "warning": "",
                    "message": "Unable to start GitHub authorization.",
                }
            )
            return _flow_snapshot_unlocked()

    with _AUTH_FLOW_LOCK:
        _AUTH_FLOW_STATE.update(
            {
                "state": "pending",
                "flow_id": flow_id,
                "started_at": started_at,
                "expires_at": expires_at,
                "poll_interval_seconds": interval,
                "verification_uri": verification_uri,
                "verification_uri_complete": verification_uri_complete if isinstance(verification_uri_complete, str) else None,
                "user_code": user_code,
                "error": "",
                "warning": "",
                "message": "Open GitHub in the browser and enter the code shown here.",
            }
        )

    Thread(
        target=_complete_browser_auth_flow,
        kwargs={
            "flow_id": flow_id,
            "device_code": device_code,
            "interval": interval,
            "expires_in": expires_in,
        },
        daemon=True,
    ).start()

    with _AUTH_FLOW_LOCK:
        return _flow_snapshot_unlocked()


def get_api_key(*, interactive: bool = False, force_refresh: bool = False) -> str:
    """Returns a valid GHCP API key, refreshing transparently when expired."""
    key = None if force_refresh else load_api_key()
    if key:
        return key
    return refresh_api_key(interactive=interactive)


def ensure_authenticated():
    """
    Called at startup — before the server accepts any requests.
    Runs the full auth flow interactively in the terminal if needed.
    """
    print("Checking GitHub Copilot authentication...", flush=True)
    try:
        key = get_api_key(interactive=True)
        print("Authenticated. GHCP API key valid.", flush=True)
        return key
    except Exception as e:
        print(f"\nAuthentication failed: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
