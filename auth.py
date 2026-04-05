"""GitHub OAuth device flow, token management, API key handling, and client configuration management."""

import glob
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import httpx
from fastapi import HTTPException

from constants import (
    GITHUB_CLIENT_ID, GITHUB_DEVICE_CODE_URL, GITHUB_ACCESS_TOKEN_URL,
    GITHUB_API_KEY_URL,
    TOKEN_DIR, ACCESS_TOKEN_FILE, BILLING_TOKEN_FILE, API_KEY_FILE,
    PROXY_BASE_URL, CODEX_PROXY_BASE_URL, DASHBOARD_BASE_URL,
    CODEX_CONFIG_DIR, CODEX_CONFIG_FILE, CLAUDE_CONFIG_DIR, CLAUDE_SETTINGS_FILE,
    CODEX_PROXY_CONFIG, CLAUDE_PROXY_SETTINGS, CLAUDE_MAX_CONTEXT_TOKENS,
    CLAUDE_MAX_OUTPUT_TOKENS,
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


def load_billing_token() -> str | None:
    env_token = os.environ.get("GHCP_GITHUB_BILLING_TOKEN", "").strip()
    if env_token:
        return env_token

    try:
        with open(BILLING_TOKEN_FILE, encoding="utf-8") as f:
            tok = f.read().strip()
        return tok or None
    except OSError:
        return None


def save_billing_token(token: str):
    os.makedirs(TOKEN_DIR, exist_ok=True)
    with open(BILLING_TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(token.strip())


def clear_billing_token():
    try:
        os.remove(BILLING_TOKEN_FILE)
    except OSError:
        pass


def billing_token_status() -> dict[str, bool | str]:
    env_token = os.environ.get("GHCP_GITHUB_BILLING_TOKEN", "").strip()
    if env_token:
        return {"configured": True, "source": "environment", "readonly": True}

    try:
        with open(BILLING_TOKEN_FILE, encoding="utf-8") as f:
            tok = f.read().strip()
    except OSError:
        tok = ""
    return {"configured": bool(tok), "source": "file" if tok else "none", "readonly": False}


def _backup_config_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None

    os.makedirs(os.path.dirname(path), exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.ghcp-proxy.bak.{timestamp}"
    attempt = 1
    while os.path.exists(backup_path):
        attempt += 1
        backup_path = f"{path}.ghcp-proxy.bak.{timestamp}.{attempt}"

    shutil.copy2(path, backup_path)
    return backup_path


def _latest_backup_path(path: str) -> str | None:
    backups = [entry for entry in glob.glob(f"{path}.ghcp-proxy.bak.*") if os.path.isfile(entry)]
    if not backups:
        return None
    return max(backups, key=lambda entry: os.path.getmtime(entry))


def _parse_toml_values(content: str) -> dict:
    current_section: str | None = None
    data: dict[str, object] = {}
    sections: dict[str, dict[str, str]] = {}

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            sections.setdefault(current_section, {})
            continue

        if "=" not in line:
            continue

        key, value = [part.strip() for part in line.split("=", 1)]
        value = value.split("#", 1)[0].strip()
        if not value:
            continue

        if (value[0] in {"'", "\""} and value[-1] == value[0]):
            value = value[1:-1]

        if current_section is None:
            data[key] = value
        else:
            section = sections.setdefault(current_section, {})
            section[key] = value
            data[current_section] = section

    return data


def codex_proxy_status() -> dict[str, bool | str | None]:
    status = {
        "client": "codex",
        "configured": False,
        "exists": False,
        "path": CODEX_CONFIG_FILE,
        "backup_path": None,
        "error": "",
        "status_message": "config file not found",
    }

    if not os.path.exists(CODEX_CONFIG_FILE):
        return status

    status["exists"] = True
    status["status_message"] = "exists but not configured for proxy"

    try:
        with open(CODEX_CONFIG_FILE, encoding="utf-8") as f:
            parsed = _parse_toml_values(f.read())
    except OSError as exc:
        status["error"] = f"failed to read {CODEX_CONFIG_FILE}: {exc}"
        return status
    except Exception as exc:
        status["error"] = f"failed to parse {CODEX_CONFIG_FILE}: {exc}"
        return status

    model_providers = parsed.get("model_providers.custom")
    provider_cfg = model_providers if isinstance(model_providers, dict) else {}
    active = (
        parsed.get("model_provider") == "custom"
        and isinstance(provider_cfg, dict)
        and provider_cfg.get("name") == "OpenAI"
        and provider_cfg.get("base_url") == CODEX_PROXY_BASE_URL
        and provider_cfg.get("wire_api") == "responses"
    )
    status["configured"] = bool(active)
    if active:
        status["status_message"] = "proxy configured"
    return status


def empty_proxy_status(client: str, path: str) -> dict[str, bool | str | None]:
    return {
        "client": client,
        "configured": False,
        "exists": False,
        "path": path,
        "backup_path": None,
        "error": "",
        "status_message": "unknown",
    }


def claude_proxy_status() -> dict[str, bool | str | None]:
    status = {
        "client": "claude",
        "configured": False,
        "exists": False,
        "path": CLAUDE_SETTINGS_FILE,
        "backup_path": None,
        "error": "",
        "status_message": "settings file not found",
    }

    if not os.path.exists(CLAUDE_SETTINGS_FILE):
        return status

    status["exists"] = True
    status["status_message"] = "exists but not configured for proxy"

    try:
        with open(CLAUDE_SETTINGS_FILE, encoding="utf-8") as f:
            payload = json.load(f)
    except OSError as exc:
        status["error"] = f"failed to read {CLAUDE_SETTINGS_FILE}: {exc}"
        return status
    except json.JSONDecodeError as exc:
        status["error"] = f"failed to parse {CLAUDE_SETTINGS_FILE}: {exc}"
        return status
    except Exception as exc:
        status["error"] = f"failed to parse {CLAUDE_SETTINGS_FILE}: {exc}"
        return status

    env = payload.get("env") if isinstance(payload, dict) else None
    has_proxy_env = (
        isinstance(env, dict)
        and env.get("ANTHROPIC_BASE_URL") == DASHBOARD_BASE_URL
        and env.get("CLAUDE_CODE_DISABLE_1M_CONTEXT") == "1"
        and isinstance(env.get("ANTHROPIC_AUTH_TOKEN"), str)
        and env.get("ANTHROPIC_AUTH_TOKEN") != ""
    )
    has_context_cap = (
        isinstance(env, dict)
        and str(env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS") or "") == CLAUDE_MAX_CONTEXT_TOKENS
    )
    has_output_cap = (
        isinstance(env, dict)
        and str(env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS") or "") == CLAUDE_MAX_OUTPUT_TOKENS
    )
    active = has_proxy_env and has_context_cap and has_output_cap
    status["configured"] = bool(active)
    if active:
        status["status_message"] = "proxy configured"
    elif has_proxy_env:
        missing = []
        if not has_context_cap:
            missing.append("context cap")
        if not has_output_cap:
            missing.append("output cap")
        status["status_message"] = f"proxy configured, missing {' and '.join(missing)}"
    return status


def _merged_claude_proxy_settings(existing_payload: dict | None) -> dict:
    merged = dict(existing_payload) if isinstance(existing_payload, dict) else {}
    existing_env = merged.get("env")
    merged_env = dict(existing_env) if isinstance(existing_env, dict) else {}
    merged_env.update(CLAUDE_PROXY_SETTINGS.get("env", {}))
    merged["env"] = merged_env

    for key, value in CLAUDE_PROXY_SETTINGS.items():
        if key == "env":
            continue
        merged.setdefault(key, value)

    return merged


def write_codex_proxy_config() -> dict[str, bool | str | None]:
    status = codex_proxy_status()
    if status.get("error"):
        return status
    if status.get("configured"):
        status["backup_path"] = _latest_backup_path(CODEX_CONFIG_FILE)
        status["status_message"] = "proxy already enabled"
        return status

    backup_path = _backup_config_file(CODEX_CONFIG_FILE)
    os.makedirs(CODEX_CONFIG_DIR, exist_ok=True)
    with open(CODEX_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(CODEX_PROXY_CONFIG)
        f.write("\n")
    status = codex_proxy_status()
    status["backup_path"] = backup_path
    status["status_message"] = "installed proxy config"
    return status


def write_claude_proxy_settings() -> dict[str, bool | str | None]:
    status = claude_proxy_status()
    if status.get("error"):
        return status
    if status.get("configured"):
        status["backup_path"] = _latest_backup_path(CLAUDE_SETTINGS_FILE)
        status["status_message"] = "proxy already enabled"
        return status

    existing_payload = {}
    if status.get("exists"):
        with open(CLAUDE_SETTINGS_FILE, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            existing_payload = payload

    backup_path = _backup_config_file(CLAUDE_SETTINGS_FILE)
    os.makedirs(CLAUDE_CONFIG_DIR, exist_ok=True)
    with open(CLAUDE_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(_merged_claude_proxy_settings(existing_payload), f, indent=2)
        f.write("\n")
    status = claude_proxy_status()
    status["backup_path"] = backup_path
    status["status_message"] = "installed proxy settings"
    return status


def disable_client_proxy_config(config_path: str, status_fn) -> dict[str, bool | str | None]:
    status = status_fn()
    if not isinstance(status, dict):
        status = empty_proxy_status("unknown", config_path)
    if status.get("error"):
        return status
    if not isinstance(config_path, str) or not config_path:
        status["error"] = "invalid config path"
        return status
    if not status.get("configured"):
        status["backup_path"] = _latest_backup_path(config_path)
        status["restored_from_backup"] = False
        status["status_message"] = "proxy already disabled"
        return status

    backup_path = _latest_backup_path(config_path)
    restored_from_backup = False
    operation_message = ""

    try:
        if backup_path:
            os.replace(backup_path, config_path)
            restored_from_backup = True
            operation_message = f"restored config from backup ({backup_path})"
        else:
            if os.path.exists(config_path):
                os.remove(config_path)
                operation_message = "removed proxy config"
            else:
                operation_message = "config file already absent"
    except Exception as exc:
        status["error"] = f"failed to disable proxy config: {exc}"
        return status

    status = status_fn()
    status["backup_path"] = backup_path
    status["restored_from_backup"] = restored_from_backup
    if operation_message:
        status["status_message"] = operation_message
    if status.get("error"):
        return status
    return status


def disable_codex_proxy_config() -> dict[str, bool | str | None]:
    return disable_client_proxy_config(CODEX_CONFIG_FILE, codex_proxy_status)


def disable_claude_proxy_settings() -> dict[str, bool | str | None]:
    return disable_client_proxy_config(CLAUDE_SETTINGS_FILE, claude_proxy_status)


def proxy_client_status_payload() -> dict[str, object]:
    codex_status = codex_proxy_status()
    claude_status = claude_proxy_status()
    codex_status["backup_path"] = _latest_backup_path(CODEX_CONFIG_FILE)
    claude_status["backup_path"] = _latest_backup_path(CLAUDE_SETTINGS_FILE)
    codex_status["restored_from_backup"] = False
    claude_status["restored_from_backup"] = False
    return {"clients": {"codex": codex_status, "claude": claude_status}}


def normalize_proxy_targets(payload: dict) -> list[str]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")

    raw_targets = payload.get("targets")
    if raw_targets is None:
        raw_targets = payload.get("target")
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    elif isinstance(raw_targets, (list, tuple, set)):
        raw_targets = list(raw_targets)
    else:
        raise HTTPException(status_code=400, detail='Request body must include "targets" or "target".')

    selected = set()
    for raw in raw_targets:
        if not isinstance(raw, str):
            raise HTTPException(status_code=400, detail="Each target must be a string.")
        target = raw.strip().lower()
        if target in {"both", "all"}:
            selected.update({"codex", "claude"})
        elif target in {"codex", "claude"}:
            selected.add(target)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported target: {raw}")

    if not selected:
        raise HTTPException(status_code=400, detail="No valid targets provided.")

    return sorted(selected)


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
