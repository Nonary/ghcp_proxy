"""Explicit client proxy configuration service."""

from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import HTTPException

from constants import CODEX_PROXY_BASE_URL, DASHBOARD_BASE_URL


@dataclass(frozen=True)
class ProxyClientConfig:
    codex_config_file: str
    codex_proxy_config: str
    claude_settings_file: str
    claude_proxy_settings: dict
    claude_max_context_tokens: str
    claude_max_output_tokens: str

    @property
    def codex_config_dir(self) -> str:
        return os.path.dirname(self.codex_config_file)

    @property
    def claude_config_dir(self) -> str:
        return os.path.dirname(self.claude_settings_file)


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


class ProxyClientConfigService:
    def __init__(self, config: ProxyClientConfig):
        self._config = config

    def codex_proxy_status(self) -> dict[str, bool | str | None]:
        status = self.empty_proxy_status("codex")
        status["status_message"] = "config file not found"

        if not os.path.exists(self._config.codex_config_file):
            return status

        status["exists"] = True
        status["status_message"] = "exists but not configured for proxy"

        try:
            with open(self._config.codex_config_file, encoding="utf-8") as f:
                parsed = self._parse_toml_values(f.read())
        except OSError as exc:
            status["error"] = f"failed to read {self._config.codex_config_file}: {exc}"
            return status
        except Exception as exc:
            status["error"] = f"failed to parse {self._config.codex_config_file}: {exc}"
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

    def claude_proxy_status(self) -> dict[str, bool | str | None]:
        status = self.empty_proxy_status("claude")
        status["status_message"] = "settings file not found"

        if not os.path.exists(self._config.claude_settings_file):
            return status

        status["exists"] = True
        status["status_message"] = "exists but not configured for proxy"

        try:
            with open(self._config.claude_settings_file, encoding="utf-8") as f:
                payload = json.load(f)
        except OSError as exc:
            status["error"] = f"failed to read {self._config.claude_settings_file}: {exc}"
            return status
        except json.JSONDecodeError as exc:
            status["error"] = f"failed to parse {self._config.claude_settings_file}: {exc}"
            return status
        except Exception as exc:
            status["error"] = f"failed to parse {self._config.claude_settings_file}: {exc}"
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
            and str(env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS") or "") == self._config.claude_max_context_tokens
        )
        has_output_cap = (
            isinstance(env, dict)
            and str(env.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS") or "") == self._config.claude_max_output_tokens
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

    def write_codex_proxy_config(self) -> dict[str, bool | str | None]:
        status = self.codex_proxy_status()
        if status.get("error"):
            return status
        if status.get("configured"):
            status["backup_path"] = self._latest_backup_path(self._config.codex_config_file)
            status["status_message"] = "proxy already enabled"
            return status

        backup_path = self._backup_config_file(self._config.codex_config_file)
        os.makedirs(self._config.codex_config_dir, exist_ok=True)
        with open(self._config.codex_config_file, "w", encoding="utf-8") as f:
            f.write(self._config.codex_proxy_config)
            f.write("\n")
        status = self.codex_proxy_status()
        status["backup_path"] = backup_path
        status["status_message"] = "installed proxy config"
        return status

    def write_claude_proxy_settings(self) -> dict[str, bool | str | None]:
        status = self.claude_proxy_status()
        if status.get("error"):
            return status
        if status.get("configured"):
            status["backup_path"] = self._latest_backup_path(self._config.claude_settings_file)
            status["status_message"] = "proxy already enabled"
            return status

        existing_payload = {}
        if status.get("exists"):
            with open(self._config.claude_settings_file, encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                existing_payload = payload

        backup_path = self._backup_config_file(self._config.claude_settings_file)
        os.makedirs(self._config.claude_config_dir, exist_ok=True)
        with open(self._config.claude_settings_file, "w", encoding="utf-8") as f:
            json.dump(self._merged_claude_proxy_settings(existing_payload), f, indent=2)
            f.write("\n")
        status = self.claude_proxy_status()
        status["backup_path"] = backup_path
        status["status_message"] = "installed proxy settings"
        return status

    def disable_codex_proxy_config(self) -> dict[str, bool | str | None]:
        return self._disable_client_proxy_config(
            self._config.codex_config_file,
            self.codex_proxy_status,
        )

    def disable_claude_proxy_settings(self) -> dict[str, bool | str | None]:
        return self._disable_client_proxy_config(
            self._config.claude_settings_file,
            self.claude_proxy_status,
        )

    def enable_target(self, target: str) -> dict[str, bool | str | None]:
        if target == "codex":
            return self.write_codex_proxy_config()
        if target == "claude":
            return self.write_claude_proxy_settings()
        raise ValueError(f"Unsupported target: {target}")

    def disable_target(self, target: str) -> dict[str, bool | str | None]:
        if target == "codex":
            return self.disable_codex_proxy_config()
        if target == "claude":
            return self.disable_claude_proxy_settings()
        raise ValueError(f"Unsupported target: {target}")

    def empty_proxy_status(self, target: str) -> dict[str, bool | str | None]:
        if target == "codex":
            return {
                "client": "codex",
                "configured": False,
                "exists": False,
                "path": self._config.codex_config_file,
                "backup_path": None,
                "error": "",
                "status_message": "unknown",
            }
        if target == "claude":
            return {
                "client": "claude",
                "configured": False,
                "exists": False,
                "path": self._config.claude_settings_file,
                "backup_path": None,
                "error": "",
                "status_message": "unknown",
            }
        return {
            "client": target,
            "configured": False,
            "exists": False,
            "path": "",
            "backup_path": None,
            "error": "",
            "status_message": "unknown",
        }

    def proxy_client_status_payload(self) -> dict[str, object]:
        codex_status = self.codex_proxy_status()
        claude_status = self.claude_proxy_status()
        codex_status["backup_path"] = self._latest_backup_path(self._config.codex_config_file)
        claude_status["backup_path"] = self._latest_backup_path(self._config.claude_settings_file)
        codex_status["restored_from_backup"] = False
        claude_status["restored_from_backup"] = False
        return {"clients": {"codex": codex_status, "claude": claude_status}}

    def _disable_client_proxy_config(self, config_path: str, status_fn) -> dict[str, bool | str | None]:
        status = status_fn()
        if not isinstance(status, dict):
            status = self.empty_proxy_status("unknown")
            status["path"] = config_path
        if status.get("error"):
            return status
        if not isinstance(config_path, str) or not config_path:
            status["error"] = "invalid config path"
            return status
        if not status.get("configured"):
            status["backup_path"] = self._latest_backup_path(config_path)
            status["restored_from_backup"] = False
            status["status_message"] = "proxy already disabled"
            return status

        backup_path = self._latest_backup_path(config_path)
        restored_from_backup = False
        operation_message = ""

        try:
            if backup_path:
                shutil.copy2(backup_path, config_path)
                restored_from_backup = True
                operation_message = f"restored config from backup ({backup_path})"
                try:
                    os.remove(backup_path)
                except OSError:
                    operation_message = (
                        f"restored config from backup ({backup_path}); "
                        "backup copy retained"
                    )
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

    def _backup_config_file(self, path: str) -> str | None:
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

    def _latest_backup_path(self, path: str) -> str | None:
        backups = [entry for entry in glob.glob(f"{path}.ghcp-proxy.bak.*") if os.path.isfile(entry)]
        if not backups:
            return None
        return max(backups, key=lambda entry: os.path.getmtime(entry))

    def _parse_toml_values(self, content: str) -> dict:
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

            if value[0] in {"'", '"'} and value[-1] == value[0]:
                value = value[1:-1]

            if current_section is None:
                data[key] = value
            else:
                section = sections.setdefault(current_section, {})
                section[key] = value
                data[current_section] = section

        return data

    def _merged_claude_proxy_settings(self, existing_payload: dict | None) -> dict:
        merged = dict(existing_payload) if isinstance(existing_payload, dict) else {}
        existing_env = merged.get("env")
        merged_env = dict(existing_env) if isinstance(existing_env, dict) else {}
        merged_env.update(self._config.claude_proxy_settings.get("env", {}))
        merged["env"] = merged_env

        for key, value in self._config.claude_proxy_settings.items():
            if key == "env":
                continue
            merged.setdefault(key, value)

        return merged
