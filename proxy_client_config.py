"""Explicit client proxy configuration service."""

from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping

from fastapi import HTTPException

from constants import CODEX_PROXY_BASE_URL, DASHBOARD_BASE_URL, MODEL_PRICING, PREMIUM_REQUEST_MULTIPLIERS


_DEFAULT_CODEX_BASE_INSTRUCTIONS = (
    "You are Codex, a coding agent based on GPT-5. You share the user's workspace and "
    "collaborate to solve software tasks with direct, factual communication."
)
_REASONING_LEVEL_DESCRIPTIONS = {
    "minimal": "Minimal reasoning for the fastest responses",
    "low": "Fast responses with lighter reasoning",
    "medium": "Balances speed and reasoning depth for everyday tasks",
    "high": "Greater reasoning depth for complex problems",
    "xhigh": "Extra high reasoning depth for complex problems",
}
_DEFAULT_REASONING_EFFORTS = ["low", "medium", "high", "xhigh"]
_REASONING_EFFORT_RANK = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}
_DEFAULT_PREFERRED_REASONING = ("medium", "low", "high", "xhigh", "minimal")


def _toml_basic_string(value: str) -> str:
    return json.dumps(value)


@dataclass(frozen=True)
class ProxyClientConfig:
    codex_primary_config_file: str
    codex_managed_config_file: str
    codex_model_catalog_file: str
    codex_proxy_config: str
    codex_model_context_window: int
    codex_model_auto_compact_token_limit: int
    claude_settings_file: str
    claude_proxy_settings: dict
    claude_max_context_tokens: str
    claude_max_output_tokens: str
    client_proxy_settings_file: str

    @property
    def codex_config_dir(self) -> str:
        return os.path.dirname(self.codex_primary_config_file or self.codex_managed_config_file)

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
    def __init__(
        self,
        config: ProxyClientConfig,
        *,
        model_capabilities_provider: Callable[[], Mapping[str, Mapping[str, object]]] | None = None,
        model_routing_settings_provider: Callable[[], Mapping[str, object]] | None = None,
    ):
        self._config = config
        self._model_capabilities_provider = model_capabilities_provider
        self._model_routing_settings_provider = model_routing_settings_provider

    def _model_capabilities(self) -> Mapping[str, Mapping[str, object]]:
        if self._model_capabilities_provider is None:
            return {}
        try:
            data = self._model_capabilities_provider()
        except Exception:
            return {}
        return data if isinstance(data, Mapping) else {}

    def _model_routing_settings(self) -> Mapping[str, object]:
        if self._model_routing_settings_provider is None:
            return {}
        try:
            data = self._model_routing_settings_provider()
        except Exception:
            return {}
        return data if isinstance(data, Mapping) else {}

    def codex_proxy_status(self) -> dict[str, bool | str | None]:
        status = self.empty_proxy_status("codex")
        status["status_message"] = "managed config file not found"
        managed_config_exists = os.path.exists(self._config.codex_managed_config_file)
        catalog_exists = os.path.exists(self._config.codex_model_catalog_file)

        if not managed_config_exists:
            if catalog_exists:
                status["exists"] = True
                status["status_message"] = "model catalog present, managed config file missing"
            return status

        status["exists"] = True
        status["status_message"] = "exists but not configured for proxy"

        try:
            parsed = self._read_codex_managed_config()
        except Exception as exc:
            status["error"] = str(exc)
            return status

        provider_cfg = self._codex_provider_config(parsed)
        config_active = (
            parsed.get("model_provider") == "custom"
            and isinstance(provider_cfg, dict)
            and provider_cfg.get("name") == "OpenAI"
            and provider_cfg.get("base_url") == CODEX_PROXY_BASE_URL
            and provider_cfg.get("wire_api") == "responses"
            and parsed.get("model_catalog_json") == self._config.codex_model_catalog_file
            and parsed.get("model_context_window") == self._config.codex_model_context_window
            and parsed.get("model_auto_compact_token_limit") == self._config.codex_model_auto_compact_token_limit
        )
        catalog_valid = self._codex_model_catalog_is_valid()
        proxy_markers_present = self._codex_managed_config_targets_proxy(parsed)
        active = config_active and catalog_valid
        status["configured"] = bool(active)
        if active:
            status["status_message"] = "proxy configured"
        elif proxy_markers_present and not catalog_valid:
            status["status_message"] = "managed config present, model catalog missing or invalid"
        elif proxy_markers_present:
            status["status_message"] = "managed config present, proxy settings incomplete"
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

    def default_client_proxy_settings(self) -> dict[str, object]:
        return {
            "revert_on_shutdown": True,
            "token_tripwire_enabled": True,
            "trace_prompt_logging_enabled": False,
            "trace_prompt_logging_salt": "",
            "trace_prompt_logging_verifier": None,
            "pending_restore_targets": [],
        }

    def load_client_proxy_settings(self) -> dict[str, object]:
        payload = self._raw_client_proxy_settings()
        defaults = self.default_client_proxy_settings()
        return {
            "revert_on_shutdown": bool(
                payload.get(
                    "revert_on_shutdown",
                    defaults["revert_on_shutdown"],
                )
            ),
            "token_tripwire_enabled": bool(
                payload.get(
                    "token_tripwire_enabled",
                    defaults["token_tripwire_enabled"],
                )
            ),
            "trace_prompt_logging_enabled": bool(
                payload.get(
                    "trace_prompt_logging_enabled",
                    defaults["trace_prompt_logging_enabled"],
                )
            ),
            "trace_prompt_logging_salt": (
                payload.get("trace_prompt_logging_salt")
                if isinstance(payload.get("trace_prompt_logging_salt"), str)
                else ""
            ),
            "trace_prompt_logging_verifier": (
                payload.get("trace_prompt_logging_verifier")
                if isinstance(payload.get("trace_prompt_logging_verifier"), dict)
                else None
            ),
            "pending_restore_targets": self._normalize_restore_targets(
                payload.get("pending_restore_targets"),
            ),
        }

    def client_proxy_settings_payload(self) -> dict[str, object]:
        settings = self.load_client_proxy_settings()
        return {
            "revert_on_shutdown": settings["revert_on_shutdown"],
            "token_tripwire_enabled": settings["token_tripwire_enabled"],
            "trace_prompt_logging_enabled": settings["trace_prompt_logging_enabled"],
            "trace_prompt_logging_configured": bool(
                settings.get("trace_prompt_logging_salt")
                and isinstance(settings.get("trace_prompt_logging_verifier"), dict)
            ),
            "pending_restore_targets": settings["pending_restore_targets"],
            "path": self._config.client_proxy_settings_file,
        }

    def save_client_proxy_settings(self, payload: dict) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object")
        known_keys = {
            "revert_on_shutdown",
            "token_tripwire_enabled",
            "trace_prompt_logging_enabled",
            "trace_prompt_logging_salt",
            "trace_prompt_logging_verifier",
        }
        if not any(key in payload for key in known_keys):
            raise HTTPException(
                status_code=400,
                detail="At least one recognized setting is required.",
            )
        if "revert_on_shutdown" in payload and not isinstance(payload.get("revert_on_shutdown"), bool):
            raise HTTPException(status_code=400, detail="revert_on_shutdown must be true or false.")
        if "token_tripwire_enabled" in payload and not isinstance(payload.get("token_tripwire_enabled"), bool):
            raise HTTPException(status_code=400, detail="token_tripwire_enabled must be true or false.")
        if "trace_prompt_logging_enabled" in payload and not isinstance(payload.get("trace_prompt_logging_enabled"), bool):
            raise HTTPException(status_code=400, detail="trace_prompt_logging_enabled must be true or false.")
        if "trace_prompt_logging_salt" in payload and not isinstance(payload.get("trace_prompt_logging_salt"), str):
            raise HTTPException(status_code=400, detail="trace_prompt_logging_salt must be a string.")
        if (
            "trace_prompt_logging_verifier" in payload
            and payload.get("trace_prompt_logging_verifier") is not None
            and not isinstance(payload.get("trace_prompt_logging_verifier"), dict)
        ):
            raise HTTPException(status_code=400, detail="trace_prompt_logging_verifier must be an object.")

        existing = self.load_client_proxy_settings()
        settings = {
            "revert_on_shutdown": payload.get("revert_on_shutdown", existing.get("revert_on_shutdown", True)),
            "token_tripwire_enabled": payload.get(
                "token_tripwire_enabled",
                existing.get("token_tripwire_enabled", True),
            ),
            "trace_prompt_logging_enabled": payload.get(
                "trace_prompt_logging_enabled",
                existing.get("trace_prompt_logging_enabled", False),
            ),
            "trace_prompt_logging_salt": payload.get(
                "trace_prompt_logging_salt",
                existing.get("trace_prompt_logging_salt", ""),
            ),
            "trace_prompt_logging_verifier": payload.get(
                "trace_prompt_logging_verifier",
                existing.get("trace_prompt_logging_verifier"),
            ),
            "pending_restore_targets": existing.get("pending_restore_targets", []),
        }
        self._write_client_proxy_settings(settings)
        return self.client_proxy_settings_payload()

    def write_codex_proxy_config(self) -> dict[str, bool | str | None]:
        status = self.codex_proxy_status()
        if status.get("error"):
            return status
        if status.get("configured"):
            status["backup_path"] = self._latest_backup_path(self._config.codex_managed_config_file)
            status["status_message"] = "proxy already enabled"
            return status

        existing_primary_config = ""
        primary_config_exists = os.path.exists(self._config.codex_primary_config_file)
        if primary_config_exists:
            try:
                with open(self._config.codex_primary_config_file, encoding="utf-8") as f:
                    existing_primary_config = f.read()
            except OSError as exc:
                status["error"] = f"failed to read {self._config.codex_primary_config_file}: {exc}"
                return status

        backup_path = self._backup_config_file(self._config.codex_managed_config_file)
        primary_backup_path = self._backup_config_file(self._config.codex_primary_config_file)
        os.makedirs(self._config.codex_config_dir, exist_ok=True)
        self._write_json_atomic(
            self._config.codex_model_catalog_file,
            self._build_codex_model_catalog_payload(),
        )
        if primary_config_exists:
            self._write_text_atomic(
                self._config.codex_primary_config_file,
                self._merged_codex_primary_config(existing_primary_config),
            )
        self._write_text_atomic(
            self._config.codex_managed_config_file,
            self._render_codex_proxy_config(),
        )
        status = self.codex_proxy_status()
        status["backup_path"] = primary_backup_path or backup_path
        status["status_message"] = "installed proxy config"
        return status

    def refresh_codex_model_catalog(self) -> bool:
        if not os.path.exists(self._config.codex_model_catalog_file):
            return False
        os.makedirs(self._config.codex_config_dir, exist_ok=True)
        self._write_json_atomic(
            self._config.codex_model_catalog_file,
            self._build_codex_model_catalog_payload(),
        )
        return True

    def refresh_claude_proxy_settings(self) -> bool:
        if not os.path.exists(self._config.claude_settings_file):
            return False
        try:
            with open(self._config.claude_settings_file, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        existing_payload = payload if isinstance(payload, dict) else {}
        os.makedirs(self._config.claude_config_dir, exist_ok=True)
        self._write_json_atomic(
            self._config.claude_settings_file,
            self._merged_claude_proxy_settings(existing_payload),
        )
        return True

    def refresh_client_model_metadata(self) -> dict[str, bool]:
        return {
            "codex": self.refresh_codex_model_catalog(),
            "claude": self.refresh_claude_proxy_settings(),
        }

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
        self._write_json_atomic(
            self._config.claude_settings_file,
            self._merged_claude_proxy_settings(existing_payload),
        )
        status = self.claude_proxy_status()
        status["backup_path"] = backup_path
        status["status_message"] = "installed proxy settings"
        return status

    def disable_codex_proxy_config(self) -> dict[str, bool | str | None]:
        status = self.codex_proxy_status()
        if status.get("error"):
            return status
        managed_config_exists = os.path.exists(self._config.codex_managed_config_file)
        catalog_exists = os.path.exists(self._config.codex_model_catalog_file)
        backup_path = self._latest_backup_path(self._config.codex_managed_config_file)
        primary_backup_path = self._latest_backup_path(self._config.codex_primary_config_file)
        managed_targets_proxy = False
        if managed_config_exists:
            try:
                managed_targets_proxy = self._codex_managed_config_targets_proxy(self._read_codex_managed_config())
            except Exception as exc:
                status["error"] = str(exc)
                return status

        if not managed_targets_proxy and not catalog_exists:
            status["backup_path"] = backup_path
            status["restored_from_backup"] = False
            status["status_message"] = "proxy already disabled"
            return status

        restored_from_backup = False
        operation_message = "removed proxy-managed Codex files"
        try:
            if primary_backup_path:
                shutil.copy2(primary_backup_path, self._config.codex_primary_config_file)
                try:
                    os.remove(primary_backup_path)
                except OSError:
                    pass
            if managed_targets_proxy and backup_path:
                shutil.copy2(backup_path, self._config.codex_managed_config_file)
                restored_from_backup = True
                operation_message = f"restored managed config from backup ({backup_path})"
                try:
                    os.remove(backup_path)
                except OSError:
                    operation_message = (
                        f"restored managed config from backup ({backup_path}); "
                        "backup copy retained"
                    )
            elif managed_targets_proxy:
                self._remove_file_if_exists(self._config.codex_managed_config_file)
            self._remove_file_if_exists(self._config.codex_model_catalog_file)
        except Exception as exc:
            status["error"] = f"failed to disable proxy config: {exc}"
            return status

        status = self.codex_proxy_status()
        status["backup_path"] = primary_backup_path or backup_path
        status["restored_from_backup"] = restored_from_backup
        status["status_message"] = operation_message
        return status

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
                "path": self._config.codex_managed_config_file,
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
        codex_status["backup_path"] = self._latest_backup_path(self._config.codex_managed_config_file)
        claude_status["backup_path"] = self._latest_backup_path(self._config.claude_settings_file)
        codex_status["restored_from_backup"] = False
        claude_status["restored_from_backup"] = False
        return {
            "clients": {"codex": codex_status, "claude": claude_status},
            "settings": self.client_proxy_settings_payload(),
        }

    def revert_proxy_configs_on_shutdown(self) -> dict[str, object]:
        settings = self.load_client_proxy_settings()
        if not bool(settings.get("revert_on_shutdown")):
            self._write_client_proxy_settings(
                {
                    **settings,
                    "pending_restore_targets": [],
                }
            )
            return {
                "attempted": False,
                "reverted": False,
                "reason": "disabled",
                "clients": {},
            }

        pending_targets = self._configured_proxy_targets()
        self._write_client_proxy_settings(
            {
                **settings,
                "pending_restore_targets": pending_targets,
            }
        )
        clients = {
            "codex": self.disable_codex_proxy_config(),
            "claude": self.disable_claude_proxy_settings(),
        }
        reverted = any(
            bool(client.get("restored_from_backup"))
            or client.get("status_message") in {"removed proxy config", "removed proxy-managed Codex files"}
            for client in clients.values()
            if isinstance(client, dict)
        )
        return {
            "attempted": True,
            "reverted": reverted,
            "reason": "shutdown",
            "clients": clients,
        }

    def restore_proxy_configs_on_startup(self) -> dict[str, object]:
        settings = self.load_client_proxy_settings()
        pending_targets = self._normalize_restore_targets(settings.get("pending_restore_targets"))
        if not pending_targets:
            return {
                "attempted": False,
                "restored": False,
                "reason": "no-pending-targets",
                "clients": {},
            }

        clients = {}
        failed_targets: list[str] = []
        for target in pending_targets:
            result = self.enable_target(target)
            clients[target] = result
            if not bool(result.get("configured")):
                failed_targets.append(target)

        self._write_client_proxy_settings(
            {
                **settings,
                "pending_restore_targets": failed_targets,
            }
        )
        restored = any(bool(result.get("configured")) for result in clients.values() if isinstance(result, dict))
        return {
            "attempted": True,
            "restored": restored,
            "reason": "startup",
            "clients": clients,
        }

    def _configured_proxy_targets(self) -> list[str]:
        targets = []
        if bool(self.codex_proxy_status().get("configured")):
            targets.append("codex")
        if bool(self.claude_proxy_status().get("configured")):
            targets.append("claude")
        return targets

    def _raw_client_proxy_settings(self) -> dict[str, object]:
        try:
            with open(self._config.client_proxy_settings_file, encoding="utf-8") as f:
                payload = json.load(f)
        except OSError:
            payload = {}
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse {self._config.client_proxy_settings_file}: {exc}",
            ) from exc

        return payload if isinstance(payload, dict) else {}

    def _write_client_proxy_settings(self, payload: dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self._config.client_proxy_settings_file) or ".", exist_ok=True)
        normalized = {
            "revert_on_shutdown": bool(payload.get("revert_on_shutdown", True)),
            "token_tripwire_enabled": bool(payload.get("token_tripwire_enabled", True)),
            "trace_prompt_logging_enabled": bool(payload.get("trace_prompt_logging_enabled", False)),
            "trace_prompt_logging_salt": (
                payload.get("trace_prompt_logging_salt")
                if isinstance(payload.get("trace_prompt_logging_salt"), str)
                else ""
            ),
            "trace_prompt_logging_verifier": (
                payload.get("trace_prompt_logging_verifier")
                if isinstance(payload.get("trace_prompt_logging_verifier"), dict)
                else None
            ),
            "pending_restore_targets": self._normalize_restore_targets(payload.get("pending_restore_targets")),
        }
        self._write_json_atomic(self._config.client_proxy_settings_file, normalized)

    def _normalize_restore_targets(self, raw_targets: object) -> list[str]:
        if not isinstance(raw_targets, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for entry in raw_targets:
            if not isinstance(entry, str):
                continue
            target = entry.strip().lower()
            if target not in {"codex", "claude"} or target in seen:
                continue
            seen.add(target)
            normalized.append(target)
        return normalized

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
        parsed = tomllib.loads(content)
        return parsed if isinstance(parsed, dict) else {}

    def _merged_claude_proxy_settings(self, existing_payload: dict | None) -> dict:
        merged = dict(existing_payload) if isinstance(existing_payload, dict) else {}
        existing_env = merged.get("env")
        merged_env = dict(existing_env) if isinstance(existing_env, dict) else {}
        merged_env.update(self._config.claude_proxy_settings.get("env", {}))
        merged_env.update(self._build_claude_code_default_env())
        merged["env"] = merged_env

        for key, value in self._config.claude_proxy_settings.items():
            if key == "env":
                continue
            merged.setdefault(key, value)

        auto_compact_window = self._resolve_claude_auto_compact_window()
        if auto_compact_window is not None:
            merged["autoCompactWindow"] = auto_compact_window

        return merged

    def _read_codex_managed_config(self) -> dict:
        try:
            with open(self._config.codex_managed_config_file, encoding="utf-8") as f:
                return self._parse_toml_values(f.read())
        except OSError as exc:
            raise RuntimeError(f"failed to read {self._config.codex_managed_config_file}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"failed to parse {self._config.codex_managed_config_file}: {exc}") from exc

    def _codex_provider_config(self, parsed: dict) -> dict[str, str]:
        model_providers = parsed.get("model_providers")
        if isinstance(model_providers, dict):
            custom_provider = model_providers.get("custom")
            if isinstance(custom_provider, dict):
                return custom_provider

        legacy_provider = parsed.get("model_providers.custom")
        return legacy_provider if isinstance(legacy_provider, dict) else {}

    def _codex_managed_config_targets_proxy(self, parsed: dict) -> bool:
        provider_cfg = self._codex_provider_config(parsed)
        return (
            provider_cfg.get("base_url") == CODEX_PROXY_BASE_URL
            or parsed.get("model_catalog_json") == self._config.codex_model_catalog_file
        )

    def _codex_model_catalog_is_valid(self) -> bool:
        try:
            with open(self._config.codex_model_catalog_file, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and isinstance(payload.get("models"), list) and bool(payload["models"])

    def _render_codex_proxy_config(self) -> str:
        top_level_lines: list[str] = []
        section_lines: list[str] = []
        target = top_level_lines
        for line in self._config.codex_proxy_config.strip().splitlines():
            if line.strip().startswith("[") and line.strip().endswith("]"):
                target = section_lines
            target.append(line.rstrip())
        top_level_lines.extend(
            [
                f"model_catalog_json = {_toml_basic_string(self._config.codex_model_catalog_file)}",
                f"model_context_window = {self._config.codex_model_context_window}",
                f"model_auto_compact_token_limit = {self._config.codex_model_auto_compact_token_limit}",
            ]
        )
        lines = [*top_level_lines, ""]
        lines.extend(section_lines)
        return "\n".join(line for line in lines if line is not None).rstrip() + "\n"

    def _merged_codex_primary_config(self, existing_content: str) -> str:
        top_level_keys = {
            "model_provider": _toml_basic_string("custom"),
            "approvals_reviewer": _toml_basic_string("user"),
            "model_catalog_json": _toml_basic_string(self._config.codex_model_catalog_file),
        }
        legacy_keys_to_remove = {
            "model_context_window",
            "model_auto_compact_token_limit",
        }
        provider_keys = {
            "name": _toml_basic_string("OpenAI"),
            "base_url": _toml_basic_string(CODEX_PROXY_BASE_URL),
            "wire_api": _toml_basic_string("responses"),
        }
        provider_section_name = "model_providers.custom"
        lines = existing_content.splitlines()
        preamble: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        current_section_name: str | None = None
        current_lines: list[str] = []

        def flush_current():
            nonlocal current_section_name, current_lines
            if current_section_name is not None:
                sections.append((current_section_name, current_lines))
            current_section_name = None
            current_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                flush_current()
                current_section_name = stripped[1:-1].strip()
                current_lines = [line]
                continue
            if current_section_name is None:
                preamble.append(line)
            else:
                current_lines.append(line)
        flush_current()

        filtered_preamble: list[str] = []
        for line in preamble:
            key = self._toml_assignment_key(line)
            if key in top_level_keys or key in legacy_keys_to_remove:
                continue
            filtered_preamble.append(line)

        if filtered_preamble and filtered_preamble[-1].strip():
            filtered_preamble.append("")
        filtered_preamble.extend(
            [
                f"{key} = {value}"
                for key, value in top_level_keys.items()
            ]
        )

        rendered_sections: list[str] = []
        provider_section_found = False
        for section_name, section_lines in sections:
            if section_name != provider_section_name:
                rendered_sections.extend(section_lines)
                continue

            provider_section_found = True
            provider_body: list[str] = [section_lines[0]]
            for line in section_lines[1:]:
                key = self._toml_assignment_key(line)
                if key in provider_keys or key in top_level_keys:
                    continue
                provider_body.append(line)
            if provider_body and provider_body[-1].strip():
                provider_body.append("")
            provider_body.extend(
                [
                    f"{key} = {value}"
                    for key, value in provider_keys.items()
                ]
            )
            rendered_sections.extend(provider_body)

        if not provider_section_found:
            if filtered_preamble and filtered_preamble[-1].strip():
                filtered_preamble.append("")
            rendered_sections.extend(
                [
                    f"[{provider_section_name}]",
                    *(f"{key} = {value}" for key, value in provider_keys.items()),
                ]
            )

        output_lines = filtered_preamble[:]
        if rendered_sections:
            if output_lines and output_lines[-1].strip():
                output_lines.append("")
            output_lines.extend(rendered_sections)
        return "\n".join(output_lines).rstrip() + "\n"

    def _toml_assignment_key(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            return None
        key = stripped.split("=", 1)[0].strip()
        return key if key and " " not in key else None

    def _build_codex_model_catalog_payload(self) -> dict[str, object]:
        capabilities = self._model_capabilities()
        routing_settings = self._model_routing_settings()
        remapped_targets = self._catalog_remap_targets(routing_settings)
        default_context = self._config.codex_model_context_window
        default_compact = self._config.codex_model_auto_compact_token_limit
        available_ids = (
            set(capabilities.keys()) if isinstance(capabilities, Mapping) else set()
        )
        models = []
        for priority, model_name in enumerate(
            self._sorted_catalog_model_names(available_ids, remapped_targets)
        ):
            routed_model_name = remapped_targets.get(model_name, model_name)
            provider = str(MODEL_PRICING.get(routed_model_name, {}).get("provider") or "Unknown")
            family = self._model_family(routed_model_name)
            caps = capabilities.get(routed_model_name) if isinstance(capabilities, Mapping) else None
            caps = caps if isinstance(caps, Mapping) else {}

            context_window = self._coerce_int(caps.get("context_window"), default_context)
            max_context_window = self._coerce_int(caps.get("max_context_window"), context_window)
            auto_compact = self._resolve_auto_compact_limit(
                caps.get("auto_compact_token_limit"),
                context_window,
                default_compact,
            )
            input_modalities = self._resolve_input_modalities(caps.get("input_modalities"), caps.get("vision"))
            supported_levels, default_level = self._resolve_reasoning_levels(family, caps.get("reasoning_efforts"))
            supports_parallel_tool_calls = self._resolve_bool(
                caps.get("parallel_tool_calls"),
                default=family == "gpt",
            )
            # The proxy's chat→responses translator relays reasoning summary
            # deltas for every upstream provider it bridges (Copilot's
            # `delta.reasoning_text`, Anthropic-style `delta.thinking`,
            # OpenAI-style `delta.reasoning_content`, etc.) — see
            # `extract_reasoning_from_chat_delta` in `format_translation.py`.
            # If a model advertises any reasoning levels at all, treat that as
            # sufficient evidence that summaries can be surfaced, so Codex
            # both (a) builds `reasoning`-aware requests for the model and
            # (b) renders the incoming summary deltas in the TUI.
            supports_reasoning_summaries = self._resolve_bool(
                caps.get("supports_reasoning_summaries"),
                default=bool(supported_levels),
            )
            supports_verbosity = family == "gpt"
            multiplier = PREMIUM_REQUEST_MULTIPLIERS.get(routed_model_name, 1.0)
            if multiplier == 1.0:
                premium_text = "1 premium request"
            else:
                multiplier_str = (
                    f"{multiplier:.2f}".rstrip("0").rstrip(".")
                )
                premium_text = f"{multiplier_str} premium requests"
            if routed_model_name == model_name:
                description = (
                    f"{provider} \u00b7 {context_window:,} token context \u00b7 {premium_text}."
                )
            else:
                description = (
                    f"Routed to {routed_model_name} ({provider}) \u00b7 "
                    f"{context_window:,} token context \u00b7 {premium_text}."
                )

            entry: dict[str, object] = {
                "slug": model_name,
                "display_name": model_name,
                "description": description,
                "shell_type": "shell_command",
                "visibility": "list",
                "supported_in_api": True,
                "priority": priority,
                "additional_speed_tiers": [],
                "availability_nux": None,
                "upgrade": None,
                "base_instructions": _DEFAULT_CODEX_BASE_INSTRUCTIONS,
                "model_messages": None,
                "supports_reasoning_summaries": supports_reasoning_summaries,
                "default_reasoning_summary": "auto" if supports_reasoning_summaries else "none",
                "support_verbosity": supports_verbosity,
                "default_verbosity": "low" if supports_verbosity else None,
                "apply_patch_tool_type": "freeform",
                "web_search_tool_type": "text",
                "truncation_policy": {"mode": "bytes", "limit": 10000},
                "supports_parallel_tool_calls": supports_parallel_tool_calls,
                "supports_image_detail_original": False,
                "context_window": context_window,
                "max_context_window": max_context_window,
                "auto_compact_token_limit": auto_compact,
                "effective_context_window_percent": 95,
                "experimental_supported_tools": [],
                "input_modalities": input_modalities,
                "supports_search_tool": False,
            }
            if supported_levels:
                entry["default_reasoning_level"] = default_level
                entry["supported_reasoning_levels"] = supported_levels
            else:
                entry["default_reasoning_level"] = "medium"
                entry["supported_reasoning_levels"] = []
            models.append(entry)
        return {"models": models}

    def _write_json_atomic(self, path: str, payload: object) -> None:
        content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        self._write_text_atomic(path, content)

    def _write_text_atomic(self, path: str, content: str) -> None:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=directory,
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _model_family(self, model_name: str) -> str:
        for prefix in ("gpt", "claude", "gemini", "grok"):
            if model_name.startswith(f"{prefix}-"):
                return prefix
        return "other"

    def _coerce_int(self, value: object, default: int) -> int:
        try:
            if value is None:
                return default
            ivalue = int(value)
            return ivalue if ivalue > 0 else default
        except (TypeError, ValueError):
            return default

    def _resolve_auto_compact_limit(self, raw_value: object, context_window: int, default_compact: int) -> int:
        explicit = self._coerce_int(raw_value, 0)
        # Leave at least a small request-building buffer even when an explicit
        # provider value is present. For the 272k Codex proxy window this keeps
        # the hard ceiling at 264k, while the default below lands at 240k.
        ceiling = max(context_window - 8000, context_window // 2)
        if explicit > 0:
            return min(explicit, ceiling)
        # Auto-compact should be close to the advertised Codex model window, not
        # the old ~100k/120k-era default. Target ~88% of the model context and
        # use the global default as a floor only after clamping to the model's
        # safe ceiling so smaller-window models are never over-advertised.
        scaled = int(context_window * 0.88)
        return min(max(default_compact, scaled), ceiling)

    def _resolve_input_modalities(self, modalities: object, vision_flag: object) -> list[str]:
        if isinstance(modalities, (list, tuple)):
            cleaned = [str(item) for item in modalities if isinstance(item, str)]
            if cleaned:
                return cleaned
        if isinstance(vision_flag, bool):
            return ["text", "image"] if vision_flag else ["text"]
        return ["text", "image"]

    def _resolve_bool(self, value: object, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        return default

    def _resolve_reasoning_levels(
        self,
        family: str,
        raw_efforts: object,
    ) -> tuple[list[dict[str, str]], str | None]:
        efforts: list[str] = []
        if isinstance(raw_efforts, (list, tuple)):
            for item in raw_efforts:
                if isinstance(item, str) and item:
                    normalized = item.strip().lower()
                    if normalized in _REASONING_EFFORT_RANK and normalized not in efforts:
                        efforts.append(normalized)
        if not efforts:
            if family == "gpt":
                efforts = list(_DEFAULT_REASONING_EFFORTS)
            else:
                # Without an upstream signal, only assume reasoning support for GPT.
                return ([], None)
        efforts.sort(key=lambda effort: _REASONING_EFFORT_RANK.get(effort, 99))
        levels = [
            {
                "effort": effort,
                "description": _REASONING_LEVEL_DESCRIPTIONS.get(effort, effort),
            }
            for effort in efforts
        ]
        default_level = next((effort for effort in _DEFAULT_PREFERRED_REASONING if effort in efforts), efforts[0])
        return (levels, default_level)

    def _sorted_catalog_model_names(
        self,
        available_ids: "set[str] | None" = None,
        remapped_targets: Mapping[str, str] | None = None,
    ) -> list[str]:
        family_order = {"gpt": 0, "claude": 1, "gemini": 2, "grok": 3}
        preferred_order = {
            "gpt-5.4": -20,
            "gpt-5.3-codex": -19,
            "gpt-5.4-mini": -18,
        }

        def family_key(model_name: str) -> int:
            for prefix, order in family_order.items():
                if model_name.startswith(f"{prefix}-"):
                    return order
            return 99

        model_names = [
            model_name
            for model_name in MODEL_PRICING
            if model_name.startswith(("gpt-", "claude-", "gemini-", "grok-"))
        ]
        # Filter by what the upstream Copilot plan actually exposes via /models.
        # If the capability fetch returned nothing (auth/network blip), fall
        # back to the full pricing list rather than wiping the catalog.
        if available_ids:
            filtered = [
                name
                for name in model_names
                if name in available_ids
                or (
                    isinstance(remapped_targets, Mapping)
                    and remapped_targets.get(name) in available_ids
                )
            ]
            if filtered:
                model_names = filtered
        return sorted(model_names, key=lambda model_name: (family_key(model_name), preferred_order.get(model_name, 0), model_name))

    def _catalog_remap_targets(self, routing_settings: Mapping[str, object]) -> dict[str, str]:
        if not bool(routing_settings.get("enabled")):
            return {}
        raw_mappings = routing_settings.get("mappings")
        if not isinstance(raw_mappings, list):
            return {}

        remapped_targets: dict[str, str] = {}
        for entry in raw_mappings:
            if not isinstance(entry, Mapping):
                continue
            source_model = entry.get("source_model")
            target_model = entry.get("target_model")
            if (
                isinstance(source_model, str)
                and source_model
                and isinstance(target_model, str)
                and target_model
                and target_model in MODEL_PRICING
            ):
                remapped_targets[source_model] = target_model
        return remapped_targets

    def _build_claude_code_default_env(self) -> dict[str, str]:
        routing_settings = self._model_routing_settings()
        capabilities = self._model_capabilities()
        claude_defaults = routing_settings.get("claude_code_defaults")
        if not isinstance(claude_defaults, Mapping):
            return {}

        env_updates: dict[str, str] = {}
        slot_specs = (
            ("opus_model", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME", "ANTHROPIC_DEFAULT_OPUS_MODEL_DESCRIPTION"),
            ("sonnet_model", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME", "ANTHROPIC_DEFAULT_SONNET_MODEL_DESCRIPTION"),
            ("haiku_model", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME", "ANTHROPIC_DEFAULT_HAIKU_MODEL_DESCRIPTION"),
        )
        for slot_key, model_key, name_key, description_key in slot_specs:
            configured_model = claude_defaults.get(slot_key)
            if not isinstance(configured_model, str) or not configured_model:
                continue
            env_updates[model_key] = configured_model
            env_updates[name_key] = configured_model
            env_updates[description_key] = self._build_claude_code_model_description(
                configured_model,
                capabilities=capabilities,
                routing_settings=routing_settings,
            )
        return env_updates

    def _resolve_claude_auto_compact_window(self) -> int | None:
        routing_settings = self._model_routing_settings()
        claude_defaults = routing_settings.get("claude_code_defaults")
        if not isinstance(claude_defaults, Mapping):
            return None
        sonnet_model = claude_defaults.get("sonnet_model")
        if not isinstance(sonnet_model, str) or not sonnet_model:
            return None

        capabilities = self._model_capabilities()
        routed_target = self._resolved_catalog_target_name(sonnet_model, routing_settings)
        caps = capabilities.get(routed_target) if isinstance(capabilities, Mapping) else None
        caps = caps if isinstance(caps, Mapping) else {}
        context_window = self._coerce_int(caps.get("context_window"), self._config.codex_model_context_window)
        return self._resolve_auto_compact_limit(
            caps.get("auto_compact_token_limit"),
            context_window,
            self._config.codex_model_auto_compact_token_limit,
        )

    def _build_claude_code_model_description(
        self,
        model_name: str,
        *,
        capabilities: Mapping[str, Mapping[str, object]] | None = None,
        routing_settings: Mapping[str, object] | None = None,
    ) -> str:
        capabilities = capabilities if isinstance(capabilities, Mapping) else self._model_capabilities()
        routing_settings = routing_settings if isinstance(routing_settings, Mapping) else self._model_routing_settings()
        routed_model_name = self._resolved_catalog_target_name(model_name, routing_settings)
        provider = str(MODEL_PRICING.get(routed_model_name, {}).get("provider") or "Unknown")
        caps = capabilities.get(routed_model_name) if isinstance(capabilities, Mapping) else None
        caps = caps if isinstance(caps, Mapping) else {}
        context_window = self._coerce_int(caps.get("context_window"), self._config.codex_model_context_window)
        multiplier = PREMIUM_REQUEST_MULTIPLIERS.get(routed_model_name, 1.0)
        if multiplier == 1.0:
            premium_text = "1 premium request"
        else:
            multiplier_str = f"{multiplier:.2f}".rstrip("0").rstrip(".")
            premium_text = f"{multiplier_str} premium requests"
        if routed_model_name == model_name:
            return f"{provider} · {context_window:,} token context · {premium_text}."
        return f"Routed to {routed_model_name} ({provider}) · {context_window:,} token context · {premium_text}."

    def _resolved_catalog_target_name(
        self,
        model_name: str,
        routing_settings: Mapping[str, object] | None = None,
    ) -> str:
        remapped_targets = self._catalog_remap_targets(routing_settings or self._model_routing_settings())
        return remapped_targets.get(model_name, model_name)

    def _remove_file_if_exists(self, path: str):
        if os.path.exists(path):
            os.remove(path)
