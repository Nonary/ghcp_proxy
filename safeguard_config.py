"""Persistent safeguard configuration (request-finish cooldown seconds)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from fastapi import HTTPException

from constants import (
    SAFEGUARD_CONFIG_FILE,
    SAFEGUARD_DEFAULT_COOLDOWN_SECONDS,
    SAFEGUARD_MAX_COOLDOWN_SECONDS,
    SAFEGUARD_MIN_COOLDOWN_SECONDS,
    TOKEN_DIR,
)


@dataclass(frozen=True)
class SafeguardConfig:
    config_file: str = SAFEGUARD_CONFIG_FILE


class SafeguardConfigService:
    def __init__(self, config: SafeguardConfig):
        self._config = config

    def default_settings(self) -> dict[str, object]:
        return {
            "cooldown_seconds": float(SAFEGUARD_DEFAULT_COOLDOWN_SECONDS),
        }

    def load_settings(self) -> dict[str, object]:
        try:
            with open(self._config.config_file, encoding="utf-8") as f:
                payload = json.load(f)
        except OSError:
            return self.default_settings()
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse {self._config.config_file}: {exc}",
            ) from exc

        if not isinstance(payload, dict):
            return self.default_settings()

        return {
            "cooldown_seconds": self._coerce_cooldown(
                payload.get("cooldown_seconds"),
                default=float(SAFEGUARD_DEFAULT_COOLDOWN_SECONDS),
            ),
        }

    def config_payload(self) -> dict[str, object]:
        current = self.load_settings()
        return {
            **current,
            "default_cooldown_seconds": float(SAFEGUARD_DEFAULT_COOLDOWN_SECONDS),
            "min_cooldown_seconds": float(SAFEGUARD_MIN_COOLDOWN_SECONDS),
            "max_cooldown_seconds": float(SAFEGUARD_MAX_COOLDOWN_SECONDS),
            "path": self._config.config_file,
        }

    def save_settings(self, payload: dict) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object")

        cooldown = self._coerce_cooldown(payload.get("cooldown_seconds"), default=None)
        if cooldown is None:
            raise HTTPException(
                status_code=400,
                detail="cooldown_seconds is required and must be a number.",
            )

        os.makedirs(os.path.dirname(self._config.config_file) or TOKEN_DIR, exist_ok=True)
        with open(self._config.config_file, "w", encoding="utf-8") as f:
            json.dump({"cooldown_seconds": cooldown}, f, indent=2)
            f.write("\n")
        return self.config_payload()

    @staticmethod
    def _coerce_cooldown(value, *, default):
        if value is None:
            return default
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="cooldown_seconds must be a number.",
            ) from exc
        if seconds < SAFEGUARD_MIN_COOLDOWN_SECONDS:
            seconds = SAFEGUARD_MIN_COOLDOWN_SECONDS
        if seconds > SAFEGUARD_MAX_COOLDOWN_SECONDS:
            seconds = SAFEGUARD_MAX_COOLDOWN_SECONDS
        return round(seconds, 3)
