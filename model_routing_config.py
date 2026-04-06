"""Persistent configuration for model remapping."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from fastapi import HTTPException

import format_translation
from constants import MODEL_PRICING, MODEL_ROUTING_CONFIG_FILE, TOKEN_DIR
from util import _normalize_model_name


def normalize_routing_model_name(model_name) -> str | None:
    if not isinstance(model_name, str):
        return None

    raw = model_name.strip()
    if not raw:
        return None

    normalized = "-".join(raw.lower().replace("_", "-").split())
    resolved = format_translation.resolve_copilot_model_name(normalized)
    return _normalize_model_name(resolved or normalized)


def model_provider_family(model_name: str | None) -> str | None:
    normalized = normalize_routing_model_name(model_name)
    if not normalized:
        return None
    if normalized.startswith("claude-"):
        return "claude"
    if normalized.startswith("gpt-"):
        return "codex"
    return None


def _available_model_payloads() -> list[dict[str, str]]:
    rows = []
    for model_name in sorted(MODEL_PRICING):
        provider = model_provider_family(model_name)
        if provider is None:
            continue
        rows.append(
            {
                "model": model_name,
                "provider": provider,
            }
        )
    return rows


@dataclass(frozen=True)
class ModelRoutingConfig:
    config_file: str = MODEL_ROUTING_CONFIG_FILE


class ModelRoutingConfigService:
    def __init__(self, config: ModelRoutingConfig):
        self._config = config
        self._available_models = _available_model_payloads()
        self._known_models = {row["model"] for row in self._available_models}

    def config_payload(self) -> dict[str, object]:
        current = self.load_settings()
        return {
            "enabled": current["enabled"],
            "mappings": current["mappings"],
            "available_models": self._available_models,
            "path": self._config.config_file,
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
            raise HTTPException(
                status_code=500,
                detail=f"Invalid model remapping settings in {self._config.config_file}",
            )

        return self._normalize_settings_payload(payload)

    def save_settings(self, payload: dict) -> dict[str, object]:
        normalized = self._normalize_settings_payload(payload)
        os.makedirs(os.path.dirname(self._config.config_file) or TOKEN_DIR, exist_ok=True)
        with open(self._config.config_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "enabled": normalized["enabled"],
                    "mappings": normalized["mappings"],
                },
                f,
                indent=2,
            )
            f.write("\n")
        return self.config_payload()

    def resolve_target_model(self, requested_model: str | None) -> str | None:
        normalized_requested = normalize_routing_model_name(requested_model)
        if not normalized_requested:
            return None

        settings = self.load_settings()
        if not settings["enabled"]:
            return None

        for mapping in settings["mappings"]:
            if mapping["source_model"] == normalized_requested:
                return mapping["target_model"]
        return None

    def default_settings(self) -> dict[str, object]:
        return {
            "enabled": False,
            "mappings": [],
        }

    def _normalize_settings_payload(self, payload: dict) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object")

        enabled = bool(payload.get("enabled", False))
        raw_mappings = payload.get("mappings", [])
        if raw_mappings is None:
            raw_mappings = []
        if not isinstance(raw_mappings, list):
            raise HTTPException(status_code=400, detail='"mappings" must be a list.')

        mappings = []
        seen_sources = set()
        for index, entry in enumerate(raw_mappings, start=1):
            if not isinstance(entry, dict):
                raise HTTPException(status_code=400, detail=f"Mapping #{index} must be an object.")

            source_model = normalize_routing_model_name(entry.get("source_model") or entry.get("source"))
            target_model = normalize_routing_model_name(entry.get("target_model") or entry.get("target"))
            if not source_model:
                raise HTTPException(status_code=400, detail=f"Mapping #{index} must include a valid source_model.")
            if not target_model:
                raise HTTPException(status_code=400, detail=f"Mapping #{index} must include a valid target_model.")
            if source_model not in self._known_models:
                raise HTTPException(status_code=400, detail=f"Mapping #{index} source model is unsupported: {source_model}")
            if target_model not in self._known_models:
                raise HTTPException(status_code=400, detail=f"Mapping #{index} target model is unsupported: {target_model}")
            if source_model in seen_sources:
                raise HTTPException(status_code=400, detail=f"Duplicate mapping source_model: {source_model}")

            seen_sources.add(source_model)
            mappings.append(
                {
                    "source_model": source_model,
                    "source_provider": model_provider_family(source_model),
                    "target_model": target_model,
                    "target_provider": model_provider_family(target_model),
                }
            )

        return {
            "enabled": enabled,
            "mappings": mappings,
        }
