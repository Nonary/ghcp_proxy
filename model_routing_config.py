"""Persistent configuration for model remapping."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from fastapi import HTTPException

import format_translation
from constants import DEFAULT_COMPACT_FALLBACK_MODEL, MODEL_PRICING, MODEL_ROUTING_CONFIG_FILE, TOKEN_DIR
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
    if normalized.startswith("gemini-"):
        return "gemini"
    if normalized.startswith("grok-"):
        return "grok"
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
        self._claude_code_default_slots = ("opus_model", "sonnet_model", "haiku_model")

    def config_payload(self) -> dict[str, object]:
        current = self.load_settings()
        return {
            "enabled": current["enabled"],
            "mappings": current["mappings"],
            "approval_enabled": current["approval_enabled"],
            "approval_mappings": current["approval_mappings"],
            "claude_code_defaults": current["claude_code_defaults"],
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
                    "approval_enabled": normalized["approval_enabled"],
                    "approval_mappings": normalized["approval_mappings"],
                    "claude_code_defaults": normalized["claude_code_defaults"],
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

    def resolve_compact_fallback_model(self, requested_model: str | None) -> str | None:
        """Return the GPT model to use when a compact is requested against a
        chat-backed target (Claude/Gemini/Grok). Returns None if no remap is
        active for this model or if the resolved target is already a Codex
        model. Falls back to DEFAULT_COMPACT_FALLBACK_MODEL when the mapping
        entry has no explicit override.
        """
        normalized_requested = normalize_routing_model_name(requested_model)
        if not normalized_requested:
            return None

        settings = self.load_settings()
        if not settings["enabled"]:
            return None

        for mapping in settings["mappings"]:
            if mapping["source_model"] != normalized_requested:
                continue
            if mapping.get("target_provider") == "codex":
                return None
            override = mapping.get("compact_fallback_model")
            fallback = override or DEFAULT_COMPACT_FALLBACK_MODEL
            if model_provider_family(fallback) != "codex":
                # Guard against a misconfigured non-GPT fallback — that would
                # just reproduce the original context-overflow error.
                return DEFAULT_COMPACT_FALLBACK_MODEL
            return fallback
        return None

    def resolve_approval_target_model(self, requested_model: str | None) -> str | None:
        normalized_requested = normalize_routing_model_name(requested_model)
        if not normalized_requested:
            return None

        settings = self.load_settings()
        if not settings["approval_enabled"]:
            return None

        for mapping in settings["approval_mappings"]:
            if mapping["source_model"] == normalized_requested:
                return mapping["target_model"]
        return None

    def default_settings(self) -> dict[str, object]:
        return {
            "enabled": False,
            "mappings": [],
            "approval_enabled": False,
            "approval_mappings": [],
            "claude_code_defaults": {
                "opus_model": "",
                "sonnet_model": "",
                "haiku_model": "",
            },
        }

    def _normalize_settings_payload(self, payload: dict) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object")

        enabled = bool(payload.get("enabled", False))
        mappings = self._normalize_mapping_list(payload.get("mappings", []), label="Mapping")

        approval_enabled = bool(payload.get("approval_enabled", False))
        approval_mappings = self._normalize_mapping_list(
            payload.get("approval_mappings", []),
            label="Approval mapping",
        )
        claude_code_defaults = self._normalize_claude_code_defaults(
            payload.get("claude_code_defaults"),
        )

        return {
            "enabled": enabled,
            "mappings": mappings,
            "approval_enabled": approval_enabled,
            "approval_mappings": approval_mappings,
            "claude_code_defaults": claude_code_defaults,
        }

    def _normalize_mapping_list(self, raw_mappings, *, label: str) -> list[dict[str, str]]:
        if raw_mappings is None:
            raw_mappings = []
        if not isinstance(raw_mappings, list):
            raise HTTPException(status_code=400, detail=f'"{label.lower()}s" must be a list.')

        mappings: list[dict[str, str]] = []
        seen_sources: set[str] = set()
        for index, entry in enumerate(raw_mappings, start=1):
            if not isinstance(entry, dict):
                raise HTTPException(status_code=400, detail=f"{label} #{index} must be an object.")

            source_model = normalize_routing_model_name(entry.get("source_model") or entry.get("source"))
            target_model = normalize_routing_model_name(entry.get("target_model") or entry.get("target"))
            if not source_model:
                raise HTTPException(status_code=400, detail=f"{label} #{index} must include a valid source_model.")
            if not target_model:
                raise HTTPException(status_code=400, detail=f"{label} #{index} must include a valid target_model.")
            if source_model not in self._known_models:
                raise HTTPException(status_code=400, detail=f"{label} #{index} source model is unsupported: {source_model}")
            if target_model not in self._known_models:
                raise HTTPException(status_code=400, detail=f"{label} #{index} target model is unsupported: {target_model}")
            if source_model in seen_sources:
                raise HTTPException(status_code=400, detail=f"Duplicate {label.lower()} source_model: {source_model}")

            seen_sources.add(source_model)
            normalized_entry: dict[str, str] = {
                "source_model": source_model,
                "source_provider": model_provider_family(source_model),
                "target_model": target_model,
                "target_provider": model_provider_family(target_model),
            }
            raw_compact_fallback = entry.get("compact_fallback_model") or entry.get("compact_fallback")
            if raw_compact_fallback is not None and str(raw_compact_fallback).strip():
                compact_fallback = normalize_routing_model_name(raw_compact_fallback)
                if not compact_fallback:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{label} #{index} compact_fallback_model is not a recognized model.",
                    )
                if compact_fallback not in self._known_models:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{label} #{index} compact_fallback_model is unsupported: {compact_fallback}",
                    )
                if model_provider_family(compact_fallback) != "codex":
                    raise HTTPException(
                        status_code=400,
                        detail=f"{label} #{index} compact_fallback_model must be a GPT model (got {compact_fallback}).",
                    )
                normalized_entry["compact_fallback_model"] = compact_fallback
            mappings.append(normalized_entry)
        return mappings

    def _normalize_claude_code_defaults(self, raw_defaults: object) -> dict[str, str]:
        if raw_defaults is None:
            raw_defaults = {}
        if not isinstance(raw_defaults, dict):
            raise HTTPException(status_code=400, detail='"claude_code_defaults" must be an object.')

        normalized_defaults: dict[str, str] = {}
        for slot_key in self._claude_code_default_slots:
            raw_value = raw_defaults.get(slot_key)
            if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
                normalized_defaults[slot_key] = ""
                continue
            model_name = normalize_routing_model_name(raw_value)
            if not model_name:
                raise HTTPException(
                    status_code=400,
                    detail=f'claude_code_defaults.{slot_key} must be a recognized model.',
                )
            if model_name not in self._known_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"claude_code_defaults.{slot_key} is unsupported: {model_name}",
                )
            normalized_defaults[slot_key] = model_name

        return normalized_defaults
