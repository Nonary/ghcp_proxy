"""Persistent premium plan configuration and manual quota resync state."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import HTTPException

from constants import PREMIUM_PLAN_CONFIG_FILE, TOKEN_DIR


PREMIUM_PLAN_OPTIONS = (
    {"id": "pro", "label": "Pro", "included": 300},
    {"id": "business", "label": "Business", "included": 300},
    {"id": "enterprise", "label": "Enterprise", "included": 1000},
    {"id": "pro_plus", "label": "Pro+", "included": 1500},
)

_PREMIUM_PLAN_BY_ID = {row["id"]: row for row in PREMIUM_PLAN_OPTIONS}
_PREMIUM_PLAN_ALIASES = {
    "pro": "pro",
    "business": "business",
    "enterprise": "enterprise",
    "proplus": "pro_plus",
    "pro+": "pro_plus",
    "pro_plus": "pro_plus",
    "plus": "pro_plus",
}


def normalize_premium_plan_id(value) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = normalized.replace("__", "_")
    return _PREMIUM_PLAN_ALIASES.get(normalized)


@dataclass(frozen=True)
class PremiumPlanConfig:
    config_file: str = PREMIUM_PLAN_CONFIG_FILE


class PremiumPlanConfigService:
    def __init__(self, config: PremiumPlanConfig):
        self._config = config

    def default_settings(self) -> dict[str, object]:
        return {
            "configured": False,
            "plan": "",
            "plan_label": "",
            "included": None,
            "synced_percent": 0.0,
            "synced_used": 0.0,
            "synced_at": None,
            "synced_month": "",
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
                detail=f"Invalid premium plan settings in {self._config.config_file}",
            )

        return self._normalize_loaded_payload(payload)

    def config_payload(self) -> dict[str, object]:
        current = self.load_settings()
        return {
            **current,
            "available_plans": list(PREMIUM_PLAN_OPTIONS),
            "path": self._config.config_file,
        }

    def save_settings(self, payload: dict, *, now: datetime | None = None) -> dict[str, object]:
        normalized = self._normalize_request_payload(payload, now=now)
        os.makedirs(os.path.dirname(self._config.config_file) or TOKEN_DIR, exist_ok=True)
        with open(self._config.config_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "plan": normalized["plan"],
                    "synced_percent": normalized["synced_percent"],
                    "synced_used": normalized["synced_used"],
                    "synced_at": normalized["synced_at"],
                    "synced_month": normalized["synced_month"],
                },
                f,
                indent=2,
            )
            f.write("\n")
        return self.config_payload()

    def clear_settings(self) -> dict[str, object]:
        try:
            os.remove(self._config.config_file)
        except OSError:
            pass
        return self.config_payload()

    def _normalize_loaded_payload(self, payload: dict) -> dict[str, object]:
        plan_id = normalize_premium_plan_id(payload.get("plan"))
        if not plan_id:
            return self.default_settings()

        plan = _PREMIUM_PLAN_BY_ID[plan_id]
        synced_percent = self._coerce_percent(payload.get("synced_percent"), default=0.0)
        synced_used = payload.get("synced_used")
        if not isinstance(synced_used, (int, float)):
            synced_used = round(plan["included"] * (synced_percent / 100.0), 2)

        return {
            "configured": True,
            "plan": plan["id"],
            "plan_label": plan["label"],
            "included": int(plan["included"]),
            "synced_percent": round(float(synced_percent), 2),
            "synced_used": round(float(synced_used), 2),
            "synced_at": payload.get("synced_at"),
            "synced_month": str(payload.get("synced_month") or "").strip(),
        }

    def _normalize_request_payload(self, payload: dict, *, now: datetime | None = None) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object")

        plan_id = normalize_premium_plan_id(payload.get("plan"))
        if not plan_id:
            raise HTTPException(status_code=400, detail="A valid premium plan is required.")

        raw_percent = payload.get("current_percent")
        if raw_percent is None:
            raw_percent = payload.get("synced_percent")
        synced_percent = self._coerce_percent(raw_percent, default=None)
        if synced_percent is None:
            raise HTTPException(status_code=400, detail="A current used percentage is required.")

        current_time = now or datetime.now(timezone.utc)
        plan = _PREMIUM_PLAN_BY_ID[plan_id]
        synced_used = round(plan["included"] * (synced_percent / 100.0), 2)
        return {
            "configured": True,
            "plan": plan["id"],
            "plan_label": plan["label"],
            "included": int(plan["included"]),
            "synced_percent": round(synced_percent, 2),
            "synced_used": synced_used,
            "synced_at": current_time.isoformat(),
            "synced_month": current_time.strftime("%Y-%m"),
        }

    def _coerce_percent(self, value, *, default):
        if value is None:
            return default
        try:
            percent = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Current used percentage must be a number.") from exc
        return max(0.0, min(percent, 100.0))
