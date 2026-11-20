"""Translate client reasoning choices to native Copilot Fusion routing tiers."""

from __future__ import annotations


MODEL_ID = "fusion"
_EFFORT_TO_TIER = {"medium": "standard", "standard": "standard", "high": "high"}
_TIER_TO_EFFORT = {"standard": "medium", "high": "high"}


def is_fusion_model(model: object) -> bool:
    return isinstance(model, str) and model.strip().lower().rsplit("/", 1)[-1] == MODEL_ID


def tier_for_request(body: dict) -> str:
    """Codex uses medium/high; direct callers may also use native standard."""
    effort = body.get("reasoning_effort")
    if effort is None:
        reasoning = body.get("reasoning")
        effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    if effort is None:
        return "standard"
    normalized = effort.strip().lower() if isinstance(effort, str) else None
    if normalized not in _EFFORT_TO_TIER:
        raise ValueError(
            "Copilot Fusion supports standard (medium) and high routing tiers; "
            f"unsupported reasoning effort: {effort!r}."
        )
    return _EFFORT_TO_TIER[normalized]


def model_capabilities(catalog: dict) -> dict[str, dict]:
    """Expose only account-enabled, visible tiers from SDK models.list metadata.

    Fusion is a native alias of the virtual Auto model, not a CAPI /models
    entry. Never infer its availability from ordinary inference models.
    """
    auto = catalog.get("auto")
    tiers = auto.get("tiers") if isinstance(auto, dict) else None
    if not isinstance(tiers, list):
        return {}
    enabled = set()
    for tier in tiers:
        if not isinstance(tier, dict):
            continue
        status = tier.get("status")
        name = tier.get("fusionTier")
        if (
            isinstance(status, dict)
            and status.get("enabled") is True
            and status.get("pickerVisible") is not False
            and isinstance(name, str)
            and name in _TIER_TO_EFFORT
        ):
            enabled.add(name)
    available = [tier for tier in _TIER_TO_EFFORT if tier in enabled]
    if not available:
        return {}
    return {
        MODEL_ID: {
            "display_name": "Copilot Fusion",
            "provider": "GitHub",
            "model_picker_enabled": True,
            "fusion_tiers": available,
            "reasoning_efforts": [_TIER_TO_EFFORT[tier] for tier in available],
        }
    }
