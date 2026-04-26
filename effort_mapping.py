"""Model-aware reasoning-effort normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod


_CANONICAL = {"low", "medium", "high", "max", "xhigh"}
_COMMON_ALIASES = {
    "minimal": "low",
    "none": "low",
}
_CLAUDE_ALIASES = {
    "xhigh": "max",
    "x-high": "max",
    "extra-high": "max",
    "extra_high": "max",
}


def _normalize_model(model: str | None) -> str:
    normalized = (model or "").strip().lower()
    if normalized.startswith("anthropic/"):
        normalized = normalized.split("/", 1)[1]
    return normalized


def _canonicalize(model: str | None, effort: str | None) -> str | None:
    if not isinstance(effort, str):
        return None
    normalized = effort.strip().lower()
    if not normalized:
        return None
    normalized_model = _normalize_model(model)
    if normalized_model.startswith("claude-"):
        mapped = _CLAUDE_ALIASES.get(normalized)
        if mapped is not None:
            return mapped
    if normalized in _CANONICAL:
        return normalized
    return _COMMON_ALIASES.get(normalized)


class ModelEffortStrategy(ABC):
    @abstractmethod
    def matches(self, normalized_model: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def map(self, canonical_effort: str) -> str | None:
        raise NotImplementedError


class GptStrategy(ModelEffortStrategy):
    def matches(self, normalized_model: str) -> bool:
        return normalized_model.startswith("gpt-")

    def map(self, canonical_effort: str) -> str | None:
        if canonical_effort == "max":
            return "xhigh"
        return canonical_effort


class ClaudeOpus47Strategy(ModelEffortStrategy):
    def matches(self, normalized_model: str) -> bool:
        return normalized_model == "claude-opus-4.7"

    def map(self, canonical_effort: str) -> str | None:
        return "medium"


class ClaudeHaiku45Strategy(ModelEffortStrategy):
    def matches(self, normalized_model: str) -> bool:
        return normalized_model == "claude-haiku-4.5"

    def map(self, canonical_effort: str) -> str | None:
        return None


class PassthroughStrategy(ModelEffortStrategy):
    def matches(self, normalized_model: str) -> bool:
        return True

    def map(self, canonical_effort: str) -> str | None:
        return canonical_effort


_STRATEGIES: list[ModelEffortStrategy] = [
    GptStrategy(),
    ClaudeOpus47Strategy(),
    ClaudeHaiku45Strategy(),
    PassthroughStrategy(),
]


def _strategy_for(model: str | None) -> ModelEffortStrategy:
    normalized = _normalize_model(model)
    for strategy in _STRATEGIES:
        if strategy.matches(normalized):
            return strategy
    return _STRATEGIES[-1]


def map_effort_for_model(model: str | None, effort: str | None) -> str | None:
    canonical = _canonicalize(model, effort)
    if canonical is None:
        return None
    return _strategy_for(model).map(canonical)
