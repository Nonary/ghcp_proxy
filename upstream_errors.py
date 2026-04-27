"""Translate upstream error responses into normalized synthetic replies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import usage_tracking
import util


@dataclass(frozen=True)
class SyntheticReply:
    """Protocol-agnostic reply generated from a classified upstream error."""

    status_for_trace: int
    client_status: int
    message: str
    reason: str
    usage_shape: str = "zero"
    windows: tuple[str, ...] = ()
    reset_at_by_window: dict[str, str | None] = field(default_factory=dict)
    event_name: str | None = None
    event_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class UpstreamErrorContext:
    status_code: int
    headers: Mapping[str, str]
    upstream: Any = None


class UpstreamErrorRule(Protocol):
    def match(self, context: UpstreamErrorContext) -> SyntheticReply | None:
        ...


def _format_limit_reset_duration(seconds: float | None) -> str:
    if seconds is None:
        return "an unknown amount of time"
    seconds = max(0, int(math.ceil(float(seconds))))
    if seconds < 60:
        return "less than 1 minute"
    minutes = int(math.ceil(seconds / 60.0))
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes and len(parts) < 2:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return " ".join(parts[:2]) if parts else "less than 1 minute"


def _limit_window_is_exhausted(window: dict | None) -> bool:
    if not isinstance(window, dict):
        return False
    remaining = window.get("percent_remaining")
    if isinstance(remaining, (int, float)):
        return float(remaining) <= 0.0
    used = window.get("percent_used")
    if isinstance(used, (int, float)):
        return float(used) >= 100.0
    overage = window.get("overage")
    overage_permitted = window.get("overage_permitted")
    if isinstance(overage, (int, float)) and float(overage) > 0 and overage_permitted is False:
        return True
    return False


def _window_reset_duration_text(window: dict | None) -> str:
    reset_at = window.get("reset_at") if isinstance(window, dict) else None
    reset_dt = util._parse_iso_datetime(reset_at if isinstance(reset_at, str) else None)
    if reset_dt is None:
        return "an unknown amount of time"
    now = util.utc_now()
    if reset_dt.tzinfo is None:
        reset_dt = reset_dt.replace(tzinfo=now.tzinfo)
    return _format_limit_reset_duration((reset_dt - now).total_seconds())


class CopilotUsageLimitRule:
    """Classify Copilot 429s caused by exhausted 5h/weekly usage windows."""

    window_order = ("session", "weekly")
    window_labels = {
        "session": "5h",
        "weekly": "weekly",
    }

    def _window_label(self, window_name: str) -> str:
        return self.window_labels.get(window_name, window_name)

    def match(self, context: UpstreamErrorContext) -> SyntheticReply | None:
        if context.status_code != 429:
            return None
        windows = usage_tracking.extract_usage_ratelimits_from_headers(context.headers)
        exhausted: list[tuple[str, dict]] = []
        for name in self.window_order:
            window = windows.get(name)
            if _limit_window_is_exhausted(window):
                exhausted.append((name, window))
        if not exhausted:
            return None

        if len(exhausted) == 1:
            label = self._window_label(exhausted[0][0])
            headline = f"Your Copilot {label} usage limit has been reached."
        else:
            labels = [self._window_label(name) for name, _ in exhausted]
            headline = f"Your Copilot {' and '.join(labels)} usage limits have been reached."

        reset_parts = [
            f"The {self._window_label(name)} limit resets in about {_window_reset_duration_text(window)}"
            for name, window in exhausted
        ]
        message = f"{headline} {'; '.join(reset_parts)}. Please try again later."
        reset_at_by_window = {
            name: window.get("reset_at") if isinstance(window.get("reset_at"), str) else None
            for name, window in exhausted
        }
        window_names = tuple(name for name, _ in exhausted)
        event_payload = {
            "reason": "copilot_usage_limit",
            "upstream_status": context.status_code,
            "window": list(window_names),
            "limit": [self._window_label(name) for name in window_names],
            "reset_at": reset_at_by_window,
        }
        return SyntheticReply(
            status_for_trace=context.status_code,
            client_status=200,
            message=message,
            reason="copilot_usage_limit",
            usage_shape="zero",
            windows=window_names,
            reset_at_by_window=reset_at_by_window,
            event_name="upstream_limit_hit",
            event_payload=event_payload,
        )


DEFAULT_RULES: tuple[UpstreamErrorRule, ...] = (CopilotUsageLimitRule(),)


def translate(upstream: Any, rules: Sequence[UpstreamErrorRule] = DEFAULT_RULES) -> SyntheticReply | None:
    """Return a normalized synthetic reply for a known upstream error, if any."""

    status_code = getattr(upstream, "status_code", None)
    if not isinstance(status_code, int):
        return None
    headers = getattr(upstream, "headers", {}) or {}
    context = UpstreamErrorContext(status_code=status_code, headers=headers, upstream=upstream)
    for rule in rules:
        reply = rule.match(context)
        if reply is not None:
            return reply
    return None


def friendly_limit_message_from_upstream(upstream: Any) -> str | None:
    """Compatibility helper for callers/tests that only need the user text."""

    reply = translate(upstream, rules=(CopilotUsageLimitRule(),))
    return reply.message if reply is not None else None
