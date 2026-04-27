"""Session usage reminder notices for streaming AI responses."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from threading import Lock
from typing import Callable

from util import _parse_iso_datetime, utc_now


DEFAULT_SESSION_REMINDER_STEP_PERCENT = 10


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _window_percent_used(window: dict | None) -> float | None:
    if not isinstance(window, dict):
        return None
    percent_used = _float_or_none(window.get("percent_used"))
    if percent_used is not None:
        return max(0.0, min(percent_used, 100.0))
    percent_remaining = _float_or_none(window.get("percent_remaining"))
    if percent_remaining is not None:
        return max(0.0, min(100.0 - percent_remaining, 100.0))
    return None


def _window_percent_remaining(window: dict | None) -> float | None:
    if not isinstance(window, dict):
        return None
    percent_remaining = _float_or_none(window.get("percent_remaining"))
    if percent_remaining is not None:
        return max(0.0, min(percent_remaining, 100.0))
    percent_used = _float_or_none(window.get("percent_used"))
    if percent_used is not None:
        return max(0.0, min(100.0 - percent_used, 100.0))
    return None


def _reminder_threshold(percent_used: float | None, step_percent: int = DEFAULT_SESSION_REMINDER_STEP_PERCENT) -> int:
    if percent_used is None or step_percent <= 0:
        return 0
    threshold = int(math.floor(max(0.0, percent_used) / float(step_percent)) * step_percent)
    return min(100, threshold)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "unknown"
    rounded = round(float(value), 1)
    if rounded.is_integer():
        return f"{int(rounded)}%"
    return f"{rounded}%"


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _seconds_until(reset_at: str | None, now: datetime) -> float | None:
    reset_dt = _parse_iso_datetime(reset_at)
    if reset_dt is None:
        return None
    return max(0.0, (_ensure_aware(reset_dt) - _ensure_aware(now)).total_seconds())


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "an unknown amount of time"
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return "less than 1 minute"
    minutes = seconds // 60
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


def _window_reset_at(window: dict | None) -> str | None:
    if not isinstance(window, dict):
        return None
    reset_at = window.get("reset_at")
    return reset_at if isinstance(reset_at, str) and reset_at else None


def _usage_window_samples(
    usage_events: list[dict],
    window_name: str,
    *,
    reset_at: str | None,
) -> list[tuple[datetime, float]]:
    samples: list[tuple[datetime, float]] = []
    for event in usage_events:
        if not isinstance(event, dict):
            continue
        windows = event.get("usage_ratelimits")
        if not isinstance(windows, dict):
            continue
        window = windows.get(window_name)
        if not isinstance(window, dict):
            continue
        if reset_at is not None and _window_reset_at(window) != reset_at:
            continue
        if reset_at is None and _window_reset_at(window) is not None:
            continue
        used = _window_percent_used(window)
        if used is None:
            continue
        ts_raw = event.get("finished_at") or event.get("started_at")
        ts = _parse_iso_datetime(ts_raw) if isinstance(ts_raw, str) else None
        if ts is None:
            continue
        samples.append((_ensure_aware(ts), used))
    samples.sort(key=lambda item: item[0])
    return samples


def _history_max_threshold(
    usage_events: list[dict],
    *,
    reset_at: str | None,
    step_percent: int,
) -> int:
    max_threshold = 0
    for _ts, used in _usage_window_samples(usage_events, "session", reset_at=reset_at):
        max_threshold = max(max_threshold, _reminder_threshold(used, step_percent))
    return max_threshold


def _burn_projection_text(
    current_windows: dict,
    usage_events: list[dict],
    *,
    now: datetime,
) -> str:
    session = current_windows.get("session") if isinstance(current_windows, dict) else None
    if not isinstance(session, dict):
        return "I do not have enough session-limit data yet to estimate burn-down."

    reset_at = _window_reset_at(session)
    current_used = _window_percent_used(session)
    current_remaining = _window_percent_remaining(session)
    samples = _usage_window_samples(usage_events, "session", reset_at=reset_at)
    if current_used is not None:
        samples.append((_ensure_aware(now), current_used))
    samples.sort(key=lambda item: item[0])

    # Pick the oldest sample in the current reset window that gives a positive
    # burn delta. This keeps the reminder stable and avoids one noisy immediate
    # predecessor dominating the projection.
    burn_sample: tuple[datetime, float] | None = None
    if current_used is not None:
        for ts, used in samples:
            if ts >= _ensure_aware(now):
                continue
            if current_used - used > 0:
                burn_sample = (ts, used)
                break

    reset_suffix = ""
    seconds_to_reset = _seconds_until(reset_at, now)
    if reset_at:
        reset_suffix = f" The 5h window resets at {reset_at} (in about {_format_duration(seconds_to_reset)})."

    if burn_sample is None or current_used is None or current_remaining is None:
        return (
            "There is not enough prior 5h-limit history in this reset window to estimate a burn-down rate yet."
            f"{reset_suffix}"
        )

    start_ts, start_used = burn_sample
    elapsed_seconds = max(0.0, (_ensure_aware(now) - start_ts).total_seconds())
    delta_used = max(0.0, current_used - start_used)
    if elapsed_seconds <= 0 or delta_used <= 0:
        return (
            "Your recent recorded 5h-limit burn is flat, so you are not currently projected to run out before reset."
            f"{reset_suffix}"
        )

    rate_per_hour = delta_used / (elapsed_seconds / 3600.0)
    hours_to_empty = current_remaining / rate_per_hour if rate_per_hour > 0 else None

    if hours_to_empty is None:
        projection = "I cannot project when it will run out yet."
    elif seconds_to_reset is None:
        projection = f"At that pace, the remaining 5h limit would last about {_format_duration(hours_to_empty * 3600.0)}."
    elif hours_to_empty * 3600.0 <= seconds_to_reset:
        projection = (
            f"At that pace, it is expected to run out in about {_format_duration(hours_to_empty * 3600.0)}, "
            "before the 5h limit resets."
        )
    else:
        projection = (
            f"At that pace, it should last about {_format_duration(hours_to_empty * 3600.0)}, "
            "so it is not expected to run out before the 5h limit resets."
        )

    return (
        f"Recent burn-down is about {_format_percent(rate_per_hour)} of the 5h limit per hour, "
        f"based on a {_format_percent(delta_used)} increase over {_format_duration(elapsed_seconds)}."
        f"{reset_suffix} {projection}"
    )


def build_usage_reminder_text(
    current_windows: dict,
    usage_events: list[dict] | None = None,
    *,
    now: datetime | None = None,
    threshold_percent: int | None = None,
) -> str:
    """Build the human-readable reminder injected into the assistant stream."""
    if now is None:
        now = utc_now()
    if usage_events is None:
        usage_events = []

    session = current_windows.get("session") if isinstance(current_windows, dict) else None
    if not isinstance(session, dict):
        return ""
    session_remaining = _window_percent_remaining(session)
    session_used = _window_percent_used(session)
    if session_remaining is None and session_used is None:
        return ""

    threshold_clause = ""
    if threshold_percent:
        threshold_clause = f" This crossed the {threshold_percent}% 5h-usage reminder."

    weekly_clause = "Weekly limit: not reported by the upstream response."
    weekly = current_windows.get("weekly") if isinstance(current_windows, dict) else None
    if isinstance(weekly, dict):
        weekly_remaining = _window_percent_remaining(weekly)
        weekly_used = _window_percent_used(weekly)
        weekly_reset = _window_reset_at(weekly)
        weekly_entitlement = weekly.get("entitlement")
        entitlement_clause = ""
        if isinstance(weekly_entitlement, (int, float)) and weekly_entitlement > 0:
            entitlement_clause = f", entitlement {int(weekly_entitlement)}"
        weekly_clause = (
            f"Weekly limit: {_format_percent(weekly_remaining)} remaining "
            f"({_format_percent(weekly_used)} used{entitlement_clause})"
        )
        if weekly_reset:
            weekly_clause += f", resets at {weekly_reset}"
        weekly_clause += "."

    burn_text = _burn_projection_text(current_windows, usage_events, now=now)
    return (
        f"By the way, your 5h usage limit is currently at {_format_percent(session_remaining)} remaining "
        f"({_format_percent(session_used)} used).{threshold_clause} {burn_text} {weekly_clause}"
    )


class UsageReminderController:
    """Decides when 10%-step session usage reminders should be injected."""

    def __init__(
        self,
        snapshot_usage_events: Callable[[], list[dict]],
        *,
        now_func: Callable[[], datetime] = utc_now,
        step_percent: int = DEFAULT_SESSION_REMINDER_STEP_PERCENT,
    ):
        self.snapshot_usage_events = snapshot_usage_events
        self.now_func = now_func
        self.step_percent = max(1, int(step_percent))
        self._lock = Lock()
        self._last_notified_thresholds_by_reset: dict[str, int] = {}

    def usage_notice_text_if_due(self, current_windows: dict | None) -> str:
        if not isinstance(current_windows, dict):
            return ""
        session = current_windows.get("session")
        if not isinstance(session, dict):
            return ""
        percent_used = _window_percent_used(session)
        threshold = _reminder_threshold(percent_used, self.step_percent)
        if threshold < self.step_percent:
            return ""

        reset_at = _window_reset_at(session)
        reset_key = reset_at or "__no_reset__"
        try:
            usage_events = self.snapshot_usage_events() or []
        except Exception:
            usage_events = []

        history_threshold = _history_max_threshold(
            usage_events,
            reset_at=reset_at,
            step_percent=self.step_percent,
        )
        with self._lock:
            already_notified = max(
                history_threshold,
                int(self._last_notified_thresholds_by_reset.get(reset_key, 0) or 0),
            )
            if threshold <= already_notified:
                return ""
            self._last_notified_thresholds_by_reset[reset_key] = threshold
            # Keep the small in-memory ledger bounded across rolling resets.
            if len(self._last_notified_thresholds_by_reset) > 8:
                for key in list(self._last_notified_thresholds_by_reset.keys())[:-8]:
                    self._last_notified_thresholds_by_reset.pop(key, None)

        return build_usage_reminder_text(
            current_windows,
            usage_events,
            now=self.now_func(),
            threshold_percent=threshold,
        )
