from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import Lock

AGENT_INITIATOR = "agent"
USER_INITIATOR = "user"
AGENT_INITIATOR_PREFIX = "_"
REQUEST_FINISH_GUARD_SECONDS = 15.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_model_name(model_name: str | None) -> str | None:
    if not isinstance(model_name, str):
        return None
    normalized = model_name.strip().lower().replace("_", "-")
    if normalized.startswith("anthropic/"):
        normalized = normalized.split("/", 1)[1]
    return normalized


def _is_haiku_model(model_name: str | None) -> bool:
    normalized = _normalize_model_name(model_name)
    return isinstance(normalized, str) and "haiku" in normalized


def _parse_event_time(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _strip_agent_initiator_prefix(text: str) -> tuple[str, bool]:
    if not isinstance(text, str):
        return text, False

    stripped = text.lstrip()
    if not stripped.startswith(AGENT_INITIATOR_PREFIX):
        return text, False

    normalized = stripped[len(AGENT_INITIATOR_PREFIX) :]
    if normalized.startswith(" "):
        normalized = normalized[1:]
    return normalized, True


def _strip_agent_initiator_prefix_from_item(item) -> str:
    if not isinstance(item, dict):
        return AGENT_INITIATOR

    content = item.get("content")
    if isinstance(content, str):
        updated_text, explicit_agent = _strip_agent_initiator_prefix(content)
        if explicit_agent:
            item["content"] = updated_text
            return AGENT_INITIATOR
        return USER_INITIATOR

    if isinstance(content, list):
        for entry in content:
            if not isinstance(entry, dict):
                continue
            for key in ("text", "input_text"):
                value = entry.get(key)
                if not isinstance(value, str):
                    continue
                updated_text, explicit_agent = _strip_agent_initiator_prefix(value)
                if explicit_agent:
                    entry[key] = updated_text
                    return AGENT_INITIATOR
        return USER_INITIATOR

    for key in ("text", "input_text"):
        value = item.get(key)
        if not isinstance(value, str):
            continue
        updated_text, explicit_agent = _strip_agent_initiator_prefix(value)
        if explicit_agent:
            item[key] = updated_text
            return AGENT_INITIATOR
        return USER_INITIATOR
    return USER_INITIATOR


def _determine_responses_candidate(input_param) -> tuple[object, str]:
    if isinstance(input_param, str):
        normalized_input, explicit_agent = _strip_agent_initiator_prefix(input_param)
        return normalized_input, AGENT_INITIATOR if explicit_agent else USER_INITIATOR

    if isinstance(input_param, list):
        for item in reversed(input_param):
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "")).lower() == "user":
                return input_param, _strip_agent_initiator_prefix_from_item(item)

    return input_param, AGENT_INITIATOR


def _determine_chat_candidate(messages) -> str:
    if not isinstance(messages, list):
        return AGENT_INITIATOR

    for message in reversed(messages):
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "user":
            return _strip_agent_initiator_prefix_from_item(message)
    return AGENT_INITIATOR


def _strip_agent_initiator_prefix_from_anthropic_message(message) -> str:
    if not isinstance(message, dict):
        return AGENT_INITIATOR

    content = message.get("content")
    if isinstance(content, str):
        updated_text, explicit_agent = _strip_agent_initiator_prefix(content)
        if explicit_agent:
            message["content"] = updated_text
            return AGENT_INITIATOR
        return USER_INITIATOR

    if not isinstance(content, list):
        return USER_INITIATOR

    for item in content:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).lower() != "text":
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        updated_text, explicit_agent = _strip_agent_initiator_prefix(text)
        if explicit_agent:
            item["text"] = updated_text
            return AGENT_INITIATOR
    return USER_INITIATOR


def _determine_anthropic_candidate(messages) -> str:
    if not isinstance(messages, list):
        return AGENT_INITIATOR

    for message in reversed(messages):
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "user":
            return _strip_agent_initiator_prefix_from_anthropic_message(message)
    return AGENT_INITIATOR


class InitiatorPolicy:
    def __init__(
        self,
        *,
        request_finish_guard_seconds: float = REQUEST_FINISH_GUARD_SECONDS,
        max_events: int = 512,
    ):
        self.request_finish_guard_seconds = request_finish_guard_seconds
        self._events = deque(maxlen=max_events)
        self._lock = Lock()
        self._active_requests: dict[str, dict[str, str | None]] = {}
        self._last_activity_at: datetime | None = None
        self._seen_user_request: bool = False

    def seed_from_usage_events(self, events, *, now: datetime | None = None):
        del now
        with self._lock:
            self._events.clear()
            self._active_requests.clear()
            self._last_activity_at = None
            self._seen_user_request = False

            if not isinstance(events, list):
                return

            seeded_events = []
            for event in events:
                if not isinstance(event, dict):
                    continue
                finished_at = _parse_event_time(event.get("finished_at"))
                if finished_at is None:
                    continue
                initiator = event.get("initiator")
                if initiator not in {AGENT_INITIATOR, USER_INITIATOR}:
                    continue
                seeded_events.append((finished_at, initiator))

            seeded_events.sort(key=lambda item: item[0])
            for finished_at, initiator in seeded_events:
                self._last_activity_at = finished_at
                self._events.append(
                    {
                        "at": finished_at.isoformat(),
                        "initiator": initiator,
                        "type": "finished",
                    }
                )

    def resolve_responses_input(
        self,
        input_param,
        model_name: str | None,
        *,
        force_initiator: str | None = None,
        now: datetime | None = None,
        request_id: str | None = None,
    ) -> tuple[object, str]:
        normalized_input, candidate = _determine_responses_candidate(input_param)
        initiator = self.resolve_initiator(
            candidate,
            model_name,
            force_initiator=force_initiator,
            now=now,
            request_id=request_id,
        )
        return normalized_input, initiator

    def resolve_chat_messages(
        self,
        messages,
        model_name: str | None,
        *,
        force_initiator: str | None = None,
        now: datetime | None = None,
        request_id: str | None = None,
    ) -> str:
        candidate = _determine_chat_candidate(messages)
        return self.resolve_initiator(
            candidate,
            model_name,
            force_initiator=force_initiator,
            now=now,
            request_id=request_id,
        )

    def resolve_anthropic_messages(
        self,
        messages,
        model_name: str | None,
        *,
        force_initiator: str | None = None,
        now: datetime | None = None,
        request_id: str | None = None,
    ) -> str:
        candidate = _determine_anthropic_candidate(messages)
        return self.resolve_initiator(
            candidate,
            model_name,
            force_initiator=force_initiator,
            now=now,
            request_id=request_id,
        )

    def resolve_initiator(
        self,
        candidate_initiator: str | None,
        model_name: str | None,
        *,
        force_initiator: str | None = None,
        now: datetime | None = None,
        request_id: str | None = None,
    ) -> str:
        resolved_now = now or _utc_now()
        with self._lock:
            initiator = self._resolve_locked(candidate_initiator, model_name, resolved_now, force_initiator)
            if isinstance(request_id, str) and request_id:
                self._record_request_started_locked(request_id, initiator, resolved_now)
            return initiator

    def note_request_started(
        self,
        request_id: str | None,
        initiator: str | None,
        *,
        started_at: datetime | None = None,
    ):
        if not isinstance(request_id, str) or not request_id:
            return
        if initiator not in {AGENT_INITIATOR, USER_INITIATOR}:
            return
        event_time = started_at or _utc_now()
        with self._lock:
            self._record_request_started_locked(request_id, initiator, event_time)

    def note_request_finished(
        self,
        request_id: str | None,
        *,
        finished_at: datetime | None = None,
    ):
        if not isinstance(request_id, str) or not request_id:
            return
        event_time = finished_at or _utc_now()
        with self._lock:
            active = self._active_requests.pop(request_id, None)
            initiator = active.get("initiator") if isinstance(active, dict) else None
            if active is not None:
                self._last_activity_at = event_time
            self._events.append(
                {
                    "at": event_time.isoformat(),
                    "initiator": initiator,
                    "request_id": request_id,
                    "type": "finished",
                }
            )

    def _resolve_locked(
        self,
        candidate_initiator: str | None,
        model_name: str | None,
        now: datetime,
        force_initiator: str | None,
    ) -> str:
        if force_initiator in {AGENT_INITIATOR, USER_INITIATOR}:
            return force_initiator

        if _is_haiku_model(model_name):
            return AGENT_INITIATOR

        initiator = USER_INITIATOR if candidate_initiator == USER_INITIATOR else AGENT_INITIATOR
        if initiator == USER_INITIATOR and self._safeguard_active_locked(now):
            return AGENT_INITIATOR
        return initiator

    def _safeguard_active_locked(self, now: datetime) -> bool:
        if not self._seen_user_request:
            return False
        if self._active_requests:
            return True
        if self._last_activity_at is None:
            return False
        elapsed_seconds = (now - self._last_activity_at).total_seconds()
        return elapsed_seconds < self.request_finish_guard_seconds

    def _check_and_expire_safeguard_locked(self, now: datetime) -> bool:
        if not self._seen_user_request:
            return False
        if self._active_requests:
            return True
        if self._last_activity_at is None:
            return False
        elapsed_seconds = (now - self._last_activity_at).total_seconds()
        return elapsed_seconds < self.request_finish_guard_seconds

    def _record_request_started_locked(self, request_id: str, initiator: str, event_time: datetime):
        if initiator == USER_INITIATOR:
            self._seen_user_request = True
        elif not self._seen_user_request:
            self._events.append(
                {
                    "at": event_time.isoformat(),
                    "initiator": initiator,
                    "request_id": request_id,
                    "type": "started",
                }
            )
            return
        self._active_requests[request_id] = {
            "initiator": initiator,
            "started_at": event_time.isoformat(),
        }
        self._last_activity_at = event_time
        self._events.append(
            {
                "at": event_time.isoformat(),
                "initiator": initiator,
                "request_id": request_id,
                "type": "started",
            }
        )
