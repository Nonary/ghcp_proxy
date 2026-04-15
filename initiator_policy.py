from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Callable

AGENT_INITIATOR = "agent"
USER_INITIATOR = "user"
EXPLICIT_USER_INITIATOR = "user!"
AGENT_INITIATOR_PREFIX = "_"
USER_INITIATOR_PREFIX = "+"
REQUEST_FINISH_GUARD_SECONDS = 15.0


def utc_now() -> datetime:
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


def _is_subagent_request(subagent: str | None) -> bool:
    return isinstance(subagent, str) and bool(subagent.strip())


def _parse_event_time(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _strip_explicit_initiator_prefix(text: str) -> tuple[str, str | None]:
    if not isinstance(text, str):
        return text, None

    stripped = text.lstrip()
    if stripped.startswith(AGENT_INITIATOR_PREFIX):
        initiator = AGENT_INITIATOR
    elif stripped.startswith(USER_INITIATOR_PREFIX):
        initiator = EXPLICIT_USER_INITIATOR
    else:
        return text, None

    normalized = stripped[1:]
    if normalized.startswith(" "):
        normalized = normalized[1:]
    return normalized, initiator


def _is_claude_compaction_summary_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    normalized = text.lstrip()
    return normalized.startswith(
        "This session is being continued from a previous conversation that ran out of context."
    )


def _is_claude_meta_user_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    normalized = text.lstrip()
    if not normalized:
        return False
    if normalized.startswith((
        "<local-command-caveat>",
        "<local-command-stdout>",
        "<local-command-stderr>",
        "<task-notification>",
    )):
        return True
    if (
        normalized.startswith("<command-name>")
        and "<command-message>" in normalized
        and "<command-args>" in normalized
    ):
        return True
    return _is_claude_compaction_summary_text(normalized)


def _strip_explicit_initiator_prefix_from_item(item, *, default_initiator: str | None = USER_INITIATOR) -> str | None:
    if not isinstance(item, dict):
        return AGENT_INITIATOR

    content = item.get("content")
    if isinstance(content, str):
        updated_text, explicit_initiator = _strip_explicit_initiator_prefix(content)
        if explicit_initiator is not None:
            item["content"] = updated_text
            return explicit_initiator
        return default_initiator

    if isinstance(content, list):
        for entry in content:
            if not isinstance(entry, dict):
                continue
            for key in ("text", "input_text"):
                value = entry.get(key)
                if not isinstance(value, str):
                    continue
                updated_text, explicit_initiator = _strip_explicit_initiator_prefix(value)
                if explicit_initiator is not None:
                    entry[key] = updated_text
                    return explicit_initiator
        return default_initiator

    for key in ("text", "input_text"):
        value = item.get(key)
        if not isinstance(value, str):
            continue
        updated_text, explicit_initiator = _strip_explicit_initiator_prefix(value)
        if explicit_initiator is not None:
            item[key] = updated_text
            return explicit_initiator
        return default_initiator
    return default_initiator


def _explicit_initiators_for_responses_input(input_param) -> dict[int, str]:
    explicit_initiators: dict[int, str] = {}
    if not isinstance(input_param, list):
        return explicit_initiators

    for item in input_param:
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).lower() != "user":
            continue
        explicit_initiator = _strip_explicit_initiator_prefix_from_item(item, default_initiator=None)
        if explicit_initiator is not None:
            explicit_initiators[id(item)] = explicit_initiator
    return explicit_initiators


def _responses_item_candidate(item, *, explicit_initiators: dict[int, str] | None = None) -> str | None:
    if not isinstance(item, dict):
        return AGENT_INITIATOR

    item_type = str(item.get("type", "")).lower()
    role = str(item.get("role", "")).lower()

    if item_type in {"reasoning", "item_reference", "compaction"}:
        return None

    if item_type in {"", "message"}:
        if role in {"system", "developer"}:
            return None
        if role == "user":
            if explicit_initiators is not None and id(item) in explicit_initiators:
                return explicit_initiators[id(item)]
            return USER_INITIATOR
        if role:
            return AGENT_INITIATOR
        return AGENT_INITIATOR

    if item_type in {"function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output"}:
        return AGENT_INITIATOR

    if role == "user":
        if explicit_initiators is not None and id(item) in explicit_initiators:
            return explicit_initiators[id(item)]
        return USER_INITIATOR
    if role in {"system", "developer"}:
        return None
    return AGENT_INITIATOR


def _determine_responses_candidate(input_param) -> tuple[object, str]:
    if isinstance(input_param, str):
        normalized_input, explicit_initiator = _strip_explicit_initiator_prefix(input_param)
        return normalized_input, explicit_initiator or USER_INITIATOR

    if isinstance(input_param, list):
        explicit_initiators = _explicit_initiators_for_responses_input(input_param)
        for item in reversed(input_param):
            candidate = _responses_item_candidate(item, explicit_initiators=explicit_initiators)
            if candidate is not None:
                return input_param, candidate

    return input_param, AGENT_INITIATOR


def _responses_item_text(item) -> str:
    if not isinstance(item, dict):
        return ""

    content = item.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for entry in content:
            if not isinstance(entry, dict):
                continue
            for key in ("text", "input_text"):
                value = entry.get(key)
                if isinstance(value, str):
                    parts.append(value)
                    break
        return "".join(parts)

    for key in ("text", "input_text"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def _is_environment_context_message(item) -> bool:
    if not isinstance(item, dict):
        return False
    if str(item.get("role", "")).lower() != "user":
        return False
    text = _responses_item_text(item).lstrip()
    return "<environment_context>" in text and "</environment_context>" in text


def _is_task_title_generation_message(item) -> bool:
    if not isinstance(item, dict):
        return False
    if str(item.get("role", "")).lower() != "user":
        return False
    text = _responses_item_text(item).lstrip()
    return (
        "You are a helpful assistant. You will be presented with a user prompt" in text
        and "Generate a concise UI title" in text
        and "User prompt:" in text
    )


def _is_codex_bootstrap_mini_request(input_param, model_name: str | None) -> bool:
    if _normalize_model_name(model_name) != "gpt-5.4-mini":
        return False
    if not isinstance(input_param, list):
        return False

    message_items: list[dict] = []
    for item in input_param:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).lower()
        role = str(item.get("role", "")).lower()
        if item_type == "message" or role:
            message_items.append(item)

    if len(message_items) < 2:
        return False

    for item in message_items[:-1]:
        role = str(item.get("role", "")).lower()
        if role not in {"developer", "system"}:
            return False

    return _is_environment_context_message(message_items[-1])


def _is_codex_title_generation_mini_request(input_param, model_name: str | None) -> bool:
    normalized_model = _normalize_model_name(model_name)
    if normalized_model not in {"gpt-5.4-mini", "gpt-5.1-codex-mini"}:
        return False
    if not isinstance(input_param, list):
        return False

    message_items: list[dict] = []
    for item in input_param:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).lower()
        role = str(item.get("role", "")).lower()
        if item_type == "message" or role:
            message_items.append(item)

    if len(message_items) < 3:
        return False
    if not _is_task_title_generation_message(message_items[-1]):
        return False

    saw_developer_or_system = False
    saw_environment_context = False
    for item in message_items[:-1]:
        role = str(item.get("role", "")).lower()
        if role in {"developer", "system"}:
            saw_developer_or_system = True
        if _is_environment_context_message(item):
            saw_environment_context = True

    return saw_developer_or_system and saw_environment_context


def _determine_chat_candidate(messages) -> str:
    if not isinstance(messages, list):
        return AGENT_INITIATOR

    explicit_initiators: dict[int, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).lower() != "user":
            continue
        explicit_initiator = _strip_explicit_initiator_prefix_from_item(message, default_initiator=None)
        if explicit_initiator is not None:
            explicit_initiators[id(message)] = explicit_initiator

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        if role == "user":
            return explicit_initiators.get(id(message), USER_INITIATOR)
        if role:
            return AGENT_INITIATOR
    return AGENT_INITIATOR


def _strip_explicit_initiator_prefix_from_anthropic_message(
    message,
    *,
    default_initiator: str | None = USER_INITIATOR,
) -> str | None:
    if not isinstance(message, dict):
        return AGENT_INITIATOR

    content = message.get("content")
    if isinstance(content, str):
        updated_text, explicit_initiator = _strip_explicit_initiator_prefix(content)
        if explicit_initiator is not None:
            message["content"] = updated_text
            return explicit_initiator
        message["content"] = updated_text
        if _is_claude_meta_user_text(updated_text):
            return AGENT_INITIATOR
        return default_initiator

    if not isinstance(content, list):
        return default_initiator

    saw_tool_result = False
    saw_non_tool_result = False
    saw_agent_meta = False
    for item in content:
        if not isinstance(item, dict):
            saw_non_tool_result = True
            continue
        item_type = str(item.get("type", "")).lower()
        if item_type == "tool_result":
            saw_tool_result = True
            continue
        if item_type != "text":
            saw_non_tool_result = True
            continue
        text = item.get("text")
        if not isinstance(text, str):
            saw_non_tool_result = True
            continue
        updated_text, explicit_initiator = _strip_explicit_initiator_prefix(text)
        if explicit_initiator is not None:
            item["text"] = updated_text
            return explicit_initiator
        item["text"] = updated_text
        if not updated_text.strip():
            continue
        if _is_claude_meta_user_text(updated_text):
            saw_agent_meta = True
            continue
        saw_non_tool_result = True
    if (saw_tool_result or saw_agent_meta) and not saw_non_tool_result:
        return AGENT_INITIATOR
    return default_initiator


def _determine_anthropic_candidate(messages) -> str:
    if not isinstance(messages, list):
        return AGENT_INITIATOR

    explicit_initiators: dict[int, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).lower() != "user":
            continue
        explicit_initiator = _strip_explicit_initiator_prefix_from_anthropic_message(
            message,
            default_initiator=None,
        )
        if explicit_initiator is not None:
            explicit_initiators[id(message)] = explicit_initiator

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        if role == "user":
            return explicit_initiators.get(id(message), USER_INITIATOR)
        if role:
            return AGENT_INITIATOR
    return AGENT_INITIATOR


class InitiatorPolicy:
    def __init__(
        self,
        *,
        request_finish_guard_seconds: float = REQUEST_FINISH_GUARD_SECONDS,
        max_events: int = 512,
        on_safeguard_triggered: Callable[[dict], None] | None = None,
    ):
        self.request_finish_guard_seconds = request_finish_guard_seconds
        self.on_safeguard_triggered = on_safeguard_triggered
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
        subagent: str | None = None,
        force_initiator: str | None = None,
        now: datetime | None = None,
        request_id: str | None = None,
    ) -> tuple[object, str]:
        normalized_input, candidate = _determine_responses_candidate(input_param)
        if (
            _is_codex_bootstrap_mini_request(normalized_input, model_name)
            or _is_codex_title_generation_mini_request(normalized_input, model_name)
        ):
            candidate = AGENT_INITIATOR
        initiator = self.resolve_initiator(
            candidate,
            model_name,
            subagent=subagent,
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
        subagent: str | None = None,
        force_initiator: str | None = None,
        now: datetime | None = None,
        request_id: str | None = None,
    ) -> str:
        candidate = _determine_chat_candidate(messages)
        return self.resolve_initiator(
            candidate,
            model_name,
            subagent=subagent,
            force_initiator=force_initiator,
            now=now,
            request_id=request_id,
        )

    def resolve_anthropic_messages(
        self,
        messages,
        model_name: str | None,
        *,
        subagent: str | None = None,
        force_initiator: str | None = None,
        now: datetime | None = None,
        request_id: str | None = None,
    ) -> str:
        candidate = _determine_anthropic_candidate(messages)
        return self.resolve_initiator(
            candidate,
            model_name,
            subagent=subagent,
            force_initiator=force_initiator,
            now=now,
            request_id=request_id,
        )

    def resolve_initiator(
        self,
        candidate_initiator: str | None,
        model_name: str | None,
        *,
        subagent: str | None = None,
        force_initiator: str | None = None,
        now: datetime | None = None,
        request_id: str | None = None,
    ) -> str:
        resolved_now = now or utc_now()
        with self._lock:
            initiator, safeguard_reason = self._resolve_locked(
                candidate_initiator,
                model_name,
                subagent,
                resolved_now,
                force_initiator,
            )
            if isinstance(request_id, str) and request_id:
                self._record_request_started_locked(request_id, initiator, resolved_now)
        if safeguard_reason is not None and self.on_safeguard_triggered is not None:
            self.on_safeguard_triggered(
                {
                    "triggered_at": resolved_now.isoformat(),
                    "trigger_reason": safeguard_reason,
                    "candidate_initiator": candidate_initiator,
                    "resolved_initiator": initiator,
                    "model_name": _normalize_model_name(model_name),
                    "request_id": request_id,
                }
            )
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
        event_time = started_at or utc_now()
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
        event_time = finished_at or utc_now()
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
        subagent: str | None,
        now: datetime,
        force_initiator: str | None,
    ) -> tuple[str, str | None]:
        if force_initiator in {AGENT_INITIATOR, USER_INITIATOR}:
            return force_initiator, None

        if _is_subagent_request(subagent):
            return AGENT_INITIATOR, None

        if _is_haiku_model(model_name):
            return AGENT_INITIATOR, None

        initiator = USER_INITIATOR if candidate_initiator in {USER_INITIATOR, EXPLICIT_USER_INITIATOR} else AGENT_INITIATOR
        if initiator == USER_INITIATOR and candidate_initiator == EXPLICIT_USER_INITIATOR:
            return initiator, None
        safeguard_reason = self._safeguard_reason_locked(now)
        if initiator == USER_INITIATOR and safeguard_reason is not None:
            return AGENT_INITIATOR, safeguard_reason
        return initiator, None

    def _safeguard_reason_locked(self, now: datetime) -> str | None:
        if not self._seen_user_request:
            return None
        if self._active_requests:
            return "active_request"
        if self._last_activity_at is None:
            return None
        elapsed_seconds = (now - self._last_activity_at).total_seconds()
        if elapsed_seconds < self.request_finish_guard_seconds:
            return "cooldown"
        return None

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
