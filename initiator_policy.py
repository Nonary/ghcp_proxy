from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Callable

AGENT_INITIATOR = "agent"
USER_INITIATOR = "user"
AGENT_INITIATOR_PREFIX = "_"
USER_INITIATOR_PREFIX = "+"
_EXPLICIT_USER_INITIATOR = "__explicit_user__"
REQUEST_FINISH_GUARD_SECONDS = 15.0
_SECURITY_MONITOR_PROMPT_MARKERS = (
    "you are a security monitor for autonomous ai coding agents",
)
_CONVERSATION_SUMMARIZER_PROMPT_MARKERS = (
    "you are a helpful ai assistant tasked with summarizing conversations.",
)


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


def _match_initiator_marker(stripped: str) -> tuple[str, str | None]:
    """Match a leading initiator marker in `stripped` (whitespace already removed
    from the left). Returns (remainder, initiator) on match, else (stripped, None).

    A trailing single space after the marker is consumed if present, mirroring
    what users naturally type ("_ foo" vs "_foo" both yield "foo").
    """
    if stripped.startswith(AGENT_INITIATOR_PREFIX):
        initiator = AGENT_INITIATOR
    elif stripped.startswith(USER_INITIATOR_PREFIX):
        initiator = _EXPLICIT_USER_INITIATOR
    else:
        return stripped, None
    remainder = stripped[1:]
    if remainder.startswith(" "):
        remainder = remainder[1:]
    return remainder, initiator


def _strip_explicit_initiator_prefix(text: str) -> tuple[str, str | None]:
    if not isinstance(text, str):
        return text, None

    remainder, initiator = _match_initiator_marker(text.lstrip())
    if initiator is not None:
        return remainder, initiator
    # Trailing marker on its own line: handles harnesses (e.g. Claude Code)
    # that append <system-reminder> blocks AFTER the user's typed prompt
    # in the same content string.
    return _strip_explicit_initiator_marker_on_last_line(text)


def _strip_explicit_initiator_marker_on_last_line(text: str) -> tuple[str, str | None]:
    rstripped = text.rstrip()
    if not rstripped:
        return text, None

    last_newline = rstripped.rfind("\n")
    if last_newline == -1:
        # Single-line input — already handled by the leading-prefix check.
        return text, None

    prefix = rstripped[: last_newline + 1]
    last_line = rstripped[last_newline + 1 :]
    remainder, initiator = _match_initiator_marker(last_line.lstrip())
    if initiator is None:
        return text, None
    return prefix + remainder, initiator


def _is_user_candidate(candidate_initiator: str | None) -> bool:
    return candidate_initiator in {USER_INITIATOR, _EXPLICIT_USER_INITIATOR}


def _public_candidate_initiator(candidate_initiator: str | None) -> str:
    if _is_user_candidate(candidate_initiator):
        return USER_INITIATOR
    return AGENT_INITIATOR


def _candidate_bypasses_safeguards(candidate_initiator: str | None) -> bool:
    # Explicit "+" prefix signals "this is a user turn, I mean it" — it
    # bypasses both the post-request cooldown AND the active_request guard.
    # Hard agent rules (subagent, Haiku, codex bootstrap/title-gen) are
    # checked earlier in _resolve_locked and are NOT overridden by "+".
    return candidate_initiator == _EXPLICIT_USER_INITIATOR


def _is_claude_compaction_summary_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    normalized = text.lstrip()
    return normalized.startswith(
        "This session is being continued from a previous conversation that ran out of context."
    )


def _contains_conversation_summarizer_prompt(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    return any(marker in normalized for marker in _CONVERSATION_SUMMARIZER_PROMPT_MARKERS)


def _is_claude_meta_user_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    normalized = text.lstrip()
    if not normalized:
        return False
    if (
        normalized.startswith("The user stepped away and is coming back.")
        and "Recap in under 40 words" in normalized
    ):
        return True
    if normalized.startswith((
        "<local-command-caveat>",
        "<local-command-stdout>",
        "<local-command-stderr>",
        "<task-notification>",
        "<system-reminder>",
    )):
        return True
    if (
        normalized.startswith("<command-name>")
        and "<command-message>" in normalized
        and "<command-args>" in normalized
    ):
        return True
    return _is_claude_compaction_summary_text(normalized)


def _contains_security_monitor_prompt(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    return any(marker in normalized for marker in _SECURITY_MONITOR_PROMPT_MARKERS)


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
        ignorable_meta_messages: set[int] = set()
        for item in input_param:
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "")).lower() != "user":
                continue
            if id(item) in explicit_initiators:
                continue
            saw_agent_meta, saw_real_user_content = _responses_user_message_traits(item)
            if saw_agent_meta and not saw_real_user_content:
                ignorable_meta_messages.add(id(item))
        for item in reversed(input_param):
            if isinstance(item, dict) and id(item) in ignorable_meta_messages:
                continue
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


def _responses_system_prompt_includes_security_monitor_prompt(input_param) -> bool:
    if not isinstance(input_param, list):
        return False
    for item in input_param:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).lower()
        if role not in {"system", "developer"}:
            continue
        text = _responses_item_text(item)
        if _contains_security_monitor_prompt(text) or _contains_conversation_summarizer_prompt(text):
            return True
    return False


def _is_codex_meta_user_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    normalized = text.lstrip()
    if not normalized:
        return False
    if normalized.startswith((
        "<turn_aborted>",
        "<subagent_notification>",
        "<subagent-notification>",
        "<task-notification>",
        "<system-reminder>",
    )):
        return True
    return _is_claude_compaction_summary_text(normalized)


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


def _responses_user_message_traits(item) -> tuple[bool, bool]:
    if not isinstance(item, dict):
        return False, False
    if str(item.get("role", "")).lower() != "user":
        return False, False

    if _is_environment_context_message(item) or _is_task_title_generation_message(item):
        return True, False

    content = item.get("content")
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped:
            return False, False
        if _is_codex_meta_user_text(content):
            return True, False
        return False, True

    if isinstance(content, list):
        saw_agent_meta = False
        saw_real_user_content = False
        for entry in content:
            if not isinstance(entry, dict):
                saw_real_user_content = True
                continue
            entry_type = str(entry.get("type", "")).lower()
            if entry_type == "input_image":
                saw_real_user_content = True
                continue
            if entry_type not in {"", "text", "input_text", "output_text"}:
                saw_real_user_content = True
                continue

            text = entry.get("text")
            if not isinstance(text, str):
                text = entry.get("input_text")
            if not isinstance(text, str):
                text = entry.get("output_text")
            if not isinstance(text, str):
                saw_real_user_content = True
                continue
            if not text.strip():
                continue
            if _is_codex_meta_user_text(text):
                saw_agent_meta = True
                continue
            saw_real_user_content = True
        return saw_agent_meta, saw_real_user_content

    for key in ("text", "input_text"):
        value = item.get(key)
        if not isinstance(value, str):
            continue
        if not value.strip():
            return False, False
        if _is_codex_meta_user_text(value):
            return True, False
        return False, True
    return False, False


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


def _chat_system_prompt_includes_security_monitor_prompt(messages) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        if role not in {"system", "developer"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and _contains_security_monitor_prompt(content):
            return True
    return False


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
        message["content"] = updated_text
        if explicit_initiator is not None:
            return explicit_initiator
        return default_initiator

    if not isinstance(content, list):
        return default_initiator

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).lower()
        if item_type != "text":
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        updated_text, explicit_initiator = _strip_explicit_initiator_prefix(text)
        item["text"] = updated_text
        if explicit_initiator is not None:
            return explicit_initiator
    return default_initiator


def _anthropic_user_message_traits(message) -> tuple[bool, bool, bool]:
    if not isinstance(message, dict):
        return False, False, False

    content = message.get("content")
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped:
            return False, False, False
        if _is_claude_meta_user_text(content):
            return False, True, False
        return False, False, True

    if not isinstance(content, list):
        return False, False, False

    saw_tool_result = False
    saw_agent_meta = False
    saw_real_user_content = False
    for item in content:
        if not isinstance(item, dict):
            saw_real_user_content = True
            continue
        item_type = str(item.get("type", "")).lower()
        if item_type == "tool_result":
            saw_tool_result = True
            continue
        if item_type != "text":
            saw_real_user_content = True
            continue
        text = item.get("text")
        if not isinstance(text, str):
            saw_real_user_content = True
            continue
        if not text.strip():
            continue
        if _is_claude_meta_user_text(text):
            saw_agent_meta = True
            continue
        saw_real_user_content = True
    return saw_tool_result, saw_agent_meta, saw_real_user_content


def _determine_anthropic_candidate(messages, *, system=None) -> str:
    if _anthropic_system_includes_security_monitor_prompt(system):
        return AGENT_INITIATOR
    if not isinstance(messages, list):
        return AGENT_INITIATOR

    explicit_initiators: dict[int, str] = {}
    ignorable_meta_messages: set[int] = set()
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
            continue
        saw_tool_result, saw_agent_meta, saw_real_user_content = _anthropic_user_message_traits(message)
        if saw_agent_meta and not saw_tool_result and not saw_real_user_content:
            ignorable_meta_messages.add(id(message))

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).lower()
        if role == "user":
            if id(message) in ignorable_meta_messages:
                continue
            if id(message) in explicit_initiators:
                return explicit_initiators[id(message)]
            saw_tool_result, _saw_agent_meta, saw_real_user_content = _anthropic_user_message_traits(message)
            if saw_tool_result and not saw_real_user_content:
                return AGENT_INITIATOR
            return explicit_initiators.get(id(message), USER_INITIATOR)
        if role:
            return AGENT_INITIATOR
    return AGENT_INITIATOR


def _anthropic_system_includes_security_monitor_prompt(system) -> bool:
    if isinstance(system, str):
        return (
            _contains_security_monitor_prompt(system)
            or _contains_conversation_summarizer_prompt(system)
        )
    if not isinstance(system, list):
        return False
    text_parts: list[str] = []
    for item in system:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).lower() != "text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
    if not text_parts:
        return False
    combined = " ".join(text_parts)
    return (
        _contains_security_monitor_prompt(combined)
        or _contains_conversation_summarizer_prompt(combined)
    )


def _request_includes_security_monitor_prompt(
    *,
    inbound_protocol: str | None = None,
    body: dict | None = None,
) -> bool:
    if not isinstance(body, dict):
        return False
    if inbound_protocol == "responses":
        return _responses_system_prompt_includes_security_monitor_prompt(body.get("input"))
    if inbound_protocol == "messages":
        return _anthropic_system_includes_security_monitor_prompt(body.get("system"))
    if inbound_protocol == "chat":
        return _chat_system_prompt_includes_security_monitor_prompt(body.get("messages"))
    return False


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
        self._last_finished_at: datetime | None = None
        self._seen_user_request: bool = False

    def seed_from_usage_events(self, events, *, now: datetime | None = None):
        del now
        with self._lock:
            self._events.clear()
            self._active_requests.clear()
            self._last_activity_at = None
            self._last_finished_at = None
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
                self._last_finished_at = finished_at
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
        verdict_sink: dict | None = None,
    ) -> tuple[object, str]:
        normalized_input, candidate = _determine_responses_candidate(input_param)
        if _responses_system_prompt_includes_security_monitor_prompt(normalized_input):
            candidate = AGENT_INITIATOR
        forced_by_codex_mini = False
        if (
            _is_codex_bootstrap_mini_request(normalized_input, model_name)
            or _is_codex_title_generation_mini_request(normalized_input, model_name)
        ):
            candidate = AGENT_INITIATOR
            forced_by_codex_mini = True
        initiator = self.resolve_initiator(
            candidate,
            model_name,
            subagent=subagent,
            force_initiator=force_initiator,
            now=now,
            request_id=request_id,
            verdict_sink=verdict_sink,
        )
        if isinstance(verdict_sink, dict) and forced_by_codex_mini:
            verdict_sink["codex_mini_forced"] = True
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
        verdict_sink: dict | None = None,
    ) -> str:
        candidate = _determine_chat_candidate(messages)
        if _chat_system_prompt_includes_security_monitor_prompt(messages):
            candidate = AGENT_INITIATOR
        return self.resolve_initiator(
            candidate,
            model_name,
            subagent=subagent,
            force_initiator=force_initiator,
            now=now,
            request_id=request_id,
            verdict_sink=verdict_sink,
        )

    def resolve_anthropic_messages(
        self,
        messages,
        model_name: str | None,
        *,
        system=None,
        subagent: str | None = None,
        force_initiator: str | None = None,
        now: datetime | None = None,
        request_id: str | None = None,
        verdict_sink: dict | None = None,
    ) -> str:
        candidate = _determine_anthropic_candidate(messages, system=system)
        return self.resolve_initiator(
            candidate,
            model_name,
            subagent=subagent,
            force_initiator=force_initiator,
            now=now,
            request_id=request_id,
            verdict_sink=verdict_sink,
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
        verdict_sink: dict | None = None,
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
            active_count = len(self._active_requests)
            last_finished_at = self._last_finished_at
        if safeguard_reason is not None and self.on_safeguard_triggered is not None:
            self.on_safeguard_triggered(
                {
                    "triggered_at": resolved_now.isoformat(),
                    "trigger_reason": safeguard_reason,
                    "candidate_initiator": _public_candidate_initiator(candidate_initiator),
                    "resolved_initiator": initiator,
                    "model_name": _normalize_model_name(model_name),
                    "request_id": request_id,
                }
            )
        if isinstance(verdict_sink, dict):
            verdict_sink.update(
                {
                    "candidate_initiator": _public_candidate_initiator(candidate_initiator),
                    "resolved_initiator": initiator,
                    "safeguard_reason": safeguard_reason,
                    "force_initiator": force_initiator if force_initiator in {AGENT_INITIATOR, USER_INITIATOR} else None,
                    "subagent": subagent if _is_subagent_request(subagent) else None,
                    "haiku_forced": _is_haiku_model(model_name),
                    "explicit_user_prefix": candidate_initiator == _EXPLICIT_USER_INITIATOR,
                    "active_requests_at_decision": active_count,
                    "seconds_since_last_activity": (
                        (resolved_now - last_finished_at).total_seconds()
                        if last_finished_at is not None
                        else None
                    ),
                    "cooldown_seconds": self.request_finish_guard_seconds,
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
                self._last_finished_at = event_time
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

        initiator = _public_candidate_initiator(candidate_initiator)
        safeguard_reason = self._safeguard_reason_locked(now)
        if initiator == USER_INITIATOR and safeguard_reason is not None:
            if _candidate_bypasses_safeguards(candidate_initiator):
                return USER_INITIATOR, None
            return AGENT_INITIATOR, safeguard_reason
        return initiator, None

    def _safeguard_reason_locked(self, now: datetime) -> str | None:
        if not self._seen_user_request:
            return None
        # Note: we intentionally do NOT demote candidate=user just because
        # other requests are in flight. Clients like opencode/Copilot fan out
        # multiple parallel requests (title-gen, context-prep, the real turn)
        # at the moment the user hits enter, so "active requests exist" is
        # routinely true for the genuine user turn. The structural candidate
        # walker (_determine_*_candidate) already rejects tool_result-tail
        # continuations as agent regardless of timing, so an agent follow-up
        # that races in during an in-flight user turn will still be classified
        # correctly on its own merits. The post-finish cooldown below is
        # retained as a belt-and-suspenders guard for the narrow window
        # immediately after a request completes.
        if self._last_finished_at is None:
            return None
        elapsed_seconds = (now - self._last_finished_at).total_seconds()
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
def is_approval_agent_request(
    *,
    subagent: str | None = None,
    inbound_protocol: str | None = None,
    body: dict | None = None,
) -> bool:
    """Detect whether an incoming request originates from an approval agent.

    Covers two patterns:
      * Codex sends ``x-openai-subagent: <name>`` (e.g. ``guardian``) for
        approval / review spawns; callers surface that header via ``subagent``.
      * Approval/security-monitor prompts identify themselves explicitly in the
        system/developer instructions.
    """
    if _is_subagent_request(subagent):
        return True
    return _request_includes_security_monitor_prompt(
        inbound_protocol=inbound_protocol,
        body=body,
    )
