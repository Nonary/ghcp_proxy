"""Replay item id repair for native Responses cache lineage."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping

from request_headers import responses_replay_affinity_value


_MAX_LINEAGE_STATES = 256
_MAX_IDS_PER_STATE = 4096


def _normalized_non_empty_string(value) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _header_value(headers, name: str) -> str | None:
    if not isinstance(headers, Mapping):
        return None
    value = headers.get(name)
    normalized = _normalized_non_empty_string(value)
    if normalized:
        return normalized
    target = name.lower()
    for key, candidate in headers.items():
        if isinstance(key, str) and key.lower() == target:
            normalized = _normalized_non_empty_string(candidate)
            if normalized:
                return normalized
    return None


def lineage_key_for_body(
    body: dict | None,
    headers=None,
    *,
    subagent: str | None = None,
) -> str | None:
    """Return the stable request lineage used to persist upstream item ids."""
    effective_subagent = (
        _normalized_non_empty_string(subagent)
        or _header_value(headers, "x-openai-subagent")
    )
    if isinstance(body, dict):
        for key in ("prompt_cache_key", "promptCacheKey"):
            value = _normalized_non_empty_string(body.get(key))
            if value:
                # Header affinity gives rollout-memory writers a private
                # namespace, but replay repair used to retain their raw prompt
                # cache key. That let background JSON writers and the visible
                # conversation share repaired item IDs.
                effective_affinity = responses_replay_affinity_value(
                    body, subagent=effective_subagent
                )
                return f"prompt_cache:{effective_affinity or value}"
        for key in ("session_id", "sessionId"):
            value = _normalized_non_empty_string(body.get(key))
            if value:
                effective_affinity = responses_replay_affinity_value(
                    body, subagent=effective_subagent
                )
                return f"session:{effective_affinity or value}"

        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            for key in ("session_id", "sessionId"):
                value = _normalized_non_empty_string(metadata.get(key))
                if value:
                    effective_affinity = responses_replay_affinity_value(
                        body, subagent=effective_subagent
                    )
                    return f"session:{effective_affinity or value}"

        for key in ("previous_response_id", "previousResponseId"):
            value = _normalized_non_empty_string(body.get(key))
            if value:
                effective_affinity = responses_replay_affinity_value(
                    body, subagent=effective_subagent
                )
                return f"previous_response:{effective_affinity or value}"

    for header_name in (
        "x-client-session-id",
        "x-claude-code-session-id",
        "x-session-affinity",
        "x-opencode-session",
        "session-id",
        "session_id",
    ):
        value = _header_value(headers, header_name)
        if value:
            return f"header:{header_name}:{value}"

    return None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_hash(value) -> str:
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        serialized = repr(value)
    return _sha256_text(serialized)


def _assistant_message_fingerprint(item: dict) -> str | None:
    if not isinstance(item, dict):
        return None
    if str(item.get("type", "")).lower() != "message":
        return None
    if str(item.get("role", "")).lower() != "assistant":
        return None
    return _canonical_json_hash(
        {
            "role": "assistant",
            "content": item.get("content"),
        }
    )


def _assistant_message_has_replay_content(item: dict) -> bool:
    content = item.get("content")
    if isinstance(content, str):
        return bool(content)
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, str) and part:
            return True
        if isinstance(part, dict):
            for key in ("text", "input_text", "output_text"):
                value = part.get(key)
                if isinstance(value, str) and value:
                    return True
    return False


def _reasoning_fingerprint(item: dict) -> str | None:
    encrypted_content = item.get("encrypted_content")
    if not isinstance(encrypted_content, str) or not encrypted_content:
        return None
    return _sha256_text(encrypted_content)


def _function_item_id(call_id: str) -> str:
    return f"fc_{call_id}"


def _trim_ordered_map(mapping: OrderedDict) -> None:
    while len(mapping) > _MAX_IDS_PER_STATE:
        mapping.popitem(last=False)


class ReplayIdState:
    """Per-lineage map of Copilot output item identity."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._function_item_ids: OrderedDict[str, str] = OrderedDict()
        self._reasoning_item_ids: OrderedDict[str, str] = OrderedDict()
        self._assistant_message_item_ids: OrderedDict[str, str] = OrderedDict()
        self._assistant_fingerprint_counts: dict[str, int] = {}
        self._assistant_observed_item_fingerprints: set[tuple[str, str]] = set()

    @staticmethod
    def _assistant_key(fingerprint: str, ordinal: int) -> str:
        return f"{fingerprint}:{ordinal}"

    def _remember_function_id_locked(self, call_id: str) -> None:
        self._function_item_ids[call_id] = _function_item_id(call_id)
        self._function_item_ids.move_to_end(call_id)
        _trim_ordered_map(self._function_item_ids)

    def _remember_reasoning_id_locked(self, fingerprint: str, item_id: str) -> None:
        self._reasoning_item_ids[fingerprint] = item_id
        self._reasoning_item_ids.move_to_end(fingerprint)
        _trim_ordered_map(self._reasoning_item_ids)

    def _remember_assistant_message_id_locked(
        self,
        fingerprint: str,
        item_id: str,
        ordinal: int | None = None,
    ) -> None:
        if ordinal is None:
            ordinal = self._assistant_fingerprint_counts.get(fingerprint, 0)
            self._assistant_fingerprint_counts[fingerprint] = ordinal + 1
        key = self._assistant_key(fingerprint, ordinal)
        self._assistant_message_item_ids[key] = item_id
        self._assistant_message_item_ids.move_to_end(key)
        _trim_ordered_map(self._assistant_message_item_ids)

    def observe_input_items(self, input_items) -> None:
        """Learn IDs the client did preserve so later sparse replays can be fixed."""
        if not isinstance(input_items, list):
            return

        assistant_ordinals: dict[str, int] = {}
        with self._lock:
            for item in input_items:
                if not isinstance(item, dict):
                    continue
                item_id = _normalized_non_empty_string(item.get("id"))
                if not item_id:
                    fingerprint = _assistant_message_fingerprint(item)
                    if fingerprint is not None:
                        assistant_ordinals[fingerprint] = assistant_ordinals.get(fingerprint, 0) + 1
                    continue

                item_type = str(item.get("type", "")).lower()
                if item_type in {"function_call", "function_call_output"}:
                    call_id = _normalized_non_empty_string(item.get("call_id"))
                    if call_id:
                        self._remember_function_id_locked(call_id)
                    continue

                if item_type == "reasoning":
                    fingerprint = _reasoning_fingerprint(item)
                    if fingerprint is not None:
                        self._remember_reasoning_id_locked(fingerprint, item_id)
                    continue

                fingerprint = _assistant_message_fingerprint(item)
                if fingerprint is not None:
                    ordinal = assistant_ordinals.get(fingerprint, 0)
                    assistant_ordinals[fingerprint] = ordinal + 1
                    self._remember_assistant_message_id_locked(
                        fingerprint,
                        item_id,
                        ordinal=ordinal,
                    )

    def observe_output_item(self, item: dict | None) -> None:
        if not isinstance(item, dict):
            return
        item_id = _normalized_non_empty_string(item.get("id"))
        if not item_id:
            return

        item_type = str(item.get("type", "")).lower()
        with self._lock:
            if item_type in {"function_call", "function_call_output"}:
                call_id = _normalized_non_empty_string(item.get("call_id"))
                if call_id:
                    self._remember_function_id_locked(call_id)
                return

            if item_type == "reasoning":
                fingerprint = _reasoning_fingerprint(item)
                if fingerprint is not None:
                    self._remember_reasoning_id_locked(fingerprint, item_id)
                return

            fingerprint = _assistant_message_fingerprint(item)
            if fingerprint is not None:
                if not _assistant_message_has_replay_content(item):
                    return
                observed_key = (item_id, fingerprint)
                if observed_key in self._assistant_observed_item_fingerprints:
                    return
                self._assistant_observed_item_fingerprints.add(observed_key)
                self._remember_assistant_message_id_locked(fingerprint, item_id)

    def observe_response_payload(self, payload: dict | None) -> None:
        if not isinstance(payload, dict):
            return
        output = payload.get("output")
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    self.observe_output_item(item)

    def _repair_item_id_locked(
        self,
        item: dict,
        *,
        assistant_ordinals: dict[str, int],
    ) -> str | None:
        item_type = str(item.get("type", "")).lower()
        if item_type in {"function_call", "function_call_output"}:
            call_id = _normalized_non_empty_string(item.get("call_id"))
            if not call_id:
                return None
            return _function_item_id(call_id)

        if item_type == "reasoning":
            fingerprint = _reasoning_fingerprint(item)
            if fingerprint is None:
                return None
            return self._reasoning_item_ids.get(fingerprint)

        fingerprint = _assistant_message_fingerprint(item)
        if fingerprint is None:
            return None
        ordinal = assistant_ordinals.get(fingerprint, 0)
        assistant_ordinals[fingerprint] = ordinal + 1
        return self._assistant_message_item_ids.get(self._assistant_key(fingerprint, ordinal))

    def repair_missing_replay_ids(self, body: dict) -> tuple[dict, dict | None]:
        if not isinstance(body, dict):
            return body, None
        input_items = body.get("input")
        if not isinstance(input_items, list):
            return body, None

        self.observe_input_items(input_items)

        repaired_items = []
        changed = False
        repaired_counts: dict[str, int] = {}
        assistant_ordinals: dict[str, int] = {}

        with self._lock:
            for item in input_items:
                if not isinstance(item, dict):
                    repaired_items.append(item)
                    continue

                item_id = _normalized_non_empty_string(item.get("id"))
                item_type = str(item.get("type", "")).lower()
                if item_type in {"function_call", "function_call_output"}:
                    call_id = _normalized_non_empty_string(item.get("call_id"))
                    if call_id:
                        canonical_id = _function_item_id(call_id)
                        if item_id != canonical_id:
                            item = {**item, "id": canonical_id}
                            item_id = canonical_id
                            item_type_label = item_type or "unknown"
                            repaired_counts[item_type_label] = repaired_counts.get(item_type_label, 0) + 1
                            changed = True
                        self._remember_function_id_locked(call_id)
                        repaired_items.append(item)
                        continue
                if item_id:
                    fingerprint = _assistant_message_fingerprint(item)
                    if fingerprint is not None:
                        assistant_ordinals[fingerprint] = assistant_ordinals.get(fingerprint, 0) + 1
                    repaired_items.append(item)
                    continue

                repaired_id = self._repair_item_id_locked(
                    item,
                    assistant_ordinals=assistant_ordinals,
                )
                if repaired_id:
                    repaired_item = {**item, "id": repaired_id}
                    repaired_items.append(repaired_item)
                    item_type = str(item.get("type", "")).lower() or "unknown"
                    repaired_counts[item_type] = repaired_counts.get(item_type, 0) + 1
                    changed = True
                    continue

                repaired_items.append(item)

        if not changed:
            return body, None

        repaired_body = {**body, "input": repaired_items}
        return repaired_body, {
            "input_items": len(input_items),
            "repaired_items": sum(repaired_counts.values()),
            "repaired_by_type": repaired_counts,
        }


_states: OrderedDict[str, ReplayIdState] = OrderedDict()
_states_lock = threading.Lock()


def state_for_lineage_key(lineage_key: str | None) -> ReplayIdState | None:
    if not lineage_key:
        return None
    with _states_lock:
        state = _states.get(lineage_key)
        if state is None:
            state = ReplayIdState()
            _states[lineage_key] = state
        else:
            _states.move_to_end(lineage_key)
        while len(_states) > _MAX_LINEAGE_STATES:
            _states.popitem(last=False)
        return state


def state_for_body(
    body: dict | None,
    headers=None,
    *,
    subagent: str | None = None,
) -> tuple[str | None, ReplayIdState | None]:
    lineage_key = lineage_key_for_body(body, headers=headers, subagent=subagent)
    return lineage_key, state_for_lineage_key(lineage_key)


def repair_missing_replay_ids(
    body: dict,
    headers=None,
    *,
    subagent: str | None = None,
) -> tuple[dict, dict | None]:
    lineage_key, state = state_for_body(body, headers=headers, subagent=subagent)
    if state is None:
        return body, None
    repaired_body, trace = state.repair_missing_replay_ids(body)
    if isinstance(trace, dict):
        trace["lineage_key_kind"] = lineage_key.split(":", 1)[0] if lineage_key else None
        trace["lineage_key_sha256"] = _sha256_text(lineage_key) if lineage_key else None
    return repaired_body, trace
