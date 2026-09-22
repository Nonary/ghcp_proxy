"""Keep Copilot SDK sessions' encrypted reasoning across disk resumes.

The Copilot runtime persists a session's visible history but not the
encrypted reasoning items the model produced.  A session that is resumed from
disk -- after a proxy restart, an interrupted or failed turn, idle expiry,
capacity eviction, a configuration change, or a pending tool call abandoned by
a new user message -- therefore resends its history without them.  The
upstream prompt cache breaks at the first missing item, so the next request
re-bills everything after the conversation's first reasoning step.

The proxy sees every model call through the SDK request handler.  This ledger
records each reasoning run together with the items on either side of it and
puts the run back into a full-history request that is missing it, so a
resumed session sends the same prefix a live session would.

Sol reaches Copilot over a WebSocket that chains ``previous_response_id``;
its requests carry only new items, so its reasoning is recorded from response
output.  HTTP models (Luna) replay reasoning inside every request, so their
runs are recorded from request input.  Runs are keyed by model because
encrypted reasoning is only valid for the model that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

# Reasoning fields the runtime replays; anything else in a response item
# (status, etc.) is response metadata, not part of the replayed input item.
_REPLAYED_REASONING_FIELDS = ("content", "encrypted_content", "id", "summary", "type")
_CALL_ITEM_TYPES = {
    "function_call",
    "function_call_output",
    "custom_tool_call",
    "custom_tool_call_output",
}
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

Anchor = tuple[str, str]


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _message_texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    texts = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return texts


def anchor_key(item: Any) -> str | None:
    """Identify a history item the same way before and after a disk resume.

    Response and replayed items differ in server ids and status fields, so
    calls are keyed by ``call_id`` and messages by role and text.
    """
    if not isinstance(item, dict):
        return None
    item_type = item.get("type") or ("message" if "role" in item else None)
    if item_type == "reasoning" or not isinstance(item_type, str):
        return None
    if item_type in _CALL_ITEM_TYPES:
        call_id = item.get("call_id")
        return f"{item_type}:{call_id}" if isinstance(call_id, str) and call_id else None
    if item_type == "message":
        return f"message:{item.get('role')}:{_digest(_message_texts(item.get('content')))}"
    stable = {key: value for key, value in item.items() if key not in {"id", "status"}}
    return f"{item_type}:{_digest(stable)}"


def _replayable(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "reasoning"
        and isinstance(item.get("encrypted_content"), str)
        and bool(item["encrypted_content"])
    )


def _is_reasoning(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "reasoning"


def _replayed_form(item: dict) -> dict:
    return {key: item[key] for key in _REPLAYED_REASONING_FIELDS if key in item}


@dataclass
class _SessionRuns:
    runs: dict[str, dict[Anchor, list[dict]]] = field(default_factory=dict)
    size: int = 0
    file_size: int = 0


class ReasoningLedger:
    """Per-session reasoning runs, cached in memory and appended to disk."""

    def __init__(
        self,
        directory: Callable[[], str],
        *,
        memory_budget_bytes: int = 128 * 1024 * 1024,
        file_budget_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self._directory = directory
        self._memory_budget = memory_budget_bytes
        self._file_budget = file_budget_bytes
        self._sessions: OrderedDict[str, _SessionRuns] = OrderedDict()
        self._memory_size = 0
        self._disabled: set[tuple[str, str]] = set()

    # -- storage ---------------------------------------------------------

    def _path(self, session_id: str) -> str | None:
        if not _SAFE_SESSION_ID.match(session_id):
            return None
        return os.path.join(self._directory(), f"{session_id}.jsonl")

    def _session(self, session_id: str) -> _SessionRuns:
        entry = self._sessions.get(session_id)
        if entry is not None:
            self._sessions.move_to_end(session_id)
            return entry
        entry = _SessionRuns()
        path = self._path(session_id)
        if path is not None:
            try:
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        entry.file_size += len(line.encode("utf-8"))
                        try:
                            record = json.loads(line)
                            model, before, after, run = record["m"], record["b"], record["a"], record["r"]
                        except (ValueError, KeyError, TypeError):
                            continue
                        if isinstance(run, list) and all(_replayable(item) for item in run):
                            self._remember(entry, model, (before, after), run)
            except OSError:
                pass
        self._sessions[session_id] = entry
        self._memory_size += entry.size
        self._trim_memory(keep=session_id)
        return entry

    def _remember(self, entry: _SessionRuns, model: str, anchor: Anchor, run: list[dict]) -> bool:
        runs = entry.runs.setdefault(model, {})
        if runs.get(anchor) == run:
            return False
        size = len(json.dumps(run, separators=(",", ":")))
        if anchor in runs:
            size -= len(json.dumps(runs[anchor], separators=(",", ":")))
        runs[anchor] = run
        entry.size += size
        return True

    def _trim_memory(self, *, keep: str) -> None:
        while self._memory_size > self._memory_budget and len(self._sessions) > 1:
            session_id, entry = next(iter(self._sessions.items()))
            if session_id == keep:
                self._sessions.move_to_end(session_id)
                continue
            del self._sessions[session_id]
            self._memory_size -= entry.size

    def _store(self, session_id: str, model: str, anchor: Anchor, run: list[dict]) -> None:
        entry = self._session(session_id)
        before = entry.size
        if not self._remember(entry, model, anchor, run):
            return
        self._memory_size += entry.size - before
        path = self._path(session_id)
        if path is not None:
            line = json.dumps({"m": model, "b": anchor[0], "a": anchor[1], "r": run}, separators=(",", ":")) + "\n"
            encoded = len(line.encode("utf-8"))
            if entry.file_size + encoded <= self._file_budget:
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                        handle.write(line)
                    entry.file_size += encoded
                except OSError:
                    pass
        self._trim_memory(keep=session_id)

    # -- recording -------------------------------------------------------

    def record_input(self, session_id: str | None, model: Any, items: Any) -> None:
        """Record reasoning runs a full-history request carries."""
        if not session_id or not isinstance(model, str) or not isinstance(items, list):
            return
        before: str | None = None
        run: list[dict] = []
        for item in items:
            if _is_reasoning(item):
                if _replayable(item):
                    run.append(copy.deepcopy(item))
                continue
            after = anchor_key(item)
            if run and before and after:
                self._store(session_id, model, (before, after), run)
            run = []
            before = after

    def record_output(self, session_id: str | None, model: Any, before: str | None, output: Any) -> None:
        """Record reasoning runs from one response's output.

        ``before`` identifies the last input item of the request that produced
        the response: in the full history the output directly follows it.
        """
        if not session_id or not isinstance(model, str) or not isinstance(output, list):
            return
        run: list[dict] = []
        for item in output:
            if _is_reasoning(item):
                if _replayable(item):
                    run.append(_replayed_form(item))
                continue
            after = anchor_key(item)
            if run and before and after:
                self._store(session_id, model, (before, after), run)
            run = []
            before = after

    # -- restoration -----------------------------------------------------

    def restore(self, session_id: str | None, model: Any, items: Any) -> tuple[Any, int]:
        """Return ``items`` with recorded reasoning runs reinserted."""
        if (
            not session_id
            or not isinstance(model, str)
            or not isinstance(items, list)
            or (session_id, model) in self._disabled
        ):
            return items, 0
        entry = self._session(session_id)
        runs = entry.runs.get(model)
        if not runs:
            return items, 0
        present = {item.get("id") for item in items if _is_reasoning(item)}
        restored: list = []
        count = 0
        previous: Any = None
        for item in items:
            if previous is not None and not _is_reasoning(previous) and not _is_reasoning(item):
                run = runs.get((anchor_key(previous), anchor_key(item)))
                if run and not any(reasoning.get("id") in present for reasoning in run):
                    restored.extend(copy.deepcopy(run))
                    count += len(run)
            restored.append(item)
            previous = item
        return (restored, count) if count else (items, 0)

    def disable(self, session_id: str | None, model: Any) -> None:
        """Stop restoring for a session/model whose restored request was rejected."""
        if session_id and isinstance(model, str):
            self._disabled.add((session_id, model))

    def prune_older_than(self, cutoff: float) -> None:
        """Delete ledger files not written since ``cutoff`` (a Unix time)."""
        directory = self._directory()
        try:
            names = os.listdir(directory)
        except OSError:
            return
        stale = set()
        for name in names:
            if not name.endswith(".jsonl"):
                continue
            try:
                if os.path.getmtime(os.path.join(directory, name)) < cutoff:
                    stale.add(name[: -len(".jsonl")])
            except OSError:
                continue
        self.forget(stale)

    def forget(self, session_ids: set[str]) -> None:
        for session_id in session_ids:
            entry = self._sessions.pop(session_id, None)
            if entry is not None:
                self._memory_size -= entry.size
            self._disabled = {key for key in self._disabled if key[0] != session_id}
            path = self._path(session_id)
            if path is not None:
                try:
                    os.unlink(path)
                except OSError:
                    pass
