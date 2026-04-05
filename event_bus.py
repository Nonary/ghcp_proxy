"""Minimal in-process event bus for decoupled runtime notifications."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Callable


class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list[Callable[..., None]]] = defaultdict(list)
        self._lock = Lock()

    def subscribe(self, event_name: str, handler: Callable[..., None]):
        if not isinstance(event_name, str) or not event_name:
            raise ValueError("event_name must be a non-empty string")
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            self._subscribers[event_name].append(handler)

    def publish(self, event_name: str, *args, **kwargs):
        with self._lock:
            subscribers = list(self._subscribers.get(event_name, ()))
        for handler in subscribers:
            handler(*args, **kwargs)
