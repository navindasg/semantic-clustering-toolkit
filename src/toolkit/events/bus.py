"""In-process event bus. Sinks never break the engine: their errors are logged and dropped."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Protocol

from toolkit.models import Event, EventType

logger = logging.getLogger("toolkit.events")


class EventSink(Protocol):
    def handle(self, event: Event) -> None: ...


class CallbackSink:
    """Calls `callback(event)` for every event, optionally filtered by type."""

    def __init__(
        self, callback: Callable[[Event], None], types: Iterable[EventType | str] | None = None
    ) -> None:
        self._callback = callback
        self._types = {str(t) for t in types} if types is not None else None

    def handle(self, event: Event) -> None:
        if self._types is None or str(event.type) in self._types:
            self._callback(event)


class EventBus:
    def __init__(self, sinks: Iterable[EventSink] = ()) -> None:
        self._sinks: list[EventSink] = list(sinks)

    def subscribe(self, sink: EventSink) -> EventSink:
        self._sinks = [*self._sinks, sink]
        return sink

    def on(
        self, callback: Callable[[Event], None], types: Iterable[EventType | str] | None = None
    ) -> EventSink:
        """Register a plain callback; returns the sink so it can be unsubscribed."""
        return self.subscribe(CallbackSink(callback, types))

    def unsubscribe(self, sink: EventSink) -> None:
        self._sinks = [s for s in self._sinks if s is not sink]

    def publish(self, events: Iterable[Event]) -> None:
        for event in events:
            for sink in self._sinks:
                try:
                    sink.handle(event)
                except Exception:
                    logger.exception("event sink %r failed on %s", sink, event.type)

    def close(self) -> None:
        for sink in self._sinks:
            closer = getattr(sink, "close", None)
            if callable(closer):
                closer()
