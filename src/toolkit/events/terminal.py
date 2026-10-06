"""Terminal event log: prints lifecycle events as they happen."""

from __future__ import annotations

import sys
from typing import TextIO

from toolkit.models import Event, EventType

_DEFAULT_TYPES = frozenset(
    {
        EventType.CLUSTER_OPENED,
        EventType.CLUSTER_CLOSED,
        EventType.CLUSTER_REOPENED,
        EventType.CLUSTER_MERGED,
        EventType.CLUSTER_ARCHIVED,
        EventType.CLUSTER_SPLIT,
    }
)
_COLORS = {
    EventType.CLUSTER_OPENED: "\033[32m",
    EventType.CLUSTER_CLOSED: "\033[33m",
    EventType.CLUSTER_REOPENED: "\033[36m",
    EventType.CLUSTER_MERGED: "\033[35m",
    EventType.CLUSTER_ARCHIVED: "\033[90m",
    EventType.CLUSTER_SPLIT: "\033[34m",
}
_RESET = "\033[0m"


def describe(event: Event) -> str:
    data = event.data
    label = data.get("label") or ""
    match event.type:
        case EventType.CLUSTER_OPENED:
            detail = f"{data.get('size', '?')} items  {label}"
        case EventType.CLUSTER_MERGED:
            detail = f"into {data.get('survivor')}  ({data.get('similarity', 0):.2f})"
        case EventType.CLUSTER_CLOSED | EventType.CLUSTER_REOPENED | EventType.CLUSTER_ARCHIVED:
            detail = f"size {data.get('size', '?')}  {label}"
        case EventType.ITEM_ASSIGNED:
            detail = f"{event.item_id} score {data.get('score', 0):.2f}"
        case EventType.ITEM_ORPHANED:
            detail = f"{event.item_id}"
        case _:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(data.items()))
    return f"{event.timestamp:%Y-%m-%d %H:%M}  {event.type:<17} {event.cluster_id or '':<15} {detail}"


class TerminalSink:
    def __init__(
        self,
        stream: TextIO | None = None,
        types: frozenset[EventType] = _DEFAULT_TYPES,
        color: bool | None = None,
    ) -> None:
        self._stream = stream or sys.stdout
        self._types = types
        self._color = self._stream.isatty() if color is None else color

    def handle(self, event: Event) -> None:
        if event.type not in self._types:
            return
        line = describe(event)
        if self._color:
            line = f"{_COLORS.get(event.type, '')}{line}{_RESET}"
        print(line, file=self._stream, flush=True)
