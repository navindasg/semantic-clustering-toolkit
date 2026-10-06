"""Clock interface. Every lifecycle decision reads time from a Clock, never from the OS directly."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


def ensure_utc(value: datetime) -> datetime:
    """Treat naive datetimes as UTC and convert aware ones to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    """Wall-clock time in UTC, for live runs."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class SimulatedClock:
    """Event-time clock for replay. Time only moves when the replay advances it."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = ensure_utc(start) if start else datetime(1970, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance_to(self, moment: datetime) -> None:
        """Move forward to `moment`. Moving backward is ignored so time is monotonic."""
        moment = ensure_utc(moment)
        if moment > self._now:
            self._now = moment
