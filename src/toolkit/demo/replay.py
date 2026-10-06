"""Replay a stream in event-time order on a simulated clock.

Scheduled jobs fire at their simulated due times between batches, so the event log depends
only on the data, config and seed — never on replay speed or machine load.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from toolkit.clock import SimulatedClock
from toolkit.engine.core import ClusteringEngine
from toolkit.models import AssignmentResult, Item
from toolkit.scheduler import Scheduler

MAX_SLEEP_SECONDS = 2.0


@dataclass(frozen=True)
class ReplayStats:
    items: int
    batches: int
    ingest_seconds: float
    discoveries: int
    sweeps: int
    start: datetime | None
    end: datetime | None
    batch_latencies: tuple[float, ...] = ()

    @property
    def items_per_second(self) -> float:
        return self.items / self.ingest_seconds if self.ingest_seconds else 0.0


def make_batches(items: Sequence[Item], window: timedelta, max_batch: int) -> list[list[Item]]:
    """Group time-ordered items into batches spanning at most `window` and `max_batch` items."""
    batches: list[list[Item]] = []
    current: list[Item] = []
    for item in items:
        if current and (item.timestamp - current[0].timestamp >= window or len(current) >= max_batch):
            batches.append(current)
            current = []
        current.append(item)
    if current:
        batches.append(current)
    return batches


def _run_until(scheduler: Scheduler, clock: SimulatedClock, moment: datetime) -> None:
    while scheduler.next_due() <= moment:
        due = scheduler.next_due()
        clock.advance_to(due)
        scheduler.run_due(due)


def replay(
    engine: ClusteringEngine,
    clock: SimulatedClock,
    items: Sequence[Item],
    *,
    batch_window: timedelta = timedelta(minutes=5),
    max_batch: int = 256,
    speed: float | None = None,
    drain: timedelta | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_batch: Callable[[list[Item], list[AssignmentResult]], None] | None = None,
) -> ReplayStats:
    """Feed `items` through the engine. `speed` is simulated seconds per real second (None = max).

    `drain` keeps running scheduled jobs for that long after the last item so quiet clusters
    get to close; the drain is never paced.
    """
    ordered = sorted(items, key=lambda i: (i.timestamp, i.id))
    if not ordered:
        return ReplayStats(0, 0, 0.0, 0, 0, None, None)
    clock.advance_to(ordered[0].timestamp)
    scheduler = Scheduler(engine)
    scheduler.start(clock.now())
    ingest_seconds, latencies = 0.0, []
    previous = ordered[0].timestamp
    batches = make_batches(ordered, batch_window, max_batch)
    for batch in batches:
        _run_until(scheduler, clock, batch[0].timestamp)
        last = batch[-1].timestamp
        if speed:
            sleep(min((last - previous).total_seconds() / speed, MAX_SLEEP_SECONDS))
        previous = last
        clock.advance_to(last)
        started = time.perf_counter()
        results = engine.ingest(batch)
        elapsed = time.perf_counter() - started
        ingest_seconds += elapsed
        latencies.append(elapsed / len(batch))
        scheduler.check_buffer(clock.now())
        if on_batch is not None:
            on_batch(batch, results)
    if drain:
        _run_until(scheduler, clock, ordered[-1].timestamp + drain)
    return ReplayStats(
        items=len(ordered),
        batches=len(batches),
        ingest_seconds=ingest_seconds,
        discoveries=scheduler.counters.discoveries,
        sweeps=scheduler.counters.sweeps,
        start=ordered[0].timestamp,
        end=clock.now(),
        batch_latencies=tuple(latencies),
    )
