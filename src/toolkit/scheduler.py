"""In-process scheduler for discovery and sweeps, driven by the engine's clock.

Discovery fires every `discovery_interval` or as soon as the buffer reaches
`discovery_min_buffer`, whichever comes first. Sweeps fire every `sweep_interval`.
In production, cron or a worker queue can call `engine.discover()` / `engine.sweep()` instead.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime

from toolkit.engine.core import ClusteringEngine
from toolkit.models import DiscoveryReport, SweepReport

logger = logging.getLogger("toolkit.scheduler")


@dataclass
class JobCounters:
    discoveries: int = 0
    sweeps: int = 0
    discovery_reports: list[DiscoveryReport] = field(default_factory=list)
    sweep_reports: list[SweepReport] = field(default_factory=list)


class Scheduler:
    def __init__(self, engine: ClusteringEngine) -> None:
        self.engine = engine
        cfg = engine.config
        self._discovery_interval = cfg.discovery_interval
        self._sweep_interval = cfg.sweep_interval
        self._min_buffer = cfg.discovery_min_buffer
        self.next_discovery: datetime | None = None
        self.next_sweep: datetime | None = None
        self.counters = JobCounters()
        self._residual = 0

    def start(self, now: datetime) -> None:
        self.next_discovery = now + self._discovery_interval
        self.next_sweep = now + self._sweep_interval

    def next_due(self) -> datetime:
        if self.next_discovery is None or self.next_sweep is None:
            raise RuntimeError("scheduler not started")
        return min(self.next_discovery, self.next_sweep)

    def _discover(self, now: datetime) -> None:
        report = self.engine.discover()
        self.counters.discoveries += 1
        self.counters.discovery_reports.append(report)
        self.next_discovery = now + self._discovery_interval
        self._residual = self.engine.buffer_size()

    def _sweep(self, now: datetime) -> None:
        report = self.engine.sweep()
        self.counters.sweeps += 1
        self.counters.sweep_reports.append(report)
        self.next_sweep = now + self._sweep_interval

    def run_due(self, now: datetime) -> None:
        """Run every job due at `now` (discovery before sweep, so new clusters get swept)."""
        if self.next_discovery is None or self.next_sweep is None:
            self.start(now)
        assert self.next_discovery is not None and self.next_sweep is not None
        if now >= self.next_discovery:
            self._discover(now)
        if now >= self.next_sweep:
            self._sweep(now)

    def check_buffer(self, now: datetime) -> bool:
        """Size-based trigger: run discovery early when the buffer is full enough.

        Items discovery left behind as noise don't count twice: after a run, the buffer must
        grow by half of `discovery_min_buffer` again before the size trigger fires.
        """
        size = self.engine.buffer_size()
        if size >= self._min_buffer and size >= self._residual + self._min_buffer // 2:
            self._discover(now)
            return True
        return False


class BackgroundScheduler:
    """Live mode: a daemon thread runs due jobs against the engine's (system) clock."""

    def __init__(self, engine: ClusteringEngine, poll_seconds: float = 1.0) -> None:
        self.scheduler = Scheduler(engine)
        self._engine = engine
        self._poll = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.scheduler.start(self._engine.clock.now())
        self._thread = threading.Thread(target=self._loop, name="toolkit-scheduler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = self._engine.clock.now()
            try:
                self.scheduler.run_due(now)
                self.scheduler.check_buffer(now)
            except Exception:
                logger.exception("scheduled job failed; will retry on the next tick")
            self._stop.wait(self._poll)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
            self._thread = None
