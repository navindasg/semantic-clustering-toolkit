"""Wire an engine from a Config: store, embedder, clock and event sinks."""

from __future__ import annotations

import logging

from toolkit.clock import Clock, SystemClock
from toolkit.config import Config
from toolkit.embedders import Embedder
from toolkit.engine.core import ClusteringEngine
from toolkit.events import EventBus, EventSink, TerminalSink, WebhookSink
from toolkit.store import open_store


def build_bus(config: Config, terminal: bool = False, extra: list[EventSink] | None = None) -> EventBus:
    sinks: list[EventSink] = [WebhookSink.from_config(w) for w in config.webhooks]
    if terminal:
        sinks.append(TerminalSink())
    return EventBus([*sinks, *(extra or [])])


def build_engine(
    config: Config,
    *,
    clock: Clock | None = None,
    terminal: bool = False,
    embedder: Embedder | None = None,
) -> ClusteringEngine:
    return ClusteringEngine(
        config,
        store=open_store(config.store),
        embedder=embedder,
        clock=clock or SystemClock(),
        bus=build_bus(config, terminal=terminal),
    )


def configure_logging(level: str = "WARNING") -> None:
    """Structured key=value logs on stderr."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format="ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s",
    )
