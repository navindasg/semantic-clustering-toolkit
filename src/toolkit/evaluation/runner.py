"""Labeled replay: run a labeled file through the same engine and score the result."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from toolkit.clock import SimulatedClock
from toolkit.config import Config
from toolkit.demo.replay import ReplayStats, replay
from toolkit.embedders import Embedder, make_embedder
from toolkit.engine.core import ClusteringEngine
from toolkit.evaluation.metrics import DEFAULT_NOISE_LABEL, Metrics, compute_metrics
from toolkit.events.bus import EventBus
from toolkit.io import LabeledItem
from toolkit.store import open_store


@dataclass(frozen=True)
class EvalResult:
    metrics: Metrics
    stats: ReplayStats
    engine: ClusteringEngine


def default_drain(config: Config) -> timedelta:
    """Long enough after the last item for quiet clusters to close."""
    return config.close_after + config.sweep_interval * 2


def run_eval(
    config: Config,
    rows: Sequence[LabeledItem],
    *,
    embedder: Embedder | None = None,
    store_url: str = "memory://",
    bus: EventBus | None = None,
    drain: timedelta | None = None,
    speed: float | None = None,
    noise_label: str = DEFAULT_NOISE_LABEL,
) -> EvalResult:
    labeled = [r for r in rows if r.label is not None]
    if not labeled:
        raise ValueError("evaluation needs ground-truth labels; none found in the data")
    clock = SimulatedClock()
    engine = ClusteringEngine(
        config,
        store=open_store(store_url),
        embedder=embedder or make_embedder(config.embedder, config.model_dir, config.embed_batch_size),
        clock=clock,
        bus=bus,
    )
    stats = replay(
        engine,
        clock,
        [r.item for r in rows],
        drain=drain if drain is not None else default_drain(config),
        speed=speed,
    )
    metrics = compute_metrics(
        engine.store,
        {r.item.id: str(r.label) for r in labeled},
        config.close_after,
        noise_label=noise_label,
        items_per_second=stats.items_per_second,
        latencies_seconds=list(stats.batch_latencies),
    )
    return EvalResult(metrics, stats, engine)
