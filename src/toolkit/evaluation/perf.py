"""Performance checks against the non-functional targets.

* Assignment: p95 per-item latency, embedding included, against N open clusters (target < 50 ms
  with 1,000 clusters).
* Discovery: wall time to cluster an N-item buffer (target < 2 minutes for 20,000 items).
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from toolkit.clock import SimulatedClock
from toolkit.config import Config
from toolkit.demo.topics import PREFIXES, SLOTS, SUFFIXES, TOPICS
from toolkit.embedders import Embedder, normalize
from toolkit.engine.core import ClusteringEngine
from toolkit.models import Cluster, ClusterStatus, Exemplar, Item, ItemStatus, StoredItem
from toolkit.store import MemoryStore

START = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class PerfReport:
    model_id: str
    open_clusters: int
    assign_samples: int
    assign_p50_ms: float
    assign_p95_ms: float
    discovery_items: int
    discovery_seconds: float | None
    discovery_clusters: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def synthetic_texts(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    topics = list(TOPICS.values())
    out = []
    for _ in range(n):
        template = rng.choice(rng.choice(topics))
        filled = template.format(**{k: rng.choice(v) for k, v in SLOTS.items()})
        out.append(f"{rng.choice(PREFIXES)}{filled}{rng.choice(SUFFIXES)} #{rng.randint(1, 9999)}")
    return out


def _seed_clusters(engine: ClusteringEngine, n_clusters: int, dim: int, per: int) -> None:
    rng = np.random.default_rng(0)
    store = engine.store
    for c in range(n_clusters):
        center = normalize(rng.normal(size=(1, dim)))[0]
        vectors = normalize(center + 0.3 * normalize(rng.normal(size=(per, dim))))
        cluster = Cluster(
            id=f"cl_perf{c:05d}",
            status=ClusterStatus.OPEN,
            threshold=0.9,
            size=per,
            first_seen=START,
            last_seen=START,
            opened_at=START,
            seen_count=per,
        )
        store.add_cluster(cluster, [Exemplar(f"ex{c}_{i}", v) for i, v in enumerate(vectors)])
    engine.ctx.invalidate_index()


def measure_assignment(
    embedder: Embedder, config: Config, n_clusters: int = 1000, samples: int = 200
) -> tuple[float, float]:
    clock = SimulatedClock(START)
    engine = ClusteringEngine(config, store=MemoryStore(), embedder=embedder, clock=clock)
    dim = embedder.embed(["warm up"]).shape[1]
    _seed_clusters(engine, n_clusters, dim, config.exemplars_per_cluster)
    texts = synthetic_texts(samples, seed=1)
    timings = []
    for n, text in enumerate(texts):
        item = Item(f"perf_{n}", text, START + timedelta(seconds=n))
        started = time.perf_counter()
        engine.ingest([item])
        timings.append((time.perf_counter() - started) * 1000)
    return float(np.percentile(timings, 50)), float(np.percentile(timings, 95))


def measure_discovery(embedder: Embedder, config: Config, n_items: int) -> tuple[float, int]:
    clock = SimulatedClock(START)
    engine = ClusteringEngine(config, store=MemoryStore(), embedder=embedder, clock=clock)
    texts = synthetic_texts(n_items, seed=2)
    vectors = np.concatenate([embedder.embed(texts[i : i + 1024]) for i in range(0, n_items, 1024)])
    engine.store.add_items(
        [
            StoredItem(
                id=f"d_{i}",
                text=t,
                timestamp=START + timedelta(seconds=i),
                metadata={},
                embedding=v,
                model_id=embedder.model_id,
                status=ItemStatus.BUFFERED,
                cluster_id=None,
                seq=0,
            )
            for i, (t, v) in enumerate(zip(texts, vectors, strict=True))
        ]
    )
    clock.advance_to(START + timedelta(seconds=n_items))
    started = time.perf_counter()
    report = engine.discover()
    return time.perf_counter() - started, len(report.opened)


def run_perf(
    embedder: Embedder,
    config: Config,
    n_clusters: int = 1000,
    samples: int = 200,
    discovery_items: int = 20000,
) -> PerfReport:
    p50, p95 = measure_assignment(embedder, config, n_clusters, samples)
    seconds, opened = (None, None)
    if discovery_items:
        seconds, opened = measure_discovery(embedder, config, discovery_items)
    return PerfReport(
        model_id=embedder.model_id,
        open_clusters=n_clusters,
        assign_samples=samples,
        assign_p50_ms=round(p50, 2),
        assign_p95_ms=round(p95, 2),
        discovery_items=discovery_items,
        discovery_seconds=round(seconds, 1) if seconds is not None else None,
        discovery_clusters=opened,
    )
