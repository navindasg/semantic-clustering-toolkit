"""Representative samples of a cluster and its activity over time."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from enum import StrEnum

import numpy as np

from toolkit.engine.context import EngineContext
from toolkit.models import StoredItem

CENTRAL_POOL = 5000


class SampleStrategy(StrEnum):
    CENTRAL = "central"
    RANDOM = "random"
    RECENT = "recent"
    MIXED = "mixed"


def _central(members: list[StoredItem], n: int) -> list[StoredItem]:
    pool = members[:CENTRAL_POOL]
    vectors = np.stack([m.embedding for m in pool])
    centroid = vectors.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    order = np.argsort(-(vectors @ centroid), kind="stable")
    return [pool[i] for i in order[:n]]


def _random(members: list[StoredItem], n: int, rng: np.random.Generator) -> list[StoredItem]:
    ordered = sorted(members, key=lambda m: m.id)
    if len(ordered) <= n:
        return ordered
    picked = rng.choice(len(ordered), n, replace=False)
    return [ordered[i] for i in sorted(picked)]


def sample_items(
    members: list[StoredItem], n: int, strategy: SampleStrategy | str, rng: np.random.Generator
) -> list[StoredItem]:
    """Pick `n` representative members (`members` newest first)."""
    if n <= 0:
        raise ValueError("n must be positive")
    strategy = SampleStrategy(strategy)
    if not members:
        return []
    if strategy == SampleStrategy.RECENT:
        return members[:n]
    if strategy == SampleStrategy.CENTRAL:
        return _central(members, n)
    if strategy == SampleStrategy.RANDOM:
        return _random(members, n, rng)
    picked: dict[str, StoredItem] = {}
    third = max(1, n // 3)
    for group in (_central(members, third), members[:third], _random(members, n, rng)):
        for item in group:
            if len(picked) < n:
                picked.setdefault(item.id, item)
    return list(picked.values())


def sample_cluster(
    ctx: EngineContext, cluster_id: str, n: int = 10, strategy: SampleStrategy | str = "mixed"
) -> list[StoredItem]:
    if ctx.store.get_cluster(cluster_id) is None:
        raise KeyError(f"unknown cluster {cluster_id}")
    members = ctx.store.cluster_items(cluster_id)  # newest first
    return sample_items(members, n, strategy, ctx.rng("sample", cluster_id, len(members)))


def counts_over_time(
    members: list[StoredItem], bucket: timedelta | None = None
) -> list[tuple[datetime, int]]:
    """Item counts per time bucket (hourly for spans under two days, otherwise daily)."""
    if not members:
        return []
    start = min(m.timestamp for m in members)
    end = max(m.timestamp for m in members)
    if bucket is None:
        bucket = timedelta(hours=1) if end - start < timedelta(days=2) else timedelta(days=1)
    seconds = bucket.total_seconds()
    origin = start.replace(minute=0, second=0, microsecond=0)
    if bucket >= timedelta(days=1):
        origin = origin.replace(hour=0)
    counts = Counter(int((m.timestamp - origin).total_seconds() // seconds) for m in members)
    last = max(counts)
    return [(origin + bucket * k, counts.get(k, 0)) for k in range(last + 1)]
