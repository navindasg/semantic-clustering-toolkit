"""Shared engine state: dependencies, the unit of work, event recording and the exemplar index cache."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from toolkit.clock import Clock
from toolkit.config import Config
from toolkit.embedders.base import Embedder
from toolkit.engine.scoring import ExemplarIndex, StoreExemplarIndex, make_rng
from toolkit.events.bus import EventBus
from toolkit.labeling.base import Labeler
from toolkit.models import Cluster, ClusterStatus, Event, EventType, Exemplar
from toolkit.store.base import Store

INDEX_VERSION_KEY = "index_version"


def cluster_id_for(member_ids: list[str]) -> str:
    """Permanent, content-derived cluster ID: the same founding members give the same ID."""
    digest = hashlib.blake2b("\n".join(sorted(member_ids)).encode(), digest_size=6).hexdigest()
    return f"cl_{digest}"


@dataclass
class _IndexCache:
    version: str
    valid_until: datetime | None
    index: ExemplarIndex | StoreExemplarIndex
    thresholds: np.ndarray
    clusters: dict[str, Cluster] = field(default_factory=dict)


@dataclass
class EngineContext:
    config: Config
    store: Store
    embedder: Embedder
    clock: Clock
    bus: EventBus
    labeler: Labeler
    _pending: list[Event] = field(default_factory=list)
    _depth: int = 0
    _cache: _IndexCache | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # --- unit of work ----------------------------------------------------
    @contextmanager
    def unit_of_work(self) -> Iterator[None]:
        """One store transaction; recorded events are stored inside it and published after commit."""
        with self._lock:
            outer = self._depth == 0
            self._depth += 1
            stored: list[Event] = []
            try:
                with self.store.transaction():
                    yield
                    if outer and self._pending:
                        stored = self.store.append_events(self._pending)
            except BaseException:
                if outer:
                    self._pending = []
                    self._cache = None
                raise
            finally:
                self._depth -= 1
            if outer:
                self._pending = []
                self.bus.publish(stored)

    def emit(
        self,
        type: EventType,
        cluster_id: str | None = None,
        item_id: str | None = None,
        **data: Any,
    ) -> None:
        self._pending.append(
            Event(
                seq=0,
                type=type,
                timestamp=self.clock.now(),
                cluster_id=cluster_id,
                item_id=item_id,
                data=data,
            )
        )

    # --- randomness ------------------------------------------------------
    def rng(self, *parts: object) -> np.random.Generator:
        return make_rng(self.config.random_seed, *parts)

    def sample_exemplars(self, member_ids: list[str], vectors: np.ndarray, salt: object) -> list[Exemplar]:
        size = self.config.exemplars_per_cluster
        order = sorted(range(len(member_ids)), key=lambda i: member_ids[i])
        if len(order) > size:
            picked = self.rng("exemplars", salt).choice(len(order), size=size, replace=False)
            order = [order[i] for i in sorted(picked)]
        return [Exemplar(member_ids[i], vectors[i]) for i in order]

    # --- exemplar index --------------------------------------------------
    def invalidate_index(self) -> None:
        """Mark the matching structures stale for this process and any other worker."""
        current = int(self.store.get_meta(INDEX_VERSION_KEY) or 0)
        self.store.set_meta(INDEX_VERSION_KEY, str(current + 1))
        self._cache = None

    def is_matchable(self, cluster: Cluster, now: datetime) -> bool:
        if cluster.status == ClusterStatus.OPEN:
            return True
        if cluster.status != ClusterStatus.CLOSED or cluster.status_locked:
            return False
        return cluster.closed_at is not None and (
            now - cluster.closed_at <= self.config.effective_reopen_grace
        )

    def active_index(self) -> _IndexCache:
        """Open clusters plus closed clusters still inside the reopen grace window."""
        now = self.clock.now()
        version = self.store.get_meta(INDEX_VERSION_KEY) or "0"
        cache = self._cache
        if (
            cache is not None
            and cache.version == version
            and (cache.valid_until is None or now <= cache.valid_until)
        ):
            return cache
        candidates = self.store.list_clusters([ClusterStatus.OPEN, ClusterStatus.CLOSED])
        live = [c for c in candidates if self.is_matchable(c, now)]
        expiries = [
            c.closed_at + self.config.effective_reopen_grace
            for c in live
            if c.status == ClusterStatus.CLOSED and c.closed_at is not None
        ]
        if self.config.exemplar_index == "store" and hasattr(self.store, "search_exemplars"):
            return self._store_index(version, live, expiries, now)
        exemplar_sets, kept = [], []
        for cluster in live:
            exemplars = self.store.get_exemplars(cluster.id)
            if exemplars:
                kept.append(cluster)
                exemplar_sets.append(np.stack([e.embedding for e in exemplars]))
        self._cache = _IndexCache(
            version=version,
            valid_until=min(expiries) if expiries else None,
            index=ExemplarIndex([c.id for c in kept], exemplar_sets),
            thresholds=np.array([c.threshold for c in kept], dtype=np.float32),
            clusters={c.id: c for c in kept},
        )
        return self._cache

    def _store_index(
        self, version: str, live: list[Cluster], expiries: list[datetime], now: datetime
    ) -> _IndexCache:
        index = StoreExemplarIndex(
            self.store,
            [c.id for c in live],
            self.config.exemplar_neighbors,
            now,
            self.config.effective_reopen_grace,
        )
        self._cache = _IndexCache(
            version=version,
            valid_until=min(expiries) if expiries else None,
            index=index,
            thresholds=np.array([c.threshold for c in live], dtype=np.float32),
            clusters={c.id: c for c in live},
        )
        return self._cache
