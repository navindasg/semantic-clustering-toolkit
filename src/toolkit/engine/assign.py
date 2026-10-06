"""Assignment: score items against live clusters, attach on a match, otherwise buffer."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np

from toolkit.engine.context import EngineContext
from toolkit.models import (
    AssignmentResult,
    Cluster,
    ClusterStatus,
    EventType,
    Exemplar,
    ItemStatus,
    StoredItem,
)


def decide(ctx: EngineContext, item_ids: list[str], vectors: np.ndarray) -> list[AssignmentResult]:
    """Pure scoring step: best and runner-up cluster per item, against the live index."""
    cache = ctx.active_index()
    if len(cache.index) == 0:
        return [AssignmentResult(i, None, None, None, None) for i in item_ids]
    scores = cache.index.scores(vectors, ctx.config.match_top_k)
    ids = cache.index.cluster_ids
    results = []
    for row, item_id in enumerate(item_ids):
        order = np.argsort(-scores[row], kind="stable")
        best = int(order[0])
        best_score = float(scores[row, best])
        runner = int(order[1]) if len(order) > 1 else None
        matched = best_score >= float(cache.thresholds[best])
        results.append(
            AssignmentResult(
                item_id=item_id,
                cluster_id=ids[best] if matched else None,
                score=round(best_score, 6),
                runner_up_cluster_id=ids[runner] if runner is not None else None,
                runner_up_score=round(float(scores[row, runner]), 6) if runner is not None else None,
            )
        )
    return results


class ClusterWriter:
    """Accumulates cluster changes inside a unit of work, then writes them once."""

    def __init__(self, ctx: EngineContext) -> None:
        self.ctx = ctx
        self._clusters: dict[str, Cluster] = {}
        self._exemplars: dict[str, list[Exemplar]] = {}
        self._new: set[str] = set()
        self._structural = False

    def get(self, cluster_id: str) -> Cluster:
        if cluster_id not in self._clusters:
            cluster = self.ctx.store.get_cluster(cluster_id)
            if cluster is None:
                raise KeyError(f"unknown cluster {cluster_id}")
            self._clusters[cluster_id] = cluster
        return self._clusters[cluster_id]

    def exemplars(self, cluster_id: str) -> list[Exemplar]:
        if cluster_id not in self._exemplars:
            self._exemplars[cluster_id] = self.ctx.store.get_exemplars(cluster_id)
        return self._exemplars[cluster_id]

    def put(self, cluster: Cluster, structural: bool = True) -> None:
        self._clusters[cluster.id] = cluster
        self._structural = self._structural or structural

    def set_exemplars(self, cluster_id: str, exemplars: list[Exemplar]) -> None:
        self._exemplars[cluster_id] = list(exemplars)
        self._structural = True

    def create(self, cluster: Cluster, exemplars: list[Exemplar]) -> None:
        self._clusters[cluster.id] = cluster
        self._exemplars[cluster.id] = list(exemplars)
        self._new.add(cluster.id)
        self._structural = True

    def attach(
        self,
        cluster_id: str,
        item_id: str,
        embedding: np.ndarray,
        timestamp: datetime,
        score: float | None,
        **event_data: object,
    ) -> None:
        cluster = self.get(cluster_id)
        if cluster.status == ClusterStatus.CLOSED:
            cluster = replace(cluster, status=ClusterStatus.OPEN, closed_at=None)
            self._structural = True
            self.ctx.emit(
                EventType.CLUSTER_REOPENED,
                cluster.id,
                item_id,
                size=cluster.size,
                label=cluster.label,
            )
        seen = cluster.seen_count + 1
        cluster = replace(
            cluster,
            size=cluster.size + 1,
            seen_count=seen,
            first_seen=min(cluster.first_seen, timestamp),
            last_seen=max(cluster.last_seen, timestamp),
        )
        self._clusters[cluster_id] = cluster
        self._reservoir(cluster_id, seen, Exemplar(item_id, embedding))
        data = {"score": score} if score is not None else {}
        self.ctx.emit(EventType.ITEM_ASSIGNED, cluster_id, item_id, **data, **event_data)

    def _reservoir(self, cluster_id: str, seen: int, exemplar: Exemplar) -> None:
        """Reservoir sampling keeps a uniform, fixed-size sample of everything the cluster saw."""
        current = self.exemplars(cluster_id)
        limit = self.ctx.config.exemplars_per_cluster
        if len(current) < limit:
            self.set_exemplars(cluster_id, [*current, exemplar])
            return
        slot = int(self.ctx.rng("reservoir", cluster_id, seen).integers(0, seen))
        if slot < limit:
            self.set_exemplars(cluster_id, [*current[:slot], exemplar, *current[slot + 1 :]])

    def flush(self) -> None:
        store = self.ctx.store
        for cluster_id, cluster in self._clusters.items():
            if cluster_id in self._new:
                store.add_cluster(cluster, self._exemplars.get(cluster_id, []))
            else:
                store.update_cluster(cluster)
        for cluster_id, exemplars in self._exemplars.items():
            if cluster_id not in self._new:
                store.set_exemplars(cluster_id, exemplars)
        if self._structural:
            self.ctx.invalidate_index()
        self._clusters, self._exemplars, self._new = {}, {}, set()
        self._structural = False


def reassign_buffered(ctx: EngineContext, items: list[StoredItem]) -> int:
    """Try buffered items against the current clusters; returns how many attached."""
    candidates = [i for i in items if i.status == ItemStatus.BUFFERED and not i.locked]
    if not candidates:
        return 0
    vectors = np.stack([i.embedding for i in candidates])
    decisions = decide(ctx, [i.id for i in candidates], vectors)
    writer = ClusterWriter(ctx)
    updated = []
    for item, decision in zip(candidates, decisions, strict=True):
        if decision.cluster_id is None:
            continue
        writer.attach(
            decision.cluster_id,
            item.id,
            item.embedding,
            item.timestamp,
            decision.score,
            reassigned=True,
        )
        updated.append(
            replace(
                item,
                status=ItemStatus.ASSIGNED,
                cluster_id=decision.cluster_id,
                score=decision.score,
            )
        )
    ctx.store.update_items(updated)
    writer.flush()
    return len(updated)
