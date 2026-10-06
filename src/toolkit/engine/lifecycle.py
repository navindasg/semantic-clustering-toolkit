"""Sweep: close quiet clusters, archive expired ones, merge converged ones, orphan stale buffer items."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from toolkit.engine.assign import ClusterWriter
from toolkit.engine.context import EngineContext
from toolkit.engine.labels import label_clusters
from toolkit.engine.scoring import cluster_similarity
from toolkit.models import Cluster, ClusterStatus, EventType, ItemStatus, SweepReport

CENTROID_PREFILTER_MARGIN = 0.15


def merge_into(
    ctx: EngineContext,
    writer: ClusterWriter,
    survivor_id: str,
    absorbed_id: str,
    similarity: float | None,
    manual: bool = False,
) -> None:
    """Move every member of `absorbed` into `survivor`; the absorbed cluster points to the survivor."""
    survivor, absorbed = writer.get(survivor_id), writer.get(absorbed_id)
    members = ctx.store.cluster_items(absorbed_id)
    ctx.store.update_items([replace(m, cluster_id=survivor_id) for m in members])
    pool = writer.exemplars(survivor_id) + writer.exemplars(absorbed_id)
    vectors = np.stack([e.embedding for e in pool]) if pool else np.zeros((0, 0))
    exemplars = ctx.sample_exemplars([e.item_id for e in pool], vectors, ("merge", survivor_id, absorbed_id))
    writer.set_exemplars(survivor_id, exemplars)
    writer.put(
        replace(
            survivor,
            size=survivor.size + len(members),
            seen_count=survivor.seen_count + absorbed.seen_count,
            first_seen=min(survivor.first_seen, absorbed.first_seen),
            last_seen=max(survivor.last_seen, absorbed.last_seen),
            threshold=min(survivor.threshold, absorbed.threshold),
        )
    )
    writer.put(
        replace(
            absorbed,
            status=ClusterStatus.MERGED,
            merged_into=survivor_id,
            size=0,
            closed_at=ctx.clock.now(),
        )
    )
    writer.set_exemplars(absorbed_id, [])
    ctx.emit(
        EventType.CLUSTER_MERGED,
        absorbed_id,
        survivor=survivor_id,
        moved=len(members),
        similarity=round(similarity, 6) if similarity is not None else None,
        manual=manual,
    )


def pick_survivor(a: Cluster, b: Cluster) -> tuple[Cluster, Cluster]:
    """Larger cluster survives; ties go to the older, then the lexically smaller ID."""
    first, second = sorted((a, b), key=lambda c: (-c.size, c.opened_at, c.id))
    return first, second


def _merge_candidates(ctx: EngineContext, clusters: list[Cluster]) -> list[tuple[float, str, str]]:
    cfg = ctx.config
    sets = {c.id: ctx.store.get_exemplars(c.id) for c in clusters}
    usable = [c for c in clusters if sets[c.id]]
    if len(usable) < 2:
        return []
    matrices = {c.id: np.stack([e.embedding for e in sets[c.id]]) for c in usable}
    centroids = np.stack([matrices[c.id].mean(axis=0) for c in usable])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    centroid_sims = centroids @ centroids.T
    blocked = ctx.store.no_merge_pairs()
    pairs = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            if centroid_sims[i, j] < cfg.merge_threshold - CENTROID_PREFILTER_MARGIN:
                continue
            a, b = usable[i].id, usable[j].id
            if frozenset((a, b)) in blocked:
                continue
            sim = cluster_similarity(matrices[a], matrices[b], cfg.match_top_k)
            if sim >= cfg.merge_threshold:
                pairs.append((sim, a, b))
    return sorted(pairs, key=lambda p: (-p[0], p[1], p[2]))


def _run_merges(ctx: EngineContext, writer: ClusterWriter) -> list[tuple[str, str]]:
    open_clusters = [writer.get(c.id) for c in ctx.store.list_clusters([ClusterStatus.OPEN])]
    open_clusters = [c for c in open_clusters if c.status == ClusterStatus.OPEN and not c.status_locked]
    touched: set[str] = set()
    merged: list[tuple[str, str]] = []
    for sim, a, b in _merge_candidates(ctx, open_clusters):
        if a in touched or b in touched:
            continue
        survivor, absorbed = pick_survivor(writer.get(a), writer.get(b))
        merge_into(ctx, writer, survivor.id, absorbed.id, sim)
        touched.update((a, b))
        merged.append((absorbed.id, survivor.id))
    return merged


def _close_quiet(ctx: EngineContext, writer: ClusterWriter) -> list[str]:
    cfg, now = ctx.config, ctx.clock.now()
    cutoff = now - cfg.close_after
    closed = []
    for cluster in ctx.store.list_clusters([ClusterStatus.OPEN]):
        if cluster.status_locked:
            continue
        if cluster.last_seen > cutoff and cfg.keep_open_min_items <= 1:
            continue
        if ctx.store.count_cluster_items_since(cluster.id, cutoff) >= cfg.keep_open_min_items:
            continue
        writer.put(replace(cluster, status=ClusterStatus.CLOSED, closed_at=now))
        ctx.emit(
            EventType.CLUSTER_CLOSED,
            cluster.id,
            size=cluster.size,
            label=cluster.label,
            last_seen=cluster.last_seen.isoformat(),
        )
        closed.append(cluster.id)
    return closed


def _archive_expired(ctx: EngineContext, writer: ClusterWriter) -> list[str]:
    now, grace = ctx.clock.now(), ctx.config.effective_reopen_grace
    archived = []
    for cluster in ctx.store.list_clusters([ClusterStatus.CLOSED]):
        current = writer.get(cluster.id)
        if current.status != ClusterStatus.CLOSED or current.closed_at is None:
            continue
        if now - current.closed_at > grace:
            writer.put(replace(current, status=ClusterStatus.ARCHIVED))
            ctx.emit(EventType.CLUSTER_ARCHIVED, cluster.id, size=cluster.size, label=cluster.label)
            archived.append(cluster.id)
    return archived


def _orphan_stale(ctx: EngineContext) -> int:
    cutoff = ctx.clock.now() - ctx.config.buffer_max_age
    stale = [i for i in ctx.store.items_by_status(ItemStatus.BUFFERED) if i.timestamp < cutoff]
    ctx.store.update_items([replace(i, status=ItemStatus.ORPHANED) for i in stale])
    for item in stale:
        ctx.emit(EventType.ITEM_ORPHANED, item_id=item.id, timestamp=item.timestamp.isoformat())
    return len(stale)


def _relabel_grown(ctx: EngineContext, writer: ClusterWriter) -> list[str]:
    growth = ctx.config.relabel_growth
    due = [
        writer.get(c.id)
        for c in ctx.store.list_clusters([ClusterStatus.OPEN])
        if c.size >= max(1, c.label_size) * growth
    ]
    due = [c for c in due if c.status == ClusterStatus.OPEN]
    if not due:
        return []
    texts = {c.id: [m.text for m in ctx.store.cluster_items(c.id, limit=500)] for c in due}
    labels = label_clusters(ctx, texts)
    relabeled = []
    for cluster in due:
        label = labels.get(cluster.id)
        if not label:
            continue
        writer.put(replace(cluster, label=label, label_size=cluster.size), structural=False)
        if label != cluster.label:
            ctx.emit(EventType.CLUSTER_RELABELED, cluster.id, label=label, previous=cluster.label)
            relabeled.append(cluster.id)
    return relabeled


def run_sweep(ctx: EngineContext) -> SweepReport:
    """All sweep steps share one transaction; rerunning a sweep at the same time is a no-op."""
    with ctx.unit_of_work():
        writer = ClusterWriter(ctx)
        closed = _close_quiet(ctx, writer)
        archived = _archive_expired(ctx, writer)
        merged = _run_merges(ctx, writer)
        orphaned = _orphan_stale(ctx)
        relabeled = _relabel_grown(ctx, writer)
        writer.flush()
    return SweepReport(tuple(closed), tuple(archived), tuple(merged), orphaned, tuple(relabeled))
