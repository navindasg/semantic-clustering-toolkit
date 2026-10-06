"""Discovery: cluster a snapshot of the buffer (UMAP then HDBSCAN) without touching history."""

from __future__ import annotations

import logging
import uuid
import warnings
from dataclasses import replace

import numpy as np
from sklearn.cluster import HDBSCAN

from toolkit.config import Config
from toolkit.engine.assign import ClusterWriter, reassign_buffered
from toolkit.engine.context import EngineContext, cluster_id_for
from toolkit.engine.labels import label_clusters
from toolkit.engine.scoring import cluster_similarity, member_scores, threshold_from_scores
from toolkit.models import (
    Cluster,
    ClusterStatus,
    DiscoveryReport,
    EventType,
    ItemStatus,
    StoredItem,
)

logger = logging.getLogger("toolkit.discover")
LOCK_NAME = "discovery"
LOCK_TTL_SECONDS = 3600.0
UMAP_MIN_ITEMS = 50


def reduce(vectors: np.ndarray, config: Config) -> np.ndarray:
    """UMAP to a few dimensions so density is meaningful for HDBSCAN.

    Small buffers skip it: UMAP's neighbour graph is unreliable on a few dozen points, and
    HDBSCAN copes fine with the raw normalized vectors at that size.
    """
    n = len(vectors)
    components = min(config.umap_components, n - 2)
    if n < max(UMAP_MIN_ITEMS, 3 * config.umap_components) or components < 2:
        return vectors
    import umap  # imported lazily: numba compilation is slow and only discovery needs it

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=components,
            n_neighbors=max(2, min(config.umap_neighbors, n - 1)),
            metric="cosine",
            min_dist=0.0,
            random_state=config.random_seed,
            init="spectral" if n >= 100 else "random",
        )
        return np.asarray(reducer.fit_transform(vectors), dtype=np.float64)


def cluster_labels(vectors: np.ndarray, config: Config) -> np.ndarray:
    """HDBSCAN labels for the vectors; -1 means noise."""
    if len(vectors) < config.min_cluster_size:
        return np.full(len(vectors), -1)
    reduced = reduce(vectors, config)

    def fit(allow_single: bool) -> np.ndarray:
        return HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=min(config.min_samples, config.min_cluster_size),
            allow_single_cluster=allow_single,
            copy=True,
        ).fit_predict(reduced)

    labels = fit(False)
    if labels.max() < 0:
        # A buffer holding one topic has no split for HDBSCAN to find; allow a single cluster.
        # The cohesion gate in discovery still rejects it if it is a loose catch-all.
        labels = fit(True)
    return labels


def _fit_sample(snapshot: list[StoredItem], ctx: EngineContext) -> list[StoredItem]:
    limit = ctx.config.discovery_sample_size
    if len(snapshot) <= limit:
        return snapshot
    picked = ctx.rng("discovery-sample", snapshot[-1].seq).choice(len(snapshot), limit, replace=False)
    return [snapshot[i] for i in sorted(picked)]


def _groups(items: list[StoredItem], labels: np.ndarray) -> list[list[StoredItem]]:
    by_label: dict[int, list[StoredItem]] = {}
    for item, label in zip(items, labels, strict=True):
        if label >= 0:
            by_label.setdefault(int(label), []).append(item)
    return sorted(by_label.values(), key=lambda g: min(i.seq for i in g))


def _best_overlap(
    ctx: EngineContext, writer: ClusterWriter, candidate: np.ndarray, open_ids: list[str]
) -> tuple[str | None, float]:
    best_id, best = None, -1.0
    for cluster_id in open_ids:
        exemplars = writer.exemplars(cluster_id)
        if not exemplars:
            continue
        sim = cluster_similarity(
            candidate, np.stack([e.embedding for e in exemplars]), ctx.config.match_top_k
        )
        if sim > best:
            best_id, best = cluster_id, sim
    return best_id, best


def free_cluster_id(ctx: EngineContext, member_ids: list[str]) -> str:
    """Content-derived ID; if the same members founded a cluster before (e.g. a repeated split),
    salt the hash deterministically until the ID is free."""
    cid, salt = cluster_id_for(member_ids), 0
    while ctx.store.get_cluster(cid) is not None:
        salt += 1
        cid = cluster_id_for([*member_ids, f"#{salt}"])
    return cid


def build_cluster(
    ctx: EngineContext, members: list[StoredItem], cluster_id: str | None = None
) -> tuple[Cluster, list, float]:
    """New open cluster from members: permanent ID, sampled exemplars, own threshold.

    Also returns the unclamped cohesion (the threshold percentile of member scores), which
    discovery uses to reject candidates too loose to be a single topic.
    """
    cfg = ctx.config
    ids = [m.id for m in members]
    vectors = np.stack([m.embedding for m in members])
    cid = cluster_id or free_cluster_id(ctx, ids)
    exemplars = ctx.sample_exemplars(ids, vectors, cid)
    scores = member_scores(
        vectors,
        ids,
        np.stack([e.embedding for e in exemplars]),
        [e.item_id for e in exemplars],
        cfg.match_top_k,
    )
    cohesion = float(np.percentile(scores, cfg.threshold_percentile))
    threshold = threshold_from_scores(scores, cfg.threshold_percentile, cfg.threshold_bounds)
    now = ctx.clock.now()
    cluster = Cluster(
        id=cid,
        status=ClusterStatus.OPEN,
        threshold=round(threshold, 6),
        size=len(members),
        first_seen=min(m.timestamp for m in members),
        last_seen=max(m.timestamp for m in members),
        opened_at=now,
        seen_count=len(members),
    )
    return cluster, exemplars, cohesion


def _create(
    ctx: EngineContext, writer: ClusterWriter, group: list[StoredItem]
) -> tuple[Cluster, list[StoredItem]] | None:
    """Open a cluster for the group, or return None when it is not cohesive enough."""
    cluster, exemplars, cohesion = build_cluster(ctx, group)
    if cohesion < ctx.config.threshold_bounds[0]:
        logger.debug("rejected loose candidate of %d items (cohesion %.3f)", len(group), cohesion)
        return None
    writer.create(cluster, exemplars)
    updated = [replace(m, status=ItemStatus.ASSIGNED, cluster_id=cluster.id, score=None) for m in group]
    return cluster, updated


def run_discovery(ctx: EngineContext) -> DiscoveryReport:
    """Snapshot the buffer, find candidate groups, fold or open clusters, re-assign late arrivals.

    Nothing is written until every candidate has been computed, and all writes share one
    transaction, so a failed run leaves the buffer untouched.
    """
    cfg = ctx.config
    owner = uuid.uuid4().hex
    if not ctx.store.acquire_lock(LOCK_NAME, owner, ctx.clock.now(), LOCK_TTL_SECONDS):
        return DiscoveryReport(0, (), (), 0, 0, skipped_reason="another discovery run holds the lock")
    try:
        snapshot = ctx.store.items_by_status(ItemStatus.BUFFERED)
        snapshot = [i for i in snapshot if not i.locked]
        if len(snapshot) < cfg.min_cluster_size:
            return DiscoveryReport(len(snapshot), (), (), len(snapshot), 0, "buffer too small")
        snapshot_seq = max(i.seq for i in snapshot)
        fit_items = _fit_sample(snapshot, ctx)
        labels = cluster_labels(np.stack([i.embedding for i in fit_items]), cfg)
        groups = _groups(fit_items, labels)
        opened, folded, rejected = [], [], 0
        with ctx.unit_of_work():
            still_buffered = {
                i.id for i in ctx.store.get_items(i.id for i in fit_items) if i.status == ItemStatus.BUFFERED
            }
            writer = ClusterWriter(ctx)
            open_ids = [c.id for c in ctx.store.list_clusters([ClusterStatus.OPEN])]
            updates: list[StoredItem] = []
            new_clusters: list[Cluster] = []
            for raw_group in groups:
                group = [i for i in raw_group if i.id in still_buffered]
                if len(group) < cfg.min_cluster_size:
                    continue
                vectors = np.stack([m.embedding for m in group])
                sample = ctx.sample_exemplars([m.id for m in group], vectors, "candidate")
                target, sim = _best_overlap(ctx, writer, np.stack([e.embedding for e in sample]), open_ids)
                if target is not None and sim >= cfg.merge_threshold:
                    for m in group:
                        writer.attach(target, m.id, m.embedding, m.timestamp, None, folded=True)
                    updates.extend(replace(m, status=ItemStatus.ASSIGNED, cluster_id=target) for m in group)
                    folded.append(target)
                    continue
                created = _create(ctx, writer, group)
                if created is None:
                    rejected += 1
                    continue
                cluster, members = created
                updates.extend(members)
                new_clusters.append(cluster)
                opened.append(cluster.id)
            ctx.store.update_items(updates)
            labels_by_id = label_clusters(
                ctx, {c.id: [u.text for u in updates if u.cluster_id == c.id] for c in new_clusters}
            )
            for cluster in new_clusters:
                labeled = replace(
                    writer.get(cluster.id),
                    label=labels_by_id.get(cluster.id, ""),
                    label_size=cluster.size,
                )
                writer.put(labeled)
                ctx.emit(
                    EventType.CLUSTER_OPENED,
                    cluster.id,
                    size=cluster.size,
                    threshold=cluster.threshold,
                    label=labeled.label,
                )
                for member in (u for u in updates if u.cluster_id == cluster.id):
                    ctx.emit(EventType.ITEM_ASSIGNED, cluster.id, member.id, discovered=True)
            writer.flush()
            fit_ids = {i.id for i in fit_items}
            leftovers = [
                i
                for i in ctx.store.items_by_status(ItemStatus.BUFFERED)
                if i.seq > snapshot_seq or i.id not in fit_ids
            ]
            reassigned = reassign_buffered(ctx, leftovers) if (opened or folded) else 0
        noise = int(sum(1 for label in labels if label < 0))
        logger.info(
            "discovery: snapshot=%d opened=%d folded=%d rejected=%d noise=%d reassigned=%d",
            len(snapshot),
            len(opened),
            len(folded),
            rejected,
            noise,
            reassigned,
        )
        return DiscoveryReport(
            len(snapshot), tuple(opened), tuple(folded), noise, reassigned, rejected=rejected
        )
    finally:
        ctx.store.release_lock(LOCK_NAME, owner)
