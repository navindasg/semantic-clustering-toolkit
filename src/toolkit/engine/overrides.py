"""Operator overrides. Each one sets a lock so automation never undoes it:

* moved and split-off items are `locked`, so re-assignment never moves them;
* manually closed clusters are `status_locked`, so matches never reopen them;
* manually reopened clusters are `status_locked`, so sweeps never close or auto-merge them;
* split pairs go on a no-merge list, so the merge check never rejoins them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from toolkit.engine.assign import ClusterWriter
from toolkit.engine.context import EngineContext
from toolkit.engine.discover import build_cluster
from toolkit.engine.labels import label_clusters
from toolkit.engine.lifecycle import merge_into, pick_survivor
from toolkit.engine.scoring import member_scores, threshold_from_scores
from toolkit.models import Cluster, ClusterStatus, EventType, ItemStatus

_EDITABLE = {ClusterStatus.OPEN, ClusterStatus.CLOSED}


def _require(writer: ClusterWriter, cluster_id: str, allowed=_EDITABLE) -> Cluster:
    """Fresh cluster state, read inside the caller's unit of work so it can't go stale."""
    cluster = writer.get(cluster_id)
    if cluster.status not in allowed:
        raise ValueError(
            f"cluster {cluster_id} is {cluster.status}; expected one of {', '.join(sorted(allowed))}"
        )
    return cluster


def _refresh(ctx: EngineContext, writer: ClusterWriter, cluster_id: str) -> None:
    """Rebuild a cluster's exemplars and threshold from its current members."""
    cfg = ctx.config
    members = ctx.store.cluster_items(cluster_id)
    cluster = writer.get(cluster_id)
    if not members:
        writer.set_exemplars(cluster_id, [])
        writer.put(replace(cluster, size=0))
        return
    ids = [m.id for m in members]
    vectors = np.stack([m.embedding for m in members])
    exemplars = ctx.sample_exemplars(ids, vectors, ("refresh", cluster_id, len(ids)))
    threshold = cluster.threshold
    if len(exemplars) > 1:
        scores = member_scores(
            vectors,
            ids,
            np.stack([e.embedding for e in exemplars]),
            [e.item_id for e in exemplars],
            cfg.match_top_k,
        )
        threshold = threshold_from_scores(scores, cfg.threshold_percentile, cfg.threshold_bounds)
    writer.set_exemplars(cluster_id, exemplars)
    writer.put(
        replace(
            cluster,
            size=len(members),
            threshold=round(threshold, 6),
            first_seen=min(m.timestamp for m in members),
            last_seen=max(m.timestamp for m in members),
        )
    )


def move_item(ctx: EngineContext, item_id: str, cluster_id: str) -> None:
    with ctx.unit_of_work():
        writer = ClusterWriter(ctx)
        target = _require(writer, cluster_id)
        item = ctx.store.get_item(item_id)
        if item is None:
            raise KeyError(f"unknown item {item_id}")
        if item.cluster_id == cluster_id:
            ctx.store.update_items([replace(item, locked=True)])
            return
        source = item.cluster_id
        ctx.store.update_items(
            [replace(item, status=ItemStatus.ASSIGNED, cluster_id=cluster_id, locked=True)]
        )
        if source is not None:
            _refresh(ctx, writer, source)
        _refresh(ctx, writer, cluster_id)
        refreshed = writer.get(cluster_id)
        writer.put(replace(refreshed, seen_count=target.seen_count + 1))
        ctx.emit(EventType.ITEM_MOVED, cluster_id, item_id, previous=source, manual=True)
        writer.flush()


def merge_clusters(ctx: EngineContext, a: str, b: str, survivor_id: str | None = None) -> str:
    if a == b:
        raise ValueError("cannot merge a cluster with itself")
    if survivor_id is not None and survivor_id not in (a, b):
        raise ValueError("survivor must be one of the merged clusters")
    with ctx.unit_of_work():
        writer = ClusterWriter(ctx)
        first, second = _require(writer, a), _require(writer, b)
        if survivor_id is None:
            survivor, absorbed = pick_survivor(first, second)
        else:
            survivor, absorbed = (first, second) if survivor_id == a else (second, first)
        if absorbed.status == ClusterStatus.OPEN and survivor.status == ClusterStatus.CLOSED:
            writer.put(replace(survivor, status=ClusterStatus.OPEN, closed_at=None))
        merge_into(ctx, writer, survivor.id, absorbed.id, None, manual=True)
        writer.flush()
    return survivor.id


def split_cluster(ctx: EngineContext, cluster_id: str, item_ids: list[str]) -> str:
    """Move `item_ids` out of `cluster_id` into a new cluster; returns the new cluster's ID."""
    wanted = list(dict.fromkeys(item_ids))
    with ctx.unit_of_work():
        writer = ClusterWriter(ctx)
        _require(writer, cluster_id)
        members = [m for m in ctx.store.get_items(wanted) if m.cluster_id == cluster_id]
        if len(members) != len(wanted):
            raise ValueError("every split item must currently belong to the cluster")
        remaining = ctx.store.cluster_items(cluster_id)
        if len(remaining) - len(members) < 1:
            raise ValueError("a split must leave at least one item behind")
        new_cluster, exemplars, _ = build_cluster(ctx, members)
        writer.create(new_cluster, exemplars)
        ctx.store.update_items([replace(m, cluster_id=new_cluster.id, locked=True) for m in members])
        _refresh(ctx, writer, cluster_id)
        ctx.store.add_no_merge(cluster_id, new_cluster.id)
        labels = label_clusters(
            ctx,
            {
                new_cluster.id: [m.text for m in members],
                cluster_id: [m.text for m in ctx.store.cluster_items(cluster_id, limit=500)],
            },
        )
        for cid in (new_cluster.id, cluster_id):
            current = writer.get(cid)
            writer.put(replace(current, label=labels.get(cid, current.label), label_size=current.size))
        ctx.emit(
            EventType.CLUSTER_SPLIT,
            cluster_id,
            new_cluster=new_cluster.id,
            moved=len(members),
            manual=True,
        )
        ctx.emit(
            EventType.CLUSTER_OPENED,
            new_cluster.id,
            size=len(members),
            threshold=new_cluster.threshold,
            label=labels.get(new_cluster.id, ""),
            split_from=cluster_id,
        )
        writer.flush()
    return new_cluster.id


def close_cluster(ctx: EngineContext, cluster_id: str) -> None:
    with ctx.unit_of_work():
        writer = ClusterWriter(ctx)
        cluster = _require(writer, cluster_id, {ClusterStatus.OPEN, ClusterStatus.CLOSED})
        closed_at = cluster.closed_at if cluster.status == ClusterStatus.CLOSED else ctx.clock.now()
        writer.put(replace(cluster, status=ClusterStatus.CLOSED, closed_at=closed_at, status_locked=True))
        if cluster.status == ClusterStatus.OPEN:
            ctx.emit(
                EventType.CLUSTER_CLOSED,
                cluster_id,
                size=cluster.size,
                label=cluster.label,
                manual=True,
            )
        writer.flush()


def reopen_cluster(ctx: EngineContext, cluster_id: str) -> None:
    allowed = {ClusterStatus.OPEN, ClusterStatus.CLOSED, ClusterStatus.ARCHIVED}
    with ctx.unit_of_work():
        writer = ClusterWriter(ctx)
        cluster = _require(writer, cluster_id, allowed)
        writer.put(replace(cluster, status=ClusterStatus.OPEN, closed_at=None, status_locked=True))
        if cluster.status != ClusterStatus.OPEN:
            ctx.emit(
                EventType.CLUSTER_REOPENED,
                cluster_id,
                size=cluster.size,
                label=cluster.label,
                manual=True,
            )
        writer.flush()


def unlock_cluster(ctx: EngineContext, cluster_id: str) -> None:
    """Hand a cluster's status back to automation."""
    with ctx.unit_of_work():
        writer = ClusterWriter(ctx)
        cluster = _require(writer, cluster_id, set(ClusterStatus))
        writer.put(replace(cluster, status_locked=False))
        writer.flush()
