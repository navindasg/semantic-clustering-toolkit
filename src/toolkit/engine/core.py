"""The public engine: ingest/assign, discover, sweep, sample, query and override."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from toolkit.clock import Clock, SystemClock, ensure_utc
from toolkit.config import Config
from toolkit.embedders import Embedder, ModelMismatchError, make_embedder
from toolkit.engine import overrides
from toolkit.engine.assign import ClusterWriter, decide
from toolkit.engine.context import EngineContext
from toolkit.engine.discover import run_discovery
from toolkit.engine.lifecycle import run_sweep
from toolkit.engine.sampling import SampleStrategy, counts_over_time, sample_cluster
from toolkit.engine.scoring import member_scores, threshold_from_scores
from toolkit.events.bus import EventBus
from toolkit.labeling import Labeler, make_labeler
from toolkit.models import (
    AssignmentResult,
    Cluster,
    ClusterStatus,
    DiscoveryReport,
    Event,
    Item,
    ItemStatus,
    StoredItem,
    SweepReport,
)
from toolkit.preprocess import Preprocessor, build_preprocessor
from toolkit.store import Store, open_store

logger = logging.getLogger("toolkit.engine")
MODEL_META_KEY = "model_id"


@dataclass(frozen=True)
class ClusterDetail:
    cluster: Cluster
    counts_over_time: list[tuple[datetime, int]]
    sample: list[StoredItem]


def _validate(items: Iterable[Item]) -> list[Item]:
    cleaned = []
    for item in items:
        if not isinstance(item, Item):
            raise TypeError(f"expected Item, got {type(item).__name__}")
        if not str(item.id).strip():
            raise ValueError("item id must be non-empty")
        if not isinstance(item.text, str) or not item.text.strip():
            raise ValueError(f"item {item.id} has empty text")
        if not isinstance(item.timestamp, datetime):
            raise TypeError(f"item {item.id} timestamp must be a datetime")
        cleaned.append(replace(item, id=str(item.id), timestamp=ensure_utc(item.timestamp)))
    return cleaned


class ClusteringEngine:
    """Pure-Python engine. All I/O goes through the store, embedder, clock and event bus adapters."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        store: Store | None = None,
        embedder: Embedder | None = None,
        clock: Clock | None = None,
        bus: EventBus | None = None,
        labeler: Labeler | None = None,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        cfg = config or Config()
        self.config = cfg
        self.ctx = EngineContext(
            config=cfg,
            store=store if store is not None else open_store(cfg.store),
            embedder=embedder or make_embedder(cfg.embedder, cfg.model_dir, cfg.embed_batch_size),
            clock=clock or SystemClock(),
            bus=bus or EventBus(),
            labeler=labeler or make_labeler(cfg.labeler, cfg.label_terms),
        )
        self._preprocess = build_preprocessor(cfg.preprocess, preprocessor)
        self._discovery_lock = threading.Lock()
        self._check_model()

    # --- wiring ----------------------------------------------------------
    @property
    def store(self) -> Store:
        return self.ctx.store

    @property
    def bus(self) -> EventBus:
        return self.ctx.bus

    @property
    def clock(self) -> Clock:
        return self.ctx.clock

    def close(self) -> None:
        self.ctx.bus.close()
        self.ctx.store.close()

    def _check_model(self) -> None:
        stored = self.store.get_meta(MODEL_META_KEY)
        current = self.ctx.embedder.model_id
        if stored is None:
            self.store.set_meta(MODEL_META_KEY, current)
        elif stored != current:
            raise ModelMismatchError(
                f"store holds vectors from {stored!r} but the engine uses {current!r}; "
                f"run `toolkit reembed` to migrate, or configure embedder: {stored}"
            )

    # --- assignment ------------------------------------------------------
    def ingest(self, items: Iterable[Item]) -> list[AssignmentResult]:
        """Embed and assign a batch. Re-ingesting a known item ID is a no-op."""
        batch = _validate(items)
        known = self.store.existing_item_ids(i.id for i in batch)
        fresh: dict[str, Item] = {}
        for item in batch:
            if item.id not in known and item.id not in fresh:
                fresh[item.id] = item
        new_items = list(fresh.values())
        results: dict[str, AssignmentResult] = {}
        if new_items:
            results = {r.item_id: r for r in self._assign_new(new_items)}
        raced = {i for i, r in results.items() if r.duplicate}
        stored = {
            i.id: i for i in self.store.get_items(i.id for i in batch if i.id in known or i.id in raced)
        }
        results = {
            i: replace(r, cluster_id=stored[i].cluster_id, score=None) if i in raced else r
            for i, r in results.items()
        }
        out, answered = [], set()
        for item in batch:
            if item.id in results and item.id not in answered:
                out.append(results[item.id])
                answered.add(item.id)
                continue
            existing = stored.get(item.id)
            cluster_id = existing.cluster_id if existing else results[item.id].cluster_id
            out.append(AssignmentResult(item.id, cluster_id, None, None, None, duplicate=True))
        return out

    def _assign_new(self, items: list[Item]) -> list[AssignmentResult]:
        texts = [self._preprocess(i.text) if self._preprocess else i.text for i in items]
        embedder = self.ctx.embedder  # capture first: a migration may swap it while we embed
        vectors = embedder.embed(texts)
        model_id = embedder.model_id
        with self.ctx.unit_of_work():
            if self.ctx.embedder.model_id != model_id:
                # A re-embed migration committed while we were embedding: redo it with the new model.
                vectors = self.ctx.embedder.embed(texts)
                model_id = self.ctx.embedder.model_id
            decisions = decide(self.ctx, [i.id for i in items], vectors)
            writer = ClusterWriter(self.ctx)
            records = []
            for item, text, vector, decision in zip(items, texts, vectors, decisions, strict=True):
                assigned = decision.cluster_id is not None
                records.append(
                    StoredItem(
                        id=item.id,
                        text=text,
                        timestamp=item.timestamp,
                        metadata=dict(item.metadata),
                        embedding=vector,
                        model_id=model_id,
                        status=ItemStatus.ASSIGNED if assigned else ItemStatus.BUFFERED,
                        cluster_id=decision.cluster_id,
                        seq=0,
                        score=decision.score,
                    )
                )
            # Another writer may have ingested some of these IDs since our pre-check; only the
            # records actually inserted here may touch clusters, so ingestion stays idempotent.
            inserted = {r.id for r in self.store.add_items(records)}
            decisions = [d if d.item_id in inserted else replace(d, duplicate=True) for d in decisions]
            for record, decision in zip(records, decisions, strict=True):
                if decision.cluster_id is not None and not decision.duplicate:
                    writer.attach(
                        decision.cluster_id,
                        record.id,
                        record.embedding,
                        record.timestamp,
                        decision.score,
                    )
            writer.flush()
        return decisions

    # --- scheduled jobs --------------------------------------------------
    def buffer_size(self) -> int:
        return self.store.count_items_by_status()[ItemStatus.BUFFERED]

    def discover(self) -> DiscoveryReport:
        """Cluster the buffer. Only one run at a time; a concurrent call returns a skipped report."""
        if not self._discovery_lock.acquire(blocking=False):
            return DiscoveryReport(0, (), (), 0, 0, skipped_reason="discovery already running")
        try:
            return run_discovery(self.ctx)
        finally:
            self._discovery_lock.release()

    def sweep(self) -> SweepReport:
        return run_sweep(self.ctx)

    # --- queries ---------------------------------------------------------
    def list_clusters(self, status: ClusterStatus | str | Iterable | None = None) -> list[Cluster]:
        if status is None:
            return self.store.list_clusters()
        if isinstance(status, ClusterStatus | str):
            return self.store.list_clusters([ClusterStatus(status)])
        return self.store.list_clusters([ClusterStatus(s) for s in status])

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        return self.store.get_cluster(cluster_id)

    def resolve(self, cluster_id: str) -> Cluster | None:
        """Follow merge pointers to the surviving cluster."""
        seen: set[str] = set()
        cluster = self.store.get_cluster(cluster_id)
        while cluster is not None and cluster.merged_into and cluster.id not in seen:
            seen.add(cluster.id)
            cluster = self.store.get_cluster(cluster.merged_into)
        return cluster

    def cluster_detail(
        self, cluster_id: str, sample_size: int = 10, bucket: timedelta | None = None
    ) -> ClusterDetail:
        cluster = self.store.get_cluster(cluster_id)
        if cluster is None:
            raise KeyError(f"unknown cluster {cluster_id}")
        members = self.store.cluster_items(cluster_id)
        sample = sample_cluster(self.ctx, cluster_id, sample_size) if members else []
        return ClusterDetail(cluster, counts_over_time(members, bucket), sample)

    def sample(
        self, cluster_id: str, n: int = 10, strategy: SampleStrategy | str = "mixed"
    ) -> list[StoredItem]:
        return sample_cluster(self.ctx, cluster_id, n, strategy)

    def events(self, after_seq: int = 0, limit: int | None = None) -> list[Event]:
        return self.store.list_events(after_seq, limit)

    def stats(self) -> dict[str, Any]:
        """Counters for observability: items by status and clusters by status."""
        items = self.store.count_items_by_status()
        clusters = dict.fromkeys(ClusterStatus, 0)
        for cluster in self.store.list_clusters():
            clusters[cluster.status] += 1
        return {
            "items": {str(k): v for k, v in items.items()},
            "clusters": {str(k): v for k, v in clusters.items()},
            "model_id": self.ctx.embedder.model_id,
        }

    # --- operator overrides ---------------------------------------------
    def move_item(self, item_id: str, cluster_id: str) -> None:
        overrides.move_item(self.ctx, item_id, cluster_id)

    def merge_clusters(self, a: str, b: str, survivor: str | None = None) -> str:
        return overrides.merge_clusters(self.ctx, a, b, survivor)

    def split_cluster(self, cluster_id: str, item_ids: list[str]) -> str:
        return overrides.split_cluster(self.ctx, cluster_id, item_ids)

    def close_cluster(self, cluster_id: str) -> None:
        overrides.close_cluster(self.ctx, cluster_id)

    def reopen_cluster(self, cluster_id: str) -> None:
        overrides.reopen_cluster(self.ctx, cluster_id)

    def unlock_cluster(self, cluster_id: str) -> None:
        overrides.unlock_cluster(self.ctx, cluster_id)

    # --- model migration -------------------------------------------------
    def reembed(self, new_embedder: Embedder, batch_size: int = 256) -> int:
        """Re-embed every item and exemplar with a new model and recompute thresholds."""
        return reembed_store(self.ctx, new_embedder, batch_size)


def reembed_store(ctx: EngineContext, new_embedder: Embedder, batch_size: int = 256) -> int:
    with ctx._lock:  # held across commit, so the embedder swap below is atomic for other threads
        total = _reembed_locked(ctx, new_embedder, batch_size)
        ctx.embedder = new_embedder
        ctx.invalidate_index()
    return total


def _reembed_locked(ctx: EngineContext, new_embedder: Embedder, batch_size: int) -> int:
    cfg, store = ctx.config, ctx.store
    total = 0
    with ctx.unit_of_work():
        pending: list[StoredItem] = []
        for item in store.iter_items(batch_size):
            pending.append(item)
            if len(pending) >= batch_size:
                total += _reembed_batch(store, new_embedder, pending)
                pending = []
        total += _reembed_batch(store, new_embedder, pending)
        writer = ClusterWriter(ctx)
        for cluster in store.list_clusters():
            old = store.get_exemplars(cluster.id)
            if not old:
                continue
            fresh = {i.id: i for i in store.get_items(e.item_id for e in old)}
            exemplars = [replace(e, embedding=fresh[e.item_id].embedding) for e in old if e.item_id in fresh]
            writer.set_exemplars(cluster.id, exemplars)
            members = store.cluster_items(cluster.id)
            if members and len(exemplars) > 1:
                scores = member_scores(
                    np.stack([m.embedding for m in members]),
                    [m.id for m in members],
                    np.stack([e.embedding for e in exemplars]),
                    [e.item_id for e in exemplars],
                    cfg.match_top_k,
                )
                threshold = threshold_from_scores(scores, cfg.threshold_percentile, cfg.threshold_bounds)
                writer.put(replace(cluster, threshold=round(threshold, 6)))
        writer.flush()
        store.set_meta(MODEL_META_KEY, new_embedder.model_id)
    return total


def _reembed_batch(store: Store, embedder: Embedder, items: list[StoredItem]) -> int:
    if not items:
        return 0
    vectors = embedder.embed([i.text for i in items])
    store.update_items(
        [replace(i, embedding=v, model_id=embedder.model_id) for i, v in zip(items, vectors, strict=True)]
    )
    return len(items)
