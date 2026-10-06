"""In-memory store for tests and the core engine milestone."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta

from toolkit.models import Cluster, ClusterStatus, Event, Exemplar, ItemStatus, StoredItem


class MemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, StoredItem] = {}
        self._clusters: dict[str, Cluster] = {}
        self._exemplars: dict[str, list[Exemplar]] = {}
        self._events: list[Event] = []
        self._meta: dict[str, str] = {}
        self._locks: dict[str, tuple[str, datetime]] = {}
        self._no_merge: set[frozenset[str]] = set()
        self._item_seq = 0
        self._rlock = threading.RLock()
        self._depth = 0

    # --- lifecycle -------------------------------------------------------
    def _snapshot(self) -> tuple:
        return (
            dict(self._items),
            dict(self._clusters),
            dict(self._exemplars),
            list(self._events),
            dict(self._meta),
            set(self._no_merge),
            self._item_seq,
        )

    def _restore(self, snap: tuple) -> None:
        (
            self._items,
            self._clusters,
            self._exemplars,
            self._events,
            self._meta,
            self._no_merge,
            self._item_seq,
        ) = snap

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._rlock:
            snap = self._snapshot() if self._depth == 0 else None
            self._depth += 1
            try:
                yield
            except BaseException:
                if snap is not None:
                    self._restore(snap)
                raise
            finally:
                self._depth -= 1

    def close(self) -> None:
        return None

    # --- metadata and locks ---------------------------------------------
    def get_meta(self, key: str) -> str | None:
        return self._meta.get(key)

    def set_meta(self, key: str, value: str) -> None:
        self._meta[key] = value

    def acquire_lock(self, name: str, owner: str, now: datetime, ttl_seconds: float) -> bool:
        with self._rlock:
            held = self._locks.get(name)
            if held and held[0] != owner and held[1] > now:
                return False
            self._locks[name] = (owner, now + timedelta(seconds=ttl_seconds))
            return True

    def release_lock(self, name: str, owner: str) -> None:
        with self._rlock:
            if self._locks.get(name, ("",))[0] == owner:
                del self._locks[name]

    # --- items -----------------------------------------------------------
    def existing_item_ids(self, ids: Iterable[str]) -> set[str]:
        return {i for i in ids if i in self._items}

    def add_items(self, items: list[StoredItem]) -> list[StoredItem]:
        with self._rlock:
            stored = []
            for item in items:
                if item.id in self._items:
                    continue
                self._item_seq += 1
                record = replace(item, seq=self._item_seq)
                self._items[item.id] = record
                stored.append(record)
            return stored

    def update_items(self, items: list[StoredItem]) -> None:
        with self._rlock:
            for item in items:
                current = self._items[item.id]
                self._items[item.id] = replace(item, seq=current.seq)

    def get_item(self, item_id: str) -> StoredItem | None:
        return self._items.get(item_id)

    def get_items(self, ids: Iterable[str]) -> list[StoredItem]:
        return [self._items[i] for i in ids if i in self._items]

    def items_by_status(self, status: ItemStatus, after_seq: int = 0) -> list[StoredItem]:
        found = [i for i in self._items.values() if i.status == status and i.seq > after_seq]
        return sorted(found, key=lambda i: i.seq)

    def cluster_items(self, cluster_id: str, limit: int | None = None) -> list[StoredItem]:
        found = [i for i in self._items.values() if i.cluster_id == cluster_id]
        found.sort(key=lambda i: (i.timestamp, i.seq), reverse=True)
        return found[:limit] if limit is not None else found

    def count_cluster_items_since(self, cluster_id: str, since: datetime) -> int:
        return sum(1 for i in self._items.values() if i.cluster_id == cluster_id and i.timestamp > since)

    def count_items_by_status(self) -> dict[ItemStatus, int]:
        counts = dict.fromkeys(ItemStatus, 0)
        for item in self._items.values():
            counts[item.status] += 1
        return counts

    def iter_items(self, batch_size: int = 1000) -> Iterator[StoredItem]:
        yield from sorted(self._items.values(), key=lambda i: i.seq)

    def max_item_seq(self) -> int:
        return self._item_seq

    # --- clusters --------------------------------------------------------
    def add_cluster(self, cluster: Cluster, exemplars: list[Exemplar]) -> None:
        with self._rlock:
            if cluster.id in self._clusters:
                raise ValueError(f"cluster {cluster.id} already exists")
            self._clusters[cluster.id] = cluster
            self._exemplars[cluster.id] = list(exemplars)

    def update_cluster(self, cluster: Cluster) -> None:
        with self._rlock:
            if cluster.id not in self._clusters:
                raise KeyError(cluster.id)
            self._clusters[cluster.id] = cluster

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        return self._clusters.get(cluster_id)

    def list_clusters(self, statuses: Iterable[ClusterStatus] | None = None) -> list[Cluster]:
        wanted = set(statuses) if statuses is not None else None
        found = [c for c in self._clusters.values() if wanted is None or c.status in wanted]
        return sorted(found, key=lambda c: (c.opened_at, c.id))

    def get_exemplars(self, cluster_id: str) -> list[Exemplar]:
        return list(self._exemplars.get(cluster_id, []))

    def set_exemplars(self, cluster_id: str, exemplars: list[Exemplar]) -> None:
        with self._rlock:
            self._exemplars[cluster_id] = list(exemplars)

    def add_no_merge(self, a: str, b: str) -> None:
        self._no_merge.add(frozenset((a, b)))

    def no_merge_pairs(self) -> set[frozenset[str]]:
        return set(self._no_merge)

    # --- events ----------------------------------------------------------
    def append_events(self, events: list[Event]) -> list[Event]:
        with self._rlock:
            start = len(self._events)
            stored = [replace(e, seq=start + n + 1) for n, e in enumerate(events)]
            self._events.extend(stored)
            return stored

    def list_events(self, after_seq: int = 0, limit: int | None = None) -> list[Event]:
        found = [e for e in self._events if e.seq > after_seq]
        return found[:limit] if limit is not None else found
