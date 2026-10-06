"""SQLite store adapter: one local file, no server. The demo-mode default."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from toolkit.models import (
    Cluster,
    ClusterStatus,
    Event,
    EventType,
    Exemplar,
    ItemStatus,
    StoredItem,
)
from toolkit.store.codec import (
    bytes_to_vec,
    dt_to_text,
    from_json,
    text_to_dt,
    text_to_required_dt,
    to_json,
    vec_to_bytes,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL UNIQUE,
    text TEXT NOT NULL,
    ts TEXT NOT NULL,
    metadata TEXT NOT NULL,
    embedding BLOB NOT NULL,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL,
    cluster_id TEXT,
    locked INTEGER NOT NULL DEFAULT 0,
    score REAL
);
CREATE INDEX IF NOT EXISTS items_status_seq ON items(status, seq);
CREATE INDEX IF NOT EXISTS items_cluster_ts ON items(cluster_id, ts);
CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    threshold REAL NOT NULL,
    size INTEGER NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    label TEXT NOT NULL,
    closed_at TEXT,
    merged_into TEXT,
    status_locked INTEGER NOT NULL,
    label_size INTEGER NOT NULL,
    seen_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS clusters_status ON clusters(status);
CREATE TABLE IF NOT EXISTS exemplars (
    cluster_id TEXT NOT NULL,
    pos INTEGER NOT NULL,
    item_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (cluster_id, pos)
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    ts TEXT NOT NULL,
    cluster_id TEXT,
    item_id TEXT,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS locks (name TEXT PRIMARY KEY, owner TEXT NOT NULL, expires TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS no_merge (a TEXT NOT NULL, b TEXT NOT NULL, PRIMARY KEY (a, b));
"""

_ITEM_COLS = "id, seq, text, ts, metadata, embedding, model_id, status, cluster_id, locked, score"
_CLUSTER_COLS = (
    "id, status, threshold, size, first_seen, last_seen, opened_at, label, closed_at, "
    "merged_into, status_locked, label_size, seen_count"
)


def _row_to_item(row: sqlite3.Row) -> StoredItem:
    return StoredItem(
        id=row["id"],
        text=row["text"],
        timestamp=text_to_required_dt(row["ts"]),
        metadata=from_json(row["metadata"]),
        embedding=bytes_to_vec(row["embedding"]),
        model_id=row["model_id"],
        status=ItemStatus(row["status"]),
        cluster_id=row["cluster_id"],
        seq=row["seq"],
        locked=bool(row["locked"]),
        score=row["score"],
    )


def _row_to_cluster(row: sqlite3.Row) -> Cluster:
    return Cluster(
        id=row["id"],
        status=ClusterStatus(row["status"]),
        threshold=row["threshold"],
        size=row["size"],
        first_seen=text_to_required_dt(row["first_seen"]),
        last_seen=text_to_required_dt(row["last_seen"]),
        opened_at=text_to_required_dt(row["opened_at"]),
        label=row["label"],
        closed_at=text_to_dt(row["closed_at"]),
        merged_into=row["merged_into"],
        status_locked=bool(row["status_locked"]),
        label_size=row["label_size"],
        seen_count=row["seen_count"],
    )


def _cluster_params(c: Cluster) -> tuple:
    return (
        c.id,
        str(c.status),
        c.threshold,
        c.size,
        dt_to_text(c.first_seen),
        dt_to_text(c.last_seen),
        dt_to_text(c.opened_at),
        c.label,
        dt_to_text(c.closed_at),
        c.merged_into,
        int(c.status_locked),
        c.label_size,
        c.seen_count,
    )


class SQLiteStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()
        self._depth = 0

    # --- lifecycle -------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            outer = self._depth == 0
            if outer:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if outer:
                    self._conn.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outer:
                self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _query(self, sql: str, params: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def _write(self, sql: str, params: Iterable = ()) -> None:
        with self.transaction():
            self._conn.execute(sql, tuple(params))

    def _write_many(self, sql: str, rows: list[tuple]) -> None:
        if rows:
            with self.transaction():
                self._conn.executemany(sql, rows)

    # --- metadata and locks ---------------------------------------------
    def get_meta(self, key: str) -> str | None:
        rows = self._query("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def set_meta(self, key: str, value: str) -> None:
        self._write(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def acquire_lock(self, name: str, owner: str, now: datetime, ttl_seconds: float) -> bool:
        with self.transaction():
            rows = self._query("SELECT owner, expires FROM locks WHERE name = ?", (name,))
            if rows and rows[0]["owner"] != owner and text_to_required_dt(rows[0]["expires"]) > now:
                return False
            expires = dt_to_text(now + timedelta(seconds=ttl_seconds))
            self._conn.execute(
                "INSERT INTO locks(name, owner, expires) VALUES (?, ?, ?) ON CONFLICT(name) "
                "DO UPDATE SET owner = excluded.owner, expires = excluded.expires",
                (name, owner, expires),
            )
            return True

    def release_lock(self, name: str, owner: str) -> None:
        self._write("DELETE FROM locks WHERE name = ? AND owner = ?", (name, owner))

    # --- items -----------------------------------------------------------
    def existing_item_ids(self, ids: Iterable[str]) -> set[str]:
        wanted = list(dict.fromkeys(ids))
        found: set[str] = set()
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            marks = ",".join("?" * len(chunk))
            found.update(r["id"] for r in self._query(f"SELECT id FROM items WHERE id IN ({marks})", chunk))
        return found

    def max_item_seq(self) -> int:
        return self._query("SELECT COALESCE(MAX(seq), 0) AS m FROM items")[0]["m"]

    def add_items(self, items: list[StoredItem]) -> list[StoredItem]:
        with self.transaction():
            existing = self.existing_item_ids(i.id for i in items)
            seq = self.max_item_seq()
            stored: list[StoredItem] = []
            for item in items:
                if item.id in existing:
                    continue
                seq += 1
                existing.add(item.id)
                stored.append(replace(item, seq=seq))
            self._conn.executemany(
                f"INSERT INTO items({_ITEM_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        i.id,
                        i.seq,
                        i.text,
                        dt_to_text(i.timestamp),
                        to_json(i.metadata),
                        vec_to_bytes(i.embedding),
                        i.model_id,
                        str(i.status),
                        i.cluster_id,
                        int(i.locked),
                        i.score,
                    )
                    for i in stored
                ],
            )
            return stored

    def update_items(self, items: list[StoredItem]) -> None:
        self._write_many(
            "UPDATE items SET status = ?, cluster_id = ?, locked = ?, score = ?, embedding = ?, "
            "model_id = ? WHERE id = ?",
            [
                (
                    str(i.status),
                    i.cluster_id,
                    int(i.locked),
                    i.score,
                    vec_to_bytes(i.embedding),
                    i.model_id,
                    i.id,
                )
                for i in items
            ],
        )

    def get_item(self, item_id: str) -> StoredItem | None:
        rows = self._query(f"SELECT {_ITEM_COLS} FROM items WHERE id = ?", (item_id,))
        return _row_to_item(rows[0]) if rows else None

    def get_items(self, ids: Iterable[str]) -> list[StoredItem]:
        wanted = list(dict.fromkeys(ids))
        by_id: dict[str, StoredItem] = {}
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            marks = ",".join("?" * len(chunk))
            for row in self._query(f"SELECT {_ITEM_COLS} FROM items WHERE id IN ({marks})", chunk):
                by_id[row["id"]] = _row_to_item(row)
        return [by_id[i] for i in wanted if i in by_id]

    def items_by_status(self, status: ItemStatus, after_seq: int = 0) -> list[StoredItem]:
        rows = self._query(
            f"SELECT {_ITEM_COLS} FROM items WHERE status = ? AND seq > ? ORDER BY seq",
            (str(status), after_seq),
        )
        return [_row_to_item(r) for r in rows]

    def cluster_items(self, cluster_id: str, limit: int | None = None) -> list[StoredItem]:
        rows = self._query(
            f"SELECT {_ITEM_COLS} FROM items WHERE cluster_id = ? ORDER BY ts DESC, seq DESC LIMIT ?",
            (cluster_id, -1 if limit is None else limit),
        )
        return [_row_to_item(r) for r in rows]

    def count_cluster_items_since(self, cluster_id: str, since: datetime) -> int:
        rows = self._query(
            "SELECT COUNT(*) AS n FROM items WHERE cluster_id = ? AND ts > ?",
            (cluster_id, dt_to_text(since)),
        )
        return rows[0]["n"]

    def count_items_by_status(self) -> dict[ItemStatus, int]:
        counts = dict.fromkeys(ItemStatus, 0)
        for row in self._query("SELECT status, COUNT(*) AS n FROM items GROUP BY status"):
            counts[ItemStatus(row["status"])] = row["n"]
        return counts

    def iter_items(self, batch_size: int = 1000) -> Iterator[StoredItem]:
        last = 0
        while True:
            rows = self._query(
                f"SELECT {_ITEM_COLS} FROM items WHERE seq > ? ORDER BY seq LIMIT ?",
                (last, batch_size),
            )
            if not rows:
                return
            for row in rows:
                yield _row_to_item(row)
            last = rows[-1]["seq"]

    # --- clusters --------------------------------------------------------
    def add_cluster(self, cluster: Cluster, exemplars: list[Exemplar]) -> None:
        with self.transaction():
            self._conn.execute(
                f"INSERT INTO clusters({_CLUSTER_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                _cluster_params(cluster),
            )
            self.set_exemplars(cluster.id, exemplars)

    def update_cluster(self, cluster: Cluster) -> None:
        params = _cluster_params(cluster)
        with self.transaction():
            cursor = self._conn.execute(
                "UPDATE clusters SET status=?, threshold=?, size=?, first_seen=?, last_seen=?, "
                "opened_at=?, label=?, closed_at=?, merged_into=?, status_locked=?, "
                "label_size=?, seen_count=? WHERE id=?",
                (*params[1:], params[0]),
            )
            if cursor.rowcount == 0:
                raise KeyError(cluster.id)

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        rows = self._query(f"SELECT {_CLUSTER_COLS} FROM clusters WHERE id = ?", (cluster_id,))
        return _row_to_cluster(rows[0]) if rows else None

    def list_clusters(self, statuses: Iterable[ClusterStatus] | None = None) -> list[Cluster]:
        if statuses is None:
            rows = self._query(f"SELECT {_CLUSTER_COLS} FROM clusters ORDER BY opened_at, id")
        else:
            wanted = [str(s) for s in statuses]
            if not wanted:
                return []
            marks = ",".join("?" * len(wanted))
            rows = self._query(
                f"SELECT {_CLUSTER_COLS} FROM clusters WHERE status IN ({marks}) ORDER BY opened_at, id",
                wanted,
            )
        return [_row_to_cluster(r) for r in rows]

    def get_exemplars(self, cluster_id: str) -> list[Exemplar]:
        rows = self._query(
            "SELECT item_id, embedding FROM exemplars WHERE cluster_id = ? ORDER BY pos",
            (cluster_id,),
        )
        return [Exemplar(r["item_id"], bytes_to_vec(r["embedding"])) for r in rows]

    def set_exemplars(self, cluster_id: str, exemplars: list[Exemplar]) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM exemplars WHERE cluster_id = ?", (cluster_id,))
            self._conn.executemany(
                "INSERT INTO exemplars(cluster_id, pos, item_id, embedding) VALUES (?,?,?,?)",
                [(cluster_id, n, e.item_id, vec_to_bytes(e.embedding)) for n, e in enumerate(exemplars)],
            )

    def add_no_merge(self, a: str, b: str) -> None:
        low, high = sorted((a, b))
        self._write("INSERT OR IGNORE INTO no_merge(a, b) VALUES (?, ?)", (low, high))

    def no_merge_pairs(self) -> set[frozenset[str]]:
        return {frozenset((r["a"], r["b"])) for r in self._query("SELECT a, b FROM no_merge")}

    # --- events ----------------------------------------------------------
    def append_events(self, events: list[Event]) -> list[Event]:
        with self.transaction():
            start = self._query("SELECT COALESCE(MAX(seq), 0) AS m FROM events")[0]["m"]
            stored = [replace(e, seq=start + n + 1) for n, e in enumerate(events)]
            self._conn.executemany(
                "INSERT INTO events(seq, type, ts, cluster_id, item_id, data) VALUES (?,?,?,?,?,?)",
                [
                    (
                        e.seq,
                        str(e.type),
                        dt_to_text(e.timestamp),
                        e.cluster_id,
                        e.item_id,
                        to_json(e.data),
                    )
                    for e in stored
                ],
            )
            return stored

    def list_events(self, after_seq: int = 0, limit: int | None = None) -> list[Event]:
        rows = self._query(
            "SELECT seq, type, ts, cluster_id, item_id, data FROM events WHERE seq > ? ORDER BY seq LIMIT ?",
            (after_seq, -1 if limit is None else limit),
        )
        return [
            Event(
                seq=r["seq"],
                type=EventType(r["type"]),
                timestamp=text_to_required_dt(r["ts"]),
                cluster_id=r["cluster_id"],
                item_id=r["item_id"],
                data=from_json(r["data"]),
            )
            for r in rows
        ]
