"""Postgres + pgvector store adapter (STO-2). Needs `toolkit[postgres]`.

Embeddings are stored as pgvector `vector` columns. Once the dimension is known, an HNSW
cosine index on exemplars backs `search_exemplars`, which the engine uses for matching when
`exemplar_index: store` is configured.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from toolkit.clock import ensure_utc
from toolkit.models import (
    Cluster,
    ClusterStatus,
    Event,
    EventType,
    Exemplar,
    ItemStatus,
    StoredItem,
)

try:
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError("the Postgres store needs `uv pip install 'toolkit[postgres]'`") from exc

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    seq BIGINT NOT NULL UNIQUE,
    text TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL,
    embedding vector NOT NULL,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL,
    cluster_id TEXT,
    locked BOOLEAN NOT NULL DEFAULT FALSE,
    score DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS items_status_seq ON items(status, seq);
CREATE INDEX IF NOT EXISTS items_cluster_ts ON items(cluster_id, ts);
CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    size INTEGER NOT NULL,
    first_seen TIMESTAMPTZ NOT NULL,
    last_seen TIMESTAMPTZ NOT NULL,
    opened_at TIMESTAMPTZ NOT NULL,
    label TEXT NOT NULL,
    closed_at TIMESTAMPTZ,
    merged_into TEXT,
    status_locked BOOLEAN NOT NULL,
    label_size INTEGER NOT NULL,
    seen_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS clusters_status ON clusters(status);
CREATE TABLE IF NOT EXISTS exemplars (
    cluster_id TEXT NOT NULL,
    pos INTEGER NOT NULL,
    item_id TEXT NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (cluster_id, pos)
);
CREATE TABLE IF NOT EXISTS events (
    seq BIGINT PRIMARY KEY,
    type TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    cluster_id TEXT,
    item_id TEXT,
    data JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS locks (name TEXT PRIMARY KEY, owner TEXT NOT NULL, expires TIMESTAMPTZ NOT NULL);
CREATE TABLE IF NOT EXISTS no_merge (a TEXT NOT NULL, b TEXT NOT NULL, PRIMARY KEY (a, b));
"""

_ITEM_COLS = "id, seq, text, ts, metadata, embedding, model_id, status, cluster_id, locked, score"
_CLUSTER_COLS = (
    "id, status, threshold, size, first_seen, last_seen, opened_at, label, closed_at, "
    "merged_into, status_locked, label_size, seen_count"
)
DIM_META_KEY = "vector_dim"
WRITE_LOCK_KEY = 0x746F6F6C6B6974  # "toolkit"


def _vec(value) -> np.ndarray:
    if hasattr(value, "to_numpy"):  # pgvector.Vector, returned for untyped vector columns
        value = value.to_numpy()
    return np.asarray(value, dtype=np.float32)


def _item(row: dict) -> StoredItem:
    return StoredItem(
        id=row["id"],
        text=row["text"],
        timestamp=ensure_utc(row["ts"]),
        metadata=row["metadata"],
        embedding=_vec(row["embedding"]),
        model_id=row["model_id"],
        status=ItemStatus(row["status"]),
        cluster_id=row["cluster_id"],
        seq=row["seq"],
        locked=row["locked"],
        score=row["score"],
    )


def _cluster(row: dict) -> Cluster:
    closed = row["closed_at"]
    return Cluster(
        id=row["id"],
        status=ClusterStatus(row["status"]),
        threshold=row["threshold"],
        size=row["size"],
        first_seen=ensure_utc(row["first_seen"]),
        last_seen=ensure_utc(row["last_seen"]),
        opened_at=ensure_utc(row["opened_at"]),
        label=row["label"],
        closed_at=ensure_utc(closed) if closed else None,
        merged_into=row["merged_into"],
        status_locked=row["status_locked"],
        label_size=row["label_size"],
        seen_count=row["seen_count"],
    )


def _cluster_params(c: Cluster) -> tuple:
    return (
        c.id,
        str(c.status),
        c.threshold,
        c.size,
        c.first_seen,
        c.last_seen,
        c.opened_at,
        c.label,
        c.closed_at,
        c.merged_into,
        c.status_locked,
        c.label_size,
        c.seen_count,
    )


class PostgresStore:
    def __init__(self, url: str) -> None:
        self._conn = psycopg.connect(url, autocommit=True, row_factory=dict_row)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._conn.execute(SCHEMA)
        self._lock = threading.RLock()
        self._depth = 0
        self._dim: int | None = None
        stored = self.get_meta(DIM_META_KEY)
        if stored:
            self._dim = int(stored)

    # --- lifecycle -------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Outer transactions take a transaction-scoped advisory lock, so engine read-modify-write
        cycles (cluster sizes, exemplars, the index version) serialize across worker processes."""
        with self._lock:
            outer = self._depth == 0
            self._depth += 1
            try:
                with self._conn.transaction():
                    if outer:
                        self._conn.execute("SELECT pg_advisory_xact_lock(%s)", (WRITE_LOCK_KEY,))
                    yield
            except BaseException:
                if outer:  # a rolled-back transaction may have dropped the dimension and index
                    stored = self.get_meta(DIM_META_KEY)
                    self._dim = int(stored) if stored else None
                raise
            finally:
                self._depth -= 1

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _rows(self, sql: str, params: Iterable = ()) -> list[dict]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def _exec(self, sql: str, params: Iterable = ()) -> int:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).rowcount

    def _many(self, sql: str, rows: list[tuple]) -> None:
        if rows:
            with self.transaction(), self._conn.cursor() as cur:
                cur.executemany(sql, rows)

    def truncate_all(self) -> None:
        """Delete every row (tests and fresh demo runs)."""
        self._exec("TRUNCATE items, clusters, exemplars, events, meta, locks, no_merge")
        self._exec("DROP INDEX IF EXISTS exemplars_hnsw")  # it is typed to the old dimension
        self._dim = None

    # --- metadata and locks ---------------------------------------------
    def get_meta(self, key: str) -> str | None:
        rows = self._rows("SELECT value FROM meta WHERE key = %s", (key,))
        return rows[0]["value"] if rows else None

    def set_meta(self, key: str, value: str) -> None:
        self._exec(
            "INSERT INTO meta(key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )

    def acquire_lock(self, name: str, owner: str, now: datetime, ttl_seconds: float) -> bool:
        expires = now + timedelta(seconds=ttl_seconds)
        rows = self._rows(
            "INSERT INTO locks(name, owner, expires) VALUES (%s, %s, %s) "
            "ON CONFLICT (name) DO UPDATE SET owner = EXCLUDED.owner, expires = EXCLUDED.expires "
            "WHERE locks.owner = EXCLUDED.owner OR locks.expires <= %s RETURNING owner",
            (name, owner, expires, now),
        )
        return bool(rows)

    def release_lock(self, name: str, owner: str) -> None:
        self._exec("DELETE FROM locks WHERE name = %s AND owner = %s", (name, owner))

    # --- vector dimension and HNSW index --------------------------------
    def _ensure_dim(self, vector: np.ndarray) -> None:
        if self._dim is not None:
            return
        dim = int(np.asarray(vector).shape[-1])
        self.set_meta(DIM_META_KEY, str(dim))
        self._exec(
            f"CREATE INDEX IF NOT EXISTS exemplars_hnsw ON exemplars "
            f"USING hnsw ((embedding::vector({dim})) vector_cosine_ops)"
        )
        self._dim = dim

    def search_exemplars(
        self, vector: np.ndarray, limit: int, now: datetime, grace: timedelta
    ) -> list[tuple[str, float]]:
        """Nearest exemplars of matchable clusters, as (cluster_id, cosine similarity)."""
        if self._dim is None:
            return []
        expr = f"e.embedding::vector({self._dim})"
        rows = self._rows(
            f"SELECT e.cluster_id, 1 - ({expr} <=> %s::vector({self._dim})) AS sim "
            "FROM exemplars e JOIN clusters c ON c.id = e.cluster_id "
            "WHERE c.status = 'open' OR (c.status = 'closed' AND NOT c.status_locked "
            "AND c.closed_at >= %s) "
            f"ORDER BY {expr} <=> %s::vector({self._dim}) LIMIT %s",
            (_vec(vector), now - grace, _vec(vector), limit),
        )
        return [(r["cluster_id"], float(r["sim"])) for r in rows]

    def exemplar_counts(self) -> dict[str, int]:
        rows = self._rows("SELECT cluster_id, COUNT(*) AS n FROM exemplars GROUP BY cluster_id")
        return {r["cluster_id"]: r["n"] for r in rows}

    # --- items -----------------------------------------------------------
    def existing_item_ids(self, ids: Iterable[str]) -> set[str]:
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return set()
        return {r["id"] for r in self._rows("SELECT id FROM items WHERE id = ANY(%s)", (wanted,))}

    def max_item_seq(self) -> int:
        return self._rows("SELECT COALESCE(MAX(seq), 0) AS m FROM items")[0]["m"]

    def add_items(self, items: list[StoredItem]) -> list[StoredItem]:
        with self.transaction():
            self._exec("LOCK TABLE items IN SHARE ROW EXCLUSIVE MODE")
            existing = self.existing_item_ids(i.id for i in items)
            seq = self.max_item_seq()
            stored = []
            for item in items:
                if item.id in existing:
                    continue
                seq += 1
                existing.add(item.id)
                stored.append(replace(item, seq=seq))
            if stored:
                self._ensure_dim(stored[0].embedding)
            self._many(
                f"INSERT INTO items({_ITEM_COLS}) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [
                    (
                        i.id,
                        i.seq,
                        i.text,
                        i.timestamp,
                        Jsonb(i.metadata),
                        _vec(i.embedding),
                        i.model_id,
                        str(i.status),
                        i.cluster_id,
                        i.locked,
                        i.score,
                    )
                    for i in stored
                ],
            )
            return stored

    def update_items(self, items: list[StoredItem]) -> None:
        self._many(
            "UPDATE items SET status=%s, cluster_id=%s, locked=%s, score=%s, embedding=%s, "
            "model_id=%s WHERE id=%s",
            [
                (
                    str(i.status),
                    i.cluster_id,
                    i.locked,
                    i.score,
                    _vec(i.embedding),
                    i.model_id,
                    i.id,
                )
                for i in items
            ],
        )

    def get_item(self, item_id: str) -> StoredItem | None:
        rows = self._rows(f"SELECT {_ITEM_COLS} FROM items WHERE id = %s", (item_id,))
        return _item(rows[0]) if rows else None

    def get_items(self, ids: Iterable[str]) -> list[StoredItem]:
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return []
        rows = self._rows(f"SELECT {_ITEM_COLS} FROM items WHERE id = ANY(%s)", (wanted,))
        by_id = {r["id"]: _item(r) for r in rows}
        return [by_id[i] for i in wanted if i in by_id]

    def items_by_status(self, status: ItemStatus, after_seq: int = 0) -> list[StoredItem]:
        rows = self._rows(
            f"SELECT {_ITEM_COLS} FROM items WHERE status = %s AND seq > %s ORDER BY seq",
            (str(status), after_seq),
        )
        return [_item(r) for r in rows]

    def cluster_items(self, cluster_id: str, limit: int | None = None) -> list[StoredItem]:
        rows = self._rows(
            f"SELECT {_ITEM_COLS} FROM items WHERE cluster_id = %s ORDER BY ts DESC, seq DESC LIMIT %s",
            (cluster_id, limit),
        )
        return [_item(r) for r in rows]

    def count_cluster_items_since(self, cluster_id: str, since: datetime) -> int:
        rows = self._rows(
            "SELECT COUNT(*) AS n FROM items WHERE cluster_id = %s AND ts > %s", (cluster_id, since)
        )
        return rows[0]["n"]

    def count_items_by_status(self) -> dict[ItemStatus, int]:
        counts = dict.fromkeys(ItemStatus, 0)
        for row in self._rows("SELECT status, COUNT(*) AS n FROM items GROUP BY status"):
            counts[ItemStatus(row["status"])] = row["n"]
        return counts

    def iter_items(self, batch_size: int = 1000) -> Iterator[StoredItem]:
        last = 0
        while True:
            rows = self._rows(
                f"SELECT {_ITEM_COLS} FROM items WHERE seq > %s ORDER BY seq LIMIT %s",
                (last, batch_size),
            )
            if not rows:
                return
            yield from (_item(r) for r in rows)
            last = rows[-1]["seq"]

    # --- clusters --------------------------------------------------------
    def add_cluster(self, cluster: Cluster, exemplars: list[Exemplar]) -> None:
        with self.transaction():
            self._exec(
                f"INSERT INTO clusters({_CLUSTER_COLS}) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                _cluster_params(cluster),
            )
            self.set_exemplars(cluster.id, exemplars)

    def update_cluster(self, cluster: Cluster) -> None:
        p = _cluster_params(cluster)
        count = self._exec(
            "UPDATE clusters SET status=%s, threshold=%s, size=%s, first_seen=%s, last_seen=%s, "
            "opened_at=%s, label=%s, closed_at=%s, merged_into=%s, status_locked=%s, "
            "label_size=%s, seen_count=%s WHERE id=%s",
            (*p[1:], p[0]),
        )
        if count == 0:
            raise KeyError(cluster.id)

    def get_cluster(self, cluster_id: str) -> Cluster | None:
        rows = self._rows(f"SELECT {_CLUSTER_COLS} FROM clusters WHERE id = %s", (cluster_id,))
        return _cluster(rows[0]) if rows else None

    def list_clusters(self, statuses: Iterable[ClusterStatus] | None = None) -> list[Cluster]:
        if statuses is None:
            rows = self._rows(f"SELECT {_CLUSTER_COLS} FROM clusters ORDER BY opened_at, id")
        else:
            wanted = [str(s) for s in statuses]
            rows = self._rows(
                f"SELECT {_CLUSTER_COLS} FROM clusters WHERE status = ANY(%s) ORDER BY opened_at, id",
                (wanted,),
            )
        return [_cluster(r) for r in rows]

    def get_exemplars(self, cluster_id: str) -> list[Exemplar]:
        rows = self._rows(
            "SELECT item_id, embedding FROM exemplars WHERE cluster_id = %s ORDER BY pos",
            (cluster_id,),
        )
        return [Exemplar(r["item_id"], _vec(r["embedding"])) for r in rows]

    def set_exemplars(self, cluster_id: str, exemplars: list[Exemplar]) -> None:
        with self.transaction():
            self._exec("DELETE FROM exemplars WHERE cluster_id = %s", (cluster_id,))
            if exemplars:
                self._ensure_dim(exemplars[0].embedding)
            self._many(
                "INSERT INTO exemplars(cluster_id, pos, item_id, embedding) VALUES (%s,%s,%s,%s)",
                [(cluster_id, n, e.item_id, _vec(e.embedding)) for n, e in enumerate(exemplars)],
            )

    def add_no_merge(self, a: str, b: str) -> None:
        low, high = sorted((a, b))
        self._exec("INSERT INTO no_merge(a, b) VALUES (%s, %s) ON CONFLICT DO NOTHING", (low, high))

    def no_merge_pairs(self) -> set[frozenset[str]]:
        return {frozenset((r["a"], r["b"])) for r in self._rows("SELECT a, b FROM no_merge")}

    # --- events ----------------------------------------------------------
    def append_events(self, events: list[Event]) -> list[Event]:
        with self.transaction():
            self._exec("LOCK TABLE events IN SHARE ROW EXCLUSIVE MODE")
            start = self._rows("SELECT COALESCE(MAX(seq), 0) AS m FROM events")[0]["m"]
            stored = [replace(e, seq=start + n + 1) for n, e in enumerate(events)]
            self._many(
                "INSERT INTO events(seq, type, ts, cluster_id, item_id, data) VALUES (%s,%s,%s,%s,%s,%s)",
                [(e.seq, str(e.type), e.timestamp, e.cluster_id, e.item_id, Jsonb(e.data)) for e in stored],
            )
            return stored

    def list_events(self, after_seq: int = 0, limit: int | None = None) -> list[Event]:
        rows = self._rows(
            "SELECT seq, type, ts, cluster_id, item_id, data FROM events WHERE seq > %s "
            "ORDER BY seq LIMIT %s",
            (after_seq, limit),
        )
        return [
            Event(
                seq=r["seq"],
                type=EventType(r["type"]),
                timestamp=ensure_utc(r["ts"]),
                cluster_id=r["cluster_id"],
                item_id=r["item_id"],
                data=r["data"],
            )
            for r in rows
        ]
