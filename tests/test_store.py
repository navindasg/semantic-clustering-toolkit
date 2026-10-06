"""One contract, every adapter: memory, SQLite, and Postgres (when TOOLKIT_TEST_POSTGRES_URL is set)."""

from datetime import timedelta

import numpy as np
import pytest
from conftest import T0, build, make_items

from toolkit.models import (
    Cluster,
    ClusterStatus,
    Event,
    EventType,
    Exemplar,
    ItemStatus,
    StoredItem,
)
from toolkit.store import MemoryStore, SQLiteStore, open_store


def item(i, status=ItemStatus.BUFFERED, cluster=None, ts=None):
    return StoredItem(
        id=f"i{i}",
        text=f"text {i}",
        timestamp=ts or T0 + timedelta(minutes=i),
        metadata={"k": i},
        embedding=np.full(4, i, dtype=np.float32),
        model_id="m",
        status=status,
        cluster_id=cluster,
        seq=0,
    )


def cluster(cid="c1", status=ClusterStatus.OPEN, opened=T0):
    return Cluster(
        id=cid,
        status=status,
        threshold=0.5,
        size=1,
        first_seen=T0,
        last_seen=T0,
        opened_at=opened,
        label="x",
    )


def test_items_roundtrip_and_seq(store):
    stored = store.add_items([item(1), item(2), item(1)])
    assert [s.seq for s in stored] == [1, 2]
    assert store.add_items([item(2)]) == []
    got = store.get_item("i1")
    assert got.metadata == {"k": 1} and got.timestamp == T0 + timedelta(minutes=1)
    assert np.allclose(got.embedding, 1.0)
    assert store.get_item("missing") is None
    assert store.existing_item_ids(["i1", "zz"]) == {"i1"}
    assert [i.id for i in store.get_items(["i2", "i1", "zz"])] == ["i2", "i1"]
    assert store.max_item_seq() == 2


def test_status_queries(store):
    store.add_items([item(1), item(2, ItemStatus.ASSIGNED, "c1"), item(3, ItemStatus.ASSIGNED, "c1")])
    assert [i.id for i in store.items_by_status(ItemStatus.BUFFERED)] == ["i1"]
    assert [i.id for i in store.items_by_status(ItemStatus.ASSIGNED, after_seq=2)] == ["i3"]
    assert [i.id for i in store.cluster_items("c1")] == ["i3", "i2"]
    assert len(store.cluster_items("c1", limit=1)) == 1
    assert store.count_cluster_items_since("c1", T0 + timedelta(minutes=2)) == 1
    counts = store.count_items_by_status()
    assert counts[ItemStatus.ASSIGNED] == 2 and counts[ItemStatus.ORPHANED] == 0
    assert [i.id for i in store.iter_items(batch_size=2)] == ["i1", "i2", "i3"]


def test_update_items(store):
    [stored] = store.add_items([item(1)])
    from dataclasses import replace

    store.update_items([replace(stored, status=ItemStatus.ASSIGNED, cluster_id="c9", locked=True, score=0.7)])
    got = store.get_item("i1")
    assert (got.status, got.cluster_id, got.locked, got.score, got.seq) == (
        ItemStatus.ASSIGNED,
        "c9",
        True,
        pytest.approx(0.7),
        1,
    )


def test_clusters_and_exemplars(store):
    ex = [
        Exemplar("i1", np.ones(4, dtype=np.float32)),
        Exemplar("i2", np.zeros(4, dtype=np.float32)),
    ]
    store.add_cluster(cluster("c2", opened=T0 + timedelta(hours=1)), ex)
    store.add_cluster(cluster("c1"), [])
    assert [c.id for c in store.list_clusters()] == ["c1", "c2"]
    assert [e.item_id for e in store.get_exemplars("c2")] == ["i1", "i2"]
    from dataclasses import replace

    store.update_cluster(replace(cluster("c1"), status=ClusterStatus.CLOSED, closed_at=T0))
    assert store.get_cluster("c1").closed_at == T0
    assert [c.id for c in store.list_clusters([ClusterStatus.CLOSED])] == ["c1"]
    assert store.list_clusters([]) == []
    store.set_exemplars("c2", ex[:1])
    assert len(store.get_exemplars("c2")) == 1
    with pytest.raises(KeyError):
        store.update_cluster(cluster("nope"))
    assert store.get_cluster("nope") is None


def test_meta_locks_no_merge(store):
    assert store.get_meta("k") is None
    store.set_meta("k", "1")
    store.set_meta("k", "2")
    assert store.get_meta("k") == "2"
    assert store.acquire_lock("L", "a", T0, 60)
    assert store.acquire_lock("L", "a", T0, 60)  # re-entrant for the owner
    assert not store.acquire_lock("L", "b", T0, 60)
    assert store.acquire_lock("L", "b", T0 + timedelta(seconds=61), 60)  # expired
    store.release_lock("L", "b")
    assert store.acquire_lock("L", "c", T0, 60)
    store.add_no_merge("b", "a")
    store.add_no_merge("a", "b")
    assert store.no_merge_pairs() == {frozenset(("a", "b"))}


def test_events(store):
    drafts = [
        Event(0, EventType.CLUSTER_OPENED, T0, "c1", None, {"size": 3}),
        Event(0, EventType.ITEM_ASSIGNED, T0, "c1", "i1", {"score": 0.5}),
    ]
    stored = store.append_events(drafts)
    assert [e.seq for e in stored] == [1, 2]
    assert [e.seq for e in store.append_events(drafts[:1])] == [3]
    events = store.list_events()
    assert events[0].data == {"size": 3} and events[1].item_id == "i1"
    assert [e.seq for e in store.list_events(after_seq=1, limit=1)] == [2]


def test_transaction_rolls_back(store):
    with pytest.raises(RuntimeError), store.transaction():
        store.add_items([item(1)])
        store.set_meta("k", "v")
        raise RuntimeError("boom")
    assert store.get_item("i1") is None
    assert store.get_meta("k") is None


def test_engine_runs_on_every_store(store):
    engine, clock, _ = build(store=store)
    items = make_items("printer", 12, T0) + make_items("wifi", 12, T0)
    clock.advance_to(T0 + timedelta(hours=3))
    engine.ingest(items)
    assert len(engine.discover().opened) == 2
    clock.advance_to(clock.now() + timedelta(days=2))
    assert len(engine.sweep().closed) == 2


def test_open_store_urls(tmp_path):
    assert isinstance(open_store("memory://"), MemoryStore)
    assert isinstance(open_store(f"sqlite:///{tmp_path}/x.db"), SQLiteStore)
    assert isinstance(open_store("sqlite://"), SQLiteStore)
    with pytest.raises(ValueError):
        open_store("mongodb://nope")


def test_memory_duplicate_cluster():
    store = MemoryStore()
    store.add_cluster(cluster(), [])
    with pytest.raises(ValueError):
        store.add_cluster(cluster(), [])


def test_sqlite_persists_across_connections(tmp_path):
    path = tmp_path / "p.db"
    first = SQLiteStore(path)
    first.add_items([item(1)])
    first.close()
    second = SQLiteStore(path)
    assert second.get_item("i1") is not None
    second.close()


def test_postgres_search_exemplars(store):
    if not hasattr(store, "search_exemplars"):
        pytest.skip("store has no vector index")
    near = np.array([1, 0, 0, 0], dtype=np.float32)
    store.add_cluster(cluster("c1"), [Exemplar("a", near), Exemplar("b", np.array([0, 1, 0, 0], np.float32))])
    store.add_cluster(cluster("c2", ClusterStatus.ARCHIVED), [Exemplar("c", near)])
    hits = store.search_exemplars(near, 5, T0, timedelta(days=1))
    assert hits[0] == ("c1", pytest.approx(1.0))
    assert all(cid == "c1" for cid, _ in hits)  # archived clusters are not matchable
    assert store.exemplar_counts() == {"c1": 2, "c2": 1}


def test_store_backed_index_matches_numpy(store):
    if not hasattr(store, "search_exemplars"):
        pytest.skip("store has no vector index")
    from toolkit.engine.scoring import ExemplarIndex

    engine, clock, _ = build(store=store, exemplar_index="store")
    items = make_items("printer", 12, T0) + make_items("wifi", 12, T0)
    clock.advance_to(T0 + timedelta(hours=3))
    engine.ingest(items)
    engine.discover()
    probe = engine.ctx.embedder.embed([i.text for i in make_items("printer", 5, clock.now(), seed=99)])
    cache = engine.ctx.active_index()
    assert type(cache.index).__name__ == "StoreExemplarIndex"
    brute = ExemplarIndex(
        cache.index.cluster_ids,
        [np.stack([e.embedding for e in store.get_exemplars(c)]) for c in cache.index.cluster_ids],
    )
    assert np.allclose(cache.index.scores(probe, 5), brute.scores(probe, 5), atol=1e-4)
