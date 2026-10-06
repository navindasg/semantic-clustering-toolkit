from datetime import timedelta

import pytest
from conftest import seed_two_topics

from toolkit.models import ClusterStatus, Item


@pytest.fixture
def seeded(engine_bundle):
    engine, clock, rec = engine_bundle
    a, b = sorted(seed_two_topics(engine, clock))
    return engine, clock, rec, a, b


def test_move_item_pins_it(seeded):
    engine, clock, rec, a, b = seeded
    item = engine.store.cluster_items(a)[0]
    engine.move_item(item.id, b)
    moved = engine.store.get_item(item.id)
    assert moved.cluster_id == b and moved.locked
    assert engine.get_cluster(a).size == 11 and engine.get_cluster(b).size == 13
    assert item.id not in [e.item_id for e in engine.store.get_exemplars(a)]
    assert "item.moved" in rec.types()


def test_move_buffered_item(seeded):
    engine, clock, _, a, _ = seeded
    engine.ingest([Item("odd", "volcano lava eruption", clock.now())])
    engine.move_item("odd", a)
    assert engine.store.get_item("odd").cluster_id == a
    assert engine.buffer_size() == 0


def test_move_to_same_cluster_just_locks(seeded):
    engine, _, _, a, _ = seeded
    item = engine.store.cluster_items(a)[0]
    engine.move_item(item.id, a)
    assert engine.store.get_item(item.id).locked
    assert engine.get_cluster(a).size == 12


def test_move_errors(seeded):
    engine, _, _, a, _ = seeded
    with pytest.raises(KeyError):
        engine.move_item("nope", a)
    with pytest.raises(KeyError):
        engine.move_item(engine.store.cluster_items(a)[0].id, "cl_missing")


def test_manual_merge(seeded):
    engine, _, rec, a, b = seeded
    survivor = engine.merge_clusters(a, b, survivor=b)
    assert survivor == b
    assert engine.get_cluster(a).status == ClusterStatus.MERGED
    assert engine.get_cluster(b).size == 24
    assert any(e.data.get("manual") for e in rec.events if str(e.type) == "cluster.merged")


def test_merge_errors(seeded):
    engine, _, _, a, b = seeded
    with pytest.raises(ValueError):
        engine.merge_clusters(a, a)
    with pytest.raises(ValueError):
        engine.merge_clusters(a, b, survivor="other")
    engine.merge_clusters(a, b)
    with pytest.raises(ValueError):
        engine.merge_clusters(a, b)  # a is merged now


def test_merge_open_into_closed_reopens_survivor(seeded):
    engine, _, _, a, b = seeded
    engine.close_cluster(b)
    engine.merge_clusters(a, b, survivor=b)
    assert engine.get_cluster(b).status == ClusterStatus.OPEN


def test_split_creates_cluster_and_blocks_remerge(seeded):
    engine, clock, rec, a, _ = seeded
    members = [m.id for m in engine.store.cluster_items(a)]
    new_id = engine.split_cluster(a, members[:4])
    assert engine.get_cluster(new_id).size == 4
    assert engine.get_cluster(a).size == 8
    assert all(engine.store.get_item(i).locked for i in members[:4])
    assert frozenset((a, new_id)) in engine.store.no_merge_pairs()
    assert "cluster.split" in rec.types()
    engine.ctx.config = engine.config.with_overrides(merge_threshold=0.01)
    assert (new_id, a) not in engine.sweep().merged and (a, new_id) not in engine.sweep().merged


def test_split_errors(seeded):
    engine, _, _, a, b = seeded
    with pytest.raises(ValueError):
        engine.split_cluster(a, [engine.store.cluster_items(b)[0].id])
    with pytest.raises(ValueError):
        engine.split_cluster(a, [m.id for m in engine.store.cluster_items(a)])


def test_manual_close_is_not_reopened_by_matches(seeded):
    engine, clock, rec, a, _ = seeded
    sample_text = engine.store.cluster_items(a)[0].text
    engine.close_cluster(a)
    assert engine.get_cluster(a).status_locked
    [result] = engine.ingest([Item("again", sample_text, clock.now() + timedelta(minutes=1))])
    assert result.cluster_id != a
    assert engine.get_cluster(a).status == ClusterStatus.CLOSED
    engine.close_cluster(a)  # closing twice is harmless
    assert rec.types().count("cluster.closed") == 1


def test_manual_reopen_is_not_closed_by_sweep(seeded):
    engine, clock, _, a, _ = seeded
    engine.close_cluster(a)
    engine.reopen_cluster(a)
    clock.advance_to(clock.now() + timedelta(days=5))
    assert a not in engine.sweep().closed
    assert engine.get_cluster(a).status == ClusterStatus.OPEN
    engine.unlock_cluster(a)
    clock.advance_to(clock.now() + timedelta(minutes=5))
    assert a in engine.sweep().closed


def test_reopen_archived(seeded):
    engine, clock, _, a, _ = seeded
    clock.advance_to(clock.now() + timedelta(days=1, hours=1))
    engine.sweep()
    clock.advance_to(clock.now() + timedelta(days=3))
    engine.sweep()
    assert engine.get_cluster(a).status == ClusterStatus.ARCHIVED
    engine.reopen_cluster(a)
    assert engine.get_cluster(a).status == ClusterStatus.OPEN


def test_override_unknown_cluster(seeded):
    engine, *_ = seeded
    for action in (engine.close_cluster, engine.reopen_cluster, engine.unlock_cluster):
        with pytest.raises(KeyError):
            action("cl_missing")


def test_repeated_split_gets_a_fresh_id(seeded):
    engine, _, _, a, _ = seeded
    members = [m.id for m in engine.store.cluster_items(a)][:3]
    first = engine.split_cluster(a, members)
    engine.merge_clusters(a, first, survivor=a)
    second = engine.split_cluster(a, members)
    assert second != first and engine.get_cluster(second).size == 3


def test_override_reads_fresh_state(seeded):
    """close_cluster must not resurrect a cluster that was merged after the caller looked."""
    engine, _, _, a, b = seeded
    engine.merge_clusters(a, b, survivor=b)
    with pytest.raises(ValueError):
        engine.close_cluster(a)
    assert engine.get_cluster(a).status == ClusterStatus.MERGED
