from datetime import timedelta

import pytest
from conftest import T0, build, make_items, seed_two_topics

import toolkit.engine.discover as discover_mod
from toolkit.models import ClusterStatus, ItemStatus


def test_discovery_opens_one_cluster_per_topic(engine_bundle):
    engine, clock, rec = engine_bundle
    opened = seed_two_topics(engine, clock)
    assert len(opened) == 2
    assert engine.buffer_size() == 0
    clusters = engine.list_clusters(ClusterStatus.OPEN)
    assert sorted(c.size for c in clusters) == [12, 12]
    for c in clusters:
        assert c.id.startswith("cl_")
        assert 0.3 <= c.threshold <= 0.95
        assert c.label
        assert len(engine.store.get_exemplars(c.id)) == min(12, engine.config.exemplars_per_cluster)
    assert rec.types().count("cluster.opened") == 2


def test_cluster_ids_stable_across_runs():
    """Milestone 1 exit: assignment + discovery twice on a fixed dataset -> same cluster IDs."""

    def run():
        engine, clock, _ = build()
        seed_two_topics(engine, clock)
        more = make_items("printer", 5, T0 + timedelta(hours=2), seed=4) + make_items(
            "billing", 12, T0 + timedelta(hours=2), seed=4
        )
        clock.advance_to(more[-1].timestamp)
        engine.ingest(more)
        engine.discover()
        return sorted((c.id, c.size) for c in engine.list_clusters())

    assert run() == run()


def test_rerun_does_not_touch_existing_clusters(engine_bundle):
    engine, clock, _ = engine_bundle
    seed_two_topics(engine, clock)
    before = {c.id: c for c in engine.list_clusters()}
    report = engine.discover()
    assert report.skipped_reason == "buffer too small"
    assert {c.id: c for c in engine.list_clusters()} == before


def test_noise_stays_buffered(engine_bundle):
    engine, clock, _ = engine_bundle
    items = make_items("printer", 12, T0)
    from toolkit.models import Item

    odd = [
        Item("n1", "volcano lava eruption", T0),
        Item("n2", "zebra savanna stripes", T0),
        Item("n3", "quantum entanglement photon", T0),
    ]
    clock.advance_to(T0 + timedelta(hours=3))
    engine.ingest(items + odd)
    report = engine.discover()
    assert len(report.opened) == 1
    assert {i.id for i in engine.store.items_by_status(ItemStatus.BUFFERED)} >= {"n1", "n2", "n3"}


def test_candidate_folds_into_overlapping_open_cluster():
    engine, clock, rec = build(merge_threshold=0.3)
    seed_two_topics(engine, clock)
    # Items forced into the buffer: same topic, but buffered by bypassing assignment.
    more = make_items("printer", 8, T0 + timedelta(hours=1), seed=11)
    clock.advance_to(more[-1].timestamp)
    vectors = engine.ctx.embedder.embed([i.text for i in more])
    from toolkit.models import StoredItem

    engine.store.add_items(
        [
            StoredItem(i.id, i.text, i.timestamp, {}, v, "hashing:256", ItemStatus.BUFFERED, None, 0)
            for i, v in zip(more, vectors, strict=True)
        ]
    )
    report = engine.discover()
    assert report.folded and not report.opened
    assert len(engine.list_clusters()) == 2
    assert any(e.data.get("folded") for e in rec.events if str(e.type) == "item.assigned")


def test_loose_candidates_are_rejected():
    engine, clock, _ = build(threshold_bounds=(0.99, 0.995))
    items = make_items("printer", 12, T0)
    clock.advance_to(items[-1].timestamp)
    engine.ingest(items)
    report = engine.discover()
    assert report.opened == () and report.rejected >= 1
    assert engine.buffer_size() == 12


def test_failed_discovery_leaves_buffer_untouched(engine_bundle, monkeypatch):
    engine, clock, rec = engine_bundle
    items = make_items("printer", 12, T0)
    clock.advance_to(items[-1].timestamp)
    engine.ingest(items)

    def boom(*args, **kwargs):
        raise RuntimeError("labeler exploded")

    monkeypatch.setattr(discover_mod, "label_clusters", boom)
    with pytest.raises(RuntimeError):
        engine.discover()
    assert engine.buffer_size() == 12
    assert engine.list_clusters() == []
    assert rec.events == []
    # the lock was released, so a later run works
    monkeypatch.undo()
    assert engine.discover().opened


def test_lock_held_elsewhere_skips(engine_bundle):
    engine, clock, _ = engine_bundle
    engine.store.acquire_lock("discovery", "other-worker", clock.now(), 3600)
    report = engine.discover()
    assert report.skipped_reason and "lock" in report.skipped_reason


def test_concurrent_call_in_process_skips(engine_bundle):
    engine, _, _ = engine_bundle
    engine._discovery_lock.acquire()
    try:
        assert engine.discover().skipped_reason == "discovery already running"
    finally:
        engine._discovery_lock.release()


def test_late_arrivals_reassigned(engine_bundle, monkeypatch):
    engine, clock, _ = engine_bundle
    items = make_items("printer", 30, T0)
    clock.advance_to(items[-1].timestamp)
    engine.ingest(items)
    late = make_items("printer", 3, T0 + timedelta(hours=6), seed=7)
    original = discover_mod.cluster_labels

    def labels_then_arrive(vectors, config):
        labels = original(vectors, config)
        engine.ingest(late)  # arrives while discovery is "running"
        return labels

    monkeypatch.setattr(discover_mod, "cluster_labels", labels_then_arrive)
    report = engine.discover()
    late_now = engine.store.get_items(i.id for i in late)
    attached = [i for i in late_now if i.status == ItemStatus.ASSIGNED]
    assert report.reassigned == len(attached) > 0
    assert all(i.cluster_id in report.opened for i in attached)


def test_fit_sample_caps_umap_input():
    engine, clock, _ = build(discovery_sample_size=12)
    items = make_items("printer", 15, T0) + make_items("wifi", 15, T0)
    clock.advance_to(T0 + timedelta(hours=3))
    engine.ingest(items)
    report = engine.discover()
    assert report.opened
    assert report.reassigned > 0  # unsampled items matched the new clusters afterwards


def test_cluster_labels_small_input_is_noise():
    import numpy as np
    from conftest import test_config

    labels = discover_mod.cluster_labels(np.eye(3, dtype=np.float32), test_config())
    assert list(labels) == [-1, -1, -1]
