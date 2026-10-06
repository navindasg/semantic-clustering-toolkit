from datetime import timedelta

from conftest import T0, build, make_items, seed_two_topics

from toolkit.models import ClusterStatus, Item, ItemStatus, StoredItem


def _topic_cluster(engine, topic):
    return next(
        c
        for c in engine.list_clusters()
        if engine.store.cluster_items(c.id, limit=1)[0].metadata.get("topic") == topic
    )


def _buffer_directly(engine, items):
    """Insert items as buffered without assignment, to force a second same-topic cluster."""
    vectors = engine.ctx.embedder.embed([i.text for i in items])
    engine.store.add_items(
        [
            StoredItem(
                i.id,
                i.text,
                i.timestamp,
                dict(i.metadata),
                v,
                "hashing:256",
                ItemStatus.BUFFERED,
                None,
                0,
            )
            for i, v in zip(items, vectors, strict=True)
        ]
    )


def test_quiet_cluster_closes_after_close_after(engine_bundle):
    engine, clock, rec = engine_bundle
    seed_two_topics(engine, clock)
    clock.advance_to(clock.now() + timedelta(hours=23))
    assert engine.sweep().closed == ()
    clock.advance_to(clock.now() + timedelta(hours=2))
    report = engine.sweep()
    assert len(report.closed) == 2
    assert all(c.status == ClusterStatus.CLOSED for c in engine.list_clusters())
    assert rec.types().count("cluster.closed") == 2


def test_keep_open_min_items_floor():
    engine, clock, _ = build(keep_open_min_items=3)
    seed_two_topics(engine, clock)
    printer = _topic_cluster(engine, "printer")
    clock.advance_to(clock.now() + timedelta(hours=20))
    stray = make_items("printer", 1, clock.now(), seed=21)
    engine.ingest(stray)  # one stray match is not enough to keep it open
    clock.advance_to(clock.now() + timedelta(hours=6))
    closed = engine.sweep().closed
    assert printer.id in closed


def test_match_inside_grace_reopens(engine_bundle):
    engine, clock, rec = engine_bundle
    seed_two_topics(engine, clock)
    printer = _topic_cluster(engine, "printer")
    clock.advance_to(clock.now() + timedelta(days=1, hours=1))
    engine.sweep()
    assert engine.get_cluster(printer.id).status == ClusterStatus.CLOSED
    clock.advance_to(clock.now() + timedelta(hours=12))
    results = engine.ingest(make_items("printer", 2, clock.now(), seed=31))
    assert all(r.cluster_id == printer.id for r in results)
    reopened = engine.get_cluster(printer.id)
    assert reopened.status == ClusterStatus.OPEN and reopened.closed_at is None
    assert "cluster.reopened" in rec.types()


def test_archive_after_grace_and_no_more_matches(engine_bundle):
    engine, clock, rec = engine_bundle
    seed_two_topics(engine, clock)
    printer = _topic_cluster(engine, "printer")
    clock.advance_to(clock.now() + timedelta(days=1, hours=1))
    engine.sweep()
    clock.advance_to(clock.now() + timedelta(days=2, hours=1))  # grace = 2 x close_after
    # past grace but before the sweep archives it: the index already excludes it
    [late] = engine.ingest(make_items("printer", 1, clock.now(), seed=41))
    assert late.cluster_id is None
    report = engine.sweep()
    assert printer.id in report.archived
    assert engine.get_cluster(printer.id).status == ClusterStatus.ARCHIVED
    assert "cluster.archived" in rec.types()


def test_converged_clusters_merge():
    engine, clock, rec = build(merge_threshold=0.99)
    first = make_items("printer", 12, T0)
    clock.advance_to(first[-1].timestamp)
    engine.ingest(first)
    engine.discover()
    second = make_items("printer", 12, clock.now(), seed=5)
    clock.advance_to(second[-1].timestamp)
    _buffer_directly(engine, second)
    engine.discover()  # merge_threshold 0.99 prevents folding, so a twin cluster opens
    assert len(engine.list_clusters(ClusterStatus.OPEN)) >= 2
    total = sum(c.size for c in engine.list_clusters())
    engine.ctx.config = engine.config.with_overrides(merge_threshold=0.3)
    report = engine.sweep()
    assert report.merged
    assert sum(c.size for c in engine.list_clusters()) == total
    absorbed, survivor = report.merged[0]
    assert engine.get_cluster(absorbed).status == ClusterStatus.MERGED
    assert engine.get_cluster(absorbed).merged_into == survivor
    assert engine.resolve(absorbed).id == survivor
    assert engine.get_cluster(survivor).size == len(engine.store.cluster_items(survivor))
    assert engine.store.cluster_items(absorbed) == []
    assert "cluster.merged" in rec.types()


def test_stale_buffer_items_become_orphans(engine_bundle):
    engine, clock, rec = engine_bundle
    engine.ingest([Item("lonely", "volcano lava eruption", T0)])
    clock.advance_to(T0 + timedelta(days=2, minutes=1))
    report = engine.sweep()
    assert report.orphaned == 1
    assert engine.store.get_item("lonely").status == ItemStatus.ORPHANED
    assert "item.orphaned" in rec.types()


def test_relabel_after_growth():
    engine, clock, rec = build(relabel_growth=1.5)
    seed_two_topics(engine, clock)
    printer = _topic_cluster(engine, "printer")
    more = make_items("printer", 10, clock.now(), seed=51)
    clock.advance_to(more[-1].timestamp)
    engine.ingest(more)
    report = engine.sweep()
    refreshed = engine.get_cluster(printer.id)
    assert refreshed.label_size == refreshed.size
    assert printer.id in report.relabeled or refreshed.label == printer.label


def test_sweep_twice_is_safe(engine_bundle):
    engine, clock, _ = engine_bundle
    seed_two_topics(engine, clock)
    clock.advance_to(clock.now() + timedelta(days=2))
    engine.sweep()
    before = (engine.list_clusters(), engine.events())
    second = engine.sweep()
    assert second.closed == () and second.archived == () and second.merged == ()
    assert (engine.list_clusters(), engine.events()) == before


def _event_log():
    engine, clock, _ = build()
    seed_two_topics(engine, clock)
    for day in range(1, 5):
        clock.advance_to(T0 + timedelta(days=day))
        engine.ingest(make_items("billing", 6, clock.now(), seed=day))
        engine.discover()
        engine.sweep()
    return [(e.seq, str(e.type), e.timestamp, e.cluster_id, e.item_id, e.data) for e in engine.events()]


def test_replay_gives_identical_event_log():
    """Milestone 2 exit: replaying the same dataset twice yields an identical event log."""
    first = _event_log()
    assert first
    assert first == _event_log()
