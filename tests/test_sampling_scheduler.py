import time
from datetime import timedelta

import pytest
from conftest import T0, build, make_items, seed_two_topics

from toolkit.clock import SimulatedClock, SystemClock
from toolkit.engine.sampling import counts_over_time
from toolkit.scheduler import BackgroundScheduler, Scheduler


@pytest.fixture
def seeded(engine_bundle):
    engine, clock, rec = engine_bundle
    ids = seed_two_topics(engine, clock)
    return engine, clock, sorted(ids)[0]


@pytest.mark.parametrize("strategy", ["central", "random", "recent", "mixed"])
def test_sample_strategies(seeded, strategy):
    engine, _, cid = seeded
    picked = engine.sample(cid, 5, strategy)
    assert len(picked) == 5
    assert len({i.id for i in picked}) == 5
    assert all(i.cluster_id == cid for i in picked)
    assert [i.id for i in engine.sample(cid, 5, strategy)] == [i.id for i in picked]


def test_sample_recent_is_newest_first(seeded):
    engine, _, cid = seeded
    picked = engine.sample(cid, 3, "recent")
    assert picked == sorted(picked, key=lambda i: i.timestamp, reverse=True)


def test_sample_errors(seeded):
    engine, _, cid = seeded
    with pytest.raises(KeyError):
        engine.sample("cl_missing")
    with pytest.raises(ValueError):
        engine.sample(cid, 0)
    with pytest.raises(ValueError):
        engine.sample(cid, 3, "bogus")
    assert len(engine.sample(cid, 100, "random")) == engine.get_cluster(cid).size


def test_cluster_detail_counts(seeded):
    engine, _, cid = seeded
    detail = engine.cluster_detail(cid, sample_size=4)
    assert sum(n for _, n in detail.counts_over_time) == detail.cluster.size
    assert len(detail.sample) == 4
    with pytest.raises(KeyError):
        engine.cluster_detail("cl_missing")


def test_counts_over_time_buckets():
    items = make_items("printer", 3, T0, spacing=timedelta(days=1))
    import numpy as np

    from toolkit.models import ItemStatus, StoredItem

    stored = [
        StoredItem(i.id, i.text, i.timestamp, {}, np.zeros(2), "m", ItemStatus.ASSIGNED, "c", 0)
        for i in items
    ]
    daily = counts_over_time(stored)
    assert [n for _, n in daily] == [1, 1, 1]
    assert counts_over_time([]) == []
    hourly = counts_over_time(stored[:1])
    assert hourly[0][1] == 1


def test_list_clusters_filters_and_stats(seeded):
    engine, _, cid = seeded
    assert len(engine.list_clusters("open")) == 2
    assert len(engine.list_clusters(["open", "closed"])) == 2
    assert engine.list_clusters("closed") == []
    stats = engine.stats()
    assert stats["clusters"]["open"] == 2 and stats["items"]["assigned"] == 24
    assert engine.resolve(cid).id == cid
    assert engine.resolve("cl_missing") is None


def test_scheduler_interval_and_buffer_trigger():
    engine, clock, _ = build(discovery_interval="1h", sweep_interval="10m", discovery_min_buffer=10)
    sched = Scheduler(engine)
    with pytest.raises(RuntimeError):
        sched.next_due()
    sched.start(T0)
    assert sched.next_due() == T0 + timedelta(minutes=10)
    sched.run_due(T0 + timedelta(minutes=10))
    assert sched.counters.sweeps == 1 and sched.counters.discoveries == 0
    sched.run_due(T0 + timedelta(hours=1))
    assert sched.counters.discoveries == 1
    engine.ingest(make_items("printer", 4, T0))
    assert not sched.check_buffer(T0)
    from toolkit.models import Item

    noise = [Item(f"n{i}", f"unique{i} word{i} thing{i}", T0) for i in range(8)]
    engine.ingest(noise)
    assert sched.check_buffer(T0)
    # leftover noise alone must not retrigger discovery on every tick
    assert not sched.check_buffer(T0)


def test_background_scheduler_runs_jobs():
    engine, _, _ = build(clock=SimulatedClock(T0))
    engine.ctx.clock = SystemClock()
    bg = BackgroundScheduler(engine, poll_seconds=0.01)
    bg.scheduler._sweep_interval = timedelta(milliseconds=10)
    bg.start()
    bg.start()  # idempotent
    deadline = time.time() + 5
    while bg.scheduler.counters.sweeps == 0 and time.time() < deadline:
        time.sleep(0.02)
    bg.stop()
    assert bg.scheduler.counters.sweeps >= 1
