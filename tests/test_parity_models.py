"""Cross-store parity (Milestone 5 exit) and real embedding backends (opt-in, downloads models)."""

import os
from datetime import timedelta

import numpy as np
import pytest
from conftest import T0, build, make_items

from toolkit.clock import SimulatedClock
from toolkit.demo.replay import replay
from toolkit.store import SQLiteStore


def _replay_on(store):
    engine, _, _ = build(store=store)
    clock = SimulatedClock()
    engine.ctx.clock = clock
    items = []
    for n, topic in enumerate(["printer", "wifi", "billing", "shipping"]):
        items += make_items(topic, 15, T0 + timedelta(hours=8 * n), seed=n)
    replay(engine, clock, items, drain=timedelta(days=3))
    events = [(e.seq, str(e.type), e.timestamp, e.cluster_id, e.item_id, e.data) for e in engine.events()]
    clusters = [(c.id, str(c.status), c.size, round(c.threshold, 5), c.label) for c in engine.list_clusters()]
    return events, clusters


def test_sqlite_and_postgres_replays_match(tmp_path):
    url = os.environ.get("TOOLKIT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("set TOOLKIT_TEST_POSTGRES_URL to run the parity check")
    from toolkit.store.postgres import PostgresStore

    pg = PostgresStore(url)
    pg.truncate_all()
    try:
        sqlite_events, sqlite_clusters = _replay_on(SQLiteStore(tmp_path / "p.db"))
        pg_events, pg_clusters = _replay_on(pg)
    finally:
        pg.truncate_all()
        pg.close()
    assert sqlite_clusters == pg_clusters
    assert sqlite_events == pg_events


models = pytest.mark.skipif(
    not os.environ.get("TOOLKIT_TEST_MODELS"),
    reason="set TOOLKIT_TEST_MODELS=1 to download real models",
)


@models
@pytest.mark.model
@pytest.mark.parametrize("spec", ["fastembed:BAAI/bge-small-en-v1.5", "model2vec:minishlab/potion-base-8M"])
def test_real_backends(spec):
    from toolkit.embedders import make_embedder

    embedder = make_embedder(spec)
    vectors = embedder.embed(["my card was charged twice", "double charge on my card", "the app crashes"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]
    assert embedder.model_id == spec
    assert embedder.embed([]).shape[0] == 0
