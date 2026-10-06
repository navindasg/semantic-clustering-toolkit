from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from conftest import T0, build, make_items, seed_two_topics

from toolkit.config import PreprocessConfig
from toolkit.embedders import HashingEmbedder, ModelMismatchError
from toolkit.engine.core import ClusteringEngine
from toolkit.models import Item, ItemStatus
from toolkit.store import MemoryStore


def test_items_buffer_when_no_clusters(engine_bundle):
    engine, clock, rec = engine_bundle
    results = engine.ingest(make_items("printer", 3, T0))
    assert all(r.buffered and r.score is None for r in results)
    assert engine.buffer_size() == 3
    assert rec.events == []


def test_assignment_returns_best_and_runner_up(engine_bundle):
    engine, clock, rec = engine_bundle
    printer, wifi = sorted(seed_two_topics(engine, clock))
    clusters = {c.id: c for c in engine.list_clusters()}
    new = make_items("wifi", 2, T0 + timedelta(hours=2), seed=5)
    clock.advance_to(new[-1].timestamp)
    results = engine.ingest(new)
    wifi_id = next(
        cid
        for cid, c in clusters.items()
        if "router" in c.label or "wifi" in c.label or "signal" in c.label or "network" in c.label
    )
    for r in results:
        assert r.cluster_id == wifi_id
        assert r.runner_up_cluster_id is not None and r.runner_up_cluster_id != wifi_id
        assert r.score > r.runner_up_score
    assert engine.get_cluster(wifi_id).size == 14
    assert engine.get_cluster(wifi_id).last_seen == new[-1].timestamp
    assert rec.types().count("item.assigned") >= 2


def test_unrelated_item_is_buffered(engine_bundle):
    engine, clock, _ = engine_bundle
    seed_two_topics(engine, clock)
    [result] = engine.ingest([Item("odd", "volcano lava eruption magma ash", T0 + timedelta(hours=1))])
    assert result.cluster_id is None and result.buffered
    assert result.score is not None


def test_reingest_is_noop(engine_bundle):
    engine, clock, rec = engine_bundle
    items = make_items("printer", 4, T0)
    engine.ingest(items)
    before = engine.stats()
    results = engine.ingest(items + [items[0]])
    assert all(r.duplicate for r in results)
    assert engine.stats() == before


def test_duplicate_within_batch(engine_bundle):
    engine, _, _ = engine_bundle
    item = make_items("printer", 1, T0)[0]
    results = engine.ingest([item, replace(item, text="different text")])
    assert not results[0].duplicate and results[1].duplicate
    assert engine.store.get_item(item.id).text == item.text


@pytest.mark.parametrize(
    "bad",
    [
        Item("", "text", T0),
        Item("x", "   ", T0),
    ],
)
def test_validation_errors(engine_bundle, bad):
    engine, _, _ = engine_bundle
    with pytest.raises(ValueError):
        engine.ingest([bad])


def test_validation_type_errors(engine_bundle):
    engine, _, _ = engine_bundle
    with pytest.raises(TypeError):
        engine.ingest(["not an item"])
    with pytest.raises(TypeError):
        engine.ingest([Item("x", "text", "2026-01-01")])


def test_naive_timestamps_become_utc(engine_bundle):
    engine, _, _ = engine_bundle
    engine.ingest([Item("n1", "printer paper", datetime(2026, 3, 1, 12, 0))])
    assert engine.store.get_item("n1").timestamp.tzinfo is not None


def test_reservoir_keeps_fixed_size_and_is_deterministic():
    def run():
        engine, clock, _ = build(exemplars_per_cluster=8)
        seed_two_topics(engine, clock)
        target = sorted(engine.list_clusters(), key=lambda c: c.id)[0]
        topic = "printer" if "printer" in target.label or "paper" in target.label else "wifi"
        more = make_items(topic, 40, T0 + timedelta(hours=1), seed=3)
        clock.advance_to(more[-1].timestamp)
        engine.ingest(more)
        return [e.item_id for e in engine.store.get_exemplars(target.id)], engine.get_cluster(target.id)

    first, cluster = run()
    second, _ = run()
    assert len(first) == 8
    assert first == second
    assert cluster.seen_count == cluster.size


def test_model_mismatch_refused():
    store = MemoryStore()
    engine, _, _ = build(store=store)
    engine.ingest(make_items("printer", 2, T0))
    with pytest.raises(ModelMismatchError):
        ClusteringEngine(engine.config, store=store, embedder=HashingEmbedder(128))


def test_stored_items_carry_model_id(engine_bundle):
    engine, _, _ = engine_bundle
    engine.ingest(make_items("printer", 1, T0))
    item = next(engine.store.iter_items())
    assert item.model_id == "hashing:256"
    assert item.status == ItemStatus.BUFFERED


def test_preprocessing_applied_before_embedding():
    engine, _, _ = build(preprocess=PreprocessConfig(redact_pii=True, max_chars=40))
    engine.ingest([Item("p1", "printer jam, mail me at someone@example.com please", T0)])
    stored = engine.store.get_item("p1").text
    assert "@" not in stored and len(stored) <= 40


def test_custom_preprocessor_hook():
    engine = ClusteringEngine(
        build()[0].config,
        store=MemoryStore(),
        embedder=HashingEmbedder(256),
        preprocessor=str.upper,
    )
    engine.ingest([Item("c1", "printer jam", T0)])
    assert engine.store.get_item("c1").text == "PRINTER JAM"


def test_concurrent_duplicate_is_not_counted_twice(engine_bundle, monkeypatch):
    """Two writers race on the same new ID: the loser's pre-check misses, but it must not
    touch cluster counts (review finding)."""
    engine, clock, rec = engine_bundle
    seed_two_topics(engine, clock)
    [item] = make_items("printer", 1, clock.now(), seed=77)
    [first] = engine.ingest([item])
    cid = first.cluster_id
    size = engine.get_cluster(cid).size if cid else None
    events = len(rec.events)
    monkeypatch.setattr(engine.store, "existing_item_ids", lambda ids: set())
    [second] = engine.ingest([item])
    assert second.duplicate and second.cluster_id == cid
    if cid:
        assert engine.get_cluster(cid).size == size
    assert len(rec.events) == events


def test_ingest_reembeds_if_model_swapped_mid_batch(engine_bundle):
    engine, clock, _ = engine_bundle
    swapped = HashingEmbedder(128)

    class Swapping(HashingEmbedder):
        def embed(self, texts):
            out = super().embed(texts)
            engine.ctx.embedder = swapped  # a migration commits while we embed
            return out

    engine.ctx.embedder = Swapping(256)
    engine.ingest(make_items("printer", 2, T0))
    stored = engine.store.get_item("printer-0-0")
    assert stored.model_id == "hashing:128" and stored.embedding.shape == (128,)
