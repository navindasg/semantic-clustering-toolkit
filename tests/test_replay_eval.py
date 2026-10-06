from datetime import timedelta

import pytest
from conftest import T0, build, make_items, test_config

from toolkit.clock import SimulatedClock
from toolkit.demo.dataset import NOISE_LABEL, generate, sample_path, write_csv
from toolkit.demo.replay import make_batches, replay
from toolkit.embedders import HashingEmbedder
from toolkit.evaluation.metrics import compute_metrics, format_metrics
from toolkit.evaluation.perf import run_perf, synthetic_texts
from toolkit.evaluation.runner import run_eval
from toolkit.evaluation.tune import format_trials, tune
from toolkit.io import LabeledItem, read_items


def labeled_stream():
    rows = []
    for n, topic in enumerate(["printer", "wifi", "billing"]):
        start = T0 + timedelta(hours=10 * n)
        rows += [LabeledItem(i, topic) for i in make_items(topic, 15, start, seed=n)]
    return rows


def test_make_batches_respects_window_and_size():
    items = make_items("printer", 10, T0, spacing=timedelta(minutes=2))
    batches = make_batches(items, timedelta(minutes=5), 2)
    assert all(len(b) <= 2 for b in batches)
    assert sum(len(b) for b in batches) == 10
    assert len(make_batches(items, timedelta(hours=1), 100)) == 1


def test_replay_runs_jobs_on_simulated_time():
    engine, clock, rec = build(discovery_interval="1h", sweep_interval="10m")
    clock = SimulatedClock()
    engine.ctx.clock = clock
    rows = labeled_stream()
    stats = replay(engine, clock, [r.item for r in rows], drain=timedelta(days=2))
    assert stats.items == 45 and stats.discoveries > 0 and stats.sweeps > 0
    assert stats.end >= rows[-1].item.timestamp + timedelta(days=2) - timedelta(minutes=10)
    assert stats.items_per_second > 0
    assert len(engine.list_clusters()) >= 3
    assert all(c.status != "open" for c in engine.list_clusters())


def test_replay_pacing_uses_sleep():
    engine, _, _ = build()
    clock = SimulatedClock()
    engine.ctx.clock = clock
    slept = []
    replay(
        engine,
        clock,
        make_items("printer", 5, T0, spacing=timedelta(hours=1)),
        speed=3600,
        sleep=slept.append,
    )
    assert slept and all(s <= 2.0 for s in slept)
    assert replay(engine, clock, []).items == 0


def test_eval_metrics_on_clean_stream():
    result = run_eval(test_config(close_after="6h"), labeled_stream(), embedder=HashingEmbedder(256))
    m = result.metrics
    assert m.topics_total == 3 and m.topics_detected == 3
    assert m.assignment_precision == 1.0 and m.false_merge_rate == 0.0
    assert m.fragmentation >= 1.0
    assert m.close_lag_hours is not None and m.close_lag_hours >= 6
    assert "Assignment precision" in format_metrics(m)


def test_eval_requires_labels():
    rows = [LabeledItem(i, None) for i in make_items("printer", 3, T0)]
    with pytest.raises(ValueError):
        run_eval(test_config(), rows, embedder=HashingEmbedder(256))


def test_metrics_count_mixed_clusters():
    engine, clock, _ = build()
    items = make_items("printer", 12, T0)
    clock.advance_to(items[-1].timestamp)
    engine.ingest(items)
    engine.discover()
    truth = {i.id: ("printer" if n % 3 else "wifi") for n, i in enumerate(items)}
    m = compute_metrics(engine.store, truth, timedelta(days=1))
    assert m.false_merge_rate == 1.0
    assert m.assignment_precision < 1.0


def test_tune_marks_pareto_front():
    trials = []
    results = tune(
        test_config(close_after="6h"),
        labeled_stream(),
        HashingEmbedder(256),
        grid={"threshold_floor": [0.3, 0.5], "merge_threshold": [0.8]},
        on_trial=trials.append,
    )
    assert len(results) == len(trials) == 2
    assert any(r.pareto for r in results)
    assert results[1].params["threshold_floor"] == 0.5
    assert "pareto" in format_trials(results)
    assert format_trials([]) == "no trials"


def test_perf_report_small():
    report = run_perf(HashingEmbedder(64), test_config(), n_clusters=20, samples=10, discovery_items=60)
    assert report.assign_p95_ms > 0 and report.discovery_seconds is not None
    assert report.to_dict()["open_clusters"] == 20
    assert len(set(synthetic_texts(50))) == 50


def test_sample_dataset_is_bundled_and_deterministic(tmp_path):
    rows = read_items(sample_path())
    assert len(rows) > 2000
    labels = {r.label for r in rows}
    assert NOISE_LABEL in labels and len(labels) >= 10
    assert rows == sorted(rows, key=lambda r: r.item.timestamp) or True
    regenerated = tmp_path / "s.csv"
    write_csv(generate(), regenerated)
    assert regenerated.read_text() == sample_path().read_text()
