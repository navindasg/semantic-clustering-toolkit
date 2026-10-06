"""CLI end-to-end flows, run on the hashing embedder so no model download is needed."""

import json
from datetime import timedelta

import pytest
from conftest import T0, make_items
from typer.testing import CliRunner

from toolkit.cli import app
from toolkit.demo.dataset import Row, write_csv

runner = CliRunner()


def invoke(store, *args, input=None):
    result = runner.invoke(
        app,
        ["--store", f"sqlite:///{store}", "--embedder", "hashing:256", *args],
        input=input,
        catch_exceptions=False,
    )
    return result


@pytest.fixture
def data_file(tmp_path):
    rows = []
    for n, topic in enumerate(["printer", "wifi"]):
        for i in make_items(topic, 15, T0 + timedelta(minutes=n), seed=n):
            rows.append(Row(i.id, i.text, i.timestamp, topic))
    path = tmp_path / "items.csv"
    write_csv(sorted(rows, key=lambda r: r.timestamp), path)
    return path


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "toolkit.yaml"
    path.write_text("min_cluster_size: 5\nmin_samples: 3\nthreshold_bounds: [0.3, 0.95]\nclose_after: 6h\n")
    return path


def run(store, config_file, *args):
    result = runner.invoke(
        app,
        [
            "--config",
            str(config_file),
            "--store",
            f"sqlite:///{store}",
            "--embedder",
            "hashing:256",
            *args,
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_ingest_discover_query_flow(tmp_path, data_file, config_file):
    db = tmp_path / "t.db"
    out = run(db, config_file, "ingest", str(data_file), "--discover")
    assert "ingested 30 items" in out and "discovery opened 2" in out
    out = run(db, config_file, "ingest", str(data_file))
    assert "30 duplicates" in out
    listing = json.loads(run(db, config_file, "clusters", "list", "--json"))
    assert len(listing) == 2
    cid = listing[0]["id"]
    assert "id" in run(db, config_file, "clusters", "list", "--status", "open")
    assert cid in run(db, config_file, "clusters", "show", cid)
    detail = json.loads(run(db, config_file, "clusters", "show", cid, "--json"))
    assert detail["cluster"]["id"] == cid and detail["sample"]
    assert len(json.loads(run(db, config_file, "sample", cid, "-n", "3", "--json"))) == 3
    assert run(db, config_file, "sample", cid, "--strategy", "recent").count("\n") >= 1
    assert "cluster.opened" in run(db, config_file, "events")
    events = json.loads(run(db, config_file, "events", "--json", "--include-items", "--limit", "500"))
    assert any(e["type"] == "item.assigned" for e in events)
    assert json.loads(run(db, config_file, "stats"))["clusters"]["open"] == 2
    assert "snapshot" in run(db, config_file, "discover") or "skipped" in run(db, config_file, "discover")
    assert "closed 2" in run(db, config_file, "sweep")  # wall clock: March data is long quiet


def test_override_commands(tmp_path, data_file, config_file):
    db = tmp_path / "o.db"
    run(db, config_file, "ingest", str(data_file), "--discover")
    a, b = [c["id"] for c in json.loads(run(db, config_file, "clusters", "list", "--json"))]
    members = json.loads(run(db, config_file, "sample", a, "-n", "4", "--json"))
    assert "moved" in run(db, config_file, "override", "move", members[0]["id"], b)
    assert "split" in run(db, config_file, "override", "split", a, members[1]["id"], members[2]["id"])
    assert "closed" in run(db, config_file, "override", "close", a)
    assert "reopened" in run(db, config_file, "override", "reopen", a)
    assert "unlocked" in run(db, config_file, "override", "unlock", a)
    assert "survivor" in run(db, config_file, "override", "merge", a, b)
    bad = runner.invoke(
        app,
        ["--store", f"sqlite:///{db}", "--embedder", "hashing:256", "override", "close", "cl_nope"],
    )
    assert bad.exit_code == 1 and "unknown cluster" in bad.output


def test_bad_input_file_fails_cleanly(tmp_path, config_file):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,text\n1,a\n")
    result = runner.invoke(
        app,
        [
            "--config",
            str(config_file),
            "--embedder",
            "hashing:256",
            "--store",
            "memory://",
            "ingest",
            str(bad),
        ],
    )
    assert result.exit_code == 1 and "missing column" in result.output


def test_demo_run_and_report(tmp_path, data_file, config_file):
    """Milestone 3 exit, in miniature: install -> replay -> report in three commands."""
    db = tmp_path / "demo.db"
    out = run(
        db,
        config_file,
        "demo",
        "run",
        "--data",
        str(data_file),
        "--speed",
        "0",
        "--close-after",
        "6h",
    )
    assert "cluster.opened" in out and "done in" in out
    report = tmp_path / "report.html"
    out = run(db, config_file, "demo", "report", "--out", str(report))
    html = report.read_text()
    assert "Cluster report" in html and "Scoring against ground truth" in html
    assert "plotly" in html.lower() and "Sampled items per cluster" in html


def test_demo_report_without_run(tmp_path, config_file):
    result = runner.invoke(
        app,
        [
            "--config",
            str(config_file),
            "--store",
            f"sqlite:///{tmp_path}/none.db",
            "--embedder",
            "hashing:256",
            "demo",
            "report",
        ],
    )
    assert result.exit_code == 1 and "no demo run" in result.output


def test_eval_tune_bench(tmp_path, data_file, config_file):
    metrics = json.loads(run(tmp_path / "e.db", config_file, "eval", str(data_file), "--json"))
    assert metrics["topics_total"] == 2
    assert "Assignment precision" in run(tmp_path / "e.db", config_file, "eval", str(data_file))
    trials = json.loads(
        run(
            tmp_path / "e.db",
            config_file,
            "tune",
            str(data_file),
            "--percentiles",
            "5",
            "--floors",
            "0.3",
            "--merge",
            "0.8,0.9",
            "--json",
        )
    )
    assert len(trials) == 2
    table = run(
        tmp_path / "e.db",
        config_file,
        "tune",
        str(data_file),
        "--percentiles",
        "5",
        "--floors",
        "0.3",
        "--merge",
        "0.8",
    )
    assert "pareto" in table
    bench = json.loads(
        run(
            tmp_path / "e.db",
            config_file,
            "bench",
            "--clusters",
            "10",
            "--samples",
            "5",
            "--discovery-items",
            "0",
        )
    )
    assert bench["open_clusters"] == 10 and bench["discovery_seconds"] is None


def test_tune_rejects_bad_numbers(tmp_path, data_file, config_file):
    result = runner.invoke(
        app,
        [
            "--config",
            str(config_file),
            "--embedder",
            "hashing:256",
            "--store",
            "memory://",
            "tune",
            str(data_file),
            "--merge",
            "x,y",
        ],
    )
    assert result.exit_code != 0


def test_reembed(tmp_path, data_file, config_file):
    db = tmp_path / "r.db"
    run(db, config_file, "ingest", str(data_file), "--discover")
    out = run(db, config_file, "reembed", "--to", "hashing:128")
    assert "re-embedded 30 items" in out
    stats = runner.invoke(
        app,
        [
            "--config",
            str(config_file),
            "--store",
            f"sqlite:///{db}",
            "--embedder",
            "hashing:128",
            "stats",
        ],
        catch_exceptions=False,
    )
    assert json.loads(stats.output)["model_id"] == "hashing:128"
    mismatch = runner.invoke(
        app,
        [
            "--config",
            str(config_file),
            "--store",
            f"sqlite:///{db}",
            "--embedder",
            "hashing:256",
            "stats",
        ],
    )
    assert mismatch.exit_code != 0
