"""Evaluation commands: eval (labeled replay), tune (grid search), bench (performance checks)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from toolkit.cli.common import emit_json, fail, state
from toolkit.demo.dataset import sample_path
from toolkit.embedders import make_embedder
from toolkit.io import InputError, read_items


def _rows(ctx: typer.Context, data: str, **extra):
    cfg = state(ctx).config(**extra)
    path = sample_path() if data == "sample" else Path(data)
    try:
        return cfg, read_items(path, cfg.columns)
    except (InputError, FileNotFoundError) as exc:
        fail(str(exc))
    raise AssertionError("unreachable")


def _floats(text: str | None) -> list[float] | None:
    if not text:
        return None
    try:
        return [float(v) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise typer.BadParameter(f"expected comma-separated numbers, got {text!r}") from exc


def eval_command(
    ctx: typer.Context,
    data: Annotated[str, typer.Argument(help="'sample' or a labeled CSV/JSONL path.")] = "sample",
    close_after: Annotated[str | None, typer.Option(help="Override close_after, e.g. 2d.")] = None,
    min_cluster_size: Annotated[int | None, typer.Option()] = None,
    noise_label: Annotated[str, typer.Option(help="Ground-truth label that marks one-off noise.")] = "noise",
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Replay a labeled file through the engine and print quality metrics."""
    from toolkit.evaluation.metrics import format_metrics
    from toolkit.evaluation.runner import run_eval

    cfg, rows = _rows(ctx, data, close_after=close_after, min_cluster_size=min_cluster_size)
    cfg = cfg.with_overrides(store="memory://")
    try:
        result = run_eval(cfg, rows, noise_label=noise_label)
    except ValueError as exc:
        fail(str(exc))
    if as_json:
        emit_json(result.metrics.to_dict())
    else:
        typer.echo(format_metrics(result.metrics))


def tune_command(
    ctx: typer.Context,
    data: Annotated[str, typer.Argument(help="'sample' or a labeled CSV/JSONL path.")] = "sample",
    percentiles: Annotated[str | None, typer.Option(help="threshold_percentile values, e.g. 5,10")] = None,
    floors: Annotated[str | None, typer.Option(help="threshold floor values, e.g. 0.72,0.75")] = None,
    merge: Annotated[str | None, typer.Option(help="merge_threshold values, e.g. 0.84,0.86")] = None,
    min_cluster_sizes: Annotated[str | None, typer.Option(help="min_cluster_size values, e.g. 5,10")] = None,
    close_after: Annotated[str | None, typer.Option()] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Sweep thresholds and HDBSCAN settings; shows false merges vs fragmentation."""
    from toolkit.evaluation.tune import default_grid, format_trials, tune

    cfg, rows = _rows(ctx, data, close_after=close_after)
    cfg = cfg.with_overrides(store="memory://")
    grid = default_grid(cfg)
    for key, values in (
        ("threshold_percentile", _floats(percentiles)),
        ("threshold_floor", _floats(floors)),
        ("merge_threshold", _floats(merge)),
        ("min_cluster_size", [int(v) for v in _floats(min_cluster_sizes) or []] or None),
    ):
        if values:
            grid[key] = values
    total = 1
    for values in grid.values():
        total *= len(values)
    if not as_json:
        typer.echo(f"running {total} trials…", err=True)
    embedder = make_embedder(cfg.embedder, cfg.model_dir, cfg.embed_batch_size)
    results = tune(
        cfg,
        rows,
        embedder,
        grid,
        on_trial=None if as_json else lambda t: typer.echo(f"  {t.params}", err=True),
    )
    if as_json:
        emit_json([{"params": r.params, "metrics": r.metrics.to_dict(), "pareto": r.pareto} for r in results])
    else:
        typer.echo(format_trials(results))


def bench_command(
    ctx: typer.Context,
    clusters: Annotated[int, typer.Option(help="Open clusters to score against.")] = 1000,
    samples: Annotated[int, typer.Option(help="Single-item assignments to time.")] = 200,
    discovery_items: Annotated[
        int, typer.Option(help="Buffer size for the discovery timing; 0 skips it.")
    ] = 20000,
) -> None:
    """Check assignment latency and discovery time against the non-functional targets."""
    from toolkit.evaluation.perf import run_perf

    cfg = state(ctx).config().with_overrides(store="memory://")
    embedder = make_embedder(cfg.embedder, cfg.model_dir, cfg.embed_batch_size)
    report = run_perf(embedder, cfg, clusters, samples, discovery_items)
    emit_json(report.to_dict())


def register(app: typer.Typer) -> None:
    app.command("eval")(eval_command)
    app.command("tune")(tune_command)
    app.command("bench")(bench_command)
