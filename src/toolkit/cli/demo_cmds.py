"""Demo mode: replay a stream on a simulated clock, then build a static HTML report."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from toolkit.cli.common import fail, state
from toolkit.clock import SimulatedClock
from toolkit.config import Config, format_duration, parse_duration
from toolkit.demo.dataset import sample_path
from toolkit.demo.replay import replay
from toolkit.evaluation.runner import default_drain
from toolkit.factory import build_engine
from toolkit.io import InputError, read_items

demo_app = typer.Typer(help="Demo mode: replay a stream and report on it.", no_args_is_help=True)

DEMO_STORE = "sqlite:///toolkit-demo.db"
DEMO_DEFAULTS = {"min_cluster_size": 5, "store": DEMO_STORE}
GROUND_TRUTH_KEY = "ground_truth"
STATS_META_KEY = "demo_stats"
CONFIG_META_KEY = "demo_config"


def demo_config(base: Config, **flags) -> Config:
    """Demo defaults apply only to settings the user did not set in a file or env var."""
    defaults = {k: v for k, v in DEMO_DEFAULTS.items() if k not in base.model_fields_set}
    return base.with_overrides(**defaults, **flags)


def _reset_sqlite(url: str) -> None:
    if not url.startswith("sqlite:///"):
        return
    path = Path(url.removeprefix("sqlite:///"))
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


@demo_app.command("run")
def run(
    ctx: typer.Context,
    data: Annotated[str, typer.Option(help="'sample' or a CSV/JSONL path.")] = "sample",
    speed: Annotated[
        float, typer.Option(help="Simulated seconds per real second; 0 = as fast as possible.")
    ] = 3600,
    close_after: Annotated[
        str | None, typer.Option(help="Quiet period before a cluster closes, e.g. 2d.")
    ] = None,
    drain: Annotated[
        str | None, typer.Option(help="Keep the clock running this long after the last item.")
    ] = None,
    fresh: Annotated[bool, typer.Option(help="Start from an empty demo store.")] = True,
) -> None:
    """Replay a stream through the full engine and print lifecycle events as they happen."""
    try:
        cfg = demo_config(state(ctx).config(), close_after=close_after)
        drain_for = parse_duration(drain) if drain else default_drain(cfg)
    except ValueError as exc:
        fail(str(exc))
    path = sample_path() if data == "sample" else Path(data)
    try:
        rows = read_items(path, cfg.columns)
    except (InputError, FileNotFoundError) as exc:
        fail(str(exc))
    items = [
        replace(r.item, metadata={**r.item.metadata, GROUND_TRUTH_KEY: r.label}) if r.label else r.item
        for r in rows
    ]
    if fresh:
        _reset_sqlite(cfg.store)
    typer.echo(
        f"replaying {len(items)} items from {path.name} at {speed:g}x "
        f"(close_after={format_duration(cfg.close_after)}, store={cfg.store})"
    )
    typer.echo("loading embedding model…")
    clock = SimulatedClock()
    engine = build_engine(cfg, clock=clock, terminal=True)
    started = time.perf_counter()
    try:
        stats = replay(engine, clock, items, speed=speed or None, drain=drain_for)
        engine.store.set_meta(
            STATS_META_KEY,
            json.dumps(
                {
                    "items": stats.items,
                    "batches": stats.batches,
                    "ingest_seconds": stats.ingest_seconds,
                    "discoveries": stats.discoveries,
                    "sweeps": stats.sweeps,
                    "items_per_second": stats.items_per_second,
                    "batch_latencies": list(stats.batch_latencies),
                    "wall_seconds": time.perf_counter() - started,
                    "data": str(path),
                }
            ),
        )
        engine.store.set_meta(CONFIG_META_KEY, cfg.model_dump_json())
        summary = engine.stats()
    finally:
        engine.close()
    typer.echo(
        f"\ndone in {time.perf_counter() - started:.0f}s: {summary['clusters']} clusters, "
        f"items {summary['items']}"
    )
    typer.echo("next: toolkit demo report --out report.html")


@demo_app.command("report")
def report(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option(help="Output HTML file.")] = Path("report.html"),
) -> None:
    """Write one static HTML report: cluster table, timeline, 2D map, samples, metrics."""
    try:
        from toolkit.demo.report import build_report
    except ImportError:
        fail("the report needs plotly: `uv pip install 'toolkit[demo]'`")
    cfg = demo_config(state(ctx).config())
    from toolkit.store import open_store

    store = open_store(cfg.store)
    try:
        saved = store.get_meta(CONFIG_META_KEY)
        if saved is None:
            fail(f"no demo run found in {cfg.store}; run `toolkit demo run` first")
        html = build_report(store, Config.model_validate_json(saved), store.get_meta(STATS_META_KEY))
    finally:
        store.close()
    out.write_text(html, encoding="utf-8")
    typer.echo(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
