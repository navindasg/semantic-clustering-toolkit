"""Core commands: ingest, discover, sweep, clusters, sample, events, stats, reembed, serve."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from toolkit.cli.common import (
    cluster_dict,
    emit_json,
    engine_for,
    fail,
    item_dict,
    state,
    table,
    when,
)
from toolkit.clock import SystemClock
from toolkit.embedders import make_embedder
from toolkit.engine.core import MODEL_META_KEY, ClusteringEngine
from toolkit.engine.sampling import SampleStrategy
from toolkit.events.terminal import describe
from toolkit.io import InputError, read_items
from toolkit.models import ClusterStatus, EventType
from toolkit.store import open_store

INGEST_BATCH = 256


def ingest(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="CSV, TSV or JSONL file of items.")],
    discover: Annotated[bool, typer.Option(help="Run discovery after ingesting.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print per-item results as JSON.")] = False,
) -> None:
    """Ingest a batch of items (ING-1). Re-ingesting known IDs is a no-op."""
    cfg = state(ctx).config()
    try:
        rows = read_items(path, cfg.columns)
    except (InputError, FileNotFoundError) as exc:
        fail(str(exc))
    with engine_for(ctx, terminal=not as_json) as engine:
        results = []
        for start in range(0, len(rows), INGEST_BATCH):
            results.extend(engine.ingest([r.item for r in rows[start : start + INGEST_BATCH]]))
        report = engine.discover() if discover else None
    if as_json:
        emit_json([r.__dict__ for r in results])
        return
    counts = Counter(
        "duplicate" if r.duplicate else ("assigned" if r.cluster_id else "buffered") for r in results
    )
    typer.echo(
        f"ingested {len(results)} items: {counts['assigned']} assigned, "
        f"{counts['buffered']} buffered, {counts['duplicate']} duplicates"
    )
    if report is not None:
        typer.echo(f"discovery opened {len(report.opened)} clusters, folded {len(report.folded)}")


def discover(ctx: typer.Context) -> None:
    """Run discovery on the buffer now."""
    with engine_for(ctx, terminal=True) as engine:
        report = engine.discover()
    if report.skipped_reason:
        typer.echo(f"discovery skipped: {report.skipped_reason}")
        return
    typer.echo(
        f"snapshot {report.snapshot_size}: opened {len(report.opened)}, folded {len(report.folded)}, "
        f"rejected {report.rejected}, noise {report.noise}, reassigned {report.reassigned}"
    )


def sweep(ctx: typer.Context) -> None:
    """Close quiet clusters, archive expired ones, merge, and orphan stale buffer items."""
    with engine_for(ctx, terminal=True) as engine:
        report = engine.sweep()
    typer.echo(
        f"closed {len(report.closed)}, archived {len(report.archived)}, merged {len(report.merged)}, "
        f"orphaned {report.orphaned}, relabeled {len(report.relabeled)}"
    )


clusters_app = typer.Typer(help="List clusters and inspect one cluster.", no_args_is_help=True)


@clusters_app.command("list")
def clusters_list(
    ctx: typer.Context,
    status: Annotated[list[ClusterStatus] | None, typer.Option(help="Filter by status (repeatable).")] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List clusters, optionally filtered by status (INT-3)."""
    with engine_for(ctx) as engine:
        clusters = engine.list_clusters(status or None)
    if as_json:
        emit_json([cluster_dict(c) for c in clusters])
        return
    rows = [
        [c.id, str(c.status), str(c.size), when(c.first_seen), when(c.last_seen), c.label]
        for c in sorted(clusters, key=lambda c: -c.size)
    ]
    typer.echo(table(rows, ["id", "status", "size", "first seen", "last seen", "label"]))


@clusters_app.command("show")
def clusters_show(
    ctx: typer.Context,
    cluster_id: str,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """One cluster's detail: counts over time and a sample."""
    with engine_for(ctx) as engine:
        try:
            detail = engine.cluster_detail(cluster_id)
        except KeyError as exc:
            fail(str(exc))
    if as_json:
        emit_json(
            {
                "cluster": cluster_dict(detail.cluster),
                "counts_over_time": [{"start": s, "count": n} for s, n in detail.counts_over_time],
                "sample": [item_dict(i) for i in detail.sample],
            }
        )
        return
    c = detail.cluster
    typer.echo(f"{c.id}  [{c.status}]  size {c.size}  threshold {c.threshold:.3f}")
    typer.echo(f"label: {c.label}")
    typer.echo(f"seen {when(c.first_seen)} -> {when(c.last_seen)}; opened {when(c.opened_at)}")
    if c.merged_into:
        typer.echo(f"merged into {c.merged_into}")
    peak = max((n for _, n in detail.counts_over_time), default=0)
    for start, count in detail.counts_over_time:
        bar = "#" * (round(40 * count / peak) if peak else 0)
        typer.echo(f"  {when(start)}  {count:>5}  {bar}")
    typer.echo("sample:")
    for item in detail.sample:
        typer.echo(f"  - {item.text}")


def sample(
    ctx: typer.Context,
    cluster_id: str,
    n: Annotated[int, typer.Option("-n", min=1, help="How many items.")] = 10,
    strategy: Annotated[
        SampleStrategy, typer.Option(help="central, random, recent or mixed.")
    ] = SampleStrategy.MIXED,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Representative items from a cluster (SMP-1)."""
    with engine_for(ctx) as engine:
        try:
            items = engine.sample(cluster_id, n, strategy)
        except KeyError as exc:
            fail(str(exc))
    if as_json:
        emit_json([item_dict(i) for i in items])
        return
    for item in items:
        typer.echo(f"{when(item.timestamp)}  {item.id}  {item.text}")


def events(
    ctx: typer.Context,
    after: Annotated[int, typer.Option(help="Only events after this sequence number.")] = 0,
    limit: Annotated[int, typer.Option(min=1)] = 100,
    include_items: Annotated[bool, typer.Option(help="Include item.assigned events.")] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """The stored lifecycle event log."""
    with engine_for(ctx) as engine:
        found = engine.events(after)
    if not include_items:
        found = [e for e in found if e.type != EventType.ITEM_ASSIGNED]
    found = found[:limit]
    if as_json:
        emit_json([e.to_dict() for e in found])
        return
    for event in found:
        typer.echo(f"{event.seq:>6}  {describe(event)}")


def stats(ctx: typer.Context) -> None:
    """Counters: items by status and clusters by status."""
    with engine_for(ctx) as engine:
        emit_json(engine.stats())


def reembed(
    ctx: typer.Context,
    to: Annotated[str, typer.Option("--to", help="New embedder spec.")],
    to_model_dir: Annotated[str | None, typer.Option(help="Local directory for the new model.")] = None,
) -> None:
    """Migrate the store to a new embedding model (EMB-3): re-embed items and exemplars."""
    cfg = state(ctx).config()
    store = open_store(cfg.store)
    current = store.get_meta(MODEL_META_KEY) or cfg.embedder
    # The old model is never called during migration; a stand-in carrying its ID opens the store.
    engine = ClusteringEngine(
        cfg.with_overrides(embedder=current),
        store=store,
        embedder=_StoredModel(current),
        clock=SystemClock(),
    )
    try:
        count = engine.reembed(make_embedder(to, to_model_dir, cfg.embed_batch_size))
    finally:
        engine.close()
    typer.echo(f"re-embedded {count} items with {to}; set `embedder: {to}` in your config")


class _StoredModel:
    def __init__(self, model_id: str) -> None:
        self._id = model_id

    @property
    def model_id(self) -> str:
        return self._id

    def embed(self, texts):  # pragma: no cover - never called during migration
        raise RuntimeError("the previous model is not loaded during migration")


def serve(
    ctx: typer.Context,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    scheduler: Annotated[bool, typer.Option(help="Run discovery and sweeps in-process.")] = True,
) -> None:
    """Serve the HTTP API (ING-3), optionally with the in-process scheduler."""
    try:
        import uvicorn

        from toolkit.service import create_app
    except ImportError:
        fail("the HTTP service needs `uv pip install 'toolkit[service]'`")
    app = create_app(state(ctx).config(), run_scheduler=scheduler)
    uvicorn.run(app, host=host, port=port)


def register(app: typer.Typer) -> None:
    for command in (ingest, discover, sweep, sample, events, stats, reembed, serve):
        app.command()(command)
    app.add_typer(clusters_app, name="clusters")
