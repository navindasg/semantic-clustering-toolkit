"""Shared CLI plumbing: config resolution, engine construction, output helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from toolkit.config import Config, load_config
from toolkit.engine.core import ClusteringEngine
from toolkit.factory import build_engine
from toolkit.models import Cluster, StoredItem


@dataclass
class State:
    config_path: Path | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def config(self, **extra: Any) -> Config:
        try:
            return load_config(self.config_path, **{**self.overrides, **extra})
        except (ValueError, FileNotFoundError) as exc:
            raise typer.BadParameter(str(exc)) from exc


def state(ctx: typer.Context) -> State:
    if ctx.obj is None:
        ctx.obj = State()
    return ctx.obj


@contextmanager
def engine_for(ctx: typer.Context, terminal: bool = False, **extra: Any) -> Iterator[ClusteringEngine]:
    engine = build_engine(state(ctx).config(**extra), terminal=terminal)
    try:
        yield engine
    finally:
        engine.close()


def fail(message: str, code: int = 1) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def emit_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=_default, sort_keys=True))


def cluster_dict(cluster: Cluster) -> dict[str, Any]:
    return {
        "id": cluster.id,
        "status": str(cluster.status),
        "size": cluster.size,
        "label": cluster.label,
        "threshold": cluster.threshold,
        "first_seen": cluster.first_seen,
        "last_seen": cluster.last_seen,
        "opened_at": cluster.opened_at,
        "closed_at": cluster.closed_at,
        "merged_into": cluster.merged_into,
        "status_locked": cluster.status_locked,
    }


def item_dict(item: StoredItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "text": item.text,
        "timestamp": item.timestamp,
        "status": str(item.status),
        "cluster_id": item.cluster_id,
        "score": item.score,
        "metadata": item.metadata,
    }


def table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    widths[-1] = min(widths[-1], 70)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*["-" * w for w in widths])]
    lines.extend(fmt.format(*[str(c)[:w] for c, w in zip(r, widths, strict=True)]) for r in rows)
    return "\n".join(lines)


def when(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value else "-"
