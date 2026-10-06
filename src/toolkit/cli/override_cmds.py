"""Operator overrides (INT-4). Automation never undoes these."""

from __future__ import annotations

from typing import Annotated

import typer

from toolkit.cli.common import engine_for, fail

override_app = typer.Typer(
    help="Operator overrides: move items, merge, split, close, reopen, unlock.",
    no_args_is_help=True,
)


def _run(ctx: typer.Context, action, *args) -> object:
    with engine_for(ctx, terminal=True) as engine:
        try:
            return action(engine, *args)
        except (KeyError, ValueError) as exc:
            fail(str(exc).strip("'\""))
    return None


@override_app.command("move")
def move(ctx: typer.Context, item_id: str, cluster_id: str) -> None:
    """Move one item into a cluster and pin it there."""
    _run(ctx, lambda e, i, c: e.move_item(i, c), item_id, cluster_id)
    typer.echo(f"moved {item_id} -> {cluster_id}")


@override_app.command("merge")
def merge(
    ctx: typer.Context,
    a: str,
    b: str,
    survivor: Annotated[str | None, typer.Option(help="Which cluster survives (default: larger).")] = None,
) -> None:
    """Merge two clusters by hand."""
    kept = _run(ctx, lambda e, x, y: e.merge_clusters(x, y, survivor), a, b)
    typer.echo(f"merged; survivor {kept}")


@override_app.command("split")
def split(
    ctx: typer.Context,
    cluster_id: str,
    item_ids: Annotated[list[str], typer.Argument(help="Items to move into the new cluster.")],
) -> None:
    """Split items out of a cluster into a new one; the pair never auto-merges again."""
    new_id = _run(ctx, lambda e, c, ids: e.split_cluster(c, ids), cluster_id, item_ids)
    typer.echo(f"split {len(item_ids)} items into {new_id}")


@override_app.command("close")
def close(ctx: typer.Context, cluster_id: str) -> None:
    """Close a cluster; matches will not reopen it."""
    _run(ctx, lambda e, c: e.close_cluster(c), cluster_id)
    typer.echo(f"closed {cluster_id}")


@override_app.command("reopen")
def reopen(ctx: typer.Context, cluster_id: str) -> None:
    """Reopen a closed or archived cluster; sweeps will not close it."""
    _run(ctx, lambda e, c: e.reopen_cluster(c), cluster_id)
    typer.echo(f"reopened {cluster_id}")


@override_app.command("unlock")
def unlock(ctx: typer.Context, cluster_id: str) -> None:
    """Hand a cluster's status back to automation."""
    _run(ctx, lambda e, c: e.unlock_cluster(c), cluster_id)
    typer.echo(f"unlocked {cluster_id}")


def register(app: typer.Typer) -> None:
    app.add_typer(override_app, name="override")
