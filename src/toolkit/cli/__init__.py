"""`toolkit` command line."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from toolkit.cli import core_cmds, demo_cmds, eval_cmds, override_cmds
from toolkit.cli.common import State
from toolkit.factory import configure_logging

app = typer.Typer(
    help="Semantic clustering toolkit: stable, incremental clusters with a lifecycle.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="YAML config file.", envvar="TOOLKIT_CONFIG"),
    ] = None,
    store: Annotated[str | None, typer.Option(help="Store URL, e.g. sqlite:///toolkit.db")] = None,
    embedder: Annotated[
        str | None, typer.Option(help="Embedder spec, e.g. model2vec:minishlab/potion-base-8M")
    ] = None,
    model_dir: Annotated[
        str | None, typer.Option(help="Load the embedding model from this local directory.")
    ] = None,
    log_level: Annotated[str, typer.Option(help="Log level for structured logs on stderr.")] = "WARNING",
) -> None:
    configure_logging(log_level)
    ctx.obj = State(
        config_path=config,
        overrides={
            k: v for k, v in {"store": store, "embedder": embedder, "model_dir": model_dir}.items() if v
        },
    )


core_cmds.register(app)
override_cmds.register(app)
eval_cmds.register(app)
app.add_typer(demo_cmds.demo_app, name="demo")

__all__ = ["app"]
