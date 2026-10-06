"""Grid search over thresholds and HDBSCAN settings: the false-merge vs fragmentation trade-off."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from toolkit.config import Config
from toolkit.embedders import CachedEmbedder, Embedder
from toolkit.evaluation.metrics import Metrics
from toolkit.evaluation.runner import run_eval
from toolkit.io import LabeledItem


@dataclass(frozen=True)
class TrialResult:
    params: dict[str, Any]
    metrics: Metrics
    pareto: bool = False


def default_grid(config: Config) -> dict[str, list[Any]]:
    floor, ceiling = config.threshold_bounds
    merge = config.merge_threshold
    return {
        "threshold_percentile": [5, 10],
        "threshold_floor": [round(floor - 0.03, 3), floor, round(floor + 0.03, 3)],
        "merge_threshold": [round(merge - 0.03, 3), merge, round(min(merge + 0.03, 0.99), 3)],
    }


def _apply(config: Config, params: dict[str, Any]) -> Config:
    overrides = dict(params)
    floor = overrides.pop("threshold_floor", None)
    if floor is not None:
        overrides["threshold_bounds"] = (floor, max(floor, config.threshold_bounds[1]))
    return config.with_overrides(**overrides)


def _pareto(results: list[TrialResult]) -> list[TrialResult]:
    """Mark trials no other trial beats on both false merges and fragmentation distance from 1."""

    def cost(r: TrialResult) -> tuple[float, float]:
        return r.metrics.false_merge_rate, abs(r.metrics.fragmentation - 1.0)

    marked = []
    for r in results:
        a = cost(r)
        dominated = any((b := cost(o))[0] <= a[0] and b[1] <= a[1] and b != a for o in results if o is not r)
        marked.append(TrialResult(r.params, r.metrics, pareto=not dominated))
    return marked


def tune(
    config: Config,
    rows: Sequence[LabeledItem],
    embedder: Embedder,
    grid: dict[str, list[Any]] | None = None,
    on_trial: Callable[[TrialResult], None] | None = None,
) -> list[TrialResult]:
    """Replay the labeled rows once per grid point. Texts are embedded only once."""
    grid = grid or default_grid(config)
    cached = embedder if isinstance(embedder, CachedEmbedder) else CachedEmbedder(embedder)
    cached.warm([r.item.text for r in rows])
    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        metrics = run_eval(_apply(config, params), rows, embedder=cached).metrics
        trial = TrialResult(params, metrics)
        results.append(trial)
        if on_trial is not None:
            on_trial(trial)
    return _pareto(results)


def format_trials(results: list[TrialResult]) -> str:
    if not results:
        return "no trials"
    keys = list(results[0].params)
    header = [*keys, "precision", "false_merge", "fragment", "orphan", "detect_h", ""]
    rows = [
        [
            *(str(r.params[k]) for k in keys),
            f"{r.metrics.assignment_precision:.3f}",
            f"{r.metrics.false_merge_rate:.3f}",
            f"{r.metrics.fragmentation:.2f}",
            f"{r.metrics.orphan_rate:.3f}",
            str(r.metrics.time_to_detect_hours),
            "* pareto" if r.pareto else "",
        ]
        for r in sorted(results, key=lambda r: (r.metrics.false_merge_rate, r.metrics.fragmentation))
    ]
    widths = [max(len(x) for x in col) for col in zip(header, *rows, strict=True)]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))

    return "\n".join([line(header), line(["-" * w for w in widths]), *map(line, rows)])
