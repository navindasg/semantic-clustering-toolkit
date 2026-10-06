"""Plotly figures for the demo report: cluster timeline and 2D item map."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import plotly.graph_objects as go

from toolkit.models import Cluster, ClusterStatus, Event, EventType, StoredItem

# Validated categorical order (fixed, never cycled); the 9th+ cluster folds into "Other".
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#a3a29c"
UNASSIGNED = "#d6d5cf"
STATUS_COLORS = {
    ClusterStatus.OPEN: "#2a78d6",
    ClusterStatus.CLOSED: "#eb6834",
    ClusterStatus.ARCHIVED: "#a3a29c",
    ClusterStatus.MERGED: "#4a3aa7",
}
MAP_SAMPLE = 4000
TIMELINE_LIMIT = 40
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=10, r=10, t=10, b=10),
    hoverlabel=dict(font_size=12),
)


@dataclass(frozen=True)
class Segment:
    cluster_id: str
    start: datetime
    end: datetime
    open_at_end: bool


def open_segments(events: list[Event], end: datetime) -> list[Segment]:
    """Intervals during which each cluster was open, reconstructed from the event log."""
    started: dict[str, datetime] = {}
    segments: list[Segment] = []
    for event in events:
        cid = event.cluster_id
        if cid is None:
            continue
        if event.type in (EventType.CLUSTER_OPENED, EventType.CLUSTER_REOPENED):
            started.setdefault(cid, event.timestamp)
        elif event.type in (EventType.CLUSTER_CLOSED, EventType.CLUSTER_MERGED) and cid in started:
            segments.append(Segment(cid, started.pop(cid), event.timestamp, False))
    segments.extend(Segment(cid, start, end, True) for cid, start in started.items())
    return segments


def short_label(cluster: Cluster, width: int = 38) -> str:
    text = cluster.label or cluster.id
    return text if len(text) <= width else text[: width - 1] + "…"


def timeline_figure(clusters: list[Cluster], events: list[Event], end: datetime) -> go.Figure:
    by_id = {c.id: c for c in clusters}
    biggest = sorted(
        (c for c in clusters if c.status != ClusterStatus.MERGED),
        key=lambda c: (-c.size, c.id),
    )[:TIMELINE_LIMIT]
    keep = {c.id for c in biggest}
    rows = {c.id: f"{short_label(c)} ({c.id[-6:]})" for c in biggest}
    fig = go.Figure()
    for status, color in STATUS_COLORS.items():
        segs = [
            s
            for s in open_segments(events, end)
            if s.cluster_id in keep and by_id[s.cluster_id].status == status
        ]
        if not segs:
            continue
        fig.add_trace(
            go.Bar(
                name=f"now {status}",
                orientation="h",
                y=[rows[s.cluster_id] for s in segs],
                base=[s.start.isoformat() for s in segs],
                x=[(s.end - s.start).total_seconds() * 1000 for s in segs],
                marker=dict(color=color, line=dict(width=0)),
                customdata=[
                    [
                        s.cluster_id,
                        by_id[s.cluster_id].size,
                        f"{s.start:%b %d %H:%M}",
                        f"{s.end:%b %d %H:%M}",
                    ]
                    for s in segs
                ],
                hovertemplate="<b>%{y}</b><br>open %{customdata[2]} → %{customdata[3]}"
                "<br>size %{customdata[1]}<extra></extra>",
            )
        )
    order = [rows[c.id] for c in sorted(biggest, key=lambda c: c.opened_at, reverse=True)]
    fig.update_layout(
        **_LAYOUT,
        barmode="overlay",
        bargap=0.35,
        height=max(260, 22 * len(biggest) + 80),
        xaxis=dict(type="date", showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(categoryorder="array", categoryarray=order, automargin=True),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
    )
    return fig


def project_2d(vectors: np.ndarray, seed: int | None) -> np.ndarray:
    if len(vectors) < 10:
        return np.zeros((len(vectors), 2))
    import umap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(vectors) - 1),
            metric="cosine",
            min_dist=0.1,
            random_state=seed,
        )
        return reducer.fit_transform(vectors)


def map_figure(items: list[StoredItem], clusters: list[Cluster], seed: int | None) -> go.Figure:
    rng = np.random.default_rng(seed or 0)
    if len(items) > MAP_SAMPLE:
        items = [items[i] for i in sorted(rng.choice(len(items), MAP_SAMPLE, replace=False))]
    coords = project_2d(np.stack([i.embedding for i in items]), seed) if items else np.zeros((0, 2))
    ranked = sorted((c for c in clusters if c.size), key=lambda c: (-c.size, c.id))
    colored = {c.id: (short_label(c, 30), SERIES[n]) for n, c in enumerate(ranked[: len(SERIES)])}
    groups: dict[str, tuple[str, list[int]]] = {}
    for idx, item in enumerate(items):
        if item.cluster_id in colored:
            name, color = colored[item.cluster_id]
        elif item.cluster_id:
            name, color = "Other clusters", OTHER
        else:
            name, color = "Unassigned", UNASSIGNED
        groups.setdefault(name, (color, []))[1].append(idx)
    fig = go.Figure()
    order = [colored[c.id][0] for c in ranked[: len(SERIES)]] + ["Other clusters", "Unassigned"]
    for name in order:
        if name not in groups:
            continue
        color, idxs = groups[name]
        fig.add_trace(
            go.Scattergl(
                name=name,
                mode="markers",
                x=coords[idxs, 0],
                y=coords[idxs, 1],
                marker=dict(color=color, size=7, line=dict(width=0.5, color="rgba(255,255,255,0.6)")),
                text=[items[i].text for i in idxs],
                customdata=[items[i].cluster_id or "buffer" for i in idxs],
                hovertemplate="%{text}<br><i>%{customdata}</i><extra></extra>",
            )
        )
    fig.update_layout(
        **_LAYOUT,
        height=560,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        legend=dict(itemsizing="constant"),
    )
    return fig
