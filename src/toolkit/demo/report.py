"""One static HTML report: summary, metrics, timeline, 2D map, cluster table and samples."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape

from toolkit.config import Config, format_duration
from toolkit.demo.report_charts import map_figure, timeline_figure
from toolkit.engine.sampling import sample_items
from toolkit.engine.scoring import make_rng
from toolkit.evaluation.metrics import METRIC_ROWS, compute_metrics
from toolkit.models import Cluster, ClusterStatus, ItemStatus, StoredItem
from toolkit.store.base import Store

GROUND_TRUTH_KEY = "ground_truth"
SAMPLED_CLUSTERS = 30
SAMPLES_PER_CLUSTER = 6

CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;
--line:#e4e3de;--accent:#2a78d6;--open:#2a78d6;--closed:#eb6834;--archived:#8a8984;--merged:#4a3aa7}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--muted:#8a8984;--line:#2e2e2b;--accent:#3987e5;--open:#3987e5;
--closed:#d95926;--merged:#9085e9}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#2e2e2b;
--accent:#3987e5;--open:#3987e5;--closed:#d95926;--merged:#9085e9}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.5 Inter,system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:17px;margin:0 0 4px}
.sub{color:var(--ink2);margin:0 0 24px}.note{color:var(--muted);font-size:12.5px;margin:0 0 12px}
section{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile b{display:block;font-size:24px;font-variant-numeric:tabular-nums}.tile span{color:var(--ink2);font-size:12.5px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600;font-size:12.5px;cursor:pointer;white-space:nowrap;user-select:none}
td.num,th.num{text-align:right}code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink2)}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--ink2)}
.pill::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--c)}
details{border-top:1px solid var(--line);padding:10px 0}details:first-of-type{border-top:0}
summary{cursor:pointer;font-weight:600}summary small{color:var(--ink2);font-weight:400;margin-left:8px}
ul{margin:8px 0 0;padding-left:20px}li{margin:3px 0;color:var(--ink2)}
"""

SCRIPT = """
document.querySelectorAll('table.sortable th').forEach((th, col) => th.addEventListener('click', () => {
  const body = th.closest('table').tBodies[0];
  const asc = th.dataset.dir !== 'asc'; th.dataset.dir = asc ? 'asc' : 'desc';
  const key = r => { const t = r.cells[col].dataset.v ?? r.cells[col].textContent; const n = parseFloat(t);
    return isNaN(n) ? t : n; };
  [...body.rows].sort((a, b) => (key(a) > key(b) ? 1 : key(a) < key(b) ? -1 : 0) * (asc ? 1 : -1))
    .forEach(r => body.appendChild(r));
}));
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
if (dark && window.Plotly) document.querySelectorAll('.js-plotly-plot').forEach(el =>
  Plotly.relayout(el, {'font.color': '#c3c2b7', 'xaxis.gridcolor': 'rgba(255,255,255,0.08)'}));
"""


def _status_pill(status: ClusterStatus) -> str:
    return f'<span class="pill" style="--c:var(--{status})">{status}</span>'


def _tiles(stats: dict, clusters: list[Cluster], run: dict) -> str:
    by_status = {s: sum(1 for c in clusters if c.status == s) for s in ClusterStatus}
    total = sum(stats.values()) or 1
    tiles = [
        (f"{total:,}", "items replayed"),
        (f"{len(clusters) - by_status[ClusterStatus.MERGED]}", "clusters discovered"),
        (
            f"{by_status[ClusterStatus.OPEN]} / {by_status[ClusterStatus.CLOSED]} / "
            f"{by_status[ClusterStatus.ARCHIVED]}",
            "open / closed / archived",
        ),
        (f"{by_status[ClusterStatus.MERGED]}", "merged away"),
        (f"{stats[ItemStatus.ASSIGNED] / total:.0%}", "items in a cluster"),
        (f"{run.get('discoveries', '–')}", "discovery runs"),
    ]
    return (
        '<div class="tiles">'
        + "".join(f'<div class="tile"><b>{escape(v)}</b><span>{escape(k)}</span></div>' for v, k in tiles)
        + "</div>"
    )


def _metrics_section(store: Store, items: list[StoredItem], cfg: Config, run: dict) -> str:
    truth = {i.id: str(i.metadata[GROUND_TRUTH_KEY]) for i in items if i.metadata.get(GROUND_TRUTH_KEY)}
    if not truth:
        return ""
    metrics = compute_metrics(
        store,
        truth,
        cfg.close_after,
        items_per_second=run.get("items_per_second"),
        latencies_seconds=run.get("batch_latencies"),
    ).to_dict()
    rows = "".join(
        f"<tr><td>{escape(title)}</td><td class='num'>{'n/a' if metrics[key] is None else metrics[key]}"
        f"</td><td>{escape(meaning)}</td></tr>"
        for key, title, meaning in METRIC_ROWS
    )
    return (
        "<section><h2>Scoring against ground truth</h2>"
        f"<p class='note'>{metrics['topics_detected']} of {metrics['topics_total']} true topics detected. "
        "Similarity settings depend on the model; use <code>toolkit tune</code> to trade false merges "
        "against fragmentation.</p><div class='scroll'><table><thead><tr><th>Metric</th>"
        f"<th class='num'>Value</th><th>Meaning</th></tr></thead><tbody>{rows}</tbody></table></div></section>"
    )


def _cluster_table(clusters: list[Cluster]) -> str:
    rows = []
    for c in sorted(clusters, key=lambda c: (-c.size, c.id)):
        if c.status == ClusterStatus.MERGED:
            continue
        rows.append(
            f"<tr><td><code>{c.id}</code></td><td data-v='{c.status}'>{_status_pill(c.status)}</td>"
            f"<td class='num'>{c.size}</td><td>{escape(c.label)}</td>"
            f"<td data-v='{c.first_seen.isoformat()}'>{c.first_seen:%b %d %H:%M}</td>"
            f"<td data-v='{c.last_seen.isoformat()}'>{c.last_seen:%b %d %H:%M}</td>"
            f"<td class='num'>{c.threshold:.3f}</td></tr>"
        )
    return (
        "<section><h2>Clusters</h2><p class='note'>Click a column to sort. Merged clusters are omitted; "
        "their items live in the survivor.</p><div class='scroll'><table class='sortable'><thead><tr>"
        "<th>ID</th><th>Status</th><th class='num'>Size</th><th>Label (c-TF-IDF)</th><th>First seen</th>"
        f"<th>Last seen</th><th class='num'>Threshold</th></tr></thead><tbody>{''.join(rows)}</tbody>"
        "</table></div></section>"
    )


def _samples(store: Store, clusters: list[Cluster], seed: int | None) -> str:
    blocks = []
    live = [c for c in clusters if c.size and c.status != ClusterStatus.MERGED]
    for c in sorted(live, key=lambda c: (-c.size, c.id))[:SAMPLED_CLUSTERS]:
        members = store.cluster_items(c.id)
        picked = sample_items(members, SAMPLES_PER_CLUSTER, "mixed", make_rng(seed, "report", c.id))
        items = "".join(f"<li>{escape(i.text)}</li>" for i in picked)
        blocks.append(
            f"<details><summary>{escape(c.label or c.id)}<small>{c.size} items · {c.status}</small>"
            f"</summary><ul>{items}</ul></details>"
        )
    return (
        "<section><h2>Sampled items per cluster</h2><p class='note'>Mixed sample: central, recent and "
        f"random members of the {len(blocks)} largest clusters.</p>{''.join(blocks)}</section>"
    )


def build_report(store: Store, cfg: Config, run_json: str | None) -> str:
    run = json.loads(run_json) if run_json else {}
    clusters = store.list_clusters()
    items = list(store.iter_items())
    events = store.list_events()
    end = max((e.timestamp for e in events), default=datetime.now(UTC))
    stats = store.count_items_by_status()
    timeline = timeline_figure(clusters, events, end)
    live = [c for c in clusters if c.status != ClusterStatus.MERGED]
    scatter = map_figure(items, live, cfg.random_seed) if items else None
    timeline_html = timeline.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False})
    map_html = (
        scatter.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})
        if scatter
        else ""
    )
    model = store.get_meta("model_id") or cfg.embedder
    span = ""
    if items:
        first = min(i.timestamp for i in items)
        last = max(i.timestamp for i in items)
        span = f"{first:%b %d} – {last:%b %d, %Y} · "
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Cluster Report</title>
<style>{CSS}</style></head><body><main>
<h1>Cluster report</h1>
<p class="sub">{escape(span)}model <code>{escape(model)}</code> · close_after {format_duration(cfg.close_after)} ·
reopen grace {format_duration(cfg.effective_reopen_grace)} · min_cluster_size {cfg.min_cluster_size}</p>
{_tiles(stats, clusters, run)}
{_metrics_section(store, items, cfg, run)}
<section><h2>When each cluster was open</h2><p class="note">Largest {min(40, len(live))} clusters,
one bar per open interval (a gap means it closed and later reopened). Color is the cluster's status now.</p>
{timeline_html}</section>
<section><h2>Items in 2D</h2><p class="note">UMAP projection of item embeddings (up to 4,000 items),
colored by cluster; the 8 largest get their own color. Hover a point to read it.</p>{map_html}</section>
{_cluster_table(clusters)}
{_samples(store, clusters, cfg.random_seed)}
</main><script>{SCRIPT}</script></body></html>"""
