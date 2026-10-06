# Semantic Clustering Toolkit

Group a stream of short text items into semantic clusters that keep **permanent IDs**, grow
**incrementally**, and follow a **lifecycle**: open → closed after a quiet period → reopened
inside a grace window → archived, with duplicates merged along the way.

It runs end to end on a CPU-only laptop (no PyTorch in the default install) and moves to
production by swapping adapters (store, scheduler, clock, event sink), not by rewriting the
engine. Items are just text, a timestamp, an ID and optional metadata, so the same engine fits
complaints, support tickets, feedback, alerts or any similar stream.

> "toolkit" is the PRD's placeholder name; the import package and CLI are both `toolkit`.

## Quickstart: the demo in three commands

```bash
uv pip install "semantic-clustering-toolkit[demo]"     # from a checkout: uv pip install -e ".[demo]"
toolkit demo run --data sample --speed 3600 --close-after 2d   # 1 simulated hour per second
toolkit demo report --out report.html
```

`demo run` replays the bundled synthetic stream (2,330 feedback items over 8 days, 12 hidden
topics plus one-off noise) on a simulated clock and prints every open / close / reopen / merge as
it happens. `--speed 0` replays as fast as possible (~45 s on a laptop, including the model
download on first run). `demo report` writes one static HTML file: KPI tiles, scoring against
the ground truth, a timeline of when each cluster was open, a 2D map of items, a sortable
cluster table, and sampled items per cluster.

Bring your own data with `--data path.csv` (or `.jsonl`); column names are configurable.

## How it works

Three jobs share one store:

| Job | When | What |
|---|---|---|
| **Assign** | every batch | Embed items; score each against the exemplars of open clusters and of closed clusters still inside the grace window (mean of the top-k cosine similarities). If the best score clears that cluster's own threshold the item joins it; otherwise it goes to the buffer. |
| **Discover** | hourly, or when the buffer fills | Snapshot the buffer, reduce with UMAP, cluster with HDBSCAN. Each candidate is folded into an overlapping open cluster or becomes a new cluster with a permanent ID, sampled exemplars and its own threshold. Noise stays buffered; items that arrived during the run are re-assigned afterwards. |
| **Sweep** | every few minutes | Close clusters quiet for `close_after`, archive closed clusters past `reopen_grace`, merge clusters whose exemplars converged, and turn buffered items older than `buffer_max_age` into orphans. |

All cluster state lives in raw embedding space; UMAP exists only inside discovery. That is what
keeps cluster IDs stable however often discovery reruns. Cluster IDs are derived from the
founding members, and every lifecycle decision reads time from a `Clock`, so the same data,
config and seed give the same clusters and the same event log, whether replayed or live.

## Python API

```python
from datetime import datetime, UTC
from toolkit import ClusteringEngine, Config, Item

engine = ClusteringEngine(Config(store="sqlite:///toolkit.db", close_after="2d"))
engine.bus.on(lambda e: print(e.type, e.cluster_id), types=["cluster.opened", "cluster.closed"])

results = engine.ingest([Item("t-1", "I was charged twice this month", datetime.now(UTC))])
results[0].cluster_id, results[0].score, results[0].runner_up_cluster_id   # or None -> buffered

engine.discover()          # normally run by a scheduler
engine.sweep()
engine.list_clusters("open")
engine.cluster_detail("cl_3fa2b1c9d0e4")       # counts over time + sample
engine.sample("cl_3fa2b1c9d0e4", n=10, strategy="mixed")   # central | random | recent | mixed

# operator overrides; automation never undoes them
engine.move_item("t-1", "cl_3fa2b1c9d0e4")
engine.merge_clusters("cl_a", "cl_b")
engine.split_cluster("cl_a", ["t-7", "t-9"])
engine.close_cluster("cl_a"); engine.reopen_cluster("cl_a"); engine.unlock_cluster("cl_a")
```

Lifecycle events: `cluster.opened`, `item.assigned`, `cluster.closed`, `cluster.reopened`,
`cluster.merged`, `item.orphaned`, plus `cluster.archived`, `cluster.split`, `cluster.relabeled`
and `item.moved`. Events are stored in the same transaction as the change and published to
in-process callbacks after commit; webhooks receive the same payloads.

For live use, `toolkit.scheduler.BackgroundScheduler(engine).start()` runs discovery and sweeps
in-process; in production, call `toolkit discover` / `toolkit sweep` from cron or a worker queue.

## CLI

```text
toolkit ingest items.csv [--discover]          # CSV / TSV / JSONL
toolkit discover | sweep
toolkit clusters list [--status open] [--json]
toolkit clusters show <cluster-id>
toolkit sample <cluster-id> -n 10 --strategy central
toolkit events [--include-items] [--json]
toolkit stats
toolkit override move|merge|split|close|reopen|unlock ...
toolkit reembed --to model2vec:minishlab/potion-base-8M
toolkit serve [--port 8000]                    # HTTP API + in-process scheduler
toolkit eval sample | labeled.csv              # replay labeled data, print metrics
toolkit tune sample --merge 0.84,0.86,0.88     # false merges vs fragmentation grid
toolkit bench                                  # latency / discovery-time checks
toolkit demo run | report
```

Global options: `--config toolkit.yaml`, `--store`, `--embedder`, `--model-dir`, `--log-level`.

## Configuration

One YAML file sets every parameter; `TOOLKIT_<NAME>` environment variables override it, and
CLI flags override both. See [`toolkit.example.yaml`](toolkit.example.yaml) for every key.

| Parameter | Default | Controls |
|---|---|---|
| `embedder` | `fastembed:BAAI/bge-small-en-v1.5` | model used for every embedding |
| `exemplars_per_cluster` | 32 | size of each cluster's matching sample (reservoir-sampled) |
| `match_top_k` | 5 | closest exemplars averaged when scoring |
| `threshold_percentile` | 5 | percentile of member-to-exemplar similarity used as a cluster's threshold |
| `threshold_bounds` | model profile (0.55–0.90 generic) | floor and ceiling on any threshold |
| `discovery_interval` / `discovery_min_buffer` | 1h / 200 | discovery schedule / size trigger |
| `min_cluster_size` / `min_samples` | 10 (demo 5) / 5 | HDBSCAN settings |
| `umap_components` / `umap_neighbors` | 5 / 15 | reduction inside discovery |
| `close_after` | 7d | quiet period before a cluster closes |
| `keep_open_min_items` | 1 | items needed within `close_after` to stay open |
| `reopen_grace` | 2 × `close_after` | how long a closed cluster can reopen |
| `merge_threshold` | model profile (0.85 generic) | exemplar similarity at which open clusters merge |
| `buffer_max_age` | 7d | age at which a buffered item becomes an orphan |
| `random_seed` | 42 | makes discovery reproducible |

**Similarity numbers depend on the model.** Known models carry a calibrated profile that applies
only when you leave `threshold_bounds` / `merge_threshold` unset (bge-small: 0.75–0.92, merge
0.86; potion-8M: the generic 0.55–0.90, merge 0.85). Use `toolkit tune` on your own labeled data
to set them. Discovery also refuses candidates whose own cohesion falls below the threshold
floor, so a loose group can't become a catch-all cluster.

## Embedders

| Install | Spec | Notes |
|---|---|---|
| default | `fastembed:BAAI/bge-small-en-v1.5` | ONNX Runtime, no PyTorch |
| `[light]` | `model2vec:minishlab/potion-base-8M` | NumPy only, ~15× faster; for weak machines or huge backfills |
| `[torch]` | `sentence-transformers:google/embeddinggemma-300m` | best quality, needs PyTorch |

Offline or locked-down machines: download the model once and point `model_dir` (or
`--model-dir`) at the local directory. Every vector is stored with its model ID; the engine
refuses to mix models and `toolkit reembed --to <spec>` migrates a store.

## Production

| Component | Demo | Production |
|---|---|---|
| Store | SQLite file | Postgres + pgvector (`toolkit[postgres]`, `store: postgresql://...`) |
| Exemplar index | NumPy brute force | pgvector HNSW (`exemplar_index: store`) |
| Scheduler | in-process | cron / worker queue calling `toolkit discover` / `toolkit sweep` |
| Clock | simulated replay | system clock, event time |
| Event sink | terminal + callbacks | webhooks (`webhooks:` in config, HMAC-signed, retried) |
| Labeler | c-TF-IDF | c-TF-IDF or `labeler: llm` (`toolkit[llm]`, Claude) |
| Ingestion | Python / CLI | plus HTTP (`toolkit[service]`, `toolkit serve`) |

The HTTP API (`POST /items`, `GET /clusters`, `GET /clusters/{id}`, `/sample`, `/events`,
`/stats`, `POST /discover`, `/sweep`, override endpoints) returns
`{"success": bool, "data": ..., "error": ...}`. Set `TOOLKIT_API_KEY` to require an
`X-API-Key` header, and `TOOLKIT_RATE_LIMIT` (requests/minute per client, default 600).

Discovery runs one at a time behind a store-level lock while assignment keeps running; a failed
discovery run leaves the buffer untouched. Buffers above `discovery_sample_size` (50k) are fit
on a sample and the rest is matched to the new clusters.

## Evaluation

```bash
toolkit eval sample --close-after 2d --min-cluster-size 5
toolkit tune sample --close-after 2d --merge 0.84,0.86,0.88 --floors 0.72,0.75,0.78
toolkit bench --clusters 1000 --discovery-items 20000
```

Metrics: assignment precision, false merge rate, fragmentation, time to detect, orphan rate,
close lag, throughput and p95 latency. Baselines and the performance checks against the
non-functional targets are in [`docs/BASELINE.md`](docs/BASELINE.md).

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest                      # 190+ tests, hashing embedder, no downloads
docker run -d -p 55432:5432 -e POSTGRES_USER=toolkit -e POSTGRES_PASSWORD=toolkit \
  -e POSTGRES_DB=toolkit pgvector/pgvector:pg16
TOOLKIT_TEST_POSTGRES_URL=postgresql://toolkit:toolkit@127.0.0.1:55432/toolkit \
TOOLKIT_TEST_MODELS=1 .venv/bin/python -m pytest --cov=toolkit   # adds Postgres + real models
.venv/bin/ruff check src tests
```

## Project layout

```text
src/toolkit/
  engine/      assign, discover, lifecycle (sweep/merge), overrides, sampling, scoring, core facade
  store/       interface + memory, SQLite, Postgres/pgvector adapters
  embedders/   interface + fastembed, model2vec, sentence-transformers, hashing (tests)
  events/      bus, terminal log, webhooks
  labeling/    c-TF-IDF, optional LLM labeler
  demo/        sample dataset, replay runner, HTML report
  evaluation/  metrics, labeled replay, tuning, performance checks
  cli/         Typer commands
  service.py   FastAPI endpoint
```

## Open questions carried from the PRD

The PRD's open questions are still open; this build takes these defaults: matches inside
`reopen_grace` (2 × `close_after`) reopen a closed cluster; `close_after` is fixed per config,
not scaled by cluster size; English-only default model; operators get both merge and split;
webhooks are best effort with bounded retries (failures are counted and logged).
