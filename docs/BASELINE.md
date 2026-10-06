# Baseline metrics and performance checks (Milestone 4)

Recorded 2026-10-05 on an Apple-silicon laptop (14 cores, CPU only), Python 3.12, with the
bundled sample stream (`toolkit eval sample`, 2,330 items, 12 true topics + 50 one-off noise
items, `close_after=2d`, `random_seed=42`). The sample is synthetic, so treat these as a
regression baseline for the engine, not as quality claims about real data. The PRD's open
question about which labeled dataset sets the real targets still stands; rerun
`toolkit eval your-labeled.csv` and `toolkit tune` once that dataset exists.

## Quality

| Metric | bge-small, mcs=5 (demo) | bge-small, mcs=10 | potion-8M, mcs=5 | potion-8M, mcs=10 |
|---|---|---|---|---|
| Assignment precision | 0.908 | 0.895 | 0.882 | 0.929 |
| False merge rate | 0.115 | 0.190 | 0.093 | 0.065 |
| Fragmentation (clusters/topic) | 2.17 | 1.75 | 3.58 | 2.58 |
| Topics detected | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 |
| Time to detect (median, h) | 6.6 | 5.4 | 5.9 | 10.7 |
| Orphan rate (all / topical) | 0.037 / 0.018 | 0.038 / 0.021 | 0.058 / 0.043 | 0.066 / 0.051 |
| Close lag (median, h) | 48.1 | 48.0 | 48.1 | 48.0 |
| Throughput (items/s, incl. embedding) | 394 | 395 | 5,612 | 6,379 |

bge-small = `fastembed:BAAI/bge-small-en-v1.5` (default), potion-8M =
`model2vec:minishlab/potion-base-8M` (`toolkit[light]`); mcs = `min_cluster_size`.

Reading the numbers:

* **Close lag equals `close_after`** (48 h) in every run: clusters close exactly one quiet
  period after their topic's last item, so stray matches are not keeping clusters alive.
* **Fragmentation is the main cost.** Most extra clusters are real sub-issues of a coarse
  synthetic topic (e.g. "address autocomplete puts the pin in the wrong place" vs "saved
  addresses disappeared"), which the merge check correctly leaves apart at the calibrated
  `merge_threshold`. Lowering `merge_threshold` trades fragmentation for false merges; that
  is the dial `toolkit tune` sweeps.
* **False merges** cluster around genuinely adjacent topics (card declined vs double charge,
  cold food vs late delivery).

### Targets for the default model (bge-small)

Set from this baseline; the bar a change must not regress on the sample stream:

| Metric | Target |
|---|---|
| Assignment precision | ≥ 0.88 |
| False merge rate | ≤ 0.15 at the demo settings |
| Topics detected | 12 / 12 |
| Fragmentation | ≤ 2.5 |
| Topic orphan rate | ≤ 0.05 |
| Close lag | within 1 h of `close_after` |

### Calibration behind the model profiles

`bge-small` similarities are compressed: same-topic pairs have median cosine 0.75 while
cross-topic p95 is 0.70. Per-topic, a pure cluster's member-to-exemplar p5 sits at 0.79–0.85
and the p99 score of other topics' items against it at 0.72–0.81. The generic 0.55 floor
therefore let loose early clusters swallow everything (precision 0.29 in the first run), so
the bge-small profile uses `threshold_bounds = (0.75, 0.92)` and `merge_threshold = 0.86`.
potion-8M's scale fits the PRD's generic 0.55–0.90 / 0.85.

## Performance against the non-functional targets

`toolkit bench --clusters 1000 --samples 200 --discovery-items 20000` (bge-small):

| Target | Measured | Status |
|---|---|---|
| Assignment p95 < 50 ms per item, embedding included, 1,000 open clusters | p50 3.7 ms, p95 4.3 ms | met |
| 20,000-item buffer discovers in < 2 min | 20.1 s (695 clusters) | met |
| No PyTorch in the default install | `torch` absent from a clean `pip install .` | met |
| Under 1 GB on disk including the default model | 407 MB environment + 64 MB model | met |
| 8 GB RAM laptop | peak RSS 1.5 GB during the 20k discovery bench | met |
| Demo: replay + report in < 5 min from three commands | ~45 s replay at `--speed 0` + ~10 s report | met |
| Reproducibility: same data/config/seed → same clusters and event log | `test_replay_gives_identical_event_log`, `test_cluster_ids_stable_across_runs` | met |
| Same replay on SQLite and Postgres | `test_sqlite_and_postgres_replays_match` (identical events and clusters) | met |

At `--speed 3600` the 8-day sample takes about 3.5 minutes of wall time by design (one
simulated hour per second); the drain after the last item is never paced.
