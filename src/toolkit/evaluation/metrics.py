"""Quality metrics computed by replaying labeled data through the engine."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from statistics import median

import numpy as np

from toolkit.models import ClusterStatus, EventType, ItemStatus
from toolkit.store.base import Store

DEFAULT_NOISE_LABEL = "noise"
DEFAULT_MIX_TOLERANCE = 0.10


@dataclass(frozen=True)
class Metrics:
    items: int
    clusters: int
    assignment_precision: float
    false_merge_rate: float
    fragmentation: float
    topics_detected: int
    topics_total: int
    time_to_detect_hours: float | None
    orphan_rate: float
    topic_orphan_rate: float
    close_lag_hours: float | None
    close_after_hours: float
    items_per_second: float | None = None
    p95_latency_ms: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _hours(delta: timedelta) -> float:
    return delta.total_seconds() / 3600


def _majorities(
    membership: dict[str, list[str]], noise_label: str, tolerance: float
) -> tuple[dict[str, str], int]:
    """Majority true topic per cluster, plus how many clusters mix two or more topics."""
    majority: dict[str, str] = {}
    mixed = 0
    for cluster_id, labels in membership.items():
        counts = Counter(labels)
        majority[cluster_id] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        topical = Counter(lbl for lbl in labels if lbl != noise_label)
        if len(topical) >= 2:
            second = sorted(topical.values(), reverse=True)[1]
            if second / max(1, sum(topical.values())) >= tolerance:
                mixed += 1
    return majority, mixed


def compute_metrics(
    store: Store,
    truth: dict[str, str],
    close_after: timedelta,
    *,
    noise_label: str = DEFAULT_NOISE_LABEL,
    mix_tolerance: float = DEFAULT_MIX_TOLERANCE,
    items_per_second: float | None = None,
    latencies_seconds: list[float] | None = None,
) -> Metrics:
    """Score the store's final state and event log against ground-truth labels."""
    items = [i for i in store.iter_items() if i.id in truth]
    membership: dict[str, list[str]] = defaultdict(list)
    first_seen: dict[str, datetime] = {}
    last_seen: dict[str, datetime] = {}
    for item in items:
        label = truth[item.id]
        first_seen[label] = min(first_seen.get(label, item.timestamp), item.timestamp)
        last_seen[label] = max(last_seen.get(label, item.timestamp), item.timestamp)
        if item.status == ItemStatus.ASSIGNED and item.cluster_id:
            membership[item.cluster_id].append(label)

    majority, mixed = _majorities(membership, noise_label, mix_tolerance)
    assigned = sum(len(v) for v in membership.values())
    correct = sum(sum(1 for lbl in labels if lbl == majority[cid]) for cid, labels in membership.items())
    topics = sorted({lbl for lbl in truth.values() if lbl != noise_label})
    per_topic: dict[str, list[str]] = defaultdict(list)
    for cluster_id, label in majority.items():
        if label != noise_label:
            per_topic[label].append(cluster_id)

    clusters = {c.id: c for c in store.list_clusters()}
    detect, lags = [], []
    close_times: dict[str, datetime] = {}
    for event in store.list_events():
        if event.type == EventType.CLUSTER_CLOSED and event.cluster_id:
            close_times[event.cluster_id] = event.timestamp
        elif event.type == EventType.CLUSTER_REOPENED and event.cluster_id:
            close_times.pop(event.cluster_id, None)
    for topic in topics:
        owned = [clusters[c] for c in per_topic.get(topic, []) if c in clusters]
        if not owned:
            continue
        detect.append(_hours(min(c.opened_at for c in owned) - first_seen[topic]))
        final_closes = [
            close_times[c.id]
            for c in owned
            if c.id in close_times and c.status in (ClusterStatus.CLOSED, ClusterStatus.ARCHIVED)
        ]
        if final_closes and len(final_closes) == len(owned):
            lags.append(_hours(max(final_closes) - last_seen[topic]))

    unassigned = [i for i in items if i.status != ItemStatus.ASSIGNED]
    topical_items = [i for i in items if truth[i.id] != noise_label]
    topical_unassigned = [i for i in unassigned if truth[i.id] != noise_label]
    p95 = float(np.percentile(latencies_seconds, 95) * 1000) if latencies_seconds else None
    return Metrics(
        items=len(items),
        clusters=len(membership),
        assignment_precision=round(correct / assigned, 4) if assigned else 0.0,
        false_merge_rate=round(mixed / len(membership), 4) if membership else 0.0,
        fragmentation=round(sum(len(v) for v in per_topic.values()) / len(per_topic), 3)
        if per_topic
        else 0.0,
        topics_detected=len(per_topic),
        topics_total=len(topics),
        time_to_detect_hours=round(median(detect), 2) if detect else None,
        orphan_rate=round(len(unassigned) / len(items), 4) if items else 0.0,
        topic_orphan_rate=round(len(topical_unassigned) / len(topical_items), 4) if topical_items else 0.0,
        close_lag_hours=round(median(lags), 2) if lags else None,
        close_after_hours=round(_hours(close_after), 2),
        items_per_second=round(items_per_second, 1) if items_per_second else None,
        p95_latency_ms=round(p95, 2) if p95 is not None else None,
    )


METRIC_ROWS = (
    (
        "assignment_precision",
        "Assignment precision",
        "share of assigned items matching their cluster's majority topic",
    ),
    ("false_merge_rate", "False merge rate", "share of clusters mixing two or more true topics"),
    ("fragmentation", "Fragmentation", "clusters per true topic (1.0 is ideal)"),
    ("time_to_detect_hours", "Time to detect (h)", "median first item -> cluster opened"),
    ("orphan_rate", "Orphan rate", "share of items never joining a cluster"),
    ("topic_orphan_rate", "Topic orphan rate", "same, excluding ground-truth noise items"),
    ("close_lag_hours", "Close lag (h)", "median last topic item -> cluster closed"),
    ("items_per_second", "Throughput (items/s)", "ingest throughput, embedding included"),
    ("p95_latency_ms", "p95 latency (ms/item)", "per-item assignment latency, embedding included"),
)


def format_metrics(metrics: Metrics) -> str:
    lines = [
        f"items={metrics.items} clusters={metrics.clusters} "
        f"topics detected={metrics.topics_detected}/{metrics.topics_total} "
        f"close_after={metrics.close_after_hours}h",
    ]
    values = metrics.to_dict()
    for key, title, meaning in METRIC_ROWS:
        value = values[key]
        shown = "n/a" if value is None else f"{value}"
        lines.append(f"  {title:<24} {shown:>10}   {meaning}")
    return "\n".join(lines)
