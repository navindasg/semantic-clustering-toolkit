"""Core domain records. All records are immutable; changes produce new instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

import numpy as np


class ClusterStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    ARCHIVED = "archived"
    MERGED = "merged"


class ItemStatus(StrEnum):
    ASSIGNED = "assigned"
    BUFFERED = "buffered"
    ORPHANED = "orphaned"


class EventType(StrEnum):
    CLUSTER_OPENED = "cluster.opened"
    ITEM_ASSIGNED = "item.assigned"
    CLUSTER_CLOSED = "cluster.closed"
    CLUSTER_REOPENED = "cluster.reopened"
    CLUSTER_MERGED = "cluster.merged"
    CLUSTER_ARCHIVED = "cluster.archived"
    CLUSTER_SPLIT = "cluster.split"
    CLUSTER_RELABELED = "cluster.relabeled"
    ITEM_ORPHANED = "item.orphaned"
    ITEM_MOVED = "item.moved"


@dataclass(frozen=True)
class Item:
    """One text record as handed to the toolkit."""

    id: str
    text: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredItem:
    """An ingested item plus its embedding and assignment state."""

    id: str
    text: str
    timestamp: datetime
    metadata: dict[str, Any]
    embedding: np.ndarray
    model_id: str
    status: ItemStatus
    cluster_id: str | None
    seq: int
    locked: bool = False
    score: float | None = None


@dataclass(frozen=True)
class Exemplar:
    item_id: str
    embedding: np.ndarray


@dataclass(frozen=True)
class Cluster:
    id: str
    status: ClusterStatus
    threshold: float
    size: int
    first_seen: datetime
    last_seen: datetime
    opened_at: datetime
    label: str = ""
    closed_at: datetime | None = None
    merged_into: str | None = None
    status_locked: bool = False
    label_size: int = 0
    seen_count: int = 0
    """Members ever seen, used as the reservoir-sampling counter (never decreases on merge)."""


@dataclass(frozen=True)
class AssignmentResult:
    item_id: str
    cluster_id: str | None
    score: float | None
    runner_up_cluster_id: str | None
    runner_up_score: float | None
    duplicate: bool = False

    @property
    def buffered(self) -> bool:
        return self.cluster_id is None and not self.duplicate


@dataclass(frozen=True)
class Event:
    seq: int
    type: EventType
    timestamp: datetime
    cluster_id: str | None = None
    item_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "type": str(self.type),
            "timestamp": self.timestamp.isoformat(),
            "cluster_id": self.cluster_id,
            "item_id": self.item_id,
            "data": self.data,
        }


@dataclass(frozen=True)
class DiscoveryReport:
    snapshot_size: int
    opened: tuple[str, ...]
    folded: tuple[str, ...]
    noise: int
    reassigned: int
    skipped_reason: str | None = None
    rejected: int = 0
    """Candidate groups dropped because they were not cohesive enough."""


@dataclass(frozen=True)
class SweepReport:
    closed: tuple[str, ...]
    archived: tuple[str, ...]
    merged: tuple[tuple[str, str], ...]
    orphaned: int
    relabeled: tuple[str, ...]
