"""Shared fixtures: a deterministic engine on the hashing embedder and tight synthetic topics."""

from __future__ import annotations

import os
import random
from datetime import UTC, datetime, timedelta

import pytest

from toolkit.clock import SimulatedClock
from toolkit.config import Config
from toolkit.embedders import HashingEmbedder
from toolkit.engine.core import ClusteringEngine
from toolkit.events.bus import EventBus
from toolkit.models import Event, Item
from toolkit.store import MemoryStore, SQLiteStore

T0 = datetime(2026, 3, 1, tzinfo=UTC)

TOPIC_WORDS = {
    "printer": ["printer", "paper", "jam", "tray", "toner", "cartridge", "printing", "stuck"],
    "wifi": ["wifi", "router", "signal", "network", "dropping", "connection", "wireless", "bars"],
    "billing": [
        "invoice",
        "billing",
        "charged",
        "refund",
        "payment",
        "overcharged",
        "statement",
        "fee",
    ],
    "shipping": [
        "package",
        "shipping",
        "courier",
        "tracking",
        "parcel",
        "delivered",
        "lost",
        "box",
    ],
}


def topic_text(topic: str, rng: random.Random) -> str:
    words = TOPIC_WORDS[topic]
    return " ".join(rng.sample(words, 5))


def make_items(
    topic: str,
    n: int,
    start: datetime,
    spacing: timedelta = timedelta(minutes=10),
    seed: int = 0,
    prefix: str | None = None,
) -> list[Item]:
    rng = random.Random(f"{topic}-{seed}")
    tag = prefix or topic
    return [
        Item(f"{tag}-{seed}-{i}", topic_text(topic, rng), start + spacing * i, {"topic": topic})
        for i in range(n)
    ]


def test_config(**overrides) -> Config:
    base = dict(
        embedder="hashing:256",
        store="memory://",
        min_cluster_size=5,
        min_samples=3,
        threshold_bounds=(0.3, 0.95),
        merge_threshold=0.8,
        close_after="1d",
        buffer_max_age="2d",
        discovery_min_buffer=20,
        exemplars_per_cluster=16,
    )
    base.update(overrides)
    return Config(**base)


test_config.__test__ = False  # not a test


class Recorder:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def handle(self, event: Event) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [str(e.type) for e in self.events]


def build(store=None, clock=None, **overrides) -> tuple[ClusteringEngine, SimulatedClock, Recorder]:
    clock = clock or SimulatedClock(T0)
    recorder = Recorder()
    engine = ClusteringEngine(
        test_config(**overrides),
        store=store if store is not None else MemoryStore(),
        embedder=HashingEmbedder(256),
        clock=clock,
        bus=EventBus([recorder]),
    )
    return engine, clock, recorder


@pytest.fixture
def engine_bundle():
    engine, clock, recorder = build()
    yield engine, clock, recorder
    engine.close()


def _postgres_url() -> str | None:
    return os.environ.get("TOOLKIT_TEST_POSTGRES_URL")


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield MemoryStore()
        return
    if request.param == "sqlite":
        s = SQLiteStore(tmp_path / "t.db")
        yield s
        s.close()
        return
    url = _postgres_url()
    if not url:
        pytest.skip("set TOOLKIT_TEST_POSTGRES_URL to run Postgres tests")
    from toolkit.store.postgres import PostgresStore

    s = PostgresStore(url)
    s.truncate_all()
    yield s
    s.truncate_all()
    s.close()


def seed_two_topics(engine: ClusteringEngine, clock: SimulatedClock) -> list[str]:
    """Ingest two topics, discover, and return the new cluster IDs."""
    items = make_items("printer", 12, T0) + make_items("wifi", 12, T0 + timedelta(minutes=5))
    clock.advance_to(max(i.timestamp for i in items))
    engine.ingest(items)
    report = engine.discover()
    return list(report.opened)
