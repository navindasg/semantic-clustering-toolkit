"""Synthetic demo stream: bursty topics with hidden ground-truth labels, plus one-off noise.

Topics arrive in bursts and go quiet, some come back later, so every lifecycle state shows up
when replayed with close_after=2d.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path

from toolkit.demo.topics import NOISE, PREFIXES, SLOTS, SUFFIXES, TOPICS

START = datetime(2026, 9, 1, tzinfo=UTC)
NOISE_LABEL = "noise"
SAMPLE_FILE = "sample.csv"

# topic -> list of (start_hour, duration_hours, count)
SCHEDULE: dict[str, list[tuple[float, float, int]]] = {
    "double_charge": [(6, 30, 220), (110, 18, 90)],
    "login_loop": [(2, 50, 260)],
    "late_delivery": [(0, 190, 300)],
    "missing_items": [(20, 120, 200)],
    "promo_code": [(48, 20, 150)],
    "app_crash": [(72, 36, 260)],
    "refund_delay": [(30, 140, 160)],
    "driver_rude": [(10, 180, 60)],
    "address_bug": [(96, 40, 140)],
    "notification_spam": [(130, 30, 120)],
    "card_declined": [(24, 16, 140), (150, 12, 60)],
    "cold_food": [(40, 100, 120)],
}
SPAN_HOURS = 192


@dataclass(frozen=True)
class Row:
    id: str
    text: str
    timestamp: datetime
    label: str


def _typo(text: str, rng: random.Random) -> str:
    letters = [i for i, ch in enumerate(text[:-1]) if ch.isalpha() and text[i + 1].isalpha()]
    if not letters:
        return text
    i = rng.choice(letters)
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def _render(template: str, rng: random.Random) -> str:
    filled = template.format(**{k: rng.choice(v) for k, v in SLOTS.items()})
    text = f"{rng.choice(PREFIXES)}{filled}{rng.choice(SUFFIXES)}"
    if rng.random() < 0.3:
        text = text.lower()
    if rng.random() < 0.15:
        text = _typo(text, rng)
    return text


def generate(seed: int = 7) -> list[Row]:
    """Deterministically generate the sample stream, sorted by event time."""
    rng = random.Random(seed)
    drafts: list[tuple[datetime, str, str]] = []
    for topic, phases in SCHEDULE.items():
        for start, duration, count in phases:
            for _ in range(count):
                offset = rng.triangular(0, duration, duration * 0.3)
                when = START + timedelta(hours=start + offset)
                drafts.append((when, _render(rng.choice(TOPICS[topic]), rng), topic))
    for question in NOISE:  # one-offs: each noise question appears exactly once
        when = START + timedelta(hours=rng.uniform(0, SPAN_HOURS))
        drafts.append((when, _render(question, rng), NOISE_LABEL))
    drafts.sort(key=lambda d: (d[0], d[1]))
    return [
        Row(f"it_{n:05d}", text, when.replace(microsecond=0), label)
        for n, (when, text, label) in enumerate(drafts, start=1)
    ]


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", "timestamp", "label"])
        for row in rows:
            writer.writerow([row.id, row.text, row.timestamp.isoformat(), row.label])


def sample_path() -> Path:
    """Path of the bundled sample dataset."""
    return Path(str(resources.files("toolkit.demo") / "data" / SAMPLE_FILE))


if __name__ == "__main__":  # regenerate the bundled file
    write_csv(generate(), sample_path())
