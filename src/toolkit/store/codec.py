"""Serialization helpers shared by the SQL store adapters."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np

from toolkit.clock import ensure_utc

_FORMAT = "%Y-%m-%dT%H:%M:%S.%f+00:00"


def dt_to_text(value: datetime | None) -> str | None:
    """Fixed-width UTC text, so lexical order equals time order."""
    return None if value is None else ensure_utc(value).strftime(_FORMAT)


def text_to_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.strptime(value, _FORMAT).replace(tzinfo=UTC)


def text_to_required_dt(value: str) -> datetime:
    return datetime.strptime(value, _FORMAT).replace(tzinfo=UTC)


def vec_to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def bytes_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def to_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def from_json(value: str | None) -> dict[str, Any]:
    return json.loads(value) if value else {}
