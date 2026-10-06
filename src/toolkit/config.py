"""Configuration: one YAML file, overridden by environment variables, then by explicit overrides."""

from __future__ import annotations

import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ENV_PREFIX = "TOOLKIT_"
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(w|d|h|m|s)")
_ISO_DURATION = re.compile(
    r"p(?:(\d+(?:\.\d+)?)w)?(?:(\d+(?:\.\d+)?)d)?"
    r"(?:t(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?)?"
)
_UNIT_SECONDS = {"w": 604800, "d": 86400, "h": 3600, "m": 60, "s": 1}


def parse_duration(value: str | int | float | timedelta) -> timedelta:
    """Parse '7d', '1h30m', '90s' or a number of seconds into a timedelta."""
    if isinstance(value, timedelta):
        return value
    if isinstance(value, int | float):
        return timedelta(seconds=value)
    text = str(value).strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return timedelta(seconds=float(text))
    iso = _ISO_DURATION.fullmatch(text)
    if iso and any(iso.groups()):
        weeks, days, hours, minutes, seconds = (float(g) if g else 0.0 for g in iso.groups())
        return timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds)
    parts = _DURATION_PART.findall(text)
    if not parts or _DURATION_PART.sub("", text).strip():
        raise ValueError(f"invalid duration {value!r}; use forms like '7d', '1h30m', '90s'")
    return timedelta(seconds=sum(float(n) * _UNIT_SECONDS[u] for n, u in parts))


def format_duration(value: timedelta) -> str:
    seconds = int(value.total_seconds())
    for unit in ("d", "h", "m"):
        size = _UNIT_SECONDS[unit]
        if seconds and seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


# Similarity scales differ per embedding model. These were calibrated on the bundled sample
# (see docs/BASELINE.md) and apply only when the config leaves the field unset; anything else
# falls back to the generic defaults on Config. `toolkit tune` recalibrates on your data.
MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "fastembed:BAAI/bge-small-en-v1.5": {"threshold_bounds": (0.75, 0.92), "merge_threshold": 0.86},
    "model2vec:minishlab/potion-base-8M": {
        "threshold_bounds": (0.55, 0.90),
        "merge_threshold": 0.85,
    },
}


class ColumnMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = "id"
    text: str = "text"
    timestamp: str = "timestamp"
    label: str | None = "label"


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    normalize: bool = False
    redact_pii: bool = False
    max_chars: int | None = Field(default=None, gt=0)


class WebhookConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    secret_env: str | None = None
    event_types: tuple[str, ...] = ()
    max_retries: int = Field(default=5, ge=0)
    timeout_seconds: float = Field(default=5.0, gt=0)


class Config(BaseModel):
    """Every engine parameter. Similarity defaults are model-dependent starting points."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedder: str = "fastembed:BAAI/bge-small-en-v1.5"
    model_dir: str | None = None
    embed_batch_size: int = Field(default=64, gt=0)
    exemplars_per_cluster: int = Field(default=32, gt=0)
    match_top_k: int = Field(default=5, gt=0)
    threshold_percentile: float = Field(default=5, ge=0, le=100)
    threshold_bounds: tuple[float, float] = (0.55, 0.90)
    discovery_interval: timedelta = timedelta(hours=1)
    discovery_min_buffer: int = Field(default=200, gt=0)
    discovery_sample_size: int = Field(default=50_000, gt=0)
    """Fit UMAP/HDBSCAN on at most this many buffered items; the rest match the new clusters."""
    min_cluster_size: int = Field(default=10, ge=2)
    min_samples: int = Field(default=5, ge=1)
    umap_components: int = Field(default=5, ge=2)
    umap_neighbors: int = Field(default=15, ge=2)
    close_after: timedelta = timedelta(days=7)
    keep_open_min_items: int = Field(default=1, ge=1)
    reopen_grace: timedelta | None = None
    merge_threshold: float = Field(default=0.85, gt=0, le=1)
    buffer_max_age: timedelta = timedelta(days=7)
    sweep_interval: timedelta = timedelta(minutes=5)
    relabel_growth: float = Field(default=1.5, gt=1)
    label_terms: int = Field(default=5, gt=0)
    random_seed: int | None = 42
    store: str = "sqlite:///toolkit.db"
    columns: ColumnMap = ColumnMap()
    preprocess: PreprocessConfig = PreprocessConfig()
    webhooks: tuple[WebhookConfig, ...] = ()
    labeler: str = "ctfidf"
    exemplar_index: Literal["numpy", "store"] = "numpy"
    """'numpy' brute force in memory, or 'store' to use the store's vector index (pgvector HNSW)."""
    exemplar_neighbors: int = Field(default=200, gt=0)
    """Nearest exemplars fetched per item when exemplar_index is 'store'."""

    @field_validator(
        "discovery_interval",
        "close_after",
        "reopen_grace",
        "buffer_max_age",
        "sweep_interval",
        mode="before",
    )
    @classmethod
    def _durations(cls, value: Any) -> Any:
        return None if value is None else parse_duration(value)

    @field_validator("threshold_bounds", mode="before")
    @classmethod
    def _bounds(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = [float(v) for v in re.split(r"[,\s]+(?:to\s+)?", value.strip()) if v]
        return value

    @model_validator(mode="before")
    @classmethod
    def _model_profile(cls, data: Any) -> Any:
        """Fill model-dependent similarity settings the caller did not set explicitly."""
        if not isinstance(data, dict):
            return data
        embedder = data.get("embedder", cls.model_fields["embedder"].default)
        profile = MODEL_PROFILES.get(str(embedder), {})
        return {**profile, **data}

    @model_validator(mode="after")
    def _check(self) -> Config:
        low, high = self.threshold_bounds
        if not 0 < low <= high <= 1:
            raise ValueError("threshold_bounds must satisfy 0 < floor <= ceiling <= 1")
        return self

    @property
    def effective_reopen_grace(self) -> timedelta:
        return self.reopen_grace if self.reopen_grace is not None else 2 * self.close_after

    def with_overrides(self, **overrides: Any) -> Config:
        """Return a new validated Config with the given non-None values replaced.

        Only explicitly set fields carry over, so switching `embedder` re-applies its profile.
        """
        data = self.model_dump(exclude_unset=True)
        new_embedder = overrides.get("embedder")
        if new_embedder is not None and new_embedder != self.embedder:
            for key, value in MODEL_PROFILES.get(self.embedder, {}).items():
                if data.get(key) == value:
                    data.pop(key)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return Config.model_validate(data)


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    fields = Config.model_fields
    overrides: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        name = key[len(ENV_PREFIX) :].lower()
        if name in fields and name not in {"columns", "preprocess", "webhooks"}:
            overrides[name] = yaml.safe_load(raw) if raw else raw
    return overrides


def load_config(
    path: str | Path | None = None,
    environ: dict[str, str] | None = None,
    **overrides: Any,
) -> Config:
    """Load config from YAML (optional), then TOOLKIT_* env vars, then explicit overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"config file not found: {file_path}")
        loaded = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config file {file_path} must contain a mapping")
        data.update(loaded)
    data.update(_env_overrides(dict(os.environ) if environ is None else environ))
    data.update({k: v for k, v in overrides.items() if v is not None})
    return Config.model_validate(data)
