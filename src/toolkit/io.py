"""Read item batches from CSV or JSONL with configurable column names."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from toolkit.clock import ensure_utc
from toolkit.config import ColumnMap
from toolkit.models import Item


@dataclass(frozen=True)
class LabeledItem:
    item: Item
    label: str | None


class InputError(ValueError):
    """A record in an input file is malformed."""


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    try:
        return datetime.fromtimestamp(float(text), tz=UTC)
    except ValueError:
        pass
    return ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))


def _to_item(record: dict[str, Any], columns: ColumnMap, where: str) -> LabeledItem:
    missing = [c for c in (columns.id, columns.text, columns.timestamp) if c not in record]
    if missing:
        raise InputError(f"{where}: missing column(s) {', '.join(missing)}")
    text = str(record[columns.text] or "").strip()
    if not text:
        raise InputError(f"{where}: empty text")
    try:
        timestamp = parse_timestamp(record[columns.timestamp])
    except (ValueError, OverflowError, OSError) as exc:
        raise InputError(f"{where}: bad timestamp {record[columns.timestamp]!r}") from exc
    reserved = {columns.id, columns.text, columns.timestamp, columns.label}
    metadata = {k: v for k, v in record.items() if k not in reserved and v not in (None, "")}
    label = record.get(columns.label) if columns.label else None
    item = Item(id=str(record[columns.id]), text=text, timestamp=timestamp, metadata=metadata)
    return LabeledItem(item, str(label) if label not in (None, "") else None)


def iter_records(path: Path, columns: ColumnMap | None = None) -> Iterator[LabeledItem]:
    columns = columns or ColumnMap()
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputError(f"{path}:{line_no}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise InputError(f"{path}:{line_no}: expected a JSON object")
                yield _to_item(record, columns, f"{path}:{line_no}")
    elif suffix in {".csv", ".tsv"}:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
            for line_no, record in enumerate(reader, start=2):
                yield _to_item(record, columns, f"{path}:{line_no}")
    else:
        raise InputError(f"unsupported file type {path.suffix!r}; use .csv, .tsv or .jsonl")


def read_items(path: str | Path, columns: ColumnMap | None = None) -> list[LabeledItem]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"input file not found: {file_path}")
    return list(iter_records(file_path, columns))
