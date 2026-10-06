"""Store adapters and a factory that picks one from a URL."""

from __future__ import annotations

from toolkit.store.base import Store
from toolkit.store.memory import MemoryStore
from toolkit.store.sqlite import SQLiteStore

__all__ = ["MemoryStore", "SQLiteStore", "Store", "open_store"]


def open_store(url: str) -> Store:
    """Open a store from 'memory://', 'sqlite:///path.db' or 'postgresql://...'."""
    if url in {"memory", "memory://"}:
        return MemoryStore()
    if url.startswith("sqlite://"):
        path = (
            url.removeprefix("sqlite:///") if url.startswith("sqlite:///") else url.removeprefix("sqlite://")
        )
        return SQLiteStore(path or ":memory:")
    if url.startswith(("postgres://", "postgresql://")):
        from toolkit.store.postgres import PostgresStore

        return PostgresStore(url)
    raise ValueError(f"unsupported store url {url!r}; use memory://, sqlite:///file.db or postgresql://")
