"""Memoizing wrapper so evaluation and tuning embed each text once."""

from __future__ import annotations

import numpy as np

from toolkit.embedders.base import Embedder


class CachedEmbedder:
    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def warm(self, texts: list[str]) -> None:
        self.embed(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            vectors = self._inner.embed(missing)
            self._cache.update(zip(missing, vectors, strict=True))
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])
