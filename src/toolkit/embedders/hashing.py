"""Deterministic hashing embedder: no download, no model. For tests and smoke runs only."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from toolkit.embedders.base import normalize

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Signed feature hashing over word unigrams, bigrams and character trigrams."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def model_id(self) -> str:
        return f"hashing:{self._dim}"

    def _features(self, text: str) -> list[str]:
        words = _TOKEN.findall(text.lower())
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        trigrams = [w[i : i + 3] for w in words if len(w) > 3 for i in range(len(w) - 2)]
        return words + bigrams + trigrams

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                sign = 1.0 if value & 1 else -1.0
                out[row, (value >> 1) % self._dim] += sign
        return normalize(out)
