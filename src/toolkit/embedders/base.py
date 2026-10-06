"""Embedder interface shared by every backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_id(self) -> str:
        """Stable identifier stored with every vector, e.g. 'fastembed:BAAI/bge-small-en-v1.5'."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero rows stay zero."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.where(norms == 0, 1.0, norms)


class ModelMismatchError(RuntimeError):
    """Raised when vectors from different embedding models would be mixed."""
