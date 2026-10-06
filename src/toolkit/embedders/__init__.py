"""Embedder backends and the factory that builds one from a spec like 'fastembed:BAAI/bge-small-en-v1.5'."""

from __future__ import annotations

from toolkit.embedders.base import Embedder, ModelMismatchError, normalize
from toolkit.embedders.cached import CachedEmbedder
from toolkit.embedders.hashing import HashingEmbedder

__all__ = [
    "CachedEmbedder",
    "Embedder",
    "HashingEmbedder",
    "ModelMismatchError",
    "make_embedder",
    "normalize",
]

BACKENDS = ("fastembed", "model2vec", "sentence-transformers", "hashing")


def make_embedder(spec: str, model_dir: str | None = None, batch_size: int = 64) -> Embedder:
    """Build an embedder from '<backend>:<model>'. `model_dir` loads the model from local files."""
    backend, sep, name = spec.partition(":")
    if not sep:
        raise ValueError(f"embedder spec {spec!r} must look like '<backend>:<model>'")
    if backend == "fastembed":
        from toolkit.embedders.fastembed_backend import FastEmbedEmbedder

        return FastEmbedEmbedder(name, model_dir=model_dir, batch_size=batch_size)
    if backend == "model2vec":
        from toolkit.embedders.model2vec_backend import Model2VecEmbedder

        return Model2VecEmbedder(name, model_dir=model_dir)
    if backend in {"sentence-transformers", "st"}:
        from toolkit.embedders.sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(name, model_dir=model_dir, batch_size=batch_size)
    if backend == "hashing":
        return HashingEmbedder(int(name) if name else 256)
    raise ValueError(f"unknown embedder backend {backend!r}; choose from {', '.join(BACKENDS)}")
