"""Similarity primitives. All vectors are L2-normalized, so dot product equals cosine."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

_CHUNK = 256


def stable_seed(*parts: object) -> int:
    """Deterministic 63-bit seed from arbitrary parts."""
    digest = hashlib.blake2b(":".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


def make_rng(random_seed: int | None, *parts: object) -> np.random.Generator:
    if random_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(stable_seed(random_seed, *parts))


def topk_mean(vectors: np.ndarray, exemplars: np.ndarray, k: int) -> np.ndarray:
    """Mean of the top-k cosine similarities of each vector against one exemplar set."""
    if len(vectors) == 0:
        return np.zeros(0, dtype=np.float32)
    sims = vectors @ exemplars.T
    k_eff = min(k, exemplars.shape[0])
    top = -np.partition(-sims, k_eff - 1, axis=1)[:, :k_eff]
    return top.mean(axis=1)


def member_scores(
    members: np.ndarray,
    member_ids: Sequence[str],
    exemplars: np.ndarray,
    exemplar_ids: Sequence[str],
    k: int,
) -> np.ndarray:
    """Top-k mean score of each member against the exemplars, ignoring its own exemplar entry."""
    sims = members @ exemplars.T
    position = {eid: n for n, eid in enumerate(exemplar_ids)}
    is_self = np.zeros_like(sims, dtype=bool)
    for row, mid in enumerate(member_ids):
        col = position.get(mid)
        if col is not None:
            is_self[row, col] = True
    sims = np.where(is_self, -np.inf, sims)
    available = exemplars.shape[0] - is_self.sum(axis=1)
    k_eff = int(max(1, min(k, available.min()))) if len(available) else 1
    top = -np.partition(-sims, k_eff - 1, axis=1)[:, :k_eff]
    return np.where(np.isfinite(top), top, 0.0).mean(axis=1)


def cluster_similarity(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Symmetric exemplar-set similarity: average of each side's mean top-k score to the other."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(0.5 * (topk_mean(a, b, k).mean() + topk_mean(b, a, k).mean()))


def threshold_from_scores(scores: np.ndarray, percentile: float, bounds: tuple[float, float]) -> float:
    if len(scores) == 0:
        return float(bounds[1])
    value = float(np.percentile(scores, percentile))
    return float(min(max(value, bounds[0]), bounds[1]))


class ExemplarIndex:
    """Brute-force NumPy nearest-exemplar lookup over a fixed set of clusters."""

    def __init__(self, cluster_ids: list[str], exemplar_sets: list[np.ndarray]) -> None:
        if len(cluster_ids) != len(exemplar_sets):
            raise ValueError("cluster_ids and exemplar_sets must align")
        self.cluster_ids = list(cluster_ids)
        self._counts = np.array([len(e) for e in exemplar_sets], dtype=np.int64)
        if not cluster_ids:
            self._tensor = np.zeros((0, 0, 0), dtype=np.float32)
            return
        width = int(self._counts.max())
        dim = exemplar_sets[0].shape[1]
        tensor = np.zeros((len(cluster_ids), width, dim), dtype=np.float32)
        for n, exemplars in enumerate(exemplar_sets):
            tensor[n, : len(exemplars)] = exemplars
        self._tensor = tensor
        self._mask = np.arange(width)[None, :] < self._counts[:, None]

    def __len__(self) -> int:
        return len(self.cluster_ids)

    def scores(self, vectors: np.ndarray, k: int) -> np.ndarray:
        """(n_vectors, n_clusters) matrix of top-k mean similarity."""
        n_clusters = len(self.cluster_ids)
        if n_clusters == 0 or len(vectors) == 0:
            return np.zeros((len(vectors), n_clusters), dtype=np.float32)
        c, m, d = self._tensor.shape
        flat = self._tensor.reshape(c * m, d)
        k_eff = np.minimum(k, self._counts)
        kmax = int(k_eff.max())
        out = np.empty((len(vectors), c), dtype=np.float32)
        rank = np.arange(kmax)[None, None, :]
        for start in range(0, len(vectors), _CHUNK):
            chunk = vectors[start : start + _CHUNK]
            sims = (chunk @ flat.T).reshape(len(chunk), c, m)
            sims = np.where(self._mask[None], sims, -np.inf)
            top = -np.sort(-sims, axis=2)[:, :, :kmax]
            keep = rank < k_eff[None, :, None]
            out[start : start + _CHUNK] = np.where(keep, top, 0.0).sum(axis=2) / k_eff[None, :]
        return out


class StoreExemplarIndex:
    """Approximate lookup through the store's vector index (e.g. pgvector HNSW).

    Each item fetches its nearest exemplars across all matchable clusters; a cluster's score is
    the top-k mean of the neighbours found for it (missing neighbours count as zero, so scores
    are never overestimated).
    """

    def __init__(self, store, cluster_ids: list[str], neighbors: int, now, grace) -> None:
        self.cluster_ids = list(cluster_ids)
        self._column = {cid: n for n, cid in enumerate(self.cluster_ids)}
        self._store = store
        self._neighbors = neighbors
        self._now = now
        self._grace = grace
        counts = store.exemplar_counts()
        self._counts = np.array([counts.get(c, 0) for c in self.cluster_ids], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.cluster_ids)

    def scores(self, vectors: np.ndarray, k: int) -> np.ndarray:
        out = np.zeros((len(vectors), len(self.cluster_ids)), dtype=np.float32)
        for row, vector in enumerate(vectors):
            found: dict[int, list[float]] = {}
            for cluster_id, sim in self._store.search_exemplars(
                vector, self._neighbors, self._now, self._grace
            ):
                col = self._column.get(cluster_id)
                if col is not None:
                    found.setdefault(col, []).append(sim)
            for col, sims in found.items():
                k_eff = max(1, min(k, int(self._counts[col])))
                out[row, col] = sum(sorted(sims, reverse=True)[:k_eff]) / k_eff
        return out
