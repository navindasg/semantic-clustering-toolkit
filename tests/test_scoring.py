import numpy as np
import pytest

from toolkit.embedders import CachedEmbedder, HashingEmbedder, make_embedder, normalize
from toolkit.engine.scoring import (
    ExemplarIndex,
    cluster_similarity,
    make_rng,
    member_scores,
    stable_seed,
    threshold_from_scores,
    topk_mean,
)


def rand(n, d=16, seed=0):
    return normalize(np.random.default_rng(seed).normal(size=(n, d)))


def test_topk_mean_matches_naive():
    vecs, ex = rand(4), rand(7, seed=1)
    sims = vecs @ ex.T
    expected = np.sort(sims, axis=1)[:, ::-1][:, :3].mean(axis=1)
    assert np.allclose(topk_mean(vecs, ex, 3), expected)
    assert len(topk_mean(np.zeros((0, 16)), ex, 3)) == 0


def test_topk_mean_k_larger_than_exemplars():
    vecs, ex = rand(2), rand(2, seed=1)
    assert np.allclose(topk_mean(vecs, ex, 10), (vecs @ ex.T).mean(axis=1))


def test_member_scores_exclude_self():
    vecs = rand(5)
    ids = [f"m{i}" for i in range(5)]
    scores = member_scores(vecs, ids, vecs[:3], ids[:3], k=1)
    assert np.all(scores < 0.999)  # never matched against itself


def test_cluster_similarity_symmetric():
    a, b = rand(5), rand(6, seed=3)
    assert cluster_similarity(a, b, 3) == pytest.approx(cluster_similarity(b, a, 3))
    assert cluster_similarity(a, a, 1) == pytest.approx(1.0)
    assert cluster_similarity(a, np.zeros((0, 16)), 3) == 0.0


def test_threshold_clamped():
    assert threshold_from_scores(np.array([0.1, 0.2]), 5, (0.5, 0.9)) == 0.5
    assert threshold_from_scores(np.array([0.95, 0.99]), 5, (0.5, 0.9)) == 0.9
    assert threshold_from_scores(np.array([]), 5, (0.5, 0.9)) == 0.9


def test_exemplar_index_matches_topk_mean():
    sets = [rand(3, seed=1), rand(8, seed=2), rand(5, seed=3)]
    index = ExemplarIndex(["a", "b", "c"], sets)
    vecs = rand(300, seed=9)  # spans more than one chunk
    scores = index.scores(vecs, 4)
    for col, ex in enumerate(sets):
        assert np.allclose(scores[:, col], topk_mean(vecs, ex, 4), atol=1e-5)
    assert len(index) == 3


def test_exemplar_index_empty_and_mismatch():
    empty = ExemplarIndex([], [])
    assert empty.scores(rand(2), 3).shape == (2, 0)
    with pytest.raises(ValueError):
        ExemplarIndex(["a"], [])


def test_seeds_deterministic():
    assert stable_seed(1, "x") == stable_seed(1, "x") != stable_seed(2, "x")
    assert make_rng(1, "a").integers(1000) == make_rng(1, "a").integers(1000)
    assert isinstance(make_rng(None, "a"), np.random.Generator)


def test_hashing_embedder_properties():
    emb = HashingEmbedder(64)
    out = emb.embed(["printer paper jam", "printer paper jam", "wifi router"])
    assert out.shape == (3, 64)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.allclose(out[0], out[1])
    assert emb.model_id == "hashing:64"
    with pytest.raises(ValueError):
        HashingEmbedder(0)


def test_cached_embedder_embeds_once():
    calls = []

    class Counting(HashingEmbedder):
        def embed(self, texts):
            calls.append(list(texts))
            return super().embed(texts)

    cached = CachedEmbedder(Counting(32))
    cached.warm(["a b", "c d"])
    cached.embed(["a b", "c d", "a b"])
    assert calls == [["a b", "c d"]]
    assert cached.model_id == "hashing:32"
    assert cached.embed([]).shape == (0, 0)


def test_make_embedder_errors():
    assert make_embedder("hashing:").model_id == "hashing:256"
    with pytest.raises(ValueError):
        make_embedder("nocolon")
    with pytest.raises(ValueError):
        make_embedder("unknown:model")


def test_normalize_handles_zero_and_1d():
    out = normalize(np.array([0.0, 0.0]))
    assert out.shape == (1, 2) and np.all(out == 0)
