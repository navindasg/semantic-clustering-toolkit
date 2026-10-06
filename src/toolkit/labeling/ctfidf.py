"""Class-based TF-IDF labels: the terms most specific to each cluster versus the rest."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class CTfidfLabeler:
    def __init__(self, top_n: int = 5, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.top_n = top_n
        self.ngram_range = ngram_range

    def top_terms(self, docs: dict[str, list[str]]) -> dict[str, list[str]]:
        ids = [cid for cid, texts in docs.items() if texts]
        if not ids:
            return {}
        corpus = [" ".join(docs[cid]) for cid in ids]
        vectorizer = CountVectorizer(stop_words="english", ngram_range=self.ngram_range)
        try:
            counts = vectorizer.fit_transform(corpus)
        except ValueError:  # only stop words or empty text
            return {cid: [] for cid in ids}
        tf = counts.toarray().astype(np.float64)
        tf /= np.maximum(tf.sum(axis=1, keepdims=True), 1.0)
        avg_words = counts.sum() / len(ids)
        frequency = np.asarray(counts.sum(axis=0)).ravel()
        idf = np.log(1.0 + avg_words / np.maximum(frequency, 1.0))
        scores = tf * idf
        terms = vectorizer.get_feature_names_out()
        result: dict[str, list[str]] = {}
        for row, cid in enumerate(ids):
            order = np.lexsort((terms, -scores[row]))
            picked: list[str] = []
            for idx in order:
                if scores[row, idx] <= 0 or len(picked) >= self.top_n:
                    break
                term = str(terms[idx])
                if not any(term in p or p in term for p in picked):
                    picked.append(term)
            result[cid] = picked
        return result

    def label(self, targets: dict[str, list[str]], context: dict[str, list[str]]) -> dict[str, str]:
        docs = {**context, **targets}
        terms = self.top_terms(docs)
        return {cid: " · ".join(terms.get(cid, [])) for cid in targets}
