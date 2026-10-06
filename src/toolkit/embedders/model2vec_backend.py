"""model2vec backend: static embeddings, NumPy only, fastest on CPU."""

from __future__ import annotations

import numpy as np

from toolkit.embedders.base import normalize


class Model2VecEmbedder:
    def __init__(self, model_name: str, model_dir: str | None = None) -> None:
        try:
            from model2vec import StaticModel
        except ImportError as exc:
            raise ImportError("model2vec is not installed; `uv pip install 'toolkit[light]'`") from exc
        self._model = StaticModel.from_pretrained(model_dir or model_name)
        self._name = model_name

    @property
    def model_id(self) -> str:
        return f"model2vec:{self._name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return normalize(self._model.encode(list(texts)))
