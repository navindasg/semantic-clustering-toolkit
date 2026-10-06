"""sentence-transformers backend (needs PyTorch via `toolkit[torch]`)."""

from __future__ import annotations

import numpy as np

from toolkit.embedders.base import normalize


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, model_dir: str | None = None, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed; `uv pip install 'toolkit[torch]'`"
            ) from exc
        self._model = SentenceTransformer(model_dir or model_name)
        self._name = model_name
        self._batch_size = batch_size

    @property
    def model_id(self) -> str:
        return f"sentence-transformers:{self._name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._model.encode(list(texts), batch_size=self._batch_size, normalize_embeddings=True)
        return normalize(vectors)
