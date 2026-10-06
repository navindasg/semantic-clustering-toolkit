"""fastembed backend (ONNX Runtime, no PyTorch). The default."""

from __future__ import annotations

import numpy as np

from toolkit.embedders.base import normalize


class FastEmbedEmbedder:
    def __init__(self, model_name: str, model_dir: str | None = None, batch_size: int = 64) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - fastembed is a core dependency
            raise ImportError("fastembed is not installed; `uv pip install toolkit`") from exc
        kwargs: dict = {"model_name": model_name}
        if model_dir:
            kwargs.update(specific_model_path=model_dir, local_files_only=True)
        self._model = TextEmbedding(**kwargs)
        self._name = model_name
        self._batch_size = batch_size

    @property
    def model_id(self) -> str:
        return f"fastembed:{self._name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = list(self._model.embed(texts, batch_size=self._batch_size))
        return normalize(np.stack(vectors))
