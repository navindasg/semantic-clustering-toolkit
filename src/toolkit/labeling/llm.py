"""Optional LLM labeler (SMP-3): a one-line title from sampled items, via the Claude API.

Needs `toolkit[llm]` and Anthropic credentials (ANTHROPIC_API_KEY or an `ant auth login`
profile). Only used when `labeler: llm` is configured; nothing is sent otherwise. Any API
error or refusal falls back to the c-TF-IDF label for that cluster.
"""

from __future__ import annotations

import logging
from typing import Any

from toolkit.labeling.base import Labeler

logger = logging.getLogger("toolkit.labeling.llm")

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TITLE_CHARS = 80
SAMPLES_IN_PROMPT = 15
SYSTEM_PROMPT = (
    "You name clusters of short user reports. Reply with one concise title (at most 8 words) "
    "describing what the reports have in common. No quotes, no trailing punctuation, no preamble."
)


class LLMLabeler:
    def __init__(
        self,
        model: str | None = None,
        fallback: Labeler | None = None,
        client: Any | None = None,
        effort: str = "low",
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.fallback = fallback
        self.effort = effort
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError("the LLM labeler needs `uv pip install 'toolkit[llm]'`") from exc
            client = anthropic.Anthropic()
        self._client = client

    def _title(self, texts: list[str]) -> str | None:
        sample = "\n".join(f"- {t}" for t in texts[:SAMPLES_IN_PROMPT])
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=[{"role": "user", "content": f"Reports:\n{sample}"}],
        )
        if response.stop_reason == "refusal":
            logger.warning("labeler request was declined; using the fallback label")
            return None
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        title = text.splitlines()[0].strip().strip('"').rstrip(".") if text else ""
        return title[:MAX_TITLE_CHARS] or None

    def label(self, targets: dict[str, list[str]], context: dict[str, list[str]]) -> dict[str, str]:
        fallback = self.fallback.label(targets, context) if self.fallback else {}
        labels: dict[str, str] = {}
        for cluster_id, texts in targets.items():
            title = None
            if texts:
                try:
                    title = self._title(texts)
                except Exception:  # API errors must never break discovery or sweeps
                    logger.exception("LLM labeling failed for %s; using the fallback label", cluster_id)
            labels[cluster_id] = title or fallback.get(cluster_id, "")
        return labels
