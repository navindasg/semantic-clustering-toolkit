"""Optional preprocessing before embedding: normalization, PII redaction, truncation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from toolkit.config import PreprocessConfig

Preprocessor = Callable[[str], str]

_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    (re.compile(r"(?<!\w)\+?\d{1,3}?[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"https?://\S+"), "[URL]"),
)
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def redact_pii(text: str) -> str:
    for pattern, token in _PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


def build_preprocessor(cfg: PreprocessConfig, extra: Preprocessor | None = None) -> Preprocessor | None:
    """Compose the configured steps (and an optional custom step, run last)."""
    steps: list[Preprocessor] = []
    if cfg.normalize:
        steps.append(normalize_text)
    if cfg.redact_pii:
        steps.append(redact_pii)
    if extra is not None:
        steps.append(extra)
    if cfg.max_chars:
        limit = cfg.max_chars
        steps.append(lambda text: text[:limit])
    if not steps:
        return None

    def run(text: str) -> str:
        for step in steps:
            text = step(text)
        return text

    return run
