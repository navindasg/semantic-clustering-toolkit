"""Cluster labelers and a factory."""

from __future__ import annotations

from toolkit.labeling.base import Labeler
from toolkit.labeling.ctfidf import CTfidfLabeler

__all__ = ["CTfidfLabeler", "Labeler", "make_labeler"]


def make_labeler(spec: str, top_n: int = 5) -> Labeler:
    """'ctfidf' (default) or 'llm[:model]' (needs toolkit[llm] and ANTHROPIC_API_KEY)."""
    name, _, arg = spec.partition(":")
    if name == "ctfidf":
        return CTfidfLabeler(top_n=top_n)
    if name == "llm":
        from toolkit.labeling.llm import LLMLabeler

        return LLMLabeler(model=arg or None, fallback=CTfidfLabeler(top_n=top_n))
    raise ValueError(f"unknown labeler {spec!r}; use 'ctfidf' or 'llm[:model]'")
