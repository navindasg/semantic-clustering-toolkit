"""Labeler interface: turns sampled cluster texts into a short label."""

from __future__ import annotations

from typing import Protocol


class Labeler(Protocol):
    def label(self, targets: dict[str, list[str]], context: dict[str, list[str]]) -> dict[str, str]:
        """Label each target cluster. `context` holds texts of all live clusters (targets
        included) for labelers that contrast clusters against each other."""
        ...
