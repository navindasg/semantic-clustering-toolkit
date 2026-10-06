"""Labeling glue: gathers sampled texts and calls the configured labeler."""

from __future__ import annotations

import logging

from toolkit.engine.context import EngineContext
from toolkit.models import ClusterStatus

logger = logging.getLogger("toolkit.labels")
TEXTS_PER_CLUSTER = 100


def _sample(texts: list[str], ctx: EngineContext, salt: str) -> list[str]:
    if len(texts) <= TEXTS_PER_CLUSTER:
        return texts
    picked = ctx.rng("label-sample", salt).choice(len(texts), TEXTS_PER_CLUSTER, replace=False)
    return [texts[i] for i in sorted(picked)]


def label_clusters(ctx: EngineContext, targets: dict[str, list[str]]) -> dict[str, str]:
    """Label target clusters, contrasting them with every other live cluster."""
    if not targets:
        return {}
    sampled = {cid: _sample(texts, ctx, cid) for cid, texts in targets.items()}
    context: dict[str, list[str]] = {}
    for cluster in ctx.store.list_clusters([ClusterStatus.OPEN, ClusterStatus.CLOSED]):
        if cluster.id not in sampled:
            members = ctx.store.cluster_items(cluster.id, limit=TEXTS_PER_CLUSTER)
            context[cluster.id] = [m.text for m in members]
    try:
        return ctx.labeler.label(sampled, {**context, **sampled})
    except Exception:
        logger.exception("labeler failed; clusters keep their previous labels")
        return {}
