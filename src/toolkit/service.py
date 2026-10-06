"""Optional HTTP API (ING-3): the same batches, queries and overrides as the Python API.

Security: set TOOLKIT_API_KEY to require an `X-API-Key` header on every endpoint except
/health. Requests are rate limited per client (TOOLKIT_RATE_LIMIT requests per minute,
default 600). Batches are capped at MAX_BATCH items.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from toolkit.config import Config
from toolkit.engine.core import ClusteringEngine
from toolkit.engine.sampling import SampleStrategy
from toolkit.factory import build_engine
from toolkit.models import ClusterStatus, Item
from toolkit.scheduler import BackgroundScheduler

logger = logging.getLogger("toolkit.service")
MAX_BATCH = 1000
API_KEY_ENV = "TOOLKIT_API_KEY"
RATE_LIMIT_ENV = "TOOLKIT_RATE_LIMIT"


class ItemIn(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=20_000)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class Batch(BaseModel):
    items: list[ItemIn] = Field(min_length=1, max_length=MAX_BATCH)


class MoveIn(BaseModel):
    item_id: str
    cluster_id: str


class MergeIn(BaseModel):
    a: str
    b: str
    survivor: str | None = None


class SplitIn(BaseModel):
    cluster_id: str
    item_ids: list[str] = Field(min_length=1)


def ok(data: Any) -> dict[str, Any]:
    return {"success": True, "data": data}


class RateLimiter:
    """Token bucket per client, with a bounded number of tracked clients (least recently seen
    are evicted), so unique client keys can't grow memory without limit."""

    def __init__(self, per_minute: int, max_clients: int = 10_000) -> None:
        self.rate = per_minute / 60.0
        self.capacity = float(per_minute)
        self.max_clients = max_clients
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.pop(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            allowed = tokens >= 1
            self._buckets[key] = (tokens - 1 if allowed else tokens, now)
            while len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
            return allowed


def _cluster(c) -> dict[str, Any]:
    return {
        "id": c.id,
        "status": str(c.status),
        "size": c.size,
        "label": c.label,
        "threshold": c.threshold,
        "first_seen": c.first_seen,
        "last_seen": c.last_seen,
        "opened_at": c.opened_at,
        "closed_at": c.closed_at,
        "merged_into": c.merged_into,
        "status_locked": c.status_locked,
    }


def _item(i) -> dict[str, Any]:
    return {
        "id": i.id,
        "text": i.text,
        "timestamp": i.timestamp,
        "status": str(i.status),
        "cluster_id": i.cluster_id,
        "score": i.score,
        "metadata": i.metadata,
    }


def create_app(
    config: Config, *, engine: ClusteringEngine | None = None, run_scheduler: bool = True
) -> FastAPI:
    holder: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        holder["engine"] = engine or build_engine(config)
        if run_scheduler:
            holder["scheduler"] = BackgroundScheduler(holder["engine"])
            holder["scheduler"].start()
        yield
        if "scheduler" in holder:
            holder["scheduler"].stop()
        if engine is None:
            holder["engine"].close()

    app = FastAPI(title="Semantic Clustering Toolkit", version="0.1.0", lifespan=lifespan)
    api_key = os.environ.get(API_KEY_ENV)
    limiter = RateLimiter(int(os.environ.get(RATE_LIMIT_ENV, "600")))
    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def guard(request: Request, key: str | None = Depends(header)) -> None:
        # Limit by client address before checking the key: a client-chosen header can't buy a
        # fresh bucket, and failed key guesses are throttled too.
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        if api_key and not (key and hmac.compare_digest(key, api_key)):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def eng() -> ClusteringEngine:
        return holder["engine"]

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"success": False, "error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(KeyError)
    async def not_found(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse({"success": False, "error": str(exc).strip("'\"")}, status_code=404)

    @app.exception_handler(ValueError)
    async def bad_request(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)

    @app.exception_handler(Exception)
    async def internal(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return JSONResponse({"success": False, "error": "internal error"}, status_code=500)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return ok({"status": "ok"})

    deps = [Depends(guard)]

    @app.post("/items", dependencies=deps)
    def ingest(batch: Batch) -> dict[str, Any]:
        items = [Item(i.id, i.text, i.timestamp, i.metadata) for i in batch.items]
        return ok([r.__dict__ for r in eng().ingest(items)])

    @app.get("/clusters", dependencies=deps)
    def clusters(status: list[ClusterStatus] | None = Query(default=None)) -> dict[str, Any]:
        return ok([_cluster(c) for c in eng().list_clusters(status or None)])

    @app.get("/clusters/{cluster_id}", dependencies=deps)
    def cluster(cluster_id: str, sample_size: int = Query(10, ge=0, le=100)) -> dict[str, Any]:
        detail = eng().cluster_detail(cluster_id, sample_size=max(sample_size, 1))
        return ok(
            {
                "cluster": _cluster(detail.cluster),
                "counts_over_time": [{"start": s, "count": n} for s, n in detail.counts_over_time],
                "sample": [_item(i) for i in detail.sample][:sample_size],
            }
        )

    @app.get("/clusters/{cluster_id}/sample", dependencies=deps)
    def sample(
        cluster_id: str,
        n: int = Query(10, ge=1, le=500),
        strategy: SampleStrategy = SampleStrategy.MIXED,
    ) -> dict[str, Any]:
        return ok([_item(i) for i in eng().sample(cluster_id, n, strategy)])

    @app.get("/events", dependencies=deps)
    def events(after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
        return ok([e.to_dict() for e in eng().events(after, limit)])

    @app.get("/stats", dependencies=deps)
    def stats() -> dict[str, Any]:
        return ok(eng().stats())

    @app.post("/discover", dependencies=deps)
    def discover() -> dict[str, Any]:
        return ok(eng().discover().__dict__)

    @app.post("/sweep", dependencies=deps)
    def sweep() -> dict[str, Any]:
        return ok(eng().sweep().__dict__)

    @app.post("/overrides/move", dependencies=deps)
    def move(body: MoveIn) -> dict[str, Any]:
        eng().move_item(body.item_id, body.cluster_id)
        return ok(None)

    @app.post("/overrides/merge", dependencies=deps)
    def merge(body: MergeIn) -> dict[str, Any]:
        return ok({"survivor": eng().merge_clusters(body.a, body.b, body.survivor)})

    @app.post("/overrides/split", dependencies=deps)
    def split(body: SplitIn) -> dict[str, Any]:
        return ok({"new_cluster": eng().split_cluster(body.cluster_id, body.item_ids)})

    @app.post("/clusters/{cluster_id}/close", dependencies=deps)
    def close(cluster_id: str) -> dict[str, Any]:
        eng().close_cluster(cluster_id)
        return ok(None)

    @app.post("/clusters/{cluster_id}/reopen", dependencies=deps)
    def reopen(cluster_id: str) -> dict[str, Any]:
        eng().reopen_cluster(cluster_id)
        return ok(None)

    @app.post("/clusters/{cluster_id}/unlock", dependencies=deps)
    def unlock(cluster_id: str) -> dict[str, Any]:
        eng().unlock_cluster(cluster_id)
        return ok(None)

    return app
