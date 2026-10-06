"""Webhook delivery with retries. Runs on a background thread so it never blocks assignment.

Delivery is best effort with bounded retries (exponential backoff); events that exhaust their
retries are logged and counted in `failed`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Iterable

import httpx

from toolkit.models import Event

logger = logging.getLogger("toolkit.webhook")
SIGNATURE_HEADER = "X-Toolkit-Signature"
_STOP = object()


def sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class WebhookSink:
    def __init__(
        self,
        url: str,
        secret: str | None = None,
        event_types: Iterable[str] = (),
        max_retries: int = 5,
        timeout_seconds: float = 5.0,
        backoff_seconds: float = 0.5,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"webhook url must be http(s): {url!r}")
        self.url = url
        self._secret = secret
        self._types = set(event_types)
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._sleep = sleep
        self._queue: queue.Queue = queue.Queue()
        self.delivered = 0
        self.failed = 0
        self._worker = threading.Thread(target=self._run, name="toolkit-webhook", daemon=True)
        self._worker.start()

    @classmethod
    def from_config(cls, cfg) -> WebhookSink:
        secret = os.environ.get(cfg.secret_env) if cfg.secret_env else None
        if cfg.secret_env and not secret:
            raise ValueError(f"webhook secret env var {cfg.secret_env} is not set")
        return cls(
            cfg.url,
            secret=secret,
            event_types=cfg.event_types,
            max_retries=cfg.max_retries,
            timeout_seconds=cfg.timeout_seconds,
        )

    def handle(self, event: Event) -> None:
        if not self._types or str(event.type) in self._types:
            self._queue.put(event)

    def _deliver(self, event: Event) -> bool:
        body = json.dumps(event.to_dict(), sort_keys=True).encode()
        headers = {"Content-Type": "application/json", "X-Toolkit-Event": str(event.type)}
        if self._secret:
            headers[SIGNATURE_HEADER] = sign(body, self._secret)
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(self.url, content=body, headers=headers)
                if response.status_code < 400:
                    return True
                retryable = response.status_code >= 500 or response.status_code == 429
                logger.warning("webhook %s returned %s", self.url, response.status_code)
                if not retryable:
                    return False
            except httpx.HTTPError as exc:
                logger.warning("webhook %s failed: %s", self.url, exc)
            if attempt < self._max_retries:
                self._sleep(self._backoff * (2**attempt))
        return False

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            try:
                if event is _STOP:
                    return
                try:
                    ok = self._deliver(event)
                except Exception:  # never let one bad event kill the worker and strand the queue
                    logger.exception("webhook delivery crashed on event %s", event.seq)
                    ok = False
                if ok:
                    self.delivered += 1
                else:
                    self.failed += 1
                    logger.error("webhook gave up on event %s (%s)", event.seq, event.type)
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Block until every queued event has been delivered or given up on."""
        self._queue.join()

    def close(self) -> None:
        self._queue.put(_STOP)
        self._worker.join(timeout=30)
        if self._worker.is_alive():
            logger.warning("webhook worker still delivering after 30s; leaving its client open")
            return
        self._client.close()
