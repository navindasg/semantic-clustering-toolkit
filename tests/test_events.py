import io
import json

import httpx
import pytest
from conftest import T0

from toolkit.config import WebhookConfig
from toolkit.events import CallbackSink, EventBus, TerminalSink, WebhookSink
from toolkit.events.terminal import describe
from toolkit.events.webhook import SIGNATURE_HEADER, sign
from toolkit.models import Event, EventType


def ev(t=EventType.CLUSTER_OPENED, **data):
    return Event(1, t, T0, "cl_x", "it_1", data)


def test_bus_isolates_failing_sinks(caplog):
    seen = []

    class Broken:
        def handle(self, event):
            raise RuntimeError("sink down")

    bus = EventBus([Broken()])
    bus.on(seen.append)
    bus.publish([ev()])
    assert len(seen) == 1
    assert "sink" in caplog.text


def test_callback_filter_and_unsubscribe():
    seen = []
    bus = EventBus()
    sink = bus.on(seen.append, types=[EventType.CLUSTER_CLOSED])
    bus.publish([ev(), ev(EventType.CLUSTER_CLOSED)])
    assert [str(e.type) for e in seen] == ["cluster.closed"]
    bus.unsubscribe(sink)
    bus.publish([ev(EventType.CLUSTER_CLOSED)])
    assert len(seen) == 1
    assert isinstance(sink, CallbackSink)


def test_event_to_dict():
    data = ev(size=3).to_dict()
    assert data["type"] == "cluster.opened" and data["data"] == {"size": 3}
    assert data["timestamp"].startswith("2026-03-01")


@pytest.mark.parametrize(
    "event",
    [
        ev(size=3, label="printer"),
        ev(EventType.CLUSTER_MERGED, survivor="cl_y", similarity=0.9),
        ev(EventType.CLUSTER_CLOSED, size=4, label="x"),
        ev(EventType.ITEM_ASSIGNED, score=0.7),
        ev(EventType.ITEM_ORPHANED),
        ev(EventType.CLUSTER_SPLIT, new_cluster="cl_z"),
    ],
)
def test_describe_every_type(event):
    assert str(event.type) in describe(event)


def test_terminal_sink_filters_item_events():
    out = io.StringIO()
    sink = TerminalSink(stream=out, color=True)
    sink.handle(ev(EventType.ITEM_ASSIGNED, score=0.5))
    sink.handle(ev(size=2, label="printer"))
    text = out.getvalue()
    assert "item.assigned" not in text and "cluster.opened" in text and "\033[" in text


def _sink(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return WebhookSink("https://hooks.example/x", client=client, sleep=lambda s: None, **kwargs)


def test_webhook_delivers_with_signature():
    received = []

    def handler(request):
        received.append(request)
        return httpx.Response(200)

    sink = _sink(handler, secret="s3cret")
    sink.handle(ev(size=1))
    sink.flush()
    sink.close()
    [request] = received
    assert request.headers[SIGNATURE_HEADER] == sign(request.content, "s3cret")
    assert json.loads(request.content)["type"] == "cluster.opened"
    assert sink.delivered == 1 and sink.failed == 0


def test_webhook_retries_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503 if len(calls) < 3 else 200)

    sink = _sink(handler, max_retries=5)
    sink.handle(ev())
    sink.flush()
    sink.close()
    assert len(calls) == 3 and sink.delivered == 1


def test_webhook_gives_up_and_counts_failure():
    def handler(request):
        raise httpx.ConnectError("down")

    sink = _sink(handler, max_retries=2)
    sink.handle(ev())
    sink.flush()
    sink.close()
    assert sink.failed == 1


def test_webhook_does_not_retry_client_errors():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400)

    sink = _sink(handler, max_retries=5)
    sink.handle(ev())
    sink.flush()
    sink.close()
    assert len(calls) == 1 and sink.failed == 1


def test_webhook_event_type_filter():
    calls = []
    sink = _sink(lambda r: calls.append(1) or httpx.Response(200), event_types=["cluster.closed"])
    sink.handle(ev())
    sink.handle(ev(EventType.CLUSTER_CLOSED))
    sink.flush()
    sink.close()
    assert len(calls) == 1


def test_webhook_validation_and_config(monkeypatch):
    with pytest.raises(ValueError):
        WebhookSink("ftp://nope")
    with pytest.raises(ValueError):
        WebhookSink.from_config(WebhookConfig(url="https://x.example", secret_env="MISSING_SECRET"))
    monkeypatch.setenv("HOOK_SECRET", "abc")
    sink = WebhookSink.from_config(WebhookConfig(url="https://x.example", secret_env="HOOK_SECRET"))
    sink.close()


def test_engine_publishes_after_commit(engine_bundle):
    from conftest import seed_two_topics

    engine, clock, rec = engine_bundle
    seed_two_topics(engine, clock)
    stored = engine.events()
    published = [e for e in rec.events]
    assert [e.seq for e in stored] == [e.seq for e in published]
    assert all(e.seq > 0 for e in published)


def test_webhook_worker_survives_unexpected_errors():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("not an httpx error")
        return httpx.Response(200)

    sink = _sink(handler, max_retries=0)
    sink.handle(ev())
    sink.handle(ev())
    sink.flush()
    sink.close()
    assert sink.failed == 1 and sink.delivered == 1
