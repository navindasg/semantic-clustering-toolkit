from datetime import timedelta

import pytest
from conftest import T0, build, make_items
from fastapi.testclient import TestClient

from toolkit.service import RateLimiter, create_app


def payload(items):
    return {"items": [{"id": i.id, "text": i.text, "timestamp": i.timestamp.isoformat()} for i in items]}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("TOOLKIT_API_KEY", raising=False)
    engine, clock, _ = build()
    clock.advance_to(T0 + timedelta(hours=3))
    app = create_app(engine.config, engine=engine, run_scheduler=False)
    with TestClient(app) as c:
        yield c


def test_full_http_flow(client):
    items = make_items("printer", 12, T0) + make_items("wifi", 12, T0)
    resp = client.post("/items", json=payload(items))
    assert resp.status_code == 200 and resp.json()["success"]
    assert len(resp.json()["data"]) == 24
    opened = client.post("/discover").json()["data"]["opened"]
    assert len(opened) == 2
    clusters = client.get("/clusters", params={"status": "open"}).json()["data"]
    cid = clusters[0]["id"]
    detail = client.get(f"/clusters/{cid}", params={"sample_size": 3}).json()["data"]
    assert len(detail["sample"]) == 3 and detail["counts_over_time"]
    sample = client.get(f"/clusters/{cid}/sample", params={"n": 2, "strategy": "central"}).json()["data"]
    assert len(sample) == 2
    assert client.get("/events", params={"limit": 5}).json()["data"]
    assert client.get("/stats").json()["data"]["clusters"]["open"] == 2
    assert client.post("/sweep").json()["success"]
    other = clusters[1]["id"]
    assert client.post("/overrides/move", json={"item_id": sample[0]["id"], "cluster_id": other}).json()[
        "success"
    ]
    members = client.get(f"/clusters/{cid}/sample", params={"n": 2}).json()["data"]
    new = client.post("/overrides/split", json={"cluster_id": cid, "item_ids": [members[0]["id"]]}).json()
    assert new["data"]["new_cluster"].startswith("cl_")
    for action in ("close", "reopen", "unlock"):
        assert client.post(f"/clusters/{cid}/{action}").json()["success"]
    assert client.post("/overrides/merge", json={"a": cid, "b": other}).json()["data"]["survivor"]


def test_errors_use_api_envelope(client):
    missing = client.get("/clusters/cl_missing")
    assert missing.status_code == 404 and missing.json() == {
        "success": False,
        "error": "unknown cluster cl_missing",
    }
    assert client.post("/items", json={"items": []}).status_code == 422
    bad = client.post("/overrides/merge", json={"a": "x", "b": "x"})
    assert bad.status_code == 400 and not bad.json()["success"]
    assert client.get("/health").json()["data"]["status"] == "ok"


def test_api_key_required(monkeypatch):
    monkeypatch.setenv("TOOLKIT_API_KEY", "secret-key")
    engine, _, _ = build()
    with TestClient(create_app(engine.config, engine=engine, run_scheduler=False)) as c:
        assert c.get("/stats").status_code == 401
        assert c.get("/stats", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/stats", headers={"X-API-Key": "secret-key"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_rate_limit(monkeypatch):
    monkeypatch.delenv("TOOLKIT_API_KEY", raising=False)
    monkeypatch.setenv("TOOLKIT_RATE_LIMIT", "2")
    engine, _, _ = build()
    with TestClient(create_app(engine.config, engine=engine, run_scheduler=False)) as c:
        codes = [c.get("/stats").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rate_limiter_refills():
    limiter = RateLimiter(60)
    assert all(limiter.allow("a") for _ in range(60))
    assert not limiter.allow("a")
    assert limiter.allow("b")


def test_lifespan_builds_engine_and_scheduler(tmp_path, monkeypatch):
    monkeypatch.delenv("TOOLKIT_API_KEY", raising=False)
    engine, _, _ = build()
    config = engine.config.with_overrides(store=f"sqlite:///{tmp_path}/svc.db")
    with TestClient(create_app(config, run_scheduler=True)) as c:
        assert c.get("/stats").status_code == 200


def test_rate_limit_ignores_client_chosen_keys(monkeypatch):
    monkeypatch.delenv("TOOLKIT_API_KEY", raising=False)
    monkeypatch.setenv("TOOLKIT_RATE_LIMIT", "2")
    engine, _, _ = build()
    with TestClient(create_app(engine.config, engine=engine, run_scheduler=False)) as c:
        codes = [c.get("/stats", headers={"X-API-Key": f"k{n}"}).status_code for n in range(3)]
    assert codes == [200, 200, 429]


def test_failed_key_guesses_are_throttled(monkeypatch):
    monkeypatch.setenv("TOOLKIT_API_KEY", "secret")
    monkeypatch.setenv("TOOLKIT_RATE_LIMIT", "2")
    engine, _, _ = build()
    with TestClient(create_app(engine.config, engine=engine, run_scheduler=False)) as c:
        codes = [c.get("/stats", headers={"X-API-Key": f"guess{n}"}).status_code for n in range(3)]
    assert codes == [401, 401, 429]


def test_rate_limiter_bounds_tracked_clients():
    limiter = RateLimiter(10, max_clients=3)
    for n in range(10):
        limiter.allow(f"c{n}")
    assert len(limiter._buckets) == 3
