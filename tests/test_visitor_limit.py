import json
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from whiteout import server

REQ = SimpleNamespace(headers={"x-forwarded-for": "203.0.113.7"}, client=None)
OTHER = SimpleNamespace(headers={"x-forwarded-for": "203.0.113.8"}, client=None)


@pytest.fixture()
def fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "VISITOR_USD", 50.0)
    monkeypatch.setattr(server, "DAILY_USD", 1000.0)
    monkeypatch.setattr(server, "_start", time.time() - 1)
    monkeypatch.setattr(server, "_ledger", {"jobs": {}, "calls": 0.0})
    monkeypatch.setattr(server, "_ledger_file", tmp_path / "main.json")
    monkeypatch.setattr(server, "_ledger_ok", True)
    monkeypatch.setattr(server, "_saved", "")
    monkeypatch.setattr(server.STORE, "jobs", {})
    return tmp_path


def test_one_address_stops_at_the_limit_and_others_do_not(fresh):
    key = server._visitor_ok(REQ)
    server._ledger["job_owner"] = {"a": key}
    server._ledger["jobs"]["a"] = 49.0
    server._record(SimpleNamespace(usage=SimpleNamespace(usd=1.0)))
    with pytest.raises(HTTPException) as e:
        server._visitor_ok(REQ)
    assert e.value.status_code == 429
    server._visitor_ok(OTHER)


def test_limit_counts_both_apps(fresh):
    key = server._visitor_key(REQ)
    (fresh / "backup.json").write_text(json.dumps({"jobs": {"b": 30.0}, "job_owner": {"b": key}, "visitor_calls": {key: 20.0}}))
    with pytest.raises(HTTPException):
        server._visitor_ok(REQ)


def test_address_is_not_stored(fresh):
    server._visitor_ok(REQ)
    server._record(SimpleNamespace(usage=SimpleNamespace(usd=1.0)))
    assert "203.0.113.7" not in (fresh / "main.json").read_text()


def test_blocked_visitor_can_still_open_samples(fresh, monkeypatch):
    monkeypatch.setattr(server, "VISITOR_USD", 0.0)
    client = TestClient(server.app)
    assert client.post("/api/jobs", files={"file": ("x.png", b"x")}).status_code == 429
    assert client.post("/api/jobs", data={"sample": "apollo"}).status_code == 200
