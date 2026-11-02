import json
import time
from collections import deque

from fastapi.testclient import TestClient

from whiteout import jobs, server


def _fresh(monkeypatch, tmp_path, daily=7.0):
    monkeypatch.setattr(server, "DAILY_USD", daily)
    monkeypatch.setattr(server, "_start", time.time() - 1)
    monkeypatch.setattr(server, "_ledger", {"jobs": {}, "calls": 0.0})
    monkeypatch.setattr(server, "_ledger_file", tmp_path / "svc.json")
    monkeypatch.setattr(server, "_ledger_ok", True)
    monkeypatch.setattr(server, "_saved", "")
    monkeypatch.setattr(server, "_history", deque())
    monkeypatch.setattr(server.STORE, "jobs", {})


def test_budget_used_up_alerts_once(monkeypatch, tmp_path, capsys):
    _fresh(monkeypatch, tmp_path)
    server._ledger["calls"] = 7.5
    server._check_alerts()
    server._check_alerts()
    out = capsys.readouterr().out
    assert out.count("used up") == 1
    assert "used up" in out


def test_alert_is_not_repeated_after_a_restart(monkeypatch, tmp_path, capsys):
    _fresh(monkeypatch, tmp_path)
    server._ledger["calls"] = 7.5
    server._check_alerts()
    capsys.readouterr()
    saved = json.loads((tmp_path / "svc.json").read_text())
    monkeypatch.setattr(server, "_ledger", saved)
    server._check_alerts()
    assert "used up" not in capsys.readouterr().out


def test_fast_spending_alerts(monkeypatch, tmp_path, capsys):
    _fresh(monkeypatch, tmp_path, daily=100.0)
    server._check_alerts()
    server._ledger["calls"] = 6.0
    server._check_alerts()
    assert "in the last hour" in capsys.readouterr().out


def test_credit_thresholds_add_up_both_apps(monkeypatch, tmp_path, capsys):
    _fresh(monkeypatch, tmp_path, daily=100.0)
    monkeypatch.setattr(server, "CREDIT_USD", 68.03)
    monkeypatch.setattr(server, "FAST_USD", 999.0)
    (tmp_path / "other.json").write_text(json.dumps({"jobs": {"a": 30.0}, "calls": 0.0}))
    server._ledger["calls"] = 10.0
    server._check_alerts()
    out = capsys.readouterr().out
    assert "50% of" in out and "80% of" not in out


def test_copies_keep_the_sample_files(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", jobs.JobStore(tmp_path))
    client = TestClient(server.app)
    for sid, jid in (("apollo", "e9f770195cba"), ("nist-904", "bd20fca1920f")):
        copy = client.post("/api/jobs", data={"sample": sid}).json()["id"]
        want = client.get(f"/api/jobs/{jid}/supplier-samples").json()
        assert want and client.get(f"/api/jobs/{copy}/supplier-samples").json() == want
    copy = client.post("/api/jobs", data={"sample": "nist-904"}).json()["id"]
    want = client.get("/api/jobs/bd20fca1920f/model-samples").json()
    assert want and client.get(f"/api/jobs/{copy}/model-samples").json() == want
