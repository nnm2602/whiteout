import json
import time
from types import SimpleNamespace

from whiteout import server


def _fresh(monkeypatch, tmp_path, start_days_ago, daily=7.0, days=7):
    monkeypatch.setattr(server, "DAILY_USD", daily)
    monkeypatch.setattr(server, "BUDGET_DAYS", days)
    monkeypatch.setattr(server, "_start", time.time() - start_days_ago * 86400 - 1)
    monkeypatch.setattr(server, "_ledger", {"jobs": {}, "calls": 0.0})
    monkeypatch.setattr(server, "_ledger_file", tmp_path / "svc.json")
    monkeypatch.setattr(server, "_ledger_ok", True)
    monkeypatch.setattr(server, "_saved", "")
    monkeypatch.setattr(server.STORE, "jobs", {})


def test_unused_money_carries_to_the_next_day(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, 2)
    assert server._allowance() == 21
    server._record(SimpleNamespace(usage=SimpleNamespace(usd=5.0)))
    assert server._budget_left() == 16


def test_allowance_stops_growing(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, 30)
    assert server._allowance() == 49


def test_before_the_start_date_there_is_one_day(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, -2)
    assert server._allowance() == 7


def test_spend_is_saved_and_counts_jobs_since_the_start(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, 0)
    now = time.time()
    monkeypatch.setattr(server.STORE, "jobs", {
        "new": SimpleNamespace(id="new", usage={"usd": 2.5}, created=now),
        "sample": SimpleNamespace(id="sample", usage={"usd": 9.0}, created=now - 86400 * 9),
    })
    assert server._spent() == 2.5
    assert json.loads((tmp_path / "svc.json").read_text())["jobs"] == {"new": 2.5}


def test_unknown_spend_blocks_paid_calls(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, 3)
    monkeypatch.setattr(server, "_ledger_ok", False)
    assert server._budget_left() == 0
