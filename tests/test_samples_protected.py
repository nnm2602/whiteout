import json

from fastapi.testclient import TestClient

from whiteout import jobs, server

SAMPLE_ID = "e9f770195cba"


def _store(monkeypatch, tmp_path):
    store = jobs.JobStore(tmp_path) if hasattr(jobs, "JobStore") else type(jobs.STORE)(tmp_path)
    monkeypatch.setattr(server, "STORE", store)
    return store


def test_writes_to_the_original_are_refused():
    client = TestClient(server.app)
    r = client.put(f"/api/jobs/{SAMPLE_ID}/results", json={"results": {"1.1": "9"}})
    assert r.status_code == 403
    assert client.post(f"/api/jobs/{SAMPLE_ID}/chars", json={"text": "x"}).status_code == 403


def test_each_visit_gets_a_private_copy_and_edits_stay_there(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    client = TestClient(server.app)
    original = (jobs.SAMPLE_JOBS / SAMPLE_ID / "job.json").read_bytes()
    a = client.post("/api/jobs", data={"sample": "apollo"}).json()["id"]
    b = client.post("/api/jobs", data={"sample": "apollo"}).json()["id"]
    assert a != b and SAMPLE_ID not in (a, b)
    assert client.put(f"/api/jobs/{a}/results", json={"results": {"1.1": "9"}}).status_code == 200
    assert client.get(f"/api/jobs/{a}").json()["results"] == {"1.1": "9"}
    assert client.get(f"/api/jobs/{b}").json()["results"] == {}
    assert client.get(f"/api/jobs/{SAMPLE_ID}").json()["results"] == {}
    assert client.get(f"/api/jobs/{a}/img/sheet0.png").status_code == 200
    assert (jobs.SAMPLE_JOBS / SAMPLE_ID / "job.json").read_bytes() == original
    assert json.loads((tmp_path / a / "job.json").read_text())["created"] < server._start  # not counted as new spend


def test_old_copies_are_removed_and_the_original_survives(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    monkeypatch.setattr(jobs, "MAX_FORKS", 2)
    ids = [store.fork(SAMPLE_ID).id for _ in range(3)]
    assert not (tmp_path / ids[0]).exists() and (tmp_path / ids[2]).exists()
    assert (jobs.SAMPLE_JOBS / SAMPLE_ID / "sheet0.png").exists()
