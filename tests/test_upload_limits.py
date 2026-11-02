from fastapi.testclient import TestClient

from whiteout import server


def _post(client, ip, name="x.txt"):
    return client.post("/api/jobs", files={"file": (name, b"x")}, headers={"x-forwarded-for": ip})


def _reset():
    server._recent.clear()
    server._all_uploads.clear()


def test_written_address_does_not_dodge_the_limit():
    _reset()
    client = TestClient(server.app)
    codes = [_post(client, f"10.0.0.{i}, 203.0.113.9").status_code for i in range(server.UPLOADS_PER_HOUR + 1)]
    assert codes[:-1] == [400] * server.UPLOADS_PER_HOUR
    assert codes[-1] == 429


def test_many_addresses_hit_the_total_limit(monkeypatch):
    _reset()
    monkeypatch.setattr(server, "TOTAL_UPLOADS_PER_HOUR", 2)
    monkeypatch.setattr(server.STORE, "create", lambda *a, **k: type("J", (), {"id": "j"})())
    client = TestClient(server.app)
    codes = [_post(client, f"203.0.113.{i}", "x.png").status_code for i in range(3)]
    assert codes == [200, 200, 429]
