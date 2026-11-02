from types import SimpleNamespace

from fastapi.testclient import TestClient

from whiteout import jobs, server
from whiteout.forms import drawing_revision, form3_rows
from whiteout.pipeline import _drop_repeats
from whiteout.types import Requirement
from whiteout.types import Characteristic


def _char(cid, text, flags=(), kind="dimension"):
    return Characteristic(id=cid, kind=kind, text=text, multiplier=1, box=(0, 0, 9, 9), raw_box=(0, 0, 9, 9), sheet=0, flags=list(flags))


def test_unmatched_repeat_is_dropped_but_kept_when_alone():
    a, b, c = _char("a", "3.375"), _char("b", "3.375", ["box not matched to ink"]), _char("c", "1.250", ["box not matched to ink"])
    _drop_repeats([a, b, c])
    assert not a.ignored and b.ignored and not c.ignored


def test_two_unmatched_repeats_are_both_kept():
    a, b = _char("a", "16", ["box not matched to ink"]), _char("b", "16", ["box not matched to ink"])
    _drop_repeats([a, b])
    assert not a.ignored and not b.ignored


def test_range_rule_has_no_pass_or_fail():
    c = _char("a", ".016R TO .062R +/- .010")
    c.number = "5"
    c.requirements = [Requirement(text=c.text, type="other", lower=-0.01, upper=0.01)]
    row = form3_rows([c], {"5": ".030"})[0]
    assert row["lower"] in ("", None) and row["status"] == "needs review"


def test_real_limits_still_give_pass_or_fail():
    c = _char("a", ".370-.380")
    c.number = "5"
    c.requirements = [Requirement(text=c.text, type="size", lower=0.37, upper=0.38)]
    assert form3_rows([c], {"5": ".375"})[0]["status"] == "pass"
    assert form3_rows([c], {"5": "1.38"})[0]["status"] == "fail"


def test_drawing_revision_is_the_latest_letter():
    sheets = [SimpleNamespace(info=SimpleNamespace(revision=r)) for r in ("A", "B", "A")]
    assert drawing_revision(sheets) == "B"
    assert drawing_revision([SimpleNamespace(info=SimpleNamespace(revision=""))]) == ""


def test_remove_and_undo_a_line(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "STORE", jobs.JobStore(tmp_path))
    client = TestClient(server.app)
    j = client.post("/api/jobs", data={"sample": "apollo"}).json()["id"]
    d = client.get(f"/api/jobs/{j}").json()
    n = len(d["rows"])
    cid = d["rows"][0]["char_id"]
    r = client.patch(f"/api/jobs/{j}/chars/{cid}", json={"ignored": True}).json()
    assert len(r["rows"]) < n and all(x["char_id"] != cid for x in r["rows"])
    assert client.get(f"/api/jobs/{j}/forms.xlsx").status_code == 200
    assert client.get(f"/api/jobs/{j}/ballooned.pdf").status_code == 200
    r = client.patch(f"/api/jobs/{j}/chars/{cid}", json={"ignored": False}).json()
    assert len(r["rows"]) == n
