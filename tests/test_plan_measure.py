import asyncio
import json
from pathlib import Path

from PIL import Image

from whiteout.ask import stackup
from whiteout.dmis import ascii_text, check_program, make_program
from whiteout.gauge import decode, segment_read, to_units
from whiteout.planner import make_plan, plan_rows
from whiteout.types import Characteristic, Requirement
from whiteout.voice import parse_value

ROOT = Path(__file__).resolve().parents[1]
NIST = ROOT / "samples" / "jobs" / "bd20fca1920f" / "job.json"


def _char(text, req, n="1", mult=1):
    return Characteristic(id="c" + n, number=n, text=text, box=(0, 0, 10, 10), multiplier=mult, requirements=[req])


def test_instrument_follows_the_tolerance_band():
    wide = _char("2.500 ± .010", Requirement(type="size", text="2.500 ± .010", lower=2.49, upper=2.51))
    tight = _char("2.5000 ± .0010", Requirement(type="size", text="2.5000 ± .0010", lower=2.499, upper=2.501), "2")
    hole = _char("Ø.182 - .192", Requirement(type="size", text="Ø.182 - .192", lower=0.182, upper=0.192), "3")
    basic = _char("1.250", Requirement(type="basic", text="1.250"), "4")
    steps = {s["row_id"]: s for s in plan_rows([wide, tight, hole, basic])}
    assert steps["1"]["instrument"] == "caliper"
    assert steps["2"]["instrument"] == "micrometer"
    assert "pin gauges" in steps["3"]["instrument"] and "0.182" in steps["3"]["method"]
    assert steps["4"]["setup"] == "Not measured"


def test_plan_and_program_on_the_nist_sample():
    job = json.loads(NIST.read_text())
    plan = asyncio.run(make_plan(job, None))
    assert plan["to_measure"] > 30
    order = [g["setup"] for g in plan["setups"]]
    assert order.index("CMM") > order.index("Hand tools")
    prog = make_program(job, plan)
    assert check_program(prog["program"]) == []
    assert prog["program"].isascii()
    assert "TOL/POS,2D,0.0050,MMC" in prog["program"]


def test_program_uses_model_hole_positions():
    job = json.loads(NIST.read_text())
    plan = asyncio.run(make_plan(job, None))
    holes = [{"xyz": [1.0, 2.0, 0.0], "axis": [0, 0, 1], "diameter": 0.187}]
    prog = make_program(job, plan, holes)
    assert prog["from_model"] == 1
    assert "CART,1.0000,2.0000,0.0000" in prog["program"]


def test_check_program_catches_mistakes():
    bad = "DMISMN/'x',05.3\nOUTPUT/FA(H1),TA(T1)\nMEAS/CIRCLE,F(H1),4\n"
    problems = check_program(bad)
    assert any("before it is defined" in p for p in problems)
    assert any("ENDFIL" in p for p in problems)
    assert ascii_text("⊕ Ø.005Ⓜ A") == "POS DIA .005(M) A"


def test_stackup_adds_limits_in_code():
    rows = [{"row_id": "1", "requirement": "1.000 ± .005", "lower": "0.995", "upper": "1.005"},
            {"row_id": "2", "requirement": ".250 ± .002", "lower": "0.248", "upper": "0.252"}]
    r = stackup(rows, [{"row_id": "1", "sign": 1}, {"row_id": "2", "sign": -1}], [{"what": "shim", "value": ".100", "sign": 1}])
    assert r["worst_case"] == {"min": "0.843", "max": "0.857"}
    assert r["nominal"] == "0.85"
    assert r["rss"]["plus_minus"] == "0.0054"


def test_spoken_value_parser():
    assert parse_value("point one eight eight five") == ".1885"
    assert parse_value("conforms") == "OK"
    assert parse_value("minus point zero zero two") == "-.002"
    assert parse_value("um") is None


def test_gauge_helpers():
    assert decode([{"lit": "bc", "point_after": True}, {"lit": "abdeg", "point_after": False}]) == "1.2"
    assert decode([{"lit": "xyz", "point_after": False}]) == "?"
    assert abs(to_units("25.40", "mm", False, "in") - 1.0) < 1e-9
    assert to_units("0.1885", "in", True, "in") == -0.1885


def test_code_reader_on_a_synthetic_display():
    import random
    import sys

    sys.path.insert(0, str(ROOT / "eval"))
    from make_gauge_photos import instrument

    for reading in ("0.9995", "3.5545", "27.65"):
        im = instrument(reading, "in", "caliper", random.Random(1))
        got = segment_read(im) or ""
        assert "".join(ch for ch in got if ch.isdigit()).lstrip("0") == reading.replace(".", "").lstrip("0"), got
