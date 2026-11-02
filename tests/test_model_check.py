import importlib.util
from pathlib import Path

import pytest

from whiteout.model_check import (
    Enum, Ref, Step, Typed, _decode, _parse_args, _parse_dim_text, compare, gdt_kind, read_step,
    requirements_from_text,
)
from whiteout.types import Characteristic, Requirement

ROOT = Path(__file__).resolve().parents[1]

# A small AP242 part: one 10 mm hole with a size dimension and a position frame, a flatness frame, datums A and B.
PART = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('part.stp','2026-10-28',(''),(''),'','','');
FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }'));
ENDSEC;
DATA;
#1=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));
#2=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));
#3=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNIT_ASSIGNED_CONTEXT((#1,#2))REPRESENTATION_CONTEXT('',''));
#10=SHAPE_ASPECT('','hole',#99,.F.);
#11=DIMENSIONAL_SIZE(#10,'diameter');
#12=(LENGTH_MEASURE_WITH_UNIT()MEASURE_REPRESENTATION_ITEM()MEASURE_WITH_UNIT(LENGTH_MEASURE(10.),#1)
REPRESENTATION_ITEM('nominal value'));
#13=SHAPE_DIMENSION_REPRESENTATION('',(#12),#3);
#14=DIMENSIONAL_CHARACTERISTIC_REPRESENTATION(#11,#13);
#15=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(-0.1),#1);
#16=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.1),#1);
#17=TOLERANCE_VALUE(#15,#16);
#18=PLUS_MINUS_TOLERANCE(#17,#11);
#20=DATUM('',$,#99,.F.,'A');
#21=DATUM_REFERENCE_COMPARTMENT('',$,#99,.F.,#20,$);
#22=DATUM('',$,#99,.F.,'B');
#23=DATUM_REFERENCE_COMPARTMENT('',$,#99,.F.,#22,(SIMPLE_DATUM_REFERENCE_MODIFIER(.MAXIMUM_MATERIAL_REQUIREMENT.)));
#24=DATUM_SYSTEM('',$,#99,.F.,(#21,#23));
#25=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.2),#1);
#26=(GEOMETRIC_TOLERANCE('Position.1','',#25,#10)GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE((#24))
GEOMETRIC_TOLERANCE_WITH_MODIFIERS((.MAXIMUM_MATERIAL_REQUIREMENT.))POSITION_TOLERANCE());
#27=TOLERANCE_ZONE_FORM('cylindrical or circular');
#28=TOLERANCE_ZONE('',$,#99,.F.,(#26),#27);
#29=FLATNESS_TOLERANCE('Flatness.1','',#30,#31);
#30=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.05),#1);
#31=SHAPE_ASPECT('','top',#99,.F.);
#32=DESCRIPTIVE_REPRESENTATION_ITEM('','ALL HOLES TO BE DEBURRED');
#40=CARTESIAN_POINT('',(0.,0.,0.));
#41=DIRECTION('',(0.,0.,1.));
#42=DIRECTION('',(1.,0.,0.));
#43=AXIS2_PLACEMENT_3D('',#40,#41,#42);
#44=CYLINDRICAL_SURFACE('',#43,5.);
#45=CARTESIAN_POINT('',(5.,0.,0.));
#46=VERTEX_POINT('',#45);
#47=CIRCLE('',#43,5.);
#48=EDGE_CURVE('',#46,#46,#47,.T.);
#49=ORIENTED_EDGE('',*,*,#48,.T.);
#50=EDGE_LOOP('',(#49));
#51=FACE_BOUND('',#50,.T.);
#52=CARTESIAN_POINT('',(0.,0.,-10.));
#53=AXIS2_PLACEMENT_3D('',#52,#41,#42);
#54=CARTESIAN_POINT('',(5.,0.,-10.));
#55=VERTEX_POINT('',#54);
#56=CIRCLE('',#53,5.);
#57=EDGE_CURVE('',#55,#55,#56,.T.);
#58=ORIENTED_EDGE('',*,*,#57,.F.);
#59=EDGE_LOOP('',(#58));
#60=FACE_BOUND('',#59,.T.);
#61=ADVANCED_FACE('',(#51,#60),#44,.F.);
#62=GEOMETRIC_ITEM_SPECIFIC_USAGE('','',#10,#3,#61);
ENDSEC;
END-ISO-10303-21;
"""

INCH_UNITS = """#1=(CONVERSION_BASED_UNIT('INCH',#4)LENGTH_UNIT()NAMED_UNIT(#5));
#4=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#6);
#5=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);
#6=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));"""


@pytest.fixture
def part(tmp_path):
    p = tmp_path / "part.stp"
    p.write_text(PART)
    return p


def char(cid, text, kind, reqs, mult=1):
    return Characteristic(id=cid, text=text, kind=kind, box=(0, 0, 10, 10), multiplier=mult, requirements=reqs)


def drawing(size_tol=0.1, pos="⌖|⌀0.2Ⓜ|A|BⓂ", flat=True, extra=False):
    chars = [
        char("s0c0", f"⌀10 ±{size_tol:g}", "hole_callout",
             [Requirement(type="size", text=f"⌀10 ±{size_tol:g}", nominal=10, lower=10 - size_tol,
                          upper=10 + size_tol, tolerance_source="printed")]),
        char("s0c1", pos, "gdt_frame", requirements_from_text(pos, "gdt_frame")),
    ]
    if flat:
        chars.append(char("s0c2", "⏥ 0.05", "gdt_frame",
                          [Requirement(type="gdt", text="⏥ 0.05", gdt_symbol="flatness", nominal=0.05, lower=0,
                                       upper=0.05, tolerance_source="printed")]))
    if extra:
        chars.append(char("s0c3", "25 ±0.2", "dimension", requirements_from_text("25 ±0.2")))
    return chars


# ---- parsing ------------------------------------------------------------------------------------------------------

def test_parse_args_values():
    args, _ = _parse_args("('a''b',#12,.T.,$,*,(1.,-2.5E-1),LENGTH_MEASURE(0.5),())")
    assert args == ["a'b", Ref(12), Enum("T"), None, "*", [1.0, -0.25], Typed("LENGTH_MEASURE", 0.5), []]


def test_decode_unicode():
    assert _decode("\\X2\\2300\\X0\\10") == "⌀10"


def test_complex_entity_and_users():
    s = Step(PART)
    assert set(s.get(26)) >= {"GEOMETRIC_TOLERANCE", "POSITION_TOLERANCE", "GEOMETRIC_TOLERANCE_WITH_MODIFIERS"}
    assert 26 in s.types("POSITION_TOLERANCE")
    assert 18 in s.users(11)


# ---- reader -------------------------------------------------------------------------------------------------------

def test_read_step_pmi(part):
    r = read_step(part)
    assert r["units"]["length"] == "mm"
    (d,) = r["dimensions"]
    assert d["type"] == "size" and d["nominal"] == 10 and d["lower"] == pytest.approx(9.9)
    assert d["upper"] == pytest.approx(10.1) and d["text"] == "⌀10 ±0.1"
    pos = next(t for t in r["tolerances"] if t["characteristic"] == "position")
    assert pos["text"] == "⌖|⌀0.2Ⓜ|A|BⓂ" and pos["zone"] == "⌀"
    assert [x["label"] for x in pos["datums"]] == ["A", "B"]
    flat = next(t for t in r["tolerances"] if t["characteristic"] == "flatness")
    assert flat["value"] == 0.05 and flat["datums"] == []
    assert r["datums"]["labels"] == ["A", "B"]
    assert r["notes"] == ["ALL HOLES TO BE DEBURRED"]


def test_read_step_holes(part):
    r = read_step(part)
    (h,) = r["holes"]
    assert h["diameter"] == 10 and h["depth"] == 10
    assert h["xyz"] == [0, 0, 0] and h["bottom"] == [0, 0, -10] and h["axis"] == [0, 0, 1]
    assert set(h["pmi"]) == {"#11", "#26"}
    assert r["hole_groups"][0]["count"] == 1


def test_inch_units(tmp_path):
    text = PART.replace("#1=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));", INCH_UNITS)
    p = tmp_path / "inch.stp"
    p.write_text(text)
    r = read_step(p)
    assert r["units"] == {"length": "in", "angle": "deg", "mm_per_unit": 25.4}
    assert r["dimensions"][0]["nominal"] == 10  # values stay in the file's own unit


# ---- printed text -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expect", [
    ("⌀20 +0.05/-0.10", (20, 19.9, 20.05)),
    ("Ø35 0 -0.2", (35, 34.8, 35)),
    ("⌀35.2/34.8", (35, 34.8, 35.2)),
    ("60° ±0.5°", (60, 59.5, 60.5)),
    ("4X ⌀.415 ±.008", (0.415, 0.407, 0.423)),
    ("1.680", (1.68, None, None)),
])
def test_parse_dim_text(text, expect):
    got = _parse_dim_text(text)
    assert got[0] == pytest.approx(expect[0])
    for g, e in zip(got[1:], expect[1:], strict=True):
        assert (g is None and e is None) or g == pytest.approx(e)


def test_requirements_from_text_splits_callouts():
    reqs = requirements_from_text("⌴ ⌀.245 - .255 ↧.100 ± .010", "hole_callout")
    assert [r.type for r in reqs] == ["size", "depth"]
    reqs = requirements_from_text("30X ⌵⌀20 ±0.3 X 90° ±1°", "hole_callout")
    assert [r.type for r in reqs] == ["size", "angle"]
    (f,) = requirements_from_text("⌖|⌀.025Ⓜ|A|B|C", "gdt_frame")
    assert f.type == "gdt" and f.nominal == 0.025 and f.datums == ["A", "B", "C"]
    assert requirements_from_text("NOTE 1. DEBURR", "note") == []


def test_gdt_kind():
    assert gdt_kind("⊕") == "position"
    assert gdt_kind("profile of a surface") == "surface_profile"
    assert gdt_kind("⌰") == "total_runout"


# ---- compare ------------------------------------------------------------------------------------------------------

def test_compare_all_agree(part):
    res = compare(part, drawing())
    s = res["summary"]
    assert (s["matched"], s["differs"], s["model_only"], s["drawing_only"]) == (3, 0, 0, 0)
    assert res["drawing_units"] == "mm"


def test_compare_reports_value_and_tolerance(part):
    res = compare(part, drawing(size_tol=0.2, pos="⌖|⌀0.3Ⓜ|A|BⓂ"))
    by = {d["model"]["id"]: d for d in res["differs"]}
    assert [f["field"] for f in by["#11"]["fields"]] == ["tolerance"]
    assert by["#11"]["drawing"]["id"] == "s0c0" and by["#11"]["drawing"]["box"] == [0, 0, 10, 10]
    assert by["#26"]["fields"][0] == {"field": "tolerance value", "model": "0.2", "drawing": "0.3"}


def test_compare_datum_modifier_and_only_lists(part):
    res = compare(part, drawing(pos="⌖|⌀0.2Ⓜ|A|B", flat=False, extra=True))
    (d,) = res["differs"]
    assert d["fields"][0]["field"] == "datums"
    assert [m["id"] for m in res["model_only"]] == ["#29"]
    assert [x["text"] for x in res["drawing_only"]] == ["25 ±0.2"]


def test_compare_converts_units(part):
    inch = [char("s0c0", "⌀.394 ±.004", "hole_callout", requirements_from_text("⌀.394 ±.004")),
            char("s0c1", "⌖|⌀.008Ⓜ|A|BⓂ", "gdt_frame", requirements_from_text("⌖|⌀.008Ⓜ|A|BⓂ", "gdt_frame"))]
    res = compare(part, inch)
    assert res["drawing_units"] == "in"
    assert res["summary"]["matched"] == 2


def test_compare_accepts_char_dicts(part):
    res = compare(part, [c.model_dump() for c in drawing()])
    assert res["summary"]["matched"] == 3


# ---- planted edits ------------------------------------------------------------------------------------------------

def test_editor_changes_and_removes(tmp_path):
    spec = importlib.util.spec_from_file_location("make_model_edits", ROOT / "eval" / "make_model_edits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ed = mod.Editor(PART)
    ed.set_measure(17, 16, 0.3)  # upper bound of the size tolerance
    ed.set_measure(26, 25, 0.4)  # position value
    ed.delete(29)  # flatness gone
    p = tmp_path / "edited.stp"
    p.write_text(ed.finish())
    r = read_step(p)
    assert r["dimensions"][0]["upper"] == pytest.approx(10.3)
    assert [t["value"] for t in r["tolerances"]] == [0.4]
    res = compare(p, drawing())
    assert {d["model"]["id"] for d in res["differs"]} == {"#11", "#26"}
    assert [x["id"] for x in res["drawing_only"]] == ["s0c2"]


# ---- NIST files, when downloaded ----------------------------------------------------------------------------------

NIST = ROOT / "data/raw/nist_pmi_step/nist_ctc_01_asme1_ap242-e1.stp"


@pytest.mark.skipif(not NIST.exists(), reason="NIST AP242 files not downloaded")
def test_nist_ctc_01():
    r = read_step(NIST)
    texts = {x["text"] for x in r["dimensions"] + r["tolerances"]}
    assert {"⌀25 ±0.15", "60° ±0.5°", "⌀35.2/34.8", "⌓|1.25|A|B|C", "⊥|1.5|A", "⏥|0.2"} <= texts
    assert {g["diameter"]: g["count"] for g in r["hole_groups"]} == {35.0: 4, 25.0: 4, 20.0: 2}
