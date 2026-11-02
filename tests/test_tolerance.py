from whiteout.tolerance import build_requirement, evaluate, parse_limits


def lim(t):
    n, lo, hi, how = parse_limits(t)
    return (round(n, 6) if n is not None else None, round(lo, 6) if lo is not None else None,
            round(hi, 6) if hi is not None else None, how)


def test_plus_minus():
    assert lim("18.900±.002") == (18.9, 18.898, 18.902, "plus_minus")
    assert lim("4X R.032 ±.010") == (0.032, 0.022, 0.042, "plus_minus")
    assert lim("↧.250 ± .010") == (0.25, 0.24, 0.26, "plus_minus")


def test_limits():
    assert lim("2X Ø.370 - .380") == (0.375, 0.37, 0.38, "limits")
    assert lim(".3910/.3895 DIA HOLE") == (0.39025, 0.3895, 0.391, "limits")
    assert lim("⊔ Ø.245 - .255") == (0.25, 0.245, 0.255, "limits")
    assert lim(".190-.196 (#11 DRILL)") == (0.193, 0.19, 0.196, "limits")


def test_unilateral_and_plain():
    assert lim("1.250 +.002 -.000") == (1.25, 1.25, 1.252, "unilateral")
    assert lim("2.500") == (2.5, None, None, "nominal_only")
    assert lim("2 9/16") == (2.5625, None, None, "nominal_only")
    assert lim("5X 45°")[0] == 45


def test_general_tolerance_by_places():
    gen = [{"applies_to": ".XX", "plus_minus": ".01"}, {"applies_to": ".XXX", "plus_minus": ".005"},
           {"applies_to": "ANGLES", "plus_minus": "0.5"}]
    q = build_requirement({"type": "size", "text": ".768"}, "4X .768", gen)
    assert (q.lower, q.upper, q.tolerance_source) == (0.763, 0.773, "general")
    q = build_requirement({"type": "angle", "text": "45°"}, "5X 45°", gen)
    assert (q.lower, q.upper) == (44.5, 45.5)


def test_no_false_flags_on_clean_reads():
    q = build_requirement({"type": "basic", "text": "2.500", "nominal": "2.500"}, "2.500", [])
    assert q.flags == []
    q = build_requirement({"type": "size", "text": "2X Ø.370 - .380", "nominal": ".375"}, "2X Ø.370 - .380", [])
    assert q.flags == [] and (q.lower, q.upper) == (0.37, 0.38)


def test_made_up_number_is_flagged():
    q = build_requirement({"type": "size", "text": "", "nominal": ".500"}, "2X Ø.370 - .380", [])
    assert any("differs" in f or "not found" in f for f in q.flags)


def test_gdt_and_evaluate():
    q = build_requirement({"type": "gdt", "text": "⊕ Ø.005Ⓜ A B C", "nominal": ".005", "gdt_symbol": "position",
                           "datums": ["A", "B", "C"]}, "⊕ | Ø.005Ⓜ | A | B | C", [])
    assert (q.lower, q.upper) == (0.0, 0.005) and not q.flags
    assert evaluate(q, "0.004") == "pass" and evaluate(q, ".0051") == "fail"
    n = build_requirement({"type": "note", "text": "BREAK ALL EDGES .015-.031"}, "BREAK ALL EDGES .015-.031", [])
    assert evaluate(n, "ok") in ("pass",) or n.lower is not None
