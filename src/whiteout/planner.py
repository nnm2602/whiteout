"""Measuring plan: an instrument and a method for every Form 3 line.

Code picks the instrument from the requirement type and the width of the tolerance band (the instrument should
resolve a tenth of the band). Muse Spark only sorts the notes, which have no number, into how they are verified.
Lines are then grouped into setups so the part is picked up as few times as possible.
"""

from __future__ import annotations

import re

from .forms import _num_key, fmt
from .model_client import VisionClient
from .prompts import _obj
from .types import Characteristic, Requirement

MM_PER_IN = 25.4

# Setups in the order a shop usually works: paperwork and eyes first, hand tools, surface plate, CMM last.
SETUPS = ["Documents", "Visual", "Hand tools", "Gauges", "Surface plate", "Optical comparator", "Surface finish",
          "CMM", "Not measured"]

NOTE_METHODS = ["certificate", "visual", "functional", "process record", "measure", "not inspectable"]

NOTE_SCHEMA = _obj({"items": {"type": "array", "items": _obj({
    "row_id": {"type": "string"},
    "method": {"type": "string", "enum": NOTE_METHODS},
    "instrument": {"type": "string"},
    "how": {"type": "string"},
})}})

NOTE_PROMPT = (
    "These are notes from an aerospace engineering drawing. For first article inspection, say how each note is "
    "verified: 'certificate' (material, finish, heat treat or plating specs proven by a supplier certificate), "
    "'visual' (marking, burrs, broken edges, workmanship), 'functional' (a fit or function test), 'process record' "
    "(a process document or log shows it), 'measure' (it states a number to measure, such as a flatness or an edge "
    "break size), or 'not inspectable' (general information, reference only, a heading). instrument: the tool or "
    "document, in a few words. how: one short sentence an inspector can follow. Keep each row_id as given.\n\nNotes:\n"
)

_HOLE = re.compile(r"\b(hole|bore|thru|counterbore|cbore|csk|countersink|slot|pin hole|reamed)\b|[Ø⌀∅]|\bDIA\b", re.I)
_INTERNAL_THREAD = re.compile(r"\b\d*[- ]?\d+\s*(UNC|UNF|UNEF|UNJ\w*|NPT|M\d)[\w-]*\s*-?\s*\d?B\b|\bTAP\b|\b\d[A-Z]?B\b", re.I)


def _band_in(q: Requirement, units: str) -> float | None:
    if q.lower is None or q.upper is None:
        return None
    band = abs(q.upper - q.lower)
    return band / MM_PER_IN if units == "mm" else band


def _num(v: float | None, units: str) -> str:
    return "" if v is None else fmt(v) + (" mm" if units == "mm" else "")


def _pick(c: Characteristic, q: Requirement, units: str) -> tuple[str, str, str, str]:
    """(setup, instrument, method, why) for one requirement."""
    t = q.type
    band = _band_in(q, units)
    text = f"{c.text} {q.text} {q.label}"
    sym = (q.gdt_symbol or "").lower()
    if t == "basic":
        return ("Not measured", "", "Basic dimension: no tolerance of its own. It is checked through the GD&T frame "
                "that refers to it.", "basic")
    if t == "reference":
        return ("Not measured", "", "Reference dimension: for information, not inspected.", "reference")
    if t == "thread":
        if _INTERNAL_THREAD.search(text):
            return ("Gauges", "thread plug gauge, GO and NO-GO", f"GO gauge must thread in fully; NO-GO must not "
                    f"enter more than two turns. Thread: {q.text or c.text}.", "internal thread")
        return ("Gauges", "thread ring gauge, GO and NO-GO", f"GO ring must thread on fully; NO-GO must not. Thread: "
                f"{q.text or c.text}.", "external thread")
    if t == "surface_finish":
        return ("Surface finish", "profilometer", f"Measure Ra across the lay at three places; the highest reading "
                f"must not exceed {_num(q.upper, units) or 'the printed value'}.", "surface finish")
    if t == "gdt":
        datums = "|".join(q.datums)
        ref = f" relative to datums {datums}" if datums else ""
        if any(k in sym for k in ("runout", "↗", "⌰")):
            return ("Surface plate", "dial indicator on V-blocks or between centers",
                    f"Rotate the part on datum {datums or 'axis'}; total indicator reading must not exceed "
                    f"{_num(q.upper, units)}.", "runout")
        if "flat" in sym or "⏥" in sym:
            return ("Surface plate", "surface plate and dial indicator",
                    f"Support the face on three adjustable points, zero it, sweep the indicator; full range must not "
                    f"exceed {_num(q.upper, units)}.", "flatness")
        if "straight" in sym or "⏤" in sym:
            return ("Surface plate", "surface plate and dial indicator",
                    f"Sweep the line element; full range must not exceed {_num(q.upper, units)}.", "straightness")
        if any(k in sym for k in ("circular", "cylindric", "○", "⌭")):
            return ("CMM", "CMM or roundness tester", f"Scan the feature; form error must not exceed "
                    f"{_num(q.upper, units)}.", "form")
        if any(k in sym for k in ("parallel", "perpend", "angular", "∥", "⟂", "∠")):
            return ("CMM", "CMM (or surface plate and indicator)", f"Align on {datums or 'the datum'}; the "
                    f"orientation error must not exceed {_num(q.upper, units)}.", "orientation")
        # position, profile, concentricity, symmetry and anything unnamed: needs a datum reference frame
        what = "position" if ("pos" in sym or "⊕" in sym or "⌖" in sym or "⊕" in c.text) else (
            "profile" if ("profile" in sym or "⌓" in sym or "⌒" in sym or "⌓" in c.text) else "the tolerance")
        mod = " Bonus tolerance applies with the MMC modifier: use the measured size." if (
            "MMC" in " ".join(q.modifiers).upper() or "Ⓜ" in c.text) else ""
        return ("CMM", "CMM", f"Align on the datum reference frame{ref}; evaluate {what} against "
                f"{_num(q.upper, units)}.{mod}", what)
    if t == "angle":
        if band is not None and band >= 1.0:
            return ("Hand tools", "universal bevel protractor", "Read the angle; it must fall within the limits.",
                    "angle, wide band")
        return ("CMM", "CMM or sine bar with gauge blocks", "Angle band under one degree needs a CMM or a sine bar.",
                "angle, tight band")
    if t == "radius":
        return ("Optical comparator", "radius gauges or optical comparator",
                "Check the radius against the gauge leaves for the low and high limit, or on the comparator chart.",
                "radius")
    if t == "chamfer":
        return ("Optical comparator", "optical comparator or chamfer gauge", "Measure the chamfer size and angle.",
                "chamfer")
    if q.attribute or t in ("note", "other") and band is None:
        return ("Visual", "visual", "Check by inspection.", "attribute")
    if band is None:
        return ("Hand tools", "caliper", "No tolerance found: confirm the limits before measuring.", "no limits")
    if t == "depth":
        if band >= 0.005:
            return ("Hand tools", "depth caliper", "Measure from the face to the bottom of the feature.", "depth")
        return ("Hand tools", "depth micrometer", "Tight depth band: use the depth micrometer.", "depth, tight")
    if t == "location":
        if band >= 0.005:
            return ("Surface plate", "height gauge on the surface plate", "Rest the part on its datum face and "
                    "measure to the feature.", "location")
        return ("CMM", "CMM", "Tight location band: measure on the CMM from the datums.", "location, tight")
    hole = bool(_HOLE.search(text))
    if hole:
        lo_in = q.lower / MM_PER_IN if units == "mm" and q.lower else q.lower
        if band >= 0.010 and (lo_in is None or lo_in >= 0.5):
            return ("Hand tools", "caliper, inside jaws", "Measure the hole at two angles 90 degrees apart.",
                    "hole, wide band")
        if band >= 0.0005 and lo_in is not None and lo_in < 1.0:
            go, nogo = q.lower, q.upper
            return ("Gauges", "pin gauges, GO and NO-GO", f"GO pin {_num(go, units)} must enter; NO-GO pin "
                    f"{_num(nogo, units)} must not. Record the largest pin that enters.", "small hole")
        if band >= 0.0005:
            return ("Hand tools", "bore gauge, set with a ring or micrometer", "Measure at two depths and two angles.",
                    "large hole")
        return ("CMM", "CMM or air gauge", "Band under half a thousandth: hand tools cannot resolve it.",
                "hole, very tight")
    if band >= 0.005:
        return ("Hand tools", "caliper", "Measure at the feature; take the worst of three readings.", "size, wide band")
    if band >= 0.001:
        return ("Hand tools", "micrometer", "Tight band: micrometer, not a caliper.", "size, tight")
    return ("CMM", "CMM or gauge blocks and indicator", "Band under a thousandth: needs a CMM or comparison to "
            "gauge blocks.", "size, very tight")


def _note_setup(method: str, instrument: str) -> str:
    fixed = {"certificate": "Documents", "process record": "Documents", "visual": "Visual",
             "not inspectable": "Not measured"}
    if method in fixed:
        return fixed[method]
    i = instrument.lower()
    for key, setup in (("cmm", "CMM"), ("comparator", "Optical comparator"), ("profilometer", "Surface finish"),
                       ("indicator", "Surface plate"), ("height gauge", "Surface plate"), ("gauge", "Gauges"),
                       ("pin", "Gauges")):
        if key in i:
            return setup
    return "Hand tools"


def plan_rows(chars: list[Characteristic], units: str = "in", note_methods: dict | None = None) -> list[dict]:
    """One step per Form 3 row (row ids match forms.form3_rows)."""
    note_methods = note_methods or {}
    steps = []
    for c in sorted((c for c in chars if not c.ignored and c.number), key=_num_key):
        reqs = c.requirements or []
        for qi, q in enumerate(reqs):
            rid0 = c.number if len(reqs) == 1 else f"{c.number}.{qi + 1}"
            setup, instrument, method, why = _pick(c, q, units)
            nm = note_methods.get(rid0)
            if nm and (q.type in ("note", "other") or q.attribute):
                setup = _note_setup(nm["method"], nm["instrument"])
                instrument = "" if setup == "Not measured" else nm["instrument"]
                method, why = nm["how"], f"note: {nm['method']}"
            for inst in range(1, c.multiplier + 1):
                rid = rid0 + (f"-{inst}" if c.multiplier > 1 else "")
                steps.append({
                    "row_id": rid, "char_id": c.id, "number": c.number, "sheet": c.sheet, "zone": c.zone,
                    "requirement": q.text if len(reqs) > 1 else c.text, "type": q.type,
                    "lower": fmt(q.lower), "upper": fmt(q.upper), "datums": q.datums,
                    "setup": setup, "instrument": instrument, "method": method, "why": why,
                })
    return steps


async def note_methods(chars: list[Characteristic], client: VisionClient | None) -> dict:
    """Ask Muse Spark how each note is verified. Returns {row_id: {method, instrument, how}}."""
    notes = []
    for c in chars:
        if c.ignored or not c.number:
            continue
        reqs = c.requirements or []
        for qi, q in enumerate(reqs):
            if q.type in ("note", "other") or (q.attribute and q.type not in ("thread", "surface_finish")):
                rid = c.number if len(reqs) == 1 else f"{c.number}.{qi + 1}"
                notes.append((rid, (q.text or c.text)[:300]))
    if not notes or client is None:
        return {}
    prompt = NOTE_PROMPT + "\n".join(f"{rid}: {txt}" for rid, txt in notes)
    data = await client.ask_json([], prompt, NOTE_SCHEMA, name="note_methods", effort="low")
    known = {rid for rid, _ in notes}
    return {it["row_id"]: it for it in data.get("items", []) if it["row_id"] in known
            and it["method"] in NOTE_METHODS}


def sequence(steps: list[dict]) -> list[dict]:
    """Group steps into setups, in shop order, keeping balloon order inside each setup."""
    groups = []
    for name in SETUPS:
        rows = [s for s in steps if s["setup"] == name]
        if not rows:
            continue
        instruments = sorted({s["instrument"] for s in rows if s["instrument"]})
        groups.append({"setup": name, "instruments": instruments, "rows": [s["row_id"] for s in rows],
                       "count": len(rows)})
    return groups


async def make_plan(job_json: dict, client: VisionClient | None = None) -> dict:
    chars = [Characteristic(**c) for s in job_json["sheets"] for c in s["chars"]]
    units = (job_json["sheets"][0]["info"].get("units") if job_json["sheets"] else "in") or "in"
    nm = await note_methods(chars, client)
    steps = plan_rows(chars, units, nm)
    groups = sequence(steps)
    order = [rid for g in groups for rid in g["rows"]]
    by = {s["row_id"]: s for s in steps}
    return {
        "units": units,
        "steps": [by[r] for r in order],
        "setups": groups,
        "to_measure": sum(1 for s in steps if s["setup"] != "Not measured"),
        "instruments": sorted({s["instrument"] for s in steps if s["instrument"]}),
    }
