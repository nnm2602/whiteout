"""Planted disagreements: copy the NIST STEP models and change some PMI values, recording every edit.

Targets are model PMI that agree with the drawing before the edit (compare() on the unedited pair), so each edit
should turn an agreement into a reported disagreement. Per model, up to five edits on different PMI:
  tolerance      a plus/minus tolerance doubled (both bounds)
  nominal        a toleranced dimension's nominal value moved by 5%
  gdt_value      a geometric tolerance value doubled
  no_tolerance   a plus/minus tolerance removed (the dimension stays, untoleranced)
  removed_gdt    a geometric tolerance removed from the model

Edited copies go to data/raw/model_edits/ (not in git; rerun this script to rebuild them), the edit list to
eval/model_edits/<case>.json.

uv run python eval/make_model_edits.py
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from run_model_check import CASES, OUT  # noqa: E402

from whiteout.model_check import Ref, Step, Typed, compare, load_chars, read_step, to_json  # noqa: E402

EDIT_JSON = ROOT / "eval" / "model_edits"
EDIT_STP = ROOT / "data" / "raw" / "model_edits"


class Editor:
    """Text edits on a STEP file: replace a measure value by cloning its record, delete records."""

    def __init__(self, text: str):
        self.text = text
        self.step = Step(text)
        self.next_id = max(self.step.raw) + 1
        self.added: list[str] = []

    def record_span(self, i: int) -> tuple[int, int]:
        m = re.search(rf"(?m)^\s*#{i}\s*=", self.text)
        if not m:
            raise KeyError(i)
        rec = re.compile(r"(?:[^;']|'(?:[^']|'')*')*;", re.S).match(self.text, m.end())
        return m.start(), rec.end()

    def record(self, i: int) -> str:
        a, b = self.record_span(i)
        return self.text[a:b]

    def replace_record(self, i: int, new: str) -> None:
        a, b = self.record_span(i)
        self.text = self.text[:a] + new + self.text[b:]

    def delete(self, i: int) -> None:
        a, b = self.record_span(i)
        end = b + 1 if self.text[b:b + 1] == "\n" else b
        self.text = self.text[:a] + self.text[end:]

    def set_measure(self, owner: int, ref: int, value: float) -> str:
        """Point owner at a copy of measure #ref with a new value; returns the new id."""
        rec = self.record(ref).strip()
        body = rec[rec.find("=") + 1:]
        new_body, n = re.subn(r"((?:LENGTH|PLANE_ANGLE|POSITIVE_LENGTH)_MEASURE\s*\(\s*)([-+0-9.Ee]+)(\s*\))",
                              lambda m: f"{m.group(1)}{_num(value)}{m.group(3)}", body, count=1)
        if n != 1:
            raise ValueError(f"no measure value in #{ref}")
        nid = self.next_id
        self.next_id += 1
        self.added.append(f"#{nid}={new_body.strip()}")
        own = self.record(owner)
        own_new = re.sub(rf"#{ref}(?!\d)", f"#{nid}", own)
        self.replace_record(owner, own_new)
        return f"#{nid}"

    def finish(self) -> str:
        if not self.added:
            return self.text
        i = self.text.rfind("ENDSEC;", 0, self.text.find("END-ISO-10303-21"))
        return self.text[:i] + "\n".join(self.added) + "\n" + self.text[i:]


def _num(v: float) -> str:
    s = f"{v:.10g}"
    return s if any(c in s for c in ".Ee") else s + "."


def _refs(v) -> list[int]:
    if isinstance(v, Ref):
        return [v.id]
    if isinstance(v, Typed):
        return _refs(v.value)
    if isinstance(v, list):
        return [r for x in v for r in _refs(x)]
    return []


def plan(case: str, rng: random.Random) -> tuple[list[dict], dict]:
    step_path, _ = CASES[case]
    model = read_step(ROOT / step_path)
    chars, units = load_chars(OUT / "drawings" / f"{case}.json")
    res = compare(model, chars, units)
    agree = {m["model"]["id"]: m["drawing"] for m in res["matched"]}
    st = Step.read(ROOT / step_path)
    pm_tol = {}
    for i in st.types("PLUS_MINUS_TOLERANCE"):
        rng_ref, dim = st.get(i)["PLUS_MINUS_TOLERANCE"]
        if isinstance(dim, Ref) and "TOLERANCE_VALUE" in st.get(rng_ref):
            pm_tol[f"#{dim.id}"] = (i, rng_ref.id)
    dims = [d for d in model["dimensions"] if d["id"] in agree and d["tolerance"] == "plus_minus"
            and d["id"] in pm_tol and d["nominal"]]
    gdts = [t for t in model["tolerances"] if t["id"] in agree and t["value"]]
    rng.shuffle(dims)
    rng.shuffle(gdts)
    edits: list[dict] = []
    used: set[str] = set()

    def take(pool, pred=lambda x: True):
        for x in pool:
            if x["id"] not in used and pred(x):
                used.add(x["id"])
                return x
        return None

    d = take(dims, lambda x: abs((x["plus"] or 0) - (x["minus"] or 0)) > 0)
    if d:
        edits.append({"edit": "tolerance", "target": d["id"], "before": d["text"], "expect": "differs",
                      "field": "tolerance", "factor": 2.0})
    d = take(dims)
    if d:
        edits.append({"edit": "nominal", "target": d["id"], "before": d["text"], "expect": "differs",
                      "field": "nominal", "factor": 1.05})
    t = take(gdts, lambda x: x["value"] > 0)
    if t:
        edits.append({"edit": "gdt_value", "target": t["id"], "before": t["text"], "expect": "differs",
                      "field": "tolerance value", "factor": 2.0})
    d = take(dims)
    if d:
        edits.append({"edit": "no_tolerance", "target": d["id"], "before": d["text"], "expect": "differs",
                      "field": "tolerance"})
    t = take(gdts, lambda x: not x.get("composite_of") and not any(
        y.get("composite_of") == x["id"] for y in model["tolerances"]))
    if t:
        edits.append({"edit": "removed_gdt", "target": t["id"], "before": t["text"], "expect": "drawing_only"})
    for e in edits:
        e["drawing"] = {k: agree[e["target"]][k] for k in ("id", "text", "requirement_text", "sheet")}
    return edits, pm_tol


def apply(case: str, edits: list[dict], pm_tol: dict) -> Path:
    step_path, _ = CASES[case]
    src = ROOT / step_path
    ed = Editor(src.read_text(errors="replace"))
    st = ed.step
    for e in edits:
        tid = int(e["target"][1:])
        if e["edit"] == "tolerance":
            _, rng_id = pm_tol[e["target"]]
            lo, hi = _refs(st.get(rng_id)["TOLERANCE_VALUE"])[:2]
            vals = []
            for ref in (lo, hi):
                v = _measure_value(st, ref)
                vals.append(v * e["factor"])
            ed.set_measure(rng_id, lo, vals[0])
            # the second bound is still referenced by the original record text
            ed.set_measure(rng_id, hi, vals[1])
            e["after"] = f"bounds {vals[0]:g} / {vals[1]:g}"
        elif e["edit"] == "nominal":
            for dcr in st.types("DIMENSIONAL_CHARACTERISTIC_REPRESENTATION"):
                dim, rep = st.get(dcr)["DIMENSIONAL_CHARACTERISTIC_REPRESENTATION"]
                if dim.id != tid:
                    continue
                for item in _refs(list(st.get(rep).values())[0][1:2]):
                    ent = st.get(item)
                    label = (ent.get("REPRESENTATION_ITEM") or [""])[0] or ""
                    v = _measure_value(st, item)
                    if v is not None and "limit" not in label.lower():
                        ed.set_measure(rep.id, item, v * e["factor"])
                        e["after"] = f"nominal {v * e['factor']:g} (was {v:g})"
                        break
                break
        elif e["edit"] == "gdt_value":
            ent = st.get(tid)
            base = ent.get("GEOMETRIC_TOLERANCE") or next(v for k, v in ent.items() if k.endswith("_TOLERANCE"))
            mag = base[2].id
            v = _measure_value(st, mag)
            ed.set_measure(tid, mag, v * e["factor"])
            e["after"] = f"value {v * e['factor']:g} (was {v:g})"
        elif e["edit"] == "no_tolerance":
            pmt, _ = pm_tol[e["target"]]
            ed.delete(pmt)
            e["after"] = f"plus/minus tolerance #{pmt} deleted"
        elif e["edit"] == "removed_gdt":
            ed.delete(tid)
            e["after"] = f"geometric tolerance #{tid} deleted"
    EDIT_STP.mkdir(parents=True, exist_ok=True)
    out = EDIT_STP / f"{case}_edited.stp"
    out.write_text(ed.finish())
    return out


def _measure_value(st: Step, ref: int) -> float | None:
    for args in st.get(ref).values():
        for a in args:
            if isinstance(a, Typed) and isinstance(a.value, float):
                return a.value
    return None


def main() -> None:
    rng = random.Random(20261028)
    EDIT_JSON.mkdir(parents=True, exist_ok=True)
    total = 0
    for case in CASES:
        if not (OUT / "drawings" / f"{case}.json").exists():
            print(f"{case}: no drawing read yet, skipped")
            continue
        edits, pm_tol = plan(case, rng)
        path = apply(case, edits, pm_tol)
        check = read_step(path)  # the edited copy must still parse
        rec = {"case": case, "source": CASES[case][0], "edited": str(path.relative_to(ROOT)),
               "pmi_after": check["counts"]["semantic_pmi"], "edits": edits}
        (EDIT_JSON / f"{case}.json").write_text(to_json(rec))
        total += len(edits)
        print(f"{case}: {len(edits)} edits -> {path.name}")
    print(f"{total} edits")


if __name__ == "__main__":
    main()
