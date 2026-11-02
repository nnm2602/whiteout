"""A starting CMM program in DMIS for the lines the plan sends to the CMM.

The drawing gives sizes, tolerances and datums but not where a hole sits in 3D. When the part's STEP model is
available, hole centres and axes come from it (matched by diameter); otherwise the program runs the features in
manual mode and the operator probes them. A CMM programmer reviews the program before it runs.
"""

from __future__ import annotations

import re
from datetime import date

from .types import Characteristic

MM_PER_IN = 25.4
MAJOR = {"DMISMN", "FILNAM", "UNITS", "WKPLAN", "PRCOMP", "FLY", "SNSET", "MODE", "FEAT", "MEAS", "PTMEAS", "ENDMES",
         "DATDEF", "DATSET", "TOL", "OUTPUT", "ENDFIL", "DISPLY", "TEXT", "RECALL"}


def _label(s: str) -> str:
    return re.sub(r"[^A-Z0-9_]", "_", s.upper())[:16]


_ASCII = {"Ø": "DIA ", "⌀": "DIA ", "∅": "DIA ", "⊔": "CBORE ", "⌵": "CSK ", "↧": "DEPTH ", "⊕": "POS ", "⌖": "POS ",
          "±": "+/-", "°": " DEG", "Ⓜ": "(M)", "Ⓛ": "(L)", "⌓": "PROFILE ", "⏥": "FLAT ", "⊥": "PERP ",
          "∠": "ANGULARITY ", "▱": "FLAT ", "◡": "PROFILE ", "|": " "}


def ascii_text(s: str) -> str:
    """DMIS files are plain ASCII; drawing symbols become words."""
    for k, v in _ASCII.items():
        s = s.replace(k, v)
    s = s.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


def _f(v: float) -> str:
    return f"{v + 0.0:.4f}".replace("-0.0000", "0.0000")


def _match_holes(diameter: float, holes: list[dict], used: set[int], units: str, count: int) -> list[dict]:
    """Model holes with this diameter (within 0.0015 in), not used yet, up to count."""
    tol = 0.0015 * (MM_PER_IN if units == "mm" else 1)
    out = []
    for i, h in enumerate(holes):
        if i in used:
            continue
        if abs(h["diameter"] - diameter) <= tol:
            out.append({**h, "_i": i})
            if len(out) == count:
                break
    for h in out:
        used.add(h["_i"])
    return out


def _material(text: str) -> str:
    if re.search(r"Ⓜ|\(M\)|\bMMC\b", text):
        return ",MMC"
    if re.search(r"Ⓛ|\(L\)|\bLMC\b", text):
        return ",LMC"
    return ",RFS"


def _nearest_hole(c: Characteristic, chars: dict, defined: dict[str, list[str]], reach: int = 450) -> list[str]:
    """The hole a position frame controls: the hole callout closest to the frame on the same sheet."""
    cx, cy = (c.box[0] + c.box[2]) / 2, (c.box[1] + c.box[3]) / 2
    best, dist = None, reach
    for cid, labels in defined.items():
        h = chars[cid]
        if h.sheet != c.sheet:
            continue
        d = max(0.0, max(h.box[0] - cx, cx - h.box[2])) + max(0.0, max(h.box[1] - cy, cy - h.box[3]))
        if h.number.isdigit() and c.number.split("-")[0].isdigit() and int(h.number) == int(c.number.split("-")[0]) - 1:
            d *= 0.3  # the callout ballooned just before the frame is usually the one it sits under
        if d < dist:
            best, dist = labels, d
    return best or []


def make_program(job_json: dict, plan: dict, holes: list[dict] | None = None, part: str = "") -> dict:
    """Return {"program": str, "features": int, "from_model": int, "manual": int}.

    holes: [{"xyz": [x, y, z], "axis": [i, j, k], "diameter": d}] in the drawing's units.
    """
    holes = holes or []
    units = plan.get("units", "in")
    chars = {c["id"]: Characteristic(**c) for s in job_json["sheets"] for c in s["chars"]}
    info = job_json["sheets"][0]["info"] if job_json["sheets"] else {}
    part = part or info.get("drawing_number") or info.get("part_number") or "PART"
    cmm = [s for s in plan["steps"] if s["setup"] == "CMM" or s["type"] in ("size",) and "hole" in s["why"]]
    datums = sorted({d.strip()[:1].upper() for s in cmm for d in s.get("datums", []) if d.strip()[:1].isalpha()})

    L: list[str] = []
    L.append(f"DMISMN/'Whiteout starting program for {ascii_text(part)}',05.3")
    L.append(f"$$ Made {date.today().isoformat()} from the ballooned drawing. Review before running.")
    L.append(f"FILNAM/'{_label(part)}',05.3")
    L.append("UNITS/MM,ANGDEC" if units == "mm" else "UNITS/INCH,ANGDEC")
    L.append("WKPLAN/XYPLAN")
    L.append("PRCOMP/ON")
    L.append("SNSET/APPRCH,0.1" if units == "in" else "SNSET/APPRCH,2.5")
    L.append("SNSET/RETRCT,0.1" if units == "in" else "SNSET/RETRCT,2.5")

    if datums:
        L.append("$$ Datum reference frame: probe the datum features by hand, then align.")
        L.append("MODE/MAN")
        kinds = ["PLANE", "PLANE", "PLANE"]
        for k, d in enumerate(datums[:3]):
            L.append(f"F(DAT_{d})=FEAT/{kinds[k]},CART,0,0,0,0,0,1")
            L.append(f"MEAS/{kinds[k]},F(DAT_{d}),{4 if k == 0 else 3}")
            L.append("ENDMES")
            L.append(f"DATDEF/FA(DAT_{d}),DAT({d})")
        order = ["ZDIR,ZORIG", "XDIR,YORIG", "XORIG"]
        parts = [f"DAT({d}),{order[k]}" for k, d in enumerate(datums[:3])]
        L.append(f"D(DRF)=DATSET/{','.join(parts)}")

    used: set[int] = set()
    n_feat = n_model = n_manual = 0
    defined: dict[str, list[str]] = {}  # char id -> hole feature labels
    for s in cmm:
        c = chars.get(s["char_id"])
        if c is None:
            continue
        lab = _label(f"B{s['row_id']}")
        lo, hi = s["lower"], s["upper"]
        if s["type"] == "size" and lo and hi:
            lo_v, hi_v = float(lo), float(hi)
            nom = (lo_v + hi_v) / 2
            inst = 1
            m = re.search(r"-(\d+)$", s["row_id"])
            if m:
                inst = int(m.group(1))
            found = _match_holes(nom, holes, used, units, 1) if holes else []
            L.append(f"$$ Balloon {s['row_id']}: {ascii_text(s['requirement'])}")
            if found:
                h = found[0]
                x, y, z = h["xyz"]
                i, j, k = h["axis"]
                L.append("MODE/AUTO,PROG,MAN")
                L.append(f"F({lab})=FEAT/CIRCLE,INNER,CART,{_f(x)},{_f(y)},{_f(z)},{_f(i)},{_f(j)},{_f(k)},{_f(nom)}")
                n_model += 1
            else:
                L.append("MODE/MAN")
                L.append(f"F({lab})=FEAT/CIRCLE,INNER,CART,0,0,0,0,0,1,{_f(nom)}")
                L.append("$$ Location is not on the drawing: probe this hole by hand.")
                n_manual += 1
            L.append(f"MEAS/CIRCLE,F({lab}),4")
            L.append("ENDMES")
            L.append(f"T(D{lab[1:]})=TOL/DIAM,{_f(lo_v - nom)},{_f(hi_v - nom)}")
            L.append(f"OUTPUT/FA({lab}),TA(D{lab[1:]})")
            defined.setdefault(c.id, []).append(lab)
            n_feat += 1
        elif s["type"] == "gdt" and hi:
            t = float(hi)
            why = s["why"]
            L.append(f"$$ Balloon {s['row_id']}: {ascii_text(s['requirement'])}")
            targets = _nearest_hole(c, chars, defined)
            if why == "position" and targets:
                L.append(f"T(P{lab[1:]})=TOL/POS,2D,{_f(t)}{_material(s['requirement'])}")
                for target in targets:
                    L.append(f"OUTPUT/FA({target}),TA(P{lab[1:]})")
                n_feat += 1
            elif why in ("flatness", "orientation", "profile", "form", "position"):
                L.append("MODE/MAN")
                kind = {"flatness": "FLAT", "orientation": "PERP" if "⊥" in s["requirement"] else "ANGLB",
                        "profile": "PROFS", "form": "CIRLTY", "position": "POS"}[why]
                L.append(f"F({lab})=FEAT/PLANE,CART,0,0,0,0,0,1" if kind != "POS" else
                         f"F({lab})=FEAT/CIRCLE,INNER,CART,0,0,0,0,0,1,0")
                L.append(f"MEAS/{'PLANE' if kind != 'POS' else 'CIRCLE'},F({lab}),{6 if kind != 'POS' else 4}")
                L.append("ENDMES")
                tol = {"FLAT": f"TOL/FLAT,{_f(t)}", "PERP": f"TOL/PERP,{_f(t)},DAT({datums[0] if datums else 'A'})",
                       "ANGLB": f"TOL/ANGLB,{_f(t)},DAT({datums[0] if datums else 'A'})",
                       "PROFS": f"TOL/PROFS,{_f(-t / 2)},{_f(t / 2)}", "CIRLTY": f"TOL/CIRLTY,{_f(t)}",
                       "POS": f"TOL/POS,2D,{_f(t)}{_material(s['requirement'])}"}[kind]
                L.append(f"T(G{lab[1:]})={tol}")
                L.append(f"OUTPUT/FA({lab}),TA(G{lab[1:]})")
                n_manual += 1
                n_feat += 1
            else:
                L.append(f"$$ Not programmed: {why}. Measure per the plan.")
        else:
            L.append(f"$$ Balloon {s['row_id']}: {ascii_text(s['requirement'])} ({s['instrument']}): measure per the plan.")
    L.append("ENDFIL")
    return {"program": "\n".join(L) + "\n", "features": n_feat, "from_model": n_model, "manual": n_manual,
            "units": units, "datums": datums}


def check_program(text: str) -> list[str]:
    """Structural checks: known major words, labels defined before use, MEAS/ENDMES balanced, ENDFIL last."""
    problems = []
    feats, tols, dats = set(), set(), set()
    open_meas = False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for n, ln in enumerate(lines, 1):
        if ln.startswith("$$"):
            continue
        m = re.match(r"^(?:([FTD])\(([A-Z0-9_]+)\)=)?([A-Z]+)/?(.*)$", ln)
        if not m:
            problems.append(f"line {n}: not a DMIS statement: {ln}")
            continue
        kind, label, major, rest = m.groups()
        if major not in MAJOR:
            problems.append(f"line {n}: unknown statement {major}")
        for ref in re.findall(r"\bFA?\(([A-Z0-9_]+)\)", rest):
            if ref not in feats:
                problems.append(f"line {n}: feature {ref} used before it is defined")
        for ref in re.findall(r"\bTA\(([A-Z0-9_]+)\)", rest):
            if ref not in tols:
                problems.append(f"line {n}: tolerance {ref} used before it is defined")
        for ref in re.findall(r"\bDAT\(([A-Z])\)", rest):
            if ref not in dats and major != "DATDEF":
                problems.append(f"line {n}: datum {ref} used before it is defined")
        if major == "DATDEF":
            dats.update(re.findall(r"DAT\(([A-Z])\)", rest))
        if kind == "F":
            feats.add(label)
        if kind == "T":
            tols.add(label)
        if major == "MEAS":
            if open_meas:
                problems.append(f"line {n}: MEAS inside another MEAS")
            open_meas = True
        if major == "ENDMES":
            if not open_meas:
                problems.append(f"line {n}: ENDMES without MEAS")
            open_meas = False
    if open_meas:
        problems.append("MEAS without ENDMES")
    if not lines or lines[-1].strip() != "ENDFIL":
        problems.append("program does not end with ENDFIL")
    return problems

