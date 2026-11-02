"""Supplier-style AS9102 Form 3 reports for testing the report check (whiteout.fair_check).

From each finished sample job this writes a correct supplier report in two layouts that differ from ours (other
column order and header wording, merged group headers, nominal and tolerance columns), one that uses the supplier's
own balloon numbers, and seeded copies with planted errors of every kind. The planted errors are listed in
answers.json. A few reports are also drawn as table images and saved as PDFs, like a scanned report.

uv run python eval/make_supplier_reports.py --out eval/supplier_reports
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side
from PIL import Image, ImageDraw, ImageFont

from whiteout.tolerance import parse_limits

ROOT = Path(__file__).resolve().parents[1]
JOBS = {"nist": "bd20fca1920f", "apollo": "e9f770195cba"}
FONT = "/Library/Fonts/Arial Unicode.ttf"
KINDS = ["missing_row", "requirement_mismatch", "accepted_out_of_tolerance", "missing_result", "duplicate_number",
         "extra_row"]


def f2(s: str) -> float | None:
    try:
        return float(s) if s not in ("", None) else None
    except ValueError:
        return None


def places(text: str) -> int:
    ds = [len(m) for m in re.findall(r"\.(\d+)", text or "")]
    return max(ds) if ds else 3


def num(v: float, p: int, lead_zero: bool) -> str:
    s = f"{v:.{p}f}"
    if not lead_zero and s.startswith("0."):
        s = s[1:]
    return s


# ---- supplier rows from the drawing ----------------------------------------------------------------------------
def retype(text: str, rtype: str, style: str) -> str:
    """The requirement as a supplier would type it: style A drops counts and note numbers and adds leading zeros."""
    t = " ".join((text or "").split())
    if style == "B":
        return t
    t = re.sub(r"^\s*\d+\s*X\s+", "", t)
    t = re.sub(r"\s*\(\s*\d+\s*\)", "", t)
    t = t.replace(" | ", " ").replace("|", " ").replace("Ⓜ", "(M)").replace("Ⓛ", "(L)")
    if rtype == "note":
        rest = re.sub(r"^\s*\d{1,2}[.)]?\s+(?=[A-Z])", "", t)
        t = rest if len(rest) > 20 else t
    t = re.sub(r"(?<![\d.])\.(\d)", r"0.\1", t)
    return " ".join(t.split())


def measured(row: dict, rng: random.Random, style: str) -> str:
    lo, hi = f2(row["lower"]), f2(row["upper"])
    rtype, text = row["type"], row["requirement"]
    lead = style == "A"
    if rtype == "reference":
        return "REF"
    if row["attribute"] or rtype in ("note", "thread", "surface_finish", "other"):
        return "Conforms" if style == "A" else "C"
    p = min(places(text) + 1, 4)
    if lo is not None and hi is not None:
        if lo < 0:  # a table of general tolerances, not one feature
            return "Conforms" if style == "A" else "C"
        if rtype == "gdt":
            return num(hi * rng.uniform(0.15, 0.7), 4, lead)
        band = hi - lo
        v = min(max(rng.gauss((lo + hi) / 2, band / 8), lo + 0.1 * band), hi - 0.1 * band)
        return num(v, p, lead) + ("°" if rtype == "angle" else "")
    decs = re.findall(r"\d*\.\d+", text)  # "-33 1.1700": the dash number is not the size
    nom = float(decs[-1]) if decs else parse_limits(text)[0]
    if nom is None:
        return "Conforms" if style == "A" else "C"
    if rtype == "angle" or "°" in text:
        return f"{nom + rng.choice([-0.1, 0.0, 0.1]):.1f}°"
    return num(nom + rng.uniform(-0.0008, 0.0008), max(p, 4), lead)


def supplier_rows(job: dict, style: str, rng: random.Random, own_numbers: bool = False) -> list[dict]:
    rows = [r for r in job["rows"]]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["number"], []).append(r)
    order = list(groups)
    if own_numbers:  # the supplier ballooned the drawing in its own order
        rng.shuffle(order)
    out = []
    for gi, number in enumerate(order, start=1):
        grp = groups[number]
        base = str(gi) if own_numbers else number
        for k, r in enumerate(grp):
            if len(grp) == 1:
                cno = base
            elif style == "A" and all("-" in x["row_id"] and "." not in x["row_id"] for x in grp):
                cno = base + "ABCDEFGHIJKLMNOP"[k]
            else:
                cno = f"{base}.{k + 1}"
            lo, hi = f2(r["lower"]), f2(r["upper"])
            nom = None
            if lo is not None and hi is not None:
                n, _, _, _ = parse_limits(r["requirement"])
                nom = n if (n is not None and lo - 1e-9 <= n <= hi + 1e-9 and r["type"] != "gdt") else None
                if nom is None and r["type"] != "gdt":
                    nom = (lo + hi) / 2
            out.append({
                "row_id": r["row_id"], "number": r["number"], "drawing_text": r["requirement"], "char_no": cno,
                "type": r["type"],
                "zone": f"{r['sheet']}/{r['zone']}", "text": retype(r["requirement"], r["type"], style),
                "lower": lo, "upper": hi, "nominal": nom, "places": places(r["requirement"]),
                "result": measured(r, rng, style), "accept": "", "nc": "", "attribute": r["attribute"],
            })
    for s in out:
        s["accept"] = "ACC" if style == "A" else ""
    if own_numbers:
        out.sort(key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s["char_no"])])
    return out


def numeric_rows(rows: list[dict]) -> list[dict]:
    return [s for s in rows if s["lower"] is not None and s["upper"] is not None and s["lower"] >= 0
            and not s["attribute"] and re.search(r"\d", s["result"])]


def outside(s: dict, rng: random.Random, style: str) -> str:
    lo, hi = s["lower"], s["upper"]
    band = max(hi - lo, 0.001)
    above = s["type"] == "gdt" or rng.random() < 0.5
    v = hi + band * rng.uniform(0.2, 0.6) if above else lo - band * rng.uniform(0.2, 0.6)
    p = 4 if s["type"] == "gdt" else min(s["places"] + 1, 4)
    return num(max(v, 0.0), p, style == "A") + ("°" if s["type"] == "angle" else "")


# ---- planting errors ----------------------------------------------------------------------------------------------
def _bump_digit(tok: str) -> str:
    d = int(tok[-1])
    return tok[:-1] + str((d + 5) % 10)


def change_requirement(s: dict, rng: random.Random) -> str | None:
    """Copy error in the requirement text or limits. Returns a description, or None if this row cannot take one."""
    t = s["text"]
    if s["type"] == "note" or (s["attribute"] and len(t) > 25):
        toks = [m for m in re.finditer(r"\d+", t)]
        if toks:
            m = rng.choice(toks)
            new = t[:m.start()] + _bump_digit(m.group(0)) + t[m.end():]
        else:
            words = [m for m in re.finditer(r"\b[A-Z]{4,}\b", t)]
            if len(words) < 3:
                return None
            m = words[min(2, len(words) - 1)]
            new = (t[:m.start()] + t[m.end():]).replace("  ", " ")
        s["text"] = new
        return f"note text changed: {t!r} -> {new!r}"
    toks = list(re.finditer(r"\d*\.\d+", t))
    if not toks:
        return None
    if s["type"] == "gdt":
        m = toks[0]
        new_tok = _bump_digit(m.group(0))
        s["text"] = t[:m.start()] + new_tok + t[m.end():]
        s["upper"] = float(new_tok)
        return f"tolerance {m.group(0)} copied as {new_tok}"
    if s["lower"] is not None and s["upper"] is not None:
        m = toks[-1]  # the upper limit or the ± value
        new_tok = _bump_digit(m.group(0))
        s["text"] = t[:m.start()] + new_tok + t[m.end():]
        _, lo, hi, how = parse_limits(s["text"])
        if how in ("plus_minus", "unilateral", "limits"):
            s["lower"], s["upper"] = lo, hi
            if s["nominal"] is not None and not (lo <= s["nominal"] <= hi):
                s["nominal"] = (lo + hi) / 2
        else:
            s["upper"] = float(new_tok) if abs(float(new_tok) - s["upper"]) < 1 else s["upper"] + 0.005
        return f"{m.group(0)} copied as {new_tok}"
    m = toks[-1]  # basic or untoleranced: transpose or bump the last digits
    g = m.group(0)
    new_tok = g[:-2] + g[-1] + g[-2] if len(g) > 2 and g[-1] != g[-2] and g[-2] != "." else _bump_digit(g)
    s["text"] = t[:m.start()] + new_tok + t[m.end():]
    return f"{g} copied as {new_tok}"


def record(kind: str, s: dict, detail: str, char_no: str | None = None) -> dict:
    return {"kind": kind, "row_id": s["row_id"], "number": s["number"], "drawing_text": s["drawing_text"],
            "char_no": char_no or s["char_no"], "detail": detail}


def plant(rows: list[dict], rng: random.Random, style: str, per_kind: int = 2) -> tuple[list[dict], list[dict]]:
    rows = [dict(r) for r in rows]
    errors = []
    used_numbers: set[str] = set()

    def pick(pool, n):
        pool = [s for s in pool if s["number"] not in used_numbers]
        rng.shuffle(pool)
        got = []
        for s in pool:
            if s["number"] in used_numbers:
                continue
            got.append(s)
            used_numbers.add(s["number"])
            if len(got) == n:
                break
        return got

    for s in pick(numeric_rows(rows), per_kind):
        s["result"] = outside(s, rng, style)
        s["accept"] = "ACC" if style == "A" else ""
        errors.append(record("accepted_out_of_tolerance", s,
                             f"result {s['result']} outside {s['lower']}..{s['upper']}, accepted"))
    limit_pool = [s for s in rows if s["lower"] is not None and s["upper"] is not None and s["type"] != "gdt"
                  and s["lower"] >= 0]
    other_pool = [s for s in rows if not (s["lower"] is not None and s["type"] != "gdt")]
    for pool in ([limit_pool, other_pool] * per_kind)[:per_kind]:
        for s in pick(pool, 5):
            before = dict(s)
            what = change_requirement(s, rng)
            if what:
                errors.append(record("requirement_mismatch", s, what))
                break
            s.update(before)
            used_numbers.discard(s["number"])
    for s in pick([s for s in rows if s["result"] not in ("REF",)], per_kind):
        s["result"] = ""
        errors.append(record("missing_result", s, "result left blank"))
    removed = pick(rows, per_kind)
    for s in removed:
        rows.remove(s)
        errors.append(record("missing_row", s, f"row for {s['text'][:40]!r} left out"))
    dups = pick(rows, per_kind)
    for k, s in enumerate(dups):
        if k % 2 == 0:  # the same row typed twice
            copy = dict(s)
            copy["result"] = measured({"lower": fmtv(s["lower"]), "upper": fmtv(s["upper"]), "type": s["type"],
                                       "requirement": s["text"], "attribute": s["attribute"]}, rng, style)
            rows.insert(rows.index(s) + 1, copy)
            errors.append(record("duplicate_number", s, "row typed twice"))
        else:  # the next characteristic typed under the same number
            i = rows.index(s)
            nxt = next((r for r in rows[i + 1:] if r["number"] != s["number"] and r["number"] not in used_numbers
                        and len([x for x in rows if x["number"] == r["number"]]) == 1), None)
            if nxt is None:
                continue
            used_numbers.add(nxt["number"])
            errors.append(record("duplicate_number", nxt, f"characteristic {nxt['char_no']} numbered {s['char_no']}",
                                 char_no=s["char_no"]))
            nxt["char_no"] = s["char_no"]
    top = max(int(re.match(r"\d+", r["char_no"]).group(0)) for r in rows + removed)
    extras = [("Ø0.3125 ±0.005" if style == "A" else "Ø.3125 ±.005", 0.3075, 0.3175, 0.3125),
              ("PASSIVATE PER AMS 2700, METHOD 1, TYPE 2", None, None, None),
              ("1.500 ±.010", 1.49, 1.51, 1.5)][:per_kind]
    for k, (text, lo, hi, nom) in enumerate(extras, start=1):
        r = {"row_id": "", "number": "", "drawing_text": "", "char_no": str(top + k),
             "type": "note" if lo is None else "size", "zone": "", "text": text, "lower": lo, "upper": hi,
             "nominal": nom, "places": 3,
             "result": ("Conforms" if style == "A" else "C") if lo is None else num(nom + 0.0012, 4, style == "A"),
             "accept": "ACC" if style == "A" else "", "nc": "", "attribute": lo is None}
        rows.append(r)
        errors.append(record("extra_row", r, f"{text!r} not on drawing"))
    return rows, errors


def fmtv(v: float | None) -> str:
    return "" if v is None else f"{v:.6f}".rstrip("0").rstrip(".")


def add_rejected(rows: list[dict], rng: random.Random, style: str) -> None:
    """A result outside the limits that the supplier correctly rejected: not an error."""
    s = rng.choice(numeric_rows(rows))
    s["result"] = outside(s, rng, style)
    s["accept"] = "REJ" if style == "A" else ""
    s["nc"] = "NCR-2026-0147"


# ---- layouts ----------------------------------------------------------------------------------------------------
THIN = Side(style="thin", color="888888")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def cells_A(s: dict) -> list:
    lead = True
    lo = "" if s["lower"] is None else num(s["lower"], 4 if s["type"] == "gdt" else s["places"], lead)
    hi = "" if s["upper"] is None else num(s["upper"], 4 if s["type"] == "gdt" else s["places"], lead)
    return [s["char_no"], s["zone"], s["text"], "", lo, hi, s["result"], "CMM" if lo else "Visual", s["accept"],
            s["nc"], ""]


HEAD_A = ["Bubble No.", "Sheet / Zone", "Characteristic Description", "Key Char.", "Min", "Max",
          "Actual Measurement", "Gauge Used", "Acc / Rej", "NCR #", "Remarks"]
GROUPS_A = [("Characteristic Accountability", 2, 4), ("Inspection / Test Results", 5, 8), ("Evaluation", 9, 11)]


def cells_B(s: dict) -> list:
    nom = plus = minus = ""
    if s["nominal"] is not None and s["lower"] is not None and s["upper"] is not None:
        p = s["places"]
        while p < 5 and any(abs(round(v, p) - v) > 1e-9 for v in (s["nominal"], s["upper"], s["lower"])):
            p += 1
        nom = num(s["nominal"], p, False)
        plus = num(s["upper"] - s["nominal"], p, False)
        minus = num(s["nominal"] - s["lower"], p, False)
    return [s["text"], nom, plus, minus, s["result"], "CMM" if nom else "Visual", s["char_no"], s["nc"], ""]


HEAD_B = ["Drawing Requirement", "Nominal", "+", "-", "Measured Value", "Inspection Method", "Item",
          "Nonconformance Report No.", "Comments"]


def write_xlsx(rows: list[dict], style: str, path: Path, part: str) -> None:
    wb = Workbook()
    info = wb.active
    info.title = "Part Info"
    info.append(["Part Number", part])
    info.append(["Report", "FAI-2026-0311"])
    ws = wb.create_sheet("FAIR Characteristics")
    bold = Font(bold=True)
    wrap = Alignment(wrap_text=True, vertical="top")
    if style == "A":
        ws.append(["FIRST ARTICLE INSPECTION REPORT (AS9102)  CHARACTERISTIC ACCOUNTABILITY"])
        ws.merge_cells("A1:K1")
        ws.append(["Part Number:", part, "", "FAI Report No.:", "FAI-2026-0311"])
        ws.append(["Supplier:", "Example Machining LLC", "", "P.O. Number:", "4500018821"])
        ws.append([])
        ws.append(["Bubble No."] + [""] * 10)
        for name, a, b in GROUPS_A:
            ws.cell(5, a, name)
            ws.merge_cells(start_row=5, start_column=a, end_row=5, end_column=b)
        ws.append([""] + HEAD_A[1:])
        ws.merge_cells("A5:A6")
        hdr = (5, 6)
        for s in rows:
            ws.append(cells_A(s))
        widths = [10, 10, 48, 8, 10, 10, 14, 10, 9, 14, 16]
    else:
        ws.append(["FAI FORM 3  DESIGN CHARACTERISTICS"])
        ws.append(["P/N", part, "Rev", "", "Inspector", "QC-07"])
        ws.append([])
        ws.append([])
        ws.append(HEAD_B)
        ws.append(["", "", "+", "-"] + [""] * 5)
        ws.cell(5, 3, "Tolerance")
        ws.merge_cells("C5:D5")
        for col in (1, 2, 5, 6, 7, 8, 9):
            ws.merge_cells(start_row=5, start_column=col, end_row=6, end_column=col)
        hdr = (5, 6)
        for s in rows:
            ws.append(cells_B(s))
        widths = [52, 9, 8, 8, 14, 12, 7, 18, 14]
    for r in hdr:
        for c in ws[r]:
            c.font, c.alignment = bold, wrap
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[chr(64 + i)].width = w
    for row in ws.iter_rows(min_row=hdr[1] + 1, max_row=ws.max_row):
        for c in row:
            c.border, c.alignment = BOX, wrap
    ws.append([])
    ws.append(["Inspected by: J. Rivera", "", "", "", "", "", "Date: 2026-10-26"])
    wb.save(path)


# ---- table image and PDF ----------------------------------------------------------------------------------------
def render_pdf(rows: list[dict], style: str, path: Path, part: str) -> int:
    W, H, M = 2000, 1500, 40
    font = ImageFont.truetype(FONT, 17)
    bold = ImageFont.truetype(FONT, 18)
    title = ImageFont.truetype(FONT, 24)
    if style == "A":
        head, widths, cells = HEAD_A, [90, 110, 640, 70, 100, 100, 160, 110, 90, 150, 300], cells_A
    else:
        head, widths, cells = (["Drawing Requirement", "Nominal", "Tol +", "Tol -", "Measured Value",
                                "Inspection Method", "Item", "Nonconformance Report No.", "Comments"],
                               [700, 110, 100, 100, 170, 150, 80, 250, 260], cells_B)
    scale = (W - 2 * M) / sum(widths)
    widths = [int(w * scale) for w in widths]
    xs = [M]
    for w in widths:
        xs.append(xs[-1] + w)

    def wrap(text: str, width: int, f) -> list[str]:
        lines, cur = [], ""
        for word in str(text).split():
            t = (cur + " " + word).strip()
            if f.getlength(t) <= width - 10 or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = word
        return lines + ([cur] if cur else []) or [""]

    pages = []
    i = 0
    while i < len(rows) or not pages:
        im = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(im)
        d.text((M, 20), "FIRST ARTICLE INSPECTION REPORT  FORM 3   Part " + part + f"   page {len(pages) + 1}",
               font=title, fill=0)
        y = 70
        hl = [wrap(h, w, bold) for h, w in zip(head, widths)]
        hh = 26 * max(len(x) for x in hl) + 10
        for x0, x1, lines in zip(xs, xs[1:], hl):
            d.rectangle([x0, y, x1, y + hh], outline=0, width=2)
            for k, ln in enumerate(lines):
                d.text((x0 + 5, y + 5 + 26 * k), ln, font=bold, fill=0)
        y += hh
        while i < len(rows):
            vals = cells(rows[i])
            wl = [wrap(v, w, font) for v, w in zip(vals, widths)]
            rh = 22 * max(len(x) for x in wl) + 10
            if y + rh > H - M:
                break
            for x0, x1, lines in zip(xs, xs[1:], wl):
                d.rectangle([x0, y, x1, y + rh], outline=90, width=1)
                for k, ln in enumerate(lines):
                    d.text((x0 + 5, y + 5 + 22 * k), ln, font=font, fill=0)
            y += rh
            i += 1
        pages.append(im.convert("RGB"))
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=150)
    return len(pages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "supplier_reports")
    ap.add_argument("--seeded", type=int, default=3, help="seeded reports per drawing")
    ap.add_argument("--seed", default="", help="change every random draw (for repeat runs)")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    answers = []
    pdfs = {("nist", "clean_A"), ("nist", "seeded1_A"), ("apollo", "clean_B"), ("apollo", "seeded2_B")}
    for name, jid in JOBS.items():
        job = json.loads((ROOT / "samples" / "jobs" / jid / "job.json").read_text())
        part = job["sheets"][0]["info"].get("drawing_number") or job["sheets"][0]["info"].get("part_number") or name
        variants = []
        for style in ("A", "B"):
            rng = random.Random(f"{a.seed}{name}-clean-{style}")
            rows = supplier_rows(job, style, rng)
            add_rejected(rows, rng, style)
            variants.append((f"clean_{style}", style, rows, []))
        rng = random.Random(f"{a.seed}{name}-own")
        variants.append(("clean_own_numbers_A", "A", supplier_rows(job, "A", rng, own_numbers=True), []))
        for k in range(1, a.seeded + 1):
            style = "A" if k % 2 else "B"
            rng = random.Random(f"{a.seed}{name}-seeded-{k}")
            rows, errors = plant(supplier_rows(job, style, rng), rng, style)
            variants.append((f"seeded{k}_{style}", style, rows, errors))
        rng = random.Random(f"{a.seed}{name}-seeded-own")
        rows, errors = plant(supplier_rows(job, "A", rng, own_numbers=True), rng, "A")
        variants.append((f"seeded{a.seeded + 1}_own_numbers_A", "A", rows, errors))
        for vname, style, rows, errors in variants:
            stem = f"{name}_{vname}"
            write_xlsx(rows, style, a.out / f"{stem}.xlsx", part)
            files = [f"{stem}.xlsx"]
            if (name, vname) in pdfs and not a.no_pdf:
                n = render_pdf(rows, style, a.out / f"{stem}.pdf", part)
                files.append(f"{stem}.pdf")
                print(f"{stem}.pdf: {n} pages")
            for f in files:
                answers.append({"file": f, "job": jid, "drawing": name, "layout": style,
                                "numbering": "own" if "own" in vname else "drawing", "clean": not errors,
                                "rows": len(rows), "errors": errors})
            print(f"{stem}: {len(rows)} rows, {len(errors)} planted errors")
    (a.out / "answers.json").write_text(json.dumps(answers, indent=1))


if __name__ == "__main__":
    main()
