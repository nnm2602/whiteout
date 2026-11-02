"""Check a supplier's AS9102 Form 3 against the drawing Whiteout read.

A buyer receives a first article report from a supplier. This module reads the supplier's Form 3 (a spreadsheet in
any reasonable layout, or a PDF or image of the table read by Muse Spark), matches each row to a requirement on the
drawing, and lists what is wrong: requirements with no row, rows that copy the requirement or its limits wrongly,
results outside the drawing limits that are marked accepted, missing results, repeated characteristic numbers, and
rows that match nothing on the drawing. Limits and pass or fail come from the drawing, in code.

uv run python -m whiteout.fair_check <job_dir> <report.xlsx|pdf|png|jpg> [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

from .forms import fmt
from .tolerance import evaluate, parse_limits, to_float
from .types import Characteristic, Requirement

KINDS = ["missing_row", "requirement_mismatch", "accepted_out_of_tolerance", "missing_result", "duplicate_number",
         "extra_row", "number_mismatch"]
KIND_WORDS = {
    "missing_row": ("requirement with no row", "requirements with no row"),
    "requirement_mismatch": ("row that differs from the drawing", "rows that differ from the drawing"),
    "accepted_out_of_tolerance": ("result outside the limits marked accepted",
                                  "results outside the limits marked accepted"),
    "missing_result": ("missing result", "missing results"),
    "duplicate_number": ("repeated characteristic number", "repeated characteristic numbers"),
    "extra_row": ("row that matches nothing on the drawing", "rows that match nothing on the drawing"),
    "number_mismatch": ("row under a different number", "rows under a different number"),
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
EPS = 1e-6


# ---- drawing side -----------------------------------------------------------------------------------------------
@dataclass
class DrawingLine:
    idx: int
    row_id: str
    base: int | None
    text: str
    req: Requirement
    char: Characteristic
    pos: int = 0  # position among lines with the same balloon number


def _num_key(c: Characteristic):
    try:
        return (c.sheet, int(c.number))
    except ValueError:
        return (c.sheet, 10**6)


def drawing_lines(job_json: dict) -> list[DrawingLine]:
    """One line per requirement per instance, numbered the way forms.form3_rows numbers them."""
    chars = [Characteristic(**c) for s in job_json.get("sheets", []) for c in s.get("chars", [])]
    out: list[DrawingLine] = []
    for c in sorted((c for c in chars if not c.ignored and c.number), key=_num_key):
        reqs = c.requirements or []
        pos = 0
        for qi, q in enumerate(reqs):
            for inst in range(1, c.multiplier + 1):
                rid = c.number if len(reqs) == 1 else f"{c.number}.{qi + 1}"
                if c.multiplier > 1:
                    rid += f"-{inst}"
                out.append(DrawingLine(len(out), rid, _base(c.number), q.text if len(reqs) > 1 else c.text, q, c, pos))
                pos += 1
    return out


# ---- supplier side ----------------------------------------------------------------------------------------------
def _half_unit(v) -> float:
    """Half a unit in the last printed decimal place: a limit rounded for printing still agrees."""
    m = re.search(r"\.(\d+)", str(v or ""))
    return 0.5 * 10 ** -len(m.group(1)) + EPS if m and len(m.group(1)) < 6 else EPS


@dataclass
class SupplierRow:
    idx: int
    where: str  # "row 14" in a spreadsheet, "page 2 line 5" in a PDF
    char_no: str = ""
    text: str = ""
    lower: float | None = None
    upper: float | None = None
    result: str = ""
    accept: str = ""
    nc: str = ""
    base: int | None = None
    pos: int = 0
    slack: float = EPS  # rounding allowed when comparing limits
    raw: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"index": self.idx, "where": self.where, "char_no": self.char_no, "requirement": self.text,
                "lower": fmt(self.lower), "upper": fmt(self.upper), "result": self.result, "accept": self.accept,
                "nonconformance": self.nc}


def _suffix_pos(char_no: str, default: int) -> int:
    """Place within a balloon from the number's suffix: 40.2, 40-2 and 40B are the second place."""
    m = re.match(r"^\W*\d+\s*(?:[.\-/]\s*(\d+)|([A-Za-z]))\s*$", char_no or "")
    if not m:
        return default
    return int(m.group(1)) - 1 if m.group(1) else ord(m.group(2).upper()) - 65


def _key(char_no: str) -> str:
    return re.sub(r"[\s.]+$", "", (char_no or "").upper().replace(" ", ""))


def _base(num: str) -> int | None:
    m = re.search(r"\d+", num or "")
    return int(m.group(0)) if m else None


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return str(v).strip()


# Header roles, most specific first. Each column takes the first role whose pattern matches its header text.
ROLES = [
    ("skip", r"designator|\bkey\b|critical|\bclass\b|tool|gage|gauge|equipment|instrument|method|remark|comment"
             r"|additional data|\bnotes?\b|found by|^\s*check\s*$|tolerance from|^\s*type\s*$"),
    ("nc", r"non.?conform|\bncr?\b|\bnc\s*(no|#|num)|deviation|\bmrb\b"),
    ("accept", r"accept|reject|\bacc\b|\brej\b|disposition|conform|pass\s*/\s*fail|\bstatus\b|\bok\b|\by\s*/\s*n\b"),
    ("lower", r"lower|\bmin(imum)?\b|\blsl\b|low\s*limit"),
    ("upper", r"upper|\bmax(imum)?\b|\busl\b|high\s*limit"),
    ("plus", r"\+\s*tol|tol\w*\s*\(?\s*\+|^\s*\+\s*$|\bplus\b"),
    ("minus", r"-\s*tol|tol\w*\s*\(?\s*-|^\s*-\s*$|\bminus\b"),
    ("result", r"result|measured|actual|reading|value\s*found|inspection\s*value"),
    ("number", r"char\w*\.?\s*(no|num|#)|balloon|bubble|\bitem\b|^\s*(no\.?|#)\s*$|feature\s*(no|#|id)|^\s*5\.\s"),
    ("nominal", r"\bnominal\s*$|^\s*nom\.?\s*$|^\s*nominal\s*$"),
    ("requirement", r"requirement|designed\s*char|specification|\bspec\b|description|dimension|characteristic"
                    r"|nominal\s*(&|and)\s*tol|^\s*8\.\s"),
    ("zone", r"zone|location|\bloc\b"),
    ("sheet", r"\bsheet\b|\bsht\b|\bpage\b"),
]
CORE = {"number", "requirement", "result", "accept", "nc", "lower", "upper", "nominal"}


def _role(text: str) -> str | None:
    t = (text or "").lower().replace("\n", " ")
    if not t.strip():
        return None
    for role, pat in ROLES:
        if re.search(pat, t):
            return role
    return None


def _grid(ws) -> dict[tuple[int, int], str]:
    """Cell text by (row, col), with merged ranges filled from their top-left cell."""
    g = {}
    for row in ws.iter_rows():
        for c in row:
            if c.value is not None:
                g[(c.row, c.column)] = _cell(c.value)
    for rng in ws.merged_cells.ranges:
        v = g.get((rng.min_row, rng.min_col), "")
        for r in range(rng.min_row, rng.max_row + 1):
            for col in range(rng.min_col, rng.max_col + 1):
                g[(r, col)] = v
    return g


def _columns(g: dict, r: int, ncol: int) -> dict[str, int]:
    """Map roles to columns for header row r; a column with no role of its own uses the group label above it."""
    cols: dict[str, int] = {}
    for c in range(1, ncol + 1):
        own = g.get((r, c), "")
        role = _role(own)
        if role is None and not own.strip():
            role = _role(g.get((r - 1, c), ""))
        if role and role != "skip" and role not in cols:
            cols[role] = c
    return cols


def find_layout(wb) -> tuple[object, int, dict[str, int]]:
    """Pick the sheet and header row whose header names the most Form 3 columns."""
    best = (-1, None, 0, {})
    for ws in wb.worksheets:
        g = _grid(ws)
        ncol = ws.max_column
        for r in range(1, min(ws.max_row, 40) + 1):
            if sum(1 for c in range(1, ncol + 1) if _num(g.get((r, c), "")) is not None) > 1:
                continue  # a data row, not a header
            cols = _columns(g, r, ncol)
            if "number" not in cols and "requirement" not in cols:
                continue
            if "result" not in cols:
                continue
            # A lower header row under group labels names more columns; on a tie the lower row wins.
            named = len(cols) + len(CORE & set(cols)) + (0.5 if re.search(r"3|char", ws.title, re.I) else 0)
            if named >= best[0] and (best[1] is None or best[1] is ws or named > best[0]):
                best = (named, ws, r, cols)
    if best[1] is None:
        raise ValueError("no Form 3 table found: need a header row with a characteristic number or requirement "
                         "column and a results column")
    return best[1], best[2], best[3]


NUMBER_LIKE = re.compile(r"^\W*[A-Za-z]{0,4}\s*[-#]?\s*\d+[\w.\-/ ]{0,8}$")


def read_xlsx(path: Path) -> tuple[list[SupplierRow], dict]:
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True)
    ws, hr, cols = find_layout(wb)
    g = _grid(ws)
    header_texts = {g.get((hr, c), "") for c in range(1, ws.max_column + 1)}
    rows: list[SupplierRow] = []
    for r in range(hr + 1, ws.max_row + 1):
        val = {k: g.get((r, c), "") for k, c in cols.items()}
        num, text = val.get("number", ""), val.get("requirement", "")
        if not any(val.values()):
            continue
        if num and num in header_texts and text in header_texts:
            continue  # header repeated after a page break
        if num and not NUMBER_LIKE.match(num):
            continue  # signature lines, footers
        if not num and not text:
            continue
        rows.append(_make_row(len(rows), f"row {r}", val))
    layout = {"source": "xlsx", "sheet": ws.title, "header_row": hr,
              "columns": {k: g.get((hr, c), "") or g.get((hr - 1, c), "") for k, c in cols.items()}}
    return rows, layout


def _make_row(idx: int, where: str, val: dict) -> SupplierRow:
    s = SupplierRow(idx, where, char_no=val.get("number", "").strip(),
                    text=" ".join(val.get("requirement", "").split()),
                    result=val.get("result", "").strip(), accept=val.get("accept", "").strip(),
                    nc=val.get("nc", "").strip(), raw=val)
    lo, hi = _num(val.get("lower")), _num(val.get("upper"))
    s.slack = max(_half_unit(val.get("lower")), _half_unit(val.get("upper")))
    if lo is None and hi is None:
        nom, plus, minus = _num(val.get("nominal")), _num(val.get("plus")), _num(val.get("minus"))
        if nom is not None and (plus is not None or minus is not None):
            lo, hi = nom - abs(minus or 0.0), nom + abs(plus or 0.0)
            s.slack = _half_unit(val.get("nominal")) + max(_half_unit(val.get("plus")), _half_unit(val.get("minus")))
    if lo is None and hi is None:
        _, plo, phi, how = parse_limits(s.text)
        if how in ("plus_minus", "unilateral", "limits"):
            lo, hi, s.slack = plo, phi, EPS
    s.lower, s.upper = lo, hi
    s.base = _base(s.char_no)
    return s


def _num(v) -> float | None:
    t = (v or "").strip().replace("°", "").replace("±", "").replace("−", "-")
    t = re.sub(r"\s*(in|mm|deg)\.?$", "", t, flags=re.I)
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return to_float(t)


# ---- reading a PDF or image with Muse Spark ---------------------------------------------------------------------
def _o(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


_S = {"type": "string"}
TABLE_SCHEMA = _o({
    "columns": {"type": "array", "items": _S},
    "rows": {"type": "array", "items": _o({
        "char_no": _S, "requirement": _S, "nominal": _S, "plus_tol": _S, "minus_tol": _S, "lower": _S, "upper": _S,
        "result": _S, "accept": _S, "nonconformance": _S,
    })},
})

TABLE_PROMPT = (
    "This image is one page of a supplier's AS9102 first article inspection report, Form 3 (characteristic "
    "accountability). Read the table of characteristics row by row, top to bottom, and return every data row. "
    "For each row give: char_no (the characteristic or balloon number cell as printed), requirement (the designed "
    "characteristic or requirement cell, the whole text with its symbols and tolerances), nominal, plus_tol and "
    "minus_tol (only when the table has separate nominal and tolerance columns), lower and upper (only when the table "
    "has separate lower or minimum and upper or maximum limit columns), result (the measured result cell), accept "
    "(the accept or reject, conformance or status cell) and nonconformance (the nonconformance or NCR number cell). "
    "Copy every character exactly as printed, even when a value looks wrong or inconsistent: never correct, round, "
    "complete or move a value. Use an empty string for an empty cell or a column the table does not have. Skip "
    "title rows, header rows, part information and signature lines. Also list the column headers as printed."
)


def page_images(path: Path) -> list:
    """Pages of a PDF or one image, at most about 3 megapixels each so the model reads them at full size."""
    from PIL import Image

    imgs = []
    if path.suffix.lower() == ".pdf":
        import pymupdf

        doc = pymupdf.open(path)
        for page in doc:
            w, h = page.rect.width, page.rect.height
            zoom = min(150 / 72, (3.0e6 / (w * h)) ** 0.5)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY)
            imgs.append(Image.frombytes("L", (pix.width, pix.height), pix.samples).convert("RGB"))
    else:
        im = Image.open(path).convert("RGB")
        scale = min(1.0, (3.0e6 / (im.width * im.height)) ** 0.5)
        if scale < 1:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        imgs.append(im)
    return imgs


async def read_table_image(path: Path, client) -> tuple[list[SupplierRow], dict]:
    imgs = page_images(path)
    pages = await asyncio.gather(*(client.ask_json([im], TABLE_PROMPT, TABLE_SCHEMA, name="form3_table",
                                                   effort="low") for im in imgs))
    rows: list[SupplierRow] = []
    for p, res in enumerate(pages, start=1):
        for i, r in enumerate(res.get("rows", []), start=1):
            val = {"number": r.get("char_no", ""), "requirement": r.get("requirement", ""),
                   "nominal": r.get("nominal", ""), "plus": r.get("plus_tol", ""), "minus": r.get("minus_tol", ""),
                   "lower": r.get("lower", ""), "upper": r.get("upper", ""), "result": r.get("result", ""),
                   "accept": r.get("accept", ""), "nc": r.get("nonconformance", "")}
            if not (val["number"] or val["requirement"]):
                continue
            if val["number"] and not NUMBER_LIKE.match(val["number"]):
                continue
            where = f"page {p} line {i}" if len(pages) > 1 else f"line {i}"
            rows.append(_make_row(len(rows), where, val))
    layout = {"source": "pdf" if path.suffix.lower() == ".pdf" else "image", "pages": len(imgs),
              "columns": {"read": (pages[0].get("columns") if pages else [])}}
    return rows, layout


# ---- comparing text ---------------------------------------------------------------------------------------------
_SYMBOLS = str.maketrans({"\u2212": "-", "\u2013": "-", "\u2014": "-", "⌀": "Ø", "∅": "Ø", "Ⓜ": " M ",
                          "Ⓛ": " L ", "Ⓢ": " S ", "Ⓟ": " P ", "Ⓕ": " F ", "|": " ", "[": " ", "]": " "})
_COUNT = [r"\(\s*\d+\s*of\s*\d+\s*\)", r"^\s*\d+\s*[Xx]\s+", r"\(\s*\d+\s*(?:PL(?:ACES|CS)?|X)?\s*\)",
          r"\bTYP\.?(?:\s*\d+\s*PL(?:ACES|CS)?)?", r"\b\d+\s*PL(?:ACES|CS|\.)?\b"]
_ENUM = re.compile(r"^\s*(?:NOTE\s*)?\d{1,2}\s*[.)]?\s+(?=[A-Z])")


def _clean(text: str, note: bool) -> str:
    t = (text or "").upper().translate(_SYMBOLS)
    t = " ".join(t.split())
    for pat in _COUNT:
        t = re.sub(pat, " ", t, flags=re.I)
    if note:
        rest = _ENUM.sub("", t, count=1)
        if len(rest) > 20:
            t = rest
    return " ".join(t.split())


def _numbers(t: str) -> Counter:
    out = Counter()
    for tok in re.findall(r"\d+\s+\d+/\d+|\d+/\d+|\d*\.\d+|\d+", t):
        f = to_float(tok)
        if f is not None:
            out[round(f, 6)] += 1
    return out


_IGNORED_WORDS = {"DIA", "DIAM", "TYP", "REF", "BASIC", "PLACES", "PLCS", "PL", "THRU", "INCH", "INCHES", "IN", "MM",
                  "POSITION", "PROFILE", "FLATNESS", "PERPENDICULARITY", "ANGULARITY", "SURFACE", "LINE", "OF", "A"}


def _words(t: str, note: bool = False) -> Counter:
    return Counter(w for w in re.findall(r"[A-Z]{2,}", t) if note or w not in _IGNORED_WORDS)


def _datums(t: str) -> set[str]:
    return {w for w in re.findall(r"(?<![A-Z])[A-Z](?![A-Z])", t) if w not in "MLSXPFR"}


def _modifiers(t: str) -> set[str]:
    return set(re.findall(r"(?<![A-Z])\(?\s*([ML])\s*\)?(?![A-Z])", t)) if re.search(r"Ø|\bDIA", t) else set()


def _gdt_symbols(t: str) -> set[str]:
    return set(re.findall(r"[⊕⌖⌓⌒⊥∠▱⏥◡○⌭↗⌰⫽∥]", t))


def _canon(t: str) -> str:
    """Cleaned text with every number written one way, for a character-level comparison."""
    t = re.sub(r"\(\s*([MLSPF])\s*\)", r" \1 ", t)
    t = re.sub(r"\d*\.\d+|\d+", lambda m: fmt(to_float(m.group(0))), t)
    return " ".join(t.split())


def _is_note(d: DrawingLine) -> bool:
    return d.req.type in ("note", "other") or (d.req.attribute and len(re.findall(r"[A-Z]{2,}", d.text.upper())) >= 4)


def _same(a: float | None, b: float | None, slack: float = EPS) -> bool:
    return a is not None and b is not None and abs(a - b) <= slack


def _limits_agree(s: "SupplierRow", d: "DrawingLine") -> bool:
    return _same(s.lower, d.req.lower, s.slack) and _same(s.upper, d.req.upper, s.slack)


def similarity(s: SupplierRow, d: DrawingLine) -> float:
    """0 to 1: how well a supplier row's requirement agrees with a drawing line."""
    note = _is_note(d)
    st, dt = _clean(s.text, note), _clean(d.text, note)
    sn, dn = _numbers(st), _numbers(dt)
    sw, dw = _words(st, note), _words(dt, note)
    if sn or dn:
        inter = sum((sn & dn).values())
        ns = inter / max(sum((sn | dn).values()), 1)
    else:
        ns = 1.0
    if sw or dw:
        ws = fuzz.token_sort_ratio(" ".join(sorted(sw.elements())), " ".join(sorted(dw.elements()))) / 100
    else:
        ws = 1.0
    if note:
        sim = 0.4 * ns + 0.6 * ws
    else:
        sim = 0.6 * ns + 0.2 * ws + 0.2 * fuzz.ratio(_canon(st), _canon(dt)) / 100
        if _datums(st) != _datums(dt):
            sim -= 0.2
        for a, b in ((_modifiers(st), _modifiers(dt)), (_gdt_symbols(st), _gdt_symbols(dt))):
            if a and b and a != b:
                sim -= 0.2
    if _limits_known(s) and _limits_known_d(d):
        sim += 0.1 if _limits_agree(s, d) else -0.1
    return max(0.0, sim)  # above 1 when limits agree too, so the best of several equal texts wins


def _limits_known(s: SupplierRow) -> bool:
    return s.lower is not None and s.upper is not None


def _limits_known_d(d: DrawingLine) -> bool:
    return d.req.lower is not None and d.req.upper is not None


def differences(s: SupplierRow, d: DrawingLine) -> list[str]:
    """Plain reasons why a matched row does not say what the drawing says; empty when it agrees."""
    note = _is_note(d)
    st, dt = _clean(s.text, note), _clean(d.text, note)
    out = []
    lim_s, lim_d = _limits_known(s), _limits_known_d(d)
    if lim_s and lim_d and not _limits_agree(s, d):
        out.append(f"limits {fmt(s.lower)} to {fmt(s.upper)} in the report, {fmt(d.req.lower)} to "
                   f"{fmt(d.req.upper)} on the drawing")
    sn, dn = _numbers(st), _numbers(dt)
    if sn != dn:
        # The same limits written another way (".375±.005" for ".370-.380") are not a difference.
        _, plo, phi, how = parse_limits(s.text)
        reexpressed = how in ("plus_minus", "unilateral", "limits") and lim_d and _same(plo, d.req.lower) \
            and _same(phi, d.req.upper)
        if not reexpressed and not out:
            only_d, only_s = dn - sn, sn - dn
            bits = []
            if only_d:
                bits.append(", ".join(fmt(v) for v in sorted(only_d.elements())) + " on the drawing")
            if only_s:
                bits.append(", ".join(fmt(v) for v in sorted(only_s.elements())) + " in the report")
            out.append("numbers differ: " + " but ".join(bits))
    if note:
        sw, dw = _words(st, True), _words(dt, True)
        if sw != dw:
            gone, added = sorted((dw - sw).elements()), sorted((sw - dw).elements())
            bits = []
            if gone:
                bits.append("drawing words missing: " + " ".join(gone))
            if added:
                bits.append("words not on the drawing: " + " ".join(added))
            out.append("; ".join(bits))
    elif d.req.type == "gdt":
        sd, dd = _datums(st), _datums(dt)
        if sd and dd and sd != dd:
            out.append(f"datums {' '.join(sorted(sd))} in the report, {' '.join(sorted(dd))} on the drawing")
        sm, dm = _modifiers(st), _modifiers(dt)
        if sm and dm and sm != dm:
            names = {"M": "MMC", "L": "LMC"}
            out.append(f"{'/'.join(names[m] for m in sorted(sm))} in the report, "
                       f"{'/'.join(names[m] for m in sorted(dm))} on the drawing")
    return out


# ---- matching ---------------------------------------------------------------------------------------------------
def _assign(pairs: list[tuple[float, int, int]], used_s: set, used_d: set) -> list[tuple[int, int, float]]:
    out = []
    for strength, si, di in sorted(pairs, key=lambda p: -p[0]):
        if si in used_s or di in used_d:
            continue
        used_s.add(si)
        used_d.add(di)
        out.append((si, di, strength))
    return out


def match_rows(rows: list[SupplierRow], lines: list[DrawingLine]) -> tuple[dict[int, int], dict[int, str], str]:
    """Match supplier rows to drawing lines one to one. Returns (row -> line, row -> how, numbering)."""
    by_base: dict[int, list[DrawingLine]] = {}
    for d in lines:
        if d.base is not None:
            by_base.setdefault(d.base, []).append(d)
    # A row typed twice (same number, same text) is one characteristic: only the first copy is matched.
    first: dict[tuple[str, str], int] = {}
    copies = set()
    for s in rows:
        key = (_key(s.char_no), _canon(_clean(s.text, True)))
        if key[0] and key in first:
            copies.add(s.idx)
        first.setdefault(key, s.idx)
    rows = [s for s in rows if s.idx not in copies]
    seen: Counter = Counter()
    for s in rows:
        if s.base is not None:
            s.pos = _suffix_pos(s.char_no, seen[s.base])
            seen[s.base] += 1

    # 1. Same balloon number: text and limits pick the line, then order within the number.
    pairs = []
    for s in rows:
        for d in by_base.get(s.base, []) if s.base is not None else []:
            n = max(len(by_base[s.base]), 1)
            pairs.append((similarity(s, d) + 0.05 * (1 - abs(s.pos - d.pos) / n), s.idx, d.idx))
    used_s: set = set()
    used_d: set = set()
    by_number = _assign(pairs, used_s, used_d)
    by_idx = {s.idx: s for s in rows}
    agree = sum(1 for si, di, _ in by_number if similarity(by_idx[si], lines[di]) >= 0.6)
    numbering = "drawing"
    if len(rows) >= 5 and agree < 0.5 * max(len(by_number), 1):
        # The supplier used its own balloon numbers: match on text and limits alone, in reading order.
        numbering = "own"
        used_s, used_d = set(), set()
        pairs = []
        for k, s in enumerate(rows):
            for d in lines:
                sim = similarity(s, d)
                if sim >= 0.7:
                    pairs.append((sim - 0.05 * abs(k / len(rows) - d.idx / len(lines)), s.idx, d.idx))
        found = _assign(pairs, used_s, used_d)
        # Rows left over that read almost like a line left over are that line, copied wrongly.
        pairs = []
        for s in rows:
            if s.idx in used_s:
                continue
            for d in lines:
                if d.idx in used_d:
                    continue
                note = _is_note(d)
                r = fuzz.ratio(_canon(_clean(s.text, note)), _canon(_clean(d.text, note))) / 100
                if r >= 0.75:
                    pairs.append((r, s.idx, d.idx))
        found += _assign(pairs, used_s, used_d)
        return {si: di for si, di, _ in found}, {si: "text" for si, _, _ in found}, numbering

    match = {si: di for si, di, _ in by_number}
    how = {si: "number" for si in match}
    # 2. Rows left over: a near exact text and limits match under another number, at about the same place in the
    # report as the balloon is on the drawing.
    pairs = []
    for k, s in enumerate(rows):
        if s.idx in used_s:
            continue
        for d in lines:
            if d.idx in used_d:
                continue
            gap = abs(k / len(rows) - d.idx / len(lines))
            if gap > 0.1:
                continue
            sim = similarity(s, d)
            if sim >= 0.9 and not differences(s, d):
                pairs.append((sim - gap, s.idx, d.idx))
    for si, di, _ in _assign(pairs, used_s, used_d):
        match[si], how[si] = di, "text"
    return match, how, numbering


# ---- findings ---------------------------------------------------------------------------------------------------
ACCEPT_WORDS = {"a", "acc", "accept", "accepted", "ok", "pass", "passed", "y", "yes", "c", "conform", "conforms",
                "conforming", "good", "g", "✓", "✔", "x"}
REJECT_WORDS = {"r", "rej", "reject", "rejected", "nc", "fail", "failed", "n", "no", "nonconforming",
                "non-conforming", "nonconform", "does not conform", "nok", "ng", "mrb"}
EMPTY_WORDS = {"", "-", "--", "\u2014", "n/a", "na", "tbd", "?", "none", "nil", "pending", "open"}


def _accepted(s: SupplierRow) -> bool | None:
    """True if the row is accepted, False if rejected, None if it says neither."""
    nc = s.nc.strip().lower()
    if nc and nc not in EMPTY_WORDS and nc not in ("0", "no", "n"):
        return False
    a = s.accept.strip().lower().rstrip(".")
    if a in REJECT_WORDS or a.startswith("rej") or a.startswith("non"):
        return False
    if a in ACCEPT_WORDS or a.startswith("acc"):
        return True
    return None


def _result_status(req: Requirement, result: str) -> str:
    """Drawing-limit status of a result; several numbers in one cell fail if any fails."""
    r = (result or "").strip()
    if r.lower() in EMPTY_WORDS:
        return ""
    vals = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+(?:\.\d*)?", r.replace("−", "-"))
    numeric = vals and re.fullmatch(r"[\s\d.+\-−,;/°\"a-zA-Z]*", r) and not re.search(r"[a-zA-Z]{3,}", r)
    if numeric and not req.attribute and req.lower is not None and req.upper is not None:
        sts = [evaluate(req, v) for v in vals]
        return "fail" if "fail" in sts else ("pass" if "pass" in sts else "needs review")
    return evaluate(req, r)


def _dinfo(d: DrawingLine) -> dict:
    c = d.char
    return {"char_id": c.id, "number": c.number, "row_id": d.row_id, "requirement": d.text, "type": d.req.type,
            "lower": fmt(d.req.lower), "upper": fmt(d.req.upper), "box": list(c.box), "sheet": c.sheet,
            "zone": c.zone}


def _q(t: str, n: int = 70) -> str:
    t = " ".join((t or "").split())
    return f"\"{t[:n]}{'...' if len(t) > n else ''}\""


def _label(s: SupplierRow) -> str:
    return f"Row {s.char_no}" if s.char_no else f"The row at {s.where}"


def check_rows(job_json: dict, rows: list[SupplierRow], layout: dict | None = None) -> dict:
    lines = drawing_lines(job_json)
    match, how, numbering = match_rows(rows, lines)
    findings: list[dict] = []

    def add(kind, msg, s=None, d=None):
        findings.append({"kind": kind, "message": msg, "supplier_row": s.as_dict() if s else None,
                         "drawing": _dinfo(d) if d else None})

    # Repeated characteristic numbers.
    groups: dict[str, list[SupplierRow]] = {}
    for s in rows:
        if _key(s.char_no):
            groups.setdefault(_key(s.char_no), []).append(s)
    dup_rows = set()
    for grp in groups.values():
        if len(grp) > 1:
            dup_rows.update(g.idx for g in grp)
            for s in grp[1:]:
                d = lines[match[s.idx]] if s.idx in match else None
                add("duplicate_number", f"Characteristic number {grp[0].char_no} is used on {len(grp)} rows "
                    f"({', '.join(g.where for g in grp)}); every characteristic needs its own number.", s, d)

    matched_lines = set(match.values())
    for d in lines:
        if d.idx not in matched_lines:
            inst = f", place {d.row_id.rsplit('-', 1)[1]} of {d.char.multiplier}" if d.char.multiplier > 1 else ""
            add("missing_row", f"Balloon {d.row_id} {_q(d.text)} (sheet {d.char.sheet + 1}"
                f"{', zone ' + d.char.zone if d.char.zone else ''}{inst}) is on the drawing but has no row in the "
                "report.", None, d)

    rejected = 0
    for s in rows:
        if s.idx not in match:
            if s.idx in dup_rows:
                continue
            add("extra_row", f"{_label(s)} {_q(s.text)} ({s.where}) matches no requirement on the drawing.", s)
            continue
        d = lines[match[s.idx]]
        if how[s.idx] == "text" and numbering == "drawing" and s.idx not in dup_rows:
            add("number_mismatch", f"{_label(s)} {_q(s.text)} matches balloon {d.row_id} on the drawing, which has "
                "a different number.", s, d)
        diff = differences(s, d)
        if diff:
            add("requirement_mismatch", f"{_label(s)} says {_q(s.text)} but balloon {d.row_id} on the drawing says "
                f"{_q(d.text)}: {'; '.join(diff)}.", s, d)
        acc = _accepted(s)
        if acc is False:
            rejected += 1
        if s.result.strip().lower() in EMPTY_WORDS:
            if d.req.type != "reference":
                add("missing_result", f"{_label(s)} {_q(s.text, 50)} has no measured result.", s, d)
            continue
        st = _result_status(d.req, s.result)
        if st == "fail" and acc is not False:
            lim = (f"outside the drawing limits {fmt(d.req.lower)} to {fmt(d.req.upper)}"
                   if d.req.lower is not None and d.req.upper is not None else "a nonconforming result")
            marked = "is marked accepted" if acc else "has no reject or nonconformance number"
            add("accepted_out_of_tolerance", f"{_label(s)} records {s.result}, {lim} for balloon {d.row_id}, but "
                f"{marked}.", s, d)

    summary = {k: 0 for k in KINDS}
    for f in findings:
        summary[f["kind"]] += 1
    n = len(findings)
    if n == 0:
        verdict = (f"No problems found: all {len(lines)} drawing requirements have a matching row and every recorded "
                   "result is within the drawing limits.")
        if rejected:
            verdict += f" {rejected} row{'s are' if rejected > 1 else ' is'} rejected with a nonconformance record."
    else:
        parts = [f"{v} {KIND_WORDS[k][0] if v == 1 else KIND_WORDS[k][1]}" for k, v in summary.items() if v]
        verdict = f"Return to supplier: {n} problem{'s' if n > 1 else ''} found ({', '.join(parts)})."
    return {
        "ok": n == 0,
        "verdict": verdict,
        "summary": summary,
        "findings": findings,
        "numbering": numbering,
        "rows_read": len(rows),
        "drawing_lines": len(lines),
        "matched": len(match),
        "rejected_rows": rejected,
        "layout": layout or {},
        "matches": [{"supplier_index": si, "char_no": rows[si].char_no, "row_id": lines[di].row_id, "how": how[si]}
                    for si, di in sorted(match.items())],
    }


async def check_report(job_json: dict, report_path: Path, client=None) -> dict:
    """Check a supplier Form 3 (xlsx, PDF or image) against a finished drawing job (job.json as a dict)."""
    report_path = Path(report_path)
    suf = report_path.suffix.lower()
    usage = {}
    if suf in (".xlsx", ".xlsm"):
        rows, layout = read_xlsx(report_path)
    elif suf == ".pdf" or suf in IMAGE_SUFFIXES:
        if client is None:
            from .model_client import ModelConfig, VisionClient

            client = VisionClient(ModelConfig(effort="low"))
        rows, layout = await read_table_image(report_path, client)
        usage = client.usage.as_dict()
    else:
        raise ValueError(f"unsupported report file type: {suf}")
    out = check_rows(job_json, rows, layout)
    out["report"] = report_path.name
    out["usage"] = usage
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Check a supplier's AS9102 Form 3 against a Whiteout drawing job.")
    ap.add_argument("job_dir", type=Path, help="job folder with job.json")
    ap.add_argument("report", type=Path, help="supplier Form 3: .xlsx, .pdf or an image")
    ap.add_argument("--json", type=Path, help="write the full result here")
    a = ap.parse_args(argv)
    job = json.loads((a.job_dir / "job.json").read_text())
    res = asyncio.run(check_report(job, a.report))
    print(res["verdict"])
    print(f"rows read {res['rows_read']}, drawing lines {res['drawing_lines']}, matched {res['matched']}, "
          f"numbering: {res['numbering']}")
    for f in res["findings"]:
        print(f"- [{f['kind']}] {f['message']}")
    if res.get("usage"):
        print(f"model usage: {res['usage']}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
