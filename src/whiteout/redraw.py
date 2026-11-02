"""Redraw an old scanned drawing as a clean modern sheet.

Muse Spark reads every text string on the sheet, tile by tile. The text ink is then removed in code; what is left
(the line work) is cleaned, thinned to one pixel and traced into straight lines, arcs, circles and filled marks.
Nothing is drawn that is not on the scan: every line follows ink, and every string is set where its ink was, at the
measured letter height and rotation.

uv run python -m whiteout.redraw <image> [--job <job_dir>] --out <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from rapidfuzz import fuzz

from . import ingest, prompts
from .geometry import Box, Raw, SheetInk, Snap, area, expand, gap, inter, merge, norm_text, overlap_min, text_sim
from .model_client import ModelConfig, VisionClient
from .tiling import Tile, grid

ROTATIONS = [0, 90, 180, 270]  # counterclockwise; 90 reads bottom to top
TEXT_KINDS = ["dimension", "note", "title_block", "table_cell", "label", "zone_marker", "other"]

TEXT_SCHEMA = prompts._obj({"items": {"type": "array", "items": prompts._obj({
    "text": prompts.S,
    "box": prompts.BOX,
    "rotation": {"type": "integer", "enum": ROTATIONS},
    "kind": {"type": "string", "enum": TEXT_KINDS},
})}})

TEXT_PROMPT = (
    "You are an object grounding expert transcribing one tile of a scanned engineering drawing so that it can be "
    "redrawn. List EVERY piece of printed or hand-lettered text in this image: dimensions and tolerances, notes, "
    "title block and revision block entries, table headers and table cells, view and section labels, zone letters "
    "and numbers along the border, part and drawing numbers, letters inside circles or flags, and stamps. Do not skip "
    "small, faint, crowded, rotated or vertical text. "
    "One item per line of text: a note printed on three lines is three items; stacked limits (one value above "
    "another) are two items; each table cell is its own item. "
    "Give the text verbatim as printed (keep symbols such as Ø ± ° and fractions such as 1/32). Give a tight box "
    "[x_min, y_min, x_max, y_max] around the ink of that one line on a 0-1000 grid of this image, and its rotation: "
    "0 for normal text, 90 for text that reads from bottom to top, 270 for text that reads from top to bottom, 180 "
    "for upside-down text. Skip text that is cut off at the image edge (it is read from the next tile). Do not list "
    "lines, arrows or symbols that are not text, and never invent text that is not printed."
)

# Helvetica covers Latin-1; these are spelled with what the font has
SUBST = {"\u2300": "Ø", "\u2205": "Ø", "\u2104": "CL", "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2019": "'",
         "\u2018": "'", "\u201c": '"', "\u201d": '"'}
CAP = 0.718  # Helvetica cap height per em
PNG9 = [cv2.IMWRITE_PNG_COMPRESSION, 9]


@dataclass
class TextItem:
    text: str
    box: Box  # sheet pixels, tight on the ink when snapped
    rotation: int = 0
    kind: str = "other"
    source: str = "read"  # "read" (all-text read) or "job" (an existing characteristic)
    snapped: bool = False
    cap: float = 0.0  # letter height in pixels
    base: float = 0.0  # baseline: y for 0 and 180, x for 90 and 270
    raw_box: Box | None = None  # the model's box, before snapping


# ---------------------------------------------------------------- text


def infer_rotation(text: str, box: Box, rotation: int = 0) -> int:
    """Trust the model, but a long string in a tall thin box is vertical even if the model said 0."""
    w, h = box[2] - box[0], box[3] - box[1]
    n = len(norm_text(text))
    if rotation in (0, 180) and n >= 3 and h > 1.8 * w:
        return 90
    if rotation in (90, 270) and n >= 3 and w > 1.8 * h:
        return 0
    return rotation if rotation in ROTATIONS else 0


def dedupe(items: list[TextItem], margins: list[int] | None = None) -> list[TextItem]:
    """One copy per string: overlapping reads of similar text from neighbouring tiles; the read farthest from a
    tile edge wins."""
    order = sorted(range(len(items)), key=lambda i: -(margins[i] if margins else 0))
    kept: list[int] = []
    for i in order:
        a = items[i]
        dup = False
        for k in kept:
            b = items[k]
            ov = overlap_min(a.box, b.box)
            if ov > 0.85 or (ov > 0.4 and text_sim(a.text, b.text) >= 75):
                dup = True
                break
        if not dup:
            kept.append(i)
    return [items[i] for i in sorted(kept)]


def tighten(ink: SheetInk, b: Box) -> tuple[Box, bool]:
    """Shrink a box onto the glyph ink that lies mostly inside it (rotated strings, where blocks do not apply)."""
    pad = max(4, int(0.5 * ink.gh))
    x0, y0, x1, y1 = expand(b, pad, pad, ink.W, ink.H)
    sub = ink.text[y0:y1, x0:x1]
    if not sub.any():
        return b, False
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    keep = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        cb = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
        if a >= 3 and inter(cb, b) >= 0.5 * area(cb):
            keep.append(cb)
    if not keep:
        return b, False
    u = keep[0]
    for cb in keep[1:]:
        u = (min(u[0], cb[0]), min(u[1], cb[1]), max(u[2], cb[2]), max(u[3], cb[3]))
    return u, True


def glyph_metrics(ink: SheetInk, it: TextItem) -> None:
    """Letter height and baseline from the glyph components inside the box."""
    b = it.box
    x0, y0, x1, y1 = expand(b, 2, 2, ink.W, ink.H)
    sub = ink.text[y0:y1, x0:x1]
    vertical = it.rotation in (90, 270)
    span = (b[2] - b[0]) if vertical else (b[3] - b[1])
    hs, lows = [], []
    if sub.any():
        n, _, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < 3:
                continue
            if vertical:
                hs.append(w)
                lows.append(x0 + x + w if it.rotation == 90 else x0 + x)
            else:
                hs.append(h)
                lows.append(y0 + y + h if it.rotation == 0 else y0 + y)
    if len(hs) >= 2 and it.snapped:
        med = float(np.median(hs))
        tall = [h for h in hs if h >= 0.5 * med]
        top = float(np.median(tall))  # letters, not the leader or arrow that touches the string
        cap = float(np.clip(top, 0.45 * span, span))
        big = [lo for h, lo in zip(hs, lows) if 0.6 * top <= h <= 1.4 * top]
        base = float(np.median(big)) if big else None
    else:
        cap, base = min(0.8 * span, 1.5 * ink.gh), None
    if base is None:
        base = {0: b[3], 180: b[1], 90: b[2], 270: b[0]}[it.rotation]
    it.cap, it.base = max(4.0, cap), float(base)


def job_items(job: dict | None, sheet: int = 0) -> list[TextItem]:
    """Requirements an earlier Whiteout run already read and matched to ink."""
    if not job:
        return []
    sheets = job.get("sheets") or []
    if sheet >= len(sheets):
        return []
    out = []
    for c in sheets[sheet].get("chars", []):
        if c.get("ignored") or not c.get("text") or not c.get("box"):
            continue
        b = tuple(int(v) for v in c["box"])
        t = c["text"].strip()
        out.append(TextItem(text=t, box=b, rotation=infer_rotation(t, b), kind=c.get("kind", "other"), source="job",
                            snapped="box not matched to ink" not in (c.get("flags") or [])))
    return out


def ink_rows(ink: SheetInk, b: Box) -> list[Box]:
    """Rows of text blocks lying inside a box, top to bottom."""
    ids = [i for i in ink.blocks_in(b) if overlap_min(ink.blocks[i], b) > 0.6]
    rows: list[Box] = []
    for i in sorted(ids, key=lambda i: (ink.blocks[i][1] + ink.blocks[i][3]) / 2):
        bb = ink.blocks[i]
        cy = (bb[1] + bb[3]) / 2
        if rows and abs(cy - (rows[-1][1] + rows[-1][3]) / 2) < 0.6 * ink.gh:
            r = rows[-1]
            rows[-1] = (min(r[0], bb[0]), min(r[1], bb[1]), max(r[2], bb[2]), max(r[3], bb[3]))
        else:
            rows.append(bb)
    return rows


def split_words(words: list[str], widths: list[int]) -> list[str]:
    """Share words out over rows in proportion to each row's printed width."""
    if not words:
        return [""] * len(widths)
    total = sum(len(w) + 1 for w in words)
    cum = (np.cumsum([len(w) + 1 for w in words]) - np.array([(len(w) + 1) / 2 for w in words])) / total  # midpoints
    edges = np.cumsum(widths) / max(1, sum(widths))
    out, k = [], 0
    for r, e in enumerate(edges):
        take = []
        while k < len(words) and (cum[k] <= e + 1e-6 or r == len(edges) - 1):
            take.append(words[k])
            k += 1
        out.append(" ".join(take))
    return out


def split_lines(ink: SheetInk, it: TextItem) -> list[TextItem]:
    """A string printed on several lines becomes one item per line, placed on the rows of text blocks in its box."""
    if it.rotation != 0:
        return [TextItem(**{**asdict(it), "text": " ".join(it.text.split())})]
    lines = [s.strip() for s in it.text.split("\n") if s.strip()]
    rows = ink_rows(ink, it.box)
    if len(lines) <= 1 and len(rows) <= 1:
        return [TextItem(**{**asdict(it), "text": " ".join(lines) or it.text})]
    snapped = it.snapped
    if len(lines) <= 1 and len(rows) > 1:
        lines = split_words((lines[0] if lines else it.text).split(), [r[2] - r[0] for r in rows])
    elif len(rows) != len(lines):
        h = (it.box[3] - it.box[1]) / len(lines)
        rows = [(it.box[0], round(it.box[1] + k * h), it.box[2], round(it.box[1] + (k + 1) * h)) for k in range(len(lines))]
        snapped = False
    return [TextItem(text=t, box=r, rotation=0, kind=it.kind, source=it.source, snapped=snapped)
            for t, r in zip(lines, rows) if t]


def combine(read: list[TextItem], jobs: list[TextItem], gh: int) -> list[TextItem]:
    """The read gives one item per printed line. A job requirement on the same line with the same characters
    replaces the read's text (it was read more carefully); a job requirement that sits on ink and that no read line
    covers fills the gap."""
    out = [TextItem(**asdict(r)) for r in read]
    for j in jobs:
        hits = [r for r in out if overlap_min(j.box, r.box) > 0.5]
        if not hits:
            near = any(gap(j.box, r.box) < 1.5 * gh and text_sim(j.text, r.text) >= 80 for r in out)
            touch = any(overlap_min(j.box, r.box) > 0.1 for r in out)
            if j.snapped and not near and not touch:
                out.append(j)
            continue
        if len(hits) == 1:
            r = hits[0]
            same_box = inter(j.box, r.box) > 0.6 * max(area(j.box), area(r.box))
            nj, nr = norm_text(j.text), norm_text(r.text)
            if same_box and len(nj) == len(nr) and fuzz.ratio(nj, nr) >= 80:
                r.text, r.source = j.text, "job"
    return out


def strip_joined(items: list[TextItem]) -> int:
    """A read that joined a neighbour's text ("-63 1.2730" across two table cells) gives that part back when the
    neighbour was read on its own and took its ink."""
    n = 0
    for a in items:
        if a.rotation or not a.raw_box:
            continue
        words = a.text.split()
        for b in items:
            if b is a or b.rotation or not b.snapped or len(words) < 2:
                continue
            if overlap_min(b.box, a.raw_box) < 0.7 or overlap_min(b.box, a.box) > 0.2:
                continue
            bw = b.text.split()
            k = len(bw)
            if k < len(words) and [norm_text(w) for w in words[:k]] == [norm_text(w) for w in bw]:
                words = words[k:]
            elif k < len(words) and [norm_text(w) for w in words[-k:]] == [norm_text(w) for w in bw]:
                words = words[:-k]
            else:
                continue
            a.text = " ".join(words)
            n += 1
    return n


def fix_order(items: list[TextItem], gh: int) -> int:
    """Stacked strings ("2.101" over "2.100") whose snapped boxes came out in the opposite order to the model's
    boxes trade places."""
    n = 0
    hz = [it for it in items if it.snapped and not it.rotation and it.raw_box]
    for a in hz:
        for b in hz:
            if a is b:
                continue
            ox = min(a.box[2], b.box[2]) - max(a.box[0], b.box[0])
            if ox < 0.5 * min(a.box[2] - a.box[0], b.box[2] - b.box[0]):
                continue
            ra, rb = (a.raw_box[1] + a.raw_box[3]) / 2, (b.raw_box[1] + b.raw_box[3]) / 2
            sa, sb = (a.box[1] + a.box[3]) / 2, (b.box[1] + b.box[3]) / 2
            if ra < rb - 0.5 * gh and sa > sb + 0.5 * gh and abs(sa - sb) < 2.5 * gh:
                a.box, b.box = b.box, a.box
                n += 1
    return n


def one_row(ink: SheetInk, items: list[TextItem], lh: int) -> int:
    """A one-line string snapped onto two rows of text keeps the row nearest the model's box; the other row is
    left for its own string."""
    n = 0
    for it in items:
        if it.rotation or not it.snapped or not it.raw_box or it.box[3] - it.box[1] < 1.6 * lh:
            continue
        if it.raw_box[3] - it.raw_box[1] > 1.4 * lh:
            continue
        rows = ink_rows(ink, it.box)
        if len(rows) < 2:
            continue
        cy = (it.raw_box[1] + it.raw_box[3]) / 2
        it.box = min(rows, key=lambda r: abs((r[1] + r[3]) / 2 - cy))
        n += 1
    return n


def resnap_free(ink: SheetInk, items: list[TextItem]) -> int:
    """Unsnapped horizontal strings take the glyph ink inside their box that no snapped string owns."""
    n = 0
    for it in items:
        if it.snapped or it.rotation:
            continue
        b, ok = tighten(ink, it.box)
        if ok and not any(o.snapped and overlap_min(o.box, b) > 0.3 for o in items if o is not it):
            it.box, it.snapped = b, True
            n += 1
    return n


def advance(items: list[TextItem]) -> float:
    """Typical width per character on this sheet, from snapped strings."""
    v = [(it.box[2] - it.box[0]) / len(it.text.strip()) for it in items
         if it.snapped and not it.rotation and len(it.text.strip()) >= 4]
    return float(np.median(v)) if len(v) >= 5 else 0.0


def grow(ink: SheetInk, items: list[TextItem]) -> int:
    """A snap that took only part of a string grows over unclaimed glyph blocks on its line: those inside the
    model's box, then neighbours along the line while the box is shorter than the string needs. No letters are
    left behind to be traced as lines."""
    owner = np.zeros(len(ink.blocks), np.int32) - 1
    for k, it in enumerate(items):
        if it.snapped:
            for i in ink.blocks_in(it.box):
                if overlap_min(ink.blocks[i], it.box) > 0.6:
                    owner[i] = k
    adv = advance(items)
    gh = ink.gh
    n = 0
    for k, it in enumerate(items):
        if it.rotation or not it.snapped:
            continue
        b = it.box
        cy, h = (b[1] + b[3]) / 2, max(b[3] - b[1], gh)

        def on_line(blk):
            return abs((blk[1] + blk[3]) / 2 - cy) <= 0.5 * h and blk[3] - blk[1] <= 1.6 * h

        nb = b
        if it.raw_box:
            zone = expand(it.raw_box, int(0.4 * gh), int(0.2 * gh), ink.W, ink.H)
            for i in ink.blocks_in(zone):
                blk = ink.blocks[i]
                if owner[i] < 0 and area(blk) and inter(blk, zone) >= 0.7 * area(blk) and on_line(blk):
                    nb = (min(nb[0], blk[0]), min(nb[1], blk[1]), max(nb[2], blk[2]), max(nb[3], blk[3]))
                    owner[i] = k
        want = adv * len(it.text.strip())
        while adv and nb[2] - nb[0] < 0.85 * want:
            look = (max(0, nb[0] - int(1.2 * gh)), nb[1], min(ink.W, nb[2] + int(1.2 * gh)), nb[3])
            cands = [i for i in ink.blocks_in(look) if owner[i] < 0 and on_line(ink.blocks[i])]
            cands = [i for i in cands if max(nb[0], ink.blocks[i][0]) > min(nb[2], ink.blocks[i][2]) - 2
                     or inter(ink.blocks[i], nb) == 0]
            if not cands:
                break
            i = min(cands, key=lambda i: gap(ink.blocks[i], nb))
            blk = ink.blocks[i]
            u = (min(nb[0], blk[0]), min(nb[1], blk[1]), max(nb[2], blk[2]), max(nb[3], blk[3]))
            if u[2] - u[0] > 1.25 * want:
                break
            nb = u
            owner[i] = k
        if nb != b:
            it.box = nb
            n += 1
    return n


def fit_width(ink: SheetInk, items: list[TextItem]) -> int:
    """A snap far wider than its string needs (it took a leader or arrowhead) is replaced by the model's box
    tightened onto glyph ink, when that fits the string clearly better."""
    adv = advance(items)
    if not adv:
        return 0
    n = 0
    for it in items:
        if it.rotation or not it.snapped or not it.raw_box:
            continue
        want = adv * max(1, len(it.text.strip()))
        if it.box[2] - it.box[0] <= 2.0 * want:
            continue
        t, ok = tighten(ink, it.raw_box)
        if ok and abs(math.log(max(1, t[2] - t[0]) / want)) < 0.5 * abs(math.log((it.box[2] - it.box[0]) / want)):
            it.box = t
            n += 1
    return n


def resolve_overlaps(items: list[TextItem]) -> list[TextItem]:
    """Two different strings set on the same ink: one of them is misplaced. Keep the one that fits best (on ink,
    from the read, width matching its length); the other's ink stays and is traced as it is."""
    adv = advance(items) or 1.0

    def fit(it: TextItem) -> tuple:
        w = (it.box[3] - it.box[1]) if it.rotation in (90, 270) else (it.box[2] - it.box[0])
        return (it.snapped, it.source == "read", -abs(math.log(max(1, w) / (adv * max(1, len(it.text.strip()))))))

    drop = set()
    for i, a in enumerate(items):
        for j in range(i + 1, len(items)):
            b = items[j]
            if i in drop or j in drop or overlap_min(a.box, b.box) <= 0.4:
                continue
            drop.add(j if fit(a) >= fit(b) else i)
    return [it for k, it in enumerate(items) if k not in drop]


def drop_floating(ink: SheetInk, items: list[TextItem]) -> list[TextItem]:
    """Unsnapped strings stay only where there is ink under them and no snapped copy nearby."""
    out = []
    for it in items:
        if not it.snapped:
            x0, y0, x1, y1 = it.box
            if (ink.gray[y0:y1, x0:x1] < 180).mean() < 0.04 if area(it.box) else True:
                continue
            if any(o.snapped and gap(o.box, it.box) < 4 * ink.gh and text_sim(o.text, it.text) >= 95 for o in items):
                continue
        out.append(it)
    return out


def dedupe_near(items: list[TextItem], gh: int) -> list[TextItem]:
    """Two placed strings on the same ink with the same characters are one string; keep the longer, snapped one."""
    order = sorted(range(len(items)), key=lambda i: (not items[i].snapped, -len(norm_text(items[i].text))))
    kept: list[int] = []
    for i in order:
        a = items[i]
        if any((overlap_min(a.box, items[k].box) > 0.3 or gap(a.box, items[k].box) < 0.6 * gh)
               and text_sim(a.text, items[k].text) >= 90 for k in kept):
            continue
        kept.append(i)
    return [items[i] for i in sorted(kept)]


async def read_tiles(client: VisionClient, img: Image.Image, effort: str = "low") -> list[tuple[Tile, list[dict]]]:
    """Every text line on every tile, as the model returned it."""
    W, H = img.size

    async def one(t: Tile):
        res = await client.ask_json([img.crop(t.box)], TEXT_PROMPT, TEXT_SCHEMA, "all_text", effort)
        return t, res.get("items", [])

    return list(await asyncio.gather(*(one(t) for t in grid(W, H))))


def place(tiles: list[tuple[Tile, list[dict]]], ink: SheetInk) -> tuple[list[TextItem], dict]:
    """Model boxes (0-1000 grid of each tile) to sheet pixels, duplicates merged, snapped onto the glyph ink."""
    W, H = ink.W, ink.H
    n_raw, rot_raws, rot_margins, raws = 0, [], [], []
    for t, items in tiles:
        for d in items:
            text = (d.get("text") or "").strip()
            b = t.to_sheet(d.get("box", []))
            if not text or b[2] - b[0] < 3 or b[3] - b[1] < 3:
                continue
            n_raw += 1
            rot = infer_rotation(text, b, int(d.get("rotation") or 0))
            margin = t.edge_margin(b, W, H)
            if rot == 0:
                raws.append(Raw(data={"text": text, "kind": d.get("kind", "other"), "confidence": "high"}, box=b,
                                raw_box=b, margin=margin, tile=(t.x, t.y)))
            else:
                rot_raws.append(TextItem(text=text, box=b, rotation=rot, kind=d.get("kind", "other")))
                rot_margins.append(margin)
    # horizontal lines: the pipeline's merge (duplicates across tiles, then a shared assignment of ink blocks)
    items = []
    for r in merge(raws, ink):
        items.append(TextItem(text=r.text, box=r.box, rotation=0, kind=r.data.get("kind", "other"),
                              snapped=bool(r.snap and r.snap.snapped), raw_box=r.raw_box))
    for it in dedupe(rot_raws, rot_margins):
        it.raw_box = it.box
        it.box, it.snapped = tighten(ink, it.box)
        items.append(it)
    return items, {"tiles": len(tiles), "raw_items": n_raw, "lines_read": len(items)}


def letter_height(items: list[TextItem]) -> int:
    """Median height of snapped one-line horizontal strings: the sheet's letter height as the reads see it."""
    hs = [it.box[3] - it.box[1] for it in items if it.snapped and not it.rotation and len(norm_text(it.text)) >= 2]
    return int(np.median(hs)) if len(hs) >= 10 else 0


def retune(ink: SheetInk, gh: int) -> None:
    """Rebuild the glyph mask and text blocks for a letter height (same steps as SheetInk, plus runs of touching
    letters, minus hollow circles). On faint scans the height estimated from the ink alone can be too small, and
    then touching letter pairs count as line work."""
    gray = ink.gray
    raw = (gray < 180).astype(np.uint8)
    L = max(25, int(2.6 * gh))
    hl = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vl = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    lines = cv2.dilate(cv2.bitwise_or(hl, vl), np.ones((3, 3), np.uint8))
    text = cv2.bitwise_and(raw, cv2.bitwise_not(lines) & 1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
    keep = np.zeros(n, bool)
    w_, h_, a_ = st[:, 2], st[:, 3], st[:, 4]
    w_, h_, a_ = w_[1:], h_[1:], a_[1:]
    glyph = (np.maximum(w_, h_) <= 2.6 * gh) & ~((w_ > 1.8 * gh) & (h_ < 3))
    run = (h_ >= 0.5 * gh) & (h_ <= 1.6 * gh) & (w_ <= 6 * gh)  # touching letters (".066-")
    ring = ((np.minimum(w_, h_) >= 1.3 * gh) & (np.maximum(w_, h_) <= 3.2 * gh)
            & (np.abs(w_ - h_) <= 0.3 * np.maximum(w_, h_)) & (a_ < 0.35 * w_ * h_))  # balloon circles stay lines
    keep[1:] = (a_ >= 3) & (glyph | run) & ~ring
    ink.text = keep[lab].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.9 * gh)), 1))
    nb, blab, bst, _ = cv2.connectedComponentsWithStats(cv2.dilate(ink.text, k), connectivity=8)
    ink.block_lab = blab
    ink.blocks = [(0, 0, 0, 0)] + [(int(x + k.shape[1] // 2), int(y + k.shape[0] // 2), int(x + w - k.shape[1] // 2),
                                    int(y + h - k.shape[0] // 2)) for x, y, w, h, _ in bst[1:]]
    ink.gh = gh


def erase_text(ink: SheetInk, gray: np.ndarray, items: list[TextItem]) -> np.ndarray:
    snaps = [Snap(it.box, [], True) for it in items if it.snapped]
    fallback = [it.box for it in items if not it.snapped]
    return ink.erase(gray, snaps, fallback)


# ---------------------------------------------------------------- line work


def frame_region(gray: np.ndarray, border: Box | None, texts: list | None = None) -> Box:
    """The drawing sheet: the inner frame found on the ink near the given border, widened by the zone strip and the
    zone labels. Scanner and aperture card margins fall outside."""
    H, W = gray.shape
    if not border:
        return (0, 0, W, H)
    bx0, by0, bx1, by1 = border
    bw, bh = bx1 - bx0, by1 - by0
    if bw < 0.2 * W or bh < 0.2 * H:
        return (0, 0, W, H)
    pad = int(0.04 * max(bw, bh))
    ink = (gray < 160).astype(np.uint8)
    # frame lines on old scans are broken: count ink in a 5 px band across the border's span
    rows = cv2.dilate(ink[:, bx0:bx1], np.ones((5, 1), np.uint8)).mean(1)
    cols = cv2.dilate(ink[by0:by1, :], np.ones((1, 5), np.uint8)).mean(0)

    def pick(prof: np.ndarray, at: int) -> int:
        lo, hi = max(0, at - pad), min(len(prof), at + pad)
        seg = prof[lo:hi]
        good = np.nonzero(seg >= max(0.5, 0.8 * seg.max()))[0] if len(seg) else []
        if not len(good):
            return at
        return int(lo + good[np.argmin(np.abs(lo + good - at))])

    fx0, fx1, fy0, fy1 = pick(cols, bx0), pick(cols, bx1), pick(rows, by0), pick(rows, by1)
    s = int(0.018 * max(bw, bh))
    x0, y0, x1, y1 = fx0 - s, fy0 - s, fx1 + s, fy1 + s
    for t in texts or []:
        b = t.box
        if t.kind == "zone_marker" and overlap_min(b, (fx0 - 3 * s, fy0 - 3 * s, fx1 + 3 * s, fy1 + 3 * s)) > 0.9:
            x0, y0, x1, y1 = min(x0, b[0] - 10), min(y0, b[1] - 10), max(x1, b[2] + 10), max(y1, b[3] + 10)
    m = 8
    return (int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m)), int(min(H, y1 + m)))


def despeckle(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = st[:, 4] >= min_area
    keep[0] = False
    return keep[lab].astype(np.uint8)


def _zs_luts() -> tuple[np.ndarray, np.ndarray]:
    l1 = np.zeros(256, bool)
    l2 = np.zeros(256, bool)
    for c in range(256):
        p = [(c >> i) & 1 for i in range(8)]  # P2..P9: N NE E SE S SW W NW
        b = sum(p)
        a = sum(1 for i in range(8) if p[i] == 0 and p[(i + 1) % 8] == 1)
        if 2 <= b <= 6 and a == 1:
            l1[c] = p[0] * p[2] * p[4] == 0 and p[2] * p[4] * p[6] == 0
            l2[c] = p[0] * p[2] * p[6] == 0 and p[0] * p[4] * p[6] == 0
    return l1, l2


_ZS1, _ZS2 = _zs_luts()
_NK = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]], np.float32)  # neighbour code kernel, bit i = P(i+2)


def _codes(img: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img.astype(np.float32), -1, _NK, borderType=cv2.BORDER_CONSTANT).astype(np.int32)


def thin(mask: np.ndarray, max_iter: int = 60) -> np.ndarray:
    """Zhang-Suen thinning to a one pixel skeleton, then removal of staircase corners."""
    img = (mask > 0).astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for lut in (_ZS1, _ZS2):
            d = lut[_codes(img)] & (img > 0)
            if d.any():
                img[d] = 0
                changed = True
        if not changed:
            break
    # a pixel with two 4-neighbours at a right angle and nothing on the other side is a redundant stair step
    for need, empty in ((1 | 4, 16 | 32 | 64), (4 | 16, 1 | 64 | 128), (16 | 64, 1 | 2 | 4), (64 | 1, 4 | 8 | 16)):
        c = _codes(img)
        d = ((c & need) == need) & ((c & empty) == 0) & (img > 0)
        img[d] = 0
    return img


_OFFS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # 4-neighbours first


def trace(skel: np.ndarray, min_spur: float = 4.0) -> list[np.ndarray]:
    """Skeleton to ordered polylines (x, y). Junction clusters are cut out, and each branch end that touched one is
    extended to the cluster centre so lines stay connected."""
    sk = (skel > 0).astype(np.uint8)
    nb = cv2.filter2D(sk.astype(np.float32), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    junc = ((nb >= 3) & (sk > 0)).astype(np.uint8)
    nj, jlab, _, jcen = cv2.connectedComponentsWithStats(junc, connectivity=8)
    branch = (sk & (1 - junc)).astype(np.uint8)
    n, lab = cv2.connectedComponents(branch, connectivity=8)
    ys, xs = np.nonzero(lab)
    order = np.argsort(lab[ys, xs], kind="stable")
    ys, xs = ys[order], xs[order]
    labs = lab[ys, xs]
    starts = np.searchsorted(labs, np.arange(1, n + 1))
    ends = np.append(starts[1:], len(labs))
    H, W = sk.shape
    out = []
    for k in range(n - 1):
        py, px = ys[starts[k]:ends[k]], xs[starts[k]:ends[k]]
        pts = set(zip(py.tolist(), px.tolist()))

        def nbrs(p):
            return [(p[0] + dy, p[1] + dx) for dy, dx in _OFFS if (p[0] + dy, p[1] + dx) in pts]

        start = next((p for p in pts if len(nbrs(p)) <= 1), None)
        cyc = start is None
        if cyc:
            start = min(pts)
        path, seen, cur = [start], {start}, start
        while True:
            nxt = next((q for q in nbrs(cur) if q not in seen), None)
            if nxt is None:
                break
            path.append(nxt)
            seen.add(nxt)
            cur = nxt
        arr = np.array([(x, y) for y, x in path], np.float32)

        def jcent(p):
            y0, x0 = p
            win = jlab[max(0, y0 - 1):y0 + 2, max(0, x0 - 1):x0 + 2]
            ids = win[win > 0]
            return jcen[int(ids[0])].astype(np.float32) if len(ids) else None

        head = jcent(path[0])
        tail = jcent(path[-1]) if len(path) > 1 or head is None else None
        if not cyc:
            # a short branch with a free end at a junction is a thinning spur
            if (head is None) != (tail is None) and len(path) < min_spur:
                continue
            if head is not None:
                arr = np.vstack([head[None], arr])
            if tail is not None:
                arr = np.vstack([arr, tail[None]])
        elif len(path) > 2:
            arr = np.vstack([arr, arr[:1]])
        out.append(arr)
    return out


def fit_circle(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Least squares circle (Kasa); returns cx, cy, r, rms distance from the circle."""
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(1e-9, sol[2] + cx * cx + cy * cy))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), float(r), rms


def line_rms(pts: np.ndarray) -> float:
    c = pts.mean(0)
    _, s, _ = np.linalg.svd(pts - c, full_matrices=False)
    return float(s[-1] / math.sqrt(len(pts))) if len(s) > 1 else 0.0


def sweep(pts: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """Start angle and signed sweep (degrees, image coordinates) of points walked along an arc."""
    ang = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    return float(np.degrees(ang[0])), float(np.degrees(ang[-1] - ang[0]))


def snap_hv(x0: float, y0: float, x1: float, y1: float, tol: float = 1.6) -> tuple[float, float, float, float]:
    """Make a nearly horizontal or vertical segment exact when that moves no end more than tol pixels."""
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 6:
        return x0, y0, x1, y1
    t = max(tol, 0.0025 * L)
    if abs(y1 - y0) <= 2 * t and abs(y1 - y0) < abs(x1 - x0):
        y = (y0 + y1) / 2
        return x0, y, x1, y
    if abs(x1 - x0) <= 2 * t and abs(x1 - x0) < abs(y1 - y0):
        x = (x0 + x1) / 2
        return x, y0, x, y1
    return x0, y0, x1, y1


def merge_collinear(lines: list[list[float]], dist: float = 1.6, join: float = 3.0, ang: float = 1.5) -> list[list[float]]:
    """Join segments that lie on one line and touch or overlap (pieces cut at junctions, doubled traces).
    Each line is [x0, y0, x1, y1, width]."""
    segs = [list(map(float, s)) for s in lines]
    changed = True
    while changed:
        changed = False
        # bucket by direction and offset so only plausible pairs are compared
        buckets: dict[tuple[int, int], list[int]] = {}
        info = []
        for i, (x0, y0, x1, y1, _) in enumerate(segs):
            a = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180
            th = math.radians(a)
            off = -math.sin(th) * x0 + math.cos(th) * y0  # signed distance of the line from the origin
            info.append((a, off, th))
            buckets.setdefault((int(a // 3), int(off // 6)), []).append(i)
        alive = [True] * len(segs)
        for i in range(len(segs)):
            if not alive[i]:
                continue
            a, off, th = info[i]
            for da in (-1, 0, 1):
                for do in (-1, 0, 1):
                    for j in buckets.get((int((a // 3) + da) % 60, int(off // 6) + do), []):
                        if j <= i or not alive[j] or not alive[i]:
                            continue
                        aj, offj, _ = info[j]
                        d = abs(a - aj)
                        if min(d, 180 - d) > ang:
                            continue
                        si, sj = segs[i], segs[j]
                        # distance of j's ends from line i, and gap along the line
                        ux, uy = math.cos(th), math.sin(th)
                        p = [((sj[0] - si[0]) * ux + (sj[1] - si[1]) * uy), ((sj[2] - si[0]) * ux + (sj[3] - si[1]) * uy)]
                        q = [abs(-(sj[0] - si[0]) * uy + (sj[1] - si[1]) * ux), abs(-(sj[2] - si[0]) * uy + (sj[3] - si[1]) * ux)]
                        if max(q) > dist:
                            continue
                        li = (si[2] - si[0]) * ux + (si[3] - si[1]) * uy
                        lo, hi = min(p), max(p)
                        if lo > li + join or hi < -join:
                            continue
                        t0, t1 = min(0.0, lo), max(li, hi)
                        Li, Lj = math.hypot(si[2] - si[0], si[3] - si[1]), math.hypot(sj[2] - sj[0], sj[3] - sj[1])
                        # the merged line keeps the longer piece's offset; width is length weighted
                        base = si if Li >= Lj else sj
                        if base is sj:
                            ux2, uy2 = (sj[2] - sj[0]) / max(Lj, 1e-6), (sj[3] - sj[1]) / max(Lj, 1e-6)
                            if ux2 * ux + uy2 * uy < 0:
                                ux2, uy2 = -ux2, -uy2
                            # project the extremes onto j's line
                            ox, oy = si[0] + t0 * ux, si[1] + t0 * uy
                            ex, ey = si[0] + t1 * ux, si[1] + t1 * uy
                            s0 = (ox - sj[0]) * ux2 + (oy - sj[1]) * uy2
                            s1 = (ex - sj[0]) * ux2 + (ey - sj[1]) * uy2
                            nx0, ny0 = sj[0] + s0 * ux2, sj[1] + s0 * uy2
                            nx1, ny1 = sj[0] + s1 * ux2, sj[1] + s1 * uy2
                        else:
                            nx0, ny0 = si[0] + t0 * ux, si[1] + t0 * uy
                            nx1, ny1 = si[0] + t1 * ux, si[1] + t1 * uy
                        w = (si[4] * Li + sj[4] * Lj) / max(Li + Lj, 1e-6)
                        segs[i] = list(snap_hv(nx0, ny0, nx1, ny1)) + [w]
                        alive[j] = False
                        changed = True
                        a, off, th = (math.degrees(math.atan2(segs[i][3] - segs[i][1], segs[i][2] - segs[i][0])) % 180,
                                      off, th)
        segs = [s for s, ok in zip(segs, alive) if ok]
    return segs


def merge_arcs(arcs: list[dict], tol: float = 2.0) -> list[dict]:
    """Arcs cut apart at junctions that share one circle are joined; nearly full coverage becomes a circle."""
    groups: list[list[dict]] = []
    for a in arcs:
        for g in groups:
            b = g[0]
            if math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) <= tol and abs(a["r"] - b["r"]) <= tol:
                g.append(a)
                break
        else:
            groups.append([a])
    out = []
    for g in groups:
        cov = np.zeros(360, bool)
        bridge = int(math.degrees(4.0 / max(1.0, g[0]["r"]))) + 2  # junction gaps where another line crossed
        for a in g:
            s, e = (a["a0"], a["a0"] + a["sweep"]) if a["sweep"] >= 0 else (a["a0"] + a["sweep"], a["a0"])
            idx = np.arange(math.floor(s), math.ceil(e) + 1) % 360
            cov[idx] = True
        if not cov.all():
            holes = ~cov
            start = int(np.argmax(cov))
            k = 0
            while k < 360:
                i = (start + k) % 360
                if holes[i]:
                    j = k
                    while j < 360 and holes[(start + j) % 360]:
                        j += 1
                    if j - k <= bridge and j < 360:
                        cov[[(start + t) % 360 for t in range(k, j)]] = True
                    k = j
                else:
                    k += 1
        L = sum(abs(a["sweep"]) * a["r"] for a in g)
        w = sum(a["w"] * abs(a["sweep"]) * a["r"] for a in g) / max(L, 1e-6)
        cx = float(np.mean([a["cx"] for a in g]))
        cy = float(np.mean([a["cy"] for a in g]))
        r = float(np.mean([a["r"] for a in g]))
        if cov.sum() >= 340:
            out.append({"t": "circle", "cx": cx, "cy": cy, "r": r, "w": w})
            continue
        # contiguous covered runs become arcs
        if cov.all():
            continue
        startpos = int(np.argmin(cov))  # an uncovered degree, so runs do not wrap
        rolled = np.roll(cov, -startpos)
        k = 0
        while k < 360:
            if rolled[k]:
                j = k
                while j < 360 and rolled[j]:
                    j += 1
                out.append({"t": "arc", "cx": cx, "cy": cy, "r": r, "a0": float((k + startpos) % 360),
                            "sweep": float(j - k - 1), "w": w})
                k = j
            else:
                k += 1
    return out


def sharpen(dp: np.ndarray, short: float, closed: bool) -> np.ndarray:
    """Corners that thinning rounded off (long, short, long) become one sharp corner where the long sides meet,
    when that point is within a short distance of the cut."""
    pts = [np.asarray(v, float) for v in dp]
    changed = True
    while changed and len(pts) >= (4 if not closed else 5):
        changed = False
        n = len(pts)
        rng = range(n) if closed else range(1, n - 2)
        for i in rng:
            a, b, c, d = pts[(i - 1) % n], pts[i % n], pts[(i + 1) % n], pts[(i + 2) % n]
            if np.hypot(*(c - b)) >= short or np.hypot(*(b - a)) < 2 * short or np.hypot(*(d - c)) < 2 * short:
                continue
            u, v = b - a, d - c
            den = u[0] * v[1] - u[1] * v[0]
            if abs(den) < 1e-6 * np.hypot(*u) * np.hypot(*v):
                continue
            t = ((c[0] - a[0]) * v[1] - (c[1] - a[1]) * v[0]) / den
            x = a + t * u
            if np.hypot(*(x - (b + c) / 2)) > 1.5 * short:
                continue
            pts[i % n] = x
            del pts[(i + 1) % n]
            changed = True
            break
    return np.array(pts, np.float32)


def fills(ink: np.ndarray, line_w: float) -> tuple[list[np.ndarray], np.ndarray]:
    """Solid marks wider than any line (arrowheads, dots, filled triangles) as polygons, and their mask."""
    k = max(5, int(round(1.9 * line_w)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.morphologyEx(ink, cv2.MORPH_OPEN, se)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    polys, mask = [], np.zeros_like(ink)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < k * k * 0.6:
            continue
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if max(rw, rh) > 8 * max(1.0, min(rw, rh)):
            continue  # a thick line, traced as a line
        poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(np.float32) + np.float32([x, y])
        polys.append(poly)
        mask[y:y + h, x:x + w] |= comp
    return polys, mask


def vectorize(line_ink: np.ndarray, eps: float = 1.2) -> tuple[list[dict], dict]:
    """Line ink (uint8 0/1, text removed) to primitives: lines, arcs, circles, polylines, fills."""
    dt = cv2.distanceTransform(line_ink, cv2.DIST_L2, 5)
    skel0 = thin(line_ink)
    widths = 2 * dt[skel0 > 0] - 1  # the centre pixel's distance counts it once
    line_w = float(np.median(widths)) if len(widths) else 3.0
    fill_polys, fmask = fills(line_ink, line_w)
    skel = skel0 & (1 - cv2.dilate(fmask, np.ones((3, 3), np.uint8)))
    paths = trace(skel, min_spur=max(4.0, 1.5 * line_w))
    lines, arcs, polys = [], [], []
    for p in paths:
        if len(p) < 2:
            continue
        L = float(np.sum(np.hypot(*np.diff(p, axis=0).T)))
        if L < max(3.0, 0.8 * line_w):
            continue
        ip = np.clip(np.round(p).astype(int), 0, [line_ink.shape[1] - 1, line_ink.shape[0] - 1])
        w = max(1.0, float(np.median(2 * dt[ip[:, 1], ip[:, 0]])) - 1)
        closed = len(p) > 8 and np.hypot(*(p[0] - p[-1])) < 2.5
        q = p[:-1] if closed and np.hypot(*(p[0] - p[-1])) < 1e-3 else p
        dp = cv2.approxPolyDP(q.reshape(-1, 1, 2), eps, closed).reshape(-1, 2)
        if len(dp) == 2 and not closed:
            lines.append([*snap_hv(*dp[0], *dp[-1]), w])
            continue
        if len(p) >= 10:
            cx, cy, r, rms = fit_circle(p)
            a0, sw = sweep(p, cx, cy)
            if closed:
                sw = 360.0
            if (3 <= r <= 0.25 * max(line_ink.shape) and rms <= max(0.75, 0.012 * r) and abs(sw) >= 30
                    and line_rms(p) > 2.5 * rms):
                if abs(sw) >= 340:
                    arcs.append({"t": "arc", "cx": cx, "cy": cy, "r": r, "a0": 0.0, "sweep": 360.0, "w": w})
                else:
                    arcs.append({"t": "arc", "cx": cx, "cy": cy, "r": r, "a0": a0, "sweep": sw, "w": w})
                continue
        dp = sharpen(dp, max(4.0, 1.2 * line_w), closed)
        seglen = np.hypot(*np.diff(dp, axis=0).T) if len(dp) > 1 else np.zeros(1)
        if len(dp) <= 4 or seglen[seglen >= 10].sum() >= 0.8 * seglen.sum():  # mostly straight runs
            for a, b in zip(dp[:-1], dp[1:]):
                lines.append([*snap_hv(*a, *b), w])
            if closed:
                lines.append([*snap_hv(*dp[-1], *dp[0]), w])
        else:
            polys.append({"t": "poly", "pts": dp.tolist() + ([dp[0].tolist()] if closed else []), "w": w})
    lines = merge_collinear(lines)
    prims = [{"t": "line", "p": [round(v, 2) for v in s[:4]], "w": s[4]} for s in lines]
    prims += merge_arcs(arcs)
    prims += polys
    prims += [{"t": "fill", "pts": f.tolist()} for f in fill_polys]
    thin_w, thick_w = weights(prims, line_w)
    for q in prims:
        if "w" in q:
            q["w"] = thick_w if q["w"] > 0.5 * (thin_w + thick_w) and thick_w > thin_w else thin_w
    stats = {"line_width_px": round(line_w, 2), "thin_px": round(thin_w, 2), "thick_px": round(thick_w, 2),
             "lines": sum(q["t"] == "line" for q in prims), "arcs": sum(q["t"] == "arc" for q in prims),
             "circles": sum(q["t"] == "circle" for q in prims), "polylines": len(polys), "fills": len(fill_polys)}
    return prims, stats


def plen(q: dict) -> float:
    if q["t"] == "line":
        x0, y0, x1, y1 = q["p"]
        return math.hypot(x1 - x0, y1 - y0)
    if q["t"] == "arc":
        return abs(math.radians(q["sweep"])) * q["r"]
    if q["t"] == "circle":
        return 2 * math.pi * q["r"]
    if q["t"] == "poly":
        p = np.array(q["pts"])
        return float(np.sum(np.hypot(*np.diff(p, axis=0).T)))
    return 0.0


def weights(prims: list[dict], line_w: float) -> tuple[float, float]:
    """Two pen weights, as on a CAD sheet: the length-weighted medians of the thin and the thick strokes."""
    ws = np.array([q["w"] for q in prims if "w" in q])
    ls = np.array([plen(q) for q in prims if "w" in q])
    if not len(ws):
        return line_w, line_w

    def wmed(v, wt):
        o = np.argsort(v)
        c = np.cumsum(wt[o])
        return float(v[o][np.searchsorted(c, c[-1] / 2)])

    med = wmed(ws, ls)
    thick = ws > 1.45 * med
    tw = wmed(ws[~thick], ls[~thick]) if (~thick).any() else med
    kw = wmed(ws[thick], ls[thick]) if thick.sum() and ls[thick].sum() > 0.05 * ls.sum() else tw
    return max(1.0, tw), max(1.0, kw)


# ---------------------------------------------------------------- output


def arc_points(cx: float, cy: float, r: float, a0: float, sw: float) -> np.ndarray:
    n = max(4, int(abs(sw) / 3) + 1)
    t = np.radians(np.linspace(a0, a0 + sw, n))
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def render_lines(prims: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """The traced line work as a raster at sheet size (uint8 0/1), for measuring."""
    out = np.zeros(shape, np.uint8)
    S = 16  # 4 bits of sub-pixel precision
    for q in prims:
        t = q["t"]
        if t == "fill":
            cv2.fillPoly(out, [np.round(np.array(q["pts"]) * S).astype(np.int32)], 1, shift=4)
            continue
        th = max(1, int(round(q["w"])))
        if t == "line":
            x0, y0, x1, y1 = q["p"]
            cv2.line(out, (round(x0 * S), round(y0 * S)), (round(x1 * S), round(y1 * S)), 1, th, shift=4)
        elif t == "circle":
            cv2.circle(out, (round(q["cx"] * S), round(q["cy"] * S)), round(q["r"] * S), 1, th, shift=4)
        else:
            pts = q["pts"] if t == "poly" else arc_points(q["cx"], q["cy"], q["r"], q["a0"], q["sweep"])
            cv2.polylines(out, [np.round(np.array(pts) * S).astype(np.int32)], False, 1, th, shift=4)
    return out


def _font_text(s: str) -> str:
    return "".join(SUBST.get(ch, ch) for ch in s)


def text_anchor(it: TextItem) -> tuple[float, float, float]:
    """Start point of the baseline and the length the string must fill, in sheet pixels."""
    x0, y0, x1, y1 = it.box
    if it.rotation == 0:
        return x0, it.base, x1 - x0
    if it.rotation == 180:
        return x1, it.base, x1 - x0
    if it.rotation == 90:
        return it.base, y1, y1 - y0
    return it.base, y0, y1 - y0


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def write_svg(path: Path, W: int, H: int, region: Box, prims: list[dict], texts: list[TextItem]) -> None:
    rx0, ry0, rx1, ry1 = region
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{rx1 - rx0}" height="{ry1 - ry0}" '
           f'viewBox="{rx0} {ry0} {rx1 - rx0} {ry1 - ry0}">',
           f'<rect x="{rx0}" y="{ry0}" width="{rx1 - rx0}" height="{ry1 - ry0}" fill="#fff"/>']
    by_w: dict[float, list[str]] = {}
    for q in prims:
        t = q["t"]
        if t == "fill":
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in q["pts"])
            by_w.setdefault(-1.0, []).append(f'<polygon points="{pts}"/>')
            continue
        if t == "line":
            x0, y0, x1, y1 = q["p"]
            d = f"M{x0:.1f} {y0:.1f}L{x1:.1f} {y1:.1f}"
        elif t == "circle":
            by_w.setdefault(round(q["w"], 2), []).append(
                f'<circle cx="{q["cx"]:.1f}" cy="{q["cy"]:.1f}" r="{q["r"]:.1f}"/>')
            continue
        elif t == "arc":
            p = arc_points(q["cx"], q["cy"], q["r"], q["a0"], q["sweep"])
            large = 1 if abs(q["sweep"]) > 180 else 0
            sweepf = 1 if q["sweep"] > 0 else 0
            d = (f"M{p[0][0]:.1f} {p[0][1]:.1f}A{q['r']:.1f} {q['r']:.1f} 0 {large} {sweepf} "
                 f"{p[-1][0]:.1f} {p[-1][1]:.1f}")
        else:
            d = "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in q["pts"])
        by_w.setdefault(round(q["w"], 2), []).append(f'<path d="{d}"/>')
    for w, els in sorted(by_w.items()):
        if w < 0:
            out.append('<g fill="#000" stroke="none">' + "".join(els) + "</g>")
        else:
            out.append(f'<g fill="none" stroke="#000" stroke-width="{w}" stroke-linecap="round" '
                       f'stroke-linejoin="round">' + "".join(els) + "</g>")
    out.append('<g font-family="Helvetica, Arial, sans-serif" fill="#000">')
    for it in texts:
        x, y, L = text_anchor(it)
        fs = it.cap / CAP
        tr = f' transform="rotate({-it.rotation} {x:.1f} {y:.1f})"' if it.rotation else ""
        out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{fs:.1f}" textLength="{max(1.0, L):.1f}" '
                   f'lengthAdjust="spacingAndGlyphs"{tr}>{_esc(it.text)}</text>')
    out.append("</g></svg>")
    path.write_text("\n".join(out))


def write_pdf(path: Path, region: Box, prims: list[dict], texts: list[TextItem], scale: float) -> dict:
    """The same drawing as a PDF page at the sheet's printed size (scale = points per pixel)."""
    rx0, ry0, rx1, ry1 = region
    doc = fitz.open()
    page = doc.new_page(width=(rx1 - rx0) * scale, height=(ry1 - ry0) * scale)
    helv = fitz.Font("helv")
    alt = fitz.Font("cjk")  # symbols Helvetica lacks (triangles, angle, perpendicular); embedded only when used
    if any(not helv.has_glyph(ord(ch)) for it in texts for ch in _font_text(it.text) if ch != " "):
        page.insert_font(fontname="alt", fontbuffer=alt.buffer)

    def P(x, y):
        return fitz.Point((x - rx0) * scale, (y - ry0) * scale)

    shape = page.new_shape()
    for q in prims:
        t = q["t"]
        if t == "fill":
            shape.draw_polyline([P(x, y) for x, y in q["pts"]] + [P(*q["pts"][0])])
            shape.finish(color=None, fill=(0, 0, 0), width=0, closePath=True)
            continue
        if t == "line":
            x0, y0, x1, y1 = q["p"]
            shape.draw_line(P(x0, y0), P(x1, y1))
        elif t == "circle":
            shape.draw_circle(P(q["cx"], q["cy"]), q["r"] * scale)
        else:
            pts = q["pts"] if t == "poly" else arc_points(q["cx"], q["cy"], q["r"], q["a0"], q["sweep"])
            shape.draw_polyline([P(x, y) for x, y in pts])
        shape.finish(color=(0, 0, 0), width=q["w"] * scale, lineCap=1, lineJoin=1, closePath=False)
    shape.commit()
    missing = 0
    for it in texts:
        s = _font_text(it.text)
        use_alt = any(not helv.has_glyph(ord(ch)) for ch in s if ch != " ")
        if use_alt:
            bad = [ch for ch in s if ch != " " and not helv.has_glyph(ord(ch)) and not alt.has_glyph(ord(ch))]
            missing += len(bad)
        font, fname = (alt, "alt") if use_alt else (helv, "helv")
        x, y, L = text_anchor(it)
        fs = it.cap / CAP * scale
        nat = font.text_length(s, fontsize=fs)
        sx = float(np.clip((L * scale) / nat, 0.4, 2.5)) if nat > 0 else 1.0
        pt = P(x, y)
        page.insert_text(pt, s, fontsize=fs, fontname=fname, color=(0, 0, 0),
                         morph=(pt, fitz.Matrix(sx, 1) * fitz.Matrix(it.rotation)))
    try:
        doc.subset_fonts()
    except Exception:  # subsetting is only a size saving
        pass
    doc.save(path, garbage=3, deflate=True)
    return {"glyphs_not_in_font": missing}


def render_pdf(pdf: Path, width_px: int) -> np.ndarray:
    doc = fitz.open(pdf)
    page = doc[0]
    z = width_px / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(z, z), colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def side_by_side(a: np.ndarray, b: np.ndarray, max_w: int = 3000, gap_px: int = 24) -> np.ndarray:
    """Before (left) and after (right) at one scale, with a thin grey divider."""
    h = max(a.shape[0], b.shape[0])
    total = a.shape[1] + b.shape[1] + gap_px
    f = min(1.0, max_w / total)

    def rs(x):
        return cv2.resize(x, (max(1, round(x.shape[1] * f)), max(1, round(x.shape[0] * f))), interpolation=cv2.INTER_AREA)

    a2, b2 = rs(a), rs(b)
    H2 = round(h * f)
    g = max(4, round(gap_px * f))
    canvas = np.full((H2, a2.shape[1] + g + b2.shape[1]), 255, np.uint8)
    canvas[:a2.shape[0], :a2.shape[1]] = a2
    canvas[:, a2.shape[1] + g // 2 - 1:a2.shape[1] + g // 2 + 1] = 160
    canvas[:b2.shape[0], a2.shape[1] + g:] = b2
    return canvas


def detail_box(texts: list[TextItem], region: Box, w: int = 1500, h: int = 1000) -> Box:
    """The window with the most re-typeset text, for a zoomed before and after."""
    rx0, ry0, rx1, ry1 = region
    w, h = min(w, rx1 - rx0), min(h, ry1 - ry0)
    best, bn = (rx0, ry0, rx0 + w, ry0 + h), -1
    for x in range(rx0, rx1 - w + 1, max(1, w // 4)):
        for y in range(ry0, ry1 - h + 1, max(1, h // 4)):
            b = (x, y, x + w, y + h)
            n = sum(1 for t in texts if overlap_min(t.box, b) > 0.9 and t.kind != "title_block")
            if n > bn:
                best, bn = b, n
    return best


# ---------------------------------------------------------------- measuring


def coverage(a: np.ndarray, b: np.ndarray, tol: int = 3) -> float:
    """Fraction of a's pixels within tol pixels of a pixel of b."""
    n = int(a.sum())
    if n == 0:
        return 1.0
    near = cv2.dilate(b.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1)))
    return float((a & near).sum() / n)


# ---------------------------------------------------------------- main entry


def sheet_border(job: dict | None, sheet: int) -> Box | None:
    try:
        b = job["sheets"][sheet]["info"].get("border_box")
        return tuple(int(v) for v in b) if b else None
    except (KeyError, IndexError, TypeError):
        return None


async def overview_border(client: VisionClient, img: Image.Image) -> Box | None:
    """Ask for the drawing border the same way the pipeline's overview does (same call, so it is often cached)."""
    from .pipeline import fit, grid_box

    small, _ = fit(img, 1790, 1790)
    ov = await client.ask_json([small], prompts.OVERVIEW_PROMPT, prompts.OVERVIEW_SCHEMA, "overview")
    return grid_box(ov.get("border_box", []), *img.size)


async def redraw(image_path, job_json: dict | None, client: VisionClient | None, out_dir: str | Path | None = None,
                 sheet: int = 0, effort: str = "low", ink_threshold: int = 165) -> dict:
    """Redraw one sheet of image_path (scan, photo or PDF) into out_dir.

    job_json: a Whiteout job (job.json) whose requirements are reused, or None. client: a VisionClient, or None to
    trace the line work and set only the job's text. Returns paths (svg, pdf, png: the redraw at most 3000 px wide,
    before_after, before_after_detail, texts, lines), width and height of the redrawn area in sheet pixels, region
    (its box on the loaded sheet) and stats (text counts, line counts, faithfulness, usd)."""
    image_path = Path(image_path)
    out = Path(out_dir) if out_dir else image_path.parent / f"{image_path.stem}_redraw"
    out.mkdir(parents=True, exist_ok=True)
    usd0 = client.usage.usd if client else 0.0
    loaded = ingest.load(image_path)[sheet]
    img = loaded.image
    gray = np.array(img)
    H, W = gray.shape
    ink = await asyncio.to_thread(SheetInk, gray, loaded.dpi, loaded.dpi_trusted)

    border = sheet_border(job_json, sheet)
    if border is None and client is not None:
        border = await overview_border(client, img)

    # 1. text
    read, rstats = ([], {"tiles": 0, "raw_items": 0, "lines_read": 0})
    if client is not None:
        tiles = await read_tiles(client, img, effort)
        read, rstats = place(tiles, ink)
        lh = letter_height(read)
        rstats["glyph_height_px"] = [ink.gh, lh]
        await asyncio.to_thread(retune, ink, lh if lh > 1.2 * ink.gh else ink.gh)
        read, rstats2 = place(tiles, ink)
        rstats.update(rstats2)
    else:
        await asyncio.to_thread(retune, ink, ink.gh)
    jobs = [s for it in job_items(job_json, sheet) for s in split_lines(ink, it)]
    texts = combine(read, jobs, ink.gh)
    rstats["joined_split"] = strip_joined(texts)
    lh = letter_height(texts) or ink.gh
    rstats["order_fixed"] = fix_order(texts, lh)
    rstats["cut_to_one_row"] = one_row(ink, texts, lh)
    rstats["resnapped"] = resnap_free(ink, texts)
    rstats["grown"] = grow(ink, texts)
    rstats["width_fixed"] = fit_width(ink, texts)
    n0 = len(texts)
    texts = drop_floating(ink, resolve_overlaps(dedupe_near([t for t in texts if t.text.strip()], ink.gh)))
    rstats["dropped_copies_or_no_ink"] = n0 - len(texts)
    region = frame_region(gray, border, texts)
    rx0, ry0, rx1, ry1 = region
    texts = [t for t in texts if overlap_min(t.box, region) > 0.5]
    for t in texts:
        glyph_metrics(ink, t)

    # 2. line work: text ink removed, specks dropped, small gaps closed, then traced
    erased = erase_text(ink, gray, texts)
    inside = np.zeros_like(gray, np.uint8)
    inside[ry0:ry1, rx0:rx1] = 1
    line_ink0 = ((erased < ink_threshold) & (inside > 0)).astype(np.uint8)
    min_area = max(6, int(0.06 * ink.gh * ink.gh))
    line_ink = despeckle(line_ink0, min_area)
    line_ink = cv2.morphologyEx(line_ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    prims, vstats = await asyncio.to_thread(vectorize, line_ink)

    # 3. outputs
    scale = 72.0 / loaded.dpi if 60 <= loaded.dpi <= 1200 else 1.0
    svg_p, pdf_p = out / "redraw.svg", out / "redraw.pdf"
    write_svg(svg_p, W, H, region, prims, texts)
    pstats = write_pdf(pdf_p, region, prims, texts, scale)
    full = render_pdf(pdf_p, rx1 - rx0)  # sheet resolution, for measuring
    if full.shape != (ry1 - ry0, rx1 - rx0):
        full = cv2.resize(full, (rx1 - rx0, ry1 - ry0), interpolation=cv2.INTER_AREA)
    before = gray[ry0:ry1, rx0:rx1]
    f = min(1.0, 3000 / (rx1 - rx0))
    png_p = out / "redraw.png"
    cv2.imwrite(str(png_p), cv2.resize(full, (round(full.shape[1] * f), round(full.shape[0] * f)),
                                       interpolation=cv2.INTER_AREA), PNG9)
    ba_p = out / "before_after.png"
    cv2.imwrite(str(ba_p), side_by_side(before, full), PNG9)
    db = detail_box(texts, region)
    dx0, dy0, dx1, dy1 = (db[0] - rx0, db[1] - ry0, db[2] - rx0, db[3] - ry0)
    bad_p = out / "before_after_detail.png"
    cv2.imwrite(str(bad_p), side_by_side(before[dy0:dy1, dx0:dx1], full[dy0:dy1, dx0:dx1]), PNG9)

    # 4. faithfulness: traced lines against the scan's line ink (both within 3 px)
    drawn = render_lines(prims, (H, W))[ry0:ry1, rx0:rx1]
    orig_lines = line_ink0[ry0:ry1, rx0:rx1]
    orig_all = ((gray < ink_threshold).astype(np.uint8))[ry0:ry1, rx0:rx1]
    texts_p = out / "texts.json"
    texts_p.write_text(json.dumps([asdict(t) for t in texts], indent=0, default=lambda o: o.item()))
    prims_p = out / "lines.json"
    prims_p.write_text(json.dumps(prims, separators=(",", ":"), default=lambda o: o.item()))
    stats = {
        "sheet_px": [W, H],
        "region": list(region),
        "text": {**rstats, "from_job": sum(t.source == "job" for t in texts), "placed": len(texts),
                 "placed_snapped": sum(t.snapped for t in texts),
                 "rotated": sum(t.rotation != 0 for t in texts), **pstats},
        "lines": vstats,
        "faithfulness": {"line_ink_covered": round(coverage(orig_lines, drawn), 4),
                         "drawn_on_line_ink": round(coverage(drawn, orig_lines), 4),
                         "drawn_on_any_ink": round(coverage(drawn, orig_all), 4), "tolerance_px": 3},
        "usd": round((client.usage.usd - usd0) if client else 0.0, 4),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=lambda o: o.item()))
    return {"svg": str(svg_p), "pdf": str(pdf_p), "png": str(png_p), "before_after": str(ba_p),
            "before_after_detail": str(bad_p), "texts": str(texts_p), "lines": str(prims_p),
            "width": rx1 - rx0, "height": ry1 - ry0, "region": list(region), "stats": stats}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Redraw a scanned drawing as a clean vector sheet.")
    ap.add_argument("image")
    ap.add_argument("--job", help="a Whiteout job directory (job.json) whose requirements are reused")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", type=int, default=0)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--no-model", action="store_true", help="line work and job text only, no model calls")
    a = ap.parse_args(argv)
    job = json.loads((Path(a.job) / "job.json").read_text()) if a.job else None
    client = None if a.no_model else VisionClient(ModelConfig())
    res = asyncio.run(redraw(a.image, job, client, a.out, sheet=a.sheet, effort=a.effort))
    print(json.dumps({k: v for k, v in res.items() if k != "stats"}, indent=1))
    print(json.dumps(res["stats"], indent=1))
    if client:
        print("usage", client.usage.as_dict())


if __name__ == "__main__":
    main()
