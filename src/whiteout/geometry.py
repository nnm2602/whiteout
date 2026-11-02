"""Box geometry, snapping model boxes onto real ink, and merging tile reads.

Model boxes drift by 10 to 30 px on a 3400 px sheet (measured Oct 27 2026). Before anything is erased or ballooned,
each box is snapped onto the connected ink components (glyphs, frames) that sit around it, in code.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
from rapidfuzz import fuzz

Box = tuple[int, int, int, int]


def area(b: Box) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a: Box, b: Box) -> int:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def overlap_min(a: Box, b: Box) -> float:
    m = min(area(a), area(b))
    return inter(a, b) / m if m else 0.0


def union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def expand(b: Box, dx: int, dy: int, w: int, h: int) -> Box:
    return (max(0, b[0] - dx), max(0, b[1] - dy), min(w, b[2] + dx), min(h, b[3] + dy))


def gap(a: Box, b: Box) -> float:
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float((dx * dx + dy * dy) ** 0.5)


def norm_text(s: str) -> str:
    s = (s or "").upper().replace("Ø", "D").replace("⌀", "D").replace("∅", "D").replace("±", "PM")
    return re.sub(r"[^A-Z0-9.]", "", s)


def ocr_norm(s: str) -> str:
    """Fold the characters Tesseract confuses on drawings so OCR and model text can be compared."""
    t = norm_text(s)
    return t.translate(str.maketrans({"O": "0", "S": "5", "I": "1", "L": "1", "B": "8", "Z": "2"}))


def ocr_words(gray: np.ndarray) -> list[tuple[Box, str]]:
    """Tesseract words (sparse text mode, 2x upscale) in sheet pixels. Empty if Tesseract is not installed."""
    exe = shutil.which("tesseract")
    if not exe:
        return []
    big = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        cv2.imwrite(f.name, big)
        out = subprocess.run([exe, f.name, "stdout", "--psm", "11", "tsv"], capture_output=True, text=True,
                             timeout=120).stdout
    words = []
    for r in csv.DictReader(io.StringIO(out), delimiter="\t"):
        t = (r.get("text") or "").strip()
        try:
            conf = float(r.get("conf") or -1)
        except ValueError:
            continue
        if t and conf > 20 and re.search(r"[0-9A-Za-z]", t):
            x, y, w, h = (int(r[k]) // 2 for k in ("left", "top", "width", "height"))
            words.append(((x, y, x + w, y + h), t))
    return words


def text_sim(a: str, b: str) -> float:
    na, nb = norm_text(a), norm_text(b)
    if not na or not nb:
        return 0.0
    return float(fuzz.partial_ratio(na, nb))


@dataclass
class Snap:
    box: Box
    blocks: list[int]  # ids of text blocks in the SheetInk map
    snapped: bool
    cost: float = 9.0


def estimate_glyph_height(ink: np.ndarray, dpi: float, dpi_trusted: bool = False) -> int:
    """Typical letter height on this sheet, measured from the ink (scans rarely carry a true dpi)."""
    guess = max(8, int(round(0.085 * dpi)))
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    hs = st[1:, 3]
    ws = st[1:, 2]
    sel = hs[(hs >= 7) & (hs <= 120) & (ws >= 2) & (ws <= hs * 1.6) & (st[1:, 4] >= 12)]
    if len(sel) < 50:
        return guess
    # glyphs form the dominant peak of component heights
    hist = np.bincount(sel).astype(float)
    # weight by height: dashes and specks are many but short; letters are the tall peak
    peak = int(np.argmax(np.convolve(hist, np.ones(5), "same") * np.arange(len(hist))))
    est = int(round(peak * 1.05))
    lo, hi = (0.8, 1.25) if dpi_trusted else (0.6, 2.5)
    return int(min(max(est, int(lo * guess)), int(hi * guess)))


class SheetInk:
    """Text blocks of a sheet, found once: drawing lines are removed, glyph-sized components are grouped into blocks.

    Model boxes are then matched to these blocks, so balloons and the whiteout land on real ink.
    """

    def __init__(self, gray: np.ndarray, dpi: float, dpi_trusted: bool = False):
        self.gray = gray
        H, W = gray.shape
        ink = (gray < 180).astype(np.uint8)  # includes faint letters on faded copies
        self.gh = gh = estimate_glyph_height((gray < 150).astype(np.uint8), dpi, dpi_trusted)
        L = max(25, int(2.6 * gh))
        hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
        vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
        lines = cv2.dilate(cv2.bitwise_or(hl, vl), np.ones((3, 3), np.uint8))
        text = cv2.bitwise_and(ink, cv2.bitwise_not(lines) & 1)
        n, lab, st, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
        keep = np.zeros(n, bool)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a >= 3 and max(w, h) <= 2.6 * gh and not (w > 1.8 * gh and h < 3):
                keep[i] = True
        self.text = keep[lab].astype(np.uint8)
        # join letters along a line only; lines above and below stay separate blocks
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.9 * gh)), 1))
        grouped = cv2.dilate(self.text, k)
        nb, blab, bst, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
        self.block_lab = blab
        self.blocks: list[Box] = [(0, 0, 0, 0)]
        for i in range(1, nb):
            x, y, w, h, _ = bst[i]
            # shrink back by the dilation margin
            self.blocks.append((int(x + k.shape[1] // 2), int(y + k.shape[0] // 2),
                                int(x + w - k.shape[1] // 2), int(y + h - k.shape[0] // 2)))
        self.W, self.H = W, H
        self.words = ocr_words(gray)

    def ocr_text_in(self, b: Box) -> str:
        return " ".join(t for wb, t in self.words
                        if b[0] <= (wb[0] + wb[2]) / 2 <= b[2] and b[1] <= (wb[1] + wb[3]) / 2 <= b[3])

    def blocks_in(self, b: Box) -> list[int]:
        x0, y0, x1, y1 = (max(0, b[0]), max(0, b[1]), min(self.W, b[2]), min(self.H, b[3]))
        if x1 <= x0 or y1 <= y0:
            return []
        ids = np.unique(self.block_lab[y0:y1, x0:x1])
        return [int(i) for i in ids if i != 0]

    def candidates(self, b: Box, lines_expected: int = 1, exclude: frozenset = frozenset(), text: str = "",
                   reach: float = 1.0) -> list[Snap]:
        """Pick the run of text blocks that best matches the model box in place and size.

        Candidates: the blocks mostly inside the box (works when the box is close), and every contiguous run of
        blocks on one text line near the box (works when the box drifted). Cost = size mismatch + offset.
        """
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bw <= 0 or bh <= 0:
            return Snap(b, [], False)
        gh = self.gh
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

        def cost(ub: Box) -> float:
            rw, rh = max(1, ub[2] - ub[0]), max(1, ub[3] - ub[1])
            dx = abs((ub[0] + ub[2]) / 2 - cx) / max(bw, gh)
            dy = abs((ub[1] + ub[3]) / 2 - cy) / max(bh, gh)
            return abs(np.log(rw / bw)) + 1.0 * abs(np.log(rh / bh)) + 0.8 * (dx + dy)

        cands: list[tuple[float, list[int], Box]] = []
        pad = int(0.5 * gh)
        inner = expand(b, pad, pad, self.W, self.H)
        mostly = [i for i in self.blocks_in(inner) if i not in exclude
                  and area(self.blocks[i]) and inter(self.blocks[i], inner) / area(self.blocks[i]) >= 0.4]
        if mostly:
            ub = self.blocks[mostly[0]]
            for i in mostly[1:]:
                ub = union(ub, self.blocks[i])
            cands.append((cost(ub) - 0.05, mostly, ub))

        zone = expand(b, int(1.6 * gh * reach), int(1.3 * gh * reach), self.W, self.H)
        near = [i for i in self.blocks_in(zone) if i not in exclude and area(self.blocks[i]) <= 6 * area(b) + 3000]
        lines: list[list[int]] = []
        for i in sorted(near, key=lambda i: self.blocks[i][0]):
            bi = self.blocks[i]
            placed = False
            for ln in lines:
                bl = self.blocks[ln[-1]]
                h = max(bi[3] - bi[1], bl[3] - bl[1], 1)
                if abs((bi[1] + bi[3]) / 2 - (bl[1] + bl[3]) / 2) < 0.5 * h and bi[0] - bl[2] < 2.4 * gh:
                    ln.append(i)
                    placed = True
                    break
            if not placed:
                lines.append([i])
        for ln in lines:
            for s0 in range(len(ln)):
                ub = self.blocks[ln[s0]]
                for s1 in range(s0, min(len(ln), s0 + 12)):
                    if s1 > s0:
                        ub = union(ub, self.blocks[ln[s1]])
                    if inter(ub, zone) / max(1, area(ub)) < 0.5:
                        continue
                    cands.append((cost(ub), ln[s0:s1 + 1], ub))
        if not cands:
            return []

        def nlines(ids: list[int]) -> int:
            ys = sorted((self.blocks[i][1] + self.blocks[i][3]) / 2 for i in ids)
            return 1 + sum(1 for a, b2 in zip(ys, ys[1:]) if b2 - a > 0.9 * gh)

        # Text check: does Tesseract see the same characters inside the candidate as the model read?
        want = ocr_norm(text)
        any_ocr = bool(want) and bool(self.ocr_text_in(zone))

        def text_pen(ub: Box) -> float:
            if not any_ocr:
                return 0.0
            got = ocr_norm(self.ocr_text_in(ub))
            if not got:
                return 0.6
            return 1.5 * (1 - fuzz.partial_ratio(want, got) / 100)

        out = sorted(((c + 0.9 * max(0, nlines(ids) - lines_expected) + text_pen(ub), ids, ub) for c, ids, ub in cands),
                     key=lambda t: t[0])
        return [Snap(ub, list(ids), True, float(c)) for c, ids, ub in out[:10] if c <= 2.8]

    def snap(self, b: Box, lines_expected: int = 1, exclude: frozenset = frozenset(), text: str = "") -> Snap:
        c = self.candidates(b, lines_expected, exclude, text)
        return c[0] if c else Snap(b, [], False)

    def erase(self, img: np.ndarray, snaps: list[Snap], fallback: list[Box]) -> np.ndarray:
        """Paint found text white: every glyph-sized mark inside each found box (drawing lines are kept)."""
        out = img.copy()
        m = np.zeros(self.text.shape, np.uint8)
        pad = max(2, int(0.3 * self.gh))
        for s in snaps:
            if not s.snapped:
                continue
            x0, y0, x1, y1 = expand(s.box, pad, pad, self.W, self.H)
            m[y0:y1, x0:x1] |= self.text[y0:y1, x0:x1]
        if m.any():
            m = cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool)
            out[m] = 255
        # light grey edges and faint strokes left inside found boxes (drawing lines are dark and stay)
        for sn in snaps:
            if not sn.snapped:
                continue
            x0, y0, x1, y1 = expand(sn.box, pad, pad, self.W, self.H)
            r = out[y0:y1, x0:x1]
            r[(r >= 150) & (r < 235)] = 255
        # Boxes that could not be matched to ink: erase only text blocks lying mostly inside them, never a whole
        # rectangle, so a missed neighbour is never wiped off the page.
        ids = []
        for b in fallback:
            for i in self.blocks_in(b):
                blk = self.blocks[i]
                if area(blk) and inter(blk, b) / area(blk) >= 0.6:
                    ids.append(i)
        if ids:
            sel = np.isin(self.block_lab, np.array(sorted(set(ids)))) & (self.text > 0)
            out[cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = 255
        return out


@dataclass
class Raw:
    """One item as read from one tile, mapped to the sheet."""

    data: dict
    box: Box
    raw_box: Box
    margin: int
    tile: tuple[int, int]
    snap: Snap | None = None

    @property
    def text(self) -> str:
        return self.data.get("text", "")

    @property
    def truncated(self) -> bool:
        return self.margin < 8


def lines_expected(data: dict) -> int:
    t = data.get("text") or ""
    return max(1, t.count("\n") + 1, round(len(t) / 60) + 1 if data.get("kind") == "note" else 1)


def merge(raws: list[Raw], ink: "SheetInk | None" = None) -> list[Raw]:
    """One copy per item, then a shared assignment of ink to items.

    1. Duplicates (the same text read by overlapping tiles) are merged using the model's own boxes; complete,
       central reads win over reads cut by a tile edge.
    2. Every item proposes its best-fitting runs of text blocks; runs are handed out cheapest first and no block
       is given to two items, so one drifted box cannot steal its neighbour's ink.
    """

    def key(r: Raw):
        return (r.truncated, r.data.get("confidence") != "high", -min(r.margin, 400), -len(norm_text(r.text)))

    near_px = int(1.8 * ink.gh) if ink is not None else 30
    kept: list[Raw] = []
    absorbed: set[tuple[int, tuple[int, int]]] = set()  # (kept index, tile) pairs already merged
    for r in sorted(raws, key=key):
        best, bfit = None, 0.0
        for ki, k in enumerate(kept):
            sim = text_sim(r.text, k.text)
            ov = overlap_min(r.raw_box, k.raw_box)
            close = ov > 0.3 or gap(r.raw_box, k.raw_box) < near_px
            same = (close and sim >= 85) or (r.truncated and ov > 0.1 and sim >= 70) or ov > 0.9
            # Two items from the same tile are two items (stacked identical frames), never one.
            if same and k.tile != r.tile and (ki, r.tile) not in absorbed:
                fit = sim / 100 + ov
                if fit > bfit:
                    best, bfit = ki, fit
        if best is None:
            kept.append(r)
        else:
            absorbed.add((best, r.tile))

    if ink is None:
        return kept
    pairs = []
    for idx, r in enumerate(kept):
        for c in ink.candidates(r.raw_box, lines_expected(r.data), text=r.text):
            pairs.append((c.cost, idx, c))
    pairs.sort(key=lambda t: t[0])
    used: set[int] = set()
    done: set[int] = set()
    for cost, idx, c in pairs:
        if idx in done or used & set(c.blocks):
            continue
        kept[idx].snap = c
        kept[idx].box = c.box
        used |= set(c.blocks)
        done.add(idx)
    # Second chance for items that did not snap: shift their box by the typical drift of their tile (measured on
    # items that did snap), and search a wider area with the text check deciding.
    drift: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for idx in done:
        r = kept[idx]
        drift.setdefault(r.tile, []).append((((r.box[0] + r.box[2]) - (r.raw_box[0] + r.raw_box[2])) / 2,
                                            ((r.box[1] + r.box[3]) - (r.raw_box[1] + r.raw_box[3])) / 2))
    allv = [v for vs in drift.values() for v in vs]
    retry = []
    for idx, r in enumerate(kept):
        if idx in done:
            continue
        vs = drift.get(r.tile) if len(drift.get(r.tile, [])) >= 3 else allv
        dx, dy = (float(np.median([v[0] for v in vs])), float(np.median([v[1] for v in vs]))) if vs else (0.0, 0.0)
        b = r.raw_box
        shifted = (int(b[0] + dx), int(b[1] + dy), int(b[2] + dx), int(b[3] + dy))
        for c in ink.candidates(shifted, lines_expected(r.data), text=r.text, reach=2.5):
            retry.append((c.cost, idx, c))
    retry.sort(key=lambda t: t[0])
    for cost, idx, c in retry:
        if idx in done or used & set(c.blocks):
            continue
        kept[idx].snap = c
        kept[idx].box = c.box
        used |= set(c.blocks)
        done.add(idx)
    for idx, r in enumerate(kept):
        if idx not in done:
            r.snap = Snap(r.raw_box, [], False)
            r.box = r.raw_box
    # Unsnapped near-copies of a snapped item are the same text read with a bad box.
    out = []
    for r in kept:
        if not r.snap.snapped and any(k.snap.snapped and gap(r.box, k.box) < 50 and text_sim(r.text, k.text) >= 95
                                      for k in kept if k is not r):
            continue
        out.append(r)
    return out
