"""Plant known requirements into real drawings, so recall can be measured on items whose truth is certain.

Each planted item is drafting-style text (OpenCV Hershey single-stroke font, close to CAD lettering) drawn in one of
four hard conditions: plain whitespace, small, rotated 90 degrees, or sitting on a drawing line.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

TEMPLATES = [
    ("dimension", ".625"), ("dimension", "1.250"), ("dimension", "2X .188"), ("dimension", "3.375"),
    ("dimension", ".094"), ("dimension", "4X .312"), ("dimension", "1.500 +/-.005"), ("dimension", ".750 +.002 -.000"),
    ("radius", "R.125"), ("radius", "2X R.062"), ("radius", "R.250 MAX"),
    ("hole_callout", "DIA .257-.261"), ("hole_callout", "2X DIA .136 THRU"), ("hole_callout", "DIA .4375-.4385"),
    ("angle", "30 DEG"), ("chamfer", ".030 X 45 DEG"), ("thread", ".250-20 UNC-2B"), ("thread", "4X #10-32 UNF-2B"),
    ("note", "BREAK SHARP EDGES .005-.015"), ("note", "DEBURR ALL EDGES"), ("note", "MARK PART NO PER MIL-STD-130"),
    ("note", "SURFACE ROUGHNESS 63 MAX"), ("note", "ALL FILLETS R.030"), ("note", "MASK BEFORE ANODIZE"),
]
CONDITIONS = ["plain", "small", "rotated", "on_line", "faint", "tiny"]


def _render_text(text: str, height: int, thick: int) -> np.ndarray:
    """Black text on white, tightly cropped, about `height` px cap height."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = height / 22.0
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    img = np.full((h + base + 2 * thick + 4, w + 2 * thick + 4), 255, np.uint8)
    cv2.putText(img, text, (thick + 2, h + thick + 2), font, scale, 0, thick, cv2.LINE_AA)
    ys, xs = np.where(img < 160)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _free(ink: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad: int) -> bool:
    H, W = ink.shape
    a, b, c, d = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    return c > a and d > b and ink[b:d, a:c].mean() < 0.002


def plant(gray: np.ndarray, glyph_h: int, n: int, seed: int, region: tuple[int, int, int, int] | None = None):
    """Return (new image, list of planted items with exact boxes)."""
    rnd = random.Random(seed)
    out = gray.copy()
    H, W = gray.shape
    ink = (gray < 150).astype(np.float32)
    x_lo, y_lo, x_hi, y_hi = region or (int(0.06 * W), int(0.06 * H), int(0.94 * W), int(0.88 * H))
    # long horizontal lines, for the on_line condition
    hl = cv2.morphologyEx((gray < 150).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(8 * glyph_h), 1)))
    line_pts = np.argwhere(hl > 0)
    items = []
    tries = 0
    while len(items) < n and tries < 4000:
        tries += 1
        kind, text = rnd.choice(TEMPLATES)
        cond = CONDITIONS[len(items) % len(CONDITIONS)]
        h = {"small": max(8, int(round(glyph_h * 0.7))), "tiny": max(7, int(round(glyph_h * 0.5)))}.get(cond, glyph_h)
        g = _render_text(text, h, max(1, round(h / 12)))
        if cond == "faint":  # light grey ink, as on a faded copy
            g = (255 - (255 - g.astype(np.int32)) * 0.38).astype(np.uint8)
        if cond == "rotated":
            g = np.rot90(g).copy()
        gh_, gw_ = g.shape
        if cond == "on_line":
            if len(line_pts) == 0:
                continue
            y, x = line_pts[rnd.randrange(len(line_pts))]
            x0, y0 = int(x - gw_ // 2), int(y - gh_ - max(1, h // 8))  # text sits on the line
            if not (x_lo <= x0 and x0 + gw_ <= x_hi and y_lo <= y0 and y0 + gh_ <= y_hi):
                continue
            if not _free(ink, x0, y0, x0 + gw_, y0 + gh_ - 2, 2):
                continue
        else:
            x0 = rnd.randrange(x_lo, max(x_lo + 1, x_hi - gw_))
            y0 = rnd.randrange(y_lo, max(y_lo + 1, y_hi - gh_))
            if not _free(ink, x0, y0, x0 + gw_, y0 + gh_, int(1.2 * glyph_h)):
                continue
        region_ = out[y0:y0 + gh_, x0:x0 + gw_]
        np.minimum(region_, g, out=region_)
        ink[max(0, y0 - glyph_h):y0 + gh_ + glyph_h, max(0, x0 - glyph_h):x0 + gw_ + glyph_h] = 1  # keep plants apart
        items.append({"text": text, "kind": kind, "condition": cond, "box": [x0, y0, x0 + gw_, y0 + gh_]})
    return out, items
