"""Zone lookup, balloon numbering in reading order, and balloon placement off the ink."""

from __future__ import annotations

import math

import numpy as np

from .geometry import Box
from .types import Characteristic, SheetInfo


def balloon_radius(info: SheetInfo) -> int:
    return max(14, round(0.11 * info.dpi))  # about 0.22 in diameter on paper


def _zone_index(v: float, lo: int, hi: int, n: int) -> int:
    if n <= 0 or hi <= lo:
        return -1
    return int(min(n - 1, max(0, (v - lo) * n // (hi - lo))))


def assign_zones(chars: list[Characteristic], info: SheetInfo) -> None:
    b = info.border_box or (0, 0, info.width, info.height)
    cols, rows = info.zone_columns, info.zone_rows
    for c in chars:
        cx, cy = (c.box[0] + c.box[2]) / 2, (c.box[1] + c.box[3]) / 2
        ci = _zone_index(cx, b[0], b[2], len(cols))
        ri = _zone_index(cy, b[1], b[3], len(rows))
        c.zone = (rows[ri] if ri >= 0 else "") + (cols[ci] if ci >= 0 else "")


def reading_order(chars: list[Characteristic], info: SheetInfo) -> list[Characteristic]:
    """Top to bottom in bands, left to right within a band (zone rows when the border has them)."""
    b = info.border_box or (0, 0, info.width, info.height)
    nrows = max(4, len(info.zone_rows) * 2)
    band = max(1, (b[3] - b[1]) / nrows)

    def key(c: Characteristic):
        cy = (c.box[1] + c.box[3]) / 2
        return (int((cy - b[1]) // band), c.box[0], c.box[1])

    return sorted(chars, key=key)


def number_and_place(chars: list[Characteristic], gray: np.ndarray, info: SheetInfo, start: int = 1) -> int:
    active = [c for c in chars if not c.ignored]
    r = balloon_radius(info)
    ink = (gray < 150).astype(np.int32)
    integral = ink.cumsum(0).cumsum(1)
    taken: list[tuple[int, int]] = []
    boxes = [c.box for c in active]
    n = start
    for c in reading_order(active, info):
        c.number = str(n)
        n += 1
        c.balloon_xy = _place(c.box, r, integral, info, taken, boxes)
        taken.append(c.balloon_xy)
    return n


def _ink_in(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
    H, W = integral.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def _place(b: Box, r: int, integral: np.ndarray, info: SheetInfo, taken: list, boxes: list[Box]) -> tuple[int, int]:
    x0, y0, x1, y1 = b
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d = r + 6
    cands = [
        (x0 - d, cy), (x0 - d, y0 - d), (cx, y0 - d), (x1 + d, y0 - d), (x1 + d, cy),
        (x1 + d, y1 + d), (cx, y1 + d), (x0 - d, y1 + d), (x0 - 2.2 * d, cy), (x1 + 2.2 * d, cy),
        (cx, y0 - 2.2 * d), (cx, y1 + 2.2 * d),
    ]
    W, H = info.width, info.height
    best, bcost = None, math.inf
    for i, (px, py) in enumerate(cands):
        px, py = int(px), int(py)
        if px - r < 0 or py - r < 0 or px + r >= W or py + r >= H:
            continue
        cost = _ink_in(integral, px - r, py - r, px + r, py + r) * 1.0
        for tx, ty in taken:
            if (tx - px) ** 2 + (ty - py) ** 2 < (2.2 * r) ** 2:
                cost += 5000
        for ob in boxes:
            if ob is b:
                continue
            if ob[0] - r < px < ob[2] + r and ob[1] - r < py < ob[3] + r:
                cost += 3000
        cost += i * 3  # prefer the left of the text, as inspectors usually draw them
        if cost < bcost:
            best, bcost = (px, py), cost
    return best or (int(max(r, x0 - d)), int(max(r, cy)))
