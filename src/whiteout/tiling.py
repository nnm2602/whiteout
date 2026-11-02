"""Cut a sheet into overlapping tiles that Muse Spark reads at full resolution.

Spark downscales anything above about 3.2 MP (measured Oct 27 2026), so tiles stay at or under 1775 x 1175 px.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TILE_W = 1775
TILE_H = 1175
OVERLAP = 300


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    w: int
    h: int
    row: int
    col: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def to_sheet(self, b: list[int] | tuple[int, ...]) -> tuple[int, int, int, int]:
        """Map a 0-1000 grid box on this tile to sheet pixels (clamped)."""
        x0, y0, x1, y1 = (max(0, min(1000, int(v))) for v in b)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return (
            self.x + round(x0 * self.w / 1000),
            self.y + round(y0 * self.h / 1000),
            self.x + round(x1 * self.w / 1000),
            self.y + round(y1 * self.h / 1000),
        )

    def edge_margin(self, b: tuple[int, int, int, int], sheet_w: int, sheet_h: int) -> int:
        """Distance from a sheet box to the nearest tile edge that is not also a sheet edge."""
        m = []
        if self.x > 0:
            m.append(b[0] - self.x)
        if self.y > 0:
            m.append(b[1] - self.y)
        if self.x + self.w < sheet_w:
            m.append(self.x + self.w - b[2])
        if self.y + self.h < sheet_h:
            m.append(self.y + self.h - b[3])
        return min(m) if m else 10**6


def grid(width: int, height: int, tw: int = TILE_W, th: int = TILE_H, overlap: int = OVERLAP, shift: bool = False) -> list[Tile]:
    """Even grid of tiles covering the sheet. shift=True offsets the grid by half a step (used by the audit pass)."""
    tw, th = min(tw, width), min(th, height)

    def starts(total: int, size: int) -> list[int]:
        if total <= size:
            return [0]
        n = max(1, math.ceil((total - overlap) / (size - overlap)))
        if n == 1:
            return [0]
        s = [round(i * (total - size) / (n - 1)) for i in range(n)]
        if shift:
            step = (total - size) / (n - 1)
            mids = [round(v + step / 2) for v in s[:-1]]
            s = sorted(set([0] + mids + [total - size]))
        return s

    xs, ys = starts(width, tw), starts(height, th)
    return [Tile(x, y, tw, th, r, c) for r, y in enumerate(ys) for c, x in enumerate(xs)]
