"""Proofread a redraw against the scan it came from.

Muse Spark sees the same region of the scan and of the redraw side by side and lists every text that differs: a wrong
digit, a word set one row off, text missing or added. Each difference is marked on the redraw, so a person checks those
places against the scan before anyone uses the redrawn sheet.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .model_client import VisionClient
from .pipeline import grid_box
from .prompts import _obj
from .tiling import grid

PROOF_SCHEMA = _obj({"differences": {"type": "array", "items": _obj({
    "box": {"type": "array", "items": {"type": "integer"}},
    "redraw": {"type": "string"},
    "scan": {"type": "string"},
    "kind": {"type": "string", "enum": ["wrong text", "missing", "added", "moved"]},
})}})

PROOF_PROMPT = (
    "The first image is a region of a scanned engineering drawing. The second image is the same region redrawn by a "
    "program, at the same scale and position. Proofread the redraw against the scan, text only (ignore line weight, "
    "font, small shifts and the drawing lines). List every place where the text in the redraw does not say exactly "
    "what the scan says: a wrong digit or letter, a value set in the wrong row or cell, scan text missing from the "
    "redraw, text in the redraw that the scan does not have, or garbled or overlapping text. Check every number in "
    "tables cell by cell. box: [x_min, y_min, x_max, y_max] on a 0-1000 grid of the second image, around the place in "
    "the redraw. redraw: what the redraw says there (empty if missing). scan: what the scan says there. Return an empty "
    "list if the text matches everywhere."
)


def _tiles(region: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = region
    return [(x0 + t.box[0], y0 + t.box[1], x0 + t.box[2], y0 + t.box[3]) for t in grid(x1 - x0, y1 - y0)]


async def proofread(scan: np.ndarray, redraw: np.ndarray, region, client: VisionClient, effort: str = "medium") -> list[dict]:
    """scan: the sheet in grayscale; redraw: the redraw at sheet resolution covering region. Boxes come back in sheet
    pixels."""
    rx0, ry0 = region[0], region[1]

    async def one(b):
        x0, y0, x1, y1 = b
        a = Image.fromarray(scan[y0:y1, x0:x1])
        r = Image.fromarray(redraw[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0])
        out = await client.ask_json([a, r], PROOF_PROMPT, PROOF_SCHEMA, name="proofread", effort=effort)
        found = []
        for d in out.get("differences", []):
            g = grid_box(d["box"], x1 - x0, y1 - y0)
            if g is None:
                continue
            found.append({**d, "box": [x0 + g[0], y0 + g[1], x0 + g[2], y0 + g[3]]})
        return found

    parts = await asyncio.gather(*(one(b) for b in _tiles(tuple(region))))
    diffs = [d for p in parts for d in p]
    # tiles overlap: keep one of two differences that cover the same place
    keep: list[dict] = []
    for d in sorted(diffs, key=lambda d: (d["box"][1], d["box"][0])):
        if any(_iou(d["box"], k["box"]) > 0.3 for k in keep):
            continue
        keep.append(d)
    return keep


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


def mark(redraw_png: Path, region, diffs: list[dict], out: Path) -> None:
    """Red boxes on the redraw PNG (which may be scaled down from sheet resolution)."""
    im = cv2.imread(str(redraw_png), cv2.IMREAD_COLOR)
    k = im.shape[1] / (region[2] - region[0])
    for d in diffs:
        x0, y0, x1, y1 = d["box"]
        p0 = (int((x0 - region[0]) * k) - 4, int((y0 - region[1]) * k) - 4)
        p1 = (int((x1 - region[0]) * k) + 4, int((y1 - region[1]) * k) + 4)
        cv2.rectangle(im, p0, p1, (30, 38, 215), max(2, int(3 * k)))
    cv2.imwrite(str(out), im)


async def proofread_dir(scan_gray: np.ndarray, out_dir: Path, client: VisionClient) -> dict:
    """Run on a redraw folder written by redraw.redraw (needs stats.json and redraw.pdf)."""
    from .redraw import render_pdf

    stats = json.loads((out_dir / "stats.json").read_text())
    region = stats["region"]
    full = render_pdf(out_dir / "redraw.pdf", region[2] - region[0])
    h, w = region[3] - region[1], region[2] - region[0]
    if full.shape != (h, w):
        full = cv2.resize(full, (w, h), interpolation=cv2.INTER_AREA)
    diffs = await proofread(scan_gray, full, region, client)
    (out_dir / "proof.json").write_text(json.dumps(diffs, indent=1))
    mark(out_dir / "redraw.png", region, diffs, out_dir / "redraw_checked.png")
    return {"differences": diffs, "checked_png": str(out_dir / "redraw_checked.png")}
