"""Draw balloons on a sheet, and draw the residue page with what the audit found."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .balloon import balloon_radius
from .types import AuditPass, Characteristic, SheetInfo

BLUE = (25, 90, 200)
RED = (215, 30, 30)
GREY = (150, 150, 150)


def font(size: int) -> ImageFont.ImageFont:
    for path in ("DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def ballooned(gray: np.ndarray, chars: list[Characteristic], info: SheetInfo) -> Image.Image:
    img = Image.fromarray(gray).convert("RGB")
    d = ImageDraw.Draw(img)
    r = balloon_radius(info)
    f = font(int(r * 0.95))
    lw = max(2, r // 8)
    for c in chars:
        if c.ignored or not c.balloon_xy:
            continue
        color = RED if c.found_in.startswith("audit") else BLUE
        bx, by = c.balloon_xy
        tx = min(max(bx, c.box[0]), c.box[2])
        ty = min(max(by, c.box[1]), c.box[3])
        d.line([(bx, by), (tx, ty)], fill=color, width=lw)
        d.rectangle(c.box, outline=color, width=max(1, lw // 2))
        d.ellipse([bx - r, by - r, bx + r, by + r], fill=(255, 255, 255), outline=color, width=lw)
        d.text((bx, by), c.number, fill=color, font=f, anchor="mm")
    return img


def residue_view(residue: np.ndarray, chars: list[Characteristic], audit: list[AuditPass]) -> Image.Image:
    img = Image.fromarray(residue).convert("RGB")
    d = ImageDraw.Draw(img)
    for p in audit:
        for fnd in p.finds:
            if fnd.verdict == "new":
                d.rectangle(fnd.box, outline=RED, width=6)
            elif fnd.verdict == "not_requirement":
                d.rectangle(fnd.box, outline=GREY, width=2)
    return img
