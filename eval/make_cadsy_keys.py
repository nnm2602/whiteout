"""Convert Cadsy official labels (data/raw/cadsy/part_*_labels.json) into eval answer keys.

Kept from the labels: dimension, callout, feature_control_frame (one item per frame row) and the title-block
"General tolerance" value (ISO sheets). Dropped: border zone labels, title block labels and admin values,
projection symbols, datum definitions, view headings and view references.

Boxes in the labels are normalised to the PNG size; the PNG loads through whiteout.ingest at its own size, so
pixels = normalised x width (or height). Run from the whiteout folder:  uv run python eval/make_cadsy_keys.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from whiteout.ingest import load

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cadsy"
KEYS = ROOT / "eval/keys"

FCF_SYMBOL = {
    "cylindricity": "⌭", "perpendicularity": "⊥", "parallelism": "∥", "flatness": "⏥", "straightness": "⏤",
    "circularity": "○", "position": "⌖", "concentricity": "◎", "symmetry": "⌯", "angularity": "∠",
    "profile_of_a_surface": "⌓", "profile_of_a_line": "⌒", "circular_runout": "↗", "total_runout": "⌰",
}
COUNT = re.compile(r"^\s*(\d+)\s*[X×]\s*")


def multiplier(text: str) -> int:
    m = COUNT.match(text)
    return int(m.group(1)) if m else 1


def callout_kind(text: str) -> str:
    body = COUNT.sub("", text).strip()
    if body.startswith("R"):
        return "radius"
    if re.match(r"C\d", body):
        return "chamfer"
    if re.match(r"M\d", body):
        return "thread"
    if body.startswith("⌀") and re.search(r"THRU|↧|⌴|⌵|PCD|B\.C\.|DEEP", body):
        return "hole_callout"
    return "dimension"  # plain ⌀80, ⌀20 ±0.1, H 6,03 (height), 30 X 40 X 4 DEEP pockets


def convert(png: Path) -> dict:
    labels = json.loads(png.with_name(png.stem + "_labels.json").read_text())["annotations"]
    sheet = load(png)[0]
    W, H = sheet.image.size

    def px(b):
        return [round(b[0] * W), round(b[1] * H), round(b[2] * W), round(b[3] * H)]

    callouts = [a for a in labels if a["type"] == "callout"]
    items, doubts = [], []
    for a in labels:
        t, text = a["type"], a["text"].strip()
        if t == "dimension":
            items.append({"text": text, "kind": "dimension", "multiplier": multiplier(text), "box": px(a["box"])})
        elif t == "callout":
            items.append({"text": text, "kind": callout_kind(text), "multiplier": multiplier(text), "box": px(a["box"])})
        elif t == "feature_control_frame":
            rows = [r for r in text.split("\n") if r.strip()]
            x0, y0, x1, y1 = a["box"]
            # A frame stacked right under an "nX" callout applies to all n features.
            mult = 1
            for c in callouts:
                cx0, _, cx1, cy1 = c["box"]
                if abs(cy1 - y0) < 0.01 and abs(cx0 - x0) < 0.01 and multiplier(c["text"]) > 1:
                    mult = multiplier(c["text"])
            for i, r in enumerate(rows):
                cells = r.split("|")
                sym = FCF_SYMBOL.get(cells[0].strip().lower())
                if sym is None:
                    doubts.append(f"unknown frame characteristic {cells[0]!r}, kept as text")
                    sym = cells[0]
                ry0 = y0 + (y1 - y0) * i / len(rows)
                ry1 = y0 + (y1 - y0) * (i + 1) / len(rows)
                items.append({"text": "|".join([sym] + cells[1:]), "kind": "gdt_frame", "multiplier": mult,
                              "box": px([x0, ry0, x1, ry1])})
    # General tolerance: the title block value that sits in the "General tolerance" cell.
    tbl = [a for a in labels if a["type"] == "title_block_label" and a["text"] == "General tolerance"]
    for lab in tbl:
        lx0, ly0, lx1, ly1 = lab["box"]
        best = None
        for v in labels:
            if v["type"] != "title_block_value":
                continue
            vx0, vy0, vx1, vy1 = v["box"]
            # value is printed in the same cell row, just above the label line
            if 0 < ly0 - vy1 < 0.03 and vx0 > lx0:
                d = ly0 - vy1
                if best is None or d < best[0]:
                    best = (d, v)
        if best:
            v = best[1]
            items.append({"text": f"General tolerance {v['text']}", "kind": "note", "multiplier": 1,
                          "box": px(v["box"]), "title_block": True})
    items.sort(key=lambda it: (it["box"][1] // 40, it["box"][0]))
    for i, it in enumerate(items, 1):
        it["key"] = f"k{i}"
    items = [{"key": it.pop("key"), **it} for it in items]
    std = png.stem.split("_")[-1]
    return {
        "id": f"cadsy_{png.stem}", "page": 1, "width": W, "height": H, "dpi": sheet.dpi,
        "file": str(png.relative_to(ROOT)), "source": "official", "standard": std.upper(),
        "doubts": doubts + [
            "official Cadsy labels converted by script; frame rows split from one labelled frame box into equal "
            "row heights; kinds inferred from the callout text",
            "dpi is the loader default for PNGs without dpi metadata (300), not a physical scale",
        ],
        "items": items,
    }


def main() -> None:
    pngs = sorted(RAW.glob("part_*_*.png"))[:25]
    tot = 0
    for p in pngs:
        k = convert(p)
        out = KEYS / f"{k['id']}__p1.json"
        out.write_text(json.dumps(k, indent=1, ensure_ascii=False) + "\n")
        tot += len(k["items"])
        print(out.name, len(k["items"]))
    print(len(pngs), "keys", tot, "items")


if __name__ == "__main__":
    main()
