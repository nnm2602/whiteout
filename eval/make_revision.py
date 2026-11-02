"""Make a known revision of NIST D2MI 827-9999-904 for the revision-delta demo and test.

Changes on sheet 2 (all known exactly): 2.375 -> 2.385, 4X .768 -> 4X .770, .268 -> .272, .134 removed,
a new note BREAK SHARP EDGES .005-.015 added, and the title block revision A -> B.
"""

import json
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/nist_d2mi/827-9999-904 rev c.pdf"
OUT = ROOT / "samples/nist-904-revB.pdf"
S = 72 / 200  # key pixels (200 dpi) to PDF points

EDITS = [  # (box in 200-dpi pixels, new text or None to remove)
    ((803, 878, 900, 912), "2.385"),
    ((2964, 520, 3094, 554), "4X .770"),
    ((408, 1218, 490, 1252), ".272"),
    ((2831, 1251, 2914, 1282), None),
]
ADD = ((2380, 1960), "BREAK SHARP EDGES .005-.015")
REV_BOX = (3240, 2060, 3275, 2098)


def main() -> None:
    doc = fitz.open(SRC)
    page = doc[1]
    for (x0, y0, x1, y1), text in EDITS:
        r = fitz.Rect(x0 * S + 1.2, y0 * S + 1.2, x1 * S - 1.2, y1 * S - 1.2)
        page.draw_rect(r, color=None, fill=(1, 1, 1))
        if text:
            page.insert_text((r.x0 + 1, r.y1 - 2.3), text, fontsize=8.6, fontname="helv", color=(0, 0, 0))
    (x, y), text = ADD
    page.insert_text((x * S, y * S), text, fontsize=8.6, fontname="helv")
    r = fitz.Rect(*(v * S for v in REV_BOX))
    page.draw_rect(r, color=None, fill=(1, 1, 1))
    page.insert_text((r.x0 + 1, r.y1 - 2), "B", fontsize=10, fontname="helv")
    OUT.parent.mkdir(exist_ok=True)
    doc.save(OUT, garbage=3, deflate=True)
    (ROOT / "samples/nist-904-revB.changes.json").write_text(json.dumps({
        "changed": ["2.375 -> 2.385", "4X .768 -> 4X .770", ".268 -> .272"], "removed": [".134"],
        "added": [text], "title_block": "rev A -> B"}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
