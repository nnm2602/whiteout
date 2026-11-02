"""Planted-requirement test: how many planted items the first read finds, and how many the whiteout search adds.

uv run python eval/run_seeded.py --sheets eval/seeded_sheets.json --per-sheet 8 --out eval/results/seeded
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from whiteout import ingest
from whiteout.geometry import SheetInk, overlap_min, text_sim
from whiteout.model_client import ModelConfig, VisionClient
from whiteout.pipeline import process_sheet

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed import plant  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def text_height(gray: np.ndarray, dpi: float) -> int:
    ink = SheetInk(gray, dpi, True)
    hs = sorted(b[3] - b[1] for b in ink.blocks[1:] if 6 <= b[3] - b[1] <= 4 * ink.gh)
    return int(np.median(hs)) if hs else ink.gh


async def one(client: VisionClient, src: dict, n: int, out: Path, idx: int) -> dict:
    sheets = ingest.load(ROOT / src["file"], max_pages=src.get("page", 1))
    sh = sheets[src.get("page", 1) - 1]
    gray = np.array(sh.image)
    gh = text_height(gray, sh.dpi)
    planted, items = plant(gray, gh, n, seed=1000 + idx)
    sh.image = Image.fromarray(planted)
    sh.words = []  # planted text has no text layer; read only from pixels
    name = f"{idx:02d}_{Path(src['file']).stem}_p{src.get('page', 1)}"
    sh.image.save(out / f"{name}.png")
    res = await process_sheet(client, sh, 0)
    chars = [c for c in res.chars if not c.ignored]
    for it in items:
        hit = None
        for c in chars:
            if overlap_min(tuple(it["box"]), c.box) > 0.3 and text_sim(it["text"].replace("DIA ", ""), c.text) >= 50:
                hit = c
                break
        it["found"] = hit is not None
        it["found_in"] = hit.found_in if hit else ""
        it["read_as"] = hit.text if hit else ""
    resid = (res.residue < 235) if res.residue is not None else (planted < 235)  # light ink counts (faint copies)
    ink0 = planted < 235
    for it in items:
        x0, y0, x1, y1 = it["box"]
        o = ink0[y0:y1, x0:x1].sum()
        it["visible_on_erased_page"] = bool(o) and float(resid[y0:y1, x0:x1].sum() / o) > 0.5
    Image.fromarray(res.residue if res.residue is not None else planted).save(out / f"{name}_erased.png")
    seeds = [tuple(i["box"]) for i in items]
    other_new = [{"text": f.text, "box": f.box, "pass": p.pass_no} for p in res.audit for f in p.finds
                 if f.verdict == "new" and not any(overlap_min(f.box, s) > 0.3 for s in seeds)]
    rec = {"name": name, "source": src, "glyph_h": gh, "items": items, "chars": len(chars),
           "audit_new_not_planted": other_new, "passes": len(res.audit), "timings": res.timings,
           "audit_finds": [{"pass": p.pass_no, **f.model_dump()} for p in res.audit for f in p.finds]}
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
    return rec


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", required=True)
    ap.add_argument("--per-sheet", type=int, default=8)
    ap.add_argument("--out", default="eval/results/seeded")
    ap.add_argument("--effort", default="medium")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    srcs = json.loads(Path(a.sheets).read_text())
    client = VisionClient(ModelConfig(effort=a.effort, concurrency=12))
    recs = []
    for i in range(0, len(srcs), 3):  # three sheets at a time
        recs += await asyncio.gather(*(one(client, s, a.per_sheet, out, i + j) for j, s in enumerate(srcs[i:i + 3])))
        print(f"{len(recs)}/{len(srcs)} sheets, usage {client.usage.as_dict()}", flush=True)
    items = [it for r in recs for it in r["items"]]
    n = len(items)
    first = sum(1 for it in items if it["found_in"] == "read")
    final = sum(1 for it in items if it["found"])
    by = {}
    for cond in sorted({it["condition"] for it in items}):
        g = [it for it in items if it["condition"] == cond]
        by[cond] = {"n": len(g), "first_read": sum(1 for it in g if it["found_in"] == "read"),
                    "after_whiteout": sum(1 for it in g if it["found"])}
    summary = {
        "sheets": len(recs), "planted": n,
        "first_read": first, "first_read_rate": first / n if n else 0, "first_read_ci95": wilson(first, n),
        "after_whiteout": final, "after_whiteout_rate": final / n if n else 0, "after_whiteout_ci95": wilson(final, n),
        "caught_by_whiteout": final - first,
        "missed_visible_on_erased_page": sum(1 for it in items if not it["found"] and it.get("visible_on_erased_page")),
        "found_but_still_visible": sum(1 for it in items if it["found"] and it.get("visible_on_erased_page")),
        "still_missed": [{"sheet": r["name"], **it} for r in recs for it in r["items"] if not it["found"]],
        "by_condition": by,
        "audit_new_not_planted": [{"sheet": r["name"], **x} for r in recs for x in r["audit_new_not_planted"]],
        "usage": client.usage.as_dict(),
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("still_missed", "audit_new_not_planted")}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
