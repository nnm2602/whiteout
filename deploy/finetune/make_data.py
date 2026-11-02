"""Training data for the Glimmer fine-tune: Muse Spark's cached tile reads, cut into smaller crops.

Every first-read tile of a training sheet (answer-key sheets minus the 6 held-out sheets and every page of their
drawings) is read from the call cache, so this costs nothing. Each 1775 x 1175 tile is cut into crops of
--crop size; an item is kept when its box lies inside the crop and its box is moved onto the crop's 0-1000 grid.
Items cut by the crop edge are left out, as the READ prompt asks ("text that is cut off at the image edge").

  PYTHONPATH=src .venv/bin/python deploy/finetune/make_data.py --crop 1344x672 --out .cache/ft/data_1344x672

Writes train.jsonl ({"image", "prompt", "answer", "sheet", "items"}) and images/*.png.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from whiteout import prompts
from whiteout.tiling import grid

from spark_reads import ROOT, load_sheet, sheet_reads, spark_client, training_keys

ITEM_ORDER = list(prompts.ITEM["properties"])
REQ_ORDER = list(prompts.REQUIREMENT["properties"])
SLACK = 6  # px a model box may poke past the crop edge and still count as inside


def offline_prompt(prompt: str, schema: dict) -> str:
    """Exactly the text the offline client sends (model_client.VisionClient._openai)."""
    return prompt + "\n\nReply with JSON only, matching this schema:\n" + json.dumps(schema)


def ordered(item: dict, box: list[int]) -> dict:
    out = {}
    for k in ITEM_ORDER:
        if k == "box":
            out[k] = box
        elif k == "requirements":
            out[k] = [{q: r.get(q, [] if q in ("datums", "modifiers") else "") for q in REQ_ORDER}
                      for r in item.get("requirements") or []]
        else:
            out[k] = item.get(k, 1 if k == "multiplier" else "")
    return out


def crop_items(items: list[dict], tile_wh: tuple[int, int], crop: tuple[int, int, int, int]) -> list[dict] | None:
    """Items inside the crop with boxes on the crop's 0-1000 grid; None if the crop cuts an item."""
    tw, th = tile_wh
    cx0, cy0, cx1, cy1 = crop
    cw, ch = cx1 - cx0, cy1 - cy0
    out = []
    for it in items:
        b = [max(0, min(1000, int(v))) for v in (list(it.get("box") or []) + [0, 0, 0, 0])[:4]]
        x0, y0, x1, y1 = b[0] * tw / 1000, b[1] * th / 1000, b[2] * tw / 1000, b[3] * th / 1000
        if x1 <= x0 or y1 <= y0:
            continue
        if x1 <= cx0 or x0 >= cx1 or y1 <= cy0 or y0 >= cy1:
            continue  # outside
        if x0 < cx0 - SLACK or y0 < cy0 - SLACK or x1 > cx1 + SLACK or y1 > cy1 + SLACK:
            continue  # cut by the crop edge: the READ prompt says leave it out
        nb = [round(max(0, min(1000, (x0 - cx0) * 1000 / cw))), round(max(0, min(1000, (y0 - cy0) * 1000 / ch))),
              round(max(0, min(1000, (x1 - cx0) * 1000 / cw))), round(max(0, min(1000, (y1 - cy0) * 1000 / ch)))]
        out.append(ordered(it, nb))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", default="1344x672")
    ap.add_argument("--overlap", type=int, default=300)
    ap.add_argument("--empty-keep", type=float, default=0.1)
    ap.add_argument("--out", default=".cache/ft/data_1344x672")
    a = ap.parse_args()
    cw, ch = (int(v) for v in a.crop.lower().split("x"))
    out = ROOT / a.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    client = spark_client()
    text = offline_prompt(prompts.READ_PROMPT, prompts.READ_SCHEMA)
    rows, sheets, skipped = [], 0, 0
    for kp in training_keys():
        img = load_sheet(kp)
        reads = await sheet_reads(client, img)
        if not any(r is not None for _, r in reads):
            skipped += 1
            continue
        sheets += 1
        for ti, (tbox, res) in enumerate(reads):
            if res is None:
                continue
            tile = img.crop(tbox)
            tw, th = tile.size
            for c in grid(tw, th, cw, ch, a.overlap):
                items = crop_items(res.get("items", []), (tw, th), c.box)
                if not items and rng.random() > a.empty_keep:
                    continue
                name = f"{kp.stem}_t{ti}_{c.x}_{c.y}.png"
                tile.crop(c.box).save(out / "images" / name)
                rows.append({"image": f"images/{name}", "prompt": text,
                             "answer": json.dumps({"items": items}, ensure_ascii=False, separators=(",", ":")),
                             "sheet": kp.stem, "items": len(items)})
        print(f"{kp.stem}: {len(rows)} crops so far", flush=True)
    rng.shuffle(rows)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_items = sum(r["items"] for r in rows)
    lens = sorted(len(r["answer"]) for r in rows)
    meta = {"crop": [cw, ch], "overlap": a.overlap, "sheets": sheets, "sheets_without_cache": skipped,
            "crops": len(rows), "empty_crops": sum(1 for r in rows if not r["items"]), "items": n_items,
            "answer_chars_p50_p90_max": [lens[len(lens) // 2], lens[int(len(lens) * 0.9)], lens[-1]],
            "prompt_chars": len(text)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
