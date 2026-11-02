"""Redraw test: turn old Apollo scans into clean vector sheets and measure what the redraw keeps.

Per sheet:
- legibility: plain Tesseract word recovery against the answer-key requirement texts, on the original scan and on
  the redraw: (1) one pass over the whole sheet as geometry.ocr_words does it, words counted only near the key
  item's box; (2) Tesseract on a crop around each key item, the same crop on both images;
- key text typeset: key items whose full text appears, re-typeset, on the redraw at the key's place;
- faithfulness: share of the scan's line ink (text removed) within 3 px of a traced line, and share of traced line
  pixels within 3 px of the scan's line ink (from redraw stats);
- text placed against text read.

Requirements already read by Whiteout are reused: the sample job for LDW270M10003, and for the other sheets the
read stage of the pipeline rebuilt from cached model calls (no new spend; skipped if not cached).

uv run python eval/run_redraw.py --out eval/results/redraw
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

from whiteout import ingest
from whiteout.geometry import SheetInk, merge, norm_text, ocr_norm, ocr_words, overlap_min
from whiteout.model_client import ModelConfig, VisionClient
from whiteout.pipeline import overview, read_tiles, to_char
from whiteout import prompts
from whiteout.model_client import png_data_url
from whiteout.redraw import TEXT_PROMPT, TEXT_SCHEMA, redraw, render_pdf
from whiteout.tiling import grid

ROOT = Path(__file__).resolve().parents[1]
SHEETS = ["apollo_b503_n0006", "apollo_b503_n0013", "apollo_b502_n0014", "apollo_b503_n0024", "apollo_b503_n0057",
          "apollo_b503_n0128"]
SAMPLE_JOB = {"apollo_b503_n0006": ROOT / "samples/jobs/e9f770195cba/job.json"}


class CacheOnly(VisionClient):
    """Answers only from the call cache; an uncached call fails instead of spending."""

    async def _meta(self, *a, **k):
        raise RuntimeError("not cached")


async def cached_job(sid: str) -> dict | None:
    """The pipeline's first read of a sheet, rebuilt from cached calls, in job.json shape."""
    sh = ingest.load(ROOT / f"data/raw/apollo/{sid}.jpg")[0]
    c = CacheOnly(ModelConfig())
    try:
        info = await overview(c, sh, 0)
        gray = np.array(sh.image)
        ink = SheetInk(gray, sh.dpi, sh.dpi_trusted)
        raws = await read_tiles(c, sh.image, ink, grid(*sh.image.size), prompts.READ_PROMPT, prompts.READ_SCHEMA,
                                None, "read")
    except RuntimeError:
        return None
    kept = merge(raws, ink)
    chars = [to_char(r, f"s0c{i}", info, "read").model_dump() for i, r in enumerate(kept)]
    return {"sheets": [{"info": info.model_dump(), "chars": chars}]}


def crop_words(gray: np.ndarray, b: list[int], pad: int) -> list[str]:
    """Tesseract on one crop around a key item (2x upscale, uniform block mode)."""
    exe = shutil.which("tesseract")
    H, W = gray.shape
    x0, y0, x1, y1 = max(0, b[0] - pad), max(0, b[1] - pad), min(W, b[2] + pad), min(H, b[3] + pad)
    if not exe or x1 <= x0 or y1 <= y0:
        return []
    c = cv2.resize(gray[y0:y1, x0:x1], ((x1 - x0) * 2, (y1 - y0) * 2), interpolation=cv2.INTER_CUBIC)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        cv2.imwrite(f.name, c)
        txt = subprocess.run([exe, f.name, "stdout", "--psm", "6"], capture_output=True, text=True, timeout=60).stdout
    return [w for w in re.split(r"[\s|]+", txt) if w]


def match_tokens(toks: list[str], words: list[str]) -> int:
    pool = [ocr_norm(w) for w in words]
    hit = 0
    for t in toks:
        for i, w in enumerate(pool):
            if w == t or (len(t) >= 4 and t in w):
                hit += 1
                pool.pop(i)
                break
    return hit


def crop_recall(key: dict, gray: np.ndarray, pad: int) -> tuple[int, int]:
    """(recovered words, key items with every word recovered) reading each key item's crop on its own."""
    got = full = 0
    for it in key["items"]:
        toks = tokens(it["text"])
        if not it.get("box") or not toks:
            continue
        hit = match_tokens(toks, crop_words(gray, it["box"], pad))
        got += hit
        full += hit == len(toks)
    return got, full


def read_cost(client: VisionClient, image: Path, effort: str = "low") -> float:
    """What the all-text read of a sheet cost when its calls were made (from the usage stored with each cached
    call), so a rerun from the cache still reports the real price."""
    import hashlib

    img = ingest.load(image)[0].image
    usd = 0.0
    for t in grid(*img.size):
        payload = {"backend": client.cfg.backend, "model": client.cfg.model, "effort": effort, "prompt": TEXT_PROMPT,
                   "schema": TEXT_SCHEMA, "images": [hashlib.sha256(png_data_url(img.crop(t.box)).encode()).hexdigest()]}
        cp = client._cache_path(payload)
        if cp.exists():
            usd += json.loads(cp.read_text()).get("usage", {}).get("usd", 0.0)
    return round(usd, 4)


def tokens(text: str) -> list[str]:
    out = []
    for w in re.split(r"[\s|/]+", text):
        t = ocr_norm(w)
        if len(t) >= 2 and re.search(r"[0-9A-Z]", t):
            out.append(t)
    return out


def word_recall(key: dict, words: list, pad: int) -> tuple[int, int, int]:
    """(key words, recovered words, key items with every word recovered), words counted only near each item."""
    n = got = full = 0
    for it in key["items"]:
        b = it.get("box")
        toks = tokens(it["text"])
        if not b or not toks:
            continue
        x0, y0, x1, y1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
        hit = match_tokens(toks, [t for wb, t in words
                                  if x0 <= (wb[0] + wb[2]) / 2 <= x1 and y0 <= (wb[1] + wb[3]) / 2 <= y1])
        n += len(toks)
        got += hit
        full += hit == len(toks)
    return n, got, full


def typeset(key: dict, texts: list[dict], pad: int) -> tuple[int, int, int]:
    """(key items, items whose text is re-typeset exactly, items at 90% similarity or better)."""
    n = exact = close = 0
    for it in key["items"]:
        b = it.get("box")
        if not b:
            continue
        n += 1
        zone = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
        inside = [t for t in texts if overlap_min(tuple(t["box"]), zone) > 0.6]
        inside.sort(key=lambda t: (round(t["box"][1] / max(1, pad)), t["box"][0]))
        want = norm_text(it["text"].replace("|", ""))
        got = norm_text(" ".join(t["text"] for t in inside))
        if not want:
            continue
        if want in got:
            exact += 1
            close += 1
        elif got and fuzz.partial_ratio(want, got) >= 90:
            close += 1
    return n, exact, close


async def one(client: VisionClient, sid: str, out: Path) -> dict:
    key = json.loads((ROOT / f"eval/keys/{sid}__p1.json").read_text())
    image = ROOT / f"data/raw/apollo/{sid}.jpg"
    if sid in SAMPLE_JOB:
        job, job_src = json.loads(SAMPLE_JOB[sid].read_text()), "sample job"
    else:
        job = await cached_job(sid)
        job_src = "cached first read" if job else "none"
    t0 = time.time()
    res = await redraw(image, job, client, out / sid)
    secs = time.time() - t0
    st = res["stats"]
    sh = ingest.load(image)[0]
    gray = np.array(sh.image)
    H, W = gray.shape
    rx0, ry0, rx1, ry1 = res["region"]
    full = np.full((H, W), 255, np.uint8)
    r = render_pdf(Path(res["pdf"]), rx1 - rx0)
    r = cv2.resize(r, (rx1 - rx0, ry1 - ry0), interpolation=cv2.INTER_AREA) if r.shape != (ry1 - ry0, rx1 - rx0) else r
    full[ry0:ry1, rx0:rx1] = r
    pad = 15
    w_orig, w_new = ocr_words(gray), ocr_words(full)
    n, got_o, full_o = word_recall(key, w_orig, pad)
    _, got_n, full_n = word_recall(key, w_new, pad)
    cg_o, cf_o = crop_recall(key, gray, pad)
    cg_n, cf_n = crop_recall(key, full, pad)
    texts = json.loads(Path(res["texts"]).read_text())
    kn, kex, kcl = typeset(key, texts, pad)
    items = sum(1 for it in key["items"] if it.get("box") and tokens(it["text"]))
    # the redraw folder keeps what a reader needs: PNG previews, SVG, PDF, the placed text and the counts
    Path(res["lines"]).unlink(missing_ok=True)
    return {
        "sheet": sid, "job": job_src, "seconds": round(secs, 1), "usd_this_run": st["usd"],
        "usd": read_cost(client, image),
        "legibility": {"key_words": n, "original_words": got_o, "redraw_words": got_n,
                       "original_word_recall": round(got_o / max(1, n), 4), "redraw_word_recall": round(got_n / max(1, n), 4),
                       "key_items": items, "original_items_all_words": full_o, "redraw_items_all_words": full_n,
                       "crop_original_words": cg_o, "crop_redraw_words": cg_n,
                       "crop_original_word_recall": round(cg_o / max(1, n), 4),
                       "crop_redraw_word_recall": round(cg_n / max(1, n), 4),
                       "crop_original_items_all_words": cf_o, "crop_redraw_items_all_words": cf_n},
        "key_text_typeset": {"key_items": kn, "exact": kex, "close_90": kcl},
        "faithfulness": st["faithfulness"],
        "text": st["text"],
        "lines": st["lines"],
        "region": res["region"],
    }


async def main_async(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    client = VisionClient(ModelConfig())
    recs = []
    for sid in a.sheets:
        rec = await one(client, sid, out)
        print(json.dumps({k: rec[k] for k in ("sheet", "legibility", "key_text_typeset", "faithfulness", "usd")}),
              flush=True)
        recs.append(rec)

    def tot(path: list[str]) -> float:
        v = 0
        for r in recs:
            x = r
            for p in path:
                x = x[p]
            v += x
        return v

    kw = tot(["legibility", "key_words"])
    ki = tot(["legibility", "key_items"])
    summary = {
        "what": "Apollo LM scans (1967) redrawn as vector sheets: line work traced in code, every text line read by "
                "Muse Spark 1.3 (effort low) and re-typeset. Word counts are plain Tesseract words near each "
                "answer-key requirement: one whole-sheet pass (sparse text mode, as geometry.ocr_words), and one "
                "pass per key item crop. Faithfulness is within 3 px.",
        "overall": {
            "sheets": len(recs),
            "key_words": kw,
            "original_word_recall": round(tot(["legibility", "original_words"]) / max(1, kw), 4),
            "redraw_word_recall": round(tot(["legibility", "redraw_words"]) / max(1, kw), 4),
            "crop_original_word_recall": round(tot(["legibility", "crop_original_words"]) / max(1, kw), 4),
            "crop_redraw_word_recall": round(tot(["legibility", "crop_redraw_words"]) / max(1, kw), 4),
            "key_items": ki,
            "crop_original_items_all_words": round(tot(["legibility", "crop_original_items_all_words"]) / max(1, ki), 4),
            "crop_redraw_items_all_words": round(tot(["legibility", "crop_redraw_items_all_words"]) / max(1, ki), 4),
            "original_items_all_words": round(tot(["legibility", "original_items_all_words"]) / max(1, ki), 4),
            "redraw_items_all_words": round(tot(["legibility", "redraw_items_all_words"]) / max(1, ki), 4),
            "key_text_typeset_exact": round(tot(["key_text_typeset", "exact"]) / max(1, tot(["key_text_typeset", "key_items"])), 4),
            "key_text_typeset_close_90": round(tot(["key_text_typeset", "close_90"]) / max(1, tot(["key_text_typeset", "key_items"])), 4),
            "line_ink_covered_mean": round(float(np.mean([r["faithfulness"]["line_ink_covered"] for r in recs])), 4),
            "drawn_on_line_ink_mean": round(float(np.mean([r["faithfulness"]["drawn_on_line_ink"] for r in recs])), 4),
            "text_lines_read": tot(["text", "lines_read"]),
            "text_placed": tot(["text", "placed"]),
            "text_placed_on_ink": tot(["text", "placed_snapped"]),
            "usd": round(tot(["usd"]), 4),
        },
        "sheets": recs,
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["overall"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "eval/results/redraw"))
    ap.add_argument("--sheets", nargs="*", default=SHEETS)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
