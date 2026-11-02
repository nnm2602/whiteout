"""Collect Muse Spark's cached tile reads for the answer-key sheets (no API calls: a cache miss is skipped).

Shared by make_data.py (training crops) and the stats run. Held-out sheets, and every other page of the same
drawings, are excluded.
"""

from __future__ import annotations

import asyncio
import glob
import json
from pathlib import Path

from PIL import Image

from whiteout import ingest, prompts
from whiteout.model_client import ModelConfig, VisionClient
from whiteout.tiling import grid

ROOT = Path(__file__).resolve().parents[2]
# The 6 sheets measured with stock Glimmer (eval/results/keys_glimmer/SUMMARY.json).
HELD_OUT_SHEETS = [r["sheet"] for r in json.loads((ROOT / "eval/results/keys_glimmer/SUMMARY.json").read_text())["rows"]]


def drawing_of(key_id: str) -> str:
    return key_id.split("__")[0]


class CacheOnly(VisionClient):
    """Answers only from the on-disk cache; never calls a model."""

    async def _meta(self, *a, **k):
        raise LookupError("not cached")

    async def _openai(self, *a, **k):
        raise LookupError("not cached")


def training_keys() -> list[Path]:
    held = {drawing_of(s) for s in HELD_OUT_SHEETS}
    out = []
    for kp in sorted(glob.glob(str(ROOT / "eval/keys/*.json"))):
        if drawing_of(Path(kp).stem) in held:
            continue
        out.append(Path(kp))
    return out


def load_sheet(kp: Path) -> Image.Image:
    import sys

    sys.path.insert(0, str(ROOT / "eval"))
    from run_keys import key_file

    key = json.loads(kp.read_text())
    page = key.get("page", 1)
    return ingest.load(key_file(key), max_pages=page)[page - 1].image


async def sheet_reads(client: VisionClient, img: Image.Image) -> list[tuple[tuple[int, int, int, int], dict | None]]:
    """(tile box in sheet pixels, Spark's READ answer or None when not cached) for every first-read tile."""
    W, H = img.size
    out = []
    for t in grid(W, H):
        try:
            res = await client.ask_json([img.crop(t.box)], prompts.READ_PROMPT, prompts.READ_SCHEMA, "read")
        except LookupError:
            res = None
        out.append((t.box, res))
    return out


def spark_client() -> CacheOnly:
    return CacheOnly(ModelConfig(effort="medium"))


async def _stats() -> None:
    client = spark_client()
    tot = hit = 0
    widths, heights, per_tile = [], [], []
    for kp in training_keys():
        img = load_sheet(kp)
        reads = await sheet_reads(client, img)
        h = sum(r is not None for _, r in reads)
        tot += len(reads)
        hit += h
        for box, r in reads:
            if r is None:
                continue
            tw, th = box[2] - box[0], box[3] - box[1]
            per_tile.append(len(r.get("items", [])))
            for it in r.get("items", []):
                b = it.get("box") or [0, 0, 0, 0]
                widths.append((b[2] - b[0]) * tw / 1000)
                heights.append((b[3] - b[1]) * th / 1000)
        print(f"{kp.stem}: {img.size} tiles {len(reads)} cached {h}", flush=True)
    import numpy as np

    print("sheets", len(training_keys()), "tiles", tot, "cached", hit)
    print("items", len(widths), "items per tile p50/p90/max", np.percentile(per_tile, [50, 90, 100]))
    print("width p50/p90/p95/p99/max", np.percentile(widths, [50, 90, 95, 99, 100]).round())
    print("height p50/p90/p95/p99/max", np.percentile(heights, [50, 90, 95, 99, 100]).round())


if __name__ == "__main__":
    asyncio.run(_stats())
