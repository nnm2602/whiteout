"""Proofread each redraw against its scan and see whether the key items the redraw got wrong are marked.

An item counts as wrong on the redraw when its key text is not re-typeset exactly near its place (the same test as
run_redraw.py), and as marked when a difference box from the proofreading pass touches it.

    uv run python eval/run_proof.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from whiteout import ingest
from whiteout.geometry import norm_text, overlap_min
from whiteout.model_client import VisionClient
from whiteout.proof import proofread_dir

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results" / "redraw"
SHEETS = ["apollo_b503_n0006", "apollo_b503_n0013", "apollo_b502_n0014", "apollo_b503_n0024", "apollo_b503_n0057",
          "apollo_b503_n0128"]


def touches(a, b, pad: int) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


async def one(client: VisionClient, sid: str) -> dict:
    out = RES / sid
    gray = np.array(ingest.load(ROOT / f"data/raw/apollo/{sid}.jpg")[0].image.convert("L"))
    r = await proofread_dir(gray, out, client)
    diffs = r["differences"]
    texts = json.loads((out / "texts.json").read_text())
    key = json.loads((ROOT / f"eval/keys/{sid}__p1.json").read_text())
    pad = 12
    wrong = right = wrong_marked = right_marked = 0
    for it in key["items"]:
        b = it.get("box")
        want = norm_text((it.get("text") or "").replace("|", ""))
        if not b or not want:
            continue
        zone = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
        inside = sorted((t for t in texts if overlap_min(tuple(t["box"]), zone) > 0.6),
                        key=lambda t: (round(t["box"][1] / pad), t["box"][0]))
        exact = want in norm_text(" ".join(t["text"] for t in inside))
        marked = any(touches(d["box"], b, pad) for d in diffs)
        if exact:
            right += 1
            right_marked += marked
        else:
            wrong += 1
            wrong_marked += marked
    return {"sheet": sid, "differences": len(diffs), "key_items_wrong_on_redraw": wrong,
            "of_those_marked": wrong_marked, "key_items_right": right, "right_but_marked": right_marked}


async def main() -> None:
    client = VisionClient()
    rows = [await one(client, s) for s in SHEETS]
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if k != "sheet"}
    summary = {"sheets": rows, "total": tot, "usd": round(client.usage.usd, 4)}
    (RES / "PROOF_SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
