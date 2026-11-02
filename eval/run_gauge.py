"""Read every synthetic gauge photo with Muse Spark and compare with the known reading.

What matters most is the reading that is wrong but marked sure: that is the only kind that could reach a report
without a person looking at it.

    uv run python eval/run_gauge.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from whiteout.gauge import read_gauge
from whiteout.model_client import VisionClient

HERE = Path(__file__).resolve().parent
PHOTOS = HERE / "gauge_photos"
OUT = HERE / "results" / "gauge"


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit()).lstrip("0")


async def main() -> None:
    answers = json.loads((PHOTOS / "answers.json").read_text())
    client = VisionClient()
    rows = []

    async def one(a):
        units = a["unit"]  # read in the display's own unit so the digits can be compared directly
        r = await read_gauge((PHOTOS / a["file"]).read_bytes(), client, units)
        want = float(a["display"]) * (-1 if a["negative"] else 1)
        right = r["value"] is not None and abs(r["value"] - want) < 1e-6 and r["unit"] == a["unit"]
        return {"file": a["file"], "want": a["display"], "negative": a["negative"], "unit": a["unit"],
                "glare": a["glare"], "read": r["display"], "read_negative": r["negative"], "read_unit": r["unit"],
                "sure": r["sure"], "right": right, "code_reading": r["code_reading"], "note": r["note"]}

    rows = await asyncio.gather(*(one(a) for a in answers))
    clear = [r for r in rows if not r["glare"]]
    glare = [r for r in rows if r["glare"]]
    summary = {
        "photos": len(rows),
        "clear_photos": len(clear),
        "clear_right_and_sure": sum(r["right"] and r["sure"] for r in clear),
        "clear_unsure": sum(not r["sure"] for r in clear),
        "glare_photos": len(glare),
        "glare_marked_unsure": sum(not r["sure"] for r in glare),
        "wrong_but_sure": sum((not r["right"]) and r["sure"] for r in rows),
        "model_right": sum(r["right"] for r in rows),
        "code_reader_digits_right": sum(_digits(r["code_reading"]) == _digits(r["want"]) for r in rows),
        "usd": round(client.usage.usd, 4),
        "note": "synthetic photos made by eval/make_gauge_photos.py",
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    for r in rows:
        if not r["right"] and r["sure"]:
            print(r)


if __name__ == "__main__":
    asyncio.run(main())
