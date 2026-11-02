"""Questions with answers worked out in code from the Form 3 lines, asked of the three sample drawings.

For list questions the balloons Muse Spark cites are compared with the right set; for limit questions the answer must
cite the balloon and give both limits; for stack-ups the chain must be the right lines with the right signs.

    uv run python eval/run_ask.py
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path

from whiteout.ask import _rows_text, ask
from whiteout.model_client import VisionClient
from whiteout.types import Characteristic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JOBS = ["bd20fca1920f", "e9f770195cba", "880a653a1d04"]
OUT = HERE / "results" / "ask"


def base(rid: str) -> str:
    return re.split(r"[.-]", rid)[0]


def questions(job: dict, rng: random.Random) -> list[dict]:
    chars = [Characteristic(**c) for s in job["sheets"] for c in s["chars"]]
    _, rows = _rows_text(job)
    qs = []
    datums = {}
    for c in chars:
        for q in c.requirements:
            for d in q.datums:
                datums.setdefault(d.strip()[:1].upper(), set()).add(c.number)
    for d in sorted(k for k in datums if k.isalpha())[:2]:
        qs.append({"kind": "list", "q": f"Which balloons refer to datum {d}?", "want": sorted(datums[d])})
    for t, word in (("radius", "radius"), ("gdt", "GD&T feature control frame"), ("thread", "thread")):
        want = sorted({r["number"] for r in rows if r["type"] == t})
        if want:
            qs.append({"kind": "list", "q": f"Which balloons are {word} requirements?", "want": want})
    sized = [r for r in rows if r["lower"] and r["upper"] and r["type"] in ("size", "depth", "location", "radius")]
    if sized:
        band = min(float(r["upper"]) - float(r["lower"]) for r in sized)
        want = sorted({r["number"] for r in sized if abs(float(r["upper"]) - float(r["lower"]) - band) < 1e-9})
        qs.append({"kind": "list", "q": "Among the size, depth, location and radius requirements (not GD&T frames), "
                   "which balloons have the narrowest tolerance band?", "want": want})
    for r in rng.sample(sized, min(3, len(sized))):
        qs.append({"kind": "limits", "q": f"What are the lower and upper limits of line {r['row_id']}?",
                   "want": [r["row_id"]], "lower": float(r["lower"]), "upper": float(r["upper"])})
    if len(sized) >= 2:
        a, b = rng.sample(sized, 2)
        qs.append({"kind": "stack", "q": f"What is the worst-case range of line {a['row_id']} plus line {b['row_id']}?",
                   "chain": {a["row_id"]: 1, b["row_id"]: 1}})
        qs.append({"kind": "stack", "q": f"What is the worst-case range of line {a['row_id']} minus line {b['row_id']}?",
                   "chain": {a["row_id"]: 1, b["row_id"]: -1}})
    return qs


def nums(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d*\.\d+|\d+", text)]


async def main() -> None:
    rng = random.Random(3)
    client = VisionClient()
    results = []
    for jid in JOBS:
        jdir = ROOT / "samples" / "jobs" / jid
        job = json.loads((jdir / "job.json").read_text())
        for q in questions(job, rng):
            r = await ask(job, jdir, q["q"], client)
            got = sorted({base(x) for x in r["rows"]})
            row = {"job": jid, **q, "answer": r["answer"], "cited": got}
            if q["kind"] == "list":
                want = set(q["want"])
                row["found"] = len(want & set(got))
                row["extra"] = len(set(got) - want)
                row["right"] = set(got) == want
            elif q["kind"] == "limits":
                ns = nums(r["answer"])
                row["right"] = base(q["want"][0]) in got and any(abs(n - q["lower"]) < 1e-6 for n in ns) and any(
                    abs(n - q["upper"]) < 1e-6 for n in ns)
            else:
                st = r.get("stackup") or {}
                chain = {x["row_id"]: x["sign"] for x in st.get("links", [])}
                row["chain"] = chain
                row["right"] = chain == q["chain"]
            results.append(row)
            print(jid, q["kind"], row["right"], q["q"][:70])
    by = {}
    for k in ("list", "limits", "stack"):
        rs = [r for r in results if r["kind"] == k]
        by[k] = {"questions": len(rs), "right": sum(r["right"] for r in rs)}
    lists = [r for r in results if r["kind"] == "list"]
    summary = {"questions": len(results), "right": sum(r["right"] for r in results), "by_kind": by,
               "list_balloons_wanted": sum(len(r["want"]) for r in lists),
               "list_balloons_found": sum(r["found"] for r in lists),
               "list_balloons_extra": sum(r["extra"] for r in lists),
               "usd": round(client.usage.usd, 4), "results": results}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
