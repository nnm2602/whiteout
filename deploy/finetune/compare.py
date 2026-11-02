"""Side by side on the 6 held-out answer-key sheets: fine-tuned Glimmer, stock Glimmer, Muse Spark.

  .venv/bin/python deploy/finetune/compare.py --ft eval/results/keys_glimmer_ft --calls .cache/calls_ft
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--ft", default="eval/results/keys_glimmer_ft")
ap.add_argument("--calls", default=".cache/calls_ft", help="call cache of the fine-tuned run (request times)")
ap.add_argument("--label", default="Muse Glimmer 30B fine-tuned on Spark tile reads (Q4_K_M + stock mmproj + DFlash)")
a = ap.parse_args()

stock = json.loads((ROOT / "eval/results/keys_glimmer/SUMMARY.json").read_text())
rows = []
for s in stock["rows"]:
    p = ROOT / a.ft / f"{s['sheet']}.json"
    if not p.exists():
        continue
    ft = json.loads(p.read_text())
    sp = json.loads((ROOT / "eval/results/keys_main" / f"{s['sheet']}.json").read_text())
    rows.append({
        "sheet": s["sheet"], "items": s["items"],
        "ft_found": ft["final_hits"], "stock_found": s["glimmer_found"], "spark_found": sp["final_hits"],
        "ft_first_read": ft["first_read_hits"], "stock_first_read": s["glimmer_first_read"],
        "ft_extra": len(ft["extra"]), "stock_extra": s["glimmer_extra"], "spark_extra": len(sp["extra"]),
        "ft_minutes": round(ft["timings"]["total"] / 60, 1), "stock_minutes": s["minutes"],
    })

secs, outs = [], []
cdir = ROOT / a.calls
if cdir.exists():
    for f in cdir.glob("*.json"):
        u = json.loads(f.read_text()).get("usage", {})
        if u.get("calls"):
            secs.append(u.get("seconds", 0))
            outs.append(u.get("output_tokens", 0))


def tot(k):
    return sum(r[k] for r in rows)


n = tot("items")
summary = {
    "engine": a.label,
    "sheets": len(rows), "items": n,
    "ft_found": tot("ft_found"), "ft_pct": round(100 * tot("ft_found") / max(n, 1), 1),
    "stock_found": tot("stock_found"), "stock_pct": round(100 * tot("stock_found") / max(n, 1), 1),
    "spark_found": tot("spark_found"), "spark_pct": round(100 * tot("spark_found") / max(n, 1), 1),
    "ft_extra_total": tot("ft_extra"), "stock_extra_total": tot("stock_extra"), "spark_extra_total": tot("spark_extra"),
    "ft_median_minutes_per_sheet": round(statistics.median([r["ft_minutes"] for r in rows]), 1) if rows else None,
    "stock_median_minutes_per_sheet": round(statistics.median([r["stock_minutes"] for r in rows]), 1) if rows else None,
    "ft_requests": len(secs),
    "ft_median_request_seconds": round(statistics.median(secs), 1) if secs else None,
    "ft_median_output_tokens": statistics.median(outs) if outs else None,
    "rows": rows,
}
(ROOT / a.ft / "COMPARISON.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
for r in rows:
    print(r)
