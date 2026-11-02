"""Run the report check on every supplier report and count planted errors found and false findings.

A planted error is found when the check reports a finding of the same kind on the same drawing line (missing row,
requirement differs, accepted out of tolerance, missing result) or the same supplier characteristic number
(repeated number, row not on the drawing). Every finding on a clean report is a false finding; findings on a seeded
report that match no planted error are counted as unexplained.

uv run python eval/run_fair_check.py [--no-model]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from whiteout.fair_check import check_report
from whiteout.model_client import ModelConfig, VisionClient

ROOT = Path(__file__).resolve().parents[1]
KINDS = ["missing_row", "requirement_mismatch", "accepted_out_of_tolerance", "missing_result", "duplicate_number",
         "extra_row"]
BY_ROW = {"missing_row", "requirement_mismatch", "accepted_out_of_tolerance", "missing_result"}


def flat(t: str | None) -> str:
    """Requirement text without spaces, cell bars or a leading count, so "4X .050" and "5X .050" read alike."""
    return re.sub(r"^\d+X", "", re.sub(r"[\s|]+", "", t or "").upper())


def explains(err: dict, f: dict, own_numbers: bool) -> bool:
    if f["kind"] != err["kind"]:
        return False
    d, s = f.get("drawing") or {}, f.get("supplier_row") or {}
    if err["kind"] in BY_ROW:
        if d.get("row_id") == err["row_id"]:
            return True
        # Places of one balloon with the same text (4X ...) are interchangeable. Without the drawing's balloon
        # numbers, so are requirements printed identically under different balloons.
        same_text = flat(d.get("requirement")) == flat(err["drawing_text"])
        return same_text and (own_numbers or d.get("number") == err["number"])
    return s.get("char_no") == err["char_no"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=ROOT / "eval" / "supplier_reports")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "fair_check")
    ap.add_argument("--no-model", action="store_true", help="skip the PDF reports (they are read by Muse Spark)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    answers = json.loads((a.reports / "answers.json").read_text())
    client = VisionClient(ModelConfig(effort="low"))
    jobs: dict[str, dict] = {}
    per_file = []
    totals = {k: {"planted": 0, "found": 0} for k in KINDS}
    by_source = {}
    false_on_clean: dict[str, int] = {}
    unexplained = 0
    for ans in answers:
        path = a.reports / ans["file"]
        src = "pdf" if path.suffix == ".pdf" else "xlsx"
        if src == "pdf" and a.no_model:
            continue
        job = jobs.setdefault(ans["job"], json.loads((ROOT / "samples" / "jobs" / ans["job"] / "job.json").read_text()))
        res = await check_report(job, path, client)
        own = ans["numbering"] == "own"
        found, missed = [], []
        for err in ans["errors"]:
            hit = any(explains(err, f, own) for f in res["findings"])
            (found if hit else missed).append(err)
            totals[err["kind"]]["planted"] += 1
            totals[err["kind"]]["found"] += int(hit)
            bs = by_source.setdefault(src, {"planted": 0, "found": 0, "false_on_clean": 0, "clean_reports": 0})
            bs["planted"] += 1
            bs["found"] += int(hit)
        extra = [f for f in res["findings"] if not any(explains(e, f, own) for e in ans["errors"])]
        if ans["clean"]:
            bs = by_source.setdefault(src, {"planted": 0, "found": 0, "false_on_clean": 0, "clean_reports": 0})
            bs["clean_reports"] += 1
            bs["false_on_clean"] += len(extra)
            for f in extra:
                false_on_clean[f["kind"]] = false_on_clean.get(f["kind"], 0) + 1
        else:
            unexplained += len(extra)
        (a.out / (ans["file"].replace(".", "_") + ".json")).write_text(json.dumps(res, indent=1))
        per_file.append({"file": ans["file"], "clean": ans["clean"], "numbering_planted": ans["numbering"],
                         "numbering_seen": res["numbering"], "rows": ans["rows"], "rows_read": res["rows_read"],
                         "matched": res["matched"], "planted": len(ans["errors"]), "found": len(found),
                         "findings": len(res["findings"]), "unexpected_findings": len(extra),
                         "missed": missed, "unexpected": [{"kind": f["kind"], "message": f["message"]} for f in extra],
                         "verdict": res["verdict"]})
        print(f"{ans['file']:34s} planted {len(ans['errors']):2d} found {len(found):2d}  "
              f"other findings {len(extra):2d}  rows {res['rows_read']}/{ans['rows']}  {res['numbering']}")
    planted = sum(t["planted"] for t in totals.values())
    found = sum(t["found"] for t in totals.values())
    summary = {
        "reports": len(per_file),
        "clean_reports": sum(1 for p in per_file if p["clean"]),
        "seeded_reports": sum(1 for p in per_file if not p["clean"]),
        "planted": planted,
        "found": found,
        "found_rate": round(found / planted, 4) if planted else None,
        "per_kind": {k: {**v, "rate": round(v["found"] / v["planted"], 4) if v["planted"] else None}
                     for k, v in totals.items()},
        "false_findings_on_clean_reports": sum(false_on_clean.values()),
        "false_findings_on_clean_by_kind": false_on_clean,
        "unexplained_findings_on_seeded_reports": unexplained,
        "by_source": by_source,
        "model_usage": client.usage.as_dict(),
        "files": per_file,
    }
    (a.out / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
