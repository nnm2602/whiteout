"""Answer-key test: run Whiteout on public drawings and compare with keys written before any run.

Reports, per source and overall: recall of the first read, recall after the whiteout search, precision, exact text,
and the same for a plain OCR baseline (Tesseract words + a dimension pattern), with Wilson 95% intervals.

uv run python eval/run_keys.py --keys 'eval/keys/*.json' --out eval/results/keys
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from whiteout import ingest
from whiteout.geometry import SheetInk, norm_text, ocr_words, overlap_min, text_sim
from whiteout.model_client import ModelConfig, VisionClient
from whiteout.pipeline import process_sheet

ROOT = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def key_file(key: dict) -> Path:
    if key.get("file"):
        return ROOT / key["file"]
    kid = key["id"]
    if kid.startswith("nist_d2mi_"):
        return ROOT / f"data/raw/nist_d2mi/827-9999-{kid.split('_')[-1]} rev c.pdf"
    if kid.startswith("apollo_"):
        return ROOT / f"data/raw/apollo/{kid}.jpg"
    if kid.startswith("milstd_"):
        hits = [p for p in (ROOT / "data/raw/milstd").iterdir() if p.stem.lower() == kid.split("_", 1)[1].lower()]
        return hits[0]
    raise FileNotFoundError(kid)


def match(key_items: list[dict], found: list[dict]) -> list[tuple[int, int]]:
    """One-to-one matching of key items to found items: by place when the key has a box, else by text."""
    pairs = []
    for ki, k in enumerate(key_items):
        for fi, f in enumerate(found):
            sim = text_sim(k["text"], f["text"])
            if k.get("box"):
                ov = overlap_min(tuple(k["box"]), tuple(f["box"]))
                if ov > 0.25 and sim >= 50:
                    pairs.append((ov + sim / 100, ki, fi))
            elif sim >= 85:
                pairs.append((sim / 100, ki, fi))
    pairs.sort(reverse=True)
    used_k, used_f, out = set(), set(), []
    for _, ki, fi in pairs:
        if ki in used_k or fi in used_f:
            continue
        used_k.add(ki)
        used_f.add(fi)
        out.append((ki, fi))
    return out


DIM = re.compile(r"(?:\d+\s*[Xx]\s*)?[ØR]?\s*\d*\.\d+(?:\s*(?:±|\+/-|-|/)\s*\d*\.\d+)?|\d+\s*°|\d+/\d+")


def baseline(gray: np.ndarray) -> list[dict]:
    """Plain OCR baseline: Tesseract words that look like dimensions, merged with their right-hand neighbours."""
    words = ocr_words(gray)
    out = []
    for b, t in words:
        if DIM.search(t):
            out.append({"text": t, "box": list(b)})
    return out


async def one(client: VisionClient, kp: str, out: Path) -> dict:
    key = json.loads(Path(kp).read_text())
    path = key_file(key)
    page = key.get("page", 1)
    sheets = ingest.load(path, max_pages=page)
    sh = sheets[page - 1]
    if sh.image.size != (key["width"], key["height"]):
        print(f"warning: {kp} key size {key['width']}x{key['height']} != image {sh.image.size}", file=sys.stderr)
    res = await process_sheet(client, sh, 0)
    found = [{"text": c.text, "box": list(c.box), "found_in": c.found_in, "multiplier": c.multiplier}
             for c in res.chars if not c.ignored]
    first = [f for f in found if f["found_in"] == "read"]
    m_final = match(key["items"], found)
    m_first = match(key["items"], first)
    # The key lists each line of a multi-line callout separately; Whiteout reads the callout as one item and splits it
    # into Form 3 lines. A key line contained in a found callout counts as found ("within" list, reported separately).
    def within(m, pool):
        got = {ki for ki, _ in m}
        extra = []
        for ki, k in enumerate(key["items"]):
            if ki in got or not k.get("box"):
                continue
            for fi, f in enumerate(pool):
                if overlap_min(tuple(k["box"]), tuple(f["box"])) > 0.5 and text_sim(k["text"], f["text"]) >= 80 \
                        and len(norm_text(f["text"])) > len(norm_text(k["text"])):
                    extra.append((ki, fi))
                    break
        return extra
    w_final = within(m_final, found)
    w_first = within(m_first, first)
    base = baseline(np.array(sh.image))
    m_base = match(key["items"], base)
    def same_text(a: str, b: str) -> bool:
        na, nb = norm_text(a), norm_text(b)
        return na == nb or (len(na) >= 3 and na in nb and len(nb) <= len(na) + 6)

    exact = sum(1 for ki, fi in m_final if same_text(key["items"][ki]["text"], found[fi]["text"]))
    mult = sum(1 for ki, fi in m_final if int(key["items"][ki].get("multiplier") or 1) == int(found[fi]["multiplier"]))
    matched_k = {ki for ki, _ in m_final} | {ki for ki, _ in w_final}
    matched_f = {fi for _, fi in m_final}
    # A key item for a whole notes block may be read as several notes: pieces inside it are not extras.
    for fi, f in enumerate(found):
        if fi in matched_f:
            continue
        for k in key["items"]:
            if k.get("box") and overlap_min(tuple(f["box"]), tuple(k["box"])) > 0.7 and \
                    (k["box"][2] - k["box"][0]) * (k["box"][3] - k["box"][1]) > 2.5 * max(1, (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1])):
                matched_f.add(fi)
                break
    # Is each missed requirement still visible on the final erased page? (ink kept in its key box)
    orig = np.array(sh.image) < 150
    resid = (res.residue < 150) if res.residue is not None else orig
    visible = 0
    for i in range(len(key["items"])):
        if i in matched_k or not key["items"][i].get("box"):
            continue
        x0, y0, x1, y1 = key["items"][i]["box"]
        o = orig[y0:y1, x0:x1].sum()
        if o and resid[y0:y1, x0:x1].sum() / o > 0.5:
            visible += 1
    true_missed = [i for i in range(len(key["items"])) if i not in matched_k and not any(
        text_sim(key["items"][i]["text"], found[fi]["text"]) >= 80 and key["items"][i].get("box")
        and overlap_min(tuple(key["items"][i]["box"]), tuple(found[fi]["box"])) > 0 for fi in range(len(found)))]
    rec = {
        "missed_visible_on_erased_page": visible,
        "true_missed": len(true_missed),
        "true_missed_visible": sum(1 for i in true_missed if key["items"][i].get("box") and _visible(orig, resid, key["items"][i]["box"])),
        "key": Path(kp).name, "kind": key["id"].split("_")[0] if not key["id"].startswith("nist") else "_".join(key["id"].split("_")[:2]),
        "source": key.get("source"), "items": len(key["items"]), "found": len(found),
        "first_read_hits": len(m_first) + len(w_first), "final_hits": len(m_final) + len(w_final),
        "hits_inside_larger_callout": len(w_final), "baseline_hits": len(m_base),
        "baseline_found": len(base), "exact_text": exact, "multiplier_right": mult,
        "audit_added": sum(1 for f in found if f["found_in"].startswith("audit")),
        "audit_added_correct": sum(1 for _, fi in m_final if found[fi]["found_in"].startswith("audit")),
        "missed": [key["items"][i] for i in range(len(key["items"])) if i not in matched_k],
        "extra": [found[i] for i in range(len(found)) if i not in matched_f],
        "timings": res.timings,
    }
    (out / Path(kp).name).write_text(json.dumps(rec, indent=1))
    return rec


def _visible(orig, resid, b) -> bool:
    x0, y0, x1, y1 = b
    o = orig[y0:y1, x0:x1].sum()
    return bool(o) and resid[y0:y1, x0:x1].sum() / o > 0.5


def totals(recs: list[dict]) -> dict:
    n = sum(r["items"] for r in recs)
    f = sum(r["found"] for r in recs)
    t = {
        "sheets": len(recs), "key_items": n,
        "first_read_recall": sum(r["first_read_hits"] for r in recs) / max(n, 1),
        "final_recall": sum(r["final_hits"] for r in recs) / max(n, 1),
        "final_recall_ci95": wilson(sum(r["final_hits"] for r in recs), n),
        "precision": sum(r["final_hits"] for r in recs) / max(f, 1),
        "exact_text_of_matched": sum(r["exact_text"] for r in recs) / max(sum(r["final_hits"] for r in recs), 1),
        "multiplier_right_of_matched": sum(r["multiplier_right"] for r in recs) / max(sum(r["final_hits"] for r in recs), 1),
        "baseline_recall": sum(r["baseline_hits"] for r in recs) / max(n, 1),
        "baseline_precision": sum(r["baseline_hits"] for r in recs) / max(sum(r["baseline_found"] for r in recs), 1),
        "audit_added": sum(r["audit_added"] for r in recs),
        "missed": sum(r["items"] - r["final_hits"] for r in recs),
        "missed_visible_on_erased_page": sum(r.get("missed_visible_on_erased_page", 0) for r in recs),
        "true_missed": sum(r.get("true_missed", 0) for r in recs),
        "true_missed_visible": sum(r.get("true_missed_visible", 0) for r in recs),
        "hits_inside_larger_callout": sum(r.get("hits_inside_larger_callout", 0) for r in recs),
        "audit_added_correct": sum(r["audit_added_correct"] for r in recs),
    }
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in t.items()}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="eval/keys/*.json")
    ap.add_argument("--out", default="eval/results/keys")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--engine", choices=["spark", "offline"], default="spark")
    ap.add_argument("--batch", type=int, default=3)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    kps = sorted({p for pat in a.keys.split(",") for p in glob.glob(str(ROOT / pat.strip()))})
    if a.engine == "offline":
        from whiteout.cli import offline_config

        cfg = offline_config()
        cfg.extra = {"max_tokens": 6000}
        cfg.timeout = 1500
        client = VisionClient(cfg)
    else:
        client = VisionClient(ModelConfig(effort=a.effort, concurrency=12))
    recs = []
    B = a.batch
    for i in range(0, len(kps), B):
        got = await asyncio.gather(*(one(client, kp, out) for kp in kps[i:i + B]), return_exceptions=True)
        for kp, g in zip(kps[i:i + B], got):
            if isinstance(g, Exception):
                print(f"failed {kp}: {type(g).__name__}: {g}", file=sys.stderr)
            else:
                recs.append(g)
        print(f"{len(recs)}/{len(kps)} sheets, ${client.usage.usd:.2f}", flush=True)
    by = {}
    for kind in sorted({r["kind"] for r in recs}):
        by[kind] = totals([r for r in recs if r["kind"] == kind])
    summary = {"overall": totals(recs), "by_source": by, "usage": client.usage.as_dict()}
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
