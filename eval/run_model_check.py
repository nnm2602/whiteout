"""Drawing against model: run compare() on NIST drawings and their STEP AP242 models, original and edited.

1. Drawing characteristics: every sheet of each matching PDF is read with the pipeline (first read, no erased-page
   search; model calls are cached) and saved to eval/results/model_check/drawings/<case>.json.
2. Reader check: the model PMI is matched against NIST's element lists (official ids in eval/keys) and the
   specifications in the NIST PMI definition spreadsheets.
3. compare() on the unedited models (real disagreements) and on the edited copies from eval/make_model_edits.py
   (planted ones), written to eval/results/model_check/SUMMARY.json.

The CTC and FTC STEP files in data/raw/nist_ctc and data/raw/nist_ftc carry geometry only (their README says so).
The AP242 files with semantic PMI come from NIST's "STEP files with PMI" archive,
https://www.nist.gov/document/nist-pmi-step-files (NIST-PMI-STEP-Files.zip, public domain); the eleven
nist_*_ap242-e*.stp files go in data/raw/nist_pmi_step/. D2MI 827-9999-904 rev C is AP242 already.

uv run python eval/run_model_check.py [--stage drawings|all] [--cap 2.5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results" / "model_check"
EDITS = ROOT / "eval" / "model_edits"

# case id (as in eval/keys) -> (STEP with semantic PMI, drawing PDF)
CASES = {
    "nist_ctc_01": ("data/raw/nist_pmi_step/nist_ctc_01_asme1_ap242-e1.stp",
                    "data/raw/nist_ctc/nist_ctc_01_asme1_rd.pdf"),
    "nist_ctc_02": ("data/raw/nist_pmi_step/nist_ctc_02_asme1_ap242-e2.stp",
                    "data/raw/nist_ctc/nist_ctc_02_asme1_rc.pdf"),
    "nist_ctc_03": ("data/raw/nist_pmi_step/nist_ctc_03_asme1_ap242-e2.stp",
                    "data/raw/nist_ctc/nist_ctc_03_asme1_rc.pdf"),
    "nist_ctc_04": ("data/raw/nist_pmi_step/nist_ctc_04_asme1_ap242-e1.stp",
                    "data/raw/nist_ctc/nist_ctc_04_asme1_rd.pdf"),
    "nist_ctc_05": ("data/raw/nist_pmi_step/nist_ctc_05_asme1_ap242-e1.stp",
                    "data/raw/nist_ctc/nist_ctc_05_asme1_rd.pdf"),
    "nist_ftc_06": ("data/raw/nist_pmi_step/nist_ftc_06_asme1_ap242-e2.stp",
                    "data/raw/nist_ftc/nist_ftc_06_asme1_rd.pdf"),
    "nist_ftc_07": ("data/raw/nist_pmi_step/nist_ftc_07_asme1_ap242-e2.stp",
                    "data/raw/nist_ftc/nist_ftc_07_asme1_rd.pdf"),
    "nist_ftc_08": ("data/raw/nist_pmi_step/nist_ftc_08_asme1_ap242-e2.stp",
                    "data/raw/nist_ftc/nist_ftc_08_asme1_rc.pdf"),
    "nist_ftc_09": ("data/raw/nist_pmi_step/nist_ftc_09_asme1_ap242-e1.stp",
                    "data/raw/nist_ftc/nist_ftc_09_asme1_rd.pdf"),
    "nist_ftc_10": ("data/raw/nist_pmi_step/nist_ftc_10_asme1_ap242-e2.stp",
                    "data/raw/nist_ftc/nist_ftc_10_asme1_rb.pdf"),
    "nist_ftc_11": ("data/raw/nist_pmi_step/nist_ftc_11_asme1_ap242-e2.stp",
                    "data/raw/nist_ftc/nist_ftc_11_asme1_rb.pdf"),
    "nist_d2mi_904": ("data/raw/nist_d2mi/827-9999-904 rev c.stp",
                      "data/raw/nist_d2mi/827-9999-904 rev c.pdf"),
}


# ---------------------------------------------------------------------------------------------------------------
# 1. Drawing characteristics


async def read_drawing(case: str, pdf: Path, cap: float, spent: list[float]) -> Path:
    """Read every sheet of the PDF once (cached calls are free) and save chars like the CLI's result.json."""
    from whiteout import ingest
    from whiteout.model_client import ModelConfig, VisionClient
    from whiteout.pipeline import process_sheet

    dest = OUT / "drawings" / f"{case}.json"
    if dest.exists():
        return dest
    sheets = ingest.load(pdf)
    out = {"source": str(pdf.relative_to(ROOT)), "passes": 0, "sheets": [], "usage": {}}
    usage = {"calls": 0, "cached": 0, "usd": 0.0}
    general: list[dict] = []
    for i, sh in enumerate(sheets):
        if spent[0] >= cap:
            raise SystemExit(f"spend cap {cap} reached before {case} sheet {i + 1}")
        client = VisionClient(ModelConfig(effort="medium", concurrency=12))
        # index 0 for every page, as eval/run_keys.py does, so cached calls are reused
        r = await process_sheet(client, sh, 0, audit_passes=0, general_fallback=general or None)
        if i == 0:
            general = list(r.info.general_tolerances)
        for n, c in enumerate(r.chars):
            c.id, c.sheet = f"s{i}c{n}", i
        r.info.index = i
        out["sheets"].append({"info": r.info.model_dump(), "chars": [c.model_dump() for c in r.chars]})
        u = client.usage.as_dict()
        for k in usage:
            usage[k] += u.get(k, 0)
        spent[0] += u.get("usd", 0.0)
        print(f"{case} sheet {i + 1}/{len(sheets)}: {len(r.chars)} chars, calls {u['calls']}, cached {u['cached']}, "
              f"usd {u['usd']:.3f} (total {spent[0]:.3f})", flush=True)
    out["usage"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in usage.items()}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    return dest


async def read_all(cap: float) -> float:
    spent = [0.0]
    for case, (_, pdf) in CASES.items():
        await read_drawing(case, ROOT / pdf, cap, spent)
    return spent[0]


# ---------------------------------------------------------------------------------------------------------------
# 2. Reader check against NIST's lists


def key_chars(case: str) -> tuple[list, list[dict]]:
    """Answer-key items as characteristics (requirements parsed from the printed text in code)."""
    from whiteout.model_check import requirements_from_text
    from whiteout.types import Characteristic

    chars, items = [], []
    for kp in sorted((ROOT / "eval" / "keys").glob(f"{case}__p*.json")):
        key = json.loads(kp.read_text())
        for it in key["items"]:
            oid = it.get("official_id") or ""
            c = Characteristic(id=f"p{key['page']}:{it['key']}", kind=it["kind"], text=it["text"],
                               multiplier=int(it.get("multiplier") or 1), box=tuple(it.get("box") or (0, 0, 0, 0)),
                               sheet=key["page"] - 1, requirements=requirements_from_text(it["text"], it["kind"]))
            chars.append(c)
            items.append({"char": c.id, "official_id": oid, "text": it["text"], "kind": it["kind"],
                          "source": key.get("source", ""), "sheet": key["page"] - 1, "box": it.get("box")})
    return chars, items


def element_id(item: dict) -> str:
    """NIST element id (D9 for D9-1) for dimension and tolerance elements; hand keys use the item itself."""
    oid = item["official_id"]
    if oid:
        return re.sub(r"-\d+$", "", oid) if re.match(r"^[DT]\d", oid) else ""
    return item["char"]


def reader_check(case: str, model: dict) -> dict:
    from whiteout.model_check import compare

    chars, items = key_chars(case)
    res = compare(model, chars)
    status: dict[str, str] = {}
    for m in res["matched"]:
        status[m["drawing"]["id"]] = "agree"
    for m in res["differs"]:
        status.setdefault(m["drawing"]["id"], "differs")
    for m in res["repeats"]["drawing"]:
        status.setdefault(m["id"], "agree")
    compared = {d["id"] for d in res["drawing_only"]} | set(status)
    by_el: dict[str, list[str]] = {}
    for it in items:
        el = element_id(it)
        if el and it["char"] in compared:
            by_el.setdefault(el, []).append(status.get(it["char"], "missing"))
    best = {el: ("agree" if "agree" in v else "differs" if "differs" in v else "missing") for el, v in by_el.items()}
    official = any(it["official_id"] for it in items)
    out = {
        "list": "NIST element ids (elem_ids drawings, FSI)" if official else "hand key",
        "elements": len(best),
        "recovered": sum(v != "missing" for v in best.values()),
        "agree": sum(v == "agree" for v in best.values()),
        "differs": [{"element": el, "fields": [f for m in res["differs"] if m["drawing"]["id"] in
                                                [it["char"] for it in items if element_id(it) == el]
                                                for f in m["fields"]]}
                    for el, v in sorted(best.items()) if v == "differs"],
        "missing": sorted(el for el, v in best.items() if v == "missing"),
        "missing_text": sorted({it["text"] for it in items if element_id(it) in
                                {el for el, v in best.items() if v == "missing"}}),
        "model_pmi_not_in_list": [m["text"] for m in res["model_only"]],
    }
    out["model_agree_ids"] = sorted({m["model"]["id"] for m in res["matched"]} |
                                    {m["id"] for m in res["repeats"]["model"]})
    out["model_differ_ids"] = sorted({m["model"]["id"] for m in res["differs"]})
    reqs = []  # each key requirement with its own status, for findings on one line of a callout
    for st, lst in (("agree", [m["drawing"] for m in res["matched"]] + res["repeats"]["drawing"]),
                    ("differs", [m["drawing"] for m in res["differs"]]), ("missing", res["drawing_only"])):
        reqs += [{"char": d["id"], "nominal": d["nominal"], "status": st} for d in lst]
    out["key_reqs"] = reqs
    out["key_items"] = items
    out["key_status"] = {it["char"]: ("agree" if status.get(it["char"]) == "agree" else status.get(it["char"]) or
                                      ("missing" if it["char"] in compared else "not compared")) for it in items}
    return out


def spreadsheet_specs() -> dict[str, list[str]]:
    """Specification strings per case from the NIST PMI definition spreadsheets."""
    import openpyxl

    out: dict[str, list[str]] = {}
    for f, prefix in ((ROOT / "data/raw/nist_ctc/NIST-CTC-PMI-Definitions.xlsx", "nist_ctc_"),
                      (ROOT / "data/raw/nist_ftc/NIST-FTC-PMI-Definitions.xlsx", "nist_ftc_")):
        ws = openpyxl.load_workbook(f, read_only=True).worksheets[0]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not isinstance(row[0], (int, float)) or not row[4]:
                continue
            out.setdefault(f"{prefix}{int(row[0]):02d}", []).append(str(row[4]))
    return out


def spec_pieces(spec: str) -> list[str]:
    pieces = []
    for p in re.split(r"\s*&\s*|\n", spec):
        p = p.strip()
        if not p:
            continue
        if "|" in p:
            p = re.sub(r"\s{2,}.*$", "", p)  # trailing 'All Around', 'A<->B', datum feature letters
            p = "|".join(x.strip() for x in p.split("|"))
        pieces.append(p)
    return pieces


def spreadsheet_check(case: str, model: dict, specs: list[str]) -> dict:
    from whiteout.model_check import compare, requirements_from_text
    from whiteout.types import Characteristic

    chars = []
    for i, spec in enumerate(specs):
        for j, p in enumerate(spec_pieces(spec)):
            kind = "gdt_frame" if "|" in p else ("basic_dimension" if "basic" in p.lower() else "dimension")
            text = re.sub(r"(?i)\s*-?\s*basic", "", p)
            reqs = requirements_from_text(text, kind)
            m = re.match(r"^\s*(\d+)X", text)
            chars.append(Characteristic(id=f"r{i}.{j}", kind=kind, text=text, multiplier=int(m.group(1)) if m else 1,
                                        box=(0, 0, 0, 0), requirements=reqs))
    res = compare(model, chars)
    ok = {m["drawing"]["id"] for m in res["matched"]} | {m["id"] for m in res["repeats"]["drawing"]}
    near = {m["drawing"]["id"] for m in res["differs"]}
    compared = ok | near | {d["id"] for d in res["drawing_only"]}
    return {"specifications": len(compared), "agree": len(ok & compared), "differs": len(near - ok),
            "missing": sorted(c.text for c in chars if c.id in compared and c.id not in ok | near),
            "differ_detail": [{"spec": m["drawing"]["text"], "model": m["model"]["text"], "fields": m["fields"]}
                              for m in res["differs"]]}


# ---------------------------------------------------------------------------------------------------------------
# 3. Drawing against model, original and edited


def triage(res: dict, rc: dict) -> None:
    """Say for each finding whether NIST's own list sides with the model (so the drawing read is at fault)."""
    from whiteout.geometry import overlap_min

    agree_ids, differ_ids = set(rc["model_agree_ids"]), set(rc["model_differ_ids"])
    keyed = [it for it in rc["key_items"] if it.get("box")]

    def key_for(dv: dict) -> dict | None:
        best = None
        for it in keyed:
            if it["sheet"] != dv["sheet"]:
                continue
            ov = overlap_min(tuple(it["box"]), tuple(dv["box"]))
            if ov > 0.3 and (best is None or ov > best[0]):
                best = (ov, it)
        return best[1] if best else None

    for f in res["differs"]:
        mid = f["model"]["id"]
        f["cause"] = ("drawing read: the model agrees with NIST's list" if mid in agree_ids else
                      "model differs from NIST's list too" if mid in differ_ids else "model PMI not in NIST's list")
        k = key_for(f["drawing"])
        f["key_text"] = k["text"] if k else None
    for m in res["model_only"]:
        m["cause"] = ("drawing read missed or misread it: NIST's list has it" if m["id"] in agree_ids | differ_ids
                      else "not in NIST's list for the drawing")
    for d in res["drawing_only"]:
        k = key_for(d)
        st = rc["key_status"].get(k["char"]) if k else None
        same = [r for r in rc["key_reqs"] if k and r["char"] == k["char"] and r["nominal"] is not None
                and d["nominal"] is not None and abs(r["nominal"] - d["nominal"]) < 1e-6]
        if same:
            st = same[0]["status"]
        d["key_text"] = k["text"] if k else None
        d["cause"] = ("not a listed element (extra read)" if k is None else
                      "listed element missing from the model" if st == "missing" else
                      "drawing read: the listed element matches model PMI" if st in ("agree", "differs") else
                      "listed element not compared")


def brief(res: dict) -> dict:
    return {
        "summary": res["summary"], "drawing_units": res["drawing_units"], "model_units": res["model_units"],
        "warnings": res["warnings"],
        "differs": [{"model": f["model"]["text"], "model_id": f["model"]["id"], "drawing": f["drawing"]["text"],
                     "drawing_requirement": f["drawing"]["requirement_text"],
                     "drawing_id": f["drawing"]["id"], "sheet": f["drawing"]["sheet"] + 1, "fields": f["fields"],
                     "cause": f.get("cause"), "key_text": f.get("key_text")} for f in res["differs"]],
        "model_only": [{"model": m["text"], "model_id": m["id"], "cause": m.get("cause")} for m in res["model_only"]],
        "drawing_only": [{"drawing": d["text"], "drawing_requirement": d["requirement_text"], "drawing_id": d["id"],
                          "sheet": d["sheet"] + 1,
                          "cause": d.get("cause"), "key_text": d.get("key_text")} for d in res["drawing_only"]],
    }


def seeded(case: str, chars: list, units: str | None, orig: dict) -> dict | None:
    from whiteout.model_check import compare, to_json

    ep = EDITS / f"{case}.json"
    if not ep.exists():
        return None
    rec = json.loads(ep.read_text())
    res = compare(ROOT / rec["edited"], chars, units)
    (OUT / f"{case}_edited.json").write_text(to_json(res))
    before = {(f["model"]["id"], f["drawing"]["id"]) for f in orig["differs"]} | \
        {m["id"] for m in orig["model_only"]} | {d["id"] for d in orig["drawing_only"]}
    new_drawing_only = [d for d in res["drawing_only"] if d["id"] not in before]
    results, claimed = [], set()
    for e in rec["edits"]:
        how = "missed"
        if e["expect"] == "differs":
            for f in res["differs"]:
                if f["model"]["id"] == e["target"] and any(x["field"] == e["field"] for x in f["fields"]):
                    how = ("differs, on its own drawing item" if f["drawing"]["id"] == e["drawing"]["id"] else
                           "differs, on an identical drawing item"
                           if f["drawing"]["requirement_text"] == e["drawing"]["requirement_text"]
                           else f"differs, on drawing item {f['drawing']['text']!r}")
                    claimed.add(f["drawing"]["id"])
            if how == "missed" and any(m["id"] == e["target"] for m in res["model_only"]):
                how = "model only (an identical model PMI took its drawing item)"
        else:
            if any(d["id"] == e["drawing"]["id"] for d in res["drawing_only"]):
                how = "drawing only, on its own drawing item"
                claimed.add(e["drawing"]["id"])
            elif len(new_drawing_only) == 1 and sum(x["edit"] == "removed_gdt" for x in rec["edits"]) == 1:
                how = f"drawing only, on drawing item {new_drawing_only[0]['text']!r}"
                claimed.add(new_drawing_only[0]["id"])
            elif any(d["text"] == e["drawing"]["text"] for d in res["repeats"]["drawing"]):
                how = "missed: the drawing item now counts as a repeat of an identical model PMI"
        results.append({**{k: e[k] for k in ("edit", "target", "before", "after")}, "drawing": e["drawing"],
                        "found": not how.startswith("missed"), "how": how})
    # findings in the edited run that are neither planted nor present before
    targets = {e["target"] for e in rec["edits"]} | {e["drawing"]["id"] for e in rec["edits"]} | claimed
    extra = [f"differs {f['model']['text']} / {f['drawing']['text']}" for f in res["differs"]
             if (f["model"]["id"], f["drawing"]["id"]) not in before and f["model"]["id"] not in targets
             and f["drawing"]["id"] not in targets]
    extra += [f"model only {m['text']}" for m in res["model_only"] if m["id"] not in before and m["id"] not in targets]
    extra += [f"drawing only {d['text']}" for d in res["drawing_only"] if d["id"] not in before
              and d["id"] not in targets]
    return {"planted": len(results), "found": sum(r["found"] for r in results),
            "found_on_its_own_drawing_item": sum("its own" in r["how"] for r in results),
            "edits": results,
            "other_new_findings": extra, "summary": res["summary"]}


def evaluate() -> dict:
    from whiteout.model_check import compare, load_chars, read_step, to_json

    specs = spreadsheet_specs()
    out: dict = {"cases": {}}
    for case, (step, pdf) in CASES.items():
        model = read_step(ROOT / step)
        chars, units = load_chars(OUT / "drawings" / f"{case}.json")
        rc = reader_check(case, model)
        sc = spreadsheet_check(case, model, specs[case]) if case in specs else None
        res = compare(model, chars, units)
        triage(res, rc)
        (OUT / f"{case}.json").write_text(to_json(res))
        sd = seeded(case, chars, units, res)
        out["cases"][case] = {
            "step": step, "drawing": pdf, "schema": model["schema"], "model_units": model["units"]["length"],
            "model": {**model["counts"], "datums": model["datums"]["labels"], **model["cylinders"]},
            "reader_vs_nist": {k: v for k, v in rc.items() if k not in ("model_agree_ids", "model_differ_ids",
                                                                         "key_items", "key_status", "key_reqs")},
            "reader_vs_spreadsheet": sc,
            "unedited": brief(res),
            "seeded": sd,
        }
        s = res["summary"]
        print(f"{case}: list {rc['recovered']}/{rc['elements']} (agree {rc['agree']})"
              + (f", sheet {sc['agree'] + sc['differs']}/{sc['specifications']}" if sc else "")
              + f" | matched {s['matched']} differ {s['differs']} model-only {s['model_only']} "
                f"drawing-only {s['drawing_only']}"
              + (f" | seeded {sd['found']}/{sd['planted']}" if sd else ""), flush=True)
    cs = out["cases"].values()
    nist = [c for c in cs if c["reader_vs_nist"]["list"].startswith("NIST")]
    out["reader_vs_nist"] = {
        "cases": len(nist),
        "elements": sum(c["reader_vs_nist"]["elements"] for c in nist),
        "recovered": sum(c["reader_vs_nist"]["recovered"] for c in nist),
        "agree": sum(c["reader_vs_nist"]["agree"] for c in nist),
    }
    sheets = [c["reader_vs_spreadsheet"] for c in cs if c["reader_vs_spreadsheet"]]
    out["reader_vs_spreadsheet"] = {"specifications": sum(x["specifications"] for x in sheets),
                                    "recovered": sum(x["agree"] + x["differs"] for x in sheets),
                                    "agree": sum(x["agree"] for x in sheets)}
    un = [c["unedited"] for c in cs]
    out["unedited"] = {k: sum(u["summary"][k] for u in un) for k in
                       ("model_pmi", "drawing_requirements", "matched", "differs", "model_only", "drawing_only")}
    causes: dict[str, int] = {}
    for u in un:
        for f in u["differs"] + u["model_only"] + u["drawing_only"]:
            causes[f["cause"]] = causes.get(f["cause"], 0) + 1
    out["unedited"]["by_cause"] = dict(sorted(causes.items(), key=lambda x: -x[1]))
    sd = [c["seeded"] for c in cs if c["seeded"]]
    hows: dict[str, int] = {}
    for x in sd:
        for e in x["edits"]:
            k = re.sub(r" '.*'$| on drawing item.*$", " on another drawing item", e["how"])
            hows[k] = hows.get(k, 0) + 1
    by_edit: dict[str, list[int]] = {}
    for x in sd:
        for e in x["edits"]:
            v = by_edit.setdefault(e["edit"], [0, 0])
            v[0] += e["found"]
            v[1] += 1
    out["seeded"] = {"planted": sum(x["planted"] for x in sd), "found": sum(x["found"] for x in sd),
                     "found_on_its_own_drawing_item": sum(x["found_on_its_own_drawing_item"] for x in sd),
                     "by_edit": {k: {"found": v[0], "planted": v[1]} for k, v in by_edit.items()},
                     "how": dict(sorted(hows.items(), key=lambda x: -x[1])),
                     "other_new_findings": sum(len(x["other_new_findings"]) for x in sd)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["drawings", "all"], default="all")
    ap.add_argument("--cap", type=float, default=2.5, help="USD limit for uncached model calls")
    a = ap.parse_args()
    from whiteout.model_check import to_json

    spent = asyncio.run(read_all(a.cap))
    print(f"drawing reads this run: usd {spent:.3f}")
    if a.stage == "drawings":
        return
    out = evaluate()
    reads = [json.loads((OUT / "drawings" / f"{c}.json").read_text())["usage"] for c in CASES]
    out["drawing_reads"] = {"sheets": sum(len(json.loads((OUT / "drawings" / f"{c}.json").read_text())["sheets"])
                                          for c in CASES), "passes": 0,
                            "model_calls": sum(u.get("calls", 0) for u in reads),
                            "cached_calls": sum(u.get("cached", 0) for u in reads),
                            "usd": round(sum(u.get("usd", 0.0) for u in reads), 4)}
    (OUT / "SUMMARY.json").write_text(to_json(out))
    print(to_json({k: v for k, v in out.items() if k != "cases"}))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
