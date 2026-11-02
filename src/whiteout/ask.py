"""Ask the drawing: questions answered by Muse Spark from the ballooned sheets and the Form 3 lines.

The model answers in words and names the balloons it used. Arithmetic is never left to the model: for a tolerance
stack-up it only chooses the chain of lines and the direction of each, and code adds up the limits.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from .forms import fmt, form3_rows
from .model_client import VisionClient
from .prompts import _obj
from .types import Characteristic

ASK_SCHEMA = _obj({
    "answer": {"type": "string"},
    "rows": {"type": "array", "items": {"type": "string"}},
    "stackup": _obj({
        "wanted": {"type": "boolean"},
        "chain": {"type": "array", "items": _obj({"row_id": {"type": "string"}, "sign": {"type": "integer"}})},
        "given": {"type": "array", "items": _obj({"what": {"type": "string"}, "value": {"type": "string"},
                                                  "sign": {"type": "integer"}})},
        "what": {"type": "string"},
    }),
})

ASK_PROMPT = (
    "You help a quality inspector with an aerospace engineering drawing. The images are its sheets with every "
    "requirement ballooned (blue circles with numbers). Below are the drawing details and every Form 3 line: "
    "row id | type | printed requirement | lower limit | upper limit | datums | sheet and zone | measured value | "
    "pass or fail.\n\n"
    "Answer the question in at most four short sentences, plainly, citing balloon numbers. rows: the row ids your "
    "answer relies on, exactly as listed. Use only what is on the drawing and in the lines; if the drawing does not "
    "say, answer that it does not say. Do not compute sums yourself.\n"
    "If the question asks for a tolerance stack-up, a gap, a clearance or a total of several dimensions, set "
    "stackup.wanted true and give the chain: the row ids in order, with sign +1 for a dimension that adds to the "
    "result and -1 for one that subtracts; stackup.what names the result (for example 'wall between hole 18 and the "
    "edge'). A value the question itself states that is not on the drawing (such as an assumed thickness) goes in "
    "stackup.given with its sign, as printed in the question. Otherwise stackup.wanted is false and the chain and "
    "given are empty.\n\n"
)


def _rows_text(job_json: dict) -> tuple[str, list[dict]]:
    chars = [Characteristic(**c) for s in job_json["sheets"] for c in s["chars"]]
    rows = form3_rows(chars, job_json.get("results") or {})
    datums = {}
    for c in chars:
        for qi, q in enumerate(c.requirements):
            rid = c.number if len(c.requirements) == 1 else f"{c.number}.{qi + 1}"
            datums[rid] = "|".join(q.datums)
    lines = []
    for r in rows:
        base = r["row_id"].split("-")[0]
        lines.append(" | ".join([r["row_id"], r["type"], r["requirement"].replace("\n", " "), r["lower"], r["upper"],
                                 datums.get(base, ""), f"sheet {r['sheet']} {r['zone']}", r["measured"], r["status"]]))
    return "\n".join(lines), rows


def _details(job_json: dict) -> str:
    out = []
    for i, s in enumerate(job_json["sheets"]):
        inf = s["info"]
        out.append(f"Sheet {i + 1}: drawing {inf.get('drawing_number', '')} rev {inf.get('revision', '')}, "
                   f"part {inf.get('part_name', '')}, units {inf.get('units', 'in')}, material {inf.get('material', '')}")
    return "\n".join(out)


def stackup(rows: list[dict], chain: list[dict], given: list[dict] | None = None) -> dict:
    """Worst case and root sum square for a chain of Form 3 lines with signs, plus exact values the user gave."""
    from .tolerance import to_float

    by = {r["row_id"]: r for r in rows}
    used, missing, fixed = [], [], []
    lo = hi = nom = 0.0
    sq = 0.0
    for g in given or []:
        v = to_float(str(g.get("value", "")))
        if v is None:
            continue
        sign = 1 if g.get("sign", 1) >= 0 else -1
        lo, hi, nom = lo + sign * v, hi + sign * v, nom + sign * v
        fixed.append({"what": g.get("what", ""), "value": fmt(v), "sign": sign})
    for link in chain:
        r = by.get(link["row_id"])
        if r is None or r["lower"] == "" or r["upper"] == "":
            missing.append(link["row_id"])
            continue
        a, b = float(r["lower"]), float(r["upper"])
        sign = 1 if link.get("sign", 1) >= 0 else -1
        mid, half = (a + b) / 2, (b - a) / 2
        if sign > 0:
            lo, hi = lo + a, hi + b
        else:
            lo, hi = lo - b, hi - a
        nom += sign * mid
        sq += half * half
        used.append({"row_id": r["row_id"], "sign": sign, "requirement": r["requirement"], "lower": r["lower"],
                     "upper": r["upper"]})
    rss = math.sqrt(sq)
    return {"links": used, "given": fixed, "missing": missing, "worst_case": {"min": fmt(lo), "max": fmt(hi)},
            "nominal": fmt(nom), "rss": {"min": fmt(nom - rss), "max": fmt(nom + rss), "plus_minus": fmt(rss)}}


def _sheet_images(job_dir: Path, n: int, max_side: int = 1775) -> list[Image.Image]:
    ims = []
    for i in range(min(n, 4)):
        for name in (f"ballooned{i}.png", f"sheet{i}.png"):
            p = job_dir / name
            if p.exists():
                im = Image.open(p).convert("L")
                im.thumbnail((max_side, max_side))
                ims.append(im)
                break
    return ims


async def ask(job_json: dict, job_dir: Path, question: str, client: VisionClient, history: list[dict] | None = None) -> dict:
    table, rows = _rows_text(job_json)
    convo = ""
    for h in (history or [])[-4:]:
        convo += f"Earlier question: {h.get('q', '')}\nEarlier answer: {h.get('a', '')}\n"
    prompt = (ASK_PROMPT + "Drawing details:\n" + _details(job_json) + "\n\nForm 3 lines:\n" + table + "\n\n" + convo
              + "Question: " + question.strip()[:600])
    ims = _sheet_images(job_dir, len(job_json["sheets"]))
    r = await client.ask_json(ims, prompt, ASK_SCHEMA, name="ask", effort="medium")
    valid = {x["row_id"] for x in rows}
    out = {"answer": r["answer"], "rows": [x for x in r["rows"] if x in valid]}
    st = r.get("stackup") or {}
    if st.get("wanted") and st.get("chain"):
        out["stackup"] = {"what": st.get("what", ""), **stackup(rows, st["chain"], st.get("given"))}
    char_of = {x["row_id"]: x["char_id"] for x in rows}
    out["chars"] = sorted({char_of[x] for x in out["rows"]})
    return out
