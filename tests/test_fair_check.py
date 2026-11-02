import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from openpyxl import Workbook
from PIL import Image

from whiteout.fair_check import SupplierRow, check_report, check_rows, read_xlsx
from whiteout.forms import write_workbook
from whiteout.model_client import Usage
from whiteout.types import Characteristic, Requirement, SheetInfo

ROOT = Path(__file__).resolve().parents[1]


def char(cid, number, text, reqs, mult=1):
    return Characteristic(id=cid, number=number, text=text, multiplier=mult, box=(10, 20, 30, 40), zone="B3",
                          requirements=reqs).model_dump()


def small_job() -> dict:
    hole = Requirement(type="size", text="Ø.182 - .192", lower=0.182, upper=0.192, tolerance_source="printed")
    depth = Requirement(type="depth", text="↧.250 ± .010", lower=0.24, upper=0.26, tolerance_source="printed")
    pos = Requirement(type="gdt", text="⊕ | Ø.005Ⓜ | A | B | C", lower=0.0, upper=0.005,
                      tolerance_source="printed")
    note = Requirement(type="note", text="2. FINISH: ANODIZE PER MIL-A-8625, TYPE II, CLASS 1.", attribute=True)
    width = Requirement(type="size", text="2X 1.250 ±.005", lower=1.245, upper=1.255, tolerance_source="printed")
    chars = [
        char("c1", "1", "2. FINISH: ANODIZE PER MIL-A-8625, TYPE II, CLASS 1.", [note]),
        char("c2", "2", "Ø.182 - .192 ↧.250 ± .010", [hole, depth]),
        char("c3", "3", "⊕ | Ø.005Ⓜ | A | B | C", [pos]),
        char("c4", "4", "2X 1.250 ±.005", [width], mult=2),
    ]
    return {"sheets": [{"info": {"index": 0, "width": 100, "height": 100}, "chars": chars}]}


CLEAN = [  # supplier's own wording: no note number, no 2X, leading zeros, letters for instances
    ["1", "FINISH: ANODIZE PER MIL-A-8625, TYPE II, CLASS 1.", "", "", "Conforms", "ACC", ""],
    ["2.1", "Ø0.182-0.192", "0.182", "0.192", "0.1871", "ACC", ""],
    ["2.2", "↧0.250 ±0.010", "0.240", "0.260", "0.2530", "ACC", ""],
    ["3", "⊕ Ø0.005 (M) A B C", "0", "0.005", "0.0021", "ACC", ""],
    ["4A", "1.250 ±0.005", "1.245", "1.255", "1.2512", "ACC", ""],
    ["4B", "1.250 ±0.005", "1.245", "1.255", "1.2561", "REJ", "NCR-12"],
]


def write_supplier(path: Path, rows: list[list[str]]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "FAIR"
    ws.append(["FIRST ARTICLE INSPECTION REPORT"])
    ws.append([])
    ws.append(["Balloon #", "Characteristic Accountability", "", "", "Inspection Results", "Evaluation", ""])
    ws.merge_cells("B3:D3")
    ws.append(["", "Designed Characteristic", "Min", "Max", "Actual Result", "Accept / Reject", "NCR No."])
    ws.merge_cells("A3:A4")
    for r in rows:
        ws.append(r)
    ws.append([])
    ws.append(["Inspector: A. Person", "", "", "", "Date: 2026-10-27"])
    wb.save(path)


def kinds(res: dict) -> list[str]:
    return sorted(f["kind"] for f in res["findings"])


def test_clean_report_in_another_layout(tmp_path):
    p = tmp_path / "clean.xlsx"
    write_supplier(p, CLEAN)
    rows, layout = read_xlsx(p)
    assert layout["header_row"] == 4 and len(rows) == 6
    assert {"number", "requirement", "lower", "upper", "result", "accept", "nc"} <= set(layout["columns"])
    res = check_rows(small_job(), rows, layout)
    assert res["findings"] == [] and res["ok"] and res["rejected_rows"] == 1
    assert res["matched"] == 6 and res["numbering"] == "drawing"


def test_planted_errors_of_every_kind(tmp_path):
    bad = [list(r) for r in CLEAN]
    bad[1][1], bad[1][3] = "Ø0.182-0.197", "0.197"  # limit copied wrongly
    bad[2][4] = "0.2710"  # out of tolerance, accepted
    bad[3][4] = ""  # no result
    del bad[4]  # 4A left out
    bad.append(["3", "⊕ Ø0.005 (M) A B C", "0", "0.005", "0.0019", "ACC", ""])  # typed twice
    bad.append(["9", "R0.500 ±0.010", "0.490", "0.510", "0.5004", "ACC", ""])  # not on the drawing
    p = tmp_path / "bad.xlsx"
    write_supplier(p, bad)
    res = asyncio.run(check_report(small_job(), p))
    assert kinds(res) == sorted(["requirement_mismatch", "accepted_out_of_tolerance", "missing_result", "missing_row",
                                 "duplicate_number", "extra_row"])
    by = {f["kind"]: f for f in res["findings"]}
    assert by["requirement_mismatch"]["drawing"]["row_id"] == "2.1"
    assert "0.197" in by["requirement_mismatch"]["message"]
    assert by["accepted_out_of_tolerance"]["drawing"]["row_id"] == "2.2"
    assert by["missing_row"]["drawing"]["char_id"] == "c4" and by["missing_row"]["supplier_row"] is None
    assert by["missing_row"]["drawing"]["box"] == [10, 20, 30, 40]
    assert by["extra_row"]["supplier_row"]["char_no"] == "9" and by["extra_row"]["drawing"] is None
    assert not res["ok"] and res["verdict"].startswith("Return to supplier: 6 problems")
    assert res["summary"]["missing_row"] == 1


def test_note_word_and_datum_changes_are_caught():
    job = small_job()
    rows = [SupplierRow(0, "row 1", "1", "FINISH: ANODIZE PER MIL-A-8625, TYPE II.", result="C", accept="ACC"),
            SupplierRow(1, "row 2", "3", "⊕ Ø.005 (M) A B", result=".001", accept="ACC")]
    for s in rows:
        s.base = int(s.char_no)
    res = check_rows(job, rows)
    mism = [f for f in res["findings"] if f["kind"] == "requirement_mismatch"]
    assert len(mism) == 2
    assert "CLASS" in mism[0]["message"] and "datums" in mism[1]["message"]


def test_own_balloon_numbers_match_on_text(tmp_path):
    shuffled = [["7", *CLEAN[3][1:]], ["8", *CLEAN[0][1:]], ["9A", *CLEAN[4][1:]], ["9B", *CLEAN[5][1:]],
                ["10", *CLEAN[2][1:]], ["11", *CLEAN[1][1:]]]
    p = tmp_path / "own.xlsx"
    write_supplier(p, shuffled)
    res = asyncio.run(check_report(small_job(), p))
    assert res["numbering"] == "own" and res["findings"] == []


def test_reads_our_own_workbook(tmp_path):
    job = small_job()
    chars = [Characteristic(**c) for c in job["sheets"][0]["chars"]]
    sheet = SimpleNamespace(info=SheetInfo(index=0, width=100, height=100), chars=chars)
    p = tmp_path / "ours.xlsx"
    results = {"1": "OK", "2.1": ".188", "2.2": ".251", "3": ".002", "4-1": "1.251", "4-2": "1.249"}
    write_workbook([sheet], p, results=results)
    rows, layout = read_xlsx(p)
    assert layout["sheet"] == "Form 3" and len(rows) == 6
    res = check_rows(job, rows, layout)
    assert res["findings"] == [], res["findings"]


class FakeClient:
    def __init__(self, rows):
        self.rows, self.usage, self.calls = rows, Usage(), []

    async def ask_json(self, images, prompt, schema, name="result", effort=None):
        self.calls.append((len(images), effort, schema["properties"]["rows"]["items"]["required"]))
        return {"columns": ["No.", "Requirement", "Result"], "rows": self.rows}


def test_image_report_goes_through_the_model(tmp_path):
    p = tmp_path / "scan.png"
    Image.new("RGB", (800, 600), "white").save(p)
    empty = {"nominal": "", "plus_tol": "", "minus_tol": "", "lower": "", "upper": "", "nonconformance": ""}
    rows = [{**empty, "char_no": r[0], "requirement": r[1], "lower": r[2], "upper": r[3], "result": r[4],
             "accept": r[5], "nonconformance": r[6]} for r in CLEAN[:5]]
    client = FakeClient(rows)
    res = asyncio.run(check_report(small_job(), p, client))
    assert client.calls and client.calls[0][1] == "low" and "char_no" in client.calls[0][2]
    assert res["layout"]["source"] == "image"
    assert kinds(res) == ["missing_row"]  # 4B was not in the rows the model returned
    assert res["findings"][0]["drawing"]["row_id"] == "4-2"


def test_generated_reports_match_their_answers():
    rep = ROOT / "eval" / "supplier_reports"
    answers = json.loads((rep / "answers.json").read_text())
    jobs = {}
    for ans in answers:
        if not ans["file"].endswith(".xlsx") or ans["numbering"] != "drawing":
            continue
        job = jobs.setdefault(ans["job"], json.loads((ROOT / "samples" / "jobs" / ans["job"] / "job.json").read_text()))
        res = asyncio.run(check_report(job, rep / ans["file"]))
        assert len(res["findings"]) == len(ans["errors"]), ans["file"]
        assert sorted(f["kind"] for f in res["findings"]) == sorted(e["kind"] for e in ans["errors"]), ans["file"]
