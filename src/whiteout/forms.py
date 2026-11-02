"""AS9102 Rev C Forms 1, 2 and 3 as an Excel workbook, plus the Form 3 rows the app shows."""

from __future__ import annotations

import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .tolerance import evaluate
from .types import Characteristic


def _num_key(c: Characteristic):
    try:
        return (c.sheet, int(c.number))
    except ValueError:
        return (c.sheet, 10**6)


def fmt(v: float | None) -> str:
    if v is None:
        return ""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _no_range_limits(q):
    """A negative lower limit is a bare +/- rule for a range of sizes (".016R TO .062R +/- .010"), not a size limit."""
    if q.lower is not None and q.lower < 0:
        return q.model_copy(update={"lower": None, "upper": None})
    return q


def drawing_revision(sheet_results) -> str:
    """Sheets can carry different revision letters; the drawing is at the latest one."""
    revs = [s.info.revision for s in sheet_results if s.info.revision]
    letters = [r.strip().upper() for r in revs if r.strip().isalpha()]
    return max(letters, key=lambda r: (len(r), r)) if letters else (revs[0] if revs else "")


def form3_rows(chars: list[Characteristic], results: dict | None = None) -> list[dict]:
    """One row per requirement per instance. results maps row_id -> measured text."""
    results = results or {}
    rows = []
    for c in sorted((c for c in chars if not c.ignored and c.number), key=_num_key):
        reqs = c.requirements or []
        for qi, q in enumerate(reqs):
            for inst in range(1, c.multiplier + 1):
                rid = c.number if len(reqs) == 1 else f"{c.number}.{qi + 1}"
                if c.multiplier > 1:
                    rid += f"-{inst}"
                measured = results.get(rid, "")
                q = _no_range_limits(q)
                designator = ""
                rows.append({
                    "row_id": rid,
                    "char_id": c.id,
                    "number": c.number,
                    "sheet": c.sheet + 1,
                    "zone": c.zone,
                    "designator": designator,
                    "requirement": q.text if len(reqs) > 1 else c.text,
                    "label": q.label,
                    "type": q.type,
                    "lower": fmt(q.lower),
                    "upper": fmt(q.upper),
                    "tolerance_source": q.tolerance_source,
                    "attribute": q.attribute,
                    "instance": f"{inst} of {c.multiplier}" if c.multiplier > 1 else "",
                    "found_in": c.found_in,
                    "flags": sorted(set(c.flags + q.flags)),
                    "measured": measured,
                    "status": evaluate(q, measured),
                })
    return rows


HEAD_FILL = PatternFill("solid", fgColor="DCE6F2")
BOLD = Font(bold=True)


def write_workbook(sheet_results, path: Path, header: dict | None = None, results: dict | None = None) -> None:
    header = header or {}
    wb = Workbook()
    info = sheet_results[0].info if sheet_results else None

    f1 = wb.active
    f1.title = "Form 1"
    f1.append(["AS9102 Rev C  Form 1: Part Number Accountability"])
    f1["A1"].font = Font(bold=True, size=13)
    fields = [
        ("1. Part Number", (info.part_number or info.drawing_number) if info else ""),
        ("2. Part Name", info.part_name if info else ""),
        ("3. Serial Number", header.get("serial", "")),
        ("4. FAI Report Number", header.get("report", "")),
        ("5. Part Revision Level", drawing_revision(sheet_results)),
        ("6. Drawing Number", info.drawing_number if info else ""),
        ("7. Drawing Revision Level", drawing_revision(sheet_results)),
        ("8. Additional Changes", header.get("changes", "")),
        ("9. Manufacturing Process Reference", header.get("process", "")),
        ("10. Organization Name", header.get("org", "")),
        ("11. Supplier Code", header.get("supplier", "")),
        ("12. P.O. Number", header.get("po", "")),
        ("13. Detail FAI / Assembly FAI", header.get("detail", "Detail FAI")),
        ("14. Full FAI / Partial FAI", header.get("full", "Full FAI")),
        ("    Reason for partial", header.get("partial_reason", "")),
        ("19. FAI complete / not complete", header.get("complete", "")),
    ]
    for k, v in fields:
        f1.append([k, v])
    f1.column_dimensions["A"].width = 38
    f1.column_dimensions["B"].width = 40

    f2 = wb.create_sheet("Form 2")
    f2.append(["AS9102 Rev C  Form 2: Product Accountability (raw material, special processes, functional testing)"])
    f2["A1"].font = Font(bold=True, size=13)
    f2.append(["6. Material or Process Name", "7. Specification Number", "8. Code", "9. Special Process Supplier Code",
               "10. Customer Approval Verification", "11. Certificate of Conformance Number", "Source on drawing"])
    for c in f2[2]:
        c.font, c.fill = BOLD, HEAD_FILL
    for r in sheet_results:
        if r.info.material:
            f2.append([r.info.material, _spec(r.info.material), "", "", "", "", "title block"])
        if r.info.finish:
            f2.append([r.info.finish, _spec(r.info.finish), "", "", "", "", "title block"])
        for c in r.chars:
            for q in c.requirements:
                if q.type == "note" and re.search(r"\b(MIL|AMS|ASTM|ANSI|QQ|SAE|NAS|PLATE|ANODIZ|PASSIVAT|HEAT TREAT|PAINT|PRIME)", c.text, re.I):
                    f2.append([c.text[:120], _spec(c.text), "", "", "", "", f"note, balloon {c.number}"])
    for i, w in enumerate([46, 24, 10, 18, 18, 22, 18], start=1):
        f2.column_dimensions[get_column_letter(i)].width = w

    f3 = wb.create_sheet("Form 3")
    f3.append(["AS9102 Rev C  Form 3: Characteristic Accountability, Verification and Compatibility Evaluation"])
    f3["A1"].font = Font(bold=True, size=13)
    head = ["5. Char. No.", "6. Reference Location", "7. Characteristic Designator", "8. Requirement", "9. Results",
            "10. Designed / Qualified Tooling", "11. Nonconformance Number", "14. Additional Data / Comments",
            "Lower limit", "Upper limit", "Tolerance from", "Status", "Sheet", "Found by", "Check"]
    f3.append(head)
    for c in f3[2]:
        c.font, c.fill = BOLD, HEAD_FILL
        c.alignment = Alignment(wrap_text=True, vertical="top")
    all_chars = [c for r in sheet_results for c in r.chars]
    for row in form3_rows(all_chars, results):
        req = row["requirement"] + (f"  ({row['instance']})" if row["instance"] else "")
        f3.append([row["row_id"], row["zone"], row["designator"], req, row["measured"], "", "", row["label"],
                   row["lower"], row["upper"], row["tolerance_source"], row["status"], row["sheet"],
                   "erased-page search" if row["found_in"].startswith("audit") else ("text layer" if row["found_in"] == "text-layer" else "first read"),
                   "; ".join(row["flags"])])
    for i, w in enumerate([10, 10, 12, 44, 12, 14, 14, 24, 11, 11, 12, 10, 7, 16, 40], start=1):
        f3.column_dimensions[get_column_letter(i)].width = w
    f3.freeze_panes = "A3"
    wb.save(path)


def _spec(text: str) -> str:
    m = re.search(r"\b(?:MIL|AMS|ASTM|ANSI|QQ|SAE|NAS|AMS-QQ)[-\s]?[A-Z0-9][A-Z0-9.\-/]*", text or "", re.I)
    return m.group(0) if m else ""
