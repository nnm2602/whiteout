"""Write data/MANIFEST.json: one entry per drawing file used by an answer key in eval/keys.

Run from the whiteout folder:  uv run python eval/make_manifest.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
KEYS = ROOT / "eval/keys"

APOLLO_LICENCE = ("Public Domain Mark 1.0 (http://creativecommons.org/publicdomain/mark/1.0/) per the archive.org item; "
                  "Grumman drawings held by NARA (RG 255), scanned and uploaded by the Virtual AGC project")
NIST_LICENCE = ("US Government work (17 USC 105), public domain; NIST asks for acknowledgement and does not allow use "
                "of the NIST logo")
MS_LICENCE = ("US Government work (17 USC 105); military standards are public documents. Copy from the everyspec.com "
              "mirror (adds a 'Downloaded from' line); official copy on ASSIST")
CADSY_LICENCE = "Apache License 2.0 (data/raw/cadsy/LICENSE); models from ADSKAILab/Zero-To-CAD-100k, Apache 2.0"
NIST_ZIP = {
    "nist_ftc": "https://www.nist.gov/system/files/documents/noindex/2022/07/19/NIST_MBE_PMI_FTC_Definitions.zip",
    "nist_ctc": "https://www.nist.gov/system/files/documents/noindex/2022/07/19/NIST_MBE_PMI_CTC_Definitions.zip",
    "nist_d2mi": "https://www.nist.gov/system/files/documents/el/msid/infotest/NIST-D2MI-Models.zip",
}


def describe(file: str) -> dict:
    p = ROOT / file
    folder = p.parent.name
    pages = len(fitz.open(p)) if p.suffix.lower() == ".pdf" else 1
    if folder == "apollo":
        m = re.match(r"apollo_b(\d+)_n(\d+)", p.stem)
        box, n = m.group(1), int(m.group(2))
        item = f"apertureCardBox{box}NARASW_images"
        return {"source_url": f"https://archive.org/download/{item}/page/n{n}.jpg",
                "source_item": f"https://archive.org/details/{item}", "licence": APOLLO_LICENCE,
                "kind": "scanned aperture card, 1-bit JPEG, Grumman Apollo LM drawing (1960s)", "pages": pages}
    if folder == "cadsy":
        return {"source_url": "https://huggingface.co/datasets/cadsy/cad-technical-drawings",
                "source_path": f"samples/{p.stem.rsplit('_', 1)[0]}/{p.stem.rsplit('_', 1)[1]}.png",
                "licence": CADSY_LICENCE, "kind": "synthetic CAD drawing, clean PNG 2480 x 1754, labelled", "pages": pages}
    if folder in ("nist_ftc", "nist_ctc", "nist_d2mi"):
        kind = {"nist_ftc": "NIST PMI fully toleranced test case drawing, vector PDF",
                "nist_ctc": "NIST PMI complex test case drawing, vector PDF",
                "nist_d2mi": "NIST D2MI mechanical reference dataset drawing, vector PDF (NX)"}[folder]
        return {"source_url": NIST_ZIP[folder], "source_path": p.name, "licence": NIST_LICENCE, "kind": kind,
                "pages": pages}
    if folder == "milstd":
        return {"source_url": "https://everyspec.com/MS-Specs/", "official_source": "https://quicksearch.dla.mil/",
                "document": p.stem, "licence": MS_LICENCE,
                "kind": "US military standard part sheet, scanned PDF", "pages": pages}
    raise ValueError(file)


def main() -> None:
    used: dict[str, dict] = {}
    keyed = defaultdict(set)
    stamp = {}
    for k in sorted(KEYS.glob("*.json")):
        d = json.loads(k.read_text())
        f = d["file"]
        keyed[f].add(d["page"])
        used.setdefault(f, d)
        if "stamp_not_for_photo_reproduction" in d:
            stamp[f] = d["stamp_not_for_photo_reproduction"]
    out = []
    for f in sorted(used):
        e = {"id": used[f]["id"], "file": f, **describe(f),
             "pages_keyed": sorted(keyed[f])}
        if f in stamp:
            e["stamp_not_for_photo_reproduction"] = stamp[f]
        out.append(e)
    (ROOT / "data/MANIFEST.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(len(out), "files")


if __name__ == "__main__":
    main()
