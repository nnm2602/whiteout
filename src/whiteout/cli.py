"""Command line: whiteout run DRAWING [--out DIR] [--pages N] [--passes N]"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from . import ingest
from .forms import write_workbook
from .model_client import ModelConfig, VisionClient
from .pipeline import process_sheet
from .render import ballooned, residue_view


def offline_config() -> ModelConfig:
    import os

    url = os.environ.get("WHITEOUT_OFFLINE_URL", "")
    if not url:
        raise SystemExit("set WHITEOUT_OFFLINE_URL to the Glimmer server, e.g. http://HOST:8000/v1")
    if not os.environ.get("OFFLINE_API_KEY"):
        keyf = Path("~/.whiteout_offline_key").expanduser()
        if keyf.exists():
            os.environ["OFFLINE_API_KEY"] = keyf.read_text().strip()
    # WHITEOUT_OFFLINE_TILE=1344x672 (or 1344x672x300 with overlap): tile size the fine-tuned Glimmer was trained on
    tile = tuple(int(v) for v in os.environ.get("WHITEOUT_OFFLINE_TILE", "").lower().split("x") if v.strip())
    return ModelConfig(backend="openai", model="muse-glimmer", base_url=url, concurrency=2,
                       label="Muse Glimmer 30B (open weights, rented GPU)", tile=tile)


async def run(path: str, out: Path, pages: int, passes: int, effort: str, only: int | None = None,
              engine: str = "spark") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    client = VisionClient(offline_config() if engine == "offline" else ModelConfig(effort=effort))
    sheets = ingest.load(path, max_pages=pages)
    if only:
        sheets = [sheets[only - 1]]
    results = []
    for i, s in enumerate(sheets):
        def emit(ev):
            if ev["type"] in ("stage", "read_done", "audit_pass", "sheet_done"):
                print({k: v for k, v in ev.items() if k not in ("finds", "info", "tiles")}, flush=True)

        r = await process_sheet(client, s, i, emit, audit_passes=passes)
        results.append(r)
        import numpy as np

        ballooned(np.array(s.image), r.chars, r.info).save(out / f"sheet{i + 1}_ballooned.png")
        if r.residue is not None:
            residue_view(r.residue, r.chars, r.audit).save(out / f"sheet{i + 1}_residue.png")
    summary = {
        "source": str(path),
        "sheets": [
            {
                "info": r.info.model_dump(),
                "chars": [c.model_dump() for c in r.chars],
                "audit": [a.model_dump() for a in r.audit],
                "ignored": [a.model_dump() for a in r.ignored],
                "timings": r.timings,
            }
            for r in results
        ],
        "usage": client.usage.as_dict(),
    }
    (out / "result.json").write_text(json.dumps(summary, indent=1))
    write_workbook(results, out / "AS9102_forms.xlsx")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="whiteout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("drawing")
    r.add_argument("--out", default="out")
    r.add_argument("--pages", type=int, default=12)
    r.add_argument("--passes", type=int, default=3)
    r.add_argument("--effort", default="medium")
    r.add_argument("--sheet", type=int, default=None)
    r.add_argument("--engine", choices=["spark", "offline"], default="spark")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    if a.cmd == "run":
        summary = asyncio.run(run(a.drawing, Path(a.out), a.pages, a.passes, a.effort, a.sheet, a.engine))
        n = sum(len(s["chars"]) for s in summary["sheets"])
        print(f"{n} characteristics; usage {summary['usage']}")
    else:
        import uvicorn

        uvicorn.run("whiteout.server:app", host="0.0.0.0", port=a.port)
