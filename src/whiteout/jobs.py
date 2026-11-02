"""Jobs: one uploaded drawing, processed in the background, saved to disk as JSON and PNGs."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import ingest
from .forms import form3_rows, write_workbook
from .model_client import ModelConfig, VisionClient
from .pipeline import SheetResult, process_sheet
from .balloon import number_and_place
from .render import ballooned
from .types import AuditFind, AuditPass, Characteristic, SheetInfo

JOBS_DIR = Path(os.environ.get("WHITEOUT_JOBS", Path(__file__).resolve().parents[2] / ".jobs"))
SAMPLE_JOBS = Path(os.environ.get("WHITEOUT_SAMPLE_JOBS", Path(__file__).resolve().parents[2] / "samples" / "jobs"))
MAX_SHEETS = int(os.environ.get("WHITEOUT_MAX_SHEETS", "6"))
MAX_FORKS = int(os.environ.get("WHITEOUT_MAX_FORKS", "300"))


@dataclass
class Job:
    id: str
    name: str
    dir: Path
    status: str = "queued"  # queued, running, done, error
    error: str = ""
    created: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)
    sheets: list[SheetResult] = field(default_factory=list)
    results: dict[str, str] = field(default_factory=dict)  # Form 3 row id -> measured value
    header: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    model: str = ""
    notes: list[str] = field(default_factory=list)
    revision_of: str = ""  # job id of the earlier revision, if any
    delta: dict = field(default_factory=dict)
    listeners: list[asyncio.Queue] = field(default_factory=list)

    # ---- persistence -------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "created": self.created,
            "model": self.model,
            "notes": self.notes,
            "usage": self.usage,
            "header": self.header,
            "results": self.results,
            "revision_of": self.revision_of,
            "delta": self.delta,
            "sheets": [
                {
                    "info": s.info.model_dump(),
                    "chars": [c.model_dump() for c in s.chars],
                    "audit": [a.model_dump() for a in s.audit],
                    "ignored": [a.model_dump() for a in s.ignored],
                    "timings": s.timings,
                    "residue_passes": len(s.residues),
                }
                for s in self.sheets
            ],
            "rows": form3_rows([c for s in self.sheets for c in s.chars], self.results),
        }

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = self.to_json()
        data["events"] = self.events
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(json.dumps(data))
        tmp.replace(self.dir / "job.json")

    @classmethod
    def load(cls, d: Path) -> "Job":
        data = json.loads((d / "job.json").read_text())
        j = cls(id=data["id"], name=data["name"], dir=d, status=data["status"], error=data.get("error", ""),
                created=data.get("created", 0), events=data.get("events", []), results=data.get("results", {}),
                header=data.get("header", {}), usage=data.get("usage", {}), model=data.get("model", ""),
                notes=data.get("notes", []), revision_of=data.get("revision_of", ""), delta=data.get("delta", {}))
        for s in data.get("sheets", []):
            r = SheetResult(info=SheetInfo(**s["info"]))
            r.chars = [Characteristic(**c) for c in s["chars"]]
            r.audit = [AuditPass(**a) for a in s["audit"]]
            r.ignored = [AuditFind(**a) for a in s["ignored"]]
            r.timings = s.get("timings", {})
            j.sheets.append(r)
        if j.status in ("queued", "running"):
            j.status, j.error = "error", "interrupted by a restart"
        return j

    async def emit(self, ev: dict) -> None:
        ev = {**ev, "t": round(time.time() - self.created, 2)}
        self.events.append(ev)
        for q in list(self.listeners):
            q.put_nowait(ev)


class JobStore:
    def __init__(self, root: Path = JOBS_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._forks: deque = deque()
        self._tasks: set[asyncio.Task] = set()

    def get(self, job_id: str) -> Job | None:
        if job_id in self.jobs:
            return self.jobs[job_id]
        if not job_id.replace("-", "").isalnum():
            return None
        for base in (self.root, SAMPLE_JOBS):
            d = base / job_id
            if (d / "job.json").exists():
                j = Job.load(d)
                self.jobs[job_id] = j
                return j
        return None

    def fork(self, job_id: str) -> Job:
        """A visitor's private copy of a saved job, so edits never reach the original. Images stay shared."""
        src = SAMPLE_JOBS / job_id
        jid = uuid.uuid4().hex[:12]
        d = self.root / jid
        d.mkdir(parents=True)
        for f in src.iterdir():
            if f.is_dir():
                shutil.copytree(f, d / f.name)
            elif f.name != "job.json" and (f.suffix == ".png" or f.stem == "source"):
                (d / f.name).symlink_to(f.resolve())
            elif f.name != "job.json":
                shutil.copy(f, d / f.name)
        data = json.loads((src / "job.json").read_text())
        data["id"] = jid
        (d / "job.json").write_text(json.dumps(data))
        (d / "sample_of").write_text(job_id)
        job = Job.load(d)
        self.jobs[jid] = job
        now = time.time()
        self._forks.append((now, jid))
        while len(self._forks) > MAX_FORKS or now - self._forks[0][0] > 86400:
            old = self._forks.popleft()[1]
            self.jobs.pop(old, None)
            shutil.rmtree(self.root / old, ignore_errors=True)
        return job

    def create(self, src: Path, name: str, cfg: ModelConfig | None = None, revision_of: str = "") -> Job:
        jid = uuid.uuid4().hex[:12]
        d = self.root / jid
        d.mkdir(parents=True)
        dst = d / ("source" + src.suffix.lower())
        shutil.copy(src, dst)
        job = Job(id=jid, name=name, dir=d, revision_of=revision_of)
        self.jobs[jid] = job
        job.save()
        t = asyncio.create_task(self._run(job, dst, cfg or ModelConfig()))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job: Job, path: Path, cfg: ModelConfig) -> None:
        job.status = "running"
        job.model = cfg.label
        client = VisionClient(cfg)
        try:
            sheets = await asyncio.to_thread(ingest.load, path, MAX_SHEETS)
            await job.emit({"type": "loaded", "sheets": [{"width": s.image.width, "height": s.image.height,
                                                          "notes": s.notes, "text_layer": bool(s.words)}
                                                         for s in sheets]})
            for i, s in enumerate(sheets):
                s.image.save(job.dir / f"sheet{i}.png", optimize=False)
                job.notes += [f"sheet {i + 1}: {n}" for n in s.notes]
            # Sheets run in parallel; balloons are numbered across the whole drawing once all are read.
            sem = asyncio.Semaphore(3)

            async def run_sheet(i: int, s):
                def save_residue(p: int, arr: np.ndarray, i: int = i) -> None:
                    Image.fromarray(arr).save(job.dir / (f"residue{i}_{p}.png" if p else f"residue{i}_final.png"))

                async with sem:
                    return await process_sheet(client, s, i, job.emit, on_residue=save_residue)

            results = await asyncio.gather(*(run_sheet(i, s) for i, s in enumerate(sheets)))
            n = 1
            for i, (s, r) in enumerate(zip(sheets, results)):
                n = number_and_place(r.chars, np.array(s.image), r.info, start=n)
                ballooned(np.array(s.image), r.chars, r.info).save(job.dir / f"ballooned{i}.png")
                job.sheets.append(r)
            job.usage = client.usage.as_dict()
            job.save()
            if job.revision_of:
                from .revision import compare_jobs

                base = STORE.get(job.revision_of)
                if base is not None:
                    job.delta = await asyncio.to_thread(compare_jobs, base, job)
                    for rid, v in job.delta.get("carried", {}).items():
                        job.results.setdefault(rid, v)
            job.status = "done"
            job.usage = client.usage.as_dict()
            await job.emit({"type": "done", "usage": job.usage})
        except Exception as e:  # noqa: BLE001 - surface any failure to the page
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
            await job.emit({"type": "error", "error": job.error})
        finally:
            job.usage = client.usage.as_dict()
            job.save()
            for i, r in enumerate(job.sheets):
                r.residues = []  # free memory; images are on disk

    def workbook(self, job: Job) -> Path:
        out = job.dir / "AS9102_forms.xlsx"
        write_workbook(job.sheets, out, job.header, job.results)
        return out


STORE = JobStore()
