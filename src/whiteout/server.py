"""HTTP API and the web page."""

from __future__ import annotations

import asyncio
import calendar
import contextvars
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .forms import form3_rows
from .jobs import STORE, Job
from .model_client import ModelConfig
from .types import Characteristic

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
SAMPLES = Path(os.environ.get("WHITEOUT_SAMPLES", ROOT.parents[1] / "samples"))
UPLOAD_LIMIT = 25 * 1024 * 1024
UPLOADS_PER_HOUR = int(os.environ.get("WHITEOUT_UPLOADS_PER_HOUR", "6"))
TOTAL_UPLOADS_PER_HOUR = int(os.environ.get("WHITEOUT_TOTAL_UPLOADS_PER_HOUR", "20"))
DAILY_USD = float(os.environ.get("WHITEOUT_DAILY_USD", "12"))  # protects the model credit on the public demo
BUDGET_START = os.environ.get("WHITEOUT_BUDGET_START", "")  # first budget day, YYYY-MM-DD in UTC
BUDGET_DAYS = int(os.environ.get("WHITEOUT_BUDGET_DAYS", "7"))  # the allowance stops growing after this many days
BUDGET_DIR = os.environ.get("WHITEOUT_BUDGET_DIR", "")  # lasting storage, so a restart does not forget the spend
CREDIT_USD = float(os.environ.get("WHITEOUT_CREDIT_USD", "0"))  # model credit left when the demo opened; 0 turns this alert off
VISITOR_USD = float(os.environ.get("WHITEOUT_VISITOR_USD", "50"))  # per address per day, across both apps
FAST_USD = float(os.environ.get("WHITEOUT_FAST_USD", "5"))  # an alert when this much goes within an hour
OFFLINE_URL = os.environ.get("WHITEOUT_OFFLINE_URL", "")  # OpenAI-compatible Glimmer server, if running

app = FastAPI(title="Whiteout")


class _KeepSamples:
    """Saved samples are read only: nothing but a read may touch the originals."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] not in ("GET", "HEAD"):
            m = re.match(r"/api/jobs/([^/]+)", scope["path"])
            if m and any(s.get("job") == m.group(1) for s in _samples()):
                await JSONResponse({"error": "this saved sample is read only"}, status_code=403)(scope, receive, send)
                return
        await self.inner(scope, receive, send)


app.add_middleware(_KeepSamples)
_recent: dict[str, deque] = defaultdict(deque)
_all_uploads: deque = deque()
_asks: dict[str, deque] = defaultdict(deque)
ASKS_PER_HOUR = int(os.environ.get("WHITEOUT_ASKS_PER_HOUR", "40"))


_start = calendar.timegm(time.strptime(BUDGET_START, "%Y-%m-%d")) if BUDGET_START else time.time()
_ledger_file = Path(BUDGET_DIR) / (os.environ.get("K_SERVICE", "local") + ".json") if BUDGET_DIR else None
_ledger: dict = {"jobs": {}, "calls": 0.0}
_ledger_ok = True
_saved = ""
_ledger_lock = threading.Lock()
_history: deque = deque()
if _ledger_file:
    try:
        if _ledger_file.exists():
            _ledger = json.loads(_ledger_file.read_text())
    except (OSError, ValueError):
        _ledger_ok = False  # spend unknown, so paid calls stay off until this is looked at


def _allowance() -> float:
    """Unused money carries over: the allowance grows by one day's amount each day, up to BUDGET_DAYS days."""
    days = int((time.time() - _start) // 86400) + 1
    return DAILY_USD * min(BUDGET_DAYS, max(1, days))


def _spent() -> float:
    global _ledger_ok, _saved
    with _ledger_lock:
        for j in list(STORE.jobs.values()):
            usd = float(j.usage.get("usd", 0) or 0)
            if usd and j.created >= _start:
                _ledger["jobs"][j.id] = max(usd, _ledger["jobs"].get(j.id, 0))
        now = json.dumps(_ledger, sort_keys=True)
        if _ledger_file and _ledger_ok and now != _saved:
            try:
                tmp = _ledger_file.with_suffix(".tmp")
                tmp.write_text(now)
                tmp.replace(_ledger_file)
                _saved = now
            except OSError:
                _ledger_ok = False
        return sum(_ledger["jobs"].values()) + _ledger["calls"]


def _budget_left() -> float:
    return _allowance() - _spent() if _ledger_ok else 0.0


_visitor: contextvars.ContextVar[str] = contextvars.ContextVar("visitor", default="")


def _visitor_key(req: Request) -> str:
    """One visitor for one day: the address is kept only as a hash."""
    ip = hashlib.sha256(_client_ip(req).encode()).hexdigest()[:16]
    return f"{int(time.time() // 86400)}:{ip}"


def _visitor_spent(key: str) -> float:
    _spent()

    def of(d: dict) -> float:
        jobs = sum(d.get("jobs", {}).get(j, 0) for j, o in d.get("job_owner", {}).items() if o == key)
        return jobs + d.get("visitor_calls", {}).get(key, 0)

    total = of(_ledger)
    if _ledger_file:
        for f in _ledger_file.parent.glob("*.json"):
            if f != _ledger_file:
                try:
                    total += of(json.loads(f.read_text()))
                except (OSError, ValueError):
                    pass
    return total


def _visitor_ok(req: Request) -> str:
    key = _visitor_key(req)
    _visitor.set(key)
    if _visitor_spent(key) >= VISITOR_USD:
        _alert(f"visitor-{key}", f"One address reached its ${VISITOR_USD:.2f} daily limit. It is blocked until midnight UTC; everyone else is unaffected.")
        raise HTTPException(429, "this connection has reached today's limit; samples still work, and the limit resets at midnight UTC")
    return key


def _alert(key: str, message: str) -> None:
    """One log line per event, once ever; a log-based alert in Cloud Monitoring turns it into an email."""
    if key not in _ledger.setdefault("alerted", []):
        _ledger["alerted"].append(key)
        print(f"WHITEOUT_ALERT {message}", flush=True)


def _check_alerts() -> None:
    service = os.environ.get("K_SERVICE", "local")
    now = time.time()
    if not _ledger_ok:
        _alert("ledger", f"{service}: the spend record cannot be read or saved, so uploads and questions are off.")
    spent = _spent()
    left = _allowance() - spent
    if left <= 0:
        _alert(f"empty-{int(now // 86400)}", f"{service}: the model budget is used up (${spent:.2f} of ${_allowance():.2f}). "
               "Uploads and questions are off until the next top-up at midnight UTC. Samples still work.")
    _history.append((now, spent))
    while now - _history[0][0] > 3600:
        _history.popleft()
    if spent - _history[0][1] >= FAST_USD:
        _alert(f"fast-{int(now // 3600)}", f"{service}: ${spent - _history[0][1]:.2f} was spent in the last hour "
               f"(${spent:.2f} so far, ${max(left, 0):.2f} left).")
    if CREDIT_USD and _ledger_file:
        total = 0.0
        for f in _ledger_file.parent.glob("*.json"):
            d = json.loads(f.read_text())
            total += sum(d.get("jobs", {}).values()) + d.get("calls", 0)
        for pct in (50, 80, 100):
            if total >= CREDIT_USD * pct / 100:
                _alert(f"credit-{pct}", f"The demo has spent ${total:.2f}, {pct}% of the ${CREDIT_USD:.2f} free Meta credit. "
                       "After the credit, usage bills to the card on the Meta account.")
    _spent()


def _save_loop() -> None:
    while True:
        time.sleep(30)
        try:
            _check_alerts()
        except Exception as e:  # noqa: BLE001  the loop must outlive a bad read
            print(f"alert check failed: {e}", flush=True)


if _ledger_file:
    threading.Thread(target=_save_loop, daemon=True).start()


def _model_call_ok(req: Request) -> None:
    """Budget and per-visitor limit for the small model calls the page makes."""
    if _budget_left() <= 0:
        raise HTTPException(429, "the model budget for the public demo is used up for now")
    _visitor_ok(req)
    q = _asks[_client_ip(req)]
    now = time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= ASKS_PER_HOUR:
        raise HTTPException(429, "question limit reached for this hour")
    q.append(now)


def _record(client) -> None:
    if client.usage.usd:
        with _ledger_lock:
            _ledger["calls"] += client.usage.usd
            key = _visitor.get()
            if key:
                calls = _ledger.setdefault("visitor_calls", {})
                calls[key] = calls.get(key, 0) + client.usage.usd
        _spent()


def _client_ip(req: Request) -> str:
    # Cloud Run appends the real address, so only the last entry is trustworthy; a visitor can write the earlier ones.
    fwd = req.headers.get("x-forwarded-for", "")
    return fwd.split(",")[-1].strip() if fwd else (req.client.host if req.client else "?")


def _rate_ok(ip: str) -> bool:
    q = _recent[ip]
    now = time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= UPLOADS_PER_HOUR:
        return False
    q.append(now)
    return True


def _total_ok() -> bool:
    """A limit across all visitors, for anyone using many addresses."""
    now = time.time()
    while _all_uploads and now - _all_uploads[0] > 3600:
        _all_uploads.popleft()
    if len(_all_uploads) >= TOTAL_UPLOADS_PER_HOUR:
        return False
    _all_uploads.append(now)
    return True


def _sample_key(job_id: str) -> str:
    """The saved sample a private copy came from, for finding that sample's example files."""
    j = STORE.get(job_id)
    marker = j.dir / "sample_of" if j else None
    return marker.read_text().strip() if marker and marker.exists() else job_id


def _job(job_id: str) -> Job:
    j = STORE.get(job_id)
    if j is None:
        raise HTTPException(404, "no such job")
    return j


def _model_cfg(engine: str) -> ModelConfig:
    if engine == "offline":
        if not OFFLINE_URL:
            raise HTTPException(400, "the offline Glimmer server is not running right now")
        return ModelConfig(backend="openai", model=os.environ.get("WHITEOUT_OFFLINE_MODEL", "muse-glimmer"),
                           base_url=OFFLINE_URL, concurrency=2, timeout=1500, extra={"max_tokens": 6000},
                           label="Muse Glimmer 30B (open weights, rented GPU)")
    return ModelConfig()


# ---- samples ----------------------------------------------------------------------------------------------------
def _samples() -> list[dict]:
    path = SAMPLES / "samples.json"
    return json.loads(path.read_text()) if path.exists() else []


@app.get("/api/samples")
def samples():
    return _samples()


@app.get("/api/samples/{sid}/thumb.png")
def sample_thumb(sid: str):
    s = next((s for s in _samples() if s["id"] == sid), None)
    if not s:
        raise HTTPException(404)
    return FileResponse(SAMPLES / s["thumb"])


# ---- jobs -------------------------------------------------------------------------------------------------------
@app.post("/api/jobs")
async def create_job(request: Request, file: UploadFile | None = File(None), sample: str = Form(""),
                     engine: str = Form("spark"), revision_of: str = Form("")):
    if sample:
        s = next((s for s in _samples() if s["id"] == sample), None)
        if not s:
            raise HTTPException(404, "unknown sample")
        # Samples are precomputed: hand back a private copy so the page is instant, costs nothing and
        # one visitor's edits never reach the next visitor.
        if s.get("job") and STORE.get(s["job"]):
            return {"id": STORE.fork(s["job"]).id, "cached": True}
        src, name = SAMPLES / s["file"], s["name"]
    else:
        if file is None:
            raise HTTPException(400, "send a file or a sample id")
        if _budget_left() <= 0:
            raise HTTPException(429, "the model budget for the public demo is used up for now; open a sample drawing")
        owner = _visitor_ok(request)
        if not _rate_ok(_client_ip(request)):
            raise HTTPException(429, "upload limit reached for this hour; try a sample drawing")
        data = await file.read()
        if len(data) > UPLOAD_LIMIT:
            raise HTTPException(413, "file over 25 MB")
        suffix = Path(file.filename or "x.pdf").suffix.lower()
        if suffix not in (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
            raise HTTPException(400, "PDF, PNG, JPEG or TIFF only")
        if not _total_ok():
            raise HTTPException(429, "the demo's upload limit is reached for this hour; try a sample drawing")
        tmp = Path(tempfile.mkdtemp()) / ("upload" + suffix)
        tmp.write_bytes(data)
        src, name = tmp, re.sub(r"[^\w .()-]", "", file.filename or "drawing")[:80]
    if revision_of:
        _job(revision_of)
    job = STORE.create(src, name, _model_cfg(engine), revision_of=revision_of)
    if not sample:
        with _ledger_lock:
            _ledger.setdefault("job_owner", {})[job.id] = owner
    return {"id": job.id, "cached": False}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job(job_id).to_json()


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, since: int = 0):
    job = _job(job_id)

    async def gen():
        q: asyncio.Queue = asyncio.Queue()
        for ev in job.events[since:]:
            yield f"data: {json.dumps(ev)}\n\n"
        if job.status in ("done", "error"):
            yield "data: {\"type\": \"end\"}\n\n"
            return
        job.listeners.append(q)
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("type") in ("done", "error"):
                    yield "data: {\"type\": \"end\"}\n\n"
                    return
        finally:
            job.listeners.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


_IMG = re.compile(r"^(sheet\d+|ballooned\d+|residue\d+_(?:\d+|final))\.png$")


@app.get("/api/jobs/{job_id}/img/{name}")
def job_image(job_id: str, name: str):
    job = _job(job_id)
    if not _IMG.match(name) or not (job.dir / name).exists():
        raise HTTPException(404)
    return FileResponse(job.dir / name, headers={"Cache-Control": "public, max-age=86400"})


@app.put("/api/jobs/{job_id}/results")
async def put_results(job_id: str, request: Request):
    job = _job(job_id)
    body = await request.json()
    for k, v in (body.get("results") or {}).items():
        job.results[str(k)[:20]] = str(v)[:40]
    if "header" in body:
        job.header.update({str(k)[:40]: str(v)[:200] for k, v in body["header"].items()})
    job.save()
    return {"rows": form3_rows([c for s in job.sheets for c in s.chars], job.results)}


@app.post("/api/jobs/{job_id}/cmm")
async def import_cmm(job_id: str, file: UploadFile = File(...)):
    """CSV with a balloon / characteristic number column and a measured value column."""
    job = _job(job_id)
    text = (await file.read()).decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise HTTPException(400, "empty file")
    head = [h.strip().lower() for h in rows[0]]
    ci = next((i for i, h in enumerate(head) if re.search(r"char|balloon|feature|no\b|id", h)), 0)
    vi = next((i for i, h in enumerate(head) if re.search(r"meas|actual|value|result", h)), 1)
    start = 1 if not re.match(r"^\s*[\d.]+", rows[0][ci] if len(rows[0]) > ci else "") else 0
    n = 0
    valid = {r["row_id"] for r in form3_rows([c for s in job.sheets for c in s.chars])}
    for r in rows[start:]:
        if len(r) <= max(ci, vi):
            continue
        rid = r[ci].strip()
        if rid in valid:
            job.results[rid] = r[vi].strip()
            n += 1
    job.save()
    return {"imported": n, "rows": form3_rows([c for s in job.sheets for c in s.chars], job.results)}


@app.patch("/api/jobs/{job_id}/chars/{char_id}")
async def patch_char(job_id: str, char_id: str, request: Request):
    """The inspector corrects a characteristic: ignore it, restore it, or fix its text."""
    job = _job(job_id)
    body = await request.json()
    for s in job.sheets:
        for c in s.chars:
            if c.id == char_id:
                if "ignored" in body:
                    c.ignored = bool(body["ignored"])
                if "text" in body:
                    from .tolerance import build_requirement

                    c.text = str(body["text"])[:200]
                    c.requirements = [build_requirement({"type": "other", "text": c.text}, c.text,
                                                        s.info.general_tolerances)]
                    c.flags = ["edited by inspector"]
                job.save()
                (job.dir / "plan.json").unlink(missing_ok=True)
                return {"ok": True, "rows": form3_rows([x for sh in job.sheets for x in sh.chars], job.results)}
    raise HTTPException(404, "no such characteristic")


@app.post("/api/jobs/{job_id}/chars")
async def add_char(job_id: str, request: Request):
    """The inspector adds a characteristic the system missed: a box drawn on the sheet plus its text."""
    from .balloon import number_and_place
    from .tolerance import build_requirement
    import numpy as np
    from PIL import Image

    job = _job(job_id)
    body = await request.json()
    si = int(body.get("sheet", 0))
    s = job.sheets[si]
    box = tuple(int(v) for v in body["box"])
    text = str(body.get("text", ""))[:200]
    c = Characteristic(id=f"s{si}m{len(s.chars)}", text=text, box=box, sheet=si, found_in="manual")
    c.requirements = [build_requirement({"type": "other", "text": text}, text, s.info.general_tolerances)]
    s.chars.append(c)
    gray = np.array(Image.open(job.dir / f"sheet{si}.png").convert("L"))
    n0 = max([int(x.number) for x in s.chars if x.number.isdigit()] or [0])
    number_and_place([c], gray, s.info, start=n0 + 1)
    job.save()
    (job.dir / "plan.json").unlink(missing_ok=True)
    return {"id": c.id}


@app.get("/api/jobs/{job_id}/forms.xlsx")
def forms(job_id: str):
    job = _job(job_id)
    path = STORE.workbook(job)
    stem = re.sub(r"[^\w-]", "_", Path(job.name).stem)[:40] or "drawing"
    return FileResponse(path, filename=f"{stem}_AS9102.xlsx")


@app.get("/api/jobs/{job_id}/ballooned.pdf")
def ballooned_pdf(job_id: str):
    import numpy as np
    from PIL import Image

    from .render import ballooned

    job = _job(job_id)
    pages = []
    for i, s in enumerate(job.sheets):
        gray = np.array(Image.open(job.dir / f"sheet{i}.png").convert("L"))
        pages.append(ballooned(gray, s.chars, s.info))
    out = job.dir / "ballooned.pdf"
    if pages:
        pages[0].save(out, save_all=True, append_images=pages[1:], resolution=200.0)
    stem = re.sub(r"[^\w-]", "_", Path(job.name).stem)[:40] or "drawing"
    return FileResponse(out, filename=f"{stem}_ballooned.pdf")


@app.post("/api/jobs/{job_id}/voice")
async def voice(job_id: str, request: Request, audio: UploadFile = File(...), row: str = Form("")):
    """Push-to-talk: 'balloon 14, point two five one two' fills a measured value. In guided measuring the line is
    already chosen (row), so the inspector says only the value."""
    from .voice import parse_dictation, parse_value, transcribe

    job = _job(job_id)
    _model_call_ok(request)
    wav = await audio.read()
    if len(wav) > 8 * 1024 * 1024:
        raise HTTPException(413, "recording too long")
    text = await transcribe(wav)
    valid = [r["row_id"] for r in form3_rows([c for s in job.sheets for c in s.chars])]
    if row:
        if row not in valid:
            raise HTTPException(400, "no such Form 3 line")
        v = parse_value(text)
        entries = [(row, v)] if v is not None else []
    else:
        entries = parse_dictation(text, valid)
    for rid, v in entries:
        job.results[rid] = v
    job.save()
    return {"transcript": text, "entries": entries,
            "rows": form3_rows([c for s in job.sheets for c in s.chars], job.results)}


def _units(job: Job) -> str:
    return (job.sheets[0].info.units if job.sheets else "in") or "in"


@app.get("/api/jobs/{job_id}/plan")
async def plan(job_id: str, request: Request):
    """Measuring plan: instrument, method and setup for every Form 3 line."""
    from .model_client import VisionClient
    from .planner import make_plan

    job = _job(job_id)
    cached = job.dir / "plan.json"
    if cached.exists():
        return json.loads(cached.read_text())
    _model_call_ok(request)
    client = VisionClient()
    p = await make_plan(job.to_json(), client)
    _record(client)
    try:
        cached.write_text(json.dumps(p))
    except OSError:
        pass
    return p


@app.get("/api/jobs/{job_id}/program.dmi")
async def dmis_program(job_id: str, request: Request):
    """A starting CMM program (DMIS) for the lines the plan sends to the CMM."""
    from .dmis import make_program

    job = _job(job_id)
    p = await plan(job_id, request)
    holes = []
    mfile = job.dir / "model.json"
    if mfile.exists():
        holes = json.loads(mfile.read_text()).get("holes", [])
    prog = make_program(job.to_json(), p, holes)
    stem = re.sub(r"[^\w-]", "_", Path(job.name).stem)[:40] or "drawing"
    path = job.dir / "program.dmi"
    path.write_text(prog["program"])
    return FileResponse(path, filename=f"{stem}_CMM.dmi", media_type="text/plain")


@app.post("/api/jobs/{job_id}/gauge")
async def gauge(job_id: str, request: Request, photo: UploadFile = File(...), row: str = Form("")):
    """A phone photo of a caliper or micrometer display: Muse Spark reads it; the value goes on the chosen line."""
    from .gauge import read_gauge
    from .model_client import VisionClient

    job = _job(job_id)
    _model_call_ok(request)
    data = await photo.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "photo over 15 MB")
    client = VisionClient()
    try:
        r = await read_gauge(data, client, _units(job))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not read the photo: {type(e).__name__}")
    _record(client)
    rows = form3_rows([c for s in job.sheets for c in s.chars], job.results)
    if row and r["sure"] and r["value_text"] and any(x["row_id"] == row for x in rows):
        job.results[row] = r["value_text"]
        job.save()
        rows = form3_rows([c for s in job.sheets for c in s.chars], job.results)
        r["written"] = True
    else:
        r["written"] = False
    return {"reading": r, "rows": rows}


@app.post("/api/jobs/{job_id}/ask")
async def ask_drawing(job_id: str, request: Request):
    """Ask the drawing a question; a stack-up is added up in code."""
    from .ask import ask
    from .model_client import VisionClient

    job = _job(job_id)
    body = await request.json()
    q = str(body.get("question", "")).strip()
    if not q:
        raise HTTPException(400, "ask a question")
    _model_call_ok(request)
    client = VisionClient()
    r = await ask(job.to_json(), job.dir, q, client, body.get("history") or [])
    _record(client)
    return r


@app.get("/api/jobs/{job_id}/supplier-samples")
def supplier_samples(job_id: str):
    d = SAMPLES / "supplier" / _sample_key(job_id)
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


@app.post("/api/jobs/{job_id}/supplier-report")
async def supplier_report(job_id: str, request: Request, file: UploadFile | None = File(None), sample: str = Form("")):
    """Check a supplier's AS9102 report (xlsx, PDF or photo of the table) against this drawing."""
    from .fair_check import check_report
    from .model_client import VisionClient

    job = _job(job_id)
    if sample:
        path = SAMPLES / "supplier" / _sample_key(job_id) / Path(sample).name
        if not path.is_file():
            raise HTTPException(404, "no such sample report")
    else:
        if file is None:
            raise HTTPException(400, "send a report")
        data = await file.read()
        if len(data) > UPLOAD_LIMIT:
            raise HTTPException(413, "file over 25 MB")
        suffix = Path(file.filename or "x.xlsx").suffix.lower()
        if suffix not in (".xlsx", ".xlsm", ".pdf", ".png", ".jpg", ".jpeg"):
            raise HTTPException(400, "Excel, PDF, PNG or JPEG only")
        path = Path(tempfile.mkdtemp()) / ("report" + suffix)
        path.write_bytes(data)
    client = None
    if path.suffix.lower() not in (".xlsx", ".xlsm"):
        _model_call_ok(request)
        client = VisionClient()
    try:
        r = await check_report(job.to_json(), path, client)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not read the report: {type(e).__name__}: {e}"[:300])
    if client is not None:
        _record(client)
    r["report"] = Path(sample).name if sample else (file.filename if file else "")
    return r


@app.get("/api/jobs/{job_id}/model-samples")
def model_samples(job_id: str):
    d = SAMPLES / "models" / _sample_key(job_id)
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


@app.post("/api/jobs/{job_id}/model")
async def model_check(job_id: str, file: UploadFile | None = File(None), sample: str = Form("")):
    """Compare the drawing with the part's STEP AP242 model; keep its holes for the CMM program."""
    from .model_check import compare, read_step

    job = _job(job_id)
    if sample:
        path = SAMPLES / "models" / _sample_key(job_id) / Path(sample).name
        if not path.is_file():
            raise HTTPException(404, "no such sample model")
    else:
        if file is None:
            raise HTTPException(400, "send a STEP file")
        data = await file.read()
        if len(data) > 60 * 1024 * 1024:
            raise HTTPException(413, "file over 60 MB")
        if Path(file.filename or "").suffix.lower() not in (".stp", ".step"):
            raise HTTPException(400, "STEP (.stp or .step) only")
        path = Path(tempfile.mkdtemp()) / "model.stp"
        path.write_bytes(data)
    try:
        model = await asyncio.to_thread(read_step, path)
        chars = [c for s in job.sheets for c in s.chars if not c.ignored]
        r = await asyncio.to_thread(compare, model, chars, _units(job))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not read the model: {type(e).__name__}: {e}"[:300])
    k = model["units"]["mm_per_unit"] / (25.4 if _units(job) == "in" else 1.0)
    holes = [{"id": h["id"], "xyz": [v * k for v in h["xyz"]], "axis": h["axis"], "diameter": h["diameter"] * k,
              "depth": (h.get("depth") or 0) * k} for h in model.get("holes", [])]
    (job.dir / "model.json").write_text(json.dumps({"file": Path(sample or (file.filename if file else "")).name,
                                                    "holes": holes, "summary": r["summary"]}))
    r["holes"] = holes
    r["model_file"] = Path(sample).name if sample else (file.filename if file else "")
    r["counts"] = model.get("counts", {})
    return r


def _redraw_dir(job: Job, sheet: int) -> Path:
    if not 0 <= sheet < len(job.sheets):
        raise HTTPException(404, "no such sheet")
    return job.dir / f"redraw{sheet}"


def _redraw_info(d: Path) -> dict:
    stats = json.loads((d / "stats.json").read_text())
    proof = json.loads((d / "proof.json").read_text()) if (d / "proof.json").exists() else []
    return {"region": stats["region"], "faithfulness": stats.get("faithfulness", {}),
            "placed": stats.get("text", {}).get("placed", 0), "differences": proof}


@app.get("/api/jobs/{job_id}/redraw/{sheet}")
def redraw_get(job_id: str, sheet: int):
    d = _redraw_dir(_job(job_id), sheet)
    if not (d / "stats.json").exists():
        raise HTTPException(404, "not redrawn yet")
    return _redraw_info(d)


@app.get("/api/jobs/{job_id}/redraw/{sheet}/{name}")
def redraw_file(job_id: str, sheet: int, name: str):
    if name not in ("redraw.png", "redraw.pdf", "redraw.svg"):
        raise HTTPException(404, "no such file")
    p = _redraw_dir(_job(job_id), sheet) / name
    if not p.exists():
        raise HTTPException(404, "not redrawn yet")
    return FileResponse(p, filename=f"redrawn_sheet{sheet + 1}{Path(name).suffix}" if name != "redraw.png" else None)


@app.post("/api/jobs/{job_id}/redraw/{sheet}")
async def redraw_post(job_id: str, sheet: int, request: Request):
    """Redraw a scanned sheet as a clean vector drawing, then proofread it against the scan with Muse Spark."""
    import numpy as np
    from PIL import Image

    from .model_client import VisionClient
    from .proof import proofread_dir
    from .redraw import redraw

    job = _job(job_id)
    d = _redraw_dir(job, sheet)
    if (d / "stats.json").exists():
        return _redraw_info(d)
    _model_call_ok(request)
    if _budget_left() <= 0.8:
        raise HTTPException(429, "the model budget for the public demo is used up for now")
    src = job.dir / f"sheet{sheet}.png"
    one = {**job.to_json(), "sheets": [job.to_json()["sheets"][sheet]]}
    client = VisionClient()
    try:
        await redraw(src, one, client, d, sheet=0)
        gray = np.array(Image.open(src).convert("L"))
        await proofread_dir(gray, d, client)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"redraw failed: {type(e).__name__}: {e}"[:300])
    finally:
        _record(client)
    return _redraw_info(d)


@app.get("/api/engines")
def engines():
    return {"spark": True, "offline": bool(OFFLINE_URL)}


@app.get("/health")
def health():
    return {"ok": True}


if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")


@app.get("/{path:path}", response_class=HTMLResponse)
def index(path: str):
    f = STATIC / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>Whiteout</h1><p>Front end not built. Run npm run build in web/.</p>")
    if path and (STATIC / path).is_file() and ".." not in path:
        return FileResponse(STATIC / path)
    return HTMLResponse(f.read_text())


@app.exception_handler(HTTPException)
async def http_err(_, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
