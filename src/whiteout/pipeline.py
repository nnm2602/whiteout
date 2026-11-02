"""The Whiteout pipeline: read, merge, snap, balloon, then erase and search what is left."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
from PIL import Image

from . import prompts
from .balloon import assign_zones, number_and_place
from .geometry import Raw, SheetInk, expand, gap, lines_expected, merge, norm_text, overlap_min, text_sim
from .ingest import LoadedSheet
from .model_client import VisionClient
from .tiling import Tile, grid
from .tolerance import build_requirement
from .types import AuditFind, AuditPass, Characteristic, SheetInfo

Emit = Callable[[dict], Awaitable[None] | None]

MAX_AUDIT_PASSES = 3


async def _emit(emit: Emit | None, ev: dict) -> None:
    if emit is None:
        return
    r = emit(ev)
    if asyncio.iscoroutine(r):
        await r


def fit(img: Image.Image, max_w: int = 1775, max_h: int = 1775) -> tuple[Image.Image, float]:
    f = min(1.0, max_w / img.width, max_h / img.height)
    if f >= 1.0:
        return img, 1.0
    return img.resize((max(1, round(img.width * f)), max(1, round(img.height * f))), Image.LANCZOS), f


def grid_box(b, w: int, h: int):
    x0, y0, x1, y1 = (max(0, min(1000, int(v))) for v in (list(b) + [0, 0, 0, 0])[:4])
    if x1 <= x0 or y1 <= y0:
        return None
    return (round(x0 * w / 1000), round(y0 * h / 1000), round(x1 * w / 1000), round(y1 * h / 1000))


@dataclass
class SheetResult:
    info: SheetInfo
    chars: list[Characteristic] = field(default_factory=list)
    audit: list[AuditPass] = field(default_factory=list)
    ignored: list[AuditFind] = field(default_factory=list)
    residue: np.ndarray | None = None
    residues: list[np.ndarray] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


async def overview(client: VisionClient, sheet: LoadedSheet, index: int) -> SheetInfo:
    img = sheet.image
    W, H = img.size
    small, _ = fit(img, 1790, 1790)
    ov = await client.ask_json([small], prompts.OVERVIEW_PROMPT, prompts.OVERVIEW_SCHEMA, "overview")
    info = SheetInfo(index=index, width=W, height=H, dpi=sheet.dpi, has_text_layer=bool(sheet.words))
    info.border_box = grid_box(ov.get("border_box", []), W, H)
    info.title_block_box = grid_box(ov.get("title_block_box", []), W, H)
    info.zone_columns = [z for z in ov.get("zone_columns", []) if z]
    info.zone_rows = [z for z in ov.get("zone_rows", []) if z]
    crops = []
    for key in ("title_block_box", "general_notes_box"):
        b = grid_box(ov.get(key, []), W, H)
        if b and (b[2] - b[0]) * (b[3] - b[1]) > 2000:
            b = expand(b, 20, 20, W, H)
            c, _ = fit(img.crop(b), 1775, 1175)
            crops.append(c)
    if crops:
        tb = await client.ask_json(crops, prompts.TITLE_PROMPT, prompts.TITLE_SCHEMA, "title_block")
        for k in ("part_number", "part_name", "drawing_number", "revision", "material", "finish"):
            setattr(info, k, (tb.get(k) or "").strip())
        info.units = tb.get("units") if tb.get("units") in ("in", "mm") else "in"
        info.general_tolerances = [g for g in tb.get("general_tolerances", []) if g.get("plus_minus")]
    return info


def _to_raw(item: dict, tile: Tile, W: int, H: int, ink: SheetInk) -> Raw | None:
    b = tile.to_sheet(item.get("box", []))
    if b[2] - b[0] < 3 or b[3] - b[1] < 3:
        return None
    r = Raw(data=item, box=b, raw_box=b, margin=tile.edge_margin(b, W, H), tile=(tile.x, tile.y))
    r.snap = ink.snap(b, lines_expected(item), text=item.get("text") or "")
    if r.snap.snapped:
        r.box = r.snap.box
    return r


async def read_tiles(client: VisionClient, img: Image.Image, ink: SheetInk, tiles: list[Tile], prompt: str,
                     schema: dict, emit: Emit | None, stage: str, effort: str | None = None) -> list[Raw]:
    W, H = img.size
    raws: list[Raw] = []

    async def one(t: Tile):
        res = await client.ask_json([img.crop(t.box)], prompt, schema, stage.split("-")[0], effort)
        items = [r for r in (_to_raw(it, t, W, H, ink) for it in res.get("items", [])) if r]
        await _emit(emit, {"type": "tile", "stage": stage, "tile": t.box, "items": [r.box for r in items],
                           "texts": [r.text for r in items],
                           "whats": [r.data.get("what", "requirement") for r in items]})
        return items

    for items in await asyncio.gather(*(one(t) for t in tiles)):
        raws.extend(items)
    return raws


def to_char(r: Raw, idx: str, info: SheetInfo, found_in: str) -> Characteristic:
    d = r.data
    c = Characteristic(
        id=idx,
        kind=d.get("kind", "dimension"),
        text=(d.get("text") or "").strip(),
        multiplier=max(1, int(d.get("multiplier") or 1)),
        box=r.box,
        raw_box=r.raw_box,
        sheet=info.index,
        confidence=d.get("confidence", "high") if d.get("confidence") in ("high", "low") else "high",
        found_in=found_in,
    )
    reqs = d.get("requirements") or [{"type": "other", "text": c.text}]
    c.requirements = [build_requirement(q, c.text, info.general_tolerances) for q in reqs]
    if r.truncated:
        c.flags.append("read at a tile edge: check the full text")
    if r.snap is not None and not r.snap.snapped:
        c.flags.append("box not matched to ink")
    for q in c.requirements:
        c.flags.extend(q.flags)
    return c


def _drop_repeats(chars: list[Characteristic]) -> None:
    """A read whose box lands on no ink, with the same text as a read that does, is the same item seen twice."""
    unmatched = "box not matched to ink"
    placed = {(norm_text(c.text), c.kind) for c in chars if unmatched not in c.flags and not c.ignored}
    for c in chars:
        if unmatched in c.flags and norm_text(c.text) and (norm_text(c.text), c.kind) in placed:
            c.ignored = True


async def process_sheet(client: VisionClient, sheet: LoadedSheet, index: int, emit: Emit | None = None,
                        audit_passes: int = MAX_AUDIT_PASSES, general_fallback: list[dict] | None = None,
                        on_residue: Callable[[int, np.ndarray], None] | None = None,
                        number_start: int = 1) -> SheetResult:
    t0 = time.time()
    img = sheet.image
    gray = np.array(img)
    W, H = img.size
    await _emit(emit, {"type": "stage", "stage": "overview", "sheet": index})
    info = await overview(client, sheet, index)
    if not info.general_tolerances and general_fallback:
        info.general_tolerances = list(general_fallback)  # stated once on sheet 1, applies to every sheet
    await _emit(emit, {"type": "overview", "sheet": index, "info": info.model_dump()})
    res = SheetResult(info=info)
    res.timings["overview"] = time.time() - t0

    t1 = time.time()
    ink = SheetInk(gray, sheet.dpi, sheet.dpi_trusted)
    tiles = grid(W, H, *client.cfg.tile)
    await _emit(emit, {"type": "stage", "stage": "read", "sheet": index, "tiles": [t.box for t in tiles]})
    raws = await read_tiles(client, img, ink, tiles, prompts.READ_PROMPT, prompts.READ_SCHEMA, emit, "read")
    kept = merge(raws, ink)
    chars = [to_char(r, f"s{index}c{i}", info, "read") for i, r in enumerate(kept)]
    snaps = {c.id: r.snap for c, r in zip(chars, kept)}
    res.timings["read"] = time.time() - t1
    await _emit(emit, {"type": "read_done", "sheet": index, "raw": len(raws), "kept": len(chars)})

    # Whiteout: erase everything found, then search what is left until a pass finds nothing new.
    t2 = time.time()
    residue = ink.erase(gray, [s for s in snaps.values() if s and s.snapped],
                        [c.box for c in chars if not (snaps.get(c.id) and snaps[c.id].snapped)])
    ignored: list[AuditFind] = []
    for p in range(1, audit_passes + 1):
        await _emit(emit, {"type": "stage", "stage": "audit", "sheet": index, "pass": p})
        rimg = Image.fromarray(residue)
        res.residues.append(residue.copy())
        if on_residue is not None:
            await asyncio.to_thread(on_residue, p, residue.copy())
            await _emit(emit, {"type": "residue", "sheet": index, "pass": p})
        tiles = grid(W, H, *client.cfg.tile, shift=(p % 2 == 1))
        araws = await read_tiles(client, rimg, ink, tiles, prompts.AUDIT_PROMPT, prompts.AUDIT_SCHEMA, emit, f"audit-{p}",
                                 client.cfg.audit_effort)
        araws = merge(araws, ink)
        ap = AuditPass(pass_no=p)
        new_snaps, new_fallback = [], []
        cands: list[tuple[Raw, AuditFind]] = []
        for r in araws:
            what = r.data.get("what", "requirement")
            find = AuditFind(pass_no=p, text=r.text, box=r.box, verdict="new", reason=r.data.get("reason", ""))
            near = _nearest(chars, r)
            if what == "not_requirement":
                find.verdict = "not_requirement"
            elif what == "erased_fragment" or (near is not None and _is_dup(near, r)):
                find.verdict = "fragment" if what == "erased_fragment" else "duplicate"
                find.char_id = near.id if near else None
                if find.verdict == "duplicate" and near is not None and r.snap and r.snap.snapped:
                    # The audit saw ink we failed to erase: grow the requirement's box to cover it.
                    near.box = (min(near.box[0], r.box[0]), min(near.box[1], r.box[1]),
                                max(near.box[2], r.box[2]), max(near.box[3], r.box[3]))
            else:
                cands.append((r, find))
            ap.finds.append(find)
            if r.snap and r.snap.snapped:
                new_snaps.append(r.snap)
            else:
                new_fallback.append(r.box)
            if find.verdict in ("not_requirement", "fragment"):
                ignored.append(find)
        # Every candidate miss is confirmed on the original drawing, next to what is already listed.
        confirmed = await asyncio.gather(*(confirm(client, img, chars, r) for r, _ in cands))
        for (r, find), (verdict, data, same) in zip(cands, confirmed):
            if verdict == "separate_requirement":
                if data.get("full_text"):
                    r.data = {**r.data, "text": data["full_text"], "kind": data.get("kind") or r.data.get("kind"),
                              "multiplier": data.get("multiplier") or r.data.get("multiplier", 1),
                              "requirements": data.get("requirements") or r.data.get("requirements", [])}
                c = to_char(r, f"s{index}c{len(chars)}", info, f"audit-{p}")
                c.flags.append("found on the erased page: confirm it is a requirement")
                chars.append(c)
                find.char_id = c.id
                find.reason = (data.get("reason") or find.reason)[:200]
                ap.new_count += 1
            else:
                find.verdict = "duplicate" if verdict == "same_as_listed" else "not_requirement"
                find.reason = "checked on the original: " + (data.get("reason") or "")[:180]
                find.char_id = same
                ignored.append(find)
        res.audit.append(ap)
        residue = ink.erase(residue, new_snaps, new_fallback)
        await _emit(emit, {"type": "audit_pass", "sheet": index, "pass": p, "new": ap.new_count,
                           "finds": [f.model_dump() for f in ap.finds]})
        if ap.new_count == 0:
            break
    res.residue = residue
    res.ignored = ignored
    if on_residue is not None:
        await asyncio.to_thread(on_residue, 0, residue.copy())
        await _emit(emit, {"type": "residue", "sheet": index, "pass": "final"})
    res.timings["audit"] = time.time() - t2

    if sheet.words:
        _text_layer_check(sheet, chars, info, res)

    assign_zones(chars, info)
    _drop_repeats(chars)
    number_and_place(chars, gray, info, start=number_start)
    res.chars = chars
    res.timings["total"] = time.time() - t0
    await _emit(emit, {"type": "sheet_done", "sheet": index, "count": len(chars)})
    return res


def _nearest(chars: list[Characteristic], r: Raw) -> Characteristic | None:
    """The found requirement this audit find most likely repeats: nearby, or a long note it is a piece of."""
    best, bd = None, 1e9
    for c in chars:
        d = gap(c.box, r.box)
        if d < bd:
            best, bd = c, d
    if best is not None and bd < 110:
        return best
    piece = norm_text(r.text)
    if len(piece) >= 4:
        for c in chars:
            if len(norm_text(c.text)) > len(piece) + 4 and text_sim(c.text, r.text) >= 95 and gap(c.box, r.box) < 400:
                return c
    return None


def _is_dup(c: Characteristic, r: Raw) -> bool:
    """An audit find that repeats a found requirement whose ink was not fully erased."""
    if overlap_min(c.box, r.box) > 0.3:
        return True
    d = gap(c.box, r.box)
    if (d < 40 and text_sim(c.text, r.text) >= 70) or (d < 110 and text_sim(c.text, r.text) >= 90):
        return True
    # an unerased line of a multi-line note that was already found
    return len(norm_text(c.text)) > len(norm_text(r.text)) + 4 and text_sim(c.text, r.text) >= 95 and d < 400


def _text_layer_check(sheet: LoadedSheet, chars: list[Characteristic], info: SheetInfo, res: SheetResult) -> None:
    """Vector PDFs: every word with a digit must sit inside a found requirement, the title block, or the border."""
    import re

    misses = []
    for b, w in sheet.words:
        if not re.search(r"\d", w):
            continue
        if info.title_block_box and overlap_min(b, info.title_block_box) > 0.5:
            continue
        if info.border_box and not overlap_min(b, info.border_box) > 0.5:
            continue
        if any(overlap_min(b, expand(c.box, 6, 6, info.width, info.height)) > 0.5 for c in chars):
            continue
        misses.append((b, w))
    for b, w in misses:
        res.ignored.append(AuditFind(pass_no=0, text=w, box=b, verdict="not_requirement",
                                     reason="text layer word outside every found requirement: review"))
    res.timings["text_layer_unmatched"] = len(misses)


async def confirm(client: VisionClient, img: Image.Image, chars: list[Characteristic], r: Raw):
    """Show the candidate on the original drawing with nearby listed requirements; return (verdict, data, char id)."""
    from PIL import ImageDraw

    W, H = img.size
    b = r.box
    bw, bh = b[2] - b[0], b[3] - b[1]
    win = expand(b, max(350, int(2.0 * bw)), max(260, int(5 * bh)), W, H)
    crop = img.crop(win).convert("RGB")
    d = ImageDraw.Draw(crop)
    listed = []
    for c in chars:
        if c.ignored or overlap_min(c.box, win) < 0.5:
            continue
        k = str(len(listed) + 1)
        x0, y0, x1, y1 = (c.box[0] - win[0] - 3, c.box[1] - win[1] - 3, c.box[2] - win[0] + 3, c.box[3] - win[1] + 3)
        d.rectangle((x0, y0, x1, y1), outline=(30, 90, 220), width=3)
        d.text((x0, max(0, y0 - 14)), k, fill=(30, 90, 220))
        listed.append((k, c.text, c.id))
    d.rectangle((b[0] - win[0] - 5, b[1] - win[1] - 5, b[2] - win[0] + 5, b[3] - win[1] + 5), outline=(220, 30, 30), width=4)
    crop, _ = fit(crop, 1775, 1175)
    out = await client.ask_json([crop], prompts.confirm_prompt([(k, t) for k, t, _ in listed]),
                                prompts.CONFIRM_SCHEMA, "confirm")
    same = next((cid for k, _, cid in listed if k == (out.get("same_as") or "").strip("[] ")), None)
    return out.get("verdict", "not_requirement"), out, same
