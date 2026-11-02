"""Revision delta: what changed between an inspected revision and a new one, and which results carry forward.

Sheets are aligned in code (ORB features + a RANSAC similarity transform), then every requirement on the new
revision is matched to the old one by position and text. Unchanged requirements keep their measured values,
so a partial first article covers only what changed.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .forms import form3_rows
from .geometry import gap, norm_text, overlap_min, text_sim


def _align(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """2x3 matrix mapping old sheet pixels to new sheet pixels."""
    f_old = 1600 / max(old.shape)
    f_new = 1600 / max(new.shape)
    a = cv2.resize(old, None, fx=f_old, fy=f_old, interpolation=cv2.INTER_AREA)
    b = cv2.resize(new, None, fx=f_new, fy=f_new, interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(6000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    ident = np.array([[f_old / f_new, 0, 0], [0, f_old / f_new, 0]], dtype=np.float64)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return ident
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    matches = sorted(matches, key=lambda m: m.distance)[:1500]
    if len(matches) < 20:
        return ident
    pa = np.float32([ka[m.queryIdx].pt for m in matches]) / f_old
    pb = np.float32([kb[m.trainIdx].pt for m in matches]) / f_new
    m, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=6.0, maxIters=4000)
    if m is None or inl is None or inl.sum() < 15:
        return ident
    return m


def _map_box(m: np.ndarray, b) -> tuple[int, int, int, int]:
    pts = np.array([[b[0], b[1], 1], [b[2], b[3], 1]], dtype=np.float64)
    q = pts @ m.T
    return (int(min(q[:, 0])), int(min(q[:, 1])), int(max(q[:, 0])), int(max(q[:, 1])))


def _diff_regions(old: np.ndarray, new: np.ndarray, m: np.ndarray) -> list[tuple[int, int, int, int]]:
    warped = cv2.warpAffine(old, m, (new.shape[1], new.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255)
    a = (warped < 150).astype(np.uint8)
    b = (new < 150).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    # ink present in one and absent (even nearby) in the other
    only_new = b & (1 - cv2.dilate(a, k))
    only_old = a & (1 - cv2.dilate(b, k))
    d = cv2.dilate(only_new | only_old, np.ones((25, 25), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(d)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area > 1500:
            out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


def compare_jobs(base, new) -> dict:
    sheets = []
    carried: dict[str, str] = {}
    base_rows = {r["char_id"]: [] for r in form3_rows([c for s in base.sheets for c in s.chars])}
    for r in form3_rows([c for s in base.sheets for c in s.chars], base.results):
        base_rows[r["char_id"]].append(r)
    new_rows: dict[str, list] = {}
    for r in form3_rows([c for s in new.sheets for c in s.chars]):
        new_rows.setdefault(r["char_id"], []).append(r)

    for i, ns in enumerate(new.sheets):
        if i >= len(base.sheets):
            sheets.append({"unchanged": [], "changed": [], "added": [c.id for c in ns.chars if not c.ignored],
                           "new_in_report": [], "removed": [], "diff_regions": []})
            continue
        bs = base.sheets[i]
        old_img = np.array(Image.open(base.dir / f"sheet{i}.png").convert("L"))
        new_img = np.array(Image.open(new.dir / f"sheet{i}.png").convert("L"))
        m = _align(old_img, new_img)
        mapped = {c.id: _map_box(m, c.box) for c in bs.chars if not c.ignored}
        diffs = _diff_regions(old_img, new_img, m)

        def in_diff(b) -> bool:
            return any(overlap_min(b, d) > 0.2 or (gap(b, d) == 0 and overlap_min(b, d) > 0) for d in diffs)

        # Match new requirements to old ones, best pairs first (place, then text).
        pairs = []
        news = [c for c in ns.chars if not c.ignored]
        olds = [c for c in bs.chars if not c.ignored]
        for c in news:
            for oc in olds:
                mb = mapped[oc.id]
                ov = overlap_min(mb, c.box)
                if ov < 0.2 and gap(mb, c.box) > 40:
                    continue
                pairs.append((ov + text_sim(oc.text, c.text) / 100, c, oc))
        pairs.sort(key=lambda t: -t[0])
        used_new: set[str] = set()
        used_old: set[str] = set()
        match: dict[str, object] = {}
        for _, c, oc in pairs:
            if c.id in used_new or oc.id in used_old:
                continue
            used_new.add(c.id)
            used_old.add(oc.id)
            match[c.id] = oc
        unchanged, changed, added, new_in_report = [], [], [], []
        for c in news:
            oc = match.get(c.id)
            touched = in_diff(c.box)
            if oc is not None and not touched:
                unchanged.append({"new": c.id, "old": oc.id})  # the pixels did not change here
            elif oc is not None:
                if _same_requirement(oc, c):
                    unchanged.append({"new": c.id, "old": oc.id})
                else:
                    changed.append({"new": c.id, "old": oc.id, "old_text": oc.text, "new_text": c.text})
            elif touched:
                added.append(c.id)
            else:
                new_in_report.append(c.id)  # drawing unchanged here, but the earlier report did not list it
        for u in unchanged:
            olds_rows, news_rows = base_rows.get(u["old"], []), new_rows.get(u["new"], [])
            if len(olds_rows) == len(news_rows):
                for o, n in zip(olds_rows, news_rows):
                    if o.get("measured"):
                        carried[n["row_id"]] = o["measured"]
        removed = [{"old": oc.id, "text": oc.text, "box": mapped[oc.id]} for oc in olds
                   if oc.id not in used_old and in_diff(mapped[oc.id])]
        sheets.append({"unchanged": unchanged, "changed": changed, "added": added, "new_in_report": new_in_report,
                       "removed": removed, "diff_regions": diffs, "transform": m.tolist()})
    total = sum(len(s["unchanged"]) + len(s["changed"]) + len(s["added"]) for s in sheets)
    measure = sum(len(s["changed"]) + len(s["added"]) + len(s["new_in_report"]) for s in sheets)
    return {"base_job": base.id, "sheets": sheets, "carried": carried,
            "summary": {"requirements": total, "to_measure": measure,
                        "unchanged": sum(len(s["unchanged"]) for s in sheets),
                        "changed": sum(len(s["changed"]) for s in sheets),
                        "added": sum(len(s["added"]) for s in sheets),
                        "removed": sum(len(s["removed"]) for s in sheets),
                        "new_in_report": sum(len(s["new_in_report"]) for s in sheets),
                        "carried_values": len(carried)}}


def _same_requirement(a, b) -> bool:
    """Same printed requirement even if the two reads spelled it differently (symbols, spacing)."""
    if norm_text(a.text) == norm_text(b.text):
        return True
    if a.multiplier != b.multiplier or len(a.requirements) != len(b.requirements):
        return False
    for qa, qb in zip(a.requirements, b.requirements):
        for x, y in ((qa.lower, qb.lower), (qa.upper, qb.upper), (qa.nominal, qb.nominal)):
            if (x is None) != (y is None) or (x is not None and abs(x - y) > 1e-9):
                return False
        if qa.attribute != qb.attribute or sorted(qa.datums) != sorted(qb.datums):
            return False
    return True
