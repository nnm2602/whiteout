"""Read a measuring instrument's display from a phone photo with Muse Spark.

The model reports the digits exactly as shown and how sure it is; code turns that into a value in the drawing's units
and checks it against the line being measured. A reading the model is unsure of is never written without a person
confirming it.
"""

from __future__ import annotations

import asyncio
import io
import re

import cv2
import numpy as np
from PIL import Image, ImageOps

from .model_client import VisionClient
from .prompts import _obj

MM_PER_IN = 25.4

FIND_SCHEMA = _obj({"found": {"type": "boolean"}, "box": {"type": "array", "items": {"type": "integer"}}})
FIND_PROMPT = (
    "Find the digital display (the LCD window with the digits) of the measuring instrument in this photo. box: "
    "[x_min, y_min, x_max, y_max] on a 0-1000 grid of this image, around the whole display window including the unit marker. found: "
    "false if there is no digital display."
)

SEGMENTS = {"abcdef": "0", "bc": "1", "abdeg": "2", "abcdg": "3", "bcfg": "4", "acdfg": "5", "acdefg": "6",
            "abc": "7", "abcdefg": "8", "abcdfg": "9", "abcf": "7"}

DIGITS_SCHEMA = _obj({
    "digits": {"type": "array", "items": _obj({
        "lit": {"type": "string"},
        "point_after": {"type": "boolean"},
    })},
    "minus": {"type": "boolean"},
    "display": {"type": "string"},
    "unit": {"type": "string", "enum": ["in", "mm", "unknown"]},
    "sure": {"type": "boolean"},
    "note": {"type": "string"},
})
DIGITS_PROMPT = (
    "This is the display of a digital measuring instrument, cropped from a photo. It shows seven-segment digits. "
    "Segments are named: a top, b upper right, c lower right, d bottom, e lower left, f upper left, g middle. For each "
    "digit position from left to right, list which segments are lit (letters in alphabetical order, for example "
    "'abcdfg'), and whether a decimal point is lit right after it. A '1' lights only b and c. minus: true if a minus "
    "sign is lit before the digits. display: the number you read, as shown, without sign or unit. unit: the unit "
    "marker shown (in or mm), or unknown. sure: false if any segment is hidden by glare, blurred or ambiguous. note: "
    "a few words on anything doubtful."
)


def decode(digits: list[dict]) -> str:
    out = ""
    for d in digits:
        lit = "".join(sorted(set(re.sub(r"[^a-g]", "", d.get("lit", "").lower()))))
        out += SEGMENTS.get(lit, "?")
        if d.get("point_after"):
            out += "."
    return out.strip()


def _same(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return False


GAUGE_SCHEMA = _obj({
    "instrument": {"type": "string", "enum": ["digital caliper", "digital micrometer", "dial indicator",
                                              "height gauge", "depth gauge", "other", "none"]},
    "display": {"type": "string"},
    "unit": {"type": "string", "enum": ["in", "mm", "unknown"]},
    "negative": {"type": "boolean"},
    "sure": {"type": "boolean"},
    "note": {"type": "string"},
})

GAUGE_PROMPT = (
    "This is a photo of a measuring instrument in an inspection lab. Read its digital display. display: the number "
    "exactly as the digits show it, with the decimal point where the display puts it, no unit, no sign (for example "
    "'0.1885' or '12.37'). unit: the unit shown on the display (in or mm), or unknown. negative: true if a minus sign "
    "is lit. sure: false if any digit is blurred, hidden, reflected or ambiguous, or if there is no readable display; "
    "never guess a digit. note: a few words on anything that makes the reading doubtful. instrument: what the tool is."
)


def prepare(data: bytes, max_side: int = 1600) -> Image.Image:
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


def to_units(display: str, unit: str, negative: bool, drawing_units: str) -> float | None:
    m = re.search(r"\d*\.\d+|\d+", display or "")
    if not m:
        return None
    v = float(m.group(0)) * (-1 if negative else 1)
    if unit == "mm" and drawing_units == "in":
        v /= MM_PER_IN
    elif unit == "in" and drawing_units == "mm":
        v *= MM_PER_IN
    return v


async def read_gauge(data: bytes, client: VisionClient, drawing_units: str = "in") -> dict:
    """Find the display, crop and enlarge it, read the segments of each digit; code decodes the segments, and a
    disagreement with the model's own reading makes the result unsure."""
    im = prepare(data)
    f = await client.ask_json([im], FIND_PROMPT, FIND_SCHEMA, name="find_display", effort="low")
    crop = im
    if f["found"] and len(f["box"]) == 4:
        x0, y0, x1, y1 = (round(v * s / 1000) for v, s in zip(f["box"], (im.width, im.height, im.width, im.height)))
        mx, my = int((x1 - x0) * 0.08) + 8, int((y1 - y0) * 0.15) + 8
        x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(im.width, x1 + mx), min(im.height, y1 + my)
        if x1 - x0 > 40 and y1 - y0 > 20:
            crop = im.crop((x0, y0, x1, y1))
            k = 1400 / crop.width
            if k > 1:
                crop = crop.resize((1400, max(1, int(crop.height * k))), Image.LANCZOS)
    d = await client.ask_json([crop], DIGITS_PROMPT, DIGITS_SCHEMA, name="gauge", effort="medium")
    decoded = decode(d["digits"])
    shown = re.sub(r"[^0-9.]", "", d["display"])
    agree = bool(decoded) and "?" not in decoded and _same(decoded, shown)
    code = await asyncio.to_thread(segment_read, crop)
    # the code reader checks the digits; the decimal point comes from the model's two readings agreeing
    code_agrees = bool(code) and re.sub(r"\D", "", code).lstrip("0") == re.sub(r"\D", "", shown).lstrip("0") != ""
    r = {"instrument": "", "display": shown, "unit": d["unit"], "negative": d["minus"],
         "sure": bool(d["sure"] and agree and code_agrees and f["found"]), "note": d["note"], "segments": decoded,
         "code_reading": code or ""}
    if not agree:
        r["note"] = (r["note"] + "; " if r["note"] else "") + f"digit shapes read as {decoded or 'nothing'}, not {shown}"
    elif not code_agrees:
        r["note"] = (r["note"] + "; " if r["note"] else "") + f"the second reader saw {code or 'nothing'}: confirm by eye"
    value = to_units(r["display"], r["unit"], r["negative"], drawing_units)
    decimals = 4 if drawing_units == "in" else 3
    text = ""
    if value is not None:
        text = f"{value:.{decimals}f}"
        if 0 <= value < 1 and drawing_units == "in":
            text = text[1:]  # ".1885", the way inch drawings write it
    return {**r, "value": None if value is None else round(value, decimals), "value_text": text,
            "converted": r["unit"] not in ("unknown", drawing_units)}


# ---- a second reader in code: classic seven-segment decoding on the cropped display ------------------------------
def _lcd_window(gray: np.ndarray) -> np.ndarray | None:
    """The light LCD window inside the crop, rotated upright."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(light)
    if n < 2:
        return None
    # the LCD sits inside the dark bezel; the steel body around it touches the edge of the crop
    h, w = gray.shape
    inner = [i for i in range(1, n) if stats[i, 0] > 2 and stats[i, 1] > 2 and stats[i, 0] + stats[i, 2] < w - 2
             and stats[i, 1] + stats[i, 3] < h - 2]
    pool = inner or list(range(1, n))
    best = max(pool, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    pts = np.column_stack(np.where(lab == best))[:, ::-1].astype(np.float32)
    if len(pts) < 100:
        return None
    hull = cv2.convexHull(pts)
    quad = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(quad) != 4:
        (cx, cy), (w, h), a = cv2.minAreaRect(pts)
        quad = cv2.boxPoints(((cx, cy), (w, h), a))
    s = quad.sum(1)
    d = np.diff(quad, axis=1).ravel()
    tl, br, tr, bl = quad[np.argmin(s)], quad[np.argmax(s)], quad[np.argmin(d)], quad[np.argmax(d)]
    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if W < 60 or H < 20 or W < H:
        return None
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    win = cv2.warpPerspective(gray, M, (W, H))
    m = int(0.03 * H) + 2
    return win[m:H - m, m:W - m]


def _segments(dark: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> str:
    w, h = x1 - x0, y1 - y0
    t = max(2, int(0.16 * w))
    regions = {
        "a": (x0 + t, y0, x1 - t, y0 + int(0.15 * h)),
        "d": (x0 + t, y1 - int(0.15 * h), x1 - t, y1),
        "g": (x0 + t, y0 + int(0.42 * h), x1 - t, y0 + int(0.58 * h)),
        "f": (x0, y0 + int(0.12 * h), x0 + t, y0 + int(0.42 * h)),
        "b": (x1 - t, y0 + int(0.12 * h), x1, y0 + int(0.42 * h)),
        "e": (x0, y0 + int(0.58 * h), x0 + t, y0 + int(0.88 * h)),
        "c": (x1 - t, y0 + int(0.58 * h), x1, y0 + int(0.88 * h)),
    }
    lit = ""
    for k, (a, b, c, d) in regions.items():
        cell = dark[max(0, b):max(b + 1, d), max(0, a):max(a + 1, c)]
        if cell.size and cell.mean() > 0.35 * 255:
            lit += k
    return "".join(sorted(lit))


def segment_read(crop: Image.Image) -> str | None:
    """Decode the digits on a cropped instrument display in code. None if the display cannot be found."""
    gray = np.array(crop.convert("L"))
    win = _lcd_window(gray)
    if win is None:
        return None
    H, W = win.shape
    dark = cv2.adaptiveThreshold(win, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 2 * (H // 4) + 1, 12)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # digits are separated by empty columns; a decimal point is a short run of its own in the gap
    cols = (dark > 0).sum(0) > max(2, int(0.02 * H))
    runs, x = [], 0
    while x < W:
        if cols[x]:
            x1 = x
            while x1 < W and cols[x1]:
                x1 += 1
            rows = np.where((dark[:, x:x1] > 0).sum(1) > 0)[0]
            if len(rows):
                runs.append((x, x1, int(rows[0]), int(rows[-1]) + 1))
            x = x1
        else:
            x += 1
    if not runs:
        return None
    dh = max(r[3] - r[2] for r in runs)
    tall = [r for r in runs if r[3] - r[2] > 0.6 * dh]
    if not tall:
        return None
    top = int(np.median([r[2] for r in tall]))
    bot = int(np.median([r[3] for r in tall]))
    # a decimal point that touches a digit: peel off edge columns whose ink is only near the bottom
    low = top + 0.72 * (bot - top)
    split = []
    for x0, x1, y0, y1 in runs:
        if y1 - y0 > 0.6 * dh:
            first = [int(np.argmax(dark[:, c] > 0)) if (dark[:, c] > 0).any() else H for c in range(x0, x1)]
            a = 0
            while a < len(first) and first[a] >= low:
                a += 1
            b = len(first)
            while b > a and first[b - 1] >= low:
                b -= 1
            if a > 0:
                split.append((x0, x0 + a, int(low), y1))
            split.append((x0 + a, x0 + b, y0, y1))
            if b < len(first):
                split.append((x0 + b, x1, int(low), y1))
        else:
            split.append((x0, x1, y0, y1))
    runs = [r for r in split if r[1] > r[0]]
    tall = [r for r in runs if r[3] - r[2] > 0.6 * dh]
    dw = int(np.median([r[1] - r[0] for r in tall if r[1] - r[0] > 0.4 * max(q[1] - q[0] for q in tall)]))
    out = []
    for x0, x1, y0, y1 in runs:
        w, h = x1 - x0, y1 - y0
        if h > 0.6 * dh:
            if w < 0.4 * dw:  # a '1' lights only the right side
                out.append((x0, "1"))
            else:
                out.append((x0, SEGMENTS.get(_segments(dark, x1 - dw, top, x1, bot), "?")))
        elif h < 0.25 * dh and w < 0.3 * dw and y0 > top + 0.6 * (bot - top):
            out.append((x0, "."))
        elif h < 0.25 * dh and w > 0.25 * dw and top + 0.3 * (bot - top) < y0 < top + 0.7 * (bot - top):
            out.append((x0, "-"))
    txt = "".join(ch for _, ch in sorted(out))
    txt = re.sub(r"^-", "", txt)
    return txt or None
