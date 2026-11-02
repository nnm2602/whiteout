"""Synthetic phone photos of digital caliper and micrometer displays with known readings.

Each photo is a drawn instrument (steel body, LCD with seven-segment digits, unit marker) placed on a bench at an
angle, with blur, noise and JPEG compression. Some have glare over a digit; for those the right answer is "not sure".
Answers go to eval/gauge_photos/answers.json.

    uv run python eval/make_gauge_photos.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent / "gauge_photos"

# segments a..g for each digit
SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc",
       "8": "abcdefg", "9": "abcdfg", "-": "g"}


def seven_seg(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, ch: str, ink, t: int) -> None:
    hh = h // 2
    pts = {
        "a": [(x + t, y), (x + w - t, y), (x + w - 2 * t, y + t), (x + 2 * t, y + t)],
        "d": [(x + 2 * t, y + h - t), (x + w - 2 * t, y + h - t), (x + w - t, y + h), (x + t, y + h)],
        "g": [(x + t, y + hh), (x + 2 * t, y + hh - t // 2), (x + w - 2 * t, y + hh - t // 2), (x + w - t, y + hh),
              (x + w - 2 * t, y + hh + t // 2), (x + 2 * t, y + hh + t // 2)],
        "f": [(x, y + t), (x + t, y + 2 * t), (x + t, y + hh - t), (x, y + hh)],
        "b": [(x + w, y + t), (x + w, y + hh), (x + w - t, y + hh - t), (x + w - t, y + 2 * t)],
        "e": [(x, y + hh), (x + t, y + hh + t), (x + t, y + h - 2 * t), (x, y + h - t)],
        "c": [(x + w, y + hh), (x + w, y + h - t), (x + w - t, y + h - 2 * t), (x + w - t, y + hh + t)],
    }
    for s in SEG.get(ch, ""):
        draw.polygon(pts[s], fill=ink)


def instrument(reading: str, unit: str, kind: str, rng: random.Random) -> Image.Image:
    W, H = 1400, 520
    im = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    # brushed steel body with a gradient
    body = np.zeros((H, W, 3), np.float32)
    grad = np.linspace(150, 215, H)[:, None]
    body[:] = grad[..., None]
    body += np.random.default_rng(rng.randint(0, 9999)).normal(0, 6, (H, W, 1))
    im = Image.fromarray(np.clip(body, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    if kind == "caliper":
        d.rectangle([0, 360, W, 440], fill=(190, 192, 196))  # beam
        for i in range(0, W, 14):
            d.line([(i, 360), (i, 372 if i % 70 else 385)], fill=(40, 40, 40), width=2)
    # display housing and LCD
    d.rounded_rectangle([110, 30, 1240, 345], 26, fill=(32, 34, 38))
    lcd = (rng.randint(150, 175), rng.randint(165, 185), rng.randint(140, 160))
    d.rounded_rectangle([150, 60, 1050, 265], 10, fill=lcd)
    ink = (25, 30, 25)
    digits = reading.replace("-", "")
    neg = reading.startswith("-")
    cw, ch, gap, t = 104, 160, 30, 17
    total = len(digits.replace(".", "")) * (cw + gap)
    x = 1030 - total
    y = 82
    if neg:
        d.rectangle([x - 70, y + ch // 2 - 8, x - 22, y + ch // 2 + 8], fill=ink)
    for c in digits:
        if c == ".":
            d.rectangle([x - gap + 6, y + ch - t, x - gap + 6 + t, y + ch], fill=ink)
            continue
        seven_seg(d, x, y, cw, ch, c, ink, t)
        x += cw + gap
    d.text((1070, 200), unit, fill=(230, 230, 230), font_size=46)
    for i, lab in enumerate(["ON/OFF", "ZERO", "in/mm"]):
        d.ellipse([300 + i * 260, 282, 344 + i * 260, 326], fill=(200, 60, 50) if i == 0 else (70, 70, 75))
        d.text((352 + i * 260, 292), lab, fill=(200, 200, 200), font_size=22)
    return im


def photo(inst: Image.Image, rng: random.Random, glare_digit: int | None) -> Image.Image:
    W, H = 1600, 1200
    bench = np.random.default_rng(rng.randint(0, 9999)).normal(rng.randint(70, 140), 18, (H, W, 1))
    bench = np.repeat(bench, 3, axis=2) * np.array([1.0, 0.95, 0.85])
    bench = cv2.GaussianBlur(bench.astype(np.float32), (0, 0), 3)
    src = np.array(inst)
    h, w = src.shape[:2]
    cx, cy = W / 2 + rng.uniform(-120, 120), H / 2 + rng.uniform(-90, 90)
    sc = rng.uniform(0.75, 0.95)
    ang = np.deg2rad(rng.uniform(-12, 12))
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) * sc
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = corners @ R.T + [cx, cy]
    dst += np.array([[rng.uniform(-40, 40), rng.uniform(-30, 30)] for _ in range(4)])  # perspective
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(dst))
    warped = cv2.warpPerspective(src, M, (W, H))
    mask = cv2.warpPerspective(np.ones((h, w), np.uint8) * 255, M, (W, H))
    out = np.where(mask[..., None] > 0, warped, bench).astype(np.float32)
    # light falloff
    yy, xx = np.mgrid[0:H, 0:W]
    out *= (1.0 - 0.35 * (((xx - W * rng.uniform(0.3, 0.7)) / W) ** 2 + ((yy - H * 0.4) / H) ** 2))[..., None]
    if glare_digit is not None:
        # a bright reflection over one digit of the display
        gx = 1030 - (glare_digit + 1) * 134 + 52
        p = M @ np.array([gx, 162, 1.0])
        p = p[:2] / p[2]
        g = np.exp(-(((xx - p[0]) / 75) ** 2 + ((yy - p[1]) / 95) ** 2))
        out = out * (1 - g[..., None]) + 255 * g[..., None]
    out += np.random.default_rng(rng.randint(0, 9999)).normal(0, rng.uniform(3, 9), out.shape)
    im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return im.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.8)))


def main() -> None:
    rng = random.Random(7)
    OUT.mkdir(parents=True, exist_ok=True)
    answers = []
    for i in range(36):
        unit = "in" if i % 3 else "mm"
        kind = "caliper" if i % 4 else "micrometer"
        if unit == "in":
            v = rng.uniform(0.04, 3.9)
            reading = f"{v:.4f}" if kind == "micrometer" or i % 2 else f"{v:.3f}5"[:6]
        else:
            v = rng.uniform(1.0, 99.0)
            reading = f"{v:.2f}" if kind == "caliper" else f"{v:.3f}"
        if i % 11 == 5:
            reading = "-" + reading
        glare = rng.randrange(0, 3) if i % 6 == 3 else None
        im = photo(instrument(reading, unit, kind, rng), rng, glare)
        name = f"gauge_{i:02d}.jpg"
        im.save(OUT / name, quality=rng.randint(70, 90))
        answers.append({"file": name, "display": reading.lstrip("-"), "negative": reading.startswith("-"),
                        "unit": unit, "instrument": kind, "glare": glare is not None})
    (OUT / "answers.json").write_text(json.dumps(answers, indent=1))
    print(f"{len(answers)} photos in {OUT}")


if __name__ == "__main__":
    main()
