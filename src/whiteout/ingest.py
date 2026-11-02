"""Load drawings: vector or scanned PDF, TIFF/PNG/JPEG scans, and phone photos.

Every sheet becomes a grayscale image (white paper, dark ink) plus, for vector PDFs, the words of the text layer
in sheet pixel coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

MAX_SIDE = 6400  # longest side in pixels after loading
PDF_DPI = 200


@dataclass
class LoadedSheet:
    image: Image.Image  # mode "L"
    dpi: float
    source: str
    words: list[tuple[tuple[int, int, int, int], str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dpi_trusted: bool = False  # true for PDF pages rendered at a known scale


def load(path: str | Path, max_pages: int = 12) -> list[LoadedSheet]:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path, max_pages)
    return [_load_image(path)]


def _load_pdf(path: Path, max_pages: int) -> list[LoadedSheet]:
    doc = fitz.open(path)
    out = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        w_in, h_in = page.rect.width / 72, page.rect.height / 72
        dpi = min(PDF_DPI, MAX_SIDE / max(w_in, h_in))
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        scale = dpi / 72
        words = []
        for x0, y0, x1, y1, w, *_ in page.get_text("words"):
            if w.strip():
                words.append(((int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)), w))
        sheet = LoadedSheet(image=img, dpi=dpi, source=f"{path.name}#page{i + 1}", words=words, dpi_trusted=True)
        if not words:
            # A PDF page that is one big image is a scan: clean it like one.
            if len(page.get_images()) >= 1 and not page.get_drawings():
                sheet.image, notes = clean_scan(np.array(img))
                sheet.notes += ["scanned PDF page"] + notes
                sheet.dpi_trusted = False
        out.append(sheet)
    return out


def _load_image(path: Path) -> LoadedSheet:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    dpi = float((img.info.get("dpi") or (300, 300))[0] or 300)
    g = img.convert("L")
    if max(g.size) > MAX_SIDE:
        f = MAX_SIDE / max(g.size)
        g = g.resize((round(g.width * f), round(g.height * f)), Image.LANCZOS)
        dpi *= f
    arr = np.array(g)
    notes = []
    if looks_like_photo(arr):
        arr, n = rectify_photo(arr)
        notes += n
    cleaned, n2 = clean_scan(arr)
    return LoadedSheet(image=cleaned, dpi=dpi, source=path.name, notes=notes + n2)


def looks_like_photo(arr: np.ndarray) -> bool:
    """Photos have uneven lighting: the paper background varies a lot across the frame."""
    small = cv2.resize(arr, (64, 64), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return float(np.percentile(bg, 95) - np.percentile(bg, 20)) > 45


def rectify_photo(arr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Find the sheet outline in a photo and warp it flat. Leaves the image alone if no clear outline."""
    h, w = arr.shape
    f = 1000 / max(h, w)
    small = cv2.resize(arr, (int(w * f), int(h * f)))
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.25 * small.shape[0] * small.shape[1]:
            best = approx.reshape(4, 2).astype(np.float32) / f
            break
    if best is None:
        return arr, ["photo: no sheet outline found, used as is"]
    s = best.sum(1)
    d = np.diff(best, axis=1).ravel()
    tl, br = best[np.argmin(s)], best[np.argmax(s)]
    tr, bl = best[np.argmin(d)], best[np.argmax(d)]
    wid = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    hei = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    m = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [wid, 0], [wid, hei], [0, hei]]))
    return cv2.warpPerspective(arr, m, (wid, hei), flags=cv2.INTER_CUBIC, borderValue=255), ["photo: sheet outline flattened"]


def clean_scan(arr: np.ndarray) -> tuple[Image.Image, list[str]]:
    """Even out the paper background and straighten small skew. Ink stays grayscale (no hard threshold)."""
    notes = []
    k = max(31, (min(arr.shape) // 40) | 1)
    bg = cv2.medianBlur(cv2.resize(arr, (arr.shape[1] // 4, arr.shape[0] // 4)), min(k // 4 * 2 + 1, 255))
    bg = cv2.resize(bg, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)
    norm = cv2.divide(arr, np.maximum(bg, 1), scale=255)
    angle = estimate_skew(norm)
    if 0.15 <= abs(angle) <= 5:
        h, w = norm.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        norm = cv2.warpAffine(norm, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
        notes.append(f"deskewed {angle:.2f} degrees")
    return Image.fromarray(norm), notes


def estimate_skew(arr: np.ndarray) -> float:
    """Median angle of long near-horizontal lines (drawing border and dimension lines)."""
    h, w = arr.shape
    f = 2000 / max(h, w)
    small = cv2.resize(arr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    ink = (small < 128).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(ink, 1, np.pi / 1800, threshold=200, minLineLength=int(0.25 * small.shape[1]), maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        a = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        if abs(a) < 6:
            angles.append(a)
    return float(np.median(angles)) if angles else 0.0
