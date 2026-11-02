"""Turn printed requirement text into numeric limits, in code.

The model splits a callout into requirements and proposes limits; this module recomputes the limits from the printed
text and flags any disagreement, so a number the model made up never reaches Form 3 silently.
"""

from __future__ import annotations

import re
from fractions import Fraction

from .types import Requirement

NUM = r"[-+]?(?:\d+\s+\d+/\d+|\d+/\d+|\d*\.\d+|\d+)"
_PREFIX = re.compile(r"^[^\w.]*(?:\d+\s*[Xx]\s+|\d+[Xx](?=[^\w]))?[^\w.]*(?:DIA\.?|SØ|SR|CR|R|V|Ø|⌀|∅)?[^\w.]*", re.I)


def to_float(s: str) -> float | None:
    s = (s or "").strip().replace("°", "").replace("\"", "")
    s = re.sub(r"^[ØØ⌀∅RSCR]+", "", s, flags=re.I).strip()
    if not s:
        return None
    try:
        if re.fullmatch(r"\d+\s+\d+/\d+", s):
            whole, frac = s.split(None, 1)
            return float(int(whole) + Fraction(frac))
        if re.fullmatch(r"[-+]?\d+/\d+", s):
            return float(Fraction(s))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def decimals(s: str) -> int | None:
    m = re.search(r"\.(\d+)", s or "")
    return len(m.group(1)) if m else (0 if re.search(r"\d", s or "") else None)


def parse_limits(text: str) -> tuple[float | None, float | None, float | None, str]:
    """Return (nominal, lower, upper, how) from the printed text of a single requirement."""
    t = (text or "").replace("−", "-").replace("–", "-").replace("±", " ± ").replace("+/-", " ± ")
    t = re.sub(r"\((?:TYP|REF|BASIC)[^)]*\)", "", t, flags=re.I)
    t = re.sub(r"\bTYP\.?\b|\bREF\b", "", t, flags=re.I)
    t = _PREFIX.sub("", t, count=1).strip()

    m = re.search(rf"({NUM})\s*±\s*({NUM})", t)
    if m:
        n, d = to_float(m.group(1)), to_float(m.group(2))
        if n is not None and d is not None:
            return n, n - abs(d), n + abs(d), "plus_minus"

    m = re.search(rf"({NUM})\s*\+\s*({NUM})\s*-\s*({NUM})", t)
    if m:
        n, p, q = (to_float(g) for g in m.groups())
        if None not in (n, p, q):
            return n, n - abs(q), n + abs(p), "unilateral"

    # Limit dimension: ".370 - .380", ".3910/.3895", ".190-.196"
    m = re.search(rf"^\s*(\d*\.\d+|\d+\.\d*)\s*[-/]\s*(\d*\.\d+|\d+\.\d*)\b", t)
    if m:
        a, b = to_float(m.group(1)), to_float(m.group(2))
        if a is not None and b is not None and a > 0 and 0.5 < b / a < 2:
            lo, hi = min(a, b), max(a, b)
            return (lo + hi) / 2, lo, hi, "limits"

    m = re.search(rf"({NUM})", t)
    if m:
        n = to_float(m.group(1))
        return n, None, None, "nominal_only"
    return None, None, None, "none"


def general_tolerance(nominal_text: str, req_type: str, general: list[dict]) -> float | None:
    """Pick the title-block tolerance for an untoleranced value: by decimal places, angle, or fraction."""
    if not general:
        return None
    is_angle = req_type == "angle" or "°" in (nominal_text or "")
    is_frac = bool(re.search(r"\d/\d", nominal_text or ""))
    places = decimals(nominal_text)
    for g in general:
        key = (g.get("applies_to") or "").upper()
        pm = to_float((g.get("plus_minus") or "").replace("±", ""))
        if pm is None:
            continue
        if is_angle and "ANG" in key:
            return pm
        if is_frac and "FRAC" in key:
            return pm
        if not is_angle and not is_frac and places is not None and "X" in key:
            after = key.split(".", 1)[1] if "." in key else ""
            if after and after.count("X") == places:
                return pm
            if not after and places == 0 and "X" in key:
                return pm
    return None


ATTRIBUTE_TYPES = {"note", "thread", "surface_finish", "other"}


def build_requirement(raw: dict, item_text: str, general: list[dict]) -> Requirement:
    """Combine the model's proposed requirement with limits recomputed in code."""
    rtype = raw.get("type") or "other"
    rtext = (raw.get("text") or "").strip() or item_text
    req = Requirement(
        type=rtype,
        label=(raw.get("label") or "").strip(),
        text=rtext,
        gdt_symbol=(raw.get("gdt_symbol") or "").strip(),
        datums=[d for d in (raw.get("datums") or []) if d],
        modifiers=[m for m in (raw.get("modifiers") or []) if m],
    )
    m_nom, m_lo, m_hi = to_float(raw.get("nominal", "")), to_float(raw.get("lower", "")), to_float(raw.get("upper", ""))

    if rtype == "gdt":
        tol = m_nom if m_nom is not None else m_hi
        # The tolerance value must be printed in the frame text.
        if tol is not None and not _number_in_text(tol, rtext + " " + item_text):
            req.flags.append("tolerance value not found in printed frame")
        req.nominal, req.lower, req.upper = tol, 0.0, tol
        req.tolerance_source = "printed" if tol is not None else "none"
        return req

    if rtype in ATTRIBUTE_TYPES and m_nom is None:
        req.attribute = True
        return req

    n, lo, hi, how = parse_limits(rtext)
    if n is None and rtext != item_text:
        n, lo, hi, how = parse_limits(item_text)
    req.nominal = n if n is not None else m_nom
    if n is not None and m_nom is not None and abs(n - m_nom) > 1e-6:
        req.flags.append(f"model nominal {m_nom:g} differs from printed {n:g}")
    printed = rtext + " " + item_text
    if how == "limits":
        if not (_number_in_text(lo, printed) and _number_in_text(hi, printed)):
            req.flags.append("limits not found in printed text")
    elif req.nominal is not None and not _number_in_text(req.nominal, printed):
        req.flags.append("nominal not found in printed text")

    if lo is not None and hi is not None:
        req.lower, req.upper, req.tolerance_source = lo, hi, "printed"
        if m_lo is not None and m_hi is not None and (abs(m_lo - lo) > 1e-6 or abs(m_hi - hi) > 1e-6):
            req.flags.append("model limits differ from printed text; printed text used")
    elif rtype in ("basic", "reference"):
        req.tolerance_source = "none"
    elif req.nominal is not None:
        g = general_tolerance(_nominal_token(rtext), rtype, general)
        if g is not None:
            req.lower, req.upper, req.tolerance_source = req.nominal - g, req.nominal + g, "general"
        elif m_lo is not None and m_hi is not None:
            req.lower, req.upper, req.tolerance_source = m_lo, m_hi, "model"
            req.flags.append("limits from the model, not printed: check")
    else:
        req.attribute = True
    return req


def _nominal_token(text: str) -> str:
    t = _PREFIX.sub("", text or "", count=1)
    m = re.search(NUM + r"°?", t)
    return m.group(0) if m else ""


def _number_in_text(v: float, text: str) -> bool:
    t = text or ""
    for tok in re.findall(r"\d+/\d+|\d*\.\d+|\d+", t):
        f = to_float(tok)
        if f is not None and abs(f - v) < 1e-9:
            return True
    # mixed fractions like "2 9/16"
    for tok in re.findall(r"\d+\s+\d+/\d+", text or ""):
        f = to_float(tok)
        if f is not None and abs(f - v) < 1e-9:
            return True
    return False


def evaluate(req: Requirement, measured: str) -> str:
    """'pass', 'fail' or 'needs review' for a measured value against a requirement."""
    s = (measured or "").strip().lower()
    if not s:
        return ""
    if req.attribute or req.lower is None or req.upper is None:
        if s in ("ok", "pass", "conforms", "c", "yes", "y", "accept"):
            return "pass"
        if s in ("nc", "fail", "no", "n", "reject", "does not conform"):
            return "fail"
        return "needs review"
    v = to_float(s)
    if v is None:
        return "needs review"
    eps = 1e-9
    return "pass" if req.lower - eps <= v <= req.upper + eps else "fail"
