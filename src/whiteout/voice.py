"""Muse Voice Transcribe for push-to-talk measurements, and a strict parser for what the inspector said.

The parser only accepts "<balloon number> <value>" pairs that match a real Form 3 row; anything else is ignored,
so a misheard word never lands in the report.
"""

from __future__ import annotations

import json
import re

import httpx

from .model_client import META_BASE, api_key

KEYWORDS = ["balloon", "point", "datum", "true position", "flatness", "conforms", "thou", "minus", "plus"]

_ONES = {"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
         "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


async def transcribe(wav: bytes) -> str:
    req = {"model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV", "keywords": KEYWORDS}
    files = {"request": (None, json.dumps(req), "application/json"), "audio": ("measure.wav", wav, "audio/wav")}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{META_BASE}/asr/transcribe", headers={"Authorization": f"Bearer {api_key()}"},
                         files=files)
    if r.status_code != 200:
        raise RuntimeError(f"transcribe failed: HTTP {r.status_code} {r.text[:200]}")
    try:
        j = r.json()
    except ValueError:
        return r.text.strip()
    return (j.get("text") or j.get("transcript") or " ".join(s.get("text", "") for s in j.get("segments", []))).strip()


def words_to_digits(text: str) -> str:
    """'point two five one two' -> '.2512'; 'fourteen' -> '14'; numbers already written as digits stay as they are."""
    t = text.lower().replace("-", " ")
    t = re.sub(r"\.(\s|$)", r" , \1", t)  # sentence-ending periods, not decimal points
    t = re.sub(r"[,;:]", " , ", t)
    toks = t.split()
    out: list[str] = []
    spoken: list[str] = []  # a run of spoken digits and "point", joined into one number

    def flush():
        if spoken:
            out.append("".join(spoken))
            spoken.clear()

    i = 0
    while i < len(toks):
        w = toks[i]
        if w in _TENS:
            v = _TENS[w]
            if i + 1 < len(toks) and toks[i + 1] in _ONES and _ONES[toks[i + 1]] < 10:
                v += _ONES[toks[i + 1]]
                i += 1
            spoken.append(str(v))
        elif w in _ONES:
            v = _ONES[w]
            # "fourteen" after a finished number starts a new one
            if v >= 10 and spoken:
                flush()
            spoken.append(str(v))
        elif w in ("point", "dot", "decimal"):
            spoken.append(".")
        else:
            flush()
            out.append(w)
        i += 1
    flush()
    return " ".join(out)


def parse_dictation(text: str, valid_rows: list[str]) -> list[tuple[str, str]]:
    s = words_to_digits(text)
    s = re.sub(r"\b(balloon|number|characteristic|char|item|row)\b", " # ", s)
    s = s.replace("conforms", " OK ").replace("conform", " OK ").replace("good", " OK ")
    entries = []
    valid = set(valid_rows)
    for part in re.split(r"#|\bnext\b", s):
        part = part.replace(",", " ")
        m = re.search(r"(\d+(?:\.\d+)?(?:-\d+)?)\s+(?:is\s+|measures\s+|reads\s+)?(-?\d*\.\d+|\d+|OK)\b", part)
        if not m:
            continue
        rid, val = m.group(1), m.group(2)
        if rid not in valid:
            # "9 2" spoken for row 9.2
            continue
        entries.append((rid, val))
    return entries


def parse_value(text: str) -> str | None:
    """Guided measuring: the line is already chosen, so the inspector says only the value ("point one eight eight
    five", "conforms"). Returns the value text, or None if nothing usable was heard."""
    s = words_to_digits(text).replace(",", " ")
    if re.search(r"\b(conforms?|good|ok|okay|accept(ed)?|pass(es)?)\b", s, re.I):
        return "OK"
    if re.search(r"\b(reject(ed)?|fail(s|ed)?|no good|does not conform|nonconform\w*)\b", s, re.I):
        return "NC"
    neg = re.search(r"\bminus\b", s)
    m = re.search(r"-?\d*\.\d+|-?\d+", s)
    if not m:
        return None
    v = m.group(0)
    return ("-" + v.lstrip("-")) if neg else v
