"""Vision model client with an on-disk cache.

Two backends share one interface:
- "meta": Meta Model API Responses endpoint (Muse Spark), strict JSON schema output.
- "openai": any OpenAI-compatible chat server (llama.cpp serving Muse Glimmer offline).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from PIL import Image

META_BASE = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
CACHE_DIR = Path(os.environ.get("WHITEOUT_CACHE", Path(__file__).resolve().parents[2] / ".cache" / "calls"))

# USD per 1M tokens (input, output), from the Meta pricing page, Oct 2026.
PRICES = {
    "muse-spark-1.3": (1.25, 4.25),
    "muse-spark-1.2": (1.25, 4.25),
}


def api_key() -> str:
    key = os.environ.get("MODEL_API_KEY")
    if key:
        return key.strip()
    path = Path(os.environ.get("MODEL_API_KEY_FILE", "~/.config/whiteout/meta_api_key")).expanduser()
    return path.read_text().strip()


def png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0
    usd: float = 0.0

    def add(self, other: "Usage") -> None:
        for k in ("calls", "cached", "input_tokens", "output_tokens", "reasoning_tokens", "seconds", "usd"):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def as_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


@dataclass
class ModelConfig:
    backend: str = "meta"  # "meta" or "openai"
    model: str = "muse-spark-1.3"
    effort: str = "medium"
    base_url: str = META_BASE
    concurrency: int = 8
    timeout: float = 600.0
    label: str = "Muse Spark 1.3"
    audit_effort: str = "low"  # the erased-page search; the confirm step uses the main effort
    extra: dict = field(default_factory=dict)
    tile: tuple[int, ...] = ()  # read tiles (width, height[, overlap]); () = the default 1775 x 1175 grid


class ModelError(RuntimeError):
    pass


class VisionClient:
    def __init__(self, cfg: ModelConfig | None = None, use_cache: bool = True):
        self.cfg = cfg or ModelConfig()
        self.use_cache = use_cache
        self.usage = Usage()
        self._sem = asyncio.Semaphore(self.cfg.concurrency)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, payload: dict) -> Path:
        h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return CACHE_DIR / f"{h}.json"

    async def ask_json(self, images: list[Image.Image], prompt: str, schema: dict, name: str = "result",
                       effort: str | None = None) -> dict:
        """Send images + prompt, return parsed JSON matching schema."""
        urls = [png_data_url(im) for im in images]
        effort = effort or self.cfg.effort
        key_payload = {
            "backend": self.cfg.backend,
            "model": self.cfg.model,
            "effort": effort,
            "prompt": prompt,
            "schema": schema,
            "images": [hashlib.sha256(u.encode()).hexdigest() for u in urls],
        }
        cpath = self._cache_path(key_payload)
        if self.use_cache and cpath.exists():
            rec = json.loads(cpath.read_text())
            u = Usage(cached=1)
            self.usage.add(u)
            return rec["result"]

        async with self._sem:
            t0 = time.time()
            if self.cfg.backend == "meta":
                result, usage = await self._meta(urls, prompt, schema, name, effort)
            else:
                result, usage = await self._openai(urls, prompt, schema, name)
            usage.seconds = time.time() - t0
            usage.calls = 1
        self.usage.add(usage)
        cpath.write_text(json.dumps({"result": result, "usage": usage.as_dict(), "model": self.cfg.model}))
        return result

    async def _post(self, url: str, body: dict, headers: dict) -> dict:
        delay = 2.0
        last = None
        async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
            for attempt in range(6):
                try:
                    r = await client.post(url, json=body, headers=headers)
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last = f"{type(e).__name__}: {e}"
                else:
                    if r.status_code == 200:
                        return r.json()
                    last = f"HTTP {r.status_code}: {r.text[:400]}"
                    if r.status_code in (401, 402, 403) or re.search(r"credit|billing|quota|insufficient", r.text, re.I):
                        print(f"WHITEOUT_ALERT Meta refused a model call ({last[:160]}). Uploads and questions may be failing; check the API key and credits.", flush=True)
                    if r.status_code not in (408, 409, 429, 500, 502, 503, 504):
                        raise ModelError(last)
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 30)
        raise ModelError(f"gave up after retries: {last}")

    async def _meta(self, urls, prompt, schema, name, effort):
        content = [{"type": "input_image", "image_url": u, "detail": "original"} for u in urls]
        content.append({"type": "input_text", "text": prompt})
        body = {
            "model": self.cfg.model,
            "reasoning": {"effort": effort},
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        }
        headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
        r = await self._post(f"{self.cfg.base_url}/responses", body, headers)
        txt = "".join(
            c.get("text", "")
            for o in r.get("output", [])
            if o.get("type") == "message"
            for c in o.get("content", [])
        )
        u = r.get("usage", {}) or {}
        usage = Usage(
            input_tokens=u.get("input_tokens", 0) or 0,
            output_tokens=u.get("output_tokens", 0) or 0,
            reasoning_tokens=(u.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
        )
        pin, pout = PRICES.get(self.cfg.model, (1.25, 4.25))
        usage.usd = usage.input_tokens * pin / 1e6 + usage.output_tokens * pout / 1e6
        try:
            return json.loads(txt), usage
        except json.JSONDecodeError as e:
            raise ModelError(f"bad JSON from model: {txt[:300]}") from e

    async def _openai(self, urls, prompt, schema, name):
        content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
        content.append({"type": "text", "text": prompt + "\n\nReply with JSON only, matching this schema:\n" + json.dumps(schema)})
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
            **self.cfg.extra,
        }
        headers = {"Content-Type": "application/json"}
        if os.environ.get("OFFLINE_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['OFFLINE_API_KEY']}"
        r = await self._post(f"{self.cfg.base_url}/chat/completions", body, headers)
        txt = r["choices"][0]["message"].get("content") or ""
        u = r.get("usage", {}) or {}
        usage = Usage(input_tokens=u.get("prompt_tokens", 0) or 0, output_tokens=u.get("completion_tokens", 0) or 0)
        return parse_loose_json(txt), usage


def parse_loose_json(txt: str) -> dict:
    """Parse JSON that may be wrapped in a code fence or preceded by reasoning text."""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, flags=re.S)
    if m:
        txt = m.group(1)
    start = txt.find("{")
    end = txt.rfind("}")
    if start < 0 or end < 0:
        raise ModelError(f"no JSON object in reply: {txt[:300]}")
    return json.loads(txt[start : end + 1])
