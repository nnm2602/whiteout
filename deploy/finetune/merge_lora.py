"""Merge the LoRA adapter into the bf16 Muse Glimmer checkpoint, one tensor at a time (fits in 32 GB of RAM).

  python merge_lora.py --base /data/base --adapter /data/ft/adapter --out /data/merged

Only language-model weights carry LoRA (the vision tower was frozen), so the stock mmproj stays valid.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/data/base")
ap.add_argument("--adapter", default="/data/ft/adapter")
ap.add_argument("--out", default="/data/merged")
ap.add_argument("--shard-gb", type=float, default=4.0)
a = ap.parse_args()
base, ad, out = Path(a.base), Path(a.adapter), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

cfg = json.loads((ad / "adapter_config.json").read_text())
r, alpha = cfg["r"], cfg["lora_alpha"]
scale = alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)
lora: dict[str, dict[str, torch.Tensor]] = {}
with safe_open(str(ad / "adapter_model.safetensors"), "pt") as f:
    for k in f.keys():
        m = re.match(r"(.*)\.lora_([AB])(?:\.default)?\.weight$", k)
        if not m:
            raise SystemExit(f"unexpected adapter tensor {k}")
        lora.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k)


DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
          "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
          "BOOL": torch.bool}


def read_shard(path: Path):
    """Yield (name, tensor) from a safetensors file with plain reads. safe_open memory-maps the whole file, and the
    kernel refuses a 50 GB private mapping on a 32 GB machine."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        start = 8 + n
        for k, v in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            b0, b1 = v["data_offsets"]
            f.seek(start + b0)
            buf = bytearray(f.read(b1 - b0))
            dt = DTYPES[v["dtype"]]
            t = torch.frombuffer(buf, dtype=dt) if buf else torch.empty(0, dtype=dt)
            yield k, t.reshape(v["shape"])


def tail(name: str) -> str:
    """'...layers.12.self_attn.q_proj' -> 'layers.12.self_attn.q_proj'"""
    i = name.find("layers.")
    return name[i:] if i >= 0 else name


index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
text_keys = {tail(k[: -len(".weight")]): k for k in index if k.endswith(".weight") and "layers." in k
             and "vision" not in k and "visual" not in k}
plan = {}
for mod in lora:
    if "vision" in mod or "visual" in mod:
        raise SystemExit(f"adapter touches the vision tower: {mod}")
    t = tail(mod)
    if t not in text_keys:
        raise SystemExit(f"no base weight for adapter module {mod} (tail {t})")
    plan[text_keys[t]] = mod
print(f"{len(lora)} LoRA modules, scale {scale}, e.g. {next(iter(plan.items()))}", flush=True)

new_index, shard, shard_bytes, n_shard, done = {}, {}, 0, 0, 0


def flush():
    global shard, shard_bytes, n_shard
    if not shard:
        return
    n_shard += 1
    name = f"model-{n_shard:05d}.safetensors"
    save_file(shard, str(out / name), metadata={"format": "pt"})
    for k in shard:
        new_index[k] = name
    shard, shard_bytes = {}, 0


for fname in sorted(set(index.values())):
    for k, t in read_shard(base / fname):
        if k in plan:
            ab = lora[plan[k]]
            delta = (ab["B"].float() @ ab["A"].float()) * scale
            t = (t.float() + delta).to(t.dtype)
            done += 1
        shard[k] = t.contiguous()
        shard_bytes += t.numel() * t.element_size()
        if shard_bytes >= a.shard_gb * 2**30:
            flush()
flush()
names = sorted(set(new_index.values()))
final = {}
for i, n in enumerate(names, 1):
    nn = f"model-{i:05d}-of-{len(names):05d}.safetensors"
    (out / n).rename(out / nn)
    final[n] = nn
(out / "model.safetensors.index.json").write_text(json.dumps(
    {"metadata": {}, "weight_map": {k: final[v] for k, v in new_index.items()}}, indent=1))
for p in base.iterdir():
    if p.is_file() and not p.name.endswith(".safetensors") and p.name != "model.safetensors.index.json":
        shutil.copy(p, out / p.name)
print(f"merged {done} of {len(plan)} weights into {len(names)} shards", flush=True)
if done != len(plan):
    raise SystemExit("not every adapter module was merged")
