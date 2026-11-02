"""QLoRA fine-tune of Muse Glimmer 30B on drawing crops (Unsloth, one 24 GB L4, vision tower frozen).

The answer is trained as a direct reply (assistant to=user, no reasoning turn), so the tuned model writes the JSON
straight away. Loss is on the reply only.

  python train.py --data /data/ft/data_1344x672 --model /data/bnb4 --out /data/ft/adapter --max-steps 150
  python train.py ... --probe        # 3 steps on the longest examples: peak memory and seconds per step
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="/data/bnb4")
ap.add_argument("--out", default="/data/ft/adapter")
ap.add_argument("--max-steps", type=int, default=150)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--max-len", type=int, default=3200)
ap.add_argument("--minutes", type=float, default=170, help="stop training after this many minutes")
ap.add_argument("--probe", action="store_true")
ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint next to --out, if there is one")
ap.add_argument("--offload-layers", default="auto", help='Unsloth block swap: "auto" keeps layers the GPU cannot hold in host RAM')
a = ap.parse_args()

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from PIL import Image  # noqa: E402
from unsloth import FastVisionModel  # noqa: E402
from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

t_load = time.time()
kw = dict(load_in_4bit=True, use_gradient_checkpointing="unsloth", max_seq_length=a.max_len)
if a.offload_layers != "none":
    kw["offload_layers"] = a.offload_layers
try:
    model, processor = FastVisionModel.from_pretrained(a.model, offload_embedding=True, **kw)
except TypeError as e:
    print("offload_embedding not accepted:", e, flush=True)
    model, processor = FastVisionModel.from_pretrained(a.model, **kw)
print(f"loaded in {time.time() - t_load:.0f}s, GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

model = FastVisionModel.get_peft_model(
    model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
    finetune_mlp_modules=True, r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none", random_state=3407)

data = Path(a.data)
rows = [json.loads(line) for line in open(data / "train.jsonl")]
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
prompt_tokens = len(tok(rows[0]["prompt"], add_special_tokens=False)["input_ids"])
for r in rows:
    r["answer_tokens"] = len(tok(r["answer"], add_special_tokens=False)["input_ids"])
w, h = Image.open(data / rows[0]["image"]).size
vis = math.ceil(w / 28) * math.ceil(h / 28)
budget = a.max_len - prompt_tokens - vis - 80  # chat template and system turn
fit = [r for r in rows if r["answer_tokens"] <= budget]
print(f"{len(rows)} examples, {len(fit)} fit {a.max_len} tokens (prompt {prompt_tokens}, image {vis}, "
      f"answer budget {budget})", flush=True)
if a.probe:
    fit = sorted(fit, key=lambda r: -r["answer_tokens"])[:3]
    a.max_steps, a.accum = 3, 1
else:
    random.Random(0).shuffle(fit)
    fit = fit[: a.max_steps * a.accum]
print(f"training on {len(fit)} examples, answer tokens {sum(r['answer_tokens'] for r in fit)}", flush=True)


def to_messages(batch):
    """Lazy transform: build chat messages with the crop image only when a row is read."""
    out = []
    for i in batch["idx"]:
        r = fit[i]
        img = Image.open(data / r["image"]).convert("RGB")
        out.append([
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": r["prompt"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]},
        ])
    return {"messages": out}


train_ds = Dataset.from_list([{"idx": i} for i in range(len(fit))])
train_ds.set_transform(to_messages)


class Budget(TrainerCallback):
    def __init__(self):
        self.t0 = time.time()
        self.log = open(Path(a.out).parent / "train_log.jsonl", "a")

    def on_log(self, args, state, control, logs=None, **kw):
        rec = {"step": state.global_step, "min": round((time.time() - self.t0) / 60, 2),
               "peak_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2), **(logs or {})}
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()
        print(rec, flush=True)

    def on_step_end(self, args, state, control, **kw):
        if (time.time() - self.t0) / 60 > a.minutes:
            print("time budget reached, stopping", flush=True)
            control.should_training_stop = True
            control.should_save = True


Path(a.out).parent.mkdir(parents=True, exist_ok=True)
FastVisionModel.for_training(model)
collator = UnslothVisionDataCollator(model, processor, resize="max", train_on_responses_only=True,
                                     instruction_part="<|start|>user<|message|>", response_part="<|start|>assistant")
trainer = SFTTrainer(
    model=model, tokenizer=processor, data_collator=collator, train_dataset=train_ds, callbacks=[Budget()],
    args=SFTConfig(
        per_device_train_batch_size=1, gradient_accumulation_steps=a.accum, warmup_steps=min(8, a.max_steps // 10),
        max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine", optim="adamw_8bit",
        weight_decay=0.0, logging_steps=1, seed=3407, output_dir=str(Path(a.out).parent / "checkpoints"),
        save_strategy="steps", save_steps=40, save_total_limit=2, report_to="none", bf16=True,
        remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
        max_length=a.max_len, dataloader_num_workers=2,
    ),
)
t0 = time.time()
resume = a.resume and any((Path(a.out).parent / "checkpoints").glob("checkpoint-*"))
print("resuming from the newest checkpoint" if resume else "training from step 0", flush=True)
stats = trainer.train(resume_from_checkpoint=True if resume else None)
print(f"trained {stats.global_step} steps in {(time.time() - t0) / 60:.1f} min, "
      f"peak {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB", flush=True)
if a.probe:
    raise SystemExit(0)
model.save_pretrained(a.out)
processor.save_pretrained(a.out)
json.dump({"steps": stats.global_step, "examples": len(fit), "minutes": round((time.time() - t0) / 60, 1),
           "peak_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2), "rank": a.rank, "lr": a.lr,
           "accum": a.accum, "max_len": a.max_len, "train_loss": stats.training_loss},
          open(Path(a.out) / "train_summary.json", "w"), indent=1)

# Sanity check on one training crop: the reply should start with the JSON, no reasoning turn.
FastVisionModel.for_inference(model)
r = fit[0]
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["prompt"]}]}]
text = processor.apply_chat_template(msgs, add_generation_prompt=True)
inp = processor(images=[Image.open(data / r["image"]).convert("RGB")], text=[text], add_special_tokens=False,
                return_tensors="pt").to("cuda")
out = model.generate(**inp, max_new_tokens=200, do_sample=False)
print("SAMPLE:", repr(tok.decode(out[0][inp["input_ids"].shape[1]:])[:400]), flush=True)
