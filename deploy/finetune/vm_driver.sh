#!/bin/bash
# Startup script of the Glimmer fine-tune VM. Runs the whole job on the VM by itself:
#   setup -> training data -> train (checkpoints copied to the bucket as they are saved) -> merge + GGUF ->
#   llama.cpp server -> answer-key test on the 6 held-out sheets -> results and logs to the bucket -> VM deletes
#   itself (falls back to shutdown -h). Each stage appends a line to $BUCKET/status.txt.
# Safe to run again after a reboot: finished stages are skipped, training resumes from its newest checkpoint and
# the test skips sheets that already have a result. The VM's --max-run-duration is the hard cost stop; a watchdog
# stops the work 25 minutes before it and uploads what exists.
# Inputs (uploaded by launch.sh): $BUCKET/inputs/whiteout_bundle.tgz, $BUCKET/inputs/data_1344x672.tgz
# No API keys are needed here: the test talks only to the local llama.cpp server.

BUCKET=gs://YOUR_BUCKET
MD=http://metadata.google.internal/computeMetadata/v1/instance
mdget() { curl -sf -H "Metadata-Flavor: Google" "$MD/$1"; }

# Detach from the startup-script runner, so the job never depends on it.
if [ "${WO_DETACHED:-}" != 1 ]; then
  mkdir -p /data
  mdget attributes/startup-script > /data/driver.sh
  WO_DETACHED=1 setsid nohup bash /data/driver.sh >> /data/driver.log 2>&1 < /dev/null &
  exit 0
fi
exec 9> /data/driver.lock
flock -n 9 || { echo "driver already running"; exit 0; }

export HOME=/root
export PATH=/root/.local/bin:/usr/lib/google-cloud-sdk/bin:/snap/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export DEBIAN_FRONTEND=noninteractive
STEPS=$(mdget attributes/ft-steps || echo 160)
MINUTES=$(mdget attributes/ft-minutes || echo 150)
MAX_HOURS=$(mdget attributes/ft-max-hours || echo 11)
TILE=1344x672x300
NAME=$(mdget name)
ZONE=$(mdget zone | awk -F/ '{print $NF}')
RUN=/data/ft/run
AD=$RUN/adapter
W=/data/whiteout
RES=eval/results/keys_glimmer_ft
ST=/data/stages
mkdir -p $ST /data/ft /data/gguf
[ -f /data/t0 ] || date +%s > /data/t0
T0=$(cat /data/t0)
DEADLINE=$(( T0 + MAX_HOURS * 3600 - 25 * 60 ))
set -x

status() {
  echo "$(date -u '+%F %T UTC') $*" >> /data/status.txt
  gcloud storage cp -q /data/status.txt $BUCKET/status.txt > /dev/null 2>&1 || true
}
stage_done() { touch "$ST/$1"; }
is_done() { [ -f "$ST/$1" ]; }
past_deadline() { [ -f $ST/DEADLINE ]; }

upload_logs() {
  mkdir -p /data/logs_out
  nvidia-smi > /data/logs_out/nvidia-smi.txt 2>&1
  uv pip freeze --python /data/venv/bin/python > /data/logs_out/pip_freeze_train.txt 2>/dev/null
  uv pip freeze --python $W/.venv/bin/python > /data/logs_out/pip_freeze_eval.txt 2>/dev/null
  docker logs glimmer > /data/logs_out/llama_server.log 2>&1
  df -h /data > /data/logs_out/disk.txt
  for f in /data/*.log /data/ft/*.log $RUN/train_log.jsonl $RUN/train.out; do [ -f "$f" ] && cp "$f" /data/logs_out/; done
  gcloud storage cp -q -r /data/logs_out/* $BUCKET/logs/ || true
}

finish() {
  status "$1"
  kill $WD $SYNC $ESYNC 2>/dev/null
  upload_logs
  # the GGUF upload started after the export may still be running: give it up to 40 minutes
  for _ in $(seq 80); do [ -f $ST/gguf_upload_running ] || break; sleep 30; done
  status "all uploads done, deleting this VM"
  # (a plain shutdown would leave a stopped VM whose 400 GB SSD keeps billing)
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet --delete-disks=all > /data/delete.log 2>&1 && sleep 600
  status "self-delete failed (see logs/delete.log), shutting down instead"
  gcloud storage cp -q /data/delete.log $BUCKET/logs/ || true
  shutdown -h now
  exit 0
}

# Watchdog: 25 minutes before the hard stop, stop the work and let the main flow upload what exists.
WD="" SYNC="" ESYNC=""
(
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 60; done
  touch $ST/DEADLINE
  status "time cap near: stopping training/test, uploading partial results"
  pkill -f /data/ft/train.py; pkill -f eval/run_keys.py
  sleep 60
  [ -d $RUN/checkpoints ] && gcloud storage rsync -q -r $RUN/checkpoints $BUCKET/ft/checkpoints
  [ -d $W/$RES ] && gcloud storage rsync -q -r $W/$RES $BUCKET/results/keys_glimmer_ft
  upload_logs
) &
WD=$!

status "driver start on $NAME ($ZONE), steps $STEPS, hard stop $(date -u -d @$((T0 + MAX_HOURS * 3600)) '+%F %T UTC')"

# ---------------------------------------------------------------- setup
until nvidia-smi > /dev/null 2>&1; do
  [ $(( $(date +%s) - T0 )) -gt 2400 ] && finish "FAILED: no GPU driver after 40 minutes"
  sleep 15
done

if ! is_done inputs; then
  status "setup: fetching inputs"
  gcloud storage cp $BUCKET/inputs/whiteout_bundle.tgz $BUCKET/inputs/data_1344x672.tgz /data/ \
    || finish "FAILED: could not fetch inputs from the bucket"
  rm -rf $W && mkdir -p $W && tar -xzf /data/whiteout_bundle.tgz -C $W
  tar -xzf /data/data_1344x672.tgz -C /data/ft
  cp $W/deploy/finetune/*.py $W/deploy/finetune/*.sh /data/ft/
  stage_done inputs
fi

if ! is_done data; then
  # Training crops come from Muse Spark's cached reads, made on the orchestrating machine (make_data.py) and
  # uploaded as inputs; nothing here can call Spark. Check they arrived whole.
  n=$(wc -l < /data/ft/data_1344x672/train.jsonl 2>/dev/null || echo 0)
  imgs=$(ls /data/ft/data_1344x672/images 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && [ "$n" -eq "$imgs" ] || finish "FAILED: training data incomplete ($n rows, $imgs images)"
  status "data: $n training crops ready"
  stage_done data
fi

dl_base() {
  for i in 1 2 3; do hf download meta-models/Muse-Glimmer-30B --local-dir /data/base >> /data/dl_base.log 2>&1 && { stage_done dl_base; return; }; sleep 60; done
}
dl_gguf() {
  for i in 1 2 3; do
    ok=1
    for f in mmproj-Muse-Glimmer-30B-Q4_K_M.gguf dflash-Muse-Glimmer-30B-Q4_K_M.gguf Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf; do
      hf download meta-models/Muse-Glimmer-30B-GGUF $f --local-dir /data/gguf >> /data/dl_gguf.log 2>&1 || ok=0
    done
    [ $ok = 1 ] && { stage_done dl_gguf; return; }; sleep 60
  done
}
docker_setup() {
  echo 'DPkg::Lock::Timeout "1800";' > /etc/apt/apt.conf.d/99wait-for-lock
  if ! command -v docker; then
    curl -fsSL https://get.docker.com | sh
  fi
  if ! command -v nvidia-ctk; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get -o DPkg::Lock::Timeout=900 update -q && apt-get -o DPkg::Lock::Timeout=900 install -yq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
  fi
  docker pull ghcr.io/ggml-org/llama.cpp:server-cuda && docker pull ghcr.io/ggml-org/llama.cpp:full-cuda && stage_done docker
}
eval_env() {
  # Tesseract 5 (the Mac runs 5.5; Ubuntu 22.04 ships 4.1), then the Whiteout package from uv.lock
  apt-get install -yq software-properties-common
  add-apt-repository -y ppa:alex-p/tesseract-ocr5 || true
  apt-get update -q
  apt-get install -yq tesseract-ocr
  tesseract --version
  cd $W && { uv sync --frozen || uv sync; } && .venv/bin/python -c "import whiteout.cli, whiteout.pipeline" \
    && command -v tesseract && stage_done eval_env
  cd /
}

if ! is_done setup; then
  status "setup: Python env with Unsloth, 4-bit checkpoint"
  command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  [ -x /data/venv/bin/python ] || uv venv -p 3.12 /data/venv
  uv pip install --python /data/venv/bin/python unsloth "huggingface_hub[hf_xet]" hf_transfer > /data/setup_pip.log 2>&1 \
    || finish "FAILED: pip install (see logs/setup_pip.log)"
  stage_done setup
fi
export PATH=/data/venv/bin:$PATH
# background: bf16 checkpoint + stock GGUFs (for the merge and the server), Docker + llama.cpp, test env
is_done dl_base || (dl_base) &
is_done dl_gguf || (dl_gguf) &
# one after the other: both use apt
( is_done docker || docker_setup > /data/docker_setup.log 2>&1; is_done eval_env || eval_env > /data/eval_env.log 2>&1 ) &

# a VM after one that finished training: take the adapter from the bucket instead of training again
if ! is_done train && gcloud storage ls $BUCKET/ft/adapter/adapter_model.safetensors > /dev/null 2>&1; then
  mkdir -p $RUN && gcloud storage cp -r $BUCKET/ft/adapter $RUN/ && gcloud storage cp $BUCKET/ft/train_log.jsonl $RUN/ \
    && [ -f $AD/adapter_model.safetensors ] && stage_done train && status "train: reusing the trained adapter from $BUCKET/ft/adapter/"
fi

if ! is_done train && ! is_done dl_bnb4; then
  for i in 1 2 3; do
    hf download unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit --local-dir /data/bnb4 >> /data/dl_bnb4.log 2>&1 && { stage_done dl_bnb4; break; }
    sleep 60
  done
  is_done dl_bnb4 || finish "FAILED: 4-bit checkpoint download (see logs/dl_bnb4.log)"
fi

# ---------------------------------------------------------------- train
train_sync() {
  last=""
  while true; do
    sleep 300
    [ -d $RUN/checkpoints ] && gcloud storage rsync -q -r --delete-unmatched-destination-objects \
      $RUN/checkpoints $BUCKET/ft/checkpoints > /dev/null 2>&1
    [ -f $RUN/train_log.jsonl ] && gcloud storage cp -q $RUN/train_log.jsonl $BUCKET/ft/train_log.jsonl > /dev/null 2>&1
    p=$(python3 - <<EOF 2>/dev/null
import json
rows = [json.loads(l) for l in open("$RUN/train_log.jsonl") if '"loss"' in l]
r = rows[-1]
print(f"step {r['step']}/$STEPS loss {r['loss']:.3f} ({r['min']:.0f} min, peak {r['peak_gib']} GiB)")
EOF
)
    [ -n "$p" ] && [ "$p" != "$last" ] && status "train: $p" && last=$p
  done
}

if ! is_done train && ! past_deadline; then
  mkdir -p $RUN
  # a fresh VM after a lost one: continue from the checkpoints already in the bucket
  if ! ls -d $RUN/checkpoints/checkpoint-* > /dev/null 2>&1 && gcloud storage ls $BUCKET/ft/checkpoints/ > /dev/null 2>&1; then
    gcloud storage cp -r $BUCKET/ft/checkpoints $RUN/ && gcloud storage cp $BUCKET/ft/train_log.jsonl $RUN/ || true
  fi
  status "train: starting ($STEPS steps, about 42 s per step)"
  train_sync &
  SYNC=$!
  cd /data/ft
  python /data/ft/train.py --data /data/ft/data_1344x672 --model /data/bnb4 --out $AD \
    --max-steps $STEPS --accum 4 --lr 1.5e-4 --rank 16 --max-len 3200 --minutes $MINUTES --resume > $RUN/train.out 2>&1
  rc=$?
  cd /
  kill $SYNC 2>/dev/null
  grep -E "trained|SAMPLE|Error|error" $RUN/train.out | tail -5
  gcloud storage rsync -q -r --delete-unmatched-destination-objects $RUN/checkpoints $BUCKET/ft/checkpoints > /dev/null 2>&1
  gcloud storage cp -q $RUN/train_log.jsonl $RUN/train.out $BUCKET/ft/ > /dev/null 2>&1
  if [ -f $AD/adapter_model.safetensors ]; then
    gcloud storage cp -q -r $AD $BUCKET/ft/ || status "warning: adapter upload failed"
    status "train: done (exit $rc), adapter uploaded to $BUCKET/ft/adapter/ ($(grep -o 'trained [0-9]* steps in [0-9.]* min' $RUN/train.out))"
    stage_done train
  else
    upload_logs
    finish "FAILED: training exit $rc, no adapter (see ft/train.out)"
  fi
fi
past_deadline && finish "STOPPED at the time cap after training"

# ---------------------------------------------------------------- merge + GGUF
status "export: waiting for the bf16 checkpoint, stock GGUFs, Docker"
for _ in $(seq 240); do is_done dl_base && is_done dl_gguf && is_done docker && break; sleep 30; done
is_done dl_gguf && is_done docker || finish "FAILED: stock GGUF or Docker setup missing after 2 h of waiting"

gguf_upload() {
  touch $ST/gguf_upload_running
  gcloud storage cp -q "$1" $BUCKET/ft/gguf/ > /data/gguf_upload.log 2>&1
  rm -f $ST/gguf_upload_running
}

if ! is_done export && ! past_deadline; then
  if is_done dl_base; then
    status "export: merging the adapter into bf16, converting, quantizing to Q4_K_M"
    AD=$AD bash /data/ft/export_gguf.sh > /data/export.log 2>&1 && echo merged > $ST/mode
  fi
  if [ ! -f $ST/mode ]; then
    status "export: merge path failed or base missing, converting the adapter alone for llama-server --lora"
    LORA=1 AD=$AD bash /data/ft/export_gguf.sh > /data/export_lora.log 2>&1 && echo lora > $ST/mode \
      || finish "FAILED: both GGUF exports failed (see logs/export.log, logs/export_lora.log)"
  fi
  rm -rf /data/merged
  stage_done export
  if [ "$(cat $ST/mode)" = merged ]; then
    (gguf_upload /data/gguf/Muse-Glimmer-30B-whiteout-ft-Q4_K_M.gguf) &
  else
    (gguf_upload /data/gguf/ft-lora-f16.gguf) &
  fi
  status "export: done ($(cat $ST/mode)), GGUF upload to $BUCKET/ft/gguf/ started"
fi
past_deadline && finish "STOPPED at the time cap after export"

# ---------------------------------------------------------------- serve + test
serve() {
  if [ "$(cat $ST/mode)" = merged ]; then
    MODEL=Muse-Glimmer-30B-whiteout-ft-Q4_K_M.gguf timeout 1800 bash /data/ft/serve.sh
  else
    MODEL=Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf LORA=ft-lora-f16.gguf timeout 1800 bash /data/ft/serve.sh
  fi
}
status "serve: starting llama.cpp ($(cat $ST/mode))"
if ! serve > /data/serve.log 2>&1; then
  docker logs glimmer > /data/serve_fail_merged.log 2>&1
  if [ "$(cat $ST/mode)" = merged ]; then
    status "serve: merged model failed to start, trying the adapter with --lora"
    LORA=1 AD=$AD bash /data/ft/export_gguf.sh > /data/export_lora.log 2>&1 && echo lora > $ST/mode \
      && serve >> /data/serve.log 2>&1 || finish "FAILED: llama.cpp server did not start (see logs/serve.log)"
  else
    finish "FAILED: llama.cpp server did not start (see logs/serve.log)"
  fi
fi
for _ in $(seq 60); do is_done eval_env && break; sleep 30; done
is_done eval_env || finish "FAILED: test environment setup (see logs/eval_env.log)"

LABEL="Muse Glimmer 30B fine-tuned on Spark tile reads (Q4_K_M + stock mmproj + DFlash)"
[ "$(cat $ST/mode)" = lora ] && LABEL="Muse Glimmer 30B stock Q4_K_M + fine-tuned LoRA (f16 GGUF) + stock mmproj + DFlash"
compare() {
  cd $W && .venv/bin/python deploy/finetune/compare.py --ft $RES --calls .cache/calls_ft --label "$LABEL" > /data/compare.log 2>&1
  cd /
}
eval_sync() {
  last=-1
  while true; do
    sleep 300
    n=$(ls $W/$RES/*__p*.json 2>/dev/null | wc -l)
    gcloud storage rsync -q -r $W/$RES $BUCKET/results/keys_glimmer_ft > /dev/null 2>&1
    if [ "$n" != "$last" ]; then
      [ "$n" -gt 0 ] && compare && gcloud storage cp -q $W/$RES/COMPARISON.json $BUCKET/results/keys_glimmer_ft/ > /dev/null 2>&1
      status "test: $n of 6 sheets done"
      last=$n
    fi
  done
}

if ! is_done eval && ! past_deadline; then
  cd $W
  KEYS=$(python3 -c "
import json, os
rows = json.load(open('eval/results/keys_glimmer/SUMMARY.json'))['rows']
print(','.join('eval/keys/' + r['sheet'] + '.json' for r in rows if not os.path.exists('$RES/' + r['sheet'] + '.json')))")
  cd /
  status "test: answer-key run on the 6 held-out sheets, tile $TILE (remaining: ${KEYS:-none})"
  eval_sync &
  ESYNC=$!
  if [ -n "$KEYS" ]; then
    cd $W
    env -u OFFLINE_API_KEY WHITEOUT_CACHE=$W/.cache/calls_ft WHITEOUT_OFFLINE_URL=http://127.0.0.1:8000/v1 \
      WHITEOUT_OFFLINE_TILE=$TILE .venv/bin/python eval/run_keys.py --engine offline --keys "$KEYS" --out $RES \
      --batch 1 >> /data/eval.log 2>&1
    echo "run_keys exit $?" >> /data/eval.log
    cd /
  fi
  kill $ESYNC 2>/dev/null
  compare
  gcloud storage rsync -q -r $W/$RES $BUCKET/results/keys_glimmer_ft > /dev/null 2>&1
  past_deadline || stage_done eval
fi
summary=$(python3 -c "
import json
d = json.load(open('$W/$RES/COMPARISON.json'))
print(f\"{d['sheets']} sheets: fine-tuned {d['ft_found']}/{d['items']}, stock {d['stock_found']}/{d['items']}, Spark {d['spark_found']}/{d['items']}; extras ft {d['ft_extra_total']} vs stock {d['stock_extra_total']}; median min/sheet ft {d['ft_median_minutes_per_sheet']} vs stock {d['stock_median_minutes_per_sheet']}\")
" 2>/dev/null)
finish "DONE: ${summary:-no comparison written (see logs/eval.log)}"
