#!/bin/bash
# On the orchestrating machine: SSH tunnel to the VM's llama-server, then the answer-key test on the 6 sheets that
# stock Glimmer was measured on, with the tuned model's tile size. Results: eval/results/keys_glimmer_ft/.
set -euo pipefail
cd "$(dirname "$0")/../.."
VM=${VM:-whiteout-ft}; ZONE=${ZONE:-northamerica-northeast2-a}; PROJECT=${PROJECT:?set PROJECT to your Google Cloud project}
PORT=${PORT:-18000}
OUT=${OUT:-eval/results/keys_glimmer_ft}
KEYS=$(python3 -c "import json;print(','.join('eval/keys/'+r['sheet']+'.json' for r in json.load(open('eval/results/keys_glimmer/SUMMARY.json'))['rows']))")
# keep the tunnel up for the whole run
( while true; do gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" -- -N -o ServerAliveInterval=30 \
    -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:8000"; sleep 5; done ) &
TUN=$!
trap 'kill $TUN 2>/dev/null; pkill -f "L $PORT:127.0.0.1:8000" 2>/dev/null' EXIT
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 3; done
# separate call cache, so no answer of the stock model is reused
WHITEOUT_CACHE=.cache/calls_ft WHITEOUT_OFFLINE_URL="http://127.0.0.1:$PORT/v1" \
  WHITEOUT_OFFLINE_TILE=${TILE:-1344x672x300} \
  caffeinate -i .venv/bin/python eval/run_keys.py --engine offline --keys "$KEYS" --out "$OUT" --batch 1
.venv/bin/python deploy/finetune/compare.py --ft "$OUT" --calls .cache/calls_ft
