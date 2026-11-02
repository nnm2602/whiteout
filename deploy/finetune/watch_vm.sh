#!/bin/bash
# Runs on the orchestrating machine. Deletes the fine-tune VM when the heartbeat file goes stale
# (the machine that drives the run has stopped), and exits once the VM is gone.
# Usage: deploy/finetune/watch_vm.sh NAME ZONE HEARTBEAT_FILE [STALE_MINUTES]
PROJECT=${PROJECT:?set PROJECT to your Google Cloud project}
NAME=$1; ZONE=$2; HB=$3; STALE=${4:-90}
while true; do
  st=$(gcloud compute instances describe "$NAME" --zone "$ZONE" --project "$PROJECT" --format='value(status)' 2>/dev/null)
  if [ -z "$st" ]; then echo "$(date) $NAME gone, watcher exits"; exit 0; fi
  age=$(( ( $(date +%s) - $(stat -f %m "$HB" 2>/dev/null || echo 0) ) / 60 ))
  if [ "$age" -ge "$STALE" ]; then
    echo "$(date) heartbeat ${age} min old: deleting $NAME"
    gcloud compute instances delete "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet
  fi
  sleep 120
done
