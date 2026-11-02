#!/bin/bash
# One L4 VM (g2-standard-8) for the Glimmer fine-tune. Hard stop: the VM deletes itself after MAX_RUN.
# g2-standard-8 rather than g2-standard-4: training keeps offloaded layers in host RAM and the merge needs ~32 GB.
# Usage: deploy/finetune/vm_create.sh ZONE [NAME]     STARTUP=deploy/finetune/vm_driver.sh runs the whole job on boot
set -euo pipefail
PROJECT=${PROJECT:?set PROJECT to your Google Cloud project}
ZONE=${1:?zone}
NAME=${2:-whiteout-ft}
MAX_RUN=${MAX_RUN:-11h}
EXTRA=()
[ -n "${STARTUP:-}" ] && EXTRA+=(--metadata-from-file "startup-script=$STARTUP")
gcloud compute instances create "$NAME" --project "$PROJECT" --zone "$ZONE" \
  --machine-type g2-standard-8 --accelerator type=nvidia-l4,count=1 \
  --maintenance-policy TERMINATE --provisioning-model STANDARD \
  --max-run-duration "$MAX_RUN" --instance-termination-action DELETE \
  --image-family common-cu129-ubuntu-2204-nvidia-580 --image-project deeplearning-platform-release \
  --boot-disk-size 400GB --boot-disk-type pd-ssd \
  --metadata "install-nvidia-driver=True,ft-max-hours=${MAX_RUN%h}" "${EXTRA[@]}" \
  --scopes storage-rw,logging-write,compute-rw \
  --labels project=whiteout,purpose=glimmer-finetune
gcloud compute disks add-labels "$NAME" --project "$PROJECT" --zone "$ZONE" --labels project=whiteout,purpose=glimmer-finetune
