#!/bin/bash
# On the orchestrating machine, once: package the inputs, upload them, create the VM with vm_driver.sh as its
# startup script. After that the VM needs nothing from here. Status: gcloud storage cat gs://YOUR_BUCKET/status.txt
# Usage: deploy/finetune/launch.sh [ZONE] [NAME]
set -euo pipefail
cd "$(dirname "$0")/../.."
PROJECT=${PROJECT:?set PROJECT to your Google Cloud project}
ZONE=${1:-northamerica-northeast2-a}
NAME=${2:-whiteout-ft}
BUCKET=gs://YOUR_BUCKET
SA=$(gcloud compute project-info describe --project "$PROJECT" --format='value(defaultServiceAccount)')

# one GPU at a time
busy=$(gcloud compute instances list --project "$PROJECT" --filter='status!=TERMINATED' \
  --format='value(name,guestAccelerators[0].acceleratorType.basename())' | awk 'NF>1' || true)
[ -z "$busy" ] || { echo "a GPU VM is already up: $busy"; exit 1; }

# inputs: code, the 6 held-out sheets with their keys and baseline results, training crops (no key files, no call cache)
TMP=$(mktemp -d)
SHEETS=$(python3 -c "import json;print(' '.join(r['sheet'] for r in json.load(open('eval/results/keys_glimmer/SUMMARY.json'))['rows']))")
FILES=(pyproject.toml uv.lock README.md .python-version src eval/run_keys.py eval/results/keys_glimmer/SUMMARY.json
       deploy/finetune/*.py deploy/finetune/*.sh)
for s in $SHEETS; do
  FILES+=("eval/keys/$s.json" "eval/results/keys_main/$s.json")
  FILES+=("$(python3 -c "import json;print(json.load(open('eval/keys/$s.json'))['file'])")")
done
COPYFILE_DISABLE=1 tar -czf "$TMP/whiteout_bundle.tgz" --exclude __pycache__ --exclude '*.pyc' "${FILES[@]}"
COPYFILE_DISABLE=1 tar -czf "$TMP/data_1344x672.tgz" -C .cache/ft data_1344x672
if tar -tzf "$TMP/whiteout_bundle.tgz" | grep -iE 'key$|_key|token|secret|\.env|\.config' | grep -vE '^eval/keys/|^eval/run_keys.py$'; then
  echo "bundle has a file that looks like a credential, stopping"; exit 1
fi
tar -tzf "$TMP/whiteout_bundle.tgz" | sort -u | sed 's/^/  /'
gcloud storage cp "$TMP/whiteout_bundle.tgz" "$TMP/data_1344x672.tgz" $BUCKET/inputs/
rm -rf "$TMP"

# the VM's service account writes status, checkpoints and results to this bucket only
gcloud storage buckets add-iam-policy-binding $BUCKET --member "serviceAccount:$SA" --role roles/storage.objectAdmin > /dev/null
echo "$(date -u '+%F %T UTC') launch: inputs uploaded, creating $NAME in $ZONE" | gcloud storage cp - $BUCKET/status.txt

STARTUP=deploy/finetune/vm_driver.sh deploy/finetune/vm_create.sh "$ZONE" "$NAME"
# let the VM delete itself (and its disk) when the job ends: rights on this one VM and its disk only
gcloud compute instances add-iam-policy-binding "$NAME" --project "$PROJECT" --zone "$ZONE" \
  --member "serviceAccount:$SA" --role roles/compute.instanceAdmin.v1 > /dev/null
gcloud compute disks add-iam-policy-binding "$NAME" --project "$PROJECT" --zone "$ZONE" \
  --member "serviceAccount:$SA" --role roles/compute.storageAdmin > /dev/null
gcloud compute instances describe "$NAME" --project "$PROJECT" --zone "$ZONE" \
  --format='value(name,status,creationTimestamp,scheduling.maxRunDuration.seconds,labels)'
