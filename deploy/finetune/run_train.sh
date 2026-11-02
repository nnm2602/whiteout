#!/bin/bash
# On the VM: train, then copy the adapter and logs to the project bucket.
cd /data/ft
/data/venv/bin/python train.py --data /data/ft/data_1344x672 --model /data/bnb4 --out /data/ft/run1/adapter \
  --max-steps ${STEPS:-160} --accum 4 --lr 1.5e-4 --rank 16 --max-len 3200 --minutes ${MINUTES:-130}
echo "TRAIN_EXIT $?"
gcloud storage cp -r /data/ft/run1/adapter /data/ft/run1/train_log.jsonl gs://YOUR_BUCKET/run1/
echo "UPLOAD_EXIT $?"
