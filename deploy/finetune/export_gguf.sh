#!/bin/bash
# On the VM: merge the adapter into the bf16 checkpoint, convert to GGUF and quantize to Q4_K_M (stock mmproj kept).
# Fallback when the merge or conversion fails: LORA=1 converts the adapter alone for llama-server --lora.
set -euxo pipefail
IMG=ghcr.io/ggml-org/llama.cpp:full-cuda
AD=${AD:-/data/ft/run1/adapter}
if [ "${LORA:-0}" = 1 ]; then
  # the image's Glimmer converter asks adapter tensors for .ndim, which LoraTorchTensor does not have
  sudo docker run --rm -v /data:/data --entrypoint bash $IMG -c "sed -i 's/tensor\.ndim/len(tensor.shape)/g' \
    /app/conversion/muse_glimmer.py && python3 /app/convert_lora_to_gguf.py '$AD' --base /data/base --outtype f16 \
    --outfile /data/gguf/ft-lora-f16.gguf"
  exit 0
fi
/data/venv/bin/python /data/ft/merge_lora.py --base /data/base --adapter "$AD" --out /data/merged
sudo docker run --rm -v /data:/data --entrypoint python3 $IMG /app/convert_hf_to_gguf.py /data/merged \
  --outtype bf16 --outfile /data/gguf/ft-bf16.gguf
sudo docker run --rm -v /data:/data --entrypoint /app/llama-quantize $IMG /data/gguf/ft-bf16.gguf \
  /data/gguf/Muse-Glimmer-30B-whiteout-ft-Q4_K_M.gguf Q4_K_M 8
rm -f /data/gguf/ft-bf16.gguf
ls -la /data/gguf
