#!/bin/bash
# Muse Glimmer 30B Q4_K_M + vision projector, served by llama.cpp (OpenAI API) on the VM's L4 GPU.
set -x
KEY=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/offline-key")
MODEL=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/model-file")
MODEL=${MODEL:-Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf}
mkdir -p /data && cd /data
for f in "$MODEL" mmproj-Muse-Glimmer-30B-Q4_K_M.gguf; do
  [ -s "$f" ] || curl -sL --retry 5 -o "$f" "https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/resolve/main/$f"
done
until nvidia-smi; do sleep 10; done
docker run -d --name glimmer --restart unless-stopped --gpus all -p 8000:8000 -v /data:/w \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m "/w/$MODEL" --mmproj /w/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8000 -ngl 999 -c 32768 -np 2 --jinja --api-key "$KEY" --alias muse-glimmer \
  --temp 1.0 --top-p 0.95 --top-k 64 --image-max-tokens 4096
