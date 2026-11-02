#!/bin/bash
# On the fine-tune VM: Python env with Unsloth, model downloads, llama.cpp images. Logs in /data/*.log.
set -x
sudo mkdir -p /data && sudo chown "$USER" /data
cd /data
until nvidia-smi; do sleep 10; done
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv -p 3.12 /data/venv
. /data/venv/bin/activate
uv pip install unsloth "huggingface_hub[hf_xet]" hf_transfer
# 4-bit checkpoint for training (Unsloth dynamic bnb 4-bit, from meta-models/Muse-Glimmer-30B)
hf download unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit --local-dir /data/bnb4 > /data/dl_bnb4.log 2>&1
# bf16 checkpoint (for the merge) and the stock GGUF files, in the background
nohup hf download meta-models/Muse-Glimmer-30B --local-dir /data/base > /data/dl_base.log 2>&1 &
nohup sh -c 'for f in mmproj-Muse-Glimmer-30B-Q4_K_M.gguf dflash-Muse-Glimmer-30B-Q4_K_M.gguf Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf; do hf download meta-models/Muse-Glimmer-30B-GGUF $f --local-dir /data/gguf; done' > /data/dl_gguf.log 2>&1 &
# Docker + NVIDIA container toolkit (the Deep Learning VM image ships neither)
if ! command -v docker; then
  curl -fsSL https://get.docker.com | sudo sh
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update -q && sudo apt-get install -yq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi
nohup sh -c 'sudo docker pull ghcr.io/ggml-org/llama.cpp:server-cuda; sudo docker pull ghcr.io/ggml-org/llama.cpp:full-cuda' > /data/docker_pull.log 2>&1 &
echo SETUP_DONE
