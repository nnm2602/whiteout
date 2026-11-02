"""Muse Glimmer 30B (Meta open weights, Q4_K_M GGUF + vision projector) served by llama.cpp on a Modal L4.

The server speaks the OpenAI chat API, so Whiteout's offline engine points at it with WHITEOUT_OFFLINE_URL.
Drawings sent to it go to a rented GPU container, not to any model API.

  modal run deploy/glimmer_modal.py::download     # once: weights into a Modal Volume
  modal deploy deploy/glimmer_modal.py             # serve (scales to zero when idle)
"""

import os
import subprocess

import modal

APP = "whiteout-glimmer"
REPO = "meta-models/Muse-Glimmer-30B-GGUF"
MODEL = "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
MMPROJ = "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf"
ADAPTER = os.environ.get("WHITEOUT_GLIMMER_MODEL", MODEL)  # a fine-tuned GGUF can replace the stock one

vol = modal.Volume.from_name("whiteout-glimmer-weights", create_if_missing=True)
app = modal.App(APP)

dl_image = modal.Image.debian_slim().pip_install("huggingface_hub[hf_transfer]==0.36.0").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
server_image = modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12").entrypoint([])


@app.function(image=dl_image, volumes={"/w": vol}, timeout=3600)
def download():
    from huggingface_hub import hf_hub_download

    for f in (MODEL, MMPROJ):
        hf_hub_download(REPO, f, local_dir="/w")
    vol.commit()
    print(sorted(os.listdir("/w")))


@app.function(image=server_image, gpu="L4", volumes={"/w": vol}, timeout=3600, scaledown_window=300,
              secrets=[modal.Secret.from_name("whiteout-offline-key")], min_containers=0, max_containers=1)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    cmd = [
        "/app/llama-server", "-m", f"/w/{ADAPTER}", "--mmproj", f"/w/{MMPROJ}",
        "--host", "0.0.0.0", "--port", "8000", "-ngl", "999", "-c", "32768", "-np", "2",
        "--jinja", "--api-key", os.environ["OFFLINE_API_KEY"], "--alias", "muse-glimmer",
        "--temp", "1.0", "--top-p", "0.95", "--top-k", "64", "--image-max-tokens", "4096",
    ]
    subprocess.Popen(cmd)
