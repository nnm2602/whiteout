#!/bin/bash
# On the VM: serve a Glimmer GGUF with the stock vision projector and the DFlash draft model on 127.0.0.1:8000
# (reach it through an SSH tunnel). Same server flags as deploy/glimmer_vm_startup.sh.
#   MODEL=Muse-Glimmer-30B-whiteout-ft-Q4_K_M.gguf bash serve.sh
#   MODEL=Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf LORA=ft-lora-f16.gguf bash serve.sh
MODEL=${MODEL:-Muse-Glimmer-30B-whiteout-ft-Q4_K_M.gguf}
EXTRA=()
[ -n "${LORA:-}" ] && EXTRA+=(--lora "/w/$LORA")
[ "${DRAFT:-1}" = 1 ] && EXTRA+=(-md /w/dflash-Muse-Glimmer-30B-Q4_K_M.gguf --spec-type draft-dflash)
sudo docker rm -f glimmer 2>/dev/null
sudo docker run -d --name glimmer --gpus all -p 127.0.0.1:8000:8000 -v /data/gguf:/w \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m "/w/$MODEL" --mmproj /w/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8000 -ngl 999 -c 32768 -np 2 --jinja --alias muse-glimmer \
  --temp 1.0 --top-p 0.95 --top-k 64 --image-max-tokens 4096 "${EXTRA[@]}"
until curl -sf http://127.0.0.1:8000/health; do sleep 5; sudo docker ps -q -f name=glimmer | grep -q . || { sudo docker logs --tail 40 glimmer; exit 1; }; done
echo; echo SERVING "$MODEL" "${EXTRA[@]}"
