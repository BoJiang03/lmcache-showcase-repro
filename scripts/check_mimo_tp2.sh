#!/bin/bash
# Sanity check for the vLLM MiMo TP2 fix: plain vLLM TP2 + DFlash, greedy text and spec acceptance.
# Without patches/vllm the text is garbage and the acceptance length is 1.00; with it, ~4.9.
source "$(dirname "$0")/common.sh"
M=$W/models/MiMo-V2.6-Flash-RL; PORT=${PORT:-19417}; OUT=$RESULTS/check_mimo_tp2; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=${GPUS:-0,1} HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$W/vllm-cache \
  $W/venv/bin/vllm serve "$M" --served-model-name mimo --port $PORT --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 --max-model-len 65536 --trust-remote-code --language-model-only \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$M/dflash\",\"num_speculative_tokens\":7}" > "$OUT/vllm.log" 2>&1 &
VP=$!
until curl -s localhost:$PORT/v1/models >/dev/null; do sleep 10; kill -0 $VP || exit 1; done
for p in "The capital of France is" "Q: What is 17*23? Show the steps.\nA:"; do
  curl -s localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"mimo\",\"prompt\":\"$p\",\"max_tokens\":48,\"temperature\":0}" | $PY -c "import sys,json;print(repr(json.load(sys.stdin)['choices'][0]['text']))"
done
for i in 1 2 3 4; do curl -s localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"mimo\",\"prompt\":$($PY -c "import json;print(json.dumps(open('$W/venv/lib/python3.12/site-packages/vllm/config/speculative.py').read()[:20000]))"),\"max_tokens\":256,\"temperature\":0}" >/dev/null; done
sleep 12; grep "SpecDecoding metrics" "$OUT/vllm.log" | tail -1 | grep -oE "Mean acceptance length: [0-9.]+"
kill -INT $VP; wait $VP
