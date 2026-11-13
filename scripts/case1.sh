#!/bin/bash
# Case 1: GLM-5.2-FP8 DEP8 (8x H200), LMCache shared L1 1.2 TB vs vLLM native CPU offload 150 GiB per rank.
# Usage: case1.sh <lmcache|native> [tag]
source "$(dirname "$0")/common.sh"
MODE=$1; TAG=${2:-prt}
cd "$REPO/showcase"
if [ "$MODE" = lmcache ]; then OUT=$RESULTS/${TAG}_glm_lmc_l1200_n96; X=(--l1-gb 1200)
else OUT=$RESULTS/${TAG}_glm_native_150g_n96; X=(); fi   # native default: 150 GiB per DP rank
$PY run_one.py --mode "$MODE" "${X[@]}" --concurrency 96 --duration 960 \
  --mbt 2048 --max-num-seqs 16 --gpu-util 0.94 --vllm-arg=--max-cudagraph-capture-size=32 \
  --out "$OUT" > "$W/logs/$(basename "$OUT").log" 2>&1
echo "case1 $MODE rc=$? -> $OUT"
