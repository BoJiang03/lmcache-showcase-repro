#!/bin/bash
# Case 2: MiniMax-M3.1-Flash TP8 + DSpark, LMCache L1 1.5 TB (GQA replica dedup) vs vLLM native CPU offload 1.5 TB.
# Usage: case2.sh <lmcache|native> [tag]
source "$(dirname "$0")/common.sh"
MODE=$1; TAG=${2:-prt}
cd "$REPO/showcase"
if [ "$MODE" = lmcache ]; then OUT=$RESULTS/${TAG}_mm_lmc_dedup_l11500_n160; X=(--lmcache-path "${LMCP:-$W/lmc-dedup}" --l1-gb 1500)
else OUT=$RESULTS/${TAG}_mm_native_cpu1500_n160; X=(--native-cpu-bytes 1610612736000); fi
$PY run_one.py --mode "$MODE" --model minimax --tp 8 "${X[@]}" \
  --concurrency 160 --duration 960 --gpu-util 0.9 --mbt 8192 --max-num-seqs 64 --spec-method dspark \
  --vllm-arg=--moe-backend=marlin \
  --out "$OUT" > "$W/logs/$(basename "$OUT").log" 2>&1
echo "case2 $MODE rc=$? -> $OUT"
