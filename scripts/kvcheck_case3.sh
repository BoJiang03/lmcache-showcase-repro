#!/bin/bash
# KV correctness smoke for case 3 (stored KV must hit and match a recompute).
# Usage: kvcheck_case3.sh <eager|lazy>   Env: GPUS (0,1), PORTOFF (0), LMCP, L2D
source "$(dirname "$0")/common.sh"
MODE=$1; OUT=$RESULTS/kvcheck_tp2_$MODE; L2D=${L2D:-$L2_ROOT/l2_kvcheck}
mkdir -p "$L2D"; find "$L2D" -mindepth 1 -delete
if [ "$MODE" = lazy ]; then NR=256; EXTRA='{"lmcache.mp.lazy_offload":true}'; else NR=0; EXTRA='{}'; fi
L2="{\"type\":\"fs_native\",\"base_path\":\"$L2D\",\"num_workers\":32,\"use_odirect\":true,\"num_read_workers\":$NR,\"max_capacity_gb\":1500,\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.9,\"eviction_ratio\":0.1}}"
P=$((${PORTOFF:-0}))
CLIENT="$PY $REPO/showcase/kv_check.py --model mimo --tokenizer $W/models/MiMo-V2.6-Flash-RL --corpus $W/venv/lib/python3.12/site-packages/vllm/v1 --l2-dir $L2D --out $OUT/kv_check.json --url http://127.0.0.1:$((18000+P)) --lmc-http http://127.0.0.1:$((18080+P))"
cd "$REPO/showcase"
$PY run_one.py --mode lmcache --tp 2 --model mimo26 --gpus "${GPUS:-0,1}" --lmcache-path "${LMCP:-$W/lmc-lazy-fix}" \
  --kv-dtype auto --spec-method dflash --spec-model dflash --mtp 7 --concurrency 1 --duration 60 \
  --l1-gb 64 --gpu-util 0.90 --mbt 8192 --max-num-seqs 64 --lmc-extra "$EXTRA" \
  "--lmcache-arg=--l2-adapter=$L2" --lmcache-arg=--eviction-trigger-watermark=0.8 --lmcache-arg=--eviction-ratio=0.2 \
  --vllm-arg=--kv-cache-memory-bytes=8589934592 --port-offset $P --client "$CLIENT" --out "$OUT" > "$W/logs/kvcheck_tp2_$MODE.log" 2>&1
echo "rc=$?"; tail -20 "$OUT/client.log"
