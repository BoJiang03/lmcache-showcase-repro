#!/bin/bash
# Case 3: MiMo-V2.6-Flash-RL TP2 + DFlash, LMCache small L1 + fs_native L2.
# Baseline (eager store, num_read_workers=0) vs tuned (lazy offload, num_read_workers=256).
# Usage: case3.sh <eager|lazy> <tag>
# Env: CONC (88), L2CAP (GB, 282 = GPU memory of 2x H200), L1 (GB, 64), DUR (960), GPUS (0,1),
#      PORTOFF (0), LMCP (LMCache build to use), L2D (L2 dir; wiped first).
source "$(dirname "$0")/common.sh"
MODE=$1; TAG=$2
OUT=$RESULTS/c3_tp2_${MODE}_${TAG}; L2D=${L2D:-$L2_ROOT/l2_c3}
mkdir -p "$L2D"; find "$L2D" -mindepth 1 -delete
if [ "$MODE" = lazy ]; then NR=256; EXTRA='{"lmcache.mp.lazy_offload":true}'; else NR=0; EXTRA='{}'; fi
L2="{\"type\":\"fs_native\",\"base_path\":\"$L2D\",\"num_workers\":32,\"use_odirect\":true,\"num_read_workers\":$NR,\"max_capacity_gb\":${L2CAP:-282},\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.9,\"eviction_ratio\":0.1}}"
mkdir -p "$OUT"
( while true; do echo "$(date +%s) $(du -sb "$L2D" 2>/dev/null | cut -f1)"; sleep 30; done ) > "$OUT/l2_du.txt" &
SAMP=$!
cd "$REPO/showcase"
$PY run_one.py --mode lmcache --tp 2 --model mimo26 --lmcache-path "${LMCP:-$W/lmc-lazy-fix}" \
  --kv-dtype auto --spec-method dflash --spec-model dflash --mtp 7 --concurrency "${CONC:-88}" --duration "${DUR:-960}" \
  --l1-gb "${L1:-64}" --gpu-util 0.90 --mbt 8192 --max-num-seqs 128 --lmc-extra "$EXTRA" \
  --port-offset "${PORTOFF:-0}" --gpus "${GPUS:-0,1}" \
  "--lmcache-arg=--l2-adapter=$L2" --lmcache-arg=--eviction-trigger-watermark=0.8 --lmcache-arg=--eviction-ratio=0.2 \
  --out "$OUT" > "$W/logs/c3_${MODE}_${TAG}.log" 2>&1
rc=$?; kill $SAMP; echo "case3 $MODE rc=$rc -> $OUT"
