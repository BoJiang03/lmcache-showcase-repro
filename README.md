# LMCache agentic-workload showcases: reproduction kit

Three showcases on one 8x H200 node, each an A/B of KV-cache tiers under an agentic
multi-turn replay. Numbers and analysis: [RESULTS.md](RESULTS.md).

| case | model / parallelism | A | B |
|---|---|---|---|
| 1 | GLM-5.2-FP8, DP8 + EP | vLLM native CPU offload, 150 GiB per rank | LMCache L1 1.2 TB shared by all ranks |
| 2 | MiniMax-M3.1-Flash, TP8 | vLLM native CPU offload, 1.5 TB | LMCache L1 1.5 TB, GQA replica dedup |
| 3 | MiMo-V2.6-Flash-RL, TP2 | LMCache baseline (eager store) | LMCache tuned (lazy offload, parallel L2 reads) |

Case 3 gives both arms a small L1 (64 GB) and an fs_native L2 on NVMe capped at the GPU memory
size (282 GB), so the L2 footprint per token decides who keeps the multi-turn working set.

## Layout

```
showcase/run_one.py    driver: starts LMCache server + vLLM + AIPerf, samples metrics, scores the window
showcase/kv_check.py   KV correctness smoke (hit from LMCache, values vs recompute noise)
runner/                sampler (Prometheus) and scorer (TTFT, decode, token-source shares)
aiperf_plugins/        AIPerf plugins used by the replay (weka_framing, live_history)
workloads/agentic/     fetch_trace.py for the Weka agentic trace
scripts/               case1.sh case2.sh case3.sh run_all.sh kvcheck_case3.sh check_mimo_tp2.sh
analysis/              live_ws.py (live working set), lru_curve.py (L2 size -> reuse coverage)
patches/vllm/          MiMo TP2 fix for vLLM (required for case 3)
patches/lmcache/       the two LMCache builds (lazy offload + SWA fix; GQA dedup)
results/               result.json / meta.json of every run reported in RESULTS.md
```

## Software

| component | version |
|---|---|
| vLLM | nightly `0.34.0rc1.dev156+g4e3463c87` (cu130) + `patches/vllm/0001-*.patch` |
| LMCache, case 1 | `LMCache/LMCache` dev `29340c7c` |
| LMCache, case 2 | dev `b8f34d05` + `patches/lmcache/gqa-tp-replica-dedup/*.patch` (branch `gqa-tp-replica-dedup`) |
| LMCache, case 3 | dev `6d8376b1` + `patches/lmcache/lazy-final-gqadedup/lazy-final-gqadedup.diff` (branch `lazy-final-gqadedup`; commit list in `COMMITS.txt`) |
| AIPerf | 0.13.0 with `aiperf_plugins/` installed |

Build LMCache from source with gcc-toolset-12 (host gcc 10 cannot build the torch C++20
extensions). Each build gets its own install prefix and is passed with `--lmcache-path`
(`LMCP=` in the scripts); case 1 uses the LMCache installed in the vLLM venv.

The vLLM patch matters only for MiMo at TP2: without it, torch.compile produces garbage
output there (speculative acceptance length 1.00). `scripts/check_mimo_tp2.sh` checks it:
the greedy text must be sensible and the DFlash acceptance length ~4.9.

## Workspace

Everything heavy lives under `$PRT_W` (default in `scripts/common.sh`):

```
$PRT_W/venv          vLLM + LMCache (case 1) venv        $PRT_W/venv-aiperf   AIPerf venv
$PRT_W/models/       GLM-5.2-FP8, MiniMax-M3.1-Flash, MiMo-V2.6-Flash-RL (with dflash/)
$PRT_W/hf-alias/     HF cache aliases for the pinned revisions
$PRT_W/trace/weka-062126-256k-dir   agentic trace (workloads/agentic/fetch_trace.py)
$PRT_W/lmc-dedup, $PRT_W/lmc-lazy-fix   LMCache builds for case 2 and case 3
$PRT_W/results, $PRT_W/logs
```

Case 3's L2 goes to `$L2_ROOT` (an NVMe; default `/mnt/nvme1/$USER`). Case 1 native needs a
large `/dev/shm` (`mount -o remount,size=1800G /dev/shm`); case 2 needs ~1.6 TB of host RAM.

## Running

```
scripts/run_all.sh                 # all ten arms in sequence, ~9 h
scripts/case3.sh eager formal_c88  # one arm
scripts/case3.sh lazy  formal_c88
scripts/case1.sh lmcache ; scripts/case1.sh native          # CONC=75 for the lower-load point
scripts/case2.sh lmcache ; scripts/case2.sh native          # CONC=130 for the lower-load point
```

Run arms one at a time: the two case-3 arms must see the same NVMe, and cases 1 and 2 use
all eight GPUs. Each run writes `result.json` (scored over the 960 s window) under `$RESULTS`.

## Deployment settings (aligned with LMCache PRT, PR #5744)

- LMCache server: `--max-workers` = number of ranks (the default 1 serializes every rank's
  stores and loads on one thread), `--l1-read-ttl-seconds 3600`, chunk 256.
- vLLM: `VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_WORKER_MULTIPROC_METHOD=fork`,
  `--disable-cascade-attn`, `compilation-config` `fi_allreduce_fusion_max_size_mb`
  (TP8: 1, TP4: 2). MiniMax follows its recipe: no EP, `--disable-hybrid-kv-cache-manager`,
  fp8 indexer, `FULL_AND_PIECEWISE`, Rust frontend.
- AIPerf: agentic replay, `--cache-bust system_suffix`, default staggered warmup (no burst).
- All of this is in `showcase/run_one.py`; the case scripts only pick the model, connector and sizes.

## Sizing case 3

`analysis/lru_curve.py <run dir> <KB per token>` replays a run's requests through a per-session
LRU and prints reuse coverage against L2 size; `analysis/live_ws.py` prints the live vs
cumulative working set. The c48 pilot put the tuned build at ~31 KB/token and the baseline at
~143 KB/token, which led to conc 88 (GPU KV peak ~100%) and L2 = 282 GB.
