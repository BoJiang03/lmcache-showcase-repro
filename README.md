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

## Getting the code

```
git clone --recursive https://github.com/BoJiang03/lmcache-showcase-repro.git
```

`third_party/LMCache` and `third_party/vllm` are submodules pinned to the exact commits the
runs used. The vLLM checkout is large; `git submodule update --init --filter=blob:none`
keeps it smaller, or skip it and apply `patches/vllm/` to the nightly wheel (below).

## Layout

```
showcase/run_one.py    driver: starts LMCache server + vLLM + AIPerf, samples metrics, scores the window
showcase/kv_check.py   KV correctness smoke (hit from LMCache, values vs recompute noise)
runner/                sampler (Prometheus) and scorer (TTFT, decode, token-source shares)
aiperf_plugins/        AIPerf plugins used by the replay (weka_framing, live_history)
workloads/agentic/     fetch_trace.py for the Weka agentic trace
scripts/               case1.sh case2.sh case3.sh run_all.sh kvcheck_case3.sh check_mimo_tp2.sh
analysis/              live_ws.py (live working set), lru_curve.py (L2 size -> reuse coverage)
third_party/LMCache    BoJiang03/LMCache; tags showcase-20261113-case{1,2,3} = the three builds
third_party/vllm       BoJiang03/vllm, branch mimo-tp2-split-qkv-inductor-fix (MiMo TP2 fix)
patches/vllm/          the same vLLM fix as a patch for the nightly wheel (required for case 3)
results/               raw data of every run reported in RESULTS.md (see below)
```

## Software

| component | version |
|---|---|
| vLLM | nightly `0.34.0rc1.dev156+g4e3463c87` (cu130) + `patches/vllm/0001-*.patch` (= `third_party/vllm`, one commit on top of `4e3463c87`) |
| LMCache, case 1 | tag `showcase-20261113-case1` = `LMCache/LMCache` dev `29340c7c` |
| LMCache, case 2 | tag `showcase-20261113-case2` = dev `b8f34d05` + GQA KV-head replica dedup (`0da765cb`) |
| LMCache, case 3 | tag `showcase-20261113-case3` = lazy offload + GQA dedup + sliding-window tail fix (`5b8d231a`) |
| AIPerf | 0.13.0 with `aiperf_plugins/` installed |

`scripts/build_lmcache.sh` builds the three tags from the submodule with gcc-toolset-12 (host
gcc 10 cannot build the torch C++20 extensions): case 1 into the vLLM venv, cases 2 and 3
into their own `--target` prefixes, which the scripts pass with `--lmcache-path` (`LMCP=`).

The case-3 build is the code the runs used. A later review found two bugs in its sliding-window
tail pinning (a GPU block pin could leak; an op could be dropped once vLLM hashed a partial tail
block), both of which can only hurt the tuned arm; the fix is `768dee69` on `BoJiang03/LMCache`
branch `lazy-final`.

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
$PRT_W/lmc-dedup, $PRT_W/lmc-lazy-fix   LMCache builds for case 2 and case 3 (scripts/build_lmcache.sh)
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

## Raw data

`results/<run>/` holds everything a run wrote except AIPerf's dataset mmap cache:

- `result.json`: the scored 960 s window (the numbers in RESULTS.md); `meta.json`: exact
  commands, environment and GPUs.
- `aiperf/profile_export.jsonl.gz`: one record per request (timestamps, ISL/OSL, TTFT, ITL);
  `aiperf/profile_export_aiperf.{json,csv}` and `profile_export_console.txt`: AIPerf's summary;
  `aiperf/server_metrics_export.*`: Prometheus scrapes taken by AIPerf.
- `samples.jsonl.gz`: vLLM metrics sampled by `runner/sampler.py` (KV usage, running/waiting,
  prefix and external hits, spec decoding counters).
- `vllm.log.gz`, `lmcache.log.gz`, `aiperf.log.gz`: process logs.
- case 3 only: `l2_du.txt`, the L2 directory size every 30 s (`<unix time> <bytes>`).

Files over 200 KB are gzipped; `gunzip -k` them before pointing `runner/score.py` or
`analysis/*.py` at a run directory.

