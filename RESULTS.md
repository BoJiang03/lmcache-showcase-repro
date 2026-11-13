# Results (2026-11-13, 8x H200, one run per arm)

Agentic multi-turn replay (Weka trace, contexts 44K-256K), AIPerf agentic warmup, then a
960 s measurement window. vLLM default routing; speculative decoding on in every arm. Every
arm in a case shares the same vLLM flags (PRT-aligned, see README) except the KV connector.
Raw `result.json` / `meta.json` per run are in `results/`.

"Recomputed" is the share of prompt tokens prefilled from scratch; the rest came from the GPU
prefix cache or from the external KV tier (LMCache or vLLM native offload).

## Case 3: MiMo-V2.6-Flash-RL TP2, LMCache tuned vs LMCache baseline (L2 = GPU memory)

Both arms: LMCache L1 64 GB + fs_native L2 capped at 282 GB (= 2 x 141 GB of H200 memory),
conc 88, DFlash 7 tokens. Baseline = eager store, `num_read_workers=0`. Tuned = lazy offload
(stores a block only when the GPU is about to drop it, skips hashless sliding-window blocks)
+ `num_read_workers=256`.

| metric | baseline | tuned | tuned / baseline |
|---|---|---|---|
| requests completed | 374 | 1185 | 3.17x |
| TTFT p50 (s) | 38.5 | 2.7 | 14.3x lower |
| TTFT p90 (s) | 63.4 | 10.9 | 5.8x lower |
| TTFT p95 (s) | 79.7 | 15.7 | 5.1x lower |
| decode p50 (tok/s/user) | 11.0 | 38.9 | 3.5x |
| interactivity mean | 15.7 | 60.5 | 3.9x |
| total tok/s | 30,812 | 101,612 | 3.3x |
| output tok/s | 193 | 927 | 4.8x |
| prompt tokens recomputed | 46.1% | 6.7% | |
| prompt tokens from LMCache | 45.9% | 74.6% | |
| spec accept length | 2.81 | 3.40 | |

Why: per token the baseline stores ~143 KB in L2 (a 128-token sliding-window tail of 39
layers for every 256-token chunk, plus the drafter layers), the tuned build ~31 KB (window
tails only where vLLM can hit). Both fill the 282 GB cap and evict, but the tuned build's
working set fits: an LRU replay of the c48 pilot trace gives 99.8% vs 90.7% reuse coverage at
250 GB (`analysis/lru_curve.py`). The baseline recomputes 46% of prompt tokens, which inflates
TTFT and starves decode.

## Case 1: GLM-5.2-FP8 DEP8, LMCache shared L1 vs vLLM native offload

LMCache L1 1.2 TB shared by the 8 DP ranks vs vLLM native CPU offload 150 GiB per rank (not
shared). conc 96, MTP 1.

| metric | native | LMCache | LMCache / native |
|---|---|---|---|
| requests completed | 178 | 621 | 3.5x |
| TTFT p50 (s) | 102.1 | 18.1 | 5.6x lower |
| TTFT p90 (s) | 290.6 | 131.0 | 2.2x lower |
| TTFT mean (s) | 127.4 | 45.0 | 2.8x lower |
| decode p50 (tok/s/user) | 5.8 | 28.2 | 4.9x |
| interactivity mean | 9.0 | 28.5 | 3.2x |
| total tok/s | 14,128 | 46,367 | 3.3x |
| prompt tokens recomputed | 61.5% | 6.1% | |

The earlier (pre-PRT-alignment) pair gave the same picture: native 189 req / TTFT p50 55.5 s,
LMCache 620 req / 15.8 s. Native TTFT p50 moves a lot between runs (its queueing is in the
hundreds of seconds); p90, throughput and hit share are stable.

## Case 2: MiniMax-M3.1-Flash TP8, LMCache (GQA dedup) vs vLLM native offload

LMCache L1 1.5 TB with GQA KV-head replica dedup vs vLLM native CPU offload 1.5 TB. conc 160,
DSpark.

| metric | native | LMCache | LMCache / native |
|---|---|---|---|
| requests completed | 636 | 1535 | 2.4x |
| TTFT p50 (s) | 81.5 | 29.6 | 2.8x lower |
| TTFT p90 (s) | 125.5 | 45.9 | 2.7x lower |
| TTFT mean (s) | 80.1 | 28.1 | 2.9x lower |
| decode p50 (tok/s/user) | 17.1 | 39.0 | 2.3x |
| interactivity mean | 22.1 | 42.2 | 1.9x |
| total tok/s | 52,884 | 127,518 | 2.4x |
| output tok/s | 474 | 1,408 | 3.0x |
| prompt tokens recomputed | 26.5% | 6.8% | |

PRT alignment helped both arms. LMCache: `--max-workers 8` (it was 1, serializing every
rank's loads on one thread) halved TTFT p50 (60.7 -> 29.6 s) and raised throughput 33%.
Native: the MiniMax recipe (`--disable-hybrid-kv-cache-manager`, no EP, ...) lifted its
external hit share from 23% to 61% and its completed requests from 277 to 636, so the
advantage is now 2.4-2.9x instead of the earlier 3-4x.

## KV correctness (case 3 tuned build, TP2)

`scripts/kvcheck_case3.sh lazy`: all 8 probes (24,576 tokens) hit LMCache for the whole prompt
with the GPU prefix cache cold; first-token logprob differs from the first pass by at most
0.16 (corrupt KV measured 1.8-4.5 nats). `results/kvcheck_tp2_lazy.json`.
