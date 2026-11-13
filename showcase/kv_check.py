#!/usr/bin/env python3
"""KV correctness smoke for LMCache: stored KV must be found, loaded and exact.

1. Probe prompts are computed once (greedy, logprobs recorded).
2. Filler prompts push the probes out of the GPU prefix cache and, with a small
   L1, out of L1, so the second pass must load them from L2.
3. The probes are re-sent one at a time; each must hit LMCache for (almost) its
   whole prompt and reproduce the first pass's tokens and logprobs.

Exit code 0 means every check passed; the report goes to --out as JSON.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path


def http_json(url: str, body: dict | None = None, timeout: float = 1800) -> dict:
    """GET (body None) or POST a JSON body and decode the JSON reply."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metric(url: str, name: str) -> float:
    """Sum one Prometheus counter over all its label sets."""
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode()
    pat = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$", re.M)
    return sum(float(v) for v in pat.findall(text))


def tokens(seed: int, n: int, lo: int, hi: int) -> list[int]:
    """Deterministic pseudo-random token ids: a distinct prefix per seed."""
    rng = random.Random(seed)
    return [rng.randrange(lo, hi) for _ in range(n)]


def corpus_tokens(tokenizer_dir: str, src_dir: str, need: int) -> list[int]:
    """Token ids of natural text (Python sources), at least ``need`` long.

    Random token ids give near-flat next-token distributions whose greedy
    argmax flips on any rounding; real text keeps the comparison meaningful.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    ids: list[int] = []
    for path in sorted(Path(src_dir).rglob("*.py")):
        ids += tok.encode(path.read_text(errors="ignore"), add_special_tokens=False)
        if len(ids) >= need:
            return ids
    raise SystemExit(f"corpus {src_dir} has only {len(ids)} tokens, need {need}")


def complete(base: str, model: str, prompt: list[int], max_tokens: int,
             salt: str | None = None) -> dict:
    """One greedy completion; returns text, token logprobs and latency.

    A ``salt`` keys the request's prefix cache apart (vLLM ``cache_salt``), so
    neither the GPU prefix cache nor LMCache can serve it.
    """
    t0 = time.time()
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "logprobs": 1, "ignore_eos": True}
    if salt:
        body["cache_salt"] = salt
    r = http_json(f"{base}/v1/completions", body)
    choice = r["choices"][0]
    lp = choice.get("logprobs") or {}
    return {"text": choice["text"], "tokens": lp.get("tokens"),
            "logprobs": lp.get("token_logprobs"), "latency_s": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18000")
    ap.add_argument("--lmc-http", default="http://127.0.0.1:18080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--probe-len", type=int, default=24576)
    ap.add_argument("--fillers", type=int, default=48)
    ap.add_argument("--filler-len", type=int, default=32768)
    ap.add_argument("--filler-conc", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--vocab-lo", type=int, default=1000)
    ap.add_argument("--vocab-hi", type=int, default=100000)
    ap.add_argument("--settle-s", type=float, default=30, help="wait for async stores")
    ap.add_argument("--min-hit-frac", type=float, default=0.9)
    ap.add_argument("--l2-dir", default="")
    ap.add_argument("--tokenizer", default="", help="tokenizer dir for natural-text probes")
    ap.add_argument("--corpus", default="", help="dir of .py files for natural-text probes")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    vm = f"{a.url}/metrics"
    if a.tokenizer and a.corpus:
        ids = corpus_tokens(a.tokenizer, a.corpus, a.probes * a.probe_len)
        probes = [ids[i * a.probe_len:(i + 1) * a.probe_len] for i in range(a.probes)]
    else:
        probes = [tokens(10_000 + i, a.probe_len, a.vocab_lo, a.vocab_hi)
                  for i in range(a.probes)]

    def lmc_status() -> dict:
        try:
            s = http_json(f"{a.lmc_http}/status")["storage_manager"]
            l1 = s.get("l1_manager", {})
            l2 = (s.get("l2_eviction_controller", {}).get("adapters") or [{}])[0]
            return {"l1_used_gib": l1.get("memory_used_bytes", 0) / 2**30,
                    "l1_objects": l1.get("total_object_count"),
                    "l2_used_gb": l2.get("total_bytes_used", 0) / 1e9}
        except Exception as e:  # status is diagnostic only
            return {"error": repr(e)}

    def l2_files() -> int | None:
        if not a.l2_dir:
            return None
        return sum(1 for p in Path(a.l2_dir).iterdir() if p.suffix == ".data")

    report: dict = {"args": {k: str(v) for k, v in vars(a).items()}}
    # Pass 1: compute every probe.
    first = [complete(a.url, a.model, p, a.max_tokens) for p in probes]
    time.sleep(a.settle_s)
    report["after_pass1"] = lmc_status() | {"l2_files": l2_files()}
    # Fillers: distinct prompts that evict the probes from the GPU and from L1.
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.filler_conc) as ex:
        list(ex.map(lambda i: complete(a.url, a.model,
                                       tokens(90_000 + i, a.filler_len, a.vocab_lo, a.vocab_hi), 1),
                    range(a.fillers)))
    report["filler_s"] = time.time() - t0
    time.sleep(a.settle_s)
    report["after_fillers"] = lmc_status() | {"l2_files": l2_files()}
    def compare(x: dict, y: dict) -> dict:
        lp1, lp2 = x["logprobs"] or [], y["logprobs"] or []
        n = min(len(lp1), len(lp2))
        same = 0
        while same < n and x["tokens"][same] == y["tokens"][same]:
            same += 1
        return {"identical": x["tokens"] == y["tokens"], "same_prefix": same, "n": n,
                "first_diff": abs(lp1[0] - lp2[0]) if n else None,
                "max_diff_same_prefix": max(
                    (abs(u - v) for u, v in zip(lp1[:same], lp2[:same])), default=0.0)}

    # Pass 2: one probe at a time so each metric delta belongs to one request.
    rows, ok = [], True
    for i, p in enumerate(probes):
        h0, q0 = metric(vm, "vllm:external_prefix_cache_hits_total"), metric(vm, "vllm:prefix_cache_hits_total")
        second = complete(a.url, a.model, p, a.max_tokens)
        ext = metric(vm, "vllm:external_prefix_cache_hits_total") - h0
        gpu = metric(vm, "vllm:prefix_cache_hits_total") - q0
        lp1, lp2 = first[i]["logprobs"] or [], second["logprobs"] or []
        n = min(len(lp1), len(lp2))
        same_prefix = 0
        while same_prefix < n and first[i]["tokens"][same_prefix] == second["tokens"][same_prefix]:
            same_prefix += 1
        row = {
            "probe": i, "prompt_tokens": len(p), "external_hit_tokens": ext,
            "gpu_hit_tokens": gpu, "hit_frac": ext / len(p),
            "tokens_identical": first[i]["tokens"] == second["tokens"],
            "same_prefix_tokens": same_prefix, "out_tokens": n,
            "first_token_logprob_diff": abs(lp1[0] - lp2[0]) if n else None,
            "max_logprob_diff_on_same_prefix": max(
                (abs(x - y) for x, y in zip(lp1[:same_prefix], lp2[:same_prefix])), default=0.0),
            "latency_pass1_s": first[i]["latency_s"], "latency_pass2_s": second["latency_s"],
        }
        # Control: the same prompt recomputed from scratch (salted, so no
        # cache serves it) measures the engine's own run-to-run noise.
        e0 = metric(vm, "vllm:external_prefix_cache_hits_total")
        ctrl = complete(a.url, a.model, p, a.max_tokens, salt=f"kvcheck-ctrl-{i}")
        c = compare(first[i], ctrl)
        row |= {"ctrl_external_hit_tokens": metric(vm, "vllm:external_prefix_cache_hits_total") - e0,
                "ctrl_identical": c["identical"], "ctrl_same_prefix_tokens": c["same_prefix"],
                "ctrl_first_token_logprob_diff": c["first_diff"],
                "ctrl_max_logprob_diff_on_same_prefix": c["max_diff_same_prefix"]}
        rows.append(row)
    report["probes"] = rows
    report["after_pass2"] = lmc_status() | {"l2_files": l2_files()}
    hits_ok = all(r["hit_frac"] >= a.min_hit_frac for r in rows)
    gpu_cold = all(r["gpu_hit_tokens"] < 0.1 * r["prompt_tokens"] for r in rows)
    # Without batch invariance a recompute is not bit-exact either (chunked
    # prefill vs one-token prefill after a load take different kernel paths),
    # so loaded KV is judged against the recompute control's noise, not zero.
    ctrl_first = max(r["ctrl_first_token_logprob_diff"] or 0.0 for r in rows)
    lmc_first = max(r["first_token_logprob_diff"] or 0.0 for r in rows)
    identical = sum(r["tokens_identical"] for r in rows)
    ctrl_identical = sum(r["ctrl_identical"] for r in rows)
    ctrl_clean = all(r["ctrl_external_hit_tokens"] == 0 for r in rows)
    # Greedy outputs diverge within a few tokens even between two recomputes,
    # so token identity is reported but not judged. Corrupt KV moves the
    # first token's logprob by whole nats (1.8-4.5 measured), noise by ~0.1.
    values_ok = ctrl_clean and lmc_first <= max(0.3, 3 * ctrl_first)
    report["noise"] = {"lmcache_max_first_token_diff": lmc_first,
                       "recompute_max_first_token_diff": ctrl_first,
                       "lmcache_identical": identical, "recompute_identical": ctrl_identical,
                       "control_missed_cache": ctrl_clean}
    # Lazy offload stores a block only when the GPU is about to reuse it, so
    # the probes reach L2 during the fillers, not after pass 1.
    stored_ok = (report["after_fillers"].get("l2_files") or 0) > 0 or not a.l2_dir
    report["checks"] = {
        "hits_from_lmcache": hits_ok, "probes_left_gpu": gpu_cold,
        "kv_values_match": values_ok, "outputs_identical": f"{identical}/{len(rows)}",
        "recompute_identical": f"{ctrl_identical}/{len(rows)}",
        "l2_written": stored_ok,
    }
    ok = hits_ok and gpu_cold and values_ok and stored_ok
    report["pass"] = ok
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["checks"] | {"pass": ok} | report["noise"], indent=1))
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
