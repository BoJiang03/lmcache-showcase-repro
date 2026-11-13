# SPDX-License-Identifier: Apache-2.0
"""Post-run check for the aiperf_prt_live_history plugin.

usage: check_live_history.py RUN_DIR [BASELINE_RUN_DIR] [--trace-dir DIR]

RUN_DIR is the AIPerf artifact dir of a run_aiperf_step_livehist.sh run; it
must contain profile_export.jsonl and prt_live_history/*.jsonl. All numbers are
over in-window requests: profile_export records with benchmark_phase=profiling,
not cancelled, joined to the plugin log on x_request_id.

Reports
  1. plugin active: share of requests whose history holds our output for the
     previous turn (newest_subbed), any of our outputs (subbed > 0), and tagged
     messages that fell back to trace text (no output on this worker: the source
     turn ran in warmup, on another worker, or was cut off).
  2. round trip: share of stored outputs whose decode->re-encode reproduces the
     generated ids exactly (first 2048 tokens checked), and mean matched prefix
     / checked tokens; a mismatch at token p caps that output's cache hit at p.
  3. ISL vs trace: isl_delta = our ISL - the ISL the trace text would give,
     overall and by turn-index bucket; for weka_main turns also server ISL minus
     the trace's recorded `in`.
  4. repetition: distinct 4-gram ratio of each output by turn index (bucketed),
     and per session the change from its first to its last profiled output.
  5. prefill: mean prompt, cached and computed (prompt - cached) tokens per
     request; with BASELINE_RUN_DIR, the same for the baseline and the delta.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def load_records(run_dir: str) -> dict[str, dict]:
    out = {}
    with open(os.path.join(run_dir, "profile_export.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            md = r["metadata"]
            if md.get("benchmark_phase") != "profiling" or md.get("was_cancelled"):
                continue
            if r.get("error"):
                continue
            out[md["x_request_id"]] = r
    return out


def metric(r: dict, name: str) -> float | None:
    m = r["metrics"].get(name)
    return None if m is None else m["value"]


def load_plugin_log(run_dir: str) -> dict[str, dict]:
    out = {}
    for p in glob.glob(os.path.join(run_dir, "prt_live_history", "*.jsonl")):
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                out[rec["x_request_id"]] = rec
    return out


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def dist(xs: list[float]) -> str:
    if not xs:
        return "n=0"
    return (
        f"n={len(xs)} mean={statistics.fmean(xs):.1f} p5={pct(xs, .05):.0f} "
        f"p50={pct(xs, .5):.0f} p95={pct(xs, .95):.0f} min={min(xs):.0f} max={max(xs):.0f}"
    )


def bucket(turn: int) -> str:
    for hi in (1, 2, 5, 10, 20, 50, 100):
        if turn < hi:
            return f"<{hi}"
    return ">=100"


BUCKETS = ["<1", "<2", "<5", "<10", "<20", "<50", "<100", ">=100"]


def prefill_summary(recs: dict[str, dict]) -> tuple[float, float, float]:
    p = [metric(r, "usage_prompt_tokens") or 0 for r in recs.values()]
    c = [metric(r, "usage_prompt_cache_read_tokens") or 0 for r in recs.values()]
    n = max(1, len(p))
    return sum(p) / n, sum(c) / n, (sum(p) - sum(c)) / n


def trace_in(trace_dir: str) -> dict[tuple[str, int], int]:
    out = {}
    for p in glob.glob(os.path.join(trace_dir, "*.json")):
        t = json.load(open(p))
        for i, r in enumerate(t["requests"]):
            if "in" in r:
                out[(t["id"], i)] = r["in"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("baseline_dir", nargs="?")
    ap.add_argument("--trace-dir", default="/raid/bo/prt-cache/work/weka-062126-256k-dir")
    a = ap.parse_args()

    recs = load_records(a.run_dir)
    log = load_plugin_log(a.run_dir)
    joined = [(recs[k], log[k]) for k in recs if k in log]
    print(f"in-window records {len(recs)}, with plugin log {len(joined)}")
    if not joined:
        print("NO PLUGIN RECORDS: plugin inactive (wrong venv, or PRT_LIVE_HISTORY_LOG_DIR unset)")
        return

    # 1. plugin active
    n = len(joined)
    first_turn = sum(lg["turn"] == 0 for _, lg in joined)
    newest_tagged = sum(lg.get("newest_tagged", False) for _, lg in joined)
    newest_subbed = sum(lg.get("newest_subbed", False) for _, lg in joined)
    any_sub = sum(lg.get("subbed", 0) > 0 for _, lg in joined)
    tagged = sum(lg.get("tagged", 0) for _, lg in joined)
    subbed = sum(lg.get("subbed", 0) for _, lg in joined)
    print("\n[1] plugin active")
    print(f"  requests whose previous-turn output is ours: {newest_subbed}/{n} = {newest_subbed / n:.3f}"
          f"  (of those with a tagged previous turn: {newest_subbed}/{newest_tagged})")
    print(f"  requests with any of our outputs in history: {any_sub}/{n} = {any_sub / n:.3f}")
    print(f"  tagged assistant messages substituted: {subbed}/{tagged}"
          f" = {subbed / max(1, tagged):.3f}  (rest fell back to trace text)")
    print(f"  turn-0 requests (no previous turn): {first_turn}")
    by_depth = defaultdict(lambda: [0, 0])
    for _, lg in joined:
        if lg.get("newest_tagged"):
            by_depth[lg["depth"]][0] += 1
            by_depth[lg["depth"]][1] += lg.get("newest_subbed", False)
    for d in sorted(by_depth):
        t, s = by_depth[d]
        print(f"  agent_depth={d}: previous-turn substituted {s}/{t}")

    # 2. round trip (all stored outputs, incl. those finishing after the window)
    stored = [lg for lg in log.values() if lg.get("stored")]
    exact = sum(bool(lg["rt_exact"]) for lg in stored)
    frac = [min(1.0, lg["rt_lcp"] / lg["rt_checked"]) for lg in stored if lg.get("rt_checked")]
    print("\n[2] token round trip of stored outputs")
    print(f"  exact: {exact}/{len(stored)} = {exact / max(1, len(stored)):.3f};"
          f" matched prefix / checked mean={statistics.fmean(frac) if frac else float('nan'):.3f}")
    print(f"  matched tokens per output: {dist([lg['rt_lcp'] for lg in stored])}")
    print(f"  n_out per output: {dist([lg['n_out'] for lg in stored])}")

    # 3. ISL vs trace
    print("\n[3] ISL vs trace")
    print(f"  isl_delta (ours - trace text), all: {dist([lg['isl_delta'] for _, lg in joined])}")
    print(f"  isl_delta, requests with substitution: "
          f"{dist([lg['isl_delta'] for _, lg in joined if lg.get('subbed')])}")
    per_b = defaultdict(list)
    for _, lg in joined:
        per_b[bucket(lg["turn"])].append(lg["isl_delta"])
    for b in BUCKETS:
        if per_b[b]:
            print(f"    turn {b:>5}: {dist(per_b[b])}")
    if os.path.isdir(a.trace_dir):
        tin = trace_in(a.trace_dir)
        diffs = []
        for r, _ in joined:
            md = r["metadata"]
            if md.get("source_kind") != "weka_main":
                continue
            want = tin.get((md.get("source_trace_id"), md.get("source_outer_idx")))
            got = metric(r, "usage_prompt_tokens")
            if want is not None and got is not None:
                diffs.append(got - want)
        print(f"  weka_main server ISL - trace `in`: {dist(diffs)}")

    # 4. repetition
    print("\n[4] distinct 4-gram ratio of outputs (1.0 = no repeats)")
    rep_b = defaultdict(list)
    by_sess = defaultdict(list)
    for lg in log.values():
        if lg.get("phase") != "profiling" or lg.get("distinct4") is None:
            continue
        rep_b[bucket(lg["turn"])].append(lg["distinct4"])
        by_sess[(lg["root"], lg["conv"])].append((lg["turn"], lg["distinct4"], lg.get("subbed", 0)))
    for b in BUCKETS:
        xs = rep_b[b]
        if xs:
            print(f"    turn {b:>5}: n={len(xs)} mean={statistics.fmean(xs):.3f} "
                  f"p10={pct(xs, .1):.3f} p50={pct(xs, .5):.3f}")
    drift = []
    for seq in by_sess.values():
        seq.sort()
        if len(seq) >= 3:
            drift.append(seq[-1][1] - seq[0][1])
    if drift:
        print(f"  per session (>=3 outputs) last - first: n={len(drift)} "
              f"mean={statistics.fmean(drift):+.3f} p10={pct(drift, .1):+.3f} p50={pct(drift, .5):+.3f}")
    with_sub = [lg["distinct4"] for lg in log.values()
                if lg.get("phase") == "profiling" and lg.get("distinct4") is not None and lg.get("subbed")]
    without = [lg["distinct4"] for lg in log.values()
               if lg.get("phase") == "profiling" and lg.get("distinct4") is not None and not lg.get("subbed")]
    if with_sub and without:
        print(f"  history with our outputs: mean={statistics.fmean(with_sub):.3f} (n={len(with_sub)}); "
              f"trace-only history: mean={statistics.fmean(without):.3f} (n={len(without)})")

    # 5. prefill
    print("\n[5] prefill per request (server usage)")
    p, c, comp = prefill_summary(recs)
    print(f"  this run: prompt={p:.0f} cached={c:.0f} computed={comp:.0f}  (n={len(recs)})")
    if a.baseline_dir:
        brecs = load_records(a.baseline_dir)
        bp, bc, bcomp = prefill_summary(brecs)
        print(f"  baseline: prompt={bp:.0f} cached={bc:.0f} computed={bcomp:.0f}  (n={len(brecs)})")
        print(f"  computed delta (this - baseline): {comp - bcomp:+.0f} tokens/request")


if __name__ == "__main__":
    main()
