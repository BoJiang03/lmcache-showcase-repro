# SPDX-License-Identifier: Apache-2.0
"""Score one run: the same metric set for every workload.

Inputs in the run dir: ``aiperf/profile_export.jsonl`` (one record per
request) and ``samples.jsonl`` (:mod:`sampler`). The window comes from the
workload's ``scoring.window``:

* ``anchor``: ``first_profiling_start`` (profiling-phase records only) or
  ``first_request_start`` (all records); the window opens ``offset_s`` later.
* ``length_s``: window length; absent means "until the last request ends".
* ``count_by``: a request is in the window if its ``start`` or ``end`` is.

Per-request decode rate is ``(output tokens - 1) / (latency - TTFT)``, for
requests with more than one output token. Percentiles are linear-interpolated.
Server counters are interpolated at the window edges.

Definitions follow prt-cache analyze_goodput.py (agentic) and nqa_ana.py
(nqa), with two unifications: nqa percentiles were nearest-rank, and errors
are now counted inside the window for both.
"""

# Standard
from pathlib import Path
from typing import Any
import json

_SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")


def pct(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile.

    Args:
        xs: Values, any order.
        q: Quantile in [0, 1].

    Returns:
        The percentile, or None for an empty list.
    """
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def load_requests(path: Path) -> list[dict[str, Any]]:
    """Read AIPerf per-request records into flat dicts (seconds, tokens).

    Args:
        path: ``profile_export.jsonl``.

    Returns:
        One dict per record: phase, start, end, ttft, lat, osl, rate, err.
    """
    out = []
    with open(path) as f:
        for line in f:
            x = json.loads(line)
            m, me = x["metadata"], x.get("metrics", {})

            def val(k: str, me: dict[str, Any] = me) -> Any:
                v = me.get(k)
                return v.get("value") if isinstance(v, dict) else None

            ttft, lat = val("time_to_first_token"), val("request_latency")
            ttft = ttft / 1e3 if ttft is not None else None
            lat = lat / 1e3 if lat is not None else None
            osl = val("usage_completion_tokens") or val("output_sequence_length") or 0
            rate = None
            if ttft is not None and lat is not None and osl > 1 and lat > ttft:
                rate = (osl - 1) / (lat - ttft)
            out.append(
                {
                    "phase": m.get("benchmark_phase"),
                    "start": m["request_start_ns"] / 1e9,
                    "end": m["request_end_ns"] / 1e9,
                    "ttft": ttft,
                    "lat": lat,
                    "osl": osl,
                    "rate": rate,
                    "err": bool(x.get("error")) or m.get("was_cancelled", False),
                }
            )
    return out


def window_bounds(
    reqs: list[dict[str, Any]], window: dict[str, Any]
) -> tuple[float, float, list[dict[str, Any]]]:
    """Compute the scored window and the requests that fall in it.

    Args:
        reqs: Output of :func:`load_requests`.
        window: The workload's ``scoring.window`` table.

    Returns:
        (start, end, requests in the window, errored ones included).

    Raises:
        ValueError: on an unknown anchor.
    """
    if window["anchor"] == "first_profiling_start":
        pop = [r for r in reqs if r["phase"] == "profiling"]
    elif window["anchor"] == "first_request_start":
        pop = reqs
    else:
        raise ValueError(f"unknown window anchor {window['anchor']!r}")
    ws = min(r["start"] for r in pop) + window["offset_s"]
    if "length_s" in window:
        we = ws + window["length_s"]
    else:
        we = max(r["end"] for r in pop)
    by = window["count_by"]
    return ws, we, [r for r in pop if ws <= r[by] <= we]


def _at(samples: list[dict[str, Any]], t: float, key: str) -> float:
    prev = None
    for s in samples:
        if key not in s:
            continue
        if s["t"] >= t:
            if prev is None:
                return s[key]
            f = (t - prev["t"]) / (s["t"] - prev["t"])
            return prev[key] + f * (s[key] - prev[key])
        prev = s
    if prev is None:
        raise KeyError(f"{key} never sampled")
    return prev[key]


def score(
    run_dir: Path,
    window: dict[str, Any],
    l2_read: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute the run's metrics.

    Args:
        run_dir: Run dir holding ``aiperf/`` and ``samples.jsonl``.
        window: The workload's ``scoring.window`` table.
        l2_read: The deployment's ``l2_read`` table, or None.

    Returns:
        Metrics, keyed by name.

    Raises:
        KeyError: a server counter was never sampled.
    """
    reqs = load_requests(run_dir / "aiperf" / "profile_export.jsonl")
    ws, we, win = window_bounds(reqs, window)
    ok = [r for r in win if not r["err"] and r["ttft"] is not None]
    ttft = [r["ttft"] for r in ok]
    rates = [r["rate"] for r in ok if r["rate"] is not None]
    with open(run_dir / "samples.jsonl") as f:
        samples = [json.loads(line) for line in f]

    def delta(key: str) -> float:
        return _at(samples, we, key) - _at(samples, ws, key)

    src = {k: delta(f"vllm:prompt_tokens_by_source/{k}") for k in _SOURCES}
    total = sum(src.values())
    drafts = delta("vllm:spec_decode_num_drafts_total")
    out: dict[str, Any] = {
        "window_start": ws,
        "window_s": we - ws,
        "requests": len(ok),
        "errors": sum(r["err"] for r in win),
        "ttft_p50": pct(ttft, 0.5),
        "ttft_p90": pct(ttft, 0.9),
        "ttft_p95": pct(ttft, 0.95),
        "decode_p10": pct(rates, 0.1),
        "decode_p50": pct(rates, 0.5),
        "prompt_tokens_by_source": src,
        "prompt_token_share": {k: v / total for k, v in src.items()} if total else None,
        # Served by the cache layer (L1 and L2 together), as vLLM counts it;
        # the same meter for every deployment.
        "external_token_share": src["external_kv_transfer"] / total if total else None,
        "preemptions": delta("vllm:num_preemptions_total"),
        "mtp_mean_accept_len": (
            1 + delta("vllm:spec_decode_num_accepted_tokens_total") / drafts
            if drafts
            else None
        ),
    }
    if l2_read is not None:
        read = sum(delta(f"io_rbytes/{c}") for c in l2_read["containers"])
        out["l2_read_gb"] = read / 1e9
    return out
