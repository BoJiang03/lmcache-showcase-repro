# SPDX-License-Identifier: Apache-2.0
"""Background sampler: server counters and process I/O, one JSON line per tick.

Each line holds ``t`` (unix seconds), the value of every Prometheus sample
whose name is in :data:`METRICS`, summed over labels and URLs
(``vllm:prompt_tokens_by_source`` is split by its ``source`` label), and
``io_rbytes/<container>``: bytes the container's cgroup read from one block
device (``io.stat``), for the containers it was given. Scoring interpolates
these counters at the window edges. Same metric set as prt-cache sampler.py.
"""

# Standard
from pathlib import Path
from typing import Any
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

METRICS = (
    frozenset(
        f"vllm:{m}"
        for m in (
            "prompt_tokens_total",
            "generation_tokens_total",
            "prefix_cache_hits_total",
            "prefix_cache_queries_total",
            "external_prefix_cache_hits_total",
            "external_prefix_cache_queries_total",
            "spec_decode_num_drafts_total",
            "spec_decode_num_draft_tokens_total",
            "spec_decode_num_accepted_tokens_total",
            "request_success_total",
            "num_requests_running",
            "num_requests_waiting",
            "kv_cache_usage_perc",
            "num_preemptions_total",
        )
    )
    | frozenset(
        f"lmcache_mp_{m}"
        for m in (
            "l1_write_chunks_total",
            "l1_evicted_chunks_total",
            "l1_read_chunks_total",
            "l1_memory_usage_bytes",
            "l1_usage_ratio",
            "lookup_hit_tokens_total",
            "lookup_requested_tokens_total",
            "num_finished_stores_total",
            "num_finished_retrieves_total",
        )
    )
    | frozenset(
        # Mooncake master (once) and clients (summed over the four): whether puts
        # reach SSD before DRAM eviction drops them.
        (
            "master_key_count",
            "master_allocated_bytes",
            "master_total_capacity_bytes",
            "master_put_start_requests_total",
            "master_put_start_alloc_failures_total",
            "master_evicted_key_count",
            "master_evicted_size_bytes",
            "master_offload_enqueued_total",
            "master_offload_completed_total",
            "master_offload_failed_total",
            "master_offload_enqueue_rejected_total",
            "master_evict_disk_replica_requests_total",
            "mooncake_ssd_write_bytes_total",
            "mooncake_ssd_read_bytes_total",
            "mooncake_ssd_read_ops_total",
        )
    )
)
BY_SOURCE = ("vllm:prompt_tokens_by_source", "vllm:prompt_tokens_by_source_total")
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def _scrape(url: str, rec: dict[str, Any]) -> None:
    with urllib.request.urlopen(url, timeout=4) as r:
        text = r.read().decode()
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        if name in BY_SOURCE:
            src = re.search(r'source="([^"]+)"', labels or "")
            if src is None:
                continue
            key = f"vllm:prompt_tokens_by_source/{src.group(1)}"
        elif name in METRICS:
            key = name
        else:
            continue
        rec[key] = rec.get(key, 0.0) + float(value)


def _status(url: str, rec: dict[str, Any]) -> None:
    """Record L1 and L2 fill from an LMCache MP server's ``GET /status``.

    The server runs with observability off (it costs throughput), which
    removes the lmcache_mp_* Prometheus metrics; /status still reports usage.
    """
    with urllib.request.urlopen(url, timeout=4) as r:
        sm = json.loads(r.read())["storage_manager"]
    l1 = sm["l1_manager"]
    rec["lmcache_status/l1_used_bytes"] = l1["memory_used_bytes"]
    rec["lmcache_status/l1_objects"] = l1["total_object_count"]
    rec["lmcache_status/l2_used_bytes"] = sum(
        a["total_bytes_used"] for a in sm["l2_eviction_controller"]["adapters"]
    )


def _rbytes(io_stat: Path, dev: int) -> int:
    key = f"{os.major(dev)}:{os.minor(dev)} "
    with open(io_stat) as f:
        for line in f:
            if line.startswith(key):
                return int(line.split("rbytes=")[1].split()[0])
    # A cgroup lists a device only after its first I/O to it.
    return 0


class Sampler:
    """Samples metrics URLs and process read_bytes into a JSONL file."""

    def __init__(
        self,
        out: Path,
        urls: list[str],
        io: dict[str, tuple[Path, int]],
        status_urls: list[str],
        interval_s: float = 5.0,
        status_every: int = 6,
    ) -> None:
        """Prepare a sampler; call :meth:`start` to begin.

        Args:
            out: JSONL file to append to.
            urls: Prometheus endpoints to scrape.
            io: Container role -> (its cgroup ``io.stat``, device number
                ``st_dev``) whose read bytes to record.
            status_urls: LMCache ``/status`` endpoints (see :func:`_status`).
            interval_s: Seconds between ticks.
            status_every: Query ``status_urls`` every this many ticks; the
                call takes ~0.15 s inside the server, so not every tick.
        """
        self.out = out
        self.urls = urls
        self.io = io
        self.status_urls = status_urls
        self.interval_s = interval_s
        self.status_every = status_every
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        """Start sampling in a background thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the last line to be written."""
        self._stop.set()
        self._thread.join()

    def _loop(self) -> None:
        with open(self.out, "a") as f:
            tick = 0
            while not self._stop.is_set():
                t = time.time()
                rec: dict[str, Any] = {"t": t}
                for url in self.urls:
                    try:
                        _scrape(url, rec)
                    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                        rec[f"err {url}"] = str(e)
                for url in self.status_urls if tick % self.status_every == 0 else []:
                    try:
                        _status(url, rec)
                    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                        rec[f"err {url}"] = str(e)
                tick += 1
                for name, (io_stat, dev) in self.io.items():
                    try:
                        rec[f"io_rbytes/{name}"] = _rbytes(io_stat, dev)
                    except OSError as e:
                        rec[f"err io_rbytes/{name}"] = str(e)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                self._stop.wait(max(0.0, self.interval_s - (time.time() - t)))
