# SPDX-License-Identifier: Apache-2.0
"""One GLM-5.2-FP8 DEP8 run without docker: vLLM native offload vs LMCache.

    python3 -m glm_dep8.run_one --mode lmcache|native --concurrency N --out DIR
        [--duration 960] [--max-model-len 202752] [--vllm-arg --flag=value ...]

Both modes share every vLLM setting (DP8 + EP8, default DP routing, i.e. no
prefix affinity); they differ only in the KV connector:

* native:  OffloadingConnector, CPUOffloadingSpec, 150 GB of CPU memory per DP
           rank (8 private CPU caches, nothing shared between ranks).
* lmcache: LMCacheMPConnector -> one LMCache MP server, L1 1.2 TB shared by all
           8 ranks. No L2 on either side.

Output in DIR: vllm.log, lmcache.log, aiperf.log, aiperf/, samples.jsonl,
meta.json, result.json (runner.score metrics + throughput/interactivity).
"""

from __future__ import annotations

import argparse
import json
import shlex
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from runner.sampler import Sampler  # noqa: E402
from runner.score import load_requests, pct, score, window_bounds  # noqa: E402

W = Path(os.environ.get("PRT_W", "/mnt/nvme0/yuhan/tmp/nvidia/bo-prt-glm52"))
# Per-model settings; select_model() sets MODEL/HF_REPO/SERVED/REV/PARSERS from --model.
MODELS = {
    "glm52": dict(dir="GLM-5.2-FP8", repo="zai-org/GLM-5.2-FP8", served="glm52",
                  rev="f33c6dc501ee5a2c7e35155653b1b1abbc320951",
                  extra=["--reasoning-parser", "glm45", "--tool-call-parser", "glm47"]),
    # Case 2: TP8 + EP, 4 KV heads -> every KV head lives on 2 TP ranks.
    "minimax": dict(dir="MiniMax-M3.1-Flash", repo="MiniMaxAI/MiniMax-M3.1-Flash",
                    served="minimax", rev="ce7f4df87d2e05262f5775f373dac4772f134041",
                    extra=["--reasoning-parser", "minimax_m3", "--tool-call-parser",
                           "minimax_m3", "--language-model-only",
                           # required: M3's indexer uses sparse_block_size 128
                           # (LMCache docs/source/recipes/minimax_m3.rst)
                           "--block-size", "128",
                           # The rest as PRT's minimax31 (vLLM recipe, Hopper TP8).
                           "--disable-hybrid-kv-cache-manager",
                           "--attention-config", '{"indexer_kv_dtype":"fp8"}'],
                    ep=False,
                    compile='{"cudagraph_mode":"FULL_AND_PIECEWISE",'
                            '"pass_config":{"fi_allreduce_fusion_max_size_mb":1}}',
                    env={"VLLM_USE_RUST_FRONTEND": "1"}),
    # Case 3: SWA hybrid (9 full + 39 sliding-window layers, window 128), dflash drafter.
    "mimo26": dict(dir="MiMo-V2.6-Flash-RL", repo="XiaomiMiMo/MiMo-V2.6-Flash-RL",
                   served="mimo", rev="f657ba2f7d24c9adc5730675baaa2d34de878a49",
                   extra=["--reasoning-parser", "mimo", "--tool-call-parser", "mimo",
                          "--language-model-only"],
                   ep=False),  # the vLLM recipe serves it without EP
}
MODEL = W / "models" / "GLM-5.2-FP8"
HF_REPO = "zai-org/GLM-5.2-FP8"
SERVED = "glm52"
REV = MODELS["glm52"]["rev"]
MODEL_EXTRA: list[str] = MODELS["glm52"]["extra"]
MODEL_CFG: dict = MODELS["glm52"]


def select_model(name: str) -> None:
    global MODEL, HF_REPO, SERVED, REV, MODEL_EXTRA, MODEL_CFG
    m = MODEL_CFG = MODELS[name]
    MODEL, HF_REPO, SERVED, REV, MODEL_EXTRA = (
        W / "models" / m["dir"], m["repo"], m["served"], m["rev"], m["extra"])


def tokenizer_dir() -> Path:
    org, name = HF_REPO.split("/")
    return W / "hf-alias" / "hub" / f"models--{org}--{name}" / "snapshots" / REV
VLLM_PORT = 18000
LMC_PORT, LMC_HTTP = 16555, 18080
CPU_PER_RANK = 150 * 2**30  # 150 GiB per DP rank; 8 x 150 = 1200 GiB = LMCache L1
L1_GB = 1200  # LMCache GiB-based flag; see meta for bytes
DP = 8

VLLM_ENV = {
    "VLLM_SERVER_DEV_MODE": "1",  # /reset_prefix_cache
    "PYTHONHASHSEED": "0",
    "HF_HUB_OFFLINE": "1",
    "VLLM_CACHE_ROOT": str(W / "vllm-cache"),
    "TORCHINDUCTOR_CACHE_DIR": str(W / "vllm-cache" / "inductor"),
    "FLASHINFER_WORKSPACE_BASE": str(W / "fi_ws"),
    "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "600",
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",  # 704 GB cold load takes >600 s
    "CUDA_HOME": "/usr/local/cuda",
    "TMPDIR": str(W / "tmp"),
    # As PRT's [vllm.env].
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "fork",
}
# PRT's H200 default per TP size; unset, the API server initializes CUDA to look it up.
FUSION_MB = {8: 1, 4: 2}


def vllm_args(mode: str, a: argparse.Namespace) -> list[str]:
    args = [
        str(MODEL),
        "--served-model-name", SERVED,
        "--port", str(VLLM_PORT),
        *(["--tensor-parallel-size", str(a.tp)] if a.tp > 1
          else ["--data-parallel-size", str(DP)]),
        *(["--enable-expert-parallel"]
          if {"on": True, "off": False}.get(a.ep, MODEL_CFG.get("ep", True)) else []),
        "--kv-cache-dtype", a.kv_dtype,
        "--gpu-memory-utilization", str(a.gpu_util),
        "--max-num-batched-tokens", str(a.mbt),
        "--max-num-seqs", str(a.max_num_seqs),
        "--max-model-len", str(a.max_model_len),
        "--enable-prefix-caching",
        "--trust-remote-code",
        *MODEL_EXTRA,
        "--enable-auto-tool-choice",
        "--enable-prompt-tokens-details",
        "--generation-config", "vllm",
        "--disable-cascade-attn",
    ]
    compile_cfg = MODEL_CFG.get("compile") or (
        json.dumps({"pass_config": {"fi_allreduce_fusion_max_size_mb": FUSION_MB[a.tp]}})
        if a.tp in FUSION_MB else "")
    if compile_cfg and not any(x.startswith("--compilation-config") for x in a.vllm_arg):
        args += ["--compilation-config", compile_cfg]
    if a.spec_method:
        # e.g. dspark: MiniMax-M3 ships the draft in its dspark/ subfolder and the
        # draft carries its own width (dspark_block_size), so no token count.
        spec = {"method": a.spec_method}
        if a.spec_model:  # drafter in a subfolder of the target repo (e.g. MiMo dflash/)
            spec |= {"model": str(MODEL / a.spec_model), "num_speculative_tokens": a.mtp}
        args += ["--speculative-config", json.dumps(spec)]
    elif a.mtp:
        args += ["--speculative-config", json.dumps(
            {"method": "mtp", "num_speculative_tokens": a.mtp})]
    if mode == "native":
        cfg = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "spec_name": "CPUOffloadingSpec",
                "block_size": a.offload_block,
                "cpu_bytes_to_use": a.native_cpu_bytes,
                "eviction_policy": "lru",
            },
        }
    else:
        cfg = {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": LMC_PORT,
                "lmcache.mp.mq_timeout": 1500,
                **json.loads(a.lmc_extra),
            },
        }
    args += ["--kv-transfer-config", json.dumps(cfg)]
    return args + a.vllm_arg


def lmcache_args(a: argparse.Namespace) -> list[str]:
    return [
        "server", "--host", "127.0.0.1", "--port", str(LMC_PORT),
        "--http-port", str(LMC_HTTP),
        "--chunk-size", str(a.chunk),
        "--separate-object-groups",
        "--l1-size-gb", str(a.l1_gb),
        "--eviction-policy", "LRU",
        "--disable-observability",
        "--eviction-trigger-watermark", "0.95",
        "--eviction-ratio", "0.05",
        # One worker per rank (default 1: every rank's stores and loads queue on
        # one thread), and read locks that outlive long queues (PRT).
        "--max-workers", str(a.tp if a.tp > 1 else DP),
        "--l1-read-ttl-seconds", "3600",
    ] + a.lmcache_arg


def aiperf_args(a: argparse.Namespace, out: Path, mode: str) -> list[str]:
    sm = [f"http://127.0.0.1:{VLLM_PORT}/metrics"]
    if mode == "lmcache":
        sm.append(f"http://127.0.0.1:{LMC_HTTP}/metrics")
    args = [
        "profile",
        "--url", f"http://127.0.0.1:{VLLM_PORT}",
        "--model", SERVED,
        "--endpoint-type", "chat",
        "--streaming",
        "--extra-inputs", "ignore_eos:true",
        "--use-server-token-count",
        "--random-seed", "0",
        "--export-level", "records",
        "--slice-duration", "30",
        "--no-gpu-telemetry",
        "--scenario", "inferencex-agentx-mvp",
        "--unsafe-override",
        "--tokenizer", HF_REPO,
        "--custom-dataset-type", "weka_trace",
        "--input-file", str(W / "trace" / "weka-062126-256k-dir"),
        "--max-context-length", str(a.max_context),
        "--prompt-corpus", "coding",
        "--cache-bust", "system_suffix",
        "--benchmark-duration", str(a.duration),
        "--benchmark-grace-period", "60",
        "--ui", "simple",
        "--concurrency", str(a.concurrency),
        "--server-metrics", *sm,
        "--artifact-dir", str(out / "aiperf"),
    ]
    return args + a.aiperf_arg


def wait_http(url: str, timeout: float, procs: list[subprocess.Popen]) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        for p in procs:
            if p.poll() is not None:
                return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def post(url: str) -> int:
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status


def free_gpus() -> list[int]:
    q = lambda arg: subprocess.run(["nvidia-smi", arg, "--format=csv,noheader"],
                                   capture_output=True, text=True, check=True).stdout.split("\n")
    busy = {u.strip() for u in q("--query-compute-apps=gpu_uuid") if u.strip()}
    out = []
    for row in q("--query-gpu=index,uuid,memory.used"):
        if not row.strip():
            continue
        i, u, mem = [x.strip() for x in row.split(",")]
        if u not in busy or int(mem.split()[0]) < 2048:
            out.append(int(i))
    return out


def reap_stragglers(cvd: str, timeout: float = 120) -> None:
    """Kill our own vLLM workers / engine cores that outlived the server's group.

    vLLM TP workers are launched via numactl and can survive the SIGINT/SIGKILL of
    the `vllm serve` process group, keeping a GPU (seen: 131 GiB on one GPU after a
    native run); the next run would then wait forever for free GPUs. Only processes
    started with this run's CUDA_VISIBLE_DEVICES are ours: another run may share the
    host on other GPUs.
    """
    me = str(os.getuid())
    t0 = time.time()
    while time.time() - t0 < timeout:
        left = []
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/status") as f:
                    if next(l for l in f if l.startswith("Uid:")).split()[1] != me:
                        continue
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
                with open(f"/proc/{pid}/environ", "rb") as f:
                    env = dict(e.split(b"=", 1) for e in f.read().split(b"\0") if b"=" in e)
            except (OSError, StopIteration):
                continue
            if env.get(b"CUDA_VISIBLE_DEVICES", b"").decode() != cvd:
                continue
            if cmd.startswith(("VLLM::", "EngineCore")) or "bo-prt-glm52/venv/bin/vllm" in cmd:
                left.append(int(pid))
        if not left:
            return
        for pid in left:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(5)


def start(cmd: list[str], env: dict[str, str], log: Path) -> subprocess.Popen:
    f = open(log, "w")
    return subprocess.Popen(cmd, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def stop(p: subprocess.Popen | None, grace: float = 60) -> None:
    if p is None or p.poll() is not None:
        return
    os.killpg(p.pid, signal.SIGINT)
    try:
        p.wait(grace)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def extra_metrics(run_dir: Path, window: dict) -> dict:
    """Throughput and interactivity over the same window as runner.score."""
    recs = []
    with open(run_dir / "aiperf" / "profile_export.jsonl") as f:
        for line in f:
            x = json.loads(line)
            me = x.get("metrics", {})
            v = lambda k: (me.get(k) or {}).get("value") if isinstance(me.get(k), dict) else None
            recs.append(v("usage_prompt_tokens") or v("input_sequence_length") or 0)
    reqs = load_requests(run_dir / "aiperf" / "profile_export.jsonl")
    for r, isl in zip(reqs, recs):
        r["isl"] = isl
    ws, we, win = window_bounds(reqs, window)
    ok = [r for r in win if not r["err"] and r["ttft"] is not None]
    span = we - ws
    isl = sum(r["isl"] for r in ok)
    osl = sum(r["osl"] for r in ok)
    inter = [r["rate"] for r in ok if r["rate"] is not None]
    return {
        "input_tok_per_s": isl / span,
        "output_tok_per_s": osl / span,
        "total_tok_per_s": (isl + osl) / span,
        "interactivity_mean": sum(inter) / len(inter) if inter else None,
        "interactivity_p50": pct(inter, 0.5),
        "ttft_mean": sum(r["ttft"] for r in ok) / len(ok) if ok else None,
        "req_per_s": len(ok) / span,
    }


def main() -> int:
    global VLLM_PORT, LMC_PORT, LMC_HTTP
    global DP
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "lmcache"], required=True)
    ap.add_argument("--model", choices=sorted(MODELS), default="glm52")
    ap.add_argument("--model-dir", default="", help="override the model path (tokenizer stays the HF alias)")
    ap.add_argument("--dp", type=int, default=DP, help="data-parallel size (smoke on fewer GPUs)")
    ap.add_argument("--tp", type=int, default=1, help=">1: TP<n> + EP instead of DP8 + EP")
    ap.add_argument("--lmcache-path", default="",
                    help="dir prepended to PYTHONPATH (vLLM connector and LMCache server)")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--duration", type=int, default=960)
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--gpu-util", type=float, default=0.96)
    ap.add_argument("--mbt", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--mtp", type=int, default=1)
    ap.add_argument("--spec-method", default="", help="speculative method instead of MTP, e.g. dspark")
    ap.add_argument("--spec-model", default="", help="drafter subdir of the model dir, e.g. dflash")
    ap.add_argument("--kv-dtype", default="fp8")
    ap.add_argument("--lmc-extra", default="{}",
                    help="JSON merged into LMCacheMPConnector extra config, e.g. lazy offload")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--ep", choices=["auto", "on", "off"], default="auto",
                    help="expert parallel; auto follows the model's recipe")
    ap.add_argument("--vllm-env", action="append", default=[], help="K=V on top of the vLLM env")
    ap.add_argument("--gpus", default="", help="comma-separated GPU ids instead of the first free ones")
    ap.add_argument("--port-offset", type=int, default=0,
                    help="shift every server port, to run two experiments side by side")
    ap.add_argument("--offload-block", type=int, default=256)
    ap.add_argument("--l1-gb", type=int, default=L1_GB)
    ap.add_argument("--native-cpu-bytes", type=int, default=CPU_PER_RANK,
                    help="value passed as cpu_bytes_to_use (see meta: per-rank semantics)")
    ap.add_argument("--vllm-arg", action="append", default=[])
    ap.add_argument("--lmcache-arg", action="append", default=[])
    ap.add_argument("--aiperf-arg", action="append", default=[])
    ap.add_argument("--startup-timeout", type=int, default=3600)
    ap.add_argument("--client", default="",
                    help="command run instead of AIPerf once the servers are up")
    ap.add_argument("--keep-up", action="store_true", help="leave servers running (debug)")
    ap.add_argument("--reuse", action="store_true", help="use servers left up by --keep-up")
    a = ap.parse_args()
    DP = a.dp
    VLLM_PORT, LMC_PORT, LMC_HTTP = (p + a.port_offset for p in (VLLM_PORT, LMC_PORT, LMC_HTTP))
    ngpu = a.tp if a.tp > 1 else DP  # TP<n>: one engine on n GPUs
    select_model(a.model)
    if a.model_dir:  # e.g. a config-only dir for --load-format dummy smoke tests
        global MODEL
        MODEL = Path(a.model_dir)

    out: Path = a.out
    out.mkdir(parents=True, exist_ok=False)
    want = [int(g) for g in a.gpus.split(",")] if a.gpus else []
    pick = (lambda: [g for g in free_gpus() if g in want]) if want else free_gpus
    gpus = list(range(ngpu)) if a.reuse else pick()
    t_wait = time.time()
    while len(gpus) < ngpu and time.time() - t_wait < 900:  # shared host: brief foreign use
        time.sleep(15)
        gpus = pick()
    if len(gpus) < ngpu:
        print(f"only {len(gpus)} free GPUs: {gpus}", file=sys.stderr)
        return 3
    venv = W / "venv" / "bin"
    gcc = "/opt/rh/gcc-toolset-12/root/usr/bin"  # host gcc 10 is too old for torch's C++20 JIT
    path = f"{venv}:{gcc}:/usr/local/cuda/bin:{os.environ['PATH']}"
    env = {**VLLM_ENV, **MODEL_CFG.get("env", {}), "PATH": path, "CC": f"{gcc}/gcc", "CXX": f"{gcc}/g++",
           "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus[:ngpu])),
           **dict(kv.split("=", 1) for kv in a.vllm_env)}
    if a.lmcache_path:
        env["PYTHONPATH"] = a.lmcache_path
    for d in ("vllm-cache/inductor", "fi_ws", "tmp"):
        (W / d).mkdir(parents=True, exist_ok=True)

    cmds = {"vllm": [str(venv / "vllm"), "serve", *vllm_args(a.mode, a)]}
    if a.mode == "lmcache":
        cmds["lmcache"] = [str(venv / "lmcache"), *lmcache_args(a)]
    ap_env = {
        "PATH": f"{W / 'venv-aiperf' / 'bin'}:{os.environ['PATH']}",
        "HF_HOME": str(W / "hf-alias"),
        "HF_HUB_OFFLINE": "1",
        "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "3600",
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "3600",
        "AIPERF_DATASET_WEKA_PARALLEL_WORKERS": "1",
        "AIPERF_DATASET_MMAP_BASE_PATH": str(out / "aiperf_mmap" / "base"),
        "AIPERF_DATASET_MMAP_CACHE_DIR": str(out / "aiperf_mmap" / "cache"),
        "PRT_LIVE_HISTORY_TOKENIZER": str(tokenizer_dir()),
        "PRT_LIVE_HISTORY_LOG_DIR": str(out / "aiperf" / "prt_live_history"),
        "TMPDIR": str(W / "tmp"),
    }
    cmds["aiperf"] = [str(W / "venv-aiperf" / "bin" / "aiperf"), *aiperf_args(a, out, a.mode)]
    for sub in ("aiperf", "aiperf_mmap/base", "aiperf_mmap/cache"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(a) | {"out": str(out)}, "gpus": gpus[:ngpu], "cmds": cmds,
            "vllm_env": env, "aiperf_env": ap_env, "start": time.time()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    procs: dict[str, subprocess.Popen] = {}
    sampler = None
    rc = 1
    try:
        if a.reuse:
            # servers left up by an earlier --keep-up run with the same settings
            if not wait_http(f"http://127.0.0.1:{VLLM_PORT}/health", 30, []):
                print("--reuse: vllm not up", file=sys.stderr)
                return 4
        if a.mode == "lmcache" and not a.reuse:
            procs["lmcache"] = start(cmds["lmcache"], {**env, "LMCACHE_LOG_LEVEL": "WARNING"},
                                     out / "lmcache.log")
            if not wait_http(f"http://127.0.0.1:{LMC_HTTP}/metrics", 600, [procs["lmcache"]]):
                print("lmcache server not up", file=sys.stderr)
                return 4
        if not a.reuse:
            procs["vllm"] = start(cmds["vllm"], env, out / "vllm.log")
            t0 = time.time()
            if not wait_http(f"http://127.0.0.1:{VLLM_PORT}/health", a.startup_timeout,
                             list(procs.values())):
                print("vllm not up", file=sys.stderr)
                return 4
            meta["vllm_startup_s"] = time.time() - t0
        post(f"http://127.0.0.1:{VLLM_PORT}/reset_prefix_cache")
        status_urls = []
        if a.mode == "lmcache":
            post(f"http://127.0.0.1:{LMC_HTTP}/cache/clear")
            status_urls = [f"http://127.0.0.1:{LMC_HTTP}/status"]
        urls = [f"http://127.0.0.1:{VLLM_PORT}/metrics"]
        if a.mode == "lmcache":
            urls.append(f"http://127.0.0.1:{LMC_HTTP}/metrics")
        sampler = Sampler(out / "samples.jsonl", urls, {}, status_urls)
        sampler.start()
        if a.client:  # e.g. the KV correctness check instead of the benchmark
            p = start(shlex.split(a.client), ap_env, out / "client.log")
        else:
            p = start(cmds["aiperf"], ap_env, out / "aiperf.log")
        rc = p.wait(a.duration + 7200)
        meta["aiperf_rc"] = rc
    finally:
        if sampler:
            sampler.stop()
        if not a.keep_up:
            for k in ("vllm", "lmcache"):
                stop(procs.get(k))
            reap_stragglers(env["CUDA_VISIBLE_DEVICES"])
        meta["end"] = time.time()
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    window = {"anchor": "first_profiling_start", "offset_s": 0,
              "length_s": a.duration, "count_by": "end"}
    res = {}
    try:
        res = score(out, window, None)
    except Exception as e:  # keep going: extra metrics may still work
        res["score_error"] = repr(e)
    try:
        res.update(extra_metrics(out, window))
    except Exception as e:
        res["extra_error"] = repr(e)
    res |= {"mode": a.mode, "concurrency": a.concurrency, "aiperf_rc": rc}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if rc == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
