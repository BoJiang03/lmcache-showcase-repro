# SPDX-License-Identifier: Apache-2.0
"""Build or verify the agentic workload's trace directory.

The trace is the HF dataset ``semianalysisai/cc-traces-weka-062126-256k`` at a
pinned revision: one ``traces.jsonl``, one session tree per line. AIPerf's
``weka_trace`` loader reads a directory with one ``<id>.json`` per tree, so
this script splits the file by ``id``, writing each line unchanged.

Usage::

    fetch_trace.py build OUT_DIR    # download (HF cache), verify, split
    fetch_trace.py check TRACE_DIR  # verify an existing directory
    fetch_trace.py digest TRACE_DIR # print dir_digest (no download)

Both verify ``traces.jsonl`` against the pinned sha256 first. ``check`` then
requires TRACE_DIR to hold exactly one file per tree, each parsing to the same
JSON as its line (content, not bytes, so a directory written by another
splitter still passes). ``digest`` is the byte-level check the runner does
before every run without network or HF cache; config.toml pins its value for
the directory ``build`` writes.
"""

# Standard
import hashlib
import json
import os
import sys

REPO_ID = "semianalysisai/cc-traces-weka-062126-256k"
REVISION = "8fecd2fc56694469f758f0afbbb6335ad3043740"
TRACES_SHA256 = "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef"
NUM_TREES = 393


def dir_digest(trace_dir: str) -> str:
    """Return sha256 over the sorted (file name, file sha256) pairs of a dir.

    Args:
        trace_dir: Directory of trace files.

    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    for name in sorted(os.listdir(trace_dir)):
        with open(os.path.join(trace_dir, name), "rb") as f:
            h.update(f"{name}\0{hashlib.sha256(f.read()).hexdigest()}\n".encode())
    return h.hexdigest()


def load_lines() -> list[str]:
    """Download the pinned ``traces.jsonl`` and return its lines.

    Returns:
        One JSON string per session tree.

    Raises:
        SystemExit: when the file's sha256 or line count differs from the pin.
    """
    # Imported here so `digest` (used by the runner) needs only the stdlib.
    # Third Party
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        REPO_ID, "traces.jsonl", repo_type="dataset", revision=REVISION
    )
    with open(path, "rb") as f:
        data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != TRACES_SHA256:
        sys.exit(f"traces.jsonl sha256 {digest} != pinned {TRACES_SHA256}")
    lines = data.decode().splitlines()
    if len(lines) != NUM_TREES:
        sys.exit(f"{len(lines)} trees, expected {NUM_TREES}")
    return lines


def build(out_dir: str) -> None:
    """Write one ``<id>.json`` per tree into a new directory ``out_dir``.

    Args:
        out_dir: Directory to create; must not exist.
    """
    lines = load_lines()
    os.makedirs(out_dir)
    for line in lines:
        with open(os.path.join(out_dir, json.loads(line)["id"] + ".json"), "w") as f:
            f.write(line)
    print(f"built {out_dir}: {len(lines)} trees")


def check(trace_dir: str) -> None:
    """Verify ``trace_dir`` against the pinned trace.

    Args:
        trace_dir: Directory produced by :func:`build` or an equivalent split.

    Raises:
        SystemExit: on a missing, extra or differing file.
    """
    want = {}
    for line in load_lines():
        tree = json.loads(line)
        want[tree["id"] + ".json"] = tree
    have = set(os.listdir(trace_dir))
    if have != set(want):
        sys.exit(
            f"file set differs: {len(have - set(want))} extra, "
            f"{len(set(want) - have)} missing"
        )
    for name, tree in want.items():
        with open(os.path.join(trace_dir, name)) as f:
            if json.load(f) != tree:
                sys.exit(f"{name} differs from the pinned trace")
    print(f"ok {trace_dir}: {len(want)} trees match {REPO_ID}@{REVISION[:8]}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("build", "check", "digest"):
        sys.exit(__doc__)
    if sys.argv[1] == "digest":
        print(dir_digest(sys.argv[2]))
    else:
        {"build": build, "check": check}[sys.argv[1]](sys.argv[2])
