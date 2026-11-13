#!/usr/bin/env bash
# Build the three LMCache commits the runs used, from the third_party/LMCache submodule.
#   case 1 -> installed into $PRT_W/venv      (tag showcase-20261113-case1, dev 29340c7c)
#   case 2 -> $PRT_W/lmc-dedup                (tag showcase-20261113-case2, GQA replica dedup)
#   case 3 -> $PRT_W/lmc-lazy-fix             (tag showcase-20261113-case3, lazy offload + SWA fix)
# Needs gcc-toolset-12 (host gcc 10 cannot build the torch C++20 extensions) and uv.
set -euo pipefail
source "$(dirname "$0")/common.sh"
SRC=$REPO/third_party/LMCache
G=${GCC_BIN:-/opt/rh/gcc-toolset-12/root/usr/bin}
export PATH=$G:/usr/local/cuda/bin:$PATH CC=$G/gcc CXX=$G/g++ CUDAHOSTCXX=$G/g++ NVCC_CCBIN=$G/g++
git -C "$SRC" fetch --quiet --tags origin

build() {  # <tag> [uv pip install args...]
    local tag=$1 wt=$REPO/third_party/LMCache-$1; shift
    [ -d "$wt" ] || git -C "$SRC" worktree add --quiet --detach "$wt" "$tag"
    (cd "$wt" && uv pip install --python "$PY" --no-build-isolation --no-deps "$@" .)
}
build showcase-20261113-case1
build showcase-20261113-case2 --target "$W/lmc-dedup"
build showcase-20261113-case3 --target "$W/lmc-lazy-fix"
