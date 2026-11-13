# Sourced by the case scripts. Override any of these in the environment.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PRT_W=${PRT_W:-/mnt/nvme0/yuhan/tmp/nvidia/bo-prt-glm52}   # workspace: venvs, models, trace, results
W=$PRT_W
PY=$W/venv/bin/python
RESULTS=${RESULTS:-$W/results}
L2_ROOT=${L2_ROOT:-/mnt/nvme1/$USER}                                # NVMe for the case-3 fs_native L2
mkdir -p "$RESULTS" "$W/logs"
