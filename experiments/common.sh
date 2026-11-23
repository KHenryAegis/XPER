# Shared settings of experiments/*.sh. Sourced by every step, not run on its own.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
: "${XPER_HOME:?set XPER_HOME to the directory that holds XPER/, models/ and miniconda3/ (README, Layout)}"
export XPER_HOME
export XPER_PYTHON=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
source scripts/activate_xper.sh    # PYTHONPATH, TAU2_DATA_DIR, XPER_MATH_DATA

PY=$XPER_PYTHON
H=$XPER_HOME/models/hf             # downloaded checkpoints (data/datasets.json)
O=$XPER_HOME/models/xper           # checkpoints written by these steps
SRC_TEL='results/src/4b-inst.shard*.db'          # stored telecom source log (2,400 episodes, 223 successes)
SRC_MATH='results/math/src/4b-inst.shard*.db'    # stored math source log (2,000 episodes, 452 successes)
MATH="--env math --max-steps 12 --max-tokens 2048 --env-workers 48"

# The paper ran on 8-GPU nodes; two 4-GPU training runs shared a node. Override for other layouts.
HALF_A=${HALF_A:-0 1 2 3}
HALF_B=${HALF_B:-4 5 6 7}
comma() { echo "${1// /,}"; }      # run_rft.sh takes comma-separated GPUs, run_grpo*.sh space-separated

mkdir -p results/eval logs/queue
step() { printf '\n=== %s %s\n' "$(date '+%m-%d %H:%M')" "$*"; }
summary() { grep -h '^{"model' "$@" || true; }   # the one-line result every evaluation log ends with
# wait for every background job of this shell; fail if any of them failed
wait_all() { local rc=0 p; for p in "$@"; do wait "$p" || rc=1; done; return $rc; }
