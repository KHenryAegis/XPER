#!/usr/bin/env bash
# Clone tau2-bench (MIT) at the commit used for the paper into third_party/tau2-bench and install it
# into the active Python environment. The experiments used this commit without local modifications.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${XPER_PYTHON:-python3}
COMMIT=$("$PY" -c "import json; print(json.load(open('data/datasets.json'))['datasets']['tau2-bench-telecom']['commit'])")
URL=$("$PY" -c "import json; print(json.load(open('data/datasets.json'))['datasets']['tau2-bench-telecom']['repository'])")
if [ ! -d third_party/tau2-bench/.git ]; then
  git clone "$URL" third_party/tau2-bench
fi
git -C third_party/tau2-bench fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C third_party/tau2-bench checkout --quiet "$COMMIT"
"$PY" -m pip install -e third_party/tau2-bench
echo "tau2-bench at $(git -C third_party/tau2-bench rev-parse HEAD)"
