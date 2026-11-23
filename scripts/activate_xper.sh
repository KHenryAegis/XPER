#!/usr/bin/env bash
# Source after activating the desired Python environment on the destination.
XPER_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export XPER_ROOT
export PYTHONPATH="$XPER_ROOT:$XPER_ROOT/third_party/tau2-bench/src${PYTHONPATH:+:$PYTHONPATH}"
export TAU2_DATA_DIR="$XPER_ROOT/third_party/tau2-bench/data"
export XPER_MATH_DATA="${XPER_MATH_DATA:-$XPER_ROOT/data/external/math/train.jsonl}"
export XPER_PYTHON="${XPER_PYTHON:-$(command -v python)}"
export PYTHONDONTWRITEBYTECODE=1
# Set XPER_MODEL_DIR and START explicitly when running new GPU experiments.
