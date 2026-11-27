#!/usr/bin/env bash
# Paper comparisons of source reuse. Run from an activated experiment environment.
source "$(dirname "$0")/common.sh"

if (( $# > 1 )); then
  echo "Usage: $0 [A|B|C|math]" >&2
  exit 2
fi
part=${1:-all}
case "$part" in
  A|B|C|math|all) ;;
  *) echo "Usage: $0 [A|B|C|math]" >&2; exit 2 ;;
esac

read -ra ga <<< "$HALF_A"
read -ra gb <<< "$HALF_B"
TR=${XPER_TORCHRUN:-$(dirname "$PY")/torchrun}
mkdir -p logs/reuse
unset INJECT_SELF PREFIX_SRC HINT_SRC ENVARGS EVALARGS

eval_model() {
  local gpu=$1 model=$2 run=$3
  shift 3
  if [[ -e "results/eval/$run.db" || -e "results/eval/$run.log" ]]; then
    echo "Existing evaluation for $run. Use a clean results directory." >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/stage_guard.py --gpus "$gpu" --kind eval \
    --log "results/eval/$run.log" -- "$PY" scripts/rollout.py --chatml --model "$model" \
    --model-id "$run" --split test "$@" --out "results/eval/$run.db"
  grep '^{"model"' "results/eval/$run.log"
}

sft() {
  local gpus=$1 ranks=$2 source=$3 model=$4 out=$5 label=$6
  CUDA_VISIBLE_DEVICES="$gpus" "$PY" scripts/stage_guard.py --gpus "$gpus" --kind update \
    --log "logs/reuse/$label.train.log" -- "$TR" --standalone --nproc_per_node "$ranks" scripts/bc_train.py \
    --full --src "$source" --model "$model" --out "$out" --epochs 2 --lr 1e-5 --grad-accum 2 --seed 0
}

run_a() {
  local driver first second
  test "${#ga[@]}" -eq 4 && test "${#gb[@]}" -eq 4
  test -f "$O/8b-xfer/final/model.safetensors.index.json"
  for seed in 601 602 603; do
    if [[ -e "results/xs-grpo-tel.s$seed" || -e "results/eval/xs-grpo-tel-s$seed.db" ||
          -e "$O/xs-grpo-tel-s$seed-0" || -e "$O/xs-grpo-tel-s$seed-1" || -e "$O/xs-grpo-tel-s$seed-final" ]]; then
      echo "Existing A outputs for seed $seed. Use a clean results and model directory." >&2
      return 1
    fi
  done
  # Reuse the cohort scheduler's guard around vLLM and FSDP stages.
  driver=$(mktemp "$ROOT/results/.reuse-a.XXXXXX")
  trap 'rm -f -- "$driver"' EXIT
  PYTHONPATH="$ROOT/scripts:$PYTHONPATH" "$PY" - "$driver" <<'PY'
import sys
from pathlib import Path

from recover_strengthen import guard_driver

Path(sys.argv[1]).write_text(guard_driver(Path("scripts/run_grpo.sh").read_text()))
PY
  train_a() {
    local seed=$1 gpus=$2 clean=$3
    env START="$O/8b-xfer/final" TAG=xs-grpo-tel SEED="$seed" GPUS="$gpus" BUDGET=44414 MAX_IT=400 \
      THINK= TASKS=16 GROUP=8 LR=1e-6 SPLIT=train_src_solved LEAN=1 INJECT= INJECT_MODE=fail \
      INJECT_LIMIT=0 INJECT_CONTROL=none SRC_TASK_FRAC=1.0 CLEAN_FINAL="$clean" \
      bash "$driver" > "logs/reuse/xs-grpo-tel.s$seed.driver.log" 2>&1
  }
  train_a 601 "$HALF_A" "" &
  first=$!
  train_a 602 "$HALF_B" 1 &
  second=$!
  wait_all "$first" "$second"
  train_a 603 "$HALF_A" 1
  rm -f -- "$driver"
  trap - EXIT
  summary results/eval/xs-grpo-tel-s60{1,2,3}.log
}

run_b() {
  local keep=results/icl_keep_mobile_data_issue.txt seed
  local -a pids=()
  "$PY" - "$keep" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
splits = json.loads(Path("data/splits/telecom_extra_splits.json").read_text())
tasks = [t for t in splits["train_src_solved"] if t.startswith("[mobile_data_issue]")]
if len(tasks) != 24:
    raise ValueError("expected 24 source-solved mobile-data tasks")
out.write_text("\n".join(tasks) + "\n")
PY
  for seed in 0 1 2; do
    eval_model "${gb[$seed]}" "$H/Qwen3-8B" "8b-instruct-icl6-mobile-s$seed" --n 8 --no-think \
      --icl-src "$SRC_TEL" --icl-k 6 --icl-seed "$seed" --icl-keep "$keep" &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
}

run_c() {
  local hermes="$H/Hermes-3-Llama-3.1-8B" first second
  test "${#gb[@]}" -eq 4
  "$PY" - "$hermes" <<'PY'
import sys

from transformers import AutoTokenizer
from xper.chat import render

tok = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
delimiters = {}
for text in ("<|im_start|>", "<|im_end|>"):
    ids = tok.encode(text, add_special_tokens=False)
    if ids != [tok.convert_tokens_to_ids(text)] or tok.decode(ids) != text:
        raise ValueError(f"invalid Hermes ChatML delimiter {text}: {ids}")
    delimiters[text] = ids[0]
prompt = render([{"role": "system", "content": "test"}])
prompt_ids = tok.encode(prompt, add_special_tokens=False)
if (prompt_ids.count(delimiters["<|im_start|>"]) != 2 or
        prompt_ids.count(delimiters["<|im_end|>"]) != 1 or tok.eos_token_id != delimiters["<|im_end|>"]):
    raise ValueError("Hermes ChatML rendering or EOS mismatch")
print(f"Hermes delimiters {delimiters}, EOS {tok.eos_token_id}")
PY
  eval_model "${gb[0]}" "$hermes" hermes8b-start --n 8 &
  first=$!
  eval_model "${gb[1]}" "$hermes" hermes8b-icl6 --n 8 --icl-src "$SRC_TEL" --icl-k 6 --icl-seed 0 &
  second=$!
  wait_all "$first" "$second"
  sft "$(comma "$HALF_B")" 4 "$SRC_TEL" "$hermes" "$O/hermes8b-xfer" hermes8b-xfer
  eval_model "${gb[0]}" "$O/hermes8b-xfer/final" hermes8b-xfer --n 8
}

run_math() (
  export XPER_STRICT_MATH=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
  local workers="${gb[0]} ${gb[1]} ${gb[2]}"
  # Step 00 initializes the same base with --mode eos as the matched cohort.
  sft "$(comma "$workers")" 3 "$SRC_MATH" "$O/qwen3-8b-base-init-eos" "$O/math/sc-math-xfer" sc-math-xfer
  eval_model "${gb[0]}" "$O/math/sc-math-xfer/final" sc-math-xfer \
    --env math --n 4 --max-steps 12 --max-tokens 2048 --env-workers 48
)

case "$part" in
  all) run_a; run_b; run_c; run_math ;;
  A) run_a ;;
  B) run_b ;;
  C) run_c ;;
  math) run_math ;;
esac
