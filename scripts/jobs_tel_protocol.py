"""Telecom weak-target protocol-vs-strategy jobs (gpu_queue.py; one host): the uncapped controls of Section 6.1.

Question: at cold start, does an injected source success transfer interaction protocol (tool-call format,
how to end an episode) or task strategy (the right actions for this state)?
Weak start 8b-commonsft, train_src_solved (80 tasks), 44,414 target env calls, fail-only injection.
  mismatch : a source success of a different task, with this task's system prompt (grpo_advantages.py)
  shuffle  : this task's source success with its non-final actions permuted
Reference arms already run (same recipe): none 0.0 / 0.0 (2 seeds), inject 42.2 / 45.3 / 40.6 (3 seeds).
LEAN=1 keeps only the per-iteration stores + logs and the final policy (~31G per run instead of ~95G + 0.3G).
Writes logs/queue/jobs_tel_protocol.jsonl.
"""

import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = os.environ["XPER_HOME"] + "/models/xper/8b-commonsft/final"
SRC = "results/src/4b-inst.shard*.db"
ARMS = {
    "mismatch": f"INJECT='{SRC}' INJECT_MODE=fail INJECT_CONTROL=mismatch ",
    "shuffle": f"INJECT='{SRC}' INJECT_MODE=fail INJECT_CONTROL=shuffle ",
}
RUNS = [("mismatch", 601), ("shuffle", 601), ("mismatch", 602), ("shuffle", 602)]


def main():
    out = ROOT / "logs/queue/jobs_tel_protocol.jsonl"
    with open(out, "w") as f:
        for arm, seed in RUNS:
            tag = f"tp-{arm}"
            # Compile caches need local rename/file-handle semantics. DOP-FUSE
            # raised ESTALE when concurrent vLLM engines compiled under kh/.cache.
            cache = f"/tmp/xper-tel-protocol-cache/{tag}-s{seed}"
            cmd = (f"cd {ROOT} && VLLM_CACHE_ROOT={cache}/vllm TRITON_CACHE_DIR={cache}/triton "
                   f"TORCHINDUCTOR_CACHE_DIR={cache}/inductor CUDA_CACHE_PATH={cache}/cuda "
                   f"GPUS=$GPUS START={START} TAG={tag} THINK='' SEED={seed} TASKS=16 GROUP=8 "
                   f"BUDGET=44414 SPLIT=train_src_solved LEAN=1 {ARMS[arm]}bash scripts/run_grpo.sh")
            f.write(json.dumps({"name": f"{tag}-s{seed}", "gpus": 4, "cmd": cmd}) + "\n")
    print(f"wrote {len(RUNS)} jobs to {out}")


if __name__ == "__main__":
    main()
