"""Write the step 07 training and validation jobs for the reported arms."""
import os
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KH = Path(os.environ.get("XPER_HOME", ROOT.parent))
PY = Path(os.environ.get("XPER_PYTHON", KH / "miniconda3/envs/xper/bin/python"))
O = KH / "models/xper"
HF = KH / "models/hf"
BASE = ROOT / "logs/queue/spec_20261117"
TEL_WEAK = O / "8b-commonsft/final"
MATH_WEAK = O / "spec-math-start"           # Qwen3-8B-Base, eos delimiter init (rebuilt, same as cohort)
TEL_SRC = "results/src/4b-inst.shard*.db"
def env(name, **kw):
    cache = f"/tmp/xper-spec-cache/{name}"
    base = dict(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                VLLM_CACHE_ROOT=f"{cache}/vllm", TRITON_CACHE_DIR=f"{cache}/triton",
                TORCHINDUCTOR_CACHE_DIR=f"{cache}/inductor", CUDA_CACHE_PATH=f"{cache}/cuda",
                OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", XPER_STRICT_MATH="1")
    base.update(kw)
    return " ".join(f"{k}={shlex.quote(str(v))}" for k, v in base.items())


def grpo(name, start, seed, budget=44414, extra=None, script="run_grpo.sh"):
    kw = dict(START=start, TAG=name, THINK="", SEED=seed, TASKS=16, GROUP=8, LR="1e-6", MAX_IT=400,
              SPLIT="train_src_solved", BUDGET=budget, LEAN=1, CLEAN_FINAL=1,
              INJECT="", INJECT_MODE="fail", INJECT_LIMIT=-1, INJECT_CONTROL="none", SRC_TASK_FRAC="1.0")
    kw.update(extra or {})
    return f"cd {ROOT} && {env(f'{name}-s{seed}', **kw)} GPUS=\"$GPUS\" bash scripts/{script}"


def probe(name, model, env_args, think=""):
    return (f"cd {ROOT} && export {env(name)} && CUDA_VISIBLE_DEVICES=${{GPUS%% *}} {PY} scripts/rollout.py --chatml "
            f"{think} --model {model} --model-id {name} --split val_probe --n 4 --seed 20261117 {env_args} "
            f"--out results/probe/{name}.db > results/probe/{name}.log 2>&1 && grep -h '^{{\"model' results/probe/{name}.log")


def jobs():
    J = []
    def add(name, gpus, cmd, after=()):
        J.append(dict(name=name, gpus=gpus, cmd=cmd, after=list(after)))
    # Order matters: gpu_queue starts jobs in file order.
    # E2 (P0): no injection at total-cost budget
    # Keep the reported tags so that result files can be matched to the paper.
    for seed in (601, 602, 603):
        tag = "e2-tel-r3" if seed == 601 else "e2-tel"
        name = "e2-tel-s601-r3" if seed == 601 else f"e2-tel-s{seed}"
        driver = "run_grpo_p1.sh" if seed == 601 else "run_grpo.sh"
        add(name, 4, grpo(tag, TEL_WEAK, seed, budget=88828, script=driver))
    # E3 (P0): probes, 1 GPU each
    tel_env, math_env = "--env-workers 24", "--env math --max-steps 12 --max-tokens 2048 --env-workers 48"
    add("math-start", 0, f"cd {ROOT} && [ -f {MATH_WEAK}/xper_init.json ] || {env('math-start')} {PY} "
        f"scripts/prep_init.py --base {HF}/Qwen3-8B-Base --out {MATH_WEAK} --mode eos")
    add("e3-tel-weak", 1, probe("e3-tel-weak", TEL_WEAK, tel_env))
    add("e3-tel-strong", 1, probe("e3-tel-strong", HF / "Qwen3-8B", tel_env, "--no-think"))
    add("e3-tel-src", 1, probe("e3-tel-src", HF / "Qwen3-4B-Instruct-2507", tel_env))
    add("e3-math-weak", 1, probe("e3-math-weak", MATH_WEAK, math_env), ["math-start"])
    add("e3-math-strong", 1, probe("e3-math-strong", HF / "Qwen3-8B", math_env, "--no-think"))
    add("e3-math-src", 1, probe("e3-math-src", HF / "Qwen3-4B-Instruct-2507", math_env))
    # E4 / E5 (P1): weak telecom start, same pool, seeds and budget as the injection cohort
    p1 = dict(
        e4a=dict(RELIFT_SRC=TEL_SRC, RELIFT_SIZE=64, RELIFT_CAP=100, RELIFT_LR="1e-6"),
        e4b=dict(INJECT=TEL_SRC, INJECT_MODE="all", GROUP=7, NO_STD=1, OFF_SHAPING="0.1"),
        e5=dict(INJECT_SELF=1, OWN_LIMIT=100),
    )
    repeated = {"e4a": {602}, "e4b": {601,603}, "e5": {602,603}}
    for arm, extra in p1.items():
        for seed in (601,602,603):
            suffix = "-r2" if seed in repeated[arm] else ""
            name = f"{arm}-tel-s{seed}{suffix}"
            tag = f"{arm}-tel{suffix}"
            add(name, 4, grpo(tag, TEL_WEAK, seed, extra=extra, script="run_grpo_p1.sh"))
    return J


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    (ROOT / "results/probe").mkdir(parents=True, exist_ok=True)
    out = BASE / "jobs.jsonl"
    J = jobs()
    tmp = out.with_suffix(".tmp")  # atomic: the running queues reread jobs.jsonl every 30 s
    tmp.write_text("".join(json.dumps(j) + "\n" for j in J))
    tmp.replace(out)
    print(f"wrote {len(J)} jobs to {out}")


if __name__ == "__main__":
    main()
