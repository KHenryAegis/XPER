"""Finite telecom dose queue on host A. New outputs only; verified evals precede cleanup.

run: create an immutable job manifest and own queue, watch through completion.
worker: a single fresh run, verify budget/eval, then remove only its new models/cache.
status: read-only progress, including independent SQLite checks for finished jobs.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
KH = Path(os.environ.get("XPER_HOME", str(ROOT.parent)))
_LEGACY_PY = KH / "miniconda3/envs/xper/bin/python"
PY = Path(os.environ.get("XPER_PYTHON", str(_LEGACY_PY if _LEGACY_PY.exists() else sys.executable)))
MODELS = Path(os.environ.get("XPER_MODEL_DIR", str(KH / "models/xper")))
START = MODELS / "8b-commonsft/final"
BATCH = "tel_dose_20261112"
BASE = ROOT / "logs/queue" / BATCH
REPORT = ROOT / "results" / f"{BATCH}_report.json"
CACHE = Path("/tmp/xper-tel-dose-20261112")
ITER = re.compile(r"^\s*it (\d+): (\{.*\}) \| cumulative env calls (\d+)$", re.M)
RUNS = [(dose, seed) for seed in (601, 602) for dose in (5, 20, 100, 0)]


def name_for(dose, seed):
    if dose not in (0, 5, 20, 100) or seed not in (601, 602):
        raise ValueError("unregistered experiment")
    return f"td{dose}-s{seed}"


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def runtime_env():
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR="/tmp", MAX_IT="400",
               XDG_CACHE_HOME=str(KH / ".cache"), XDG_CONFIG_HOME=str(KH / ".config"),
               HF_HOME=str(KH / ".cache/huggingface"), TORCH_HOME=str(KH / ".cache/torch"),
               VLLM_CONFIG_ROOT=str(KH / ".config/vllm"),
               NUMBA_CACHE_DIR=str(KH / ".cache/numba"), MPLCONFIGDIR=str(KH / ".cache/matplotlib"),
               OUTLINES_CACHE_DIR=str(KH / ".cache/outlines"),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    return env


def read_iterations(name):
    log = BASE / f"{name}.log"
    return [dict(iteration=int(it), env_calls=int(calls), **json.loads(stats))
            for it, stats, calls in ITER.findall(log.read_text() if log.exists() else "")]


def first_round_audit(dose, seed):
    """Fail-only injection excludes groups already solved by a target sample."""
    groups = collections.defaultdict(list)
    for path in sorted((ROOT / "results" / f"td{dose}.s{seed}").glob("it0.shard*.db")):
        with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as db:
            for task, reward in db.execute("SELECT task_id,reward FROM trajectories"):
                groups[task].append(reward)
    if len(groups) != 16 or any(len(v) != 8 for v in groups.values()):
        raise ValueError("first-round DB is not 16 tasks x 8 samples")
    solved_by_source = set()
    for path in sorted((ROOT / "results/src").glob("4b-inst.shard*.db")):
        with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as db:
            solved_by_source.update(t for (t,) in db.execute(
                "SELECT DISTINCT task_id FROM trajectories WHERE reward >= 1"))
    eligible = sum(task in solved_by_source and not any(r > 0 for r in rewards)
                   for task, rewards in groups.items())
    return dict(eligible_groups=eligible, expected_injected=min(dose, eligible),
                target_successes=sum(r >= 1 for rewards in groups.values() for r in rewards))


def verify(dose, seed):
    name = name_for(dose, seed)
    iterations = read_iterations(name)
    tag = f"td{dose}"
    stats = json.loads((ROOT / "results" / f"{tag}.s{seed}" / "stats.json").read_text())
    if not iterations or stats["env_calls"] < 44414 or stats["env_calls"] != iterations[-1]["env_calls"]:
        raise ValueError("training budget/log mismatch")
    if any(r["episodes"] != 128 or r["groups"] != 16 or r["injected_own"] for r in iterations):
        raise ValueError("training recipe mismatch")
    total = sum(r["injected"] for r in iterations)
    if not 0 <= total <= dose:
        raise ValueError("cumulative source dose exceeded")
    log = (ROOT / "results/eval" / f"{name}.log").read_text()
    metric = [json.loads(line) for line in log.splitlines() if line.startswith('{"model"')][-1]
    path = ROOT / "results/eval" / f"{name}.db"
    with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as db:
        rows = db.execute("SELECT task_id,reward FROM trajectories").fetchall()
    by_task = collections.defaultdict(list)
    for task, reward in rows:
        by_task[task].append(reward >= 1)
    if len(rows) != 320 or len(by_task) != 40 or any(len(v) != 8 for v in by_task.values()):
        raise ValueError("evaluation must contain 40 tasks x 8 samples")
    p1 = sum(sum(v) for v in by_task.values()) / 320
    p8 = sum(any(v) for v in by_task.values()) / 40
    if (metric["model"] != name or metric["split"] != "test" or metric["n"] != 8
            or metric["tasks"] != 40 or abs(metric["pass@1"] - p1) > .0001
            or abs(metric["pass@8"] - p8) > .0001):
        raise ValueError("evaluation DB/log mismatch")
    return dict(verified=stamp(), requested_dose=dose, injected_total=total,
                dose_fully_used=(total == dose), training=stats,
                takeoff_iteration=next((r["iteration"] for r in iterations if r["mean_reward"] >= .1), None),
                eval=dict(metric, db_pass1=p1, db_pass8=p8))


def worker(dose, seed):
    name = name_for(dose, seed)
    manifest = json.loads((BASE / "launch.json").read_text())
    if manifest["host"] != socket.gethostname() or manifest["batch"] != BATCH:
        raise RuntimeError("manifest/host mismatch")
    tag = f"td{dose}"
    result = ROOT / "results" / f"{tag}.s{seed}"
    own_models = [MODELS / f"{name}-{suffix}" for suffix in ("0", "1", "final")]
    own_cache = CACHE / name
    fresh = [result, *own_models, own_cache, ROOT / "results/eval" / f"{name}.db",
             ROOT / "results/eval" / f"{name}.log", BASE / f"{name}.verified.json"]
    if any(p.exists() or p.is_symlink() for p in fresh):
        raise RuntimeError("refusing to overwrite existing run artifacts")
    result.mkdir()
    atomic_json(result / "owner.json", dict(batch=BATCH, name=name, pid=os.getpid(), started=stamp()))
    env = runtime_env()
    # Clear optional controls so a parent shell cannot silently alter this recipe.
    for key in ("INJECT_SELF", "PREFIX_SRC", "HINT_SRC", "ENVARGS", "EVALARGS"):
        env.pop(key, None)
    env.update(START=str(START), TAG=tag, THINK="", SEED=str(seed), TASKS="16", GROUP="8",
               BUDGET="44414", SPLIT="train_src_solved", LEAN="1", LR="1e-6",
               INJECT="results/src/4b-inst.shard*.db" if dose else "",
               INJECT_MODE="fail", INJECT_CONTROL="none", INJECT_LIMIT=str(dose), SRC_TASK_FRAC="1.0",
               VLLM_CACHE_ROOT=str(own_cache / "vllm"), TRITON_CACHE_DIR=str(own_cache / "triton"),
               TORCHINDUCTOR_CACHE_DIR=str(own_cache / "inductor"), CUDA_CACHE_PATH=str(own_cache / "cuda"))
    atomic_json(result / "recipe.json", {k: env[k] for k in (
        "START", "TAG", "THINK", "SEED", "TASKS", "GROUP", "BUDGET", "SPLIT", "LEAN", "LR",
        "INJECT", "INJECT_MODE", "INJECT_CONTROL", "INJECT_LIMIT", "MAX_IT", "SRC_TASK_FRAC", "GPUS")})
    subprocess.run(["bash", "scripts/run_grpo.sh"], cwd=ROOT, env=env, check=True)
    checked = verify(dose, seed)
    # Write the verified record *before* removing any of this worker's artifacts.
    checked["cleanup"] = "pending"
    receipt = BASE / f"{name}.verified.json"
    atomic_json(receipt, checked)
    owner = json.loads((result / "owner.json").read_text())
    if owner["pid"] != os.getpid() or owner["batch"] != BATCH or owner["name"] != name:
        raise RuntimeError("cleanup ownership mismatch")
    removed = []
    for p in [*own_models, own_cache]:
        if p.is_symlink():
            raise RuntimeError(f"refusing symlink cleanup: {p}")
        expected_parent = MODELS if p in own_models else CACHE
        if p.parent.resolve() != expected_parent.resolve() or p.resolve() == START.resolve():
            raise RuntimeError("cleanup scope mismatch")
        if p.exists():
            shutil.rmtree(p)
            removed.append(str(p))
    checked.update(cleanup="done", removed=removed)
    atomic_json(receipt, checked)
    print(json.dumps(dict(name=name, **checked), ensure_ascii=False), flush=True)


def snapshot():
    statepath = BASE / "state.json"
    state = json.loads(statepath.read_text()) if statepath.exists() else {}
    jobs = {}
    for dose, seed in RUNS:
        name = name_for(dose, seed)
        rec = dict(state.get(name, {"status": "pending"}), requested_dose=dose, errors=[])
        iterations = read_iterations(name)
        rec.update(iterations=iterations, injected_total=sum(r["injected"] for r in iterations),
                   takeoff_iteration=next((r["iteration"] for r in iterations if r["mean_reward"] >= .1), None))
        if rec["injected_total"] > dose:
            rec["errors"].append("source dose exceeded")
        if iterations and dose:
            try:
                audit = first_round_audit(dose, seed)
                rec["first_round_audit"] = audit
                if iterations[0]["injected"] != audit["expected_injected"]:
                    rec["errors"].append("first-round injection does not match eligible all-fail groups")
            except (OSError, ValueError, sqlite3.Error) as exc:
                rec["errors"].append(f"first-round validation: {exc}")
        if rec["status"] == "ok":
            try:
                rec.update(verify(dose, seed))
                receipt = json.loads((BASE / f"{name}.verified.json").read_text())
                rec["cleanup"] = receipt["cleanup"]
                if receipt["cleanup"] != "done":
                    rec["errors"].append("worker cleanup incomplete")
            except (OSError, ValueError, KeyError, IndexError, sqlite3.Error) as exc:
                rec["errors"].append(str(exc))
        jobs[name] = rec
    terminal = all(r["status"] == "ok" or r["status"].startswith(("fail", "skipped")) for r in jobs.values())
    return dict(updated=stamp(), complete=all(r["status"] == "ok" and not r["errors"] for r in jobs.values()),
                terminal=terminal, jobs=jobs)


def run():
    if socket.gethostname().split(".")[0] != os.environ.get("XPER_HOST", socket.gethostname().split(".")[0]):
        raise RuntimeError("run only on the host named by XPER_HOST")
    if not START.is_dir() or not list((ROOT / "results/src").glob("4b-inst.shard*.db")):
        raise RuntimeError("missing local start model/source stores")
    BASE.mkdir(parents=True, exist_ok=True)
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
        "scripts/tel_dose.py", "scripts/grpo_advantages.py", "scripts/run_grpo.sh", "scripts/gpu_queue.py")}
    manifest = dict(batch=BATCH, host=socket.gethostname(), pid=os.getpid(), start=stamp(),
                    proc_start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
                    runs=RUNS, hashes=hashes, model_cleanup="after verified eval only")
    # Never re-use a previous queue or reset its claims/state: run_grpo starts at it0.
    with (BASE / "launch.json").open("x") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    jobs = []
    for dose, seed in RUNS:
        cmd = shlex.join([str(PY), "-B", "scripts/tel_dose.py", "worker", "--dose", str(dose), "--seed", str(seed)])
        jobs.append(dict(name=name_for(dose, seed), gpus=4, cmd=cmd))
    with (BASE / "jobs.jsonl").open("x") as f:
        f.write("".join(json.dumps(j) + "\n" for j in jobs))
    queue_args = [str(PY), "-B", "scripts/gpu_queue.py", "--jobs", str(BASE / "jobs.jsonl"),
                  "--state", str(BASE / "state.json"), "--claim-dir", str(BASE / "claims"),
                  "--host-label", "74", "--poll", "30"]
    with (BASE / "queue.log").open("a") as log:
        queue = subprocess.Popen(queue_args, cwd=ROOT, env=runtime_env(), stdout=log, stderr=subprocess.STDOUT)
        manifest["queue_pid"] = queue.pid
        atomic_json(BASE / "launch.json", manifest)
        previous = None
        try:
            while True:
                try:
                    report = snapshot()
                except json.JSONDecodeError:
                    time.sleep(5)
                    continue
                atomic_json(REPORT, report)
                brief = {name: dict(status=r["status"], last=r["iterations"][-1] if r["iterations"] else None,
                                    injected_total=r["injected_total"], errors=r["errors"], eval=r.get("eval"))
                         for name, r in report["jobs"].items()}
                current = json.dumps(brief, ensure_ascii=False)
                if current != previous:
                    print(stamp(), current, flush=True)
                    previous = current
                if report["terminal"]:
                    # This exact Popen child was started above; all jobs have exited.
                    queue.terminate()
                    queue.wait(timeout=30)
                    return 0 if report["complete"] else 1
                if queue.poll() is not None:
                    raise RuntimeError(f"our queue exited unexpectedly: {queue.returncode}")
                time.sleep(30)
        except BaseException:
            # Leave active workers untouched for inspection; record their authoritative pgids.
            print("Controller stopped; inspect state.json before any restart. Workers were not signalled.", flush=True)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker", "status"))
    parser.add_argument("--dose", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.mode == "run":
        sys.exit(run())
    elif args.mode == "worker":
        worker(args.dose, args.seed)
    else:
        print(json.dumps(snapshot(), ensure_ascii=False, indent=2))
