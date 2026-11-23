"""Mechanism controls + math weak-start replication; one active queue on host A or host B."""
import argparse
import collections
import fcntl
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tel_dose import ROOT, KH, PY, MODELS, ITER, atomic_json, runtime_env, stamp

BATCH = "strengthen_20261113"
BASE = ROOT / "logs/queue" / BATCH
REPORT = ROOT / "results" / f"{BATCH}_report.json"
CACHE = Path("/tmp/xper-strengthen-20261113")
MATH_START = MODELS / "sc-math-start-20261113"
BASE_MODEL = KH / "models/hf/Qwen3-8B-Base"
SPECS = {"sc-math-init": dict(kind="prep", gpus=0),
         "sc-math-headroom": dict(kind="headroom", gpus=1, after=["sc-math-init"])}
for seed in (601, 602):
    for env, arm in (("tel", "obs_mask"), ("tel", "schema_random"), ("math", "real"), ("math", "none")):
        name = f"sc-{env}-{arm}-s{seed}"
        SPECS[name] = dict(kind="train", env=env, arm=arm, seed=seed, gpus=4,
                           after=["sc-math-headroom"] if env == "math" else [])


def fresh(paths):
    if any(p.exists() or p.is_symlink() for p in paths):
        raise RuntimeError("refusing to overwrite existing experiment artifacts")


def iterations(name):
    paths = [BASE / f"{name}.log"]
    recoveries = [p for p in BASE.glob(f"recover*-{name}.log") if re.match(r"^recover\d+-", p.name)]
    paths += sorted(recoveries, key=lambda p: int(re.match(r"^recover(\d+)-", p.name)[1]))
    rounds = {}
    for p in paths:
        for i, payload, calls in ITER.findall(p.read_text() if p.exists() else ""):
            row = dict(iteration=int(i), env_calls=int(calls), **json.loads(payload))
            if int(i) in rounds and rounds[int(i)] != row:
                raise ValueError(f"conflicting records for {name} iteration {i}; refusing to double count")
            rounds[int(i)] = row
    return [rounds[i] for i in sorted(rounds)]


def evaluate_record(name, count, samples):
    p = ROOT / "results/eval" / f"{name}.db"
    with sqlite3.connect(f"file:{p}?mode=ro&immutable=1", uri=True) as db:
        rows = db.execute("SELECT task_id,reward FROM trajectories").fetchall()
    groups = collections.defaultdict(list)
    for task, reward in rows:
        groups[task].append(reward >= 1)
    if len(groups) != count or len(rows) != count * samples or any(len(v) != samples for v in groups.values()):
        raise ValueError("evaluation task/sample count mismatch")
    logs = (ROOT / "results/eval" / f"{name}.log").read_text().splitlines()
    metric = [json.loads(l) for l in logs if l.startswith('{"model"')][-1]
    p1, pk = sum(sum(v) for v in groups.values()) / len(rows), sum(any(v) for v in groups.values()) / count
    if (metric["model"] != name or metric["split"] != "test" or metric["tasks"] != count
            or metric["n"] != samples or abs(metric["pass@1"] - p1) > .0001
            or abs(metric[f"pass@{samples}"] - pk) > .0001):
        raise ValueError("evaluation DB/log mismatch")
    return dict(metric, db_pass1=p1, db_passk=pk)


def verify(name):
    spec = SPECS[name]
    if spec["kind"] == "prep":
        index = json.loads((MATH_START / "model.safetensors.index.json").read_text())
        if not all((MATH_START / f).is_file() for f in set(index["weight_map"].values())):
            raise ValueError("incomplete initialized math checkpoint")
        return dict(verified=stamp(), initializer=json.loads((MATH_START / "xper_init.json").read_text()))
    if spec["kind"] == "headroom":
        return dict(verified=stamp(), eval=evaluate_record(name, 300, 4))
    tag, seed = name.rsplit("-s", 1)
    result = ROOT / "results" / f"{tag}.s{seed}"
    stats = json.loads((result / "stats.json").read_text())
    history = iterations(name)
    budget = 10403 if spec["env"] == "math" else 44414
    if not history or stats["env_calls"] < budget or stats["env_calls"] != history[-1]["env_calls"]:
        raise ValueError("training budget/log mismatch")
    if any(r["episodes"] != 128 or r["groups"] != 16 or r["injected_own"] for r in history):
        raise ValueError("training recipe mismatch")
    dose = sum(r["injected"] for r in history)
    if dose > 100 or (spec["arm"] == "none" and dose):
        raise ValueError("injection cap exceeded")
    # These are dataset token totals, not a claim of exact FLOPs matching.
    updates = []
    for p in result.glob("it*.train.log"):
        for line in p.read_text().splitlines():
            if line.startswith('{"saved"'):
                updates.append(json.loads(line))
    return dict(verified=stamp(), training=stats, injected_total=dose,
                takeoff_iteration=next((r["iteration"] for r in history if r["mean_reward"] >= .1), None),
                updates=len(updates), dataset_supervised_tokens=sum(r["supervised_tokens"] for r in updates),
                eval=evaluate_record(name, 300 if spec["env"] == "math" else 40, 4 if spec["env"] == "math" else 8))


def finalize_verified_run(name):
    """Finalize a completed eval whose worker failed only in post-run validation."""
    checked = verify(name)  # must pass before touching any checkpoint/cache
    manifest = json.loads((BASE / "launch.json").read_text())
    if manifest["active_deployment"]["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise RuntimeError("finalize on the active host only")
    spec = SPECS[name]
    if spec["kind"] != "train":
        raise ValueError("only training runs need this reconciliation")
    mappings = json.loads((BASE / "recoveries.json").read_text())
    state = json.loads((BASE / "state.json").read_text())
    key = mappings.get(name, {}).get("job", name)
    if not state.get(key, {}).get("status", "").startswith("fail"):
        raise RuntimeError("refusing to finalize an active/pending worker")
    tag, seed = name.rsplit("-s", 1)
    result = ROOT / "results" / f"{tag}.s{seed}"
    owner = json.loads((result / "owner.json").read_text())
    if owner["batch"] != BATCH or owner["name"] != name:
        raise RuntimeError("finalization ownership mismatch")
    receipt_path = BASE / f"{name}.verified.json"
    prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if prior and prior["cleanup"] == "done":
        return prior
    final = MODELS / f"{name}-final"
    if prior is None:
        from recover_strengthen import complete_model
        if not complete_model(final / "final"):
            raise ValueError("completed model checkpoint is not intact")
        checked["checkpoint_metadata"] = json.loads((final / "data_stats.json").read_text())
    else:
        checked["checkpoint_metadata"] = prior.get("checkpoint_metadata")
    checked.update(cleanup="pending", postcheck_reconciled=True,
                   reconciliation="Merged original and recovery logs; completed training/eval were not rerun")
    atomic_json(receipt_path, checked)
    paths = [MODELS / f"{name}-{x}" for x in ("0", "1", "final")] + [CACHE / name]
    removed = list(prior.get("removed", [])) if prior else []
    for path in paths:
        if path.is_symlink() or path.parent.resolve() not in (MODELS.resolve(), CACHE.resolve()):
            raise RuntimeError("finalization cleanup scope mismatch")
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    checked.update(cleanup="done", removed=removed)
    atomic_json(receipt_path, checked)
    return checked


def worker(name, recovery_plan=None):
    spec = SPECS[name]
    manifest = json.loads((BASE / "launch.json").read_text())
    if manifest["host"] != socket.gethostname() or manifest["specs"] != SPECS:
        raise RuntimeError("host/manifest mismatch")
    active = manifest.get("active_deployment", {})
    if active and active["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise RuntimeError("worker is running on the previous deployment's machine")
    env = runtime_env()
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", XPER_STRICT_MATH="1")
    for key in ("INJECT_SELF", "PREFIX_SRC", "HINT_SRC", "ENVARGS", "EVALARGS", "THINK", "SRC_TASK_FRAC"):
        env.pop(key, None)
    recovery = json.loads(recovery_plan.read_text()) if recovery_plan else None
    if recovery and recovery["name"] != name:
        raise RuntimeError("recovery plan/task mismatch")
    if recovery:
        index = Path(recovery["policy"]) / "model.safetensors.index.json"
        if hashlib.sha256(index.read_bytes()).hexdigest() != recovery["policy_index_sha256"]:
            raise RuntimeError("recovery policy changed after planning")
    cache = CACHE / name
    fresh(([cache] if not recovery else []) + [BASE / f"{name}.verified.json"])
    env.update(VLLM_CACHE_ROOT=str(cache / "vllm"), TRITON_CACHE_DIR=str(cache / "triton"),
               TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), CUDA_CACHE_PATH=str(cache / "cuda"))
    cleanup = [cache]
    if spec["kind"] == "prep":
        fresh([MATH_START])
        subprocess.run([str(PY), "-B", "scripts/prep_init.py", "--base", str(BASE_MODEL),
                        "--out", str(MATH_START), "--mode", "eos"], cwd=ROOT, env=env, check=True)
        atomic_json(MATH_START / "strengthen_owner.json", dict(batch=BATCH, pid=os.getpid(), created=stamp()))
    elif spec["kind"] == "headroom":
        fresh([ROOT / "results/eval" / f"{name}.db", ROOT / "results/eval" / f"{name}.log"])
        first = env["GPUS"].split()[0]
        env["CUDA_VISIBLE_DEVICES"] = first
        with (ROOT / "results/eval" / f"{name}.log").open("x") as log:
            subprocess.run([str(PY), "-B", "scripts/rollout.py", "--chatml", "--env", "math",
                            "--model", str(MATH_START), "--model-id", name, "--split", "test", "--n", "4",
                            "--max-steps", "12", "--max-tokens", "2048", "--env-workers", "48",
                            "--out", f"results/eval/{name}.db"], cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    else:
        tag, seed = name.rsplit("-s", 1)
        result = ROOT / "results" / f"{tag}.s{seed}"
        own_models = [MODELS / f"{name}-{s}" for s in ("0", "1", "final")]
        if recovery:
            owner = json.loads((result / "owner.json").read_text())
            if owner["batch"] != BATCH or owner["name"] != name:
                raise RuntimeError("recovery ownership mismatch")
            env["XPER_RECOVERY_ATTEMPT"] = str(recovery["attempt"])
            fresh([ROOT / "results/eval" / f"{name}.db", ROOT / "results/eval" / f"{name}.log"])
        else:
            fresh([result, *own_models, ROOT / "results/eval" / f"{name}.db", ROOT / "results/eval" / f"{name}.log"])
            result.mkdir()
            atomic_json(result / "owner.json", dict(batch=BATCH, name=name, pid=os.getpid(), created=stamp()))
        math = spec["env"] == "math"
        env.update(START=str(MATH_START if math else MODELS / "8b-commonsft/final"), TAG=tag,
                   THINK="", SEED=seed, TASKS="16", GROUP="8", MAX_IT="400", LEAN="1", LR="1e-6",
                   SPLIT="train_src_solved", BUDGET="10403" if math else "44414", SRC_TASK_FRAC="1.0",
                   INJECT="" if spec["arm"] == "none" else "results/math/src/4b-inst.shard*.db" if math else "results/src/4b-inst.shard*.db",
                   INJECT_MODE="fail", INJECT_LIMIT=str(spec.get("injection_cap", 100)),
                   INJECT_CONTROL=spec["arm"] if spec["arm"] in ("mismatch", "shuffle", "obs_mask", "schema_random") else "none")
        if math:
            env.update(ENVARGS="--env math --max-steps 12 --max-tokens 1024 --env-workers 48",
                       EVALARGS="--n 4 --max-tokens 2048")
        driver = (ROOT / "scripts/run_grpo.sh").read_text()
        anchor = "    policy=$out/final\n"
        assert driver.count(anchor) == 1
        driver = driver.replace('cd "$(dirname "$0")/.."', f"cd {shlex.quote(str(ROOT))}", 1)
        # The previous policy is deleted only after torchrun has exited and
        # the replacement policy was saved. Restrict removal to this job's two slots.
        driver = driver.replace(anchor, """    previous_policy=$policy
    policy=$out/final
    if [ -n "$LEAN" ] && { [ "$previous_policy" = "$O/$TAG-s$SEED-0/final" ] || [ "$previous_policy" = "$O/$TAG-s$SEED-1/final" ]; }; then
      rm -rf -- "$(dirname "$previous_policy")"
    fi
""", 1)
        from recover_strengthen import guard_driver
        driver = guard_driver(driver, recovery, recovery_plan)
        suffix = f".recovery{recovery['attempt']}" if recovery else ""
        path = result / f"driver{suffix}.sh"
        path.write_text(driver)
        atomic_json(result / f"recipe{suffix}.json", dict(spec=spec, recovery=recovery, driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),
                    environment={k: v for k, v in env.items() if k in (
                        "START", "TAG", "SEED", "TASKS", "GROUP", "MAX_IT", "LEAN", "LR", "SPLIT", "BUDGET",
                        "INJECT", "INJECT_MODE", "INJECT_LIMIT", "INJECT_CONTROL", "ENVARGS", "EVALARGS", "GPUS", "XPER_STRICT_MATH")}))
        subprocess.run(["bash", str(path)], cwd=ROOT, env=env, check=True)
        cleanup += own_models
    checked = verify(name)
    checked["cleanup"] = "pending"
    atomic_json(BASE / f"{name}.verified.json", checked)
    removed = []
    for path in cleanup:
        if path.is_symlink() or path.parent.resolve() not in (MODELS.resolve(), CACHE.resolve()):
            raise RuntimeError("cleanup scope mismatch")
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    checked.update(cleanup="done", removed=removed)
    atomic_json(BASE / f"{name}.verified.json", checked)
    print(json.dumps(dict(name=name, **checked), ensure_ascii=False), flush=True)


def snapshot():
    p = BASE / "state.json"
    state = json.loads(p.read_text()) if p.exists() else {}
    recoveries_path = BASE / "recoveries.json"
    recoveries = json.loads(recoveries_path.read_text()) if recoveries_path.exists() else {}
    jobs = {}
    for name, spec in SPECS.items():
        rec = dict(state.get(name, {"status": "pending"}), spec=spec, errors=[])
        recovery = recoveries.get(name)
        if recovery and "job" in recovery:
            rec = dict(state.get(recovery["job"], {"status": "pending"}), spec=spec, errors=[],
                       recovery_job=recovery["job"], initial_failure=recovery["initial_failure"],
                       recovery_attempt=recovery["attempt"])
            if recovery.get("verified_completion"):
                rec["executor_status"] = rec["status"]
                rec["status"] = "ok"
                rec["postcheck_reconciled"] = True
        if recovery and recovery.get("planning_error"):
            rec["errors"].append(recovery["planning_error"])
        if rec["status"].startswith("fail"):
            rec["errors"].append("job exited unsuccessfully; inspect the job/stage logs")
        history = iterations(name)
        if history:
            rec.update(last=history[-1], injected_total=sum(r["injected"] for r in history),
                       takeoff_iteration=next((r["iteration"] for r in history if r["mean_reward"] >= .1), None))
        if rec["status"] == "ok":
            try:
                receipt = json.loads((BASE / f"{name}.verified.json").read_text())
                rec.update(receipt)
                # Prep may have been cleaned at finalization; its saved audit
                # is enough then. Evals are independently read again each poll.
                if spec["kind"] != "prep":
                    rec.update(verify(name))
                if receipt["cleanup"] != "done":
                    rec["errors"].append("cleanup incomplete")
            except (OSError, ValueError, KeyError, IndexError, sqlite3.Error) as exc:
                rec["errors"].append(str(exc))
        jobs[name] = rec
    terminal = all(r["status"] == "ok" or r["status"].startswith(("fail", "skipped")) for r in jobs.values())
    return dict(updated=stamp(), terminal=terminal,
                complete=all(r["status"] == "ok" and not r["errors"] for r in jobs.values()), jobs=jobs)


def run(host_label="74", resume=False):
    if socket.gethostname().split(".")[0] != os.environ.get("XPER_HOST", socket.gethostname().split(".")[0]):
        raise RuntimeError("unexpected GPU host (see XPER_HOST)")
    BASE.mkdir(parents=True, exist_ok=True)
    lock_handle = (BASE / "controller.lock").open("a")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    scripts = ["scripts/strengthen.py", "scripts/run_grpo.sh", "scripts/grpo_advantages.py", "scripts/bc_train.py",
               "scripts/prep_init.py", "xper/injection_controls.py", "xper/mathenv.py", "xper/math_seccomp.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in scripts}
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if resume:
        migration = json.loads((BASE / "migration_to_152.json").read_text())
        manifest = json.loads((BASE / "launch.json").read_text())
        state = json.loads((BASE / "state.json").read_text())
        if (host_label != "152" or migration["status"] != "old_idle_queue_stopped"
                or migration["from_boot_id"] == boot_id or manifest["specs"] != SPECS
                or any(v["status"] != "ok" for v in state.values())):
            raise RuntimeError("migration preconditions not met; no state reset or rerun attempted")
        verify("sc-math-init")  # completed shared checkpoint must be intact
        manifest.setdefault("deployments", []).append(dict(
            host=manifest["host"], pid=manifest["pid"], queue_pid=manifest["queue_pid"],
            host_label="74", stopped=migration["when"], hashes=manifest["hashes"]))
        manifest.update(host=socket.gethostname(), pid=os.getpid(), hashes=hashes)
    else:
        fresh([BASE / "launch.json", BASE / "jobs.jsonl", BASE / "state.json", MATH_START])
        manifest = dict(batch=BATCH, host=socket.gethostname(), pid=os.getpid(), started=stamp(), specs=SPECS,
                        hashes=hashes)
        with (BASE / "jobs.jsonl").open("x") as f:
            for name, spec in SPECS.items():
                f.write(json.dumps(dict(name=name, gpus=spec["gpus"], after=spec.get("after", []),
                    cmd=shlex.join([str(PY), "-B", "scripts/strengthen.py", "worker", "--name", name]))) + "\n")
    manifest["active_deployment"] = dict(host_label=host_label, boot_id=boot_id, started=stamp(),
                                         controller_pid=os.getpid(), host=socket.gethostname())
    atomic_json(BASE / "launch.json", manifest)
    with (BASE / "queue.log").open("a") as log:
        queue = subprocess.Popen([str(PY), "-B", "scripts/gpu_queue.py", "--jobs", str(BASE / "jobs.jsonl"),
                    "--state", str(BASE / "state.json"), "--claim-dir", str(BASE / "claims"),
                    "--host-label", host_label, "--poll", "30"], cwd=ROOT, env=runtime_env(), stdout=log, stderr=subprocess.STDOUT)
        manifest["queue_pid"] = queue.pid
        manifest["active_deployment"]["queue_pid"] = queue.pid
        atomic_json(BASE / "launch.json", manifest)
        previous = None
        while True:
            try:
                report = snapshot()
            except json.JSONDecodeError:
                time.sleep(5)
                continue
            atomic_json(REPORT, report)
            brief = {n: {k: v for k, v in r.items() if k in ("status", "last", "injected_total", "errors", "eval")}
                     for n, r in report["jobs"].items()}
            current = json.dumps(brief, ensure_ascii=False)
            if current != previous:
                print(stamp(), current, flush=True)
                previous = current
            if report["terminal"]:
                queue.terminate()  # this Popen child only; every queued job has exited
                queue.wait(timeout=30)
                if report["complete"]:
                    owner = json.loads((MATH_START / "strengthen_owner.json").read_text())
                    if owner["batch"] != BATCH or MATH_START.is_symlink():
                        raise RuntimeError("math-start ownership mismatch")
                    shutil.rmtree(MATH_START)
                    report["math_start_cleanup"] = "done"
                    atomic_json(REPORT, report)
                return 0 if report["complete"] else 1
            if queue.poll() is not None:
                raise RuntimeError("our queue exited; inspect state before restarting, workers left untouched")
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "resume", "worker", "status"))
    parser.add_argument("--host-label", choices=("74", "152"), default="74")
    parser.add_argument("--name", choices=list(SPECS))
    parser.add_argument("--recovery-plan", type=Path)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.name, args.recovery_plan)
    elif args.mode in ("run", "resume"):
        sys.exit(run(args.host_label, resume=args.mode == "resume"))
    else:
        print(json.dumps(snapshot(), ensure_ascii=False, indent=2))
