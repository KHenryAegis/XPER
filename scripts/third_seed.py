"""Isolated third-rollout-seed cohort on host B; finite queue with owned cleanup.

Uses the audited strengthening worker/guard/recovery code with a separate module
namespace, manifest, claims, results and caches. `run` is fresh-only: it never
resets a queue or repeats an existing experiment. `status` is read-only.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tel_dose import ROOT, MODELS, PY, atomic_json, runtime_env, stamp

BATCH = "third_seed_20261115"
BASE = ROOT / "logs/queue" / BATCH
REPORT = ROOT / "results" / f"{BATCH}_report.json"
CACHE = Path("/tmp/xper-third-seed-20261115")
MATH_START = MODELS / "sc-math-start-20261115-s603"
EXPECTED_BOOT = "60061f08-a2d6-49e1-b0b6-bb6f85189f69"
PREP = "s603-math-init"
SPECS = {
    PREP: dict(kind="prep", gpus=0),
    "td100-s603": dict(kind="train", env="tel", arm="real", seed=603, gpus=4, after=[], injection_cap=100),
    "td0-s603": dict(kind="train", env="tel", arm="none", seed=603, gpus=4, after=[], injection_cap=0),
    "sc-math-real-s603": dict(kind="train", env="math", arm="real", seed=603, gpus=4, after=[PREP], injection_cap=100),
    "sc-math-none-s603": dict(kind="train", env="math", arm="none", seed=603, gpus=4, after=[PREP], injection_cap=100),
}


def cohort():
    # Load in a fresh namespace so importing this script cannot reconfigure the
    # historical strengthening cohort or its reporting functions.
    spec = importlib.util.spec_from_file_location("_third_seed_strengthen", ROOT / "scripts/strengthen.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    for key, value in dict(BATCH=BATCH, BASE=BASE, REPORT=REPORT, CACHE=CACHE,
                           MATH_START=MATH_START, SPECS=SPECS).items():
        setattr(s, key, value)
    original_verify = s.verify

    def verify(name):
        rec = original_verify(name)
        accounting = BASE / f"{name}.failed_sampling_cost.json"
        if accounting.exists():
            rec["budget_accounting"] = json.loads(accounting.read_text())
            if "training" in rec:
                calls = rec["training"]["env_calls"]
                rec["budget_accounting"]["successful_rollout_calls"] = calls
                rec["budget_accounting"]["total_calls_lower_bound"] = calls + rec["budget_accounting"]["failed_calls_lower_bound"]
                rec["budget_accounting"]["total_calls_conservatively_charged"] = calls + rec["budget_accounting"]["failed_calls_upper_bound"]
        if name == PREP:
            previous = json.loads((ROOT / "logs/queue/strengthen_20261113/sc-math-init.verified.json").read_text())
            if rec["initializer"] != previous["initializer"]:
                raise ValueError("rebuilt Math initializer differs from seeds 601/602; do not train")
            rec["matches_previous_initializer"] = True
            rec["headroom_reference"] = "results/eval/sc-math-headroom.db"
            rec["headroom_note"] = "Previous full-test evaluation; not a new evaluation of this rebuilt checkpoint"
            rec["index_sha256"] = hashlib.sha256((s.MATH_START / "model.safetensors.index.json").read_bytes()).hexdigest()
        return rec

    s.verify = verify
    return s


def preflight(s):
    for batch in ("tel_dose_20261112", "strengthen_20261113"):
        d = json.loads((ROOT / "results" / f"{batch}_report.json").read_text())
        if not d["complete"] or not d["terminal"]:
            raise ValueError(f"previous cohort is not complete: {batch}")
    old = json.loads((ROOT / "logs/queue/strengthen_20261113/launch.json").read_text())
    # No optimizer, rollout driver, advantage construction, or init change since
    # the two-seed cohort. Only orchestration gains optional cohort arguments.
    for path in ("scripts/run_grpo.sh", "scripts/grpo_advantages.py", "scripts/bc_train.py", "scripts/prep_init.py",
                 "xper/mathenv.py", "xper/math_seccomp.py"):
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != old["hashes"][path]:
            raise ValueError(f"training/init implementation changed: {path}")
    from recover_strengthen import complete_model
    for path in (MODELS / "8b-commonsft/final", s.BASE_MODEL):
        if not complete_model(path):
            raise ValueError(f"missing/incomplete original start: {path}")
    candidates = [BASE / "launch.json", BASE / "jobs.jsonl", BASE / "state.json", MATH_START]
    for name, spec in SPECS.items():
        candidates += [CACHE / name, BASE / f"{name}.verified.json"]
        if spec["kind"] == "train":
            tag, seed = name.rsplit("-s", 1)
            candidates += [ROOT / "results" / f"{tag}.s{seed}",
                           ROOT / "results/eval" / f"{name}.db", ROOT / "results/eval" / f"{name}.log"]
            candidates += [MODELS / f"{name}-{slot}" for slot in ("0", "1", "final")]
    s.fresh(candidates)
    return dict(training_recipe_matches_20261113=True,
                math_initializer_reference="logs/queue/strengthen_20261113/sc-math-init.verified.json",
                third_seed=603, optimizer_seed=0, strict_math=True,
                source_collection_calls=dict(tel=44414, math=10403),
                cleanup="verified per-job checkpoints/cache, then owned shared Math start",
                estimated_transient_checkpoint_peak_gib=140)


def run():
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    # The original run was pinned to one machine (EXPECTED_BOOT). Set XPER_EXPECTED_BOOT / XPER_HOST to pin a rerun.
    if boot != os.environ.get("XPER_EXPECTED_BOOT", boot) or socket.gethostname().split(".")[0] != os.environ.get("XPER_HOST", socket.gethostname().split(".")[0]):
        raise RuntimeError("run only on the deployment named by XPER_HOST / XPER_EXPECTED_BOOT")
    BASE.mkdir(parents=True, exist_ok=True)
    lock = (BASE / "controller.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    s = cohort()
    checked = preflight(s)
    paths = ("scripts/third_seed.py", "scripts/strengthen.py", "scripts/recover_strengthen.py",
             "scripts/stage_guard.py", "scripts/gpu_queue.py", "scripts/run_grpo.sh",
             "scripts/grpo_advantages.py", "scripts/bc_train.py", "scripts/prep_init.py",
             "xper/injection_controls.py", "xper/mathenv.py", "xper/math_seccomp.py")
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    manifest = dict(batch=BATCH, host=socket.gethostname(), pid=os.getpid(), started=stamp(),
                    specs=SPECS, hashes=hashes, preflight=checked,
                    active_deployment=dict(host_label="152", boot_id=boot, controller_pid=os.getpid()))
    with (BASE / "jobs.jsonl").open("x") as f:
        for name, spec in SPECS.items():
            f.write(json.dumps(dict(name=name, gpus=spec["gpus"], after=spec.get("after", []),
                cmd=shlex.join([str(PY), "-B", "scripts/third_seed.py", "worker", "--name", name]))) + "\n")
    atomic_json(BASE / "launch.json", manifest)
    with (BASE / "queue.log").open("a") as output:
        queue = subprocess.Popen([str(PY), "-B", "scripts/gpu_queue.py", "--jobs", str(BASE / "jobs.jsonl"),
            "--state", str(BASE / "state.json"), "--claim-dir", str(BASE / "claims"),
            "--host-label", "152", "--poll", "20"], cwd=ROOT, env=runtime_env(),
            stdout=output, stderr=subprocess.STDOUT)
    manifest["queue_pid"] = queue.pid
    manifest["active_deployment"]["queue_pid"] = queue.pid
    atomic_json(BASE / "launch.json", manifest)
    previous = None
    while True:
        try:
            if (BASE / "state.json").exists():
                from recover_strengthen import queue_recoveries
                queue_recoveries(s, "scripts/third_seed.py")
            report = s.snapshot()
            state = json.loads((BASE / "state.json").read_text()) if (BASE / "state.json").exists() else {}
        except json.JSONDecodeError:
            time.sleep(5)
            continue
        atomic_json(REPORT, report)
        brief = {n: {k: v for k, v in r.items() if k in ("status", "last", "injected_total", "errors", "eval", "recovery_job")}
                 for n, r in report["jobs"].items()}
        current = json.dumps(brief, ensure_ascii=False)
        if current != previous:
            print(stamp(), current, flush=True)
            previous = current
        if report["terminal"] and not any(v["status"] == "running" for v in state.values()):
            queue.terminate()  # this controller's Popen child; no GPU worker remains active
            queue.wait(timeout=30)
            if report["complete"]:
                owner = json.loads((MATH_START / "strengthen_owner.json").read_text())
                if owner["batch"] != BATCH or MATH_START.is_symlink() or MATH_START.parent.resolve() != MODELS.resolve():
                    raise RuntimeError("shared Math-start cleanup ownership mismatch")
                shutil.rmtree(MATH_START)
                report["math_start_cleanup"] = "done"
                atomic_json(REPORT, report)
            return 0 if report["complete"] else 1
        if queue.poll() is not None:
            raise RuntimeError("owned queue exited; no state reset and no worker restart attempted")
        time.sleep(20)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "worker", "status", "preflight", "watch"))
    p.add_argument("--name", choices=list(SPECS))
    p.add_argument("--recovery-plan", type=Path)
    args = p.parse_args()
    if args.mode == "run":
        raise SystemExit(run())
    s = cohort()
    if args.mode == "watch":
        from recover_strengthen import watch
        raise SystemExit(watch(s, "scripts/third_seed.py"))
    elif args.mode == "worker":
        if args.name is None:
            p.error("worker requires --name")
        s.worker(args.name, args.recovery_plan)
    elif args.mode == "preflight":
        print(json.dumps(preflight(s), indent=2))
    else:
        print(json.dumps(s.snapshot(), ensure_ascii=False, indent=2))
