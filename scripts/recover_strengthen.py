"""Conservative recovery of interrupted strengthening runs; never reset claims/state."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import sys
import signal
import shutil
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))


def complete_model(path):
    try:
        index = json.loads((path / "model.safetensors.index.json").read_text())
        return all((path / f).is_file() and (path / f).stat().st_size > 0 for f in set(index["weight_map"].values()))
    except (OSError, ValueError, KeyError):
        return False


def shard_status(result, iteration, shard):
    """0=complete/reusable, 1=safe to run, 2=unsafe (possible lost calls)."""
    path = result / f"it{iteration}.shard{shard}.db"
    log = path.with_suffix(".log")
    convs = result / f"it{iteration}.shard{shard}.convs.jsonl"
    rows = []
    if path.exists():
        try:
            with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as db:
                rows = db.execute("SELECT task_id,env_calls,model_id FROM trajectories").fetchall()
        except sqlite3.Error:
            return 2
    if len(rows) == 32 and convs.exists():
        model = result.name.rsplit(".s", 1)[0] + f"-it{iteration}"
        counts = {task: sum(t == task for t, _, _ in rows) for task, _, _ in rows}
        records = [json.loads(l) for l in convs.read_text().splitlines() if l.strip()]
        if (len(counts) == 4 and set(counts.values()) == {8} and all(m == model for _, _, m in rows)
                and len(records) == 32 and sorted(r["task"] for r in records) == sorted(t for t, _, _ in rows)):
            return 0
    if rows:
        return 2
    # The outer driver's one-time retry renames .log to .try1.log before it
    # invokes this check again. Keep that reset evidence: an empty DB after
    # reset does not imply that no environment calls were spent.
    attempt_logs = [log, result / f"it{iteration}.shard{shard}.try1.log",
                    log.with_name(log.name + ".oom-first"), log.with_name(log.name + ".oom-last"),
                    log.with_name(log.name + ".before-recovery")]
    if any(p.exists() and re.search(r"^reset \d+ episodes", p.read_text(errors="replace"), re.M)
           for p in attempt_logs):
        return 2
    return 1


def plan(name, cohort=None):
    if cohort is None:
        import strengthen as cohort
    s = cohort
    spec = s.SPECS[name]
    if spec["kind"] != "train":
        raise ValueError("only training recoveries are supported")
    tag, seed = name.rsplit("-s", 1)
    result = s.ROOT / "results" / f"{tag}.s{seed}"
    owner = json.loads((result / "owner.json").read_text())
    if owner["batch"] != s.BATCH or owner["name"] != name:
        raise ValueError("result ownership mismatch")
    history = s.iterations(name)
    saved = []
    for p in result.glob("it*.train.log"):
        it = int(p.name.split(".")[0][2:])
        for line in p.read_text().splitlines():
            if line.startswith('{"saved"'):
                record = json.loads(line)
                saved.append((it, record))
    if saved:
        last, record = max(saved, key=lambda row: row[0])
        policy = Path(record["saved"])
        metadata = json.loads(policy.parent.joinpath("data_stats.json").read_text())
        if not complete_model(policy) or metadata["src"] != record["src"]:
            raise ValueError("last saved policy is incomplete or no longer corresponds to the logged update")
    else:
        last = -1
        policy = s.MATH_START if spec["env"] == "math" else s.MODELS / "8b-commonsft/final"
        if not complete_model(policy):
            raise ValueError("shared start incomplete")
    iteration = last + 1
    pending = next((r for r in history if r["iteration"] == iteration), None)
    if any(r["iteration"] > iteration for r in history):
        raise ValueError("history extends beyond the last committed update; manual audit required")
    phase = "update" if pending else "rollout"
    statuses = [shard_status(result, iteration, i) for i in range(4)]
    if 2 in statuses or (pending and any(v != 0 for v in statuses)):
        raise ValueError("incomplete rollout with possible spent calls; refusing to discard it")
    previous = [r for r in history if r["iteration"] < iteration]
    return dict(name=name, phase=phase, iteration=iteration, policy=str(policy),
                spent_prior=previous[-1]["env_calls"] if previous else 0,
                src_used_prior=sum(r["injected"] - r["injected_own"] for r in previous),
                logged_round=pending, shard_status=statuses,
                policy_index_sha256=hashlib.sha256((policy / "model.safetensors.index.json").read_bytes()).hexdigest())


def guard_driver(driver, recovery=None, plan_path=None):
    # These replacements change scheduling/recovery only, not model, data or optimizer settings.
    needle = "  shard() {  # one rollout shard; retried once below (vLLM's internal port can collide with another engine)\n"
    if driver.count(needle) != 1:
        raise ValueError("unrecognized rollout driver")
    initial_check = '''for g in "${G[@]}"; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$mem" -lt 1000 ] || die "GPU $g already has ${mem}MiB in use"
done'''
    if initial_check not in driver:
        raise ValueError("unrecognized initial GPU check")
    driver = driver.replace(initial_check, "# stage_guard waits before every GPU stage, including startup.", 1)
    driver = driver.replace(needle, needle + '''    if $PY scripts/recover_strengthen.py shard-ready --result "$R" --iteration "$it" --shard "$1"; then
      echo "  reusing completed it $it shard $1"; return 0
    else
      ready=$?; [ "$ready" -eq 1 ] || die "unsafe partial rollout: it $it shard $1"
    fi
''', 1)
    driver = driver.replace('CUDA_VISIBLE_DEVICES=${G[$1]} $PY scripts/rollout.py',
        'CUDA_VISIBLE_DEVICES=${G[$1]} $PY scripts/stage_guard.py --gpus "${G[$1]}" --kind rollout --log "$R/it$it.shard$1.log" -- $PY scripts/rollout.py', 1)
    driver = driver.replace(' > $R/it$it.shard$1.log 2>&1', '', 1)
    driver = driver.replace('CUDA_VISIBLE_DEVICES=$T $TR --nproc_per_node',
        'CUDA_VISIBLE_DEVICES=$T $PY scripts/stage_guard.py --gpus "$T" --kind update --log "$R/it$it.train.log" -- $TR --nproc_per_node', 1)
    driver = driver.replace('> $R/it$it.train.log 2>&1', '', 1)
    driver = driver.replace('CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/rollout.py',
        'CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/stage_guard.py --gpus "${G[0]}" --kind eval --log "results/eval/$m.log" -- $PY scripts/rollout.py', 1)
    driver = driver.replace('> results/eval/$m.log 2>&1', '', 1)
    if recovery:
        r = recovery
        driver = driver.replace('policy=$START; it=0; spent=0',
            f"policy={shlex.quote(r['policy'])}; it={r['iteration']}; spent={r['spent_prior']}; src_used={r['src_used_prior']}", 1)
        echo = '  echo "  it $it: $adv | cumulative env calls $spent"'
        check = f'  $PY scripts/recover_strengthen.py assert-adv --plan {shlex.quote(str(plan_path))} --iteration "$it" --stats "$adv" || die "recovery advantage mismatch"\n'
        if r["phase"] == "update":
            driver = driver.replace(echo, check + f'  if [ "$it" -ne {r["iteration"]} ]; then\n{echo}\n  else echo "  continuing previously logged it $it update"; fi', 1)
        else:
            driver = driver.replace(echo, check + echo, 1)
    return driver


def queue_recoveries(cohort=None, worker_script="scripts/strengthen.py"):
    if cohort is None:
        import strengthen as cohort
    s = cohort
    state = json.loads((s.BASE / "state.json").read_text())
    mapping_path = s.BASE / "recoveries.json"
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    jobs_path = s.BASE / "jobs.jsonl"
    jobs = [json.loads(l) for l in jobs_path.read_text().splitlines() if l.strip()]
    added = []
    for name, spec in s.SPECS.items():
        old = mapping.get(name, {})
        current = state.get(old.get("job", name), {})
        if spec["kind"] != "train" or not current.get("status", "").startswith("fail"):
            continue
        if old.get("verified_completion"):
            continue
        try:
            s.finalize_verified_run(name)
        except (OSError, ValueError, KeyError, IndexError, sqlite3.Error):
            pass  # still incomplete: proceed to conservative recovery planning
        else:
            reconciled = dict(old, verified_completion=True, reconciled=s.stamp())
            if "planning_error" in reconciled:
                reconciled["resolved_planning_error"] = reconciled.pop("planning_error")
            mapping[name] = reconciled
            continue
        if old.get("planning_error") or old.get("attempt", 0) >= 3:
            continue
        attempt = old.get("attempt", 0) + 1
        try:
            recovery = plan(name, s)
            recovery["attempt"] = attempt
            folder = s.BASE / "recovery_plans"
            folder.mkdir(exist_ok=True)
            path = folder / f"{name}.r{attempt}.json"
            if path.exists():
                raise ValueError("recovery plan already exists; inspect before retrying")
            s.atomic_json(path, recovery)
            key = f"recover{attempt}-{name}"
            mapping[name] = dict(job=key, attempt=attempt, plan=str(path), queued=s.stamp(),
                                 initial_failure=old.get("initial_failure", current),
                                 previous_recovery=old.get("job"))
            added.append(dict(name=key, gpus=4, after=spec.get("after", []),
                cmd=shlex.join([str(s.PY), "-B", worker_script, "worker", "--name", name,
                                "--recovery-plan", str(path)])))
        except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
            mapping[name] = dict(old, planning_error=str(exc))
    if added:
        # Prioritize recoveries closest to their pre-registered call budget.
        def fraction(job):
            canonical = next(n for n, rec in mapping.items() if rec.get("job") == job["name"])
            r = json.loads(Path(mapping[canonical]["plan"]).read_text())
            calls = r["logged_round"]["env_calls"] if r["logged_round"] else r["spent_prior"]
            return calls / (10403 if s.SPECS[canonical]["env"] == "math" else 44414)
        added.sort(key=fraction, reverse=True)
        s.atomic_json(mapping_path, mapping)
        tmp = jobs_path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(j) + "\n" for j in added + jobs))
        tmp.replace(jobs_path)
    elif mapping:
        s.atomic_json(mapping_path, mapping)


def watch(cohort=None, worker_script="scripts/strengthen.py"):
    """Attach to the existing owned queue; active GPU jobs are never restarted."""
    if cohort is None:
        import strengthen as cohort
    s = cohort
    lock = (s.BASE / "controller.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((s.BASE / "launch.json").read_text())
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if manifest["active_deployment"]["boot_id"] != boot:
        raise RuntimeError("watch on the active deployment host only")
    queue_pid = manifest["queue_pid"]
    queue_proc = Path(f"/proc/{queue_pid}")
    queue_cmd = (queue_proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    expected = "scripts/gpu_queue.py --jobs " + str(s.BASE / "jobs.jsonl")
    if expected not in queue_cmd or (queue_proc / "cwd").resolve() != s.ROOT:
        raise RuntimeError("queue identity mismatch")
    start_ticks = (queue_proc / "stat").read_text().split()[21]
    manifest.setdefault("monitor_revisions", []).append(dict(
        updated=s.stamp(), previous_controller=manifest["pid"], controller=os.getpid(),
        reason="Attach recovery monitor; retain active workers and all existing state/claims",
        hashes={p: hashlib.sha256((s.ROOT / p).read_bytes()).hexdigest() for p in dict.fromkeys((
            "scripts/strengthen.py", "scripts/recover_strengthen.py", "scripts/stage_guard.py",
            worker_script, "xper/actions.py"))}))
    manifest["pid"] = os.getpid()
    manifest["active_deployment"]["controller_pid"] = os.getpid()
    s.atomic_json(s.BASE / "launch.json", manifest)
    previous = None
    while True:
        try:
            queue_recoveries(s, worker_script)
            report = s.snapshot()
            state = json.loads((s.BASE / "state.json").read_text())
        except json.JSONDecodeError:
            time.sleep(5)
            continue
        s.atomic_json(s.REPORT, report)
        brief = {n: {k: v for k, v in r.items() if k in (
            "status", "last", "errors", "recovery_job", "eval")} for n, r in report["jobs"].items()}
        current = json.dumps(brief, ensure_ascii=False)
        if current != previous:
            print(s.stamp(), current, flush=True)
            previous = current
        if report["terminal"] and not any(r["status"] == "running" for r in state.values()):
            if (queue_proc / "stat").read_text().split()[21] != start_ticks:
                raise RuntimeError("queue PID identity changed; refusing to signal")
            os.kill(queue_pid, signal.SIGTERM)  # verified queue created by this experiment
            if report["complete"]:
                owner = json.loads((s.MATH_START / "strengthen_owner.json").read_text())
                if owner["batch"] != s.BATCH or s.MATH_START.is_symlink():
                    raise RuntimeError("math-start ownership mismatch")
                shutil.rmtree(s.MATH_START)
                report["math_start_cleanup"] = "done"
                s.atomic_json(s.REPORT, report)
            return 0 if report["complete"] else 1
        if not queue_proc.exists():
            raise RuntimeError("owned queue exited; active workers were left untouched")
        time.sleep(20)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "shard-ready", "assert-adv", "watch"))
    p.add_argument("--name")
    p.add_argument("--result", type=Path)
    p.add_argument("--iteration", type=int)
    p.add_argument("--shard", type=int)
    p.add_argument("--plan", type=Path)
    p.add_argument("--stats")
    a = p.parse_args()
    if a.mode == "plan":
        print(json.dumps(plan(a.name), indent=2))
    elif a.mode == "shard-ready":
        raise SystemExit(shard_status(a.result, a.iteration, a.shard))
    elif a.mode == "assert-adv":
        r = json.loads(a.plan.read_text())
        if a.iteration == r["iteration"] and r["logged_round"]:
            expected = {k: v for k, v in r["logged_round"].items() if k not in ("iteration", "env_calls")}
            if expected != json.loads(a.stats):
                raise SystemExit("resumed advantages differ from the original logged round")
    else:
        raise SystemExit(watch())
