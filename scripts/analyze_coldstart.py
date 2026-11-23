"""Audit of finished cold-start cohort runs (library for scripts/analyze_paper.py and scripts/analyze_alternatives.py).

Reads evaluation and training stores read-only and checks every run's budget, injections and recovery history.
Inputs are opened read-only. No checkpoints, trajectories, or bootstrap draws are written.
Intervals resample held-out tasks jointly across arms and observed rollout seeds;
they are conditional on these trained policies, not intervals over future training seeds.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (601, 602)
BOOT_SEED = 20261115
DRAWS = 20000
ITER = re.compile(r"^\s*it (\d+): (\{.*\}) \| cumulative env calls (\d+)$", re.M)
ARMS = {
    "tel_none": ("telecom", "No injection", "td0"),
    "tel_d5": ("telecom", "Cap 5", "td5"),
    "tel_d20": ("telecom", "Cap 20", "td20"),
    "tel_real": ("telecom", "Real source", "td100"),
    "tel_mask": ("telecom", "Masked observations", "sc-tel-obs_mask"),
    "tel_schema": ("telecom", "Random schema", "sc-tel-schema_random"),
    "tel_mismatch": ("telecom", "Different-task source", "tp-mismatch"),
    "tel_shuffle": ("telecom", "Shuffled actions", "tp-shuffle"),
    "math_none": ("math", "No injection", "sc-math-none"),
    "math_real": ("math", "Real source", "sc-math-real"),
}


def readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_eval(path, tasks, samples, model):
    with readonly(path) as db:
        rows = db.execute("SELECT task_id,reward,success,model_id,meta FROM trajectories").fetchall()
    groups = collections.defaultdict(list)
    for task, reward, success, model_id, meta in rows:
        if reward not in (0, 1) or success != int(reward >= 1) or model_id != model:
            raise ValueError(f"invalid outcome/model in {path}")
        groups[task].append((int(reward >= 1), json.loads(meta)["k"]))
    if len(groups) != tasks or any(sorted(k for _, k in g) != list(range(samples)) for g in groups.values()):
        raise ValueError(f"missing/duplicate tasks or samples in {path}")
    ids = sorted(groups)
    rates = np.array([sum(x for x, _ in groups[t]) / samples for t in ids])
    covered = np.array([any(x for x, _ in groups[t]) for t in ids], dtype=float)
    return ids, rates, covered


def merge_history(paths):
    rounds = {}
    for path in paths:
        for i, stats, calls in ITER.findall(path.read_text()):
            row = dict(iteration=int(i), env_calls=int(calls), **json.loads(stats))
            if int(i) in rounds and rounds[int(i)] != row:
                raise ValueError(f"conflicting recovered iteration {i}: {path}")
            rounds[int(i)] = row
    if not rounds or sorted(rounds) != list(range(max(rounds) + 1)):
        raise ValueError("missing training iterations")
    return [rounds[i] for i in sorted(rounds)]


def audit_training(directory, history, record):
    """Check each plotted on-policy reward against saved target rollout DBs."""
    rounds = collections.defaultdict(list)
    for path in directory.glob("it*.shard*.db"):
        m = re.fullmatch(r"it(\d+)\.shard(\d+)\.db", path.name)
        if m:
            rounds[int(m[1])].append((int(m[2]), path))
    if sorted(rounds) != [r["iteration"] for r in history]:
        raise ValueError(f"training DB/history coverage mismatch: {directory}")
    total_successes = total_calls = total_episodes = 0
    previous_calls = 0
    for row in history:
        shards = rounds[row["iteration"]]
        if sorted(s for s, _ in shards) != list(range(4)):
            raise ValueError(f"missing/duplicate shard: {directory}, {row['iteration']}")
        count = successes = calls = 0
        for _, path in shards:
            with readonly(path) as db:
                n, solved, charged, valid = db.execute(
                    "SELECT COUNT(*),SUM(reward>=1),SUM(env_calls),MIN(success=(reward>=1)) FROM trajectories"
                ).fetchone()
            if n != 32 or valid != 1:
                raise ValueError(f"invalid target training shard: {path}")
            count += n
            successes += solved
            calls += charged
        total_calls += calls
        total_successes += successes
        total_episodes += count
        if (count != row["episodes"] or row["groups"] != 16 or row["injected_own"] != 0
                or abs(row["mean_reward"] - successes / count) > 0.000051
                or row["env_calls"] != total_calls or total_calls <= previous_calls):
            raise ValueError(f"target DB/log mismatch: {directory}, iteration {row['iteration']}")
        row["target_successes"] = successes
        row["target_success_rate"] = successes / count
        previous_calls = total_calls
    expected = record["training"]
    if (total_calls, total_episodes, total_successes) != (
            expected["env_calls"], expected["episodes"], expected["successes"]):
        raise ValueError(f"training report mismatch: {directory}")


def after_injection(history):
    injected = [r for r in history if r["injected"]]
    last = injected[-1] if injected else None
    post = [r for r in history if last and r["iteration"] > last["iteration"]]
    tail = history[-5:]
    return {
        "last_injection_iteration": last["iteration"] if last else None,
        "calls_at_last_injection": last["env_calls"] if last else None,
        "post_injection_rounds": len(post),
        "post_injection_calls": history[-1]["env_calls"] - last["env_calls"] if last else None,
        "post_injection_target_success_rate": (
            sum(r["target_successes"] for r in post) / sum(r["episodes"] for r in post) if post else None),
        "last_five_target_success_rate": sum(r["target_successes"] for r in tail) / sum(r["episodes"] for r in tail),
    }


def bootstrap_arms(arrays, draws=DRAWS, seed=BOOT_SEED):
    """arrays[arm, seed, task]; shared task indices preserve pairing and clusters."""
    a = np.asarray(arrays, dtype=float)
    if a.ndim != 3 or not np.isfinite(a).all() or min(a.shape) == 0:
        raise ValueError("expected finite arm x seed x task data")
    per_task = a.mean(axis=1)
    rng = np.random.default_rng(seed)
    out = np.empty((a.shape[0], draws))
    for start in range(0, draws, 512):
        ids = rng.integers(0, a.shape[2], size=(min(512, draws - start), a.shape[2]))
        out[:, start:start + len(ids)] = per_task[:, ids].mean(axis=2)
    return out


def ci(values):
    return np.quantile(values, [.025, .975]).tolist()


def collect():
    reports = {}
    manifest = {}
    for fn in ("tel_protocol_report.json", "tel_dose_20261112_report.json", "strengthen_20261113_report.json"):
        path = ROOT / "results" / fn
        report = json.loads(path.read_text())
        if not report["complete"] or not report["terminal"]:
            raise ValueError(f"unfinished cohort: {path}")
        reports.update(report["jobs"])
        manifest[str(path.relative_to(ROOT))] = digest(path)
    arms, values = {}, {}
    task_ids = {}
    for key, (env, label, tag) in ARMS.items():
        n_tasks, samples, budget, source_cost = (40, 8, 44414, 44414) if env == "telecom" else (300, 4, 10403, 10403)
        runs, rates = [], []
        for seed in SEEDS:
            name = f"{tag}-s{seed}"
            rec = reports[name]
            if rec["status"] != "ok" or rec["errors"]:
                raise ValueError(f"unverified job {name}")
            db = ROOT / "results/eval" / f"{name}.db"
            ids, rate, pk = load_eval(db, n_tasks, samples, name)
            if env in task_ids and task_ids[env] != ids:
                raise ValueError(f"test task identities differ for {name}")
            task_ids[env] = ids
            if abs(rate.mean() - rec["eval"]["pass@1"]) > .0001:
                raise ValueError(f"eval/report mismatch for {name}")
            if abs(pk.mean() - rec["eval"][f"pass@{samples}"]) > .0001:
                raise ValueError(f"pass@k/report mismatch for {name}")
            manifest[str(db.relative_to(ROOT))] = digest(db)
            if tag.startswith("sc-"):
                base = ROOT / "logs/queue/strengthen_20261113"
                logs = [base / f"{name}.log"] + sorted(base.glob(f"recover*-{name}.log"),
                    key=lambda p: int(re.match(r"recover(\d+)-", p.name)[1]))
            elif tag.startswith("td"):
                logs = [ROOT / "logs/queue/tel_dose_20261112" / f"{name}.log"]
            else:
                logs = [ROOT / "logs/queue" / f"{name}.log"]
            history = merge_history(logs)
            directory = ROOT / "results" / f"{tag}.s{seed}"
            audit_training(directory, history, rec)
            injected = sum(r["injected"] for r in history)
            if injected != rec["injected_total"] or history[-1]["env_calls"] < budget:
                raise ValueError(f"dose/budget mismatch {name}")
            for path in logs:
                manifest[str(path.relative_to(ROOT))] = digest(path)
            supervised_tokens = 0
            updates = 0
            for path in directory.glob("it*.train.log"):
                for line in path.read_text().splitlines():
                    if line.startswith('{"saved"'):
                        stats = json.loads(line)
                        supervised_tokens += stats["supervised_tokens"]
                        updates += 1
            rates.append(rate)
            runs.append(dict(name=name, seed=seed, pass1=float(rate.mean()), passk=float(pk.mean()),
                target_calls=history[-1]["env_calls"], source_injections=injected,
                target_episodes=rec["training"]["episodes"], updates=updates,
                dataset_supervised_tokens=supervised_tokens, history=history, **after_injection(history)))
            print(f"audited {name}: test={rate.mean():.6f}, calls={history[-1]['env_calls']}", flush=True)
        values[key] = np.stack(rates)
        arms[key] = dict(environment=env, label=label, tasks=n_tasks, samples=samples, source_calls=source_cost,
                         runs=runs, mean_pass1=float(np.mean([r["pass1"] for r in runs])),
                         sd_pass1=float(np.std([r["pass1"] for r in runs], ddof=1)),
                         mean_passk=float(np.mean([r["passk"] for r in runs])))
    comparisons = {}
    for env in ("telecom", "math"):
        keys = [k for k, a in arms.items() if a["environment"] == env]
        boot = bootstrap_arms(np.stack([values[k] for k in keys]))
        for i, key in enumerate(keys):
            arms[key]["task_bootstrap_ci95_pass1"] = ci(boot[i])
        pairs = [(k, "tel_none" if env == "telecom" else "math_none") for k in keys if not k.endswith("none")]
        if env == "telecom":
            pairs += [("tel_real", "tel_mask"), ("tel_real", "tel_schema"), ("tel_mask", "tel_schema")]
        for a, b in pairs:
            delta = values[a] - values[b]
            comparison = dict(arm=a, reference=b, mean_difference=float(delta.mean()),
                task_bootstrap_ci95=ci(boot[keys.index(a)] - boot[keys.index(b)]),
                seed_differences=[float(x) for x in delta.mean(axis=1)],
                tasks=delta.shape[1], matched_injection_cap=not ("mismatch" in a or "shuffle" in a))
            comparison["per_seed_task_ci95"] = [
                ci(bootstrap_arms(delta[s][None, None, :], seed=BOOT_SEED + s)[0]) for s in range(2)]
            comparisons[f"{a}_vs_{b}"] = comparison
    # A new full-test start is kept separate from the old 100-task pilot.
    ids, rate, pk = load_eval(ROOT / "results/eval/sc-math-headroom.db", 300, 4, "sc-math-headroom")
    if ids != task_ids["math"]:
        raise ValueError("math start and final evaluation tasks differ")
    start_path = ROOT / "results/eval/sc-math-headroom.db"
    manifest[str(start_path.relative_to(ROOT))] = digest(start_path)
    sources = {}
    for env, pattern, expected in (("telecom", "results/src/4b-inst.shard*.db", (2400, 44414, 223, 80)),
                                   ("math", "results/math/src/4b-inst.shard*.db", (2000, 10403, 452, 452))):
        count = calls = successes = 0
        all_tasks, solved_tasks = set(), set()
        paths = sorted(ROOT.glob(pattern))
        if not paths:
            raise ValueError(f"missing source store: {env}")
        for path in paths:
            with readonly(path) as db:
                for task, reward, charged in db.execute("SELECT task_id,reward,env_calls FROM trajectories"):
                    all_tasks.add(task)
                    count += 1
                    calls += charged
                    successes += reward >= 1
                    if reward >= 1:
                        solved_tasks.add(task)
            manifest[str(path.relative_to(ROOT))] = digest(path)
        if (count, calls, successes, len(solved_tasks)) != expected or all_tasks.intersection(task_ids[env]):
            raise ValueError(f"source inventory/leakage audit failed: {env}")
        sources[env] = dict(episodes=count, calls=calls, successes=successes,
                           solved_tasks=len(solved_tasks), test_overlap=0)
    manifest[str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__))
    return dict(schema_version=1, analysis_seed=BOOT_SEED, bootstrap_draws=DRAWS,
        interval_method="paired percentile task-cluster bootstrap; seeds 601/602 held fixed, shared task draws",
        limitations=["Intervals condition on the two observed trained policies; they do not estimate training-seed population uncertainty.",
                     "Intervals are marginal, with no multiple-comparison correction.",
                     "An all-zero empirical cell has a degenerate task-bootstrap interval, not proof of zero population success.",
                     "Training curves are on-policy training success, not held-out learning curves.",
                     "Environment-call budgets are matched at round boundaries, not tokens or FLOPs.",
                     "Protocol mismatch/shuffle runs have unrestricted, unequal injection exposure.",
                     "Supervised-token totals count completed datasets, not repeated failed optimizer work or full compute."],
        math_start=dict(tasks=300, samples=4, pass1=float(rate.mean()), passk=float(pk.mean())),
        task_ids=task_ids, sources=sources, arms=arms, comparisons=comparisons, input_sha256=manifest)


def pp(x):
    return f"{100*x:.2f}"


def interval(xs):
    return f"[{pp(xs[0])}, {pp(xs[1])}]"
