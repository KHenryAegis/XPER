"""Group-relative advantages for one GRPO iteration.

Input: conversations saved by ``rollout.py --save-convs`` (G samples per task).
Output: jsonl {"task", "weight", "messages"} for ``bc_train.py --weighted``, where
weight = (r - mean_group(r)) / (std_group(r) + eps), clipped to [-clip, clip].
Groups with zero variance (all solved or all failed) carry no signal and are dropped.

Off-policy experience injection (``--inject-src``): one stored source success for the task is added
to the group as an extra member with reward 1 before advantages are computed.
  --inject-mode fail  only into groups where every target sample failed (the target cannot find the
                      success itself; groups it already solves stay purely on-policy)
  --inject-mode all   into every group whose task the source solved (LUFFY-style mixing)
Injected trajectories cost the target no env calls (the source already paid for them).

Bootstrapping (``--inject-convs``): the target's own earlier successes (rollout.py --save-convs files) form a
second pool. For an all-fail group the target's own success is injected when it has one, the source's only
otherwise. ``--src-task-frac F`` keeps source successes for a fixed fraction F of the tasks (chosen by a hash
of the task id), so the source can be rationed while self-injection covers what the target learns to solve.

Protocol-vs-strategy controls (``--inject-control``, source injection only):
  mismatch  inject a source success of a *different* task, with its system prompt replaced by this task's:
            valid format and tool protocol, wrong content for this state
  shuffle   inject this task's source success with its non-final actions permuted (``done`` stays last,
            observations stay in place): same actions and format, state -> action mapping destroyed
  obs_mask  keep source actions but mask observations (generic action strategy remains)
  schema_random  same turn template, schema-sampled calls and masked observations; synthetic positive label
These controls distinguish components of the signal; success alone does not prove pure-protocol transfer.

Self-replay only (``--inject-convs`` without ``--inject-src``, ExGRPO/RLEP-style): ``--max-own-injections`` caps the
target's own injections in this call; the output then reports ``all_fail_groups`` and ``own_possible`` (all-fail
groups for which an own earlier success existed).
``--no-std``: advantage r - mean without the std division (LUFFY's grpo_use_std=False).
"""

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--convs", nargs="+", required=True)
p.add_argument("--out", required=True)
p.add_argument("--clip", type=float, default=3.0)
p.add_argument("--eps", type=float, default=1e-4)
p.add_argument("--inject-src", default="", help="glob of source stores (successes are injected)")
p.add_argument("--inject-convs", default="", help="glob of convs JSONL files (self-injection: target's own successes)")
p.add_argument("--inject-mode", choices=["fail", "all"], default="fail")
p.add_argument("--src-task-frac", type=float, default=1.0, help="keep source successes for this fraction of tasks")
p.add_argument("--inject-control", choices=["none", "mismatch", "shuffle", "obs_mask", "schema_random"], default="none")
p.add_argument("--max-src-injections", type=int, default=-1,
               help="maximum source trajectories this invocation; -1 is unlimited (self-injection is separate)")
p.add_argument("--max-own-injections", type=int, default=-1, help="cap on the target's own injections; -1 = unlimited")
p.add_argument("--no-std", action="store_true", help="advantage r - mean (LUFFY grpo_use_std=False)")
p.add_argument("--obs-chars", type=int, default=4000)
p.add_argument("--seed", type=int, default=0, help="which source success to inject when there are several")
a = p.parse_args()
if a.max_src_injections < -1:
    p.error("--max-src-injections must be -1 or nonnegative")

groups = collections.defaultdict(list)
for path in a.convs:
    for line in open(path):
        r = json.loads(line)
        # an episode needs at least one target turn: PrefixRL episodes the env ended during the replayed
        # source prefix contain only source actions and carry no signal about the policy
        if any(m["role"] == "assistant" and not m.get("prefix") for m in r["messages"]):
            groups[r["task"]].append(r)

# Injection pools: source stores (--inject-src) and the target's own successes (--inject-convs)
import hashlib


def _src_kept(task):
    return int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < a.src_task_frac


src, own = collections.defaultdict(list), collections.defaultdict(list)
if a.inject_src:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bc_train import build_conversations
    convs, tasks = build_conversations(a.inject_src, a.obs_chars)
    for c, t in zip(convs, tasks):
        if _src_kept(t):
            src[t].append(c)
if a.inject_convs:
    import glob as _glob
    for path in sorted(_glob.glob(a.inject_convs)):
        for line in open(path):
            r = json.loads(line)
            if r["reward"] >= 1.0 and not r.get("n_prefix"):
                own[r["task"]].append(r["messages"])

def _shuffled(msgs):
    """Permute the assistant turns before the last one; system, observations and the final turn stay put."""
    idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][:-1]
    acts = [msgs[i]["content"] for i in idx]
    shuf_rng.shuffle(acts)
    out = [dict(m) for m in msgs]
    for i, c in zip(idx, acts):
        out[i]["content"] = c
    return out


injected = injected_own = 0
rng = random.Random(a.seed)
shuf_rng = random.Random(a.seed + 7919)  # separate stream: shuffle picks the same trajectories as real injection
control_rng = random.Random(a.seed + 15485863)
src_tasks = sorted(src)
source_allowed = None
if a.max_src_injections >= 0:
    eligible = [task for task, eps in groups.items()
                if task in src and not (a.inject_mode == "fail" and any(e["reward"] > 0 for e in eps))
                and (a.inject_control == "mismatch" or not own.get(task))
                and (a.inject_control != "mismatch" or len(src_tasks) > 1)]
    # Independent stream: avoid favouring the first task family when the dose
    # runs out partway through a round; preserve the legacy trajectory RNG.
    source_allowed = set(random.Random(a.seed + 104729).sample(
        eligible, min(a.max_src_injections, len(eligible))))
all_fail = [task for task, eps in groups.items() if not any(e["reward"] > 0 for e in eps)]
own_possible = [task for task in all_fail if own.get(task)]
own_allowed = None
if a.max_own_injections >= 0:
    own_allowed = set(random.Random(a.seed + 1299709).sample(
        own_possible, min(a.max_own_injections, len(own_possible))))
for task, eps in groups.items():
    if a.inject_mode == "fail" and any(e["reward"] > 0 for e in eps):
        continue
    if a.inject_control == "mismatch":
        # same eligibility as real injection (the source solved this task), content from another task
        others = [t for t in src_tasks if t != task]
        if task not in src or not others:
            continue
        if source_allowed is not None and task not in source_allowed:
            continue
        msgs = [dict(m) for m in rng.choice(src[rng.choice(others)])]
        msgs[0] = dict(eps[0]["messages"][0])  # this task's system prompt (policy, tools, ticket)
    else:
        pool = own.get(task) if own_allowed is None or task in own_allowed else None
        pool = pool or src.get(task)
        if not pool:
            continue
        if pool is not own.get(task) and source_allowed is not None and task not in source_allowed:
            continue
        injected_own += pool is own.get(task)
        msgs = rng.choice(pool)
        if a.inject_control == "shuffle" and pool is not own.get(task):
            msgs = _shuffled(msgs)
        if a.inject_control in ("obs_mask", "schema_random") and pool is not own.get(task):
            from xper.injection_controls import mask_observations, schema_random
            msgs = (mask_observations(msgs) if a.inject_control == "obs_mask"
                    else schema_random(msgs, eps[0]["messages"][0]["content"], control_rng))
    eps.append({"task": task, "reward": 1.0, "messages": msgs, "injected": True,
                "injection_control": a.inject_control,
                "synthetic_label": a.inject_control != "none"})
    injected += 1

kept, n_eps, rewards = [], 0, []
for task, eps in groups.items():
    rs = [e["reward"] for e in eps]
    on = [e["reward"] for e in eps if not e.get("injected")]
    n_eps += len(on)
    rewards += on
    mu = sum(rs) / len(rs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in rs) / len(rs))
    if sd < 1e-8:
        continue
    for e in eps:
        adv = e["reward"] - mu if a.no_std else (e["reward"] - mu) / (sd + a.eps)
        w = max(-a.clip, min(a.clip, adv))
        record = {"task": task, "weight": w, "messages": e["messages"]}
        if e.get("injected"):
            record.update(injected=True, injection_control=e["injection_control"], synthetic_label=e["synthetic_label"])
        kept.append(record)

with open(a.out, "w") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
mixed = sum(1 for eps in groups.values() if len({e["reward"] for e in eps}) > 1)
stats = {"groups": len(groups), "mixed_groups": mixed, "episodes": n_eps, "injected": injected, "injected_own": injected_own,
         "train_seqs": len(kept), "mean_reward": round(sum(rewards) / max(1, len(rewards)), 4)}
if a.inject_convs:
    stats.update(all_fail_groups=len(all_fail), own_possible=len(own_possible))
print(json.dumps(stats))
