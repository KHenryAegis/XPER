"""Exact-replay coverage of several target stores by one source store, on the tasks both share.

Like scripts/coverage.py, but each side is a glob over shard stores, and the result also reports
how many target env calls exact reuse could have served and where target episodes first leave
the source trie.

Usage: python scripts/coverage_multi.py --src 'results/src/4b-inst.shard*.db' \
           --tgt strong='results/pipeline_b2s/selfexp.s401.r*.shard*.db' ...
"""

import argparse
import glob
import json
import sqlite3
from collections import Counter, defaultdict


def load(pattern):
    nodes, tasks, eps = set(), set(), []
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no rollout stores match {pattern}")
    for path in paths:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        nodes.update(r[0] for r in db.execute("SELECT node_id FROM nodes"))
        steps = defaultdict(list)
        for tid, t, child in db.execute("SELECT traj_id, t, child_id FROM steps ORDER BY traj_id, t"):
            steps[tid].append(child)
        for tid, task, succ in db.execute("SELECT traj_id, task_id, success FROM trajectories"):
            tasks.add(task)
            eps.append((task, succ, steps[tid]))
        db.close()
    return nodes, tasks, eps


def coverage(src, tgt, max_depth=12):
    src_nodes, src_tasks, _ = src
    _, _, eps = tgt
    by_depth = defaultdict(lambda: [0, 0])
    first_miss = Counter()
    n_eps = full = steps = hits = 0
    succ_eps = succ_full = 0
    for task, succ, chain in eps:
        if task not in src_tasks or not chain:
            continue
        n_eps += 1
        miss = None
        for t, child in enumerate(chain):
            hit = child in src_nodes
            by_depth[min(t, max_depth)][0] += hit
            by_depth[min(t, max_depth)][1] += 1
            if not hit and miss is None:
                miss = t
        # a step is covered only if its whole prefix is, so hits = steps before the first miss
        k = len(chain) if miss is None else miss
        steps += len(chain)
        hits += k
        full += miss is None
        first_miss[min(k, max_depth)] += 1
        succ_eps += succ
        succ_full += succ and miss is None
    return {
        "episodes": n_eps,
        "shared_tasks": len({e[0] for e in eps} & src_tasks),
        "step_coverage": round(hits / max(1, steps), 4),
        "episode_full_coverage": round(full / max(1, n_eps), 4),
        "successful_episodes": succ_eps,
        "successful_full_coverage": round(succ_full / max(1, succ_eps), 4),
        "by_depth": {d: round(c / n, 4) for d, (c, n) in sorted(by_depth.items()) if n},
        "covered_prefix_len": {d: round(c / n_eps, 4) for d, c in sorted(first_miss.items())},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", nargs="+", required=True, help="name=glob")
    a = p.parse_args()
    src = load(a.src)
    out = {"src_nodes": len(src[0]), "src_tasks": len(src[1]), "src_episodes": len(src[2])}
    for spec in a.tgt:
        name, pat = spec.split("=", 1)
        out[name] = coverage(src, load(pat))
    print(json.dumps(out, indent=1))
