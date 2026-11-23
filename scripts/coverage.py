"""Gate G2: how much of a target model's experience is already in a source store?

For each target step (history, action), check whether the source trie contains
the same history and the same canonical action. Reported per depth, because
coverage can only shrink with depth (a step is covered only if its whole
prefix is). Also reports the fraction of target episodes fully covered, i.e.
episodes that would have cost zero env calls under exact reuse.
"""

import argparse
import json
from collections import defaultdict

from xper.store import ExperienceStore


def coverage(src_path: str, tgt_path: str, max_depth: int = 10):
    src = ExperienceStore(src_path)
    tgt = ExperienceStore(tgt_path)
    src_nodes = {r[0] for r in src.db.execute("SELECT node_id FROM nodes").fetchall()}
    src_tasks = {r[0] for r in src.db.execute("SELECT DISTINCT task_id FROM trajectories").fetchall()}

    by_depth = defaultdict(lambda: [0, 0])  # depth -> [covered, total]
    episodes = covered_eps = 0
    steps_total = steps_cov = 0
    for traj_id, task_id, *_ in tgt.trajectories():
        if task_id not in src_tasks:
            continue
        episodes += 1
        full = True
        for t, node_id, child, *_ in tgt.steps(traj_id):
            hit = child in src_nodes
            d = min(t, max_depth)
            by_depth[d][0] += hit
            by_depth[d][1] += 1
            steps_total += 1
            steps_cov += hit
            full &= hit
        covered_eps += full
    first3 = [by_depth[d] for d in range(3)]
    return {
        "episodes": episodes,
        "step_coverage": steps_cov / max(1, steps_total),
        "first3_coverage": sum(c for c, _ in first3) / max(1, sum(n for _, n in first3)),
        "episode_full_coverage": covered_eps / max(1, episodes),
        "by_depth": {d: round(c / n, 4) for d, (c, n) in sorted(by_depth.items()) if n},
        "src_nodes": len(src_nodes),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    a = p.parse_args()
    print(json.dumps(coverage(a.src, a.tgt), indent=1))
