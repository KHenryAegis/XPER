"""Count episodes / env calls / successes across ExperienceStore shards matching a glob."""

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from xper.store import ExperienceStore  # noqa: E402

pattern = sys.argv[1]
tot = calls = succ = 0
tasks, fam = set(), {}
for p in sorted(glob.glob(pattern)):
    try:
        rows = ExperienceStore(p).trajectories()
    except Exception as e:  # shard still being written
        print(f"skip {p}: {e}")
        continue
    for _id, task, _m, r, _s, _n, ec in rows:
        tot += 1
        calls += ec
        if r >= 1:
            succ += 1
            tasks.add(task)
            f = task.split("]")[0].strip("[") if task.startswith("[") else "all"  # math ids: no family
            fam[f] = fam.get(f, 0) + 1
print(json.dumps({"shards": len(glob.glob(pattern)), "episodes": tot, "env_calls": calls, "successes": succ,
                  "tasks_solved": len(tasks), "by_family": fam, "success_rate": round(succ / max(1, tot), 5)}))
