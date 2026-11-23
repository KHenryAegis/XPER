"""ReLIFT-style interleaved SFT buffer for one GRPO iteration (Ma et al., arXiv 2506.07527).

Official recipe (TheRoadQaQ/ReLIFT, verl/relift/trainer.py): every task whose whole group failed is appended to a
buffer (one entry per occurrence); once the buffer holds sft_data_size entries, one SFT update on demonstrations for
the first sft_data_size entries is run and those entries are popped. Here the demonstration is a stored source
success for the task, and the cumulative number of SFT sequences (exposures) is capped by --cap; the last batch is
shrunk to the remaining cap so the cap is met exactly.

State: --state JSON {"buffer": [task, ...], "exposures": int}. Output: --out jsonl {"task", "weight": 1.0,
"messages"} for bc_train.py --weighted (constant lr, plain CE), written only when an SFT step is due.
Prints {"all_fail": n, "buffered": len(buffer), "sft_seqs": n, "exposures": total}.
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--convs", nargs="+", required=True)
p.add_argument("--src", required=True, help="glob of source stores")
p.add_argument("--state", required=True)
p.add_argument("--out", required=True)
p.add_argument("--size", type=int, default=64, help="sft_data_size: buffer entries per SFT step")
p.add_argument("--cap", type=int, default=100, help="cumulative SFT exposures; -1 = unlimited")
p.add_argument("--obs-chars", type=int, default=4000)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bc_train import build_conversations

groups = collections.defaultdict(list)
for path in a.convs:
    for line in open(path):
        r = json.loads(line)
        groups[r["task"]].append(r["reward"])
src = collections.defaultdict(list)
for c, t in zip(*build_conversations(a.src, a.obs_chars)):
    src[t].append(c)

state_p = Path(a.state)
state = json.loads(state_p.read_text()) if state_p.exists() else {"buffer": [], "exposures": 0}
all_fail = sorted(t for t, rs in groups.items() if not any(r > 0 for r in rs))
left = a.cap - state["exposures"] if a.cap >= 0 else None
if left is None or left > 0:
    state["buffer"] += [t for t in all_fail if t in src]  # a task without a demonstration cannot be buffered
n = a.size if left is None else min(a.size, left)
batch = []
if n > 0 and len(state["buffer"]) >= n:
    batch, state["buffer"] = state["buffer"][:n], state["buffer"][n:]
    rng = random.Random(a.seed)
    with open(a.out, "w") as f:
        for t in batch:
            f.write(json.dumps({"task": t, "weight": 1.0, "messages": rng.choice(src[t])}, ensure_ascii=False) + "\n")
    state["exposures"] += len(batch)
if left is not None and state["exposures"] >= a.cap:
    state["buffer"] = []
state_p.write_text(json.dumps(state))
print(json.dumps({"all_fail": len(all_fail), "buffered": len(state["buffer"]), "sft_seqs": len(batch),
                  "exposures": state["exposures"]}))
