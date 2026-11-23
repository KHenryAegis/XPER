"""Batched rollouts on tau2 telecom (solo), written to an ExperienceStore.

Architecture:
  * vLLM runs in the driver process and batches generation across episodes.
  * tau2 environments run in a *process* pool, created before CUDA init so
    forked children never inherit the parent's CUDA context. Threads do not work
    here: tau2's per-episode busy-wait thread thrashes the GIL.
  * Each sample gets its own sampling seed. A single global seed makes vLLM
    return near-identical text for identical prompts, which silently collapses
    pass@k onto pass@1.

Used for G1 (headroom: pass@1/pass@k) and as the source/target stores for G2
(cross-model coverage). Every env step here is a real call; reuse is measured
offline by scripts/coverage.py.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Self-contained launch: running `python scripts/rollout.py` puts scripts/ on sys.path, not the
# repo root, so `import xper` fails unless PYTHONPATH happens to be set.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from vllm import LLM, SamplingParams

from xper.actions import Action, parse_model_output
from xper.chat import render
from xper.pool import ProcEnvPool
from xper.store import ExperienceStore, Step

BAD_FORMAT = "Error: could not parse a tool call. Reply with exactly:\nAction:\n{\"tool\": \"<name>\", \"arguments\": {...}}"
MAX_BAD = 3


def load_env_module(env: str):
    if env == "tau":
        from xper.tau import split, system_prompt, FullTasksAgentGymEnv as EnvClass
    elif env == "math":
        from xper.mathenv import split, system_prompt, MathToolEnv as EnvClass
    else:
        raise ValueError(f"unknown env: {env}")
    return split, system_prompt, EnvClass


def stratified(ids: list[str], n: int, seed: int) -> list[str]:
    """n tasks with an equal quota per issue family ([mms_issue], ...), deterministic.

    telecom train_full is 88% mms_issue while test is 40%, so uniform sampling
    would teach mostly one family.
    """
    import random
    fams: dict[str, list[str]] = {}
    for t in sorted(ids):
        fams.setdefault(t.split("]")[0], []).append(t)
    rng = random.Random(seed)
    for v in fams.values():
        rng.shuffle(v)
    out, k = [], 0
    longest = max((len(v) for v in fams.values()), default=0)
    # stop once every family is exhausted (n may exceed the number of tasks)
    while len(out) < n and k < longest:
        for v in fams.values():
            if k < len(v) and len(out) < n:
                out.append(v[k])
        k += 1
    return sorted(out)


def ep_seed(base: int, key: str) -> int:
    return int(hashlib.sha256(f"{base}|{key}".encode()).hexdigest()[:8], 16)


def run(args):
    split, system_prompt, _ = load_env_module(args.env)
    domain = args.domain if hasattr(args, 'domain') and args.domain else "telecom"
    if args.env == "math":
        ids = split(args.split)
    else:
        ids = split(domain, args.split)
    if args.stratify and args.env == "math":
        import random
        ids = sorted(random.Random(args.seed).sample(ids, min(args.stratify, len(ids))))
    elif args.stratify:
        ids = stratified(ids, args.stratify, args.seed)
    if args.limit:
        ids = ids[: args.limit]
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        ids = ids[i::n]
    pairs = [(tid, k) for tid in ids for k in range(args.n)]
    store = ExperienceStore(args.out)

    # Pool first: forked children must not inherit a CUDA context.
    with ProcEnvPool(args.env_workers, start_method="fork") as pool:
        # bf16 explicitly: FSDP-saved checkpoints are fp32 on disk
        llm = LLM(args.model, max_model_len=args.max_len, gpu_memory_utilization=args.gpu_util,
                  enable_prefix_caching=True, tensor_parallel_size=args.tp, dtype="bfloat16")
        tok = llm.get_tokenizer()
        # Qwen: <|im_end|> (151645). Other families (Llama) stop on their own eos / eot ids.
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end == 151645:
            stop_strs, stop_ids = ["<|im_end|>"], [151645]
        else:
            stop_strs = []
            stop_ids = sorted({i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
                               if isinstance(i, int) and i >= 0 and i != tok.unk_token_id})

        block = ""
        test_ids = set(split("test") if args.env == "math" else split(domain, "test"))
        if args.icl_src:
            from xper.icl import examples_block, pick_examples
            keep = set(Path(args.icl_keep).read_text().split()) if args.icl_keep else None
            ex = pick_examples(args.icl_src, args.icl_k, args.icl_seed, exclude=test_ids, keep=keep)
            block = examples_block(ex, args.icl_obs_chars)
            print(f"ICL: {len(ex)} examples, {len(tok.encode(block))} tokens", flush=True)

        def sys_msg(tid):
            s = system_prompt(tid) if args.env == "math" else system_prompt(tid, domain)
            if block:
                from xper.icl import with_examples
                s = with_examples(s, block)
            return s

        eps = {}
        for tid, k in pairs:
            eid = f"{tid}|{k}"
            eps[eid] = {"tid": tid, "k": k, "done": False, "terminated": False, "t": 0, "n_bad": 0,
                        "env_calls": 0, "reward": 0.0, "steps": [], "domain": domain,
                        "msgs": [{"role": "system", "content": sys_msg(tid)}]}
        if args.env == "math":
            specs = [(eid, {"kind": args.env, "task_id": e["tid"], "max_steps": args.max_steps})
                     for eid, e in eps.items()]
        else:
            specs = [(eid, {"kind": "tau_solo", "task_id": e["tid"], "domain": e["domain"],
                            "max_steps": args.max_steps})
                     for eid, e in eps.items()]
        t0 = time.perf_counter()
        reset_res = pool.reset(specs)
        bad = {eid: r[0] for eid, r in reset_res.items() if str(r[0]).startswith("ERROR")}
        if bad:
            eid, msg = next(iter(bad.items()))
            raise SystemExit(f"{len(bad)}/{len(specs)} resets failed, e.g. {eid}: {msg}")
        print(f"reset {len(specs)} episodes in {time.perf_counter()-t0:.1f}s", flush=True)

        timing = {"gen": 0.0, "env": 0.0, "turns": 0, "gens": 0, "hit_max_tokens": 0}
        while True:
            live = [(eid, e) for eid, e in eps.items() if not e["done"]]
            if not live:
                break

            if args.chatml:
                prompts = [render(e["msgs"], no_think=args.no_think) for _, e in live]
            else:
                prompts = [tok.apply_chat_template(e["msgs"], tokenize=False, add_generation_prompt=True,
                                                   **({"enable_thinking": False} if args.no_think else {}))
                           for _, e in live]
            # A context that no longer fits (long ICL demos + a long episode) would kill the vLLM engine;
            # end such episodes here like a max-steps truncation (finalized and scored below).
            fits = [len(tok.encode(pr, add_special_tokens=False)) < args.max_len for pr in prompts]
            if not all(fits):
                for (_, e), ok in zip(live, fits):
                    if not ok:
                        e["done"] = True
                        timing["context_full"] = timing.get("context_full", 0) + 1
                live = [x for x, ok in zip(live, fits) if ok]
                prompts = [pr for pr, ok in zip(prompts, fits) if ok]
                if not live:
                    continue
            # <|im_end|> stop: Base checkpoints' eos is <|endoftext|>, so they would run on otherwise.
            # stop_token_ids ensures vLLM stops even if the model never emits the string "<|im_end|>".
            sps = [SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens,
                                  stop=stop_strs, stop_token_ids=stop_ids,
                                  seed=ep_seed(args.seed, eid)) for eid, _ in live]
            t = time.perf_counter()
            outs = llm.generate(prompts, sps, use_tqdm=False)
            timing["gen"] += time.perf_counter() - t
            # A model that never emits <|im_end|> runs every turn to max_tokens (seen with LoRA on Base:
            # the untrained <|im_end|> lm_head row cannot be raised by LoRA). Surface it as a metric.
            timing["gens"] += len(outs)
            timing["hit_max_tokens"] += sum(o.outputs[0].finish_reason == "length" for o in outs)

            actions, acts, local = {}, {}, {}
            for (eid, e), out in zip(live, outs):
                text = out.outputs[0].text
                e["msgs"].append({"role": "assistant", "content": text})
                e["t"] += 1
                act = parse_model_output(text)
                if act is None or not act.is_tool or act.tool == "done":
                    if act is not None and act.tool == "done":
                        actions[eid], acts[eid] = act.to_tau2(), act   # terminates + scores the episode
                    else:
                        e["n_bad"] += 1
                        if args.dump_bad and timing.setdefault("dumped", 0) < args.dump_bad:
                            timing["dumped"] += 1
                            print(f"--- bad format ({out.outputs[0].finish_reason}, "
                                  f"{len(out.outputs[0].token_ids)} tok):\n{text[:1500]}\n---", flush=True)
                        if e["n_bad"] >= MAX_BAD or e["t"] >= args.max_steps:
                            stop = Action.tool_call("done")
                            actions[eid], acts[eid] = stop.to_tau2(), stop
                        else:
                            local[eid] = BAD_FORMAT
                else:
                    actions[eid], acts[eid] = act.to_tau2(), act

            t = time.perf_counter()
            res = pool.step(actions) if actions else {}
            timing["env"] += time.perf_counter() - t
            timing["turns"] += 1

            for eid, e in live:
                if eid in res:
                    obs, reward, done = res[eid]
                    e["env_calls"] += 1
                    e["steps"].append(Step(acts[eid], obs[: args.obs_chars]))
                    e["reward"] = reward
                    if done:
                        e["terminated"] = e["done"] = True
                    elif e["t"] >= args.max_steps:
                        e["done"] = True   # env left open; finalized below so it still gets scored
                    elif not e["msgs"][-1]["content"].startswith("Error: could not parse"):
                        e["msgs"].append({"role": "user", "content": f"Observation:\n{obs[: args.obs_chars]}"})
                else:
                    e["msgs"].append({"role": "user", "content": local[eid]})
                    if e["t"] >= args.max_steps:
                        e["done"] = True
            print(f"[{time.perf_counter()-t0:6.0f}s] live {len(live):4d} -> "
                  f"{sum(not e['done'] for e in eps.values()):4d} | gen {timing['gen']:5.1f}s "
                  f"env {timing['env']:5.1f}s", flush=True)

        # Finalize truncated episodes: ask the env to score them instead of assuming 0.
        open_eps = {eid: e for eid, e in eps.items() if not e["terminated"] and e["steps"]}
        if open_eps:
            res = pool.step({eid: Action.tool_call("done").to_tau2() for eid in open_eps})
            for eid, e in open_eps.items():
                if eid in res:
                    obs, reward, done = res[eid]
                    e["env_calls"] += 1
                    e["steps"].append(Step(Action.tool_call("done"), obs[: args.obs_chars]))
                    e["reward"] = reward
                    e["terminated"] = True
            print(f"finalized {len(open_eps)} truncated episodes", flush=True)

    rows = []
    for eid, e in eps.items():
        store.add_trajectory(e["tid"], e["steps"], e["reward"], e["reward"] >= 1.0, args.model_id,
                             args.policy_version, meta={"k": e["k"], "bad_format": e["n_bad"]})
        rows.append({"task": e["tid"], "k": e["k"], "reward": e["reward"], "steps": len(e["steps"]),
                     "env_calls": e["env_calls"], "bad_format": e["n_bad"]})
    Path(args.out).with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    if args.save_convs:
        # exact sampled conversations (incl. bad-format turns), for on-policy RL updates
        with open(args.save_convs, "w") as f:
            for e in eps.values():
                f.write(json.dumps({"task": e["tid"], "k": e["k"], "reward": e["reward"], "n_prefix": 0,
                                    "messages": e["msgs"]}, ensure_ascii=False) + "\n")
    summarize(rows, args)
    print(json.dumps({"timing": {k: round(v, 1) for k, v in timing.items()}}))


def summarize(rows, args):
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["reward"] >= 1)
    n = len(by_task)
    p1 = sum(sum(v) / len(v) for v in by_task.values()) / n
    pk = sum(any(v) for v in by_task.values()) / n
    mixed = sum(0 < sum(v) < len(v) for v in by_task.values())
    print(json.dumps({"model": args.model_id, "split": args.split, "tasks": n, "n": args.n,
                      "pass@1": round(p1, 4), f"pass@{args.n}": round(pk, 4), "mixed_tasks": mixed,
                      "env_calls": sum(r["env_calls"] for r in rows),
                      "mean_steps": round(sum(r["steps"] for r in rows) / len(rows), 2),
                      "bad_format_rate": round(sum(r["bad_format"] for r in rows) /
                                               max(1, sum(r["steps"] + r["bad_format"] for r in rows)), 4)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--env", choices=["tau", "math"], default="tau",
                   help="tau: tau2 solo mode (--domain); math: DAPO-Math with a sandboxed python tool")
    p.add_argument("--domain", default="telecom", help="tau2 domain (telecom, retail, airline, etc.)")
    p.add_argument("--policy-version", default="init")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stratify", type=int, default=0, help="sample N tasks, equal quota per issue family")
    p.add_argument("--shard", default="", help="i/n: keep every n-th task starting at i")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--max-tokens", type=int, default=512, help="per-turn generation cap (math: ~1024 for code)")
    p.add_argument("--dump-bad", type=int, default=0, help="print the raw text of the first N unparseable outputs")
    p.add_argument("--save-convs", default="", help="write every episode's sampled conversation + reward (jsonl)")
    p.add_argument("--max-len", type=int, default=32768)
    p.add_argument("--obs-chars", type=int, default=4000)
    p.add_argument("--gpu-util", type=float, default=0.8)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--env-workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--chatml", action="store_true",
                   help="render prompts with xper.chat.render (identical to bc_train) instead of HF template")
    p.add_argument("--icl-src", default="", help="few-shot baseline: glob of source stores to draw examples from")
    p.add_argument("--icl-k", type=int, default=6)
    p.add_argument("--icl-seed", type=int, default=0)
    p.add_argument("--icl-obs-chars", type=int, default=400)
    p.add_argument("--icl-keep", default="", help="file of task ids: draw ICL examples only from these tasks")
    p.add_argument("--out", required=True)
    run(p.parse_args())