"""Trajectory SFT (behavior cloning) of a target model on stored source experience.

One training sequence per successful source trajectory, rendered with
``xper.chat.render`` exactly as ``rollout.py --chatml`` renders prompts:

  system: real tau2 solo prompt (policy + tools + ticket, ~9.3k tokens)
  assistant: "Action:\\n{json}"   <- supervised, incl. <|im_end|>
  user: "Observation:\\n..."      <- masked
  ...

Only stored steps are replayed (format-error turns were never env steps, so
they are absent), which makes these cleaned demonstrations.

Full-parameter FSDP training, launched with torchrun:
  torchrun --nproc_per_node 4 scripts/bc_train.py --full --src 'results/src/4b-inst.shard*.db' \\
      --model .../Qwen3-8B-Base --out .../8b-xfer
"""

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

# Self-contained launch (torchrun puts scripts/ on sys.path, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_jsonl_conversations(path: str):
    """Pre-rendered conversations ({"messages": [...]} per line), e.g. data/external/common_sft.jsonl.

    Optional per line: "task" (task id), "weight" (per-sequence loss weight, e.g. a GRPO
    advantage; used with --weighted) and "injected" (an off-policy source trajectory; shaped by
    --off-policy-shaping)."""
    rows = [json.loads(line) for line in open(path)]
    convs = [r["messages"] for r in rows]
    tasks = [r.get("task", f"jsonl:{path}") for r in rows]
    weights = [float(r.get("weight", 1.0)) for r in rows]
    off = [bool(r.get("injected")) for r in rows]
    return convs, tasks, weights, off


def build_conversations(src_glob: str, obs_chars: int, only_success: bool = True,
                        shuffle_actions: bool = False, seed: int = 0):
    """Read stored trajectories into chat conversations.

    ``shuffle_actions`` is the **format control**: the same trajectories, same
    multiset of actions, same observations in the same order, but the action
    order is permuted, so the state -> action mapping is destroyed while the
    format and the action/observation marginals are preserved. If training on
    permuted data matches training on the real order, the gain comes from
    learning the output format and task priors, not from experience content.
    """

    from xper import mathenv
    from xper.actions import Action
    from xper.store import ExperienceStore
    from xper.tau import split, system_prompt as tau_system_prompt

    def is_math(tid: str) -> bool:
        return tid.startswith("math-")

    def system_prompt(tid: str) -> str:
        return mathenv.system_prompt(tid) if is_math(tid) else tau_system_prompt(tid)

    rng = random.Random(seed)
    test = set(split("telecom", "test")) | set(mathenv.split("test"))
    convs, tasks = [], []  # tasks[i] is the task id of convs[i]
    for path in sorted(glob.glob(src_glob)):
        st = ExperienceStore(path)
        for traj_id, task_id, _m, reward, _s, _n, _e in st.trajectories():
            if only_success and reward < 1.0:
                continue
            if task_id in test:
                raise SystemExit(f"refusing to train on a test task: {task_id} ({path})")
            rows = list(st.steps(traj_id))
            acts = [Action.from_json(r[3]) for r in rows]
            if shuffle_actions:
                perm = list(range(len(acts)))
                rng.shuffle(perm)
                acts = [acts[i] for i in perm]
            msgs = [{"role": "system", "content": system_prompt(task_id)}]
            for act, row in zip(acts, rows):
                obs = row[4]
                msgs.append({"role": "assistant", "content": act.to_text()})
                if act.tool != "done":
                    msgs.append({"role": "user", "content": f"Observation:\n{(obs or '')[:obs_chars]}"})
            if msgs[-1]["role"] == "user":
                msgs.pop()
            convs.append(msgs)
            tasks.append(task_id)
    return convs, tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="glob of source ExperienceStore .db files, or a .jsonl of pre-rendered conversations")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=16384)
    p.add_argument("--obs-chars", type=int, default=4000)
    p.add_argument("--dry-run", action="store_true", help="build + check data, no training")
    p.add_argument("--full", action="store_true", required=True,
                   help="full-parameter FSDP training, the only mode (launch with torchrun)")
    p.add_argument("--max-trajs", type=int, default=0, help="seeded subsample size for the scaling experiment")
    p.add_argument("--subsample-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-think", action="store_true",
                   help="DISABLED: puts the empty think block only on the last (usually `done`) turn, "
                        "so the model learns 'block -> done' and quits early at --no-think eval. "
                        "Train bare; eval with rollout --no-think (Pipeline B2 seeds recipe).")
    p.add_argument("--format-control", action="store_true",
                   help="train on action-order-permuted trajectories (decides format vs experience)")
    p.add_argument("--weighted", action="store_true",
                   help="policy-gradient update: scale each sequence's CE by its jsonl 'weight' (advantage)")
    p.add_argument("--off-policy-shaping", type=float, default=0.0,
                   help="LUFFY policy shaping (Yan et al. 2025, mix_core_alg.py p_div_p_GAMMA): on jsonl rows with "
                        "'injected', the per-token -log p is replaced by -p/(p+GAMMA), no clip; 0 = off")
    a = p.parse_args()
    if a.no_think:
        p.error("--no-think trains a think-block -> done shortcut (B2 transfer 0%, ablation p50 0.3%); "
                "train bare and evaluate with rollout.py --no-think")
    import torch
    from transformers import AutoTokenizer

    from xper.chat import tokenize_for_training

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.src.endswith(".jsonl"):
        convs, tasks, weights, off = load_jsonl_conversations(a.src)
    else:
        convs, tasks = build_conversations(a.src, a.obs_chars, shuffle_actions=a.format_control, seed=a.seed)
        weights = [1.0] * len(convs)
        off = [False] * len(convs)
    if a.off_policy_shaping and not a.weighted:
        p.error("--off-policy-shaping needs --weighted")
    if a.weighted and not a.src.endswith(".jsonl"):
        p.error("--weighted needs a .jsonl with per-line weights (scripts/grpo_advantages.py)")
    if a.max_trajs:
        if not 0 < a.max_trajs <= len(convs):
            p.error("--max-trajs must fit the available trajectories")
        keep = random.Random(a.subsample_seed).sample(range(len(convs)), a.max_trajs)
        convs, tasks, weights, off = ([values[i] for i in keep] for values in (convs, tasks, weights, off))
    data = [dict(tokenize_for_training(c, tok, a.max_len, no_think=a.no_think), weight=w, off_policy=o)
            for c, w, o in zip(convs, weights, off)]
    data = [d for d in data if any(x != -100 for x in d["labels"])]  # truncation can drop all targets
    n_sup = [sum(x != -100 for x in d["labels"]) for d in data]
    lens = [len(d["input_ids"]) for d in data]
    truncated = sum(len(tokenize_for_training(c, tok, 10**9, no_think=a.no_think)["input_ids"]) > a.max_len
                    for c in convs)
    stats = {"trajectories": len(data), "tasks": len(set(tasks)),"tokens_mean": sum(lens) // max(1, len(lens)),
             "tokens_max": max(lens, default=0), "supervised_tokens": sum(n_sup), "truncated": truncated}
    print(json.dumps(stats), flush=True)
    assert data and min(n_sup) > 0, "some trajectory has no supervised tokens"
    if a.dry_run:
        ex = data[0]
        sup = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
        print("supervised text of example 0:\n" + sup[:600])
        return

    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch):
        # Batch size 1 by design: sequences are ~10-13k tokens. The LM head is applied only at
        # positions whose next token is supervised (~3% of positions): full logits at 13k x 152k
        # vocab in fp32 alone need ~40 GB.
        assert len(batch) == 1, "per_device_train_batch_size must be 1"
        ids = torch.tensor(batch[0]["input_ids"]).unsqueeze(0)
        lab = torch.tensor(batch[0]["labels"])
        keep = (lab[1:] != -100).nonzero().squeeze(-1)  # position p predicts token p+1
        # No attention_mask: one unpadded sequence, so SDPA takes the causal fused kernel.
        # An explicit mask makes HF build a 4D mask and SDPA falls back to O(n^2) memory.
        out = {"input_ids": ids, "logits_to_keep": keep, "targets": lab[keep + 1],
               "weight": torch.tensor(batch[0]["weight"], dtype=torch.float32),
               "off_policy": torch.tensor(bool(batch[0]["off_policy"]))}
        return out

    class SupervisedOnlyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            targets = inputs.pop("targets")
            weight = inputs.pop("weight")
            off_policy = bool(inputs.pop("off_policy"))
            out = model(**inputs)
            logits = out.logits.view(-1, out.logits.size(-1)).float()
            if off_policy and a.off_policy_shaping:
                # LUFFY off-policy token loss -A * f(p), f(p) = p / (p + gamma); weight (A) applied below
                prob = torch.exp(-torch.nn.functional.cross_entropy(logits, targets.to(logits.device), reduction="none"))
                loss = -(prob / (prob + a.off_policy_shaping)).mean()
            else:
                loss = torch.nn.functional.cross_entropy(logits, targets.to(logits.device))
            if a.weighted:
                # policy gradient: minimize -A * mean_t log p(a_t) == A * CE (A may be negative)
                loss = weight.to(loss.device) * loss
            return (loss, out) if return_outputs else loss

    if "LOCAL_RANK" not in os.environ:
        raise SystemExit("--full needs torchrun (FSDP), e.g. torchrun --nproc_per_node 4 scripts/bc_train.py --full ...")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    # Full-parameter SFT, FSDP full_shard. Optimizer states fp32 via Trainer (bf16 mixed precision).
    fsdp_kw = {"fsdp": "full_shard auto_wrap",
               "fsdp_config": {"transformer_layer_cls_to_wrap": [type(model.model.layers[0]).__name__],
                               "use_orig_params": True,
                               "state_dict_type": "FULL_STATE_DICT"}}
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=1,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr,
        # RL updates: a constant lr per iteration (a cosine/warmup schedule would restart every step)
        lr_scheduler_type="constant" if a.weighted else "cosine",
        warmup_ratio=0.0 if a.weighted else 0.05, bf16=True, logging_steps=5, save_strategy="no", report_to=[],
        group_by_length=True, dataloader_num_workers=2, remove_unused_columns=False,
        weight_decay=0.0, seed=a.seed,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        **fsdp_kw)
    trainer = SupervisedOnlyTrainer(model=model, args=args, train_dataset=data, data_collator=collate)
    trainer.train()

    final = Path(a.out) / "final"
    trainer.save_model(final)  # consolidated HF weights (vLLM-loadable)
    if trainer.is_world_process_zero():
        tok.save_pretrained(final)
        meta = {**stats, "src": a.src, "model": a.model, "mode": "full",
                "lr": a.lr, "epochs": a.epochs, "grad_accum": a.grad_accum,
                "world_size": int(os.environ.get("WORLD_SIZE", 1)), "seed": a.seed}
        (Path(a.out) / "data_stats.json").write_text(json.dumps(meta))
        print(json.dumps({"saved": str(final), **meta}))
    # release NCCL/CUDA state explicitly; skipping this left GPU memory held after exit
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
