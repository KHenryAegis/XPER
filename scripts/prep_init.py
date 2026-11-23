"""Mean-initialize the ChatML special-token rows of a Qwen3 Base checkpoint.

Qwen3 Base checkpoints never trained <|im_start|>/<|im_end|>: their embedding and
lm_head rows have 0.23-0.31x the median row norm. After full-parameter BC
(lr 1e-5, 56 steps) the 8B lm_head <|im_end|> row moved by only 0.013 and the
model still assigned p~0.007 to <|im_end|> where it is the target (NLL 5.05 vs
0.011 on other action tokens), so it never stopped generating.

Setting these rows to the mean of the ordinary-token rows is the standard init
for new tokens (HF resize_token_embeddings does the same). Apply it to every
Base checkpoint before any training, so all methods share the same start.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHAT_TOKENS = ["<|im_start|>", "<|im_end|>"]

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--out", required=True)
p.add_argument("--mode", choices=["mean", "eos"], default="eos",
               help="mean: mean of ordinary rows; eos: copy the trained <|endoftext|> rows (an 'end' prior)")
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16)
ids = tok.convert_tokens_to_ids(CHAT_TOKENS)
n_ordinary = tok.convert_tokens_to_ids("<|endoftext|>")  # ids below this are ordinary BPE tokens

report = {}
with torch.no_grad():
    mats = {"embed_tokens": model.get_input_embeddings().weight}
    out_emb = model.get_output_embeddings().weight
    if out_emb.data_ptr() != mats["embed_tokens"].data_ptr():  # 4B ties them, 8B does not
        mats["lm_head"] = out_emb
    for name, W in mats.items():
        mean = W[:n_ordinary].float().mean(0)
        eos = W[n_ordinary].float()
        before = [W[i].float().norm().item() for i in ids]
        # eos mode: <|im_end|> inherits the trained end-of-text row ("stop here" prior);
        # <|im_start|> gets the mean so the two stay distinguishable as inputs (4B ties embed/lm_head).
        src = {ids[0]: mean, ids[1]: eos if a.mode == "eos" else mean}
        for i in ids:
            W[i] = src[i].to(W.dtype)
        report[name] = {"before_norm": before, "after_norm": [W[i].float().norm().item() for i in ids],
                        "median_norm": W[:n_ordinary].float().norm(dim=1).median().item()}

model.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
meta = {"base": a.base, "mode": a.mode, "tokens": dict(zip(CHAT_TOKENS, ids)), "tied": "lm_head" not in report, "rows": report}
Path(a.out, "xper_init.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta))
