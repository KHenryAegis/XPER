"""One ChatML renderer shared by rollout and training.

HF chat templates differ across Qwen3 checkpoints: the hybrid Qwen3-8B template
rewrites earlier assistant turns (think tags), so a prefix rendered at rollout
time is not a prefix of the rendered full conversation. Training on such text
teaches a context the policy never sees. Rendering ChatML ourselves keeps
rollout prompts and training sequences byte-identical for every Qwen3 tokenizer
(they share <|im_start|>/<|im_end|>). No <think> blocks are emitted.
"""

from __future__ import annotations

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
# Hybrid Qwen3 checkpoints (Qwen3-8B, Qwen3-4B-Instruct-2507) reason by default; their template
# opens the assistant turn with an empty think block when thinking is off. Without the same block
# here the model fills the whole budget with  thinking and never emits an action.
THINK_OPEN, THINK_CLOSE = "<" + "think>", "</" + "think>"
NO_THINK_BLOCK = f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"


def templates_think(tokenizer) -> bool:
    """Does this checkpoint's template emit a think block when thinking is disabled?"""
    try:
        text = tokenizer.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return False
    return THINK_OPEN in text


def render(msgs: list[dict], add_generation_prompt: bool = True, no_think: bool = False) -> str:
    # The official hybrid template adds the empty think block only to the *last* assistant turn
    # (which is the generation prompt); earlier assistant turns are rendered bare. Verified
    # against Qwen3-8B's own template in tests/test_chat.py.
    out = "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in msgs)
    if add_generation_prompt:
        out += f"{IM_START}assistant\n{NO_THINK_BLOCK if no_think else ''}"
    return out


def tokenize_for_training(msgs: list[dict], tokenizer, max_len: int, no_think: bool = False) -> dict:
    """input_ids + labels supervising every assistant turn (content + <|im_end|>).

    Assistant turns marked ``"prefix": True`` are rendered but not supervised.
    Built segment by segment so label boundaries are exact token boundaries. With ``no_think``
    the *final* assistant turn (the one being trained on) carries the empty think block, masked
    like the role header; earlier assistant turns are bare, matching ``render`` exactly.
    Truncation keeps the head (system prompt + early turns).
    """
    last_assistant = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)
    ids, labels = [], []
    for i, m in enumerate(msgs):
        head = f"{IM_START}{m['role']}\n"
        if no_think and i == last_assistant:
            head += NO_THINK_BLOCK
        head_ids = tokenizer.encode(head, add_special_tokens=False)
        body = tokenizer.encode(f"{m['content']}{IM_END}", add_special_tokens=False)
        tail = tokenizer.encode("\n", add_special_tokens=False)
        ids += head_ids + body + tail
        # "prefix" turns are replayed source actions (PrefixRL): context for the target, never trained on
        sup = m["role"] == "assistant" and not m.get("prefix")
        labels += [-100] * len(head_ids) + (body if sup else [-100] * len(body)) + [-100] * len(tail)
    return {"input_ids": ids[:max_len], "labels": labels[:max_len]}
