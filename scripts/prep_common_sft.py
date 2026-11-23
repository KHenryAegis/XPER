"""Build the common agent SFT set from ToolACE: leak-free, same surface form as tau2 rollouts.

What it teaches: ChatML, reading a tool list, emitting exactly one
``Action:\\n{"tool": ..., "arguments": {...}}`` per turn, reading ``Observation:`` turns,
and answering in plain text. What it must not teach: anything about tau2 telecom.

Conversion per ToolACE conversation:
  * system  -> generic agent instruction + one JSON tool schema per line (same layout as
               xper.tau.system_prompt, different tools)
  * assistant "[F(a=1), G(b='x')]" -> one Action turn per call; the matching tool results
               are split into one Observation turn each (calls and results are matched by order)
  * assistant plain text -> kept as plain text
Conversations are dropped if a call cannot be parsed, call/result counts disagree, or any
telecom leak term / tau2 tool name appears anywhere.

Output: data/external/common_sft.jsonl, one {"messages": [...]} per line, plus data/external/common_sft.stats.json.
"""

import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

INSTRUCTION = """You are an agent that completes the user's request with the tools listed in <tools>.
To call a tool, reply with exactly one tool call and nothing else:

Action:
{"tool": "<tool name>", "arguments": {<arguments>}}

Tool results come back as "Observation:". When no tool is needed, reply to the user in plain text."""

LEAK_TERMS = ["mms", "apn", "roaming", "sim card", "simcard", "airplane mode", "wifi calling",
              "wi-fi calling", "mobile data", "data saver", "network mode", "cellular", "telecom",
              "signal strength", "reboot device", "data plan", "phone line"]


def tau2_tool_names() -> set[str]:
    from tau2.registry import registry

    env = registry.get_env_constructor("telecom")(solo_mode=True)
    names = {t.name for t in env.get_tools()}
    if env.user_tools:
        names |= {t.name for t in env.get_user_tools()}
    return {n.lower() for n in names}


def split_calls(s: str) -> list[str] | None:
    """'[F(a=1), G(b="x, y")]' -> ['F(a=1)', 'G(b="x, y")'] (top-level commas only)."""
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    body, out, depth, quote, cur = s[1:-1], [], 0, None, ""
    for i, ch in enumerate(body):
        if quote:
            cur += ch
            if ch == quote and body[i - 1] != "\\":
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out or None


def parse_call(c: str) -> tuple[str, dict] | None:
    m = re.match(r"^([^()]+?)\s*\((.*)\)$", c, re.DOTALL)
    if not m:
        return None
    name, args = m.group(1).strip(), m.group(2)
    try:
        node = ast.parse(f"f({args})", mode="eval").body
        # nested parens in the call text can make the parse something other than a Call
        if not isinstance(node, ast.Call) or node.args or any(k.arg is None for k in node.keywords):
            return None
        return name, {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None


def pydantic_style(schema: dict) -> dict:
    """Render a ToolACE JSON-schema the way tau2 renders its Pydantic tool schemas.

    Without this, the common-SFT model has never seen a schema with `title`/`$defs`/`$ref`,
    and at tau2 time it copies those keys into its own action (`{"tool": ..., "$defs": ...}`)
    instead of `arguments`. This changes the *surface form* only; no tau2 knowledge is added.
    """
    def fix(node):
        if isinstance(node, list):
            return [fix(x) for x in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for k, v in node.items():
            if k == "type" and v == "dict":
                out[k] = "object"
            else:
                out[k] = fix(v)
        if out.get("type") == "object" and "properties" in out:
            for name, prop in out["properties"].items():
                if isinstance(prop, dict) and "title" not in prop:
                    prop["title"] = name.replace("_", " ").title()
            out.setdefault("title", "parameters")
        return out

    return fix(schema)


def tools_block(system: str) -> str | None:
    # The prompt also contains "[func1(...)]" format text after the schema list, so decode
    # only the first JSON array instead of slicing to the last "]".
    i = system.find("[{")
    if i < 0:
        return None
    try:
        tools, _ = json.JSONDecoder().raw_decode(system[i:])
    except json.JSONDecodeError:
        return None
    if not isinstance(tools, list) or not all(isinstance(t, dict) and "name" in t for t in tools):
        return None
    return "\n".join(json.dumps(pydantic_style(t), ensure_ascii=False) for t in tools)


def action_text(name: str, args: dict) -> str:
    return "Action:\n" + json.dumps({"tool": name, "arguments": args}, ensure_ascii=False, indent=2)


def convert(conv: dict):
    tools = tools_block(conv["system"])
    if tools is None:
        return None, "bad_tools"
    msgs = [{"role": "system", "content": f"<instructions>\n{INSTRUCTION}\n</instructions>\n<tools>\n{tools}\n</tools>"}]
    pending = []  # calls awaiting results
    for t in conv["conversations"]:
        role, val = t["from"], t["value"]
        if role == "user":
            if pending:
                return None, "unanswered_call"
            msgs.append({"role": "user", "content": val})
        elif role == "assistant":
            calls = split_calls(val)
            if calls is None:
                if pending:
                    return None, "unanswered_call"
                msgs.append({"role": "assistant", "content": val})
                continue
            parsed = [parse_call(c) for c in calls]
            if any(p is None for p in parsed):
                return None, "unparsable_call"
            pending = parsed
        elif role == "tool":
            try:
                results = json.loads(val)
            except json.JSONDecodeError:
                return None, "bad_result"
            if not isinstance(results, list) or len(results) != len(pending):
                return None, "call_result_mismatch"
            for (name, args), res in zip(pending, results):
                msgs.append({"role": "assistant", "content": action_text(name, args)})
                msgs.append({"role": "user", "content": "Observation:\n" + json.dumps(res, ensure_ascii=False)})
            pending = []
        else:
            return None, f"role_{role}"
    # 8,712/11,300 ToolACE conversations end on the agent's final call with no result. That call
    # is still a valid supervised target ("pick the right tool + args"); keep the first one only,
    # since the format is one call per turn and later calls would follow an observation we lack.
    if pending:
        name, args = pending[0]
        msgs.append({"role": "assistant", "content": action_text(name, args)})
    while msgs[-1]["role"] != "assistant":
        msgs.pop()
        if len(msgs) == 1:
            return None, "no_assistant"
    return msgs, None


def main():
    data = json.load(open("data/external/toolace.json"))
    tau_names = tau2_tool_names()
    kept, why = [], Counter()
    for conv in data:
        blob = json.dumps(conv, ensure_ascii=False).lower()
        if any(w in blob for w in LEAK_TERMS):
            why["leak_term"] += 1
            continue
        if any(re.search(rf"\b{re.escape(n)}\b", blob) for n in tau_names):
            why["tau2_tool_name"] += 1
            continue
        msgs, err = convert(conv)
        if err:
            why[err] += 1
            continue
        kept.append(msgs)
    n_act = sum(m["content"].startswith("Action:") for c in kept for m in c if m["role"] == "assistant")
    n_txt = sum(not m["content"].startswith("Action:") for c in kept for m in c if m["role"] == "assistant")
    stats = {"source": len(data), "kept": len(kept), "dropped": dict(why),
             "assistant_action_turns": n_act, "assistant_text_turns": n_txt}
    with open("data/external/common_sft.jsonl", "w") as f:
        for c in kept:
            f.write(json.dumps({"messages": c}, ensure_ascii=False) + "\n")
    Path("data/external/common_sft.stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
    ex = next(c for c in kept if any(m["content"].startswith("Action:") for m in c))
    for m in ex[1:5]:
        print(f"--- {m['role']}\n{m['content'][:300]}")


if __name__ == "__main__":
    main()
