"""Second environment: multi-turn math with a sandboxed python tool (deterministic, rule reward).

Problems: DAPO-Math-17k as packaged by rStar2-Agent (14,116 problems, integer answers).
Tools: ``python(code)`` runs a fresh interpreter and returns stdout/stderr;
``done(answer)`` submits an integer and ends the episode (reward 1 iff it matches).

Same ``reset``/``step`` interface as ``tau.TauSoloEnv`` so the rollout loop, process pool,
experience store and SFT code are shared unchanged. Sandbox: no network (unshare -n), CPU time,
address-space and output limits, private temp dir. Code is model-generated, so treat it as hostile.
"""

from __future__ import annotations

import json
import os
import random
import re
import resource
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLED_DATA = _ROOT / "data/external/math/train.jsonl"
DATA = Path(os.environ.get("XPER_MATH_DATA", str(_BUNDLED_DATA if _BUNDLED_DATA.exists()
            else Path(os.environ.get("XPER_HOME", str(_ROOT.parent))) / "datasets/rStar2-Agent/train.jsonl")))
N_TEST, SPLIT_SEED = 300, 0
TIMEOUT_S, MEM_BYTES, OUT_CHARS = 10, 4 << 30, 2000

INSTRUCTION = """You solve a competition math problem. You may run Python (sympy, numpy, scipy are available)
to compute or check results. Every reply must be exactly one tool call:

Action:
{"tool": "<tool name>", "arguments": {<arguments>}}

and nothing else. Each python call starts a fresh interpreter (no state is kept between calls) and
returns what the code prints. When you are confident, call "done" with the final integer answer."""

TOOLS = [
    {"name": "python", "description": "Run a Python 3 program in a fresh interpreter; returns stdout and stderr.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"name": "done", "description": "Submit the final answer (an integer) and finish.",
     "parameters": {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}},
]


@lru_cache(maxsize=1)
def problems() -> dict[str, dict]:
    out = {}
    for line in DATA.open():
        r = json.loads(line)
        i = r["extra_info"]["index"]
        out[f"math-{i}"] = {"question": r["extra_info"]["question"], "answer": str(r["reward_model"]["ground_truth"]).strip()}
    return out


def split(name: str) -> list[str]:
    extra = Path(__file__).resolve().parents[1] / "data/splits" / "math_extra_splits.json"
    if extra.exists() and name in (ex := json.loads(extra.read_text())):
        return ex[name]  # train tasks the source explored / solved (scripts: GRPO with injection)
    ids = sorted(problems(), key=lambda s: int(s.split("-")[1]))
    rng = random.Random(SPLIT_SEED)
    test = set(rng.sample(ids, N_TEST))
    if name == "test":
        return [i for i in ids if i in test]
    if name in ("train", "train_full"):
        return [i for i in ids if i not in test]
    raise KeyError(name)


def system_prompt(task_id: str) -> str:
    tool_txt = "\n".join(json.dumps(t, ensure_ascii=False) for t in TOOLS)
    return (f"<instructions>\n{INSTRUCTION}\n</instructions>\n<tools>\n{tool_txt}\n</tools>\n"
            f"<problem>\n{problems()[task_id]['question']}\n</problem>")


_PRELOADED = False


def _preload():
    """Import the heavy libraries once per env worker; forked children inherit them.
    Importing sympy/scipy from the shared filesystem costs 1-7 s per fresh interpreter."""
    global _PRELOADED
    if not _PRELOADED:
        import fractions, itertools, math  # noqa: F401
        import numpy, scipy, scipy.optimize, sympy  # noqa: F401
        _PRELOADED = True


def _child(code: str, d: str, out_fd: int):
    """Runs in the forked child: isolate, limit, execute, write output, never return."""
    try:
        strict = os.environ.get("XPER_STRICT_MATH") == "1"
        os.setsid()
        os.unshare(os.CLONE_NEWNET)  # no network: fresh namespace with only a down loopback
        os.chdir(d)
        # soft < hard so the kernel sends SIGXCPU at the limit (soft == hard sends SIGKILL here)
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S + 1))
        # budget on top of the inherited mappings: numpy/openblas alone reserve ~4.7 GB of address
        # space (79 MB resident), so a flat 4 GB cap made every sympy.solve raise MemoryError
        vm = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmSize")) << 10
        resource.setrlimit(resource.RLIMIT_AS, (vm + MEM_BYTES, vm + MEM_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        os.environ.clear()
        os.environ.update({"PATH": "/usr/bin:/bin", "HOME": d, "OMP_NUM_THREADS": "1"})
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0)
        os.dup2(out_fd, 1)
        os.dup2(out_fd, 2)
        os.closerange(3, 65536)  # drop inherited pool pipes, sqlite handles, sockets
        sys.stdout = open(1, "w", buffering=1, closefd=False)
        sys.stderr = open(2, "w", buffering=1, closefd=False)
        if strict:
            try:
                from .math_seccomp import install
                install()
            except BaseException as exc:
                print(f"[sandbox error] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                return  # fail closed; never execute model code without the requested filter
        g = {"__name__": "__main__", "__builtins__": __builtins__}
        try:
            exec(compile(code, "<python>", "exec"), g)
        except BaseException as e:
            import traceback
            sys.stdout.flush()
            # show the model only frames of its own code, plus the exception line
            src = code.splitlines()  # exec'd source is not in linecache, so take lines from it
            tb = [f for f in traceback.extract_tb(e.__traceback__, limit=None) if f.filename == "<python>"]
            lines = [f"  line {f.lineno}: {src[f.lineno - 1].strip()}" for f in tb[-3:]
                     if f.lineno and 0 < f.lineno <= len(src)]
            if isinstance(e, SyntaxError):
                lines.append(f"  line {e.lineno}: {(e.text or '').rstrip()}")
            msg = "".join(traceback.format_exception_only(type(e), e)).strip()
            print("[error]\n" + "\n".join(lines + [msg]), file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0)


def run_python(code: str) -> str:
    """Execute model code in a forked, network-less, resource-limited child; return its output."""
    import select
    import signal
    import time

    _preload()
    with tempfile.TemporaryDirectory(prefix="xper_math_") as d:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            _child(code, d, w)
        os.close(w)
        chunks, size, deadline, timed_out = [], 0, time.monotonic() + TIMEOUT_S + 2, False
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([r], [], [], left)
            if not ready:
                continue
            b = os.read(r, 65536)
            if not b:
                break
            if size < 4 * OUT_CHARS:
                chunks.append(b)
            size += len(b)
        os.close(r)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _, status = os.waitpid(pid, 0)
    out = b"".join(chunks).decode("utf-8", "replace").strip()
    killed = os.WIFSIGNALED(status) and os.WTERMSIG(status) in (signal.SIGXCPU, signal.SIGKILL)
    if timed_out or killed:
        out = (out + "\n" if out else "") + f"[timeout after {TIMEOUT_S}s]"
    out = out or "[no output; use print()]"
    return out[:OUT_CHARS] + (" ...[truncated]" if len(out) > OUT_CHARS else "")


def _as_int(x) -> str | None:
    s = str(x).strip().replace(",", "")
    m = re.fullmatch(r"[+-]?\d+(?:\.0+)?", s)
    return str(int(float(s))) if m else None


class MathToolEnv:
    """One math episode. ``step`` takes a tau2-style action string {"name": ..., "arguments": ...}."""

    def __init__(self, task_id: str, max_steps: int = 12):
        self.task_id, self.max_steps = task_id, max_steps
        self.answer = problems()[task_id]["answer"]

    def reset(self) -> str:
        self.t = 0
        return ""

    def step(self, action: str):
        self.t += 1
        try:
            a = json.loads(action)
            name, args = a.get("name"), a.get("arguments") or {}
        except (json.JSONDecodeError, AttributeError):
            name, args = None, {}
        if name == "done":
            got = _as_int(args.get("answer", ""))
            return "", float(got is not None and got == _as_int(self.answer)), True, {}
        if name == "python":
            obs = "tool: " + run_python(str(args.get("code", "")))
        else:
            obs = f'tool: error: unknown tool "{name}". Available: python, done.'
        if self.t >= self.max_steps:
            return obs, 0.0, True, {}
        return obs, 0.0, False, {}
