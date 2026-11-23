"""Sandbox and scoring checks for xper.mathenv (run: python tests/test_mathenv.py)."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

pytest.importorskip("scipy", reason="the math sandbox preloads SciPy (requirements.txt)")
from xper.mathenv import DATA, MathToolEnv, _as_int, problems, run_python, split  # noqa: E402

needs_data = pytest.mark.skipif(not DATA.exists(), reason="needs data/external/math/train.jsonl")


def test_runs_code_with_preloaded_libs():
    assert run_python("print(2 + 2)") == "4"
    assert "{3: 1, 5: 1" in run_python("import sympy; print(sympy.factorint(2**64 - 1))")


def test_fresh_state_per_call():
    run_python("x = 41")
    assert "NameError" in run_python("print(x)")


def test_errors_are_reported():
    assert "ZeroDivisionError" in run_python("print(1/0)")
    assert "SyntaxError" in run_python("def(")


def test_timeout_and_cpu_limit():
    t = time.time()
    out = run_python("while True: pass")
    assert "[timeout" in out and time.time() - t < 15, out


def test_no_network():
    out = run_python("import socket; socket.create_connection(('8.8.8.8', 53), timeout=3); print('OK' + 'NET')")
    assert "OKNET" not in out and "Network is unreachable" in out, out


def test_memory_limit():
    out = run_python("x = bytearray(8 << 30); print('OK' + 'ALLOC')")
    assert "OKALLOC" not in out and "MemoryError" in out, out
    # ordinary symbolic work must fit (a flat 4 GB address-space cap broke this)
    assert run_python("import sympy; x = sympy.Symbol('x'); print(sympy.solve(x**2 - 2, x))") == "[-sqrt(2), sqrt(2)]"


def test_error_message_shows_user_code_only():
    out = run_python("def f(n):\n    return 1 / n\nprint(f(0))")
    assert "ZeroDivisionError" in out and "line 2:" in out and "mathenv.py" not in out, out


def test_output_truncated():
    out = run_python("print('a' * 100000)")
    assert len(out) < 2100 and out.endswith("[truncated]")


def test_inherited_fds_closed():
    out = run_python("import os; print(sorted(int(f) for f in os.listdir('/proc/self/fd')))")
    assert set(json.loads(out)) <= {0, 1, 2, 3}, out  # 3 = the listdir handle itself


def test_answer_parsing():
    assert _as_int("42") == "42" and _as_int("1,000") == "1000" and _as_int("7.0") == "7"
    assert _as_int("7.5") is None and _as_int("x") is None


@needs_data
def test_split_disjoint_and_scoring():
    test, train = set(split("test")), set(split("train"))
    assert len(test) == 300 and not test & train and len(test | train) == len(problems())
    tid = sorted(test)[0]
    env = MathToolEnv(tid, max_steps=3)
    env.reset()
    obs, r, done, _ = env.step(json.dumps({"name": "python", "arguments": {"code": "print(1)"}}))
    assert obs == "tool: 1" and r == 0 and not done
    ans = problems()[tid]["answer"]
    assert env.step(json.dumps({"name": "done", "arguments": {"answer": int(ans)}}))[1:3] == (1.0, True)
    env.reset()
    assert env.step(json.dumps({"name": "done", "arguments": {"answer": int(ans) + 1}}))[1] == 0.0


@needs_data
def test_step_cap():
    env = MathToolEnv(split("test")[0], max_steps=2)
    env.reset()
    env.step(json.dumps({"name": "python", "arguments": {"code": "print(1)"}}))
    assert env.step(json.dumps({"name": "python", "arguments": {"code": "print(1)"}}))[2] is True


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn()
            print(f"ok   {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {e!r}"[:300])
    sys.exit(1 if fails else 0)
