import sys

import pytest

if sys.platform != "linux":
    pytest.skip("the strict math sandbox needs Linux (unshare, libseccomp)", allow_module_level=True)
pytest.importorskip("scipy", reason="the math sandbox preloads SciPy (requirements.txt)")
from xper.mathenv import run_python  # noqa: E402


def test_strict_math_computes_and_denies_file_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("XPER_STRICT_MATH", "1")
    assert run_python("import sympy; x=sympy.Symbol('x'); print(sympy.solve(x*x-2,x))") == "[-sqrt(2), sqrt(2)]"
    target = tmp_path / "owned-probe"
    target.write_text("keep")
    assert "PermissionError" in run_python(f"open({str(target)!r}, 'w').write('changed')")
    assert "PermissionError" in run_python(f"import os; os.unlink({str(target)!r})")
    assert target.read_text() == "keep"


def test_strict_math_cannot_signal_or_spawn(monkeypatch):
    monkeypatch.setenv("XPER_STRICT_MATH", "1")
    # Signal 0 only probes permissions; it cannot terminate the parent even if
    # the filter regresses. The child must still reject this syscall.
    assert "PermissionError" in run_python("import os; os.kill(os.getppid(), 0)")
    assert "PermissionError" in run_python("import subprocess; subprocess.run(['true'])")
