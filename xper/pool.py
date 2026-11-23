"""Process-based environment pool.

tau2's gym env runs a busy-wait orchestrator thread per episode
(``Event.wait(0.01)`` in a loop). Under threads those pollers plus the driver
thrash the GIL: measured ~1 core of total work from 362 threads, and 320 episode
resets failed to finish in 16 minutes. Processes give real parallelism.

Episodes are pinned to a worker (``hash(eid) % n``) so each worker keeps the
state of its own episodes and no env object is ever passed between processes —
only canonical action JSON and observation strings cross the boundary.

Set ``MP_START=spawn`` when the parent has already initialized CUDA (vLLM), so
children never inherit the parent's CUDA context.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
from typing import Callable, Iterable, Optional


def _make_env(spec: dict):
    kind = spec["kind"]
    if kind == "tau_solo":
        from .tau import TauSoloEnv

        return TauSoloEnv(spec["task_id"], domain=spec.get("domain", "telecom"),
                          max_steps=spec.get("max_steps", 30), lean=spec.get("lean", True))
    if kind == "math":
        from .mathenv import MathToolEnv

        return MathToolEnv(spec["task_id"], max_steps=spec.get("max_steps", 12))
    if kind == "toy":
        from tests.test_core import ToyEnv

        return ToyEnv(spec["task_id"])
    raise ValueError(f"unknown env kind {kind}")


def _worker(req_q, resp_q, init_fn):
    if init_fn is not None:
        init_fn()
    envs: dict[str, object] = {}
    while True:
        msg = req_q.get()
        if msg is None:
            return
        batch_id, ops = msg
        out = []
        for eid, op, payload in ops:
            try:
                if op == "reset":
                    env = _make_env(payload)
                    envs[eid] = env
                    out.append((eid, env.reset(), 0.0, False))
                elif op == "step":
                    env = envs.get(eid)
                    if env is None:
                        out.append((eid, "ERROR: env missing", 0.0, True))
                        continue
                    obs, reward, done, _ = env.step(payload)
                    if done:
                        envs.pop(eid, None)
                    out.append((eid, obs, float(reward), bool(done)))
                else:
                    envs.pop(eid, None)
            except Exception as e:  # a broken episode must not kill the worker
                envs.pop(eid, None)
                out.append((eid, f"ERROR: {type(e).__name__}: {e}", 0.0, True))
        resp_q.put((batch_id, out))


class ProcEnvPool:
    """Parallel env driver. Use as a context manager."""

    def __init__(self, n: int, init_fn: Optional[Callable[[], None]] = None, start_method: Optional[str] = None):
        self.n = n
        ctx = mp.get_context(start_method or os.environ.get("MP_START", "fork"))
        self.req_qs, self.resp_q, self.procs = [], ctx.Queue(), []
        for _ in range(n):
            rq = ctx.Queue()
            p = ctx.Process(target=_worker, args=(rq, self.resp_q, init_fn), daemon=True)
            p.start()
            self.req_qs.append(rq)
            self.procs.append(p)
        self._batch = 0
        self._pending: dict[str, int] = {}
        self._results: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _worker_of(self, eid: str) -> int:
        return hash(eid) % self.n

    def submit(self, ops: Iterable[tuple[str, str, dict]]) -> None:
        """Queue ops for one turn; each op is (episode_id, op, payload)."""
        with self._lock:
            self._batch += 1
            bid = self._batch
            per_worker: dict[int, list] = {}
            for eid, op, payload in ops:
                per_worker.setdefault(self._worker_of(eid), []).append((eid, op, payload))
                self._pending[eid] = bid
            for w, batch in per_worker.items():
                self.req_qs[w].put((bid, batch))

    def gather(self) -> dict[str, tuple]:
        """Block until every submitted op returns; return {episode_id: (obs, reward, done)}."""
        with self._lock:
            n_pending = len(self._pending)
        while len(self._results) < n_pending:
            bid, out = self.resp_q.get()
            for eid, obs, reward, done in out:
                self._results[eid] = (obs, reward, done)
        with self._lock:
            res = {eid: self._results.pop(eid) for eid in self._pending}
            self._pending.clear()
        return res

    def reset(self, eids_specs: Iterable[tuple[str, dict]], chunk: int = 256) -> dict[str, tuple]:
        """Reset many episodes (specs are per-episode env dicts)."""
        out: dict[str, tuple] = {}
        specs = list(eids_specs)
        for i in range(0, len(specs), chunk):
            self.submit((eid, "reset", spec) for eid, spec in specs[i:i + chunk])
            out.update(self.gather())
        return out

    def step(self, actions: dict[str, str]) -> dict[str, tuple]:
        self.submit((eid, "step", a) for eid, a in actions.items())
        return self.gather()

    def close(self) -> None:
        for rq in self.req_qs:
            try:
                rq.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            if p.is_alive():
                p.kill()