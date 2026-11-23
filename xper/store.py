"""Persistent experience store: trajectories + a per-task prefix trie.

A node is an interaction history ``h_t`` identified by the sequence of
canonical actions taken from the task's initial state. In a deterministic
environment (tau2 solo mode, or a temperature-0 user simulator), the
observation after ``(h_t, a_t)`` is fixed, so any model reaching the same
history and choosing the same action can reuse the stored observation instead
of calling the environment. Per-node Monte-Carlo statistics (visits, mean
return) give a model-agnostic value estimate for every stored history.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .actions import Action

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    parent_id TEXT,
    action_json TEXT,
    depth INTEGER NOT NULL,
    observation TEXT,
    n_obs_variants INTEGER NOT NULL DEFAULT 0,
    visits INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 0,
    return_sum REAL NOT NULL DEFAULT 0,
    terminal_visits INTEGER NOT NULL DEFAULT 0,
    terminal_return_sum REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS nodes_parent ON nodes(parent_id);
CREATE TABLE IF NOT EXISTS trajectories (
    traj_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    reward REAL NOT NULL,
    success INTEGER NOT NULL,
    n_steps INTEGER NOT NULL,
    env_calls INTEGER NOT NULL,
    meta TEXT,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    traj_id INTEGER NOT NULL,
    t INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    action_json TEXT NOT NULL,
    observation TEXT,
    from_cache INTEGER NOT NULL,
    logprob_init REAL,
    logprob_rl REAL,
    PRIMARY KEY (traj_id, t)
);
CREATE INDEX IF NOT EXISTS steps_node ON steps(node_id);
"""


def root_id(task_id: str) -> str:
    return hashlib.sha1(f"root\x00{task_id}".encode()).hexdigest()[:20]


def child_id(parent: str, action: Action) -> str:
    return hashlib.sha1(f"{parent}\x00{action.key}".encode()).hexdigest()[:20]


@dataclass
class Step:
    action: Action
    observation: Optional[str]
    from_cache: bool = False
    logprob_init: Optional[float] = None
    logprob_rl: Optional[float] = None


@dataclass
class NodeStats:
    node_id: str
    action: Optional[Action]
    depth: int
    observation: Optional[str]
    n_obs_variants: int
    visits: int
    successes: int
    return_sum: float

    @property
    def value(self) -> float:
        return self.return_sum / self.visits if self.visits else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.visits if self.visits else 0.0

    @property
    def consistent(self) -> bool:
        """Every visit observed the same environment response."""
        return self.n_obs_variants <= 1


def _node(row) -> NodeStats:
    nid, aj, depth, obs, nvar, visits, succ, rsum = row
    return NodeStats(nid, Action.from_json(aj) if aj else None, depth, obs, nvar, visits, succ, rsum)


_NODE_COLS = "node_id, action_json, depth, observation, n_obs_variants, visits, successes, return_sum"


class _LockedConnection:
    """sqlite3 connection shared across rollout threads; every call is serialized."""

    def __init__(self, path: str, readonly: bool = False):
        if readonly:  # immutable: no locks, no -wal/-shm writes (safe on the shared FS for finished stores)
            self._conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
        else:
            self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()

    def execute(self, *a):
        with self._lock:
            return _Rows(self._conn.execute(*a))

    def executescript(self, s):
        with self._lock:
            return self._conn.executescript(s)

    def close(self):
        with self._lock:
            self._conn.close()

    def __enter__(self):
        self._lock.acquire()
        self._conn.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            return self._conn.__exit__(*exc)
        finally:
            self._lock.release()


class _Rows:
    """Materialized cursor result so rows can be read after the lock is released."""

    def __init__(self, cur):
        self.lastrowid = cur.lastrowid
        self._rows = cur.fetchall()

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class ExperienceStore:
    def __init__(self, path: str = ":memory:", readonly: bool = False):
        """readonly: open a finished store (e.g. a source store read by many shards at once) without writing."""
        self.db = _LockedConnection(path, readonly)
        if not readonly:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(_SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ---------------- writes ----------------

    def add_trajectory(
        self,
        task_id: str,
        steps: Sequence[Step],
        reward: float,
        success: bool,
        model_id: str,
        policy_version: str = "",
        meta: Optional[dict] = None,
    ) -> int:
        env_calls = sum(1 for s in steps if not s.from_cache)
        with self.db:
            cur = self.db.execute(
                "INSERT INTO trajectories (task_id, model_id, policy_version, reward, success,"
                " n_steps, env_calls, meta, created) VALUES (?,?,?,?,?,?,?,?,?)",
                (task_id, model_id, policy_version, reward, int(success), len(steps), env_calls,
                 json.dumps(meta or {}), time.time()),
            )
            traj_id = cur.lastrowid
            parent = root_id(task_id)
            self._touch(parent, task_id, None, None, 0, None, reward, success)
            for t, s in enumerate(steps):
                cid = child_id(parent, s.action)
                self._touch(cid, task_id, parent, s.action, t + 1, s.observation, reward, success)
                self.db.execute(
                    "INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?)",
                    (traj_id, t, parent, cid, s.action.canonical(), s.observation,
                     int(s.from_cache), s.logprob_init, s.logprob_rl),
                )
                parent = cid
            self.db.execute(
                "UPDATE nodes SET terminal_visits=terminal_visits+1, terminal_return_sum=terminal_return_sum+?"
                " WHERE node_id=?", (reward, parent),
            )
        return traj_id

    def terminal_reward(self, node_id: str) -> Optional[tuple[int, float]]:
        """(times an episode ended here, mean reward at termination), or None."""
        row = self.db.execute(
            "SELECT terminal_visits, terminal_return_sum FROM nodes WHERE node_id=?", (node_id,)
        ).fetchone()
        if not row or row[0] == 0:
            return None
        return row[0], row[1] / row[0]

    def _touch(self, nid, task_id, parent, action, depth, obs, reward, success):
        row = self.db.execute("SELECT observation FROM nodes WHERE node_id=?", (nid,)).fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,0,0)",
                (nid, task_id, parent, action.canonical() if action else None, depth, obs,
                 1 if obs is not None else 0, 1, int(success), reward),
            )
            return
        variant = 1 if (obs is not None and row[0] is not None and obs != row[0]) else 0
        self.db.execute(
            "UPDATE nodes SET visits=visits+1, successes=successes+?, return_sum=return_sum+?,"
            " n_obs_variants=n_obs_variants+?, observation=COALESCE(observation, ?) WHERE node_id=?",
            (int(success), reward, variant, obs, nid),
        )

    def set_logprobs(self, traj_id: int, t: int, logprob_init: Optional[float], logprob_rl: Optional[float]):
        with self.db:
            self.db.execute(
                "UPDATE steps SET logprob_init=?, logprob_rl=? WHERE traj_id=? AND t=?",
                (logprob_init, logprob_rl, traj_id, t),
            )

    # ---------------- reads ----------------

    def node(self, node_id: str) -> Optional[NodeStats]:
        row = self.db.execute(f"SELECT {_NODE_COLS} FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        return _node(row) if row else None

    def children(self, node_id: str) -> list[NodeStats]:
        rows = self.db.execute(
            f"SELECT {_NODE_COLS} FROM nodes WHERE parent_id=? ORDER BY visits DESC", (node_id,)
        ).fetchall()
        return [_node(r) for r in rows]

    def lookup(self, node_id: str, action: Action) -> Optional[NodeStats]:
        """Stored outcome of taking ``action`` at ``node_id``, if ever observed."""
        return self.node(child_id(node_id, action))

    def branch_nodes(self, min_children: int = 2) -> list[str]:
        """Histories where stored experience contains at least ``min_children`` distinct actions."""
        rows = self.db.execute(
            "SELECT parent_id FROM nodes WHERE parent_id IS NOT NULL GROUP BY parent_id HAVING COUNT(*)>=?",
            (min_children,),
        ).fetchall()
        return [r[0] for r in rows]

    def history(self, node_id: str) -> list[NodeStats]:
        """Path from (excluding) root to ``node_id``."""
        path = []
        cur = node_id
        while True:
            row = self.db.execute("SELECT parent_id FROM nodes WHERE node_id=?", (cur,)).fetchone()
            if row is None or row[0] is None:
                break
            path.append(self.node(cur))
            cur = row[0]
        return path[::-1]

    def trajectories(self, task_id: Optional[str] = None, success: Optional[bool] = None) -> Iterable[tuple]:
        q, args = "SELECT traj_id, task_id, model_id, reward, success, n_steps, env_calls FROM trajectories WHERE 1=1", []
        if task_id is not None:
            q += " AND task_id=?"
            args.append(task_id)
        if success is not None:
            q += " AND success=?"
            args.append(int(success))
        return self.db.execute(q, args).fetchall()

    def steps(self, traj_id: int) -> list[tuple]:
        return self.db.execute(
            "SELECT t, node_id, child_id, action_json, observation, from_cache, logprob_init, logprob_rl"
            " FROM steps WHERE traj_id=? ORDER BY t", (traj_id,)
        ).fetchall()

    def env_call_total(self, model_id: Optional[str] = None) -> int:
        q, args = "SELECT COALESCE(SUM(env_calls),0) FROM trajectories", []
        if model_id is not None:
            q += " WHERE model_id=?"
            args.append(model_id)
        return self.db.execute(q, args).fetchone()[0]
