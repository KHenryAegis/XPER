"""Selective re-exploration: serve covered steps from the store, call the env otherwise.

``ReuseEnv`` wraps a real environment. When the agent's (history, action)
matches a stored, consistent transition, the stored observation is returned
and no environment call is made. On the first uncovered action, the real
environment is materialized by replaying the cached action prefix, then the
new action is executed for real. Replayed calls are counted as real calls, so
``real_calls`` is an honest interaction count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from .actions import Action
from .store import ExperienceStore, Step, child_id, root_id


class Env(Protocol):
    def reset(self) -> str: ...

    def step(self, action: str) -> tuple[str, float, bool, dict]: ...


@dataclass
class EpisodeLog:
    steps: list[Step] = field(default_factory=list)
    real_calls: int = 0  # env.step calls actually executed (new + replayed)
    replayed_calls: int = 0
    resets: int = 0
    reward: float = 0.0
    done: bool = False


class ReuseEnv:
    def __init__(
        self,
        store: ExperienceStore,
        make_env: Callable[[str], Env],
        task_id: str,
        reuse: bool = True,
        min_visits: int = 1,
    ):
        self.store = store
        self.make_env = make_env
        self.task_id = task_id
        self.reuse = reuse
        self.min_visits = min_visits

    # The initial observation is stored on the root node's ``observation`` column.
    def reset(self) -> str:
        self.log = EpisodeLog()
        self.node = root_id(self.task_id)
        self.pending: list[Action] = []
        self.real: Optional[Env] = None
        root = self.store.node(self.node)
        if self.reuse and root is not None and root.observation is not None:
            return root.observation
        return self._materialize()

    def _materialize(self) -> str:
        self.real = self.make_env(self.task_id)
        obs = self.real.reset()
        self.log.resets += 1
        for a in self.pending:
            obs, _, _, _ = self.real.step(a.to_tau2())
            self.log.real_calls += 1
            self.log.replayed_calls += 1
        self.pending = []
        return obs

    def _cached(self, action: Action):
        if not self.reuse:
            return None
        c = self.store.lookup(self.node, action)
        if c is None or c.visits < self.min_visits or not c.consistent or c.observation is None:
            return None
        return c

    def step(self, action: Action) -> tuple[str, float, bool, dict[str, Any]]:
        c = self._cached(action) if self.real is None else None
        if c is not None:
            term = self.store.terminal_reward(c.node_id)
            done = term is not None
            reward = term[1] if term else 0.0
            self.pending.append(action)
            self.node = c.node_id
            self.log.steps.append(Step(action, c.observation, from_cache=True))
            self._finish(reward, done)
            return c.observation, reward, done, {"from_cache": True}
        if self.real is None:
            self._materialize()
        obs, reward, done, info = self.real.step(action.to_tau2())
        self.log.real_calls += 1
        self.node = child_id(self.node, action)
        self.log.steps.append(Step(action, obs, from_cache=False))
        self._finish(reward, done)
        return obs, reward, done, {**info, "from_cache": False}

    def _finish(self, reward: float, done: bool):
        if done:
            self.log.reward, self.log.done = reward, True

    def commit(self, model_id: str, policy_version: str = "", initial_observation: Optional[str] = None,
               success_threshold: float = 1.0) -> int:
        """Write the finished episode to the store."""
        log = self.log
        tid = self.store.add_trajectory(
            self.task_id, log.steps, log.reward, log.reward >= success_threshold, model_id, policy_version,
            meta={"real_calls": log.real_calls, "replayed": log.replayed_calls, "resets": log.resets},
        )
        if initial_observation is not None:
            with self.store.db:
                self.store.db.execute(
                    "UPDATE nodes SET observation=COALESCE(observation, ?) WHERE node_id=?",
                    (initial_observation, root_id(self.task_id)),
                )
        # trajectories.env_calls counts non-cached steps; overwrite with the honest count
        with self.store.db:
            self.store.db.execute("UPDATE trajectories SET env_calls=? WHERE traj_id=?", (log.real_calls, tid))
        return tid
