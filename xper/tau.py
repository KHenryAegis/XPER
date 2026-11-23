"""tau2-bench telecom (solo mode) adapter.

Wraps ``AgentGymEnv`` into the ``Env`` protocol used by ``ReuseEnv`` and builds
a model-agnostic prompt: every model sees the same system text and the same
"Action: {json}" surface form, so stored experience is comparable across
tokenizers.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from loguru import logger

logger.remove()
logging.disable(logging.CRITICAL)

from tau2.gym.gym_agent import AgentGymEnv  # noqa: E402
from tau2.registry import registry  # noqa: E402

DOMAINS = Path(__file__).resolve().parents[1] / "third_party/tau2-bench/data/tau2/domains"

INSTRUCTION = """You are a customer service agent that resolves the ticket below according to the <policy>.
You cannot talk to the user; you can only call tools. Call exactly one tool per turn by replying with

Action:
{"tool": "<tool name>", "arguments": {<arguments>}}

and nothing else. When the ticket is resolved, call the tool "done" with no arguments."""


def split(domain: str = "telecom", name: str = "test") -> list[str]:
    s = json.loads((DOMAINS / domain / "split_tasks.json").read_text())
    extra = Path(__file__).resolve().parents[1] / "data/splits" / f"{domain}_extra_splits.json"
    if extra.exists() and name in (ex := json.loads(extra.read_text())):
        return ex[name]  # held-out states never touched by any strong-target or source run
    if domain == "telecom" and name == "train_full":
        test = set(s["test"])
        return [t for t in s["full"] if t not in test]
    return s[name]


@lru_cache(maxsize=4)
def _tasks(domain: str):
    loader_name = "telecom_full" if domain == "telecom" else domain
    return {t.id: t for t in registry.get_tasks_loader(loader_name)()}


class FullTasksAgentGymEnv(AgentGymEnv):
    """Resolve task ids against the full task set (telecom_full, 2285 tasks).

    Stock ``_get_task`` uses ``registry.get_tasks_loader(domain)``, which for
    telecom is the 114-task base set, so every train_full id fails at reset.
    """

    def _get_task(self):
        try:
            return _tasks(self.domain)[self.task_id]
        except KeyError:
            raise ValueError(f"No task {self.task_id} in {self.domain} (full set)") from None


class LeanAgentGymEnv(FullTasksAgentGymEnv):
    """AgentGymEnv without the per-step evaluator call and full-run JSON dump.

    Stock ``step()`` runs ``evaluate_simulation`` and serializes the whole
    simulation run on *every* step (~470 ms); the episode reward only matters at
    the end. ``done`` comes from the orchestrator, so termination is unaffected.
    Measured 47x faster per step.
    """

    def _get_reward(self):
        if self._simulation_run is None:
            return 0.0, "{}"
        return super()._get_reward()  # real reward, computed once, at the end

    def _get_info(self):
        return {}

    def _get_simulation_run(self):
        return "{}"

    def _get_tools(self):
        return []

    def _get_policy(self):
        return ""


class TauSoloEnv:
    """One tau2 solo-mode episode. ``step`` takes a tau2 action string."""

    def __init__(self, task_id: str, domain: str = "telecom", max_steps: int = 30, lean: bool = True):
        self.task_id, self.domain, self.max_steps = task_id, domain, max_steps
        self.env_cls = LeanAgentGymEnv if lean else FullTasksAgentGymEnv

    def reset(self) -> str:
        self.env = self.env_cls(domain=self.domain, task_id=self.task_id, solo_mode=True)
        _, info = self.env.reset()
        self._prev = ""
        self.t = 0
        return ""  # solo mode: no initial observation

    def step(self, action: str):
        obs, reward, done, _, _ = self.env.step(action)
        self.t += 1
        new = obs[len(self._prev):].strip() if obs.startswith(self._prev) else obs
        self._prev = obs
        if not done and self.t >= self.max_steps:
            obs2, reward, done, _, _ = self.env.step('{"name": "done", "arguments": {}}')
            done = True
        return new, float(reward), bool(done), {}


@lru_cache(maxsize=4096)
def system_prompt(task_id: str, domain: str = "telecom") -> str:
    """Same tools/policy the gym agent receives in solo mode (agent + task user tools)."""
    environment = registry.get_env_constructor(domain)(solo_mode=True)
    task = _tasks(domain)[task_id]
    tools = environment.get_tools()
    if environment.user_tools:
        tools = tools + environment.get_user_tools(include=task.user_tools)
    schemas = [t.openai_schema["function"] for t in tools]
    schemas.append({"name": "done", "description": "Call when the ticket is resolved.",
                    "parameters": {"type": "object", "properties": {}}})
    tool_txt = "\n".join(json.dumps(t, ensure_ascii=False) for t in schemas)
    return (f"<instructions>\n{INSTRUCTION}\n</instructions>\n<policy>\n{environment.get_policy()}\n</policy>\n"
            f"<tools>\n{tool_txt}\n</tools>\n<ticket>\n{task.ticket}\n</ticket>")
