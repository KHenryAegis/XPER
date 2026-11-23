"""Integration tests against real tau2 telecom (solo mode). CPU only."""

import unittest

import pytest

pytest.importorskip("tau2", reason="needs tau2-bench (tools/fetch_tau2.sh)")
from xper.actions import Action  # noqa: E402
from xper.env import ReuseEnv
from xper.store import ExperienceStore
from xper.tau import TauSoloEnv, _tasks, split, system_prompt


class TestTau(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_ids = split("telecom", "test")
        cls.train_ids = split("telecom", "train_full")

    def test_splits_disjoint(self):
        self.assertEqual(len(self.test_ids), 40)
        self.assertEqual(len(self.train_ids), 2245)
        self.assertFalse(set(self.test_ids) & set(self.train_ids))

    def test_prompt_contains_ticket_and_done(self):
        p = system_prompt(self.test_ids[0])
        self.assertIn(_tasks("telecom")[self.test_ids[0]].ticket, p)
        self.assertIn('"name": "done"', p)

    def test_lean_env_matches_stock_reward(self):
        """The lean env must produce the same reward as the stock AgentGymEnv."""
        tid = self.test_ids[1]
        task = _tasks("telecom")[tid]
        gold = [Action.tool_call(a.name, a.arguments) for a in task.evaluation_criteria.actions]
        gold.append(Action.tool_call("done"))
        rewards = []
        for lean in (False, True):
            env = ReuseEnv(ExperienceStore(), lambda t: TauSoloEnv(t, lean=lean), tid, reuse=False)
            env.reset()
            for a in gold:
                _, _, done, _ = env.step(a)
                if done:
                    break
            rewards.append(env.log.reward)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards, [1.0, 1.0])

    def test_train_full_task_gold_replay(self):
        """train_full ids are outside tau2's default 114-task set; reset + reward must work."""
        tid = next(t for t in self.train_ids if t not in set(split("telecom", "train")))
        task = _tasks("telecom")[tid]
        gold = [Action.tool_call(a.name, a.arguments) for a in task.evaluation_criteria.actions]
        env = ReuseEnv(ExperienceStore(), lambda t: TauSoloEnv(t), tid, reuse=False)
        env.reset()
        for a in gold + [Action.tool_call("done")]:
            obs, _, done, _ = env.step(a)
            self.assertFalse(str(obs).startswith("ERROR"), obs)
            if done:
                break
        self.assertEqual(env.log.reward, 1.0)

    def test_gold_replay_then_free_reuse(self):
        tid = self.test_ids[0]
        task = _tasks("telecom")[tid]
        gold = [Action.tool_call(a.name, a.arguments) for a in task.evaluation_criteria.actions]
        gold.append(Action.tool_call("done"))
        store = ExperienceStore()
        logs = []
        for model in ("src", "tgt"):
            env = ReuseEnv(store, lambda t: TauSoloEnv(t), tid)
            obs0 = env.reset()
            for a in gold:
                _, _, done, _ = env.step(a)
                if done:
                    break
            env.commit(model, initial_observation=obs0)
            logs.append(env.log)
        self.assertEqual(logs[0].reward, 1.0)
        self.assertEqual(logs[0].real_calls, len(gold))
        self.assertEqual((logs[1].reward, logs[1].real_calls), (1.0, 0))


if __name__ == "__main__":
    unittest.main()
