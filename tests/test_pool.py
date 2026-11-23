"""Tests for the process env pool, using the deterministic toy env."""

import unittest

from xper.pool import ProcEnvPool


class TestPool(unittest.TestCase):
    def test_reset_step_and_terminal(self):
        with ProcEnvPool(4, start_method="fork") as pool:
            specs = [(f"e{i}", {"kind": "toy", "task_id": "t0"}) for i in range(6)]
            res = pool.reset(specs)
            self.assertEqual(len(res), 6)
            self.assertTrue(all(done is False for _, _, done in res.values()))

            # lookup -> refund succeeds; lookup -> cancel fails
            acts = {f"e{i}": '{"name": "lookup", "arguments": {"id": 1}}' for i in range(6)}
            res = pool.step(acts)
            self.assertTrue(all("found" in obs for obs, _, _ in res.values()))

            acts = {f"e{i}": ('{"name": "refund", "arguments": {"id": 1}}' if i % 2 else
                              '{"name": "cancel", "arguments": {"id": 1}}') for i in range(6)}
            res = pool.step(acts)
            for i in range(6):
                obs, reward, done = res[f"e{i}"]
                self.assertTrue(done)
                self.assertEqual(reward, 1.0 if i % 2 else 0.0, f"episode {i}")

    def test_state_stays_on_its_worker(self):
        """Episode state must not be lost: multi-step history is required for success."""
        with ProcEnvPool(3, start_method="fork") as pool:
            specs = [(f"e{i}", {"kind": "toy", "task_id": "t0"}) for i in range(9)]
            pool.reset(specs)
            pool.step({f"e{i}": '{"name": "lookup", "arguments": {"id": 1}}' for i in range(9)})
            res = pool.step({f"e{i}": '{"name": "refund", "arguments": {"id": 1}}' for i in range(9)})
            self.assertTrue(all(reward == 1.0 for _, reward, _ in res.values()))

    def test_bad_action_does_not_kill_worker(self):
        with ProcEnvPool(2, start_method="fork") as pool:
            pool.reset([("e0", {"kind": "toy", "task_id": "t0"})])
            res = pool.step({"e0": "not json at all"})
            self.assertTrue(res["e0"][2])  # episode reported done with an error
            pool.reset([("e1", {"kind": "toy", "task_id": "t0"})])
            self.assertIn("refund order", pool.reset([("e2", {"kind": "toy", "task_id": "t0"})])["e2"][0])

    def test_many_episodes_dispatch(self):
        with ProcEnvPool(8, start_method="fork") as pool:
            specs = [(f"e{i}", {"kind": "toy", "task_id": "t0"}) for i in range(50)]
            self.assertEqual(len(pool.reset(specs)), 50)


if __name__ == "__main__":
    unittest.main()