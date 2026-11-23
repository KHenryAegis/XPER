import json
import unittest

from xper.actions import Action, parse_model_output
from xper.env import ReuseEnv
from xper.store import ExperienceStore, Step, child_id, root_id


class ToyEnv:
    """Deterministic: success iff the agent calls lookup(id=1) then refund(id=1)."""

    instances = 0

    def __init__(self, task_id):
        ToyEnv.instances += 1
        self.calls = 0

    def reset(self):
        self.hist = []
        return "user: refund order 1"

    def step(self, action):
        self.calls += 1
        d = json.loads(action)
        self.hist.append((d["name"], d["arguments"].get("id")))
        if d["name"] == "lookup":
            return f"tool: order {d['arguments']['id']} found", 0.0, False, {}
        done = d["name"] in ("refund", "cancel")
        reward = 1.0 if self.hist == [("lookup", 1), ("refund", 1)] else 0.0
        return f"tool: {d['name']} done", reward if done else 0.0, done, {}


LOOKUP = Action.tool_call("lookup", {"id": 1})
REFUND = Action.tool_call("refund", {"id": 1})
CANCEL = Action.tool_call("cancel", {"id": 1})


def run(store, actions, model, reuse=True):
    env = ReuseEnv(store, ToyEnv, "t0", reuse=reuse)
    obs0 = env.reset()
    for a in actions:
        _, _, done, _ = env.step(a)
        if done:
            break
    env.commit(model, initial_observation=obs0)
    return env.log


class TestActions(unittest.TestCase):
    def test_canonical_is_order_and_whitespace_invariant(self):
        a = Action.tool_call("f", {"b": " x ", "a": 1.0})
        b = Action.tool_call("f", {"a": 1, "b": "x"})
        self.assertEqual(a.key, b.key)

    def test_parse_variants(self):
        self.assertEqual(parse_model_output('Action:\n{"tool": "f", "arguments": {"a": 1}}').key,
                         Action.tool_call("f", {"a": 1}).key)
        self.assertEqual(parse_model_output('```json\n{"name": "f", "arguments": "{\\"a\\": 1}"}\n```').key,
                         Action.tool_call("f", {"a": 1}).key)
        self.assertEqual(parse_model_output("Sure, one moment.").kind, "message")
        self.assertIsNone(parse_model_output('Action: {"tool": 3}'))
        self.assertIsNone(parse_model_output("Action: not json"))

    def test_roundtrip(self):
        a = Action.tool_call("f", {"a": [1, 2]})
        self.assertEqual(Action.from_json(a.canonical()).key, a.key)
        self.assertEqual(json.loads(a.to_tau2()), {"name": "f", "arguments": {"a": [1, 2]}})


class TestStoreAndReuse(unittest.TestCase):
    def setUp(self):
        self.store = ExperienceStore()

    def test_first_episode_hits_env(self):
        log = run(self.store, [LOOKUP, REFUND], "src")
        self.assertEqual((log.real_calls, log.reward), (2, 1.0))

    def test_identical_episode_is_free_and_gets_reward(self):
        run(self.store, [LOOKUP, REFUND], "src")
        ToyEnv.instances = 0
        log = run(self.store, [LOOKUP, REFUND], "tgt")
        self.assertEqual((log.real_calls, log.reward, log.done), (0, 1.0, True))
        self.assertEqual(ToyEnv.instances, 0)

    def test_divergence_replays_prefix_honestly(self):
        run(self.store, [LOOKUP, REFUND], "src")
        log = run(self.store, [LOOKUP, CANCEL], "tgt")
        # lookup served from cache, then replayed (1) + cancel (1)
        self.assertEqual((log.real_calls, log.replayed_calls, log.reward), (2, 1, 0.0))
        self.assertEqual([s.from_cache for s in log.steps], [True, False])

    def test_no_reuse_baseline(self):
        run(self.store, [LOOKUP, REFUND], "src")
        log = run(self.store, [LOOKUP, REFUND], "tgt", reuse=False)
        self.assertEqual(log.real_calls, 2)

    def test_inconsistent_observation_disables_reuse(self):
        s = self.store
        s.add_trajectory("t1", [Step(LOOKUP, "A")], 0.0, False, "m")
        s.add_trajectory("t1", [Step(LOOKUP, "B")], 0.0, False, "m")
        self.assertFalse(s.lookup(root_id("t1"), LOOKUP).consistent)

    def test_cross_store_coverage(self):
        import os, sys, tempfile
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from coverage import coverage
        d = tempfile.mkdtemp()
        src, tgt = ExperienceStore(f"{d}/s.db"), ExperienceStore(f"{d}/t.db")
        run(src, [LOOKUP, REFUND], "src", reuse=False)
        run(tgt, [LOOKUP, REFUND], "tgt", reuse=False)
        run(tgt, [LOOKUP, CANCEL], "tgt", reuse=False)
        src.close(); tgt.close()
        c = coverage(f"{d}/s.db", f"{d}/t.db")
        self.assertEqual(c["by_depth"], {0: 1.0, 1: 0.5})
        self.assertEqual(c["episode_full_coverage"], 0.5)
        self.assertAlmostEqual(c["step_coverage"], 0.75)

    def test_history_path(self):
        run(self.store, [LOOKUP, REFUND], "src")
        leaf = child_id(child_id(root_id("t0"), LOOKUP), REFUND)
        self.assertEqual([n.action.tool for n in self.store.history(leaf)], ["lookup", "refund"])


if __name__ == "__main__":
    unittest.main()
