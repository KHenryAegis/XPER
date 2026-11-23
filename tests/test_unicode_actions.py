import hashlib
import json

from xper.actions import Action
from xper.store import ExperienceStore, Step, root_id


def test_normal_unicode_keeps_original_keys():
    action = Action.tool_call("python", {"code": "print('中文😀')"})
    original = json.dumps({"tool": "python", "arguments": action.arguments},
                          sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert action.canonical() == original
    assert action.key == hashlib.sha1(original.encode()).hexdigest()[:16]


def test_lone_surrogate_roundtrips_without_dropping_argument_content():
    code = "x = '" + chr(0xD83D) + "'"
    action = Action.tool_call("python", {"code": code})
    assert "\\ud83d" in action.canonical()
    assert Action.from_json(action.canonical()).arguments["code"] == code
    assert Action.from_json(action.canonical()).key == action.key


def test_surrogate_action_can_be_committed_and_looked_up(tmp_path):
    action = Action.tool_call("python", {"code": chr(0xD83D)})
    store = ExperienceStore(str(tmp_path / "rollout.db"))
    store.add_trajectory("task", [Step(action, "tool error")], 0, False, "target")
    node = store.lookup(root_id("task"), action)
    assert node is not None and node.observation == "tool error"
    assert node.action.arguments == action.arguments
    assert store.db.execute("SELECT COUNT(*),SUM(env_calls) FROM trajectories").fetchone() == (1, 1)
    store.close()
