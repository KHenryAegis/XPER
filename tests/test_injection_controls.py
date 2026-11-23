import json
import random

from jsonschema import validate
from xper.actions import Action, parse_model_output
from xper.injection_controls import MASK, mask_observations, prompt_tools, schema_random


def fixture():
    tools = [{"name": "inspect", "parameters": {"type": "object", "properties": {}}},
             {"name": "set_mode", "parameters": {
                 "type": "object", "$defs": {"Mode": {"enum": ["a", "b"], "type": "string"}},
                 "properties": {"mode": {"$ref": "#/$defs/Mode"}}, "required": ["mode"]}},
             {"name": "done", "parameters": {"type": "object", "properties": {}}}]
    system = "<tools>\n" + "\n".join(map(json.dumps, tools)) + "\n</tools>"
    messages = [{"role": "system", "content": system}]
    for i in range(12):
        messages.extend([{"role": "assistant", "content": Action.tool_call("inspect").to_text()},
                         {"role": "user", "content": f"secret observation {i}"}])
    messages.append({"role": "assistant", "content": Action.tool_call("done").to_text()})
    return system, messages


def test_mask_keeps_actions_and_does_not_mutate_source():
    _, messages = fixture()
    before = json.dumps(messages)
    out = mask_observations(messages)
    assert json.dumps(messages) == before
    assert out[0] == messages[0]
    assert [m for m in out if m["role"] == "assistant"] == [m for m in messages if m["role"] == "assistant"]
    assert all(m["content"] == MASK for m in out if m["role"] == "user")


def test_schema_control_is_valid_and_source_content_independent():
    system, messages = fixture()
    changed = [dict(m, content="unrelated source text") if m["role"] != "system" else m for m in messages]
    out = schema_random(messages, system, random.Random(7))
    assert out == schema_random(changed, system, random.Random(7))
    assert [m["role"] for m in out] == [m["role"] for m in messages]
    schemas = {t["name"]: t["parameters"] for t in prompt_tools(system)}
    actions = [parse_model_output(m["content"]) for m in out if m["role"] == "assistant"]
    assert actions[-1].tool == "done" and all(a.tool != "done" for a in actions[:-1])
    assert {a.tool for a in actions[:-1]} == {"inspect", "set_mode"}
    for a in actions:
        validate(a.arguments, schemas[a.tool])
    assert all(m["content"] == MASK for m in out if m["role"] == "user")
