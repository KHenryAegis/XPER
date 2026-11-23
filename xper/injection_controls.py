"""Content controls for offline injection; synthetic calls are never executed."""
import copy
import json

MASK = "Observation:\n[observation withheld]"


def mask_observations(messages):
    out = copy.deepcopy(messages)
    for m in out[1:]:
        if m["role"] in ("user", "tool"):
            m["content"] = MASK
    return out


def prompt_tools(system):
    body = system.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def schema_value(schema, rng, root=None):
    root = root or schema
    if "$ref" in schema:
        node = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            node = node[part.replace("~1", "/").replace("~0", "~")]
        return schema_value(node, rng, root)
    if "enum" in schema:
        return rng.choice(schema["enum"])
    if "const" in schema:
        return schema["const"]
    if "anyOf" in schema or "oneOf" in schema:
        choices = schema.get("anyOf", schema.get("oneOf"))
        return schema_value(choices[0], rng, root)
    typ = schema.get("type", "object" if "properties" in schema else "string")
    if typ == "object":
        return {k: schema_value(v, rng, root) for k, v in schema.get("properties", {}).items()
                if k in schema.get("required", [])}
    if typ == "array":
        return [schema_value(schema.get("items", {}), rng, root)
                for _ in range(schema.get("minItems", 0))]
    if typ == "boolean":
        return bool(rng.randrange(2))
    if typ in ("number", "integer"):
        low, high = int(schema.get("minimum", 0)), int(schema.get("maximum", 9))
        return rng.randint(low, max(low, high))
    if typ == "null":
        return None
    return "synthetic"


def schema_random(messages, system, rng):
    """Same role/turn template; uniformly sampled schema calls, no source values/observations.

    The positive training label is deliberately synthetic. This is not an
    environment success and must never enter source-success or reward metrics.
    """
    from .actions import Action
    out = mask_observations(messages)
    out[0] = dict(out[0], content=system)
    tools = prompt_tools(system)
    ordinary = [t for t in tools if t["name"] != "done"]
    done = next(t for t in tools if t["name"] == "done")
    indices = [i for i, m in enumerate(out) if m["role"] == "assistant"]
    if not indices or not ordinary:
        raise ValueError("synthetic control requires assistant turns and tools")
    for i in indices:
        tool = done if i == indices[-1] else rng.choice(ordinary)
        args = schema_value(tool["parameters"], rng)
        out[i]["content"] = Action.tool_call(tool["name"], args).to_text()
    return out
