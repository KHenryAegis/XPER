"""Model-agnostic agent action space.

Every model emits either a tool call ``{"tool": name, "arguments": {...}}`` or a
plain message to the user. Actions are canonicalized so that two models (with
different tokenizers / chat templates) producing the same decision map to the
same key. This key is what the experience store and policy-shift transfer use.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _normalize(value: Any) -> Any:
    """Normalize argument values so trivially different encodings compare equal."""
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


@dataclass(frozen=True)
class Action:
    kind: str  # "tool" or "message"
    tool: Optional[str] = None
    arguments: dict = field(default_factory=dict)
    text: Optional[str] = None

    def __post_init__(self):
        if self.kind not in ("tool", "message"):
            raise ValueError(f"unknown action kind: {self.kind}")
        if self.kind == "tool" and not self.tool:
            raise ValueError("tool action requires a tool name")

    def canonical(self) -> str:
        if self.kind == "tool":
            body = {"tool": self.tool.strip(), "arguments": _normalize(self.arguments)}
        else:
            body = {"message": (self.text or "").strip()}
        text = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        # json.loads accepts escaped lone surrogates in model-written code.
        # Keep ordinary Unicode and its existing trie keys unchanged, but
        # escape non-UTF-8 code points so hashing/SQLite can persist the action.
        return text.encode("utf-8", errors="backslashreplace").decode("utf-8")

    @property
    def key(self) -> str:
        """Exact-match key. Messages are keyed by full text (only matches verbatim)."""
        return hashlib.sha1(self.canonical().encode()).hexdigest()[:16]

    @property
    def is_tool(self) -> bool:
        return self.kind == "tool"

    def to_tau2(self) -> str:
        """Action string accepted by tau2 ``AgentGymEnv.step``."""
        if self.is_tool:
            return json.dumps({"name": self.tool, "arguments": self.arguments}, ensure_ascii=False)
        return self.text or ""

    def to_text(self) -> str:
        """Surface form shown to / scored under a model (identical across models)."""
        if self.is_tool:
            return "Action:\n" + json.dumps(
                {"tool": self.tool, "arguments": self.arguments}, ensure_ascii=False, indent=2
            )
        return self.text or ""

    @classmethod
    def tool_call(cls, tool: str, arguments: Optional[dict] = None) -> "Action":
        return cls(kind="tool", tool=tool, arguments=dict(arguments or {}))

    @classmethod
    def message(cls, text: str) -> "Action":
        return cls(kind="message", text=text)

    @classmethod
    def from_json(cls, s: str) -> "Action":
        d = json.loads(s)
        if "message" in d:
            return cls.message(d["message"])
        return cls.tool_call(d["tool"], d.get("arguments", {}))


def _first_json_object(text: str) -> Optional[dict]:
    """Return the first top-level JSON object found in ``text``."""
    for m in _FENCE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_model_output(text: str) -> Optional[Action]:
    """Parse raw model output into an Action.

    Returns ``None`` for malformed tool calls (an ``Action:`` marker or a JSON
    object with a tool key, but unusable). Anything else is a user message.
    """
    stripped = text.strip()
    if not stripped:
        return None
    wants_tool = stripped.startswith("Action:") or stripped.startswith("{") or "```" in stripped
    obj = _first_json_object(stripped) if wants_tool else None
    if obj is not None:
        name = obj.get("tool", obj.get("name"))
        args = obj.get("arguments", obj.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if isinstance(name, str) and name and isinstance(args, dict):
            return Action.tool_call(name, args)
        return None
    if wants_tool:
        return None
    return Action.message(stripped)
