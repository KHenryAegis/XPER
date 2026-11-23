"""xper.chat must render byte-identically to the official Qwen3 templates."""

import os
import unittest

import pytest

pytest.importorskip("transformers", reason="needs the GPU environment (requirements.txt)")
from transformers import AutoTokenizer  # noqa: E402

from xper.chat import NO_THINK_BLOCK, render, templates_think

M = os.environ.get("XPER_HOME", "") + "/models/hf/"
MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
MULTI = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
         {"role": "assistant", "content": "A1"}, {"role": "user", "content": "O1"}]


class TestChat(unittest.TestCase):
    def test_no_think_matches_hybrid_template(self):
        tok = AutoTokenizer.from_pretrained(M + "Qwen3-8B")
        ref = tok.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(templates_think(tok))
        self.assertEqual(render(MSGS, no_think=True), ref)

    def test_no_think_matches_with_history(self):
        """History assistant turns must also carry the empty think block."""
        tok = AutoTokenizer.from_pretrained(M + "Qwen3-8B")
        ref = tok.apply_chat_template(MULTI, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.assertEqual(render(MULTI, no_think=True), ref)

    def test_plain_matches_instruct_2507(self):
        tok = AutoTokenizer.from_pretrained(M + "Qwen3-4B-Instruct-2507")
        ref = tok.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True)
        self.assertFalse(templates_think(tok))
        self.assertEqual(render(MSGS), ref)

    def test_block_is_real_think_tags(self):
        self.assertTrue(NO_THINK_BLOCK.startswith("<think>") and "</think>" in NO_THINK_BLOCK)


if __name__ == "__main__":
    unittest.main()
