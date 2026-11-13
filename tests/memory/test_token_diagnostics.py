"""Token-level evidence checks run without model weights or PyTorch."""

from __future__ import annotations

import unittest

from lightnav.inference.token_diagnostics import memory_token_checks, token_ids_sha256


class _ReversibleTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in value]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return "".join(chr(value) for value in token_ids)


class TokenDiagnosticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _ReversibleTokenizer()
        self.line = "Current RGB observed instruction landmark UNIQUE_MEMORY_MARKER; source=rgb."
        self.memory = "[Navigation memory; uncertain RGB observations]\n" + self.line
        self.trace = {
            "original_instruction": "Find the TV",
            "memory_text": self.memory,
            "accepted_records": [{"evidence_ids": ["event_7"], "text": self.line}],
        }

    def test_actual_prompt_tokens_link_memory_line_to_evidence(self) -> None:
        prompt = (
            "<navigation_task>Find the TV</navigation_task>. "
            + self.memory + "\nContinue.<|im_start|>assistant\n"
        )
        token_ids = self.tokenizer.encode(prompt)
        result = memory_token_checks(self.tokenizer, token_ids, self.trace)
        self.assertTrue(result["instruction_text_present"])
        self.assertTrue(result["assistant_generation_boundary_present"])
        self.assertTrue(result["memory_text_present"])
        self.assertEqual(result["model_visible_evidence_ids"], ["event_7"])
        self.assertEqual(result["literal_model_visible_evidence_ids"], [])
        self.assertTrue(result["accepted_records"][0]["present_in_final_tokens"])
        self.assertEqual(token_ids_sha256(token_ids), token_ids_sha256(list(token_ids)))

    def test_memory_words_elsewhere_do_not_count_as_injection(self) -> None:
        prompt = (
            "<navigation_task>Find the TV; " + self.memory
            + "</navigation_task>. Continue.<|im_start|>assistant\n"
        )
        result = memory_token_checks(self.tokenizer, self.tokenizer.encode(prompt), self.trace)
        self.assertFalse(result["memory_text_present"])
        self.assertEqual(result["model_visible_evidence_ids"], [])

    def test_missing_answer_boundary_is_reported(self) -> None:
        prompt = "<navigation_task>Find the TV</navigation_task>. " + self.memory + "\nContinue."
        result = memory_token_checks(self.tokenizer, self.tokenizer.encode(prompt), self.trace)
        self.assertTrue(result["memory_text_present"])
        self.assertFalse(result["assistant_generation_boundary_present"])

    def test_v3_evidence_key_must_be_literal_in_visible_record(self) -> None:
        evidence_key = '["scene","episode",0,"event_7"]'
        line = f"[OBSERVED] archway; Evidence: {evidence_key}."
        memory = "[Navigation memory; evidence-linked observations]\n" + line
        trace = {
            "original_instruction": "Find the archway",
            "memory_text": memory,
            "accepted_records": [{"evidence_ids": [evidence_key], "text": line}],
        }
        prompt = (
            "<navigation_task>Find the archway</navigation_task>. "
            + memory + "\nContinue.<|im_start|>assistant\n"
        )
        result = memory_token_checks(self.tokenizer, self.tokenizer.encode(prompt), trace)
        self.assertEqual(result["literal_model_visible_evidence_ids"], [evidence_key])
        self.assertEqual(
            result["accepted_records"][0]["literal_evidence_ids_in_final_tokens"],
            [evidence_key],
        )


if __name__ == "__main__":
    unittest.main()
