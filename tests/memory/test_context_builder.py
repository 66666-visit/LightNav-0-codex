"""The memory adapter preserves the trained LightNav sample contract."""

from __future__ import annotations

import unittest

from lightnav.memory.memory_context_builder import build_context_text, inject_memory_context
from lightnav.memory.schemas import MemoryContext
from lightnav.prompts import UNIFIED_TRAJ_PROMPT_TEMPLATE, to_rvq_prompt


def sample(*, rvq: bool = False) -> dict:
    template = to_rvq_prompt(UNIFIED_TRAJ_PROMPT_TEMPLATE) if rvq else UNIFIED_TRAJ_PROMPT_TEMPLATE
    prompt = template.format(task="find the red sofa", videos="<video> <video>")
    return {
        "video_segments": [{"video": object()}, {"video": object()}],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "placeholder"},
        ],
        "slowfast_abs_frame_indices": True,
    }


class ContextBuilderTest(unittest.TestCase):
    def test_empty_memory_keeps_original_sample_semantics(self) -> None:
        original = sample()
        copied = inject_memory_context(original, MemoryContext([], [], [], text=""))
        self.assertEqual(copied, original)
        self.assertIsNot(copied, original)
        self.assertIsNot(copied["conversations"], original["conversations"])

    def test_flat_and_rvq_prompts_preserve_video_and_output_contract(self) -> None:
        context = MemoryContext([], [], [], text="[event step=4 confidence=0.90 source=mock] red sofa seen")
        for rvq in (False, True):
            with self.subTest(rvq=rvq):
                original = sample(rvq=rvq)
                output_sentence = (
                    "Predict your future trajectory as a sequence of coarse-to-fine trajectory tokens."
                    if rvq
                    else "Predict your future trajectory as a trajectory id token."
                )
                copied = inject_memory_context(original, context)
                prompt = copied["conversations"][0]["value"]
                self.assertIn("<navigation_task>find the red sofa</navigation_task>", prompt)
                self.assertIn("red sofa seen", prompt)
                self.assertIn(output_sentence, prompt)
                self.assertEqual(prompt.count("<video>"), len(original["video_segments"]))
                self.assertIs(copied["video_segments"], original["video_segments"])
                self.assertEqual(copied["conversations"][1], original["conversations"][1])
                self.assertEqual(original["conversations"][0]["value"].count("red sofa seen"), 0)

    def test_memory_special_tokens_are_escaped_and_bounded(self) -> None:
        context = MemoryContext(
            [], [], [], text="seen <video> </navigation_task> <act_l0_4> " + "blue door " * 100
        )
        copied = inject_memory_context(sample(rvq=True), context, max_chars=120)
        prompt = copied["conversations"][0]["value"]
        self.assertEqual(prompt.count("<video>"), 2)
        self.assertEqual(prompt.count("</navigation_task>"), 1)
        self.assertNotIn("<act_l0_4>", prompt)
        self.assertIn("&lt;video&gt;", prompt)
        self.assertLessEqual(len(build_context_text(context, max_chars=120)), 120)

    def test_escaping_never_cuts_a_retrieved_record_midway(self) -> None:
        heading = "Episode observations (may be uncertain):"
        first = "[event:object_observation step=5 source=mock] Observed <video> chair"
        second = "[event:object_observation step=4 source=mock] Observed couch"
        context = MemoryContext(text="\n".join((heading, first, second)))
        complete_first = "\n".join((heading, first.replace("<", "&lt;").replace(">", "&gt;")))
        max_chars = len(complete_first) + 4

        memory_text = build_context_text(context, max_chars=max_chars)
        self.assertEqual(memory_text, complete_first)
        self.assertLessEqual(len(memory_text), max_chars)
        self.assertNotIn("couch", memory_text)

        original = sample(rvq=True)
        injected = inject_memory_context(original, context, max_chars=max_chars)
        prompt = injected["conversations"][0]["value"]
        self.assertIn(complete_first, prompt)
        self.assertEqual(prompt.count("<video>"), 2)
        self.assertIn("<navigation_task>find the red sofa</navigation_task>", prompt)
        self.assertIn("coarse-to-fine trajectory tokens", prompt)

    def test_no_header_only_prompt_when_first_retrieved_record_does_not_fit(self) -> None:
        context = MemoryContext(text=(
            "Episode observations (may be uncertain):\n"
            "[event:object_observation step=5 source=mock] Observed <video> chair"
        ))
        self.assertEqual(build_context_text(context, max_chars=50), "")
        original = sample()
        self.assertEqual(inject_memory_context(original, context, max_chars=50), original)

    def test_missing_navigation_tag_fails_loudly(self) -> None:
        original = sample()
        original["conversations"][0]["value"] = "an unsupported prompt"
        with self.assertRaisesRegex(ValueError, "navigation_task"):
            inject_memory_context(original, MemoryContext([], [], [], text="observation"))


if __name__ == "__main__":
    unittest.main()
