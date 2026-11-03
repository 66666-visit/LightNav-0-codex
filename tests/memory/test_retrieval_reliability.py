"""Evidence labels and empty-context behavior at the retrieval boundary."""

from __future__ import annotations

import unittest

import numpy as np

from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_context_builder import build_context_text, inject_memory_context
from lightnav.memory.retriever import MemoryRetriever
from lightnav.memory.schemas import MemoryContext, MemoryQuery, ObservationPacket, SceneEdge
from lightnav.prompts import UNIFIED_TRAJ_PROMPT_TEMPLATE, to_rvq_prompt


def _sample(*, rvq: bool = False) -> dict:
    template = to_rvq_prompt(UNIFIED_TRAJ_PROMPT_TEMPLATE) if rvq else UNIFIED_TRAJ_PROMPT_TEMPLATE
    return {
        "video_segments": [{"video": object()}, {"video": object()}],
        "conversations": [
            {
                "from": "human",
                "value": template.format(task="find the red sofa", videos="<video> <video>"),
            },
            {"from": "gpt", "value": "placeholder"},
        ],
        "slowfast_abs_frame_indices": True,
        "_allow_vit_cache": True,
        "_skip_normalize": True,
    }


class RetrievalReliabilityTest(unittest.TestCase):
    def test_relation_text_separates_hypothesis_traversal_and_visibility(self) -> None:
        edges = [
            SceneEdge(
                "possible", "hall", "kitchen", "possible_transition", 0.4,
                "action_and_rgb_change", 3, "appearance changed after a command",
            ),
            SceneEdge(
                "connected", "hall", "kitchen", "connected_to", 0.9,
                "verified_traversal", 4, "doorway traversal confirmed",
            ),
            SceneEdge(
                "visible", "hall", "sofa", "visible_from", 0.8,
                "rgb_extractor", 5, "sofa appears in the RGB view",
            ),
        ]
        context = MemoryRetriever().retrieve(
            MemoryQuery("kitchen sofa", current_step=5, top_k=3, max_chars=600),
            [], [], edges,
        )

        self.assertEqual(len(context.edges), 3)
        self.assertIn("relation:possible_transition status=unverified_hypothesis", context.text)
        self.assertIn("relation:connected_to status=verified_traversal", context.text)
        self.assertIn("relation:visible_from status=visual_observation", context.text)
        self.assertIn("sofa was visible from hall", context.text)
        self.assertNotIn("relation:contains", context.text)

    def test_direct_unverified_connection_record_is_not_presented_as_verified(self) -> None:
        edge = SceneEdge(
            "untrusted", "hall", "kitchen", "connected_to", 0.7,
            "issued_command_and_rgb_semantics", 2, "appearance changed",
        )
        context = MemoryRetriever().retrieve(
            MemoryQuery("kitchen", current_step=2, top_k=1), [], [], [edge],
        )
        self.assertIn("relation:connected_to status=unverified_claim", context.text)
        self.assertNotIn("status=verified_traversal", context.text)

    def test_null_extractor_first_frame_has_no_semantic_context(self) -> None:
        manager = MemoryManager()
        manager.reset_episode("episode-1", "find the red sofa")
        try:
            manager.observe(ObservationPacket(
                step_id=0,
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                instruction="find the red sofa",
            ))
            context = manager.get_context()
            self.assertEqual(context.text, "")
            self.assertEqual(context.events, [])
            self.assertEqual(context.nodes, [])
            self.assertEqual(context.edges, [])
            self.assertEqual(manager.scene_graph.nodes, [])
        finally:
            manager.close_episode()

    def test_empty_memory_leaves_instruction_video_and_slowfast_sample_intact(self) -> None:
        for rvq in (False, True):
            with self.subTest(rvq=rvq):
                original = _sample(rvq=rvq)
                copied = inject_memory_context(original, MemoryContext(text=""))
                self.assertEqual(copied["conversations"], original["conversations"])
                self.assertIs(copied["video_segments"], original["video_segments"])
                self.assertTrue(copied["slowfast_abs_frame_indices"])
                self.assertTrue(copied["_allow_vit_cache"])
                self.assertTrue(copied["_skip_normalize"])

    def test_long_memory_is_char_bounded_without_changing_prompt_contract(self) -> None:
        context = MemoryContext(text="[event] remembered sofa near door. " * 100)
        self.assertLessEqual(len(build_context_text(context)), 600)
        for rvq in (False, True):
            with self.subTest(rvq=rvq):
                original = _sample(rvq=rvq)
                copied = inject_memory_context(original, context)
                prompt = copied["conversations"][0]["value"]
                base = original["conversations"][0]["value"]
                self.assertIn("<navigation_task>find the red sofa</navigation_task>. ", prompt)
                self.assertEqual(prompt.count("<video>"), base.count("<video>"))
                self.assertEqual(prompt.count("</navigation_task>"), 1)
                self.assertIn(base.split("</navigation_task>. ", 1)[1], prompt)
                self.assertLessEqual(len(prompt) - len(base), 602)
                self.assertIs(copied["video_segments"], original["video_segments"])
                self.assertEqual(copied["conversations"][1], original["conversations"][1])
                self.assertTrue(copied["slowfast_abs_frame_indices"])


if __name__ == "__main__":
    unittest.main()
