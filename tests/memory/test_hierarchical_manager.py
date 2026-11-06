"""One-extraction, three-layer episode tests through the real MemoryManager API."""

from __future__ import annotations

import unittest

import numpy as np

from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.schemas import DetectedObject, MemoryQuery, ObservationPacket, SceneExtraction


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _action() -> dict:
    return {
        "action": "velocity_control",
        "action_args": {"linear_velocity": 0.2, "angular_velocity": 0.0},
    }


class CountingExtractor(MockSceneExtractor):
    def __init__(self, by_step: dict[int, SceneExtraction]) -> None:
        super().__init__(by_step)
        self.calls: list[int] = []

    def extract(self, packet: ObservationPacket) -> SceneExtraction:
        self.calls.append(packet.step_id)
        return super().extract(packet)


class HierarchicalManagerTests(unittest.TestCase):
    def test_one_extraction_per_observation_and_step_five_couch_survives_to_twenty(self) -> None:
        extractor = CountingExtractor({
            5: SceneExtraction(
                detections=[DetectedObject("couch", 0.82, (1, 1, 6, 6))],
                confidence=0.82,
            ),
        })
        manager = MemoryManager(extractor=extractor, strategy="hierarchical")
        instruction = "find the couch seen earlier"
        manager.reset_episode("first", instruction)
        for step in range(21):
            if step:
                manager.record_action(_action())
            manager.observe(ObservationPacket(step, _frame(), instruction))
        self.assertEqual(extractor.calls, list(range(21)))
        observations = manager.episodic.query(kind="object_observation")
        self.assertEqual(len(observations), 1)
        self.assertEqual(len(manager.scene_graph.query_nodes(kind="object")), 1)
        self.assertEqual(manager.scene_graph.edges, [])

        layers = manager.retrieve_layers()
        self.assertFalse(any("couch" in item.description for item in layers["short"]))
        self.assertTrue(any("couch" in item.description for item in layers["mid"]))
        self.assertTrue(any(item.label == "couch" for item in layers["long"]))
        for layer in ("mid", "long"):
            couch_refs = [
                ref for item in layers[layer] for ref in item.evidence_refs
                if ref.label == "couch"
            ]
            self.assertEqual(len(couch_refs), 1)
            self.assertEqual(couch_refs[0].event_id, observations[0].event_id)
            self.assertEqual(couch_refs[0].node_id, observations[0].metadata["node_id"])
        fused = manager.get_context()
        self.assertIn("couch", fused.text)
        self.assertEqual(
            sum(ref.event_id == observations[0].event_id for item in fused.layer_items
                for ref in item.evidence_refs), 1,
        )
        manager.close_episode()
        self.assertEqual(manager.semantic_index.entries, [])
        self.assertEqual(manager.working.recent_observations, ())
        self.assertEqual(manager.episodic.events, [])

        manager.reset_episode("second", "find a table")
        manager.observe(ObservationPacket(0, _frame(), "find a table"))
        self.assertEqual(manager.get_context().text, "")
        self.assertEqual(manager.scene_graph.nodes, [])
        manager.close_episode()

    def test_all_seven_layer_ablations_and_no_semantics(self) -> None:
        subsets = (
            ("short",), ("mid",), ("long",), ("short", "mid"),
            ("short", "long"), ("mid", "long"), ("short", "mid", "long"),
        )
        for subset in subsets:
            with self.subTest(subset=subset):
                extractor = CountingExtractor({
                    0: SceneExtraction(
                        detections=[DetectedObject("chair", 0.77)], confidence=0.77,
                    ),
                })
                manager = MemoryManager(
                    extractor=extractor, strategy="hierarchical", layers=subset,
                )
                manager.reset_episode("ablation", "find chair")
                manager.observe(ObservationPacket(0, _frame(), "find chair"))
                layers = manager.retrieve_layers()
                self.assertEqual(extractor.calls, [0])
                for layer in ("short", "mid", "long"):
                    self.assertEqual(bool(layers[layer]), layer in subset)
                self.assertEqual(bool(manager.working.recent_observations), "short" in subset)
                self.assertEqual(bool(manager.episodic.key_events), "mid" in subset)
                self.assertEqual(bool(manager.semantic_index.entries), "long" in subset)
                context = manager.get_context()
                self.assertTrue(context.text)
                self.assertTrue(all(item.layer in subset for item in context.layer_items))
                manager.close_episode()

        empty = MemoryManager(strategy="hierarchical")
        empty.reset_episode("empty", "find chair")
        empty.observe(ObservationPacket(0, _frame(), "find chair"))
        self.assertEqual(empty.get_context().text, "")
        empty.close_episode()

    def test_flat_path_still_uses_original_retriever(self) -> None:
        manager = MemoryManager(extractor=MockSceneExtractor({
            0: SceneExtraction(detections=[DetectedObject("couch", 0.84)], confidence=0.84),
        }))
        manager.reset_episode("flat", "find couch")
        manager.observe(ObservationPacket(0, _frame(), "find couch"))
        expected = manager.retriever.retrieve(
            MemoryQuery("find couch", 0, top_k=5, max_chars=600), manager.episodic.events,
            manager.scene_graph.nodes, manager.scene_graph.edges,
        )
        self.assertEqual(manager.get_context(), expected)
        self.assertEqual(manager.get_context().layer_items, [])
        self.assertEqual(manager.last_layer_results, {})
        manager.close_episode()


if __name__ == "__main__":
    unittest.main()
