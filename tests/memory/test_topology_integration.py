"""Manager integration for a descriptor anchored, episode-local place graph.

These tests use scripted visual descriptors and YOLO-shaped detections. They do
not claim that the pretrained DINOv2 model or Habitat has run in this suite.
"""

from __future__ import annotations

import unittest

import numpy as np

from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.place_recognition import MockPlaceDescriptorExtractor
from lightnav.memory.schemas import DetectedObject, MemoryQuery, ObservationPacket, SceneExtraction


def _frame() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _action() -> dict:
    return {
        "action": "velocity_control",
        "action_args": {"linear_velocity": 0.2, "angular_velocity": 0.0},
    }


class TopologyManagerIntegrationTest(unittest.TestCase):
    def test_loop_closure_categories_and_evidence_use_same_rgb_steps(self) -> None:
        descriptor = MockPlaceDescriptorExtractor({
            0: (1.0, 0.0, 0.0),
            5: (0.0, 1.0, 0.0),
            10: (0.0, 0.0, 1.0),
            20: (1.0, 0.0, 0.0),
        })
        visual = MockSceneExtractor({
            0: SceneExtraction(detections=[
                DetectedObject("chair", 0.81), DetectedObject("chair", 0.69),
            ], confidence=0.81),
            5: SceneExtraction(detections=[DetectedObject("table", 0.78)], confidence=0.78),
            20: SceneExtraction(detections=[DetectedObject("couch", 0.88)], confidence=0.88),
        })
        manager = MemoryManager(
            extractor=visual, strategy="hierarchical", place_enabled=True,
            place_extractor=descriptor, place_every_n_steps=5,
            place_similarity_threshold=0.85,
        )
        instruction = "return to the chair, then find the couch"
        manager.reset_episode("loop", instruction)
        try:
            for step in range(21):
                if step:
                    manager.record_action(_action())
                manager.observe(ObservationPacket(step, _frame(), instruction))
                if step in {0, 5, 10, 20}:
                    # The same step's writer events are available before place
                    # association and before the policy calls retrieve().
                    self.assertIsNotNone(manager.last_place_result)
                if step == 15:
                    self.assertIsNone(manager.last_place_result)
                    self.assertEqual(len(manager.topological_memory.nodes), 3)
                    self.assertEqual(len(manager.topological_memory.edges), 2)

            self.assertEqual(descriptor.calls, [0, 5, 10, 15, 20])
            self.assertTrue(manager.last_place_result.revisited)
            self.assertAlmostEqual(manager.last_place_result.top1_similarity, 1.0)
            graph = manager.topological_memory
            self.assertEqual(len(graph.nodes), 3)
            self.assertEqual(len(graph.edges), 3)
            a, b, c = graph.nodes
            self.assertEqual(graph.current_place_id, a.place_id)
            self.assertEqual((a.visit_count, a.observation_count), (2, 2))
            self.assertEqual((b.visit_count, c.visit_count), (1, 1))
            self.assertEqual(
                [(edge.source_place_id, edge.target_place_id) for edge in graph.edges],
                [(a.place_id, b.place_id), (b.place_id, c.place_id), (c.place_id, a.place_id)],
            )
            self.assertTrue(all(
                edge.relation_type == "observed_transition" for edge in graph.edges
            ))
            self.assertEqual(manager.scene_graph.query_edges(relation="connected_to"), [])
            self.assertEqual(manager.scene_graph.query_edges(relation="contains"), [])

            chair = a.observed_categories["chair"]
            self.assertEqual(chair.observation_count, 2)
            self.assertEqual(len(chair.evidence_event_ids), 2)
            self.assertEqual(
                len(manager.scene_graph.query_nodes(kind="object", label="chair")), 2,
            )
            self.assertIn("couch", a.observed_categories)
            self.assertIn("table", b.observed_categories)
            self.assertNotIn("table", a.observed_categories)
            self.assertEqual(len(manager.episodic.query(kind="place_observation")), 4)
            self.assertTrue(
                manager.episodic.query(kind="place_observation")[-1].metadata["place_matched"]
            )
            logged_ids = {event.event_id for event in manager.episodic.events}
            for node in graph.nodes:
                for event_id in node.evidence_event_ids:
                    self.assertIn(event_id, logged_ids)
            for edge in graph.edges:
                self.assertTrue(edge.evidence_event_ids)
                self.assertTrue(set(edge.evidence_event_ids) <= logged_ids)

            long_items = manager.retrieve_layers()["long"]
            self.assertTrue(any(item.kind == "current_place" for item in long_items))
            self.assertTrue(any(item.kind == "observed_transition" for item in long_items))
            self.assertTrue(any(item.kind == "place_semantics" for item in long_items))
            historical = manager.retrieve_layers(MemoryQuery(
                "find the table", current_step=20, top_k=10,
            ))["long"]
            self.assertTrue(any(
                item.kind == "place_semantics" and item.label == "table"
                and item.metadata.get("place_id") == b.place_id
                for item in historical
            ))
            fused = manager.get_context()
            self.assertIn("[long:current_place", fused.text)
            self.assertTrue(fused.text.index("[long:current_place") <
                            fused.text.index("[short:current_observation"))
            couch_event_id = manager.episodic.query(kind="object_observation")[-1].event_id
            self.assertEqual(sum(
                ref.event_id == couch_event_id for item in fused.layer_items
                for ref in item.evidence_refs
            ), 1)
        finally:
            manager.close_episode()
        self.assertEqual(manager.topological_memory.nodes, [])
        self.assertEqual(manager.topological_memory.edges, [])
        self.assertIsNone(manager.topological_memory.current_place_id)

    def test_no_executed_motion_means_no_observed_transition_edge(self) -> None:
        descriptor = MockPlaceDescriptorExtractor({0: (1.0, 0.0), 5: (0.0, 1.0)})
        manager = MemoryManager(
            strategy="hierarchical", layers=("long",), place_enabled=True,
            place_extractor=descriptor, place_every_n_steps=5,
        )
        manager.reset_episode("stationary", "look around")
        try:
            for step in range(6):
                manager.observe(ObservationPacket(step, _frame(), "look around"))
            self.assertEqual(len(manager.topological_memory.nodes), 2)
            self.assertEqual(manager.topological_memory.edges, [])
        finally:
            manager.close_episode()

    def test_stop_or_pure_turn_does_not_support_transition_edge(self) -> None:
        commands = (
            {"action": "velocity_control", "action_args": {
                "linear_velocity": -1.0, "angular_velocity": 0.0,
            }},
            {"action": "velocity_control", "action_args": {
                "linear_velocity": -1.0, "angular_velocity": 0.4,
            }},
        )
        for command in commands:
            with self.subTest(command=command):
                manager = MemoryManager(
                    strategy="hierarchical", layers=("long",), place_enabled=True,
                    place_extractor=MockPlaceDescriptorExtractor({
                        0: (1.0, 0.0), 5: (0.0, 1.0),
                    }),
                )
                manager.reset_episode("no_forward_motion", "look around")
                try:
                    for step in range(6):
                        if step == 5:
                            manager.record_action(command)
                        manager.observe(ObservationPacket(step, _frame(), "look around"))
                    self.assertEqual(len(manager.topological_memory.nodes), 2)
                    self.assertEqual(manager.topological_memory.edges, [])
                finally:
                    manager.close_episode()

    def test_flat_and_long_disabled_never_invoke_place_descriptor(self) -> None:
        for strategy, layers in (("flat", ("short", "mid", "long")),
                                 ("hierarchical", ("short", "mid"))):
            with self.subTest(strategy=strategy, layers=layers):
                descriptor = MockPlaceDescriptorExtractor({0: (1.0, 0.0)})
                manager = MemoryManager(
                    strategy=strategy, layers=layers, place_enabled=True,
                    place_extractor=descriptor,
                )
                manager.reset_episode("disabled", "find chair")
                try:
                    manager.observe(ObservationPacket(0, _frame(), "find chair"))
                    self.assertEqual(descriptor.calls, [])
                    self.assertEqual(manager.topological_memory.nodes, [])
                finally:
                    manager.close_episode()

    def test_descriptor_failure_warns_without_false_place_and_can_retry(self) -> None:
        class FailsOnce(MockPlaceDescriptorExtractor):
            def __init__(self) -> None:
                super().__init__({5: (1.0, 0.0)})
                self.failed = False

            def extract(self, packet: ObservationPacket) -> tuple[float, ...] | None:
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("synthetic DINO inference failure")
                return super().extract(packet)

        manager = MemoryManager(
            strategy="hierarchical", layers=("long",), place_enabled=True,
            place_extractor=FailsOnce(),
        )
        manager.reset_episode("retry", "find chair")
        try:
            manager.observe(ObservationPacket(0, _frame(), "find chair"))
            self.assertEqual(manager.topological_memory.nodes, [])
            self.assertEqual(manager.episodic.query(kind="place_observation"), [])
            self.assertIn("synthetic DINO inference failure", manager.last_place_error)
            for step in range(1, 6):
                manager.observe(ObservationPacket(step, _frame(), "find chair"))
            self.assertEqual(len(manager.topological_memory.nodes), 1)
            self.assertIsNone(manager.last_place_error)
            self.assertEqual(len(manager.episodic.query(kind="place_observation")), 1)
        finally:
            manager.close_episode()

    def test_reset_clears_previous_place_and_semantics(self) -> None:
        descriptor = MockPlaceDescriptorExtractor({0: (1.0, 0.0)})
        manager = MemoryManager(
            strategy="hierarchical", layers=("long",), place_enabled=True,
            place_extractor=descriptor,
            extractor=MockSceneExtractor({
                0: SceneExtraction(detections=[DetectedObject("chair", 0.8)], confidence=0.8),
            }),
        )
        manager.reset_episode("first", "find chair")
        manager.observe(ObservationPacket(0, _frame(), "find chair"))
        self.assertEqual(len(manager.topological_memory.nodes), 1)
        manager.reset_episode("second", "look at table")
        self.assertEqual(manager.topological_memory.nodes, [])
        self.assertEqual(manager.topological_memory.edges, [])
        self.assertIsNone(manager.topological_memory.current_place_id)
        self.assertEqual(manager.episodic.events, [])
        manager.close_episode()


if __name__ == "__main__":
    unittest.main()
