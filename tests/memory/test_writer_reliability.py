"""Regression checks for conservative memory writes and extractor failures."""

from __future__ import annotations

import unittest

import numpy as np

from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor, NullSceneExtractor
from lightnav.memory.schemas import DetectedObject, ObservationPacket, SceneExtraction


def _rgb(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def _forward_action() -> dict:
    return {
        "action": "velocity_control",
        "action_args": {"linear_velocity": 0.6, "angular_velocity": 0.0},
    }


class MemoryWriterReliabilityTests(unittest.TestCase):
    def test_null_first_frame_leaves_no_memory_or_place(self) -> None:
        manager = MemoryManager()
        manager.reset_episode("episode", "find the chair")

        self.assertEqual(manager.observe(ObservationPacket(0, _rgb(20), "find the chair")), [])
        self.assertEqual(manager.episodic.events, [])
        self.assertEqual(manager.scene_graph.nodes, [])
        self.assertEqual(manager.scene_graph.edges, [])
        self.assertIsNone(manager.working.state.place_id)
        self.assertEqual(manager.get_context().text, "")

    def test_command_and_color_change_are_only_uncertain_events(self) -> None:
        manager = MemoryManager()
        manager.reset_episode("episode", "find the chair")
        manager.observe(ObservationPacket(0, _rgb(0), "find the chair"))
        manager.record_action(_forward_action())
        manager.observe(ObservationPacket(1, _rgb(255), "find the chair"))

        self.assertEqual(manager.scene_graph.nodes, [])
        self.assertEqual(manager.scene_graph.query_edges(relation="connected_to"), [])
        transitions = manager.episodic.query(kind="possible_transition")
        self.assertEqual(len(transitions), 1)
        self.assertLess(transitions[0].confidence, 0.5)
        self.assertIn("unverified", transitions[0].description)
        self.assertIsNone(transitions[0].place_id)

    def test_semantic_label_change_only_adds_possible_transition(self) -> None:
        extractor = MockSceneExtractor({
            0: SceneExtraction(place_label="hall", confidence=0.9, source="script"),
            1: SceneExtraction(place_label="kitchen", confidence=0.9, source="script"),
        })
        manager = MemoryManager(extractor=extractor)
        manager.reset_episode("episode", "enter the kitchen")
        manager.observe(ObservationPacket(0, _rgb(0), "enter the kitchen"))
        manager.record_action(_forward_action())
        manager.observe(ObservationPacket(1, _rgb(255), "enter the kitchen"))

        self.assertEqual(len(manager.scene_graph.query_nodes(kind="place")), 2)
        self.assertEqual(manager.scene_graph.query_edges(relation="connected_to"), [])
        transitions = manager.scene_graph.query_edges(relation="possible_transition")
        self.assertEqual(len(transitions), 1)
        self.assertIn("unverified", transitions[0].evidence)

    def test_label_change_without_motion_does_not_create_new_place(self) -> None:
        extractor = MockSceneExtractor({
            0: SceneExtraction(place_label="hall", confidence=0.9, source="script"),
            1: SceneExtraction(place_label="kitchen", confidence=0.9, source="script"),
        })
        manager = MemoryManager(extractor=extractor)
        manager.reset_episode("episode", "find the kitchen")
        manager.observe(ObservationPacket(0, _rgb(0), "find the kitchen"))
        manager.observe(ObservationPacket(1, _rgb(0), "find the kitchen"))

        self.assertIsNone(manager.working.state.place_id)
        self.assertEqual(len(manager.scene_graph.query_nodes(kind="place")), 1)
        self.assertEqual(manager.scene_graph.edges, [])
        hypotheses = manager.episodic.query(kind="place_hypothesis")
        self.assertEqual(len(hypotheses), 1)
        self.assertIsNone(hypotheses[0].place_id)

    def test_same_place_name_after_motion_does_not_attach_new_object(self) -> None:
        extractor = MockSceneExtractor({
            0: SceneExtraction(place_label="hall", confidence=0.9, source="script"),
            1: SceneExtraction(
                place_label="hall", objects=["chair"], confidence=0.9, source="script",
            ),
            2: SceneExtraction(place_label="hall", confidence=0.9, source="script"),
        })
        manager = MemoryManager(extractor=extractor)
        manager.reset_episode("episode", "find a chair")
        manager.observe(ObservationPacket(0, _rgb(0), "find a chair"))
        place = manager.scene_graph.query_nodes(kind="place")[0]
        manager.record_action(_forward_action())
        manager.observe(ObservationPacket(1, _rgb(0), "find a chair"))

        self.assertEqual(place.last_step, 0)
        self.assertEqual(len(manager.scene_graph.query_nodes(kind="object")), 1)
        self.assertEqual(manager.scene_graph.query_edges(relation="visible_from"), [])
        observations = manager.episodic.query(kind="object_observation")
        self.assertEqual(len(observations), 1)
        self.assertIsNone(observations[0].place_id)

        manager.observe(ObservationPacket(2, _rgb(0), "find a chair"))
        later_place = manager.working.state.place_id
        self.assertNotEqual(later_place, place.node_id)
        self.assertEqual(len(manager.scene_graph.query_nodes(kind="place", label="hall")), 2)
        self.assertEqual(manager.scene_graph.query_edges(relation="connected_to"), [])

    def test_unresolved_motion_stays_unlocated_on_later_object_only_frame(self) -> None:
        extractor = MockSceneExtractor({
            0: SceneExtraction(place_label="hall", confidence=0.9, source="script"),
            2: SceneExtraction(objects=["chair"], confidence=0.9, source="script"),
        })
        manager = MemoryManager(extractor=extractor)
        manager.reset_episode("episode", "find a chair")
        manager.observe(ObservationPacket(0, _rgb(0), "find a chair"))
        old_place = manager.working.state.place_id
        manager.record_action(_forward_action())
        manager.observe(ObservationPacket(1, _rgb(0), "find a chair"))
        self.assertIsNone(manager.working.state.place_id)

        manager.observe(ObservationPacket(2, _rgb(0), "find a chair"))
        self.assertEqual(len(manager.scene_graph.query_nodes(kind="place")), 1)
        self.assertEqual(len(manager.scene_graph.query_nodes(kind="object")), 1)
        self.assertEqual(manager.scene_graph.query_edges(relation="visible_from"), [])
        observations = manager.episodic.query(kind="object_observation")
        self.assertEqual(len(observations), 1)
        self.assertIsNone(observations[0].place_id)
        self.assertEqual(manager.scene_graph.get_node(old_place).last_step, 0)

    def test_object_without_place_has_no_place_or_relation(self) -> None:
        extractor = MockSceneExtractor({
            0: SceneExtraction(objects=["chair"], confidence=0.8, source="script"),
        })
        manager = MemoryManager(extractor=extractor)
        manager.reset_episode("episode", "find the chair")
        manager.observe(ObservationPacket(0, _rgb(0), "find the chair"))

        self.assertEqual(len(manager.scene_graph.query_nodes(kind="object")), 1)
        self.assertEqual(manager.scene_graph.query_nodes(kind="place"), [])
        self.assertEqual(manager.scene_graph.edges, [])
        observations = manager.episodic.query(kind="object_observation")
        self.assertEqual(len(observations), 1)
        self.assertIsNone(observations[0].place_id)
        self.assertEqual(observations[0].source, "mock_scene_extractor")

    def test_same_name_detections_do_not_merge_objects(self) -> None:
        extractor = MockSceneExtractor({
            0: SceneExtraction(
                place_label="hall", objects=["chair", "chair"],
                confidence=0.8, source="script",
            ),
        })
        manager = MemoryManager(extractor=extractor)
        manager.reset_episode("episode", "find a chair")
        manager.observe(ObservationPacket(0, _rgb(0), "find a chair"))

        objects = manager.scene_graph.query_nodes(kind="object")
        self.assertEqual(len(objects), 2)
        self.assertNotEqual(objects[0].node_id, objects[1].node_id)
        self.assertEqual(len(manager.scene_graph.query_edges(relation="visible_from")), 2)

    def test_extractor_exception_does_not_advance_step_and_can_retry(self) -> None:
        class FlakyExtractor(NullSceneExtractor):
            def __init__(self) -> None:
                self.calls = 0

            def extract(self, packet: ObservationPacket) -> SceneExtraction:
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("temporary extraction failure")
                if self.calls == 3:
                    return SceneExtraction("hall", confidence=0.9, source="flaky")
                return SceneExtraction()

        manager = MemoryManager(extractor=FlakyExtractor())
        manager.reset_episode("episode", "find the hall")
        manager.observe(ObservationPacket(0, _rgb(0), "find the hall"))
        manager.record_action(_forward_action())
        before = manager.working.state.to_dict()

        with self.assertRaisesRegex(RuntimeError, "temporary extraction failure"):
            manager.observe(ObservationPacket(1, _rgb(255), "find the hall"))
        self.assertEqual(manager.working.state.to_dict(), before)
        self.assertEqual(manager.episodic.events, [])
        self.assertEqual(manager.scene_graph.nodes, [])

        manager.observe(ObservationPacket(1, _rgb(255), "find the hall"))
        self.assertEqual(manager.working.state.step_id, 1)
        self.assertEqual(len(manager.episodic.query(kind="action")), 1)
        self.assertEqual(len(manager.episodic.query(kind="visual_change")), 1)
        self.assertEqual(len(manager.scene_graph.query_nodes(kind="place")), 1)

    def test_invalid_extraction_does_not_advance_step(self) -> None:
        class InvalidThenValidExtractor(NullSceneExtractor):
            def __init__(self) -> None:
                self.calls = 0

            def extract(self, packet: ObservationPacket) -> SceneExtraction:
                self.calls += 1
                if self.calls == 1:
                    return SceneExtraction("hall", confidence=float("nan"), source="invalid")
                return SceneExtraction("hall", confidence=0.8, source="valid")

        manager = MemoryManager(extractor=InvalidThenValidExtractor())
        manager.reset_episode("episode", "find the hall")
        with self.assertRaisesRegex(ValueError, "extractor confidence"):
            manager.observe(ObservationPacket(0, _rgb(0), "find the hall"))

        self.assertEqual(manager.working.state.step_id, -1)
        self.assertEqual(manager.episodic.events, [])
        self.assertEqual(manager.scene_graph.nodes, [])
        manager.observe(ObservationPacket(0, _rgb(0), "find the hall"))
        self.assertEqual(manager.working.state.step_id, 0)
        self.assertEqual(len(manager.scene_graph.query_nodes(kind="place")), 1)

    def test_invalid_object_confidence_is_rejected_before_any_write(self) -> None:
        extractor = MockSceneExtractor({
            0: SceneExtraction(
                confidence=0.9, source="script",
                detections=[DetectedObject("chair", float("nan"))],
            ),
        })
        manager = MemoryManager(extractor=extractor)
        manager.reset_episode("episode", "find a chair")
        with self.assertRaisesRegex(ValueError, "detection confidence"):
            manager.observe(ObservationPacket(0, _rgb(0), "find a chair"))
        self.assertEqual(manager.working.state.step_id, -1)
        self.assertEqual(manager.episodic.events, [])
        self.assertEqual(manager.scene_graph.nodes, [])


if __name__ == "__main__":
    unittest.main()
