"""CPU checks for episode memory records, graph evidence, and lifecycle."""

from __future__ import annotations

import json
import unittest

import numpy as np

from lightnav.memory.episodic_memory import EpisodicMemory
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.scene_graph import SceneGraph
from lightnav.memory.schemas import (
    MemoryContext,
    MemoryEvent,
    MemoryQuery,
    ObservationPacket,
    SceneEdge,
    SceneExtraction,
    SceneNode,
    WorkingMemory,
)
from lightnav.memory.working_memory import WorkingMemoryStore


def _rgb(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def _forward_action() -> dict:
    return {
        "action": "velocity_control",
        "action_args": {"linear_velocity": 0.6, "angular_velocity": 0.0},
    }


class SchemaTests(unittest.TestCase):
    def test_all_records_json_roundtrip(self) -> None:
        instances = [
            WorkingMemory("ep", "walk to the chair", step_id=2, place_id="place_0"),
            MemoryEvent("e1", 2, "landmark", "Observed chair", 0.8, "mock", "place_0"),
            SceneNode("place_0", "place", "hall", 0.7, "mock", 0, 2, visited=True),
            SceneEdge("edge_0", "place_0", "object_1", "contains", 0.8, "mock", 2, "RGB"),
            MemoryQuery("find chair", 2, "place_0", top_k=3, max_chars=200),
            SceneExtraction("hall", ["chair"], ["door"], 0.8, "mock"),
        ]
        for record in instances:
            with self.subTest(type=type(record).__name__):
                encoded = json.loads(json.dumps(record.to_dict()))
                self.assertEqual(type(record).from_dict(encoded), record)

        context = MemoryContext(
            events=[instances[1]], nodes=[instances[2]], edges=[instances[3]], text="chair",
        )
        self.assertEqual(MemoryContext.from_dict(json.loads(json.dumps(context.to_dict()))), context)

        packet = ObservationPacket(2, _rgb(20), "find chair", _forward_action(), "<traj_1>")
        data = json.loads(json.dumps(packet.to_dict()))
        self.assertNotIn("rgb", data)
        restored = ObservationPacket.from_dict(data)
        self.assertIsNone(restored.rgb)
        self.assertEqual(restored.step_id, packet.step_id)
        self.assertEqual(restored.last_action, packet.last_action)


class WorkingMemoryTests(unittest.TestCase):
    def test_update_recent_events_and_action(self) -> None:
        working = WorkingMemoryStore(recent_limit=2)
        working.reset("ep1", "find chair")
        working.advance(0, place_id="p0", place_confidence=0.2)
        working.advance(1, last_action=_forward_action(), prediction="<traj_1>")
        for index in range(3):
            working.record_event(MemoryEvent(f"e{index}", 1, "action", "move", 1.0, "command"))
        working.record_instruction_match("chair")
        self.assertEqual(working.state.step_id, 1)
        self.assertEqual(working.state.place_id, "p0")
        self.assertEqual(working.state.recent_event_ids, ["e1", "e2"])
        self.assertEqual(working.state.last_action, _forward_action())
        self.assertEqual(working.state.instruction_matches, ["chair"])
        with self.assertRaises(ValueError):
            working.advance(1)
        working.close()
        self.assertIsNone(working.state)


class EpisodicMemoryTests(unittest.TestCase):
    def test_write_query_and_clear(self) -> None:
        journal = EpisodicMemory()
        journal.add(MemoryEvent("e1", 0, "landmark", "chair", 0.8, "rgb", "p0"))
        journal.add(MemoryEvent("e2", 2, "turn", "left", 1.0, "command", "p1"))
        self.assertEqual([e.event_id for e in journal.query(kind="turn")], ["e2"])
        self.assertEqual([e.event_id for e in journal.query(place_id="p0")], ["e1"])
        self.assertEqual([e.event_id for e in journal.query(since_step=1)], ["e2"])
        with self.assertRaises(ValueError):
            journal.add(MemoryEvent("e1", 3, "landmark", "table", 0.8, "rgb"))
        journal.clear()
        self.assertEqual(journal.events, [])


class SceneGraphTests(unittest.TestCase):
    def test_nodes_edges_updates_queries_and_place_identity(self) -> None:
        graph = SceneGraph()
        first = graph.add_place("hall", confidence=0.8, source="rgb_model", step_id=0, evidence="RGB")
        same_text = graph.add_place("hall", confidence=0.7, source="rgb_model", step_id=4, evidence="new RGB")
        self.assertNotEqual(first.node_id, same_text.node_id)
        self.assertEqual(first.status, "uncertain")
        chair = graph.add_object(
            "chair", place_id=first.node_id, confidence=0.7,
            source="rgb_model", step_id=1, evidence="RGB detection",
        )
        repeated = graph.add_object(
            "Chair", place_id=first.node_id, confidence=0.9,
            source="rgb_model", step_id=2, evidence="second RGB detection",
        )
        self.assertNotEqual(chair.node_id, repeated.node_id)
        self.assertEqual(chair.confidence, 0.7)
        another = graph.add_object(
            "chair", place_id=same_text.node_id, confidence=0.7,
            source="rgb_model", step_id=4, evidence="new RGB detection",
        )
        self.assertNotEqual(chair.node_id, another.node_id)
        self.assertEqual(len(graph.query_nodes(kind="place", label="hall", visited=True)), 2)
        self.assertEqual(graph.query_nodes(kind="object", place_id=first.node_id), [chair, repeated])
        self.assertEqual(len(graph.query_edges(relation="visible_from")), 3)
        with self.assertRaises(ValueError):
            graph.add_edge(
                first.node_id, same_text.node_id, "connected_to", confidence=0.6,
                source="text_similarity", step_id=4, evidence="same label",
            )
        transition = graph.add_edge(
            first.node_id, same_text.node_id, "possible_transition", confidence=0.6,
            source="executed_action_and_observation", step_id=4,
            evidence="forward command then new RGB view; displacement unverified",
        )
        self.assertEqual(graph.query_edges(relation="possible_transition"), [transition])
        self.assertEqual(graph.query_edges(relation="connected_to"), [])
        graph.update_node(first.node_id, step_id=5, attributes={"note": "seen again"})
        self.assertEqual(first.last_step, 5)
        graph.clear()
        self.assertEqual(graph.nodes, [])
        self.assertEqual(graph.edges, [])


class MemoryManagerTests(unittest.TestCase):
    def test_extractor_episode_hooks_are_called(self) -> None:
        class StatefulExtractor(MockSceneExtractor):
            def __init__(self) -> None:
                super().__init__()
                self.resets: list[str] = []
                self.closes = 0

            def reset_episode(self, episode_id: str, instruction: str) -> None:
                self.resets.append(episode_id)

            def close_episode(self) -> None:
                self.closes += 1

        extractor = StatefulExtractor()
        manager = MemoryManager(extractor=extractor)
        manager.reset_episode("one", "go")
        manager.observe(ObservationPacket(0, _rgb(0), "go"))
        manager.reset_episode("two", "stop")
        self.assertEqual(extractor.resets, ["one", "two"])
        self.assertEqual(extractor.closes, 1)
        manager.close_episode()
        manager.close_episode()
        self.assertEqual(extractor.closes, 2)

    def test_lifecycle_and_no_cross_episode_leak(self) -> None:
        manager = MemoryManager()
        manager.reset_episode("episode_one", "find the chair")
        manager.observe(ObservationPacket(0, _rgb(0), "find the chair"))
        self.assertTrue(manager.active)
        self.assertEqual(manager.episodic.events, [])
        self.assertEqual(manager.scene_graph.nodes, [])
        self.assertEqual(manager.get_context().text, "")
        manager.record_action(_forward_action(), prediction="<traj_1>")
        manager.observe(ObservationPacket(1, _rgb(255), "find the chair"))
        self.assertTrue(manager.episodic.query(kind="visual_change"))
        self.assertTrue(manager.episodic.query(kind="action"))
        self.assertEqual(manager.scene_graph.query_edges(relation="connected_to"), [])
        self.assertIn("RGB", manager.get_context().text)
        manager.close_episode()
        manager.close_episode()
        self.assertFalse(manager.active)
        self.assertEqual(manager.episodic.events, [])
        self.assertEqual(manager.scene_graph.nodes, [])
        with self.assertRaises(RuntimeError):
            manager.retrieve()
        manager.reset_episode("episode_two", "go to the table")
        manager.observe(ObservationPacket(0, _rgb(0), "go to the table"))
        self.assertFalse(any("chair" in e.description for e in manager.episodic.events))
        self.assertFalse(manager.episodic.query(kind="action"))

    def test_mock_rgb_semantic_pipeline_and_evidence(self) -> None:
        extractor = MockSceneExtractor({
            0: SceneExtraction("hall", ["red chair"], ["red chair"], 0.9, "ignored"),
            1: SceneExtraction("kitchen", ["table"], ["table"], 0.85, "ignored"),
            2: SceneExtraction("hall", ["red chair"], [], 0.8, "ignored"),
        })
        manager = MemoryManager(extractor=extractor)
        manager.reset_episode("mock_episode", "pass the red chair then enter the kitchen")
        manager.observe(ObservationPacket(0, _rgb(0), "pass the red chair then enter the kitchen"))
        first_id = manager.working.state.place_id
        self.assertEqual(manager.working.state.place_source, "mock_scene_extractor")
        self.assertEqual(manager.scene_graph.query_nodes(kind="object", place_id=first_id)[0].label, "red chair")
        self.assertEqual(manager.working.state.instruction_matches, ["red chair"])
        manager.record_action(_forward_action())
        manager.observe(ObservationPacket(1, _rgb(120), "pass the red chair then enter the kitchen"))
        second_id = manager.working.state.place_id
        self.assertNotEqual(first_id, second_id)
        manager.record_action(_forward_action())
        manager.observe(ObservationPacket(2, _rgb(0), "pass the red chair then enter the kitchen"))
        third_id = manager.working.state.place_id
        self.assertNotEqual(first_id, third_id)  # repeated label is not identity proof
        self.assertEqual(len(manager.scene_graph.query_edges(relation="possible_transition")), 2)
        self.assertEqual(manager.scene_graph.query_edges(relation="connected_to"), [])
        self.assertEqual(len(manager.scene_graph.query_nodes(kind="place", label="hall")), 2)
        self.assertTrue(manager.episodic.query(kind="instruction_match"))
        context = manager.retrieve(MemoryQuery("red chair", 2, third_id, top_k=3, max_chars=280))
        self.assertLessEqual(len(context.text), 280)
        self.assertIn("source=", context.text)
        self.assertNotIn("255, 255", context.text)

    def test_rejects_invalid_frame_and_instruction_change(self) -> None:
        manager = MemoryManager()
        manager.reset_episode("one", "turn left")
        with self.assertRaises(ValueError):
            manager.observe(ObservationPacket(0, np.zeros((4, 4), dtype=np.uint8), "turn left"))
        manager.observe(ObservationPacket(0, _rgb(0), "turn left"))
        with self.assertRaises(ValueError):
            manager.observe(ObservationPacket(1, _rgb(0), "different"))


if __name__ == "__main__":
    unittest.main()
