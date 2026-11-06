"""CPU-only tests for episode-local, category-level semantic memory."""

from __future__ import annotations

import json
import unittest

from lightnav.memory.scene_graph import SceneGraph
from lightnav.memory.schemas import MemoryEvent, MemoryQuery
from lightnav.memory.semantic_index import SemanticEntry, SemanticIndex


def _detected(
    graph: SceneGraph,
    event_id: str,
    label: str,
    step: int,
    confidence: float,
) -> MemoryEvent:
    node = graph.add_object(
        label,
        place_id=None,
        confidence=confidence,
        source="mock_rgb_detector",
        step_id=step,
        evidence=f"RGB detection at step {step}",
    )
    return MemoryEvent(
        event_id=event_id,
        step_id=step,
        kind="object_observation",
        description=f"Observed {label}",
        confidence=confidence,
        source="mock_rgb_detector",
        importance=0.6,
        metadata={"node_id": node.node_id},
    )


class SemanticIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = SceneGraph()
        self.index = SemanticIndex()

    def test_same_category_keeps_distinct_observations_and_provenance(self) -> None:
        first = _detected(self.graph, "event_1", "chair", 5, 0.49)
        second = _detected(self.graph, "event_2", "chair", 5, 0.33)
        later = _detected(self.graph, "event_3", "chair", 12, 0.80)
        match = MemoryEvent(
            event_id="match_1", step_id=5, kind="instruction_match",
            description="Observed instruction-related chair", confidence=0.49,
            source="mock_rgb_detector", importance=0.95,
            metadata={"node_id": first.metadata["node_id"]},
        )
        self.index.update([first, second, later, match], self.graph, "find the chair")
        self.index.update([first, second], self.graph, "find the chair")

        entry = self.index.get("CHAIR")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.observation_count, 3)
        self.assertEqual((entry.first_step, entry.last_step), (5, 12))
        self.assertAlmostEqual(entry.mean_confidence, (0.49 + 0.33 + 0.80) / 3)
        self.assertEqual(entry.max_confidence, 0.80)
        self.assertTrue(entry.instruction_relevant)
        self.assertEqual({obs.event_id for obs in entry.observations}, {"event_1", "event_2", "event_3"})
        self.assertEqual(len({obs.node_id for obs in entry.observations}), 3)
        self.assertTrue(all(obs.source == "mock_rgb_detector" for obs in entry.observations))
        self.assertTrue(all("RGB detection" in obs.evidence for obs in entry.observations))
        self.assertEqual(len(self.graph.edges), 0)

        restored = SemanticEntry.from_dict(json.loads(json.dumps(entry.to_dict())))
        self.assertEqual(restored, entry)

    def test_step_twenty_retrieves_step_five_couch_after_action_only_steps(self) -> None:
        couch = _detected(self.graph, "couch_at_5", "couch", 5, 0.72)
        self.index.update([couch], self.graph, "find the couch")
        for step in range(6, 21):
            command = MemoryEvent(
                event_id=f"action_{step}", step_id=step, kind="action",
                description="Issued velocity_control", confidence=1.0,
                source="issued_command_and_next_observation", importance=0.4,
            )
            self.index.update([command], self.graph, "find the couch")

        result = self.index.retrieve(MemoryQuery("find the couch", current_step=20))
        self.assertEqual(len(result), 1)
        summary = result[0]
        self.assertEqual((summary.layer, summary.kind, summary.label),
                         ("long", "semantic_summary", "couch"))
        self.assertEqual(summary.step_id, 5)
        self.assertEqual([ref.event_id for ref in summary.evidence_refs], ["couch_at_5"])
        self.assertIn("first step 5", summary.description)
        self.assertIn("instance identity unverified", summary.description)

    def test_only_existing_graph_edges_are_reported_with_evidence_status(self) -> None:
        hall = self.graph.add_place(
            "hall", confidence=0.8, source="visual_place_model", step_id=1,
            evidence="RGB place hypothesis",
        )
        room = self.graph.add_place(
            "room", confidence=0.8, source="visual_place_model", step_id=2,
            evidence="new RGB place hypothesis",
        )
        self.graph.add_edge(
            hall.node_id, room.node_id, "possible_transition", confidence=0.4,
            source="issued_command_and_rgb_semantics", step_id=2,
            evidence="different label after a command; traversal unverified",
        )
        self.graph.add_edge(
            hall.node_id, room.node_id, "connected_to", confidence=0.9,
            source="verified_traversal", step_id=3,
            evidence="observed successful traversal",
        )
        self.index.update([], self.graph, "find the room")

        relations = self.index.retrieve(MemoryQuery("room", current_step=3, top_k=5))
        by_relation = {item.label: item for item in relations}
        self.assertIn("possible_transition", by_relation)
        self.assertIn("connected_to", by_relation)
        self.assertEqual(by_relation["possible_transition"].kind, "uncertain_relation")
        self.assertIn("unverified", by_relation["possible_transition"].description)
        self.assertEqual(by_relation["connected_to"].kind, "semantic_relation")
        self.assertIn("verified traversal", by_relation["connected_to"].description)
        self.assertEqual(by_relation["connected_to"].evidence_refs[0].event_id, "edge_1")
        self.assertEqual(len(self.graph.edges), 2)

    def test_clear_discards_categories_and_graph_reference(self) -> None:
        event = _detected(self.graph, "couch", "couch", 5, 0.7)
        self.index.update([event], self.graph, "find couch")
        self.index.clear()
        self.assertEqual(self.index.entries, [])
        self.assertEqual(self.index.retrieve(MemoryQuery("couch", 20)), [])
        self.assertIsNone(self.index.get("couch"))


if __name__ == "__main__":
    unittest.main()
