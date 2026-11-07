"""Pure CPU tests for visual place identity and observed transition evidence."""

from __future__ import annotations

import json
import math
import unittest

from lightnav.memory.schemas import MemoryEvent, MemoryQuery
from lightnav.memory.topological_memory import PlaceEdge, PlaceNode, TopologicalMemory


def _seen(
    memory: TopologicalMemory,
    step: int,
    descriptor: tuple[float, ...],
    *,
    events: list[MemoryEvent] | None = None,
    action_ids: tuple[str, ...] = (),
):
    return memory.observe(
        step, descriptor, events or [], descriptor_source="mock_dinov2",
        observation_event_id=f"place_observation_{step}",
        transition_evidence_ids=action_ids,
    )


def _object(event_id: str, step: int, label: str, node_id: str, confidence: float):
    return MemoryEvent(
        event_id, step, "object_observation", f"Observed {label}",
        confidence, "mock_yolo", metadata={"node_id": node_id},
    )


class TopologicalMemoryTests(unittest.TestCase):
    def test_first_place_and_descriptor_are_normalized(self) -> None:
        memory = TopologicalMemory()
        result = _seen(memory, 0, (3.0, 4.0, 0.0))
        self.assertTrue(result.created)
        self.assertFalse(result.matched)
        self.assertIsNone(result.top1_similarity)
        self.assertEqual(result.place_id, "place_000")
        node = memory.nodes[0]
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in node.visual_descriptor)), 1.0,
        )
        self.assertEqual((node.observation_count, node.visit_count), (1, 1))
        self.assertEqual(node.evidence_event_ids, ["place_observation_0"])
        self.assertEqual(memory.edges, [])
        current = memory.retrieve(MemoryQuery("find chair", 0))[0]
        self.assertEqual(current.confidence, 0.5)
        self.assertEqual(current.evidence_refs[0].confidence, 1.0)
        self.assertEqual(PlaceNode.from_dict(json.loads(json.dumps(node.to_dict()))), node)
        self.assertEqual(memory.retrieve(MemoryQuery("find couch", 0))[0].confidence, 0.5)

    def test_threshold_and_consecutive_same_place_do_not_add_visits_or_edges(self) -> None:
        memory = TopologicalMemory(similarity_threshold=0.85)
        first = _seen(memory, 0, (1.0, 0.0))
        second = _seen(memory, 5, (0.9, 0.1), action_ids=("action_4",))
        self.assertEqual(second.place_id, first.place_id)
        self.assertTrue(second.matched)
        self.assertGreaterEqual(second.top1_similarity, memory.similarity_threshold)
        self.assertEqual((memory.nodes[0].observation_count, memory.nodes[0].visit_count), (2, 1))
        self.assertEqual(memory.edges, [])
        self.assertFalse(second.revisited)

        third = _seen(memory, 10, (0.0, 1.0), action_ids=("action_9",))
        self.assertTrue(third.created)
        self.assertLess(third.top1_similarity, memory.similarity_threshold)
        self.assertEqual(len(memory.nodes), 2)
        self.assertEqual(len(memory.edges), 1)

    def test_distinct_places_need_motion_evidence_for_directed_edge(self) -> None:
        memory = TopologicalMemory()
        a = _seen(memory, 0, (1.0, 0.0))
        b = _seen(memory, 5, (0.0, 1.0))
        self.assertEqual(memory.edges, [])
        self.assertIsNone(b.transition)

        back = _seen(memory, 10, (1.0, 0.0), action_ids=("action_8", "action_9"))
        self.assertEqual(back.place_id, a.place_id)
        self.assertTrue(back.revisited)
        self.assertEqual((memory.get_node(a.place_id).visit_count,
                          memory.get_node(a.place_id).observation_count), (2, 2))
        self.assertEqual(len(memory.edges), 1)
        edge = memory.edges[0]
        self.assertEqual((edge.source_place_id, edge.target_place_id), (b.place_id, a.place_id))
        self.assertEqual(edge.relation_type, "observed_transition")
        self.assertEqual(edge.evidence_event_ids, [
            "place_observation_5", "action_8", "action_9", "place_observation_10",
        ])
        self.assertEqual(PlaceEdge.from_dict(json.loads(json.dumps(edge.to_dict()))), edge)

    def test_visual_switch_without_motion_does_not_count_as_revisit(self) -> None:
        memory = TopologicalMemory()
        a = _seen(memory, 0, (1.0, 0.0))
        _seen(memory, 5, (0.0, 1.0))
        return_view = _seen(memory, 10, (1.0, 0.0))
        self.assertEqual(return_view.place_id, a.place_id)
        self.assertTrue(return_view.matched)
        self.assertFalse(return_view.revisited)
        self.assertEqual(memory.get_node(a.place_id).observation_count, 2)
        self.assertEqual(memory.get_node(a.place_id).visit_count, 1)
        self.assertEqual(memory.edges, [])

    def test_a_to_b_to_c_to_a_loop_closure_is_visual(self) -> None:
        memory = TopologicalMemory()
        a = _seen(memory, 0, (1.0, 0.0, 0.0))
        b = _seen(memory, 5, (0.0, 1.0, 0.0), action_ids=("action_5",))
        c = _seen(memory, 10, (0.0, 0.0, 1.0), action_ids=("action_10",))
        revisit = _seen(memory, 20, (0.99, 0.01, 0.0), action_ids=("action_20",))

        self.assertEqual(revisit.place_id, a.place_id)
        self.assertTrue(revisit.matched)
        self.assertTrue(revisit.revisited)
        self.assertFalse(revisit.created)
        self.assertEqual(len(memory.nodes), 3)
        self.assertEqual([(edge.source_place_id, edge.target_place_id) for edge in memory.edges], [
            (a.place_id, b.place_id), (b.place_id, c.place_id), (c.place_id, a.place_id),
        ])
        self.assertEqual(memory.get_node(a.place_id).revisit_steps, [20])
        self.assertEqual(memory.get_node(a.place_id).visit_count, 2)
        self.assertTrue(all(edge.confidence <= 0.7 for edge in memory.edges))

    def test_reverse_edge_is_independent_and_repeated_direction_updates_count(self) -> None:
        memory = TopologicalMemory()
        a = _seen(memory, 0, (1.0, 0.0))
        b = _seen(memory, 5, (0.0, 1.0), action_ids=("action_5",))
        _seen(memory, 10, (1.0, 0.0), action_ids=("action_10",))
        _seen(memory, 15, (0.0, 1.0), action_ids=("action_15",))
        forward = memory.get_edge(a.place_id, b.place_id)
        backward = memory.get_edge(b.place_id, a.place_id)
        self.assertIsNotNone(forward)
        self.assertIsNotNone(backward)
        self.assertEqual((forward.transition_count, backward.transition_count), (2, 1))
        self.assertEqual((forward.first_observed_step, forward.last_observed_step), (5, 15))
        self.assertEqual(len(memory.edges), 2)

    def test_yolo_sightings_bind_to_visual_place_without_merging_chair_instances(self) -> None:
        memory = TopologicalMemory()
        seen = _seen(memory, 0, (1.0, 0.0), events=[
            _object("chair_1", 0, "chair", "object_1", 0.49),
            _object("chair_2", 0, "chair", "object_2", 0.33),
            _object("couch_1", 0, "couch", "object_3", 0.81),
        ])
        node = memory.get_node(seen.place_id)
        chair = node.observed_categories["chair"]
        self.assertEqual(chair.observation_count, 2)
        self.assertEqual(chair.evidence_event_ids, ["chair_1", "chair_2"])
        self.assertEqual(chair.observed_object_node_ids, ["object_1", "object_2"])
        self.assertAlmostEqual(chair.mean_confidence, 0.41)
        self.assertEqual(node.observation_count, 1)
        self.assertEqual(node.visit_count, 1)
        self.assertEqual(len(memory.edges), 0)
        self.assertEqual(PlaceNode.from_dict(json.loads(json.dumps(node.to_dict()))), node)

        # Equal YOLO classes do not affect recognition: orthogonal RGB
        # descriptors still produce a new place.
        new = _seen(memory, 5, (0.0, 1.0), events=[
            _object("chair_3", 5, "chair", "object_4", 0.9),
        ], action_ids=("action_5",))
        self.assertNotEqual(seen.place_id, new.place_id)
        self.assertEqual(
            memory.get_node(new.place_id).observed_categories["chair"].observation_count, 1,
        )

    def test_retrieval_separates_current_place_historical_semantics_and_transition(self) -> None:
        memory = TopologicalMemory()
        a = _seen(memory, 0, (1.0, 0.0), events=[
            _object("couch_0", 0, "couch", "object_0", 0.8),
        ])
        b = _seen(memory, 5, (0.0, 1.0), action_ids=("action_5",))
        _seen(memory, 10, (1.0, 0.0), action_ids=("action_10",))
        result = memory.retrieve(MemoryQuery("find couch", 10, top_k=5))
        self.assertEqual(result[0].kind, "current_place")
        self.assertEqual(result[0].label, a.place_id)
        self.assertTrue(any(item.kind == "place_semantics" and item.label == "couch"
                            for item in result))
        self.assertTrue(any(item.kind == "observed_transition" for item in result))
        self.assertTrue(any(item.kind == "revisited_place" for item in result))
        self.assertTrue(all("traversability unverified" in item.description
                            for item in result if item.kind == "observed_transition"))
        self.assertTrue(all(
            ref.event_id.startswith("place_observation_")
            for item in result if item.kind == "observed_transition"
            for ref in item.evidence_refs
        ))
        self.assertTrue(all(
            ref.confidence == 1.0
            for item in result if item.kind == "observed_transition"
            for ref in item.evidence_refs
        ))
        self.assertTrue(any(item.evidence_refs[0].event_id == "couch_0"
                            for item in result if item.kind == "place_semantics"))
        self.assertTrue(all(item.metadata["place_id"] == a.place_id
                            for item in result if item.kind == "place_semantics"))
        self.assertIsNotNone(memory.get_edge(b.place_id, a.place_id))

        later = memory.retrieve(MemoryQuery("find couch", 12, top_k=5))
        self.assertEqual(later[0].kind, "last_recognized_place")
        self.assertIn("no new place descriptor at step 12", later[0].description)

    def test_incident_transition_survives_many_unrelated_categories(self) -> None:
        memory = TopologicalMemory()
        categories = [
            _object(f"object_{index}", 0, f"class{index}", f"node_{index}", 0.9)
            for index in range(7)
        ]
        _seen(memory, 0, (1.0, 0.0), events=categories)
        _seen(memory, 5, (0.0, 1.0), action_ids=("executed_action_5",))
        result = memory.retrieve(MemoryQuery("go forward", 5, top_k=5))
        self.assertEqual(result[0].kind, "current_place")
        self.assertTrue(any(item.kind == "observed_transition" for item in result))

    def test_revisited_place_remains_queryable_after_leaving_it(self) -> None:
        memory = TopologicalMemory()
        a = _seen(memory, 0, (1.0, 0.0, 0.0), events=[
            _object("couch_0", 0, "couch", "object_0", 0.8),
        ])
        _seen(memory, 5, (0.0, 1.0, 0.0), action_ids=("action_5",))
        _seen(memory, 10, (1.0, 0.0, 0.0), action_ids=("action_10",))
        c = _seen(memory, 15, (0.0, 0.0, 1.0), action_ids=("action_15",))
        self.assertNotEqual(c.place_id, a.place_id)
        results = memory.retrieve(MemoryQuery("find couch", 15, top_k=8))
        self.assertEqual(results[0].kind, "current_place")
        self.assertEqual(results[0].label, c.place_id)
        revisit = [item for item in results if item.kind == "revisited_place"
                   and item.label == a.place_id]
        self.assertEqual(len(revisit), 1)
        self.assertIn("Historical visual place", revisit[0].description)
        self.assertEqual(revisit[0].step_id, 10)
        self.assertEqual(revisit[0].evidence_refs[0].event_id, "place_observation_10")
        self.assertEqual(revisit[0].metadata["current"], False)
        historical_couch = [item for item in results if item.kind == "place_semantics"
                            and item.label == "couch"]
        self.assertEqual(len(historical_couch), 1)
        self.assertIn("At historical visual place", historical_couch[0].description)
        self.assertEqual(historical_couch[0].metadata["place_id"], a.place_id)
        self.assertEqual(historical_couch[0].evidence_refs[0].event_id, "couch_0")

    def test_revisit_confidence_keeps_its_original_step_provenance(self) -> None:
        memory = TopologicalMemory()
        a = _seen(memory, 0, (1.0, 0.0, 0.0))
        _seen(memory, 5, (0.0, 1.0, 0.0), action_ids=("action_5",))
        _seen(memory, 10, (1.0, 0.0, 0.0), action_ids=("action_10",))
        _seen(memory, 15, (0.9, 0.1, 0.0))
        _seen(memory, 20, (0.0, 1.0, 0.0), action_ids=("action_20",))
        revisit = next(
            item for item in memory.retrieve(MemoryQuery("find place", 20, top_k=10))
            if item.kind == "revisited_place" and item.label == a.place_id
        )
        self.assertEqual(revisit.evidence_refs[0].event_id, "place_observation_10")
        self.assertEqual(revisit.evidence_refs[0].step_id, 10)
        self.assertAlmostEqual(revisit.evidence_refs[0].confidence, 1.0)
        self.assertAlmostEqual(revisit.confidence, 1.0)
        self.assertLess(memory.get_node(a.place_id).descriptor_similarities[2], 1.0)

    def test_bad_descriptor_or_event_leaves_state_unchanged(self) -> None:
        memory = TopologicalMemory()
        _seen(memory, 0, (1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "dimension changed"):
            _seen(memory, 5, (1.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            _seen(memory, 5, (math.nan, 1.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            _seen(memory, 5, (1e308, 1e308))
        with self.assertRaisesRegex(ValueError, "observation step"):
            _seen(memory, 0, (1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "current observation step"):
            _seen(memory, 5, (0.0, 1.0), events=[
                _object("wrong_step", 4, "chair", "object_0", 0.9),
            ])
        self.assertEqual(len(memory.nodes), 1)
        self.assertEqual(memory.last_observation_step, 0)
        self.assertEqual(memory.edges, [])

    def test_clear_drops_every_place_and_transition(self) -> None:
        memory = TopologicalMemory()
        _seen(memory, 0, (1.0, 0.0))
        _seen(memory, 5, (0.0, 1.0), action_ids=("action_5",))
        memory.clear()
        self.assertEqual(memory.nodes, [])
        self.assertEqual(memory.edges, [])
        self.assertIsNone(memory.current_place_id)
        self.assertIsNone(memory.last_observation_step)
        self.assertEqual(memory.retrieve(MemoryQuery("find couch", 10)), [])

    def test_retrospective_query_never_exposes_future_place_state(self) -> None:
        memory = TopologicalMemory()
        _seen(memory, 0, (1.0, 0.0))
        _seen(memory, 5, (0.0, 1.0), action_ids=("action_5",))
        self.assertEqual(memory.retrieve(MemoryQuery("find couch", 0)), [])


if __name__ == "__main__":
    unittest.main()
