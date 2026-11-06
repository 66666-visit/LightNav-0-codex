"""Independent mid-term event index and retrieval, without RGB or model weights."""

from __future__ import annotations

import json
import unittest

from lightnav.memory.episodic_memory import EpisodicMemory, EpisodicSegment
from lightnav.memory.schemas import MemoryEvent, MemoryQuery


def _event(
    event_id: str,
    step: int,
    kind: str,
    description: str,
    *,
    confidence: float = 0.8,
    importance: float = 0.6,
    node_id: str | None = None,
) -> MemoryEvent:
    metadata = {"node_id": node_id} if node_id is not None else {}
    return MemoryEvent(
        event_id, step, kind, description, confidence, "rgb_or_action",
        importance=importance, metadata=metadata,
    )


class MidLayerTest(unittest.TestCase):
    def test_disabled_mid_index_keeps_flat_event_log_without_layer_state(self) -> None:
        memory = EpisodicMemory(index_enabled=False)
        couch = _event("seen", 5, "object_observation", "Observed couch", node_id="object_0")
        memory.add(couch)
        self.assertEqual(memory.events, [couch])
        self.assertEqual(memory.key_events, [])
        self.assertEqual(memory.segments, [])
        self.assertEqual(memory.retrieve(MemoryQuery("couch", current_step=5)), [])

    def test_raw_log_and_key_index_are_separate(self) -> None:
        memory = EpisodicMemory()
        couch = _event("seen", 5, "object_observation", "Observed couch", node_id="object_0")
        action = _event("move", 6, "action", "Issued velocity_control", confidence=1.0)
        turn = _event("turn", 7, "turn", "Issued turn", confidence=1.0)
        for record in (couch, action, turn):
            memory.add(record)

        self.assertEqual(memory.events, [couch, action, turn])
        self.assertEqual(memory.key_events, [couch, turn])
        self.assertEqual(memory.query(kind="action"), [action])
        self.assertEqual([segment.kind for segment in memory.segments], [
            "object_observation", "turn",
        ])

    def test_temporal_segments_query_time_label_instruction_and_evidence(self) -> None:
        memory = EpisodicMemory()
        memory.add(_event("couch_a", 5, "object_observation", "Observed couch",
                          node_id="object_a", confidence=0.8))
        memory.add(_event("couch_b", 6, "object_observation", "Observed couch",
                          node_id="object_b", confidence=0.6))
        memory.add(_event("couch_c", 10, "object_observation", "Observed couch",
                          node_id="object_c", confidence=0.9))

        segments = memory.query_segments(kind="object_observation", label="couch")
        self.assertEqual([(s.start_step, s.end_step) for s in segments], [(5, 6), (10, 10)])
        self.assertEqual(segments[0].event_ids, ["couch_a", "couch_b"])
        self.assertEqual(segments[0].node_ids, ["object_a", "object_b"])
        self.assertEqual(len(memory.query_segments(since_step=6, until_step=6)), 1)
        self.assertEqual(len(memory.query_segments(instruction="find the couch")), 2)
        self.assertEqual(memory.query_segments(evidence_id="object_b"), [segments[0]])
        self.assertEqual(memory.query_segments(evidence_id="couch_c"), [segments[1]])
        self.assertEqual(len(memory.query_segments(min_confidence=0.85)), 1)
        self.assertEqual(
            EpisodicSegment.from_dict(json.loads(json.dumps(segments[0].to_dict()))), segments[0],
        )

    def test_step_twenty_still_retrieves_step_five_couch_after_actions(self) -> None:
        memory = EpisodicMemory()
        couch = _event("couch", 5, "object_observation", "Observed couch",
                       confidence=0.72, node_id="object_0")
        match = _event("match", 5, "instruction_match", "Observed instruction-related couch",
                       confidence=0.72, importance=0.95, node_id="object_0")
        memory.add(couch)
        memory.add(match)
        for step in range(6, 20):
            memory.add(_event(f"action_{step}", step, "action", "Issued velocity_control",
                              confidence=1.0, importance=0.4))
        memory.add(_event("turn_19", 19, "turn", "Issued turn", confidence=1.0))

        items = memory.retrieve(MemoryQuery("find the couch", current_step=20, top_k=3))
        couch_items = [item for item in items if item.label == "couch"]
        self.assertEqual(len(couch_items), 1)
        self.assertEqual(couch_items[0].kind, "object_observation")
        self.assertEqual(couch_items[0].step_id, 5)
        self.assertEqual(couch_items[0].importance, 0.95)
        self.assertEqual(couch_items[0].evidence_refs[0].event_id, couch.event_id)
        self.assertEqual(couch_items[0].evidence_refs[0].node_id, "object_0")
        self.assertTrue(any(item.kind == "turn" for item in items))
        self.assertFalse(any(item.kind == "action" for item in items))

    def test_same_class_different_nodes_remain_distinct_observations(self) -> None:
        memory = EpisodicMemory()
        memory.add(_event("chair_a", 5, "object_observation", "Observed chair",
                          node_id="object_a"))
        memory.add(_event("chair_b", 5, "object_observation", "Observed chair",
                          node_id="object_b"))
        items = memory.retrieve(MemoryQuery("chair", current_step=5, top_k=5))
        self.assertEqual(len(items), 2)
        self.assertEqual({item.evidence_refs[0].node_id for item in items}, {
            "object_a", "object_b",
        })

    def test_place_label_is_queryable_without_claiming_identity(self) -> None:
        memory = EpisodicMemory()
        memory.add(_event(
            "place_1", 3, "place_visit", "Extractor reported kitchen (place hypothesis)",
            confidence=0.7,
        ))
        memory.add(_event(
            "place_2", 8, "place_visit", "Extractor reported kitchen (place hypothesis)",
            confidence=0.6,
        ))
        matches = memory.query_segments(instruction="go to the kitchen", label="kitchen")
        self.assertEqual([(item.start_step, item.end_step) for item in matches], [
            (3, 3), (8, 8),
        ])
        self.assertNotEqual(matches[0].event_ids, matches[1].event_ids)

    def test_uncertain_transition_stays_an_event_and_clear_removes_indexes(self) -> None:
        memory = EpisodicMemory()
        uncertain = _event(
            "possible", 4, "possible_transition",
            "RGB appearance changed after a forward command; place change unverified",
            confidence=0.35, importance=0.4,
        )
        memory.add(uncertain)
        items = memory.retrieve(MemoryQuery("continue", current_step=4))
        self.assertEqual([item.kind for item in items], ["possible_transition"])
        self.assertIn("unverified", items[0].description)
        memory.clear()
        self.assertEqual(memory.events, [])
        self.assertEqual(memory.key_events, [])
        self.assertEqual(memory.segments, [])
        self.assertEqual(memory.retrieve(MemoryQuery("continue", current_step=4)), [])


if __name__ == "__main__":
    unittest.main()
