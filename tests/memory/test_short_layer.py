"""L1 keeps bounded, step-specific semantic state without storing RGB."""

from __future__ import annotations

import unittest

from lightnav.memory.schemas import MemoryEvent, MemoryQuery
from lightnav.memory.working_memory import WorkingMemoryStore


def _object(event_id: str, step: int, label: str, node_id: str, confidence: float) -> MemoryEvent:
    return MemoryEvent(
        event_id, step, "object_observation", f"Observed {label}", confidence,
        "mock_rgb_detector", importance=0.6, metadata={"node_id": node_id},
    )


def _action(event_id: str, step: int, *, turn: bool = False) -> MemoryEvent:
    action = {"action": "velocity_control", "action_args": {"angular_velocity": 0.4 if turn else 0.0}}
    return MemoryEvent(
        event_id, step, "turn" if turn else "action",
        "Issued a turn before this view" if turn else "Issued forward command before this view",
        1.0, "issued_command_and_next_observation", importance=0.7 if turn else 0.4,
        metadata={"action": action},
    )


class ShortMemoryTests(unittest.TestCase):
    def test_current_detection_and_carried_sighting_are_distinct(self) -> None:
        store = WorkingMemoryStore(recent_limit=4)
        store.reset("ep", "find the couch")
        store.advance(0)
        store.record_observation(0, [_object("e0", 0, "couch", "object_0", 0.83)])
        current = store.get_context()
        self.assertEqual([item.kind for item in current], ["current_observation"])
        self.assertEqual(current[0].evidence_refs[0].event_id, "e0")
        self.assertEqual(current[0].evidence_refs[0].node_id, "object_0")

        store.advance(1)
        store.record_observation(1, [_action("e1", 1, turn=True)])
        carried = store.get_context()
        self.assertEqual([item.kind for item in carried], ["recent_observation", "recent_turn"])
        self.assertEqual(carried[0].step_id, 0)
        self.assertIn("no current detection at step 1", carried[0].description)
        self.assertEqual(store.recent_observations[-1].detections, ())
        self.assertIsNone(store.recent_observations[-1].place_id)
        self.assertIsNotNone(store.recent_observations[-1].motions[0].action)
        store.close()

    def test_bounded_recent_history_does_not_erase_original_event_identity(self) -> None:
        store = WorkingMemoryStore(recent_limit=3)
        store.reset("ep", "find the couch")
        store.advance(0)
        store.record_observation(0, [_object("e0", 0, "couch", "object_0", 0.75)])
        for step in range(1, 4):
            store.advance(step)
            store.record_observation(step, [])
        self.assertEqual([obs.step_id for obs in store.recent_observations], [1, 2, 3])
        self.assertEqual(store.retrieve(MemoryQuery("find couch", 3, top_k=3)), [])
        store.close()

    def test_two_chairs_in_one_frame_remain_two_observations(self) -> None:
        store = WorkingMemoryStore()
        store.reset("ep", "walk ahead")
        store.advance(5)
        store.record_observation(5, [
            _object("e0", 5, "chair", "object_0", 0.49),
            _object("e1", 5, "chair", "object_1", 0.33),
        ])
        context = store.get_context()
        self.assertEqual([item.kind for item in context], ["current_observation"] * 2)
        self.assertEqual({item.evidence_refs[0].node_id for item in context}, {"object_0", "object_1"})
        store.close()

    def test_place_is_unknown_without_evidence_and_reset_clears_short_state(self) -> None:
        store = WorkingMemoryStore()
        store.reset("ep1", "find a doorway")
        store.advance(0)
        store.record_observation(0, [])
        self.assertEqual(store.get_context(), [])
        self.assertIsNone(store.recent_observations[-1].place_id)

        store.advance(1)
        store.set_place("place_0", 0.7, "mock_rgb_detector")
        place = MemoryEvent(
            "p1", 1, "place_visit", "Extractor reported doorway (place hypothesis)",
            0.7, "mock_rgb_detector", place_id="place_0", importance=0.7,
        )
        store.record_observation(1, [place])
        self.assertEqual(store.get_context()[0].kind, "place_hypothesis")
        self.assertIn("uncertain", store.get_context()[0].description)

        store.reset("ep2", "find couch")
        self.assertEqual(store.recent_observations, ())
        store.advance(0)
        store.record_observation(0, [])
        self.assertEqual(store.get_context(), [])
        store.close()
        self.assertEqual(store.recent_observations, ())

    def test_place_hypothesis_carries_original_evidence_until_cleared(self) -> None:
        store = WorkingMemoryStore(recent_limit=2)
        store.reset("ep", "find a doorway")
        store.advance(0)
        store.set_place("place_0", 0.72, "mock_rgb_detector")
        visit = MemoryEvent(
            "p0", 0, "place_visit", "Extractor reported doorway (place hypothesis)",
            0.72, "mock_rgb_detector", place_id="place_0", importance=0.7,
        )
        store.record_observation(0, [visit])

        for step in (1, 2):
            store.advance(step)
            observation = store.record_observation(step, [])
            self.assertEqual(observation.place_event_id, "p0")
            self.assertEqual(observation.place_evidence.step_id, 0)
            self.assertEqual(observation.place_evidence.source, "mock_rgb_detector")
            self.assertEqual(observation.place_evidence.evidence, "episodic_event:p0")
            self.assertEqual(
                observation.place_event_description,
                "Extractor reported doorway (place hypothesis)",
            )
            context = store.get_context()
            self.assertEqual(len(context), 1)
            self.assertEqual(context[0].kind, "place_hypothesis")
            self.assertEqual(context[0].step_id, 0)
            self.assertEqual(context[0].source, "mock_rgb_detector")
            self.assertEqual(context[0].evidence_refs[0].event_id, "p0")
            self.assertIn("Carried place hypothesis", context[0].description)
            self.assertIn("Extractor reported doorway", context[0].description)
            self.assertIn("no new place evidence this step", context[0].description)

        # The original visit has fallen out of the bounded deque; its evidence
        # remains attached to the carried hypothesis while the place is active.
        self.assertEqual([obs.step_id for obs in store.recent_observations], [1, 2])
        store.clear_place()
        self.assertEqual(store.get_context(), [])
        store.advance(3)
        cleared = store.record_observation(3, [])
        self.assertIsNone(cleared.place_id)
        self.assertIsNone(cleared.place_evidence)
        self.assertEqual(store.get_context(), [])

        # A newly named place without a place_visit event has no provenance.
        store.advance(4)
        store.set_place("place_1", 0.6, "mock_rgb_detector")
        store.record_observation(4, [])
        self.assertEqual(store.get_context(), [])
        store.close()


if __name__ == "__main__":
    unittest.main()
