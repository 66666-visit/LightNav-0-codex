"""Write Gate decisions must remain attached to their own source event."""

from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # The repository's CPU test environment supplies NumPy.
    np = None


@unittest.skipUnless(np is not None, "NumPy is required for the real RGB writer")
class EventMetadataIsolationTests(unittest.TestCase):
    def test_accepted_observation_survives_rejected_derived_event(self) -> None:
        from lightnav.memory.manager import MemoryManager
        from lightnav.memory.memory_writer import MockSceneExtractor
        from lightnav.memory.rgm_config import RGMConfig
        from lightnav.memory.schemas import ObservationPacket, SceneExtraction

        manager = MemoryManager(
            extractor=MockSceneExtractor({
                0: SceneExtraction(objects=["couch"], confidence=0.88),
            }),
            strategy="hierarchical",
            rgm_config=RGMConfig(enabled=True),
        )
        manager.reset_episode("episode_probe", "Find the sofa")
        try:
            events = manager.observe(ObservationPacket(
                step_id=0, rgb=np.zeros((8, 8, 3), dtype=np.uint8),
                instruction="Find the sofa",
            ))
            observation = next(event for event in events if event.kind == "object_observation")
            derived = next(event for event in events if event.kind == "instruction_match")
            self.assertNotEqual(observation.event_id, derived.event_id)
            self.assertIsNot(observation.metadata, derived.metadata)
            self.assertTrue(observation.metadata["rgm_adopted"])
            self.assertEqual(observation.metadata["rgm_write_reason"],
                             "instruction_grounded_observation")
            self.assertFalse(derived.metadata["rgm_adopted"])
            self.assertEqual(derived.metadata["rgm_write_reason"], "not_navigation_landmark")
            self.assertNotIn("rgm_temporal_support", derived.metadata)

            decisions = {decision.evidence_ids[-1]: decision
                         for decision in manager.write_gate.decisions}
            recorded = {event.event_id: event for event in manager.episodic.events}
            for event in (observation, derived):
                self.assertEqual(decisions[event.event_id].accepted,
                                 recorded[event.event_id].metadata["rgm_adopted"])
            self.assertEqual(set(manager.working.state.recent_event_ids), set(recorded))
            self.assertEqual(
                [ref.event_id for ref in manager.working.recent_observations[-1].detections],
                [observation.event_id],
            )
            self.assertEqual(manager.working.state.instruction_matches, ["couch"])
            self.assertEqual(
                [ref.event_id for ref in manager.semantic_index.get("couch").observations],
                [observation.event_id],
            )
        finally:
            manager.close_episode()


if __name__ == "__main__":
    unittest.main()
