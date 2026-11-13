"""V4 navigation state uses observed evidence, never commands or evaluator GT."""

from __future__ import annotations

import unittest

from lightnav.memory.v3_schemas import EvidenceRecord
from lightnav.memory.v3_store import V3MemoryStore
from lightnav.memory.v4_navigation_state import (
    COMPLETION_UNKNOWN,
    NOT_OBSERVED,
    OBSERVED,
    POSSIBLE_REVISIT,
    PROGRESS_SUPPORTED,
    NavigationStateUpdater,
)


def evidence(scene: str, episode: str, event_id: str, step: int,
             *, event_type: str = "object_observation", **kwargs) -> EvidenceRecord:
    return EvidenceRecord(
        scene_id=scene, episode_id=episode, rollout_idx=0,
        evidence_id=event_id, step=step, event_type=event_type, **kwargs,
    )


class V4NavigationStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = V3MemoryStore(scope="scene_persistent")
        self.store.begin_episode("scene_a", "episode_1", "Go through the red door")
        self.updater = NavigationStateUpdater(self.store)
        self.updater.reset_episode("scene_a", "episode_1", "Go through the red door")

    def test_current_observation_updates_anchor_but_not_completion(self) -> None:
        observed = self.store.add(evidence(
            "scene_a", "episode_1", "door_0", 0,
            semantic_label="red door", observation_status="observed",
        ))
        state = self.updater.update(0)
        self.assertEqual(state.instruction, "Go through the red door")
        self.assertEqual(state.progress_status, OBSERVED)
        self.assertEqual(state.active_anchor, "red door")
        self.assertEqual(state.anchor_evidence["door"], (observed.key,))
        self.assertEqual(state.anchor_status["door"], OBSERVED)
        self.assertEqual(state.recent_observation_ids, (observed.key,))
        self.assertEqual(state.historical_landmark_ids, ())
        self.assertEqual(state.progress_evidence_ids, ())
        self.assertIn("unverified", state.uncertainty)
        self.assertEqual(
            (state.evidence_provenance[0].episode_id, state.evidence_provenance[0].step),
            ("episode_1", 0),
        )

    def test_same_episode_history_preserves_step_and_excludes_current(self) -> None:
        old = self.store.add(evidence(
            "scene_a", "episode_1", "door_0", 0,
            semantic_label="red door", observation_status="observed",
        ))
        self.updater.update(0)
        now = self.store.add(evidence(
            "scene_a", "episode_1", "door_2", 2,
            semantic_label="red door", observation_status="observed",
        ))
        state = self.updater.update(2)
        self.assertEqual(state.historical_landmark_ids, (old.key,))
        self.assertNotIn(now.key, state.historical_landmark_ids)
        self.assertEqual(state.anchor_evidence["door"], (old.key, now.key))
        provenance = {item.evidence_id: item for item in state.evidence_provenance}
        self.assertEqual((provenance[old.key].episode_id, provenance[old.key].step),
                         ("episode_1", 0))
        self.assertEqual((provenance[now.key].episode_id, provenance[now.key].step),
                         ("episode_1", 2))

    def test_cross_episode_history_does_not_claim_current_progress(self) -> None:
        old = self.store.add(evidence(
            "scene_a", "episode_1", "door_0", 0,
            semantic_label="red door", observation_status="observed",
        ))
        self.store.close_episode()
        self.store.begin_episode("scene_a", "episode_2", "Go through the red door")
        self.updater.reset_episode("scene_a", "episode_2", "Go through the red door")
        state = self.updater.update(0)
        self.assertEqual(state.progress_status, COMPLETION_UNKNOWN)
        self.assertEqual(state.historical_landmark_ids, (old.key,))
        self.assertEqual(state.anchor_evidence["door"], ())
        self.assertEqual(state.anchor_status["door"], NOT_OBSERVED)
        self.assertEqual((state.evidence_provenance[0].episode_id,
                          state.evidence_provenance[0].step), ("episode_1", 0))
        now = self.store.add(evidence(
            "scene_a", "episode_2", "door_0", 0,
            semantic_label="red door", observation_status="observed",
        ))
        state = self.updater.update(0)
        self.assertEqual(state.anchor_evidence["door"], (now.key,))
        self.assertEqual(state.historical_landmark_ids, (old.key,))

    def test_issued_forward_command_cannot_verify_passage(self) -> None:
        seen = self.store.add(evidence(
            "scene_a", "episode_1", "door_0", 0,
            semantic_label="red door", observation_status="observed",
        ))
        self.store.add(evidence(
            "scene_a", "episode_1", "forward_1", 1, event_type="action",
            payload={"description": "issued forward velocity control"},
            traversability_status="verified",
        ))
        state = self.updater.update(1)
        self.assertEqual(state.progress_status, OBSERVED)
        self.assertEqual(state.historical_landmark_ids, (seen.key,))
        self.assertEqual(state.progress_evidence_ids, ())

    def test_dino_matches_are_only_possible_revisits(self) -> None:
        for event_id, step, vector in (
            ("place_0", 0, (1.0, 0.0)),
            ("place_2", 2, (0.99, 0.01)),
        ):
            self.store.add(evidence(
                "scene_a", "episode_1", event_id, step,
                event_type="place_observation", visual_descriptor=vector,
                observation_status="observed",
            ))
        current = self.store.add(evidence(
            "scene_a", "episode_1", "place_4", 4,
            event_type="place_observation", visual_descriptor=(0.98, 0.02),
            observation_status="observed",
        ))
        self.store.add(evidence(
            "scene_a", "episode_1", "door_4", 4,
            semantic_label="red door", observation_status="observed",
        ))
        state = self.updater.update(4)
        self.assertEqual(state.progress_status, POSSIBLE_REVISIT)
        self.assertEqual(state.repeated_observation_count, 2)
        self.assertEqual(len(state.revisit_candidates), 1)
        candidate = state.revisit_candidates[0]
        self.assertEqual(candidate.current_evidence_id, current.key)
        self.assertEqual(candidate.independent_history_observations, 2)
        self.assertTrue(candidate.instruction_related)
        self.assertEqual(candidate.identity_status, "unknown")
        self.assertEqual(state.progress_evidence_ids, ())
        self.assertIn("does_not_verify", state.uncertainty)

    def test_single_visual_match_still_has_unknown_identity(self) -> None:
        old = self.store.add(evidence(
            "scene_a", "episode_1", "place_0", 0,
            event_type="place_observation", visual_descriptor=(1.0, 0.0),
            observation_status="observed",
        ))
        self.store.add(evidence(
            "scene_a", "episode_1", "place_1", 1,
            event_type="place_observation", visual_descriptor=(0.99, 0.01),
            observation_status="observed",
        ))
        state = self.updater.update(1)
        self.assertEqual(state.progress_status, POSSIBLE_REVISIT)
        self.assertEqual(state.revisit_candidates[0].historical_evidence_ids, (old.key,))
        self.assertEqual(state.revisit_candidates[0].independent_history_observations, 1)
        self.assertFalse(state.revisit_candidates[0].instruction_related)
        self.assertEqual(state.revisit_candidates[0].identity_status, "unknown")

    def test_insufficient_evidence_keeps_completion_unknown(self) -> None:
        state = self.updater.update(0)
        self.assertEqual(state.progress_status, COMPLETION_UNKNOWN)
        self.assertEqual(state.anchor_status["door"], NOT_OBSERVED)
        self.assertEqual(state.evidence_ids, ())
        self.assertIsNone(state.active_anchor)

    def test_inferred_visual_record_is_not_a_current_observation(self) -> None:
        self.store.add(evidence(
            "scene_a", "episode_1", "place_0", 0,
            event_type="place_observation", visual_descriptor=(1.0, 0.0),
            observation_status="observed",
        ))
        self.store.add(evidence(
            "scene_a", "episode_1", "inferred_place_1", 1,
            event_type="place_observation", visual_descriptor=(1.0, 0.0),
            observation_status="inferred",
        ))
        state = self.updater.update(1)
        self.assertEqual(state.revisit_candidates, ())
        self.assertEqual(state.progress_status, COMPLETION_UNKNOWN)

    def test_evaluator_fields_and_untrusted_events_are_ignored(self) -> None:
        self.store.add(evidence(
            "scene_a", "episode_1", "oracle_0", 0,
            event_type="oracle_goal_distance", semantic_label="red door",
            observation_status="observed",
            payload={"success": True, "goal_distance": 0.0, "spl": 1.0},
        ))
        self.store.add(evidence(
            "scene_a", "episode_1", "place_visit_0", 0,
            event_type="place_visit", semantic_label="red door",
            observation_status="observed",
        ))
        state = self.updater.update(0)
        self.assertEqual(state.progress_status, COMPLETION_UNKNOWN)
        self.assertEqual(state.evidence_ids, ())

    def test_other_scene_is_never_a_revisit(self) -> None:
        self.store.add(evidence(
            "scene_a", "episode_1", "place_0", 0,
            event_type="place_observation", visual_descriptor=(1.0, 0.0),
            observation_status="observed",
        ))
        self.store.close_episode()
        self.store.begin_episode("scene_b", "episode_2", "Go through the red door")
        self.updater.reset_episode("scene_b", "episode_2", "Go through the red door")
        self.store.add(evidence(
            "scene_b", "episode_2", "place_0", 0,
            event_type="place_observation", visual_descriptor=(1.0, 0.0),
            observation_status="observed",
        ))
        state = self.updater.update(0)
        self.assertEqual(state.revisit_candidates, ())
        self.assertEqual(state.progress_status, COMPLETION_UNKNOWN)

    def test_independently_measured_traversal_supports_progress_not_completion(self) -> None:
        door = self.store.add(evidence(
            "scene_a", "episode_1", "door_0", 0,
            semantic_label="red door", observation_status="observed",
        ))
        place = self.store.add(evidence(
            "scene_a", "episode_1", "place_1", 1,
            event_type="place_observation", visual_descriptor=(1.0, 0.0),
            observation_status="observed",
        ))
        motion = self.store.add(evidence(
            "scene_a", "episode_1", "motion_1", 1,
            event_type="motion_observation", observation_status="observed",
            payload={"measurement_source": "odometry", "displacement_m": 0.5},
        ))
        source = self.store.add_landmark(door)
        target = self.store.add_place(place)
        self.store.add_relation(
            source.node_id, target.node_id, "verified_traversal",
            evidence_keys=(door.key, place.key, motion.key),
            motion_evidence_key=motion.key,
        )
        state = self.updater.update(1)
        self.assertEqual(state.progress_status, PROGRESS_SUPPORTED)
        self.assertIn(motion.key, state.progress_evidence_ids)
        self.assertEqual(state.uncertainty, "goal_completion_unverified")


if __name__ == "__main__":
    unittest.main()
