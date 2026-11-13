"""V3 storage boundaries and physical traversal evidence, without model packages."""

from __future__ import annotations

import unittest

from lightnav.memory.v3_schemas import EvidenceRecord
from lightnav.memory.v3_store import V3MemoryStore


def record(scene: str, episode: str, event: str, step: int = 0, **kwargs) -> EvidenceRecord:
    return EvidenceRecord(
        scene_id=scene, episode_id=episode, evidence_id=event,
        rollout_idx=0, step=step,
        event_type=kwargs.pop("event_type", "object_observation"),
        **kwargs,
    )


class V3StoreTests(unittest.TestCase):
    def test_active_mid_segment_uses_only_prior_steps_without_closing_episode(self) -> None:
        store = V3MemoryStore(scope="episode_local")
        store.begin_episode("scene", "episode", "turn toward the red door")
        seen = store.add(record(
            "scene", "episode", "event_0", step=0, semantic_label="red door",
            observation_status="observed",
        ))
        action = store.add(record(
            "scene", "episode", "event_1", step=1, event_type="action",
            payload={"description": "turned left before the next view"},
        ))
        current = store.add(record(
            "scene", "episode", "event_2", step=2, semantic_label="red door",
            observation_status="observed",
        ))
        segment = store.active_segment(before_step=2)
        self.assertIsNotNone(segment)
        self.assertEqual(segment.scene_id, "scene")
        self.assertEqual(segment.episode_id, "episode")
        self.assertEqual(segment.outcome, "unknown")
        self.assertEqual(segment.actions, [action.key])
        self.assertEqual(segment.evidence_ids, [seen.key, action.key])
        self.assertNotIn(current.key, segment.evidence_ids)
        self.assertEqual(store.active_episode, ("scene", "episode", 0))
        self.assertEqual(store.episodes_for_scene("scene"), ())
        store.close_episode()
        store.begin_episode("scene", "next", "turn toward the red door")
        self.assertIsNone(store.active_segment(before_step=1))
        self.assertEqual(store.evidence_for_scene("scene"), ())
        store.close_episode()

    def test_low_confidence_is_stored_and_local_scope_clears_all_banks(self) -> None:
        store = V3MemoryStore(scope="episode_local", short_capacity=1)
        store.begin_episode("scene", "first", "find a door")
        first = store.add(record("scene", "first", "event_0", semantic_label="door",
                                 raw_detector_confidence=0.12,
                                 detector_name="yolo", observation_status="observed"))
        store.add(record("scene", "first", "event_1", step=1,
                         event_type="action", payload={"description": "turn"}))
        self.assertEqual(first.raw_detector_confidence, 0.12)
        self.assertEqual(len(store.current_short), 1)
        episode = store.close_episode(outcome="failure")
        self.assertEqual(episode.outcome, "failure")
        self.assertEqual(len(episode.actions), 1)
        self.assertEqual(store.evidence_for_scene("scene"), ())
        self.assertEqual(store.episodes_for_scene("scene"), ())

    def test_persistent_scope_uses_composite_keys_and_keeps_failure_history(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene", "first", "find door")
        old = store.add(record("scene", "first", "event_0", semantic_label="door"))
        store.close_episode(outcome="failure")
        store.begin_episode("scene", "second", "find door")
        new = store.add(record("scene", "second", "event_0", semantic_label="door"))
        self.assertNotEqual(old.key, new.key)
        self.assertEqual(new.support_count, 2)
        self.assertEqual(new.temporal_support, 1)
        self.assertEqual(new.episode_support, 2)
        self.assertEqual(store.episodes_for_scene("scene")[0].outcome, "failure")
        self.assertIsNone(store.get_evidence(record("other", "first", "event_0").key))
        store.close_episode()

    def test_instruction_match_is_not_independent_visual_support(self) -> None:
        store = V3MemoryStore()
        store.begin_episode("scene", "episode", "find door")
        direct = store.add(record("scene", "episode", "event_0", semantic_label="door",
                                  observation_status="observed"))
        derived = store.add(record(
            "scene", "episode", "event_1", event_type="instruction_match",
            semantic_label="door", source_evidence_ids=(direct.key,),
        ))
        repeated = store.add(record("scene", "episode", "event_2", step=1,
                                    semantic_label="door", observation_status="observed"))
        self.assertEqual(derived.support_count, 1)
        self.assertEqual(repeated.support_count, 2)
        self.assertEqual(repeated.temporal_support, 2)
        episode = store.close_episode()
        self.assertEqual(episode.landmarks, [direct.key, repeated.key])

    def test_visual_shift_and_command_do_not_verify_traversal(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene", "episode", "go to door")
        a = store.add(record("scene", "episode", "place_0", event_type="place_observation",
                             visual_descriptor=(1.0, 0.0), observation_status="observed"))
        b = store.add(record("scene", "episode", "place_1", step=1,
                             event_type="place_observation", visual_descriptor=(0.0, 1.0),
                             observation_status="observed"))
        command = store.add(record("scene", "episode", "event_0", step=1,
                                   event_type="action", payload={"description": "forward"}))
        a_node = store.add_place(a)
        b_node = store.add_place(b)
        uncertain = store.add_relation(a_node.node_id, b_node.node_id, "observed_transition",
                                       evidence_keys=(a.key, b.key, command.key))
        self.assertEqual(uncertain.traversability_status, "unknown")
        self.assertEqual(uncertain.traversal_count, 0)
        with self.assertRaises(ValueError):
            store.add_relation(a_node.node_id, b_node.node_id, "verified_traversal",
                               evidence_keys=(a.key, b.key, command.key),
                               motion_evidence_key=command.key)
        store.close_episode(outcome="success")
        self.assertEqual(store.edges_for_scene("scene")[0].traversability_status, "unknown")
        self.assertEqual(store.episodes_for_scene("scene")[0].outcome, "success")

    def test_independent_measured_motion_verifies_physical_traversal(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene", "episode", "find door")
        a = store.add(record("scene", "episode", "a", event_type="place_observation",
                             visual_descriptor=(1.0, 0.0), observation_status="observed"))
        b = store.add(record("scene", "episode", "b", step=1,
                             event_type="place_observation", visual_descriptor=(0.0, 1.0),
                             observation_status="observed"))
        motion = store.add(record(
            "scene", "episode", "motion", step=1, event_type="observed_motion",
            observation_status="observed",
            payload={"measurement_source": "odometry", "displacement_m": 0.5},
        ))
        node_a, node_b = store.add_place(a), store.add_place(b)
        edge = store.add_relation(
            node_a.node_id, node_b.node_id, "verified_traversal",
            evidence_keys=(a.key, b.key, motion.key), motion_evidence_key=motion.key,
        )
        self.assertEqual((edge.traversal_count, edge.successful_traversals), (1, 1))
        c = store.add(record("scene", "episode", "c", step=2,
                             event_type="place_observation", visual_descriptor=(-1.0, 0.0),
                             observation_status="observed"))
        node_c = store.add_place(c)
        with self.assertRaisesRegex(ValueError, "multiple spatial edges"):
            store.add_relation(
                node_a.node_id, node_c.node_id, "verified_traversal",
                evidence_keys=(a.key, c.key, motion.key), motion_evidence_key=motion.key,
            )
        late_motion = store.add(record(
            "scene", "episode", "late_motion", step=3, event_type="observed_motion",
            observation_status="observed",
            payload={"measurement_source": "odometry", "displacement_m": 0.5},
        ))
        with self.assertRaisesRegex(ValueError, "between endpoint observations"):
            store.add_relation(
                node_a.node_id, node_b.node_id, "verified_traversal",
                evidence_keys=(a.key, b.key, late_motion.key),
                motion_evidence_key=late_motion.key,
            )
        store.close_episode(outcome="failure")
        persisted = store.edges_for_scene("scene")[0]
        self.assertEqual((persisted.successful_traversals, persisted.failed_traversals), (1, 0))

    def test_same_scene_visual_place_updates_across_episodes(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene_a", "first", "find archway")
        first = store.add(record("scene_a", "first", "place_0",
                                 event_type="place_observation",
                                 visual_descriptor=(1.0, 0.0),
                                 observation_status="observed"))
        node = store.add_place(first)
        store.close_episode()
        store.begin_episode("scene_a", "second", "find archway")
        second = store.add(record("scene_a", "second", "place_0",
                                  event_type="place_observation",
                                  visual_descriptor=(0.99, 0.01),
                                  observation_status="observed"))
        matched = store.add_place(second)
        self.assertEqual(matched.node_id, node.node_id)
        self.assertEqual(matched.observation_count, 2)
        self.assertEqual(matched.support_episode_count, 2)
        store.close_episode()
        store.begin_episode("scene_b", "third", "find archway")
        separate = store.add(record("scene_b", "third", "place_0",
                                    event_type="place_observation",
                                    visual_descriptor=(1.0, 0.0),
                                    observation_status="observed"))
        store.add_place(separate)
        self.assertEqual(len(store.nodes_for_scene("scene_a")), 1)
        self.assertEqual(len(store.nodes_for_scene("scene_b")), 1)
        store.close_episode()


if __name__ == "__main__":
    unittest.main()
