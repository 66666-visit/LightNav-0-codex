"""Four V3 retrieval channels and scene isolation using synthetic evidence."""

from __future__ import annotations

import unittest

from lightnav.memory.v3_retrieval import V3Retriever, reciprocal_rank_fusion
from lightnav.memory.v3_schemas import EvidenceRecord, RetrievedMemory
from lightnav.memory.v3_store import V3MemoryStore


def evidence(scene: str, episode: str, event: str, step: int = 0, **kwargs) -> EvidenceRecord:
    return EvidenceRecord(
        scene_id=scene, episode_id=episode, rollout_idx=0, evidence_id=event,
        step=step, event_type=kwargs.pop("event_type", "object_observation"), **kwargs,
    )


class V3RetrievalTests(unittest.TestCase):
    def test_same_episode_prior_steps_enter_mid_and_historical_counts(self) -> None:
        store = V3MemoryStore(scope="episode_local")
        store.begin_episode("scene", "current", "turn toward the door")
        old = store.add(evidence(
            "scene", "current", "event_0", step=0, semantic_label="door",
            observation_status="observed",
        ))
        store.add(evidence(
            "scene", "current", "event_1", step=1, event_type="action",
            payload={"description": "turn left before the next view"},
        ))
        current = store.add(evidence(
            "scene", "current", "event_2", step=2, semantic_label="door",
            observation_status="observed",
        ))
        result = V3Retriever(store).retrieve(
            scene_id="scene", instruction="turn toward the door", current_step=2,
            episode_id="current", top_k=5,
        )
        self.assertGreaterEqual(result.counts["episodic_historical_candidates"], 1)
        self.assertGreaterEqual(result.counts["historical_same_episode_prior_step"], 1)
        self.assertGreaterEqual(result.counts["source_step_lt_current_step"], 1)
        self.assertEqual(result.counts["historical_cross_episode"], 0)
        self.assertGreaterEqual(result.counts["historical_final_top_k"], 1)
        segments = [memory for memory in result.records
                    if memory.memory_id.startswith("segment:")]
        self.assertEqual(len(segments), 1)
        self.assertIn("Earlier segment", segments[0].description)
        self.assertIn("ongoing", segments[0].description)
        self.assertEqual(segments[0].outcome, "unknown")
        self.assertTrue(any(old.key in memory.evidence_ids for memory in result.records))
        self.assertTrue(any(current.key in memory.evidence_ids for memory in result.records))
        store.close_episode()

    def test_cross_episode_history_is_counted_even_when_step_resets(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene_a", "old", "find the door")
        old = store.add(evidence(
            "scene_a", "old", "event_0", step=200, semantic_label="door",
            observation_status="observed",
        ))
        store.close_episode()
        store.begin_episode("scene_a", "new", "find the door")
        result = V3Retriever(store).retrieve(
            scene_id="scene_a", instruction="find the door", current_step=0,
            episode_id="new", top_k=5,
        )
        self.assertGreaterEqual(result.counts["semantic_historical_candidates"], 1)
        self.assertGreaterEqual(result.counts["episodic_historical_candidates"], 1)
        self.assertGreaterEqual(result.counts["historical_cross_episode"], 1)
        self.assertEqual(result.counts["historical_same_episode_prior_step"], 0)
        self.assertEqual(result.counts["source_step_lt_current_step"], 0)
        self.assertGreaterEqual(result.counts["historical_final_top_k"], 1)
        self.assertTrue(any(old.key in memory.evidence_ids for memory in result.records))
        self.assertTrue(all(memory.source_step is not None and memory.source_step > 0
                            for memory in result.records))
        store.close_episode()
        store.begin_episode("scene_b", "other", "find the door")
        isolated = V3Retriever(store).retrieve(
            scene_id="scene_b", instruction="find the door", current_step=0,
            episode_id="other", top_k=5,
        )
        self.assertEqual(isolated.records, ())
        self.assertEqual(isolated.counts["historical_cross_episode"], 0)
        self.assertEqual(isolated.counts["historical_final_top_k"], 0)
        store.close_episode()

    def test_all_four_channels_report_real_historical_candidates(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene", "past", "find the archway")
        landmark = store.add(evidence(
            "scene", "past", "event_0", semantic_label="archway",
            observation_status="observed",
        ))
        a = store.add(evidence(
            "scene", "past", "place_0", event_type="place_observation",
            visual_descriptor=(1.0, 0.0), observation_status="observed",
        ))
        b = store.add(evidence(
            "scene", "past", "place_1", step=1, event_type="place_observation",
            visual_descriptor=(0.0, 1.0), observation_status="observed",
        ))
        node_a, node_b = store.add_place(a), store.add_place(b)
        store.add_relation(
            node_a.node_id, node_b.node_id, "inferred_relation",
            evidence_keys=(a.key, b.key),
        )
        store.close_episode()
        store.begin_episode("scene", "current", "find the archway")
        result = V3Retriever(store).retrieve(
            scene_id="scene", instruction="find the archway", current_step=0,
            visual_descriptor=(1.0, 0.0), place_id=node_a.node_id,
            episode_id="current", top_k=5,
        )
        for channel in ("visual", "semantic", "spatial", "episodic"):
            self.assertGreaterEqual(result.counts[f"{channel}_historical_candidates"], 1)
        self.assertEqual(result.counts["historical_final_top_k"], result.final_top_k)
        self.assertTrue(any(landmark.key in memory.evidence_ids for memory in result.records))
        self.assertTrue(all(memory.traversability_status != "traversed"
                            for memory in result.records))
        store.close_episode()

    def test_semantic_visual_spatial_episodic_and_scene_isolation(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene_a", "past", "find the archway")
        landmark = store.add(evidence("scene_a", "past", "event_0", semantic_label="archway",
                                      payload={"description": "Observed archway"},
                                      observation_status="observed"))
        place_a = store.add(evidence("scene_a", "past", "place_0", event_type="place_observation",
                                     visual_descriptor=(1.0, 0.0), observation_status="observed"))
        place_b = store.add(evidence("scene_a", "past", "place_1", step=1,
                                     event_type="place_observation", visual_descriptor=(0.0, 1.0),
                                     observation_status="observed"))
        action = store.add(evidence("scene_a", "past", "event_1", step=1,
                                    event_type="action", payload={"description": "turn left"}))
        node_a, node_b = store.add_place(place_a), store.add_place(place_b)
        store.add_relation(node_a.node_id, node_b.node_id, "inferred_relation",
                           evidence_keys=(place_a.key, place_b.key, action.key))
        store.close_episode(outcome="failure")

        store.begin_episode("scene_b", "other", "find archway")
        other = store.add(evidence("scene_b", "other", "event_0", semantic_label="archway",
                                   observation_status="observed"))
        store.close_episode()
        store.begin_episode("scene_a", "current", "find archway")
        result = V3Retriever(store).retrieve(
            scene_id="scene_a", instruction="find the archway", current_step=0,
            visual_descriptor=(1.0, 0.0), place_id=node_a.node_id,
            episode_id="current", top_k=5,
        )
        self.assertGreater(result.channel_candidates["semantic"], 0)
        self.assertGreater(result.channel_candidates["visual"], 0)
        self.assertGreater(result.channel_candidates["spatial"], 0)
        self.assertGreater(result.channel_candidates["episodic"], 0)
        self.assertTrue(any(landmark.key in memory.evidence_ids for memory in result.records))
        self.assertTrue(all(memory.scene_id == "scene_a" for memory in result.records))
        self.assertTrue(all(other.key not in memory.evidence_ids for memory in result.records))
        self.assertTrue(any(memory.outcome == "failure" for memory in result.records))
        experience = next(memory for memory in result.records if memory.outcome == "failure")
        self.assertIn("actions=1", experience.description)
        self.assertIn("turn left", experience.description)
        self.assertTrue(all(memory.traversability_status != "traversed"
                            for memory in result.records))
        spatial_only = V3Retriever(store).retrieve(
            scene_id="scene_a", instruction="unrelated", current_step=0,
            place_id=node_a.node_id, episode_id="current", top_k=5,
        )
        self.assertTrue(any(memory.kind == "inferred_relation"
                            for memory in spatial_only.records))
        relation = next(memory for memory in spatial_only.records
                        if memory.kind == "inferred_relation")
        self.assertIn("traversability=unknown", relation.description)
        store.close_episode()

    def test_current_observation_wins_rrf_tie_with_past_episode(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene", "past", "find the archway")
        store.add(evidence("scene", "past", "event_0", semantic_label="archway",
                           observation_status="observed"))
        store.close_episode(outcome="failure")
        store.begin_episode("scene", "now", "find the archway")
        current = store.add(evidence("scene", "now", "event_0", semantic_label="archway",
                                     observation_status="observed"))
        derived = store.add(evidence(
            "scene", "now", "event_1", event_type="instruction_match",
            semantic_label="archway", source_evidence_ids=(current.key,),
            observation_status="unknown",
        ))
        result = V3Retriever(store).retrieve(
            scene_id="scene", instruction="find the archway", current_step=0,
            episode_id="now", top_k=1,
        )
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].evidence_ids, (current.key,))
        self.assertTrue(result.records[0].current_observation)
        self.assertNotIn(derived.key, result.records[0].evidence_ids)
        store.close_episode()

    def test_spatial_prompt_keeps_landmark_label_and_unknown_traversability(self) -> None:
        store = V3MemoryStore()
        store.begin_episode("scene", "ep", "find archway")
        landmark = store.add(evidence("scene", "ep", "landmark", semantic_label="archway",
                                      observation_status="observed"))
        place = store.add(evidence("scene", "ep", "place", event_type="place_observation",
                                   visual_descriptor=(1.0, 0.0), observation_status="observed"))
        landmark_node = store.add_landmark(landmark)
        place_node = store.add_place(place)
        store.add_relation(place_node.node_id, landmark_node.node_id, "inferred_relation",
                           evidence_keys=(place.key, landmark.key))
        result = V3Retriever(store).retrieve(
            scene_id="scene", instruction="unrelated", current_step=0,
            place_id=place_node.node_id, episode_id="ep", top_k=5,
        )
        relation = next(memory for memory in result.records if memory.kind == "inferred_relation")
        self.assertIn("archway", relation.description)
        self.assertIn("traversability=unknown", relation.description)
        store.close_episode()

    def test_low_instruction_overlap_does_not_remove_visual_match(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene", "old", "unrelated")
        place = store.add(evidence("scene", "old", "place", event_type="place_observation",
                                   visual_descriptor=(1.0, 0.0), observation_status="observed"))
        node = store.add_place(place)
        store.close_episode()
        store.begin_episode("scene", "new", "search for kitchen")
        result = V3Retriever(store).retrieve(
            scene_id="scene", instruction="search for kitchen", current_step=0,
            visual_descriptor=(1.0, 0.0), place_id=node.node_id,
            episode_id="new", top_k=5,
        )
        self.assertTrue(any(place.key in item.evidence_ids for item in result.records))
        store.close_episode()

    def test_generic_navigation_verbs_do_not_force_unrelated_history(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene", "past", "Find the sofa")
        store.add(evidence("scene", "past", "event_0", semantic_label="sofa",
                           observation_status="observed"))
        store.close_episode()
        store.begin_episode("scene", "now", "Find the kitchen")
        result = V3Retriever(store).retrieve(
            scene_id="scene", instruction="Find the kitchen", current_step=0,
            episode_id="now", top_k=5,
        )
        self.assertEqual(result.records, ())
        store.close_episode()

    def test_scene_persistent_recency_uses_write_order_across_step_resets(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene", "older", "unrelated")
        older = store.add(evidence("scene", "older", "event_0", step=200,
                                   semantic_label="door", observation_status="observed"))
        store.close_episode()
        store.begin_episode("scene", "newer", "unrelated")
        newer = store.add(evidence("scene", "newer", "event_0", step=0,
                                   semantic_label="door", observation_status="observed"))
        store.close_episode()
        store.begin_episode("scene", "current", "find door")
        result = V3Retriever(store).retrieve(
            scene_id="scene", instruction="find door", current_step=0,
            episode_id="current", top_k=1,
        )
        self.assertEqual(result.records[0].evidence_ids, (newer.key,))
        self.assertNotEqual(result.records[0].evidence_ids, (older.key,))
        store.close_episode()

    def test_rrf_rewards_multichannel_presence(self) -> None:
        common = RetrievedMemory(memory_id="common", kind="observation", scene_id="s",
                                 description="common", evidence_ids=("[\"s\",\"e\",0,\"a\"]",))
        only = RetrievedMemory(memory_id="only", kind="observation", scene_id="s",
                               description="only", evidence_ids=("[\"s\",\"e\",0,\"b\"]",))
        selected = reciprocal_rank_fusion(
            {"visual": [only, common], "semantic": [common],
             "spatial": [], "episodic": []}, top_k=2,
        )
        self.assertEqual(selected[0].memory_id, "common")
        self.assertEqual(set(selected[0].channel_ranks), {"visual", "semantic"})

    def test_invalid_provenance_cannot_reach_retrieval(self) -> None:
        store = V3MemoryStore()
        store.begin_episode("scene", "episode", "find door")
        store.add(evidence("scene", "episode", "event_0", semantic_label="door",
                           provenance_validity=False, observation_status="observed"))
        result = V3Retriever(store).retrieve(
            scene_id="scene", instruction="find door", current_step=0,
            episode_id="episode", top_k=5,
        )
        self.assertEqual(result.records, ())
        store.close_episode()


if __name__ == "__main__":
    unittest.main()
