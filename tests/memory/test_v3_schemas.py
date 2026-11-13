"""V3 evidence records remain scene-scoped and JSON-roundtrippable."""

from __future__ import annotations

import json
import math
import unittest

from lightnav.memory.v3_schemas import (
    EpisodeMemory,
    EvidenceRecord,
    RetrievedMemory,
    SpatialEdgeV3,
    SpatialNodeV3,
    canonical_evidence_key,
    parse_evidence_key,
)


def _roundtrip(record):
    return type(record).from_dict(json.loads(json.dumps(record.to_dict(), allow_nan=False)))


class V3SchemaTests(unittest.TestCase):
    def test_composite_evidence_key_prevents_cross_episode_collision(self) -> None:
        first = canonical_evidence_key("scene_a", "episode_1", 0, "event_0")
        second = canonical_evidence_key("scene_a", "episode_2", 0, "event_0")
        third = canonical_evidence_key("scene_b", "episode_1", 0, "event_0")
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual(parse_evidence_key(first), ("scene_a", "episode_1", 0, "event_0"))
        with self.assertRaises(ValueError):
            parse_evidence_key('["scene_a", "episode_1", 0, "event_0"]')

    def test_evidence_roundtrip_preserves_raw_reliability_dimensions(self) -> None:
        source = canonical_evidence_key("scene_a", "episode_1", 0, "event_0")
        record = EvidenceRecord(
            evidence_id="event_1", scene_id="scene_a", episode_id="episode_1",
            rollout_idx=0, step=5, event_type="object_observation",
            semantic_label="door", payload={"bbox_xyxy": [1, 2, 3, 4]},
            visual_descriptor=(0.6, 0.8), detector_name="yolo11",
            raw_detector_confidence=0.24, instruction_terms=("door",),
            source_evidence_ids=(source,), observation_status="observed",
            spatial_status="unknown", traversability_status="unknown",
            support_count=2, temporal_support=1, episode_support=1,
            spatial_consistency=0.3,
        )
        self.assertEqual(record.key, canonical_evidence_key(
            "scene_a", "episode_1", 0, "event_1",
        ))
        self.assertEqual((record.first_seen, record.last_seen), (5, 5))
        self.assertEqual(_roundtrip(record), record)
        self.assertEqual(record.raw_detector_confidence, 0.24)
        self.assertEqual(record.temporal_support, 1)
        self.assertTrue(record.provenance_validity)

    def test_episode_and_spatial_graph_roundtrip(self) -> None:
        ref = canonical_evidence_key("scene_a", "episode_1", 0, "event_0")
        episode = EpisodeMemory(
            scene_id="scene_a", episode_id="episode_1", instruction="Find the door",
            observations=[ref], actions=[ref], outcome="failure",
            start_step=0, end_step=3, evidence_ids=[ref],
        )
        node = SpatialNodeV3(
            node_id="place_1", scene_id="scene_a", node_type="place",
            visual_descriptor=(1.0, 0.0), observation_count=2,
            support_episode_count=1, evidence_ids=[ref],
            first_seen_step=0, last_seen_step=3, spatial_status="observed",
        )
        edge = SpatialEdgeV3(
            edge_id="edge_1", scene_id="scene_a", source_node="place_1",
            target_node="place_2", relation_type="verified_traversal",
            traversal_count=2, successful_traversals=1, failed_traversals=1,
            evidence_ids=[ref], confidence_statistics={"raw_mean": 0.6},
            spatial_status="traversed", traversability_status="traversed",
            first_seen_step=1, last_seen_step=3,
        )
        for record in (episode, node, edge):
            self.assertEqual(_roundtrip(record), record)
        self.assertEqual(episode.outcome, "failure")

    def test_retrieved_memory_keeps_provenance_and_channel_ranks(self) -> None:
        ref = canonical_evidence_key("scene_a", "episode_1", 0, "event_0")
        item = RetrievedMemory(
            memory_id="episode:1", kind="episode", scene_id="scene_a",
            description="A left turn failed", evidence_ids=(ref,),
            source_step=3,
            observation_status="observed", outcome="failure",
            score=0.02, channel_ranks={"semantic": 1, "episodic": 2},
        )
        self.assertEqual(_roundtrip(item), item)
        self.assertEqual(item.evidence_ids, (ref,))
        self.assertEqual(item.source_step, 3)

    def test_invalid_identity_payload_and_spatial_claims_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "scene_id"):
            EvidenceRecord("event_0", "", "episode_1", 0, 0, "object_observation")
        with self.assertRaisesRegex(ValueError, "raw_detector_confidence"):
            EvidenceRecord("event_0", "scene_a", "episode_1", 0, 0,
                           "object_observation", raw_detector_confidence=1.4)
        with self.assertRaisesRegex(ValueError, "JSON number"):
            EvidenceRecord("event_0", "scene_a", "episode_1", 0, 0,
                           "object_observation", payload={"value": math.nan})
        other_scene = canonical_evidence_key("scene_b", "episode_1", 0, "event_0")
        with self.assertRaisesRegex(ValueError, "crosses scene"):
            SpatialNodeV3("place_1", "scene_a", "place", evidence_ids=[other_scene])
        with self.assertRaisesRegex(ValueError, "successful traversal"):
            SpatialEdgeV3("edge_1", "scene_a", "a", "b", "verified_traversal",
                          traversal_count=1, traversability_status="traversed")
        with self.assertRaisesRegex(ValueError, "UNKNOWN traversability"):
            SpatialEdgeV3("edge_2", "scene_a", "a", "b", "observed_transition",
                          traversal_count=1, successful_traversals=1,
                          traversability_status="traversed")
        with self.assertRaisesRegex(ValueError, "crosses episode"):
            EpisodeMemory("scene_a", "episode_2", "Find door", observations=[
                canonical_evidence_key("scene_a", "episode_1", 0, "event_0"),
            ])


if __name__ == "__main__":
    unittest.main()
