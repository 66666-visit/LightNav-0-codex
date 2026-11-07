"""CPU-only tests for hierarchical evidence fusion and serialization."""

from __future__ import annotations

import json
import unittest

from lightnav.memory.episodic_memory import EpisodicMemory
from lightnav.memory.memory_context_builder import inject_memory_context
from lightnav.memory.memory_fusion import MemoryFusion
from lightnav.memory.scene_graph import SceneGraph
from lightnav.memory.schemas import (
    LayerMemoryItem,
    MemoryContext,
    MemoryEvent,
    MemoryQuery,
    ObservationEvidence,
)
from lightnav.memory.semantic_index import SemanticIndex
from lightnav.memory.topological_memory import TopologicalMemory
from lightnav.memory.working_memory import WorkingMemoryStore
from lightnav.prompts import UNIFIED_TRAJ_PROMPT_TEMPLATE, to_rvq_prompt


def _ref(event_id: str, step: int, label: str, node_id: str) -> ObservationEvidence:
    return ObservationEvidence(event_id, step, label, 0.8, "mock_rgb", node_id, "RGB detection")


def _item(layer: str, kind: str, ref: ObservationEvidence) -> LayerMemoryItem:
    return LayerMemoryItem(
        layer=layer, kind=kind, description=f"Observed {ref.label}",
        step_id=ref.step_id, confidence=ref.confidence, source=ref.source,
        evidence_refs=[ref], label=ref.label,
    )


class MemoryFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fusion = MemoryFusion()
        self.query = MemoryQuery("find couch", current_step=20, top_k=5, max_chars=600)

    def test_same_observation_is_described_once_across_three_layers(self) -> None:
        couch = _ref("event_5", 5, "couch", "object_5")
        short = _item("short", "current_observation", couch)
        mid = _item("mid", "object_observation", couch)
        long = LayerMemoryItem(
            "long", "semantic_summary", "Couch sightings", 5, 0.8, "mock_rgb",
            [couch], label="couch",
        )
        context = self.fusion.build(
            MemoryQuery("find couch", current_step=5, top_k=5, max_chars=600),
            [short], [mid], [long],
        )
        self.assertEqual(len(context.layer_items), 1)
        self.assertEqual(context.layer_items[0].layer, "short")
        self.assertEqual(context.text.count("event_5@5"), 1)

    def test_long_summary_keeps_different_time_evidence_after_overlap(self) -> None:
        old = _ref("event_5", 5, "couch", "object_5")
        current = _ref("event_20", 20, "couch", "object_20")
        short = _item("short", "current_observation", current)
        long = LayerMemoryItem(
            "long", "semantic_summary", "Couch sightings", 20, 0.8, "mock_rgb",
            [old, current], label="couch",
        )
        context = self.fusion.build(self.query, [short], [], [long])
        self.assertEqual([item.layer for item in context.layer_items], ["short", "long"])
        self.assertEqual([ref.event_id for ref in context.layer_items[1].evidence_refs], ["event_5"])
        self.assertIn("first step=5, latest step=5", context.text)
        self.assertEqual(context.text.count("event_20@20"), 1)

    def test_same_name_different_detection_ids_both_survive(self) -> None:
        first = _ref("event_a", 5, "chair", "object_a")
        second = _ref("event_b", 5, "chair", "object_b")
        context = self.fusion.build(
            self.query, [_item("short", "current_observation", first)],
            [_item("mid", "object_observation", second)], [],
        )
        self.assertEqual(len(context.layer_items), 2)
        self.assertIn("event_a@5", context.text)
        self.assertIn("event_b@5", context.text)

    def test_long_summary_lists_distinct_observation_steps(self) -> None:
        first = _ref("event_5", 5, "couch", "object_5")
        second = _ref("event_10", 10, "couch", "object_10")
        summary = LayerMemoryItem(
            "long", "semantic_summary", "Couch sightings", 10, 0.8, "mock_rgb",
            [first, second], label="couch",
        )
        context = self.fusion.build(self.query, [], [], [summary])
        self.assertIn("observation steps=5,10", context.text)
        self.assertEqual(len(context.layer_items[0].evidence_refs), 2)

    def test_place_semantics_keeps_only_uncovered_historical_detection(self) -> None:
        old = _ref("old", 5, "couch", "object_old")
        current = _ref("current", 20, "couch", "object_current")
        summary = LayerMemoryItem(
            "long", "place_semantics", "couch near visual place", 20, 0.8,
            "mock_rgb", [old, current], label="couch",
            metadata={"place_id": "place_000", "current": True},
        )
        fused = self.fusion.build(
            self.query, [_item("short", "current_observation", current)], [], [summary],
        )
        self.assertEqual([item.kind for item in fused.layer_items], [
            "current_observation", "place_semantics",
        ])
        self.assertEqual(fused.layer_items[1].evidence_refs, [old])
        self.assertIn("Historical RGB detections at visual place place_000", fused.text)
        self.assertEqual(
            MemoryContext.from_dict(json.loads(json.dumps(fused.to_dict()))), fused,
        )

    def test_place_semantics_labels_mixed_current_and_historical_sightings(self) -> None:
        old = _ref("old", 5, "couch", "object_old")
        current = _ref("current", 20, "couch", "object_current")
        summary = LayerMemoryItem(
            "long", "place_semantics", "couch near visual place", 20, 0.8,
            "mock_rgb", [old, current], label="couch",
            metadata={"place_id": "place_000", "current": True},
        )
        fused = self.fusion.build(self.query, [], [], [summary])
        self.assertIn("Current and historical RGB detections", fused.text)
        self.assertEqual(len(fused.layer_items[0].evidence_refs), 2)

    def test_current_place_and_observed_edge_fit_unchanged_character_cap(self) -> None:
        topology = TopologicalMemory()
        topology.observe(
            0, (1.0, 0.0), [], descriptor_source="mock_dino",
            observation_event_id="place_observation_0",
        )
        chair = MemoryEvent(
            "chair_5", 5, "object_observation", "Observed chair", 0.8,
            "mock_rgb", metadata={"node_id": "object_5"},
        )
        topology.observe(
            5, (0.0, 1.0), [chair], descriptor_source="mock_dino",
            observation_event_id="place_observation_5",
            transition_evidence_ids=tuple(f"action_{step}" for step in range(1, 6)),
        )
        query = MemoryQuery("find chair", 5, top_k=5, max_chars=600)
        current = _item("short", "current_observation", _ref("chair_5", 5, "chair", "object_5"))
        mid = _item("mid", "object_observation", _ref("old", 0, "table", "object_0"))
        fused = self.fusion.build(query, [current], [mid], topology.retrieve(query))
        self.assertEqual(fused.layer_items[0].kind, "current_place")
        self.assertIn("[short:current_observation", fused.text)
        self.assertIn("[long:observed_transition", fused.text)
        self.assertIn("evidence=place_observation_0->place_observation_5", fused.text)
        self.assertNotIn("place_shift_5", fused.text)
        self.assertEqual(fused.text.count("chair_5@5"), 1)
        self.assertLessEqual(len(fused.text), 600)

    def test_empty_layer_and_context_roundtrip(self) -> None:
        self.assertEqual(self.fusion.build(self.query, [], [], []).text, "")
        old = _ref("event_5", 5, "couch", "object_5")
        context = self.fusion.build(self.query, [], [_item("mid", "object_observation", old)], [])
        restored = MemoryContext.from_dict(json.loads(json.dumps(context.to_dict())))
        self.assertEqual(restored, context)

    def test_hierarchical_context_keeps_instruction_video_and_rvq_prompt(self) -> None:
        original_prompt = to_rvq_prompt(UNIFIED_TRAJ_PROMPT_TEMPLATE).format(
            task="find couch", videos="<video> <video>",
        )
        videos = [{"video": object()}, {"video": object()}]
        sample = {
            "video_segments": videos,
            "conversations": [
                {"from": "human", "value": original_prompt},
                {"from": "gpt", "value": "placeholder"},
            ],
            "slowfast_abs_frame_indices": True,
        }
        context = self.fusion.build(
            MemoryQuery("find couch", current_step=5),
            [_item("short", "current_observation", _ref("e5", 5, "couch", "object_5"))],
            [], [],
        )
        injected = inject_memory_context(sample, context)
        prompt = injected["conversations"][0]["value"]
        self.assertIn("<navigation_task>find couch</navigation_task>", prompt)
        self.assertIn("[short:current_observation", prompt)
        self.assertEqual(prompt.count("<video>"), original_prompt.count("<video>"))
        self.assertIn("coarse-to-fine trajectory tokens", prompt)
        self.assertIs(injected["video_segments"], videos)
        self.assertTrue(injected["slowfast_abs_frame_indices"])
        self.assertEqual(injected["conversations"][1], sample["conversations"][1])
        self.assertEqual(sample["conversations"][0]["value"], original_prompt)

    def test_step_five_to_twenty_uses_independent_stores_and_deduplicates_fusion(self) -> None:
        instruction = "find the couch seen earlier"
        short, mid, long, graph = (
            WorkingMemoryStore(), EpisodicMemory(), SemanticIndex(), SceneGraph(),
        )
        short.reset("episode", instruction)
        for step in range(21):
            short.advance(step)
            events = []
            if step:
                events.append(MemoryEvent(
                    f"action_{step}", step, "action", "Issued velocity_control", 1.0,
                    "command", importance=0.4,
                ))
            if step == 5:
                node = graph.add_object(
                    "couch", place_id=None, confidence=0.82, source="mock_rgb",
                    step_id=5, evidence="RGB detection at step 5",
                )
                events.append(MemoryEvent(
                    "couch_5", 5, "object_observation", "Observed couch", 0.82,
                    "mock_rgb", importance=0.6, metadata={"node_id": node.node_id},
                ))
            for event in events:
                mid.add(event)
            short.record_observation(step, events)
            long.update(events, graph, instruction)
            if step in {5, 20}:
                query = MemoryQuery(instruction, step, top_k=5, max_chars=600)
                layer_items = (short.retrieve(query), mid.retrieve(query), long.retrieve(query))
                self.assertTrue(layer_items[1] and layer_items[2])
                fused = self.fusion.build(query, *layer_items)
                self.assertEqual(
                    sum("couch" in line for line in fused.text.splitlines()), 1,
                )
                self.assertEqual(sum(
                    ref.event_id == "couch_5" for item in fused.layer_items
                    for ref in item.evidence_refs
                ), 1)
                if step == 5:
                    self.assertEqual(layer_items[0][0].kind, "current_observation")
                    self.assertIn("[short:current_observation", fused.text)
                else:
                    self.assertFalse(any(item.label == "couch" for item in layer_items[0]))
                    self.assertIn("[mid:object_observation step=5", fused.text)


if __name__ == "__main__":
    unittest.main()
