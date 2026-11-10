"""CPU-only checks for RGM evidence gates and prompt recovery."""

from __future__ import annotations

import unittest

from lightnav.memory.memory_context_builder import audit_prompt_injection, inject_memory_context
from lightnav.memory.memory_fusion import MemoryFusion
from lightnav.memory.memory_prompt_gate import MemoryPromptGate
from lightnav.memory.memory_retrieval_gate import MemoryRetrievalGate
from lightnav.memory.memory_write_gate import MemoryWriteGate, instruction_match
from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.schemas import (
    LayerMemoryItem, MemoryContext, MemoryEvent, MemoryQuery, ObservationEvidence,
)
from lightnav.memory.topological_memory import TopologicalMemory


def _event(step: int, sequence: int, label: str, confidence: float) -> MemoryEvent:
    return MemoryEvent(
        event_id=f"event_{sequence}", step_id=step, kind="object_observation",
        description=f"Observed {label}", confidence=confidence, source="mock_yolo",
    )


def _item(event_id: str, step: int = 0, label: str = "couch") -> LayerMemoryItem:
    return LayerMemoryItem(
        layer="short", kind="current_observation", description=f"Observed {label}",
        step_id=step, confidence=0.8, source="mock_yolo", label=label,
        evidence_refs=[ObservationEvidence(event_id, step, label, 0.8, "mock_yolo")],
    )


class RGMGateTests(unittest.TestCase):
    def test_instruction_synonyms_use_whole_terms(self) -> None:
        self.assertTrue(instruction_match("couch", "Walk to the sofa."))
        self.assertTrue(instruction_match("television", "Go to the TV"))
        self.assertFalse(instruction_match("sofa", "Go to the sofabed"))

    def test_raw_events_remain_auditable_and_temporal_support_is_independent(self) -> None:
        gate = MemoryWriteGate("episode_a", "Find the sofa", RGMConfig(
            enabled=True, yolo_temporal_enabled=True,
        ))
        unrelated = _event(0, 0, "bench", 0.8)
        repeated = _event(0, 1, "bench", 0.9)
        later = _event(5, 2, "bench", 0.8)
        self.assertFalse(gate.decide(unrelated).accepted)
        self.assertEqual(gate.decide(repeated).reason, "duplicate_same_step_category")
        self.assertEqual(gate.decide(later).reason, "temporally_supported_category")
        self.assertEqual(len(gate.evidence), 3)
        self.assertEqual(gate.evidence[0].episode_id, "episode_a")
        self.assertEqual(later.metadata["rgm_temporal_support"]["steps"], [0, 5])
        gate.clear()
        self.assertEqual(gate.evidence, [])

    def test_empty_yolo_extractions_expire_old_temporal_support(self) -> None:
        gate = MemoryWriteGate("episode_a", "Find the sofa", RGMConfig(
            enabled=True, yolo_temporal_enabled=True,
        ))
        gate.record_extraction(0)
        self.assertFalse(gate.decide(_event(0, 0, "bench", 0.8)).accepted)
        gate.record_extraction(5)
        gate.record_extraction(10)
        gate.record_extraction(15)
        decision = gate.decide(_event(15, 1, "bench", 0.8))
        self.assertFalse(decision.accepted)
        self.assertEqual(gate.evidence[-1].step, 15)

    def test_instruction_grounded_current_frame_and_gate_switches(self) -> None:
        gate = MemoryWriteGate("e", "Find the sofa", RGMConfig(enabled=True))
        self.assertEqual(gate.decide(_event(0, 0, "couch", 0.7)).reason,
                         "instruction_grounded_observation")
        self.assertFalse(gate.decide(_event(5, 1, "couch", 0.3)).accepted)
        disabled = MemoryWriteGate("e", "", RGMConfig(
            enabled=True, write_gate_enabled=False,
        ))
        self.assertTrue(disabled.decide(_event(0, 0, "bench", 0.1)).accepted)

    def test_c_without_temporal_consensus_still_injects_grounded_memory(self) -> None:
        config = RGMConfig(enabled=True, yolo_temporal_enabled=False)
        gate = MemoryWriteGate("episode_c", "Find the sofa", config)
        event = _event(0, 0, "couch", 0.8)
        self.assertTrue(gate.decide(event).accepted)
        query = MemoryQuery("Find the sofa", current_step=0)
        layers = {"short": [_item(event.event_id)], "mid": [], "long": []}
        filtered = MemoryRetrievalGate().filter(query, layers, {event.event_id})
        fused = MemoryFusion().build(query, **{
            "short_items": filtered["short"], "mid_items": filtered["mid"],
            "long_items": filtered["long"],
        })
        selected = MemoryPromptGate().select(query, fused, {event.event_id})
        self.assertIn("couch", selected.text)

    def test_retrieval_and_prompt_gates_reject_unadopted_evidence(self) -> None:
        query = MemoryQuery("Find the sofa", current_step=0)
        item = _item("event_0")
        layers = {"short": [item], "mid": [], "long": []}
        retrieval = MemoryRetrievalGate()
        self.assertEqual(retrieval.filter(query, layers, set())["short"], [])
        self.assertEqual(len(retrieval.filter(query, layers, {"event_0"})["short"]), 1)
        self.assertIsNot(retrieval.filter(query, layers, {"event_0"}), layers)
        prompt = MemoryPromptGate()
        context = MemoryContext(text="old", layer_items=[item])
        self.assertEqual(prompt.select(query, context, set()).text, "")
        self.assertIn("couch", prompt.select(query, context, {"event_0"}).text)
        self.assertEqual(prompt.select(query, context, {"event_0"}).layer_items, [item])
        later = MemoryQuery("Find the sofa", current_step=5)
        live = _item("event_5", step=5)
        self.assertEqual(prompt.select(later, MemoryContext(layer_items=[live]),
                                       {"event_5"}).text, "")
        self.assertIn("couch", prompt.select(
            later, MemoryContext(layer_items=[live]), {"event_5"}, {"event_5": 2},
        ).text)

    def test_retrieval_ranks_before_deduplicating_and_switches_are_independent(self) -> None:
        query = MemoryQuery("Find the sofa", current_step=5)
        weak = _item("event_0", step=0)
        strong = LayerMemoryItem(
            layer="mid", kind="object_observation", description="Observed couch",
            step_id=0, confidence=0.95, source="mock_yolo", label="couch",
            evidence_refs=weak.evidence_refs,
        )
        layers = {"short": [weak], "mid": [strong], "long": []}
        retrieval = MemoryRetrievalGate()
        filtered = retrieval.filter(query, layers, {"event_0"})
        self.assertEqual(filtered["short"], [])
        self.assertEqual(filtered["mid"], [strong])
        self.assertIn("duplicate_evidence", [row["reason"] for row in retrieval.decisions])
        self.assertIs(MemoryRetrievalGate(enabled=False).filter(query, layers, set()), layers)
        context = MemoryContext(text="preexisting fusion text", layer_items=[weak])
        self.assertIs(MemoryPromptGate(enabled=False).select(query, context, set()), context)

    def test_prompt_exact_roundtrip_and_no_memory_equality(self) -> None:
        sample = {
            "conversations": [
                {"from": "human", "value": "<navigation_task>Find sofa</navigation_task>. <video> Return <act_l0_*>."},
                {"from": "gpt", "value": ""},
            ],
            "video_segments": [object()],
        }
        empty = MemoryContext()
        self.assertEqual(inject_memory_context(sample, empty), sample)
        context = MemoryContext(text="Previously saw sofa; evidence event_1@5.")
        changed = inject_memory_context(sample, context)
        audit = audit_prompt_injection(sample, changed, context, max_chars=600)
        self.assertTrue(audit["roundtrip_ok"])
        self.assertIs(changed["video_segments"], sample["video_segments"])
        self.assertIn("Return <act_l0_*>.", changed["conversations"][0]["value"])
        self.assertEqual(sample["conversations"][0]["value"],
                         "<navigation_task>Find sofa</navigation_task>. <video> Return <act_l0_*>.")

    def test_adaptive_scheduler_is_explicitly_unavailable_in_v1(self) -> None:
        with self.assertRaises(ValueError):
            RGMConfig(adaptive_extraction_enabled=True)

    def test_forced_second_place_preserves_true_global_top_one_score(self) -> None:
        topology = TopologicalMemory()
        topology.observe(0, (1.0, 0.0), [], descriptor_source="mock_dino",
                         observation_event_id="place_0")
        second = topology.observe(5, (0.8, 0.6), [], descriptor_source="mock_dino",
                                  observation_event_id="place_5", force_new=True)
        with self.assertRaises(ValueError):
            topology.observe(10, (0.98, 0.2), [], descriptor_source="mock_dino",
                             observation_event_id="invalid", force_new=True,
                             force_match_place_id=second.place_id)
        result = topology.observe(10, (0.98, 0.2), [], descriptor_source="mock_dino",
                                  observation_event_id="place_10",
                                  force_match_place_id=second.place_id)
        self.assertEqual(result.place_id, second.place_id)
        self.assertGreater(result.top1_similarity, topology.get_node(second.place_id).last_similarity)


if __name__ == "__main__":
    unittest.main()
