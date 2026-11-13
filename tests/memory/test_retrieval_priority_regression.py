"""Legacy RGM chooses the most faithful representation of shared RGB evidence."""

from __future__ import annotations

import unittest

from lightnav.memory.memory_fusion import MemoryFusion
from lightnav.memory.memory_prompt_gate import MemoryPromptGate
from lightnav.memory.memory_retrieval_gate import MemoryRetrievalGate
from lightnav.memory.schemas import LayerMemoryItem, MemoryQuery, ObservationEvidence


def _item(layer: str, kind: str, refs: list[ObservationEvidence], *,
          confidence: float = 0.88, metadata: dict | None = None,
          label: str = "couch") -> LayerMemoryItem:
    return LayerMemoryItem(
        layer=layer, kind=kind, description=f"Observed {label}",
        step_id=0, confidence=confidence,
        source="semantic_index" if kind == "semantic_summary" else "mock_yolo",
        evidence_refs=refs, label=label, metadata=metadata or {},
    )


class RetrievalPriorityRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query = MemoryQuery("Find the sofa", current_step=0)
        self.object_ref = ObservationEvidence("event_0", 0, "couch", 0.88, "mock_yolo")

    def _select(self, layers: dict[str, list[LayerMemoryItem]],
                accepted: set[str]) -> tuple[MemoryRetrievalGate, list[LayerMemoryItem], str]:
        retrieval = MemoryRetrievalGate()
        filtered = retrieval.filter(self.query, layers, accepted)
        fused = MemoryFusion().build(
            self.query, filtered["short"], filtered["mid"], filtered["long"],
        )
        prompt = MemoryPromptGate().select(self.query, fused, accepted)
        return retrieval, prompt.layer_items, prompt.text

    def test_step0_direct_short_wins_over_same_evidence_summary(self) -> None:
        layers = {
            "short": [_item("short", "current_observation", [self.object_ref])],
            "mid": [_item("mid", "object_observation", [self.object_ref])],
            "long": [_item("long", "semantic_summary", [self.object_ref])],
        }
        gate, selected, text = self._select(layers, {"event_0"})
        self.assertEqual([item.kind for item in selected], ["current_observation"])
        self.assertIn("instruction landmark couch", text)
        self.assertEqual([ref.event_id for item in selected for ref in item.evidence_refs],
                         ["event_0"])
        self.assertEqual([row["reason"] for row in gate.decisions],
                         ["accepted", "duplicate_evidence", "duplicate_evidence"])

    def test_summary_keeps_other_evidence_without_repeating_direct_view(self) -> None:
        older_ref = ObservationEvidence("event_older", 0, "couch", 0.88, "mock_yolo")
        layers = {
            "short": [_item("short", "current_observation", [self.object_ref])],
            "mid": [],
            "long": [_item("long", "semantic_summary", [older_ref, self.object_ref])],
        }
        gate, selected, _ = self._select(layers, {"event_0", "event_older"})
        # Step 0 Prompt Gate may decline the partial summary, while Retrieval
        # must preserve its separate evidence for a later eligible query.
        selected_ids = [ref.event_id for item in selected for ref in item.evidence_refs]
        self.assertEqual(selected_ids, ["event_0"])
        self.assertEqual(len(selected_ids), len(set(selected_ids)))
        self.assertEqual([row["reason"] for row in gate.decisions],
                         ["accepted", "accepted"])
        self.assertEqual(gate.decisions[1]["selected_evidence_ids"], ["event_older"])

    def test_valid_current_place_semantics_keeps_richer_relation_once(self) -> None:
        place_ref = ObservationEvidence(
            "place_observation_0", 0, "place_000", 1.0, "mock_dino",
        )
        layers = {
            "short": [_item("short", "current_observation", [self.object_ref])],
            "mid": [_item("mid", "object_observation", [self.object_ref])],
            "long": [
                _item("long", "current_place", [place_ref], confidence=0.5,
                      metadata={"place_id": "place_000"}, label="place_000"),
                _item("long", "place_semantics", [self.object_ref],
                      metadata={"place_id": "place_000", "current": True,
                                "descriptor_event_ids": ["place_observation_0"]}),
            ],
        }
        gate, selected, text = self._select(
            layers, {"event_0", "place_observation_0"},
        )
        self.assertEqual([item.kind for item in selected], ["place_semantics"])
        self.assertIn("instruction landmark couch while visual place hypothesis", text)
        self.assertEqual([ref.event_id for item in selected for ref in item.evidence_refs],
                         ["event_0"])
        self.assertTrue(any(row["kind"] == "current_observation"
                            and row["reason"] == "duplicate_evidence"
                            for row in gate.decisions))

    def test_unsupported_place_relation_cannot_outrank_direct_observation(self) -> None:
        layers = {
            "short": [_item("short", "current_observation", [self.object_ref])],
            "mid": [],
            "long": [_item("long", "place_semantics", [self.object_ref],
                           metadata={"place_id": "place_000", "current": True,
                                     "descriptor_event_ids": []})],
        }
        gate, selected, text = self._select(layers, {"event_0"})
        self.assertEqual([item.kind for item in selected], ["current_observation"])
        self.assertIn("instruction landmark couch", text)
        self.assertIn("unverified_spatial_assertion",
                      [row["reason"] for row in gate.decisions])


if __name__ == "__main__":
    unittest.main()
