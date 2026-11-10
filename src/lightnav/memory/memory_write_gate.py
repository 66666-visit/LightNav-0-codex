"""Audit raw detector evidence before admitting it to navigational indexes."""

from __future__ import annotations

import re
from time import perf_counter

from lightnav.memory.object_association import ObjectAssociation
from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.schemas import GateDecision, MemoryEvent, RGMEvidence

_SYNONYMS = {
    "sofa": "couch", "couches": "couch", "sofas": "couch",
    "tv": "television", "tvs": "television", "televisions": "television",
    "chairs": "chair", "tables": "table", "benches": "bench",
}
_TERMS = re.compile(r"[^\W_]+", re.UNICODE)


def normalized_terms(value: str) -> tuple[str, ...]:
    return tuple(_SYNONYMS.get(term, term) for term in _TERMS.findall(value.casefold()))


def instruction_match(label: str, instruction: str) -> tuple[str, ...]:
    label_terms = set(normalized_terms(label))
    instruction_terms = set(normalized_terms(instruction))
    return tuple(sorted(label_terms)) if label_terms and label_terms <= instruction_terms else ()


class MemoryWriteGate:
    def __init__(self, episode_id: str, instruction: str, config: RGMConfig) -> None:
        self.episode_id = episode_id
        self.instruction = instruction
        self.config = config
        self.temporal = ObjectAssociation(
            window=config.temporal_window, min_support=config.temporal_min_support,
        )
        self.evidence: list[RGMEvidence] = []
        self.decisions: list[GateDecision] = []
        self._seen_ids: set[str] = set()
        self._last_step = -1

    def record_extraction(self, step: int) -> None:
        self.temporal.advance_extraction(step)

    def decide(self, event: MemoryEvent) -> GateDecision:
        if event.event_id in self._seen_ids or event.step_id < self._last_step:
            raise ValueError("RGM evidence ID duplicated or step regressed")
        self._seen_ids.add(event.event_id)
        self._last_step = event.step_id
        label = event.description.removeprefix("Observed ").strip()
        terms = instruction_match(label, self.instruction) if event.kind in {
            "object_observation", "landmark",
        } else ()
        evidence = RGMEvidence(
            episode_id=self.episode_id, step=event.step_id, event_id=event.event_id,
            kind=event.kind, source=event.source,
            payload={"label": label, "bbox_xyxy": event.metadata.get("bbox_xyxy")},
            confidence=event.confidence, related_instruction_terms=terms,
            provenance={"observation": "rgb", "pose": "unavailable", "collision": "unavailable"},
        )
        self.evidence.append(evidence)
        support = None
        if event.kind in {"object_observation", "landmark"}:
            started = perf_counter()
            support = self.temporal.observe(event)
            event.metadata["rgm_association_ms"] = (perf_counter() - started) * 1000.0
            event.metadata["rgm_temporal_support"] = {
                "event_ids": list(support.event_ids), "steps": list(support.steps),
                "category_only": True, "instance_identity": "unavailable",
            }
        if not self.config.write_gate_enabled:
            decision = GateDecision(True, event.confidence, "write_gate_disabled", (event.event_id,))
        elif event.kind not in {"object_observation", "landmark"}:
            decision = GateDecision(False, event.confidence, "not_navigation_landmark", (event.event_id,))
        elif event.confidence < self.config.min_candidate_confidence:
            decision = GateDecision(False, event.confidence, "low_detection_confidence", (event.event_id,))
        elif support is not None and support.event_ids[-1] != event.event_id:
            decision = GateDecision(False, event.confidence, "duplicate_same_step_category", (event.event_id,))
        elif terms and event.confidence >= self.config.min_instruction_landmark_confidence:
            decision = GateDecision(True, event.confidence, "instruction_grounded_observation", (event.event_id,))
        elif self.config.yolo_temporal_enabled and support is not None and support.supported:
            decision = GateDecision(True, event.confidence, "temporally_supported_category", support.event_ids)
        else:
            decision = GateDecision(False, event.confidence, "insufficient_navigation_evidence", (event.event_id,))
        event.metadata["rgm_adopted"] = decision.accepted
        event.metadata["rgm_write_reason"] = decision.reason
        self.decisions.append(decision)
        return decision

    def clear(self) -> None:
        self.temporal.clear()
        self.evidence.clear()
        self.decisions.clear()
        self._seen_ids.clear()
        self._last_step = -1
