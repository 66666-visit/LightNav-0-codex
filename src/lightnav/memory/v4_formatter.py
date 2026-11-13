"""Bounded, evidence-linked text for selective V4 memory use.

The formatter describes earlier observations and uncertainty. It never issues a
navigation command; LightNav remains the only action policy. The returned
``MemoryContext`` uses the existing safe insertion point after the original
``</navigation_task>`` marker.
"""

from __future__ import annotations

import json
from hashlib import sha256
from html import escape

from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence
from lightnav.memory.v4_context import (
    DroppedCandidate,
    NavigationFact,
    NavigationMemoryContext,
    RevisitEvidence,
)
from lightnav.memory.v4_navigation_state import EvidenceProvenance
from lightnav.memory.v4_utilization import (
    HISTORY_REFERENCE,
    NONE,
    PROGRESS_AUDIT,
    REVISIT_AUDIT,
)

_HEADER = "[Navigation Memory]"
_VISUAL_EVENTS = frozenset({
    "object_observation", "landmark", "place_visit", "place_hypothesis",
})
_ISSUED_EVENTS = frozenset({"action", "turn", "issued_command", "prediction"})
_RELATION_EVENTS = frozenset({
    "inferred_relation", "observed_transition", "spatial_relation",
    "viewpoint_connection",
})


def _clean(value: str) -> str:
    return " ".join(value.split())


def _source_ref(evidence_id: str) -> str:
    """Give the prompt a short handle; full canonical IDs remain in diagnostics."""
    return sha256(evidence_id.encode("utf-8")).hexdigest()[:12]


def _source_text(provenance: tuple[EvidenceProvenance, ...]) -> str:
    """Keep Episode/Step explicit without filling the budget with key syntax."""
    return "; ".join(
        f"Episode={json.dumps(source.episode_id, ensure_ascii=False)}, "
        f"Rollout={source.rollout_idx}, Step={source.step}, "
        f"Ref={_source_ref(source.evidence_id)}"
        for source in provenance
    )


def _fact_line(fact: NavigationFact, mode: str) -> str | None:
    if (not fact.instruction_relevant or fact.temporal_role == "CURRENT"
            or fact.kind in _ISSUED_EVENTS or not fact.evidence_ids
            or len(fact.evidence_ids) != len(fact.provenance)):
        return None
    if any(source.evidence_id != key for key, source in zip(
        fact.evidence_ids, fact.provenance,
    )):
        return None

    if fact.kind in _VISUAL_EVENTS:
        expected_status = "inferred" if fact.kind == "place_hypothesis" else "observed"
        if fact.observation_status != expected_status:
            return None
        label = _clean(fact.semantic_label or "unspecified landmark")
        if fact.kind == "place_hypothesis":
            if not fact.semantic_label:
                return None
            claim = (f"[INFERRED] Earlier RGB extractor described {label}; "
                     "place identity UNKNOWN; visit UNKNOWN")
        elif fact.kind == "place_visit":
            claim = f"[OBSERVED] Earlier RGB place-label hypothesis: {label}; visit UNKNOWN"
        else:
            claim = f"[OBSERVED] Earlier RGB detector reported {label}; visit UNKNOWN"
        if mode == PROGRESS_AUDIT:
            claim += "; passage UNKNOWN; instruction completion UNKNOWN"
    elif fact.kind == "verified_traversal":
        if (mode != PROGRESS_AUDIT or fact.visit_status != "VERIFIED"
                or fact.traversability_status not in {"traversed", "verified"}):
            return None
        detail = _clean(fact.description).split(";")[0]
        claim = (f"[VERIFIED] {detail}; "
                 "instruction completion UNKNOWN")
    elif fact.kind in _RELATION_EVENTS:
        if mode != HISTORY_REFERENCE or fact.traversability_status != "unknown":
            return None
        status = "OBSERVED" if (
            fact.kind == "observed_transition" and fact.spatial_status == "observed"
        ) else "INFERRED"
        detail = _clean(fact.description).split(";")[0]
        claim = (f"[{status}] {detail}; "
                 "passage UNKNOWN; place identity UNKNOWN")
    else:
        return None
    return f"{claim}; {_source_text(fact.provenance)}."


def _revisit_line(candidate: RevisitEvidence) -> str | None:
    if (candidate.identity_status != "unknown" or not candidate.instruction_related
            or candidate.independent_history_observations < 2
            or not candidate.historical_evidence_ids
            or len(candidate.provenance) != 1 + len(candidate.historical_evidence_ids)):
        return None
    keys = (candidate.current_evidence_id, *candidate.historical_evidence_ids)
    if any(source.evidence_id != key for key, source in zip(keys, candidate.provenance)):
        return None
    current = candidate.provenance[0]
    history = candidate.provenance[1:]
    if any((source.episode_id, source.rollout_idx, source.step) == (
        current.episode_id, current.rollout_idx, current.step,
    ) for source in history):
        return None
    return (
        "[INFERRED] Possible revisit from RGB similarity; place identity UNKNOWN; "
        f"current observation: {_source_text((current,))}; "
        f"historical observations: {_source_text(history)}."
    )


class V4PromptFormatter:
    """Serialize only utilization-selected evidence within a strict text budget."""

    def __init__(self, *, max_records: int = 3, max_chars: int = 400) -> None:
        if max_records < 1 or max_chars < 1:
            raise ValueError("V4 prompt budgets must be positive")
        self.max_records = max_records
        self.max_chars = max_chars
        self.last_dropped: list[DroppedCandidate] = []
        self.last_lines: list[str] = []
        self.last_serialized = 0

    def format(
        self, context: NavigationMemoryContext, *,
        excluded_evidence_ids: frozenset[str] = frozenset(),
    ) -> MemoryContext | None:
        """Return None for exact baseline fallback; never truncate a fact mid-line."""
        self.last_dropped = []
        self.last_lines = []
        self.last_serialized = 0
        mode = context.mode
        if mode == NONE:
            return None
        if mode not in {HISTORY_REFERENCE, REVISIT_AUDIT, PROGRESS_AUDIT}:
            raise ValueError(f"unsupported V4 memory mode: {mode}")

        if mode == REVISIT_AUDIT:
            candidates: tuple[NavigationFact | RevisitEvidence, ...] = context.selected_revisits
        else:
            candidates = context.selected_facts
        if not candidates:
            return None

        lines: list[str] = []
        items: list[LayerMemoryItem] = []
        seen = set(excluded_evidence_ids)
        used = len(escape(_HEADER, quote=False))
        for candidate in candidates:
            if len(lines) >= self.max_records:
                self.last_dropped.append(DroppedCandidate(
                    candidate.source_memory_id if isinstance(candidate, NavigationFact)
                    else "revisit",
                    candidate.evidence_ids if isinstance(candidate, NavigationFact)
                    else (candidate.current_evidence_id, *candidate.historical_evidence_ids),
                    "record_budget",
                ))
                continue
            if isinstance(candidate, NavigationFact):
                keys = candidate.evidence_ids
                memory_id = candidate.source_memory_id
                line = _fact_line(candidate, mode)
                sources = candidate.provenance
                kind = candidate.kind
                layer = "long" if candidate.kind in _RELATION_EVENTS | {"verified_traversal"} else "mid"
            else:
                keys = (candidate.current_evidence_id, *candidate.historical_evidence_ids)
                memory_id = "revisit"
                line = _revisit_line(candidate)
                sources = candidate.provenance
                kind = "possible_revisit"
                layer = "long"
            if line is None:
                self.last_dropped.append(DroppedCandidate(memory_id, keys, "invalid_for_mode"))
                continue
            if len(set(keys)) != len(keys) or any(key in seen for key in keys):
                reason = ("excluded_evidence" if any(key in excluded_evidence_ids for key in keys)
                          else "duplicate_evidence")
                self.last_dropped.append(DroppedCandidate(memory_id, keys, reason))
                continue
            safe_line = escape(line, quote=False)
            # One newline before the first item, and one between later items.
            if used + 1 + len(safe_line) > self.max_chars:
                self.last_dropped.append(DroppedCandidate(memory_id, keys, "char_budget"))
                continue
            seen.update(keys)
            used += 1 + len(safe_line)
            lines.append(line)
            self.last_lines.append(safe_line)
            items.append(LayerMemoryItem(
                layer=layer, kind=f"v4_{kind}", description=line,
                step_id=sources[0].step, confidence=0.0, source="v4_formatter",
                evidence_refs=[ObservationEvidence(
                    event_id=source.evidence_id, step_id=source.step,
                    label=kind, confidence=0.0, source="v4_evidence",
                ) for source in sources],
                metadata={
                    "memory_id": memory_id,
                    "mode": mode,
                    "evidence_ids": list(keys),
                    "provenance": [
                        {"scene_id": source.scene_id, "episode_id": source.episode_id,
                         "rollout_idx": source.rollout_idx, "step": source.step,
                         "evidence_id": source.evidence_id,
                         "prompt_ref": _source_ref(source.evidence_id)}
                        for source in sources
                    ],
                    "serialized_text": safe_line,
                },
            ))
        self.last_serialized = len(lines)
        if not lines:
            return None
        return MemoryContext(text=_HEADER + "\n" + "\n".join(lines), layer_items=items)
