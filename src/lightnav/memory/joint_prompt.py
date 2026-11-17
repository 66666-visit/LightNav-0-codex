"""Fuse organized L1 and L3 facts into one auditable navigation context.

Alignment uses only identities the memory backends already recorded:

* a trusted match is an L3 ``SpatialEvidence.linked_l1_evidence_id`` that L3
  minted at write time from a same-class, high-IoU, non-ambiguous L1 detection;
* an equal class label alone is *not* an identity, so those facts stay separate
  and are marked uncertain;
* a linked pair whose labels disagree is a conflict and is never merged.

The fusion keeps both sources and their uncertainty in the prompt. It does not
overwrite a current L1 observation with an older L3 record, does not claim a
straight-line range is a path, and does not claim object passability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import Any

from lightnav.memory.l1_prompt_organizer import L1Fact, L1Organization
from lightnav.memory.l3_prompt_organizer import L3Fact, L3Organization, _range_text
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence

HEADER = "[Navigation Context - L1+L3]"
_STATE_TEMPLATE = "Step {step}; RGB is the current view; map lines are history."
_NO_FACT_LINE = "No instruction-related L1 detection or L3 map fact yet."
_UNKNOWN_LINE = "Unknown: object passability; the map proves no path."


@dataclass(frozen=True)
class EntityAlignment:
    """One L1/L3 comparison with the evidence that justified it."""

    label: str
    status: str  # confirmed | uncertain_same_label | identity_conflict | unmatched
    l3_object_id: str
    l1_evidence_ids: tuple[str, ...]
    l3_evidence_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label, "status": self.status,
            "l3_object_id": self.l3_object_id,
            "l1_evidence_ids": list(self.l1_evidence_ids),
            "l3_evidence_ids": list(self.l3_evidence_ids),
            "reason": self.reason,
        }


@dataclass
class JointOrganization:
    """The fused card, its alignments, conflicts and layer contributions."""

    context: MemoryContext | None
    alignments: list[EntityAlignment] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    selected_kinds: list[str] = field(default_factory=list)
    merged_entities: int = 0
    duplicates_removed: int = 0
    unverified_relations: int = 0
    omitted_items: int = 0
    prompt_chars: int = 0
    l1_item_count: int = 0
    l3_item_count: int = 0
    joint_item_count: int = 0
    empty_reason: str | None = None

    def diagnostics(self) -> dict[str, Any]:
        attempted = sum(
            alignment.status != "unmatched" for alignment in self.alignments
        )
        confirmed = sum(
            alignment.status == "confirmed" for alignment in self.alignments
        )
        uncertain = sum(
            alignment.status == "uncertain_same_label" for alignment in self.alignments
        )
        return {
            "layer": "joint",
            "source": "joint_prompt_builder" if self.context is not None else None,
            "item_count": len(self.selected_kinds),
            "l1_item_count": self.l1_item_count,
            "l3_item_count": self.l3_item_count,
            "joint_item_count": self.joint_item_count,
            "merged_entities": self.merged_entities,
            "duplicates_removed": self.duplicates_removed,
            "conflict_count": len(self.conflicts),
            "conflicts": list(self.conflicts),
            "unknown_relation_count": self.unverified_relations,
            "omitted_items": self.omitted_items,
            "prompt_chars": self.prompt_chars,
            "alignment": {
                "attempted": attempted,
                "confirmed": confirmed,
                "uncertain": uncertain,
                "unmatched": len(self.alignments) - attempted,
                "success_rate": (
                    confirmed / attempted if attempted else None
                ),
                "items": [alignment.to_dict() for alignment in self.alignments],
            },
            "empty_reason": self.empty_reason,
        }


def _evidence_ref(candidate: "_Candidate", evidence_id: str) -> ObservationEvidence:
    for entry in candidate.provenance:
        if entry.get("evidence_id") == evidence_id:
            step = entry.get("step")
            return ObservationEvidence(
                event_id=evidence_id,
                step_id=step if isinstance(step, int) else candidate.step_id,
                label=candidate.label,
                confidence=candidate.confidence,
                source=str(entry.get("source") or "memory"),
            )
    return ObservationEvidence(
        event_id=evidence_id, step_id=candidate.step_id, label=candidate.label,
        confidence=candidate.confidence, source="memory",
    )


@dataclass
class _Candidate:
    kind: str  # l1_current | l1_history | l3 | joint
    priority: int
    label: str
    line: str
    evidence_ids: tuple[str, ...]
    l1_evidence_ids: tuple[str, ...]
    l3_evidence_ids: tuple[str, ...]
    provenance: tuple[dict[str, object], ...]
    confidence: float
    step_id: int
    layer: str = "long"
    aligned: bool = False


class JointPromptBuilder:
    """Build a compact, conflict-preserving L1 + L3 navigation context."""

    source = "joint_prompt_builder"

    def __init__(self, *, max_items: int = 3) -> None:
        if type(max_items) is not int or max_items < 1:
            raise ValueError("joint max_items must be a positive integer")
        self.max_items = max_items

    def build(
        self,
        l1: L1Organization,
        l3: L3Organization,
        *,
        step: int,
        max_chars: int,
    ) -> JointOrganization:
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if type(max_chars) is not int or max_chars < 1:
            raise ValueError("max_chars must be a positive integer")

        organization = JointOrganization(context=None)
        l1_facts = list(l1.facts)
        l3_facts = list(l3.facts)
        l1_by_label: dict[str, list[L1Fact]] = {}
        for fact in l1_facts:
            l1_by_label.setdefault(fact.label.casefold(), []).append(fact)
        consumed: set[tuple[str, str]] = set()
        candidates: list[_Candidate] = []
        notes: list[str] = []

        for fact in l3_facts:
            linked = set(fact.linked_l1_evidence_ids)
            matches = [
                item for item in l1_facts if linked & set(item.evidence_ids)
            ]
            same_label = l1_by_label.get(fact.label.casefold(), [])
            if matches:
                partner = matches[0]
                if partner.label.casefold() != fact.label.casefold():
                    consumed.add(partner.signature())
                    organization.alignments.append(EntityAlignment(
                        label=fact.label, status="identity_conflict",
                        l3_object_id=fact.object_id,
                        l1_evidence_ids=partner.evidence_ids,
                        l3_evidence_ids=fact.evidence_ids,
                        reason="linked_evidence_label_mismatch",
                    ))
                    organization.conflicts.append({
                        "type": "identity_mismatch", "label": fact.label,
                        "l1_label": partner.label,
                        "l1_evidence_ids": list(partner.evidence_ids),
                        "l3_evidence_ids": list(fact.evidence_ids),
                        "resolution": "both_kept_unmerged",
                    })
                    candidates.append(self._l1_candidate(
                        partner, priority=1 if partner.role == "current" else 4,
                    ))
                    candidates.append(self._l3_candidate(fact, priority=5))
                    continue
                consumed.add(partner.signature())
                organization.alignments.append(EntityAlignment(
                    label=fact.label, status="confirmed",
                    l3_object_id=fact.object_id,
                    l1_evidence_ids=partner.evidence_ids,
                    l3_evidence_ids=fact.evidence_ids,
                    reason="l3_linked_l1_evidence",
                ))
                organization.merged_entities += 1
                candidates.append(self._joint_candidate(partner, fact))
                continue

            if same_label:
                organization.alignments.append(EntityAlignment(
                    label=fact.label, status="uncertain_same_label",
                    l3_object_id=fact.object_id,
                    l1_evidence_ids=tuple(
                        evidence_id for item in same_label for evidence_id in item.evidence_ids
                    ),
                    l3_evidence_ids=fact.evidence_ids,
                    reason="same_class_without_trusted_link",
                ))
                notes.append(
                    f"Uncertain: {fact.label} may be the same object in L1 and L3 "
                    "(unconfirmed)."
                )
                if fact.geometry_verified:
                    candidates.append(self._l3_candidate(fact, priority=3))
                else:
                    organization.conflicts.append({
                        "type": "geometry_unverified", "label": fact.label,
                        "l1_evidence_ids": list(same_label[0].evidence_ids),
                        "l3_evidence_ids": list(fact.evidence_ids),
                        "resolution": "both_kept_position_unknown",
                    })
                    candidates.append(self._l3_candidate(fact, priority=6))
                continue

            organization.alignments.append(EntityAlignment(
                label=fact.label, status="unmatched", l3_object_id=fact.object_id,
                l1_evidence_ids=(), l3_evidence_ids=fact.evidence_ids,
                reason="no_l1_counterpart",
            ))
            candidates.append(self._l3_candidate(
                fact, priority=3 if fact.geometry_verified else 6,
            ))

        for fact in l1_facts:
            if fact.signature() in consumed:
                continue
            if fact.role == "current":
                candidates.append(self._l1_candidate(fact, priority=1))
            else:
                candidates.append(self._l1_candidate(fact, priority=4))

        candidates.sort(key=lambda item: (item.priority, item.label.casefold()))
        unique_candidates: list[_Candidate] = []
        seen_candidates: set[tuple[str, tuple[str, ...]]] = set()
        for candidate in candidates:
            signature = (candidate.line, candidate.evidence_ids)
            if signature in seen_candidates:
                organization.duplicates_removed += 1
                continue
            seen_candidates.add(signature)
            unique_candidates.append(candidate)
        candidates = unique_candidates

        lines = [HEADER]
        used = len(HEADER)
        items: list[LayerMemoryItem] = []

        def add_line(line: str, candidate: _Candidate | None = None) -> bool:
            nonlocal used
            safe_line = escape(line, quote=False)
            if used + 1 + len(safe_line) > max_chars:
                return False
            used += 1 + len(safe_line)
            lines.append(line)
            if candidate is not None:
                items.append(self._item(candidate, safe_line))
            return True

        if not add_line(_STATE_TEMPLATE.format(step=step)):
            organization.empty_reason = "budget_too_small_for_state"
            return organization

        selected: list[_Candidate] = []
        for candidate in candidates:
            if len(selected) >= self.max_items:
                organization.omitted_items += 1
                continue
            if not add_line(candidate.line, candidate):
                organization.omitted_items += 1
                continue
            selected.append(candidate)
            if candidate.kind == "joint":
                organization.joint_item_count += 1
            elif candidate.kind == "l3":
                organization.l3_item_count += 1
            else:
                organization.l1_item_count += 1
            if candidate.kind in {"l3", "joint"}:
                organization.unverified_relations += 1

        if not selected:
            add_line(_NO_FACT_LINE)
        elif items:
            for conflict in organization.conflicts:
                message = self._conflict_line(conflict)
                if not add_line(message):
                    break
            seen_notes: set[str] = set()
            for note in notes:
                if note in seen_notes:
                    continue
                seen_notes.add(note)
                if not add_line(note):
                    break
            if organization.unverified_relations:
                add_line(_UNKNOWN_LINE)

        organization.selected_kinds = [candidate.kind for candidate in selected]
        organization.prompt_chars = len("\n".join(lines))
        organization.context = MemoryContext(text="\n".join(lines), layer_items=items)
        return organization

    def _conflict_line(self, conflict: dict[str, Any]) -> str:
        if conflict["type"] == "identity_mismatch":
            return (
                f"Conflict: linked L1/L3 labels differ ({conflict['l1_label']} vs "
                f"{conflict['label']}); sources kept, not merged."
            )
        return (
            f"Conflict: L1 detects {conflict['label']} but L3 evidence is unverified; "
            "position unknown, both sources kept."
        )

    def _joint_candidate(self, l1: L1Fact, l3: L3Fact) -> _Candidate:
        visibility = "now visible" if l1.role == "current" else "last observed"
        sightings = f", {l1.sightings}x" if l1.sightings > 1 else ""
        views = f"{l3.viewpoints} view" + ("" if l3.viewpoints == 1 else "s")
        line = (
            f"{l1.label}: {visibility} (L1 {l1.confidence:.2f}{sightings}); "
            f"map {views}, last step {l3.last_step}"
        )
        if l3.distance_m is not None:
            line += f", {_range_text(l3.distance_m, l3.bearing_deg)}"
        elif not l3.geometry_verified:
            line += ", position unverified"
        else:
            line += ", position unknown"
        line += "; passage unknown."
        if l1.role == "history":
            line += f" L1 last step {l1.last_step}."
        provenance = (*l1.provenance, *l3.provenance)
        return _Candidate(
            kind="joint", priority=0 if l1.role == "current" else 2,
            label=l1.label, line=line,
            evidence_ids=tuple(dict.fromkeys((*l1.evidence_ids, *l3.evidence_ids))),
            l1_evidence_ids=l1.evidence_ids, l3_evidence_ids=l3.evidence_ids,
            provenance=provenance, confidence=l1.confidence, step_id=l1.last_step,
            layer="short" if l1.role == "current" else "mid",
            aligned=True,
        )

    def _l1_candidate(self, fact: L1Fact, *, priority: int) -> _Candidate:
        return _Candidate(
            kind="l1_current" if fact.role == "current" else "l1_history",
            priority=priority, label=fact.label, line=fact.line(),
            evidence_ids=fact.evidence_ids, l1_evidence_ids=fact.evidence_ids,
            l3_evidence_ids=(), provenance=fact.provenance,
            confidence=fact.confidence, step_id=fact.last_step,
            layer="short" if fact.role == "current" else "mid",
        )

    def _l3_candidate(self, fact: L3Fact, *, priority: int) -> _Candidate:
        line = fact.line()
        if not fact.geometry_verified:
            line = (
                f"{fact.label}: map evidence unverified; position and passage unknown."
            )
        return _Candidate(
            kind="l3", priority=priority, label=fact.label, line=line,
            evidence_ids=fact.evidence_ids, l1_evidence_ids=(),
            l3_evidence_ids=fact.evidence_ids, provenance=fact.provenance,
            confidence=fact.confidence, step_id=fact.last_step,
        )

    def _item(self, candidate: _Candidate, safe_line: str) -> LayerMemoryItem:
        layer_source = {
            "joint": "joint", "l3": "l3",
        }.get(candidate.kind, "l1")
        contribution = {
            "l1": candidate.kind in {"joint", "l1_current", "l1_history"},
            "l3": candidate.kind in {"joint", "l3"},
        }
        return LayerMemoryItem(
            layer=candidate.layer,
            kind=f"l1_l3_{candidate.kind}",
            description=candidate.line,
            step_id=candidate.step_id,
            confidence=candidate.confidence,
            source=self.source,
            evidence_refs=[
                _evidence_ref(candidate, evidence_id)
                for evidence_id in candidate.evidence_ids
            ],
            label=candidate.label,
            metadata={
                "layer_source": layer_source,
                "layer_contribution": contribution,
                "evidence_ids": list(candidate.evidence_ids),
                "l1_evidence_ids": list(candidate.l1_evidence_ids),
                "l3_evidence_ids": list(candidate.l3_evidence_ids),
                "serialized_text": safe_line,
                "provenance": list(candidate.provenance),
                "role": candidate.kind,
                "aligned": candidate.aligned,
                "traversability_status": "unknown",
            },
        )
