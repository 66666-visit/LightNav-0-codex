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

The historical ``build`` path serializes organized L1 facts. The supplemental
``build_preserving_l1`` path accepts the original Local Graph ``MemoryContext``
and reuses the same alignment/conflict/dedup rules without rewriting L1 text;
a confirmed L3 fact becomes a separate linked map line.
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
    aligned_pairs: int = 0
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
            "aligned_pairs": self.aligned_pairs,
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


@dataclass(frozen=True)
class _L1Ref:
    """One L1 item in the form entity alignment needs.

    ``original_item`` is set only on the lossless path. When it is present the
    serializer copies that item instead of rebuilding its text from metadata.
    """

    label: str
    role: str
    evidence_ids: tuple[str, ...]
    provenance: tuple[dict[str, object], ...]
    confidence: float
    step_id: int
    line: str
    layer: str
    source: str
    sightings: int = 1
    original_item: LayerMemoryItem | None = None

    def signature(self) -> tuple[str, str]:
        return self.role, self.label.casefold()


@dataclass(frozen=True)
class _L3Match:
    fact: L3Fact
    status: str  # confirmed | identity_conflict | uncertain_* | unmatched
    partner: _L1Ref | None = None
    same_label: tuple[_L1Ref, ...] = ()


@dataclass
class _Candidate:
    kind: str  # l1_current | l1_history | l3 | l3_aligned | joint
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
    l1_signature: tuple[str, str] | None = None
    object_id: str | None = None
    aligned_object_ids: tuple[str, ...] = ()
    original_item: LayerMemoryItem | None = None


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
        """Fuse the organized L1 card with L3 (unchanged historical behavior)."""
        self._validate_arguments(step=step, max_chars=max_chars)
        refs = [self._l1_ref_from_fact(fact) for fact in l1.facts]
        organization, consumed, matches, notes = self._align(refs, l3.facts)
        candidates: list[_Candidate] = []

        for match in matches:
            fact = match.fact
            if match.status == "identity_conflict":
                assert match.partner is not None
                candidates.append(self._l1_candidate(
                    match.partner,
                    priority=1 if match.partner.role == "current" else 4,
                ))
                candidates.append(self._l3_candidate(fact, priority=5))
            elif match.status == "confirmed":
                assert match.partner is not None
                organization.merged_entities += 1
                candidates.append(self._joint_candidate(match.partner, fact))
            elif match.status == "uncertain_verified":
                candidates.append(self._l3_candidate(fact, priority=3))
            elif match.status == "uncertain_unverified":
                candidates.append(self._l3_candidate(fact, priority=6))
            else:
                assert match.status == "unmatched"
                candidates.append(self._l3_candidate(
                    fact, priority=3 if fact.geometry_verified else 6,
                ))

        for ref in refs:
            if ref.signature() in consumed:
                continue
            candidates.append(self._l1_candidate(
                ref, priority=1 if ref.role == "current" else 4,
            ))

        return self._serialize(
            organization, candidates, step=step, max_chars=max_chars, notes=notes,
        )

    def build_preserving_l1(
        self,
        l1: MemoryContext | None,
        l3: L3Organization,
        *,
        step: int,
        max_chars: int,
    ) -> JointOrganization:
        """Fuse original L1 lines with organized L3 without rewriting L1.

        The existing alignment, conflict and duplicate rules are reused. A
        confirmed pair keeps the exact original L1 line and emits the linked L3
        map fact as a separate ``Map`` line, because the historical merged line
        is synthesized from L1 metadata and would not preserve the original
        text. Unlinked and conflicting facts keep both source lines.
        """
        self._validate_arguments(step=step, max_chars=max_chars)
        refs = [
            self._l1_ref_from_item(item)
            for item in (l1.layer_items if l1 is not None else [])
        ]
        organization, consumed, matches, notes = self._align(refs, l3.facts)
        candidates: list[_Candidate] = []

        for match in matches:
            fact = match.fact
            if match.status == "identity_conflict":
                assert match.partner is not None
                candidates.append(self._original_l1_candidate(
                    match.partner,
                    priority=0 if match.partner.role == "current" else 2,
                ))
                candidates.append(self._l3_candidate(fact, priority=5))
            elif match.status == "confirmed":
                assert match.partner is not None
                candidates.append(self._original_l1_candidate(
                    match.partner,
                    priority=0 if match.partner.role == "current" else 2,
                    aligned_object_id=fact.object_id,
                ))
                candidates.append(self._l3_candidate(
                    fact,
                    priority=1 if match.partner.role == "current" else 3,
                    aligned_object_id=fact.object_id,
                    aligned_l1_evidence_ids=match.partner.evidence_ids,
                    aligned_l1_signature=match.partner.signature(),
                ))
            elif match.status == "uncertain_verified":
                candidates.append(self._l3_candidate(fact, priority=3))
            elif match.status == "uncertain_unverified":
                candidates.append(self._l3_candidate(fact, priority=6))
            else:
                assert match.status == "unmatched"
                candidates.append(self._l3_candidate(
                    fact, priority=3 if fact.geometry_verified else 6,
                ))

        for ref in refs:
            if ref.signature() in consumed:
                continue
            # All remaining L1 lines keep priority over generic L3 history.
            candidates.append(self._original_l1_candidate(
                ref, priority=0 if ref.role == "current" else 2,
            ))

        return self._serialize(
            organization, candidates, step=step, max_chars=max_chars, notes=notes,
        )

    @staticmethod
    def _validate_arguments(*, step: int, max_chars: int) -> None:
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if type(max_chars) is not int or max_chars < 1:
            raise ValueError("max_chars must be a positive integer")

    @staticmethod
    def _l1_ref_from_fact(fact: L1Fact) -> _L1Ref:
        return _L1Ref(
            label=fact.label, role=fact.role, evidence_ids=fact.evidence_ids,
            provenance=fact.provenance, confidence=fact.confidence,
            step_id=fact.last_step, line=fact.line(),
            layer="short" if fact.role == "current" else "mid",
            source="l1_prompt_organizer", sightings=fact.sightings,
        )

    @staticmethod
    def _l1_ref_from_item(item: LayerMemoryItem) -> _L1Ref:
        role = item.metadata.get("role")
        if role not in {"current", "history"}:
            role = "current" if item.layer == "short" else "history"
        evidence_ids = tuple(ref.event_id for ref in item.evidence_refs)
        if not evidence_ids:
            evidence_ids = tuple(item.metadata.get("evidence_ids", ()))
        provenance = tuple(item.metadata.get("provenance", ()))
        try:
            sightings = int(item.metadata.get("sightings", 1) or 1)
        except (TypeError, ValueError):
            sightings = 1
        return _L1Ref(
            label=item.label or "", role=str(role), evidence_ids=evidence_ids,
            provenance=provenance, confidence=float(item.confidence),
            step_id=item.step_id, line=item.description, layer=item.layer,
            source=item.source, sightings=max(1, sightings), original_item=item,
        )

    def _align(
        self, l1_refs: list[_L1Ref], l3_facts: list[L3Fact],
    ) -> tuple[JointOrganization, set[tuple[str, str]], list[_L3Match], list[str]]:
        """Apply the historical trusted-link alignment rules to either L1 form."""
        organization = JointOrganization(context=None)
        l1_by_label: dict[str, list[_L1Ref]] = {}
        for ref in l1_refs:
            l1_by_label.setdefault(ref.label.casefold(), []).append(ref)
        consumed: set[tuple[str, str]] = set()
        matches: list[_L3Match] = []
        notes: list[str] = []

        for fact in l3_facts:
            linked = set(fact.linked_l1_evidence_ids)
            linked_matches = [
                item for item in l1_refs if linked & set(item.evidence_ids)
            ]
            same_label = l1_by_label.get(fact.label.casefold(), [])
            if linked_matches:
                partner = linked_matches[0]
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
                    matches.append(_L3Match(
                        fact, "identity_conflict", partner, tuple(same_label),
                    ))
                    continue
                consumed.add(partner.signature())
                organization.alignments.append(EntityAlignment(
                    label=fact.label, status="confirmed",
                    l3_object_id=fact.object_id,
                    l1_evidence_ids=partner.evidence_ids,
                    l3_evidence_ids=fact.evidence_ids,
                    reason="l3_linked_l1_evidence",
                ))
                matches.append(_L3Match(
                    fact, "confirmed", partner, tuple(same_label),
                ))
                continue

            if same_label:
                organization.alignments.append(EntityAlignment(
                    label=fact.label, status="uncertain_same_label",
                    l3_object_id=fact.object_id,
                    l1_evidence_ids=tuple(
                        evidence_id for item in same_label
                        for evidence_id in item.evidence_ids
                    ),
                    l3_evidence_ids=fact.evidence_ids,
                    reason="same_class_without_trusted_link",
                ))
                notes.append(
                    f"Uncertain: {fact.label} may be the same object in L1 and L3 "
                    "(unconfirmed)."
                )
                if fact.geometry_verified:
                    matches.append(_L3Match(
                        fact, "uncertain_verified", None, tuple(same_label),
                    ))
                else:
                    organization.conflicts.append({
                        "type": "geometry_unverified", "label": fact.label,
                        "l1_evidence_ids": list(same_label[0].evidence_ids),
                        "l3_evidence_ids": list(fact.evidence_ids),
                        "resolution": "both_kept_position_unknown",
                    })
                    matches.append(_L3Match(
                        fact, "uncertain_unverified", None, tuple(same_label),
                    ))
                continue

            organization.alignments.append(EntityAlignment(
                label=fact.label, status="unmatched", l3_object_id=fact.object_id,
                l1_evidence_ids=(), l3_evidence_ids=fact.evidence_ids,
                reason="no_l1_counterpart",
            ))
            matches.append(_L3Match(fact, "unmatched"))
        return organization, consumed, matches, notes

    def _serialize(
        self, organization: JointOrganization, candidates: list[_Candidate], *,
        step: int, max_chars: int, notes: list[str],
    ) -> JointOrganization:
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
            elif candidate.kind in {"l3", "l3_aligned"}:
                organization.l3_item_count += 1
            else:
                organization.l1_item_count += 1
            if candidate.kind in {"l3", "l3_aligned", "joint"}:
                organization.unverified_relations += 1

        selected_l1_signatures = {
            candidate.l1_signature for candidate in selected
            if candidate.kind in {"l1_current", "l1_history"}
            and candidate.l1_signature is not None
        }
        organization.aligned_pairs = sum(
            1 for candidate in selected
            if candidate.kind == "l3_aligned"
            and candidate.l1_signature in selected_l1_signatures
        )

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

    def _joint_candidate(self, l1: _L1Ref, l3: L3Fact) -> _Candidate:
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
            line += f" L1 last step {l1.step_id}."
        provenance = (*l1.provenance, *l3.provenance)
        return _Candidate(
            kind="joint", priority=0 if l1.role == "current" else 2,
            label=l1.label, line=line,
            evidence_ids=tuple(dict.fromkeys((*l1.evidence_ids, *l3.evidence_ids))),
            l1_evidence_ids=l1.evidence_ids, l3_evidence_ids=l3.evidence_ids,
            provenance=provenance, confidence=l1.confidence, step_id=l1.step_id,
            layer=l1.layer, aligned=True, l1_signature=l1.signature(),
            object_id=l3.object_id,
        )

    def _l1_candidate(
        self, ref: _L1Ref, *, priority: int, original_item: LayerMemoryItem | None = None,
        aligned_object_ids: tuple[str, ...] = (),
    ) -> _Candidate:
        return _Candidate(
            kind="l1_current" if ref.role == "current" else "l1_history",
            priority=priority, label=ref.label, line=ref.line,
            evidence_ids=ref.evidence_ids, l1_evidence_ids=ref.evidence_ids,
            l3_evidence_ids=(), provenance=ref.provenance,
            confidence=ref.confidence, step_id=ref.step_id, layer=ref.layer,
            l1_signature=ref.signature(), aligned_object_ids=aligned_object_ids,
            original_item=(
                original_item if original_item is not None else ref.original_item
            ),
        )

    def _original_l1_candidate(
        self, ref: _L1Ref, *, priority: int,
        aligned_object_id: str | None = None,
    ) -> _Candidate:
        if ref.original_item is None:
            raise ValueError("lossless joint mode requires original L1 layer items")
        return self._l1_candidate(
            ref, priority=priority, original_item=ref.original_item,
            aligned_object_ids=(
                (aligned_object_id,) if aligned_object_id is not None else ()
            ),
        )

    def _l3_candidate(
        self, fact: L3Fact, *, priority: int, aligned_object_id: str | None = None,
        aligned_l1_evidence_ids: tuple[str, ...] = (),
        aligned_l1_signature: tuple[str, str] | None = None,
    ) -> _Candidate:
        line = fact.line()
        if not fact.geometry_verified:
            line = (
                f"{fact.label}: map evidence unverified; position and passage unknown."
            )
        return _Candidate(
            kind="l3_aligned" if aligned_object_id is not None else "l3",
            priority=priority, label=fact.label, line=line,
            evidence_ids=fact.evidence_ids,
            l1_evidence_ids=aligned_l1_evidence_ids,
            l3_evidence_ids=fact.evidence_ids, provenance=fact.provenance,
            confidence=fact.confidence, step_id=fact.last_step,
            aligned=aligned_object_id is not None,
            l1_signature=aligned_l1_signature, object_id=aligned_object_id,
        )

    def _item(self, candidate: _Candidate, safe_line: str) -> LayerMemoryItem:
        if candidate.original_item is not None:
            original = candidate.original_item
            metadata = dict(original.metadata)
            metadata.update({
                "layer_source": "l1_original",
                "layer_contribution": {"l1": True, "l3": False},
                "evidence_ids": list(candidate.evidence_ids),
                "l1_evidence_ids": list(candidate.l1_evidence_ids),
                "l3_evidence_ids": [],
                "serialized_text": safe_line,
                "role": candidate.kind.removeprefix("l1_"),
                "l1_prompt_preserved": True,
                "original_layer_item_kind": original.kind,
            })
            if candidate.aligned_object_ids:
                metadata["aligned_l3_object_ids"] = list(
                    candidate.aligned_object_ids
                )
            return LayerMemoryItem(
                layer=original.layer,
                kind=original.kind,
                description=original.description,
                step_id=original.step_id,
                confidence=original.confidence,
                source=original.source,
                evidence_refs=list(original.evidence_refs),
                label=original.label or candidate.label,
                metadata=metadata,
            )
        layer_source = {
            "joint": "joint", "l3": "l3", "l3_aligned": "l3",
        }.get(candidate.kind, "l1")
        contribution = {
            "l1": candidate.kind in {"joint", "l1_current", "l1_history"},
            "l3": candidate.kind in {"joint", "l3", "l3_aligned"},
        }
        metadata = {
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
        }
        if candidate.kind == "l3_aligned":
            metadata["aligned_l3_object_id"] = candidate.object_id
            metadata["aligned_l1_evidence_ids"] = list(candidate.l1_evidence_ids)
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
            metadata=metadata,
        )
