"""Joint ViL3DRel proposal -> persistent ObjectNode -> Evidence selection.

The official Student scores **every** proposal.  This module does not change,
re-rank by hand or threshold those official scores.  It adds the missing
identity mapping and chooses the highest-ranked *eligible* proposal instead of
the global argmax:

    proposal_id -> persistent_object_id -> evidence_ids

A proposal is eligible only when

1. its ``persistent_object_id`` is a real, observed ``ObjectNode``;
2. every Evidence ID on that node maps to a real observed Evidence record and
   to a recorded RGB sidecar point set;
3. the existing ``SpatialGraphAdapter`` produced a formatable geometry fact for
   that ObjectNode;
4. the node label matches the instruction landmark object (when the query is a
   landmark instead of the full instruction).

The global argmax can fail (1)-(4); the next ranked proposal is then used.  No
confidence threshold and no fixed top-k are introduced.  The caller applies the
existing Prompt character/Token budget to the final selected whole lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Iterable, Sequence

from lightnav.memory.l3_instance_grounder import InstanceGroundingResult
from lightnav.memory.schemas import LayerMemoryItem
from lightnav.memory.v3_retrieval import _terms

#: Category aliases for instance identity.  Only subtype/generic relations are
#: aliased; dining table and coffee table stay distinct.  Structural passage
#: synonyms are included because COCO/ScanNet-style detectors may emit
#: ``door``/``staircase`` while the instruction says ``doorway``/``stairs``;
#: these name the same instruction landmark, not a different object category.
_CATEGORY_ALIASES = {
    "armchair": "chair",
    "armchairs": "chair",
    "sofa": "couch",
    "sofas": "couch",
    "couches": "couch",
    "tv": "television",
    "tvs": "television",
    "televisions": "television",
    "doorway": "door",
    "doorways": "door",
    "staircase": "stair",
    "staircases": "stair",
    "stairway": "stair",
    "stairways": "stair",
}


def _singular(term: str) -> str:
    """Small, deterministic plural normalizer for label matching only."""
    if len(term) > 3 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 3 and term.endswith(("ses", "xes", "zes", "ches", "shes")):
        return term[:-2]
    if len(term) > 2 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def normalized_label_terms(value: object) -> set[str]:
    """Alias/plural-normalized terms for label identity comparisons."""
    return {
        _CATEGORY_ALIASES.get(term, term)
        for term in (_singular(term) for term in _terms(value))
    }


def _normalized_terms(value: object) -> set[str]:
    return normalized_label_terms(value)


def label_relevant_to_instruction(label: str, instruction: str) -> bool:
    """Alias-aware replacement for a raw term intersection in the adapter."""
    return bool(
        normalized_label_terms(label) & normalized_label_terms(instruction)
    )


def label_matches_target(label: str, target_object: str) -> bool:
    """Return whether a detector label denotes the instruction landmark.

    Matching is set containment on normalized terms in either direction, so
    ``window`` matches ``windows`` and ``door`` matches ``double doors`` while
    ``dining table`` does not match ``coffee table``.  It is an identity check,
    not a learned or tuned confidence score.
    """
    label_terms = _normalized_terms(label)
    target_terms = _normalized_terms(target_object)
    if not label_terms or not target_terms:
        return False
    return label_terms <= target_terms or target_terms <= label_terms


@dataclass(frozen=True)
class ProposalEligibility:
    """One official proposal with its persistent object and Evidence mapping."""

    proposal_id: str
    proposal_object_id: str
    persistent_object_id: str
    class_name: str
    evidence_ids: tuple[str, ...]
    grounding_logit: float | None
    grounding_probability: float | None
    grounding_rank: int
    global_grounding_rank: int
    eligible_grounding_rank: int | None
    object_node_valid: bool
    evidence_mapping_valid: bool
    formatable: bool
    class_match: bool
    already_selected: bool = False
    rejection_reason: str | None = None

    @property
    def eligible(self) -> bool:
        return (
            self.object_node_valid
            and self.evidence_mapping_valid
            and self.formatable
            and self.class_match
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_object_id": self.proposal_object_id,
            "persistent_object_id": self.persistent_object_id,
            "class_name": self.class_name,
            "evidence_ids": list(self.evidence_ids),
            "grounding_logit": self.grounding_logit,
            "grounding_probability": self.grounding_probability,
            "grounding_rank": self.grounding_rank,
            "global_grounding_rank": self.global_grounding_rank,
            "eligible_grounding_rank": self.eligible_grounding_rank,
            "object_node_valid": self.object_node_valid,
            "evidence_mapping_valid": self.evidence_mapping_valid,
            "formatable": self.formatable,
            "class_match": self.class_match,
            "already_selected": self.already_selected,
            "eligible": self.eligible,
            "rejection_reason": self.rejection_reason,
        }


def _rejection_reason(
    *, object_node_valid: bool, evidence_mapping_valid: bool,
    formatable: bool, class_match: bool,
) -> str | None:
    if not object_node_valid:
        return "missing_object_node"
    if not evidence_mapping_valid:
        return "missing_evidence_mapping"
    if not formatable:
        return "not_formatable"
    if not class_match:
        return "class_mismatch"
    return None


def build_proposal_eligibility(
    result: InstanceGroundingResult,
    *,
    object_map: Any,
    sidecar: Any,
    items_by_object: dict[str, LayerMemoryItem],
    target_object: str | None = None,
    selected_object_ids: Iterable[str] = (),
) -> list[ProposalEligibility]:
    """Rank every proposal by the official score and record its eligibility."""
    scores = list(result.proposals)
    scored = [
        score for score in scores
        if score.probability is not None and score.logit is not None
    ]
    scored.sort(
        key=lambda score: (
            -float(score.probability),
            -float(score.logit),
            score.object_id,
        )
    )
    already = set(selected_object_ids)
    nodes = getattr(object_map, "objects", {}) or {}
    records = getattr(object_map, "evidence", {}) or {}
    eligibility: list[ProposalEligibility] = []
    eligible_rank = 0
    for rank, score in enumerate(scored, start=1):
        node = nodes.get(score.object_id)
        evidence_ids = tuple(
            node.evidence_ids if node is not None else getattr(score, "evidence_ids", ())
        )
        object_node_valid = bool(
            node is not None
            and getattr(node, "geometry_status", None) == "observed"
            and evidence_ids
        )
        evidence_mapping_valid = bool(object_node_valid) and all(
            (record := records.get(evidence_id)) is not None
            and getattr(record, "geometry_status", None) == "observed"
            and sidecar.has_evidence(evidence_id)
            for evidence_id in evidence_ids
        )
        item = items_by_object.get(score.object_id)
        formatable = bool(
            item is not None
            and item.metadata.get("evidence_ids")
        )
        class_match = (
            True if target_object is None
            else label_matches_target(score.class_name, target_object)
        )
        if (
            object_node_valid and evidence_mapping_valid
            and formatable and class_match
        ):
            eligible_rank += 1
            this_eligible_rank = eligible_rank
        else:
            this_eligible_rank = None
        eligibility.append(ProposalEligibility(
            proposal_id=getattr(score, "proposal_id", "") or f"proposal-{rank}",
            proposal_object_id=score.object_id,
            persistent_object_id=score.object_id,
            class_name=score.class_name,
            evidence_ids=evidence_ids,
            grounding_logit=(
                float(score.logit) if score.logit is not None else None
            ),
            grounding_probability=(
                float(score.probability)
                if score.probability is not None else None
            ),
            grounding_rank=rank,
            global_grounding_rank=rank,
            eligible_grounding_rank=this_eligible_rank,
            object_node_valid=object_node_valid,
            evidence_mapping_valid=evidence_mapping_valid,
            formatable=formatable,
            class_match=class_match,
            already_selected=score.object_id in already,
            rejection_reason=_rejection_reason(
                object_node_valid=object_node_valid,
                evidence_mapping_valid=evidence_mapping_valid,
                formatable=formatable,
                class_match=class_match,
            ),
        ))
    return eligibility


def eligible_object_ids(
    eligibility: Sequence[ProposalEligibility],
) -> list[str]:
    """All eligible proposals in official score order (duplicates removed)."""
    return list(dict.fromkeys(
        proposal.persistent_object_id
        for proposal in eligibility if proposal.eligible
    ))


def choose_first_eligible(
    eligibility: Sequence[ProposalEligibility],
    *, selected_object_ids: Iterable[str] = (),
) -> ProposalEligibility | None:
    """Choose the highest-ranked eligible proposal not already injected."""
    selected = set(selected_object_ids)
    for proposal in eligibility:
        if proposal.eligible and proposal.persistent_object_id not in selected:
            return proposal
    return None


@dataclass(frozen=True)
class GeometryFallbackResult:
    """Explicit outcome of the same-category Geometry On-demand fallback.

    ``reason`` is populated even when fallback is disabled or no candidate
    exists, so a diagnostic can distinguish "not attempted" from "attempted but
    no same-category item" from "item selected".
    """

    item: LayerMemoryItem | None
    attempted: bool
    reason: str | None
    same_category_item_count: int = 0
    available_item_count: int = 0

    @property
    def selected(self) -> bool:
        return self.item is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "reason": self.reason,
            "selected_object_id": (
                self.item.metadata.get("object_id")
                if self.item is not None else None
            ),
            "same_category_item_count": self.same_category_item_count,
            "available_item_count": self.available_item_count,
        }


def resolve_geometry_fallback(
    items: Sequence[LayerMemoryItem],
    *,
    target_object: str | None,
    selected_object_ids: Iterable[str] = (),
    enabled: bool = True,
) -> GeometryFallbackResult:
    """Choose the original adapter item for one landmark, with an explicit reason.

    This is still a *selection* over the unchanged ``SpatialGraphAdapter.build``
    output.  It never re-runs geometry, changes the adapter order, or relaxes
    the category check.  A cross-category item is never returned.
    """
    if not enabled:
        return GeometryFallbackResult(None, False, "fallback_disabled")
    if target_object is None:
        return GeometryFallbackResult(None, False, "no_landmark_target")
    if not items:
        return GeometryFallbackResult(None, True, "no_formatable_geometry_item")
    matching = [
        item for item in items
        if label_matches_target(item.label, target_object)
    ]
    if not matching:
        return GeometryFallbackResult(
            None, True, "no_same_category_geometry_item",
            same_category_item_count=0,
        )
    selected = set(selected_object_ids)
    available = [
        item for item in matching
        if isinstance(item.metadata.get("object_id"), str)
        and item.metadata.get("object_id") not in selected
    ]
    if not available:
        return GeometryFallbackResult(
            None, True, "same_category_already_injected",
            same_category_item_count=len(matching),
            available_item_count=0,
        )
    return GeometryFallbackResult(
        available[0], True, None,
        same_category_item_count=len(matching),
        available_item_count=len(available),
    )


def geometry_fallback_item(
    items: Sequence[LayerMemoryItem],
    *,
    target_object: str | None,
    selected_object_ids: Iterable[str] = (),
) -> LayerMemoryItem | None:
    """Reuse an existing formatable geometry fact when grounding has no result.

    ``items`` is the unchanged ``SpatialGraphAdapter.build`` output, already
    ordered by the original adapter.  This only chooses one of those existing
    facts; it never re-runs or re-implements geometry retrieval.

    ``target_object=None`` keeps the original full-instruction behaviour: the
    first unselected adapter item is used.  Landmark queries always pass a
    target object, so they can never cross categories by accident.
    """
    if target_object is None:
        selected = set(selected_object_ids)
        for item in items:
            object_id = item.metadata.get("object_id")
            if isinstance(object_id, str) and object_id not in selected:
                return item
        return None
    return resolve_geometry_fallback(
        items,
        target_object=target_object,
        selected_object_ids=selected_object_ids,
    ).item


def render_context(
    items: Sequence[LayerMemoryItem], *, header: str, max_chars: int,
) -> tuple[str | None, list[LayerMemoryItem]]:
    """Render whole existing fact lines within the existing character cap."""
    if not items or max_chars < 1:
        return None, []
    used = len(header)
    lines: list[str] = []
    included: list[LayerMemoryItem] = []
    for item in items:
        line = item.description
        if not isinstance(line, str) or not line.strip():
            continue
        safe_line = escape(line, quote=False)
        if used + len(safe_line) + 1 > max_chars:
            continue
        used += len(safe_line) + 1
        lines.append(line)
        included.append(item)
    if not included:
        return None, []
    return header + "\n" + "\n".join(lines), included


__all__ = [
    "GeometryFallbackResult",
    "ProposalEligibility",
    "build_proposal_eligibility",
    "choose_first_eligible",
    "eligible_object_ids",
    "geometry_fallback_item",
    "label_matches_target",
    "label_relevant_to_instruction",
    "normalized_label_terms",
    "render_context",
    "resolve_geometry_fallback",
]
