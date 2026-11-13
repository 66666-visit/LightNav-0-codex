"""Evidence-first filtering and reproducible priority for RGM layer results."""

from __future__ import annotations

from dataclasses import replace

from lightnav.memory.memory_write_gate import instruction_match
from lightnav.memory.schemas import LayerMemoryItem, MemoryQuery

_VISUAL_KINDS = {
    "current_observation", "recent_observation", "object_observation", "landmark",
    "semantic_summary", "place_semantics",
}
_PLACE_KINDS = {"current_place", "last_recognized_place", "revisited_place"}
_SPATIAL_KINDS = _PLACE_KINDS | {"observed_transition"}


def spatial_evidence_reason(item: LayerMemoryItem, accepted_event_ids: set[str]) -> str | None:
    """Require adopted RGB descriptors before an uncertain spatial cue is used.

    A visual place or shift is never evidence of a traversable path. The caller
    may describe the observation, but must not turn it into a navigation fact.
    """
    metadata = item.metadata
    if item.kind == "place_semantics":
        descriptor_ids = metadata.get("descriptor_event_ids")
        if (
            not metadata.get("place_id") or not isinstance(descriptor_ids, list)
            or not any(isinstance(evidence_id, str) and evidence_id in accepted_event_ids
                       for evidence_id in descriptor_ids)
        ):
            return "unverified_spatial_assertion"
    elif item.kind in _PLACE_KINDS:
        place_id = metadata.get("place_id")
        if (
            not isinstance(place_id, str) or not place_id
            or not item.evidence_refs
            or any(ref.label != place_id or ref.event_id not in accepted_event_ids
                   for ref in item.evidence_refs)
        ):
            return "unverified_spatial_assertion"
        if item.kind == "revisited_place" and metadata.get("visit_count", 0) < 2:
            return "unverified_spatial_assertion"
    elif item.kind == "observed_transition":
        source = metadata.get("source_place_id")
        target = metadata.get("target_place_id")
        motion_ids = metadata.get("motion_event_ids")
        edge_ids = metadata.get("evidence_event_ids")
        if (
            not isinstance(source, str) or not source
            or not isinstance(target, str) or not target or source == target
            or len(item.evidence_refs) != 2
            or len({ref.event_id for ref in item.evidence_refs}) != 2
            or [ref.label for ref in item.evidence_refs] != [source, target]
            or any(ref.event_id not in accepted_event_ids for ref in item.evidence_refs)
            or not isinstance(motion_ids, list) or not motion_ids
            or not all(isinstance(evidence_id, str) and evidence_id for evidence_id in motion_ids)
            or not isinstance(edge_ids, list)
            or not set(motion_ids).issubset(edge_ids)
            or not {ref.event_id for ref in item.evidence_refs}.issubset(edge_ids)
        ):
            return "unverified_spatial_assertion"
    return None


class MemoryRetrievalGate:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.decisions: list[dict] = []

    def filter(
        self, query: MemoryQuery, layers: dict[str, list[LayerMemoryItem]],
        accepted_event_ids: set[str],
    ) -> dict[str, list[LayerMemoryItem]]:
        self.decisions = []
        if not self.enabled:
            return layers
        eligible: list[tuple[tuple, LayerMemoryItem, list]] = []
        seen: set[str] = set()
        for layer in ("short", "mid", "long"):
            for item in layers[layer]:
                raw_refs = [ref.event_id for ref in item.evidence_refs]
                approved_refs = [ref for ref in item.evidence_refs
                                 if ref.event_id in accepted_event_ids]
                refs = [ref.event_id for ref in approved_refs]
                reason = "accepted"
                related = bool(item.label and instruction_match(item.label, query.instruction))
                historical = item.step_id < query.current_step
                if not raw_refs:
                    reason = "missing_evidence"
                elif item.kind in {"semantic_relation", "uncertain_relation"}:
                    reason = "unverified_spatial_assertion"
                elif item.kind not in _VISUAL_KINDS | _SPATIAL_KINDS:
                    reason = "not_navigation_landmark"
                elif not refs:
                    reason = "write_gate_rejected"
                elif item.kind in _VISUAL_KINDS and not related:
                    reason = "instruction_irrelevant"
                elif item.kind in _SPATIAL_KINDS | {"place_semantics"}:
                    reason = spatial_evidence_reason(item, accepted_event_ids) or "accepted"
                elif item.kind == "recent_observation" and not historical:
                    reason = "stale_current_observation"
                if reason != "accepted":
                    self.decisions.append({
                        "layer": layer, "kind": item.kind, "step": item.step_id,
                        "evidence_ids": raw_refs, "approved_evidence_ids": refs,
                        "selected_evidence_ids": [], "accepted": False, "reason": reason,
                    })
                    continue
                # Rank *representations* of the same evidence before deduplication.
                # A validated current place/object relation is more informative
                # than its bare detection; a direct current RGB detection is
                # more faithful than an aggregate category summary. The spatial
                # evidence check above must pass before either can win.
                representation = (
                    2 if item.kind == "place_semantics"
                    and item.metadata.get("current", False)
                    and item.step_id == query.current_step
                    else 1 if item.kind in {
                        "current_observation", "object_observation", "landmark",
                    } and item.step_id == query.current_step else 0
                )
                priority = (
                    -int(related or item.kind in _SPATIAL_KINDS),
                    -representation,
                    -min(len(refs), 3), -int(historical),
                    -float(item.confidence), -int(item.kind == "current_observation"),
                    -int(item.metadata.get("current", False)),
                    -item.step_id, min(refs), layer,
                )
                eligible.append((priority, item, approved_refs))
        eligible.sort(key=lambda entry: entry[0])
        output: dict[str, list[LayerMemoryItem]] = {"short": [], "mid": [], "long": []}
        for _, item, refs in eligible:
            fresh = [ref for ref in refs if ref.event_id not in seen]
            # A descriptor pair is the evidence for one visual shift. Dropping
            # either side would turn a supported cue into a one-sided assertion.
            if item.kind == "observed_transition" and len(fresh) != len(refs):
                fresh = []
            reason = "accepted" if fresh else "duplicate_evidence"
            self.decisions.append({
                "layer": item.layer, "kind": item.kind, "step": item.step_id,
                "evidence_ids": [ref.event_id for ref in item.evidence_refs],
                "approved_evidence_ids": [ref.event_id for ref in refs],
                "selected_evidence_ids": [ref.event_id for ref in fresh],
                "accepted": bool(fresh), "reason": reason,
            })
            if not fresh:
                continue
            seen.update(ref.event_id for ref in fresh)
            item = replace(item, evidence_refs=fresh)
            output[item.layer].append(item)
        return output
