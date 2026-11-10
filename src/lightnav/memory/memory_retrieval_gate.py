"""Evidence-first filtering and reproducible priority for RGM layer results."""

from __future__ import annotations

from dataclasses import replace

from lightnav.memory.memory_write_gate import instruction_match
from lightnav.memory.schemas import LayerMemoryItem, MemoryQuery

_VISUAL_KINDS = {
    "current_observation", "recent_observation", "object_observation", "landmark",
    "semantic_summary", "place_semantics",
}


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
                elif item.kind in {"semantic_relation", "uncertain_relation", "observed_transition",
                                 "current_place", "last_recognized_place"}:
                    reason = "unverified_spatial_assertion"
                elif item.kind not in _VISUAL_KINDS:
                    reason = "not_navigation_landmark"
                elif not refs:
                    reason = "write_gate_rejected"
                elif not related:
                    reason = "instruction_irrelevant"
                elif item.kind == "recent_observation" and not historical:
                    reason = "stale_current_observation"
                if reason != "accepted":
                    self.decisions.append({
                        "layer": layer, "kind": item.kind, "step": item.step_id,
                        "evidence_ids": refs, "accepted": False, "reason": reason,
                    })
                    continue
                # Rank before deduplication so a weaker layer item cannot
                # consume evidence needed by a stronger independent record.
                priority = (
                    -int(related), -min(len(refs), 3), -int(historical),
                    -float(item.confidence), -int(item.metadata.get("current", False)),
                    -item.step_id, min(refs), layer,
                )
                eligible.append((priority, item, approved_refs))
        eligible.sort(key=lambda entry: entry[0])
        output: dict[str, list[LayerMemoryItem]] = {"short": [], "mid": [], "long": []}
        for _, item, refs in eligible:
            fresh = [ref for ref in refs if ref.event_id not in seen]
            reason = "accepted" if fresh else "duplicate_evidence"
            self.decisions.append({
                "layer": item.layer, "kind": item.kind, "step": item.step_id,
                "evidence_ids": [ref.event_id for ref in refs],
                "accepted": bool(fresh), "reason": reason,
            })
            if not fresh:
                continue
            seen.update(ref.event_id for ref in fresh)
            item = replace(item, evidence_refs=fresh)
            output[item.layer].append(item)
        return output
