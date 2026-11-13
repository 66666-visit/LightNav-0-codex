"""Single-pass memory availability gate before the unchanged LightNav inference."""

from __future__ import annotations

from lightnav.memory.memory_retrieval_gate import spatial_evidence_reason
from lightnav.memory.memory_write_gate import instruction_match
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, MemoryQuery, MemorySelection

_HEADER = "[Navigation memory; uncertain RGB observations]\n"
_CATEGORY_KINDS = {
    "current_observation", "recent_observation", "object_observation",
    "landmark", "semantic_summary", "place_semantics",
}
_SPATIAL_KINDS = {
    "current_place", "last_recognized_place", "revisited_place", "observed_transition",
}


def _clean(value: object) -> str:
    return " ".join(str(value).split())


def _record_line(item: LayerMemoryItem, query: MemoryQuery,
                 support_by_event_id: dict[str, int]) -> str:
    """Keep one complete, explicitly uncertain line per accepted record."""
    source = _clean(item.source)
    if item.kind == "place_semantics":
        when = "Current" if item.step_id == query.current_step else "Earlier"
        return (
            f"{when} RGB detected instruction landmark {_clean(item.label)} while visual place "
            f"hypothesis {_clean(item.metadata['place_id'])} was active; place and object identity "
            f"uncertain; source={source}."
        )
    if item.kind == "current_place":
        return (
            f"Current RGB descriptor suggests visual place {_clean(item.metadata['place_id'])}; "
            f"room identity unverified; source={source}."
        )
    if item.kind == "last_recognized_place":
        return (
            f"Last RGB descriptor at step {item.step_id} suggested visual place "
            f"{_clean(item.metadata['place_id'])}; current place unverified; source={source}."
        )
    if item.kind == "revisited_place":
        return (
            f"RGB descriptor resembled previously seen visual place "
            f"{_clean(item.metadata['place_id'])}; room identity unverified; source={source}."
        )
    if item.kind == "observed_transition":
        return (
            f"RGB descriptors suggest a visual shift from {_clean(item.metadata['source_place_id'])} "
            f"to {_clean(item.metadata['target_place_id'])} after an issued motion command; "
            f"traversability unverified; source={source}."
        )
    when = "Current RGB" if item.step_id == query.current_step else "Previously"
    support = max([
        len({ref.step_id for ref in item.evidence_refs}),
        *(support_by_event_id.get(ref.event_id, 0) for ref in item.evidence_refs),
    ])
    return (
        f"{when} observed instruction landmark {_clean(item.label)}; "
        f"{support} distinct category view(s); source={source}."
    )


class MemoryPromptGate:
    def __init__(self, *, enabled: bool = True, max_records: int = 3, max_chars: int = 400,
                 step0_conditional: bool = True, min_current_confidence: float = 0.60,
                 temporal_min_support: int = 2) -> None:
        if temporal_min_support < 1:
            raise ValueError("temporal_min_support must be positive")
        self.enabled = enabled
        self.max_records = max_records
        self.max_chars = max_chars
        self.step0_conditional = step0_conditional
        self.min_current_confidence = min_current_confidence
        self.temporal_min_support = temporal_min_support
        self.last_selection = MemorySelection((), (), (), "")

    def select(self, query: MemoryQuery, context: MemoryContext,
               accepted_event_ids: set[str] | None = None,
               support_by_event_id: dict[str, int] | None = None) -> MemoryContext:
        if not self.enabled:
            self.last_selection = MemorySelection(tuple(context.layer_items), (),
                                                  ("prompt_gate_disabled",), context.text)
            return context
        accepted = []
        rejected = []
        lines = []
        used = len(_HEADER)
        for item in context.layer_items:
            ids = tuple(dict.fromkeys(ref.event_id for ref in item.evidence_refs))
            reason = None
            if not ids or not item.source or item.source == "none":
                reason = "missing_provenance"
            elif accepted_event_ids is not None and not all(
                evidence_id in accepted_event_ids for evidence_id in ids
            ):
                reason = "write_gate_rejected"
            elif item.kind not in _CATEGORY_KINDS | _SPATIAL_KINDS:
                reason = "unverified_or_nonvisual_memory"
            elif item.kind in _CATEGORY_KINDS and (
                not item.label or not instruction_match(item.label, query.instruction)
            ):
                reason = "instruction_irrelevant"
            elif item.kind in _SPATIAL_KINDS | {"place_semantics"}:
                reason = spatial_evidence_reason(item, accepted_event_ids or set())
            if reason is None and query.current_step == 0 and self.step0_conditional and (
                item.step_id != 0 or item.kind not in {
                    "current_observation", "object_observation", "landmark", "place_semantics",
                }
            ):
                reason = "step0_no_history"
            elif reason is None and query.current_step == 0 and item.confidence < self.min_current_confidence:
                reason = "step0_low_confidence"
            elif reason is None and item.kind in _CATEGORY_KINDS - {"place_semantics"} and (
                item.step_id == query.current_step and query.current_step > 0
            ) and not any(
                (support_by_event_id or {}).get(evidence_id, 0) >= self.temporal_min_support
                for evidence_id in ids
            ):
                reason = "current_view_duplicate"
            elif reason is None and len(accepted) >= self.max_records:
                reason = "record_budget"
            if reason is None:
                line = _record_line(item, query, support_by_event_id or {})
                extra = len(line) + (1 if lines else 0)
                if used + extra > self.max_chars:
                    reason = "character_budget"
            if reason is not None:
                rejected.append((ids[0] if ids else "", reason))
                continue
            lines.append(line)
            accepted.append(item)
            used += extra
        text = _HEADER + "\n".join(lines) if lines else ""
        self.last_selection = MemorySelection(
            tuple(accepted), tuple(rejected), tuple(reason for _, reason in rejected), text,
        )
        return MemoryContext(text=text, layer_items=list(accepted))
