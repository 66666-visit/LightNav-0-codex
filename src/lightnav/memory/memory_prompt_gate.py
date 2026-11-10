"""Single-pass memory availability gate before the unchanged LightNav inference."""

from __future__ import annotations

from lightnav.memory.memory_write_gate import instruction_match
from lightnav.memory.schemas import MemoryContext, MemoryQuery, MemorySelection

_HEADER = "[Navigation memory; uncertain RGB observations]\n"


class MemoryPromptGate:
    def __init__(self, *, enabled: bool = True, max_records: int = 3, max_chars: int = 400,
                 step0_conditional: bool = True, min_current_confidence: float = 0.60) -> None:
        self.enabled = enabled
        self.max_records = max_records
        self.max_chars = max_chars
        self.step0_conditional = step0_conditional
        self.min_current_confidence = min_current_confidence
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
            elif item.kind not in {
                "current_observation", "recent_observation", "object_observation",
                "landmark", "semantic_summary", "place_semantics",
            }:
                reason = "unverified_or_nonvisual_memory"
            elif not item.label or not instruction_match(item.label, query.instruction):
                reason = "instruction_irrelevant"
            elif query.current_step == 0 and self.step0_conditional and (
                item.step_id != 0 or item.kind not in {"current_observation", "object_observation", "landmark"}
            ):
                reason = "step0_no_history"
            elif query.current_step == 0 and item.confidence < self.min_current_confidence:
                reason = "step0_low_confidence"
            elif item.step_id == query.current_step and query.current_step > 0 and not any(
                (support_by_event_id or {}).get(evidence_id, 0) >= 2 for evidence_id in ids
            ):
                reason = "current_view_duplicate"
            elif len(accepted) >= self.max_records:
                reason = "record_budget"
            label = " ".join(str(item.label or "").split())
            source = " ".join(str(item.source).split())
            if reason is None:
                when = "Current RGB" if item.step_id == query.current_step else "Previously"
                support = len({ref.step_id for ref in item.evidence_refs})
                line = f"{when} observed instruction landmark {label}; {support} distinct view(s); source={source}."
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
