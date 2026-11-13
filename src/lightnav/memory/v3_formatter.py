"""Rank-preserving, evidence-linked text for the frozen LightNav prompt."""

from __future__ import annotations

from html import escape

from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence
from lightnav.memory.v3_schemas import RetrievedMemory

_HEADER = "[Navigation memory; evidence-linked observations]"


def _clean(value: object) -> str:
    return " ".join(str(value).split())


def _line(record: RetrievedMemory) -> str:
    kind = record.kind
    outcome = record.outcome
    traversability = record.traversability_status
    if kind == "episodic_experience":
        status = "EPISODIC EXPERIENCE"
    elif kind == "action_history":
        status = "ACTION HISTORY"
    elif traversability in {"traversed", "verified"}:
        status = "TRAVERSED"
    elif record.spatial_status == "inferred" or kind in {"inferred_relation", "spatial_relation"}:
        status = "INFERRED"
    elif record.observation_status == "observed":
        status = "OBSERVED"
    else:
        status = "UNKNOWN"
    evidence = record.evidence_ids[-1]
    if len(record.evidence_ids) > 1:
        evidence += f" (+{len(record.evidence_ids) - 1} refs)"
    description = _clean(record.description)
    parts = [f"[{status}] {description}", f"Evidence: {evidence}"]
    if kind == "episodic_experience":
        parts.append(f"Outcome: {outcome}")
    if kind in {"inferred_relation", "spatial_relation", "observed_transition",
                "verified_traversal", "place"}:
        parts.append(f"Traversability: {traversability}")
    return "; ".join(parts) + "."


class V3PromptFormatter:
    """Format highest-ranked complete records within the existing RGM budget."""

    def __init__(self, *, max_records: int = 3, max_chars: int = 400) -> None:
        if max_records < 1 or max_chars < 1:
            raise ValueError("prompt budgets must be positive")
        self.max_records = max_records
        self.max_chars = max_chars
        self.last_selected = 0
        self.last_serialized = 0
        self.last_candidate_line_chars: list[int] = []
        self.last_serialized_line_chars: list[int] = []

    def format(self, records: tuple[RetrievedMemory, ...] | list[RetrievedMemory],
               *, current_step: int) -> MemoryContext:
        self.last_selected = min(len(records), self.max_records)
        self.last_serialized = 0
        self.last_candidate_line_chars = []
        self.last_serialized_line_chars = []
        lines: list[str] = []
        items: list[LayerMemoryItem] = []
        seen: set[str] = set()
        used = len(_HEADER)
        for record in records:
            if len(items) >= self.max_records:
                break
            if not record.evidence_ids or record.memory_id in seen:
                continue
            line = _line(record)
            # ContextBuilder escapes text once more before injection. Budget
            # the escaped form here so it cannot drop a selected record later.
            line_chars = len(escape(line, quote=False))
            self.last_candidate_line_chars.append(line_chars)
            extra = 1 + line_chars
            if used + extra > self.max_chars:
                continue
            seen.add(record.memory_id)
            used += extra
            lines.append(line)
            self.last_serialized_line_chars.append(line_chars)
            layer = (
                "mid" if record.kind in {"episodic_experience", "action_history"}
                else "long" if record.kind in {
                    "place", "inferred_relation", "spatial_relation",
                    "observed_transition", "verified_traversal",
                }
                else "short"
            )
            items.append(LayerMemoryItem(
                layer=layer, kind=f"v3_{record.kind}", description=record.description,
                step_id=record.source_step if record.source_step is not None else current_step,
                confidence=0.0, source="v3_retrieval",
                evidence_refs=[ObservationEvidence(
                    event_id=record.evidence_ids[-1],
                    step_id=record.source_step if record.source_step is not None else current_step,
                    label=record.kind, confidence=0.0, source="v3_evidence",
                )],
                metadata={
                    "memory_id": record.memory_id,
                    "supporting_evidence_ids": list(record.evidence_ids),
                    "channel_ranks": dict(record.channel_ranks),
                    "observation_status": record.observation_status,
                    "spatial_status": record.spatial_status,
                    "traversability_status": record.traversability_status,
                    "outcome": record.outcome,
                    "rrf_score": record.score,
                    "serialized_text": escape(line, quote=False),
                },
            ))
        self.last_serialized = len(lines)
        return MemoryContext(
            text=_HEADER + "\n" + "\n".join(lines) if lines else "",
            layer_items=items,
        )
