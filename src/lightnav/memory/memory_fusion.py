"""Fuse independently retrieved episode memories without repeating evidence."""

from __future__ import annotations

from dataclasses import replace

from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, MemoryQuery, ObservationEvidence

_HEADER = "Episode observations (may be uncertain):\n"


def _clean(value: object) -> str:
    return " ".join(str(value).split())


def _evidence_key(ref: ObservationEvidence) -> tuple[str, str, int]:
    # A node id identifies one detection in the YOLO writer. The step also
    # distinguishes a later observation of an explicitly tracked graph node.
    if ref.node_id:
        return ("node", ref.node_id, ref.step_id)
    return ("event", ref.event_id, ref.step_id)


def _evidence_text(refs: list[ObservationEvidence]) -> str:
    def identifier(ref: ObservationEvidence) -> str:
        suffix = f"/{_clean(ref.node_id)}" if ref.node_id else ""
        return f"{_clean(ref.event_id)}@{ref.step_id}{suffix}"

    first = refs[0]
    if len(refs) == 1:
        return identifier(first)
    last = refs[-1]
    return f"{identifier(first)}..{identifier(last)}; count={len(refs)}"


class MemoryFusion:
    """Combine layer results while retaining every distinct observation record."""

    def build(
        self,
        query: MemoryQuery,
        short_items: list[LayerMemoryItem],
        mid_items: list[LayerMemoryItem],
        long_items: list[LayerMemoryItem],
    ) -> MemoryContext:
        if query.top_k <= 0 or query.max_chars <= len(_HEADER):
            return MemoryContext()

        current = [item for item in short_items if item.kind == "current_observation"]
        other_short = [item for item in short_items if item.kind != "current_observation"]
        selected: list[LayerMemoryItem] = []
        lines: list[str] = []
        seen: set[tuple[str, str, int]] = set()
        used = len(_HEADER)

        def add(item: LayerMemoryItem) -> bool:
            nonlocal used
            if len(lines) >= query.top_k or not item.evidence_refs:
                return False
            refs = [ref for ref in item.evidence_refs if _evidence_key(ref) not in seen]
            if not refs:
                return False
            if item.kind == "semantic_summary":
                # An aggregate can partly overlap an L1/L2 observation. Retain
                # its unseen evidence instead of dropping an earlier sighting.
                steps = sorted({ref.step_id for ref in refs})
                label = item.label or refs[0].label
                confidence = sum(ref.confidence for ref in refs) / len(refs)
                strongest = max(ref.confidence for ref in refs)
                description = (
                    f"Category {_clean(label)} observed {len(refs)} time(s), "
                    f"first step={steps[0]}, latest step={steps[-1]}, "
                    f"observation steps={','.join(str(step) for step in steps)}, "
                    f"mean confidence={confidence:.2f}, max confidence={strongest:.2f}; "
                    "physical instance and room identity unknown"
                )
                source = ",".join(sorted({ref.source for ref in refs}))
                item = replace(
                    item, description=description, step_id=steps[-1],
                    confidence=confidence, source=source, evidence_refs=refs,
                )
            elif len(refs) != len(item.evidence_refs):
                item = replace(item, evidence_refs=refs)

            line = (
                f"[{_clean(item.layer)}:{_clean(item.kind)} step={item.step_id} "
                f"confidence={item.confidence:.2f} source={_clean(item.source)} "
                f"evidence={_evidence_text(refs)}] {_clean(item.description)}"
            )
            extra = len(line) + (1 if lines else 0)
            if used + extra > query.max_chars:
                return False
            lines.append(line)
            selected.append(item)
            used += extra
            seen.update(_evidence_key(ref) for ref in refs)
            return True

        # Keep live detections prominent, but leave room for independently
        # retrieved middle and long memories when their evidence is distinct.
        reserved = min(max(query.top_k - 1, 0), bool(mid_items) + bool(long_items))
        for item in current[: max(1, query.top_k - reserved)]:
            add(item)
        for layer in (mid_items, long_items):
            for item in layer:
                if add(item):
                    break
        for item in current[max(1, query.top_k - reserved):]:
            add(item)
        # Each layer has already ranked its own results. Fill remaining slots
        # round-robin so one long event list cannot monopolize the prompt.
        queues = [other_short, mid_items, long_items]
        for index in range(max(map(len, queues), default=0)):
            for queue in queues:
                if index < len(queue):
                    add(queue[index])

        return MemoryContext(
            text=_HEADER + "\n".join(lines) if lines else "",
            layer_items=selected,
        )
