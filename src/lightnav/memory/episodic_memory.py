"""Episode-local event journal and independent mid-term key-event retrieval.

The complete event log remains intact. A separate index and temporal segments
make important observations retrievable after many ordinary commands without
claiming that similar labels refer to one physical object or place.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from lightnav.memory.schemas import LayerMemoryItem, MemoryEvent, MemoryQuery, ObservationEvidence

_KEY_KINDS = frozenset({
    "object_observation", "landmark", "instruction_match", "turn", "visual_change",
    "place_visit", "place_hypothesis", "possible_transition",
})
_VISUAL_KINDS = frozenset({"object_observation", "landmark", "instruction_match"})
_STOP_WORDS = frozenset({
    "a", "an", "and", "at", "by", "for", "from", "in", "into", "of", "on", "or",
    "the", "then", "to", "until", "go", "walk", "move", "continue", "find",
})
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def _terms(value: str) -> set[str]:
    words: set[str] = set()
    for match in _WORDS.finditer(value.casefold()):
        token = match.group()
        if token not in _STOP_WORDS:
            words.add(token)
            if len(token) > 4 and token.endswith("s"):
                words.add(token[:-1])
    return words


def _event_label(event: MemoryEvent) -> str:
    """Extract a class label for indexed observations without inventing identity."""
    if event.kind in _VISUAL_KINDS:
        description = event.description.strip()
        for prefix in ("Observed instruction-related ", "Observed "):
            if description.startswith(prefix):
                return description[len(prefix):].strip()
        return description
    if event.kind == "place_visit" and event.description.startswith("Extractor reported "):
        return event.description.removeprefix("Extractor reported ").split(" (", 1)[0].strip()
    if event.kind == "place_hypothesis" and event.description.startswith(
        "RGB extractor described "
    ):
        return event.description.removeprefix("RGB extractor described ").split(";", 1)[0].strip()
    return event.kind


def _node_id(event: MemoryEvent) -> str | None:
    value = event.metadata.get("node_id") if isinstance(event.metadata, dict) else None
    return value if isinstance(value, str) and value else None


@dataclass
class EpisodicSegment:
    """Adjacent event sightings, not an inferred object track or place visit."""

    kind: str
    label: str
    start_step: int
    end_step: int
    event_ids: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    max_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodicSegment:
        return cls(**data)


class EpisodicMemory:
    def __init__(self, *, index_enabled: bool = True) -> None:
        self.index_enabled = bool(index_enabled)
        self._events: list[MemoryEvent] = []
        self._ids: set[str] = set()
        self._key_events: list[MemoryEvent] = []
        self._segments: list[EpisodicSegment] = []
        self._latest_segment_by_key: dict[tuple[str, str], EpisodicSegment] = {}

    @property
    def events(self) -> list[MemoryEvent]:
        return list(self._events)

    @property
    def key_events(self) -> list[MemoryEvent]:
        return list(self._key_events)

    @property
    def segments(self) -> list[EpisodicSegment]:
        return deepcopy(self._segments)

    def add(self, event: MemoryEvent) -> None:
        if event.event_id in self._ids:
            raise ValueError(f"duplicate event_id: {event.event_id}")
        if not 0.0 <= event.confidence <= 1.0:
            raise ValueError("event confidence must be in [0, 1]")
        if not 0.0 <= event.importance <= 1.0:
            raise ValueError("event importance must be in [0, 1]")
        if not event.source:
            raise ValueError("event source is required")
        self._events.append(event)
        self._ids.add(event.event_id)
        if self.index_enabled and event.kind in _KEY_KINDS:
            self._key_events.append(event)
            self._add_to_segment(event)

    def _add_to_segment(self, event: MemoryEvent) -> None:
        label = _event_label(event)
        key = (event.kind, label.casefold())
        previous = self._latest_segment_by_key.get(key)
        if previous is not None and previous.end_step <= event.step_id <= previous.end_step + 1:
            segment = previous
            segment.end_step = event.step_id
        else:
            segment = EpisodicSegment(event.kind, label, event.step_id, event.step_id)
            self._segments.append(segment)
            self._latest_segment_by_key[key] = segment
        segment.event_ids.append(event.event_id)
        node_id = _node_id(event)
        if node_id is not None:
            segment.node_ids.append(node_id)
        if event.source not in segment.sources:
            segment.sources.append(event.source)
        segment.max_confidence = max(segment.max_confidence, event.confidence)

    def query(
        self,
        *,
        kind: str | None = None,
        place_id: str | None = None,
        since_step: int | None = None,
        min_confidence: float = 0.0,
    ) -> list[MemoryEvent]:
        return [
            event for event in self._events
            if (kind is None or event.kind == kind)
            and (place_id is None or event.place_id == place_id)
            and (since_step is None or event.step_id >= since_step)
            and event.confidence >= min_confidence
        ]

    def query_segments(
        self,
        *,
        since_step: int | None = None,
        until_step: int | None = None,
        kind: str | None = None,
        label: str | None = None,
        instruction: str | None = None,
        evidence_id: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[EpisodicSegment]:
        """Query indexed time spans by class, time, instruction, or event/node ref."""
        instruction_terms = _terms(instruction) if instruction else set()
        return [
            deepcopy(segment) for segment in self._segments
            if (since_step is None or segment.end_step >= since_step)
            and (until_step is None or segment.start_step <= until_step)
            and (kind is None or segment.kind == kind)
            and (label is None or segment.label.casefold() == label.casefold())
            and (not instruction_terms or bool(instruction_terms & _terms(segment.label)))
            and (evidence_id is None or evidence_id in segment.event_ids
                 or evidence_id in segment.node_ids)
            and segment.max_confidence >= min_confidence
        ]

    def retrieve(self, query: MemoryQuery) -> list[LayerMemoryItem]:
        """Rank key events independently of the flat retriever and ordinary actions."""
        if not self.index_enabled or query.top_k <= 0:
            return []
        instruction_terms = _terms(query.instruction)
        grouped: dict[tuple[str, int], list[MemoryEvent]] = {}
        candidates: list[tuple[MemoryEvent, float]] = []
        for event in self._key_events:
            if event.step_id > query.current_step:
                continue
            node_id = _node_id(event)
            if event.kind in _VISUAL_KINDS and node_id is not None:
                grouped.setdefault((node_id, event.step_id), []).append(event)
            else:
                candidates.append((event, event.importance))
        for representations in grouped.values():
            # A task-match event raises importance, but the original observation
            # remains the retrievable record and its provenance is unchanged.
            canonical = min(
                representations,
                key=lambda event: (
                    0 if event.kind in {"object_observation", "landmark"} else 1,
                    -event.confidence, event.event_id,
                ),
            )
            candidates.append((canonical, max(event.importance for event in representations)))

        def rank(value: tuple[MemoryEvent, float]) -> tuple[float, int, str]:
            event, importance = value
            label = _event_label(event)
            overlap = len(instruction_terms & _terms(label + " " + event.description))
            lexical = 3.0 * overlap / len(instruction_terms) if instruction_terms else 0.0
            age = max(0, query.current_step - event.step_id)
            recency = 1.0 / (1.0 + age / 16.0)
            place = 1.0 if query.place_id and event.place_id == query.place_id else 0.0
            visual = 0.6 if event.kind in _VISUAL_KINDS else 0.0
            score = lexical + recency + event.confidence + 0.7 * importance + place + visual
            return score, event.step_id, event.event_id

        candidates.sort(key=lambda value: (-rank(value)[0], -rank(value)[1], rank(value)[2]))
        selected: list[LayerMemoryItem] = []
        turn_seen = False
        for event, importance in candidates:
            if event.kind == "turn":
                if turn_seen:
                    continue
                turn_seen = True
            label = _event_label(event)
            ref = ObservationEvidence(
                event_id=event.event_id,
                step_id=event.step_id,
                label=label,
                confidence=event.confidence,
                source=event.source,
                node_id=_node_id(event),
                evidence=f"episodic_event:{event.event_id}",
            )
            selected.append(LayerMemoryItem(
                layer="mid", kind=event.kind, description=event.description,
                step_id=event.step_id, confidence=event.confidence,
                source=event.source, evidence_refs=[ref],
                importance=importance, label=label,
            ))
            if len(selected) >= query.top_k:
                break
        return selected

    def clear(self) -> None:
        self._events.clear()
        self._ids.clear()
        self._key_events.clear()
        self._segments.clear()
        self._latest_segment_by_key.clear()
