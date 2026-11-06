"""Episode-local category memory backed by recorded visual and graph evidence.

An index entry is a collection of sightings of a *category*. It never claims
that equal labels in different images identify the same physical object.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from lightnav.memory.scene_graph import SceneGraph
from lightnav.memory.schemas import (
    LayerMemoryItem,
    MemoryEvent,
    MemoryQuery,
    ObservationEvidence,
    SceneEdge,
)

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_OBSERVATION_KINDS = frozenset({"object_observation", "landmark"})


def _terms(value: str) -> set[str]:
    words: set[str] = set()
    for match in _WORDS.finditer(value.casefold()):
        word = match.group()
        words.add(word)
        if len(word) > 4 and word.endswith("s"):
            words.add(word[:-1])
    return words


def _one_line(value: object) -> str:
    return " ".join(str(value).split())


@dataclass
class SemanticEntry:
    """Distinct observations of one label, with no asserted instance identity."""

    label: str
    observations: list[ObservationEvidence] = field(default_factory=list)
    instruction_relevant: bool = False

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def first_step(self) -> int:
        return min(item.step_id for item in self.observations)

    @property
    def last_step(self) -> int:
        return max(item.step_id for item in self.observations)

    @property
    def mean_confidence(self) -> float:
        return sum(item.confidence for item in self.observations) / self.observation_count

    @property
    def max_confidence(self) -> float:
        return max(item.confidence for item in self.observations)

    @property
    def sources(self) -> list[str]:
        return sorted({item.source for item in self.observations})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticEntry:
        values = dict(data)
        values["observations"] = [
            ObservationEvidence.from_dict(item) for item in values.get("observations", [])
        ]
        return cls(**values)


class SemanticIndex:
    """Long-term, episode-scoped semantic index and evidence-backed relations."""

    def __init__(self) -> None:
        self._entries: dict[str, SemanticEntry] = {}
        self._event_ids: set[str] = set()
        self._graph: SceneGraph | None = None

    @property
    def entries(self) -> list[SemanticEntry]:
        return list(self._entries.values())

    def get(self, label: str) -> SemanticEntry | None:
        return self._entries.get(label.strip().casefold())

    def query(
        self,
        *,
        label: str | None = None,
        since_step: int | None = None,
        min_confidence: float = 0.0,
    ) -> list[SemanticEntry]:
        """Inspect category records without claiming any physical instance match."""
        return [
            entry for entry in self._entries.values()
            if (label is None or entry.label.casefold() == label.casefold())
            and any(
                (since_step is None or obs.step_id >= since_step)
                and obs.confidence >= min_confidence
                for obs in entry.observations
            )
        ]

    def update(self, events: list[MemoryEvent], graph: SceneGraph, instruction: str) -> None:
        """Index the writer's existing events; never run extraction or create edges."""
        self._graph = graph
        instruction_terms = _terms(instruction)
        for event in events:
            if event.kind not in _OBSERVATION_KINDS or event.event_id in self._event_ids:
                continue
            node_id = event.metadata.get("node_id")
            if not isinstance(node_id, str) or not node_id:
                node_id = None
            node = graph.get_node(node_id) if node_id else None
            if node is not None and node.kind == "object":
                label = node.label.strip()
                graph_evidence = _one_line(node.attributes.get("evidence", ""))
            else:
                label = event.description.removeprefix("Observed ").strip()
                graph_evidence = ""
            if not label:
                continue
            key = label.casefold()
            entry = self._entries.setdefault(key, SemanticEntry(label=label))
            entry.instruction_relevant |= bool(_terms(label) & instruction_terms)
            entry.observations.append(ObservationEvidence(
                event_id=event.event_id,
                step_id=event.step_id,
                label=label,
                confidence=event.confidence,
                source=event.source,
                node_id=node_id,
                evidence=graph_evidence or f"episodic event {event.event_id}",
            ))
            self._event_ids.add(event.event_id)

    def retrieve(self, query: MemoryQuery) -> list[LayerMemoryItem]:
        """Rank category summaries independently of short and episodic memory."""
        if query.top_k <= 0:
            return []
        query_terms = _terms(query.instruction)
        ranked: list[tuple[float, LayerMemoryItem]] = []
        for entry in self._entries.values():
            evidence = [obs for obs in entry.observations if obs.step_id <= query.current_step]
            if not evidence:
                continue
            first = min(obs.step_id for obs in evidence)
            latest = max(obs.step_id for obs in evidence)
            mean = sum(obs.confidence for obs in evidence) / len(evidence)
            strongest = max(obs.confidence for obs in evidence)
            label_terms = _terms(entry.label)
            relevance = len(label_terms & query_terms) / max(1, len(label_terms))
            recency = 1.0 / (1.0 + max(0, query.current_step - latest) / 12.0)
            score = 3.0 * relevance + 0.75 * strongest + 0.25 * mean + 0.25 * recency
            score += 0.25 if entry.instruction_relevant else 0.0
            description = (
                f"Category {_one_line(entry.label)}: {len(evidence)} visual sighting(s), "
                f"first step {first}, latest step {latest}, mean confidence {mean:.2f}, "
                f"max confidence {strongest:.2f}; category only, instance identity unverified"
            )
            ranked.append((score, LayerMemoryItem(
                layer="long",
                kind="semantic_summary",
                description=description,
                step_id=latest,
                confidence=mean,
                source="semantic_index",
                evidence_refs=list(evidence),
                importance=0.85 if relevance else 0.6,
                label=entry.label,
            )))

        if self._graph is not None:
            for edge in self._graph.edges:
                if edge.step_id <= query.current_step:
                    ranked.append(self._relation_item(edge, query))
        ranked.sort(key=lambda pair: (-pair[0], -pair[1].step_id, pair[1].label or ""))
        return [item for _, item in ranked[:query.top_k]]

    def clear(self) -> None:
        self._entries.clear()
        self._event_ids.clear()
        self._graph = None

    def _relation_item(self, edge: SceneEdge, query: MemoryQuery) -> tuple[float, LayerMemoryItem]:
        assert self._graph is not None
        start = self._graph.get_node(edge.source_id)
        end = self._graph.get_node(edge.target_id)
        start_label = _one_line(start.label) if start else edge.source_id
        end_label = _one_line(end.label) if end else edge.target_id
        if edge.relation == "connected_to":
            status = "verified traversal" if edge.source == "verified_traversal" else "unverified claim"
        elif edge.relation == "contains":
            status = "verified containment" if edge.source == "verified_containment" else "unverified claim"
        elif edge.relation == "possible_transition":
            status = "possible transition, unverified"
        else:
            status = "visually observed relation"
        description = (
            f"{edge.relation} ({status}): {start_label} -> {end_label}; "
            f"evidence: {_one_line(edge.evidence)}"
        )
        evidence = ObservationEvidence(
            event_id=edge.edge_id,
            step_id=edge.step_id,
            label=edge.relation,
            confidence=edge.confidence,
            source=edge.source,
            evidence=edge.evidence,
        )
        place_bonus = 1.0 if query.place_id in (edge.source_id, edge.target_id) else 0.0
        relevance = len(_terms(description) & _terms(query.instruction))
        score = place_bonus + edge.confidence + 0.5 * relevance
        return score, LayerMemoryItem(
            layer="long",
            kind="uncertain_relation" if "unverified" in status else "semantic_relation",
            description=description,
            step_id=edge.step_id,
            confidence=edge.confidence,
            source=edge.source,
            evidence_refs=[evidence],
            importance=0.5,
            label=edge.relation,
        )
