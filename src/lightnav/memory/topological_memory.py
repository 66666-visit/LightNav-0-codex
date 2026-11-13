"""Episode-local, descriptor-based place memory with observed transitions only.

Place identity here is a visual hypothesis. ``observed_transition`` means two
different descriptor clusters were observed in sequence with an intervening
executed command; it is not proof of a traversable doorway or geometry.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from lightnav.memory.schemas import LayerMemoryItem, MemoryEvent, MemoryQuery, ObservationEvidence

_OBJECT_KINDS = frozenset({"object_observation", "landmark"})
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def _terms(value: str) -> set[str]:
    return {match.group() for match in _WORDS.finditer(value.casefold())}


def _normalized(descriptor: Sequence[float], dimension: int | None = None) -> list[float]:
    if isinstance(descriptor, (str, bytes)):
        raise TypeError("place descriptor must be a numeric sequence")
    try:
        values = [float(value) for value in descriptor]
    except (TypeError, ValueError) as exc:
        raise TypeError("place descriptor must be a numeric sequence") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("place descriptor must contain finite values")
    if dimension is not None and len(values) != dimension:
        raise ValueError(f"place descriptor dimension changed: {len(values)} != {dimension}")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or not norm:
        raise ValueError("place descriptor must have finite nonzero norm")
    return [value / norm for value in values]


def _category_label(event: MemoryEvent) -> str:
    description = event.description.strip()
    return description.removeprefix("Observed ").strip()


@dataclass
class PlaceCategory:
    """Category sightings at one visual place, never an object instance track."""

    label: str
    first_seen_step: int
    last_seen_step: int
    observation_count: int = 0
    confidence_sum: float = 0.0
    max_confidence: float = 0.0
    evidence: list[ObservationEvidence] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.observation_count if self.observation_count else 0.0

    @property
    def evidence_event_ids(self) -> list[str]:
        return [ref.event_id for ref in self.evidence]

    @property
    def observed_object_node_ids(self) -> list[str]:
        return [ref.node_id for ref in self.evidence if ref.node_id is not None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaceCategory:
        values = dict(data)
        values["evidence"] = [ObservationEvidence.from_dict(ref) for ref in values["evidence"]]
        return cls(**values)


@dataclass
class PlaceNode:
    place_id: str
    visual_descriptor: list[float]
    first_seen_step: int
    last_seen_step: int
    visit_count: int
    observation_count: int
    observed_categories: dict[str, PlaceCategory] = field(default_factory=dict)
    evidence_event_ids: list[str] = field(default_factory=list)
    descriptor_event_ids: list[str] = field(default_factory=list)
    descriptor_steps: list[int] = field(default_factory=list)
    descriptor_similarities: list[float | None] = field(default_factory=list)
    revisit_steps: list[int] = field(default_factory=list)
    descriptor_source: str = ""
    last_similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaceNode:
        values = dict(data)
        values["observed_categories"] = {
            label: PlaceCategory.from_dict(category)
            for label, category in values.get("observed_categories", {}).items()
        }
        return cls(**values)


@dataclass
class PlaceEdge:
    source_place_id: str
    target_place_id: str
    relation_type: str
    first_observed_step: int
    last_observed_step: int
    transition_count: int
    confidence: float
    evidence_event_ids: list[str] = field(default_factory=list)
    motion_event_ids: list[str] = field(default_factory=list)
    last_observation_event_ids: list[str] = field(default_factory=list)
    last_observation_steps: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaceEdge:
        return cls(**data)


@dataclass(frozen=True)
class PlaceObservationResult:
    place_id: str
    top1_similarity: float | None
    matched: bool
    created: bool
    revisited: bool
    transition: PlaceEdge | None
    observation_count: int
    visit_count: int


class TopologicalMemory:
    """Nearest-neighbor visual places and directed, unverified observations."""

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        if not isinstance(similarity_threshold, (int, float)) or not math.isfinite(
            similarity_threshold
        ) or not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be finite and in [-1, 1]")
        self.similarity_threshold = float(similarity_threshold)
        self._nodes: dict[str, PlaceNode] = {}
        self._edges: dict[tuple[str, str], PlaceEdge] = {}
        self._current_place_id: str | None = None
        self._last_observation_step: int | None = None
        self._last_result: PlaceObservationResult | None = None
        self._descriptor_dimension: int | None = None
        self._descriptor_source = ""

    @property
    def nodes(self) -> list[PlaceNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[PlaceEdge]:
        return list(self._edges.values())

    @property
    def current_place_id(self) -> str | None:
        return self._current_place_id

    @property
    def last_observation_step(self) -> int | None:
        return self._last_observation_step

    @property
    def last_result(self) -> PlaceObservationResult | None:
        return self._last_result

    def get_node(self, place_id: str) -> PlaceNode | None:
        return self._nodes.get(place_id)

    def get_edge(self, source_place_id: str, target_place_id: str) -> PlaceEdge | None:
        return self._edges.get((source_place_id, target_place_id))

    def observe(
        self,
        step_id: int,
        descriptor: Sequence[float],
        events: list[MemoryEvent],
        *,
        descriptor_source: str,
        observation_event_id: str,
        transition_evidence_ids: Sequence[str] = (),
        force_match_place_id: str | None = None,
        force_new: bool = False,
    ) -> PlaceObservationResult:
        """Match one sampled RGB descriptor and bind same-frame YOLO sightings.

        Caller supplies a real episode-local ``place_observation`` event ID and
        intervening executed-command event IDs. Category names never affect
        matching. An edge requires both different place IDs and motion evidence.
        """
        if not isinstance(step_id, int) or step_id < 0:
            raise ValueError("step_id must be a nonnegative integer")
        if force_match_place_id is not None and force_new:
            raise ValueError("cannot force both a place match and a new place")
        if self._last_observation_step is not None and step_id <= self._last_observation_step:
            raise ValueError("place observation steps must increase")
        if not isinstance(descriptor_source, str) or not descriptor_source.strip():
            raise ValueError("descriptor_source is required")
        if not isinstance(observation_event_id, str) or not observation_event_id.strip():
            raise ValueError("observation_event_id is required")
        if any(observation_event_id in node.descriptor_event_ids for node in self._nodes.values()):
            raise ValueError("duplicate place observation event ID")
        if not isinstance(events, list) or any(
            not isinstance(event, MemoryEvent) or event.step_id != step_id for event in events
        ):
            raise ValueError("semantic events must belong to the current observation step")
        if any(not isinstance(event_id, str) or not event_id.strip()
               for event_id in transition_evidence_ids):
            raise ValueError("transition evidence IDs must be nonempty strings")
        motion_ids = list(dict.fromkeys(transition_evidence_ids))
        values = _normalized(descriptor, self._descriptor_dimension)
        top_node: PlaceNode | None = None
        top1_similarity: float | None = None
        for node in self._nodes.values():
            similarity = max(-1.0, min(1.0, sum(
                a * b for a, b in zip(values, node.visual_descriptor)
            )))
            if top1_similarity is None or similarity > top1_similarity:
                top_node = node
                top1_similarity = similarity
        matched_similarity = top1_similarity
        if force_match_place_id is not None:
            if force_match_place_id not in self._nodes:
                raise ValueError("forced place is not an existing visual node")
            top_node = self._nodes[force_match_place_id]
            matched_similarity = sum(a * b for a, b in zip(values, top_node.visual_descriptor))
        matched = (force_match_place_id is not None or (
            not force_new and top_node is not None
            and top1_similarity >= self.similarity_threshold
        ))
        previous_id = self._current_place_id
        if matched:
            assert top_node is not None
            node = top_node
            old_count = node.observation_count
            node.visual_descriptor = _normalized([
                (old_count * old + new) for old, new in zip(node.visual_descriptor, values)
            ])
            node.last_seen_step = step_id
            node.observation_count += 1
            revisited = (
                previous_id is not None and previous_id != node.place_id and bool(motion_ids)
            )
            if revisited:
                node.visit_count += 1
                node.revisit_steps.append(step_id)
            node.last_similarity = matched_similarity
        else:
            place_id = f"place_{len(self._nodes):03d}"
            node = PlaceNode(
                place_id=place_id, visual_descriptor=values,
                first_seen_step=step_id, last_seen_step=step_id,
                visit_count=1, observation_count=1,
                descriptor_source=descriptor_source, last_similarity=top1_similarity,
            )
            self._nodes[place_id] = node
            revisited = False
        node.evidence_event_ids.append(observation_event_id)
        node.descriptor_event_ids.append(observation_event_id)
        node.descriptor_steps.append(step_id)
        node.descriptor_similarities.append(matched_similarity if matched else None)
        seen_event_ids: set[str] = set()
        for event in events:
            if event.kind not in _OBJECT_KINDS or event.event_id in seen_event_ids:
                continue
            label = _category_label(event)
            if not label or not 0.0 <= event.confidence <= 1.0:
                continue
            seen_event_ids.add(event.event_id)
            key = label.casefold()
            category = node.observed_categories.get(key)
            if category is None:
                category = PlaceCategory(label=label, first_seen_step=step_id,
                                         last_seen_step=step_id)
                node.observed_categories[key] = category
            category.observation_count += 1
            category.confidence_sum += event.confidence
            category.max_confidence = max(category.max_confidence, event.confidence)
            category.last_seen_step = step_id
            node_id = event.metadata.get("node_id") if isinstance(event.metadata, dict) else None
            category.evidence.append(ObservationEvidence(
                event_id=event.event_id, step_id=step_id, label=label,
                confidence=event.confidence, source=event.source,
                node_id=node_id if isinstance(node_id, str) and node_id else None,
                evidence=f"episodic event {event.event_id}",
            ))
            node.evidence_event_ids.append(event.event_id)

        transition: PlaceEdge | None = None
        if previous_id is not None and previous_id != node.place_id and motion_ids:
            key = (previous_id, node.place_id)
            confidence = min(0.7, max(0.0, matched_similarity)) if matched else 0.5
            previous_descriptor_id = self._nodes[previous_id].descriptor_event_ids[-1]
            previous_descriptor_step = self._nodes[previous_id].descriptor_steps[-1]
            evidence_ids = list(dict.fromkeys([
                previous_descriptor_id, *motion_ids, observation_event_id,
            ]))
            edge = self._edges.get(key)
            if edge is None:
                edge = PlaceEdge(
                    source_place_id=previous_id, target_place_id=node.place_id,
                    relation_type="observed_transition", first_observed_step=step_id,
                    last_observed_step=step_id, transition_count=1,
                    confidence=confidence, evidence_event_ids=evidence_ids,
                    motion_event_ids=motion_ids,
                    last_observation_event_ids=[previous_descriptor_id, observation_event_id],
                    last_observation_steps=[previous_descriptor_step, step_id],
                )
                self._edges[key] = edge
            else:
                edge.last_observed_step = step_id
                edge.transition_count += 1
                edge.confidence = max(edge.confidence, confidence)
                edge.evidence_event_ids.extend(
                    event_id for event_id in evidence_ids if event_id not in edge.evidence_event_ids
                )
                edge.motion_event_ids.extend(
                    event_id for event_id in motion_ids if event_id not in edge.motion_event_ids
                )
                edge.last_observation_event_ids = [previous_descriptor_id, observation_event_id]
                edge.last_observation_steps = [previous_descriptor_step, step_id]
            transition = edge

        self._current_place_id = node.place_id
        self._last_observation_step = step_id
        self._descriptor_dimension = len(values)
        self._descriptor_source = descriptor_source
        result = PlaceObservationResult(
            place_id=node.place_id, top1_similarity=top1_similarity, matched=matched,
            created=not matched, revisited=revisited, transition=transition,
            observation_count=node.observation_count, visit_count=node.visit_count,
        )
        self._last_result = result
        return result

    def retrieve(self, query: MemoryQuery) -> list[LayerMemoryItem]:
        """Describe the latest recognized place, evidenced neighbors, and categories."""
        if query.top_k <= 0 or self._current_place_id is None:
            return []
        if (
            self._last_observation_step is not None
            and query.current_step < self._last_observation_step
        ):
            # This store tracks the latest recognized state. A retrospective
            # query cannot safely reuse its newer visit and edge counters.
            return []
        node = self._nodes[self._current_place_id]
        current = query.current_step == node.last_seen_step
        source = node.descriptor_source
        similarity = self._last_result.top1_similarity if self._last_result else None
        similarity_text = "first observation" if similarity is None else f"top1={similarity:.2f}"
        recognition_confidence = (
            max(0.0, similarity)
            if self._last_result is not None and self._last_result.matched
            and similarity is not None else 0.5
        )
        place_ref = ObservationEvidence(
            event_id=node.descriptor_event_ids[-1], step_id=node.last_seen_step,
            label=node.place_id, confidence=1.0,
            source=source, evidence="RGB place descriptor observation",
        )
        kind = "current_place" if current else "last_recognized_place"
        description = (
            f"Current visual place {node.place_id} ({similarity_text}; "
            f"visits={node.visit_count}, views={node.observation_count}); visual match only"
            if current else
            f"Last recognized visual place {node.place_id} at step {node.last_seen_step}; "
            f"no new place descriptor at step {query.current_step}"
        )
        ranked: list[tuple[float, LayerMemoryItem]] = [(100.0, LayerMemoryItem(
            layer="long", kind=kind, description=description,
            step_id=node.last_seen_step, confidence=recognition_confidence,
            source=source, evidence_refs=[place_ref], importance=0.9,
            label=node.place_id,
            metadata={
                "place_id": node.place_id, "current": current,
                "similarity": similarity, "visit_count": node.visit_count,
                "observation_count": node.observation_count,
            },
        ))]
        query_terms = _terms(query.instruction)
        for semantic_place in self._nodes.values():
            is_current_place = semantic_place.place_id == node.place_id
            for category in semantic_place.observed_categories.values():
                refs = [ref for ref in category.evidence if ref.step_id <= query.current_step]
                if not refs:
                    continue
                overlap = len(query_terms & _terms(category.label))
                mean = sum(ref.confidence for ref in refs) / len(refs)
                last_step = max(ref.step_id for ref in refs)
                description = (
                    f"{'At current' if is_current_place else 'At historical'} visual place "
                    f"{semantic_place.place_id}, observed category {category.label} "
                    f"{len(refs)} time(s), latest step {last_step}; "
                    "object instance identity unknown"
                )
                rank = (2.0 if is_current_place else 1.5) + 2.0 * overlap + mean
                ranked.append((rank, LayerMemoryItem(
                    layer="long", kind="place_semantics", description=description,
                    step_id=last_step, confidence=mean,
                    source=",".join(sorted({ref.source for ref in refs})),
                    evidence_refs=list(refs), importance=0.75 if overlap else 0.55,
                    label=category.label,
                    metadata={
                        "place_id": semantic_place.place_id,
                        "current": is_current_place and last_step == query.current_step,
                        "descriptor_event_ids": list(semantic_place.descriptor_event_ids),
                    },
                )))
        for edge in self._edges.values():
            if edge.last_observed_step > query.current_step:
                continue
            if node.place_id not in (edge.source_place_id, edge.target_place_id):
                continue
            descriptor_refs = [
                ObservationEvidence(
                    event_id=event_id, step_id=step_id,
                    label=place_id, confidence=1.0,
                    source=self._nodes[place_id].descriptor_source,
                    evidence="RGB descriptor with executed motion evidence",
                )
                for event_id, step_id, place_id in zip(
                    edge.last_observation_event_ids,
                    edge.last_observation_steps,
                    (edge.source_place_id, edge.target_place_id),
                )
            ]
            description = (
                f"Observed shift {edge.source_place_id}->{edge.target_place_id} "
                f"({edge.transition_count}x, motion={len(edge.motion_event_ids)}); "
                f"RGB evidence {edge.last_observation_event_ids[0]}->"
                f"{edge.last_observation_event_ids[1]}; "
                "traversability unverified"
            )
            ranked.append((4.0 + 0.1 * edge.transition_count, LayerMemoryItem(
                layer="long", kind="observed_transition", description=description,
                step_id=edge.last_observed_step, confidence=edge.confidence,
                source="rgb_descriptor_and_action", evidence_refs=descriptor_refs,
                importance=0.7, label=edge.target_place_id,
                metadata={
                    "source_place_id": edge.source_place_id,
                    "target_place_id": edge.target_place_id,
                    "evidence_event_ids": list(edge.evidence_event_ids),
                    "motion_event_ids": list(edge.motion_event_ids),
                    "relation_id": f"place_shift_{edge.last_observed_step}",
                    "current": edge.last_observed_step == query.current_step,
                },
            )))
        for revisited_node in self._nodes.values():
            if (
                revisited_node.visit_count <= 1
                or revisited_node.revisit_steps[-1] > query.current_step
            ):
                continue
            revisit_step = revisited_node.revisit_steps[-1]
            revisit_index = revisited_node.descriptor_steps.index(revisit_step)
            revisit_event_id = revisited_node.descriptor_event_ids[revisit_index]
            revisit_confidence = max(0.0, min(1.0, (
                revisited_node.descriptor_similarities[revisit_index] or 0.0
            )))
            revisit_ref = ObservationEvidence(
                event_id=revisit_event_id, step_id=revisit_step,
                label=revisited_node.place_id,
                confidence=1.0,
                source=revisited_node.descriptor_source,
                evidence="RGB place descriptor revisit",
            )
            is_current_place = revisited_node.place_id == node.place_id
            description = (
                f"{'Current' if is_current_place else 'Historical'} visual place "
                f"{revisited_node.place_id} revisited at step {revisit_step} "
                f"({revisited_node.visit_count} visits); descriptor match only"
            )
            recency = 1.0 / (1.0 + max(0, query.current_step - revisit_step) / 20.0)
            relevance = sum(
                bool(query_terms & _terms(category.label))
                for category in revisited_node.observed_categories.values()
            )
            ranked.append((2.0 + recency + relevance, LayerMemoryItem(
                layer="long", kind="revisited_place", description=description,
                step_id=revisit_step, confidence=revisit_confidence,
                source=revisited_node.descriptor_source,
                evidence_refs=[revisit_ref],
                importance=0.75, label=revisited_node.place_id,
                metadata={
                    "place_id": revisited_node.place_id,
                    "current": is_current_place and revisit_step == query.current_step,
                    "visit_count": revisited_node.visit_count,
                },
            )))
        ranked.sort(key=lambda pair: (-pair[0], -pair[1].step_id, pair[1].kind))
        return [item for _, item in ranked[:query.top_k]]

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._current_place_id = None
        self._last_observation_step = None
        self._last_result = None
        self._descriptor_dimension = None
        self._descriptor_source = ""
