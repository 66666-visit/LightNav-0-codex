"""Turn V3 retrieval results into source-bound V4 navigation facts.

The V3 ranker decides which stored evidence is worth inspecting. This adapter
does not use its prose descriptions as facts: aggregate node and episode
descriptions can contain actions or outcomes, and their ``source_step`` loses
cross-Episode provenance. Every statement below is reconstructed from the
original EvidenceRecord and its canonical key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from lightnav.memory.v3_retrieval import RetrievalResult, _terms
from lightnav.memory.v3_schemas import EvidenceRecord, RetrievedMemory, SpatialEdgeV3
from lightnav.memory.v3_store import V3MemoryStore
from lightnav.memory.v4_navigation_state import (
    EvidenceProvenance,
    NavigationState,
    RevisitCandidate,
)

_LANDMARK_EVENTS = frozenset({"object_observation", "landmark"})
_VISUAL_EVENTS = _LANDMARK_EVENTS | {"place_observation", "place_visit"}
_ISSUED_EVENTS = frozenset({"action", "turn", "issued_command", "prediction"})
_RELATION_KINDS = frozenset({
    "inferred_relation", "observed_transition", "verified_traversal",
    "spatial_relation", "viewpoint_connection",
})


@dataclass(frozen=True)
class NavigationFact:
    """One observation or relation; status never upgrades by wording alone."""

    kind: str
    semantic_label: str | None
    description: str
    evidence_ids: tuple[str, ...]
    provenance: tuple[EvidenceProvenance, ...]
    source_memory_id: str
    temporal_role: str  # CURRENT | HISTORICAL | MIXED
    observation_status: str
    visit_status: str  # UNKNOWN | VERIFIED
    spatial_status: str
    traversability_status: str
    instruction_relevant: bool
    source: str
    raw_detector_confidence: float | None = None
    relation_endpoints: tuple[str, str] | None = None
    source_description: str | None = None


@dataclass(frozen=True)
class RevisitEvidence:
    """Same-Scene visual candidate; identity is explicitly unresolved."""

    current_evidence_id: str
    historical_evidence_ids: tuple[str, ...]
    provenance: tuple[EvidenceProvenance, ...]
    independent_history_observations: int
    highest_visual_similarity: float
    instruction_related: bool
    identity_status: str = "unknown"


@dataclass(frozen=True)
class DroppedCandidate:
    memory_id: str
    evidence_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class NavigationMemoryContext:
    """Structured inputs for V4 utilization and selective prompt formatting."""

    mode: str = "NONE"
    current_evidence: tuple[NavigationFact, ...] = ()
    history_evidence: tuple[NavigationFact, ...] = ()
    historical_landmarks: tuple[NavigationFact, ...] = ()
    revisit_evidence: tuple[RevisitEvidence, ...] = ()
    instruction_progress: tuple[NavigationFact, ...] = ()
    uncertain_relations: tuple[NavigationFact, ...] = ()
    issued_commands: tuple[NavigationFact, ...] = ()
    evidence_provenance: tuple[EvidenceProvenance, ...] = ()
    dropped_candidates: tuple[DroppedCandidate, ...] = ()
    candidate_memory_ids: tuple[str, ...] = ()
    ordered_facts: tuple[NavigationFact, ...] = ()
    selected_facts: tuple[NavigationFact, ...] = field(default_factory=tuple)
    selected_revisits: tuple[RevisitEvidence, ...] = field(default_factory=tuple)


def _role(record: EvidenceRecord, state: NavigationState) -> str:
    if (record.episode_id, record.rollout_idx) != (state.episode_id, state.rollout_idx):
        return "HISTORICAL"
    if record.step < state.last_update_step:
        return "HISTORICAL"
    if record.step == state.last_update_step:
        return "CURRENT"
    return "FUTURE"


def _related(record: EvidenceRecord, state: NavigationState) -> bool:
    """Use the same lexical normalization as V3, but only on semantic labels."""
    label = _terms(_semantic_label(record))
    return bool(label and label.issubset(set(state.instruction_anchors)))


def _semantic_label(record: EvidenceRecord) -> str | None:
    if record.semantic_label:
        return record.semantic_label.strip()
    if record.event_type == "place_visit":
        description = record.payload.get("description")
        if isinstance(description, str):
            match = re.fullmatch(r"Extractor reported (.+) \(place hypothesis\)", description)
            if match:
                return match.group(1).strip()
    return None


def _record_fact(record: EvidenceRecord, memory: RetrievedMemory,
                 state: NavigationState) -> NavigationFact:
    role = _role(record, state)
    label = _semantic_label(record)
    # A detector reports an observation. It does not establish physical place
    # identity, successful passage, or completion of an instruction clause.
    if record.event_type in _LANDMARK_EVENTS:
        if record.observation_status == "observed":
            description = f"RGB detector reported {label}" if label else "RGB object observation"
        else:
            description = f"Unverified label candidate: {label}" if label else "Unverified label"
    elif record.event_type == "place_observation":
        description = "RGB visual place descriptor was observed; place identity unknown"
    elif record.event_type == "place_visit":
        description = (f"RGB extractor reported place label {label}; visit unverified"
                       if label else "RGB extractor reported an unresolved place hypothesis")
    elif record.event_type in _ISSUED_EVENTS:
        description = "Navigation command was issued before a later observation; movement unverified"
    else:
        description = record.payload.get("description", record.event_type)
        if not isinstance(description, str):
            description = record.event_type
    return NavigationFact(
        kind=record.event_type, semantic_label=label, description=description,
        evidence_ids=(record.key,), provenance=(EvidenceProvenance.from_record(record),),
        source_memory_id=memory.memory_id, temporal_role=role,
        observation_status=("unknown" if record.event_type in _ISSUED_EVENTS
                            else record.observation_status),
        visit_status="UNKNOWN",
        spatial_status=("unknown" if record.event_type in _ISSUED_EVENTS
                        else record.spatial_status),
        traversability_status=("unknown" if record.event_type in _ISSUED_EVENTS
                               else record.traversability_status),
        instruction_relevant=_related(record, state),
        source=record.detector_name or record.event_type,
        raw_detector_confidence=record.raw_detector_confidence,
        source_description=(record.payload.get("description")
                            if isinstance(record.payload.get("description"), str) else None),
    )


def _find_edge(memory: RetrievedMemory, store: V3MemoryStore) -> SpatialEdgeV3 | None:
    if not memory.memory_id.startswith("edge:"):
        return None
    try:
        identity = json.loads(memory.memory_id.removeprefix("edge:"))
    except json.JSONDecodeError:
        return None
    if not isinstance(identity, list) or len(identity) != 2 or identity[0] != memory.scene_id:
        return None
    return next((edge for edge in store.edges_for_scene(memory.scene_id)
                 if edge.edge_id == identity[1]), None)


def _relation_fact(edge: SpatialEdgeV3, refs: tuple[EvidenceRecord, ...],
                   memory: RetrievedMemory, state: NavigationState,
                   store: V3MemoryStore) -> NavigationFact | None:
    if set(memory.evidence_ids) != set(edge.evidence_ids):
        return None
    nodes = {node.node_id: node for node in store.nodes_for_scene(state.scene_id)}
    source_node = nodes.get(edge.source_node)
    target_node = nodes.get(edge.target_node)
    labels = tuple(label for label in (
        source_node.semantic_label if source_node else None,
        target_node.semantic_label if target_node else None,
    ) if label)
    instruction_terms = set(state.instruction_anchors)
    relevant = any(_terms(label) & instruction_terms for label in labels)
    if not relevant:
        return None
    verified = (
        edge.relation_type == "verified_traversal"
        and edge.successful_traversals > 0
        and edge.traversability_status in {"traversed", "verified"}
    )
    label_text = " and ".join(labels) if labels else "visual place candidates"
    description = (
        f"Measured traversal between {label_text}; instruction completion unknown"
        if verified else
        f"Visual relation between {label_text}; passage and place identity unknown"
    )
    roles = {_role(record, state) for record in refs}
    temporal_role = next(iter(roles)) if len(roles) == 1 else "MIXED"
    return NavigationFact(
        kind=edge.relation_type, semantic_label=None, description=description,
        evidence_ids=tuple(record.key for record in refs),
        provenance=tuple(EvidenceProvenance.from_record(record) for record in refs),
        source_memory_id=memory.memory_id, temporal_role=temporal_role,
        observation_status="unknown" if verified else "inferred",
        visit_status="VERIFIED" if verified else "UNKNOWN",
        spatial_status="traversed" if verified else edge.spatial_status,
        traversability_status=edge.traversability_status if verified else "unknown",
        instruction_relevant=True, source=edge.relation_type,
        relation_endpoints=(edge.source_node, edge.target_node),
    )


def _revisit(candidate: RevisitCandidate, store: V3MemoryStore,
             state: NavigationState) -> RevisitEvidence | None:
    current = store.get_evidence(candidate.current_evidence_id)
    history = tuple(store.get_evidence(ref) for ref in candidate.historical_evidence_ids)
    if (current is None or any(record is None for record in history)
            or _role(current, state) != "CURRENT"
            or current.scene_id != state.scene_id
            or current.event_type != "place_observation"
            or current.observation_status != "observed"
            or current.visual_descriptor is None
            or candidate.identity_status != "unknown"):
        return None
    valid_history = tuple(record for record in history if record is not None)
    if not valid_history or any(record.scene_id != state.scene_id
                                or record.event_type != "place_observation"
                                or record.observation_status != "observed"
                                or record.visual_descriptor is None
                                or _role(record, state) != "HISTORICAL"
                                for record in valid_history):
        return None
    unique_times = {(record.episode_id, record.rollout_idx, record.step)
                    for record in valid_history}
    if len(unique_times) != candidate.independent_history_observations:
        return None
    return RevisitEvidence(
        current_evidence_id=current.key,
        historical_evidence_ids=tuple(record.key for record in valid_history),
        provenance=tuple(EvidenceProvenance.from_record(record)
                         for record in (current, *valid_history)),
        independent_history_observations=candidate.independent_history_observations,
        highest_visual_similarity=candidate.highest_visual_similarity,
        instruction_related=candidate.instruction_related,
    )


def build_navigation_memory_context(
    result: RetrievalResult, store: V3MemoryStore, state: NavigationState,
) -> NavigationMemoryContext:
    """Expand existing V3 candidates into atomic, provenance-checked facts.

    Relation and revisit assertions retain all their supporting sources; the
    formatter must select disjoint evidence sets when preparing the prompt.
    """
    if store.active_episode != (state.scene_id, state.episode_id, state.rollout_idx):
        raise ValueError("V4 context requires the matching active V3 Episode")
    if state.last_update_step < 0:
        raise ValueError("update navigation state before building V4 context")

    current: list[NavigationFact] = []
    history: list[NavigationFact] = []
    landmarks: list[NavigationFact] = []
    relations: list[NavigationFact] = []
    progress: list[NavigationFact] = []
    issued: list[NavigationFact] = []
    dropped: list[DroppedCandidate] = []
    provenance: dict[str, EvidenceProvenance] = {}
    seen_atomic: set[str] = set()
    seen_relations: set[str] = set()
    ordered_facts: list[NavigationFact] = []

    for memory in result.records:
        if memory.scene_id != state.scene_id:
            dropped.append(DroppedCandidate(memory.memory_id, memory.evidence_ids, "wrong_scene"))
            continue
        refs = tuple(store.get_evidence(key) for key in dict.fromkeys(memory.evidence_ids))
        if (not refs or any(record is None or not record.provenance_validity
                            or record.scene_id != state.scene_id for record in refs)):
            dropped.append(DroppedCandidate(memory.memory_id, memory.evidence_ids,
                                            "invalid_provenance"))
            continue
        resolved = tuple(record for record in refs if record is not None)
        if any(_role(record, state) == "FUTURE" for record in resolved):
            dropped.append(DroppedCandidate(memory.memory_id, memory.evidence_ids,
                                            "future_evidence"))
            continue

        edge = _find_edge(memory, store) if memory.kind in _RELATION_KINDS else None
        if edge is not None and memory.memory_id not in seen_relations:
            relation = _relation_fact(edge, resolved, memory, state, store)
            seen_relations.add(memory.memory_id)
            if relation is None:
                dropped.append(DroppedCandidate(memory.memory_id, memory.evidence_ids,
                                                "unrelated_relation"))
            else:
                relations.append(relation)
                ordered_facts.append(relation)
                if relation.visit_status == "VERIFIED":
                    progress.append(relation)
                for source in relation.provenance:
                    provenance.setdefault(source.evidence_id, source)
            continue
        if memory.kind in _RELATION_KINDS and edge is None:
            dropped.append(DroppedCandidate(memory.memory_id, memory.evidence_ids,
                                            "missing_relation"))
            continue

        added = False
        for record in resolved:
            if record.key in seen_atomic:
                continue
            if record.event_type in _ISSUED_EVENTS:
                fact = _record_fact(record, memory, state)
                issued.append(fact)
            elif record.event_type in _VISUAL_EVENTS:
                fact = _record_fact(record, memory, state)
                if fact.temporal_role == "CURRENT":
                    current.append(fact)
                elif fact.instruction_relevant and fact.temporal_role == "HISTORICAL":
                    history.append(fact)
                    ordered_facts.append(fact)
                    if fact.kind in _LANDMARK_EVENTS:
                        landmarks.append(fact)
                elif fact.kind == "place_observation":
                    # Visual descriptors can support a state-level revisit audit,
                    # but by themselves have no instruction-grounded prose value.
                    pass
                else:
                    dropped.append(DroppedCandidate(memory.memory_id, (record.key,),
                                                    "unrelated_history"))
            else:
                dropped.append(DroppedCandidate(memory.memory_id, (record.key,),
                                                "unsupported_event_type"))
                continue
            seen_atomic.add(record.key)
            provenance.setdefault(record.key, EvidenceProvenance.from_record(record))
            added = True
        if not added and all(record.key in seen_atomic for record in resolved):
            dropped.append(DroppedCandidate(memory.memory_id, memory.evidence_ids,
                                            "duplicate_evidence"))

    revisits: list[RevisitEvidence] = []
    for candidate in state.revisit_candidates:
        converted = _revisit(candidate, store, state)
        if converted is None:
            dropped.append(DroppedCandidate("revisit", (candidate.current_evidence_id,
                                                       *candidate.historical_evidence_ids),
                                            "invalid_revisit_provenance"))
            continue
        revisits.append(converted)
        for source in converted.provenance:
            provenance.setdefault(source.evidence_id, source)

    # Keep distinct observations and relations even when they disagree. Their
    # evidence keys, temporal roles, and statuses remain available to utilization.
    return NavigationMemoryContext(
        current_evidence=tuple(current), history_evidence=tuple(history),
        historical_landmarks=tuple(landmarks), revisit_evidence=tuple(revisits),
        instruction_progress=tuple(progress), uncertain_relations=tuple(
            relation for relation in relations if relation.visit_status != "VERIFIED"
        ),
        issued_commands=tuple(issued),
        evidence_provenance=tuple(provenance.values()),
        dropped_candidates=tuple(dropped),
        candidate_memory_ids=tuple(memory.memory_id for memory in result.records),
        ordered_facts=tuple(ordered_facts),
    )
