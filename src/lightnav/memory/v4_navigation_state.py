"""Evidence-bound navigation progress for the frozen LightNav V4 policy.

This state summarizes observations. It never chooses an action or infers that a
command was executed. In particular, DINO similarity is only a revisit
hypothesis; an independently measured traversal still does not prove that the
navigation instruction is complete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lightnav.memory.v3_retrieval import _cosine, _terms
from lightnav.memory.v3_schemas import EvidenceRecord
from lightnav.memory.v3_store import V3MemoryStore, _has_observed_displacement

NOT_OBSERVED = "NOT_OBSERVED"
OBSERVED = "OBSERVED"
POSSIBLE_REVISIT = "POSSIBLE_REVISIT"
PROGRESS_SUPPORTED = "PROGRESS_SUPPORTED"
COMPLETION_UNKNOWN = "COMPLETION_UNKNOWN"

_LANDMARK_EVENTS = frozenset({"object_observation", "landmark"})
_OBSERVATION_EVENTS = _LANDMARK_EVENTS | {"place_observation"}
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def _instruction_anchors(instruction: str) -> tuple[str, ...]:
    """Reuse V3 lexical normalization without claiming a parsed subgoal order."""
    anchors: list[str] = []
    for word in _WORDS.findall(instruction.casefold()):
        anchors.extend(sorted(_terms(word)))
    return tuple(dict.fromkeys(anchors))


def _relevant_landmark(record: EvidenceRecord, instruction_terms: set[str]) -> bool:
    if record.event_type not in _LANDMARK_EVENTS or record.observation_status != "observed":
        return False
    label_terms = _terms(record.semantic_label)
    return bool(label_terms and label_terms.issubset(instruction_terms))


def _is_observation(record: EvidenceRecord) -> bool:
    return record.event_type in _OBSERVATION_EVENTS and record.observation_status == "observed"


@dataclass(frozen=True)
class EvidenceProvenance:
    evidence_id: str  # canonical Scene/Episode/rollout/event key
    scene_id: str
    episode_id: str
    rollout_idx: int
    step: int

    @classmethod
    def from_record(cls, record: EvidenceRecord) -> EvidenceProvenance:
        return cls(record.key, record.scene_id, record.episode_id,
                   record.rollout_idx, record.step)


@dataclass(frozen=True)
class RevisitCandidate:
    current_evidence_id: str
    historical_evidence_ids: tuple[str, ...]
    independent_history_observations: int
    highest_visual_similarity: float
    instruction_related: bool
    identity_status: str = "unknown"


@dataclass(frozen=True)
class NavigationState:
    instruction: str
    scene_id: str
    episode_id: str
    rollout_idx: int
    instruction_anchors: tuple[str, ...] = ()
    active_anchor: str | None = None
    anchor_evidence: dict[str, tuple[str, ...]] = field(default_factory=dict)
    anchor_status: dict[str, str] = field(default_factory=dict)
    recent_observation_ids: tuple[str, ...] = ()
    historical_landmark_ids: tuple[str, ...] = ()
    revisit_candidates: tuple[RevisitCandidate, ...] = ()
    repeated_observation_count: int = 0
    progress_status: str = COMPLETION_UNKNOWN
    uncertainty: str = "no_verified_instruction_progress"
    evidence_ids: tuple[str, ...] = ()
    evidence_provenance: tuple[EvidenceProvenance, ...] = ()
    progress_evidence_ids: tuple[str, ...] = ()
    last_update_step: int = -1


class NavigationStateUpdater:
    """Incrementally summarize the existing V3 store for one active Episode.

    Call ``reset_episode`` after ``V3MemoryStore.begin_episode`` and ``update``
    after the current step's evidence has been stored. The updater reads no
    Habitat observation or evaluator ``info`` object.
    """

    def __init__(self, store: V3MemoryStore, *, revisit_enabled: bool = True,
                 revisit_similarity_threshold: float | None = None,
                 recent_capacity: int = 8) -> None:
        if recent_capacity < 1:
            raise ValueError("recent_capacity must be positive")
        threshold = (store.place_similarity_threshold if revisit_similarity_threshold is None
                     else revisit_similarity_threshold)
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("revisit_similarity_threshold must be in [-1, 1]")
        self.store = store
        self.revisit_enabled = revisit_enabled
        self.revisit_similarity_threshold = threshold
        self.recent_capacity = recent_capacity
        self.state: NavigationState | None = None

    def reset_episode(self, scene_id: str, episode_id: str, instruction: str,
                      *, rollout_idx: int = 0) -> NavigationState:
        if self.store.active_episode != (scene_id, episode_id, rollout_idx):
            raise ValueError("navigation state must match the active V3 Episode")
        if not isinstance(instruction, str):
            raise TypeError("instruction must be a string")
        anchors = _instruction_anchors(instruction)
        self.state = NavigationState(
            instruction=instruction, scene_id=scene_id, episode_id=episode_id,
            rollout_idx=rollout_idx, instruction_anchors=anchors,
            anchor_evidence={anchor: () for anchor in anchors},
            anchor_status={anchor: NOT_OBSERVED for anchor in anchors},
        )
        return self.state

    def update(self, step: int) -> NavigationState:
        previous = self.state
        if previous is None:
            raise RuntimeError("reset_episode must be called before update")
        identity = (previous.scene_id, previous.episode_id, previous.rollout_idx)
        if self.store.active_episode != identity:
            raise RuntimeError("active V3 Episode changed without resetting navigation state")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if step < previous.last_update_step:
            raise ValueError("navigation state step regressed")

        records = tuple(record for record in self.store.evidence_for_scene(previous.scene_id)
                        if record.provenance_validity and (
                            (record.episode_id, record.rollout_idx) != identity[1:]
                            or record.step <= step
                        ))
        by_key = {record.key: record for record in records}
        active = tuple(record for record in records
                       if (record.episode_id, record.rollout_idx) == identity[1:])
        current = tuple(record for record in active if record.step == step)
        earlier = tuple(record for record in records if (
            (record.episode_id, record.rollout_idx) != identity[1:]
            or record.step < step
        ))
        instruction_terms = set(previous.instruction_anchors)

        observed_active = tuple(record for record in active if _is_observation(record))
        recent_ids = tuple(record.key for record in observed_active[-self.recent_capacity:])
        related_active = tuple(record for record in active
                               if _relevant_landmark(record, instruction_terms))
        related_history = tuple(record for record in earlier
                                if _relevant_landmark(record, instruction_terms))
        anchor_evidence = {
            anchor: tuple(record.key for record in related_active
                          if anchor in _terms(record.semantic_label))
            for anchor in previous.instruction_anchors
        }
        active_anchor = related_active[-1].semantic_label if related_active else None

        revisit_candidates: list[RevisitCandidate] = []
        repeated_observations: set[tuple[str, int, int]] = set()
        if self.revisit_enabled:
            instruction_related_steps = {
                (record.episode_id, record.rollout_idx, record.step)
                for record in (*related_active, *related_history)
            }
            for observed in current:
                if (observed.event_type != "place_observation"
                        or observed.observation_status != "observed"
                        or observed.visual_descriptor is None):
                    continue
                matches: list[tuple[float, EvidenceRecord]] = []
                for historical in earlier:
                    if (historical.event_type != "place_observation"
                            or historical.observation_status != "observed"
                            or historical.visual_descriptor is None):
                        continue
                    similarity = _cosine(observed.visual_descriptor,
                                         historical.visual_descriptor)
                    if similarity is not None and similarity >= self.revisit_similarity_threshold:
                        matches.append((similarity, historical))
                if not matches:
                    continue
                matches.sort(key=lambda pair: (-pair[0], pair[1].key))
                unique_times = {(record.episode_id, record.rollout_idx, record.step)
                                for _, record in matches}
                repeated_observations.update(unique_times)
                related = ((observed.episode_id, observed.rollout_idx, observed.step)
                           in instruction_related_steps or any(
                               (record.episode_id, record.rollout_idx, record.step)
                               in instruction_related_steps for _, record in matches
                           ))
                revisit_candidates.append(RevisitCandidate(
                    current_evidence_id=observed.key,
                    historical_evidence_ids=tuple(record.key for _, record in matches),
                    independent_history_observations=len(unique_times),
                    highest_visual_similarity=matches[0][0],
                    instruction_related=related,
                ))

        progress_refs: list[str] = []
        if related_active:
            for edge in self.store.edges_for_scene(previous.scene_id):
                if (edge.relation_type != "verified_traversal"
                        or edge.traversability_status not in {"traversed", "verified"}
                        or edge.successful_traversals < 1):
                    continue
                direct_refs = {record.key for record in related_active}
                if not direct_refs.intersection(edge.evidence_ids):
                    continue
                edge_refs = [ref for ref in edge.evidence_ids if ref in by_key and (
                    by_key[ref].episode_id, by_key[ref].rollout_idx
                ) == identity[1:]]
                if not any(_has_observed_displacement(by_key[ref]) for ref in edge_refs):
                    continue
                progress_refs.extend(edge_refs)

        if progress_refs:
            status = PROGRESS_SUPPORTED
            uncertainty = "goal_completion_unverified"
        elif revisit_candidates:
            status = POSSIBLE_REVISIT
            uncertainty = "visual_similarity_does_not_verify_place_identity"
        elif related_active:
            status = OBSERVED
            uncertainty = "landmark_observed_passage_unverified"
        else:
            status = COMPLETION_UNKNOWN
            uncertainty = "no_verified_instruction_progress"

        evidence_ids = tuple(dict.fromkeys((
            *recent_ids, *(record.key for record in related_history),
            *(candidate.current_evidence_id for candidate in revisit_candidates),
            *(ref for candidate in revisit_candidates for ref in candidate.historical_evidence_ids),
            *progress_refs,
        )))
        self.state = NavigationState(
            instruction=previous.instruction, scene_id=previous.scene_id,
            episode_id=previous.episode_id, rollout_idx=previous.rollout_idx,
            instruction_anchors=previous.instruction_anchors,
            active_anchor=active_anchor, anchor_evidence=anchor_evidence,
            anchor_status={anchor: OBSERVED if refs else NOT_OBSERVED
                           for anchor, refs in anchor_evidence.items()},
            recent_observation_ids=recent_ids,
            historical_landmark_ids=tuple(record.key for record in related_history),
            revisit_candidates=tuple(revisit_candidates),
            repeated_observation_count=len(repeated_observations),
            progress_status=status, uncertainty=uncertainty,
            evidence_ids=evidence_ids,
            evidence_provenance=tuple(EvidenceProvenance.from_record(by_key[ref])
                                      for ref in evidence_ids),
            progress_evidence_ids=tuple(dict.fromkeys(progress_refs)),
            last_update_step=step,
        )
        return self.state
