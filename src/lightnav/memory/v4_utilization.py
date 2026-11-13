"""Select when stored history adds information to LightNav's current RGB sample.

This is a use decision, not a second navigation policy. It neither writes
memory nor chooses a movement. A prior step is visually redundant only when
its exact frame ID is known and that ID is in the current processor sample.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable, Mapping

from lightnav.memory.v4_context import (
    DroppedCandidate,
    NavigationFact,
    NavigationMemoryContext,
    RevisitEvidence,
)
from lightnav.memory.v4_navigation_state import NavigationState

NONE = "NONE"
HISTORY_REFERENCE = "HISTORY_REFERENCE"
REVISIT_AUDIT = "REVISIT_AUDIT"
PROGRESS_AUDIT = "PROGRESS_AUDIT"

_VISUAL_EVENTS = frozenset({
    "object_observation", "landmark", "place_observation", "place_visit",
    "place_hypothesis",
})
_PASSAGE_TERMS = frozenset({"through", "pass", "past", "enter", "exit", "cross", "across"})
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)

# The mapping is supplied by the Policy from TrackingAgent's absolute frame
# IDs, after each act() observation. Cross-Episode records have no map entry.
FrameKey = tuple[str, str, int, int]  # scene, episode, rollout, memory step


def _visually_duplicated(
    fact: NavigationFact, state: NavigationState,
    visible_frame_ids: frozenset[int], frame_ids_by_step: Mapping[FrameKey, int],
) -> bool:
    """Prove RGB duplication by identity and exact sample frame membership."""
    if fact.kind not in _VISUAL_EVENTS or not fact.provenance:
        return False
    for source in fact.provenance:
        if (source.scene_id, source.episode_id, source.rollout_idx) != (
            state.scene_id, state.episode_id, state.rollout_idx
        ):
            return False
        frame_id = frame_ids_by_step.get((
            source.scene_id, source.episode_id, source.rollout_idx, source.step,
        ))
        if frame_id is None or frame_id not in visible_frame_ids:
            return False
    return True


def _fact_key(fact: NavigationFact) -> tuple[str, ...]:
    return fact.evidence_ids


def _revisit_key(candidate: RevisitEvidence) -> tuple[str, ...]:
    return (candidate.current_evidence_id, *candidate.historical_evidence_ids)


def _revisit_signature(candidate: RevisitEvidence) -> tuple[str, ...]:
    # A new current RGB record alone does not make the same historical revisit
    # useful again at every step. New historical support can trigger a new audit.
    return tuple(sorted(candidate.historical_evidence_ids))


class MemoryUtilizationManager:
    """Choose one evidence use mode and suppress unchanged repeated context."""

    def __init__(self) -> None:
        self._episode: tuple[str, str, int] | None = None
        self._last_signatures: dict[str, tuple[tuple[str, ...], ...]] = {}
        self._reserved_mode: str | None = None
        self._reserved_evidence_ids: set[str] = set()
        self._injected_evidence_ids: set[str] = set()

    def reset_episode(self, scene_id: str, episode_id: str, rollout_idx: int) -> None:
        self._episode = (scene_id, episode_id, rollout_idx)
        self._last_signatures = {}
        self._reserved_mode = None
        self._reserved_evidence_ids = set()
        self._injected_evidence_ids = set()

    def confirm_injected_evidence(self, evidence_ids: Iterable[str]) -> None:
        """Commit only the source IDs verified in final model input."""
        confirmed = set(evidence_ids)
        if (self._reserved_mode is None or not confirmed
                or not confirmed.issubset(self._reserved_evidence_ids)):
            raise ValueError("injected evidence must belong to the selected V4 context")
        self._injected_evidence_ids.update(confirmed)
        self._reserved_mode = None
        self._reserved_evidence_ids.clear()

    def discard_uninjected_selection(self) -> None:
        """Permit a later retry when the chosen text never reached the model."""
        if self._reserved_mode is not None:
            self._last_signatures.pop(self._reserved_mode, None)
            self._reserved_mode = None
            self._reserved_evidence_ids.clear()

    def select(
        self, context: NavigationMemoryContext, state: NavigationState, *,
        visible_frame_ids: set[int] | frozenset[int] | tuple[int, ...] = (),
        frame_ids_by_step: Mapping[FrameKey, int] | None = None,
    ) -> NavigationMemoryContext:
        identity = (state.scene_id, state.episode_id, state.rollout_idx)
        if self._episode != identity:
            raise ValueError("reset utilization for the active Scene and Episode")
        visible = frozenset(visible_frame_ids)
        mapping = frame_ids_by_step or {}
        dropped = list(context.dropped_candidates)
        useful: list[NavigationFact] = []
        for fact in context.ordered_facts:
            if not fact.instruction_relevant:
                dropped.append(DroppedCandidate(
                    fact.source_memory_id, fact.evidence_ids, "unrelated_instruction",
                ))
                continue
            if fact.temporal_role == "CURRENT":
                dropped.append(DroppedCandidate(
                    fact.source_memory_id, fact.evidence_ids, "current_observation",
                ))
                continue
            if fact.evidence_ids and set(fact.evidence_ids).issubset(
                self._injected_evidence_ids
            ):
                dropped.append(DroppedCandidate(
                    fact.source_memory_id, fact.evidence_ids,
                    "repeated_without_new_evidence",
                ))
                continue
            if fact.kind in _VISUAL_EVENTS:
                expected_status = "inferred" if fact.kind == "place_hypothesis" else "observed"
                if fact.observation_status != expected_status:
                    dropped.append(DroppedCandidate(
                        fact.source_memory_id, fact.evidence_ids, "unverified_observation",
                    ))
                    continue
                if _visually_duplicated(fact, state, visible, mapping):
                    dropped.append(DroppedCandidate(
                        fact.source_memory_id, fact.evidence_ids, "visible_in_native_rgb",
                    ))
                    continue
                useful.append(fact)
            elif fact.kind == "verified_traversal":
                if fact in context.instruction_progress:
                    useful.append(fact)
                else:
                    dropped.append(DroppedCandidate(
                        fact.source_memory_id, fact.evidence_ids, "unverified_progress_link",
                    ))
            elif fact in context.uncertain_relations:
                if fact.temporal_role in {"HISTORICAL", "MIXED"}:
                    useful.append(fact)
                else:
                    dropped.append(DroppedCandidate(
                        fact.source_memory_id, fact.evidence_ids, "current_relation",
                    ))

        credible_revisits: list[RevisitEvidence] = []
        for candidate in context.revisit_evidence:
            if candidate.historical_evidence_ids and set(
                candidate.historical_evidence_ids
            ).issubset(self._injected_evidence_ids):
                dropped.append(DroppedCandidate(
                    "revisit", _revisit_key(candidate), "repeated_without_new_evidence",
                ))
                continue
            if candidate.identity_status != "unknown" or not candidate.instruction_related:
                dropped.append(DroppedCandidate(
                    "revisit", _revisit_key(candidate), "unrelated_or_overstated_revisit",
                ))
                continue
            if candidate.independent_history_observations < 2:
                dropped.append(DroppedCandidate(
                    "revisit", _revisit_key(candidate), "insufficient_independent_views",
                ))
                continue
            credible_revisits.append(candidate)

        progress = [fact for fact in useful if fact.kind == "verified_traversal"]
        passage_instruction = bool(
            set(_WORDS.findall(state.instruction.casefold())) & _PASSAGE_TERMS
        )
        historical_observations = [fact for fact in useful if fact.kind in _VISUAL_EVENTS]
        historical_relations = [fact for fact in useful if fact in context.uncertain_relations]

        options: list[tuple[str, tuple[NavigationFact, ...], tuple[RevisitEvidence, ...]]] = []
        if credible_revisits:
            options.append((REVISIT_AUDIT, (), tuple(credible_revisits)))
        if progress or (passage_instruction and historical_observations):
            progress_facts = tuple((*progress, *(
                fact for fact in historical_observations if fact not in progress
            )))
            options.append((PROGRESS_AUDIT, progress_facts, ()))
        if historical_observations or historical_relations:
            history_facts = tuple(fact for fact in useful if (
                fact in historical_observations or fact in historical_relations
            ))
            options.append((HISTORY_REFERENCE, history_facts, ()))

        mode, selected_facts, selected_revisits = NONE, (), ()
        for candidate_mode, candidate_facts, candidate_revisits in options:
            signature = tuple(
                [_fact_key(fact) for fact in candidate_facts]
                + [_revisit_signature(item) for item in candidate_revisits]
            )
            if signature == self._last_signatures.get(candidate_mode):
                dropped.extend(DroppedCandidate(
                    fact.source_memory_id, fact.evidence_ids, "repeated_without_new_evidence",
                ) for fact in candidate_facts)
                dropped.extend(DroppedCandidate(
                    "revisit", _revisit_key(item), "repeated_without_new_evidence",
                ) for item in candidate_revisits)
                continue
            mode, selected_facts, selected_revisits = (
                candidate_mode, candidate_facts, candidate_revisits,
            )
            self._last_signatures[mode] = signature
            self._reserved_mode = mode
            self._reserved_evidence_ids = {
                evidence_id for fact in candidate_facts for evidence_id in fact.evidence_ids
            } | {
                evidence_id for revisit in candidate_revisits
                for evidence_id in _revisit_key(revisit)
            }
            break

        return replace(
            context, mode=mode, selected_facts=selected_facts,
            selected_revisits=selected_revisits, dropped_candidates=tuple(dropped),
        )
