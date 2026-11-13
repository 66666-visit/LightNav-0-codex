"""V4 use decisions require useful history and exact proof of RGB duplication."""

from __future__ import annotations

from lightnav.memory.v4_context import (
    NavigationFact,
    NavigationMemoryContext,
    RevisitEvidence,
)
from lightnav.memory.v4_navigation_state import EvidenceProvenance, NavigationState
from lightnav.memory.v4_utilization import (
    HISTORY_REFERENCE,
    NONE,
    PROGRESS_AUDIT,
    REVISIT_AUDIT,
    MemoryUtilizationManager,
)


def _source(event: str, *, episode: str = "active", step: int = 1) -> EvidenceProvenance:
    return EvidenceProvenance(
        evidence_id=event, scene_id="scene", episode_id=episode,
        rollout_idx=0, step=step,
    )


def _fact(event: str, *, episode: str = "active", step: int = 1,
          kind: str = "landmark", relevant: bool = True,
          confidence: float | None = 0.8) -> NavigationFact:
    source = _source(event, episode=episode, step=step)
    return NavigationFact(
        kind=kind, semantic_label="door", description="RGB detector reported door",
        evidence_ids=(event,), provenance=(source,), source_memory_id=f"evidence:{event}",
        temporal_role="HISTORICAL", observation_status="observed",
        visit_status="UNKNOWN", spatial_status="unknown",
        traversability_status="unknown", instruction_relevant=relevant,
        source="mock", raw_detector_confidence=confidence,
    )


def _state(instruction: str = "Find the door") -> NavigationState:
    return NavigationState(
        instruction=instruction, scene_id="scene", episode_id="active",
        rollout_idx=0, last_update_step=3,
    )


def _manager() -> MemoryUtilizationManager:
    manager = MemoryUtilizationManager()
    manager.reset_episode("scene", "active", 0)
    return manager


def _context(*facts: NavigationFact,
             revisits: tuple[RevisitEvidence, ...] = ()) -> NavigationMemoryContext:
    return NavigationMemoryContext(
        history_evidence=facts, historical_landmarks=facts,
        ordered_facts=facts, revisit_evidence=revisits,
    )


def test_none_has_no_selected_context() -> None:
    result = _manager().select(_context(), _state())
    assert result.mode == NONE
    assert result.selected_facts == ()
    assert result.selected_revisits == ()


def test_exact_same_episode_frame_in_sample_is_visually_redundant() -> None:
    fact = _fact("old-door")
    result = _manager().select(
        _context(fact), _state(), visible_frame_ids={12, 13},
        frame_ids_by_step={("scene", "active", 0, 1): 12},
    )
    assert result.mode == NONE
    assert any(drop.reason == "visible_in_native_rgb" for drop in result.dropped_candidates)


def test_step_distance_and_similarity_do_not_prove_visual_duplication() -> None:
    fact = _fact("old-door", step=1)
    result = _manager().select(
        _context(fact), _state(), visible_frame_ids={99},
        frame_ids_by_step={("scene", "active", 0, 1): 12},
    )
    assert result.mode == HISTORY_REFERENCE
    assert result.selected_facts == (fact,)
    without_map = _manager().select(
        _context(fact), _state(), visible_frame_ids={12}, frame_ids_by_step={},
    )
    assert without_map.mode == HISTORY_REFERENCE


def test_cross_episode_step_collision_is_still_useful_history() -> None:
    fact = _fact("past-door", episode="past", step=1)
    result = _manager().select(
        _context(fact), _state(), visible_frame_ids={12},
        frame_ids_by_step={("scene", "active", 0, 1): 12},
    )
    assert result.mode == HISTORY_REFERENCE


def test_passage_word_selects_progress_audit_without_claiming_completion() -> None:
    fact = _fact("old-door", confidence=0.25)
    result = _manager().select(_context(fact), _state("Go through the door"))
    assert result.mode == PROGRESS_AUDIT
    assert result.selected_facts == (fact,)
    assert fact.visit_status == "UNKNOWN"
    assert fact.traversability_status == "unknown"


def test_verified_relation_must_be_linked_to_instruction_progress() -> None:
    source = _source("motion", step=2)
    relation = NavigationFact(
        kind="verified_traversal", semantic_label=None,
        description="Measured traversal between door and hall; completion unknown",
        evidence_ids=("motion",), provenance=(source,), source_memory_id="edge:one",
        temporal_role="HISTORICAL", observation_status="unknown",
        visit_status="VERIFIED", spatial_status="traversed",
        traversability_status="verified", instruction_relevant=True,
        source="verified_traversal",
    )
    context = NavigationMemoryContext(
        ordered_facts=(relation,), instruction_progress=(relation,),
    )
    result = _manager().select(context, _state())
    assert result.mode == PROGRESS_AUDIT
    assert result.selected_facts == (relation,)
    unlinked = _manager().select(
        NavigationMemoryContext(ordered_facts=(relation,)), _state(),
    )
    assert unlinked.mode == NONE
    assert any(drop.reason == "unverified_progress_link"
               for drop in unlinked.dropped_candidates)


def test_progress_mode_keeps_historical_observation_as_fallback() -> None:
    source = _source("motion", step=2)
    progress = NavigationFact(
        kind="verified_traversal", semantic_label=None,
        description="Measured traversal between door and hall",
        evidence_ids=("motion",), provenance=(source,), source_memory_id="edge:one",
        temporal_role="HISTORICAL", observation_status="unknown",
        visit_status="VERIFIED", spatial_status="traversed",
        traversability_status="verified", instruction_relevant=True,
        source="verified_traversal",
    )
    observation = _fact("earlier-door", step=0)
    selected = _manager().select(NavigationMemoryContext(
        ordered_facts=(progress, observation), instruction_progress=(progress,),
        history_evidence=(observation,),
    ), _state())
    assert selected.mode == PROGRESS_AUDIT
    assert selected.selected_facts == (progress, observation)


def test_revisit_requires_two_independent_historical_observations() -> None:
    sources = (_source("now", step=3), _source("old-1", step=0),
               _source("old-2", step=1))
    candidate = RevisitEvidence(
        current_evidence_id="now", historical_evidence_ids=("old-1", "old-2"),
        provenance=sources, independent_history_observations=2,
        highest_visual_similarity=0.94, instruction_related=True,
    )
    result = _manager().select(_context(revisits=(candidate,)), _state())
    assert result.mode == REVISIT_AUDIT
    assert result.selected_revisits == (candidate,)
    weak = RevisitEvidence(
        current_evidence_id="now", historical_evidence_ids=("old-1",),
        provenance=sources[:2], independent_history_observations=1,
        highest_visual_similarity=0.99, instruction_related=True,
    )
    rejected = _manager().select(_context(revisits=(weak,)), _state())
    assert rejected.mode == NONE
    assert any(drop.reason == "insufficient_independent_views"
               for drop in rejected.dropped_candidates)


def test_multiple_revisit_candidates_remain_available_for_budget_retry() -> None:
    def candidate(name: str) -> RevisitEvidence:
        refs = (_source(f"{name}-now", step=3),
                _source(f"{name}-old-0", step=0),
                _source(f"{name}-old-1", step=1))
        return RevisitEvidence(
            current_evidence_id=refs[0].evidence_id,
            historical_evidence_ids=(refs[1].evidence_id, refs[2].evidence_id),
            provenance=refs, independent_history_observations=2,
            highest_visual_similarity=0.9, instruction_related=True,
        )

    first, second = candidate("long"), candidate("short")
    result = _manager().select(_context(revisits=(first, second)), _state())
    assert result.mode == REVISIT_AUDIT
    assert result.selected_revisits == (first, second)


def test_same_historical_revisit_is_not_injected_every_new_rgb_step() -> None:
    manager = _manager()
    old = (_source("old-1", step=0), _source("old-2", step=1))

    def candidate(current: str) -> RevisitEvidence:
        now = _source(current, step=3)
        return RevisitEvidence(
            current_evidence_id=now.evidence_id,
            historical_evidence_ids=tuple(source.evidence_id for source in old),
            provenance=(now, *old), independent_history_observations=2,
            highest_visual_similarity=0.9, instruction_related=True,
        )

    assert manager.select(_context(revisits=(candidate("now-3"),)), _state()).mode == (
        REVISIT_AUDIT
    )
    assert manager.select(_context(revisits=(candidate("now-4"),)), _state()).mode == NONE


def test_repeated_revisit_can_yield_new_history_without_alternating_modes() -> None:
    manager = _manager()
    old = (_source("old-1", step=0), _source("old-2", step=1))

    def candidate(current: str) -> RevisitEvidence:
        now = _source(current, step=3)
        return RevisitEvidence(
            current_evidence_id=now.evidence_id,
            historical_evidence_ids=tuple(source.evidence_id for source in old),
            provenance=(now, *old), independent_history_observations=2,
            highest_visual_similarity=0.9, instruction_related=True,
        )

    assert manager.select(_context(revisits=(candidate("now-3"),)), _state()).mode == (
        REVISIT_AUDIT
    )
    new_landmark = _fact("new-door", step=2)
    with_new_history = _context(new_landmark, revisits=(candidate("now-4"),))
    assert manager.select(with_new_history, _state()).mode == HISTORY_REFERENCE
    assert manager.select(with_new_history, _state()).mode == NONE


def test_issued_actions_and_unrelated_history_do_not_enter_prompt() -> None:
    unrelated = _fact("lamp", relevant=False)
    issued = _fact("command", kind="action")
    result = _manager().select(
        NavigationMemoryContext(
            ordered_facts=(unrelated,), issued_commands=(issued,),
        ), _state("Go through the door"),
    )
    assert result.mode == NONE
    assert result.selected_facts == ()
    assert any(drop.reason == "unrelated_instruction" for drop in result.dropped_candidates)


def test_unchanged_context_is_suppressed_until_new_evidence() -> None:
    manager = _manager()
    first = _fact("old-door")
    second = _fact("new-door", step=2)
    assert manager.select(_context(first), _state()).mode == HISTORY_REFERENCE
    repeated = manager.select(_context(first), _state()).mode
    assert repeated == NONE
    assert manager.select(_context(first, second), _state()).mode == HISTORY_REFERENCE
    manager.reset_episode("scene", "active", 0)
    assert manager.select(_context(first), _state()).mode == HISTORY_REFERENCE
