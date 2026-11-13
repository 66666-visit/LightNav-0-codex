"""The V4 formatter emits complete, source-bound facts within a small budget."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_context_builder import build_context_text
from lightnav.memory.v3_schemas import canonical_evidence_key
from lightnav.memory.v4_config import RGMV4Config
from lightnav.memory.v4_context import (
    NavigationFact,
    NavigationMemoryContext,
    RevisitEvidence,
)
from lightnav.memory.v4_formatter import V4PromptFormatter
from lightnav.memory.v4_navigation_state import EvidenceProvenance
from lightnav.memory.v4_utilization import (
    HISTORY_REFERENCE,
    PROGRESS_AUDIT,
    REVISIT_AUDIT,
)


def _source(event: str, *, episode: str = "old", step: int = 2) -> EvidenceProvenance:
    return EvidenceProvenance(
        evidence_id=canonical_evidence_key("scene", episode, 0, event),
        scene_id="scene", episode_id=episode, rollout_idx=0, step=step,
    )


def _fact(event: str, *, label: str = "door", episode: str = "old", step: int = 2,
          kind: str = "object_observation") -> NavigationFact:
    source = _source(event, episode=episode, step=step)
    return NavigationFact(
        kind=kind, semantic_label=label,
        description=f"RGB detector reported {label}",
        evidence_ids=(source.evidence_id,), provenance=(source,),
        source_memory_id=f"memory:{event}", temporal_role="HISTORICAL",
        observation_status="observed", visit_status="UNKNOWN",
        spatial_status="unknown", traversability_status="unknown",
        instruction_relevant=True, source="synthetic_rgb",
    )


def _context(mode: str, *facts: NavigationFact) -> NavigationMemoryContext:
    return NavigationMemoryContext(mode=mode, selected_facts=tuple(facts))


def _ref(source: EvidenceProvenance) -> str:
    return sha256(source.evidence_id.encode()).hexdigest()[:12]


def test_none_and_empty_selection_return_exact_no_context() -> None:
    formatter = V4PromptFormatter()
    assert formatter.format(_context("NONE", _fact("door"))) is None
    assert formatter.last_lines == []
    assert formatter.last_serialized == 0
    assert formatter.format(_context(HISTORY_REFERENCE)) is None


def test_history_statement_has_status_and_canonical_episode_step_provenance() -> None:
    formatter = V4PromptFormatter()
    fact = _fact("door_0", episode="earlier", step=7)
    output = formatter.format(_context(HISTORY_REFERENCE, fact))
    assert output is not None
    assert output.text.startswith("[Navigation Memory]\n")
    assert "[OBSERVED] Earlier RGB detector reported door" in output.text
    assert "visit UNKNOWN" in output.text
    assert 'Episode="earlier", Rollout=0, Step=7' in output.text
    assert f"Ref={_ref(fact.provenance[0])}" in output.text
    assert fact.evidence_ids[0] not in output.text
    assert output.layer_items[0].evidence_refs[0].event_id == fact.evidence_ids[0]
    assert output.layer_items[0].metadata["provenance"][0]["step"] == 7
    assert output.layer_items[0].metadata["provenance"][0]["prompt_ref"] == _ref(
        fact.provenance[0]
    )
    assert formatter.last_lines == [output.layer_items[0].metadata["serialized_text"]]


def test_progress_audit_observation_keeps_passage_and_completion_unknown() -> None:
    formatter = V4PromptFormatter()
    output = formatter.format(_context(PROGRESS_AUDIT, _fact("seen_door")))
    assert output is not None
    assert "passage UNKNOWN" in output.text
    assert "instruction completion UNKNOWN" in output.text
    assert "VERIFIED" not in output.text


def test_issued_command_is_not_formatted_as_verified_progress() -> None:
    formatter = V4PromptFormatter()
    action = replace(_fact("issued", label="door", kind="action"),
                     description="Turn left now", observation_status="unknown")
    assert formatter.format(_context(PROGRESS_AUDIT, action)) is None
    assert formatter.last_dropped[0].reason == "invalid_for_mode"
    assert formatter.last_serialized == 0


def test_verified_traversal_preserves_motion_sources_and_no_completion_claim() -> None:
    formatter = V4PromptFormatter(max_chars=400)
    start = _source("before", step=1)
    motion = _source("measured_motion", step=2)
    end = _source("after", step=3)
    fact = NavigationFact(
        kind="verified_traversal", semantic_label=None,
        description="Measured traversal between door and hall",
        evidence_ids=(start.evidence_id, motion.evidence_id, end.evidence_id),
        provenance=(start, motion, end), source_memory_id="edge:verified",
        temporal_role="HISTORICAL", observation_status="unknown",
        visit_status="VERIFIED", spatial_status="traversed",
        traversability_status="traversed", instruction_relevant=True,
        source="verified_traversal",
    )
    output = formatter.format(_context(PROGRESS_AUDIT, fact))
    assert output is not None
    assert "[VERIFIED]" in output.text
    assert "instruction completion UNKNOWN" in output.text
    assert len(output.layer_items[0].evidence_refs) == 3
    assert all(f"Ref={_ref(source)}" in output.text for source in fact.provenance)
    assert [ref.event_id for ref in output.layer_items[0].evidence_refs] == list(
        fact.evidence_ids
    )


def test_visual_relation_keeps_observed_or_inferred_status_and_unknown_passage() -> None:
    formatter = V4PromptFormatter()
    first = _source("door_view", step=1)
    second = _source("hall_view", step=2)
    inferred = NavigationFact(
        kind="inferred_relation", semantic_label=None,
        description="Visual relation between door and hall; passage unknown",
        evidence_ids=(first.evidence_id, second.evidence_id),
        provenance=(first, second), source_memory_id="edge:inferred",
        temporal_role="HISTORICAL", observation_status="inferred",
        visit_status="UNKNOWN", spatial_status="inferred",
        traversability_status="unknown", instruction_relevant=True,
        source="inferred_relation",
    )
    inferred_output = formatter.format(_context(HISTORY_REFERENCE, inferred))
    assert inferred_output is not None
    assert "[INFERRED]" in inferred_output.text
    assert "passage UNKNOWN" in inferred_output.text
    observed = replace(inferred, kind="observed_transition", spatial_status="observed")
    observed_output = formatter.format(_context(HISTORY_REFERENCE, observed))
    assert observed_output is not None
    assert "[OBSERVED]" in observed_output.text
    assert "passage UNKNOWN" in observed_output.text


def test_revisit_audit_separates_current_and_history_without_identity_claim() -> None:
    formatter = V4PromptFormatter()
    current = _source("current_view", episode="now", step=9)
    prior_1 = _source("prior_view_1", episode="old", step=2)
    prior_2 = _source("prior_view_2", episode="old", step=4)
    candidate = RevisitEvidence(
        current_evidence_id=current.evidence_id,
        historical_evidence_ids=(prior_1.evidence_id, prior_2.evidence_id),
        provenance=(current, prior_1, prior_2),
        independent_history_observations=2, highest_visual_similarity=0.91,
        instruction_related=True,
    )
    context = NavigationMemoryContext(
        mode=REVISIT_AUDIT, selected_revisits=(candidate,),
    )
    output = formatter.format(context)
    assert output is not None
    assert "[INFERRED] Possible revisit" in output.text
    assert "place identity UNKNOWN" in output.text
    assert "current observation:" in output.text
    assert "historical observations:" in output.text
    assert output.text.index(f"Ref={_ref(current)}") < output.text.index(
        f"Ref={_ref(prior_1)}"
    )
    assert len(output.layer_items[0].evidence_refs) == 3


def test_realistic_revisit_keys_fit_default_budget_with_full_diagnostic_ids() -> None:
    scene = "mp3d/abc123/abc123.glb"
    sources = tuple(EvidenceProvenance(
        evidence_id=canonical_evidence_key(
            scene, "episode_000123", 0, f"event:step:{step:04d}:place_observation",
        ),
        scene_id=scene, episode_id="episode_000123", rollout_idx=0, step=step,
    ) for step in (9, 2, 4))
    candidate = RevisitEvidence(
        current_evidence_id=sources[0].evidence_id,
        historical_evidence_ids=tuple(source.evidence_id for source in sources[1:]),
        provenance=sources, independent_history_observations=2,
        highest_visual_similarity=0.92, instruction_related=True,
    )
    output = V4PromptFormatter().format(NavigationMemoryContext(
        mode=REVISIT_AUDIT, selected_revisits=(candidate,),
    ))
    assert output is not None
    assert len(build_context_text(output, max_chars=400)) <= 400
    assert [ref.event_id for ref in output.layer_items[0].evidence_refs] == [
        source.evidence_id for source in sources
    ]
    assert all(f"Ref={_ref(source)}" in output.text for source in sources)


def test_skip_overlong_complete_item_and_try_later_candidate() -> None:
    formatter = V4PromptFormatter(max_chars=210)
    too_long = _fact("long", label="door " + "x" * 180)
    short = _fact("short", label="door")
    output = formatter.format(_context(HISTORY_REFERENCE, too_long, short))
    assert output is not None
    assert f"Ref={_ref(too_long.provenance[0])}" not in output.text
    assert f"Ref={_ref(short.provenance[0])}" in output.text
    assert formatter.last_serialized == 1
    assert [item.reason for item in formatter.last_dropped] == ["char_budget"]


def test_overlong_revisit_allows_later_independent_candidate() -> None:
    def revisit(tag: str, history_episode: str = "old") -> RevisitEvidence:
        current = _source(f"{tag}_now", episode="now", step=8)
        before = _source(f"{tag}_before", episode=history_episode, step=1)
        after = _source(f"{tag}_after", episode=history_episode, step=3)
        return RevisitEvidence(
            current_evidence_id=current.evidence_id,
            historical_evidence_ids=(before.evidence_id, after.evidence_id),
            provenance=(current, before, after),
            independent_history_observations=2,
            highest_visual_similarity=0.9, instruction_related=True,
        )

    oversized = revisit("long", history_episode="x" * 110)
    concise = revisit("door")
    formatter = V4PromptFormatter(max_chars=400)
    output = formatter.format(NavigationMemoryContext(
        mode=REVISIT_AUDIT, selected_revisits=(oversized, concise),
    ))
    assert output is not None
    assert f"Ref={_ref(oversized.provenance[0])}" not in output.text
    assert f"Ref={_ref(concise.provenance[0])}" in output.text
    assert any(item.reason == "char_budget" for item in formatter.last_dropped)


def test_overlong_revisit_falls_back_to_useful_historical_landmark() -> None:
    manager = MemoryManager(
        strategy="hierarchical", v4_config=RGMV4Config(enabled=True),
    )
    manager.reset_episode("now", "Find the door", scene_id="scene")
    state = manager.v4_state_updater.update(3)
    current = _source("now_view", episode="now", step=3)
    old_episode = "x" * 120
    history = (
        _source("first_view", episode=old_episode, step=0),
        _source("second_view", episode=old_episode, step=1),
    )
    revisit = RevisitEvidence(
        current_evidence_id=current.evidence_id,
        historical_evidence_ids=tuple(source.evidence_id for source in history),
        provenance=(current, *history), independent_history_observations=2,
        highest_visual_similarity=0.9, instruction_related=True,
    )
    landmark = _fact("door_old", episode="earlier", step=0)
    structured = NavigationMemoryContext(
        ordered_facts=(landmark,), history_evidence=(landmark,),
        historical_landmarks=(landmark,), revisit_evidence=(revisit,),
    )
    manager._v4_structured_context = structured
    selected = manager.v4_utilization.select(structured, state)
    assert selected.mode == REVISIT_AUDIT
    failed = manager.v4_formatter.format(selected)
    assert failed is None
    assert manager.v4_formatter.last_dropped[0].reason == "char_budget"
    fallback, output = manager._v4_history_fallback(selected, failed)
    assert fallback.mode == HISTORY_REFERENCE
    assert output is not None
    assert "Earlier RGB detector reported door" in output.text
    manager.close_episode()


def test_budget_counts_escaped_characters_and_keeps_full_evidence() -> None:
    formatter = V4PromptFormatter(max_chars=240)
    fact = _fact("escaped", label="<video> door")
    output = formatter.format(_context(HISTORY_REFERENCE, fact))
    assert output is not None
    safe = build_context_text(output, max_chars=240)
    assert "&lt;video&gt;" in safe
    assert len(safe) <= 240
    assert f"Ref={_ref(fact.provenance[0])}" in safe
    assert output.layer_items[0].metadata["serialized_text"] in safe


def test_deduplicate_and_exclude_evidence_across_items() -> None:
    formatter = V4PromptFormatter()
    first = _fact("same")
    duplicate = replace(first, source_memory_id="another_ranked_candidate")
    third = _fact("third")
    output = formatter.format(_context(HISTORY_REFERENCE, first, duplicate, third))
    assert output is not None
    assert len(output.layer_items) == 2
    assert [item.reason for item in formatter.last_dropped] == ["duplicate_evidence"]
    retried = formatter.format(
        _context(HISTORY_REFERENCE, first, duplicate, third),
        excluded_evidence_ids=frozenset(first.evidence_ids),
    )
    assert retried is not None
    assert len(retried.layer_items) == 1
    assert f"Ref={_ref(third.provenance[0])}" in retried.text
    assert all(item.reason == "excluded_evidence" for item in formatter.last_dropped)


def test_unrelated_or_current_fact_does_not_enter_prompt() -> None:
    formatter = V4PromptFormatter()
    historical = _fact("unrelated")
    current = replace(_fact("current"), temporal_role="CURRENT")
    assert formatter.format(_context(
        HISTORY_REFERENCE, replace(historical, instruction_relevant=False), current,
    )) is None
    assert [item.reason for item in formatter.last_dropped] == [
        "invalid_for_mode", "invalid_for_mode",
    ]
