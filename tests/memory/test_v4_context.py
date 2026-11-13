"""CPU checks for V4's provenance-preserving conversion of V3 candidates."""

from __future__ import annotations

import json

from lightnav.memory.v3_retrieval import RetrievalResult
from lightnav.memory.v3_schemas import EvidenceRecord, RetrievedMemory
from lightnav.memory.v3_store import V3MemoryStore
from lightnav.memory.v4_context import build_navigation_memory_context
from lightnav.memory.v4_navigation_state import NavigationStateUpdater


def _store(*, persistent: bool = False) -> V3MemoryStore:
    return V3MemoryStore(scope="scene_persistent" if persistent else "episode_local")


def _record(store: V3MemoryStore, *, episode: str, step: int, evidence_id: str,
            event_type: str = "object_observation", label: str | None = "door",
            payload_description: str | None = None,
            descriptor: tuple[float, ...] | None = None,
            observation_status: str = "observed") -> EvidenceRecord:
    return store.add(EvidenceRecord(
        evidence_id=evidence_id, scene_id="scene", episode_id=episode,
        rollout_idx=0, step=step, event_type=event_type, semantic_label=label,
        payload={"description": payload_description or f"Observed {label}"},
        visual_descriptor=descriptor, observation_status=observation_status,
        detector_name="synthetic_rgb" if event_type in {"object_observation", "landmark"}
        else None,
    ))


def _result(*memories: RetrievedMemory) -> RetrievalResult:
    return RetrievalResult(
        records=tuple(memories), channel_candidates={
            "visual": 0, "semantic": len(memories), "spatial": 0, "episodic": 0,
        }, merged_candidates=len(memories), final_top_k=len(memories),
    )


def _candidate(record: EvidenceRecord, *, kind: str = "observation",
               memory_id: str | None = None) -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=memory_id or f"evidence:{record.key}", kind=kind,
        scene_id=record.scene_id, description="untrusted V3 summary",
        evidence_ids=(record.key,), source_step=record.step,
    )


def _state(store: V3MemoryStore, *, episode: str, instruction: str,
           step: int):
    updater = NavigationStateUpdater(store)
    updater.reset_episode("scene", episode, instruction)
    return updater.update(step)


def test_current_and_historical_observations_keep_exact_episode_step() -> None:
    store = _store(persistent=True)
    store.begin_episode("scene", "older", "Find the door")
    old = _record(store, episode="older", step=0, evidence_id="seen")
    store.close_episode()
    store.begin_episode("scene", "now", "Find the door")
    current = _record(store, episode="now", step=0, evidence_id="seen")
    context = build_navigation_memory_context(
        _result(_candidate(current), _candidate(old)),
        store, _state(store, episode="now", instruction="Find the door", step=0),
    )
    assert [fact.evidence_ids for fact in context.current_evidence] == [(current.key,)]
    assert [fact.evidence_ids for fact in context.history_evidence] == [(old.key,)]
    assert context.history_evidence[0].provenance[0].episode_id == "older"
    assert context.history_evidence[0].provenance[0].step == 0
    assert context.current_evidence[0].provenance[0].episode_id == "now"
    assert len(context.evidence_provenance) == 2


def test_same_evidence_from_direct_and_node_candidate_is_atomic() -> None:
    store = _store()
    store.begin_episode("scene", "one", "Find the door")
    old = _record(store, episode="one", step=0, evidence_id="door_0")
    _record(store, episode="one", step=1, evidence_id="door_1")
    node = RetrievedMemory(
        memory_id='node:["scene","landmark_0000"]', kind="place", scene_id="scene",
        description="untrusted node summary", evidence_ids=(old.key,),
    )
    context = build_navigation_memory_context(
        _result(_candidate(old), node), store,
        _state(store, episode="one", instruction="Find the door", step=1),
    )
    assert [fact.evidence_ids for fact in context.history_evidence] == [(old.key,)]
    assert [fact.evidence_ids for fact in context.ordered_facts] == [(old.key,)]
    assert any(item.reason == "duplicate_evidence" for item in context.dropped_candidates)


def test_relevant_landmark_recalled_and_unrelated_history_excluded() -> None:
    store = _store()
    store.begin_episode("scene", "one", "Find the door")
    door = _record(store, episode="one", step=0, evidence_id="door_0")
    chair = _record(store, episode="one", step=0, evidence_id="chair_0", label="chair")
    context = build_navigation_memory_context(
        _result(_candidate(chair), _candidate(door)), store,
        _state(store, episode="one", instruction="Find the door", step=1),
    )
    assert [fact.semantic_label for fact in context.historical_landmarks] == ["door"]
    assert [fact.semantic_label for fact in context.history_evidence] == ["door"]
    assert any(item.reason == "unrelated_history" and chair.key in item.evidence_ids
               for item in context.dropped_candidates)


def test_place_visit_is_an_observed_hypothesis_not_a_verified_visit() -> None:
    store = _store()
    store.begin_episode("scene", "one", "Find the bedroom")
    place = _record(
        store, episode="one", step=0, evidence_id="place_label",
        event_type="place_visit", label=None,
        payload_description="Extractor reported bedroom (place hypothesis)",
    )
    context = build_navigation_memory_context(
        _result(_candidate(place)), store,
        _state(store, episode="one", instruction="Find the bedroom", step=1),
    )
    assert len(context.history_evidence) == 1
    fact = context.history_evidence[0]
    assert fact.semantic_label == "bedroom"
    assert fact.observation_status == "observed"
    assert fact.visit_status == "UNKNOWN"
    assert fact.traversability_status == "unknown"
    assert "unverified" in fact.description


def test_inferred_spatial_relation_stays_unknown_and_preserves_both_sources() -> None:
    store = _store()
    store.begin_episode("scene", "one", "Find the door")
    door = _record(store, episode="one", step=0, evidence_id="door", label="door")
    hall = _record(store, episode="one", step=1, evidence_id="hall", label="hall")
    n_door = store.add_landmark(door)
    n_hall = store.add_landmark(hall)
    edge = store.add_relation(
        n_door.node_id, n_hall.node_id, "inferred_relation",
        evidence_keys=(door.key, hall.key),
    )
    memory = RetrievedMemory(
        memory_id=f"edge:{json.dumps([edge.scene_id, edge.edge_id], separators=(',', ':'))}",
        kind="inferred_relation", scene_id="scene", description="untrusted traversed claim",
        evidence_ids=(door.key, hall.key), source_step=1,
    )
    context = build_navigation_memory_context(
        _result(memory), store,
        _state(store, episode="one", instruction="Find the door", step=2),
    )
    assert len(context.uncertain_relations) == 1
    relation = context.uncertain_relations[0]
    assert relation.evidence_ids == (door.key, hall.key)
    assert relation.spatial_status == "inferred"
    assert relation.traversability_status == "unknown"
    assert relation.visit_status == "UNKNOWN"
    assert context.instruction_progress == ()


def test_issued_forward_command_is_not_movement_or_progress() -> None:
    store = _store()
    store.begin_episode("scene", "one", "Find the door")
    issued = _record(
        store, episode="one", step=0, evidence_id="action_0", event_type="action",
        label=None, payload_description="Issued velocity control before this view",
        observation_status="unknown",
    )
    context = build_navigation_memory_context(
        _result(_candidate(issued, kind="action_history")), store,
        _state(store, episode="one", instruction="Find the door", step=1),
    )
    assert len(context.issued_commands) == 1
    assert context.issued_commands[0].visit_status == "UNKNOWN"
    assert context.issued_commands[0].traversability_status == "unknown"
    assert context.instruction_progress == ()
    assert context.history_evidence == ()


def test_visual_revisit_candidate_keeps_identity_unknown_and_provenance() -> None:
    store = _store(persistent=True)
    store.begin_episode("scene", "older", "Find the door")
    prior_place = _record(
        store, episode="older", step=0, evidence_id="place_0",
        event_type="place_observation", label=None, descriptor=(1.0, 0.0),
    )
    _record(store, episode="older", step=0, evidence_id="door_0")
    store.close_episode()
    store.begin_episode("scene", "now", "Find the door")
    now_place = _record(
        store, episode="now", step=0, evidence_id="place_0",
        event_type="place_observation", label=None, descriptor=(0.99, 0.01),
    )
    _record(store, episode="now", step=0, evidence_id="door_0")
    state = _state(store, episode="now", instruction="Find the door", step=0)
    assert state.revisit_candidates
    context = build_navigation_memory_context(
        _result(_candidate(now_place), _candidate(prior_place)), store, state,
    )
    assert len(context.revisit_evidence) == 1
    revisit = context.revisit_evidence[0]
    assert revisit.identity_status == "unknown"
    assert revisit.instruction_related
    assert revisit.current_evidence_id == now_place.key
    assert revisit.historical_evidence_ids == (prior_place.key,)
    assert {(source.episode_id, source.step) for source in revisit.provenance} == {
        ("older", 0), ("now", 0),
    }


def test_conflicting_evidence_is_not_merged_by_shared_label() -> None:
    store = _store()
    store.begin_episode("scene", "one", "Find the door")
    first = _record(
        store, episode="one", step=0, evidence_id="door_open", label="door",
        payload_description="Observed open door",
    )
    second = _record(
        store, episode="one", step=1, evidence_id="door_closed", label="door",
        payload_description="Observed closed door",
    )
    context = build_navigation_memory_context(
        _result(_candidate(first), _candidate(second)), store,
        _state(store, episode="one", instruction="Find the door", step=2),
    )
    assert len(context.history_evidence) == 2
    assert {fact.source_description for fact in context.history_evidence} == {
        "Observed open door", "Observed closed door",
    }
    assert {fact.evidence_ids for fact in context.history_evidence} == {
        (first.key,), (second.key,),
    }
    assert len(context.ordered_facts) == 2
