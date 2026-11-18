"""L1 organization, L3 organization and joint fusion contracts."""

from __future__ import annotations

from html import escape

import numpy as np
import pytest

from lightnav.memory.conceptgraphs_lite import (
    CameraIntrinsics,
    GeometryConfig,
    InstanceMask,
    ObjectMap,
    Pose3D,
    RGBDFrame,
)
from lightnav.memory.joint_prompt import JointPromptBuilder
from lightnav.memory.l1_prompt_organizer import L1Fact, L1PromptOrganizer
from lightnav.memory.l3_prompt_organizer import L3Fact, L3Organization, L3PromptOrganizer
from lightnav.memory.schemas import (
    LayerMemoryItem,
    MemoryContext,
    MemoryEvent,
    ObservationEvidence,
)
from lightnav.memory.v3_schemas import EvidenceRecord, canonical_evidence_key
from lightnav.memory.v3_store import V3MemoryStore

_IDENTITY = Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _record(
    event_id: str, step: int, *, label: str = "chair", confidence: float = 0.9,
    episode: str = "episode", kind: str = "object_observation",
) -> EvidenceRecord:
    return EvidenceRecord(
        event_id, "scene", episode, 0, step, kind, semantic_label=label,
        raw_detector_confidence=confidence if kind in {"object_observation", "landmark"} else None,
        detector_name="yolo" if kind in {"object_observation", "landmark"} else None,
        observation_status="observed",
    )


def _event(record: EvidenceRecord, box: list[float] | None = None) -> MemoryEvent:
    return MemoryEvent(
        event_id=record.evidence_id, step_id=record.step,
        kind=record.event_type, description=f"Observed {record.semantic_label}",
        confidence=record.raw_detector_confidence or 0.0, source="yolo",
        episode_id=record.episode_id, metadata={"bbox_xyxy": box} if box else {},
    )


def _frame(*, agent_x: float = 0.0, depth: np.ndarray | None = None) -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.full((8, 8), 2.0, dtype=np.float32) if depth is None else depth,
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
        agent_pose=Pose3D((agent_x, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        camera_extrinsics=_IDENTITY,
    )


def _mask(left: int, right: int) -> np.ndarray:
    value = np.zeros((8, 8), dtype=bool)
    value[2:6, left:right] = True
    return value


def _instance(
    name: str, mask: np.ndarray, step: int, index: int, *,
    confidence: float = 0.9, linked: str | None = None,
) -> InstanceMask:
    return InstanceMask(
        class_name=name, confidence=confidence, mask=mask,
        evidence_id=canonical_evidence_key(
            "scene", "episode", 0, f"l3_seg_{step}_{index}",
        ),
        linked_l1_evidence_id=linked,
    )


# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------


def test_l1_separates_now_from_history_and_folds_duplicate_sightings() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the chair and the door")
    for step in (0, 1):
        store.add(_record(f"chair_{step}", step))
    store.add(_record("door_1", 1, label="door"))
    store.add(_record("table_0", 1, label="table"))
    store.add(_record("banana_0", 1, label="banana"))
    current = store.add(_record("chair_3", 3))
    organizer = L1PromptOrganizer(max_items=4)

    organization = organizer.build(
        store, scene_id="scene", episode_id="episode", rollout_idx=0, step=3,
        instruction="find the chair and the door", rgb_width=300,
        events=[_event(current, [110.0, 10.0, 190.0, 80.0])], max_chars=320,
    )

    assert organization.context is not None
    text = organization.context.text
    assert "Now: chair (center)" in text
    assert "Earlier: door" in text
    assert "Earlier: chair" not in text
    assert "table" not in text and "banana" not in text
    assert organization.current_count == 1
    assert organization.history_count == 1
    assert organization.duplicates_removed >= 1
    chair = next(fact for fact in organization.facts if fact.role == "current")
    assert chair.sightings == 3
    assert chair.first_step == 0
    assert len(chair.evidence_ids) >= 3
    # L1 must not restate the RGB the frozen model already receives.
    for word in ("pixel", "colour", "color", "image", "scene looks"):
        assert word not in text.casefold()


def test_l1_omits_low_confidence_and_irrelevant_detections() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the chair")
    low = store.add(_record("chair_0", 0, confidence=0.3))
    store.add(_record("banana_0", 0, label="banana", confidence=0.99))
    organizer = L1PromptOrganizer()

    organization = organizer.build(
        store, scene_id="scene", episode_id="episode", rollout_idx=0, step=0,
        instruction="find the chair", rgb_width=300,
        events=[_event(low, [10.0, 10.0, 20.0, 20.0])], max_chars=320,
    )

    assert organization.context is not None
    assert organization.facts == []
    assert organization.context.layer_items == []
    assert "No instruction-related landmark" in organization.context.text
    assert "banana" not in organization.context.text


def test_l1_budget_drops_whole_lines_without_exceeding_the_original_budget() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find chair table door")
    for step, label in enumerate(("chair", "table", "door")):
        record = store.add(_record(label, step, label=label))
        assert record
    current = store.add(_record("chair_4", 4))
    organizer = L1PromptOrganizer(max_items=4)

    organization = organizer.build(
        store, scene_id="scene", episode_id="episode", rollout_idx=0, step=4,
        instruction="find chair table door", rgb_width=300,
        events=[_event(current, [10.0, 10.0, 40.0, 40.0])], max_chars=120,
    )

    if organization.context is not None:
        assert len(organization.context.text) <= 120
        assert organization.omitted_items >= 1
        for item in organization.context.layer_items:
            assert item.metadata["serialized_text"] in organization.context.text or (
                item.metadata["serialized_text"].replace("&amp;", "&")
                in organization.context.text
            )
    tiny = organizer.build(
        store, scene_id="scene", episode_id="episode", rollout_idx=0, step=4,
        instruction="find chair table door", rgb_width=300,
        events=[_event(current, [10.0, 10.0, 40.0, 40.0])], max_chars=5,
    )
    assert tiny.context is None
    assert tiny.empty_reason is not None


def test_l1_previous_episode_history_requires_persistent_scope() -> None:
    local = V3MemoryStore()
    local.begin_episode("scene", "old", "find chair")
    local.add(_record("old_chair", 0, episode="old"))
    local.close_episode()
    local.begin_episode("scene", "episode", "find chair")
    current = local.add(_record("chair_now", 0))
    local_org = L1PromptOrganizer().build(
        local, scene_id="scene", episode_id="episode", rollout_idx=0, step=0,
        instruction="find chair", rgb_width=300, events=[_event(current)], max_chars=320,
    )
    assert all(not fact.past_episode for fact in local_org.facts)

    persistent = V3MemoryStore(scope="scene_persistent")
    persistent.begin_episode("scene", "old", "find chair")
    persistent.add(_record("old_chair", 0, episode="old"))
    persistent.close_episode()
    persistent.begin_episode("scene", "episode2", "find chair")
    persistent_org = L1PromptOrganizer().build(
        persistent, scene_id="scene", episode_id="episode2", rollout_idx=0, step=0,
        instruction="find chair", rgb_width=300, events=[], max_chars=320,
    )
    past = [fact for fact in persistent_org.facts if fact.role == "history"]
    assert len(past) == 1 and past[0].past_episode
    assert "in an earlier episode" in persistent_org.context.text


# ---------------------------------------------------------------------------
# L3
# ---------------------------------------------------------------------------


def _l3_scene(*, move_for_traversal: bool = False):
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the chair and the table")
    linked = store.add(_record("chair_0", 0))
    object_map = ObjectMap(GeometryConfig())
    object_map.reset_episode("scene", "episode", 0)
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    depth[2:6, 0:4] = 1.0
    depth[2:6, 4:8] = 5.0
    object_map.observe(
        _frame(depth=depth),
        [
            _instance("chair", _mask(0, 4), 0, 0, linked=linked.key),
            _instance("table", _mask(4, 8), 0, 1),
        ],
        step_id=0,
    )
    if move_for_traversal:
        object_map.observe(
            _frame(agent_x=0.5), [_instance("chair", _mask(0, 4), 1, 0)],
            step_id=1, transition_record=True,
        )
    else:
        object_map.observe(
            _frame(depth=depth), [_instance("table", _mask(4, 8), 1, 0)],
            step_id=1,
        )
    return store, object_map


def test_l3_retrieves_by_instruction_and_recency_not_distance_alone() -> None:
    store, object_map = _l3_scene()
    organizer = L3PromptOrganizer(max_items=2)
    organization = organizer.build(
        store, object_map, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=2, instruction="find the chair and the table",
        frame=_frame(), scope="episode_local", max_chars=320, min_confidence=0.5,
    )

    labels = [fact.label for fact in organization.selected_facts]
    assert labels[0] == "table"  # seen at step 1, farther away
    text = organization.context.text
    assert "chair" in text and "table" in text
    assert "not a path" in text
    assert "passage unknown" in text
    assert "last step 1" in text
    assert organization.candidate_count == 2
    # Every referenced Evidence exists in the recorded map; nothing is invented.
    for item in organization.context.layer_items:
        for evidence_id in item.metadata["evidence_ids"]:
            assert evidence_id in object_map.evidence
    for word in ("Place", "Edge", "connected_to", "reachable"):
        assert word not in text


def test_l3_separates_observed_spatial_and_traversal_relations() -> None:
    store, object_map = _l3_scene(move_for_traversal=True)
    organization = L3PromptOrganizer(max_items=2).build(
        store, object_map, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=2, instruction="find the chair", frame=_frame(), scope="episode_local",
        max_chars=320, min_confidence=0.5,
    )

    chair = organization.selected_facts[0]
    assert chair.viewpoints == 1
    assert chair.traversal_links >= 1
    assert chair.traversal_steps
    assert "traversal recorded between observed viewpoints" in organization.context.text
    assert "passage unknown" in organization.context.text


def test_l3_marks_unverified_geometry_without_a_position_or_path() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the chair")
    object_map = ObjectMap(GeometryConfig())
    object_map.reset_episode("scene", "episode", 0)
    object_map.observe(
        _frame(), [_instance("chair", _mask(0, 4), 0, 0, confidence=0.3)],
        step_id=0,
    )
    organization = L3PromptOrganizer().build(
        store, object_map, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=2, instruction="find the chair", frame=_frame(), scope="episode_local",
        max_chars=320, min_confidence=0.5,
    )

    assert organization.unverified_count == 1
    assert organization.selected_facts[0].geometry_verified is False
    assert organization.selected_facts[0].distance_m is None
    text = organization.context.text
    assert "map evidence unverified" in text
    assert "position and passage unknown" in text
    assert "m line" not in text


def test_l3_current_step_observation_alone_is_not_history() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the chair")
    object_map = ObjectMap(GeometryConfig())
    object_map.reset_episode("scene", "episode", 0)
    object_map.observe(
        _frame(), [_instance("chair", _mask(0, 4), 0, 0)], step_id=0,
    )
    organization = L3PromptOrganizer().build(
        store, object_map, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=0, instruction="find the chair", frame=_frame(), scope="episode_local",
        max_chars=320, min_confidence=0.5,
    )
    assert organization.context is None
    assert organization.current_only_skipped == 1


# ---------------------------------------------------------------------------
# Joint
# ---------------------------------------------------------------------------


def _l1_fact(*, evidence: tuple[str, ...], role: str = "current", label: str = "chair") -> L1Fact:
    return L1Fact(
        label=label, role=role, sector="center", confidence=0.9, last_step=3,
        first_step=0, sightings=2, source="yolo", evidence_ids=evidence,
        provenance=tuple({"evidence_id": item, "source": "yolo"} for item in evidence),
        match_score=1.0,
    )


def _l3_fact(
    *, object_id: str = "obj", label: str = "chair",
    evidence: tuple[str, ...] = ("l3",), linked: tuple[str, ...] = (),
    verified: bool = True, distance: float | None = 2.0,
    viewpoints: int = 2, traversal: tuple[int, ...] = (),
) -> L3Fact:
    return L3Fact(
        object_id=object_id, label=label, evidence_ids=evidence,
        linked_l1_evidence_ids=linked,
        provenance=tuple({"evidence_id": item, "source": "yolo11n-seg"} for item in evidence),
        last_step=2, first_step=0, observation_count=2, viewpoints=viewpoints,
        traversal_links=len(traversal), confidence=0.8, source="yolo11n-seg",
        distance_m=distance if verified else None, bearing_deg=10.0 if verified else None,
        geometry_verified=verified, past_episode=False, match_score=1.0,
        traversal_steps=traversal,
    )


def _organizations(l1_facts, l3_facts, *, l1_duplicates: int = 0, l3_candidates: int = 0):
    from lightnav.memory.l1_prompt_organizer import L1Organization
    from lightnav.memory.l3_prompt_organizer import L3Organization

    l1 = L1Organization(
        context=None, facts=list(l1_facts), selected_facts=list(l1_facts),
        current_count=sum(fact.role == "current" for fact in l1_facts),
        history_count=sum(fact.role == "history" for fact in l1_facts),
        duplicates_removed=l1_duplicates,
    )
    l3 = L3Organization(
        context=None, facts=list(l3_facts), selected_facts=list(l3_facts),
        candidate_count=l3_candidates or len(l3_facts),
        verified_count=sum(fact.geometry_verified for fact in l3_facts),
        unverified_count=sum(not fact.geometry_verified for fact in l3_facts),
    )
    return l1, l3


def test_joint_confirmed_link_merges_one_fact_but_keeps_both_sources() -> None:
    l1, l3 = _organizations(
        [_l1_fact(evidence=("l1_chair",))],
        [_l3_fact(linked=("l1_chair",), evidence=("l3_chair",))],
    )
    joint = JointPromptBuilder().build(l1, l3, step=3, max_chars=320)

    assert joint.merged_entities == 1
    assert joint.alignments[0].status == "confirmed"
    assert joint.joint_item_count == 1
    assert len(joint.context.layer_items) == 1
    item = joint.context.layer_items[0]
    assert item.layer == "short"
    assert item.metadata["l1_evidence_ids"] == ["l1_chair"]
    assert item.metadata["l3_evidence_ids"] == ["l3_chair"]
    assert set(item.metadata["evidence_ids"]) == {"l1_chair", "l3_chair"}
    assert "now visible" in joint.context.text
    assert "not a path" in joint.context.text
    assert "passage unknown" in joint.context.text


def test_joint_same_label_without_link_stays_uncertain_and_unmerged() -> None:
    l1, l3 = _organizations(
        [_l1_fact(evidence=("l1_chair",))],
        [_l3_fact(linked=(), evidence=("l3_chair",))],
    )
    joint = JointPromptBuilder().build(l1, l3, step=3, max_chars=320)

    assert joint.merged_entities == 0
    assert joint.alignments[0].status == "uncertain_same_label"
    assert joint.l1_item_count == 1
    assert joint.l3_item_count == 1
    assert "Uncertain: chair may be the same object" in joint.context.text
    assert joint.diagnostics()["alignment"]["success_rate"] == 0.0


def test_joint_conflicting_link_is_preserved_not_silently_merged() -> None:
    l1, l3 = _organizations(
        [_l1_fact(evidence=("l1_chair",), label="chair")],
        [_l3_fact(linked=("l1_chair",), label="table", evidence=("l3_table",))],
    )
    joint = JointPromptBuilder().build(l1, l3, step=3, max_chars=320)

    assert joint.merged_entities == 0
    assert joint.alignments[0].status == "identity_conflict"
    assert len(joint.conflicts) == 1
    assert joint.conflicts[0]["type"] == "identity_mismatch"
    # Both independent facts remain as separate prompt items, not merged.
    labels = {item.label for item in joint.context.layer_items}
    assert {"chair", "table"} <= labels
    assert joint.joint_item_count == 0
    assert "chair" in joint.context.text and "table" in joint.context.text


def test_joint_unverified_geometry_conflict_keeps_position_unknown() -> None:
    l1, l3 = _organizations(
        [_l1_fact(evidence=("l1_chair",))],
        [_l3_fact(linked=(), evidence=("l3_chair",), verified=False)],
    )
    joint = JointPromptBuilder().build(l1, l3, step=3, max_chars=320)

    assert joint.conflicts and joint.conflicts[0]["type"] == "geometry_unverified"
    assert "position unknown" in joint.context.text
    assert "m line" not in joint.context.text


def test_joint_removes_duplicate_facts_but_keeps_distinct_objects() -> None:
    duplicate_l3 = _l3_fact(object_id="obj_a", evidence=("l3_a",))
    same_line_other_object = _l3_fact(object_id="obj_b", evidence=("l3_a",))
    distinct = _l3_fact(object_id="obj_c", label="table", evidence=("l3_c",))
    l1, l3 = _organizations(
        [_l1_fact(evidence=("l1_chair",))],
        [duplicate_l3, same_line_other_object, distinct],
    )
    joint = JointPromptBuilder().build(l1, l3, step=3, max_chars=500)

    assert joint.duplicates_removed >= 1
    labels = {item.label for item in joint.context.layer_items}
    assert "chair" in labels and "table" in labels


def test_joint_budget_and_unknown_relation_protection() -> None:
    facts = [
        _l3_fact(object_id=f"obj_{index}", evidence=(f"l3_{index}",), distance=2.0)
        for index in range(4)
    ]
    l1, l3 = _organizations([], facts)
    joint = JointPromptBuilder(max_items=3).build(l1, l3, step=3, max_chars=200)

    assert joint.context is not None
    assert len(joint.context.text) <= 200
    assert joint.omitted_items >= 1
    assert "passage unknown" in joint.context.text
    assert "Unknown: object passability" in joint.context.text or (
        joint.unverified_relations >= 1
    )
    assert not any(
        word in joint.context.text.casefold()
        for word in ("reachable", "is clear", "path is", "walkable")
    )


def test_joint_empty_inputs_still_state_current_step_without_inventing() -> None:
    l1, l3 = _organizations([], [])
    joint = JointPromptBuilder().build(l1, l3, step=7, max_chars=320)
    assert joint.context is not None
    assert "Step 7" in joint.context.text
    assert "No instruction-related L1 detection or L3 map fact" in joint.context.text
    assert joint.context.layer_items == []


@pytest.mark.parametrize("max_chars", [0, -1])
def test_joint_rejects_invalid_budget(max_chars: int) -> None:
    l1, l3 = _organizations([], [])
    with pytest.raises(ValueError):
        JointPromptBuilder().build(l1, l3, step=0, max_chars=max_chars)


# ---------------------------------------------------------------------------
# Lossless joint (original L1 + organized L3)
# ---------------------------------------------------------------------------


def _original_l1_context(
    *, label: str = "chair", role: str = "current",
    evidence: tuple[str, ...] = ("l1_chair",),
    line: str | None = None,
) -> MemoryContext:
    text = line or (
        f"Current RGB: instruction landmark {label} in the center of the current "
        "view (detected; passage unknown)."
        if role == "current"
        else f"Local history: {label} was seen near a visually similar area "
        "(possible revisit; current direction and passage unknown)."
    )
    item = LayerMemoryItem(
        layer="short" if role == "current" else "long",
        kind=f"local_graph_{role}_landmark",
        description=text,
        step_id=3,
        confidence=0.9,
        source="local_graph_fusion",
        evidence_refs=[
            ObservationEvidence(
                event_id=evidence_id, step_id=3, label=label,
                confidence=0.9, source="yolo",
            )
            for evidence_id in evidence
        ],
        label=label,
        metadata={
            "evidence_ids": list(evidence),
            "serialized_text": escape(text, quote=False),
            "provenance": [
                {"evidence_id": evidence_id, "step": 3, "source": "yolo"}
                for evidence_id in evidence
            ],
            "role": role,
            "traversability_status": "unknown",
        },
    )
    return MemoryContext(text="[Navigation Memory]\n" + text, layer_items=[item])


def _l3_organization(facts: list[L3Fact]) -> L3Organization:
    return L3Organization(
        context=None, facts=list(facts), selected_facts=list(facts),
        candidate_count=len(facts),
        verified_count=sum(fact.geometry_verified for fact in facts),
        unverified_count=sum(not fact.geometry_verified for fact in facts),
    )


def test_lossless_joint_keeps_the_exact_l1_line_and_links_l3_separately() -> None:
    original = _original_l1_context()
    l3 = _l3_organization([
        _l3_fact(linked=("l1_chair",), evidence=("l3_chair",)),
    ])

    joint = JointPromptBuilder().build_preserving_l1(
        original, l3, step=4, max_chars=500,
    )

    l1_line = original.layer_items[0].description
    assert l1_line in joint.context.text
    assert "now visible" not in joint.context.text
    assert "not a path" in joint.context.text
    assert joint.merged_entities == 0  # no synthesized L1+L3 line
    assert joint.aligned_pairs == 1
    assert joint.l1_item_count == 1 and joint.l3_item_count == 1
    assert joint.joint_item_count == 0
    assert joint.alignments[0].status == "confirmed"

    l1_item = next(
        item for item in joint.context.layer_items
        if item.metadata.get("l1_prompt_preserved") is True
    )
    assert l1_item.description == l1_line
    assert l1_item.source == "local_graph_fusion"
    assert l1_item.evidence_refs == original.layer_items[0].evidence_refs
    assert l1_item.metadata["aligned_l3_object_ids"] == ["obj"]
    assert l1_item.metadata["layer_contribution"] == {"l1": True, "l3": False}

    linked_l3 = next(
        item for item in joint.context.layer_items
        if item.metadata.get("layer_source") == "l3"
    )
    assert linked_l3.metadata["aligned"] is True
    assert linked_l3.metadata["aligned_l1_evidence_ids"] == ["l1_chair"]
    assert linked_l3.metadata["aligned_l3_object_id"] == "obj"


def test_lossless_joint_keeps_both_lines_for_an_identity_conflict() -> None:
    original = _original_l1_context(label="chair")
    l3 = _l3_organization([
        _l3_fact(linked=("l1_chair",), label="table", evidence=("l3_table",)),
    ])

    joint = JointPromptBuilder().build_preserving_l1(
        original, l3, step=4, max_chars=500,
    )

    assert joint.conflicts and joint.conflicts[0]["type"] == "identity_mismatch"
    assert original.layer_items[0].description in joint.context.text
    assert "table" in joint.context.text
    assert "sources kept, not merged" in joint.context.text
    assert joint.aligned_pairs == 0
    assert joint.l1_item_count == 1 and joint.l3_item_count == 1


def test_lossless_joint_same_label_without_link_keeps_original_and_uncertain() -> None:
    original = _original_l1_context(label="chair")
    l3 = _l3_organization([
        _l3_fact(linked=(), evidence=("l3_chair",), verified=False),
    ])

    joint = JointPromptBuilder().build_preserving_l1(
        original, l3, step=4, max_chars=500,
    )

    assert joint.merged_entities == 0
    assert joint.alignments[0].status == "uncertain_same_label"
    assert joint.conflicts and joint.conflicts[0]["type"] == "geometry_unverified"
    assert original.layer_items[0].description in joint.context.text
    assert "map evidence unverified" in joint.context.text
    assert "Uncertain: chair may be the same object" in joint.context.text
    assert "m line" not in joint.context.text


def test_lossless_joint_without_l1_still_emits_the_organized_l3_card() -> None:
    l3 = _l3_organization([
        _l3_fact(linked=(), evidence=("l3_chair",)),
    ])

    joint = JointPromptBuilder().build_preserving_l1(
        None, l3, step=4, max_chars=500,
    )

    assert joint.context is not None
    assert "chair" in joint.context.text
    assert "not a path" in joint.context.text
    assert joint.l1_item_count == 0 and joint.l3_item_count == 1


def test_lossless_joint_whole_line_budget_does_not_cut_l1_text() -> None:
    original = _original_l1_context()
    l3 = _l3_organization([
        _l3_fact(linked=("l1_chair",), evidence=("l3_chair",)),
    ])

    joint = JointPromptBuilder(max_items=3).build_preserving_l1(
        original, l3, step=4, max_chars=120,
    )

    assert joint.context is None or len(joint.context.text) <= 120
    if joint.context is not None:
        for item in joint.context.layer_items:
            assert item.metadata["serialized_text"] in escape(
                joint.context.text, quote=False
            )
