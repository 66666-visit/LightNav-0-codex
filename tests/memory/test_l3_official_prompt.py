"""Official-Prompt-Guide expression of the shared L3 candidate facts."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from lightnav.memory.conceptgraphs_lite import (
    CameraIntrinsics,
    ObjectMap,
    ObjectNode,
    ObservedFromEdge,
    Pose3D,
    RGBDFrame,
    SpatialEvidence,
    ViewpointNode,
)
from lightnav.memory.l3_official_prompt import (
    HEADER,
    L3OfficialPromptOrganizer,
    direction_phrase,
    support_relation,
)
from lightnav.memory.l3_prompt_organizer import L3PromptOrganizer
from lightnav.memory.token_budget import select_context_lines
from lightnav.memory.v3_schemas import canonical_evidence_key
from lightnav.memory.v3_store import V3MemoryStore

_IDENTITY = Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
_DEFAULT = object()


def _pose(*, x: float = 0.0, z: float = 0.0, yaw_deg: float = 0.0) -> Pose3D:
    half = math.radians(yaw_deg) / 2.0
    return Pose3D((x, 0.0, z), (math.cos(half), 0.0, math.sin(half), 0.0))


def _frame(*, pose: Pose3D | None = None) -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.full((8, 8), 2.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
        agent_pose=pose or _IDENTITY,
        camera_extrinsics=_IDENTITY,
    )


def _build_map(
    objects: list[dict[str, object]], *,
    scene: str = "scene", episode: str = "episode", rollout: int = 0,
) -> tuple[V3MemoryStore, ObjectMap]:
    """Build the exact ObjectMap/Evidence shape the L3 organizer reads."""
    store = V3MemoryStore()
    store.begin_episode(scene, episode, "find the chair")
    object_map = ObjectMap()
    object_map.reset_episode(scene, episode, rollout)
    for index, spec in enumerate(objects):
        step = int(spec.get("step", 7))
        label = str(spec["label"])
        center = tuple(float(value) for value in spec["center"])
        bbox = tuple(
            tuple(float(value) for value in point) for point in spec["bbox"]
        )
        confidence = float(spec.get("confidence", 0.9))
        viewpoint_id = json.dumps(
            [scene, episode, rollout, step], separators=(",", ":"),
        )
        object_map.viewpoints.setdefault(viewpoint_id, ViewpointNode(
            viewpoint_id, scene, episode, rollout, step, _IDENTITY, _IDENTITY,
        ))
        evidence_id = canonical_evidence_key(
            scene, episode, rollout, f"l3_seg_{step}_{index}",
        )
        object_map.evidence[evidence_id] = SpatialEvidence(
            evidence_id, scene, episode, rollout, step, label, confidence,
            "yolo11n-seg", 100, 100, "observed", None,
        )
        object_id = f"object_{index}"
        object_map.objects[object_id] = ObjectNode(
            object_id, label, center, bbox, 2, [evidence_id], "observed",
            np.empty((0, 3)),
        )
        object_map.edges.append(ObservedFromEdge(
            object_id, viewpoint_id, evidence_id,
        ))
    return store, object_map


def _official(
    store: V3MemoryStore, object_map: ObjectMap, *,
    instruction: str = "find the chair", step: int = 10,
    frame: object = _DEFAULT, max_chars: int = 2000, max_items: int = 2,
    min_confidence: float = 0.5, scope: str = "episode_local",
):
    return L3OfficialPromptOrganizer(max_items=max_items).build(
        store, object_map, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=step, instruction=instruction,
        frame=_frame() if frame is _DEFAULT else frame,
        scope=scope, max_chars=max_chars, min_confidence=min_confidence,
    )


_CHAIR_LEFT = {
    "label": "chair", "center": (-1.0, 0.5, -2.0),
    "bbox": ((-1.3, 0.0, -2.3), (-0.7, 1.0, -1.7)),
}
_CHAIR_RIGHT = {
    "label": "chair", "center": (3.5, 0.5, -3.0),
    "bbox": ((3.2, 0.0, -3.3), (3.8, 1.0, -2.7)),
}
_TABLE = {
    "label": "dining table", "center": (0.5, 0.4, -2.5),
    "bbox": ((-0.2, 0.35, -3.2), (1.2, 0.75, -1.8)),
}


# ---------------------------------------------------------------------------
# Candidate identity / historical mode
# ---------------------------------------------------------------------------


def test_candidate_facts_are_byte_identical_to_the_organized_mode() -> None:
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT, _TABLE])
    organized = L3PromptOrganizer(max_items=2).build(
        store, object_map, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=10, instruction="find the chair", frame=_frame(),
        scope="episode_local", max_chars=2000, min_confidence=0.5,
    )
    official = _official(store, object_map)

    assert [fact.to_dict() for fact in official.facts] == [
        fact.to_dict() for fact in organized.facts
    ]
    assert official.candidate_count == organized.candidate_count == 2
    assert official.verified_count == organized.verified_count == 2
    assert official.unverified_count == organized.unverified_count == 0
    assert official.current_only_skipped == organized.current_only_skipped == 0
    # The anchor table is in the prompt pool but is not a selected fact.
    assert [fact.object_id for fact in official.selected_facts] == [
        fact.object_id for fact in organized.selected_facts
    ]


def test_organized_serialization_is_unchanged_by_the_new_mode() -> None:
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT])
    organized = L3PromptOrganizer(max_items=2).build(
        store, object_map, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=10, instruction="find the chair", frame=_frame(),
        scope="episode_local", max_chars=320, min_confidence=0.5,
    )
    text = organized.context.text
    assert text.startswith("[Navigation Context - L3]")
    assert "map facts are history, not current visibility." in text
    assert "not a path" in text and "passage unknown" in text
    assert "last step 7" in text
    assert HEADER not in text
    assert organized.source == "l3_prompt_organizer"
    assert "expression" not in organized.diagnostics()


# ---------------------------------------------------------------------------
# Direction + object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("bearing", "expected"), [
    (0.0, "in front of you"),
    (22.5, "in front of you"),
    (22.6, "to your front-left"),
    (60.0, "to your front-left"),
    (90.0, "to your left"),
    (120.0, "behind you on the left"),
    (160.0, "behind you"),
    (-30.0, "to your front-right"),
    (-90.0, "to your right"),
    (-120.0, "behind you on the right"),
    (-180.0, "behind you"),
])
def test_direction_sectors_are_robot_relative(bearing: float, expected: str) -> None:
    assert direction_phrase(bearing) == expected


def test_direction_uses_the_current_robot_heading_not_world_bearing() -> None:
    objects = [{
        "label": "chair", "center": (0.0, 0.5, -2.0),
        "bbox": ((-0.3, 0.0, -2.3), (0.3, 1.0, -1.7)),
    }]
    store, object_map = _build_map(objects)
    forward = _official(store, object_map)
    assert "is in front of you" in forward.context.text

    # Turn the robot +90 deg (left): a world-forward object is now on its right.
    turned = _official(store, object_map, frame=_frame(pose=_pose(yaw_deg=90.0)))
    assert "is to your right" in turned.context.text
    assert "is in front of you" not in turned.context.text
    item = turned.context.layer_items[0]
    assert item.metadata["current_relative_bearing_deg"] == pytest.approx(-90.0)
    assert item.metadata["range_kind"] == "straight_line_not_path"


def test_no_pose_keeps_a_structured_fact_without_inventing_a_direction() -> None:
    store, object_map = _build_map([_CHAIR_LEFT])
    result = _official(store, object_map, frame=None)
    text = result.context.text
    assert "direction from you is unavailable" in text
    assert "in front" not in text and "to your" not in text
    assert "Map record (step 7" in text
    assert result.selected_facts[0].distance_m is None


# ---------------------------------------------------------------------------
# Relational anchor
# ---------------------------------------------------------------------------


def test_next_to_uses_an_observed_adjacent_large_anchor() -> None:
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT, _TABLE])
    result = _official(store, object_map)

    text = result.context.text
    assert "chair next to the dining table" in text
    assert "straight-line" in text
    assert result.expression["strategy_counts"]["relational_anchor"] == 1
    assert result.expression["anchors_used"] == [{
        "object_id": "object_2", "label": "dining table", "relation": "next to",
        "evidence_ids": ['["scene","episode",0,"l3_seg_7_2"]'],
    }]
    anchor_item = next(
        item for item in result.context.layer_items
        if item.metadata.get("expression_strategy") == "relational_anchor"
    )
    assert anchor_item.metadata["anchor_label"] == "dining table"
    assert anchor_item.metadata["relation"] == "next to"


def test_on_is_used_only_when_the_observed_boxes_support_support_geometry() -> None:
    objects = [
        {
            "label": "vase", "center": (0.0, 0.85, -2.0),
            "bbox": ((-0.1, 0.75, -2.1), (0.1, 0.95, -1.9)),
        },
        {
            "label": "vase", "center": (2.0, 0.1, -3.0),
            "bbox": ((1.9, 0.0, -3.1), (2.1, 0.2, -2.9)),
        },
        {
            "label": "dining table", "center": (0.0, 0.725, -2.0),
            "bbox": ((-0.5, 0.7, -2.5), (0.5, 0.75, -1.5)),
        },
    ]
    store, object_map = _build_map(objects)
    result = _official(store, object_map, instruction="find the vase")
    assert "vase on the dining table" in result.context.text
    assert result.expression["anchors_used"][0]["relation"] == "on"


def test_shared_relation_is_rejected_because_it_does_not_disambiguate() -> None:
    left_table_chair = {
        "label": "chair", "center": (-0.8, 0.5, -2.0),
        "bbox": ((-1.1, 0.0, -2.3), (-0.5, 1.0, -1.7)),
    }
    right_table_chair = {
        "label": "chair", "center": (0.8, 0.5, -2.0),
        "bbox": ((0.5, 0.0, -2.3), (1.1, 1.0, -1.7)),
    }
    table = {
        "label": "dining table", "center": (0.0, 0.4, -2.0),
        "bbox": ((-0.4, 0.35, -2.5), (0.4, 0.75, -1.5)),
    }
    store, object_map = _build_map([left_table_chair, right_table_chair, table])
    result = _official(store, object_map)
    assert "next to" not in result.context.text
    assert result.expression["strategy_counts"]["relational_anchor"] == 0
    # The stable left-to-right order still singles out both chairs.
    assert result.expression["strategy_counts"]["extreme"] == 2


def test_current_step_only_observation_cannot_become_an_anchor() -> None:
    table = {**_TABLE, "step": 10}  # observed only in the current frame
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT, table])
    result = _official(store, object_map)
    assert "next to" not in result.context.text
    assert result.expression["anchors_used"] == []
    assert result.candidate_count == 2


def test_support_relation_rejects_unknown_geometry() -> None:
    table = ((-0.5, 0.7, -2.5), (0.5, 0.75, -1.5))
    on_table = ((-0.1, 0.75, -2.1), (0.1, 0.95, -1.9))
    above_table = ((-0.1, 1.5, -2.1), (0.1, 1.7, -1.9))
    beside_table = ((0.7, 0.0, -2.0), (0.9, 0.3, -1.8))
    far_away = ((3.0, 0.0, -2.0), (3.2, 0.3, -1.8))
    bigger_than_table = ((-0.6, 0.7, -2.6), (0.6, 1.4, -1.4))

    assert support_relation(on_table, table) == "on"
    assert support_relation(beside_table, table) == "next to"
    assert support_relation(above_table, table) is None
    assert support_relation(far_away, table) is None
    assert support_relation(bigger_than_table, table) is None
    assert support_relation(table, on_table) is None


# ---------------------------------------------------------------------------
# Extremes / ordinals
# ---------------------------------------------------------------------------


def test_extremes_need_a_stable_left_to_right_forward_order() -> None:
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT])
    result = _official(store, object_map)
    text = result.context.text
    assert "leftmost chair of the 2 recorded in front" in text
    assert "rightmost chair of the 2 recorded in front" in text
    assert result.expression["strategy_counts"]["extreme"] == 2


def test_same_bearing_objects_do_not_get_an_unstable_extreme() -> None:
    near = {
        "label": "chair", "center": (0.0, 0.5, -2.0),
        "bbox": ((-0.3, 0.0, -2.3), (0.3, 1.0, -1.7)),
    }
    far = {
        "label": "chair", "center": (0.0, 0.5, -4.0),
        "bbox": ((-0.6, 0.0, -4.6), (0.6, 1.0, -3.4)),
    }
    store, object_map = _build_map([near, far])
    result = _official(store, object_map)
    assert "leftmost" not in result.context.text
    assert "rightmost" not in result.context.text
    assert result.expression["strategy_counts"]["direction"] == 2


def test_behind_objects_are_not_counted_in_the_front_order() -> None:
    front = {
        "label": "chair", "center": (0.0, 0.5, -2.0),
        "bbox": ((-0.3, 0.0, -2.3), (0.3, 1.0, -1.7)),
    }
    behind = {
        "label": "chair", "center": (0.0, 0.5, 2.0),
        "bbox": ((-0.3, 0.0, 1.7), (0.3, 1.0, 2.3)),
    }
    store, object_map = _build_map([front, behind])
    result = _official(store, object_map)
    assert "leftmost" not in result.context.text
    assert "behind you" in result.context.text


def test_ordinal_needs_three_forward_objects_and_a_counting_direction() -> None:
    middle = {
        "label": "chair", "center": (0.0, 0.5, -6.0),
        "bbox": ((-0.3, 0.0, -6.3), (0.3, 1.0, -5.7)),
    }
    store, object_map = _build_map([_CHAIR_LEFT, middle, _CHAIR_RIGHT])
    result = _official(store, object_map, max_items=3)
    text = result.context.text
    assert result.expression["strategy_counts"]["ordinal"] == 1
    ordinal_item = next(
        item for item in result.context.layer_items
        if item.metadata.get("expression_strategy") == "ordinal"
    )
    assert ordinal_item.metadata["ordinal"] == 2
    assert "second chair from the left among the 3 recorded in front" in text
    assert "from the left" in text
    assert "fourth" not in text and "fifth" not in text

    two = _official(store, object_map, max_items=2)
    assert "second chair from the left" not in two.context.text


# ---------------------------------------------------------------------------
# Safety, budget and dedup
# ---------------------------------------------------------------------------


def test_unverified_geometry_is_dropped_instead_of_being_dressed_up() -> None:
    low = {**_CHAIR_LEFT, "confidence": 0.3}
    store, object_map = _build_map([low])
    result = _official(store, object_map)
    assert result.context is None
    assert result.expression["unverified_omitted"] == 1
    assert result.selected_facts == []
    assert result.empty_reason == "no_expressible_l3_fact"

    organized = L3PromptOrganizer(max_items=2).build(
        store, object_map, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=10, instruction="find the chair", frame=_frame(),
        scope="episode_local", max_chars=320, min_confidence=0.5,
    )
    assert "map evidence unverified" in organized.context.text


def test_header_keeps_the_history_range_and_passage_caveats() -> None:
    store, object_map = _build_map([_CHAIR_LEFT])
    text = _official(store, object_map).context.text
    header = text.splitlines()[0]
    assert header == HEADER
    for phrase in (
        "history, not current view", "directions relative to you",
        "straight-line ranges, not paths", "passage unverified",
    ):
        assert phrase in header
    for imperative in ("go to", "walk to", "head to", "approach", "turn "):
        assert imperative not in text.casefold()
    for claim in ("reachable", "walkable", "is clear", "path is"):
        assert claim not in text.casefold()


def test_duplicate_facts_are_not_injected_twice() -> None:
    duplicate = {
        "label": "chair", "center": (-1.0, 0.5, -2.0),
        "bbox": ((-1.3, 0.0, -2.3), (-0.7, 1.0, -1.7)),
    }
    store, object_map = _build_map([_CHAIR_LEFT, duplicate])
    result = _official(store, object_map)
    assert result.context.layer_items and len(result.context.layer_items) == 1
    assert result.expression["duplicates_removed"] == 1
    assert result.context.text.count("the chair is") == 1


def test_header_is_atomic_and_the_budget_keeps_whole_lines() -> None:
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT])
    no_room = _official(store, object_map, max_chars=len(HEADER) - 1)
    assert no_room.context is None
    assert no_room.empty_reason == "budget_too_small_for_header"

    header_only = _official(store, object_map, max_chars=len(HEADER))
    assert header_only.context is None
    assert header_only.empty_reason == "no_expressible_l3_fact"

    one_line = _official(
        store, object_map, max_chars=len(HEADER) + 130,
    )
    assert one_line.context is not None
    assert len(one_line.context.text) <= len(HEADER) + 130
    assert len(one_line.context.layer_items) == 1
    item = one_line.context.layer_items[0]
    assert item.metadata["serialized_text"] in one_line.context.text
    assert one_line.prompt_chars <= len(HEADER) + 130
    assert one_line.omitted_items >= 1


def test_token_selection_keeps_the_atomic_header_and_whole_lines() -> None:
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT])
    result = _official(store, object_map, max_chars=2000)
    lines = result.context.text.splitlines()

    def count(text: str) -> int:
        return len(text.split())

    capacity = count("\n".join(lines[:2]))  # header + first fact
    selection = select_context_lines(
        result.context, capacity_tokens=capacity, count_tokens=count,
    )
    assert selection.context is not None
    assert selection.block_tokens <= capacity
    assert selection.context.text.splitlines()[0] == HEADER
    assert len(selection.context.layer_items) == 1
    assert selection.omitted_items == 1

    empty = select_context_lines(
        result.context, capacity_tokens=0, count_tokens=count,
    )
    assert empty.context is None


def test_verified_selection_is_the_organized_selection_without_unverified_lines() -> None:
    verified = _CHAIR_LEFT
    unverified_one = {**_CHAIR_RIGHT, "confidence": 0.3}
    unverified_two = {
        "label": "chair", "center": (0.0, 0.5, -6.0),
        "bbox": ((-0.3, 0.0, -6.3), (0.3, 1.0, -5.7)),
        "confidence": 0.3,
    }
    store, object_map = _build_map([verified, unverified_one, unverified_two])
    organized = L3PromptOrganizer(max_items=2).build(
        store, object_map, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=10, instruction="find the chair", frame=_frame(),
        scope="episode_local", max_chars=2000, min_confidence=0.5,
    )
    official = _official(store, object_map)

    old_verified = {
        fact.object_id for fact in organized.selected_facts if fact.geometry_verified
    }
    new_verified = {fact.object_id for fact in official.selected_facts}
    assert new_verified == old_verified == {"object_0"}
    # Both unverified candidates are omitted: neither has a trustworthy
    # position, so neither can be turned into a directional statement.
    assert official.expression["unverified_omitted"] == 2
    # The organized mode keeps one unverified line (its cap allows one); the
    # official mode cannot give it a truthful direction and omits it.
    assert len(organized.selected_facts) == 2
    assert "map evidence unverified" in organized.context.text
    assert "map evidence unverified" not in official.context.text


def test_partial_footprint_interpenetration_is_not_next_to() -> None:
    table = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    tucked_chair = ((0.9, 0.0, 0.2), (1.2, 1.0, 0.8))
    assert support_relation(tucked_chair, table) is None


def test_next_to_rejects_a_large_vertical_separation() -> None:
    table = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    floor_object = ((1.2, 0.0, 0.0), (1.5, 0.3, 0.3))
    wall_object = ((1.2, 2.0, 0.0), (1.5, 2.3, 0.3))
    assert support_relation(floor_object, table) == "next to"
    assert support_relation(wall_object, table) is None


def test_on_rejects_a_visible_gap_and_an_overhanging_object() -> None:
    table = ((-0.5, 0.7, -0.5), (0.5, 0.75, 0.5))
    floating = ((-0.1, 0.95, -0.1), (0.1, 1.15, 0.1))
    contact = ((-0.1, 0.75, -0.1), (0.1, 0.95, 0.1))
    overhanging = ((-0.4, 0.7, 0.2), (0.6, 0.9, 0.8))
    assert support_relation(floating, table) is None
    assert support_relation(contact, table) == "on"
    assert support_relation(overhanging, table) is None


def test_a_group_member_without_a_bearing_disables_extremes_and_ordinals() -> None:
    unknown_center = {
        "label": "chair", "center": (float("nan"), 0.5, -4.0),
        "bbox": ((-0.3, 0.0, -4.3), (0.3, 1.0, -3.7)),
    }
    store, object_map = _build_map(
        [_CHAIR_LEFT, _CHAIR_RIGHT, unknown_center],
    )
    result = _official(store, object_map, max_items=3)
    text = result.context.text
    assert "leftmost" not in text and "rightmost" not in text
    assert "second chair from the left" not in text
    assert result.expression["strategy_counts"]["direction_without_pose"] == 1


def test_duplicate_anchor_labels_do_not_produce_an_ambiguous_relation() -> None:
    far_table = {
        "label": "dining table", "center": (5.5, 0.4, -2.5),
        "bbox": ((5.0, 0.35, -3.2), (6.0, 0.75, -1.8)),
    }
    store, object_map = _build_map([_CHAIR_LEFT, _CHAIR_RIGHT, _TABLE, far_table])
    result = _official(store, object_map)
    assert "next to" not in result.context.text
    assert result.expression["anchors_used"] == []


def test_latest_step_is_the_newest_record_not_the_maximum_step() -> None:
    scene, episode, rollout = "scene", "episode", 0
    store = V3MemoryStore(scope="scene_persistent")
    store.begin_episode(scene, "old", "find the chair")
    store.close_episode()
    store.begin_episode(scene, episode, "find the chair")

    object_map = ObjectMap()
    object_map.reset_episode(scene, episode, rollout)
    object_id = "object_0"
    evidence_ids = []
    for index, (recorded_episode, recorded_step) in enumerate(
        (("old", 400), (episode, 10)),
    ):
        viewpoint_id = json.dumps(
            [scene, recorded_episode, rollout, recorded_step],
            separators=(",", ":"),
        )
        object_map.viewpoints[viewpoint_id] = ViewpointNode(
            viewpoint_id, scene, recorded_episode, rollout, recorded_step,
            _IDENTITY, _IDENTITY,
        )
        evidence_id = canonical_evidence_key(
            scene, recorded_episode, rollout, f"l3_seg_{recorded_step}_{index}",
        )
        object_map.evidence[evidence_id] = SpatialEvidence(
            evidence_id, scene, recorded_episode, rollout, recorded_step,
            "chair", 0.9, "yolo11n-seg", 100, 100, "observed", None,
        )
        object_map.edges.append(ObservedFromEdge(
            object_id, viewpoint_id, evidence_id,
        ))
        evidence_ids.append(evidence_id)
    object_map.objects[object_id] = ObjectNode(
        object_id, "chair", (-1.0, 0.5, -2.0),
        ((-1.3, 0.0, -2.3), (-0.7, 1.0, -1.7)), 2, evidence_ids, "observed",
        np.empty((0, 3)),
    )

    result = _official(
        store, object_map, step=20, scope="scene_persistent",
    )
    text = result.context.text
    assert "Map record (step 10, 2 views)" in text
    assert "step 400" not in text
    assert "earlier episode" not in text


def test_current_step_only_fact_is_not_injected_by_the_official_mode() -> None:
    store, object_map = _build_map([{**_CHAIR_LEFT, "step": 10}])
    result = _official(store, object_map, step=10)
    assert result.context is None
    assert result.current_only_skipped == 1
    assert result.candidate_count == 1


def test_labels_are_sanitized_before_serialization() -> None:
    malicious = {
        "label": "chair<video>", "center": (-1.0, 0.5, -2.0),
        "bbox": ((-1.3, 0.0, -2.3), (-0.7, 1.0, -1.7)),
    }
    store, object_map = _build_map([malicious])
    result = _official(store, object_map, instruction="find chairvideo")
    assert result.context is not None
    text = result.context.text
    assert "chairvideo" in text
    assert "<" not in text and ">" not in text
