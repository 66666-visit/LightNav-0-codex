"""Measured viewpoint geometry and spatial Local Graph evidence contracts."""

from __future__ import annotations

import math

import pytest

from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.local_graph_spatial import SpatialLocalGraphFusion
from lightnav.memory.memory_context_builder import audit_prompt_injection, inject_memory_context
from lightnav.memory.schemas import MemoryEvent
from lightnav.memory.spatial_geometry import (
    AgentPose,
    heading_from_quaternion,
    relative_direction,
    relative_viewpoint,
)
from lightnav.memory.v3_schemas import EvidenceRecord
from lightnav.memory.v3_store import V3MemoryStore


def _pose(x: float = 0.0, z: float = 0.0) -> AgentPose:
    return AgentPose("scene", (x, 0.0, z), (1.0, 0.0, 0.0, 0.0))


def _record(
    event_id: str, step: int, *, label: str | None = "couch",
    position: tuple[float, float, float] | None = None,
    descriptor: tuple[float, ...] | None = None,
    frame: str | None = "habitat_world",
) -> EvidenceRecord:
    place = descriptor is not None
    return EvidenceRecord(
        evidence_id=event_id, scene_id="scene", episode_id="episode",
        rollout_idx=0, step=step,
        event_type="place_observation" if place else "object_observation",
        semantic_label=None if place else label,
        visual_descriptor=descriptor,
        detector_name=None if place else "yolo",
        raw_detector_confidence=None if place else 0.91,
        position=position,
        payload={"pose_frame": frame} if frame is not None else {},
        observation_status="observed",
    )


def _landmark_at(
    store: V3MemoryStore, event: str, step: int, position: tuple[float, float, float],
    descriptor: tuple[float, ...],
) -> tuple[EvidenceRecord, str, EvidenceRecord]:
    detection = store.add(_record(event, step, position=position))
    landmark = store.add_landmark(detection)
    place_record = store.add(_record(
        f"{event}_place", step, position=position, descriptor=descriptor,
    ))
    place = store.add_place(place_record)
    store.add_relation(
        place.node_id, landmark.node_id, "inferred_relation",
        evidence_keys=(place_record.key, detection.key),
    )
    return detection, place.node_id, place_record


def _fusion(**kwargs) -> SpatialLocalGraphFusion:
    fusion = SpatialLocalGraphFusion(LocalGraphConfig(
        enabled=True, fusion_mode="spatial", **kwargs,
    ))
    fusion.reset_episode("scene", "episode", 0)
    return fusion


def _build(
    fusion: SpatialLocalGraphFusion, store: V3MemoryStore, *,
    step: int, pose: AgentPose | None, place_id: str | None = None,
    events: list[MemoryEvent] | None = None,
):
    return fusion.build(
        store, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=step, instruction="find the couch", current_place_id=place_id,
        rgb_width=300, events=events or [], current_pose=pose,
    )


def test_habitat_wxyz_heading_and_relative_viewpoint_geometry() -> None:
    root = math.sqrt(0.5)
    assert heading_from_quaternion((1, 0, 0, 0)) == pytest.approx(0)
    assert heading_from_quaternion((root, 0, root, 0)) == pytest.approx(math.pi / 2)
    assert heading_from_quaternion((root, 0, -root, 0)) == pytest.approx(-math.pi / 2)
    assert heading_from_quaternion((root * 3, 0, root * 3, 0)) == pytest.approx(math.pi / 2)
    assert heading_from_quaternion((0, 0, 1, 0)) == pytest.approx(math.pi)
    assert relative_viewpoint(
        _pose(), (3, 100, -4), target_scene_id="scene", target_frame_id="habitat_world",
    ) == pytest.approx((5, -math.atan2(3, 4)))
    assert relative_direction(-math.pi / 4, 1) == "right-front"
    assert relative_direction(math.pi / 4, 1) == "left-front"
    assert relative_direction(math.pi, 1) == "back"
    turned = AgentPose("scene", (0, 0, 0), (root, 0, root, 0))
    assert relative_viewpoint(
        turned, (-2, 0, 0), target_scene_id="scene", target_frame_id="habitat_world",
    ) == pytest.approx((2, 0))
    assert relative_viewpoint(
        _pose(), (0, 9, 0), target_scene_id="scene", target_frame_id="habitat_world",
    ) == (0, None)


def test_missing_bad_or_cross_frame_pose_cannot_generate_metric_facts() -> None:
    with pytest.raises(ValueError, match="different Scenes or coordinate frames"):
        relative_viewpoint(
            _pose(), (0, 0, -1), target_scene_id="other", target_frame_id="habitat_world",
        )
    with pytest.raises(ValueError, match="different Scenes or coordinate frames"):
        relative_viewpoint(
            _pose(), (0, 0, -1), target_scene_id="scene", target_frame_id="map_local",
        )
    with pytest.raises(ValueError, match="zero norm"):
        heading_from_quaternion((0, 0, 0, 0))
    with pytest.raises(ValueError, match="finite"):
        AgentPose("scene", (float("nan"), 0, 0), (1, 0, 0, 0))
    with pytest.raises(ValueError, match="scene differs"):
        AgentPose.from_habitat_info({
            "scene_id": "other", "agent_pose": {
                "position": [0, 0, 0], "rotation": [1, 0, 0, 0],
            },
        }, expected_scene_id="scene")
    assert AgentPose.from_habitat_info({"distance_to_goal": 0.1},
                                       expected_scene_id="scene") is None
    assert AgentPose.from_habitat_info({
        "scene_id": "scene", "agent_pose": {
            "position": [float("nan"), 0, 0], "rotation": [1, 0, 0, 0],
        },
    }, expected_scene_id="scene") is None


def test_two_couches_keep_distinct_places_and_real_viewpoint_geometry() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the couch")
    first, first_place, _ = _landmark_at(store, "couch_a", 0, (0, 0, -3), (1, 0))
    second, second_place, _ = _landmark_at(store, "couch_b", 1, (4, 0, -3), (0, 1))
    context = _build(_fusion(prompt_max_chars=500, neighbor_radius_m=7.0),
                     store, step=2, pose=_pose())
    assert context is not None
    assert len(context.layer_items) == 2
    assert context.text.count("History: couch") == 2
    fields = [item.metadata["spatial_fields"] for item in context.layer_items]
    assert {item["landmark_node_id"] for item in fields} == {
        node.node_id for node in store.nodes_for_scene("scene") if node.node_type == "landmark"
    }
    assert {item["source_place_node_id"] for item in fields} == {first_place, second_place}
    assert {first.key, second.key} <= {
        ref.event_id for item in context.layer_items for ref in item.evidence_refs
    }
    assert sorted(item["relative_distance"] for item in fields) == [3.0, 5.0]
    assert all(item["traversability_status"] == "unknown" for item in fields)
    assert len(context.text) <= 500
    default_radius = _build(_fusion(prompt_max_chars=500), store, step=2, pose=_pose())
    assert len(default_radius.layer_items) == 1
    assert default_radius.layer_items[0].metadata["spatial_fields"]["source_place_node_id"] == first_place


def test_inferred_edge_stays_unknown_until_independent_measured_traversal() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the couch")
    _, source_id, source_record = _landmark_at(store, "couch", 0, (0, 0, -2), (1, 0))
    motion = store.add(EvidenceRecord(
        "motion", "scene", "episode", 0, 1, "observed_motion",
        observation_status="observed",
        payload={"measurement_source": "odometry", "displacement_m": 2.0},
    ))
    current_record = store.add(_record(
        "current_place", 2, position=(0, 0, 0), descriptor=(0, 1),
    ))
    current_id = store.add_place(current_record).node_id
    store.add_relation(
        source_id, current_id, "inferred_relation",
        evidence_keys=(source_record.key, current_record.key),
    )
    context = _build(_fusion(), store, step=3, pose=_pose(), place_id=current_id)
    fields = context.layer_items[0].metadata["spatial_fields"]
    assert fields["graph_relation"] == "inferred_relation"
    assert fields["traversability_status"] == "unknown"
    assert "relative viewpoint 2.0m front" in context.text
    assert "passage unknown" in context.text
    video = object()
    base = {"video_segments": video, "conversations": [
        {"from": "human", "value": (
            "Visual observations <video>. <navigation_task>find the couch"
            "</navigation_task>. Predict trajectory."
        )},
        {"from": "gpt", "value": "placeholder"},
    ]}
    augmented = inject_memory_context(base, context, max_chars=320)
    assert audit_prompt_injection(base, augmented, context, max_chars=320)["roundtrip_ok"]
    assert augmented["video_segments"] is video
    assert "relative viewpoint 2.0m front" in augmented["conversations"][0]["value"]
    assert "<navigation_task>find the couch</navigation_task>" in augmented["conversations"][0]["value"]

    store.add_relation(
        source_id, current_id, "observed_transition",
        evidence_keys=(source_record.key, current_record.key),
    )
    observed = _build(_fusion(), store, step=3, pose=_pose(), place_id=current_id)
    observed_fields = observed.layer_items[0].metadata["spatial_fields"]
    assert observed_fields["graph_relation"] == "observed_transition"
    assert observed_fields["traversability_status"] == "unknown"

    store.add_relation(
        source_id, current_id, "verified_traversal",
        evidence_keys=(source_record.key, current_record.key, motion.key),
        motion_evidence_key=motion.key,
    )
    verified = _build(_fusion(), store, step=3, pose=_pose(), place_id=current_id)
    verified_fields = verified.layer_items[0].metadata["spatial_fields"]
    assert verified_fields["graph_relation"] == "verified_traversal"
    assert verified_fields["traversability_status"] == "traversed"
    assert motion.key in verified_fields["evidence_ids"]


def test_unknown_geometry_and_current_rgb_are_separate_and_bounded() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the couch")
    detection = store.add(_record("old_couch", 0, position=None, frame=None))
    landmark = store.add_landmark(detection)
    place_record = store.add(_record(
        "old_place", 0, position=None, descriptor=(1, 0), frame=None,
    ))
    place_id = store.add_place(place_record).node_id
    store.add_relation(
        place_id, landmark.node_id, "inferred_relation",
        evidence_keys=(place_record.key, detection.key),
    )
    now = store.add(_record("now_couch", 2, position=(1, 0, 0)))
    event = MemoryEvent(
        event_id=now.evidence_id, step_id=2, kind="object_observation",
        description="Observed couch", confidence=0.91, source="yolo",
        episode_id="episode", metadata={"bbox_xyxy": [0, 0, 80, 80]},
    )
    context = _build(
        _fusion(prompt_max_chars=320), store, step=2, pose=_pose(1),
        place_id=place_id, events=[event],
    )
    assert context is not None
    assert len(context.text) <= 320
    assert "Current RGB:" in context.text
    assert "History:" in context.text
    assert "distance unknown, bearing unknown" in context.text
    assert "passage unknown" in context.text
    assert "detector 0.91" in context.text
    assert context.layer_items[1].metadata["spatial_fields"]["relative_distance"] is None
    assert context.layer_items[1].metadata["spatial_fields"]["source_position"] is None
    tight = _build(
        _fusion(prompt_max_chars=220), store, step=2, pose=_pose(1),
        place_id=place_id, events=[event],
    )
    assert tight is not None
    assert len(tight.text) <= 220
    assert "History:" in tight.text


def test_metric_radius_and_frame_mismatch_do_not_fabricate_local_space() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the couch")
    _landmark_at(store, "far_couch", 0, (0, 0, -8), (1, 0))
    assert _build(_fusion(), store, step=1, pose=_pose()) is None

    detection = store.add(_record(
        "wrong_frame_couch", 1, position=(0, 0, -2), frame="another_frame",
    ))
    landmark = store.add_landmark(detection)
    place_record = store.add(_record(
        "wrong_frame_place", 1, position=(0, 0, -2), descriptor=(0, 1),
        frame="another_frame",
    ))
    place = store.add_place(place_record)
    store.add_relation(
        place.node_id, landmark.node_id, "inferred_relation",
        evidence_keys=(place_record.key, detection.key),
    )
    context = _build(_fusion(), store, step=2, pose=_pose(), place_id=place.node_id)
    assert context is not None
    assert "distance unknown, bearing unknown" in context.text
    fields = context.layer_items[0].metadata["spatial_fields"]
    assert fields["source_frame_id"] == "another_frame"
    assert fields["relative_distance"] is None


def test_only_measured_two_edge_place_path_is_reported_as_verified() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the couch")
    _, start_id, start_record = _landmark_at(store, "couch", 0, (0, 0, -2), (1, 0))
    middle_record = store.add(_record(
        "middle", 1, position=(0, 0, -1), descriptor=(0, 1),
    ))
    middle_id = store.add_place(middle_record).node_id
    end_record = store.add(_record(
        "end", 2, position=(0, 0, 0), descriptor=(-1, 0),
    ))
    end_id = store.add_place(end_record).node_id
    motion1 = store.add(EvidenceRecord(
        "motion1", "scene", "episode", 0, 1, "observed_motion",
        observation_status="observed",
        payload={"measurement_source": "odometry", "displacement_m": 1.0},
    ))
    motion2 = store.add(EvidenceRecord(
        "motion2", "scene", "episode", 0, 2, "observed_motion",
        observation_status="observed",
        payload={"measurement_source": "odometry", "displacement_m": 1.0},
    ))
    store.add_relation(
        start_id, middle_id, "verified_traversal",
        evidence_keys=(start_record.key, middle_record.key, motion1.key),
        motion_evidence_key=motion1.key,
    )
    store.add_relation(
        middle_id, end_id, "verified_traversal",
        evidence_keys=(middle_record.key, end_record.key, motion2.key),
        motion_evidence_key=motion2.key,
    )
    context = _build(_fusion(), store, step=3, pose=_pose(), place_id=end_id)
    fields = context.layer_items[0].metadata["spatial_fields"]
    assert fields["graph_relation"] == "verified_path"
    assert fields["traversability_status"] == "traversed"
    assert len(fields["graph_edge_ids"]) == 2
    assert {motion1.key, motion2.key} <= set(fields["evidence_ids"])
