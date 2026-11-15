"""CPU geometry contracts for the independent ConceptGraphs-Lite L3 map."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from lightnav.memory.conceptgraphs_lite import (
    CameraIntrinsics,
    GeometryConfig,
    InstanceMask,
    ObjectMap,
    Pose3D,
    RGBDFrame,
    _n_ratio,
    backproject_world,
)
from lightnav.memory.v3_schemas import canonical_evidence_key


def _frame(
    *, width: int = 8, height: int = 8, depth: np.ndarray | None = None,
    agent_x: float = 0.0, agent_pose: Pose3D | None = None,
    camera_extrinsics: Pose3D | None = None,
) -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        depth_m=(np.full((height, width), 2.0, dtype=np.float32)
                 if depth is None else depth),
        intrinsics=CameraIntrinsics(
            fx=float(width), fy=float(width), cx=(width - 1) / 2,
            cy=(height - 1) / 2, width=width, height=height,
        ),
        agent_pose=agent_pose or Pose3D((agent_x, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        camera_extrinsics=camera_extrinsics or Pose3D(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0),
        ),
    )


def _mask(
    left: int, right: int, *, width: int = 8, height: int = 8,
    top: int = 2, bottom: int = 6,
) -> np.ndarray:
    value = np.zeros((height, width), dtype=bool)
    value[top:bottom, left:right] = True
    return value


def _instance(name: str, mask: np.ndarray, number: int, *, scene: str = "scene") -> InstanceMask:
    return InstanceMask(
        class_name=name, confidence=0.9, mask=mask,
        evidence_id=canonical_evidence_key(scene, "episode", 0, f"seg_{number}"),
    )


def _map() -> ObjectMap:
    graph = ObjectMap(GeometryConfig())
    graph.reset_episode("scene", "episode")
    return graph


def test_rgbd_alignment_rejects_shape_and_intrinsics_mismatch() -> None:
    with pytest.raises(ValueError):
        _frame(depth=np.ones((7, 8), dtype=np.float32))
    frame = _frame()
    bad_mask = np.ones((7, 8), dtype=bool)
    with pytest.raises(ValueError):
        backproject_world(bad_mask, frame, GeometryConfig(min_points=1))
    with pytest.raises(ValueError):
        RGBDFrame(
            rgb=frame.rgb, depth_m=frame.depth_m,
            intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 7, 8),
            agent_pose=frame.agent_pose, camera_extrinsics=frame.camera_extrinsics,
        )


def test_invalid_depth_pixels_are_excluded_before_backprojection() -> None:
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    depth[0, :] = [0.0, np.nan, np.inf]
    depth[1, :] = [-1.0, 0.05, 11.0]
    frame = _frame(width=3, height=3, depth=depth)
    points = backproject_world(
        np.ones((3, 3), dtype=bool), frame, GeometryConfig(min_points=1),
    )
    assert points.shape == (3, 3)
    assert np.isfinite(points).all()
    assert np.allclose(points[:, 2], -2.0)


def test_habitat_normalized_depth_requires_calibration_and_censors_endpoints() -> None:
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    depth = np.full((3, 3), 0.5, dtype=np.float32)
    depth[0, 0] = 0.0
    depth[0, 1] = 1.0
    metadata = {
        "status": "ok", "aligned": True, "camera_frame": "habitat_right_up_back",
        "image_size_hw": [3, 3],
        "K": [[3.0, 0.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0]],
        "agent_pose": {"position": [1.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        "camera_extrinsics": {
            "position": [0.0, 0.88, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0],
        },
        "depth_encoding": "normalized_0_1", "min_depth_m": 0.5, "max_depth_m": 5.0,
    }
    frame = RGBDFrame.from_habitat({"rgb": rgb, "depth": depth}, {"spatial_rgbd": metadata})
    assert np.isnan(frame.depth_m[0, 0])
    assert np.isnan(frame.depth_m[0, 1])
    assert frame.depth_m[1, 1] == pytest.approx(2.75)
    assert frame.camera_pose.position == pytest.approx((1.0, 0.88, 0.0))
    with pytest.raises(ValueError):
        RGBDFrame.from_habitat({"rgb": rgb, "depth": depth}, {})
    with pytest.raises(ValueError):
        RGBDFrame.from_habitat(
            {"rgb": rgb, "depth": depth},
            {"spatial_rgbd": {**metadata, "image_size_hw": [3, 4]}},
        )


def test_habitat_metric_depth_filters_range_and_rejects_ambiguous_camera_frame() -> None:
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    depth = np.array([
        [0.0, 0.1, 0.5],
        [2.0, 5.0, np.inf],
        [np.nan, -1.0, 3.0],
    ], dtype=np.float32)
    metadata = {
        "status": "ok", "aligned": True, "camera_frame": "habitat_right_up_back",
        "image_size_hw": [3, 3],
        "K": [[3.0, 0.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0]],
        "agent_pose": {"position": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        "camera_extrinsics": {
            "position": [0.0, 0.88, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0],
        },
        "depth_encoding": "meters", "min_depth_m": 0.1, "max_depth_m": 5.0,
    }
    frame = RGBDFrame.from_habitat({"rgb": rgb, "depth": depth}, {"spatial_rgbd": metadata})
    assert np.isfinite(frame.depth_m).sum() == 3
    assert frame.depth_m[0, 2] == pytest.approx(0.5)
    assert frame.depth_m[1, 0] == pytest.approx(2.0)
    assert frame.depth_m[2, 2] == pytest.approx(3.0)
    assert np.isnan(frame.depth_m[0, 1]) and np.isnan(frame.depth_m[1, 1])
    with pytest.raises(ValueError, match="camera coordinate"):
        RGBDFrame.from_habitat(
            {"rgb": rgb, "depth": depth},
            {"spatial_rgbd": {**metadata, "camera_frame": "unknown"}},
        )
    with pytest.raises(ValueError, match="quaternion"):
        RGBDFrame.from_habitat(
            {"rgb": rgb, "depth": depth},
            {"spatial_rgbd": {
                **metadata,
                "agent_pose": {"position": [0, 0, 0], "rotation": [0, 0, 0, 0]},
            }},
        )


def test_intrinsics_backproject_a_known_optical_pixel() -> None:
    frame = _frame(width=3, height=3)
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 2] = True
    points = backproject_world(mask, frame, GeometryConfig(min_points=1))
    assert points.shape == (1, 3)
    # Habitat camera: +X right, +Y up, -Z forward; K is at native RGB size.
    assert points[0] == pytest.approx((2 / 3, 2 / 3, -2.0))


def test_agent_sensor_composition_and_world_transform_use_camera_pose() -> None:
    q_yaw_90 = (sqrt(0.5), 0.0, sqrt(0.5), 0.0)
    agent = Pose3D((1.0, 0.0, 0.0), q_yaw_90)
    sensor = Pose3D((0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    camera = agent.compose(sensor)
    assert camera.position == pytest.approx((1.0, 1.0, 0.0))
    assert camera.position != agent.position
    frame = _frame(
        width=3, height=3, agent_pose=agent, camera_extrinsics=sensor,
    )
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    points = backproject_world(mask, frame, GeometryConfig(min_points=1))
    assert points[0] == pytest.approx((-1.0, 1.0, 0.0), abs=1e-6)


def test_same_object_from_shifted_viewpoint_updates_one_node() -> None:
    graph = _map()
    first = _instance("chair", _mask(2, 6), 0)
    second = _instance("chair", _mask(1, 5), 1)
    first_node = graph.observe(_frame(), [first], step_id=0)[0]
    second_node = graph.observe(
        _frame(agent_x=0.25), [second], step_id=1, transition_record=True,
    )[0]
    assert second_node.object_id == first_node.object_id
    assert second_node.observation_count == 2
    assert set(second_node.evidence_ids) == {first.evidence_id, second.evidence_id}
    assert second_node.bbox_3d is not None
    assert second_node.center_world is not None
    assert len(graph.objects) == 1


def test_adjacent_same_class_instances_do_not_merge() -> None:
    graph = _map()
    nodes = graph.observe(
        _frame(), [
            _instance("chair", _mask(0, 3, top=0, bottom=8), 0),
            _instance("chair", _mask(5, 8, top=0, bottom=8), 1),
        ], step_id=0,
    )
    assert len({node.object_id for node in nodes}) == 2
    assert len(graph.objects) == 2


def test_nearby_same_class_instances_in_separate_frames_stay_distinct() -> None:
    graph = _map()
    first = graph.observe(
        _frame(), [_instance("chair", _mask(2, 4, top=0, bottom=8), 0)], step_id=0,
    )[0]
    second = graph.observe(
        _frame(), [_instance("chair", _mask(4, 6, top=0, bottom=8), 1)], step_id=1,
    )[0]
    assert first.object_id != second.object_id


def test_class_mismatch_never_forces_merge() -> None:
    graph = _map()
    first = graph.observe(_frame(), [_instance("chair", _mask(2, 6), 0)], step_id=0)[0]
    second = graph.observe(
        _frame(), [_instance("sofa", _mask(2, 6), 1)], step_id=1,
    )[0]
    assert first.object_id != second.object_id


def test_ambiguous_equal_geometry_rejects_merge() -> None:
    graph = _map()
    # At this resolution, adjacent columns are 0.01 m apart at 2 m depth.
    # Historical surfaces are 0.04 m apart, beyond the 0.03 m n_ratio radius.
    mask = _mask(95, 96, width=200, top=0, bottom=8)
    left = graph.observe(_frame(width=200), [_instance("chair", mask, 0)], step_id=0)[0]
    right = graph.observe(
        _frame(width=200, agent_x=0.04), [_instance("chair", mask, 1)], step_id=1,
    )[0]
    assert left.object_id != right.object_id
    # Every new point is 0.02 m from either old surface: both n_ratio scores
    # are 1, so the candidate must abstain rather than merge either object.
    center = graph.observe(
        _frame(width=200, agent_x=0.02), [_instance("chair", mask, 2)], step_id=2,
    )[0]
    assert center.object_id not in {left.object_id, right.object_id}
    assert len(graph.objects) == 3


def test_same_frame_competition_keeps_consumed_node_in_ambiguity_check() -> None:
    graph = _map()
    old_mask = _mask(95, 96, width=200, top=0, bottom=8)
    first = graph.observe(
        _frame(width=200), [_instance("chair", old_mask, 0)], step_id=0,
    )[0]
    second = graph.observe(
        _frame(width=200, agent_x=0.04), [_instance("chair", old_mask, 1)], step_id=1,
    )[0]
    assert first.object_id != second.object_id

    # The first current mask uniquely prefers first; the second sits halfway
    # between both historical nodes. Removing first before evaluating the
    # second would hide an equal-scoring candidate and incorrectly merge second.
    unique = _instance("chair", old_mask, 2)
    ambiguous = _instance("chair", _mask(97, 98, width=200, top=0, bottom=8), 3)
    observed = graph.observe(
        _frame(width=200), [unique, ambiguous], step_id=2,
    )
    assert observed[0].object_id == first.object_id
    assert observed[1].object_id not in {first.object_id, second.object_id}
    assert first.observation_count == 2
    assert second.observation_count == 1
    assert len(graph.objects) == 3


def test_greedy_assignment_is_one_to_one_and_prefers_higher_n_ratio() -> None:
    graph = _map()
    old_mask = _mask(95, 96, width=200, top=0, bottom=8)
    old = graph.observe(
        _frame(width=200), [_instance("chair", old_mask, 0)], step_id=0,
    )[0]

    # The wider detection is intentionally listed first. It has only a
    # partial n_ratio overlap with old, while the later detection overlaps
    # exactly. Greedy assignment must give the sole old node to the latter.
    partial = _instance("chair", _mask(96, 100, width=200, top=0, bottom=8), 1)
    exact = _instance("chair", old_mask, 2)
    observed = graph.observe(_frame(width=200), [partial, exact], step_id=1)
    assert observed[0].object_id != old.object_id
    assert observed[1].object_id == old.object_id
    assert observed[0].object_id != observed[1].object_id
    assert old.observation_count == 2
    assert observed[0].observation_count == 1
    assert len(graph.objects) == 2


def test_n_ratio_uses_new_detection_point_count_as_denominator() -> None:
    small = _mask(95, 96, width=200, top=0, bottom=8)
    wide = _mask(95, 101, width=200, top=0, bottom=8)
    config = GeometryConfig(min_overlap_ratio=0.75)

    graph = ObjectMap(config)
    graph.reset_episode("scene", "episode")
    old_small = graph.observe(
        _frame(width=200), [_instance("chair", small, 0)], step_id=0,
    )[0]
    new_wide = graph.observe(
        _frame(width=200), [_instance("chair", wide, 1)], step_id=1,
    )[0]
    assert new_wide.object_id != old_small.object_id

    reverse = ObjectMap(config)
    reverse.reset_episode("scene", "episode")
    old_wide = reverse.observe(
        _frame(width=200), [_instance("chair", wide, 0)], step_id=0,
    )[0]
    new_small = reverse.observe(
        _frame(width=200), [_instance("chair", small, 1)], step_id=1,
    )[0]
    assert new_small.object_id == old_wide.object_id


def test_n_ratio_uses_strict_voxel_distance_and_detection_denominator() -> None:
    mapped = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    detection = np.array([
        [0.0, 0.0, 0.0],
        [0.0625, 0.0, 0.0],
        [0.125, 0.0, 0.0],
        [0.25, 0.0, 0.0],
    ], dtype=np.float64)
    assert _n_ratio(detection, mapped, 0.125) == 0.5
    assert _n_ratio(mapped, detection, 0.125) == 1.0


def test_incremental_cloud_update_preserves_both_evidence_and_observations() -> None:
    graph = _map()
    mask = _mask(95, 96, width=200, top=0, bottom=8)
    first = _instance("chair", mask, 0)
    node = graph.observe(_frame(width=200), [first], step_id=0)[0]
    original_points = node.points_world.copy()
    original_max_x = node.bbox_3d[1][0]

    second = _instance("chair", mask, 1)
    updated = graph.observe(
        _frame(width=200, agent_x=0.02), [second], step_id=1,
        transition_record=True,
    )[0]
    assert updated is node
    assert updated.observation_count == 2
    assert updated.evidence_ids == [first.evidence_id, second.evidence_id]
    assert updated.bbox_3d[1][0] > original_max_x
    assert len(updated.points_world) > len(original_points)
    assert {edge.evidence_id for edge in graph.edges if edge.relation_type == "observed_from"} == {
        first.evidence_id, second.evidence_id,
    }
    for step, instance in enumerate((first, second)):
        record = graph.evidence[instance.evidence_id]
        assert record.step_id == step
        assert record.geometry_status == "observed"
        assert record.depth_point_count >= graph.config.min_points


def test_observed_from_requires_valid_geometry_and_evidence() -> None:
    graph = _map()
    first = _instance("chair", _mask(2, 6), 0)
    graph.observe(_frame(), [first], step_id=0)
    assert len(graph.viewpoints) == 1
    observed = [edge for edge in graph.edges if edge.relation_type == "observed_from"]
    assert len(observed) == 1
    assert observed[0].evidence_id == first.evidence_id
    assert graph.evidence[first.evidence_id].geometry_status == "observed"
    blank = _frame(depth=np.zeros((8, 8), dtype=np.float32))
    second = _instance("chair", _mask(2, 6), 1)
    graph.observe(blank, [second], step_id=1)
    assert len([edge for edge in graph.edges if edge.relation_type == "observed_from"]) == 1
    rejected = graph.evidence[second.evidence_id]
    assert rejected.geometry_status == "rejected_depth"
    assert rejected.mask_pixel_count == 16 and rejected.depth_point_count == 0


def test_independent_segmentation_evidence_preserves_optional_l1_link() -> None:
    graph = _map()
    l1_id = canonical_evidence_key("scene", "episode", 0, "l1_detection_0")
    l3_id = canonical_evidence_key("scene", "episode", 0, "l3_seg_0_0")
    segment = InstanceMask(
        "chair", 0.91, _mask(2, 6), l3_id,
        source="ultralytics_yolo_seg:yolo11n-seg.pt",
        linked_l1_evidence_id=l1_id,
    )
    observed = graph.observe(_frame(), [segment], step_id=0)
    assert len(observed) == 1
    assert observed[0].evidence_ids == [l3_id]
    record = graph.evidence[l3_id]
    assert record.evidence_id == l3_id
    assert (record.scene_id, record.episode_id, record.rollout_idx, record.step_id) == (
        "scene", "episode", 0, 0,
    )
    assert record.source == "ultralytics_yolo_seg:yolo11n-seg.pt"
    assert record.linked_l1_evidence_id == l1_id


def test_cross_episode_evidence_is_rejected_without_partial_graph_mutation() -> None:
    graph = _map()
    foreign = InstanceMask(
        "chair", 0.9, _mask(2, 6),
        canonical_evidence_key("scene", "other_episode", 0, "l3_seg_0_0"),
    )
    with pytest.raises(ValueError, match="Scene/Episode"):
        graph.observe(_frame(), [foreign], step_id=0)
    assert not graph.evidence and not graph.viewpoints and not graph.objects and not graph.edges


def test_traversed_to_requires_recorded_transition_and_real_pose_change() -> None:
    graph = _map()
    graph.observe(_frame(), [], step_id=0)
    graph.observe(_frame(agent_x=0.2), [], step_id=1, transition_record=False)
    assert not any(edge.relation_type == "traversed_to" for edge in graph.edges)
    graph.observe(_frame(agent_x=0.2), [], step_id=2, transition_record=True)
    assert not any(edge.relation_type == "traversed_to" for edge in graph.edges)
    graph.observe(_frame(agent_x=0.4), [], step_id=3, transition_record=True)
    assert len([edge for edge in graph.edges if edge.relation_type == "traversed_to"]) == 1
    assert {edge.relation_type for edge in graph.edges} <= {"observed_from", "traversed_to"}


def test_episode_and_scene_reset_isolates_object_identity() -> None:
    graph = _map()
    old = graph.observe(_frame(), [_instance("chair", _mask(2, 6), 0)], step_id=0)[0]
    graph.reset_episode("scene", "next_episode")
    assert not graph.objects and not graph.edges and not graph.viewpoints
    graph.reset_episode("other_scene", "episode")
    new = graph.observe(
        _frame(), [_instance("chair", _mask(2, 6), 1, scene="other_scene")], step_id=0,
    )[0]
    assert new is not old
    assert old.evidence_ids[0] not in new.evidence_ids
