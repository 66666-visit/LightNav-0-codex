"""CPU contracts for the read-only per-Evidence XYZRGB sidecar."""

from __future__ import annotations

import numpy as np
import pytest

from lightnav.memory.conceptgraphs_lite import (
    CameraIntrinsics,
    GeometryConfig,
    InstanceMask,
    ObjectNode,
    Pose3D,
    RGBDFrame,
    backproject_world,
)
from lightnav.memory.rgb_sidecar import RGBSidecar
from lightnav.memory.v3_schemas import canonical_evidence_key

SCENE = "scene"
EPISODE = "episode"


def _frame(rgb: np.ndarray, depth: np.ndarray) -> RGBDFrame:
    height, width = rgb.shape[:2]
    intrinsics = CameraIntrinsics(
        fx=2.0, fy=2.0, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
        width=width, height=height,
    )
    return RGBDFrame(
        rgb=rgb, depth_m=depth, intrinsics=intrinsics,
        agent_pose=Pose3D((0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        camera_extrinsics=Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    )


def _mask(evidence_id: str, shape=(4, 4), values=None) -> InstanceMask:
    mask = np.ones(shape, dtype=np.bool_) if values is None else values
    return InstanceMask(
        class_name="couch", confidence=0.9, mask=mask,
        evidence_id=evidence_id, source="unit-test",
    )


def _key(name: str) -> str:
    return canonical_evidence_key(SCENE, EPISODE, 0, name)


def test_sidecar_xyz_is_exactly_backproject_world_and_rgb_is_real_pixels() -> None:
    rgb = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    depth = np.full((4, 4), 2.0, dtype=np.float64)
    frame = _frame(rgb, depth)
    evidence = _key("l3_seg_1_0")
    mask = _mask(evidence)
    sidecar = RGBSidecar(GeometryConfig(min_points=1))
    sidecar.reset_episode(SCENE, EPISODE, 0)
    sidecar.record_frame(frame, [mask], step_id=1)

    expected = backproject_world(mask.mask, frame, sidecar.config)
    record = sidecar.record_for(evidence)
    assert record is not None
    assert np.array_equal(record.points_world, expected)
    assert np.array_equal(record.colors_rgb, rgb[mask.mask])
    assert record.colors_rgb.dtype == np.uint8
    assert record.point_count == int(mask.mask.sum())


def test_sidecar_uses_the_same_linspace_subsample_as_backproject_world() -> None:
    rgb = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    depth = np.full((4, 4), 2.0, dtype=np.float64)
    frame = _frame(rgb, depth)
    evidence = _key("l3_seg_1_0")
    mask_values = np.zeros((4, 4), dtype=np.bool_)
    mask_values[0, :] = True
    mask_values[2, :] = True
    mask = _mask(evidence, values=mask_values)
    config = GeometryConfig(min_points=1, max_points_per_instance=4)
    sidecar = RGBSidecar(config)
    sidecar.reset_episode(SCENE, EPISODE, 0)
    sidecar.record_frame(frame, [mask], step_id=1)

    expected = backproject_world(mask.mask, frame, config)
    rows, cols = np.nonzero(mask_values)
    indices = np.linspace(0, len(rows) - 1, 4, dtype=np.int64)
    expected_rgb = rgb[rows[indices], cols[indices]]
    record = sidecar.record_for(evidence)
    assert record is not None
    assert np.array_equal(record.points_world, expected)
    assert np.array_equal(record.colors_rgb, expected_rgb)
    assert record.point_count == 4


def test_sidecar_preserves_evidence_to_object_mapping_and_row_pairing() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(16, dtype=np.uint8).reshape(4, 4) * 10
    rgb[..., 1] = 7
    rgb[..., 2] = 3
    frame = _frame(rgb, np.full((4, 4), 2.0, dtype=np.float64))
    first, second = _key("l3_seg_1_0"), _key("l3_seg_1_1")
    mask_a = np.zeros((4, 4), dtype=np.bool_)
    mask_a[:, :2] = True
    mask_b = np.zeros((4, 4), dtype=np.bool_)
    mask_b[:, 2:] = True
    sidecar = RGBSidecar(GeometryConfig(min_points=1))
    sidecar.reset_episode(SCENE, EPISODE, 0)
    sidecar.record_frame(
        frame, [_mask(first, values=mask_a), _mask(second, values=mask_b)],
        step_id=1,
    )
    node = ObjectNode(
        "object-A", "couch", (0.0, 0.0, 0.0), ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        2, [first, second], "observed", np.zeros((2, 3)),
    )
    other = ObjectNode(
        "object-B", "chair", (1.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
        1, [first], "observed", np.zeros((1, 3)),
    )
    with pytest.raises(ValueError, match="two ObjectNodes"):
        sidecar.bind_objects([node, other])

    sidecar.bind_objects([node])
    assert sidecar.record_for(first).object_id == "object-A"
    assert sidecar.record_for(second).object_id == "object-A"
    proposal = sidecar.proposal_for(node)
    assert proposal.object_id == "object-A"
    assert proposal.point_count == 16
    # The concatenated proposal is exactly the two raw records, in evidence order.
    first_record, second_record = sidecar.record_for(first), sidecar.record_for(second)
    assert np.array_equal(
        proposal.points_world_rgb[: first_record.point_count, :3],
        first_record.points_world.astype(np.float32),
    )
    assert np.array_equal(
        proposal.points_world_rgb[: first_record.point_count, 3:],
        first_record.colors_rgb.astype(np.float32),
    )
    offset = first_record.point_count
    assert np.array_equal(
        proposal.points_world_rgb[offset:, 3:],
        second_record.colors_rgb.astype(np.float32),
    )


def test_sidecar_missing_evidence_is_reported_and_never_synthesized() -> None:
    frame = _frame(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.full((4, 4), 2.0, dtype=np.float64),
    )
    sidecar = RGBSidecar(GeometryConfig(min_points=1))
    sidecar.reset_episode(SCENE, EPISODE, 0)
    first = _key("l3_seg_1_0")
    second = _key("l3_seg_1_1")
    sidecar.record_frame(frame, [_mask(first)], step_id=1)
    node = ObjectNode(
        "object-A", "couch", (0.0, 0.0, 0.0), ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        2, [first, second], "observed", np.zeros((2, 3)),
    )
    with pytest.raises(ValueError, match="missing RGB"):
        sidecar.proposal_for(node)
    proposals, skipped = sidecar.proposals({node.object_id: node})
    assert proposals == []
    assert skipped == [{
        "object_id": "object-A",
        "class_name": "couch",
        "reason": "missing_rgb_evidence",
        "missing_evidence": [second],
    }]


def test_sidecar_records_only_when_identity_is_active_and_resets() -> None:
    frame = _frame(
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.full((4, 4), 2.0, dtype=np.float64),
    )
    sidecar = RGBSidecar(GeometryConfig(min_points=1))
    with pytest.raises(RuntimeError, match="no active Episode"):
        sidecar.record_frame(frame, [_mask(_key("l3_seg_1_0"))], step_id=1)
    sidecar.reset_episode(SCENE, EPISODE, 0)
    sidecar.record_frame(frame, [_mask(_key("l3_seg_1_0"))], step_id=1)
    assert sidecar.summary()["records"] == 1
    sidecar.reset_episode(SCENE, EPISODE, 0)
    assert sidecar.summary()["records"] == 0


def test_sidecar_excludes_invalid_and_out_of_range_depth_pixels() -> None:
    rgb = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    depth = np.full((4, 4), 2.0, dtype=np.float64)
    depth[0, 0] = np.nan
    depth[0, 1] = 0.05          # below min_depth_m (0.1)
    depth[0, 2] = 20.0          # above max_depth_m (10.0)
    depth[0, 3] = 9.99
    frame = _frame(rgb, depth)
    evidence = _key("l3_seg_1_0")
    mask = _mask(evidence)
    config = GeometryConfig(min_points=1)
    sidecar = RGBSidecar(config)
    sidecar.reset_episode(SCENE, EPISODE, 0)
    sidecar.record_frame(frame, [mask], step_id=1)
    expected = backproject_world(mask.mask, frame, config)
    valid = (
        mask.mask & np.isfinite(depth)
        & (depth >= config.min_depth_m) & (depth <= config.max_depth_m)
    )
    rows, cols = np.nonzero(valid)
    record = sidecar.record_for(evidence)
    assert record is not None
    assert np.array_equal(record.points_world, expected)
    assert np.array_equal(record.colors_rgb, rgb[rows, cols])
    assert record.point_count == 13
