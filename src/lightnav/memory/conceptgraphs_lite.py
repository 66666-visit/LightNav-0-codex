"""Conservative, evidence-linked RGB-D instance map for Habitat navigation.

Coordinates follow Habitat: world and camera frames are right-handed, +Y is up,
and a camera looks along local -Z. Quaternions are explicitly (w, x, y, z).
Only observed geometry and recorded environment transitions create graph edges.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from lightnav.memory.clip_features import fuse_clip_feature
from lightnav.memory.v3_schemas import parse_evidence_key


def _vector(values: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or len(values) != length:
        raise ValueError(f"{name} must have {length} numeric values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _quaternion_multiply(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    a, b, c, d = first
    w, x, y, z = second
    return (
        a * w - b * x - c * y - d * z,
        a * x + b * w + c * z - d * y,
        a * y - b * z + c * w + d * x,
        a * z + b * y - c * x + d * w,
    )


@dataclass(frozen=True)
class Pose3D:
    """Translation in metres and a unit quaternion in Habitat wxyz order."""

    position: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        position = _vector(self.position, 3, "pose position")
        rotation = _vector(self.rotation_wxyz, 4, "pose rotation_wxyz")
        norm = math.sqrt(sum(value * value for value in rotation))
        if not 0.99 <= norm <= 1.01:
            raise ValueError("pose quaternion must have unit length in wxyz order")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation_wxyz", tuple(value / norm for value in rotation))

    @classmethod
    def from_dict(cls, data: Any) -> Pose3D:
        if not isinstance(data, dict):
            raise ValueError("pose must be a mapping")
        # The Habitat server explicitly serializes [w,x,y,z]; do not guess xyzw.
        return cls(data["position"], data["rotation"])

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("points must be finite Nx3 coordinates")
        w, x, y, z = self.rotation_wxyz
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        return values @ rotation.T + np.asarray(self.position, dtype=np.float64)

    def compose(self, child: Pose3D) -> Pose3D:
        """Return world_T_child = world_T_self @ self_T_child."""
        translated = self.transform_points(np.asarray([child.position]))[0]
        return Pose3D(
            tuple(float(value) for value in translated),
            _quaternion_multiply(self.rotation_wxyz, child.rotation_wxyz),
        )


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int or min(
            self.width, self.height,
        ) < 1:
            raise ValueError("camera width and height must be positive integers")
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(value) for value in values) or self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be finite and positive")
        if not 0 <= self.cx < self.width or not 0 <= self.cy < self.height:
            raise ValueError("camera principal point lies outside the image")

    @classmethod
    def from_matrix(cls, matrix: Any, *, width: int, height: int) -> CameraIntrinsics:
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != (3, 3) or not np.isfinite(values).all():
            raise ValueError("K must be a finite 3x3 matrix")
        if not np.allclose(values[2], [0, 0, 1]) or not np.allclose(
            [values[0, 1], values[1, 0]], [0, 0], atol=1e-8,
        ):
            raise ValueError("K must have zero skew and homogeneous bottom row")
        return cls(float(values[0, 0]), float(values[1, 1]),
                   float(values[0, 2]), float(values[1, 2]), width, height)


@dataclass(frozen=True)
class GeometryConfig:
    min_depth_m: float = 0.1
    max_depth_m: float = 10.0
    min_points: int = 8
    max_points_per_instance: int = 256
    max_points_per_object: int = 2048
    voxel_size_m: float = 0.03
    max_center_distance_m: float = 0.75
    min_overlap_ratio: float = 0.35
    min_bbox_iou: float = 0.01
    ambiguity_margin_ratio: float = 0.12
    min_travel_distance_m: float = 0.01

    def __post_init__(self) -> None:
        for name in ("min_points", "max_points_per_instance", "max_points_per_object"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"geometry.{name} must be a positive integer")
        if not self.min_points <= self.max_points_per_instance <= self.max_points_per_object:
            raise ValueError("geometry point limits must be ordered")
        for name in (
            "min_depth_m", "max_depth_m", "voxel_size_m", "max_center_distance_m",
            "min_travel_distance_m",
        ):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"geometry.{name} must be finite and positive")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must exceed min_depth_m")
        for name in ("min_overlap_ratio", "ambiguity_margin_ratio"):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"geometry.{name} must lie in (0, 1]")
        if (not isinstance(self.min_bbox_iou, (float, int))
                or not math.isfinite(self.min_bbox_iou)
                or not 0 <= self.min_bbox_iou <= 1):
            raise ValueError("geometry.min_bbox_iou must lie in [0, 1]")


@dataclass(frozen=True)
class RGBDFrame:
    """One synchronized RGB, aligned metric depth, calibration and two poses."""

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    agent_pose: Pose3D
    camera_extrinsics: Pose3D

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB must be HWC uint8 with three channels")
        if depth.ndim != 2 or depth.shape != rgb.shape[:2] or not np.issubdtype(
            depth.dtype, np.floating,
        ):
            raise ValueError("depth must be numeric HW aligned to RGB")
        if rgb.shape[:2] != (self.intrinsics.height, self.intrinsics.width):
            raise ValueError("RGB size differs from camera intrinsics")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth.astype(np.float64, copy=False))

    @property
    def camera_pose(self) -> Pose3D:
        return self.agent_pose.compose(self.camera_extrinsics)

    @classmethod
    def from_habitat(cls, obs: dict[str, Any], info: dict[str, Any]) -> RGBDFrame:
        """Decode server metadata, rejecting missing calibration or ambiguous units."""
        if not isinstance(info, dict) or not isinstance(info.get("spatial_rgbd"), dict):
            raise ValueError("Habitat spatial_rgbd metadata is missing")
        metadata = info["spatial_rgbd"]
        if metadata.get("status") != "ok" or metadata.get("aligned") is not True:
            raise ValueError(f"RGB-D calibration unavailable: {metadata.get('reason', 'unaligned')}")
        if metadata.get("camera_frame") != "habitat_right_up_back":
            raise ValueError("RGB-D camera coordinate convention is unavailable")
        if not isinstance(obs, dict) or "rgb" not in obs or "depth" not in obs:
            raise ValueError("Habitat RGB and depth must come from the same observation")
        rgb = np.asarray(obs["rgb"])
        source_depth = np.asarray(obs["depth"])
        if not np.issubdtype(source_depth.dtype, np.floating):
            raise ValueError("Habitat depth must be a floating-point image")
        depth = source_depth.astype(np.float64)
        if rgb.ndim != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("Habitat RGB-D image shapes differ")
        shape = _vector(metadata["image_size_hw"], 2, "image_size_hw")
        height, width = int(shape[0]), int(shape[1])
        if (height, width) != rgb.shape[:2] or shape != (height, width):
            raise ValueError("Habitat calibration size differs from RGB-D")
        intrinsics = CameraIntrinsics.from_matrix(metadata["K"], width=width, height=height)
        agent_pose = Pose3D.from_dict(metadata["agent_pose"])
        extrinsics = Pose3D.from_dict(metadata["camera_extrinsics"])
        encoding = metadata.get("depth_encoding")
        near = float(metadata["min_depth_m"])
        far = float(metadata["max_depth_m"])
        if not math.isfinite(near) or not math.isfinite(far) or not 0 <= near < far:
            raise ValueError("Habitat depth range is invalid")
        if encoding == "normalized_0_1":
            # Habitat clips raw depth at near/far before normalizing; both endpoints
            # are censored and therefore cannot be treated as measured geometry.
            valid = np.isfinite(depth) & (depth > 0.0) & (depth < 1.0)
            converted = np.full(depth.shape, np.nan, dtype=np.float64)
            converted[valid] = near + depth[valid] * (far - near)
            depth = converted
        elif encoding == "meters":
            # Compare against endpoints rounded to the sensor's original dtype;
            # float32(0.1) otherwise looks slightly greater than float64(0.1).
            near_sensor = float(np.asarray(near, dtype=source_depth.dtype))
            far_sensor = float(np.asarray(far, dtype=source_depth.dtype))
            valid = np.isfinite(depth) & (depth > near_sensor) & (depth < far_sensor)
            depth = np.where(valid, depth, np.nan)
        else:
            raise ValueError("Habitat depth encoding must explicitly state metres or normalization")
        return cls(rgb, depth, intrinsics, agent_pose, extrinsics)


@dataclass(frozen=True)
class InstanceMask:
    class_name: str
    confidence: float
    mask: np.ndarray
    evidence_id: str
    source: str = "instance_segmentation"
    linked_l1_evidence_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("instance class_name must be nonempty")
        if not math.isfinite(self.confidence) or not 0 < self.confidence <= 1:
            raise ValueError("instance confidence must be in (0, 1]")
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("instance must have a canonical Evidence reference")
        parse_evidence_key(self.evidence_id)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("instance source must be nonempty")
        if self.linked_l1_evidence_id is not None:
            if parse_evidence_key(self.linked_l1_evidence_id)[:3] != parse_evidence_key(
                self.evidence_id,
            )[:3]:
                raise ValueError("linked L1 Evidence crosses Scene/Episode identity")
        values = np.asarray(self.mask)
        if values.ndim != 2 or values.dtype != np.bool_:
            raise ValueError("instance mask must be a two-dimensional boolean array")
        object.__setattr__(self, "mask", values)


def backproject_world(mask: np.ndarray, frame: RGBDFrame, config: GeometryConfig) -> np.ndarray:
    """Project valid mask pixels through K and world_T_agent @ agent_T_camera."""
    pixels = np.asarray(mask)
    if pixels.dtype != np.bool_ or pixels.shape != frame.depth_m.shape:
        raise ValueError("instance mask must be boolean and aligned to RGB-D")
    valid = (
        pixels & np.isfinite(frame.depth_m)
        & (frame.depth_m >= config.min_depth_m)
        & (frame.depth_m <= config.max_depth_m)
    )
    rows, cols = np.nonzero(valid)
    if len(rows) < config.min_points:
        return np.empty((0, 3), dtype=np.float64)
    if len(rows) > config.max_points_per_instance:
        indices = np.linspace(0, len(rows) - 1, config.max_points_per_instance, dtype=np.int64)
        rows, cols = rows[indices], cols[indices]
    z = frame.depth_m[rows, cols]
    K = frame.intrinsics
    camera = np.column_stack(((cols - K.cx) * z / K.fx, -(rows - K.cy) * z / K.fy, -z))
    return frame.camera_pose.transform_points(camera)


def _bbox(points: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return tuple(points.min(axis=0).tolist()), tuple(points.max(axis=0).tolist())


def _bbox_iou(first: np.ndarray, second: np.ndarray, padding: float) -> float:
    # Observed surfaces can be thin planes. A small symmetric padding gives
    # finite volumes without inventing an object's unseen physical extent.
    first_min, first_max = first.min(axis=0) - padding, first.max(axis=0) + padding
    second_min, second_max = second.min(axis=0) - padding, second.max(axis=0) + padding
    intersection = np.maximum(0, np.minimum(first_max, second_max) - np.maximum(first_min, second_min))
    intersection_volume = float(np.prod(intersection))
    first_volume = float(np.prod(first_max - first_min))
    second_volume = float(np.prod(second_max - second_min))
    return intersection_volume / (first_volume + second_volume - intersection_volume)


def _n_ratio(detection: np.ndarray, mapped: np.ndarray, voxel_size_m: float) -> float:
    """Fraction of new points whose nearest mapped point is within one voxel.

    This follows ConceptGraphs ``compute_overlap_matrix_2set``: the ratio is
    directional (new detection to old object) and the distance comparison is
    strictly less than the squared downsampling voxel size.
    """
    squared = np.sum((detection[:, None, :] - mapped[None, :, :]) ** 2, axis=2)
    return float(np.mean(squared.min(axis=1) < voxel_size_m * voxel_size_m))


def _downsample(points: np.ndarray, config: GeometryConfig) -> np.ndarray:
    voxels = np.floor(points / config.voxel_size_m).astype(np.int64)
    _, first = np.unique(voxels, axis=0, return_index=True)
    chosen = points[np.sort(first)]
    if len(chosen) > config.max_points_per_object:
        indices = np.linspace(0, len(chosen) - 1, config.max_points_per_object, dtype=np.int64)
        chosen = chosen[indices]
    return chosen


@dataclass
class ObjectNode:
    object_id: str
    class_name: str
    center_world: tuple[float, float, float]
    bbox_3d: tuple[tuple[float, float, float], tuple[float, float, float]]
    observation_count: int
    evidence_ids: list[str]
    geometry_status: str = "observed"
    points_world: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    # Fused, L2-normalized visual feature and its observation count. Both are
    # populated after ``ObjectMap.observe`` has finished its geometry pass; the
    # geometry, identity and evidence fields above are never derived from them.
    clip_feature: np.ndarray | None = None
    clip_feature_observations: int = 0
    clip_feature_evidence_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ViewpointNode:
    viewpoint_id: str
    scene_id: str
    episode_id: str
    rollout_idx: int
    step_id: int
    agent_pose: Pose3D
    camera_pose: Pose3D


@dataclass(frozen=True)
class ObservedFromEdge:
    object_id: str
    viewpoint_id: str
    evidence_id: str
    relation_type: str = "observed_from"


@dataclass(frozen=True)
class TraversedToEdge:
    source_viewpoint_id: str
    target_viewpoint_id: str
    transition_step: int
    relation_type: str = "traversed_to"


@dataclass(frozen=True)
class SpatialEvidence:
    evidence_id: str
    scene_id: str
    episode_id: str
    rollout_idx: int
    step_id: int
    class_name: str
    confidence: float
    source: str
    mask_pixel_count: int
    depth_point_count: int
    geometry_status: str
    linked_l1_evidence_id: str | None = None


class ObjectMap:
    """Episode-isolated graph of observed 3D instances and real viewpoints."""

    def __init__(self, config: GeometryConfig | None = None) -> None:
        self.config = config or GeometryConfig()
        self.objects: dict[str, ObjectNode] = {}
        self.viewpoints: dict[str, ViewpointNode] = {}
        self.edges: list[ObservedFromEdge | TraversedToEdge] = []
        self.evidence: dict[str, SpatialEvidence] = {}
        self._identity: tuple[str, str, int] | None = None
        self._last_viewpoint: ViewpointNode | None = None
        self._next_object_id = 0

    def reset_episode(self, scene_id: str, episode_id: str, rollout_idx: int = 0) -> None:
        if not scene_id or not episode_id or type(rollout_idx) is not int or rollout_idx < 0:
            raise ValueError("ObjectMap requires Scene/Episode/rollout identity")
        self.objects.clear()
        self.viewpoints.clear()
        self.edges.clear()
        self.evidence.clear()
        self._identity = (scene_id, episode_id, rollout_idx)
        self._last_viewpoint = None
        self._next_object_id = 0

    def _candidate_scores(
        self, name: str, points: np.ndarray, history: Sequence[ObjectNode],
    ) -> list[tuple[float, ObjectNode]]:
        """Score against the complete pre-frame map, including later occupied nodes."""
        scores: list[tuple[float, ObjectNode]] = []
        center = points.mean(axis=0)
        for node in history:
            if node.class_name.casefold() != name.casefold():
                continue
            center_distance = float(np.linalg.norm(center - np.asarray(node.center_world)))
            if center_distance > self.config.max_center_distance_m:
                continue
            iou = _bbox_iou(points, node.points_world, self.config.voxel_size_m)
            if iou < self.config.min_bbox_iou:
                continue
            scores.append((_n_ratio(points, node.points_world, self.config.voxel_size_m), node))
        scores.sort(key=lambda pair: (-pair[0], pair[1].object_id))
        return scores

    def observe(
        self, frame: RGBDFrame, instances: Sequence[InstanceMask], *,
        step_id: int, transition_record: bool = False,
    ) -> list[ObjectNode]:
        if self._identity is None:
            raise RuntimeError("ObjectMap has no active Episode")
        if type(step_id) is not int or step_id < 0:
            raise ValueError("step_id must be nonnegative")
        if type(transition_record) is not bool:
            raise ValueError("transition_record must be an explicit boolean")
        scene_id, episode_id, rollout_idx = self._identity
        viewpoint_id = json.dumps([scene_id, episode_id, rollout_idx, step_id], separators=(",", ":"))
        if viewpoint_id in self.viewpoints:
            raise ValueError("one RGB-D frame may be observed only once per step")
        if self._last_viewpoint is not None and step_id <= self._last_viewpoint.step_id:
            raise ValueError("ObjectMap steps must increase within an Episode")

        # Validate all instance/evidence inputs before mutating the graph.
        prepared: list[tuple[InstanceMask, np.ndarray]] = []
        used_evidence: set[str] = set()
        for instance in instances:
            if not isinstance(instance, InstanceMask):
                raise ValueError("instances must be InstanceMask records")
            ref_scene, ref_episode, ref_rollout, _ = parse_evidence_key(instance.evidence_id)
            if (ref_scene, ref_episode, ref_rollout) != self._identity:
                raise ValueError("instance Evidence crosses Scene/Episode identity")
            if instance.evidence_id in used_evidence:
                raise ValueError("one Evidence cannot be used by two instance masks")
            if instance.evidence_id in self.evidence:
                raise ValueError("segmentation Evidence was already recorded")
            used_evidence.add(instance.evidence_id)
            points = backproject_world(instance.mask, frame, self.config)
            prepared.append((instance, points))

        viewpoint = ViewpointNode(
            viewpoint_id, scene_id, episode_id, rollout_idx, step_id,
            frame.agent_pose, frame.camera_pose,
        )
        previous = self._last_viewpoint
        self.viewpoints[viewpoint_id] = viewpoint
        if transition_record and previous is not None and previous.step_id + 1 == step_id:
            distance = math.dist(previous.agent_pose.position, viewpoint.agent_pose.position)
            if distance >= self.config.min_travel_distance_m:
                self.edges.append(TraversedToEdge(
                    previous.viewpoint_id, viewpoint_id, step_id,
                ))
        self._last_viewpoint = viewpoint

        for instance, points in prepared:
            self.evidence[instance.evidence_id] = SpatialEvidence(
                instance.evidence_id, scene_id, episode_id, rollout_idx, step_id,
                instance.class_name, instance.confidence, instance.source,
                int(instance.mask.sum()), len(points),
                "observed" if len(points) >= self.config.min_points else "rejected_depth",
                instance.linked_l1_evidence_id,
            )

        # Score every detection against the full pre-frame history before any
        # assignment or point-cloud mutation. An occupied best node must remain
        # visible to another detection's ambiguity check.
        history = tuple(self.objects.values())
        proposals: list[tuple[float, int, ObjectNode]] = []
        for index, (instance, points) in enumerate(prepared):
            if len(points) < self.config.min_points:
                continue
            scores = self._candidate_scores(instance.class_name, points, history)
            if not scores or scores[0][0] < self.config.min_overlap_ratio:
                continue
            if (len(scores) > 1
                    and scores[0][0] - scores[1][0] < self.config.ambiguity_margin_ratio):
                continue
            proposals.append((scores[0][0], index, scores[0][1]))

        # Geometry-only greedy assignment. A detection whose preferred old
        # object was claimed by a stronger detection becomes a new node; it
        # cannot fall through to a less supported runner-up.
        proposals.sort(key=lambda item: (-item[0], item[1], item[2].object_id))
        assignments: dict[int, ObjectNode] = {}
        claimed_ids: set[str] = set()
        for _, index, node in proposals:
            if node.object_id not in claimed_ids:
                assignments[index] = node
                claimed_ids.add(node.object_id)

        observed: list[ObjectNode] = []
        for index, (instance, points) in enumerate(prepared):
            if len(points) < self.config.min_points:
                continue
            node = assignments.get(index)
            if node is None:
                self._next_object_id += 1
                object_id = json.dumps(
                    [scene_id, episode_id, rollout_idx, "object", self._next_object_id],
                    ensure_ascii=False, separators=(",", ":"),
                )
                cloud = _downsample(points, self.config)
                node = ObjectNode(
                    object_id, instance.class_name,
                    tuple(cloud.mean(axis=0).tolist()), _bbox(cloud),
                    1, [instance.evidence_id], "observed", cloud,
                )
                self.objects[object_id] = node
            else:
                cloud = _downsample(np.concatenate((node.points_world, points)), self.config)
                node.points_world = cloud
                node.center_world = tuple(cloud.mean(axis=0).tolist())
                node.bbox_3d = _bbox(cloud)
                node.observation_count += 1
                node.evidence_ids.append(instance.evidence_id)
            self.edges.append(ObservedFromEdge(node.object_id, viewpoint_id, instance.evidence_id))
            observed.append(node)
        return observed

    def attach_clip_features(self, features: Mapping[str, Any]) -> int:
        """Fuse per-observation CLIP features into existing object nodes.

        ``features`` maps an L3 Evidence ID from this frame to one normalized
        visual feature. Only nodes that already contain that Evidence are
        updated, so object identity, point clouds, boxes, centers, observation
        counts and graph edges are untouched. Fusing is idempotent per Evidence
        ID and follows ConceptGraphs ``merge_obj2_into_obj1``: detection-count
        weighted average followed by L2 normalization.

        Returns the number of newly fused observations.
        """
        if not features:
            return 0
        attached = 0
        for node in self.objects.values():
            for evidence_id in node.evidence_ids:
                if evidence_id not in features or evidence_id in node.clip_feature_evidence_ids:
                    continue
                node.clip_feature, node.clip_feature_observations = fuse_clip_feature(
                    node.clip_feature,
                    node.clip_feature_observations,
                    features[evidence_id],
                )
                node.clip_feature_evidence_ids.append(evidence_id)
                attached += 1
        return attached
