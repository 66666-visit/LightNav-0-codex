"""Read-only per-Evidence XYZRGB recorder for the ConceptGraphs-Lite map.

`ObjectMap` deliberately stores only world-space XYZ points.  The official
ViL3DRel student, however, consumes an ``(N, 6)`` XYZRGB proposal for every
object.  This sidecar records the exact pixels that produced each Evidence
mask, *before* any voxel downsampling or object fusion, so a later grounding
adapter can recover true per-point colours.

The module never mutates ``ObjectMap``, ``ObjectNode``, ``observe()``, the
geometry association, the Prompt, or any L1 state.  It is disabled by default
and only records when explicitly constructed by ``MemoryManager``.

Pixel selection is byte-for-byte the same as
``conceptgraphs_lite.backproject_world``:

1. valid = mask & finite(depth) & min_depth <= depth <= max_depth;
2. if the valid pixel count exceeds ``max_points_per_instance``, keep the
   evenly spaced indices produced by ``np.linspace``;
3. back-project through K and ``world_T_camera``.

The unit tests assert that the sidecar XYZ array is exactly equal to
``backproject_world`` for the same frame/mask, and that ``rgb[rows, cols]`` is
the matching real pixel colour.  No colour is invented, interpolated, or taken
from the downsampled ``ObjectNode.points_world``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from lightnav.memory.conceptgraphs_lite import (
    GeometryConfig,
    InstanceMask,
    ObjectNode,
    RGBDFrame,
)
from lightnav.memory.v3_schemas import parse_evidence_key


@dataclass(frozen=True)
class EvidenceRGBRecord:
    """Real XYZ and RGB for the pixels of one L3 Evidence mask."""

    evidence_id: str
    step_id: int
    scene_id: str
    episode_id: str
    rollout_idx: int
    points_world: np.ndarray
    colors_rgb: np.ndarray
    object_id: str | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points_world)
        colors = np.asarray(self.colors_rgb)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("evidence XYZ must be an (N, 3) array")
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError("evidence RGB must be an (N, 3) array")
        if points.shape[0] != colors.shape[0]:
            raise ValueError("evidence XYZ and RGB must have one row per point")
        if points.dtype != np.float64 and not np.issubdtype(points.dtype, np.floating):
            raise ValueError("evidence XYZ must be floating point")
        if colors.dtype != np.uint8:
            raise ValueError("evidence RGB must be real uint8 pixels")
        if not np.isfinite(points).all():
            raise ValueError("evidence XYZ must be finite")
        object.__setattr__(self, "points_world", points)
        object.__setattr__(self, "colors_rgb", colors)

    @property
    def point_count(self) -> int:
        return int(self.points_world.shape[0])


@dataclass
class XYZRGBProposal:
    """One ObjectNode's real, unfused XYZRGB point set.

    Coordinates remain in the Habitat world frame (``+Y`` up).  The model
    adapter is responsible for the deterministic z-up conversion.
    """

    object_id: str
    class_name: str
    points_world_rgb: np.ndarray
    evidence_ids: tuple[str, ...]
    observation_count: int

    def __post_init__(self) -> None:
        points = np.asarray(self.points_world_rgb)
        if points.ndim != 2 or points.shape[1] != 6:
            raise ValueError("proposal must be an (N, 6) XYZRGB array")
        if not np.isfinite(points).all():
            raise ValueError("proposal must contain finite XYZRGB values")
        object.__setattr__(self, "points_world_rgb", points)

    @property
    def point_count(self) -> int:
        return int(self.points_world_rgb.shape[0])


class RGBSidecar:
    """Episode-isolated evidence_id -> real XYZRGB record store."""

    def __init__(self, config: GeometryConfig | None = None) -> None:
        if config is not None and not isinstance(config, GeometryConfig):
            raise TypeError("RGBSidecar config must be GeometryConfig")
        self.config = config or GeometryConfig()
        self._identity: tuple[str, str, int] | None = None
        self._records: dict[str, EvidenceRGBRecord] = {}
        self._object_evidence: dict[str, list[str]] = {}
        self.last_diagnostic: dict[str, Any] = {}
        self.stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {
            "frames": 0,
            "masks_seen": 0,
            "records_minted": 0,
            "empty_masks": 0,
            "duplicate_evidence": 0,
            "bound_records": 0,
            "unbound_records": 0,
        }

    @property
    def identity(self) -> tuple[str, str, int] | None:
        return self._identity

    def reset_episode(self, scene_id: str, episode_id: str, rollout_idx: int = 0) -> None:
        if not scene_id or not episode_id or type(rollout_idx) is not int or rollout_idx < 0:
            raise ValueError("RGBSidecar requires Scene/Episode/rollout identity")
        self._identity = (scene_id, episode_id, rollout_idx)
        self._records.clear()
        self._object_evidence.clear()
        self.last_diagnostic = {}
        self.stats = self._empty_stats()

    def close_episode(self) -> None:
        self._identity = None
        self._records.clear()
        self._object_evidence.clear()
        self.last_diagnostic = {}
        self.stats = self._empty_stats()

    def _require_identity(self) -> tuple[str, str, int]:
        if self._identity is None:
            raise RuntimeError("RGBSidecar has no active Episode")
        return self._identity

    def record_frame(
        self,
        frame: RGBDFrame,
        masks: Sequence[InstanceMask],
        *,
        step_id: int,
    ) -> dict[str, Any]:
        """Record the exact valid pixels of every mask; no map mutation.

        Returns a small diagnostic dict.  Evidence IDs that already exist are
        not overwritten: the same Evidence can only be observed once per step
        by ``ObjectMap.observe`` and is an error here too.
        """
        if not isinstance(frame, RGBDFrame):
            raise TypeError("RGBSidecar frame must be an RGBDFrame")
        if type(step_id) is not int or step_id < 0:
            raise ValueError("RGBSidecar step_id must be a nonnegative integer")
        scene_id, episode_id, rollout_idx = self._require_identity()
        self.stats["frames"] += 1
        diagnostic = {
            "step": step_id,
            "masks_seen": 0,
            "records_minted": 0,
            "empty_masks": 0,
            "duplicate_evidence": 0,
            "errors": [],
        }
        for mask in masks:
            if not isinstance(mask, InstanceMask):
                raise TypeError("RGBSidecar masks must be InstanceMask records")
            key = parse_evidence_key(mask.evidence_id)
            if (key[0], key[1], key[2]) != (scene_id, episode_id, rollout_idx):
                raise ValueError("RGBSidecar Evidence crosses Scene/Episode identity")
            self.stats["masks_seen"] += 1
            diagnostic["masks_seen"] += 1
            if mask.evidence_id in self._records:
                self.stats["duplicate_evidence"] += 1
                diagnostic["duplicate_evidence"] += 1
                continue
            points, colors, rows, cols = self._backproject_rgb(frame, mask)
            if points.shape[0] == 0:
                self.stats["empty_masks"] += 1
                diagnostic["empty_masks"] += 1
                continue
            record = EvidenceRGBRecord(
                evidence_id=mask.evidence_id,
                step_id=step_id,
                scene_id=scene_id,
                episode_id=episode_id,
                rollout_idx=rollout_idx,
                points_world=points,
                colors_rgb=colors,
            )
            self._records[mask.evidence_id] = record
            self.stats["records_minted"] += 1
            diagnostic["records_minted"] += 1
        diagnostic["stats"] = dict(self.stats)
        self.last_diagnostic = diagnostic
        return diagnostic

    def _backproject_rgb(
        self, frame: RGBDFrame, mask: InstanceMask,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Back-project and sample real RGB with the exact ``backproject_world`` pixels."""
        pixels = np.asarray(mask.mask)
        if pixels.dtype != np.bool_ or pixels.shape != frame.depth_m.shape:
            raise ValueError("sidecar mask must be boolean and aligned to RGB-D")
        valid = (
            pixels & np.isfinite(frame.depth_m)
            & (frame.depth_m >= self.config.min_depth_m)
            & (frame.depth_m <= self.config.max_depth_m)
        )
        rows, cols = np.nonzero(valid)
        if len(rows) < self.config.min_points:
            empty_points = np.empty((0, 3), dtype=np.float64)
            empty_colors = np.empty((0, 3), dtype=np.uint8)
            empty_idx = np.empty((0,), dtype=np.int64)
            return empty_points, empty_colors, empty_idx, empty_idx
        if len(rows) > self.config.max_points_per_instance:
            indices = np.linspace(
                0, len(rows) - 1, self.config.max_points_per_instance, dtype=np.int64,
            )
            rows, cols = rows[indices], cols[indices]
        z = frame.depth_m[rows, cols]
        intrinsics = frame.intrinsics
        camera = np.column_stack((
            (cols - intrinsics.cx) * z / intrinsics.fx,
            -(rows - intrinsics.cy) * z / intrinsics.fy,
            -z,
        ))
        points = frame.camera_pose.transform_points(camera)
        colors = np.array(frame.rgb[rows, cols], dtype=np.uint8, copy=True)
        return np.asarray(points, dtype=np.float64), colors, rows, cols

    def bind_objects(self, objects: Iterable[ObjectNode]) -> dict[str, Any]:
        """Associate recorded Evidence IDs with the ObjectNodes that own them."""
        self._require_identity()
        bound = 0
        for node in objects:
            if not isinstance(node, ObjectNode):
                raise TypeError("sidecar objects must be ObjectNode records")
            known = self._object_evidence.setdefault(node.object_id, [])
            for evidence_id in node.evidence_ids:
                record = self._records.get(evidence_id)
                if record is None:
                    continue
                if record.object_id is not None and record.object_id != node.object_id:
                    raise ValueError(
                        "one Evidence RGB record cannot belong to two ObjectNodes"
                    )
                if record.object_id is None:
                    bound += 1
                    self.stats["bound_records"] += 1
                record = EvidenceRGBRecord(
                    evidence_id=record.evidence_id,
                    step_id=record.step_id,
                    scene_id=record.scene_id,
                    episode_id=record.episode_id,
                    rollout_idx=record.rollout_idx,
                    points_world=record.points_world,
                    colors_rgb=record.colors_rgb,
                    object_id=node.object_id,
                )
                self._records[evidence_id] = record
                if evidence_id not in known:
                    known.append(evidence_id)
        self._recount_unbound()
        return {
            "objects": len(self._object_evidence),
            "bound_now": bound,
            "bound_records": self.stats["bound_records"],
            "unbound_records": self.stats["unbound_records"],
        }

    def _recount_unbound(self) -> None:
        unbound = sum(1 for record in self._records.values() if record.object_id is None)
        self.stats["unbound_records"] = unbound

    def has_evidence(self, evidence_id: str) -> bool:
        return evidence_id in self._records

    def record_for(self, evidence_id: str) -> EvidenceRGBRecord | None:
        return self._records.get(evidence_id)

    def object_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._object_evidence))

    def missing_evidence(self, node: ObjectNode) -> tuple[str, ...]:
        return tuple(
            evidence_id for evidence_id in node.evidence_ids
            if evidence_id not in self._records
        )

    def proposal_for(self, node: ObjectNode) -> XYZRGBProposal:
        """Concatenate the node's *raw* Evidence XYZRGB; fail closed if incomplete.

        The concatenated points are the real per-mask back-projections.  They
        are never joined with ``node.points_world`` and never reorder XYZ
        relative to RGB.
        """
        if not isinstance(node, ObjectNode):
            raise TypeError("proposal_for requires an ObjectNode")
        missing = self.missing_evidence(node)
        if missing:
            raise ValueError(
                f"ObjectNode {node.object_id!r} is missing RGB for "
                f"{len(missing)} Evidence records"
            )
        point_blocks: list[np.ndarray] = []
        color_blocks: list[np.ndarray] = []
        for evidence_id in node.evidence_ids:
            record = self._records[evidence_id]
            if record.object_id not in (None, node.object_id):
                raise ValueError("Evidence RGB record belongs to a different ObjectNode")
            if record.point_count < 1:
                raise ValueError(f"Evidence {evidence_id!r} has no valid RGB points")
            point_blocks.append(record.points_world)
            color_blocks.append(record.colors_rgb.astype(np.float32, copy=False))
        points = np.concatenate(point_blocks, axis=0)
        colors = np.concatenate(color_blocks, axis=0)
        if points.shape[0] != colors.shape[0]:
            raise ValueError("sidecar XYZ/RGB row counts diverged")
        xyzrgb = np.concatenate((points, colors), axis=1).astype(np.float32, copy=False)
        return XYZRGBProposal(
            object_id=node.object_id,
            class_name=node.class_name,
            points_world_rgb=xyzrgb,
            evidence_ids=tuple(node.evidence_ids),
            observation_count=int(node.observation_count),
        )

    def proposals(self, objects: Mapping[str, ObjectNode]) -> tuple[
        list[XYZRGBProposal], list[dict[str, Any]],
    ]:
        """Build proposals for all complete nodes; report every skipped node."""
        proposals: list[XYZRGBProposal] = []
        skipped: list[dict[str, Any]] = []
        for object_id in sorted(objects):
            node = objects[object_id]
            missing = self.missing_evidence(node)
            if missing:
                skipped.append({
                    "object_id": object_id,
                    "class_name": node.class_name,
                    "reason": "missing_rgb_evidence",
                    "missing_evidence": list(missing),
                })
                continue
            if not node.evidence_ids:
                skipped.append({
                    "object_id": object_id,
                    "class_name": node.class_name,
                    "reason": "no_evidence",
                })
                continue
            try:
                proposal = self.proposal_for(node)
            except ValueError as exc:
                skipped.append({
                    "object_id": object_id,
                    "class_name": node.class_name,
                    "reason": "invalid_proposal",
                    "error": str(exc),
                })
                continue
            proposals.append(proposal)
        skipped.sort(key=lambda item: (item["reason"], item["object_id"]))
        return proposals, skipped

    def summary(self) -> dict[str, Any]:
        return {
            **self.stats,
            "records": len(self._records),
            "objects": len(self._object_evidence),
            "identity": list(self._identity) if self._identity is not None else None,
        }
