"""Turn verified 3D object observations into existing Local Graph prompt items."""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape

import numpy as np

from lightnav.memory.conceptgraphs_lite import InstanceMask
from lightnav.memory.local_graph_fusion import _plain_label
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence
from lightnav.memory.v3_retrieval import _terms
from lightnav.memory.v3_schemas import canonical_evidence_key
from lightnav.memory.v3_store import V3MemoryStore

_HEADER = "[Navigation Memory]"


def make_spatial_evidence(
    instances: list[object], events: list[object], store: V3MemoryStore, *,
    scene_id: str, episode_id: str, rollout_idx: int, step: int,
    source: str, min_iou: float, ambiguity_margin: float,
) -> tuple[list[InstanceMask], dict[str, int]]:
    """Mint L3 Evidence for every valid mask, optionally linking one L1 record.

    L1 linkage never gates the 3D map. It only supplies extra provenance when a
    same-class mask and detector box agree without ambiguity.
    """
    masks: list[InstanceMask] = []
    counts = {"unmatched": 0, "ambiguous": 0, "linked": 0, "minted": 0}
    used_evidence: set[str] = set()
    for index, instance in enumerate(instances):
        mask = np.asarray(instance.mask)
        if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any():
            continue
        height, width = mask.shape
        sx1, sy1, sx2, sy2 = (float(value) for value in instance.bbox_xyxy)
        if not all(math.isfinite(value) for value in (sx1, sy1, sx2, sy2)):
            continue
        seg_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
        matches: list[tuple[float, str]] = []
        for event in events:
            if event.kind not in {"object_observation", "landmark"}:
                continue
            if event.description.removeprefix("Observed ").strip().casefold() != (
                instance.class_name.casefold()
            ):
                continue
            box = event.metadata.get("bbox_xyxy")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in box)
            except (TypeError, ValueError):
                continue
            if (not all(math.isfinite(value) for value in (x1, y1, x2, y2))
                    or x1 < 0 or y1 < 0 or x2 > width or y2 > height
                    or x2 <= x1 or y2 <= y1):
                continue
            intersection = max(0.0, min(sx2, x2) - max(sx1, x1)) * max(
                0.0, min(sy2, y2) - max(sy1, y1)
            )
            union = seg_area + (x2 - x1) * (y2 - y1) - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou < min_iou:
                continue
            xlo, xhi = max(0, int(math.floor(x1))), min(width, int(math.ceil(x2)))
            ylo, yhi = max(0, int(math.floor(y1))), min(height, int(math.ceil(y2)))
            covered = float(mask[ylo:yhi, xlo:xhi].sum()) / float(mask.sum())
            if covered < 0.8:
                continue
            evidence_id = canonical_evidence_key(
                scene_id, episode_id, rollout_idx, event.event_id,
            )
            record = store.get_evidence(evidence_id)
            if (record is None or record.event_type not in {"object_observation", "landmark"}
                    or record.observation_status != "observed"
                    or not record.provenance_validity
                    or (record.semantic_label or "").casefold() != instance.class_name.casefold()
                    or record.step != event.step_id):
                continue
            matches.append((iou, evidence_id))
        matches.sort(reverse=True)
        best_id = None
        if not matches:
            counts["unmatched"] += 1
        elif (len(matches) > 1 and matches[0][0] - matches[1][0] < ambiguity_margin):
            counts["ambiguous"] += 1
        elif matches[0][1] in used_evidence:
            counts["ambiguous"] += 1
        else:
            best_id = matches[0][1]
            used_evidence.add(best_id)
            counts["linked"] += 1
        l3_id = canonical_evidence_key(
            scene_id, episode_id, rollout_idx, f"l3_seg_{step}_{index}",
        )
        masks.append(InstanceMask(
            class_name=instance.class_name, confidence=instance.confidence,
            mask=mask, evidence_id=l3_id, source=source,
            linked_l1_evidence_id=best_id,
        ))
        counts["minted"] += 1
    return masks, counts


def _relative_position(agent_pose: object, target: object) -> tuple[float, float] | None:
    """Return straight-line range and left-positive bearing in the agent frame."""
    try:
        origin = np.asarray(agent_pose.position, dtype=np.float64)
        q = np.asarray(agent_pose.rotation_wxyz, dtype=np.float64)
        center = np.asarray(target, dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return None
    if origin.shape != (3,) or center.shape != (3,) or q.shape != (4,):
        return None
    if not np.isfinite(origin).all() or not np.isfinite(center).all() or not np.isfinite(q).all():
        return None
    norm = np.linalg.norm(q)
    if not math.isfinite(norm) or norm < 1e-8:
        return None
    w, x, y, z = q / norm
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    local = rotation.T @ (center - origin)
    distance = float(np.linalg.norm(center - origin))
    if distance < 1e-8:
        return None
    # Habitat camera/agent convention: +X is right and -Z is forward.
    bearing = math.degrees(math.atan2(-float(local[0]), -float(local[2])))
    return distance, bearing


@dataclass(frozen=True)
class _SpatialCandidate:
    object_id: str
    label: str
    center_world: tuple[float, float, float]
    viewpoint_position_world: tuple[float, float, float]
    distance_m: float
    bearing_deg: float
    observation_count: int
    evidence_id: str
    evidence_step: int
    detector_name: str
    confidence: float
    provenance: dict[str, object]

    def line(self) -> str:
        center = ",".join(f"{value:.1f}" for value in self.center_world)
        return (
            f"Spatial history: {self.label} observed at world ({center}) m; "
            f"now {self.distance_m:.1f} m straight-line, bearing "
            f"{self.bearing_deg:+.0f} deg (left +); "
            f"{self.observation_count} observations; source {self.detector_name}; "
            "passage unknown."
        )


class SpatialGraphAdapter:
    """Select only instruction-related, evidence-linked historical object facts."""

    def build(
        self, store: V3MemoryStore, object_map: object, frame: object, *,
        scene_id: str, episode_id: str, rollout_idx: int, step: int,
        instruction: str, scope: str, max_landmarks: int, prompt_max_chars: int,
        min_confidence: float,
    ) -> MemoryContext | None:
        if store.active_episode != (scene_id, episode_id, rollout_idx):
            raise ValueError("spatial selection requires the active Scene/Episode")
        if store.scope != scope:
            raise ValueError("spatial selection and evidence store scopes differ")
        if scope not in {"episode_local", "scene_persistent"}:
            raise ValueError("invalid spatial scope")
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be nonnegative")
        if not isinstance(max_landmarks, int) or max_landmarks < 1 or prompt_max_chars < 1:
            raise ValueError("spatial prompt limits must be positive")

        objects = object_map.objects
        viewpoints = object_map.viewpoints
        observed_edges = [
            edge for edge in object_map.edges
            if edge.relation_type == "observed_from"
        ]
        candidates: list[_SpatialCandidate] = []
        for obj in objects.values():
            if getattr(obj, "geometry_status", None) != "observed":
                continue
            # The center and box contain every merged observation. A low-score
            # mask could move them even if an older high-score Evidence is chosen
            # for the Prompt, so the whole fused node must meet the floor.
            if any(
                (record := object_map.evidence.get(evidence_id)) is None
                or record.geometry_status != "observed"
                or record.confidence < min_confidence
                for evidence_id in obj.evidence_ids
            ):
                continue
            label = _plain_label(obj.class_name)
            if not label or not (_terms(label) & _terms(instruction)):
                continue
            center = tuple(float(value) for value in obj.center_world)
            if len(center) != 3 or not all(math.isfinite(value) for value in center):
                continue
            relative = _relative_position(frame.agent_pose, center)
            if relative is None:
                continue
            valid_history = []
            for edge in observed_edges:
                if edge.object_id != obj.object_id or edge.evidence_id not in obj.evidence_ids:
                    continue
                viewpoint = viewpoints.get(edge.viewpoint_id)
                record = object_map.evidence.get(edge.evidence_id)
                if viewpoint is None or record is None:
                    continue
                if (record.scene_id, record.episode_id, record.rollout_idx, record.step_id) != (
                    viewpoint.scene_id, viewpoint.episode_id, viewpoint.rollout_idx,
                    viewpoint.step_id,
                ):
                    continue
                if record.scene_id != scene_id or (
                    (record.episode_id, record.rollout_idx) == (episode_id, rollout_idx)
                    and record.step_id >= step
                ):
                    continue
                if scope == "episode_local" and (
                    record.episode_id, record.rollout_idx
                ) != (episode_id, rollout_idx):
                    continue
                if (record.geometry_status != "observed"
                        or record.class_name.casefold() != obj.class_name.casefold()
                        or record.confidence < min_confidence):
                    continue
                l1_id = record.linked_l1_evidence_id
                if l1_id is not None:
                    l1 = store.get_evidence(l1_id)
                    if (l1 is None or l1.scene_id != scene_id
                            or l1.episode_id != record.episode_id
                            or l1.rollout_idx != record.rollout_idx
                            or l1.step != record.step_id
                            or l1.observation_status != "observed"):
                        l1_id = None
                position = tuple(float(value) for value in viewpoint.camera_pose.position)
                if len(position) != 3 or not all(math.isfinite(value) for value in position):
                    continue
                valid_history.append((record, position, l1_id))
            if not valid_history:
                continue
            record, position, l1_id = max(
                valid_history,
                key=lambda pair: (
                    pair[0].episode_id == episode_id,
                    pair[0].rollout_idx == rollout_idx, pair[0].step_id,
                ),
            )
            candidates.append(_SpatialCandidate(
                object_id=obj.object_id, label=label, center_world=center,
                viewpoint_position_world=position,
                distance_m=relative[0], bearing_deg=relative[1],
                observation_count=obj.observation_count,
                evidence_id=record.evidence_id, evidence_step=record.step_id,
                detector_name="YOLO11n-seg",
                confidence=float(record.confidence),
                provenance={
                    "scene_id": record.scene_id, "episode_id": record.episode_id,
                    "rollout_idx": record.rollout_idx, "step": record.step_id,
                    "evidence_id": record.evidence_id,
                    "source": record.source,
                    "linked_l1_evidence_id": l1_id,
                    "geometry_status": record.geometry_status,
                },
            ))
        candidates.sort(key=lambda item: (item.distance_m, -item.observation_count, item.object_id))
        lines: list[str] = []
        items: list[LayerMemoryItem] = []
        used = len(_HEADER)
        for candidate in candidates[:max_landmarks]:
            line = candidate.line()
            safe_line = escape(line, quote=False)
            if used + len(safe_line) + 1 > prompt_max_chars:
                continue
            used += len(safe_line) + 1
            lines.append(line)
            items.append(LayerMemoryItem(
                layer="long", kind="local_graph_spatial_object",
                description=line, step_id=candidate.evidence_step,
                confidence=candidate.confidence, source="spatial_graph_adapter",
                evidence_refs=[ObservationEvidence(
                    event_id=candidate.evidence_id, step_id=candidate.evidence_step,
                    label=candidate.label, confidence=candidate.confidence,
                    source=candidate.detector_name,
                )],
                label=candidate.label,
                metadata={
                    "evidence_ids": [candidate.evidence_id],
                    "serialized_text": safe_line,
                    "provenance": [candidate.provenance],
                    "role": "spatial_history", "object_id": candidate.object_id,
                    "center_world": candidate.center_world,
                    "history_viewpoint_world": candidate.viewpoint_position_world,
                    "current_relative_distance_m": candidate.distance_m,
                    "current_relative_bearing_deg": candidate.bearing_deg,
                    "observation_count": candidate.observation_count,
                    "traversability_status": "unknown",
                },
            ))
        if not items:
            return None
        return MemoryContext(text=_HEADER + "\n" + "\n".join(lines), layer_items=items)
