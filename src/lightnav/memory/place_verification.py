"""Conservative two-stage visual place hypotheses for RGM D."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.schemas import PlaceDecision, VisualPlaceFeatures
from lightnav.memory.topological_memory import TopologicalMemory


@dataclass(frozen=True)
class LocalCheck:
    match_count: int
    inlier_ratio: float | None
    reason: str


def _points(indices: np.ndarray, features: VisualPlaceFeatures) -> np.ndarray:
    height, width = features.patch_grid
    original = np.asarray(features.patch_indices, dtype=np.int32)[indices]
    return np.column_stack(((original % width + 0.5) / width,
                            (original // width + 0.5) / height)).astype(np.float32)


def local_check(query: VisualPlaceFeatures, reference: VisualPlaceFeatures,
                config: RGMConfig) -> LocalCheck:
    """Mutual patch matches plus RANSAC affine consistency; no physical pose claim."""
    if len(query.patch_descriptors) == 0 or len(reference.patch_descriptors) == 0:
        return LocalCheck(0, None, "missing_patch_features")
    q = np.asarray(query.patch_descriptors, dtype=np.float32)
    r = np.asarray(reference.patch_descriptors, dtype=np.float32)
    similarities = q @ r.T
    forward = similarities.argmax(axis=1)
    backward = similarities.argmax(axis=0)
    pairs = np.asarray([
        (i, int(j)) for i, j in enumerate(forward)
        if backward[j] == i and similarities[i, j] >= config.place_local_cosine_min
    ], dtype=np.int32)
    count = len(pairs)
    if count < config.place_local_min_matches:
        return LocalCheck(count, None, "insufficient_mutual_patch_matches")
    try:
        import cv2
    except ImportError:
        return LocalCheck(count, None, "opencv_unavailable")
    source = _points(pairs[:, 1], reference)
    target = _points(pairs[:, 0], query)
    try:
        _, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC,
            ransacReprojThreshold=config.place_ransac_reprojection_threshold,
            maxIters=2000, confidence=0.99,
        )
    except cv2.error:
        return LocalCheck(count, None, "geometry_estimation_failed")
    if inliers is None:
        return LocalCheck(count, None, "geometry_estimation_failed")
    ratio = float(np.mean(inliers))
    reason = "local_consistent" if ratio >= config.place_local_inlier_min else "local_geometry_inconsistent"
    return LocalCheck(count, ratio, reason)


class TwoStagePlaceVerifier:
    def __init__(self, config: RGMConfig) -> None:
        self.config = config
        self._keyframes: dict[str, list[VisualPlaceFeatures]] = {}
        self._pending: tuple[int, VisualPlaceFeatures] | None = None
        self.last_topk: list[dict] = []

    def decide(self, step: int, features: VisualPlaceFeatures,
               topology: TopologicalMemory) -> PlaceDecision:
        global_vector = np.asarray(features.global_descriptor, dtype=np.float32)
        nodes = {node.place_id: node for node in topology.nodes}
        candidates = sorted(
            ((float(global_vector @ np.asarray(node.visual_descriptor, dtype=np.float32)), node.place_id)
             for node in nodes.values()), key=lambda item: (-item[0], item[1]),
        )[:self.config.place_top_k]
        self.last_topk = [{
            "place_id": place_id, "similarity": similarity,
            "descriptor_source": nodes[place_id].descriptor_source,
            "descriptor_event_ids": list(nodes[place_id].descriptor_event_ids),
            "keyframe_count": len(self._keyframes.get(place_id, [])),
        } for similarity, place_id in candidates]
        top1 = candidates[0][0] if candidates else None
        top2 = candidates[1][0] if len(candidates) > 1 else None
        def result(status: str, place_id: str | None, check: LocalCheck | None,
                   reason: str) -> PlaceDecision:
            return PlaceDecision(
                status, place_id, top1, top2,
                check.match_count if check else 0,
                check.inlier_ratio if check else None, reason,
            )
        quality = features.preprocessing_metadata.get("image_quality")
        if (not isinstance(quality, (int, float)) or not isfinite(quality)
                or quality < self.config.place_min_image_quality):
            return result("UNKNOWN", None, None, "low_or_unavailable_image_quality")
        if not candidates:
            return result("NEW", None, None, "initial_visual_reference")
        if top1 is not None and top1 >= self.config.place_global_candidate_min:
            supported: list[tuple[float, str, LocalCheck]] = []
            unresolved: list[tuple[float, str]] = []
            for index, (similarity, place_id) in enumerate(candidates):
                if similarity < self.config.place_global_candidate_min:
                    self.last_topk[index]["local_reason"] = "below_global_candidate_threshold"
                    continue
                checks = [local_check(features, keyframe, self.config)
                          for keyframe in self._keyframes.get(place_id, [])]
                if not checks:
                    self.last_topk[index]["local_reason"] = "missing_place_keyframes"
                    unresolved.append((similarity, place_id))
                    continue
                best = max(checks, key=lambda check: (
                    check.inlier_ratio if check.inlier_ratio is not None else -1.0,
                    check.match_count,
                ))
                self.last_topk[index].update({
                    "local_match_count": best.match_count,
                    "local_inlier_ratio": best.inlier_ratio,
                    "local_reason": best.reason,
                })
                if best.inlier_ratio is None:
                    unresolved.append((similarity, place_id))
                elif best.inlier_ratio >= self.config.place_local_inlier_min:
                    supported.append((similarity, place_id, best))
            if len(supported) > 1:
                return result("UNKNOWN", None, None, "ambiguous_local_candidates")
            if supported:
                similarity, place_id, best = supported[0]
                if any(other_similarity >= similarity - self.config.place_global_margin_min
                       for other_similarity, _ in unresolved):
                    return result("UNKNOWN", place_id, best, "unresolved_competing_candidate")
                self._pending = None
                return result("MATCH", place_id, best, "global_and_local_support")
            if len(unresolved) > 1 and top2 is not None and (
                top1 - top2 < self.config.place_global_margin_min
            ):
                return result("UNKNOWN", None, None, "ambiguous_global_candidates")
            first = self.last_topk[0]
            return result("UNKNOWN", candidates[0][1], LocalCheck(
                first.get("local_match_count", 0), first.get("local_inlier_ratio"),
                first.get("local_reason", "no_local_support"),
            ), first.get("local_reason", "no_local_support"))
        if top1 is not None and top1 > self.config.place_global_new_max:
            return result("UNKNOWN", None, None, "global_candidate_borderline")
        pending = self._pending
        self._pending = (step, features)
        if pending is None or pending[0] == step:
            return result("UNKNOWN", None, None, "new_place_needs_second_observation")
        pending_similarity = float(global_vector @ np.asarray(
            pending[1].global_descriptor, dtype=np.float32,
        ))
        if pending_similarity < self.config.place_new_temporal_similarity_min:
            return result("UNKNOWN", None, None, "new_place_candidate_changed")
        check = local_check(features, pending[1], self.config)
        if check.inlier_ratio is None or check.inlier_ratio < self.config.place_local_inlier_min:
            return result("UNKNOWN", None, check, check.reason)
        self._pending = None
        return result("NEW", None, check, "independent_new_place_confirmation")

    def accept(self, place_id: str, features: VisualPlaceFeatures) -> None:
        frames = self._keyframes.setdefault(place_id, [])
        frames.append(features)
        if len(frames) > self.config.place_keyframes_max:
            # Preserve the first reference, then greedily cover distinct global
            # views. Image quality and recency resolve coverage ties.
            kept = [0]
            while len(kept) < self.config.place_keyframes_max:
                candidates = [index for index in range(1, len(frames)) if index not in kept]
                chosen = max(candidates, key=lambda index: (
                    min(1.0 - float(np.dot(
                        frames[index].global_descriptor,
                        frames[other].global_descriptor,
                    )) for other in kept),
                    float(frames[index].preprocessing_metadata.get("image_quality", 0.0)),
                    index,
                ))
                kept.append(chosen)
            self._keyframes[place_id] = [frames[index] for index in sorted(kept)]

    def clear(self) -> None:
        self._keyframes.clear()
        self._pending = None
        self.last_topk = []
