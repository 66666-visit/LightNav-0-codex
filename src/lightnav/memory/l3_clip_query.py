"""Local text-to-object CLIP query over the ConceptGraphs-Lite L3 map.

The retrieval score is the official ConceptGraphs cosine similarity between a
normalized CLIP text feature and the stored normalized object feature
(``MapObjectList.compute_similarities`` in
``conceptgraph/slam/slam_classes.py``).  Descending similarity ranking follows
the official interactive visualizer
(``scripts/visualize_cfslam_results.py::color_by_clip_sim``), which computes
the same cosine score and selects its maximum.  The official repository does
not ship a separate Top-K service; this module only adds deterministic Top-K
selection over those scores.  It does not implement a new similarity model,
re-embed objects, rebuild the map, or change object identities.

There is no HTTP server and no external API: callers pass a text query and a
Top-K count, and receive existing object records plus their stored geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from lightnav.memory.clip_features import ClipFeatureEncoder, cosine_similarity, normalize_feature
from lightnav.memory.conceptgraphs_lite import ObjectMap

_NO_MATCH = "no_similarity_above_floor"


@dataclass(frozen=True)
class L3QueryCandidate:
    """One scored object, including the raw cosine and why it was rejected.

    ``similarity`` is a raw CLIP cosine score, not a calibrated confidence or a
    probability.  The rank is over every object with a valid visual feature.
    """

    rank: int
    object_id: str
    similarity: float
    class_name: str
    evidence_ids: tuple[str, ...]
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "object_id": self.object_id,
            "similarity": self.similarity,
            "class_name": self.class_name,
            "evidence_ids": list(self.evidence_ids),
            "decision": self.decision,
        }


@dataclass(frozen=True)
class L3QueryMatch:
    """One existing L3 object returned by a local CLIP query."""

    object_id: str
    rank: int
    similarity: float
    class_name: str
    center_world: tuple[float, float, float]
    bbox_3d: tuple[tuple[float, float, float], tuple[float, float, float]]
    observation_count: int
    evidence_ids: tuple[str, ...]
    geometry_status: str
    point_count: int
    clip_feature_observations: int
    latest_step: int
    detector_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "rank": self.rank,
            "similarity": self.similarity,
            "class_name": self.class_name,
            "center_world": list(self.center_world),
            "bbox_3d": [list(corner) for corner in self.bbox_3d],
            "observation_count": self.observation_count,
            "evidence_ids": list(self.evidence_ids),
            "geometry_status": self.geometry_status,
            "point_count": self.point_count,
            "clip_feature_observations": self.clip_feature_observations,
            "latest_step": self.latest_step,
            "detector_confidence": self.detector_confidence,
        }


@dataclass(frozen=True)
class L3QueryResult:
    """Result of one local query, including an explicit empty/no-match reason."""

    query: str
    top_k: int
    matches: tuple[L3QueryMatch, ...] = ()
    candidates: tuple[L3QueryCandidate, ...] = ()
    objects_total: int = 0
    objects_with_features: int = 0
    skipped_reason: str | None = None
    encode_ms: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.matches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "matches": [match.to_dict() for match in self.matches],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "objects_total": self.objects_total,
            "objects_with_features": self.objects_with_features,
            "skipped_reason": self.skipped_reason,
            "encode_ms": self.encode_ms,
        }


def _evidence_summary(object_map: ObjectMap, evidence_ids: tuple[str, ...]) -> tuple[int, float]:
    steps: list[int] = []
    confidences: list[float] = []
    for evidence_id in evidence_ids:
        record = object_map.evidence.get(evidence_id)
        if record is None:
            continue
        steps.append(int(record.step_id))
        confidences.append(float(record.confidence))
    return (max(steps) if steps else 0, max(confidences) if confidences else 0.0)


class L3ClipQuery:
    """Query one live ``ObjectMap`` by natural language; no map mutation."""

    def __init__(self, object_map: ObjectMap, encoder: ClipFeatureEncoder) -> None:
        if not isinstance(object_map, ObjectMap):
            raise TypeError("object_map must be an ObjectMap")
        for method in ("encode_text", "encode_image"):
            if not callable(getattr(encoder, method, None)):
                raise ValueError(f"CLIP encoder is missing {method}()")
        self.object_map = object_map
        self.encoder = encoder

    def query(
        self, text: str, top_k: int, *, min_similarity: float = 0.0,
    ) -> L3QueryResult:
        """Return up to ``top_k`` existing objects ranked by cosine similarity.

        ``min_similarity`` defaults to zero, the mathematical cosine boundary
        between aligned and anti-aligned vectors.  A match must also have a
        strictly positive cosine; an orthogonal (zero) score is not evidence of
        semantic relatedness.  This is not a tuned confidence threshold: the
        official ConceptGraphs code does not threshold cosine query scores at
        all.  The empty/NaN/mismatched cases return an explicit
        ``skipped_reason`` instead of raising or inventing a match.
        """
        if not isinstance(text, str):
            raise TypeError("query text must be a string")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if (
            not isinstance(min_similarity, (int, float))
            or not math.isfinite(float(min_similarity))
            or not -1.0 <= float(min_similarity) <= 1.0
        ):
            raise ValueError("min_similarity must be finite and in [-1, 1]")
        started = perf_counter()
        query = " ".join(text.split())
        objects = list(self.object_map.objects.values())
        if not query:
            return L3QueryResult(
                query="", top_k=top_k, objects_total=len(objects),
                skipped_reason="empty_query",
            )
        if not objects:
            return L3QueryResult(
                query=query, top_k=top_k, skipped_reason="empty_map",
            )

        features: list[np.ndarray] = []
        candidates: list[Any] = []
        for node in objects:
            if node.clip_feature is None or node.clip_feature_observations < 1:
                continue
            try:
                feature = normalize_feature(node.clip_feature)
            except ValueError:
                continue
            features.append(feature)
            candidates.append(node)
        if not features:
            return L3QueryResult(
                query=query, top_k=top_k, objects_total=len(objects),
                skipped_reason="no_object_features",
            )

        dimension = features[0].size
        kept = [(node, feat) for node, feat in zip(candidates, features) if feat.size == dimension]
        if not kept:
            return L3QueryResult(
                query=query, top_k=top_k, objects_total=len(objects),
                skipped_reason="no_object_features",
            )
        query_features = np.asarray(self.encoder.encode_text([query]), dtype=np.float32)
        encode_ms = (perf_counter() - started) * 1000.0
        if query_features.ndim != 2 or query_features.shape != (1, dimension):
            return L3QueryResult(
                query=query, top_k=top_k, objects_total=len(objects),
                objects_with_features=len(kept), skipped_reason="feature_dimension_mismatch",
                encode_ms=encode_ms,
            )
        try:
            query_feature = normalize_feature(query_features[0], expected_dim=dimension)
            similarities = cosine_similarity(
                query_feature, np.stack([feat for _, feat in kept]),
            )
        except ValueError:
            return L3QueryResult(
                query=query, top_k=top_k, objects_total=len(objects),
                objects_with_features=len(kept), skipped_reason="invalid_features",
                encode_ms=encode_ms,
            )
        if not np.isfinite(similarities).all():
            return L3QueryResult(
                query=query, top_k=top_k, objects_total=len(objects),
                objects_with_features=len(kept), skipped_reason="invalid_features",
                encode_ms=encode_ms,
            )

        scored = [
            (float(similarity), node)
            for similarity, (node, _) in zip(similarities, kept)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].object_id))
        accepted = [
            (similarity, node)
            for similarity, node in scored
            if similarity > 0.0 and similarity >= float(min_similarity)
        ]
        accepted_ids = {node.object_id for _, node in accepted[:top_k]}
        candidates: list[L3QueryCandidate] = []
        for rank, (similarity, node) in enumerate(scored, start=1):
            if node.object_id in accepted_ids:
                decision = "matched"
            elif similarity <= 0.0:
                decision = "below_positive_cosine"
            elif similarity < float(min_similarity):
                decision = "below_min_similarity"
            else:
                decision = "ranked_out_by_top_k"
            candidates.append(L3QueryCandidate(
                rank=rank,
                object_id=node.object_id,
                similarity=similarity,
                class_name=node.class_name,
                evidence_ids=tuple(node.evidence_ids),
                decision=decision,
            ))
        matches: list[L3QueryMatch] = []
        for similarity, node in accepted[:top_k]:
            evidence_ids = tuple(node.evidence_ids)
            latest_step, detector_confidence = _evidence_summary(
                self.object_map, evidence_ids,
            )
            rank = next(
                candidate.rank for candidate in candidates
                if candidate.object_id == node.object_id
            )
            matches.append(L3QueryMatch(
                object_id=node.object_id,
                rank=rank,
                similarity=similarity,
                class_name=node.class_name,
                center_world=tuple(float(value) for value in node.center_world),
                bbox_3d=tuple(
                    tuple(float(value) for value in corner) for corner in node.bbox_3d
                ),
                observation_count=int(node.observation_count),
                evidence_ids=evidence_ids,
                geometry_status=node.geometry_status,
                point_count=int(len(node.points_world)),
                clip_feature_observations=int(node.clip_feature_observations),
                latest_step=latest_step,
                detector_confidence=detector_confidence,
            ))
        return L3QueryResult(
            query=query,
            top_k=top_k,
            matches=tuple(matches),
            candidates=tuple(candidates),
            objects_total=len(objects),
            objects_with_features=len(kept),
            skipped_reason=None if matches else _NO_MATCH,
            encode_ms=encode_ms,
        )


def query_objects(
    object_map: ObjectMap, text: str, top_k: int, *,
    encoder: ClipFeatureEncoder, min_similarity: float = 0.0,
) -> L3QueryResult:
    """Functional form of :class:`L3ClipQuery` for a one-shot local query."""
    return L3ClipQuery(object_map, encoder).query(
        text, top_k, min_similarity=min_similarity,
    )
