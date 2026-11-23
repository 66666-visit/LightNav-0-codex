"""On-demand L3 CLIP query control and Prompt evidence formatting.

The default L1 path remains primary.  This module only runs a local CLIP query
after the existing Local Graph selector has reported an empty result, and it
never injects anything unless the query returned a valid cosine match.  It
reuses the existing ``MemoryContext``/``inject_memory_context`` contract and
the same character budget as the original navigation card.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from html import escape
from typing import Any

from lightnav.memory.clip_features import ClipFeatureEncoder
from lightnav.memory.conceptgraphs_lite import ObjectMap
from lightnav.memory.l3_clip_query import L3ClipQuery, L3QueryMatch, L3QueryResult
from lightnav.memory.local_graph_fusion import _plain_label
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence
from lightnav.memory.spatial_graph_adapter import _relative_position

_HEADER = "[Navigation Memory]"

#: The only L1 outcomes that mean "the existing retrieval path has no candidate
#: for this instruction".  ``reactivation_suppressed`` means L1 *has* a valid
#: card and intentionally repeated it; it must not trigger another memory path.
L1_EMPTY_STATUSES = frozenset({"no_candidates", "budget_excluded"})


@dataclass
class L3QueryController:
    """Rate-limit and invoke the local CLIP query for an L3 object map."""

    encoder: ClipFeatureEncoder
    top_k: int = 3
    min_similarity: float = 0.0
    _identity: tuple[str, str, int] | None = field(default=None, init=False, repr=False)
    _last_query_key: tuple[tuple[Any, ...], str] | None = field(
        default=None, init=False, repr=False,
    )
    _last_query_step: int | None = field(default=None, init=False, repr=False)
    last_diagnostic: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("L3 CLIP top_k must be a positive integer")
        if (
            not isinstance(self.min_similarity, (int, float))
            or not math.isfinite(float(self.min_similarity))
            or not -1.0 <= float(self.min_similarity) <= 1.0
        ):
            raise ValueError("L3 CLIP min_similarity must be finite and in [-1, 1]")

    def reset_episode(self, scene_id: str, episode_id: str, rollout_idx: int = 0) -> None:
        self._identity = (scene_id, episode_id, rollout_idx)
        self._last_query_key = None
        self._last_query_step = None
        self.last_diagnostic = None

    def close_episode(self) -> None:
        self._identity = None
        self._last_query_key = None
        self._last_query_step = None
        self.last_diagnostic = None

    @staticmethod
    def map_signature(object_map: ObjectMap) -> tuple[tuple[Any, ...], ...]:
        """Identify new visual evidence without counting pose-only viewpoints."""
        return tuple(sorted(
            (
                node.object_id,
                int(node.clip_feature_observations),
                tuple(node.clip_feature_evidence_ids),
            )
            for node in object_map.objects.values()
            if node.clip_feature is not None and node.clip_feature_observations >= 1
        ))

    def maybe_query(
        self, object_map: ObjectMap, *, step: int, instruction: str, l1_status: str,
        query_text: str | None = None,
    ) -> L3QueryResult | None:
        """Run one query only when L1 is empty and the L3 evidence changed.

        ``query_text`` defaults to the raw navigation instruction, matching the
        official ConceptGraphs text-query path (which accepts a caller-supplied
        text string and does not parse it).  Tests and controlled experiments may
        pass a short object landmark phrase explicitly; no parser is invented
        here.  Returns ``None`` with a diagnostic reason when no query is
        warranted or when the query produced no valid match.  A query is never
        repeated for an unchanged object map and query text.
        """
        query = (
            " ".join(query_text.split()) if isinstance(query_text, str)
            else " ".join(instruction.split()) if isinstance(instruction, str)
            else ""
        )
        query_source = "explicit" if query_text is not None else "instruction"
        diagnostic: dict[str, Any] = {
            "trigger": l1_status,
            "step": step,
            "scene_id": self._identity[0] if self._identity is not None else None,
            "episode_id": self._identity[1] if self._identity is not None else None,
            "rollout_idx": self._identity[2] if self._identity is not None else None,
            "instruction": (
                " ".join(instruction.split()) if isinstance(instruction, str) else ""
            ),
            "query": query,
            "query_source": query_source,
            "query_executed": False,
            "last_query_step": self._last_query_step,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "similarity_semantics": "raw_cosine_not_confidence",
            "candidate_count": 0,
            "scored_candidate_count": 0,
            "skipped_reason": None,
            "query_ms": 0.0,
            "matches": [],
            "candidates": [],
        }
        self.last_diagnostic = diagnostic
        if l1_status not in L1_EMPTY_STATUSES:
            diagnostic["skipped_reason"] = (
                "l1_reactivation_suppressed"
                if l1_status == "reactivation_suppressed"
                else "l1_satisfied"
            )
            return None
        if type(step) is not int or step < 0:
            raise ValueError("L3 query step must be a nonnegative integer")
        if not isinstance(instruction, str):
            raise TypeError("L3 query instruction must be a string")
        if query_text is not None and not isinstance(query_text, str):
            raise TypeError("L3 query text must be a string or None")
        if not query:
            diagnostic["skipped_reason"] = "empty_query"
            return None
        if not object_map.objects:
            diagnostic["skipped_reason"] = "empty_map"
            return None
        if not any(
            node.clip_feature is not None and node.clip_feature_observations >= 1
            for node in object_map.objects.values()
        ):
            diagnostic["skipped_reason"] = "no_object_features"
            return None

        query_key = (self.map_signature(object_map), query)
        if query_key == self._last_query_key:
            diagnostic["skipped_reason"] = "map_unchanged_since_last_query"
            if self._last_query_step is not None:
                diagnostic["steps_since_last_query"] = step - self._last_query_step
            return None
        self._last_query_key = query_key
        diagnostic["query_executed"] = True
        self._last_query_step = step
        try:
            result = L3ClipQuery(object_map, self.encoder).query(
                query, self.top_k, min_similarity=self.min_similarity,
            )
        except Exception as exc:  # noqa: BLE001 - a sidecar must not break L1
            diagnostic["skipped_reason"] = "query_error"
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            return None
        diagnostic.update({
            "candidate_count": len(result.matches),
            "scored_candidate_count": len(result.candidates),
            "query_ms": result.encode_ms,
            "objects_total": result.objects_total,
            "objects_with_features": result.objects_with_features,
            "matches": [
                {
                    "object_id": match.object_id,
                    "rank": match.rank,
                    "similarity": match.similarity,
                    "detector_confidence": match.detector_confidence,
                }
                for match in result.matches
            ],
            "candidates": [candidate.to_dict() for candidate in result.candidates],
            "skipped_reason": result.skipped_reason,
        })
        return result if result.valid else None


def _query_line(match: L3QueryMatch, agent_pose: Any) -> str:
    label = _plain_label(match.class_name) or "object"
    center = ",".join(f"{value:.1f}" for value in match.center_world)
    relative = _relative_position(agent_pose, match.center_world)
    if relative is None:
        geometry = "relative position unknown"
    else:
        bearing = 0.0 if abs(relative[1]) < 0.5 else relative[1]
        geometry = f"{relative[0]:.1f}m, {bearing:+.0f}deg (left +)"
    return (
        f"L3 CLIP cos {match.similarity:.3f}: {label} at world ({center})m; "
        f"{geometry}; {match.observation_count}obs; passage unknown."
    )


def build_l3_query_context(
    result: L3QueryResult | None, object_map: ObjectMap, *, agent_pose: Any,
    max_chars: int,
) -> MemoryContext | None:
    """Translate valid Top-K matches into the existing bounded Prompt block."""
    if result is None or not result.matches:
        return None
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("L3 query prompt budget must be a positive integer")
    lines: list[str] = []
    items: list[LayerMemoryItem] = []
    used = len(_HEADER)
    for match in result.matches:
        line = _query_line(match, agent_pose)
        safe_line = escape(line, quote=False)
        if used + 1 + len(safe_line) > max_chars:
            continue
        used += 1 + len(safe_line)
        lines.append(line)
        provenance = []
        evidence_refs = []
        for evidence_id in match.evidence_ids:
            record = object_map.evidence.get(evidence_id)
            if record is None:
                continue
            provenance.append({
                "scene_id": record.scene_id,
                "episode_id": record.episode_id,
                "rollout_idx": record.rollout_idx,
                "step": record.step_id,
                "evidence_id": record.evidence_id,
                "source": record.source,
                "confidence": record.confidence,
                "geometry_status": record.geometry_status,
            })
            evidence_refs.append(ObservationEvidence(
                event_id=record.evidence_id,
                step_id=record.step_id,
                label=record.class_name,
                confidence=record.confidence,
                source=record.source,
            ))
        items.append(LayerMemoryItem(
            layer="long",
            kind="l3_clip_query_object",
            description=line,
            step_id=match.latest_step,
            confidence=match.detector_confidence,
            source="l3_clip_query",
            evidence_refs=evidence_refs,
            label=_plain_label(match.class_name) or None,
            metadata={
                "evidence_ids": list(match.evidence_ids),
                "serialized_text": safe_line,
                "provenance": provenance,
                "role": "l3_clip_query",
                "object_id": match.object_id,
                "query": result.query,
                "rank": match.rank,
                "similarity": match.similarity,
                "similarity_semantics": "raw_cosine_not_confidence",
                "detector_confidence": match.detector_confidence,
                "center_world": list(match.center_world),
                "bbox_3d": [list(corner) for corner in match.bbox_3d],
                "observation_count": match.observation_count,
                "clip_feature_observations": match.clip_feature_observations,
                "geometry_status": match.geometry_status,
                "traversability_status": "unknown",
            },
        ))
    if not items:
        return None
    return MemoryContext(text=_HEADER + "\n" + "\n".join(lines), layer_items=items)
