"""Scene-scoped, JSON-safe records for the evidence-driven V3 memory path.

Confidence fields retain detector or similarity measurements. They are not
calibrated probabilities and must not be combined into one such value.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


_OBSERVATION_STATUSES = {"observed", "inferred", "unknown"}
_SPATIAL_STATUSES = {"observed", "inferred", "traversed", "unknown"}
_TRAVERSABILITY_STATUSES = {"unknown", "inferred", "traversed", "verified", "failed"}
_OUTCOMES = {"unknown", "success", "failure", "partial", "truncated", "aborted"}
_NODE_TYPES = {"place", "landmark", "viewpoint"}
_EDGE_TYPES = {
    "observed_transition", "inferred_relation", "verified_traversal",
    "spatial_relation", "viewpoint_connection",  # read compatibility for draft records
}


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _index(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _finite(value: float | int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _raw_confidence(value: float | None, name: str) -> None:
    if value is not None:
        _finite(value, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


def _status(value: str, allowed: set[str], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")


def _vector(value: tuple[float, ...] | None, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a nonempty numeric sequence")
    for element in value:
        _finite(element, name)
    return tuple(float(element) for element in value)


def _json_safe(value: Any) -> Any:
    """Normalize tuples to lists and reject tensors, NaN and lossy dict keys."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        _finite(value, "JSON number")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    raise TypeError(f"not JSON-safe: {type(value).__name__}")


def canonical_evidence_key(
    scene_id: str, episode_id: str, rollout_idx: int, evidence_id: str,
) -> str:
    """Return an unambiguous, cross-Episode evidence reference."""
    _nonempty(scene_id, "scene_id")
    _nonempty(episode_id, "episode_id")
    _index(rollout_idx, "rollout_idx")
    _nonempty(evidence_id, "evidence_id")
    return json.dumps(
        [scene_id, episode_id, rollout_idx, evidence_id],
        ensure_ascii=False, separators=(",", ":"),
    )


def parse_evidence_key(value: str) -> tuple[str, str, int, str]:
    """Parse only the canonical representation, so equivalent keys cannot split counts."""
    if not isinstance(value, str):
        raise ValueError("evidence reference must be a canonical JSON string")
    try:
        parts = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid evidence reference") from exc
    if not isinstance(parts, list) or len(parts) != 4:
        raise ValueError("evidence reference must have four components")
    scene_id, episode_id, rollout_idx, evidence_id = parts
    if canonical_evidence_key(scene_id, episode_id, rollout_idx, evidence_id) != value:
        raise ValueError("evidence reference is not canonical")
    return scene_id, episode_id, rollout_idx, evidence_id


def _refs(
    values: list[str] | tuple[str, ...], scene_id: str, name: str,
    *, episode_id: str | None = None, rollout_idx: int | None = None,
) -> None:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of evidence references")
    for value in values:
        ref_scene, ref_episode, ref_rollout, _ = parse_evidence_key(value)
        if ref_scene != scene_id:
            raise ValueError(f"{name} crosses scene identity")
        if episode_id is not None and (ref_episode != episode_id or ref_rollout != rollout_idx):
            raise ValueError(f"{name} crosses episode identity")


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    scene_id: str
    episode_id: str
    rollout_idx: int
    step: int
    event_type: str
    semantic_label: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    visual_descriptor: tuple[float, ...] | None = None
    detector_name: str | None = None
    raw_detector_confidence: float | None = None
    instruction_terms: tuple[str, ...] = ()
    position: tuple[float, ...] | None = None
    heading: float | None = None
    source_evidence_ids: tuple[str, ...] = ()
    observation_status: str = "unknown"
    spatial_status: str = "unknown"
    traversability_status: str = "unknown"
    outcome: str = "unknown"
    support_count: int = 1
    first_seen: int | None = None
    last_seen: int | None = None
    temporal_support: int = 1
    episode_support: int = 1
    spatial_consistency: float | None = None
    provenance_validity: bool = True
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError("EvidenceRecord schema_version must be 3")
        canonical_evidence_key(self.scene_id, self.episode_id, self.rollout_idx, self.evidence_id)
        _index(self.step, "step")
        _nonempty(self.event_type, "event_type")
        if self.semantic_label is not None and not isinstance(self.semantic_label, str):
            raise ValueError("semantic_label must be a string or None")
        if self.detector_name is not None:
            _nonempty(self.detector_name, "detector_name")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a JSON object")
        _json_safe(self.payload)
        _raw_confidence(self.raw_detector_confidence, "raw_detector_confidence")
        object.__setattr__(self, "visual_descriptor", _vector(self.visual_descriptor, "visual_descriptor"))
        object.__setattr__(self, "position", _vector(self.position, "position"))
        if self.heading is not None:
            _finite(self.heading, "heading")
        if not isinstance(self.instruction_terms, (tuple, list)) or any(
            not isinstance(term, str) or not term.strip() for term in self.instruction_terms
        ):
            raise ValueError("instruction_terms must be nonempty strings")
        object.__setattr__(self, "instruction_terms", tuple(self.instruction_terms))
        _refs(self.source_evidence_ids, self.scene_id, "source_evidence_ids")
        object.__setattr__(self, "source_evidence_ids", tuple(self.source_evidence_ids))
        _status(self.observation_status, _OBSERVATION_STATUSES, "observation_status")
        _status(self.spatial_status, _SPATIAL_STATUSES, "spatial_status")
        _status(self.traversability_status, _TRAVERSABILITY_STATUSES, "traversability_status")
        _status(self.outcome, _OUTCOMES, "outcome")
        for name in ("support_count", "temporal_support", "episode_support"):
            _index(getattr(self, name), name)
        if self.support_count < 1:
            raise ValueError("support_count must be at least one")
        first = self.step if self.first_seen is None else self.first_seen
        last = self.step if self.last_seen is None else self.last_seen
        _index(first, "first_seen")
        _index(last, "last_seen")
        if not first <= self.step <= last:
            raise ValueError("first_seen, step and last_seen must be ordered")
        object.__setattr__(self, "first_seen", first)
        object.__setattr__(self, "last_seen", last)
        if self.spatial_consistency is not None:
            _finite(self.spatial_consistency, "spatial_consistency")
        if not isinstance(self.provenance_validity, bool):
            raise ValueError("provenance_validity must be a boolean")

    @property
    def key(self) -> str:
        return canonical_evidence_key(
            self.scene_id, self.episode_id, self.rollout_idx, self.evidence_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceRecord:
        return cls(**data)


@dataclass
class EpisodeMemory:
    scene_id: str
    episode_id: str
    instruction: str
    rollout_idx: int = 0
    observations: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    transitions: list[str] = field(default_factory=list)
    landmarks: list[str] = field(default_factory=list)
    outcome: str = "unknown"
    start_step: int = 0
    end_step: int = 0
    evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _nonempty(self.scene_id, "scene_id")
        _nonempty(self.episode_id, "episode_id")
        if not isinstance(self.instruction, str):
            raise ValueError("instruction must be a string")
        _index(self.rollout_idx, "rollout_idx")
        _index(self.start_step, "start_step")
        _index(self.end_step, "end_step")
        if self.end_step < self.start_step:
            raise ValueError("end_step precedes start_step")
        _status(self.outcome, _OUTCOMES, "outcome")
        for name in ("observations", "actions", "transitions", "landmarks", "evidence_ids"):
            _refs(getattr(self, name), self.scene_id, name,
                  episode_id=self.episode_id, rollout_idx=self.rollout_idx)

    @property
    def key(self) -> str:
        return json.dumps(
            [self.scene_id, self.episode_id, self.rollout_idx],
            ensure_ascii=False, separators=(",", ":"),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeMemory:
        return cls(**data)


@dataclass
class SpatialNodeV3:
    node_id: str
    scene_id: str
    node_type: str
    visual_descriptor: tuple[float, ...] | None = None
    semantic_label: str | None = None
    position: tuple[float, ...] | None = None
    detector_confidence: float | None = None
    observation_count: int = 0
    support_episode_count: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    first_seen_step: int = 0
    last_seen_step: int = 0
    spatial_status: str = "unknown"

    def __post_init__(self) -> None:
        _nonempty(self.node_id, "node_id")
        _nonempty(self.scene_id, "scene_id")
        _status(self.node_type, _NODE_TYPES, "node_type")
        self.visual_descriptor = _vector(self.visual_descriptor, "visual_descriptor")
        self.position = _vector(self.position, "position")
        if self.semantic_label is not None and not isinstance(self.semantic_label, str):
            raise ValueError("semantic_label must be a string or None")
        _raw_confidence(self.detector_confidence, "detector_confidence")
        _index(self.observation_count, "observation_count")
        _index(self.support_episode_count, "support_episode_count")
        _refs(self.evidence_ids, self.scene_id, "evidence_ids")
        _index(self.first_seen_step, "first_seen_step")
        _index(self.last_seen_step, "last_seen_step")
        if self.last_seen_step < self.first_seen_step:
            raise ValueError("last_seen_step precedes first_seen_step")
        _status(self.spatial_status, _SPATIAL_STATUSES, "spatial_status")

    @property
    def key(self) -> str:
        return json.dumps([self.scene_id, self.node_id], ensure_ascii=False, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialNodeV3:
        return cls(**data)


@dataclass
class SpatialEdgeV3:
    edge_id: str
    scene_id: str
    source_node: str
    target_node: str
    relation_type: str
    traversal_count: int = 0
    successful_traversals: int = 0
    failed_traversals: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    confidence_statistics: dict[str, float | int] = field(default_factory=dict)
    spatial_status: str = "unknown"
    traversability_status: str = "unknown"
    first_seen_step: int = 0
    last_seen_step: int = 0

    def __post_init__(self) -> None:
        for name in ("edge_id", "scene_id", "source_node", "target_node"):
            _nonempty(getattr(self, name), name)
        _status(self.relation_type, _EDGE_TYPES, "relation_type")
        for name in ("traversal_count", "successful_traversals", "failed_traversals"):
            _index(getattr(self, name), name)
        if self.successful_traversals + self.failed_traversals > self.traversal_count:
            raise ValueError("traversal outcomes exceed traversal_count")
        _refs(self.evidence_ids, self.scene_id, "evidence_ids")
        if not isinstance(self.confidence_statistics, dict):
            raise ValueError("confidence_statistics must be an object")
        for name, value in self.confidence_statistics.items():
            _nonempty(name, "confidence_statistics key")
            _finite(value, "confidence_statistics value")
        _status(self.spatial_status, _SPATIAL_STATUSES, "spatial_status")
        _status(self.traversability_status, _TRAVERSABILITY_STATUSES, "traversability_status")
        if self.relation_type == "verified_traversal":
            if self.successful_traversals < 1 or self.traversal_count < 1:
                raise ValueError("verified traversal requires a successful traversal")
            if self.spatial_status != "traversed" or self.traversability_status not in {
                "traversed", "verified",
            }:
                raise ValueError("verified traversal requires traversed spatial status")
        elif (
            self.traversal_count or self.successful_traversals or self.failed_traversals
            or self.traversability_status != "unknown"
        ):
            raise ValueError("an unverified relation must have UNKNOWN traversability")
        _index(self.first_seen_step, "first_seen_step")
        _index(self.last_seen_step, "last_seen_step")
        if self.last_seen_step < self.first_seen_step:
            raise ValueError("last_seen_step precedes first_seen_step")

    @property
    def key(self) -> str:
        return json.dumps([self.scene_id, self.edge_id], ensure_ascii=False, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialEdgeV3:
        return cls(**data)


@dataclass(frozen=True)
class RetrievedMemory:
    memory_id: str
    kind: str
    scene_id: str
    description: str
    evidence_ids: tuple[str, ...] = ()
    source_step: int | None = None
    observation_status: str = "unknown"
    spatial_status: str = "unknown"
    traversability_status: str = "unknown"
    outcome: str = "unknown"
    score: float = 0.0
    channel_ranks: dict[str, int] = field(default_factory=dict)
    current_observation: bool = False

    def __post_init__(self) -> None:
        for name in ("memory_id", "kind", "scene_id", "description"):
            _nonempty(getattr(self, name), name)
        _refs(self.evidence_ids, self.scene_id, "evidence_ids")
        if not self.evidence_ids:
            raise ValueError("retrieved memory requires source evidence")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        if self.source_step is not None:
            _index(self.source_step, "source_step")
        _status(self.observation_status, _OBSERVATION_STATUSES, "observation_status")
        _status(self.spatial_status, _SPATIAL_STATUSES, "spatial_status")
        _status(self.traversability_status, _TRAVERSABILITY_STATUSES, "traversability_status")
        _status(self.outcome, _OUTCOMES, "outcome")
        _finite(self.score, "score")
        if not isinstance(self.current_observation, bool):
            raise ValueError("current_observation must be a boolean")
        if not isinstance(self.channel_ranks, dict):
            raise ValueError("channel_ranks must be an object")
        for channel, rank in self.channel_ranks.items():
            _nonempty(channel, "channel")
            _index(rank, "rank")
            if rank < 1:
                raise ValueError("channel rank must be positive")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievedMemory:
        return cls(**data)
