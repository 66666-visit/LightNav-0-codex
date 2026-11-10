"""JSON-compatible records for one episode of explicit navigation memory.

All persistent fields are observations, instructions, commands, or model outputs.
``ObservationPacket.rgb`` is transient and intentionally omitted from ``to_dict``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WorkingMemory:
    episode_id: str
    instruction: str
    step_id: int = -1
    place_id: str | None = None
    place_confidence: float = 0.0
    place_source: str = "none"
    recent_event_ids: list[str] = field(default_factory=list)
    task_progress: str = "not_started"
    instruction_matches: list[str] = field(default_factory=list)
    last_action: dict[str, Any] | None = None
    last_prediction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkingMemory:
        return cls(**data)


@dataclass
class MemoryEvent:
    event_id: str
    step_id: int
    kind: str
    description: str
    confidence: float
    source: str
    place_id: str | None = None
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    episode_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEvent:
        return cls(**data)


@dataclass
class SceneNode:
    node_id: str
    kind: str  # place | object
    label: str
    confidence: float
    source: str
    first_step: int
    last_step: int
    visited: bool = False
    status: str = "uncertain"  # uncertain | observed
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneNode:
        return cls(**data)


@dataclass
class SceneEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str  # visible_from | possible_transition | contains | connected_to
    confidence: float
    source: str
    step_id: int
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneEdge:
        return cls(**data)


@dataclass
class MemoryQuery:
    instruction: str
    current_step: int
    place_id: str | None = None
    top_k: int = 5
    max_chars: int = 600

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryQuery:
        return cls(**data)


@dataclass
class ObservationEvidence:
    """Reference to one recorded observation, not a persistent object identity."""

    event_id: str
    step_id: int
    label: str
    confidence: float
    source: str
    node_id: str | None = None
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationEvidence:
        return cls(**data)


@dataclass
class LayerMemoryItem:
    """One layer's retrievable statement with its original evidence references."""

    layer: str
    kind: str
    description: str
    step_id: int
    confidence: float
    source: str
    evidence_refs: list[ObservationEvidence] = field(default_factory=list)
    importance: float = 0.5
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerMemoryItem:
        values = dict(data)
        values["evidence_refs"] = [
            ObservationEvidence.from_dict(item) for item in values.get("evidence_refs", [])
        ]
        return cls(**values)


@dataclass
class MemoryContext:
    events: list[MemoryEvent] = field(default_factory=list)
    nodes: list[SceneNode] = field(default_factory=list)
    edges: list[SceneEdge] = field(default_factory=list)
    text: str = ""
    layer_items: list[LayerMemoryItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "text": self.text,
            "layer_items": [item.to_dict() for item in self.layer_items],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryContext:
        return cls(
            events=[MemoryEvent.from_dict(item) for item in data.get("events", [])],
            nodes=[SceneNode.from_dict(item) for item in data.get("nodes", [])],
            edges=[SceneEdge.from_dict(item) for item in data.get("edges", [])],
            text=data.get("text", ""),
            layer_items=[LayerMemoryItem.from_dict(item) for item in data.get("layer_items", [])],
        )


@dataclass
class ObservationPacket:
    step_id: int
    rgb: Any = None  # HWC uint8 RGB; transient input, never serialized
    instruction: str = ""
    last_action: dict[str, Any] | None = None  # previously executed command
    prediction: str | None = None  # previous model output, if available

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "instruction": self.instruction,
            "last_action": self.last_action,
            "prediction": self.prediction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationPacket:
        return cls(**data)


@dataclass
class DetectedObject:
    """One object detection from an RGB frame, without storing the frame."""

    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectedObject:
        values = dict(data)
        if values.get("bbox_xyxy") is not None:
            values["bbox_xyxy"] = tuple(values["bbox_xyxy"])
        return cls(**values)


@dataclass
class SceneExtraction:
    """Optional extractor output; no value here is inferred by MemoryManager."""

    place_label: str | None = None
    objects: list[str] = field(default_factory=list)
    landmarks: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "none"
    detections: list[DetectedObject] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneExtraction:
        values = dict(data)
        values["detections"] = [
            DetectedObject.from_dict(item) for item in values.get("detections", [])
        ]
        return cls(**values)


@dataclass(frozen=True)
class RGMEvidence:
    episode_id: str
    step: int
    event_id: str
    kind: str
    source: str
    payload: dict[str, Any]
    confidence: float | None
    related_instruction_terms: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    score: float | None
    reason: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class VisualPlaceFeatures:
    global_descriptor: tuple[float, ...]
    patch_descriptors: Any  # transient float16 ndarray in D; never serialized
    patch_grid: tuple[int, int]
    patch_indices: tuple[int, ...] = ()
    preprocessing_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlaceDecision:
    status: str  # MATCH | NEW | UNKNOWN
    candidate_place_id: str | None
    global_similarity: float | None
    second_similarity: float | None
    local_match_count: int
    local_inlier_ratio: float | None
    reason: str


@dataclass(frozen=True)
class MemorySelection:
    accepted_records: tuple[LayerMemoryItem, ...]
    rejected_records: tuple[tuple[str, str], ...]
    gate_reasons: tuple[str, ...]
    injected_text: str
