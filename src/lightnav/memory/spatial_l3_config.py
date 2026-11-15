"""Independent ConceptGraphs-Lite settings for the existing Local Graph path."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lightnav.memory.conceptgraphs_lite import GeometryConfig


@dataclass(frozen=True)
class SpatialL3Config:
    mode: str = "off"  # off | shadow | inject
    seg_model_path: str = ""
    seg_device: str = "cpu"
    seg_confidence: float = 0.25
    seg_every_n_steps: int = 5
    evidence_min_iou: float = 0.5
    evidence_ambiguity_margin: float = 0.1
    min_injection_confidence: float = 0.5
    geometry: GeometryConfig = field(default_factory=GeometryConfig)

    def __post_init__(self) -> None:
        if self.mode not in {"off", "shadow", "inject"}:
            raise ValueError("spatial_l3.mode must be off, shadow or inject")
        if self.mode != "off" and not self.seg_model_path:
            raise ValueError("spatial_l3.seg_model_path is required when enabled")
        if not isinstance(self.seg_device, str) or not self.seg_device.strip():
            raise ValueError("spatial_l3.seg_device must be nonempty")
        if not isinstance(self.seg_confidence, (float, int)) or not (
            math.isfinite(self.seg_confidence) and 0.0 < self.seg_confidence <= 1.0
        ):
            raise ValueError("spatial_l3.seg_confidence must be finite and in (0, 1]")
        if type(self.seg_every_n_steps) is not int or self.seg_every_n_steps < 1:
            raise ValueError("spatial_l3.seg_every_n_steps must be a positive integer")
        for name in (
            "evidence_min_iou", "evidence_ambiguity_margin", "min_injection_confidence",
        ):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"spatial_l3.{name} must be finite and in [0, 1]")
        if not isinstance(self.geometry, GeometryConfig):
            raise ValueError("spatial_l3.geometry must be GeometryConfig")
