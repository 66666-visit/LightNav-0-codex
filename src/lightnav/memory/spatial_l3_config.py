"""Independent ConceptGraphs-Lite settings for the existing Local Graph path."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lightnav.memory.conceptgraphs_lite import GeometryConfig


@dataclass(frozen=True)
class SpatialL3Config:
    # off | shadow | inject | on_demand | geometry_on_demand.  ``on_demand``
    # keeps L1 Original as the primary path and runs the local CLIP query only
    # after L1 reports an empty candidate set.  ``geometry_on_demand`` uses the
    # same gate but the existing geometry-only spatial candidate retrieval, so
    # it never needs CLIP weights or OpenCLIP.
    mode: str = "off"
    seg_model_path: str = ""
    seg_device: str = "cpu"
    seg_confidence: float = 0.25
    seg_every_n_steps: int = 5
    evidence_min_iou: float = 0.5
    evidence_ambiguity_margin: float = 0.1
    min_injection_confidence: float = 0.5
    # Local OpenCLIP checkpoint used by the on-demand text query.  The model
    # architecture name matches ConceptGraphs' official ViT-H-14 encoder.
    clip_model_path: str = ""
    clip_model_name: str = "ViT-H-14"
    clip_device: str = "cpu"
    clip_precision: str = "fp32"
    clip_batch_size: int = 8
    clip_top_k: int = 3
    clip_min_similarity: float = 0.0
    geometry: GeometryConfig = field(default_factory=GeometryConfig)

    def __post_init__(self) -> None:
        if self.mode not in {"off", "shadow", "inject", "on_demand", "geometry_on_demand"}:
            raise ValueError(
                "spatial_l3.mode must be off, shadow, inject, on_demand or "
                "geometry_on_demand"
            )
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
        if not isinstance(self.clip_model_path, str):
            raise ValueError("spatial_l3.clip_model_path must be a string")
        if not isinstance(self.clip_model_name, str) or not self.clip_model_name.strip():
            raise ValueError("spatial_l3.clip_model_name must be nonempty")
        if not isinstance(self.clip_device, str) or not self.clip_device.strip():
            raise ValueError("spatial_l3.clip_device must be nonempty")
        if self.clip_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("spatial_l3.clip_precision must be fp32, fp16 or bf16")
        if (
            (self.mode == "on_demand" or self.clip_model_path)
            and self.clip_device.strip().lower() == "cpu"
            and self.clip_precision != "fp32"
        ):
            raise ValueError("non-fp32 spatial_l3.clip_precision requires a CUDA device")
        if type(self.clip_batch_size) is not int or self.clip_batch_size < 1:
            raise ValueError("spatial_l3.clip_batch_size must be a positive integer")
        if type(self.clip_top_k) is not int or self.clip_top_k < 1:
            raise ValueError("spatial_l3.clip_top_k must be a positive integer")
        if (
            not isinstance(self.clip_min_similarity, (float, int))
            or not math.isfinite(self.clip_min_similarity)
            or not -1.0 <= float(self.clip_min_similarity) <= 1.0
        ):
            raise ValueError("spatial_l3.clip_min_similarity must be finite and in [-1, 1]")
        if self.mode == "on_demand" and not self.clip_model_path:
            raise ValueError("spatial_l3.clip_model_path is required for on_demand mode")
        if not isinstance(self.geometry, GeometryConfig):
            raise ValueError("spatial_l3.geometry must be GeometryConfig")
