"""Uncalibrated, episode-local RGM V1 gates and evidence thresholds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RGMConfig:
    enabled: bool = False
    step0_conditional: bool = True
    write_gate_enabled: bool = True
    retrieval_gate_enabled: bool = True
    prompt_gate_enabled: bool = True
    yolo_temporal_enabled: bool = False
    place_two_stage_enabled: bool = False
    topology_motion_validation_enabled: bool = False
    adaptive_extraction_enabled: bool = False
    min_candidate_confidence: float = 0.25
    min_instruction_landmark_confidence: float = 0.60
    temporal_window: int = 3
    temporal_min_support: int = 2
    place_top_k: int = 3
    place_global_candidate_min: float = 0.75
    place_global_new_max: float = 0.65
    place_local_cosine_min: float = 0.70
    place_local_min_matches: int = 8
    place_local_inlier_min: float = 0.50
    place_min_image_quality: float = 0.20
    place_global_margin_min: float = 0.03
    place_new_temporal_similarity_min: float = 0.85
    place_ransac_reprojection_threshold: float = 0.08
    place_keyframes_max: int = 3
    topology_visual_delta_min: float = 0.35
    prompt_max_records: int = 3
    prompt_max_chars: int = 400

    def __post_init__(self) -> None:
        if self.adaptive_extraction_enabled:
            raise ValueError("RGM V1 does not implement adaptive extraction")
        for name in (
            "min_candidate_confidence", "min_instruction_landmark_confidence",
            "place_local_inlier_min", "topology_visual_delta_min",
            "place_min_image_quality",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"rgm.{name} must be in [0, 1]")
        for name in (
            "place_global_candidate_min", "place_global_new_max",
            "place_local_cosine_min", "place_global_margin_min",
            "place_new_temporal_similarity_min",
        ):
            if not -1.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"rgm.{name} must be in [-1, 1]")
        for name in (
            "temporal_window", "temporal_min_support", "place_top_k",
            "place_local_min_matches", "place_keyframes_max", "prompt_max_records",
            "prompt_max_chars",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"rgm.{name} must be positive")
        if self.temporal_min_support > self.temporal_window:
            raise ValueError("rgm temporal_min_support exceeds temporal_window")
        if self.place_global_new_max >= self.place_global_candidate_min:
            raise ValueError("rgm new-place bound must be below candidate bound")
        if not 0.0 < self.place_ransac_reprojection_threshold <= 1.0:
            raise ValueError("rgm place_ransac_reprojection_threshold must be in (0, 1]")
