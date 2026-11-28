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
    # Opt-in read-only XYZRGB sidecar.  It records the exact pixels of each L3
    # Evidence mask so the official ViL3DRel Student can consume real RGB.
    # Disabled by default; it never changes ObjectNode or observe().
    rgb_sidecar_enabled: bool = False
    # Opt-in official ViL3DRel Nr3D Student grounding between the geometry
    # query and the existing Prompt formatter.  Requires the sidecar and a
    # locally built official pointnet2 extension.
    instance_grounder_enabled: bool = False
    grounder_checkpoint: str = ""
    grounder_config: str = ""
    grounder_device: str = "cuda"
    grounder_bert_vocab_path: str = ""
    grounder_seed: int = 0
    # Ablation switches.  Defaults reproduce the original d659e57 ViL3DRel
    # instance-disambiguation path exactly: one full-instruction query and the
    # official global argmax.  The new joint selector, multi-landmark queries
    # and geometry fallback are independent opt-ins.
    grounder_selection_mode: str = "argmax"
    grounder_landmark_queries: bool = False
    grounder_geometry_fallback: bool = False
    #: Organize the existing Navigation Memory injection point as an ordered
    #: instruction plan.  Landmarks with a verified fact keep that fact;
    #: landmarks without one keep only verbatim instruction semantics.  This
    #: is an opt-in so D/E/F keep their previously measured prompt shape.
    grounder_ordered_landmark_memory: bool = False
    #: H1/H2/H3 rendering ablation.  With the defaults below the ordered
    #: renderer is byte-identical to G: missing-Evidence landmarks keep a
    #: coordinate-free placeholder, and every verified fact uses G's compact
    #: sentence.  ``include_missing_landmark_placeholders=False`` omits those
    #: placeholder lines; the freed budget may then carry additional real fact
    #: lines (recorded in ``ordered_memory_rendering``).  Setting
    #: ``preserve_original_spatial_fact=True`` swaps only the fact body for F's
    #: original ``render_context`` sentence.
    include_missing_landmark_placeholders: bool = True
    preserve_original_spatial_fact: bool = False
    #: C arm: H1 plus one object-level admission decision at the final L3
    #: history/Prompt boundary.  ``off`` keeps every existing arm byte-identical
    #: to its previous behaviour.  ``once_per_object`` admits a stable
    #: ObjectNode ``object_id`` once per Episode; a later re-selection of the
    #: same object is refused even when its Evidence ID changed.  Candidates
    #: without a reliable object ID stay fail-open and are only counted.
    history_admission_mode: str = "off"
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
        for name in (
            "rgb_sidecar_enabled", "instance_grounder_enabled",
            "grounder_landmark_queries", "grounder_geometry_fallback",
            "grounder_ordered_landmark_memory",
            "include_missing_landmark_placeholders",
            "preserve_original_spatial_fact",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"spatial_l3.{name} must be a boolean")
        if self.grounder_selection_mode not in {"argmax", "eligible"}:
            raise ValueError(
                "spatial_l3.grounder_selection_mode must be argmax or eligible"
            )
        if (
            self.grounder_landmark_queries
            and self.grounder_selection_mode != "eligible"
        ):
            raise ValueError(
                "spatial_l3.grounder_landmark_queries requires "
                "grounder_selection_mode=eligible"
            )
        if (
            self.grounder_geometry_fallback
            and self.grounder_selection_mode != "eligible"
        ):
            raise ValueError(
                "spatial_l3.grounder_geometry_fallback requires "
                "grounder_selection_mode=eligible"
            )
        if self.grounder_ordered_landmark_memory:
            if self.grounder_selection_mode != "eligible":
                raise ValueError(
                    "spatial_l3.grounder_ordered_landmark_memory requires "
                    "grounder_selection_mode=eligible"
                )
            if not self.grounder_landmark_queries:
                raise ValueError(
                    "spatial_l3.grounder_ordered_landmark_memory requires "
                    "grounder_landmark_queries"
                )
            if not self.instance_grounder_enabled:
                raise ValueError(
                    "spatial_l3.grounder_ordered_landmark_memory requires "
                    "instance_grounder_enabled"
                )
        if not self.grounder_ordered_landmark_memory and (
            not self.include_missing_landmark_placeholders
            or self.preserve_original_spatial_fact
        ):
            raise ValueError(
                "spatial_l3.include_missing_landmark_placeholders=False or "
                "spatial_l3.preserve_original_spatial_fact=True requires "
                "grounder_ordered_landmark_memory"
            )
        if self.history_admission_mode not in {"off", "once_per_object"}:
            raise ValueError(
                "spatial_l3.history_admission_mode must be off or once_per_object"
            )
        if self.history_admission_mode == "once_per_object":
            # The C arm adds one decision to H1's final L3 history boundary.
            # Refuse configurations where that boundary is not the ordered
            # Grounding path, instead of silently running with an inert switch.
            if self.mode != "geometry_on_demand":
                raise ValueError(
                    "spatial_l3.history_admission_mode=once_per_object requires "
                    "mode=geometry_on_demand"
                )
            if not self.instance_grounder_enabled:
                raise ValueError(
                    "spatial_l3.history_admission_mode=once_per_object requires "
                    "instance_grounder_enabled"
                )
            if self.grounder_selection_mode != "eligible":
                raise ValueError(
                    "spatial_l3.history_admission_mode=once_per_object requires "
                    "grounder_selection_mode=eligible"
                )
            if not self.grounder_ordered_landmark_memory:
                raise ValueError(
                    "spatial_l3.history_admission_mode=once_per_object requires "
                    "grounder_ordered_landmark_memory"
                )
        for name in (
            "grounder_checkpoint", "grounder_config", "grounder_device",
            "grounder_bert_vocab_path",
        ):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"spatial_l3.{name} must be a string")
        if self.instance_grounder_enabled:
            if self.mode != "geometry_on_demand":
                raise ValueError(
                    "spatial_l3.instance_grounder_enabled requires "
                    "mode=geometry_on_demand"
                )
            if not self.rgb_sidecar_enabled:
                raise ValueError(
                    "spatial_l3.instance_grounder_enabled requires "
                    "rgb_sidecar_enabled"
                )
            if not self.grounder_checkpoint:
                raise ValueError(
                    "spatial_l3.grounder_checkpoint is required when the "
                    "instance grounder is enabled"
                )
            if not self.grounder_device.strip():
                raise ValueError("spatial_l3.grounder_device must be nonempty")
            if type(self.grounder_seed) is not int or self.grounder_seed < 0:
                raise ValueError(
                    "spatial_l3.grounder_seed must be a nonnegative integer"
                )
        if self.rgb_sidecar_enabled and self.mode == "off":
            raise ValueError(
                "spatial_l3.rgb_sidecar_enabled requires a non-off spatial mode"
            )
        if not isinstance(self.geometry, GeometryConfig):
            raise ValueError("spatial_l3.geometry must be GeometryConfig")
