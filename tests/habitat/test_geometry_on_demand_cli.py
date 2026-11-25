"""Configuration contracts for the Geometry-only on-demand arm."""

from __future__ import annotations

import pytest

from lightnav.cli.eval_habitat import build_parser, config_from_args
from lightnav.habitat.prompt_artifacts import build_prompt_protocol
from lightnav.habitat.runner import MemoryConfig, effective_spatial_l3_config
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.spatial_l3_config import SpatialL3Config


def _config(**overrides) -> MemoryConfig:
    values = {
        "enabled": True,
        "strategy": "hierarchical",
        "extractor": "yolo11",
        "extractor_model_path": "/fake/yolo11n.pt",
        "local_graph": LocalGraphConfig(enabled=True),
        "spatial_l3": SpatialL3Config(
            mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
            seg_every_n_steps=5,
        ),
        "prompt_mode": "l1_original_only",
    }
    values.update(overrides)
    return MemoryConfig(**values)


def test_cli_accepts_geometry_on_demand_without_clip_flags() -> None:
    parser = build_parser()
    parsed = parser.parse_args([
        "--model_path", "/fake/model",
        "--spatial_l3_mode", "geometry_on_demand",
        "--spatial_l3_seg_model_path", "/fake/seg.pt",
    ])
    assert parsed.spatial_l3_mode == "geometry_on_demand"
    assert parsed.spatial_l3_clip_model_path == ""


def test_config_accepts_geometry_on_demand_without_clip_path() -> None:
    config = _config()
    assert config.spatial_l3.clip_model_path == ""
    assert effective_spatial_l3_config(config).mode == "geometry_on_demand"


def test_geometry_mode_keeps_legacy_and_l1_original_only_only() -> None:
    _config(prompt_mode="legacy")
    with pytest.raises(ValueError, match="on-demand L3"):
        _config(prompt_mode="l1_l3_joint")


def test_geometry_protocol_records_no_clip_similarity() -> None:
    protocol = build_prompt_protocol(_config())
    block = protocol["budget"]["on_demand_l3"]
    assert block["retrieval"] == "spatial_graph_adapter"
    assert block["clip_weights_required"] is False
    assert block["clip_similarity_used"] is False
    assert protocol["layer_activation"]["l3_layer"] == (
        "geometry_on_demand_when_l1_empty"
    )


def test_geometry_mode_does_not_require_clip_precision_device_pair() -> None:
    config = SpatialL3Config(
        mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
        clip_device="cpu", clip_precision="fp16",
    )
    assert config.mode == "geometry_on_demand"


def test_config_from_args_wires_geometry_mode_without_clip() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--model_path", "/fake/model",
        "--memory_enabled",
        "--memory_strategy", "hierarchical",
        "--memory_extractor", "yolo11",
        "--memory_extractor_model_path", "/fake/yolo11n.pt",
        "--local_graph_enabled",
        "--spatial_l3_mode", "geometry_on_demand",
        "--spatial_l3_seg_model_path", "/fake/seg.pt",
        "--memory_prompt_mode", "l1_original_only",
        "--output_dir", "/tmp/geometry_on_demand_cli_test",
    ])
    config = config_from_args(args)
    assert config.memory.spatial_l3.mode == "geometry_on_demand"
    assert config.memory.spatial_l3.clip_model_path == ""
    assert config.memory.spatial_l3.seg_every_n_steps == 5


def test_cli_wires_opt_in_rgb_sidecar_and_grounder() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--model_path", "/fake/model",
        "--memory_enabled",
        "--memory_strategy", "hierarchical",
        "--memory_extractor", "yolo11",
        "--memory_extractor_model_path", "/fake/yolo11n.pt",
        "--local_graph_enabled",
        "--spatial_l3_mode", "geometry_on_demand",
        "--spatial_l3_seg_model_path", "/fake/seg.pt",
        "--memory_prompt_mode", "l1_original_only",
        "--spatial_l3_rgb_sidecar",
        "--spatial_l3_instance_grounder",
        "--spatial_l3_grounder_checkpoint", "/fake/student.pt",
        "--spatial_l3_grounder_config", "/fake/model.yaml",
        "--spatial_l3_grounder_device", "cuda:1",
        "--spatial_l3_grounder_bert_vocab", "/fake/vocab.txt",
        "--spatial_l3_grounder_seed", "7",
        "--output_dir", "/tmp/geometry_on_demand_cli_test",
    ])
    spatial = config_from_args(args).memory.spatial_l3
    assert spatial.rgb_sidecar_enabled is True
    assert spatial.instance_grounder_enabled is True
    assert spatial.grounder_checkpoint == "/fake/student.pt"
    assert spatial.grounder_config == "/fake/model.yaml"
    assert spatial.grounder_device == "cuda:1"
    assert spatial.grounder_bert_vocab_path == "/fake/vocab.txt"
    assert spatial.grounder_seed == 7


def test_grounder_requires_sidecar_and_geometry_mode() -> None:
    base = {
        "mode": "geometry_on_demand", "seg_model_path": "/fake/seg.pt",
        "instance_grounder_enabled": True,
        "grounder_checkpoint": "/fake/student.pt",
    }
    with pytest.raises(ValueError, match="rgb_sidecar_enabled"):
        SpatialL3Config(**base)
    with pytest.raises(ValueError, match="geometry_on_demand"):
        SpatialL3Config(
            **{**base, "mode": "shadow", "rgb_sidecar_enabled": True},
        )


def _grounder_ablation_base() -> list[str]:
    return [
        "--model_path", "/fake/model",
        "--memory_enabled",
        "--memory_strategy", "hierarchical",
        "--memory_extractor", "yolo11",
        "--memory_extractor_model_path", "/fake/yolo11n.pt",
        "--local_graph_enabled",
        "--spatial_l3_mode", "geometry_on_demand",
        "--spatial_l3_seg_model_path", "/fake/seg.pt",
        "--memory_prompt_mode", "l1_original_only",
        "--spatial_l3_rgb_sidecar",
        "--spatial_l3_instance_grounder",
        "--spatial_l3_grounder_checkpoint", "/fake/student.pt",
        "--output_dir", "/tmp/geometry_on_demand_cli_test",
    ]


def test_grounder_ablation_defaults_are_the_original_argmax_path() -> None:
    parser = build_parser()
    spatial = config_from_args(
        parser.parse_args(_grounder_ablation_base())
    ).memory.spatial_l3
    assert spatial.grounder_selection_mode == "argmax"
    assert spatial.grounder_landmark_queries is False
    assert spatial.grounder_geometry_fallback is False


def test_grounder_ablation_flags_are_independently_enableable() -> None:
    parser = build_parser()
    eligible = config_from_args(parser.parse_args([
        *_grounder_ablation_base(),
        "--spatial_l3_grounder_selection", "eligible",
    ])).memory.spatial_l3
    assert eligible.grounder_selection_mode == "eligible"
    assert eligible.grounder_landmark_queries is False
    assert eligible.grounder_geometry_fallback is False

    landmarks = config_from_args(parser.parse_args([
        *_grounder_ablation_base(),
        "--spatial_l3_grounder_selection", "eligible",
        "--spatial_l3_grounder_landmark_queries",
    ])).memory.spatial_l3
    assert landmarks.grounder_landmark_queries is True
    assert landmarks.grounder_geometry_fallback is False

    fallback = config_from_args(parser.parse_args([
        *_grounder_ablation_base(),
        "--spatial_l3_grounder_selection", "eligible",
        "--spatial_l3_grounder_landmark_queries",
        "--spatial_l3_grounder_geometry_fallback",
    ])).memory.spatial_l3
    assert fallback.grounder_geometry_fallback is True


def test_landmark_or_fallback_require_eligible_selection() -> None:
    base = {
        "mode": "geometry_on_demand", "seg_model_path": "/fake/seg.pt",
        "rgb_sidecar_enabled": True, "instance_grounder_enabled": True,
        "grounder_checkpoint": "/fake/student.pt",
    }
    with pytest.raises(ValueError, match="grounder_selection_mode=eligible"):
        SpatialL3Config(**{**base, "grounder_landmark_queries": True})
    with pytest.raises(ValueError, match="grounder_selection_mode=eligible"):
        SpatialL3Config(**{**base, "grounder_geometry_fallback": True})


def test_protocol_records_joint_grounding_and_geometry_fallback() -> None:
    config = _config(spatial_l3=SpatialL3Config(
        mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
        rgb_sidecar_enabled=True, instance_grounder_enabled=True,
        grounder_checkpoint="/fake/student.pt", grounder_device="cpu",
        grounder_selection_mode="eligible",
        grounder_landmark_queries=True,
        grounder_geometry_fallback=True,
    ))
    block = build_prompt_protocol(config)["budget"]["on_demand_l3"]
    assert block["retrieval"] == "spatial_graph_adapter+vil3dref_student"
    assert block["grounder_selection_mode"] == "eligible"
    assert block["proposal_object_evidence_mapping"] is True
    assert block["eligible_proposal_selection"] is True
    assert block["landmark_queries"] is True
    assert block["geometry_fallback"] == "original SpatialGraphAdapter items"
