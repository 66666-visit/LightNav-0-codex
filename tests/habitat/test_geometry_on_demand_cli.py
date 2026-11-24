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
