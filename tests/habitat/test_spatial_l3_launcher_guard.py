"""Spatial run settings get a separate marker in a fresh Local Graph directory."""

from __future__ import annotations

import json

import pytest

from lightnav.cli.eval_habitat import build_parser, config_from_args
from lightnav.habitat.runner import (
    MemoryConfig,
    _guard_non_v3_v4_output_dir,
    _prepare_local_graph_output_dir,
    _write_spatial_l3_protocol,
)
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.spatial_l3_config import SpatialL3Config


def test_spatial_marker_records_mode_and_geometry_without_replacing_local_marker(tmp_path) -> None:
    output = tmp_path / "new_run"
    _prepare_local_graph_output_dir(str(output), LocalGraphConfig(enabled=True))
    local_before = (output / "local_graph_protocol.json").read_bytes()
    config = SpatialL3Config(mode="shadow", seg_model_path="/local/yolo11n-seg.pt")

    _write_spatial_l3_protocol(str(output), config)

    marker = json.loads((output / "conceptgraphs_l3_protocol.json").read_text())
    assert marker["protocol_version"] == "conceptgraphs_lite_l3_v1_1"
    assert marker["spatial_l3"]["mode"] == "shadow"
    assert marker["spatial_l3"]["geometry"]["max_center_distance_m"] == 0.75
    assert (output / "local_graph_protocol.json").read_bytes() == local_before
    with pytest.raises(FileExistsError):
        _write_spatial_l3_protocol(str(output), config)
    with pytest.raises(ValueError):
        _guard_non_v3_v4_output_dir(str(output))


def test_off_mode_does_not_write_spatial_marker(tmp_path) -> None:
    output = tmp_path / "baseline"
    _prepare_local_graph_output_dir(str(output), LocalGraphConfig(enabled=True))
    _write_spatial_l3_protocol(str(output), SpatialL3Config())
    assert not (output / "conceptgraphs_l3_protocol.json").exists()


def test_cli_keeps_l1_and_l3_model_paths_independent() -> None:
    args = build_parser().parse_args([
        "--model_path", "/model", "--output_dir", "/tmp/spatial_new",
        "--memory_enabled", "--memory_strategy", "hierarchical",
        "--local_graph_enabled", "--memory_extractor", "yolo11",
        "--memory_extractor_model_path", "/weights/yolo11n.pt",
        "--spatial_l3_mode", "shadow",
        "--spatial_l3_seg_model_path", "/weights/yolo11n-seg.pt",
        "--spatial_l3_max_center_distance_m", "0.5",
    ])
    config = config_from_args(args).memory
    assert config.extractor_model_path == "/weights/yolo11n.pt"
    assert config.spatial_l3.seg_model_path == "/weights/yolo11n-seg.pt"
    assert config.spatial_l3.geometry.max_center_distance_m == 0.5
    with pytest.raises(ValueError, match="independent model paths"):
        MemoryConfig(
            enabled=True, strategy="hierarchical", extractor="yolo11",
            extractor_model_path="/weights/same.pt",
            local_graph=LocalGraphConfig(enabled=True),
            spatial_l3=SpatialL3Config(mode="inject", seg_model_path="/weights/same.pt"),
        )


def _on_demand_memory_config(prompt_mode: str) -> MemoryConfig:
    return MemoryConfig(
        enabled=True,
        strategy="hierarchical",
        layers=("short", "mid", "long"),
        extractor="yolo11",
        extractor_model_path="/tmp/yolo11n.pt",
        local_graph=LocalGraphConfig(enabled=True),
        spatial_l3=SpatialL3Config(
            mode="on_demand",
            seg_model_path="/tmp/yolo11n-seg.pt",
            clip_model_path="/tmp/open_clip_vit_h14.bin",
        ),
        prompt_mode=prompt_mode,
    )


@pytest.mark.parametrize("prompt_mode", ["legacy", "l1_original_only"])
def test_on_demand_accepts_legacy_and_l1_original_only(prompt_mode) -> None:
    config = _on_demand_memory_config(prompt_mode)
    assert config.spatial_l3.mode == "on_demand"
    assert config.prompt_mode == prompt_mode


def test_on_demand_rejects_other_prompt_ablation_modes() -> None:
    with pytest.raises(ValueError, match="l1_original_only"):
        _on_demand_memory_config("l1_organized_only")


def test_cli_parses_on_demand_with_l1_original_only() -> None:
    args = build_parser().parse_args([
        "--model_path", "/model", "--output_dir", "/tmp/on_demand_new",
        "--memory_enabled", "--memory_strategy", "hierarchical",
        "--local_graph_enabled", "--memory_extractor", "yolo11",
        "--memory_extractor_model_path", "/weights/yolo11n.pt",
        "--memory_prompt_mode", "l1_original_only",
        "--spatial_l3_mode", "on_demand",
        "--spatial_l3_seg_model_path", "/weights/yolo11n-seg.pt",
        "--spatial_l3_clip_model_path", "/weights/open_clip.bin",
        "--spatial_l3_clip_device", "cuda:0",
        "--spatial_l3_clip_precision", "fp16",
        "--spatial_l3_clip_batch_size", "4",
        "--spatial_l3_clip_top_k", "2",
    ])
    memory = config_from_args(args).memory
    assert memory.prompt_mode == "l1_original_only"
    assert memory.spatial_l3.mode == "on_demand"
    assert memory.spatial_l3.clip_precision == "fp16"
    assert memory.spatial_l3.clip_batch_size == 4
    assert memory.spatial_l3.clip_top_k == 2


def test_effective_spatial_config_keeps_on_demand_for_l1_original_only() -> None:
    from lightnav.habitat.runner import effective_spatial_l3_config

    config = _on_demand_memory_config("l1_original_only")
    assert effective_spatial_l3_config(config).mode == "on_demand"
    other = MemoryConfig(
        enabled=True, strategy="hierarchical",
        layers=("short", "mid", "long"),
        extractor="yolo11", extractor_model_path="/tmp/yolo11n.pt",
        local_graph=LocalGraphConfig(enabled=True),
        spatial_l3=SpatialL3Config(mode="inject", seg_model_path="/tmp/seg.pt"),
        prompt_mode="l1_organized_only",
    )
    assert effective_spatial_l3_config(other).mode == "off"
