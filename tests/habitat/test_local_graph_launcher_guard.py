"""Independent local graph switch and output isolation, without a model load."""

from __future__ import annotations

import json

import pytest

from lightnav.cli.eval_habitat import build_parser, config_from_args
from lightnav.habitat.runner import (
    MemoryConfig,
    _guard_non_v3_v4_output_dir,
    _prepare_local_graph_output_dir,
)
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.v4_config import RGMV4Config


def test_local_graph_cli_is_independent_of_fix3() -> None:
    args = build_parser().parse_args([
        "--model_path", "/not/loaded", "--output_dir", "/tmp/local_graph_new",
        "--memory_enabled", "--memory_strategy", "hierarchical",
        "--local_graph_enabled", "--local_graph_prompt_max_chars", "280",
    ])
    config = config_from_args(args)
    assert config.memory.local_graph.enabled
    assert config.memory.local_graph.prompt_max_chars == 280
    assert config.memory.local_graph.fusion_mode == "legacy"
    assert not config.memory.rgm_v4.enabled


def test_fix3_and_local_graph_cannot_share_a_run() -> None:
    with pytest.raises(ValueError, match="cannot run together"):
        MemoryConfig(
            enabled=True, strategy="hierarchical",
            rgm_v4=RGMV4Config(enabled=True),
            local_graph=LocalGraphConfig(enabled=True),
        )


def test_local_graph_protocol_requires_fresh_directory(tmp_path) -> None:
    fresh = tmp_path / "local"
    config = LocalGraphConfig(enabled=True)
    _prepare_local_graph_output_dir(str(fresh), config)
    marker = json.loads((fresh / "local_graph_protocol.json").read_text())
    assert marker["protocol_version"] == "local_graph_v1"
    assert marker["fusion_mode"] == "legacy"
    with pytest.raises(FileExistsError):
        _prepare_local_graph_output_dir(str(fresh), config)
    with pytest.raises(ValueError):
        _guard_non_v3_v4_output_dir(str(fresh))
    occupied = tmp_path / "old"
    occupied.mkdir()
    (occupied / "results.jsonl").write_text("preserve")
    with pytest.raises(FileExistsError):
        _prepare_local_graph_output_dir(str(occupied), config)
    assert (occupied / "results.jsonl").read_text() == "preserve"


def test_spatial_cli_protocol_and_legacy_result_are_isolated(tmp_path) -> None:
    args = build_parser().parse_args([
        "--model_path", "/not/loaded", "--output_dir", str(tmp_path / "spatial"),
        "--memory_enabled", "--memory_strategy", "hierarchical",
        "--local_graph_enabled", "--local_graph_fusion_mode", "spatial",
        "--local_graph_neighbor_radius_m", "4.5",
    ])
    config = config_from_args(args).memory.local_graph
    assert config.fusion_mode == "spatial"
    assert config.protocol_version == "local_graph_spatial_v1"
    assert config.neighbor_radius_m == 4.5
    with pytest.raises(ValueError, match="legacy 88-Episode result path"):
        _prepare_local_graph_output_dir(
            str(tmp_path / "local_graph_fusion_88_run2" / "shard_0"), config,
        )
    _prepare_local_graph_output_dir(str(tmp_path / "spatial"), config)
    marker = json.loads((tmp_path / "spatial" / "local_graph_protocol.json").read_text())
    assert marker["fusion_mode"] == "spatial"
    assert marker["protocol_version"] == "local_graph_spatial_v1"
    assert marker["neighbor_radius_m"] == 4.5
