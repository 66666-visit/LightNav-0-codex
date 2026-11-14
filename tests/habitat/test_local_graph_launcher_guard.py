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
