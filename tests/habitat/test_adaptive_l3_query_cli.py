"""Configuration contracts for the opt-in Adaptive L3 Query arm."""

from __future__ import annotations

import pytest

from lightnav.cli.eval_habitat import build_parser
from lightnav.habitat.prompt_artifacts import build_prompt_protocol
from lightnav.habitat.runner import MemoryConfig
from lightnav.memory.adaptive_l3_query_config import AdaptiveL3QueryConfig
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
            mode="shadow", seg_model_path="/fake/seg.pt",
            clip_model_path="/fake/clip.pt",
        ),
        "prompt_mode": "l1_original_only",
        "adaptive_l3_query": AdaptiveL3QueryConfig(enabled=True),
    }
    values.update(overrides)
    return MemoryConfig(**values)


def test_cli_flag_is_off_by_default_and_parses_explicit_enable() -> None:
    parser = build_parser()
    default = parser.parse_args(["--model_path", "/fake/model"])
    assert default.adaptive_l3_query is False
    parsed = parser.parse_args([
        "--model_path", "/fake/model",
        "--adaptive_l3_query",
        "--adaptive_l3_query_max_new_tokens", "12",
        "--spatial_l3_clip_model_path", "/fake/clip.pt",
    ])
    assert parsed.adaptive_l3_query is True
    assert parsed.adaptive_l3_query_max_new_tokens == 12


def test_config_accepts_the_shadow_l1_original_arm() -> None:
    config = _config()
    assert config.adaptive_l3_query.enabled is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enabled": False}, "memory_enabled"),
        ({"local_graph": LocalGraphConfig(enabled=False)}, "local graph"),
        (
            {"spatial_l3": SpatialL3Config(
                mode="inject", seg_model_path="/fake/seg.pt",
                clip_model_path="/fake/clip.pt",
            )},
            "shadow",
        ),
        (
            {"spatial_l3": SpatialL3Config(
                mode="shadow", seg_model_path="/fake/seg.pt",
            )},
            "clip_model_path",
        ),
        ({"prompt_mode": "l1_l3_joint"}, "prompt mode"),
    ],
)
def test_config_rejects_invalid_adaptive_combinations(overrides, message) -> None:
    if overrides.get("prompt_mode") == "l1_l3_joint":
        overrides = {
            **overrides,
            "spatial_l3": SpatialL3Config(
                mode="inject", seg_model_path="/fake/seg.pt",
                clip_model_path="/fake/clip.pt",
            ),
        }
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_prompt_protocol_records_the_adaptive_contract_only_when_enabled() -> None:
    disabled = _config(adaptive_l3_query=AdaptiveL3QueryConfig(enabled=False))
    assert "adaptive_l3_query" not in build_prompt_protocol(disabled)
    enabled = build_prompt_protocol(_config())
    assert enabled["adaptive_l3_query"]["enabled"] is True
    assert enabled["layer_activation"]["adaptive_l3"] == (
        "controller_gated_verified_facts"
    )
    assert enabled["budget"]["adaptive_l3"]["query"].startswith("existing L3ClipQuery")
