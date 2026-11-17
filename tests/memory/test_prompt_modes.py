"""Prompt-mode plans, budgets and backend validation."""

from __future__ import annotations

import pytest

from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.prompt_modes import (
    ABLATION_PROMPT_MODES,
    MEMORY_PROMPT_MODES,
    effective_layer_activation,
    mode_plan,
    resolve_prompt_budget,
    validate_mode_backing,
)
from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.spatial_l3_config import SpatialL3Config


def test_all_seven_modes_are_explicit_and_six_are_ablated() -> None:
    assert MEMORY_PROMPT_MODES == (
        "legacy", "l1_original_only", "l1_organized_only", "l3_original_only",
        "l3_organized_only", "l1_l3_original_concat", "l1_l3_joint",
    )
    assert ABLATION_PROMPT_MODES == MEMORY_PROMPT_MODES[1:]
    plans = {mode: mode_plan(mode) for mode in MEMORY_PROMPT_MODES}
    assert plans["l1_original_only"].uses_l1 and not plans["l1_original_only"].uses_l3
    assert plans["l1_organized_only"].l1_organized
    assert plans["l3_original_only"].uses_l3 and not plans["l3_original_only"].uses_l1
    assert plans["l3_organized_only"].l3_organized
    assert plans["l1_l3_original_concat"].concatenated
    assert plans["l1_l3_joint"].joint
    assert plans["l1_l3_joint"].l1_organized and plans["l1_l3_joint"].l3_organized


def test_unknown_or_non_string_mode_is_rejected() -> None:
    for mode in ("", "l2_only", None, 3):
        with pytest.raises(ValueError):
            mode_plan(mode)


def test_prompt_budget_is_the_original_minimum_not_a_new_allowance() -> None:
    assert resolve_prompt_budget(
        max_context_chars=600, local_graph_prompt_max_chars=320,
    ) == 320
    assert resolve_prompt_budget(
        max_context_chars=200, local_graph_prompt_max_chars=320,
    ) == 200
    with pytest.raises(ValueError):
        resolve_prompt_budget(max_context_chars=0, local_graph_prompt_max_chars=320)


def test_mode_backing_requires_the_selected_layer_sources() -> None:
    validate_mode_backing(
        mode_plan("l1_organized_only"),
        local_graph_enabled=True, spatial_l3_mode="off",
    )
    with pytest.raises(ValueError):
        validate_mode_backing(
            mode_plan("l1_organized_only"),
            local_graph_enabled=False, spatial_l3_mode="inject",
        )
    with pytest.raises(ValueError):
        validate_mode_backing(
            mode_plan("l3_organized_only"),
            local_graph_enabled=True, spatial_l3_mode="shadow",
        )
    validate_mode_backing(
        mode_plan("l3_organized_only"),
        local_graph_enabled=True, spatial_l3_mode="inject",
    )
    # Legacy keeps whatever backend the original run configured.
    validate_mode_backing(
        mode_plan("legacy"), local_graph_enabled=False, spatial_l3_mode="off",
    )


def test_layer_activation_is_explicit_for_every_mode() -> None:
    l1_only = effective_layer_activation(mode_plan("l1_organized_only"))
    assert l1_only["l1_prompt_enabled"] is True
    assert l1_only["l3_prompt_enabled"] is False
    assert l1_only["joint_alignment_enabled"] is False
    joint = effective_layer_activation(mode_plan("l1_l3_joint"))
    assert joint["l1_prompt_enabled"] and joint["l3_prompt_enabled"]
    assert joint["joint_alignment_enabled"] is True


def test_manager_rejects_modes_without_their_backend() -> None:
    spatial = SpatialL3Config(mode="inject", seg_model_path="/fake/seg.pt")
    with pytest.raises(ValueError):
        MemoryManager(
            strategy="hierarchical", local_graph_config=LocalGraphConfig(enabled=False),
            prompt_mode="l1_organized_only",
        )
    with pytest.raises(ValueError):
        MemoryManager(
            strategy="hierarchical", local_graph_config=LocalGraphConfig(enabled=True),
            spatial_l3_config=SpatialL3Config(mode="off"),
            prompt_mode="l3_organized_only",
        )
    with pytest.raises(ValueError):
        MemoryManager(
            strategy="hierarchical", local_graph_config=LocalGraphConfig(enabled=True),
            spatial_l3_config=spatial, spatial_extractor=object(),
            rgm_config=RGMConfig(enabled=True), prompt_mode="l1_l3_joint",
        )
    manager = MemoryManager(
        strategy="hierarchical", local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=spatial, spatial_extractor=object(),
        prompt_mode="l1_l3_joint",
    )
    assert manager.prompt_budget_chars == 320
    assert manager.plan.mode == "l1_l3_joint"
