"""Prompt protocol/diagnostics artifacts and their runner integration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from lightnav.habitat.prompt_artifacts import (
    build_prompt_protocol,
    merge_prompt_artifacts,
    prompt_diagnostic_row,
    validate_group_prompt_artifacts,
    write_or_match_prompt_protocol,
    write_per_episode_csv,
)
from lightnav.habitat.runner import HabitatEvalConfig, MemoryConfig, run_habitat_eval
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.prompt_modes import ABLATION_PROMPT_MODES
from lightnav.memory.spatial_l3_config import SpatialL3Config


def _memory(mode: str) -> MemoryConfig:
    return MemoryConfig(
        enabled=True, extractor="yolo11", extractor_model_path="/tmp/yolo11n.pt",
        strategy="hierarchical", layers=("short", "mid", "long"),
        local_graph=LocalGraphConfig(enabled=True),
        spatial_l3=SpatialL3Config(mode="inject", seg_model_path="/tmp/seg.pt"),
        prompt_mode=mode,
    )


def test_protocol_is_immutable_and_shared_budget(tmp_path: Path) -> None:
    budgets = set()
    for mode in (*ABLATION_PROMPT_MODES, "legacy"):
        protocol = build_prompt_protocol(_memory(mode))
        assert protocol["mode"] == mode
        if mode == "legacy":
            assert protocol["budget"]["strategy"] == "original_char_budget"
            assert protocol["budget"]["chars"] == 320
        else:
            assert protocol["budget"]["strategy"] == "token_aware_context"
            assert protocol["budget"]["token_aware"] is True
            assert "chars" not in protocol["budget"]
        assert protocol["budget"]["original_char_budget_chars"] == 320
        budgets.add(protocol["budget"]["original_char_budget_chars"])
        path = tmp_path / mode
        path.mkdir()
        write_or_match_prompt_protocol(path, protocol)
        assert write_or_match_prompt_protocol(path, protocol) == path / "prompt_protocol.json"
        with pytest.raises(ValueError):
            write_or_match_prompt_protocol(
                path,
                {**protocol, "mode": "legacy" if mode != "legacy" else "l1_joint"},
            )
    assert budgets == {320}


def test_prompt_diagnostic_row_records_layers_alignment_and_conflicts() -> None:
    diagnostic = {
        "step": 4,
        "prompt_mode": "l1_l3_joint",
        "prompt_protocol_version": "l1_l3_prompt_ablation_v1",
        "prompt_chars": 500,
        "memory_backend": "l1_l3_prompt:l1_l3_joint",
        "actual_injection": True,
        "actual_historical_injection": True,
        "injected_evidence_ids": ["e1"],
        "injected_historical_evidence_ids": ["e1"],
        "processor_length_rejections": 1,
        "prompt_fallback_reason": None,
        "prompt_hash_before": "a",
        "prompt_hash_after": "b",
        "prompt_context": {
            "source": "joint",
            "budget_chars": 320,
            "context_chars": 300,
            "item_count": 1,
            "injected_memory_chars": 280,
            "final_token_count": 700,
            "injected_memory_tokens": 20,
            "layers": {
                "l1": {"item_count": 1, "fact_count": 2, "current_count": 1,
                       "history_count": 1, "duplicates_removed": 1, "omitted_items": 0},
                "l3": {"candidate_count": 2, "item_count": 1, "unverified_count": 1,
                       "omitted_items": 1},
                "joint": {"merged_entities": 1, "omitted_items": 0},
            },
            "layer_item_counts": {"l1": 0, "l3": 0, "joint": 1},
            "alignment": {"attempted": 1, "confirmed": 1, "uncertain": 0,
                          "unmatched": 0, "success_rate": 1.0},
            "conflicts": [{"type": "geometry_unverified"}],
            "duplicates_removed": 1,
            "unknown_relation_count": 1,
            "omitted_items": 1,
            "memory_tokens_injected": 18,
            "budget": {
                "budget_source": "token_aware_context",
                "context_limit_tokens": 8192,
                "base_prompt_tokens": 7000,
                "generation_reserve_tokens": 64,
                "safety_margin_tokens": 1,
                "available_memory_tokens": 1127,
                "memory_block_tokens": 18,
                "token_overflow_drops": 2,
                "token_budget_unavailable_reason": None,
            },
            "roundtrip_ok": True,
        },
    }
    row = prompt_diagnostic_row(
        diagnostic, episode_id="episode_000", habitat_episode_id="ep1", scene_id="scene",
    )
    assert row["mode"] == "l1_l3_joint"
    assert row["l1_fact_count"] == 2
    assert row["l3_candidate_count"] == 2
    assert row["joint_item_count"] == 1
    assert row["l1_item_count"] == 1  # the merged item carries both layers
    assert row["l3_item_count"] == 1
    assert row["merged_entities"] == 1
    assert row["alignment_confirmed"] == 1
    assert row["conflict_count"] == 1
    assert row["prompt_tokens"] == 700
    assert row["processor_length_rejections"] == 1
    assert row["omitted_items"] == 1
    assert row["budget_source"] == "token_aware_context"
    assert row["context_limit_tokens"] == 8192
    assert row["base_prompt_tokens"] == 7000
    assert row["available_memory_tokens"] == 1127
    assert row["memory_block_tokens"] == 18
    assert row["token_overflow_drops"] == 2
    assert row["memory_tokens_injected"] == 18


def test_per_episode_csv_and_merge_of_two_shards(tmp_path: Path) -> None:
    shards = []
    for index in range(2):
        shard = tmp_path / f"shard_{index}"
        shard.mkdir()
        protocol = build_prompt_protocol(_memory("l1_organized_only"))
        write_or_match_prompt_protocol(shard, protocol)
        row = {
            "mode": "l1_organized_only", "episode_id": f"episode_{index:03d}",
            "habitat_episode_id": f"ep{index}", "scene_id": f"scene{index}", "step": 0,
            "prompt_chars": 400,
        }
        (shard / "prompt_diagnostics.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8",
        )
        shards.append(shard)
    root = tmp_path / "root"
    root.mkdir()
    (root / "results.jsonl").write_text(
        json.dumps({
            "episode_id": "episode_000", "habitat_episode_id": "ep0",
            "scene_id": "scene0", "success": 1, "oracle_success": 1, "spl": 1.0,
            "ndtw": 0.8, "final_distance": 0.5, "min_distance": 0.4, "steps": 3,
            "path_length": 4.2, "termination_reason": "agent_stop", "forced_stop": False,
        }) + "\n",
        encoding="utf-8",
    )
    summary = merge_prompt_artifacts(root, shards)
    assert summary == {"mode": "l1_organized_only", "rows": 2, "episodes": 1}
    assert len((root / "prompt_diagnostics.jsonl").read_text().splitlines()) == 2
    with (root / "per_episode.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["episode_id"] == "episode_000"
    assert rows[0]["path_length"] == "4.2"


def test_validate_group_artifacts_requires_mode_and_rows(tmp_path: Path) -> None:
    group = tmp_path / "group"
    group.mkdir()
    protocol = build_prompt_protocol(_memory("l3_organized_only"))
    write_or_match_prompt_protocol(group, protocol)
    (group / "prompt_diagnostics.jsonl").write_text(
        json.dumps({"mode": "l3_organized_only", "step": 0}) + "\n", encoding="utf-8",
    )
    write_per_episode_csv(group, [])
    summary = validate_group_prompt_artifacts(group, expected_mode="l3_organized_only")
    assert summary["prompt_diagnostic_rows"] == 1
    with pytest.raises(ValueError):
        validate_group_prompt_artifacts(group, expected_mode="l1_organized_only")
    (group / "prompt_diagnostics.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_group_prompt_artifacts(group, expected_mode="l3_organized_only")


# ---------------------------------------------------------------------------
# Runner integration with an injected policy (no GPU / no Habitat)
# ---------------------------------------------------------------------------


class _FakeEnv:
    def __init__(self) -> None:
        self.step_index = 0
        self.closed = False

    def reset(self):
        self.step_index = 0
        return (
            {"rgb": np.zeros((4, 4, 3), dtype=np.uint8),
             "instruction": {"text": "find the chair"}},
            {"episode_id": "ep1", "scene_id": "sceneA", "distance_to_goal": 3.0,
             "habitat_time_step": 1.0, "lin_vel_range": [0.0, 0.25],
             "ang_vel_range": [-30.0, 30.0]},
        )

    def step(self, action):
        self.step_index += 1
        terminated = self.step_index >= 2
        info = {
            "episode_id": "ep1", "scene_id": "sceneA", "distance_to_goal": 0.5,
            "success": 1.0, "oracle_success": 1.0, "spl": 1.0, "ndtw": 0.9,
            "path_length": 3.5, "termination_reason": "agent_stop",
        }
        if terminated:
            info["termination_details"] = {}
        return {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}, 0.0, terminated, False, info

    def close(self) -> None:
        self.closed = True


class _FakePolicy:
    memory_enabled = True

    def __init__(self) -> None:
        self.last_memory_diagnostic = None
        self.last_prompt_chars = 500
        self.last_prompt_text = "prompt"
        self.last_memory_text = ""
        self.last_model_inference_ms = 1.0
        self.steps = 0

    def reset(self, obs) -> None:
        self.steps = 0

    def act(self, obs, info) -> dict:
        self.steps += 1
        self.last_memory_diagnostic = {
            "step": self.steps - 1,
            "prompt_mode": "l1_organized_only",
            "prompt_protocol_version": "l1_l3_prompt_ablation_v1",
            "prompt_chars": 500,
            "actual_injection": True,
            "injected_evidence_ids": [],
            "prompt_context": {
                "budget_chars": 320, "context_chars": 200, "item_count": 1,
                "injected_memory_chars": 200, "final_token_count": 600,
                "layers": {"l1": {
                    "item_count": 1, "fact_count": 1, "current_count": 1,
                    "history_count": 0, "duplicates_removed": 0, "omitted_items": 0,
                }},
                "duplicates_removed": 0, "omitted_items": 0,
                "processor_length_rejections": 0, "prompt_roundtrip_ok": True,
            },
        }
        return {"action": "velocity_control", "action_args": {}}

    def get_info(self) -> dict:
        return {}

    def close_episode(self) -> None:
        pass


def test_runner_writes_prompt_artifacts_and_per_episode_csv(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = HabitatEvalConfig(
        model_path="/tmp/model", episodes=1, max_steps=2, output_dir=str(output),
        memory=_memory("l1_organized_only"),
    )
    results = run_habitat_eval(
        config,
        env_factory=lambda cfg: _FakeEnv(),
        policy_factory=lambda cfg, engine, bundle, info: _FakePolicy(),
        engine_factory=lambda cfg: (object(), object()),
    )
    assert results[0]["path_length"] == 3.5
    assert (output / "prompt_protocol.json").is_file()
    diagnostics = (output / "prompt_diagnostics.jsonl").read_text().splitlines()
    assert len(diagnostics) == 2
    with (output / "per_episode.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["episode_id"] == "episode_000"
    assert rows[0]["path_length"] == "3.5"
    protocol = json.loads((output / "prompt_protocol.json").read_text())
    assert protocol["mode"] == "l1_organized_only"


def test_legacy_prompt_row_is_measurement_only_without_layer_claims() -> None:
    from lightnav.habitat.prompt_artifacts import legacy_prompt_diagnostic_row

    row = legacy_prompt_diagnostic_row(
        {
            "step": 3,
            "prompt_chars": 500,
            "memory_backend": "local_graph",
            "actual_injection": True,
            "injected_evidence_ids": ["ev1"],
            "processor_length_rejections": 0,
            "memory_text": "[Navigation Memory]\nCurrent RGB: chair",
        },
        engine_diagnostic={"final_token_count": 640},
        episode_id="episode_000", habitat_episode_id="ep1", scene_id="scene",
        char_budget=320,
    )
    assert row["mode"] == "legacy"
    assert row["budget_source"] == "original_char_budget"
    assert row["budget_chars"] == 320
    assert row["prompt_tokens"] == 640
    assert row["injected_memory_chars"] > 0
    for absent in ("l1_item_count", "l3_candidate_count", "alignment"):
        assert absent not in row


def test_runner_writes_measurement_artifacts_for_legacy_without_changing_path(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    output = tmp_path / "legacy_run"
    memory = replace(_memory("legacy"), prompt_measurement=True)
    config = HabitatEvalConfig(
        model_path="/tmp/model", episodes=1, max_steps=2, output_dir=str(output),
        memory=memory,
    )
    results = run_habitat_eval(
        config,
        env_factory=lambda cfg: _FakeEnv(),
        policy_factory=lambda cfg, engine, bundle, info: _FakePolicy(),
        engine_factory=lambda cfg: (object(), object()),
    )
    assert len(results) == 1
    protocol = json.loads((output / "prompt_protocol.json").read_text())
    assert protocol["mode"] == "legacy"
    assert protocol["budget"]["strategy"] == "original_char_budget"
    rows = [
        json.loads(line)
        for line in (output / "prompt_diagnostics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows and all(row["mode"] == "legacy" for row in rows)
    assert all(row["budget_source"] == "original_char_budget" for row in rows)
    assert (output / "per_episode.csv").is_file()


def test_official_builder_is_declared_only_for_the_official_mode() -> None:
    organized = build_prompt_protocol(_memory("l3_organized_only"))
    assert "l3_official" not in organized["builders"]
    assert organized["layer_activation"]["l3_layer"] == "organized"

    official = build_prompt_protocol(_memory("l3_official_style_only"))
    assert official["builders"]["l3_official"] == (
        "lightnav.memory.l3_official_prompt.L3OfficialPromptOrganizer"
    )
    assert official["layer_activation"]["l3_layer"] == "official"
    assert official["layer_activation"]["l1_prompt_enabled"] is False


def test_diagnostic_row_records_the_official_expression() -> None:
    expression = {
        "expression_version": "l3_official_style_v1",
        "strategy_counts": {"relational_anchor": 1},
        "anchors_used": [{"label": "dining table", "relation": "next to"}],
    }
    row = prompt_diagnostic_row(
        {
            "step": 3,
            "prompt_mode": "l3_official_style_only",
            "prompt_context": {
                "layers": {"l3": {"expression": expression}},
                "source": "l3_official",
            },
        },
        episode_id="episode", habitat_episode_id="habitat", scene_id="scene",
    )
    assert row["l3_expression"] == expression
    assert row["source"] == "l3_official"
