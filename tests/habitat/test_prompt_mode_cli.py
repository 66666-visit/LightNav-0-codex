"""CLI and config mapping for --memory_prompt_mode."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lightnav.cli.eval_habitat import build_parser, config_from_args
from lightnav.habitat.runner import HabitatEvalConfig, MemoryConfig
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.prompt_modes import MEMORY_PROMPT_MODES
from lightnav.memory.spatial_l3_config import SpatialL3Config


def _parse(argv: list[str]):
    return build_parser().parse_args(argv)


def test_default_mode_is_legacy_and_maps_to_runner_config() -> None:
    args = _parse(["--model_path", "/tmp/model"])
    assert args.memory_prompt_mode == "legacy"
    config = config_from_args(args)
    assert config.memory.prompt_mode == "legacy"
    assert config.memory.local_graph.enabled is False


def test_every_mode_is_accepted_and_mapped() -> None:
    for mode in MEMORY_PROMPT_MODES:
        args = _parse([
            "--model_path", "/tmp/model", "--output_dir", "/tmp/out",
            "--memory_enabled", "--memory_extractor", "yolo11",
            "--memory_extractor_model_path", "/tmp/yolo11n.pt",
            "--memory_strategy", "hierarchical",
            "--memory_layers", "short,mid,long", "--local_graph_enabled",
            "--spatial_l3_mode", "inject",
            "--spatial_l3_seg_model_path", "/tmp/yolo11n-seg.pt",
            "--memory_prompt_mode", mode,
        ])
        config = config_from_args(args)
        assert config.memory.prompt_mode == mode


def test_unknown_mode_is_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        _parse(["--model_path", "/tmp/model", "--memory_prompt_mode", "l1_only"])


def test_mode_requires_explicit_output_directory() -> None:
    from lightnav.cli.eval_habitat import main

    with pytest.raises(SystemExit):
        main([
            "--model_path", "/tmp/model", "--memory_enabled",
            "--memory_extractor", "yolo11",
            "--memory_extractor_model_path", "/tmp/yolo11n.pt",
            "--memory_strategy", "hierarchical", "--memory_layers", "short,mid,long",
            "--local_graph_enabled", "--spatial_l3_mode", "inject",
            "--spatial_l3_seg_model_path", "/tmp/seg.pt",
            "--memory_prompt_mode", "l1_l3_joint",
        ])


def test_memory_config_rejects_modes_with_missing_backends() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(prompt_mode="l1_l3_joint")
    with pytest.raises(ValueError):
        MemoryConfig(
            enabled=True, local_graph=LocalGraphConfig(enabled=True),
            prompt_mode="l3_organized_only",
        )
    with pytest.raises(ValueError):
        MemoryConfig(
            enabled=True, local_graph=LocalGraphConfig(enabled=True),
            prompt_mode="l3_organized_only",
        )


def test_memory_config_accepts_joint_mode_with_both_backends() -> None:
    config = MemoryConfig(
        enabled=True, extractor="yolo11", extractor_model_path="/tmp/yolo11n.pt",
        strategy="hierarchical", layers=("short", "mid", "long"),
        local_graph=LocalGraphConfig(enabled=True),
        spatial_l3=SpatialL3Config(mode="inject", seg_model_path="/tmp/seg.pt"),
        prompt_mode="l1_l3_joint",
    )
    assert config.prompt_mode == "l1_l3_joint"


def test_habitat_config_rejects_experimental_mode_without_output_dir() -> None:
    with pytest.raises(ValueError):
        HabitatEvalConfig(
            model_path="/tmp/model",
            memory=MemoryConfig(
                enabled=True, extractor="yolo11",
                extractor_model_path="/tmp/yolo11n.pt",
                local_graph=LocalGraphConfig(enabled=True),
                spatial_l3=SpatialL3Config(mode="inject", seg_model_path="/tmp/seg.pt"),
                prompt_mode="l1_l3_joint",
            ),
        )


# ---------------------------------------------------------------------------
# scripts/eval_habitat.sh: CLIENT_ARGS -> config_snapshot.yaml mapping
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
_EVAL_LAUNCHER = _REPO / "scripts" / "eval_habitat.sh"
_PROMPT_ARGS = (
    "--memory_enabled --memory_extractor yolo11 "
    "--memory_extractor_model_path /tmp/yolo11n.pt "
    "--memory_strategy hierarchical --memory_layers short,mid,long "
    "--local_graph_enabled --local_graph_prompt_max_chars 320 "
    "--memory_max_context_chars 600 "
    "--spatial_l3_mode inject --spatial_l3_seg_model_path /tmp/seg.pt "
    "--memory_prompt_mode l1_l3_joint"
)


def _launch_eval_snapshot(tmp_path: Path, args: str, **overrides: str):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = '-c' ]; then exit 0; fi\n"
        f"if [ \"$1\" = '-' ]; then exec '{sys.executable}' \"$@\"; fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = dict(os.environ)
    env.update({
        "MODEL_PATH": str(model), "OUTPUT_ROOT": str(tmp_path / "out"),
        "CLIENT_ARGS": args, "CLIENT_PYTHON": str(fake_python),
        "HABITAT_PYTHON": str(fake_python), "GPU_IDS": "0",
        "READY_TIMEOUT_S": "0", "STRICT_FULL_EVAL": "0",
        "PILOT_MANIFEST_PATH": "", "EPISODE_MANIFEST_PATH": "", "RESUME": "0",
    })
    env.update(overrides)
    return subprocess.run(
        ["bash", str(_EVAL_LAUNCHER)], cwd=_REPO, env=env,
        text=True, capture_output=True, check=False, timeout=60,
    )


def test_eval_launcher_snapshot_records_prompt_mode_and_shared_budget(tmp_path: Path) -> None:
    result = _launch_eval_snapshot(tmp_path, _PROMPT_ARGS)
    assert result.returncode != 0  # fake servers never become ready
    snapshot = json.loads((tmp_path / "out" / "config_snapshot.yaml").read_text())
    effective = snapshot["prompt_effective"]
    assert effective["prompt_mode"] == "l1_l3_joint"
    assert effective["budget_strategy"] == "token_aware_context"
    assert effective["original_char_budget_chars"] == 320
    assert effective["layer_activation"]["joint_alignment_enabled"] is True
    assert snapshot["protocol_version"] == "local_graph_v1"
    assert snapshot["prompt_protocol_version"] == "l1_l3_prompt_ablation_v1"
    assert snapshot["memory_effective"]["prompt_mode"] == "l1_l3_joint"
    assert "--memory_prompt_mode" in snapshot["memory_args"]


def test_eval_launcher_l1_only_snapshot_reports_l3_prompt_disabled(tmp_path: Path) -> None:
    args = _PROMPT_ARGS.replace("l1_l3_joint", "l1_organized_only")
    result = _launch_eval_snapshot(tmp_path, args)
    assert result.returncode != 0
    snapshot = json.loads((tmp_path / "out" / "config_snapshot.yaml").read_text())
    activation = snapshot["prompt_effective"]["layer_activation"]
    assert activation["l1_prompt_enabled"] is True
    assert activation["l3_prompt_enabled"] is False
    # The requested spatial config remains visible, but the effective layer is off.
    assert snapshot["prompt_effective"]["prompt_mode"] == "l1_organized_only"


def test_eval_launcher_rejects_resume_for_prompt_modes(tmp_path: Path) -> None:
    result = _launch_eval_snapshot(tmp_path, _PROMPT_ARGS, RESUME="1")
    assert result.returncode != 0
    assert "cannot resume" in result.stderr


def test_eval_launcher_snapshot_records_legacy_measurement_without_token_budget(
    tmp_path: Path,
) -> None:
    args = _PROMPT_ARGS.replace(
        "--memory_prompt_mode l1_l3_joint",
        "--memory_prompt_mode legacy --prompt_measurement_enabled",
    )
    result = _launch_eval_snapshot(tmp_path, args)
    assert result.returncode != 0
    snapshot = json.loads((tmp_path / "out" / "config_snapshot.yaml").read_text())
    effective = snapshot["prompt_effective"]
    assert effective["prompt_mode"] == "legacy"
    assert effective["budget_strategy"] == "original_char_budget"
    assert effective["original_char_budget_chars"] == 320
    assert effective["layer_activation"]["joint_alignment_enabled"] is False
    assert snapshot["memory_effective"]["prompt_measurement_enabled"] is True
    assert snapshot["protocol_version"] == "local_graph_v1"
    assert snapshot["prompt_protocol_version"] == "l1_l3_prompt_ablation_v1"
    assert "--prompt_measurement_enabled" in snapshot["memory_args"]
