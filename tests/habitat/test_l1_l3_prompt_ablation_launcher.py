"""Launcher preflight and six-group consistency checks without Habitat."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lightnav.habitat.prompt_artifacts import build_prompt_protocol
from lightnav.habitat.runner import MemoryConfig
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.prompt_modes import ABLATION_PROMPT_MODES
from lightnav.memory.spatial_l3_config import SpatialL3Config

_MODES = (*ABLATION_PROMPT_MODES, "legacy")

_REPO = Path(__file__).resolve().parents[2]
_LAUNCHER = _REPO / "scripts" / "run_l1_l3_prompt_ablation.sh"
_SCENES = [f"/data/scene{index:03d}" for index in range(11)]
_GROUPS = ["short"] * 3 + ["medium"] * 3 + ["long"] * 2


def _episodes() -> list[dict[str, str]]:
    return [
        {"scene_id": scene, "episode_id": f"ep{episode:02d}"}
        for scene in _SCENES
        for episode in range(8)
    ]


def _manifest(episodes: list[dict[str, str]] | None = None) -> dict:
    rows = episodes if episodes is not None else _episodes()
    return {"split": "val_unseen", "expected_count": len(rows), "episodes": rows}


def _pilot_manifest() -> dict:
    rows = []
    for scene in _SCENES:
        for episode, group in enumerate(_GROUPS):
            rows.append({
                "scene_id": scene,
                "habitat_episode_id": f"ep{episode:02d}",
                "distance_group": group,
                "parent_manifest_sha256": "a" * 64,
                "sample_seed": 42,
                "dataset_split": "val_unseen",
            })
    return {
        "protocol_version": "rgm_pilot_88_v1",
        "dataset_split": "val_unseen",
        "sample_seed": 42,
        "expected_count": 88,
        "per_scene": 8,
        "parent_manifest_sha256": "a" * 64,
        "episodes": rows,
        "scene_quotas": {scene: 8 for scene in _SCENES},
    }


def _inputs(tmp_path: Path) -> dict[str, str]:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "eval_config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    yolo = tmp_path / "yolo11n.pt"
    yolo.write_bytes(b"l1")
    seg = tmp_path / "yolo11n-seg.pt"
    seg.write_bytes(b"l3")
    habitat_config = tmp_path / "vlnce_r2r.yaml"
    habitat_config.write_text("habitat: {}\n", encoding="utf-8")
    manifest = tmp_path / "episode_manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    pilot = tmp_path / "pilot_manifest.json"
    if not pilot.exists():
        pilot.write_text(json.dumps(_pilot_manifest()), encoding="utf-8")
    search = tmp_path / "history"
    search.mkdir(exist_ok=True)
    return {
        "MODEL_PATH": str(model),
        "YOLO_MODEL_PATH": str(yolo),
        "SEG_MODEL_PATH": str(seg),
        "EPISODE_MANIFEST_PATH": str(manifest),
        "PILOT_MANIFEST_PATH": str(pilot),
        "HABITAT_CONFIG": str(habitat_config),
        "MANIFEST_SEARCH_ROOT": str(search),
        "CLIENT_PYTHON": sys.executable,
        "GPU_IDS": "0",
    }


def _run(tmp_path: Path, env_updates: dict[str, str], *args: str):
    env = dict(os.environ)
    env.update(_inputs(tmp_path))
    env.update(env_updates)
    return subprocess.run(
        ["bash", str(_LAUNCHER), *args], cwd=_REPO, env=env,
        text=True, capture_output=True, check=False, timeout=120,
    )


def test_preflight_writes_provenance_hashes_and_never_creates_a_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    result = _run(tmp_path, {}, "--preflight-only", "--output-root", str(output))
    assert result.returncode == 0, result.stderr
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["episode_manifest_expected_count"] == 88
    assert len(provenance["episode_manifest_sha256"]) == 64
    assert len(provenance["checkpoint_sha256"]) == 64
    assert provenance["seed"] == 42
    assert provenance["legacy_group_required"] is True
    assert provenance["legacy_history_candidates"] == []
    assert "No historical 88-Episode result" in provenance["legacy_reuse_note"]
    assert len(provenance["l1_yolo_model_sha256"]) == 64
    assert len(provenance["l3_seg_model_sha256"]) == 64
    assert (output / "episode_manifest.json").is_file()
    assert (output / "manifest.sha256").read_text().startswith(
        provenance["episode_manifest_sha256"]
    )
    assert (output / "checkpoint.sha256").read_text().startswith(
        provenance["checkpoint_sha256"]
    )
    assert not any((output / mode).exists() for mode in ABLATION_PROMPT_MODES)


def test_wrong_episode_count_stops_before_writing_any_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "episode_manifest.json"
    manifest.write_text(json.dumps(_manifest(_episodes()[:87])), encoding="utf-8")
    output = tmp_path / "out"
    result = _run(
        tmp_path, {"EPISODE_MANIFEST_PATH": str(manifest)},
        "--preflight-only", "--output-root", str(output),
    )
    assert result.returncode != 0
    assert "88" in result.stderr
    assert not (output / "episode_manifest.json").exists()


def test_two_inconsistent_historical_manifests_stop_the_run(tmp_path: Path) -> None:
    history = tmp_path / "history" / "old_run"
    history.mkdir(parents=True, exist_ok=True)
    other = _episodes()
    other[0] = {"scene_id": _SCENES[0], "episode_id": "different_episode"}
    (history / "episode_manifest.json").write_text(
        json.dumps(_manifest(other)), encoding="utf-8",
    )
    output = tmp_path / "out"
    result = _run(tmp_path, {}, "--preflight-only", "--output-root", str(output))
    assert result.returncode != 0
    assert "multiple inconsistent 88-Episode manifests" in result.stderr
    assert not (output / "episode_manifest.json").exists()


def test_dry_run_writes_nothing_and_still_reports_the_six_modes(tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = _run(tmp_path, {}, "--dry-run", "--output-root", str(output))
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    for mode in ABLATION_PROMPT_MODES:
        assert mode in result.stdout


def test_override_environment_is_rejected_before_preflight(tmp_path: Path) -> None:
    for name, value, message in (
        ("SEED", "0", "SEED is fixed at 42"),
        ("CLIENT_ARGS", "--verbose", "CLIENT_ARGS is controlled"),
        ("EPISODES", "1", "EPISODES is controlled"),
    ):
        result = _run(
            tmp_path, {name: value}, "--preflight-only",
            "--output-root", str(tmp_path / f"out_{name}"),
        )
        assert result.returncode != 0
        assert message in result.stderr


def _memory_config(mode: str) -> MemoryConfig:
    return MemoryConfig(
        enabled=True, extractor="yolo11", extractor_model_path="/data/yolo11n.pt",
        strategy="hierarchical", layers=("short", "mid", "long"),
        local_graph=LocalGraphConfig(enabled=True),
        spatial_l3=SpatialL3Config(mode="inject", seg_model_path="/data/seg.pt"),
        prompt_mode=mode,
    )


def _write_provenance(root: Path, manifest: dict) -> None:
    from hashlib import sha256

    digest = sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (root / "provenance.json").write_text(json.dumps({
        "checkpoint_sha256": "e" * 64,
        "checkpoint_config_sha256": "c" * 64,
        "checkpoint_eval_config_sha256": "d" * 64,
        "habitat_config_sha256": "f" * 64,
        "episode_manifest_canonical_sha256": digest,
        "pilot_manifest_sha256": "b" * 64,
        "seed": 42,
        "gpu_ids": ["0"],
    }), encoding="utf-8")


def _group_artifacts(root: Path, mode: str, manifest: dict) -> None:
    from hashlib import sha256

    from lightnav.memory.prompt_modes import (
        PROMPT_PROTOCOL_VERSION,
        effective_layer_activation,
        mode_plan,
    )

    legacy_activation = {
        "mode": "legacy", "l1_layer": "original",
        "l3_layer": "original_spatial_when_inject_else_l1",
        "combination": "legacy", "l1_prompt_enabled": True,
        "l3_prompt_enabled": "original_inject_mode",
        "joint_alignment_enabled": False,
    }
    group = root / mode
    group.mkdir()
    (group / "episode_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    memory = _memory_config(mode)
    protocol = build_prompt_protocol(memory)
    (group / "prompt_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    config = {
        "common": {
            "task": "vlnce", "split": "val_unseen", "expected_episodes": 88,
            "max_steps": 500, "force_stop_at_max_steps": False, "seed": 42,
            "model_weights_sha256": "e" * 64,
            "model_config_sha256": "c" * 64,
            "eval_config_sha256": "d" * 64,
            "habitat_config_sha256": "f" * 64,
            "gpu_ids": ["0"],
        },
        "memory_args": [
            "--memory_enabled", "--memory_extractor", "yolo11",
            "--memory_strategy", "hierarchical", "--memory_layers",
            "short,mid,long", "--memory_top_k", "5",
            "--memory_max_context_chars", "600",
            "--local_graph_enabled", "--local_graph_prompt_max_chars", "320",
            "--spatial_l3_mode", "inject", "--spatial_l3_seg_model_path", "/data/seg.pt",
            "--memory_prompt_mode", mode, "--prompt_measurement_enabled",
        ],
        "memory_effective": {"enabled": True},
        "local_graph_effective": {"enabled": True, "prompt_max_chars": 320},
        "prompt_effective": {
            "prompt_mode": mode,
            "protocol_version": PROMPT_PROTOCOL_VERSION,
            "budget_strategy": (
                "original_char_budget" if mode == "legacy" else "token_aware_context"
            ),
            "original_char_budget_chars": 320,
            "layer_activation": (
                legacy_activation if mode == "legacy"
                else effective_layer_activation(mode_plan(mode))
            ),
        },
        "server_args": [],
        "protocol_version": "local_graph_v1",
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "pilot_manifest_path": "/data/pilot_manifest.json",
        "pilot_manifest_sha256": "b" * 64,
    }
    (group / "config_snapshot.yaml").write_text(json.dumps(config), encoding="utf-8")
    results = []
    prompt_rows = []
    for index, row in enumerate(manifest["episodes"]):
        success = float(index % 2 == 0)
        result = {
            "episode_id": f"episode_{index:03d}",
            "habitat_episode_id": row["episode_id"],
            "scene_id": row["scene_id"],
            "success": success, "oracle_success": success, "spl": success,
            "ndtw": 0.5, "final_distance": 1.0, "min_distance": 0.8,
            "steps": 1, "path_length": 2.0, "forced_stop": False,
            "inference_latency_ms_mean": 12.5,
        }
        results.append(result)
        prompt_rows.append({
            "episode_id": result["episode_id"],
            "habitat_episode_id": row["episode_id"],
            "scene_id": row["scene_id"],
            "step": 0, "mode": mode, "prompt_chars": 300,
            "injected_memory_chars": 200, "prompt_tokens": 100,
            "budget_chars": 320, "actual_injection": True,
            "l1_fact_count": 1, "l1_item_count": 1, "l1_duplicates_removed": 0,
            "l3_candidate_count": 1 if mode_plan(mode).uses_l3 else 0,
            "l3_item_count": 1 if mode_plan(mode).uses_l3 else 0,
            "l3_unverified_count": 0, "duplicates_removed": 0,
            "conflict_count": 0, "alignment_attempted": 1 if mode_plan(mode).joint else 0,
            "alignment_confirmed": 1 if mode_plan(mode).joint else 0,
            "omitted_items": 0, "processor_length_rejections": 0,
        })
    (group / "results.jsonl").write_text(
        "\n".join(json.dumps(item) for item in results) + "\n", encoding="utf-8",
    )
    (group / "prompt_diagnostics.jsonl").write_text(
        "\n".join(json.dumps(item) for item in prompt_rows) + "\n", encoding="utf-8",
    )
    (group / "summary.json").write_text(
        json.dumps({"num_episodes": len(results)}), encoding="utf-8",
    )
    (group / "per_episode.csv").write_text("episode_id\n", encoding="utf-8")
    digest = sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (group / "pilot_complete.json").write_text(
        json.dumps({"expected_count": 88, "manifest_sha256": digest}), encoding="utf-8",
    )


def test_seven_group_artifacts_pass_verification_and_compare(tmp_path: Path) -> None:
    from scripts.compare_l1_l3_prompt_ablation import compare, verify_only, write_reports

    root = tmp_path / "ablation"
    root.mkdir()
    manifest = _manifest()
    (root / "episode_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_provenance(root, manifest)
    for mode in _MODES:
        _group_artifacts(root, mode, manifest)
    assert verify_only(root) == 0
    report, paired_rows = compare(root)
    assert report["status"] == "complete"
    assert set(report["groups"]) == set(_MODES)
    assert len(report["paired_differences"]) == 21
    assert paired_rows
    assert report["checks"]["original_char_budget_chars"] == 320
    assert report["groups"]["l1_l3_joint"]["metrics"]["SR_pct"]["n"] == 88
    write_reports(root, report, paired_rows)
    for name in ("comparison.csv", "comparison.json", "paired_episode_deltas.csv",
                 "final_report.md"):
        assert (root / name).is_file()
    report_text = (root / "final_report.md").read_text()
    assert "McNemar" in report_text and "代码验收" in report_text
    assert "导航效果判定" in report_text
    assert "not_demonstrated" in report_text
    assert "Latency" in report_text
    assert report["navigation_analysis"]["status"] == "complete"
    assert report["navigation_analysis"]["baseline"] == "legacy"
    assert "l1_l3_joint" in (root / "comparison.csv").read_text()


def test_a_changed_non_prompt_memory_argument_breaks_consistency(tmp_path: Path) -> None:
    from scripts.compare_l1_l3_prompt_ablation import verify_only

    root = tmp_path / "ablation"
    root.mkdir()
    manifest = _manifest()
    (root / "episode_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_provenance(root, manifest)
    for mode in _MODES:
        _group_artifacts(root, mode, manifest)
    config_path = root / "l1_l3_joint" / "config_snapshot.yaml"
    config = json.loads(config_path.read_text())
    config["memory_args"] = [
        item for item in config["memory_args"] if item != "--memory_top_k"
    ]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="differs outside --memory_prompt_mode"):
        verify_only(root)


def test_resume_verifies_existing_provenance_and_rejects_a_different_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    created = _run(tmp_path, {}, "--preflight-only", "--output-root", str(output))
    assert created.returncode == 0, created.stderr
    resumed = _run(
        tmp_path, {}, "--resume", "--dry-run", "--output-root", str(output),
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "resume provenance verified" in resumed.stdout

    other_yolo = tmp_path / "other_yolo.pt"
    other_yolo.write_bytes(b"different L1 weights")
    changed = _run(
        tmp_path, {"YOLO_MODEL_PATH": str(other_yolo)},
        "--resume", "--dry-run", "--output-root", str(output),
    )
    assert changed.returncode != 0
    assert "resume provenance differs" in changed.stderr


def test_resume_requires_an_existing_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    result = _run(
        tmp_path, {}, "--resume", "--output-root", str(missing),
    )
    assert result.returncode != 0
    assert "resume root not found" in result.stderr


def test_comparison_recovers_old_snapshot_without_duplicated_budget_fields(
    tmp_path: Path,
) -> None:
    from scripts.compare_l1_l3_prompt_ablation import verify_only

    root = tmp_path / "ablation"
    root.mkdir()
    manifest = _manifest()
    (root / "episode_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_provenance(root, manifest)
    for mode in _MODES:
        _group_artifacts(root, mode, manifest)
    config_path = root / "l1_original_only" / "config_snapshot.yaml"
    config = json.loads(config_path.read_text())
    config["protocol_version"] = "l1_l3_prompt_ablation_v1"
    config.pop("prompt_protocol_version", None)
    config["prompt_effective"].pop("budget_strategy", None)
    config["prompt_effective"].pop("original_char_budget_chars", None)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert verify_only(root) == 0


def test_group_identity_is_bound_to_root_provenance(tmp_path: Path) -> None:
    from scripts.compare_l1_l3_prompt_ablation import verify_only

    root = tmp_path / "ablation"
    root.mkdir()
    manifest = _manifest()
    (root / "episode_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_provenance(root, manifest)
    for mode in _MODES:
        _group_artifacts(root, mode, manifest)
    config_path = root / "l1_original_only" / "config_snapshot.yaml"
    config = json.loads(config_path.read_text())
    config["common"]["model_weights_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        verify_only(root)


def test_marked_complete_group_with_incomplete_prompt_steps_is_rejected(
    tmp_path: Path,
) -> None:
    from scripts.compare_l1_l3_prompt_ablation import verify_only

    root = tmp_path / "ablation"
    root.mkdir()
    manifest = _manifest()
    (root / "episode_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_provenance(root, manifest)
    for mode in _MODES:
        _group_artifacts(root, mode, manifest)
    diagnostics = root / "l1_original_only" / "prompt_diagnostics.jsonl"
    lines = diagnostics.read_text().splitlines()
    diagnostics.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks complete prompt diagnostics"):
        verify_only(root)
