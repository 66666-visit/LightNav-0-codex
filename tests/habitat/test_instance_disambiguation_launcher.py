"""CPU-only dry-run checks for the four-Episode instance-disambiguation launcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "run_instance_disambiguation_4.sh"


def _assets(tmp_path: Path, rows: list[dict]) -> dict[str, str]:
    (tmp_path / "data").mkdir()
    (tmp_path / "scenes").mkdir()
    for name in ("model", "yolo.pt", "seg.pt", "grounder.pt", "vocab.txt"):
        (tmp_path / name).touch()
    (tmp_path / "grounder.yaml").write_text("model: {}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "dataset_split": "val_unseen",
        "expected_count": len(rows),
        "episodes": rows,
    }), encoding="utf-8")
    return {
        "MODEL_PATH": str(tmp_path / "model"),
        "YOLO_MODEL_PATH": str(tmp_path / "yolo.pt"),
        "SEG_MODEL_PATH": str(tmp_path / "seg.pt"),
        "GROUNDER_CHECKPOINT": str(tmp_path / "grounder.pt"),
        "GROUNDER_CONFIG": str(tmp_path / "grounder.yaml"),
        "GROUNDER_BERT_VOCAB": str(tmp_path / "vocab.txt"),
        "EPISODE_MANIFEST_PATH": str(manifest),
        "DATA_PATH": str(tmp_path / "data"),
        "SCENES_DIR": str(tmp_path / "scenes"),
        "CLIENT_PYTHON": sys.executable,
        "DRY_RUN": "1",
    }


def _run(tmp_path: Path, env: dict[str, str], output_name: str):
    full_env = os.environ.copy()
    full_env.update(env)
    full_env["OUTPUT_ROOT"] = str(tmp_path / output_name)
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=full_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _rows(ids: list[str]) -> list[dict]:
    return [
        {
            "scene_id": f"data/scene_datasets/mp3d/scene{index}/scene{index}.glb",
            "habitat_episode_id": episode_id,
            "rollout_idx": 0,
        }
        for index, episode_id in enumerate(ids)
    ]


def test_launcher_dry_run_resolves_four_exact_ids_without_gpu(tmp_path: Path) -> None:
    env = _assets(tmp_path, _rows(["487", "168", "1080", "1012"]))
    result = _run(tmp_path, env, "out_ok")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "DRY_RUN: no server or GPU started" in result.stdout
    resolved = json.loads(
        (tmp_path / "out_ok" / "resolved_config.json").read_text()
    )
    assert resolved["selection_source"] == "manifest_exact_id"
    assert [row["habitat_episode_id"] for row in resolved["episodes"]] == [
        "487", "168", "1080", "1012",
    ]
    assert all(row["rollout_idx"] == 0 for row in resolved["episodes"])
    assert resolved["ablation"] == {
        "id": "F",
        "semantics": (
            "joint eligible selection + multi-landmark queries + "
            "same-category geometry fallback"
        ),
        "grounder_selection_mode": "eligible",
        "grounder_landmark_queries": True,
        "grounder_geometry_fallback": True,
        "ordered_landmark_memory": False,
        "include_missing_landmark_placeholders": True,
        "preserve_original_spatial_fact": False,
    }
    assert resolved["frozen"]["episodes_per_run"] == 1
    selection = json.loads(
        (tmp_path / "out_ok" / "four_episode_selection.json").read_text()
    )
    assert len(selection["episodes"]) == 4


def test_launcher_never_treats_episode_001_as_habitat_id_1(tmp_path: Path) -> None:
    env = _assets(tmp_path, _rows(["001", "2", "3", "4"]))
    env["EPISODE_IDS"] = "1 2 3 4"
    result = _run(tmp_path, env, "out_bad")
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "matches=0" in combined or "exactly one manifest row" in combined


@pytest.mark.parametrize(
    ("group", "placeholders", "original_fact"),
    [("H1", False, False), ("H2", True, True), ("H3", False, True)],
)
def test_launcher_wires_h1_h2_h3_switches(
    tmp_path: Path, group: str, placeholders: bool, original_fact: bool,
) -> None:
    env = _assets(tmp_path, _rows(["487", "168", "1080", "1012"]))
    env["ABLATION_GROUP"] = group
    result = _run(tmp_path, env, f"out_{group}")
    assert result.returncode == 0, result.stderr + result.stdout
    resolved = json.loads(
        (tmp_path / f"out_{group}" / f"group_{group}"
         / "resolved_config.json").read_text()
    )
    ablation = resolved["ablation"]
    assert ablation["id"] == group
    assert ablation["ordered_landmark_memory"] is True
    assert ablation["include_missing_landmark_placeholders"] is placeholders
    assert ablation["preserve_original_spatial_fact"] is original_fact
    assert resolved["flags"] == {
        "spatial_l3_grounder_selection": "eligible",
        "spatial_l3_grounder_landmark_queries": True,
        "spatial_l3_grounder_geometry_fallback": True,
        "spatial_l3_grounder_ordered_landmarks": True,
        "spatial_l3_grounder_include_missing_landmark_placeholders": placeholders,
        "spatial_l3_grounder_preserve_original_spatial_fact": original_fact,
    }
