"""CPU-only dry-run checks for the 1012-only ordered-landmark launcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "run_instance_disambiguation_1012.sh"


def _assets(tmp_path: Path, rows: list[dict]) -> dict[str, str]:
    (tmp_path / "data").mkdir()
    (tmp_path / "scenes").mkdir()
    for name in ("model", "yolo.pt", "seg.pt", "grounder.pt", "vocab.txt"):
        (tmp_path / name).touch()
    (tmp_path / "grounder.yaml").write_text("model: {}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": rows}), encoding="utf-8")
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


def test_1012_launcher_dry_run_resolves_exact_id_and_ordered_flags(
    tmp_path: Path,
) -> None:
    env = _assets(tmp_path, [
        {
            "scene_id": "data/scene_datasets/mp3d/scene1012/scene1012.glb",
            "habitat_episode_id": "1012",
            "rollout_idx": 0,
        },
        {
            "scene_id": "data/scene_datasets/mp3d/scene168/scene168.glb",
            "habitat_episode_id": "168",
            "rollout_idx": 0,
        },
    ])
    result = _run(tmp_path, env, "out_1012")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "DRY_RUN: no server or GPU started" in result.stdout
    assert "episode: 1012" in result.stdout
    resolved = json.loads(
        (tmp_path / "out_1012" / "resolved_config.json").read_text()
    )
    assert resolved["protocol"] == (
        "instance_disambiguation_1012_ordered_landmarks_v1"
    )
    assert resolved["ablation_group"] == "G"
    assert resolved["episode_id"] == 1012
    assert resolved["episode_key"] == {
        "scene_id": "data/scene_datasets/mp3d/scene1012/scene1012.glb",
        "habitat_episode_id": "1012",
        "rollout_idx": 0,
    }
    tokens = resolved["client_args"]
    assert tokens[tokens.index("--spatial_l3_grounder_selection") + 1] == (
        "eligible"
    )
    for flag in (
        "--spatial_l3_grounder_landmark_queries",
        "--spatial_l3_grounder_geometry_fallback",
        "--spatial_l3_grounder_ordered_landmarks",
    ):
        assert flag in tokens
    assert "--spatial_l3_grounder_selection" in tokens
    assert resolved["frozen"]["episodes_per_run"] == 1


def test_1012_launcher_rejects_ambiguous_manifest_ids(tmp_path: Path) -> None:
    env = _assets(tmp_path, [
        {
            "scene_id": "a/scene1012.glb",
            "habitat_episode_id": "1012",
            "rollout_idx": 0,
        },
        {
            "scene_id": "b/scene1012.glb",
            "habitat_episode_id": "1012",
            "rollout_idx": 0,
        },
    ])
    result = _run(tmp_path, env, "out_bad")
    assert result.returncode != 0
    assert "matches=2" in (result.stderr + result.stdout)
