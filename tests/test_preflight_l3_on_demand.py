"""CPU-only tests for the shell-safe AutoDL preflight command generator."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight_l3_on_demand.py"


def _paths(tmp_path: Path) -> list[str]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    yolo = tmp_path / "yolo.pt"
    yolo.write_bytes(b"x")
    seg = tmp_path / "seg.pt"
    seg.write_bytes(b"x")
    clip = tmp_path / "clip.bin"
    clip.write_bytes(b"x")
    episode = tmp_path / "episode.json"
    episode.write_text("{}")
    pilot = tmp_path / "pilot.json"
    pilot.write_text("{}")
    habitat = tmp_path / "habitat.yaml"
    habitat.write_text("---\n")
    data = tmp_path / "data"
    data.mkdir()
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    return [
        "--model-path", str(model),
        "--yolo-model", str(yolo),
        "--seg-model", str(seg),
        "--clip-model", str(clip),
        "--episode-manifest", str(episode),
        "--pilot-manifest", str(pilot),
        "--habitat-config", str(habitat),
        "--data-path", str(data),
        "--scenes-dir", str(scenes),
    ]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--print-only"],
        capture_output=True, text=True, timeout=60,
    )


def test_preflight_emits_shell_safe_l1_original_only_on_demand(tmp_path: Path) -> None:
    proc = _run(*_paths(tmp_path))
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["status"] == "ok"
    args = report["client_args"]
    assert args[args.index("--memory_prompt_mode") + 1] == "l1_original_only"
    assert args[args.index("--spatial_l3_mode") + 1] == "on_demand"
    assert args[args.index("--local_graph_prompt_max_chars") + 1] == "320"
    assert args[args.index("--local_graph_max_landmarks") + 1] == "2"
    assert args[args.index("--local_graph_reactivation_steps") + 1] == "8"
    assert args[args.index("--local_graph_scope") + 1] == "episode_local"
    assert args[args.index("--memory_max_context_chars") + 1] == "600"
    assert args[args.index("--memory_extractor_device") + 1] == "cpu"
    assert args[args.index("--spatial_l3_clip_precision") + 1] == "fp16"
    assert shlex.split(report["client_args_shell"]) == args
    env = report["eval_env"]
    assert env["SEED"] == "42"
    assert env["EXPECTED_EPISODES"] == "88"
    assert env["GPU_MEM_UTIL"] == "0.65"
    assert "VLN_KV_CACHE_GIB" not in env
    assert env["BASE_PORT"] == "5555"
    assert env["RESUME"] == "0"
    assert "VLN_VLLM_ENFORCE_EAGER" not in env
    assert "--no_force_stop" in args
    assert "--verbose" in args
    assert "--prompt_measurement_enabled" in args
    assert args[args.index("--spatial_l3_seg_device") + 1] == "cpu"


def test_preflight_requires_both_manifests(tmp_path: Path) -> None:
    args = _paths(tmp_path)
    proc = _run(
        "--model-path", args[1],
        "--yolo-model", args[3],
        "--seg-model", args[5],
        "--clip-model", args[7],
        "--habitat-config", args[11],
    )
    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert report["status"] == "error"
    assert any("EPISODE_MANIFEST_PATH" in error for error in report["errors"])
    assert any("PILOT_MANIFEST_PATH" in error for error in report["errors"])


def test_preflight_rejects_missing_clip_checkpoint(tmp_path: Path) -> None:
    args = _paths(tmp_path)
    args[7] = str(tmp_path / "missing.bin")
    proc = _run(*args)
    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert any("CLIP_MODEL_PATH" in error for error in report["errors"])


def test_preflight_smoke_mode_needs_no_manifests(tmp_path: Path) -> None:
    args = _paths(tmp_path)
    proc = _run(
        "--smoke",
        "--model-path", args[1],
        "--yolo-model", args[3],
        "--seg-model", args[5],
        "--clip-model", args[7],
        "--habitat-config", args[11],
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["eval_env"]["EPISODES"] == "1"
    assert report["eval_env"]["MAX_STEPS"] == "40"
    assert report["eval_env"]["STRICT_FULL_EVAL"] == "0"
