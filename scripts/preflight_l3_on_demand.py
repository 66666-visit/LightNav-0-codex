#!/usr/bin/env python3
"""No-GPU preflight for the L1 Original + on-demand L3 CLIP experiment.

This script only inspects local paths and imports; it never loads model weights,
allocates CUDA memory, starts Habitat, or runs inference.  It prints the exact
``CLIENT_ARGS`` payload with ``shlex.join`` so callers do not hit the Bash
pitfall where ``VAR=... CLIENT_ARGS="$VAR"`` expands ``$VAR`` before the
assignment takes effect.

Typical use on AutoDL::

    PYTHONPATH=src python scripts/preflight_l3_on_demand.py \
        --model-path /root/autodl-tmp/LightNav-0 \
        --yolo-model /root/autodl-tmp/weights/yolo11n.pt \
        --seg-model /root/autodl-tmp/weights/yolo11n-seg.pt \
        --clip-model /root/autodl-tmp/weights/open_clip_ViT-H-14_laion2b_s32b_b79k.bin \
        --episode-manifest ... --pilot-manifest ... \
        --habitat-config habitat_server/configs/vlnce_r2r.yaml \
        --data-path /root/autodl-tmp/data --scenes-dir /root/autodl-tmp/scenes \
        --habitat-conda-env habitat --client-python .venv/bin/python \
        --print-only
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_client_args(
    *, yolo_model: str, seg_model: str, clip_model: str,
    l1_device: str = "cpu", seg_device: str = "cpu",
    clip_device: str = "cuda:0", clip_precision: str = "fp16",
    clip_batch_size: int = 8, clip_top_k: int = 3,
) -> list[str]:
    """Return the frozen L1 Original parameters plus on-demand L3 flags."""
    return [
        "--no_force_stop",
        "--verbose",
        "--memory_enabled",
        "--memory_strategy", "hierarchical",
        "--memory_layers", "short,mid,long",
        "--memory_top_k", "5",
        "--memory_max_context_chars", "600",
        "--memory_extractor", "yolo11",
        "--memory_extractor_model_path", yolo_model,
        "--memory_extractor_device", l1_device,
        "--memory_extractor_confidence", "0.25",
        "--memory_extract_every_n_steps", "5",
        "--local_graph_enabled",
        "--local_graph_scope", "episode_local",
        "--local_graph_prompt_max_chars", "320",
        "--local_graph_max_landmarks", "2",
        "--local_graph_reactivation_steps", "8",
        "--memory_prompt_mode", "l1_original_only",
        "--spatial_l3_mode", "on_demand",
        "--spatial_l3_seg_model_path", seg_model,
        "--spatial_l3_seg_device", seg_device,
        "--spatial_l3_seg_confidence", "0.25",
        "--spatial_l3_seg_every_n_steps", "5",
        "--spatial_l3_evidence_min_iou", "0.5",
        "--spatial_l3_evidence_ambiguity_margin", "0.1",
        "--spatial_l3_min_injection_confidence", "0.5",
        "--spatial_l3_clip_model_path", clip_model,
        "--spatial_l3_clip_model_name", "ViT-H-14",
        "--spatial_l3_clip_device", clip_device,
        "--spatial_l3_clip_precision", clip_precision,
        "--spatial_l3_clip_batch_size", str(clip_batch_size),
        "--spatial_l3_clip_top_k", str(clip_top_k),
        "--spatial_l3_clip_min_similarity", "0.0",
        "--prompt_measurement_enabled",
    ]


def build_eval_env(
    *, model_path: str, episode_manifest: str | None,
    pilot_manifest: str | None, habitat_config: str, data_path: str | None,
    scenes_dir: str | None, habitat_conda_env: str, client_python: str,
    gpu_ids: str, output_root: str, gpu_mem_util: float = 0.65,
    kv_cache_gib: float | None = None, enforce_eager: bool = False,
    smoke: bool = False,
) -> dict[str, str]:
    """Return the environment consumed by ``scripts/eval_habitat.sh``."""
    env = {
        "MODEL_PATH": model_path,
        "BACKEND": "vllm_local",
        "TASK": "vlnce",
        "SPLIT": "val_unseen",
        "GPU_IDS": gpu_ids,
        "GPU_MEM_UTIL": f"{gpu_mem_util:g}",
        "BASE_PORT": "5555",
        "RESUME": "0",
        "HABITAT_CONFIG": habitat_config,
        "HABITAT_CONDA_ENV": habitat_conda_env,
        "CLIENT_PYTHON": client_python,
        "OUTPUT_ROOT": output_root,
        "EXPECTED_EPISODES": "1" if smoke else "88",
        "STRICT_FULL_EVAL": "0" if smoke else "1",
        "NO_FORCE_STOP": "1",
        "MAX_STEPS": "40" if smoke else "500",
        "EPISODES": "1" if smoke else "-1",
        "SEED": "42",
    }
    if kv_cache_gib is not None:
        env["VLN_KV_CACHE_GIB"] = f"{kv_cache_gib:g}"
    if enforce_eager:
        env["VLN_VLLM_ENFORCE_EAGER"] = "1"
    if episode_manifest:
        env["EPISODE_MANIFEST_PATH"] = episode_manifest
    if pilot_manifest:
        env["PILOT_MANIFEST_PATH"] = pilot_manifest
    if data_path:
        env["DATA_PATH"] = data_path
    if scenes_dir:
        env["SCENES_DIR"] = scenes_dir
    return env


def _check_file(value: str, label: str, errors: list[str]) -> None:
    path = Path(value).expanduser()
    if not path.is_file():
        errors.append(f"{label} is not a file: {path}")
    elif path.stat().st_size == 0:
        errors.append(f"{label} is empty: {path}")


def _check_dir(value: str, label: str, errors: list[str]) -> None:
    path = Path(value).expanduser()
    if not path.is_dir():
        errors.append(f"{label} is not a directory: {path}")


def _check_imports(
    python: str, *, habitat: bool = False, env: dict[str, str] | None = None,
) -> dict[str, Any]:
    modules = (
        "lightnav.cli.eval_habitat, open_clip, ultralytics, zmq, torch, transformers"
        if not habitat else "habitat, habitat_sim, lightnav_habitat.serve"
    )
    try:
        proc = subprocess.run(
            [python, "-c", f"import {modules}; print('ok')"],
            capture_output=True, text=True, timeout=120, env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "python": python, "modules": modules, "ok": False,
            "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "python": python,
        "modules": modules,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip()[-2000:],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--yolo-model", required=True)
    p.add_argument("--seg-model", required=True)
    p.add_argument("--clip-model", required=True)
    p.add_argument("--episode-manifest", default=None)
    p.add_argument("--pilot-manifest", default=None)
    p.add_argument("--habitat-config", default="habitat_server/configs/vlnce_r2r.yaml")
    p.add_argument("--data-path", default=None)
    p.add_argument("--scenes-dir", default=None)
    p.add_argument("--habitat-conda-env", default="habitat")
    p.add_argument("--habitat-python", default=None)
    p.add_argument("--client-python", default=sys.executable)
    p.add_argument("--gpu-ids", default="0")
    p.add_argument("--gpu-mem-util", type=float, default=0.65)
    p.add_argument("--kv-cache-gib", type=float, default=None,
                   help="Optional OOM fallback; engine auto uses 2 GiB for batch=1.")
    p.add_argument("--enforce-eager", action="store_true",
                   help="OOM fallback only; default matches the six-group baseline.")
    p.add_argument("--output-root", default="output/l3_on_demand_l1_original_88")
    p.add_argument("--l1-device", default="cpu")
    p.add_argument("--seg-device", default="cpu")
    p.add_argument("--clip-device", default="cuda:0")
    p.add_argument("--clip-precision", default="fp16")
    p.add_argument("--clip-batch-size", type=int, default=8)
    p.add_argument("--clip-top-k", type=int, default=3)
    p.add_argument("--smoke", action="store_true",
                   help="Emit the 1-episode/40-step smoke environment.")
    p.add_argument("--print-only", action="store_true",
                   help="Skip package import checks; paths are still validated.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    habitat_env = dict(os.environ)
    habitat_prefix = str(repo_root / "habitat_server")
    habitat_env["PYTHONPATH"] = (
        f"{habitat_prefix}:{habitat_env['PYTHONPATH']}"
        if habitat_env.get("PYTHONPATH") else habitat_prefix
    )
    errors: list[str] = []
    for value, label in (
        (args.model_path, "MODEL_PATH"),
        (args.yolo_model, "YOLO_MODEL_PATH"),
        (args.seg_model, "SEG_MODEL_PATH"),
        (args.clip_model, "CLIP_MODEL_PATH"),
        (args.habitat_config, "HABITAT_CONFIG"),
    ):
        if label == "MODEL_PATH":
            _check_dir(value, label, errors)
        else:
            _check_file(value, label, errors)
    for value, label in (
        (args.episode_manifest, "EPISODE_MANIFEST_PATH"),
        (args.pilot_manifest, "PILOT_MANIFEST_PATH"),
    ):
        if value:
            _check_file(value, label, errors)
    for value, label in (
        (args.data_path, "DATA_PATH"),
        (args.scenes_dir, "SCENES_DIR"),
    ):
        if value:
            _check_dir(value, label, errors)
    if not args.smoke:
        if not args.episode_manifest:
            errors.append(
                "EPISODE_MANIFEST_PATH is required for the formal 88-episode run"
            )
        if not args.pilot_manifest:
            errors.append(
                "PILOT_MANIFEST_PATH is required for the formal 88-episode run"
            )

    client_args = build_client_args(
        yolo_model=args.yolo_model,
        seg_model=args.seg_model,
        clip_model=args.clip_model,
        l1_device=args.l1_device,
        seg_device=args.seg_device,
        clip_device=args.clip_device,
        clip_precision=args.clip_precision,
        clip_batch_size=args.clip_batch_size,
        clip_top_k=args.clip_top_k,
    )
    eval_env = build_eval_env(
        model_path=args.model_path,
        episode_manifest=args.episode_manifest,
        pilot_manifest=args.pilot_manifest,
        habitat_config=args.habitat_config,
        data_path=args.data_path,
        scenes_dir=args.scenes_dir,
        habitat_conda_env=args.habitat_conda_env,
        client_python=args.client_python,
        gpu_ids=args.gpu_ids,
        output_root=args.output_root,
        gpu_mem_util=args.gpu_mem_util,
        kv_cache_gib=args.kv_cache_gib,
        enforce_eager=args.enforce_eager,
        smoke=args.smoke,
    )
    report: dict[str, Any] = {
        "status": "error" if errors else "ok",
        "errors": errors,
        "client_args": client_args,
        "client_args_shell": shlex.join(client_args),
        "eval_env": eval_env,
        "command": "bash scripts/eval_habitat.sh",
    }
    if not args.print_only:
        report["client_imports"] = _check_imports(args.client_python)
        habitat_python = args.habitat_python
        if habitat_python is None:
            try:
                habitat_proc = subprocess.run(
                    [
                        "conda", "run", "-n", args.habitat_conda_env,
                        "python", "-c",
                        "import habitat, lightnav_habitat; print('ok')",
                    ],
                    capture_output=True, text=True, timeout=120,
                    env=habitat_env,
                )
                report["habitat_imports"] = {
                    "conda_env": args.habitat_conda_env,
                    "ok": habitat_proc.returncode == 0,
                    "stdout": habitat_proc.stdout.strip(),
                    "stderr": habitat_proc.stderr.strip()[-2000:],
                }
            except (OSError, subprocess.TimeoutExpired) as exc:
                report["habitat_imports"] = {
                    "conda_env": args.habitat_conda_env, "ok": False,
                    "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
                }
        else:
            report["habitat_imports"] = _check_imports(
                habitat_python, habitat=True, env=habitat_env,
            )
        if not report["client_imports"]["ok"] or not report["habitat_imports"]["ok"]:
            report["status"] = "error"
            errors.append("package import check failed")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
