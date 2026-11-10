#!/usr/bin/env python3
"""Inspect a local VLN checkpoint's SlowFast visual-token budget on CPU.

This diagnostic uses the checkpoint's tokenizer and video processor, but never
loads model weights. It writes nothing: all results go to stdout as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any


EPISODE_LENGTHS = (2, 10, 34, 90, 120, 256)
DEFAULT_INSTRUCTION = "walk forward through the corridor and stop"
REPO_ROOT = Path(__file__).resolve().parents[1]


def grid_patch_rows(grid: list[list[int]]) -> int:
    """Number of patch rows entering the ViT, before its spatial merger."""
    total = 0
    for row in grid:
        if len(row) != 3:
            raise ValueError(f"grid row must be [T, H, W], got {row!r}")
        t, h, w = (int(value) for value in row)
        if min(t, h, w) <= 0:
            raise ValueError(f"grid dimensions must be positive, got {row!r}")
        total += t * h * w
    return total


def grid_visual_tokens(grid: list[list[int]], merge_size: int) -> int:
    """Number of merged visual tokens represented by a processor grid."""
    if merge_size <= 0:
        raise ValueError(f"merge_size must be positive, got {merge_size}")
    total = 0
    for row in grid:
        if len(row) != 3:
            raise ValueError(f"grid row must be [T, H, W], got {row!r}")
        t, h, w = (int(value) for value in row)
        if min(t, h, w) <= 0 or h % merge_size or w % merge_size:
            raise ValueError(f"grid {row!r} is invalid for merge_size={merge_size}")
        total += t * (h // merge_size) * (w // merge_size)
    return total


def generation_prompt_length(labels: Sequence[int], ignore_index: int) -> int:
    """The answer boundary used by VLNInferenceEngine._prepare_inputs."""
    for index, label in enumerate(labels):
        if int(label) != ignore_index:
            return index
    raise ValueError("processor did not mark an assistant answer boundary")


def check_processor_counts(
    original_grid: list[list[int]],
    pooled_grid: list[list[int]],
    *,
    merge_size: int,
    pixel_rows: int,
    placeholder_tokens: int,
) -> tuple[int, int, int]:
    """Cross-check processor tensors and expanded placeholders against both grids."""
    if len(original_grid) != len(pooled_grid):
        raise ValueError("original and pooled grids have different segment counts")
    patch_rows = grid_patch_rows(original_grid)
    pre_pool_tokens = grid_visual_tokens(original_grid, merge_size)
    post_pool_tokens = grid_visual_tokens(pooled_grid, merge_size)
    if pixel_rows != patch_rows:
        raise ValueError(f"ViT patch rows: processor={pixel_rows}, original grid={patch_rows}")
    if placeholder_tokens != post_pool_tokens:
        raise ValueError(
            "expanded video placeholders: "
            f"processor={placeholder_tokens}, pooled grid={post_pool_tokens}"
        )
    if post_pool_tokens > pre_pool_tokens:
        raise ValueError("post-ViT pooling increased the visual token count")
    return patch_rows, pre_pool_tokens, post_pool_tokens


def _tensor_grid(value: Any, name: str) -> list[list[int]]:
    if value is None:
        raise RuntimeError(f"processor did not return {name}")
    if value.ndim == 3 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 2 or value.shape[1] != 3:
        raise RuntimeError(f"{name} must be [segments, 3], got {tuple(value.shape)}")
    return [[int(part) for part in row] for row in value.tolist()]


def _read_checkpoint_params(checkpoint: Path) -> dict[str, Any]:
    from lightnav.eval_config import get_task_params, resolve_inference_params

    config_path = checkpoint / "eval_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"local checkpoint config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task_params = get_task_params(config, "vlnce")
    for key in ("video_size", "pool_enable", "pool_stage", "slowfast_tiers", "video_fps"):
        if key not in task_params:
            raise ValueError(f"eval_config.json has no vlnce/common {key!r}")
    params, _ = resolve_inference_params({}, task_params)
    if not params["slowfast_tiers"]:
        raise ValueError("vlnce.slowfast_tiers is empty; this is not a SlowFast checkpoint")
    if not params["pool_enable"] or params["pool_stage"] != "post_vit":
        raise ValueError("this diagnostic requires pool_enable=true and pool_stage='post_vit'")
    size = tuple(int(part) for part in params["video_size"])
    if len(size) != 2 or min(size) <= 0:
        raise ValueError(f"invalid video_size: {params['video_size']!r}")
    params["video_size"] = size
    return params


def inspect_checkpoint(checkpoint: Path, instruction: str) -> dict[str, Any]:
    # Set before importing transformers / huggingface_hub. Every from_pretrained
    # call below is restricted to local files; Python must not create bytecode.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO_ROOT / "src"))

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"local checkpoint directory is missing: {checkpoint}")
    params = _read_checkpoint_params(checkpoint)
    os.environ["VLN_TIMESTAMP_RELATIVE"] = "1" if params["timestamp_relative"] else "0"

    import numpy as np
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    from lightnav.data_processor import IGNORE_INDEX, Qwen3VLDataProcessor
    from lightnav.inference.frame_preprocessing import rgb_frame_to_model_tensor
    from lightnav.inference.model import _get_position_id, resolve_model_paths
    from lightnav.inference.samples import build_vln_traj_sample
    from lightnav.processing import VLNQwen3VLProcessor
    from lightnav.slowfast import slowfast_segments, validate_slowfast_tiers

    tiers = validate_slowfast_tiers(params["slowfast_tiers"])
    config_dir, _, processor_dir = resolve_model_paths(str(checkpoint))
    processor = VLNQwen3VLProcessor.from_pretrained(
        processor_dir, padding_side="left", local_files_only=True
    )
    video_processor = processor.video_processor
    loaded_video_processor = {
        "patch_size": int(video_processor.patch_size),
        "merge_size": int(video_processor.merge_size),
        "temporal_patch_size": int(video_processor.temporal_patch_size),
        "do_sample_frames": bool(video_processor.do_sample_frames),
        "do_resize": bool(video_processor.do_resize),
        "do_rescale": bool(video_processor.do_rescale),
    }
    # Equivalent to inference.model._position_id_func_from_config, but every
    # config lookup is explicitly local-only and remote code is disabled.
    local_config = AutoConfig.from_pretrained(
        config_dir, local_files_only=True, trust_remote_code=False
    )
    position_id_func = partial(
        _get_position_id,
        Qwen3VLModel.get_rope_index,
        SimpleNamespace(config=local_config),
    )
    data_processor = Qwen3VLDataProcessor(
        processor=processor,
        model_max_length=None,  # measure the full expansion without silent truncation
        video_fps=int(params["video_fps"]),
        video_pool_enable=bool(params["pool_enable"]),
        video_pool_spatial=int(params["pool_spatial"]),
        video_pool_mode=str(params["pool_mode"]),
        video_pool_stage=str(params["pool_stage"]),
        video_size=params["video_size"],
        position_id_func=position_id_func,
    )
    merge_size = int(video_processor.merge_size)
    height, width = params["video_size"]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = (32, 96, 160)
    model_frame = rgb_frame_to_model_tensor(rgb, params["video_size"])
    bundle = SimpleNamespace(
        slowfast_tiers=tiers,
        video_fps=int(params["video_fps"]),
        predict_horizon=int(params["predict_horizon"]),
        action_method=(params.get("action_tokenizer") or {}).get("method") or "flat",
    )

    episodes = []
    for episode_length in EPISODE_LENGTHS:
        layout = slowfast_segments(episode_length - 1, episode_length, tiers)
        # The broadcast view represents a full RGB episode without allocating 256
        # float32 frames. slowfast_video_segments indexes the selected frames.
        episode_frames = model_frame.unsqueeze(0).expand(episode_length, -1, -1, -1)
        sample = build_vln_traj_sample(episode_frames, instruction, None, bundle)
        segments = sample["video_segments"]
        if len(layout) != len(segments):
            raise RuntimeError("sample segment count differs from slowfast_segments")
        tier_report = []
        for selected, segment in zip(layout, segments):
            if selected["frame_ids"] != segment["frame_indices"]:
                raise RuntimeError("sample frame IDs differ from slowfast_segments")
            if selected["pool_spatial"] != segment["pool_spatial"]:
                raise RuntimeError("sample pooling differs from slowfast_segments")
            tier_report.append(
                {
                    "tier_index": selected["tier"],
                    "tier_name": tiers[selected["tier"]]["name"],
                    "frame_indices": selected["frame_ids"],
                    "frame_count": len(selected["frame_ids"]),
                    "pool_spatial": selected["pool_spatial"],
                    "pool_mode": selected["pool_mode"],
                }
            )

        with torch.no_grad():
            inputs = data_processor.process_sample(sample, add_generation_prompt=True)
        original_grid = _tensor_grid(inputs.get("_original_video_grid_thw"), "original video grid")
        pooled_grid = _tensor_grid(inputs.get("video_grid_thw"), "pooled video grid")
        pixels = inputs.get("pixel_values_videos")
        if pixels is None:
            raise RuntimeError("processor did not return pixel_values_videos")
        if pixels.ndim == 3 and pixels.shape[0] == 1:
            pixels = pixels.squeeze(0)
        if pixels.ndim != 2:
            raise RuntimeError(f"pixel_values_videos must be 2D, got {tuple(pixels.shape)}")

        token_ids = inputs["input_ids"]
        labels = inputs["labels"]
        # VLNInferenceEngine._prepare_inputs trims exactly at this boundary.
        prompt_length = generation_prompt_length(labels.tolist(), IGNORE_INDEX)
        placeholder_tokens = int((token_ids[:prompt_length] == data_processor.video_token_id).sum())
        patch_rows, pre_pool_tokens, post_pool_tokens = check_processor_counts(
            original_grid,
            pooled_grid,
            merge_size=merge_size,
            pixel_rows=int(pixels.shape[0]),
            placeholder_tokens=placeholder_tokens,
        )
        if len(original_grid) != len(segments):
            raise RuntimeError("processor grid count differs from sample segment count")
        temporal_patch_size = int(video_processor.temporal_patch_size)
        for tier_result, raw_grid, final_grid in zip(tier_report, original_grid, pooled_grid):
            expected_t = (tier_result["frame_count"] + temporal_patch_size - 1) // temporal_patch_size
            if raw_grid[0] != expected_t or final_grid[0] != expected_t:
                raise RuntimeError(
                    f"tier {tier_result['tier_name']} temporal grid differs from selected frame count"
                )
            tier_result["original_grid_thw"] = raw_grid
            tier_result["vit_patch_rows"] = grid_patch_rows([raw_grid])
            tier_result["vit_output_visual_tokens_before_pool"] = grid_visual_tokens(
                [raw_grid], merge_size
            )
            tier_result["pooled_grid_thw"] = final_grid
            tier_result["llm_visual_tokens_after_pool"] = grid_visual_tokens(
                [final_grid], merge_size
            )
        configured_max = int(params["max_seq_len"])
        full_processed_length = int(token_ids.numel())
        full_sequence_fits = full_processed_length <= configured_max
        prompt_fits = prompt_length <= configured_max
        episodes.append(
            {
                "episode_length": episode_length,
                "tiers": tier_report,
                "original_video_grid_thw": original_grid,
                "vit_patch_rows": patch_rows,
                "vit_output_visual_tokens_before_pool": pre_pool_tokens,
                "pooled_video_grid_thw": pooled_grid,
                "llm_visual_tokens_after_pool": post_pool_tokens,
                "expanded_video_placeholder_tokens": placeholder_tokens,
                "placeholder_matches_pooled_grid": True,
                "generation_prompt_tokens_untruncated": prompt_length,
                "model_input_tokens": prompt_length if prompt_fits else None,
                "model_input_status": (
                    "measured from processor input_ids at the engine answer boundary"
                    if prompt_fits else
                    "unavailable: checkpoint max_seq_len would truncate the generation prompt"
                ),
                "processor_full_sequence_tokens_including_placeholder_answer": full_processed_length,
                "max_seq_len": configured_max,
                "generation_prompt_fits_max_seq_len": prompt_fits,
                "full_sequence_fits_max_seq_len": full_sequence_fits,
            }
        )
        del inputs, pixels, segments, sample, episode_frames

    return {
        "checkpoint": str(checkpoint),
        "config": str(checkpoint / "eval_config.json"),
        "model_config_dir": config_dir,
        "processor_dir": processor_dir,
        "instruction": instruction,
        "video_size_hw": [height, width],
        "pool_stage": params["pool_stage"],
        "video_fps": params["video_fps"],
        "timestamp_relative": bool(params["timestamp_relative"]),
        "slowfast_tiers": tiers,
        "processor_class": type(processor).__name__,
        "loaded_checkpoint_video_processor": loaded_video_processor,
        "effective_inference_video_processor": {
            "patch_size": int(video_processor.patch_size),
            "merge_size": merge_size,
            "temporal_patch_size": int(video_processor.temporal_patch_size),
            "do_sample_frames": bool(video_processor.do_sample_frames),
            "do_resize": bool(video_processor.do_resize),
            "do_rescale": bool(video_processor.do_rescale),
            "sample_skip_normalize": True,
        },
        "counting_note": (
            "vit_patch_rows are unpooled ViT input patches; "
            "vit_output_visual_tokens_before_pool are merger outputs; "
            "llm_visual_tokens_after_pool are LLM inputs. Post-ViT pooling does not "
            "proportionally reduce ViT work. A ViT cache may avoid recomputing "
            "some patches in a real session; this script runs no ViT."
        ),
        "episodes": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/LightNav-0"),
        help="local checkpoint directory containing eval_config.json (default: checkpoints/LightNav-0)",
    )
    parser.add_argument(
        "--instruction", default=DEFAULT_INSTRUCTION,
        help="fixed prompt text used when measuring the full input sequence length",
    )
    args = parser.parse_args()
    try:
        report = inspect_checkpoint(args.checkpoint, args.instruction)
    except Exception as exc:
        print(f"SlowFast budget inspection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
