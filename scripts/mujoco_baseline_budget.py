"""Read a measured SlowFast diagnostic and project its visual budget per step.

The projection counts selected frames, unpooled ViT patches, ViT merger outputs,
and post-ViT LLM visual tokens. It does not run the processor or the model during
an episode, so these counts are configuration-derived, not runtime measurements.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightnav.slowfast import slowfast_segments, validate_slowfast_tiers  # noqa: E402
from scripts.inspect_slowfast_budget import grid_patch_rows, grid_visual_tokens  # noqa: E402


REQUIRED_LENGTHS = (2, 10, 34, 90, 120, 256)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _grid(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be [T, H, W]")
    return [_integer(part, f"{name}[{index}]", minimum=1) for index, part in enumerate(value)]


def _grid_list(value: Any, name: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of grids")
    return [_grid(row, f"{name}[{index}]") for index, row in enumerate(value)]


def _required_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _required_count(record: dict[str, Any], key: str, expected: int, context: str) -> None:
    actual = _integer(record.get(key), f"{context}.{key}")
    if actual != expected:
        raise ValueError(f"{context}.{key}={actual} differs from expected {expected}")


def _pooled_hw(raw_hw: tuple[int, int], spatial: int, merge: int) -> tuple[int, int]:
    raw_h, raw_w = raw_hw
    if raw_h % merge or raw_w % merge:
        raise ValueError("original grid must be divisible by merge_size")
    if spatial == 1:
        return raw_hw
    return (
        ((raw_h // merge + spatial - 1) // spatial) * merge,
        ((raw_w // merge + spatial - 1) // spatial) * merge,
    )


class SlowFastBudget:
    """Validated diagnostic calibration for independent MuJoCo evaluation."""

    @classmethod
    def from_report(
        cls,
        payload: dict[str, Any],
        *,
        path: str | Path = "<in-memory>",
        sha256: str = "",
    ) -> SlowFastBudget:
        """Validate an in-memory report, useful for checkpoint-free tests."""
        return cls(Path(path), _required_dict(payload, "diagnostic"), sha256)

    def __init__(self, path: Path, payload: dict[str, Any], sha256: str) -> None:
        self.diagnostic_path = path
        self.diagnostic_sha256 = sha256
        if payload.get("pool_stage") != "post_vit":
            raise ValueError("diagnostic must use post_vit pooling")
        if payload.get("processor_class") != "VLNQwen3VLProcessor":
            raise ValueError("diagnostic must come from VLNQwen3VLProcessor")

        raw_tiers = payload.get("slowfast_tiers")
        self.tiers = validate_slowfast_tiers(raw_tiers)
        video_size = payload.get("video_size_hw")
        if not isinstance(video_size, list) or len(video_size) != 2:
            raise ValueError("video_size_hw must be [height, width]")
        height, width = (_integer(part, "video_size_hw", minimum=1) for part in video_size)
        self.video_size_hw = (height, width)
        loaded = _required_dict(payload.get("loaded_checkpoint_video_processor"), "loaded_checkpoint_video_processor")
        effective = _required_dict(payload.get("effective_inference_video_processor"), "effective_inference_video_processor")
        for key in ("patch_size", "merge_size", "temporal_patch_size"):
            if loaded.get(key) != effective.get(key):
                raise ValueError(f"checkpoint and effective video processor disagree on {key}")
        self.patch_size = _integer(effective.get("patch_size"), "patch_size", minimum=1)
        self.merge_size = _integer(effective.get("merge_size"), "merge_size", minimum=1)
        self.temporal_patch_size = _integer(effective.get("temporal_patch_size"), "temporal_patch_size", minimum=1)
        episodes = payload.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("episodes must be a non-empty list")
        # The real video processor may resize before patchification. Its measured
        # original grid is the calibration; video_size / patch_size is not always it.
        first_episode = _required_dict(episodes[0], "episodes[0]")
        first_tiers = first_episode.get("tiers")
        if not isinstance(first_tiers, list) or not first_tiers:
            raise ValueError("episodes[0].tiers must be a non-empty list")
        first_tier = _required_dict(first_tiers[0], "episodes[0].tiers[0]")
        first_grid = _grid(first_tier.get("original_grid_thw"), "episodes[0].tiers[0].original_grid_thw")
        self.raw_hw = tuple(first_grid[1:])
        if self.raw_hw[0] % self.merge_size or self.raw_hw[1] % self.merge_size:
            raise ValueError("measured original grid must be divisible by merge_size")
        lengths: set[int] = set()
        observed_tiers: set[int] = set()
        for episode in episodes:
            self._validate_episode(_required_dict(episode, "episode"), lengths, observed_tiers)
        missing = set(REQUIRED_LENGTHS) - lengths
        if missing:
            raise ValueError(f"diagnostic is missing episode lengths {sorted(missing)}")
        missing_tiers = set(range(len(self.tiers))) - observed_tiers
        if missing_tiers:
            raise ValueError(f"diagnostic never measured tiers {sorted(missing_tiers)}")

    def _validate_episode(
        self,
        episode: dict[str, Any],
        lengths: set[int],
        observed_tiers: set[int],
    ) -> None:
        length = _integer(episode.get("episode_length"), "episode_length", minimum=1)
        if length in lengths:
            raise ValueError(f"duplicate diagnostic episode length {length}")
        lengths.add(length)
        expected_segments = slowfast_segments(length - 1, length, self.tiers)
        records = episode.get("tiers")
        if not isinstance(records, list) or len(records) != len(expected_segments):
            raise ValueError(f"episode {length}: tier count differs from SlowFast")
        raw_grids: list[list[int]] = []
        pooled_grids: list[list[int]] = []
        selected_count = 0
        for index, (record_value, selected) in enumerate(zip(records, expected_segments)):
            context = f"episode {length} tier {index}"
            record = _required_dict(record_value, context)
            tier_index = selected["tier"]
            observed_tiers.add(tier_index)
            _required_count(record, "tier_index", tier_index, context)
            if record.get("tier_name") != self.tiers[tier_index]["name"]:
                raise ValueError(f"{context}: tier_name differs from SlowFast")
            indices = record.get("frame_indices")
            if indices != selected["frame_ids"] or not isinstance(indices, list):
                raise ValueError(f"{context}: frame_indices differ from SlowFast")
            _required_count(record, "frame_count", len(indices), context)
            _required_count(record, "pool_spatial", selected["pool_spatial"], context)
            if record.get("pool_mode") != selected["pool_mode"]:
                raise ValueError(f"{context}: pool_mode differs from SlowFast")
            temporal = (len(indices) + self.temporal_patch_size - 1) // self.temporal_patch_size
            raw = _grid(record.get("original_grid_thw"), f"{context}.original_grid_thw")
            pooled = _grid(record.get("pooled_grid_thw"), f"{context}.pooled_grid_thw")
            if raw != [temporal, *self.raw_hw]:
                raise ValueError(f"{context}: original grid disagrees with video processor")
            expected_pool_hw = _pooled_hw(self.raw_hw, selected["pool_spatial"], self.merge_size)
            if pooled != [temporal, *expected_pool_hw]:
                raise ValueError(f"{context}: pooled grid disagrees with post-ViT pooling")
            raw_grids.append(raw)
            pooled_grids.append(pooled)
            selected_count += len(indices)
            _required_count(record, "vit_patch_rows", grid_patch_rows([raw]), context)
            _required_count(record, "vit_output_visual_tokens_before_pool", grid_visual_tokens([raw], self.merge_size), context)
            _required_count(record, "llm_visual_tokens_after_pool", grid_visual_tokens([pooled], self.merge_size), context)
        if _grid_list(episode.get("original_video_grid_thw"), f"episode {length}.original_video_grid_thw") != raw_grids:
            raise ValueError(f"episode {length}: aggregate original grids differ from tier grids")
        if _grid_list(episode.get("pooled_video_grid_thw"), f"episode {length}.pooled_video_grid_thw") != pooled_grids:
            raise ValueError(f"episode {length}: aggregate pooled grids differ from tier grids")
        _required_count(episode, "vit_patch_rows", grid_patch_rows(raw_grids), f"episode {length}")
        _required_count(episode, "vit_output_visual_tokens_before_pool", grid_visual_tokens(raw_grids, self.merge_size), f"episode {length}")
        pooled_count = grid_visual_tokens(pooled_grids, self.merge_size)
        _required_count(episode, "llm_visual_tokens_after_pool", pooled_count, f"episode {length}")
        _required_count(episode, "expanded_video_placeholder_tokens", pooled_count, f"episode {length}")
        if episode.get("placeholder_matches_pooled_grid") is not True:
            raise ValueError(f"episode {length}: processor placeholder cross-check failed")
        estimate = self.estimate(length)
        if estimate["selected_frame_count"] != selected_count:
            raise ValueError(f"episode {length}: selected frame total differs from SlowFast")
        for key in ("vit_patch_rows", "vit_output_visual_tokens_before_pool", "llm_visual_tokens_after_pool"):
            if estimate[key] != episode[key]:
                raise ValueError(f"episode {length}: projected {key} differs from measured diagnostic")

    def estimate(self, history_frames: int) -> dict[str, int | str]:
        """Project token counts from calibrated grids; no runtime processor call."""
        length = _integer(history_frames, "history_frames", minimum=1)
        segments = slowfast_segments(length - 1, length, self.tiers)
        raw_grids: list[list[int]] = []
        pooled_grids: list[list[int]] = []
        for segment in segments:
            temporal = (len(segment["frame_ids"]) + self.temporal_patch_size - 1) // self.temporal_patch_size
            raw_grids.append([temporal, *self.raw_hw])
            pooled_hw = _pooled_hw(self.raw_hw, segment["pool_spatial"], self.merge_size)
            pooled_grids.append([temporal, *pooled_hw])
        return {
            "selected_frame_count": sum(len(segment["frame_ids"]) for segment in segments),
            "vit_patch_rows": grid_patch_rows(raw_grids),
            "vit_output_visual_tokens_before_pool": grid_visual_tokens(raw_grids, self.merge_size),
            "llm_visual_tokens_after_pool": grid_visual_tokens(pooled_grids, self.merge_size),
            "token_count_source": "diagnostic_calibrated_projection",
        }


def load_budget(path: str | Path) -> SlowFastBudget:
    """Load and strictly validate a local inspect_slowfast_budget.py JSON report."""
    diagnostic_path = Path(path).expanduser()
    raw = diagnostic_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    return SlowFastBudget.from_report(
        payload, path=diagnostic_path, sha256=hashlib.sha256(raw).hexdigest()
    )
