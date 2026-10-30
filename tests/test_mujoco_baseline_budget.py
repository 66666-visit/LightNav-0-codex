"""Checkpoint-free checks for the independent MuJoCo token-budget projection."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from lightnav.slowfast import DEFAULT_SLOWFAST_TIERS_SPAN, slowfast_segments, validate_slowfast_tiers  # noqa: E402
from scripts.mujoco_baseline_budget import SlowFastBudget  # noqa: E402


def synthetic_diagnostic() -> dict:
    """Small, internally consistent stand-in for a processor diagnostic."""
    tiers = validate_slowfast_tiers(DEFAULT_SLOWFAST_TIERS_SPAN)
    episodes = []
    for length in (2, 10, 34, 90, 120, 256):
        records = []
        raw_grids = []
        pooled_grids = []
        for segment in slowfast_segments(length - 1, length, tiers):
            tier_index = segment["tier"]
            frame_indices = segment["frame_ids"]
            t = len(frame_indices) // 2
            factor = segment["pool_spatial"]
            raw = [t, 16, 28]
            pooled = [t, 2 * ((8 + factor - 1) // factor), 2 * ((14 + factor - 1) // factor)]
            raw_grids.append(raw)
            pooled_grids.append(pooled)
            records.append({
                "tier_index": tier_index,
                "tier_name": tiers[tier_index]["name"],
                "frame_indices": frame_indices,
                "frame_count": len(frame_indices),
                "pool_spatial": factor,
                "pool_mode": segment["pool_mode"],
                "original_grid_thw": raw,
                "pooled_grid_thw": pooled,
                "vit_patch_rows": t * 16 * 28,
                "vit_output_visual_tokens_before_pool": t * 8 * 14,
                "llm_visual_tokens_after_pool": t * (pooled[1] // 2) * (pooled[2] // 2),
            })
        pooled_total = sum(row["llm_visual_tokens_after_pool"] for row in records)
        episodes.append({
            "episode_length": length,
            "tiers": records,
            "original_video_grid_thw": raw_grids,
            "pooled_video_grid_thw": pooled_grids,
            "vit_patch_rows": sum(row["vit_patch_rows"] for row in records),
            "vit_output_visual_tokens_before_pool": sum(
                row["vit_output_visual_tokens_before_pool"] for row in records
            ),
            "llm_visual_tokens_after_pool": pooled_total,
            "expanded_video_placeholder_tokens": pooled_total,
            "placeholder_matches_pooled_grid": True,
        })
    return {
        "pool_stage": "post_vit",
        "processor_class": "VLNQwen3VLProcessor",
        "video_size_hw": [256, 448],
        "slowfast_tiers": tiers,
        "loaded_checkpoint_video_processor": {
            "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
        },
        "effective_inference_video_processor": {
            "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
        },
        "episodes": episodes,
    }


class SlowFastBudgetTest(unittest.TestCase):
    def test_calibrated_projection_matches_known_counts(self) -> None:
        budget = SlowFastBudget.from_report(synthetic_diagnostic())
        two = budget.estimate(2)
        self.assertEqual(two["selected_frame_count"], 2)
        self.assertEqual(two["vit_patch_rows"], 448)
        self.assertEqual(two["vit_output_visual_tokens_before_pool"], 112)
        self.assertEqual(two["llm_visual_tokens_after_pool"], 112)

        ten = budget.estimate(10)
        self.assertEqual(ten["selected_frame_count"], 10)
        self.assertEqual(ten["vit_patch_rows"], 2240)
        self.assertEqual(ten["vit_output_visual_tokens_before_pool"], 560)
        self.assertEqual(ten["llm_visual_tokens_after_pool"], 224)
        self.assertEqual(ten["token_count_source"], "diagnostic_calibrated_projection")

    def test_projection_handles_unmeasured_length(self) -> None:
        budget = SlowFastBudget.from_report(synthetic_diagnostic())
        one = budget.estimate(1)
        self.assertEqual(one["selected_frame_count"], 2)  # dense tier pads to a pair
        self.assertEqual(one["llm_visual_tokens_after_pool"], 112)

    def test_uses_measured_grid_when_nominal_size_needs_processor_resize(self) -> None:
        report = synthetic_diagnostic()
        report["video_size_hw"] = [255, 447]
        report["loaded_checkpoint_video_processor"]["patch_size"] = 14
        report["effective_inference_video_processor"]["patch_size"] = 14
        budget = SlowFastBudget.from_report(report)
        self.assertEqual(budget.estimate(10)["llm_visual_tokens_after_pool"], 224)

        bad_grid = copy.deepcopy(report)
        bad_grid["episodes"][1]["tiers"][0]["original_grid_thw"][1] += 2
        with self.assertRaisesRegex(ValueError, "original grid"):
            SlowFastBudget.from_report(bad_grid)

    def test_rejects_corrupted_frame_indices_and_token_count(self) -> None:
        report = synthetic_diagnostic()
        bad_indices = copy.deepcopy(report)
        bad_indices["episodes"][0]["tiers"][0]["frame_indices"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "frame_indices"):
            SlowFastBudget.from_report(bad_indices)

        bad_count = copy.deepcopy(report)
        bad_count["episodes"][0]["expanded_video_placeholder_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "expanded_video_placeholder_tokens"):
            SlowFastBudget.from_report(bad_count)


if __name__ == "__main__":
    unittest.main()
