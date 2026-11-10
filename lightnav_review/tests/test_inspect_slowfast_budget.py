"""CPU-only, checkpoint-free checks for the SlowFast budget diagnostic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lightnav.slowfast import (  # noqa: E402
    DEFAULT_SLOWFAST_TIERS_SPAN,
    slowfast_segments,
    validate_slowfast_tiers,
)
from scripts.inspect_slowfast_budget import (  # noqa: E402
    EPISODE_LENGTHS,
    check_processor_counts,
    generation_prompt_length,
    grid_patch_rows,
    grid_visual_tokens,
)


class SlowFastBudgetDiagnosticTests(unittest.TestCase):
    def test_known_example_tier_frame_counts_at_requested_lengths(self) -> None:
        # These are the repository's example span tiers, not the user's checkpoint.
        expected = {
            2: [(0, 2)],
            10: [(1, 8), (0, 2)],
            34: [(4, 2), (2, 6), (1, 16), (0, 2)],
            90: [(4, 2), (2, 24), (1, 16), (0, 2)],
            120: [(4, 2), (3, 26), (2, 24), (1, 16), (0, 2)],
            256: [(4, 2), (3, 26), (2, 24), (1, 16), (0, 2)],
        }
        tiers = validate_slowfast_tiers(DEFAULT_SLOWFAST_TIERS_SPAN)
        self.assertEqual(tuple(expected), EPISODE_LENGTHS)
        for length in EPISODE_LENGTHS:
            with self.subTest(length=length):
                segments = slowfast_segments(length - 1, length, tiers)
                self.assertEqual(
                    [(segment["tier"], len(segment["frame_ids"])) for segment in segments],
                    expected[length],
                )
                self.assertEqual(segments[-1]["frame_ids"][-1], length - 1)
                self.assertTrue(all(len(segment["frame_ids"]) % 2 == 0 for segment in segments))

    def test_post_vit_grid_counts_keep_patch_rows_separate(self) -> None:
        # 256x448 with patch_size=16 gives raw H,W=(16,28), merge_size=2.
        # Post-ViT pool=4 uses ceil(14/4)=4 merged columns, hence 8 tokens.
        self.assertEqual(grid_patch_rows([[1, 16, 28]]), 448)
        self.assertEqual(grid_visual_tokens([[1, 16, 28]], merge_size=2), 112)
        self.assertEqual(grid_visual_tokens([[1, 8, 14]], merge_size=2), 28)
        self.assertEqual(grid_visual_tokens([[1, 4, 8]], merge_size=2), 8)
        self.assertEqual(
            check_processor_counts(
                [[2, 16, 28]], [[2, 8, 14]],
                merge_size=2, pixel_rows=896, placeholder_tokens=56,
            ),
            (896, 224, 56),
        )

    def test_processor_cross_checks_reject_inconsistent_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "ViT patch rows"):
            check_processor_counts(
                [[1, 16, 28]], [[1, 8, 14]],
                merge_size=2, pixel_rows=112, placeholder_tokens=28,
            )
        with self.assertRaisesRegex(ValueError, "expanded video placeholders"):
            check_processor_counts(
                [[1, 16, 28]], [[1, 8, 14]],
                merge_size=2, pixel_rows=448, placeholder_tokens=112,
            )
        with self.assertRaisesRegex(ValueError, "invalid"):
            grid_visual_tokens([[1, 15, 28]], merge_size=2)

    def test_generation_length_excludes_placeholder_answer(self) -> None:
        self.assertEqual(generation_prompt_length([-100, -100, -100, 42, 43], -100), 3)
        with self.assertRaisesRegex(ValueError, "answer boundary"):
            generation_prompt_length([-100, -100], -100)


if __name__ == "__main__":
    unittest.main()
