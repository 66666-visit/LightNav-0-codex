"""C/D CLI contract: the three D extensions are the only differing settings."""

from __future__ import annotations

import importlib.util
import unittest
from dataclasses import asdict

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(_HAS_NUMPY, "NumPy and the inference test environment are required")
class RGMConfigTests(unittest.TestCase):
    def test_c_d_differ_only_in_the_three_extensions(self) -> None:
        from lightnav.cli.eval_habitat import build_parser, config_from_args

        base = [
            "--model_path", "/model", "--memory_enabled",
            "--memory_strategy", "hierarchical", "--memory_layers", "short,mid,long",
            "--memory_place_enabled", "--memory_place_model_path", "/dino",
            "--memory_extractor", "yolo11", "--memory_extractor_model_path", "/yolo.pt",
            "--rgm_enabled", "--rgm_prompt_max_records", "3", "--rgm_prompt_max_chars", "400",
        ]
        parser = build_parser()
        c = config_from_args(parser.parse_args(base))
        d = config_from_args(parser.parse_args(base + [
            "--rgm_yolo_temporal_enabled", "--rgm_place_two_stage_enabled",
            "--rgm_topology_motion_validation_enabled",
        ]))
        self.assertEqual(c.memory.rgm.prompt_max_records, 3)
        self.assertEqual(c.max_new_tokens, d.max_new_tokens)
        self.assertEqual(c.num_history_frames, d.num_history_frames)
        self.assertEqual(c.memory.extract_every_n_steps, 5)
        self.assertEqual(c.memory.place_extract_every_n_steps, 5)
        c_memory = asdict(c.memory)
        d_memory = asdict(d.memory)
        c_rgm, d_rgm = c_memory.pop("rgm"), d_memory.pop("rgm")
        self.assertEqual(c_memory, d_memory)
        differences = {key for key in c_rgm if c_rgm[key] != d_rgm[key]}
        self.assertEqual(differences, {
            "yolo_temporal_enabled", "place_two_stage_enabled",
            "topology_motion_validation_enabled",
        })


if __name__ == "__main__":
    unittest.main()
