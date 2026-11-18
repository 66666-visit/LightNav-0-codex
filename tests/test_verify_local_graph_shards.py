"""Local Graph shard validation uses only saved protocol metadata."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_local_graph_shards import verify_local_graph_shards

_PROMPT_MODES = (
    "l1_original_only", "l1_organized_only", "l3_original_only",
    "l3_organized_only", "l1_l3_original_concat", "l1_l3_joint",
    "l1_original_l3_organized_concat", "l1_original_l3_organized_joint",
    "l3_official_style_only",
)
_PROMPT_PROTOCOL = "l1_l3_prompt_ablation_v1"


class VerifyLocalGraphShardsTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.effective = {
            "enabled": True, "scope": "episode_local", "prompt_max_chars": 320,
            "max_landmarks": 2, "reactivation_steps": 8,
            "protocol_version": "local_graph_v1",
        }
        snapshot = {
            "protocol_version": "local_graph_v1",
            "common": {"gpu_ids": ["2", "0"]},
            "local_graph_effective": self.effective,
            "memory_effective": {"local_graph_effective": self.effective},
        }
        (self.root / "config_snapshot.yaml").write_text(json.dumps(snapshot))
        self.marker = {key: value for key, value in self.effective.items() if key != "enabled"}
        for index in range(2):
            shard = self.root / f"shard_{index}"
            shard.mkdir()
            (shard / "local_graph_protocol.json").write_text(json.dumps(self.marker))

    def test_two_gpu_shards_match_root_snapshot(self) -> None:
        verify_local_graph_shards(self.root, ["2", "0"])

    def test_rejects_gpu_order_and_missing_or_extra_shard(self) -> None:
        with self.assertRaisesRegex(ValueError, "GPU order"):
            verify_local_graph_shards(self.root, ["0", "2"])
        (self.root / "shard_1" / "local_graph_protocol.json").unlink()
        with self.assertRaisesRegex(ValueError, "lacks protocol marker"):
            verify_local_graph_shards(self.root, ["2", "0"])
        (self.root / "shard_1" / "local_graph_protocol.json").write_text(json.dumps(self.marker))
        (self.root / "shard_2").mkdir()
        with self.assertRaisesRegex(ValueError, "shard set differs"):
            verify_local_graph_shards(self.root, ["2", "0"])

    def _write_prompt_snapshot(
        self, mode: str, *, old_shape: bool = False,
        prompt_protocol: str = _PROMPT_PROTOCOL,
    ) -> None:
        prompt_effective = {
            "prompt_mode": mode,
            "protocol_version": _PROMPT_PROTOCOL,
            "budget_strategy": "token_aware_context",
            "original_char_budget_chars": 320,
        }
        snapshot = {
            "protocol_version": (
                _PROMPT_PROTOCOL if old_shape else "local_graph_v1"
            ),
            "prompt_protocol_version": prompt_protocol,
            "common": {"gpu_ids": ["2", "0"]},
            "local_graph_effective": self.effective,
            "memory_effective": {
                "local_graph_effective": self.effective,
                "prompt_effective": prompt_effective,
            },
            "prompt_effective": prompt_effective,
        }
        (self.root / "config_snapshot.yaml").write_text(json.dumps(snapshot))

    def test_six_non_legacy_prompt_modes_keep_one_local_graph_protocol(self) -> None:
        for mode in _PROMPT_MODES:
            with self.subTest(mode=mode):
                self._write_prompt_snapshot(mode)
                verify_local_graph_shards(self.root, ["2", "0"])

    def test_old_prompt_snapshot_conflict_shape_is_recoverable(self) -> None:
        for mode in _PROMPT_MODES:
            with self.subTest(mode=mode):
                self._write_prompt_snapshot(mode, old_shape=True)
                verify_local_graph_shards(self.root, ["2", "0"])

    def test_prompt_snapshot_without_matching_prompt_block_is_rejected(self) -> None:
        self._write_prompt_snapshot("l1_l3_joint")
        snapshot = json.loads((self.root / "config_snapshot.yaml").read_text())
        snapshot["prompt_protocol_version"] = "different_protocol"
        (self.root / "config_snapshot.yaml").write_text(json.dumps(snapshot))
        with self.assertRaisesRegex(ValueError, "lacks one enabled Local Graph protocol"):
            verify_local_graph_shards(self.root, ["2", "0"])

        self._write_prompt_snapshot("l1_l3_joint", old_shape=True)
        snapshot = json.loads((self.root / "config_snapshot.yaml").read_text())
        snapshot.pop("prompt_effective")
        (self.root / "config_snapshot.yaml").write_text(json.dumps(snapshot))
        with self.assertRaisesRegex(ValueError, "lacks one enabled Local Graph protocol"):
            verify_local_graph_shards(self.root, ["2", "0"])

    def test_rejects_null_or_incomplete_effective_marker_values(self) -> None:
        snapshot = json.loads((self.root / "config_snapshot.yaml").read_text())
        for key, value in (
            ("protocol_version", None),
            ("scope", None),
            ("prompt_max_chars", 0),
            ("max_landmarks", None),
            ("reactivation_steps", -1),
        ):
            with self.subTest(key=key):
                mutated = json.loads(json.dumps(snapshot))
                mutated["local_graph_effective"][key] = value
                mutated["memory_effective"]["local_graph_effective"] = (
                    mutated["local_graph_effective"]
                )
                (self.root / "config_snapshot.yaml").write_text(json.dumps(mutated))
                with self.assertRaisesRegex(
                    ValueError, "lacks one enabled Local Graph protocol"
                ):
                    verify_local_graph_shards(self.root, ["2", "0"])

    def test_rejects_changed_shard_protocol_without_rewriting_it(self) -> None:
        marker_path = self.root / "shard_1" / "local_graph_protocol.json"
        changed = {**self.marker, "max_landmarks": 9}
        original = json.dumps(changed)
        marker_path.write_text(original)
        with self.assertRaisesRegex(ValueError, "differs from root snapshot"):
            verify_local_graph_shards(self.root, ["2", "0"])
        self.assertEqual(marker_path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
