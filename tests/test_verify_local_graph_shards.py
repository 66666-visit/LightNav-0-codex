"""Local Graph shard validation uses only saved protocol metadata."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_local_graph_shards import verify_local_graph_shards


class VerifyLocalGraphShardsTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.effective = {
            "enabled": True, "scope": "episode_local", "prompt_max_chars": 320,
            "max_landmarks": 2, "reactivation_steps": 8,
            "neighbor_radius_m": 3.0,
            "fusion_mode": "legacy",
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

    def test_rejects_changed_shard_protocol_without_rewriting_it(self) -> None:
        marker_path = self.root / "shard_1" / "local_graph_protocol.json"
        changed = {**self.marker, "max_landmarks": 9}
        original = json.dumps(changed)
        marker_path.write_text(original)
        with self.assertRaisesRegex(ValueError, "differs from root snapshot"):
            verify_local_graph_shards(self.root, ["2", "0"])
        self.assertEqual(marker_path.read_text(), original)

    def test_spatial_mode_requires_matching_protocol_on_every_shard(self) -> None:
        snapshot_path = self.root / "config_snapshot.yaml"
        snapshot = json.loads(snapshot_path.read_text())
        spatial = {
            **self.effective, "fusion_mode": "spatial",
            "protocol_version": "local_graph_spatial_v1",
        }
        snapshot["protocol_version"] = spatial["protocol_version"]
        snapshot["local_graph_effective"] = spatial
        snapshot["memory_effective"]["local_graph_effective"] = spatial
        snapshot_path.write_text(json.dumps(snapshot))
        with self.assertRaisesRegex(ValueError, "differs from root snapshot"):
            verify_local_graph_shards(self.root, ["2", "0"])
        marker = {key: value for key, value in spatial.items() if key != "enabled"}
        for index in range(2):
            (self.root / f"shard_{index}" / "local_graph_protocol.json").write_text(
                json.dumps(marker)
            )
        verify_local_graph_shards(self.root, ["2", "0"])


if __name__ == "__main__":
    unittest.main()
