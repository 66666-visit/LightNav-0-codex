"""A manifest ordinal, not a numeric dataset ID, identifies Episode 1."""

from __future__ import annotations

import gzip
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.resolve_single_episode import main, resolve_episode  # noqa: E402


class ResolveSingleEpisodeTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.scenes_dir = self.root / "scenes"
        self.scene = self.scenes_dir / "mp3d" / "house" / "house.glb"
        self.scene.parent.mkdir(parents=True)
        self.scene.touch()
        self.manifest = self.root / "episode_manifest.json"
        self.dataset = self.root / "val_unseen.json.gz"
        self.episodes = [self._episode(1, "Walk straight."),
                         self._episode(9, "Turn left at the chair.")]
        self._write_dataset()

    @staticmethod
    def _episode(episode_id: int, instruction: str) -> dict:
        return {
            "episode_id": episode_id,
            "scene_id": "data/scene_datasets/mp3d/house/house.glb",
            "trajectory_id": episode_id + 100,
            "instruction": {"instruction_text": instruction},
            "start_position": [1.0, 2.0, 3.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "reference_path": [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
            "goals": [{"position": [2.0, 2.0, 3.0]}],
        }

    def _write_dataset(self) -> None:
        with gzip.open(self.dataset, "wt", encoding="utf-8") as stream:
            json.dump({"episodes": self.episodes}, stream)

    def _write_manifest(self, *, pilot: bool = True, scene: str | None = None) -> None:
        rows = [{"scene_id": scene or str(self.scene),
                 "habitat_episode_id" if pilot else "episode_id": episode_id}
                for episode_id in ("9", "1")]
        metadata = {"dataset_split" if pilot else "split": "val_unseen",
                    "expected_count": 2, "episodes": rows}
        self.manifest.write_text(json.dumps(metadata), encoding="utf-8")

    def test_pilot_first_ordinal_resolves_dataset_id_nine(self) -> None:
        self._write_manifest()

        result = resolve_episode(self.manifest, self.dataset, self.scenes_dir)

        self.assertEqual(result["manifest_ordinal"], 1)
        self.assertEqual(result["dataset_episode_index"], 1)
        self.assertEqual(result["habitat_episode_id"], "9")
        self.assertEqual(result["trajectory_id"], "109")
        self.assertEqual(result["instruction_text"], "Turn left at the chair.")
        self.assertEqual(result["start_position"], [1.0, 2.0, 3.0])
        self.assertEqual(result["start_rotation"], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(result["reference_path"], [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
        self.assertEqual(result["goal_position"], [2.0, 2.0, 3.0])
        self.assertEqual(result["server_selection_args"], [
            "--single-scene-id", str(self.scene), "--single-episode-id", "9",
        ])

    def test_evaluator_manifest_and_scene_symlink(self) -> None:
        scene_alias = self.root / "alias.glb"
        scene_alias.symlink_to(self.scene)
        self._write_manifest(pilot=False, scene=str(scene_alias))

        result = resolve_episode(self.manifest, self.dataset, self.scenes_dir,
                                 episode_number=2)

        self.assertEqual(result["habitat_episode_id"], "1")
        self.assertEqual(result["dataset_episode_index"], 0)
        self.assertEqual(result["single_scene_id"], str(scene_alias))

    def test_missing_or_duplicate_source_match_fails(self) -> None:
        self._write_manifest()
        self.episodes.pop()
        self._write_dataset()
        with self.assertRaisesRegex(ValueError, "matches=0"):
            resolve_episode(self.manifest, self.dataset, self.scenes_dir)

        self.episodes.extend([self._episode(9, "First."), self._episode(9, "Second.")])
        self._write_dataset()
        with self.assertRaisesRegex(ValueError, "matches=2"):
            resolve_episode(self.manifest, self.dataset, self.scenes_dir)

    def test_rejects_bad_ordinal_and_split(self) -> None:
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "one-based"):
            resolve_episode(self.manifest, self.dataset, self.scenes_dir, 0)
        with self.assertRaisesRegex(ValueError, "only 2"):
            resolve_episode(self.manifest, self.dataset, self.scenes_dir, 3)
        metadata = json.loads(self.manifest.read_text(encoding="utf-8"))
        metadata["dataset_split"] = "train"
        self.manifest.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "val_unseen"):
            resolve_episode(self.manifest, self.dataset, self.scenes_dir)

    def test_cli_prints_json_identity(self) -> None:
        self._write_manifest()
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--manifest", str(self.manifest), "--dataset", str(self.dataset),
                  "--scenes-dir", str(self.scenes_dir), "--episode-number", "1"])
        self.assertEqual(json.loads(output.getvalue())["single_episode_id"], "9")


if __name__ == "__main__":
    unittest.main()
