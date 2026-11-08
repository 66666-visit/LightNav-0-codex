"""Deterministic pilot sampling tests using synthetic R2R dataset metadata."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts import sample_r2r_pilot as pilot


class SampleR2RPilotTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "R2R_VLNCE_v1-3_preprocessed" / "val_unseen"
        self.source.mkdir(parents=True)
        self.dataset = self.source / "val_unseen.json.gz"
        self.episodes = [
            {
                "scene_id": f"data/scene_datasets/mp3d/scene_{scene}/scene_{scene}.glb",
                "episode_id": scene * 1000 + index,
                "trajectory_id": f"scene_{scene}_traj_{index // 2}",
                "info": {"geodesic_distance": float(index + 1)},
            }
            for scene in range(11)
            for index in range(36)
        ]
        self._write()

    def _write(self) -> None:
        with gzip.open(self.dataset, "wt", encoding="utf-8") as stream:
            json.dump({"episodes": self.episodes}, stream)

    def _sample(self) -> dict:
        return pilot.sample_pilot(self.root, scenes_dir="/scenes")

    def test_stratified_11_scene_220_selection_and_unique_ids(self) -> None:
        manifest = self._sample()
        rows = manifest["episodes"]
        self.assertEqual(manifest["dataset_split"], "val_unseen")
        self.assertEqual(manifest["expected_count"], 220)
        self.assertEqual(len(manifest["full_scene_counts"]), 11)
        self.assertEqual(set(manifest["full_scene_counts"].values()), {36})
        self.assertEqual(set(manifest["scene_quotas"].values()), {20})
        self.assertEqual({row["reason"] for row in manifest["scene_allocation"].values()},
                         {"base_quota"})
        self.assertEqual(len({row["habitat_episode_id"] for row in rows}), 220)
        self.assertEqual(Counter(row["scene_id"] for row in rows), {
            f"/scenes/mp3d/scene_{scene}/scene_{scene}.glb": 20
            for scene in range(11)
        })
        for scene in manifest["full_scene_counts"]:
            chosen = [row for row in rows if row["scene_id"] == scene]
            self.assertEqual(Counter(row["distance_group"] for row in chosen), {
                "short": 7, "medium": 7, "long": 6,
            })
            self.assertTrue(all(
                row["route_length"] <= 12 for row in chosen
                if row["distance_group"] == "short"
            ))
            self.assertGreaterEqual(len({row["trajectory_id"] for row in chosen}), 18)
            self.assertTrue(all(row["sample_seed"] == 42 for row in chosen))
        self.assertEqual(len(manifest["source_sha256"]), 64)

    def test_fixed_seed_is_reproducible_independent_of_dataset_order(self) -> None:
        first = self._sample()
        self.episodes.reverse()
        self._write()
        second = self._sample()
        self.assertEqual(
            [row["habitat_episode_id"] for row in first["episodes"]],
            [row["habitat_episode_id"] for row in second["episodes"]],
        )
        self.assertNotEqual(first["source_sha256"], second["source_sha256"])

    def test_missing_distance_or_trajectory_fails(self) -> None:
        del self.episodes[0]["info"]["geodesic_distance"]
        self._write()
        with self.assertRaisesRegex(ValueError, "lacks geodesic_distance"):
            self._sample()
        self.episodes[0]["info"]["geodesic_distance"] = 1.0
        del self.episodes[0]["trajectory_id"]
        self._write()
        with self.assertRaisesRegex(ValueError, "lacks scene_id, episode_id, or trajectory_id"):
            self._sample()

    def test_inadequate_scene_fails_and_twenty_candidates_are_valid(self) -> None:
        self.episodes = [row for row in self.episodes if not row["scene_id"].endswith(
            "scene_10/scene_10.glb")]
        self._write()
        with self.assertRaisesRegex(ValueError, "found 10"):
            self._sample()
        self.episodes.extend({
            "scene_id": "data/scene_datasets/mp3d/scene_10/scene_10.glb",
            "episode_id": 10000 + index,
            "trajectory_id": f"traj_{index}",
            "geodesic_distance": index + 1,
        } for index in range(20))
        self._write()
        manifest = self._sample()
        scene = "/scenes/mp3d/scene_10/scene_10.glb"
        self.assertEqual(manifest["scene_quotas"][scene], 20)
        self.assertEqual(Counter(
            row["distance_group"] for row in manifest["episodes"]
            if row["scene_id"] == scene
        ), {"short": 7, "medium": 7, "long": 6})

    def test_real_1839_distribution_redistributes_two_seats(self) -> None:
        # Mirrors the confirmed val_unseen shape: pLe4wQe7qrG has 18 rows.
        # The other counts sum to 1,821; the two largest candidate pools
        # receive one additional seat each, retaining exactly 220 samples.
        counts = {"pLe4wQe7qrG": 18, "scene_00": 187, "scene_01": 186}
        counts.update({f"scene_{index:02d}": 181 for index in range(2, 10)})
        self.assertEqual(sum(counts.values()), 1839)
        self.episodes = [
            {
                "scene_id": f"data/scene_datasets/mp3d/{scene}/{scene}.glb",
                "episode_id": f"{scene}_{index}",
                "trajectory_id": f"{scene}_traj_{index // 2}",
                "info": {"geodesic_distance": float(index + 1)},
            }
            for scene, count in counts.items() for index in range(count)
        ]
        self._write()
        manifest = self._sample()
        scene_quotas = {
            Path(scene).stem: quota for scene, quota in manifest["scene_quotas"].items()
        }
        self.assertEqual(scene_quotas, {
            "pLe4wQe7qrG": 18, "scene_00": 21, "scene_01": 21,
            **{f"scene_{index:02d}": 20 for index in range(2, 10)},
        })
        self.assertEqual(len(manifest["episodes"]), 220)
        self.assertEqual(len({
            (row["scene_id"], row["habitat_episode_id"])
            for row in manifest["episodes"]
        }), 220)
        for scene_id, quota in manifest["scene_quotas"].items():
            chosen = [
                row for row in manifest["episodes"] if row["scene_id"] == scene_id
            ]
            self.assertEqual(len(chosen), quota)
            self.assertEqual(Counter(row["distance_group"] for row in chosen),
                             pilot._distance_quotas(quota))
            allocation = manifest["scene_allocation"][scene_id]
            self.assertEqual(allocation["selected_quota"], quota)
            self.assertEqual(allocation["available_episodes"],
                             manifest["full_scene_counts"][scene_id])
        short_scene = next(scene for scene in manifest["scene_quotas"]
                           if "pLe4wQe7qrG" in scene)
        self.assertEqual(manifest["scene_allocation"][short_scene]["reason"],
                         "all_available_below_base")
        for scene in ("scene_00", "scene_01"):
            key = next(scene_id for scene_id in manifest["scene_quotas"]
                       if Path(scene_id).stem == scene)
            self.assertEqual(manifest["scene_allocation"][key]["reason"],
                             "redistributed_shortfall")
        expected_ids = [row["habitat_episode_id"] for row in manifest["episodes"]]
        self.episodes.reverse()
        self._write()
        self.assertEqual([row["habitat_episode_id"] for row in self._sample()["episodes"]],
                         expected_ids)

    def test_redistribution_ties_break_by_scene_id(self) -> None:
        counts = {"deficit": 18, "scene_b": 30, "scene_a": 30, "scene_c": 30}
        quotas, allocation = pilot._allocate_scene_quotas(counts, 20)
        self.assertEqual(quotas, {"deficit": 18, "scene_a": 21,
                                  "scene_b": 21, "scene_c": 20})
        self.assertEqual(allocation["scene_c"]["reason"], "base_quota")

    def test_total_available_below_220_fails(self) -> None:
        self.episodes = self.episodes[:11]
        self._write()
        with self.assertRaisesRegex(ValueError, "found"):
            self._sample()
        counts = {f"scene_{index}": 19 for index in range(11)}
        with self.assertRaisesRegex(ValueError, "Pilot needs 220"):
            pilot._allocate_scene_quotas(counts, 20)

    def test_duplicate_source_episode_and_invalid_protocol_fail(self) -> None:
        self.episodes.append(dict(self.episodes[0]))
        self._write()
        with self.assertRaisesRegex(ValueError, "duplicate dataset Episode"):
            self._sample()
        with self.assertRaisesRegex(ValueError, "base target 20"):
            pilot.sample_pilot(self.root, per_scene=19)
        with self.assertRaisesRegex(ValueError, "seed 42"):
            pilot.sample_pilot(self.root, seed=43)

    def test_episode_id_is_scene_local(self) -> None:
        # The evaluator keys IDs by (scene_id, episode_id), not ID alone.
        self.episodes[36]["episode_id"] = self.episodes[0]["episode_id"]
        self._write()
        result = self._sample()
        self.assertEqual(result["expected_count"], 220)
        self.assertEqual(len({
            (row["scene_id"], row["habitat_episode_id"])
            for row in result["episodes"]
        }), 220)

    def test_cli_creates_fixed_manifest_and_refuses_overwrite(self) -> None:
        output = self.root / "pilot" / "pilot_manifest.json"
        args = ["--data-path", str(self.root), "--scenes-dir", "/scenes",
                "--output", str(output)]
        self.assertEqual(pilot.main(args), 0)
        self.assertEqual(json.loads(output.read_text())["expected_count"], 220)
        with self.assertRaises(FileExistsError):
            pilot.main(args)


if __name__ == "__main__":
    unittest.main()
