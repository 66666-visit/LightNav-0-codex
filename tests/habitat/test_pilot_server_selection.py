"""CPU checks that the Habitat server filters Pilot Episodes before sharding.

The real Habitat simulator is intentionally absent. Fake Episodes and a fake
iterator exercise the exact server-side selection methods used at startup.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


_SERVER_ROOT = Path(__file__).resolve().parents[2] / "habitat_server"
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

# Only these pure selection methods run here. In a minimal CPU test environment,
# the wrapper can be imported without installing NumPy or Habitat.
if importlib.util.find_spec("numpy") is None:
    with patch.dict(sys.modules, {"numpy": types.ModuleType("numpy")}):
        from lightnav_habitat.vlnce import VLNCEEnv
else:
    from lightnav_habitat.vlnce import VLNCEEnv


class FakeHabitatEnv:
    def __init__(self, episodes: list[types.SimpleNamespace]) -> None:
        self._episodes = list(episodes)
        self._dataset = types.SimpleNamespace(episodes=self._episodes)
        self.setter_calls = 0
        self.iterator_index = 0

    @property
    def episodes(self) -> list[types.SimpleNamespace]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: list[types.SimpleNamespace]) -> None:
        self._episodes = list(episodes)
        self._dataset.episodes = self._episodes
        self.setter_calls += 1
        self.iterator_index = 0

    def reset(self) -> types.SimpleNamespace:
        episode = self._episodes[self.iterator_index % len(self._episodes)]
        self.iterator_index += 1
        return episode


def _all_episodes() -> list[types.SimpleNamespace]:
    return [
        types.SimpleNamespace(
            scene_id=f"scene_{index % 11:02d}", episode_id=f"episode_{index:04d}",
        )
        for index in range(1839)
    ]


def _sample_rows(episodes: list[types.SimpleNamespace]) -> list[dict[str, str]]:
    rows = []
    for scene in range(11):
        matches = [ep for ep in episodes if ep.scene_id == f"scene_{scene:02d}"]
        rows.extend({
            "scene_id": ep.scene_id, "habitat_episode_id": ep.episode_id,
        } for ep in matches[:20])
    return rows


def _keys(episodes: list[types.SimpleNamespace]) -> set[tuple[str, str]]:
    return {(ep.scene_id, ep.episode_id) for ep in episodes}


class PilotServerSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.all_episodes = _all_episodes()
        self.rows = _sample_rows(self.all_episodes)
        self.manifest = self.root / "pilot_manifest.json"
        self._write_manifest(self.rows)

    def _write_manifest(self, rows: list[dict[str, str]]) -> None:
        self.manifest.write_text(json.dumps({
            "dataset_split": "val_unseen", "expected_count": 220, "episodes": rows,
        }), encoding="utf-8")

    def _wrapper(self, shard: int | None = None) -> VLNCEEnv:
        wrapper = VLNCEEnv.__new__(VLNCEEnv)
        wrapper.pilot_manifest_path = str(self.manifest)
        wrapper.split_id = shard
        wrapper.split_num = 2 if shard is not None else None
        return wrapper

    def test_selected_220_only_are_executable_and_shards_are_disjoint(self) -> None:
        expected = {(row["scene_id"], row["habitat_episode_id"]) for row in self.rows}
        shard_keys = []
        for shard_index in range(2):
            env = FakeHabitatEnv(self.all_episodes)
            wrapper = self._wrapper(shard_index)
            wrapper._apply_pilot_selection(env)
            self.assertEqual(len(env.episodes), 220)
            self.assertEqual(_keys(env.episodes), expected)
            wrapper._apply_episode_split(env)
            self.assertEqual(len(env.episodes), 110)
            self.assertEqual(env.setter_calls, 2)
            executed = [env.reset() for _ in range(110)]
            self.assertEqual(_keys(executed), _keys(env.episodes))
            self.assertTrue(_keys(executed) <= expected)
            shard_keys.append(_keys(executed))
        self.assertFalse(shard_keys[0] & shard_keys[1])
        self.assertEqual(shard_keys[0] | shard_keys[1], expected)

    def test_rgm_88_selection_precedes_sharding_and_rejects_bad_quota(self) -> None:
        rows = []
        for scene in range(11):
            scene_rows = [row for row in self.rows
                          if row["scene_id"] == f"scene_{scene:02d}"][:8]
            for index, row in enumerate(scene_rows):
                rows.append({**row, "distance_group": (
                    "short" if index < 3 else "medium" if index < 6 else "long"
                )})
        self.manifest.write_text(json.dumps({
            "protocol_version": "rgm_pilot_88_v1", "dataset_split": "val_unseen",
            "sample_seed": 42, "expected_count": 88, "per_scene": 8,
            "episodes": rows,
        }), encoding="utf-8")
        env = FakeHabitatEnv(self.all_episodes)
        wrapper = self._wrapper(0)
        wrapper._apply_pilot_selection(env)
        self.assertEqual(len(env.episodes), 88)
        wrapper._apply_episode_split(env)
        self.assertEqual(len(env.episodes), 44)
        self.assertEqual(env.setter_calls, 2)
        rows[0]["distance_group"] = "long"
        self.manifest.write_text(json.dumps({
            "protocol_version": "rgm_pilot_88_v1", "dataset_split": "val_unseen",
            "sample_seed": 42, "expected_count": 88, "per_scene": 8,
            "episodes": rows,
        }), encoding="utf-8")
        invalid = FakeHabitatEnv(self.all_episodes)
        with self.assertRaisesRegex(ValueError, "3/3/2"):
            self._wrapper()._apply_pilot_selection(invalid)
        self.assertEqual(invalid.setter_calls, 0)

    def test_pilot_missing_or_duplicate_id_fails_before_iterator_change(self) -> None:
        env = FakeHabitatEnv(self.all_episodes)
        changed = list(self.rows)
        changed[0] = {"scene_id": "scene_00", "habitat_episode_id": "unknown"}
        self._write_manifest(changed)
        with self.assertRaisesRegex(ValueError, "missing="):
            self._wrapper()._apply_pilot_selection(env)
        self.assertEqual(env.setter_calls, 0)

        changed[0] = dict(changed[1])
        self._write_manifest(changed)
        with self.assertRaisesRegex(ValueError, "duplicate Episode"):
            self._wrapper()._apply_pilot_selection(env)
        self.assertEqual(env.setter_calls, 0)

    def test_without_pilot_manifest_the_full_split_is_left_intact(self) -> None:
        env = FakeHabitatEnv(self.all_episodes)
        wrapper = self._wrapper()
        wrapper.pilot_manifest_path = None
        wrapper._apply_pilot_selection(env)
        self.assertEqual(len(env.episodes), 1839)
        self.assertEqual(env.setter_calls, 0)

        first = FakeHabitatEnv(self.all_episodes)
        second = FakeHabitatEnv(self.all_episodes)
        for index, shard_env in enumerate((first, second)):
            shard_wrapper = self._wrapper(index)
            shard_wrapper.pilot_manifest_path = None
            shard_wrapper._apply_episode_split(shard_env)
        self.assertEqual(len(first.episodes), 919)
        self.assertEqual(len(second.episodes), 920)
        self.assertEqual(_keys(first.episodes) | _keys(second.episodes),
                         _keys(self.all_episodes))
        self.assertFalse(_keys(first.episodes) & _keys(second.episodes))


class PilotScenePathNormalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        data = self.root / "actual_data"
        data.mkdir()
        (self.root / "data").symlink_to(data, target_is_directory=True)
        previous_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous_cwd)

        self.scene_paths = []
        for scene in range(11):
            path = data / "scene_datasets" / "mp3d" / f"scene_{scene:02d}" / "scene.glb"
            path.parent.mkdir(parents=True)
            path.touch()
            self.scene_paths.append(path)
        self.episodes = [
            types.SimpleNamespace(
                scene_id=f"data/scene_datasets/mp3d/scene_{index % 11:02d}/scene.glb",
                episode_id=f"episode_{index:04d}",
            )
            for index in range(1839)
        ]
        self.rows = []
        for scene, path in enumerate(self.scene_paths):
            episodes = [ep for ep in self.episodes
                        if ep.scene_id.endswith(f"scene_{scene:02d}/scene.glb")][:8]
            for index, episode in enumerate(episodes):
                self.rows.append({
                    "scene_id": str(path), "habitat_episode_id": episode.episode_id,
                    "distance_group": "short" if index < 3 else "medium" if index < 6 else "long",
                })
        self.manifest = self.root / "pilot_manifest.json"
        self._write_manifest(self.rows)

    def _write_manifest(self, rows: list[dict[str, str]]) -> None:
        self.manifest.write_text(json.dumps({
            "protocol_version": "rgm_pilot_88_v1", "dataset_split": "val_unseen",
            "sample_seed": 42, "expected_count": 88, "per_scene": 8,
            "episodes": rows,
        }), encoding="utf-8")

    def _select(self, episodes: list[types.SimpleNamespace] | None = None) -> FakeHabitatEnv:
        env = FakeHabitatEnv(self.episodes if episodes is None else episodes)
        wrapper = VLNCEEnv.__new__(VLNCEEnv)
        wrapper.pilot_manifest_path = str(self.manifest)
        wrapper._apply_pilot_selection(env)
        return env

    def test_absolute_manifest_matches_relative_symlink_scene_paths(self) -> None:
        env = self._select()
        self.assertEqual(len(env.episodes), 88)
        self.assertEqual(env.setter_calls, 1)
        self.assertEqual({ep.episode_id for ep in env.episodes},
                         {row["habitat_episode_id"] for row in self.rows})
        self.assertTrue(all(ep.scene_id.startswith("data/") for ep in env.episodes))

    def test_different_episode_id_still_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        rows[0]["habitat_episode_id"] = "different_episode"
        self._write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "missing="):
            self._select()

    def test_different_scene_file_still_rejected(self) -> None:
        other_scene = self.scene_paths[0].with_name("other.glb")
        other_scene.touch()
        rows = [dict(row) for row in self.rows]
        for row in rows[:8]:
            row["scene_id"] = str(other_scene)
        self._write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "missing="):
            self._select()

    def test_duplicate_episode_with_scene_alias_still_rejected(self) -> None:
        rows = [dict(row) for row in self.rows]
        rows[0]["scene_id"] = self.episodes[0].scene_id
        rows[0]["habitat_episode_id"] = rows[1]["habitat_episode_id"]
        self._write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "duplicate Episode"):
            self._select()

    def test_duplicate_loaded_episode_still_rejected(self) -> None:
        episodes = list(self.episodes)
        episodes[-1] = types.SimpleNamespace(
            scene_id=episodes[0].scene_id, episode_id=episodes[0].episode_id,
        )
        with self.assertRaisesRegex(ValueError, "Pilot IDs differ"):
            self._select(episodes)

    def test_rgm_88_count_and_scene_quotas_still_required(self) -> None:
        rows = [dict(row) for row in self.rows[:-1]]
        self._write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "exactly 88"):
            self._select()

        rows = [dict(row) for row in self.rows]
        rows[0]["distance_group"] = "long"
        self._write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "3/3/2"):
            self._select()


if __name__ == "__main__":
    unittest.main()
