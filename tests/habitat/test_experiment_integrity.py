"""Pure-Python checks for a complete, paired R2R episode manifest."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lightnav.habitat.experiment_integrity import (
    _key,
    _normalize_scene_id,
    archive_unfinished_diagnostics,
    check_coverage,
    check_pilot_selection,
    combine_shard_manifests,
    collect_ready_manifests,
    repair_interrupted_jsonl_tail,
    validate_manifest,
    verify_group,
    write_or_match_manifest,
)


def _manifest(episodes: list[tuple[str, str]]) -> dict:
    return {
        "split": "val_unseen",
        "expected_count": len(episodes),
        "episodes": [
            {"scene_id": scene, "episode_id": episode}
            for scene, episode in episodes
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _pilot_rows() -> list[tuple[str, str]]:
    return [
        (f"scene_{scene:02d}", f"episode_{scene:02d}_{episode:02d}")
        for scene in range(11) for episode in range(20)
    ]


class ExperimentIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_collect_shards_uses_actual_loaded_ids_and_shared_manifest(self) -> None:
        first = self.root / "shard0.ready"
        second = self.root / "shard1.ready"
        first.write_text(json.dumps(_manifest([("scene_a", "1")])))
        second.write_text(json.dumps(_manifest([("scene_b", "1")])))
        group = self.root / "group" / "episode_manifest.json"
        shared = self.root / "episode_manifest.json"
        combined = collect_ready_manifests([first, second], group, shared, 2)
        self.assertEqual(combined["expected_count"], 2)
        self.assertEqual(json.loads(group.read_text()), json.loads(shared.read_text()))
        with self.assertRaisesRegex(ValueError, "expected 3"):
            collect_ready_manifests([first, second], group, shared, 3)

    def test_duplicate_and_missing_ids_are_rejected(self) -> None:
        manifest = _manifest([("scene_a", "1"), ("scene_b", "1")])
        row = {"scene_id": "scene_a", "habitat_episode_id": "1"}
        with self.assertRaisesRegex(ValueError, "duplicate result"):
            check_coverage([row, row], manifest, complete=False)
        actual, missing = check_coverage([row], manifest, complete=False)
        self.assertEqual(actual, {(_normalize_scene_id("scene_a"), "1")})
        self.assertEqual(missing, {(_normalize_scene_id("scene_b"), "1")})
        with self.assertRaisesRegex(ValueError, "missing 1 episodes"):
            check_coverage([row], manifest, complete=True)

    def test_duplicate_episode_across_shards_fails_before_model_run(self) -> None:
        first = self.root / "shard0.ready"
        second = self.root / "shard1.ready"
        first.write_text(json.dumps(_manifest([("scene", "7")])))
        second.write_text(json.dumps(_manifest([("scene", "7")])))
        with self.assertRaisesRegex(ValueError, "duplicate episode across shards"):
            collect_ready_manifests([first, second], self.root / "group.json")

    def _pilot_selection(self) -> Path:
        path = self.root / "pilot_manifest.json"
        path.write_text(json.dumps({
            "dataset_split": "val_unseen", "expected_count": 220, "per_scene": 20,
            "episodes": [
                {"scene_id": scene, "habitat_episode_id": episode}
                for scene, episode in _pilot_rows()
            ],
        }), encoding="utf-8")
        return path

    def test_pilot_collection_checks_exact_ids_before_writing_shared_manifest(self) -> None:
        selection = self._pilot_selection()
        rows = _pilot_rows()
        first = self.root / "pilot_shard0.ready"
        second = self.root / "pilot_shard1.ready"
        first.write_text(json.dumps(_manifest(rows[:110])), encoding="utf-8")
        second.write_text(json.dumps(_manifest(rows[110:])), encoding="utf-8")
        group = self.root / "pilot_group" / "episode_manifest.json"
        shared = self.root / "pilot_shared.json"
        combined = collect_ready_manifests(
            [first, second], group, shared, 220, selection,
        )
        self.assertEqual(combined["expected_count"], 220)
        self.assertEqual(json.loads(group.read_text()), json.loads(shared.read_text()))

        replacement = list(rows)
        replacement[0] = ("scene_00", "wrong_episode")
        first.write_text(json.dumps(_manifest(replacement[:110])), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differ from fixed Pilot selection"):
            collect_ready_manifests(
                [first, second], self.root / "wrong.json", None, 220, selection,
            )
        self.assertFalse((self.root / "wrong.json").exists())

    def test_pilot_selection_detects_missing_and_duplicate_ids(self) -> None:
        selection = self._pilot_selection()
        rows = _pilot_rows()
        with self.assertRaisesRegex(ValueError, "exactly 220"):
            check_pilot_selection(_manifest(rows[:-1]), selection)
        manifest = json.loads(selection.read_text(encoding="utf-8"))
        manifest["episodes"][1] = dict(manifest["episodes"][0])
        selection.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate Episode"):
            check_pilot_selection(_manifest(rows), selection)

    def test_pilot_selection_accepts_18_21_21_scene_quotas(self) -> None:
        scenes = ["pLe4wQe7qrG", *(f"scene_{index:02d}" for index in range(1, 11))]
        full_counts = {
            scene: (18 if index == 0 else 187 if index == 1 else 186 if index == 2
                    else 181)
            for index, scene in enumerate(scenes)
        }
        self.assertEqual(sum(full_counts.values()), 1839)
        quotas = {
            scene: (18 if index == 0 else 21 if index in (1, 2) else 20)
            for index, scene in enumerate(scenes)
        }
        rows = [
            (scene, f"episode_{index:03d}")
            for scene, quota in quotas.items() for index in range(quota)
        ]
        selection = self.root / "adaptive_pilot_manifest.json"
        data = {
            "dataset_split": "val_unseen", "expected_count": 220, "per_scene": 20,
            "full_scene_counts": full_counts, "scene_quotas": quotas,
            "episodes": [
                {"scene_id": scene, "habitat_episode_id": episode}
                for scene, episode in rows
            ],
        }
        selection.write_text(json.dumps(data), encoding="utf-8")
        check_pilot_selection(_manifest(rows), selection)
        self.assertEqual(len(rows), 220)
        data["scene_quotas"]["scene_01"] = 20
        selection.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "totaling 220"):
            check_pilot_selection(_manifest(rows), selection)
        data["scene_quotas"]["scene_03"] = 21
        wrong_rows = [
            (scene, f"episode_{index:03d}")
            for scene, quota in data["scene_quotas"].items() for index in range(quota)
        ]
        data["episodes"] = [
            {"scene_id": scene, "habitat_episode_id": episode}
            for scene, episode in wrong_rows
        ]
        selection.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "deterministic allocation"):
            check_pilot_selection(_manifest(wrong_rows), selection)

    def _complete_pilot_group(self) -> tuple[Path, Path]:
        root = self.root / "pilot_group"
        root.mkdir()
        selection = self._pilot_selection()
        rows = _pilot_rows()
        manifest = _manifest(rows)
        (root / "episode_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        all_results = []
        all_diagnostics = []
        for index in range(2):
            shard = root / f"shard_{index}"
            shard.mkdir()
            shard_rows = rows[index * 110:(index + 1) * 110]
            (shard / "episode_manifest.json").write_text(
                json.dumps(_manifest(shard_rows)), encoding="utf-8",
            )
            shard_results = []
            for scene, episode in shard_rows:
                result = {
                    "scene_id": scene, "habitat_episode_id": episode,
                    "steps": 1, "episode_time_sec": 1.0,
                    "terminated": True, "truncated": False, "forced_stop": False,
                    "success": True, "oracle_success": True,
                    "spl": 0.5, "ndtw": 0.7, "final_distance": 2.0,
                }
                shard_results.append(result)
                all_results.append({**result, "shard": index})
                all_diagnostics.append({
                    "scene_id": scene, "habitat_episode_id": episode,
                    "step": 0, "inference_latency_ms": 12.0,
                    "prompt_chars": 150,
                })
            _write_jsonl(shard / "results.jsonl", shard_results)
        _write_jsonl(root / "results.jsonl", all_results)
        _write_jsonl(root / "memory_diagnostics.jsonl", all_diagnostics)
        (root / "summary.json").write_text(json.dumps({
            "num_episodes": 220,
            "metrics": {
                "SR_success_rate_pct": 100.0,
                "OS_oracle_success_pct": 100.0,
                "SPL_pct": 50.0,
                "NDTW_pct": 70.0,
                "NE_navigation_error_m": 2.0,
            },
            "episodes": [
                {"scene_id": scene, "habitat_episode_id": episode}
                for scene, episode in rows
            ],
        }), encoding="utf-8")
        return root, selection

    def test_complete_pilot_writes_only_pilot_marker(self) -> None:
        root, selection = self._complete_pilot_group()
        completed = verify_group(root, pilot_selection=selection)
        self.assertEqual(completed["expected_count"], 220)
        self.assertEqual(json.loads((root / "pilot_complete.json").read_text()), completed)
        self.assertFalse((root / "formal_complete.json").exists())

    def test_incomplete_or_duplicate_pilot_cannot_be_marked_complete(self) -> None:
        root, selection = self._complete_pilot_group()
        result_path = root / "results.jsonl"
        original = result_path.read_text(encoding="utf-8")
        lines = original.splitlines()
        result_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing 1 episodes"):
            verify_group(root, pilot_selection=selection)
        self.assertFalse((root / "pilot_complete.json").exists())
        result_path.write_text(original + lines[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate result"):
            verify_group(root, pilot_selection=selection)
        self.assertFalse((root / "pilot_complete.json").exists())

    def test_resume_archives_only_unfinished_episode_diagnostics(self) -> None:
        path = self.root / "memory_diagnostics.jsonl"
        completed = {"scene_id": "scene_a", "habitat_episode_id": "1", "step": 0}
        interrupted = {"scene_id": "scene_b", "habitat_episode_id": "2", "step": 0}
        _write_jsonl(path, [completed, interrupted])
        count = archive_unfinished_diagnostics(path, {("scene_a", "1")})
        self.assertEqual(count, 1)
        self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], [completed])
        archives = list(self.root.glob("memory_diagnostics.jsonl.interrupted_*"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(json.loads(archives[0].read_text()), interrupted)

    def test_resume_repairs_only_final_partial_jsonl_row(self) -> None:
        path = self.root / "results.jsonl"
        original = {"scene_id": "scene", "habitat_episode_id": "7"}
        path.write_bytes((json.dumps(original) + "\n").encode() + b'{"scene_id":')
        self.assertTrue(repair_interrupted_jsonl_tail(path))
        self.assertEqual(path.read_text(), json.dumps(original) + "\n")
        archives = list(self.root.glob("results.jsonl.interrupted_tail_*"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_bytes(), b'{"scene_id":')

        path.write_text(json.dumps(original), encoding="utf-8")
        self.assertFalse(repair_interrupted_jsonl_tail(path))
        self.assertEqual(path.read_text(), json.dumps(original) + "\n")

        path.write_bytes(b'{"scene_id":\n' + (json.dumps(original) + "\n").encode())
        with self.assertRaisesRegex(ValueError, "invalid non-final"):
            repair_interrupted_jsonl_tail(path)

    def _complete_group(self) -> Path:
        root = self.root / "group"
        shard = root / "shard_0"
        shard.mkdir(parents=True)
        manifest = _manifest([("scene", "7")])
        (root / "episode_manifest.json").write_text(json.dumps(manifest))
        (shard / "episode_manifest.json").write_text(json.dumps(manifest))
        result = {
            "scene_id": "scene", "habitat_episode_id": "7", "steps": 2,
            "episode_time_sec": 1.0, "terminated": True, "truncated": False,
            "forced_stop": False, "success": True, "oracle_success": True,
            "spl": 0.5, "ndtw": 0.7, "final_distance": 2.0,
        }
        _write_jsonl(root / "results.jsonl", [result])
        _write_jsonl(shard / "results.jsonl", [result])
        _write_jsonl(root / "memory_diagnostics.jsonl", [
            {"scene_id": "scene", "habitat_episode_id": "7", "step": step,
             "inference_latency_ms": 12.0, "prompt_chars": 150}
            for step in range(2)
        ])
        (root / "summary.json").write_text(json.dumps({
            "num_episodes": 1,
            "metrics": {
                "SR_success_rate_pct": 100.0,
                "OS_oracle_success_pct": 100.0,
                "SPL_pct": 50.0,
                "NDTW_pct": 70.0,
                "NE_navigation_error_m": 2.0,
            },
            "episodes": [{"scene_id": "scene", "habitat_episode_id": "7"}],
        }))
        return root

    def test_complete_group_writes_marker_only_after_exact_coverage(self) -> None:
        root = self._complete_group()
        completed = verify_group(root)
        self.assertEqual(completed["expected_count"], 1)
        self.assertEqual(json.loads((root / "formal_complete.json").read_text()), completed)

    def test_incomplete_diagnostics_never_mark_formal_completion(self) -> None:
        root = self._complete_group()
        _write_jsonl(root / "memory_diagnostics.jsonl", [
            {"scene_id": "scene", "habitat_episode_id": "7", "step": 0,
             "inference_latency_ms": 12.0, "prompt_chars": 150},
        ])
        with self.assertRaisesRegex(ValueError, "diagnostic steps incomplete"):
            verify_group(root)
        self.assertFalse((root / "formal_complete.json").exists())

    def test_terminal_signal_and_summary_metrics_are_required(self) -> None:
        root = self._complete_group()
        result_path = root / "results.jsonl"
        result = json.loads(result_path.read_text())
        result["terminated"] = False
        _write_jsonl(result_path, [result])
        _write_jsonl(root / "shard_0" / "results.jsonl", [result])
        with self.assertRaisesRegex(ValueError, "terminal signal"):
            verify_group(root)
        self.assertFalse((root / "formal_complete.json").exists())

        result["terminated"] = True
        _write_jsonl(result_path, [result])
        _write_jsonl(root / "shard_0" / "results.jsonl", [result])
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text())
        del summary["metrics"]["NDTW_pct"]
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "summary benchmark metrics"):
            verify_group(root)
        self.assertFalse((root / "formal_complete.json").exists())

        summary["metrics"]["NDTW_pct"] = 0.0
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "disagrees with Episode results"):
            verify_group(root)

    def test_merged_result_must_match_its_shard(self) -> None:
        root = self._complete_group()
        result_path = root / "results.jsonl"
        result = json.loads(result_path.read_text())
        result["spl"] = 0.9
        _write_jsonl(result_path, [result])
        with self.assertRaisesRegex(ValueError, "differs from its shard result"):
            verify_group(root)


class ScenePathIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        actual_root = self.root / "actual_data"
        actual_root.mkdir()
        alias_root = self.root / "data"
        alias_root.symlink_to(actual_root, target_is_directory=True)
        self.absolute_scenes = []
        self.relative_scenes = []
        for scene in range(11):
            suffix = Path("scene_datasets") / "mp3d" / f"scene_{scene:02d}"
            actual = actual_root / suffix / "scene.glb"
            actual.parent.mkdir(parents=True)
            (actual.parent / "nested").mkdir()
            actual.touch()
            self.absolute_scenes.append(str(actual))
            alias = alias_root / suffix / "nested" / ".." / "scene.glb"
            self.relative_scenes.append(os.path.relpath(alias, Path.cwd()))

    def _rgm_rows(self) -> list[dict[str, str]]:
        return [
            {
                "scene_id": self.absolute_scenes[scene],
                "habitat_episode_id": f"ep_{scene}_{index}",
                "distance_group": "short" if index < 3 else "medium" if index < 6 else "long",
            }
            for scene in range(11) for index in range(8)
        ]

    def _rgm_selection(self, rows: list[dict[str, str]] | None = None,
                       quotas: dict[str, int] | None = None) -> Path:
        path = self.root / "rgm_pilot_manifest.json"
        path.write_text(json.dumps({
            "protocol_version": "rgm_pilot_88_v1", "dataset_split": "val_unseen",
            "sample_seed": 42, "expected_count": 88, "per_scene": 8,
            "parent_manifest_sha256": "a" * 64,
            "scene_quotas": quotas if quotas is not None else {
                scene: 8 for scene in self.absolute_scenes
            },
            "episodes": self._rgm_rows() if rows is None else rows,
        }), encoding="utf-8")
        return path

    def _loaded_88(self) -> dict:
        return _manifest([
            (self.relative_scenes[scene], f"ep_{scene}_{index}")
            for scene in range(11) for index in range(8)
        ])

    def test_rgm_88_collect_matches_absolute_and_relative_symlink_paths(self) -> None:
        selection = self._rgm_selection(quotas={
            self.relative_scenes[scene] if scene % 2 else self.absolute_scenes[scene]: 8
            for scene in range(11)
        })
        loaded = self._loaded_88()
        first = self.root / "shard_0.ready"
        second = self.root / "shard_1.ready"
        first.write_text(json.dumps(_manifest([
            (row["scene_id"], row["episode_id"]) for row in loaded["episodes"][:44]
        ])), encoding="utf-8")
        second.write_text(json.dumps(_manifest([
            (row["scene_id"], row["episode_id"]) for row in loaded["episodes"][44:]
        ])), encoding="utf-8")
        group = self.root / "group_manifest.json"
        shared = self.root / "shared_manifest.json"
        combined = collect_ready_manifests([first, second], group, shared, 88, selection)
        self.assertEqual(combined["expected_count"], 88)
        self.assertEqual(combined["episodes"][0]["scene_id"], self.relative_scenes[0])
        self.assertEqual(json.loads(group.read_text()), json.loads(shared.read_text()))

    def test_rgm_88_still_rejects_wrong_episode_and_scene_file(self) -> None:
        selection = self._rgm_selection()
        loaded = self._loaded_88()
        wrong_id = json.loads(json.dumps(loaded))
        wrong_id["episodes"][0]["episode_id"] = "other_episode"
        with self.assertRaisesRegex(ValueError, "loaded RGM Pilot Episode IDs differ"):
            check_pilot_selection(wrong_id, selection)
        wrong_scene = json.loads(json.dumps(loaded))
        other = Path(self.absolute_scenes[0]).with_name("other.glb")
        other.touch()
        wrong_scene["episodes"][0]["scene_id"] = str(other)
        with self.assertRaisesRegex(ValueError, "loaded RGM Pilot Episode IDs differ"):
            check_pilot_selection(wrong_scene, selection)

    def test_normalized_duplicate_episode_is_rejected_in_selection_manifest_and_shards(self) -> None:
        rows = self._rgm_rows()
        rows[0] = {**rows[1], "scene_id": self.relative_scenes[0]}
        with self.assertRaisesRegex(ValueError, "8 unique Episodes"):
            check_pilot_selection(self._loaded_88(), self._rgm_selection(rows))
        duplicate = _manifest([
            (self.absolute_scenes[0], "ep"), (self.relative_scenes[0], "ep"),
        ])
        with self.assertRaisesRegex(ValueError, "duplicate episode in manifest"):
            validate_manifest(duplicate)
        first = self.root / "first.ready"
        second = self.root / "second.ready"
        first.write_text(json.dumps(_manifest([(self.absolute_scenes[0], "ep")])))
        second.write_text(json.dumps(_manifest([(self.relative_scenes[0], "ep")])))
        with self.assertRaisesRegex(ValueError, "duplicate episode across shards"):
            combine_shard_manifests([first, second])

    def test_rgm_88_count_scene_quota_and_strata_remain_strict(self) -> None:
        loaded = self._loaded_88()
        with self.assertRaisesRegex(ValueError, "88 val_unseen"):
            check_pilot_selection(loaded, self._rgm_selection(self._rgm_rows()[:-1]))
        rows = self._rgm_rows()
        rows[0]["scene_id"] = self.absolute_scenes[1]
        with self.assertRaisesRegex(ValueError, "8 unique Episodes and 3/3/2"):
            check_pilot_selection(loaded, self._rgm_selection(rows))
        rows = self._rgm_rows()
        rows[0]["distance_group"] = "long"
        with self.assertRaisesRegex(ValueError, "8 unique Episodes and 3/3/2"):
            check_pilot_selection(loaded, self._rgm_selection(rows))
        quotas = {scene: 8 for scene in self.absolute_scenes}
        quotas[self.relative_scenes[0]] = quotas.pop(self.absolute_scenes[1])
        with self.assertRaisesRegex(ValueError, "duplicate normalized scene"):
            check_pilot_selection(loaded, self._rgm_selection(quotas=quotas))

    def test_legacy_220_accepts_aliases_and_keeps_quota_validation(self) -> None:
        rows = [
            (self.absolute_scenes[scene], f"ep_{scene}_{index}")
            for scene in range(11) for index in range(20)
        ]
        selection = self.root / "pilot_220.json"
        data = {
            "dataset_split": "val_unseen", "expected_count": 220, "per_scene": 20,
            "scene_quotas": {scene: 20 for scene in self.absolute_scenes},
            "full_scene_counts": {scene: 30 for scene in self.relative_scenes},
            "episodes": [
                {"scene_id": scene, "habitat_episode_id": episode}
                for scene, episode in rows
            ],
        }
        selection.write_text(json.dumps(data), encoding="utf-8")
        loaded = _manifest([
            (self.relative_scenes[scene], f"ep_{scene}_{index}")
            for scene in range(11) for index in range(20)
        ])
        check_pilot_selection(loaded, selection)
        data["scene_quotas"][self.absolute_scenes[0]] = 19
        selection.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "totaling 220"):
            check_pilot_selection(loaded, selection)

    def test_formal_1839_manifest_identity_and_existing_bytes_are_preserved(self) -> None:
        relative = _manifest([
            (self.relative_scenes[index % 11], f"episode_{index}")
            for index in range(1839)
        ])
        absolute = _manifest([
            (self.absolute_scenes[index % 11], f"episode_{index}")
            for index in range(1839)
        ])
        target = self.root / "formal_manifest.json"
        write_or_match_manifest(target, relative)
        original = target.read_bytes()
        write_or_match_manifest(target, absolute)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(validate_manifest(relative)["episodes"][0]["scene_id"],
                         self.relative_scenes[0])
        actual, missing = check_coverage([
            {"scene_id": scene, "habitat_episode_id": episode}
            for scene, episode in (
                (self.absolute_scenes[index % 11], f"episode_{index}")
                for index in range(1839)
            )
        ], relative, complete=True)
        self.assertEqual(len(actual), 1839)
        self.assertFalse(missing)

    def test_scene_id_rejects_invalid_values_and_preserves_linux_case(self) -> None:
        for invalid in (None, "", "  ", "bad\x00path", 42):
            with self.subTest(scene_id=invalid), self.assertRaises(ValueError):
                _normalize_scene_id(invalid)
        upper = Path(self.absolute_scenes[0]).with_name("Scene.glb")
        upper.touch()
        self.assertNotEqual(_normalize_scene_id(str(upper)),
                            _normalize_scene_id(self.absolute_scenes[0]))
        self.assertEqual(_key({"scene_id": self.relative_scenes[0], "episode_id": 12}),
                         (self.absolute_scenes[0], "12"))
        with self.assertRaises(ValueError):
            _key({"scene_id": self.absolute_scenes[0], "episode_id": " "})

    def test_resume_diagnostics_accepts_an_alias_of_completed_scene(self) -> None:
        path = self.root / "memory_diagnostics.jsonl"
        row = {"scene_id": self.relative_scenes[0], "habitat_episode_id": "ep", "step": 0}
        _write_jsonl(path, [row])
        count = archive_unfinished_diagnostics(path, {(self.absolute_scenes[0], "ep")})
        self.assertEqual(count, 0)
        self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], [row])

    def test_verify_rgm_group_accepts_aliases_but_not_changed_metrics(self) -> None:
        selection = self._rgm_selection()
        group = self.root / "rgm_group"
        shard = group / "shard_0"
        shard.mkdir(parents=True)
        relative_manifest = self._loaded_88()
        absolute_manifest = _manifest([
            (self.absolute_scenes[scene], f"ep_{scene}_{index}")
            for scene in range(11) for index in range(8)
        ])
        (group / "episode_manifest.json").write_text(json.dumps(relative_manifest))
        (shard / "episode_manifest.json").write_text(json.dumps(absolute_manifest))
        shared = self.root / "shared_rgm_manifest.json"
        shared.write_text(json.dumps(absolute_manifest))

        shard_results = []
        merged_results = []
        diagnostics = []
        for scene in range(11):
            for index in range(8):
                episode_id = f"ep_{scene}_{index}"
                result = {
                    "scene_id": self.relative_scenes[scene],
                    "habitat_episode_id": episode_id, "steps": 1,
                    "episode_time_sec": 1.0, "terminated": True,
                    "truncated": False, "forced_stop": False,
                    "success": True, "oracle_success": True,
                    "spl": 0.5, "ndtw": 0.7, "final_distance": 2.0,
                }
                shard_results.append(result)
                merged_results.append({**result, "scene_id": self.absolute_scenes[scene],
                                       "shard": 0})
                diagnostics.append({
                    "scene_id": self.relative_scenes[scene],
                    "habitat_episode_id": episode_id, "step": 0,
                    "inference_latency_ms": 12.0, "prompt_chars": 150,
                })
        _write_jsonl(shard / "results.jsonl", shard_results)
        _write_jsonl(group / "results.jsonl", merged_results)
        _write_jsonl(group / "memory_diagnostics.jsonl", diagnostics)
        (group / "summary.json").write_text(json.dumps({
            "num_episodes": 88,
            "metrics": {
                "SR_success_rate_pct": 100.0, "OS_oracle_success_pct": 100.0,
                "SPL_pct": 50.0, "NDTW_pct": 70.0, "NE_navigation_error_m": 2.0,
            },
            "episodes": [
                {"scene_id": row["scene_id"],
                 "habitat_episode_id": row["habitat_episode_id"]}
                for row in merged_results
            ],
        }))
        completed = verify_group(group, shared=shared, pilot_selection=selection)
        self.assertEqual(completed["expected_count"], 88)
        self.assertTrue((group / "pilot_complete.json").is_file())
        merged_results[0]["spl"] = 0.8
        _write_jsonl(group / "results.jsonl", merged_results)
        with self.assertRaisesRegex(ValueError, "differs from its shard result"):
            verify_group(group, shared=shared, pilot_selection=selection)


if __name__ == "__main__":
    unittest.main()
