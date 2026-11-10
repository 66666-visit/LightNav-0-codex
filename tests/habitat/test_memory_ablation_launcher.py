"""Fast launcher guards and interrupted-run bookkeeping without Habitat or weights."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.sample_r2r_pilot import sample_pilot


_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "run_memory_ablation.sh"
_EVAL_SCRIPT = _REPO / "scripts" / "eval_habitat.sh"


class MemoryAblationLauncherTest(unittest.TestCase):
    def test_rgm_88_entrypoint_has_only_a_c_d_protocol(self) -> None:
        help_result = subprocess.run(
            ["bash", str(_SCRIPT), "--rgm-88", "--help"], cwd=_REPO,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--parent-manifest", help_result.stdout)
        rejected = subprocess.run(
            ["bash", str(_SCRIPT), "--rgm-88", "--groups", "B"], cwd=_REPO,
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("unknown option", rejected.stderr)

    def test_rgm_wrapper_selects_88_and_resume_keeps_immutable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            model, yolo, dino = self._fake_assets(temp)
            dataset = self._fake_pilot_dataset(temp)
            parent = temp / "parent_220.json"
            parent.write_text(json.dumps(sample_pilot(dataset)), encoding="utf-8")
            output = temp / "rgm_88"
            env = dict(os.environ)
            env.update({
                "MODEL_PATH": str(model), "YOLO_MODEL_PATH": str(yolo),
                "DINO_MODEL_PATH": str(dino), "DATA_PATH": str(dataset),
                "HABITAT_PYTHON": "/bin/false", "CLIENT_PYTHON": "/bin/false",
                "GPU_IDS": "0", "SEED": "42",
            })
            for key in ("CLIENT_ARGS", "SERVER_ARGS", "LANGUAGES", "SCENES_DIR",
                        "SUCCESS_DISTANCE", "NAVMESH_CELL_HEIGHT"):
                env.pop(key, None)
            command = ["bash", str(_SCRIPT), "--rgm-88", "--parent-manifest",
                       str(parent), "--output-root", str(output)]
            first = subprocess.run(command, cwd=_REPO, env=env, text=True,
                                   capture_output=True, check=False)
            self.assertNotEqual(first.returncode, 0)
            self.assertIn("cannot import habitat", first.stderr)
            pilot_path = output / "pilot_manifest.json"
            before = pilot_path.read_bytes()
            pilot = json.loads(before)
            self.assertEqual(pilot["expected_count"], 88)
            self.assertEqual(len(pilot["episodes"]), 88)
            self.assertEqual(json.loads((output / "ablation_run.json").read_text())["groups"],
                             ["baseline_off", "rgm_gates", "rgm_full"])
            resumed = subprocess.run([*command, "--resume"], cwd=_REPO, env=env,
                                     text=True, capture_output=True, check=False)
            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("cannot import habitat", resumed.stderr)
            self.assertEqual(pilot_path.read_bytes(), before)

    @staticmethod
    def _fake_pilot_dataset(temp: Path) -> Path:
        root = temp / "R2R_VLNCE_v1-3_preprocessed"
        split = root / "val_unseen"
        split.mkdir(parents=True)
        episodes = [
            {
                "scene_id": f"data/scene_datasets/scene_{scene:02d}/scene.glb",
                "episode_id": str(scene * 1000 + index),
                "trajectory_id": str(scene * 1000 + index),
                "geodesic_distance": float(index + 1),
            }
            for scene in range(11) for index in range(21)
        ]
        with gzip.open(split / "val_unseen.json.gz", "wt", encoding="utf-8") as stream:
            json.dump({"episodes": episodes}, stream)
        return root

    @staticmethod
    def _fake_assets(temp: Path) -> tuple[Path, Path, Path]:
        model = temp / "checkpoint"
        model.mkdir()
        yolo = temp / "yolo11n.pt"
        yolo.write_bytes(b"local fake file; model is never loaded")
        dino = temp / "dinov2-small"
        dino.mkdir()
        (dino / "config.json").write_text("{}", encoding="utf-8")
        (dino / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        (dino / "model.safetensors").write_bytes(b"fake; model is never loaded")
        return model, yolo, dino

    def test_groups_reject_empty_unknown_duplicate_and_repeated_option(self) -> None:
        for arguments, message in (
            (["--groups", ""], "nonempty"),
            (["--groups", ",baseline_off"], "nonempty"),
            (["--groups", "baseline_off,"], "nonempty"),
            (["--groups", "baseline_off,,hierarchical_topology"], "nonempty"),
            (["--groups", "unknown"], "invalid --groups entry"),
            (["--groups", "baseline_off,baseline_off"], "duplicate --groups entry"),
            (["--groups", "baseline_off", "--groups", "hierarchical_topology"],
             "--groups needs one comma-separated list"),
        ):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ["bash", str(_SCRIPT), *arguments], cwd=_REPO,
                    text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_pilot_option_rejects_wrong_groups_and_sample_size(self) -> None:
        for arguments, message in (
            (["--pilot-per-scene", ""], "requires 20"),
            (["--pilot-per-scene", "19"], "must be 20"),
            (["--pilot-per-scene", "20", "--groups", "baseline_off"],
             "Pilot requires --groups baseline_off,hierarchical_topology"),
        ):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ["bash", str(_SCRIPT), *arguments], cwd=_REPO,
                    text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_pilot_manifest_is_fixed_and_resume_does_not_resample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            model, yolo, dino = self._fake_assets(temp)
            dataset = self._fake_pilot_dataset(temp)
            output = temp / "pilot"
            env = dict(os.environ)
            env.update({
                "MODEL_PATH": str(model), "YOLO_MODEL_PATH": str(yolo),
                "DINO_MODEL_PATH": str(dino), "DATA_PATH": str(dataset),
                "HABITAT_PYTHON": "/bin/false", "CLIENT_PYTHON": "/bin/false",
                "GPU_IDS": "0", "SEED": "42",
            })
            for key in (
                "CLIENT_ARGS", "SERVER_ARGS", "LANGUAGES", "LOG_DIR", "SUCCESS_DISTANCE",
                "NAVMESH_CELL_HEIGHT", "EPISODES", "MAX_STEPS", "TASK", "SPLIT",
                "EXPECTED_EPISODES", "SCENES_DIR",
            ):
                env.pop(key, None)
            command = [
                "bash", str(_SCRIPT), "--output-root", str(output),
                "--groups", "baseline_off,hierarchical_topology",
                "--pilot-per-scene", "20",
            ]
            first = subprocess.run(command, cwd=_REPO, env=env, text=True,
                                   capture_output=True, check=False)
            self.assertNotEqual(first.returncode, 0)  # fake Habitat preflight
            manifest_path = output / "pilot_manifest.json"
            self.assertTrue(manifest_path.is_file(), first.stderr)
            before = manifest_path.read_bytes()
            pilot = json.loads(before)
            self.assertEqual(pilot["expected_count"], 220)
            self.assertEqual(len(pilot["full_scene_counts"]), 11)
            self.assertEqual(json.loads((output / "ablation_run.json").read_text())[
                "pilot_per_scene"], 20)
            self.assertFalse((output / "baseline_off" / "formal_complete.json").exists())
            self.assertFalse((output / "flat_memory").exists())

            second = subprocess.run(
                ["bash", str(_SCRIPT), "--resume", str(output)], cwd=_REPO,
                env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(manifest_path.read_bytes(), before)
            self.assertEqual(len(json.loads(
                (output / "baseline_off" / "run_timing.json").read_text()
            )["attempts"]), 2)

            with gzip.open(dataset / "val_unseen" / "val_unseen.json.gz", "at",
                           encoding="utf-8") as stream:
                stream.write(" ")
            changed = subprocess.run(
                ["bash", str(_SCRIPT), "--resume", str(output)], cwd=_REPO,
                env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("Pilot source dataset differs", changed.stderr)
            self.assertEqual(len(json.loads(
                (output / "baseline_off" / "run_timing.json").read_text()
            )["attempts"]), 2)

    def test_formal_run_rejects_short_episode_count_and_step_cap(self) -> None:
        for override, message in (
            ({"EPISODES": "1"}, "EPISODES must be -1"),
            ({"MAX_STEPS": "40"}, "MAX_STEPS must be 500"),
            ({"EXPECTED_EPISODES": "1"}, "EXPECTED_EPISODES must be 1839"),
        ):
            with self.subTest(override=override):
                env = dict(os.environ, **override)
                result = subprocess.run(
                    ["bash", str(_SCRIPT)], cwd=_REPO, env=env,
                    text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_direct_formal_eval_rejects_wrong_expected_count_before_server_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "checkpoint"
            model.mkdir()
            env = dict(os.environ)
            env.update({
                "MODEL_PATH": str(model),
                "HABITAT_CONFIG": str(_REPO / "habitat_server/configs/vlnce_r2r.yaml"),
                "STRICT_FULL_EVAL": "1", "EXPECTED_EPISODES": "1",
                "TASK": "vlnce", "SPLIT": "val_unseen", "EPISODES": "-1",
                "MAX_STEPS": "500", "NO_FORCE_STOP": "1", "SEED": "42",
                "SERVER_ARGS": "", "LANGUAGES": "",
            })
            result = subprocess.run(
                ["bash", str(_EVAL_SCRIPT)], cwd=_REPO, env=env,
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("formal R2R val_unseen requires EXPECTED_EPISODES=1839", result.stderr)

    def test_direct_eval_accepts_88_and_220_pilot_counts_before_habitat_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            model = temp / "checkpoint"
            model.mkdir()
            manifest_88 = temp / "pilot_88.json"
            manifest_220 = temp / "pilot_220.json"
            manifest_88.write_text(json.dumps({"protocol_version": "rgm_pilot_88_v1",
                                               "expected_count": 88, "episodes": []}))
            manifest_220.write_text(json.dumps({"expected_count": 220, "episodes": []}))
            base = dict(os.environ)
            base.update({
                "MODEL_PATH": str(model), "HABITAT_PYTHON": "/bin/false",
                "HABITAT_CONFIG": str(_REPO / "habitat_server/configs/vlnce_r2r.yaml"),
                "STRICT_FULL_EVAL": "1", "GPU_IDS": "0", "TASK": "vlnce",
                "SPLIT": "val_unseen", "EPISODES": "-1", "MAX_STEPS": "500",
                "NO_FORCE_STOP": "1", "SEED": "42", "SERVER_ARGS": "", "LANGUAGES": "",
            })
            for manifest, count in ((manifest_88, "88"), (manifest_220, "220")):
                with self.subTest(count=count):
                    env = dict(base, PILOT_MANIFEST_PATH=str(manifest),
                               EXPECTED_EPISODES=count)
                    result = subprocess.run(["bash", str(_EVAL_SCRIPT)], cwd=_REPO,
                                            env=env, text=True, capture_output=True,
                                            check=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("cannot import habitat", result.stderr)
                    env["EXPECTED_EPISODES"] = "220" if count == "88" else "88"
                    rejected = subprocess.run(["bash", str(_EVAL_SCRIPT)], cwd=_REPO,
                                              env=env, text=True, capture_output=True,
                                              check=False)
                    self.assertIn(f"EXPECTED_EPISODES={count}", rejected.stderr)

    def test_formal_snapshot_records_dataset_paths_and_refuses_changed_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            model = temp / "checkpoint"
            model.mkdir()
            data_path = temp / "dataset_a"
            data_path.mkdir()
            changed_data_path = temp / "dataset_b"
            changed_data_path.mkdir()
            scenes_dir = temp / "scenes"
            scenes_dir.mkdir()
            output = temp / "output"
            client = temp / "fake_client_python"
            client.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = '-c' ]; then exit 0; fi\n"
                f'exec "{sys.executable}" "$@"\n',
                encoding="utf-8",
            )
            client.chmod(0o755)
            env = dict(os.environ)
            env.update({
                "MODEL_PATH": str(model),
                "HABITAT_CONFIG": str(_REPO / "habitat_server/configs/vlnce_r2r.yaml"),
                "HABITAT_PYTHON": "/bin/true", "CLIENT_PYTHON": str(client),
                "DATA_PATH": str(data_path), "SCENES_DIR": str(scenes_dir),
                "OUTPUT_ROOT": str(output), "GPU_IDS": "0", "READY_TIMEOUT_S": "0",
                "STRICT_FULL_EVAL": "1", "EXPECTED_EPISODES": "1839",
                "TASK": "vlnce", "SPLIT": "val_unseen", "EPISODES": "-1",
                "MAX_STEPS": "500", "NO_FORCE_STOP": "1", "SEED": "42",
                "SERVER_ARGS": "", "LANGUAGES": "", "CLIENT_ARGS": "",
            })
            first = subprocess.run(
                ["bash", str(_EVAL_SCRIPT)], cwd=_REPO, env=env,
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(first.returncode, 0)  # fake server never creates ready file
            snapshot = json.loads((output / "config_snapshot.yaml").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["common"]["data_path"], str(data_path))
            self.assertEqual(snapshot["common"]["scenes_dir"], str(scenes_dir))
            self.assertFalse(snapshot["memory_effective"]["rgm"]["enabled"])

            second = subprocess.run(
                ["bash", str(_EVAL_SCRIPT)], cwd=_REPO,
                env=dict(env, DATA_PATH=str(changed_data_path)),
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("existing config snapshot differs", second.stderr)

    def test_resume_keeps_partial_files_and_records_measured_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            model, yolo, dino = self._fake_assets(temp)
            output = temp / "ablation"
            env = dict(os.environ)
            env.update({
                "MODEL_PATH": str(model),
                "YOLO_MODEL_PATH": str(yolo),
                "DINO_MODEL_PATH": str(dino),
                "HABITAT_PYTHON": "/bin/false",  # deterministic preflight failure
                "CLIENT_PYTHON": "/bin/false",
                "GPU_IDS": "0",
                "SEED": "42",
            })
            for key in (
                "CLIENT_ARGS", "SERVER_ARGS", "LANGUAGES", "LOG_DIR", "SUCCESS_DISTANCE",
                "NAVMESH_CELL_HEIGHT", "EPISODES", "MAX_STEPS", "TASK", "SPLIT",
            ):
                env.pop(key, None)

            first = subprocess.run(
                ["bash", str(_SCRIPT), "--output-root", str(output)],
                cwd=_REPO, env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(first.returncode, 0)
            self.assertTrue((output / "ablation_run.json").is_file())
            self.assertEqual(json.loads((output / "ablation_run.json").read_text())["groups"], [
                "baseline_off", "flat_memory", "hierarchical_memory", "hierarchical_topology",
            ])
            timing = output / "baseline_off" / "run_timing.json"
            attempts = json.loads(timing.read_text(encoding="utf-8"))["attempts"]
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["exit_code"], first.returncode)
            self.assertGreaterEqual(attempts[0]["wall_time_sec"], 0.0)
            self.assertFalse((output / "baseline_off" / "formal_complete.json").exists())

            partial = output / "baseline_off" / "shard_0" / "results.jsonl"
            partial.parent.mkdir()
            partial.write_text('{"episode_id":"partial"}\n', encoding="utf-8")
            second = subprocess.run(
                ["bash", str(_SCRIPT), "--resume", str(output)],
                cwd=_REPO, env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(partial.read_text(encoding="utf-8"),
                             '{"episode_id":"partial"}\n')
            attempts = json.loads(timing.read_text(encoding="utf-8"))["attempts"]
            self.assertEqual(len(attempts), 2)
            self.assertTrue(all(item["ended_at_utc"] for item in attempts))

            different_seed = dict(env, SEED="99")
            mismatch = subprocess.run(
                ["bash", str(_SCRIPT), "--resume", str(output)],
                cwd=_REPO, env=different_seed, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("resume settings differ", mismatch.stderr)
            self.assertEqual(len(json.loads(timing.read_text())["attempts"]), 2)

    def test_two_group_selection_and_resume_do_not_start_other_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            model, yolo, dino = self._fake_assets(temp)
            output = temp / "ablation_ab"
            env = dict(os.environ)
            env.update({
                "MODEL_PATH": str(model), "YOLO_MODEL_PATH": str(yolo),
                "DINO_MODEL_PATH": str(dino), "HABITAT_PYTHON": "/bin/false",
                "CLIENT_PYTHON": "/bin/false", "GPU_IDS": "0", "SEED": "42",
            })
            for key in (
                "CLIENT_ARGS", "SERVER_ARGS", "LANGUAGES", "LOG_DIR", "SUCCESS_DISTANCE",
                "NAVMESH_CELL_HEIGHT", "EPISODES", "MAX_STEPS", "TASK", "SPLIT",
            ):
                env.pop(key, None)
            chosen = "baseline_off,hierarchical_topology"
            first = subprocess.run(
                ["bash", str(_SCRIPT), "--output-root", str(output), "--groups", chosen],
                cwd=_REPO, env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(first.returncode, 0)  # fake Habitat preflight
            self.assertEqual(json.loads((output / "ablation_run.json").read_text())["groups"], [
                "baseline_off", "hierarchical_topology",
            ])
            self.assertTrue((output / "baseline_off" / "run_timing.json").is_file())
            self.assertFalse((output / "flat_memory").exists())
            self.assertFalse((output / "hierarchical_memory").exists())

            resumed = subprocess.run(
                ["bash", str(_SCRIPT), "--resume", str(output)],
                cwd=_REPO, env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(resumed.returncode, 0)
            attempts = json.loads((output / "baseline_off" / "run_timing.json").read_text())[
                "attempts"
            ]
            self.assertEqual(len(attempts), 2)
            self.assertFalse((output / "flat_memory").exists())
            self.assertFalse((output / "hierarchical_memory").exists())

            mismatch = subprocess.run(
                ["bash", str(_SCRIPT), "--resume", str(output),
                 "--groups", "baseline_off,flat_memory"],
                cwd=_REPO, env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("resume settings differ", mismatch.stderr)
            self.assertEqual(len(json.loads(
                (output / "baseline_off" / "run_timing.json").read_text()
            )["attempts"]), 2)

    def test_baseline_only_does_not_require_unused_detector_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            model = temp / "checkpoint"
            model.mkdir()
            output = temp / "baseline_only"
            env = dict(os.environ)
            env.update({
                "MODEL_PATH": str(model), "HABITAT_PYTHON": "/bin/false",
                "CLIENT_PYTHON": "/bin/false", "GPU_IDS": "0", "SEED": "42",
            })
            env.pop("YOLO_MODEL_PATH", None)
            env.pop("DINO_MODEL_PATH", None)
            for key in ("CLIENT_ARGS", "SERVER_ARGS", "LANGUAGES", "LOG_DIR"):
                env.pop(key, None)
            result = subprocess.run(
                ["bash", str(_SCRIPT), "--output-root", str(output),
                 "--groups", "baseline_off"],
                cwd=_REPO, env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)  # fake Habitat preflight
            self.assertTrue((output / "baseline_off" / "run_timing.json").is_file())
            self.assertFalse((output / "hierarchical_topology").exists())
            self.assertEqual(json.loads((output / "ablation_run.json").read_text())["groups"], [
                "baseline_off",
            ])

    def test_legacy_four_group_snapshot_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            model, yolo, dino = self._fake_assets(temp)
            output = temp / "legacy"
            env = dict(os.environ)
            env.update({
                "MODEL_PATH": str(model), "YOLO_MODEL_PATH": str(yolo),
                "DINO_MODEL_PATH": str(dino), "HABITAT_PYTHON": "/bin/false",
                "CLIENT_PYTHON": "/bin/false", "GPU_IDS": "0", "SEED": "42",
            })
            for key in ("CLIENT_ARGS", "SERVER_ARGS", "LANGUAGES", "LOG_DIR"):
                env.pop(key, None)
            subprocess.run(
                ["bash", str(_SCRIPT), "--output-root", str(output)],
                cwd=_REPO, env=env, text=True, capture_output=True, check=False,
            )
            snapshot_path = output / "ablation_run.json"
            snapshot = json.loads(snapshot_path.read_text())
            del snapshot["groups"]
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            resumed = subprocess.run(
                ["bash", str(_SCRIPT), "--resume", str(output)],
                cwd=_REPO, env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(resumed.returncode, 0)  # fake Habitat preflight
            self.assertNotIn("resume settings differ", resumed.stderr)
            self.assertEqual(len(json.loads(
                (output / "baseline_off" / "run_timing.json").read_text()
            )["attempts"]), 2)

    def test_topology_only_keeps_existing_arm_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            model, yolo, dino = self._fake_assets(temp)
            client = temp / "fake_client_python"
            client.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = '-c' ]; then exit 0; fi\n"
                f'exec "{sys.executable}" "$@"\n', encoding="utf-8",
            )
            client.chmod(0o755)
            output = temp / "topology_first"
            env = dict(os.environ)
            env.update({
                "MODEL_PATH": str(model), "YOLO_MODEL_PATH": str(yolo),
                "DINO_MODEL_PATH": str(dino), "HABITAT_PYTHON": "/bin/true",
                "CLIENT_PYTHON": str(client), "GPU_IDS": "0", "SEED": "42",
                "READY_TIMEOUT_S": "0",
            })
            for key in ("CLIENT_ARGS", "SERVER_ARGS", "LANGUAGES", "LOG_DIR"):
                env.pop(key, None)
            result = subprocess.run(
                ["bash", str(_SCRIPT), "--output-root", str(output),
                 "--groups", "hierarchical_topology,baseline_off"],
                cwd=_REPO, env=env, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)  # fake server never creates ready file
            snapshot = json.loads((output / "hierarchical_topology" / "config_snapshot.yaml").read_text())
            arguments = snapshot["memory_args"]
            for expected in (
                "--memory_enabled", "--memory_extractor", "yolo11",
                "--memory_strategy", "hierarchical", "--memory_layers", "short,mid,long",
                "--memory_place_enabled", "--memory_place_extract_every_n_steps", "5",
                "--memory_place_similarity_threshold", "0.85",
            ):
                self.assertIn(expected, arguments)
            self.assertFalse((output / "flat_memory").exists())
            self.assertFalse((output / "hierarchical_memory").exists())
            self.assertFalse((output / "baseline_off").exists())


if __name__ == "__main__":
    unittest.main()
