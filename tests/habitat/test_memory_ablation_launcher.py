"""Fast launcher guards and interrupted-run bookkeeping without Habitat or weights."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "run_memory_ablation.sh"
_EVAL_SCRIPT = _REPO / "scripts" / "eval_habitat.sh"


class MemoryAblationLauncherTest(unittest.TestCase):
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
            model = temp / "checkpoint"
            model.mkdir()
            yolo = temp / "yolo11n.pt"
            yolo.write_bytes(b"local fake file; model is never loaded")
            dino = temp / "dinov2-small"
            dino.mkdir()
            (dino / "config.json").write_text("{}", encoding="utf-8")
            (dino / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (dino / "model.safetensors").write_bytes(b"fake; model is never loaded")
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


if __name__ == "__main__":
    unittest.main()
