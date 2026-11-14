"""Local Graph launcher preflight and snapshot checks without running Habitat."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_LAUNCHER = _REPO / "scripts" / "eval_habitat.sh"
_LOCAL_ARGS = (
    "--memory_enabled --memory_strategy hierarchical "
    "--memory_layers short,mid,long --local_graph_enabled"
)


def _runtime_available() -> bool:
    try:
        import lightnav.cli.eval_habitat  # noqa: F401
    except ImportError:
        return False
    return True


_HAS_RUNTIME = _runtime_available()


class LocalGraphEvalLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.fake_python = self.root / "fake_python"
        self.fake_python.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = '-c' ]; then exit 0; fi\n"
            f"if [ \"$1\" = '-' ]; then exec '{sys.executable}' \"$@\"; fi\n"
            "exit 2\n",
            encoding="utf-8",
        )
        self.fake_python.chmod(0o755)

    def launch(self, args: str, output: str | Path,
               **overrides: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.update({
            "MODEL_PATH": str(self.model), "OUTPUT_ROOT": str(output),
            "CLIENT_ARGS": args, "CLIENT_PYTHON": str(self.fake_python),
            "HABITAT_PYTHON": str(self.fake_python), "GPU_IDS": "0",
            "READY_TIMEOUT_S": "0", "STRICT_FULL_EVAL": "0",
            "PILOT_MANIFEST_PATH": "", "EPISODE_MANIFEST_PATH": "", "RESUME": "0",
        })
        env.update(overrides)
        return subprocess.run(
            ["bash", str(_LAUNCHER)], cwd=_REPO, env=env,
            text=True, capture_output=True, check=False, timeout=30,
        )

    def test_requires_memory_and_exclusive_backend_before_writing(self) -> None:
        for name, args, message in (
            ("no_memory", "--local_graph_enabled", "requires --memory_enabled"),
            ("rgm", _LOCAL_ARGS + " --rgm_enabled", "cannot share a run"),
            ("v3", _LOCAL_ARGS + " --rgm_v3_enabled", "cannot share a run"),
            ("v4", _LOCAL_ARGS + " --rgm_v4_enabled", "cannot share a run"),
        ):
            with self.subTest(name=name):
                output = self.root / name
                result = self.launch(args, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(output.exists())

    def test_rejects_resume_and_preserves_existing_fix3_root(self) -> None:
        output = self.root / "custom_fix3_result"
        output.mkdir()
        (output / "config_snapshot.yaml").write_text("old config", encoding="utf-8")
        (output / "shard_0").mkdir()
        (output / "shard_0" / "rgm_v4_protocol.json").write_text("old marker", encoding="utf-8")
        result = self.launch(_LOCAL_ARGS, output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be empty", result.stderr)
        self.assertEqual((output / "config_snapshot.yaml").read_text(), "old config")
        self.assertEqual((output / "shard_0" / "rgm_v4_protocol.json").read_text(),
                         "old marker")

        fresh = self.root / "resume_attempt"
        result = self.launch(_LOCAL_ARGS, fresh, RESUME="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot resume", result.stderr)
        self.assertFalse(fresh.exists())

        nested = output / "new_local"
        result = self.launch(_LOCAL_ARGS, nested)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nested inside existing results", result.stderr)
        self.assertFalse(nested.exists())

        nested_shard = output / "shard_0" / "new_local"
        result = self.launch(_LOCAL_ARGS, nested_shard)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nested inside existing results", result.stderr)
        self.assertFalse(nested_shard.exists())

        reserved = self.root / "rgm_v4_pilot_88" / "new_local"
        result = self.launch(_LOCAL_ARGS, reserved)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("independent output directory", result.stderr)
        self.assertFalse(reserved.exists())
        result = self.launch(_LOCAL_ARGS, str(reserved) + "/")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("independent output directory", result.stderr)

    def test_no_l2_rejects_original_88_path_before_writing(self) -> None:
        original = self.root / "local_graph_fusion_88_run2"
        for flag in (
            "--local_graph_mid_enabled false",
            "--local_graph_mid_enabled=false",
            "--local_graph_mid_enabled False",
            "--local_graph_mid_enabled=0",
        ):
            with self.subTest(flag=flag):
                result = self.launch(_LOCAL_ARGS + " " + flag, original)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("original Local Graph 88 result path", result.stderr)
                self.assertFalse(original.exists())

    @unittest.skipUnless(_HAS_RUNTIME, "LightNav runtime dependencies unavailable")
    def test_snapshot_records_actual_local_graph_cli_settings(self) -> None:
        output = self.root / "snapshot"
        result = self.launch(
            _LOCAL_ARGS + " --local_graph_scope scene_persistent "
            "--local_graph_prompt_max_chars=280 --local_graph_max_landmarks 3 "
            "--local_graph_reactivation_steps 12",
            output,
        )
        self.assertNotEqual(result.returncode, 0)  # fake server never becomes ready
        snapshot = json.loads((output / "config_snapshot.yaml").read_text())
        expected = {
            "enabled": True, "scope": "scene_persistent", "prompt_max_chars": 280,
            "max_landmarks": 3, "reactivation_steps": 12, "mid_enabled": True,
            "protocol_version": "local_graph_v1",
        }
        self.assertEqual(snapshot["local_graph_effective"], expected)
        self.assertEqual(snapshot["memory_effective"]["local_graph_effective"], expected)
        self.assertEqual(snapshot["protocol_version"], "local_graph_v1")
        self.assertIn("--local_graph_prompt_max_chars=280", snapshot["memory_args"])
        self.assertNotIn("--local_graph_enabled", snapshot["common"]["client_non_memory_args"])

    @unittest.skipUnless(_HAS_RUNTIME, "LightNav runtime dependencies unavailable")
    def test_no_l2_snapshot_records_only_ablation_as_changed_memory_setting(self) -> None:
        output = self.root / "no_l2_snapshot"
        result = self.launch(_LOCAL_ARGS + " --local_graph_mid_enabled false", output)
        self.assertNotEqual(result.returncode, 0)  # fake server never becomes ready
        snapshot = json.loads((output / "config_snapshot.yaml").read_text())
        effective = snapshot["local_graph_effective"]
        assert effective == {
            "enabled": True, "scope": "episode_local", "prompt_max_chars": 320,
            "max_landmarks": 2, "reactivation_steps": 8, "mid_enabled": False,
            "protocol_version": "local_graph_no_l2_v1",
        }
        self.assertEqual(snapshot["protocol_version"], "local_graph_no_l2_v1")
        self.assertEqual(snapshot["memory_effective"]["local_graph_effective"], effective)
        self.assertEqual(
            snapshot["memory_args"][-2:],
            ["--local_graph_mid_enabled", "false"],
        )

    @unittest.skipUnless(_HAS_RUNTIME, "LightNav runtime dependencies unavailable")
    def test_formal_88_two_gpu_snapshot_keeps_pilot_and_shard_order(self) -> None:
        pilot = self.root / "pilot_manifest.json"
        pilot.write_text(json.dumps({"protocol_version": "rgm_pilot_88_v1"}))
        output = self.root / "formal_88"
        result = self.launch(
            _LOCAL_ARGS, output, GPU_IDS="2 0", STRICT_FULL_EVAL="1",
            EXPECTED_EPISODES="88", PILOT_MANIFEST_PATH=str(pilot),
            EPISODES="-1", MAX_STEPS="500", NO_FORCE_STOP="1", SEED="42",
        )
        self.assertNotEqual(result.returncode, 0)  # fake servers never become ready
        snapshot = json.loads((output / "config_snapshot.yaml").read_text())
        self.assertEqual(snapshot["common"]["gpu_ids"], ["2", "0"])
        self.assertEqual(snapshot["common"]["expected_episodes"], 88)
        self.assertEqual(snapshot["pilot_manifest_path"], str(pilot))
        self.assertEqual(snapshot["local_graph_effective"]["scope"], "episode_local")

    @unittest.skipUnless(_HAS_RUNTIME, "LightNav runtime dependencies unavailable")
    def test_wrong_memory_structure_rejected_before_server_start(self) -> None:
        for name, args in (
            ("flat", "--memory_enabled --local_graph_enabled"),
            ("missing_layer", "--memory_enabled --memory_strategy hierarchical "
             "--memory_layers short,long --local_graph_enabled"),
        ):
            with self.subTest(name=name):
                output = self.root / name
                result = self.launch(args, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid Local Graph client configuration", result.stderr)
                self.assertFalse((output / "config_snapshot.yaml").exists())
