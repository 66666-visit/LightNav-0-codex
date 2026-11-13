"""V3 launcher selection and Pilot protection without Habitat or model weights."""

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


class V3LauncherGuardTests(unittest.TestCase):
    @staticmethod
    def _environment(root: Path, output: Path, client_args: str) -> dict[str, str]:
        model = root / "model"
        model.mkdir()
        fake_python = root / "fake_python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = '-c' ]; then exit 0; fi\n"
            f"if [ \"$1\" = '-' ]; then exec '{sys.executable}' \"$@\"; fi\n"
            "exit 2\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        env = dict(os.environ)
        env.update({
            "MODEL_PATH": str(model), "OUTPUT_ROOT": str(output),
            "CLIENT_ARGS": client_args, "CLIENT_PYTHON": str(fake_python),
            "HABITAT_PYTHON": str(fake_python), "GPU_IDS": "0",
            "READY_TIMEOUT_S": "0", "STRICT_FULL_EVAL": "0",
            "PILOT_MANIFEST_PATH": "", "EPISODE_MANIFEST_PATH": "",
        })
        return env

    def test_tab_separated_v3_flag_rejects_legacy_pilot_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "rgm_pilot_88_v2" / "shard_0"
            env = self._environment(
                root, output,
                "--memory_enabled\t--rgm_v3_enabled\t--memory_strategy hierarchical",
            )
            result = subprocess.run(
                ["bash", str(_LAUNCHER)], cwd=_REPO, env=env,
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("independent output directory", result.stderr)
            self.assertFalse(output.exists())

    def test_v3_snapshot_selects_independent_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "rgm_v3_test"
            env = self._environment(
                root, output,
                "--memory_enabled --memory_strategy hierarchical "
                "--memory_layers short,mid,long --rgm_v3_enabled",
            )
            result = subprocess.run(
                ["bash", str(_LAUNCHER)], cwd=_REPO, env=env,
                text=True, capture_output=True, check=False,
            )
            # The fake server exits before readiness; no Habitat is launched.
            self.assertNotEqual(result.returncode, 0)
            snapshot = json.loads(
                (output / "config_snapshot.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(snapshot["protocol_version"], "rgm_v3_v1")
            self.assertTrue(snapshot["memory_effective"]["rgm_v3"]["enabled"])
            self.assertFalse(snapshot["memory_effective"]["rgm"]["enabled"])
            self.assertIn("--rgm_v3_enabled", snapshot["memory_args"])


if __name__ == "__main__":
    unittest.main()
