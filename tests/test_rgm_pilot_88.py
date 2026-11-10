"""Synthetic 220-to-88 selection and strict A compatibility checks."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.sample_r2r_pilot import sample_rgm_88
from scripts.reuse_legacy_a import inspect_legacy_a
from lightnav.habitat.experiment_integrity import check_pilot_selection


class RGMPilotTests(unittest.TestCase):
    def test_derived_selection_is_deterministic_and_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"scene_id": f"scene_{scene}", "habitat_episode_id": f"ep_{scene}_{index}",
                 "trajectory_id": f"traj_{index // 2}", "route_length": float(index + 1)}
                for scene in range(11) for index in range(20)
            ]
            parent = root / "old.json"
            parent.write_text(json.dumps({
                "dataset_split": "val_unseen", "sample_seed": 42,
                "expected_count": 220, "episodes": rows,
            }))
            first = sample_rgm_88(parent)
            second = sample_rgm_88(parent)
            self.assertEqual(first, second)
            self.assertEqual(first["expected_count"], 88)
            self.assertEqual(len({(row["scene_id"], row["habitat_episode_id"])
                                  for row in first["episodes"]}), 88)
            for scene in range(11):
                counts = Counter(row["distance_group"] for row in first["episodes"]
                                 if row["scene_id"] == f"scene_{scene}")
                self.assertEqual(counts, {"short": 3, "medium": 3, "long": 2})
            selected = root / "selected.json"
            selected.write_text(json.dumps(first))
            loaded = {
                "split": "val_unseen", "expected_count": 88,
                "episodes": [{"scene_id": row["scene_id"],
                              "episode_id": row["habitat_episode_id"]}
                             for row in first["episodes"]],
            }
            check_pilot_selection(loaded, selected)
            with self.assertRaises(ValueError):
                check_pilot_selection({**loaded, "episodes": loaded["episodes"][:-1],
                                       "expected_count": 87}, selected)
            target = root / "target_config.yaml"
            target.write_text(json.dumps({"common": {}}))
            plan = inspect_legacy_a(parent, selected, root / "missing_old_run", target)
            self.assertEqual(plan["status"], "rerun_a_required")
            self.assertIn("old_a_missing_selected_results", plan["reasons"])

    def test_complete_legacy_a_with_missing_fingerprints_is_not_auto_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"scene_id": f"scene_{scene}", "habitat_episode_id": f"ep_{scene}_{index}",
                 "trajectory_id": f"traj_{index}", "route_length": float(index + 1)}
                for scene in range(11) for index in range(20)
            ]
            parent = root / "old_pilot.json"
            parent.write_text(json.dumps({
                "dataset_split": "val_unseen", "sample_seed": 42,
                "expected_count": 220, "episodes": rows,
            }))
            selected = root / "new_pilot.json"
            selected.write_text(json.dumps(sample_rgm_88(parent)))
            old_root = root / "old_run"
            group = old_root / "baseline_off"
            group.mkdir(parents=True)
            manifest = {
                "split": "val_unseen", "expected_count": 220,
                "episodes": [{"scene_id": row["scene_id"],
                              "episode_id": row["habitat_episode_id"]} for row in rows],
            }
            (old_root / "episode_manifest.json").write_text(json.dumps(manifest))
            (group / "episode_manifest.json").write_text(json.dumps(manifest))
            result_rows = [{
                "scene_id": row["scene_id"], "habitat_episode_id": row["habitat_episode_id"],
                "success": True, "oracle_success": True, "spl": 0.5, "ndtw": 0.7,
                "final_distance": 1.0, "steps": 1, "episode_time_sec": 1.0,
                "inference_latency_ms_mean": 10.0,
            } for row in rows]
            diagnostics = [{
                "scene_id": row["scene_id"], "habitat_episode_id": row["habitat_episode_id"],
                "step": 0, "inference_latency_ms": 10.0, "prompt_chars": 100,
            } for row in rows]
            (group / "results.jsonl").write_text("".join(json.dumps(row) + "\n"
                                                       for row in result_rows))
            (group / "memory_diagnostics.jsonl").write_text("".join(
                json.dumps(row) + "\n" for row in diagnostics
            ))
            (group / "summary.json").write_text(json.dumps({
                "num_episodes": 220, "total_time_sec": 220.0,
                "metrics": {
                    "SR_success_rate_pct": 100.0, "OS_oracle_success_pct": 100.0,
                    "SPL_pct": 50.0, "NDTW_pct": 70.0,
                    "NE_navigation_error_m": 1.0,
                },
            }))
            (group / "run_timing.json").write_text(json.dumps({
                "group": "baseline_off", "attempts": [{"exit_code": 0, "wall_time_sec": 220.0}],
            }))
            (group / "pilot_complete.json").write_text(json.dumps({
                "expected_count": 220,
                "manifest_sha256": hashlib.sha256(
                    json.dumps(manifest, sort_keys=True).encode(),
                ).hexdigest(),
            }))
            common = {
                "model_path": "/model", "model_config_sha256": "m",
                "eval_config_sha256": "e", "backend": "vllm_local",
                "task": "vlnce", "split": "val_unseen", "habitat_config": "/config",
                "habitat_config_sha256": "h", "seed": 42, "episodes": -1,
                "max_steps": 500, "data_path": "/data", "scenes_dir": "/scenes",
                "force_stop_at_max_steps": False, "gpu_ids": ["0"],
                "action_tokenizer_bundle": "", "gpu_memory_utilization": 0.65,
                "client_non_memory_args": ["--no_force_stop"], "inference_env": {},
            }
            (group / "config_snapshot.yaml").write_text(json.dumps({
                "common": {**common, "expected_episodes": 220},
                "memory_args": [], "server_args": [],
            }))
            target = root / "target.yaml"
            target.write_text(json.dumps({
                "common": {**common, "expected_episodes": 88},
                "memory_args": ["--rgm_enabled"], "server_args": [],
            }))
            plan = inspect_legacy_a(parent, selected, old_root, target)
            self.assertEqual(plan["status"], "rerun_a_required")
            self.assertIn("unresolved_provenance_gap", plan["reasons"])
            self.assertEqual(plan["missing_episode_keys"], [])
            self.assertEqual(plan["reusable_count"], 0)


if __name__ == "__main__":
    unittest.main()
