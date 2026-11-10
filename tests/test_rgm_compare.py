"""Exact paired A/C/D report from synthetic 88 Episode artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from lightnav.memory.rgm_config import RGMConfig
from scripts.compare_memory_ablation import compare


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class RGMCompareTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        pilot_rows = [
            {"scene_id": f"scene_{scene}", "habitat_episode_id": f"ep_{scene}_{index}",
             "distance_group": ("short", "medium", "long")[
                 0 if index < 3 else 1 if index < 6 else 2],
             "sample_seed": 42, "dataset_split": "val_unseen"}
            for scene in range(11) for index in range(8)
        ]
        manifest = {
            "split": "val_unseen", "expected_count": 88,
            "episodes": [{"scene_id": row["scene_id"],
                          "episode_id": row["habitat_episode_id"]} for row in pilot_rows],
        }
        pilot = {
            "protocol_version": "rgm_pilot_88_v1", "dataset_split": "val_unseen",
            "sample_seed": 42, "expected_count": 88, "per_scene": 8,
            "parent_manifest_sha256": "a" * 64,
            "scene_quotas": {f"scene_{scene}": 8 for scene in range(11)},
            "episodes": pilot_rows,
        }
        _write_json(root / "episode_manifest.json", manifest)
        _write_json(root / "pilot_manifest.json", pilot)
        _write_json(root / "ablation_run.json", {
            "protocol_version": "rgm_pilot_88_v1",
            "groups": ["baseline_off", "rgm_gates", "rgm_full"],
            "parent_manifest_sha256": "a" * 64,
            "pilot_manifest_sha256": hashlib.sha256(
                (root / "pilot_manifest.json").read_bytes(),
            ).hexdigest(),
        })
        marker = {
            "expected_count": 88,
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        }
        common = {
            "model_path": "/model", "model_config_sha256": "m", "eval_config_sha256": "e",
            "backend": "vllm_local", "task": "vlnce", "split": "val_unseen",
            "habitat_config": "/config", "habitat_config_sha256": "h",
            "seed": 42, "episodes": -1, "max_steps": 500,
            "data_path": "/data", "scenes_dir": "/scenes",
            "expected_episodes": 88, "force_stop_at_max_steps": False,
            "gpu_ids": ["0"], "action_tokenizer_bundle": "",
            "gpu_memory_utilization": 0.65,
            "client_non_memory_args": ["--no_force_stop"], "inference_env": {},
        }
        c_args = [
            "--memory_enabled", "--memory_extractor", "yolo11",
            "--memory_extract_every_n_steps", "5",
            "--memory_strategy", "hierarchical",
            "--memory_layers", "short,mid,long", "--memory_place_enabled",
            "--memory_place_extract_every_n_steps", "5",
            "--rgm_enabled", "--rgm_prompt_max_records", "3",
            "--rgm_prompt_max_chars", "400",
        ]
        for name, extra, success in (
            ("baseline_off", [], False),
            ("rgm_gates", c_args, False),
            ("rgm_full", c_args + ["--rgm_yolo_temporal_enabled",
                          "--rgm_place_two_stage_enabled",
                          "--rgm_topology_motion_validation_enabled"], True),
        ):
            directory = root / name
            _write_json(directory / "episode_manifest.json", manifest)
            _write_json(directory / "config_snapshot.yaml", {
                "common": common, "memory_args": extra, "server_args": [],
                "memory_effective": {
                    "enabled": name != "baseline_off",
                    "rgm": asdict(RGMConfig(
                        enabled=name != "baseline_off",
                        yolo_temporal_enabled=name == "rgm_full",
                        place_two_stage_enabled=name == "rgm_full",
                        topology_motion_validation_enabled=name == "rgm_full",
                    )),
                },
            })
            rows = [{
                "scene_id": row["scene_id"], "habitat_episode_id": row["habitat_episode_id"],
                "success": success, "oracle_success": True, "spl": 0.5 if success else 0.0,
                "ndtw": 0.7, "final_distance": 1.0, "steps": 1,
                "episode_time_sec": 1.0, "inference_latency_ms_mean": 10.0,
            } for row in pilot_rows]
            diagnostics = [{
                "scene_id": row["scene_id"], "habitat_episode_id": row["habitat_episode_id"],
                "step": 0, "inference_latency_ms": 10.0, "prompt_chars": 100,
                "extraction_attempted": False, "injected_records": 0,
                "episode_id": f"{row['scene_id']}:{row['habitat_episode_id']}",
                "yolo_detection_ms": None, "yolo_candidate_count": 0,
                "yolo_temporal_support": [], "yolo_association_ms": None,
                "dino_global_topk": [], "dino_local_match_count": None,
                "dino_local_inlier_ratio": None, "retrieval_candidates": [],
                "fusion_selected_records": [], "prompt_fallback_reason": None,
                "injected_memory_chars": 0,
                "injected_memory_text": "", "rgm_config": {"enabled": True},
                "write_gate_decisions": [], "retrieval_gate_decisions": [],
                "prompt_gate_decision": {"accepted_records": []},
                "prompt_hash": {"roundtrip_ok": True},
                "place_decision": None, "topology_edge_status": None,
            } for row in pilot_rows]
            _write_jsonl(directory / "results.jsonl", rows)
            _write_jsonl(directory / "memory_diagnostics.jsonl", diagnostics)
            _write_json(directory / "summary.json", {
                "num_episodes": 88, "total_time_sec": 88.0,
                "metrics": {
                    "SR_success_rate_pct": 100.0 if success else 0.0,
                    "OS_oracle_success_pct": 100.0,
                    "SPL_pct": 50.0 if success else 0.0,
                    "NDTW_pct": 70.0, "NE_navigation_error_m": 1.0,
                },
            })
            _write_json(directory / "pilot_complete.json", marker)
            _write_json(directory / "run_timing.json", {
                "group": name, "attempts": [{"exit_code": 0, "wall_time_sec": 88.0}],
            })

    def test_exact_88_paired_bootstrap_and_incomplete_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            report = compare(root, samples=30, seed=42)
            self.assertEqual(report["evaluation_mode"], "rgm_pilot_88_v1")
            pair = report["paired_differences"]["rgm_full_minus_rgm_gates"]
            self.assertEqual(pair["paired_count"], 88)
            self.assertEqual(pair["metrics"]["SR_pct"]["mean"], 100.0)
            self.assertEqual(pair["metrics"]["SR_pct"]["ci95"], [100.0, 100.0])
            marker = root / "rgm_full" / "pilot_complete.json"
            marker.unlink()
            with self.assertRaises(ValueError):
                compare(root, samples=30, seed=42)

    def test_absolute_pilot_selection_matches_relative_symlink_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            real = root / "real"
            real.mkdir()
            (root / "data").symlink_to(real, target_is_directory=True)
            aliases = {}
            targets = {}
            for scene in range(11):
                name = f"scene_{scene}"
                target = real / f"{name}.glb"
                target.touch()
                targets[name] = str(target)
                aliases[name] = os.path.relpath(root / "data" / target.name, Path.cwd())

            pilot_path = root / "pilot_manifest.json"
            pilot = json.loads(pilot_path.read_text())
            pilot["scene_quotas"] = {targets[scene]: count
                                      for scene, count in pilot["scene_quotas"].items()}
            for row in pilot["episodes"]:
                row["scene_id"] = targets[row["scene_id"]]
            _write_json(pilot_path, pilot)
            launch_path = root / "ablation_run.json"
            launch = json.loads(launch_path.read_text())
            launch["pilot_manifest_sha256"] = hashlib.sha256(pilot_path.read_bytes()).hexdigest()
            _write_json(launch_path, launch)

            manifest_path = root / "episode_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for row in manifest["episodes"]:
                row["scene_id"] = aliases[row["scene_id"]]
            _write_json(manifest_path, manifest)
            marker = {
                "expected_count": 88,
                "manifest_sha256": hashlib.sha256(
                    json.dumps(manifest, sort_keys=True).encode(),
                ).hexdigest(),
            }
            for name in ("baseline_off", "rgm_gates", "rgm_full"):
                group = root / name
                _write_json(group / "episode_manifest.json", manifest)
                for filename in ("results.jsonl", "memory_diagnostics.jsonl"):
                    path = group / filename
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    for row in rows:
                        row["scene_id"] = aliases[row["scene_id"]]
                    _write_jsonl(path, rows)
                _write_json(group / "pilot_complete.json", marker)

            report = compare(root, samples=10, seed=42)
            self.assertEqual(report["evaluation_mode"], "rgm_pilot_88_v1")
            self.assertEqual(report["paired_differences"]["rgm_full_minus_rgm_gates"][
                "paired_count"], 88)

    def test_c_cannot_silently_disable_a_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            path = root / "rgm_gates" / "config_snapshot.yaml"
            config = json.loads(path.read_text())
            config["memory_effective"]["rgm"]["write_gate_enabled"] = False
            _write_json(path, config)
            with self.assertRaisesRegex(ValueError, "effective RGM settings"):
                compare(root, samples=30, seed=42)

    def test_c_d_model_schedule_and_prompt_budget_must_match(self) -> None:
        for field, value in (
            ("model_path", "/other-model"),
            ("--memory_place_extract_every_n_steps", "10"),
            ("--rgm_prompt_max_chars", "401"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._fixture(root)
                path = root / "rgm_full" / "config_snapshot.yaml"
                config = json.loads(path.read_text())
                if field.startswith("--"):
                    position = config["memory_args"].index(field) + 1
                    config["memory_args"][position] = value
                else:
                    config["common"][field] = value
                _write_json(path, config)
                with self.assertRaises(ValueError):
                    compare(root, samples=30, seed=42)


if __name__ == "__main__":
    unittest.main()
