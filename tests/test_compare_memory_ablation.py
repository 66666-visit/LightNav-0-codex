"""Pure stdlib tests for the measured, paired R2R ablation report."""

from __future__ import annotations

import gc
import hashlib
import json
import tempfile
import tracemalloc
import unittest
from pathlib import Path

from scripts import compare_memory_ablation as ablation


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CompareMemoryAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = {
            "split": "val_unseen", "expected_count": 2,
            "episodes": [
                {"scene_id": "scene_a", "episode_id": "ep_1"},
                {"scene_id": "scene_a", "episode_id": "ep_2"},
            ],
        }
        _write_json(self.root / "episode_manifest.json", self.manifest)
        for name in ablation.GROUPS:
            directory = self.root / name
            directory.mkdir()
            _write_json(directory / "episode_manifest.json", self.manifest)
            _write_json(directory / "config_snapshot.yaml", {
                "common": {
                    "model_path": "/model", "backend": "vllm_local", "task": "vlnce",
                    "model_config_sha256": "model-hash",
                    "eval_config_sha256": "eval-hash",
                    "split": "val_unseen", "habitat_config": "/r2r.yaml",
                    "data_path": "/r2r-data", "scenes_dir": "/mp3d-scenes",
                    "habitat_config_sha256": "abc", "seed": 0, "episodes": -1,
                    "max_steps": 500, "expected_episodes": 2,
                    "force_stop_at_max_steps": False,
                    "gpu_ids": [0], "action_tokenizer_bundle": "",
                    "gpu_memory_utilization": 0.65,
                    "client_non_memory_args": [], "inference_env": {},
                },
                "server_args": [], "client_args": [name],
            })
            records = []
            diagnostics = []
            for index, episode_id in enumerate(("ep_1", "ep_2")):
                records.append({
                    "scene_id": "scene_a", "habitat_episode_id": episode_id,
                    "episode_id": f"episode_{index:03d}",
                    "success": bool(index or name != "baseline_off"),
                    "oracle_success": True, "spl": 0.5 + 0.1 * index,
                    "ndtw": 0.7 + 0.1 * index, "final_distance": 3.0 - index,
                    "steps": 2, "episode_time_sec": 2.0 + index,
                    "inference_latency_ms_mean": 11.0,
                })
                for step in range(2):
                    row = {
                        "scene_id": "scene_a", "habitat_episode_id": episode_id,
                        "step": step, "inference_latency_ms": 10.0 + 2 * step,
                        "prompt_chars": 100 + step,
                    }
                    if name != "baseline_off":
                        row.update({
                            "extraction_attempted": step == 0,
                            "yolo_detection_ms": 3.0 if step == 0 else None,
                            "injected_records": 1, "injected_memory_chars": 50,
                        })
                    if name == "hierarchical_topology":
                        row.update({
                            "place_extraction_attempted": step == 0,
                            "place_extraction_ms": 7.0 if step == 0 else None,
                            "topology_nodes": 1, "topology_edges": 0,
                            "place_revisited": False, "place_transition": None,
                        })
                    diagnostics.append(row)
            _write_jsonl(directory / "results.jsonl", records)
            _write_jsonl(directory / "memory_diagnostics.jsonl", diagnostics)
            _write_json(directory / "summary.json", {
                "num_episodes": 2, "total_time_sec": 10.0, "metrics": {},
            })
            _write_json(directory / "run_timing.json", {
                "group": name,
                "attempts": [{"attempt_id": "first", "wall_time_sec": 10.0, "exit_code": 0}],
            })
            _write_json(directory / "formal_complete.json", {
                "expected_count": 2,
                "manifest_sha256": hashlib.sha256(
                    json.dumps(self.manifest, sort_keys=True).encode()
                ).hexdigest(),
            })

    def _prepare_complete_ab(self) -> None:
        selected = ("baseline_off", "hierarchical_topology")
        _write_json(self.root / "ablation_run.json", {"groups": list(selected)})
        manifest = {
            "split": "val_unseen", "expected_count": 1839,
            "episodes": [
                {"scene_id": "scene_a", "episode_id": f"ep_{index}"}
                for index in range(1839)
            ],
        }
        _write_json(self.root / "episode_manifest.json", manifest)
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        for name in selected:
            directory = self.root / name
            _write_json(directory / "episode_manifest.json", manifest)
            config_path = directory / "config_snapshot.yaml"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["common"]["expected_episodes"] = 1839
            _write_json(config_path, config)
            records = []
            diagnostics = []
            for index in range(1839):
                is_topology = name == "hierarchical_topology"
                success = (index == 1) if is_topology else (index == 0)
                records.append({
                    "scene_id": "scene_a", "habitat_episode_id": f"ep_{index}",
                    "episode_id": f"episode_{index:04d}",
                    "success": success, "oracle_success": True,
                    "spl": 0.6 if is_topology else 0.5,
                    "ndtw": 0.75 if is_topology else 0.7,
                    "final_distance": 1.8 if is_topology else 2.0,
                    "steps": 1, "episode_time_sec": 2.0 if is_topology else 1.0,
                    "inference_latency_ms_mean": 12.0 if is_topology else 10.0,
                })
                row = {
                    "scene_id": "scene_a", "habitat_episode_id": f"ep_{index}",
                    "step": 0, "inference_latency_ms": 12.0 if is_topology else 10.0,
                    "prompt_chars": 150 if is_topology else 100,
                }
                if is_topology:
                    row.update({
                        "extraction_attempted": True, "yolo_detection_ms": 3.0,
                        "injected_records": 1, "injected_memory_chars": 40,
                        "place_extraction_attempted": True, "place_extraction_ms": 7.0,
                        "topology_nodes": 1, "topology_edges": 0,
                        "place_revisited": False, "place_transition": None,
                    })
                diagnostics.append(row)
            _write_jsonl(directory / "results.jsonl", records)
            _write_jsonl(directory / "memory_diagnostics.jsonl", diagnostics)
            _write_json(directory / "summary.json", {
                "num_episodes": 1839, "total_time_sec": 10.0, "metrics": {},
            })
            _write_json(directory / "formal_complete.json", {
                "expected_count": 1839, "manifest_sha256": digest,
            })

    def _prepare_complete_pilot(self) -> None:
        selected = ("baseline_off", "hierarchical_topology")
        counts = {f"scene_{scene}": 167 for scene in range(11)}
        counts["scene_10"] = 169
        rows = []
        pilot_rows = []
        for scene in range(11):
            for index in range(20):
                scene_id, episode_id = f"scene_{scene}", f"episode_{scene}_{index}"
                rows.append({"scene_id": scene_id, "episode_id": episode_id})
                pilot_rows.append({
                    "scene_id": scene_id, "habitat_episode_id": episode_id,
                    "episode_id": episode_id, "trajectory_id": f"trajectory_{index}",
                    "distance_group": ("short" if index < 7 else
                                       "medium" if index < 14 else "long"),
                    "route_length": float(index + 1), "sample_seed": 42,
                    "dataset_split": "val_unseen", "source_dataset": "/r2r.json.gz",
                    "sampling_method": "per_scene_geodesic_tertiles_7_7_6_trajectory_diverse",
                })
        manifest = {"split": "val_unseen", "expected_count": 220, "episodes": rows}
        _write_json(self.root / "episode_manifest.json", manifest)
        _write_json(self.root / "pilot_manifest.json", {
            "dataset_split": "val_unseen", "sample_seed": 42,
            "sampling_method": "per_scene_geodesic_tertiles_7_7_6_trajectory_diverse",
            "expected_count": 220, "per_scene": 20,
            "full_scene_counts": counts, "episodes": pilot_rows,
        })
        _write_json(self.root / "ablation_run.json", {
            "groups": list(selected), "pilot_per_scene": 20,
            "pilot_manifest_sha256": hashlib.sha256(
                (self.root / "pilot_manifest.json").read_bytes()
            ).hexdigest(),
        })
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        for name in selected:
            directory = self.root / name
            _write_json(directory / "episode_manifest.json", manifest)
            config_path = directory / "config_snapshot.yaml"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["common"]["expected_episodes"] = 220
            _write_json(config_path, config)
            results = []
            diagnostics = []
            for scene in range(11):
                for index in range(20):
                    scene_id, episode_id = f"scene_{scene}", f"episode_{scene}_{index}"
                    is_agent = name == "hierarchical_topology"
                    results.append({
                        "scene_id": scene_id, "habitat_episode_id": episode_id,
                        "episode_id": f"eval_{scene}_{index}",
                        "success": index % 2 == 0 or (is_agent and index == 1),
                        "oracle_success": True,
                        "spl": 0.6 if is_agent else 0.5,
                        "ndtw": 0.8 if is_agent else 0.7,
                        "final_distance": 1.8 if is_agent else 2.0,
                        "steps": 1, "episode_time_sec": 2.0 if is_agent else 1.0,
                        "inference_latency_ms_mean": 12.0 if is_agent else 10.0,
                    })
                    diagnostic = {
                        "scene_id": scene_id, "habitat_episode_id": episode_id,
                        "step": 0, "inference_latency_ms": 12.0 if is_agent else 10.0,
                        "prompt_chars": 150 if is_agent else 100,
                    }
                    if is_agent:
                        diagnostic.update({
                            "extraction_attempted": True, "yolo_detection_ms": 3.0,
                            "injected_records": 1, "injected_memory_chars": 40,
                            "place_extraction_attempted": True, "place_extraction_ms": 7.0,
                            "topology_nodes": 1, "topology_edges": 0,
                            "place_revisited": False, "place_transition": None,
                        })
                    diagnostics.append(diagnostic)
            _write_jsonl(directory / "results.jsonl", results)
            _write_jsonl(directory / "memory_diagnostics.jsonl", diagnostics)
            _write_json(directory / "summary.json", {
                "num_episodes": 220, "total_time_sec": 10.0, "metrics": {},
            })
            (directory / "formal_complete.json").unlink()
            _write_json(directory / "pilot_complete.json", {
                "expected_count": 220, "manifest_sha256": digest,
            })

    def test_pilot_ab_report_contains_scene_and_paired_statistics(self) -> None:
        self._prepare_complete_pilot()
        report = ablation.compare(self.root, samples=30, seed=42)
        self.assertEqual(report["evaluation_mode"], "pilot")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["per_episode"]), 220)
        self.assertEqual(report["pilot"]["sampled_scene_count"], 11)
        baseline = report["groups"]["baseline_off"]
        agent = report["groups"]["hierarchical_topology"]
        self.assertEqual(baseline["success_count"], 110)
        self.assertEqual(agent["success_count"], 121)
        self.assertEqual(agent["metrics"]["SR_pct"]["mean"], 55.0)
        self.assertEqual(len(agent["pilot_scenes"]["by_scene"]), 11)
        self.assertEqual(agent["pilot_scenes"]["by_scene"]["scene_0"]["sampled_episodes"], 20)
        self.assertEqual(agent["pilot_scenes"]["scene_macro_mean"]["SR_pct"], 55.0)
        self.assertEqual(agent["pilot_scenes"]["full_val_scene_weighted_estimate"]["SR_pct"], 55.0)
        self.assertEqual(agent["runtime"]["total_episode_time_sec"], 440.0)
        self.assertEqual(agent["memory"]["yolo_detection"]["calls"], 220)
        self.assertEqual(agent["memory"]["dinov2_extraction"]["calls"], 220)
        pair = report["paired_differences"]["hierarchical_topology_minus_baseline_off"]
        self.assertEqual(pair["paired_count"], 220)
        self.assertEqual(pair["sr_flips"], {
            "success_to_failure": 0, "failure_to_success": 11,
        })
        self.assertAlmostEqual(pair["metrics"]["SR_pct"]["mean"], 5.0)
        self.assertEqual(len(pair["metrics"]["SR_pct"]["ci95"]), 2)
        self.assertEqual(pair["per_episode"][1]["delta"]["SR_pct"], 100.0)
        ablation.write_report(self.root, report)
        self.assertTrue((self.root / "comparison.json").is_file())

    def test_pilot_missing_duplicate_and_wrong_marker_cannot_complete(self) -> None:
        self._prepare_complete_pilot()
        path = self.root / "hierarchical_topology" / "results.jsonl"
        original = path.read_text(encoding="utf-8")
        path.write_text("\n".join(original.splitlines()[:-1]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing 1 manifest Episode"):
            ablation.compare(self.root, samples=10)
        path.write_text(original + original.splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate Episode ID"):
            ablation.compare(self.root, samples=10)
        path.write_text(original, encoding="utf-8")
        directory = self.root / "hierarchical_topology"
        (directory / "pilot_complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "ablation is incomplete"):
            ablation.compare(self.root, samples=10)
        _write_json(directory / "formal_complete.json", {"expected_count": 220})
        with self.assertRaisesRegex(ValueError, "wrong evaluation mode"):
            ablation.compare(self.root, samples=10)

    def test_pilot_interrupted_results_remain_provisional(self) -> None:
        self._prepare_complete_pilot()
        directory = self.root / "hierarchical_topology"
        result_path = directory / "results.jsonl"
        rows = result_path.read_text(encoding="utf-8").splitlines()
        result_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
        (directory / "pilot_complete.json").unlink()
        (directory / "summary.json").unlink()
        report = ablation.compare(self.root, allow_incomplete=True, samples=10)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["groups"]["hierarchical_topology"]["completed_episodes"], 219)
        self.assertEqual(len(report["groups"]["hierarchical_topology"]["missing_episode_ids"]), 1)
        self.assertIsNone(report["groups"]["hierarchical_topology"]["pilot_scenes"][
            "full_val_scene_weighted_estimate"]["SR_pct"])

    def test_pilot_manifest_must_match_selected_episode_ids_and_scene_quotas(self) -> None:
        self._prepare_complete_pilot()
        path = self.root / "pilot_manifest.json"
        pilot = json.loads(path.read_text(encoding="utf-8"))
        pilot["episodes"][0]["habitat_episode_id"] = "wrong"
        _write_json(path, pilot)
        with self.assertRaisesRegex(ValueError, "Episode IDs differ"):
            ablation.compare(self.root, samples=10)
        pilot["episodes"][0]["habitat_episode_id"] = "episode_0_0"
        pilot["episodes"][0]["distance_group"] = "long"
        _write_json(path, pilot)
        with self.assertRaisesRegex(ValueError, "7/7/6"):
            ablation.compare(self.root, samples=10)

    def test_pilot_manifest_valid_metadata_change_rejected_by_immutable_digest(self) -> None:
        self._prepare_complete_pilot()
        path = self.root / "pilot_manifest.json"
        pilot = json.loads(path.read_text(encoding="utf-8"))
        pilot["full_scene_counts"]["scene_0"] += 1
        pilot["full_scene_counts"]["scene_1"] -= 1
        _write_json(path, pilot)
        with self.assertRaisesRegex(ValueError, "immutable ablation_run.json digest"):
            ablation.compare(self.root, samples=10)

    def test_pilot_scene_weighting_uses_full_split_scene_counts(self) -> None:
        metrics = {
            ("one", "a"): {"SR_pct": 100.0},
            ("two", "b"): {"SR_pct": 0.0},
        }
        result = ablation._pilot_scene_statistics(
            metrics, {"one": 3, "two": 1}, per_scene=1, samples=10, seed=42,
        )
        self.assertEqual(result["scene_macro_mean"]["SR_pct"], 50.0)
        self.assertEqual(result["full_val_scene_weighted_estimate"]["SR_pct"], 75.0)

    def test_selected_ab_comparison_uses_all_1839_paired_episode_ids(self) -> None:
        self._prepare_complete_ab()
        report = ablation.compare(self.root, samples=30, seed=7)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["selected_groups"], [
            "baseline_off", "hierarchical_topology",
        ])
        self.assertEqual(set(report["groups"]), {
            "baseline_off", "hierarchical_topology",
        })
        self.assertEqual(len(report["per_episode"]), 1839)
        pair = report["paired_differences"][
            "hierarchical_topology_minus_baseline_off"
        ]
        self.assertEqual(pair["paired_count"], 1839)
        self.assertEqual(pair["sr_flips"], {
            "success_to_failure": 1, "failure_to_success": 1,
        })
        self.assertEqual(pair["per_episode"][0]["delta"]["SR_pct"], -100.0)
        self.assertEqual(pair["per_episode"][1]["delta"]["SR_pct"], 100.0)
        self.assertAlmostEqual(pair["metrics"]["SPL_pct"]["mean"], 10.0)
        self.assertAlmostEqual(pair["metrics"]["nDTW_pct"]["mean"], 5.0)
        self.assertAlmostEqual(pair["metrics"]["NE_m"]["mean"], -0.2)
        for metric in ("SR_pct", "SPL_pct", "nDTW_pct", "NE_m"):
            self.assertEqual(len(pair["metrics"][metric]["ci95"]), 2)
        topology = report["groups"]["hierarchical_topology"]
        self.assertEqual(topology["completed_episodes"], 1839)
        self.assertEqual(topology["metrics"]["avg_steps"]["mean"], 1.0)
        self.assertEqual(topology["runtime"]["total_episode_time_sec"], 3678.0)
        self.assertEqual(topology["memory"]["yolo_detection"]["calls"], 1839)
        self.assertEqual(topology["memory"]["dinov2_extraction"]["calls"], 1839)
        self.assertEqual(topology["memory"]["topology_nodes_total"], 1839)
        self.assertEqual(topology["memory"]["topology_edges_total"], 0)
        ablation.write_report(self.root, report)
        written = json.loads((self.root / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(written["selected_groups"], [
            "baseline_off", "hierarchical_topology",
        ])
        self.assertTrue((self.root / "comparison.csv").is_file())

    def test_selected_ab_missing_episode_cannot_be_formally_complete(self) -> None:
        self._prepare_complete_ab()
        path = self.root / "hierarchical_topology" / "results.jsonl"
        rows = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing 1 manifest Episode"):
            ablation.compare(self.root, samples=10)

    def test_selected_ab_requires_identical_episode_manifest(self) -> None:
        self._prepare_complete_ab()
        path = self.root / "hierarchical_topology" / "episode_manifest.json"
        changed = json.loads(path.read_text(encoding="utf-8"))
        changed["episodes"][-1]["episode_id"] = "different_episode"
        _write_json(path, changed)
        with self.assertRaisesRegex(ValueError, "differs from the shared manifest"):
            ablation.compare(self.root, samples=10)

    def test_selected_groups_reject_unknown_duplicates_and_short_manifest(self) -> None:
        for selected in (None, [], ["unknown"], ["baseline_off", "baseline_off"]):
            _write_json(self.root / "ablation_run.json", {"groups": selected})
            with self.subTest(selected=selected):
                with self.assertRaisesRegex(ValueError, "groups must be"):
                    ablation.compare(self.root)
        _write_json(self.root / "ablation_run.json", {
            "groups": ["baseline_off", "hierarchical_topology"],
        })
        with self.assertRaisesRegex(ValueError, "require 1839 Episode IDs"):
            ablation.compare(self.root)

    def test_four_groups_metrics_paired_differences_and_real_diagnostics(self) -> None:
        report = ablation.compare(self.root, samples=100, seed=3)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["selected_groups"], list(ablation.GROUPS))
        baseline = report["groups"]["baseline_off"]
        flat = report["groups"]["flat_memory"]
        topology = report["groups"]["hierarchical_topology"]
        self.assertEqual(baseline["metrics"]["SR_pct"]["mean"], 50.0)
        self.assertEqual(flat["metrics"]["SR_pct"]["mean"], 100.0)
        self.assertEqual(flat["metrics"]["OS_pct"]["mean"], 100.0)
        self.assertAlmostEqual(flat["metrics"]["SPL_pct"]["mean"], 55.0)
        self.assertAlmostEqual(flat["metrics"]["nDTW_pct"]["mean"], 75.0)
        self.assertEqual(flat["metrics"]["NE_m"]["mean"], 2.5)
        self.assertEqual(flat["metrics"]["avg_steps"]["mean"], 2.0)
        self.assertEqual(flat["metrics"]["mean_inference_latency_ms"]["mean"], 11.0)
        self.assertEqual(flat["runtime"]["total_wall_time_sec"], 10.0)
        self.assertEqual(flat["runtime"]["total_episode_time_sec"], 5.0)
        self.assertEqual(flat["memory"]["yolo_detection"], {
            "calls": 2, "total_ms": 6.0, "mean_ms": 3.0,
        })
        self.assertEqual(topology["memory"]["dinov2_extraction"], {
            "calls": 2, "total_ms": 14.0, "mean_ms": 7.0,
        })
        self.assertEqual(topology["memory"]["topology_nodes_total"], 2)
        self.assertEqual(topology["memory"]["topology_edges_total"], 0)
        self.assertEqual(topology["memory"]["place_revisits"], 0)
        self.assertEqual(topology["memory"]["place_transitions"], 0)
        self.assertEqual(topology["memory"]["mean_injected_records_per_step"], 1.0)
        self.assertEqual(topology["memory"]["mean_prompt_chars_per_step"], 100.5)
        paired = report["paired_differences"]["flat_memory_minus_baseline_off"]
        self.assertEqual(paired["paired_count"], 2)
        self.assertEqual(paired["metrics"]["SR_pct"]["mean"], 50.0)
        self.assertEqual(paired["per_episode"][0]["delta"]["SR_pct"], 100.0)
        self.assertEqual(paired["per_episode"][1]["delta"]["SR_pct"], 0.0)
        ablation.write_report(self.root, report)
        self.assertTrue((self.root / "comparison.json").is_file())
        self.assertTrue((self.root / "comparison.csv").is_file())

    def test_duplicate_and_missing_episode_are_rejected(self) -> None:
        path = self.root / "flat_memory" / "results.jsonl"
        original = path.read_text(encoding="utf-8")
        path.write_text(original + original.splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate Episode ID"):
            ablation.compare(self.root)
        path.write_text(original.splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing 1 manifest Episode"):
            ablation.compare(self.root)

    def test_manifest_mismatch_and_duplicate_manifest_are_rejected(self) -> None:
        group_manifest = self.root / "flat_memory" / "episode_manifest.json"
        _write_json(group_manifest, {**self.manifest, "split": "train"})
        with self.assertRaisesRegex(ValueError, "differs from the shared manifest"):
            ablation.compare(self.root)
        _write_json(group_manifest, self.manifest)
        _write_json(self.root / "episode_manifest.json", {
            **self.manifest,
            "episodes": [self.manifest["episodes"][0], self.manifest["episodes"][0]],
        })
        with self.assertRaisesRegex(ValueError, "duplicate Episode ID"):
            ablation.compare(self.root)

    def test_common_config_mismatch_or_forced_stop_is_rejected(self) -> None:
        path = self.root / "flat_memory" / "config_snapshot.yaml"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["common"]["seed"] = 99
        _write_json(path, config)
        with self.assertRaisesRegex(ValueError, "common checkpoint, seed"):
            ablation.compare(self.root)
        config["common"]["seed"] = 0
        config["common"]["data_path"] = "/different-r2r-data"
        _write_json(path, config)
        with self.assertRaisesRegex(ValueError, "common checkpoint, seed"):
            ablation.compare(self.root)
        config["common"]["data_path"] = "/r2r-data"
        _write_json(path, config)
        path = self.root / "flat_memory" / "results.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["forced_stop"] = True
        _write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "forced STOP"):
            ablation.compare(self.root)

    def test_published_summary_metric_must_match_episode_rows(self) -> None:
        path = self.root / "flat_memory" / "summary.json"
        _write_json(path, {
            "num_episodes": 2, "total_time_sec": 10.0,
            "metrics": {"SR_success_rate_pct": 0.0},
        })
        with self.assertRaisesRegex(ValueError, "disagrees with episode results"):
            ablation.compare(self.root)

    def test_formal_completion_marker_must_match_manifest(self) -> None:
        _write_json(self.root / "flat_memory" / "formal_complete.json", {
            "expected_count": 2, "manifest_sha256": "wrong",
        })
        with self.assertRaisesRegex(ValueError, "completion marker disagrees"):
            ablation.compare(self.root)

    def test_bootstrap_ci_is_reproducible_and_exact_for_single_episode(self) -> None:
        first = ablation.bootstrap_ci([1.0, 2.0, 5.0], samples=500, seed=42)
        self.assertEqual(first, ablation.bootstrap_ci([1.0, 2.0, 5.0], samples=500, seed=42))
        self.assertTrue(1.0 <= first[0] <= first[1] <= 5.0)
        self.assertEqual(ablation.bootstrap_ci([2.5]), [2.5, 2.5])
        self.assertIsNone(ablation.bootstrap_ci([]))
        with self.assertRaises(ValueError):
            ablation.bootstrap_ci([1.0], samples=0)

    def test_latency_mean_weights_real_calls_and_bootstraps_episode_clusters(self) -> None:
        clusters = [
            [{"inference_latency_ms": 10.0}],
            [{"inference_latency_ms": 20.0}, {"inference_latency_ms": 30.0}],
        ]
        result = ablation._latency_summary(clusters, samples=500, seed=4)
        self.assertEqual(result["mean"], 20.0)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["inference_calls"], 3)
        self.assertEqual(result["ci95"], ablation._latency_summary(
            clusters, samples=500, seed=4,
        )["ci95"])

    def test_interrupted_group_is_provisional_and_keeps_completed_rows(self) -> None:
        directory = self.root / "hierarchical_topology"
        result_path = directory / "results.jsonl"
        result_path.write_text(result_path.read_text(encoding="utf-8").splitlines()[0] + "\n",
                               encoding="utf-8")
        (directory / "summary.json").unlink()
        (directory / "formal_complete.json").unlink()
        _write_json(directory / "run_timing.json", {
            "group": "hierarchical_topology",
            "attempts": [
                {"attempt_id": "interrupted", "wall_time_sec": None, "exit_code": None},
                {"attempt_id": "resumed", "wall_time_sec": 10.0, "exit_code": 0},
            ],
        })
        with self.assertRaises(ValueError):
            ablation.compare(self.root)
        report = ablation.compare(self.root, allow_incomplete=True, samples=50)
        self.assertEqual(report["status"], "incomplete")
        topology = report["groups"]["hierarchical_topology"]
        self.assertEqual(topology["completed_episodes"], 1)
        self.assertEqual(topology["orphan_diagnostic_rows"], 2)
        self.assertEqual(topology["memory"]["yolo_detection"]["calls"], 1)
        self.assertEqual(topology["missing_episode_ids"], [
            {"scene_id": "scene_a", "episode_id": "ep_2"},
        ])
        self.assertIsNone(topology["runtime"]["total_wall_time_sec"])
        self.assertEqual(topology["runtime"]["measured_wall_time_sec"], 10.0)
        self.assertEqual(topology["runtime"]["unmeasured_attempts"], 1)
        paired = report["paired_differences"][
            "hierarchical_topology_minus_baseline_off"
        ]
        self.assertEqual(paired["paired_count"], 1)
        self.assertEqual(len(paired["per_episode"]), 1)

    def test_missing_step_diagnostic_blocks_formal_report(self) -> None:
        path = self.root / "baseline_off" / "memory_diagnostics.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        _write_jsonl(path, rows[1:])
        with self.assertRaisesRegex(ValueError, "lacks exact step diagnostics"):
            ablation.compare(self.root)
        report = ablation.compare(self.root, allow_incomplete=True, samples=50)
        self.assertEqual(report["status"], "incomplete")

    def test_diagnostic_loader_streams_large_verbose_rows(self) -> None:
        directory = self.root / "baseline_off"
        result_path = directory / "results.jsonl"
        records = [json.loads(line) for line in result_path.read_text().splitlines()]
        for record in records:
            record["steps"] = 500
        _write_jsonl(result_path, records)
        large_payload = "visual observation " * 1000
        rows = [
            {
                "scene_id": "scene_a", "habitat_episode_id": episode_id,
                "step": step, "inference_latency_ms": 10.0,
                "prompt_chars": 100, "layer_contexts": {"long": large_payload},
            }
            for episode_id in ("ep_1", "ep_2") for step in range(500)
        ]
        _write_jsonl(directory / "memory_diagnostics.jsonl", rows)
        del rows, large_payload
        gc.collect()
        tracemalloc.start()
        try:
            loaded = ablation._load_group(
                self.root, "baseline_off", self.manifest,
                [("scene_a", "ep_1"), ("scene_a", "ep_2")],
                allow_incomplete=False,
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(len(loaded.diagnostics), 2)
        self.assertEqual(sum(item.rows for item in loaded.diagnostics.values()), 1000)
        # The file is >16 MB; retaining complete diagnostic objects would exceed this.
        self.assertLess(peak, 8_000_000)


if __name__ == "__main__":
    unittest.main()
