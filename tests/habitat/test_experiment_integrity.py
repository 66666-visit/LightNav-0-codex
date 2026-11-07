"""Pure-Python checks for a complete, paired R2R episode manifest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lightnav.habitat.experiment_integrity import (
    archive_unfinished_diagnostics,
    check_coverage,
    collect_ready_manifests,
    repair_interrupted_jsonl_tail,
    verify_group,
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
        self.assertEqual(actual, {("scene_a", "1")})
        self.assertEqual(missing, {("scene_b", "1")})
        with self.assertRaisesRegex(ValueError, "missing 1 episodes"):
            check_coverage([row], manifest, complete=True)

    def test_duplicate_episode_across_shards_fails_before_model_run(self) -> None:
        first = self.root / "shard0.ready"
        second = self.root / "shard1.ready"
        first.write_text(json.dumps(_manifest([("scene", "7")])))
        second.write_text(json.dumps(_manifest([("scene", "7")])))
        with self.assertRaisesRegex(ValueError, "duplicate episode across shards"):
            collect_ready_manifests([first, second], self.root / "group.json")

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


if __name__ == "__main__":
    unittest.main()
