"""Single-Episode comparison separates policy decisions from changed observations."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_episode1_traces import compare_traces


class CompareEpisode1TracesTest(unittest.TestCase):
    def _write(self, root: Path, name: str, rgb: tuple[bytes, ...],
               actions: tuple[str, ...], velocities: tuple[float, ...] | None = None) -> Path:
        directory = root / name
        (directory / "rgb").mkdir(parents=True)
        if velocities is None:
            velocities = tuple(float(action.removesuffix(">").rsplit("_", 1)[-1])
                               for action in actions)
        with (directory / "step_trace.jsonl").open("w", encoding="utf-8") as stream:
            for step, (frame, action, velocity) in enumerate(zip(rgb, actions, velocities)):
                relative = f"rgb/step_{step:04d}.png"
                (directory / relative).write_bytes(frame)
                row = {
                    "scene_id": "scene", "habitat_episode_id": "9",
                    "instruction": "find the door", "step": step,
                    "rgb": relative, "pose_pre": {"position": [0, 0, 0]},
                    "rgb_raw_sha256": hashlib.sha256(frame).hexdigest(),
                    "rgb_shape": [1, len(frame), 1],
                    "rvq_action_tokens": [action],
                    "decoded_action": {
                        "action": "velocity_control",
                        "action_args": {"linear_velocity": velocity, "angular_velocity": 0.0},
                    },
                    "distance_to_goal_post": 5 - step,
                }
                stream.write(json.dumps(row) + "\n")
        return directory

    def test_reports_same_rgb_action_change_before_trajectories_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._write(root, "a", (b"view0", b"view1"), ("<act_l0_1>", "<act_l0_1>"))
            b = self._write(root, "b", (b"view0", b"view2"), ("<act_l0_2>", "<act_l0_2>"))
            c = self._write(root, "c", (b"view0", b"view1"), ("<act_l0_1>", "<act_l0_2>"))
            report = compare_traces(a, b, c)
            fix3 = report["comparisons"]["rgm_v4_fix3"]
            local = report["comparisons"]["local_graph"]
            self.assertEqual(fix3["first_same_rgb_action_divergence_step"], 0)
            self.assertEqual(fix3["first_rgb_divergence_step"], 1)
            self.assertEqual(local["first_same_rgb_action_divergence_step"], 1)
            self.assertIsNone(report["first_correct_route_deviation_step"])
            self.assertIsNone(local["paired_steps"][1]["wrong_turn"])

    def test_different_rvq_tokens_with_same_executed_action_are_not_action_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._write(root, "a", (b"view0",), ("<act_l0_1>",), (0.1,))
            b = self._write(root, "b", (b"view0",), ("<act_l0_2>",), (0.1,))
            c = self._write(root, "c", (b"view0",), ("<act_l0_1>",), (0.1,))
            report = compare_traces(a, b, c)
            fix3 = report["comparisons"]["rgm_v4_fix3"]
            self.assertIsNone(fix3["first_action_divergence_step"])
            self.assertIs(fix3["paired_steps"][0]["action_divergence_vs_baseline"], False)
            self.assertIs(fix3["paired_steps"][0]["rvq_token_divergence_vs_baseline"], True)

    def test_rejects_different_initial_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._write(root, "a", (b"view0",), ("<act_l0_1>",))
            b = self._write(root, "b", (b"changed",), ("<act_l0_1>",))
            c = self._write(root, "c", (b"view0",), ("<act_l0_1>",))
            with self.assertRaisesRegex(ValueError, "initial RGB"):
                compare_traces(a, b, c)

    def test_rejects_different_action_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self._write(root, "a", (b"view0",), ("<act_l0_1>",))
            b = self._write(root, "b", (b"view0",), ("<act_l0_1>",))
            c = self._write(root, "c", (b"view0",), ("<act_l0_1>",))
            path = b / "step_trace.jsonl"
            row = json.loads(path.read_text().strip())
            row["action_horizon"] = 5
            path.write_text(json.dumps(row) + "\n")
            baseline = a / "step_trace.jsonl"
            base_row = json.loads(baseline.read_text().strip())
            base_row["action_horizon"] = 10
            baseline.write_text(json.dumps(base_row) + "\n")
            with self.assertRaisesRegex(ValueError, "action_horizon"):
                compare_traces(a, b, c)


if __name__ == "__main__":
    unittest.main()
