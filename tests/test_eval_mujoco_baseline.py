"""Dependency-free checks for MuJoCo baseline scoring and protocol accounting."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.eval_mujoco_baseline import (  # noqa: E402
    DemoClient,
    ModelProxy,
    _run_navigation,
    await_post_reset_frame,
    evaluate,
    goal_distance,
    history_is_continuous,
    score_trial,
    summarize,
    validate_manifest,
    yaw_error,
)


class BaselineScoringTest(unittest.TestCase):
    def test_example_tasks_are_explicitly_unverified(self) -> None:
        path = ROOT / "scripts" / "mujoco_baseline_tasks.example.json"
        manifest = validate_manifest(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(len(manifest["tasks"]), 3)
        for task in manifest["tasks"]:
            self.assertFalse(task["goal_verified"])
            self.assertEqual(
                score_trial(False, "model_stop", 0.0, task["radius_m"], True),
                (None, "goal_unverified"),
            )

        duplicate = copy.deepcopy(manifest)
        duplicate["tasks"][1]["id"] = duplicate["tasks"][0]["id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_manifest(duplicate)

    def test_stop_requires_verified_goal_and_final_proximity(self) -> None:
        self.assertEqual(score_trial(True, "model_stop", 0.49, 0.5, True), (True, None))
        self.assertEqual(score_trial(True, "model_stop", 0.49, 0.5, False), (False, None))
        self.assertEqual(score_trial(True, "model_stop", 0.49, 0.5, None, "geometry_unavailable"),
                         (None, "geometry_unavailable"))
        self.assertEqual(score_trial(True, "model_stop", 0.51, 0.5, None), (False, None))
        self.assertEqual(score_trial(True, "timeout", 0.1, 0.5, None), (False, None))
        self.assertEqual(score_trial(True, "history_discontinuity", 0.1, 0.5, True),
                         (None, "history_discontinuity"))
        snapshot = {"simulation": {"pose": {"x": 6.5, "y": 13.8, "z": 99.0, "yaw": 0.0}}}
        self.assertEqual(goal_distance(snapshot, [7.5, 13.8]), 1.0)
        self.assertAlmostEqual(yaw_error(-3.13, 3.13), 0.023185307179586, places=6)

    def test_proxy_keeps_server_history_and_both_latency_kinds(self) -> None:
        proxy = ModelProxy("ws://127.0.0.1:8050")
        proxy._record_next(
            {"action": "next", "data": {"seq": 4}},
            {"action": "next", "data": {
                "rc": 0, "seq": 4, "actions": {"step": 1, "actions": [[0.2, 0.0, 0.0]]},
                "latency_ms": 12.5, "stop": False,
            }},
            15.0,
        )
        self.assertEqual(proxy.steps[0]["history_frames"], 1)
        self.assertEqual(proxy.steps[0]["server_latency_ms"], 12.5)
        self.assertEqual(proxy.steps[0]["proxy_roundtrip_ms"], 15.0)
        self.assertTrue(history_is_continuous(proxy.steps))
        proxy.steps.append({"history_frames": 1})
        self.assertFalse(history_is_continuous(proxy.steps))

        bad = ModelProxy("ws://127.0.0.1:8050")
        bad._record_next(
            {"action": "next", "data": {"seq": 1}},
            {"action": "next", "data": {"rc": 0, "seq": 2, "actions": {"step": 1}}},
            1.0,
        )
        self.assertIn("sequence mismatch", bad.error)

    def test_summary_reports_scoring_coverage_and_abnormal_trials(self) -> None:
        records = [
            {"goal_verified": True, "success": True, "stop_reason": "model_stop", "navigation_steps": 2,
             "duration_s": 3.0, "visual_token_totals": {"llm_visual_tokens_after_pool": 100},
             "steps": [{"rc": 0, "server_latency_ms": 10.0, "proxy_roundtrip_ms": 12.0,
                        "budget": {"selected_frame_count": 2, "vit_patch_rows": 448,
                                   "vit_output_visual_tokens_before_pool": 112,
                                   "llm_visual_tokens_after_pool": 100}}]},
            {"goal_verified": True, "success": False, "stop_reason": "timeout", "navigation_steps": 1,
             "duration_s": 5.0, "visual_token_totals": {"llm_visual_tokens_after_pool": 80},
             "steps": []},
            {"goal_verified": False, "success": None, "unscorable_reason": "goal_unverified",
             "stop_reason": "model_stop", "navigation_steps": 1,
             "duration_s": 2.0, "visual_token_totals": None, "steps": []},
        ]
        result = summarize(records)
        self.assertEqual(result["scored_trials"], 2)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["scoring_coverage"], 1.0)
        self.assertEqual(result["abnormal_trials"], 0)
        self.assertEqual(result["unscorable_reason_counts"], {"goal_unverified": 1})
        self.assertEqual(result["mean_navigation_steps"], 4 / 3)
        self.assertEqual(result["mean_server_latency_ms_per_step"], 10.0)

        records.append({"goal_verified": True, "success": None,
                        "unscorable_reason": "inference_error", "stop_reason": "inference_error",
                        "navigation_steps": 0, "duration_s": 0.5, "steps": []})
        result = summarize(records)
        self.assertEqual(result["unscorable_verified_trials"], 1)
        self.assertEqual(result["abnormal_trials"], 1)
        self.assertEqual(result["scoring_coverage"], 2 / 3)
        self.assertEqual(result["stop_reason_counts"]["inference_error"], 1)
        self.assertEqual(result["unscorable_reason_counts"], {
            "goal_unverified": 1, "inference_error": 1,
        })
        self.assertEqual(result["success_rate"], 1 / 3)
        self.assertEqual(result["valid_scored_success_rate"], 0.5)

        unverified_only = summarize([records[2]])
        self.assertIsNone(unverified_only["success_rate"])

    def test_demo_command_waits_past_stale_reset_frame(self) -> None:
        def state(stamp: int, yaw: float = 0.0, x: float = 6.5) -> dict:
            return {"simulation": {"pose": {"x": x, "y": 13.8, "z": 0.0, "yaw": yaw},
                                   "camera": {"stamp_ns": stamp}}}

        class FakeWebSocket:
            def __init__(self, *, bad_yaw: bool = False, bad_xy: bool = False) -> None:
                self.sent = []
                self.messages = [
                    {"type": "snapshot", "data": state(100)},
                    {"type": "command_result", "ok": True, "message": "simulation reset"},
                    {"type": "runtime", "data": state(100)},
                    {"type": "runtime", "data": state(2001)},
                    {"type": "runtime", "data": state(2001)},
                    {"type": "runtime", "data": state(
                        2002, 0.2 if bad_yaw else 0.0, 6.7 if bad_xy else 6.5,
                    )},
                ]

            async def receive(self):
                return SimpleNamespace(type=SimpleNamespace(name="TEXT"),
                                       data=json.dumps(self.messages.pop(0)))

            async def send_json(self, payload):
                self.sent.append(payload)

        async def check() -> None:
            websocket = FakeWebSocket()
            demo = DemoClient(websocket)
            await demo.wait_snapshot(lambda _: True, 1.0)
            await demo.command({"type": "reset"}, 1.0)
            fresh, stamps = await await_post_reset_frame(
                demo, 2000, (6.5, 13.8), 0.0, 1.0
            )
            self.assertEqual(fresh["simulation"]["camera"]["stamp_ns"], 2002)
            self.assertEqual(stamps, [2001, 2002])
            self.assertEqual(websocket.sent, [{"type": "reset"}])

        asyncio.run(check())

        async def check_wrong_yaw() -> None:
            demo = DemoClient(FakeWebSocket(bad_yaw=True))
            await demo.wait_snapshot(lambda _: True, 1.0)
            await demo.command({"type": "reset"}, 1.0)
            with self.assertRaisesRegex(RuntimeError, "spawn yaw"):
                await await_post_reset_frame(demo, 2000, (6.5, 13.8), 0.0, 1.0)

        asyncio.run(check_wrong_yaw())

        async def check_wrong_xy() -> None:
            demo = DemoClient(FakeWebSocket(bad_xy=True))
            await demo.wait_snapshot(lambda _: True, 1.0)
            await demo.command({"type": "reset"}, 1.0)
            with self.assertRaisesRegex(RuntimeError, "spawn position"):
                await await_post_reset_frame(demo, 2000, (6.5, 13.8), 0.0, 1.0)

        asyncio.run(check_wrong_xy())

    def test_model_stop_requires_demo_confirmation(self) -> None:
        proxy = ModelProxy("ws://127.0.0.1:8050")
        proxy.steps = [{"rc": 0, "history_frames": 1, "stop": True}]

        class FakeDemo:
            snapshot = {"mpc": {"error": ""}, "vln": {"state": "IDLE", "stop": True},
                        "control": {"auto": False}}

        self.assertEqual(asyncio.run(_run_navigation(FakeDemo(), proxy, 1.0, 10)), "model_stop")
        proxy.steps = [{"rc": 0, "history_frames": 1, "stop": False},
                       {"rc": 0, "history_frames": 3, "stop": False}]
        self.assertEqual(
            asyncio.run(_run_navigation(FakeDemo(), proxy, 1.0, 10)),
            "history_discontinuity",
        )

    def test_demo_records_pose_samples_for_geometry_check(self) -> None:
        def state(sim_time: float, x: float) -> dict:
            return {"simulation": {"sim_time": sim_time,
                                   "pose": {"x": x, "y": 13.8, "z": 0.033, "yaw": 0.0}}}

        class FakeWebSocket:
            def __init__(self) -> None:
                self.messages = [
                    {"type": "snapshot", "data": state(0.0, 6.5)},
                    {"type": "runtime", "data": state(0.1, 6.6)},
                    {"type": "runtime", "data": state(0.2, 6.7)},
                ]

            async def receive(self):
                return SimpleNamespace(type=SimpleNamespace(name="TEXT"),
                                       data=json.dumps(self.messages.pop(0)))

        async def check() -> None:
            demo = DemoClient(FakeWebSocket())
            await demo.wait_snapshot(lambda _: True, 1.0)
            demo.start_path([7.0, 13.8])
            await demo.read(1.0)
            await demo.read(1.0)
            demo.stop_path()
            self.assertEqual(len(demo.trajectory_samples), 3)
            self.assertAlmostEqual(demo.path_length_m, 0.2)
            self.assertAlmostEqual(demo.min_goal_distance_m_sampled, 0.3)

        asyncio.run(check())

    def test_jsonl_and_summary_use_new_output_directory(self) -> None:
        async def fake_run_trial(**kwargs):
            return {
                "task_id": kwargs["task"]["id"],
                "goal_verified": False,
                "success": None,
                "stop_reason": "model_stop",
                "navigation_steps": 1,
                "duration_s": 1.0,
                "visual_token_totals": {"llm_visual_tokens_after_pool": 112},
                "steps": [{"rc": 0, "server_latency_ms": 10.0,
                           "proxy_roundtrip_ms": 12.0,
                           "budget": {"selected_frame_count": 2, "vit_patch_rows": 448,
                                      "vit_output_visual_tokens_before_pool": 112,
                                      "llm_visual_tokens_after_pool": 112}}],
            }

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "baseline"
            args = argparse.Namespace(
                tasks=ROOT / "scripts" / "mujoco_baseline_tasks.example.json",
                budget_diagnostic=directory / "diagnostic.json",
                demo_url="http://127.0.0.1:8088",
                vln_server="ws://127.0.0.1:8050",
                model_id="fake-checkpoint",
                repeat=1,
                timeout_s=5.0,
                max_steps=10,
                output=output,
            )
            fake_budget = SimpleNamespace(diagnostic_sha256="synthetic")
            with patch("mujoco_baseline_budget.load_budget", return_value=fake_budget), \
                 patch("scripts.eval_mujoco_baseline.run_trial", side_effect=fake_run_trial), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = asyncio.run(evaluate(args))
                with self.assertRaises(FileExistsError):
                    asyncio.run(evaluate(args))
            lines = (output / "results.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIsNone(result["success_rate"])
            self.assertEqual(json.loads((output / "summary.json").read_text())["total_trials"], 3)


if __name__ == "__main__":
    unittest.main()
