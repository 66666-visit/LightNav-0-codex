"""Footprint interpolation and unknown-result checks without MuJoCo installed."""

from __future__ import annotations

import sys
import unittest
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mujoco_baseline_trajectory import MujocoFootprintProbe, check_trajectory  # noqa: E402


def pose(time_s: float, x: float, y: float) -> dict[str, float]:
    return {"sim_time": time_s, "x": x, "y": y, "z": 0.033, "yaw": 0.0}


def wall_probe(x: float, y: float, _z: float, _yaw: float) -> list[str]:
    return ["wall_collision"] if 0.14 <= x <= 0.16 and abs(y) < 0.1 else []


class TrajectoryCheckTest(unittest.TestCase):
    def test_clear_path(self) -> None:
        samples = [pose(0.0, 0.0, 1.0), pose(0.1, 0.1, 1.0), pose(0.2, 0.2, 1.0)]
        result = check_trajectory(samples, wall_probe)
        self.assertIs(result.valid, True)
        self.assertIsNone(result.reason)
        self.assertGreater(result.checked_poses, len(samples))

    def test_interpolation_detects_wall_crossing(self) -> None:
        samples = [pose(0.0, 0.0, 0.0), pose(0.1, 0.1, 0.0), pose(0.2, 0.2, 0.0)]
        result = check_trajectory(samples, wall_probe)
        self.assertIs(result.valid, False)
        self.assertEqual(result.reason, "collision_detected")
        self.assertEqual(result.collision_geom, "wall_collision")

    def test_missing_or_sparse_trajectory_is_unknown(self) -> None:
        self.assertEqual(check_trajectory(None, wall_probe).reason, "trajectory_data_missing")
        self.assertIsNone(check_trajectory([], wall_probe).valid)
        self.assertEqual(check_trajectory([pose(0.0, 0.0, 0.0)], wall_probe).reason,
                         "trajectory_data_missing")
        sparse = [pose(0.0, 0.0, 0.0), pose(0.5, 0.1, 0.0)]
        result = check_trajectory(sparse, wall_probe)
        self.assertIsNone(result.valid)
        self.assertEqual(result.reason, "trajectory_sample_gap")

    def test_same_tick_duplicate_snapshot_is_ignored(self) -> None:
        samples = [pose(0.0, 0.0, 1.0), pose(0.0, 0.0, 1.0), pose(0.1, 0.1, 1.0)]
        self.assertIs(check_trajectory(samples, wall_probe).valid, True)

        changed_without_time = [pose(0.0, 0.0, 1.0), pose(0.0, 0.1, 1.0)]
        result = check_trajectory(changed_without_time, wall_probe)
        self.assertIsNone(result.valid)
        self.assertEqual(result.reason, "trajectory_sample_gap")

    def test_missing_geometry_or_probe_failure_is_unknown(self) -> None:
        samples = [pose(0.0, 0.0, 0.0), pose(0.1, 0.1, 0.0)]
        self.assertEqual(check_trajectory(samples, None).reason, "geometry_unavailable")

        def broken(_x: float, _y: float, _z: float, _yaw: float) -> list[str]:
            raise RuntimeError("probe failed")

        self.assertEqual(check_trajectory(samples, broken).reason, "geometry_check_error")

    @unittest.skipUnless(find_spec("mujoco") and find_spec("vln_mujoco"), "MuJoCo demo environment unavailable")
    def test_bundled_scene_collision_probe_self_check(self) -> None:
        probe = MujocoFootprintProbe()
        self.assertEqual(probe(6.5, 13.8, probe.base_z, 0.0), [])
        self.assertTrue(any(name.startswith("wall_4_10_collision_") for name in
                            probe(5.518, 13.8, probe.base_z, 0.0)))


if __name__ == "__main__":
    unittest.main()
