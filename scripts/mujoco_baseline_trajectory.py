"""Independent, conservative trajectory checks for the MuJoCo TurtleBot demo.

The scene and robot are loaded into a separate MuJoCo data object. The evaluator
never changes the live simulator. Missing samples or an unverified collision
model produce an unknown result rather than a successful navigation score.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class TrajectoryCheck:
    valid: bool | None
    reason: str | None
    checked_poses: int
    collision_geom: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _pose(value: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return tuple(_number(value.get(key), key) for key in ("sim_time", "x", "y", "z", "yaw"))


def check_trajectory(
    samples: list[Mapping[str, Any]] | None,
    probe: Callable[[float, float, float, float], list[str]] | None,
    *,
    max_time_gap_s: float = 0.25,
    spatial_spacing_m: float = 0.02,
    yaw_spacing_rad: float = 0.1,
) -> TrajectoryCheck:
    """Interpolate dense footprint poses and query MuJoCo contacts.

    A clear result means no sampled penetration through collision geoms. It is
    not a continuous collision proof. Gaps, jumps and probe failures are unknown.
    """
    if not isinstance(samples, list) or len(samples) < 2:
        return TrajectoryCheck(None, "trajectory_data_missing", 0)
    try:
        poses = [_pose(sample) for sample in samples]
    except (AttributeError, TypeError, ValueError):
        return TrajectoryCheck(None, "trajectory_data_missing", 0)
    # Commands can broadcast multiple snapshots before the next physics tick.
    # Identical pose/time duplicates add no trajectory information, while a
    # changed pose at the same simulation time is inconsistent telemetry.
    unique_poses = [poses[0]]
    for pose in poses[1:]:
        previous = unique_poses[-1]
        if pose[0] == previous[0] and pose[1:] == previous[1:]:
            continue
        unique_poses.append(pose)
    if len(unique_poses) < 2:
        return TrajectoryCheck(None, "trajectory_data_missing", 0)
    poses = unique_poses
    if probe is None:
        return TrajectoryCheck(None, "geometry_unavailable", 0)
    if min(max_time_gap_s, spatial_spacing_m, yaw_spacing_rad) <= 0:
        raise ValueError("trajectory sampling limits must be positive")

    checked = 0
    previous = poses[0]
    try:
        for index, current in enumerate(poses):
            if index:
                dt = current[0] - previous[0]
                if dt <= 0 or dt > max_time_gap_s:
                    return TrajectoryCheck(None, "trajectory_sample_gap", checked)
                distance = math.hypot(current[1] - previous[1], current[2] - previous[2])
                dyaw = math.atan2(math.sin(current[4] - previous[4]), math.cos(current[4] - previous[4]))
                # The demo clips commands to 1.5 m/s and 3 rad/s. Allow slack for
                # discrete snapshots but reject teleports and external resets.
                if distance > 2.0 * dt + 0.05 or abs(dyaw) > 3.5 * dt + 0.1:
                    return TrajectoryCheck(None, "trajectory_pose_jump", checked)
                subdivisions = max(
                    1,
                    math.ceil(distance / spatial_spacing_m),
                    math.ceil(abs(dyaw) / yaw_spacing_rad),
                )
                fractions = (part / subdivisions for part in range(1, subdivisions + 1))
            else:
                fractions = (0.0,)
                dyaw = 0.0
            for fraction in fractions:
                x = previous[1] + (current[1] - previous[1]) * fraction
                y = previous[2] + (current[2] - previous[2]) * fraction
                z = previous[3] + (current[3] - previous[3]) * fraction
                yaw = previous[4] + dyaw * fraction
                collisions = probe(x, y, z, yaw)
                if not isinstance(collisions, list) or any(not isinstance(name, str) for name in collisions):
                    raise TypeError("trajectory contact probe returned malformed collisions")
                checked += 1
                if collisions:
                    return TrajectoryCheck(False, "collision_detected", checked, collisions[0])
            previous = current
    except Exception:
        return TrajectoryCheck(None, "geometry_check_error", checked)
    return TrajectoryCheck(True, None, checked)


class MujocoFootprintProbe:
    """Check the demo robot's collidable footprint against the frozen scene."""

    def __init__(self) -> None:
        import mujoco

        from vln_mujoco.robots.turtlebot import TurtleBotBackend

        backend = TurtleBotBackend()
        self.model = backend.model
        self.data = backend.data
        self.mujoco = mujoco
        self.robot_geom_ids = {
            int(self.model.geom(name).id)
            for name in ("base_collision", "left_wheel_geom", "right_wheel_geom", "caster_geom")
        }
        joint_id = int(self.model.joint("base_joint").id)
        self.base_qpos = int(self.model.jnt_qposadr[joint_id])
        self.base_z = float(self.data.qpos[self.base_qpos + 2])
        self.floor_ids = {
            geom_id for geom_id in range(self.model.ngeom)
            if int(self.model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
            or self._geom_name(geom_id) == "floor"
        }
        self.model.geom("wall_4_10_collision_0")
        if not any("wall_" in self._geom_name(i) and "_collision_" in self._geom_name(i)
                   for i in range(self.model.ngeom)):
            raise RuntimeError("scene has no wall collision geoms")
        spawn_collisions = self(6.5, 13.8, self.base_z, 0.0)
        if spawn_collisions:
            raise RuntimeError(f"spawn overlaps scene collision geometry: {spawn_collisions[:3]}")
        wall_collisions = self(5.518, 13.8, self.base_z, 0.0)
        if not any(name.startswith("wall_4_10_collision_") for name in wall_collisions):
            raise RuntimeError("wall collision geometry self-check did not detect wall_4_10")

    def _geom_name(self, geom_id: int) -> str:
        return self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""

    def __call__(self, x: float, y: float, z: float, yaw: float) -> list[str]:
        address = self.base_qpos
        self.data.qpos[address : address + 3] = (x, y, z)
        self.data.qpos[address + 3 : address + 7] = (
            math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)
        )
        self.mujoco.mj_forward(self.model, self.data)
        collisions: list[str] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.dist >= -1e-4:
                continue
            first, second = int(contact.geom1), int(contact.geom2)
            if first in self.robot_geom_ids and second not in self.robot_geom_ids:
                other = second
            elif second in self.robot_geom_ids and first not in self.robot_geom_ids:
                other = first
            else:
                continue
            if other not in self.floor_ids:
                collisions.append(self._geom_name(other) or f"geom_{other}")
        return collisions
