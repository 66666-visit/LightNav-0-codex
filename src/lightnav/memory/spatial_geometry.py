"""Measured Habitat world pose and horizontal viewpoint geometry.

The position belongs to the observing agent, never to a YOLO object. Habitat's
world frame has +Y up and the agent faces -Z at identity rotation. Heading and
bearing are positive to the left, matching Habitat's positive turn angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

HABITAT_WORLD_FRAME = "habitat_world"


def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} must contain finite numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def heading_from_quaternion(rotation_wxyz: object) -> float:
    """Return Habitat yaw in radians, positive left, from a wxyz quaternion."""
    w, x, y, z = _vector(rotation_wxyz, 4, "rotation_wxyz")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        raise ValueError("rotation_wxyz has zero norm")
    w, x, y, z = (part / norm for part in (w, x, y, z))
    forward_x = 2 * (w * y - x * z)
    forward_z = 1 - 2 * (x * x + y * y)
    if math.hypot(forward_x, forward_z) < 1e-8:
        raise ValueError("rotation_wxyz has no horizontal heading")
    return math.atan2(forward_x, forward_z)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class AgentPose:
    """One simulator-measured agent pose in a named Scene and coordinate frame."""

    scene_id: str
    position: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    frame_id: str = HABITAT_WORLD_FRAME

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise ValueError("pose scene_id must be nonempty")
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("pose frame_id must be nonempty")
        object.__setattr__(self, "position", _vector(self.position, 3, "position"))
        object.__setattr__(self, "rotation_wxyz", _vector(
            self.rotation_wxyz, 4, "rotation_wxyz",
        ))
        heading_from_quaternion(self.rotation_wxyz)

    @property
    def heading(self) -> float:
        return heading_from_quaternion(self.rotation_wxyz)

    @classmethod
    def from_habitat_info(
        cls, info: dict[str, Any] | None, *, expected_scene_id: str,
    ) -> AgentPose | None:
        """Read only the true pose/scene fields; omit evaluator metrics."""
        if not isinstance(info, dict) or not isinstance(info.get("agent_pose"), dict):
            return None
        scene_id = info.get("scene_id")
        if scene_id is None:
            return None
        if scene_id != expected_scene_id:
            raise ValueError("Habitat pose scene differs from active memory Scene")
        pose = info["agent_pose"]
        try:
            return cls(
                scene_id=scene_id, position=pose.get("position"),
                rotation_wxyz=pose.get("rotation"),
            )
        except ValueError:
            # An unavailable or malformed simulator pose is unknown geometry.
            return None


def relative_viewpoint(
    current: AgentPose, target_position: object, *,
    target_scene_id: str, target_frame_id: str,
) -> tuple[float, float | None]:
    """Return horizontal metric distance and left-positive bearing to a viewpoint.

    This straight-line bearing is neither an object bearing nor a navigable
    next action. A coincident viewpoint has no defined bearing.
    """
    if current.scene_id != target_scene_id or current.frame_id != target_frame_id:
        raise ValueError("viewpoints use different Scenes or coordinate frames")
    target = _vector(target_position, 3, "target_position")
    dx = target[0] - current.position[0]
    dz = target[2] - current.position[2]
    distance = math.hypot(dx, dz)
    if distance < 1e-6:
        return distance, None
    absolute = math.atan2(-dx, -dz)
    return distance, wrap_angle(absolute - current.heading)


def relative_direction(bearing: float | None, distance: float | None) -> str:
    if distance is None:
        return "unknown"
    if bearing is None:
        return "here"
    angle = math.degrees(bearing)
    magnitude = abs(angle)
    if magnitude <= 22.5:
        return "front"
    if magnitude >= 157.5:
        return "back"
    side = "left" if angle > 0 else "right"
    if magnitude <= 67.5:
        return f"{side}-front"
    if magnitude <= 112.5:
        return side
    return f"{side}-back"
