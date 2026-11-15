"""Same-step Habitat RGB-D calibration and sensor pose metadata, without Habitat."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

_SERVER_ROOT = Path(__file__).resolve().parents[2] / "habitat_server"
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

from lightnav_habitat.base import BaseHabitatEnv  # noqa: E402
from lightnav_habitat.vlnce import VLNCEEnv  # noqa: E402

from lightnav.memory.conceptgraphs_lite import (  # noqa: E402
    GeometryConfig,
    RGBDFrame,
    backproject_world,
)


def _rotation(w: float, x: float, y: float, z: float) -> NS:
    return NS(w=w, x=x, y=y, z=z)


def _fixture(*, normalize_depth: bool = True) -> tuple[VLNCEEnv, dict]:
    yaw = math.pi / 2.0
    agent_quat = _rotation(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    agent_pos = np.array([1.0, 2.0, 3.0])
    camera_offset = np.array([0.1, 0.88, 0.2])
    # A +90 degree Habitat Y rotation maps local +X to world -Z and +Z to +X.
    world_offset = np.array([camera_offset[2], camera_offset[1], -camera_offset[0]])
    sensor_state = NS(position=agent_pos + world_offset, rotation=agent_quat)
    state = NS(
        position=agent_pos, rotation=agent_quat,
        sensor_states={"rgb": sensor_state, "depth": sensor_state},
    )
    rgb_config = NS(height=4, width=6, hfov=90.0, sensor_subtype="PINHOLE")
    depth_config = NS(
        height=4, width=6, hfov=90.0, sensor_subtype="PINHOLE",
        min_depth=0.5, max_depth=5.0, normalize_depth=normalize_depth,
    )
    sim = NS(
        get_agent_state=lambda: state,
        habitat_config=NS(agents=NS(main_agent=NS(sim_sensors=NS(
            rgb_sensor=rgb_config, depth_sensor=depth_config,
        )))),
    )
    wrapper = VLNCEEnv.__new__(VLNCEEnv)
    wrapper._habitat_env = NS(sim=sim)
    wrapper._step_count = 0
    habitat_obs = {
        "rgb": np.zeros((4, 6, 3), dtype=np.uint8),
        "depth": np.full((4, 6, 1), 0.5, dtype=np.float32),
    }
    return wrapper, habitat_obs


def test_spatial_rgbd_info_uses_actual_sensor_pose_and_normalized_depth_range() -> None:
    wrapper, habitat_obs = _fixture()
    info = BaseHabitatEnv._compute_info(wrapper, habitat_obs)
    geometry = info["spatial_rgbd"]

    assert geometry["status"] == "ok"
    assert geometry["aligned"] is True
    assert geometry["image_size_hw"] == [4, 6]
    np.testing.assert_allclose(
        geometry["K"],
        [[3.0, 0.0, 3.0], [0.0, 3.0, 2.0], [0.0, 0.0, 1.0]],
    )
    assert geometry["agent_pose"]["position"] == [1.0, 2.0, 3.0]
    assert geometry["camera_extrinsics"]["position"] == pytest.approx([0.1, 0.88, 0.2])
    assert geometry["camera_extrinsics"]["rotation"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert geometry["depth_encoding"] == "normalized_0_1"
    assert (geometry["min_depth_m"], geometry["max_depth_m"]) == (0.5, 5.0)


def test_metric_depth_encoding_is_explicit() -> None:
    wrapper, habitat_obs = _fixture(normalize_depth=False)
    assert BaseHabitatEnv._compute_info(wrapper, habitat_obs)["spatial_rgbd"]["depth_encoding"] == "meters"


def test_server_metadata_decodes_in_client_to_metric_world_point() -> None:
    wrapper, habitat_obs = _fixture()
    info = BaseHabitatEnv._compute_info(wrapper, habitat_obs)
    obs = {
        "rgb": BaseHabitatEnv._extract_rgb(wrapper, habitat_obs),
        "depth": BaseHabitatEnv._extract_depth(wrapper, habitat_obs),
    }
    frame = RGBDFrame.from_habitat(obs, info)
    assert frame.depth_m[2, 3] == pytest.approx(2.75)
    assert frame.camera_pose.position == pytest.approx((1.2, 2.88, 2.9))

    mask = np.zeros((4, 6), dtype=np.bool_)
    mask[2, 3] = True  # optical centre: local camera point (0, 0, -2.75)
    world = backproject_world(mask, frame, GeometryConfig(min_points=1))
    np.testing.assert_allclose(world, [[-1.55, 2.88, 2.9]], atol=1e-8)


def test_depth_channel_removal_preserves_single_pixel_image_axes() -> None:
    wrapper, _ = _fixture()
    depth = BaseHabitatEnv._extract_depth(
        wrapper, {"depth": np.array([[[0.5]]], dtype=np.float32)},
    )
    assert depth.shape == (1, 1)


@pytest.mark.parametrize("invalid", ["size", "missing_depth", "misaligned_pose", "fov"])
def test_invalid_rgbd_metadata_fails_closed(invalid: str) -> None:
    wrapper, habitat_obs = _fixture()
    if invalid == "size":
        habitat_obs["depth"] = habitat_obs["depth"][:, :-1]
    elif invalid == "missing_depth":
        del habitat_obs["depth"]
    elif invalid == "misaligned_pose":
        wrapper._habitat_env.sim.get_agent_state().sensor_states["depth"] = NS(
            position=[99.0, 0.0, 0.0], rotation=_rotation(1.0, 0.0, 0.0, 0.0),
        )
    else:
        wrapper._habitat_env.sim.habitat_config.agents.main_agent.sim_sensors.depth_sensor.hfov = 80.0

    geometry = BaseHabitatEnv._compute_info(wrapper, habitat_obs)["spatial_rgbd"]
    assert geometry["status"] == "unavailable"
    assert geometry["reason"]
    assert "K" not in geometry and "camera_extrinsics" not in geometry


def test_transition_record_only_follows_a_real_environment_step() -> None:
    wrapper, _ = _fixture()
    wrapper.max_steps = 10
    wrapper._habitat_env.step = lambda action: {"rgb": np.zeros((4, 6, 3), np.uint8)}
    wrapper._convert_action = lambda action: action
    wrapper._build_observation = lambda observation: observation
    wrapper._compute_info = lambda observation: BaseHabitatEnv._compute_info(wrapper, observation)
    wrapper._compute_reward = lambda observation, info: 0.0
    wrapper._check_terminated = lambda observation, info, action: False

    _, _, _, _, info = BaseHabitatEnv.step(wrapper, "move_forward")
    assert info["spatial_transition_record"] is True
    assert info["steps"] == 1

    def failed_step(action):
        raise RuntimeError("environment did not transition")

    wrapper._habitat_env.step = failed_step
    with pytest.raises(RuntimeError, match="did not transition"):
        BaseHabitatEnv.step(wrapper, "move_forward")
