"""Base class shared by the Habitat environment wrappers (RGB/depth extraction, reset/step plumbing)."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Shape used for the zero image returned when the simulator exposes no RGB/depth
# sensor and no explicit image_size was requested (never hit with the shipped configs).
_FALLBACK_IMAGE_HW = (270, 480)


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    """Read a Hydra config or a small test double without importing Habitat."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _quaternion_wxyz(rotation: Any) -> np.ndarray:
    values = np.asarray(
        [rotation.w, rotation.x, rotation.y, rotation.z], dtype=np.float64,
    )
    norm = float(np.linalg.norm(values))
    if not np.isfinite(values).all() or not 0.99 <= norm <= 1.01:
        raise ValueError("invalid sensor or agent quaternion")
    return values / norm


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _quaternion_rotate(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pure = np.array([0.0, *vector], dtype=np.float64)
    inverse = rotation * np.array([1.0, -1.0, -1.0, -1.0])
    return _quaternion_product(_quaternion_product(rotation, pure), inverse)[1:]


class BaseHabitatEnv(ABC):
    """Common lifecycle for a wrapped ``habitat.Env``.

    Subclasses implement ``_create_habitat_env``, ``_extract_task_obs``, ``_convert_action``,
    ``_compute_reward`` and ``_check_terminated``.
    """

    # Sensor uuids accepted as the RGB / depth observation.
    RGB_KEYS = ["rgb", "head_rgb"]
    DEPTH_KEYS = ["depth", "head_depth"]

    def __init__(
        self,
        config_path: str,
        gpu_id: int = 0,
        image_size: Optional[Tuple[int, int]] = None,
        max_steps: int = 500,
        split_id: Optional[int] = None,
        split_num: Optional[int] = None,
    ):
        """
        Args:
            config_path: Path to the Habitat (Hydra) config yaml.
            gpu_id: habitat-sim render device (``habitat_sim_v0.gpu_device_id``).
            image_size: Optional (height, width) override for the RGB/depth sensors.
                ``None`` keeps the sensor size from the yaml.
            max_steps: Steps after which ``truncated`` is reported. Must not exceed the
                yaml's ``environment.max_episode_steps``.
            split_id / split_num: Shard index / count for parallel evaluation
                (scene-sorted contiguous slices of the episode list). ``None`` = no sharding.
        """
        self.config_path = config_path
        self.gpu_id = gpu_id
        self.image_size = tuple(image_size) if image_size is not None else None
        self.max_steps = max_steps
        self.split_id = split_id
        self.split_num = split_num

        self._habitat_env: Any = None
        self._step_count = 0
        self._initialized = False
        self._obs_keys_logged = False

    # -- abstract hooks --------------------------------------------------------

    @abstractmethod
    def _create_habitat_env(self) -> Any:
        """Build and return the underlying ``habitat.Env``."""

    @abstractmethod
    def _extract_task_obs(self, habitat_obs: Dict) -> Dict[str, Any]:
        """Return task-specific observation entries."""

    @abstractmethod
    def _convert_action(self, action: Any) -> Any:
        """Convert a client action into the Habitat action format."""

    @abstractmethod
    def _compute_reward(self, obs: Dict, info: Dict) -> float:
        """Return the reward for the current step."""

    @abstractmethod
    def _check_terminated(self, obs: Dict, info: Dict, action: Any) -> bool:
        """Return True when the episode ended on this step."""

    # -- observation helpers ---------------------------------------------------

    def _fallback_hw(self) -> Tuple[int, int]:
        return self.image_size if self.image_size is not None else _FALLBACK_IMAGE_HW

    def _extract_rgb(self, habitat_obs: Dict) -> np.ndarray:
        """RGB image as (H, W, 3) uint8."""
        for key in self.RGB_KEYS:
            if key in habitat_obs:
                rgb = np.asarray(habitat_obs[key])
                if rgb.dtype in (np.float32, np.float64):
                    rgb = (rgb * 255).astype(np.uint8)
                return rgb

        logger.warning(f"No RGB found in keys: {self.RGB_KEYS}")
        return np.zeros((*self._fallback_hw(), 3), dtype=np.uint8)

    def _extract_depth(self, habitat_obs: Dict) -> np.ndarray:
        """Depth image as (H, W) float32."""
        for key in self.DEPTH_KEYS:
            if key in habitat_obs:
                depth = np.asarray(habitat_obs[key])
                if depth.ndim == 3 and depth.shape[2] == 1:
                    depth = depth[:, :, 0]
                return depth.astype(np.float32)

        return np.zeros(self._fallback_hw(), dtype=np.float32)

    def _build_observation(self, habitat_obs: Dict) -> Dict[str, Any]:
        if not self._obs_keys_logged:
            logger.info(f"Available habitat_obs keys: {list(habitat_obs.keys())}")
            self._obs_keys_logged = True

        obs: Dict[str, Any] = {
            "rgb": self._extract_rgb(habitat_obs),
            "depth": self._extract_depth(habitat_obs),
        }
        obs.update(self._extract_task_obs(habitat_obs))
        return obs

    def _spatial_rgbd_info(self, habitat_obs: Dict, state: Any) -> Dict[str, Any]:
        """Describe an aligned RGB-D sample using the *current* simulator sensor poses.

        The RGB and depth arrays remain unchanged on the wire. In particular,
        Habitat-Lab normally normalizes depth after clipping it to its configured
        range; the client must decode that value before backprojection.
        """
        unavailable: Dict[str, Any] = {"status": "unavailable"}
        try:
            rgb_key = next(key for key in self.RGB_KEYS if key in habitat_obs)
            depth_key = next(key for key in self.DEPTH_KEYS if key in habitat_obs)
            rgb = np.asarray(habitat_obs[rgb_key])
            depth = np.asarray(habitat_obs[depth_key])
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]
            if (rgb.ndim != 3 or rgb.shape[2] not in (3, 4)
                    or depth.ndim != 2 or rgb.shape[:2] != depth.shape):
                raise ValueError("RGB and depth image sizes differ")
            height, width = depth.shape

            sim = self._habitat_env.sim
            simulator_config = getattr(sim, "habitat_config", None)
            if simulator_config is None:
                env_config = getattr(self._habitat_env, "_config", None)
                simulator_config = _config_value(
                    _config_value(env_config, "habitat"), "simulator",
                )
            agents = _config_value(simulator_config, "agents")
            agent_config = _config_value(agents, "main_agent")
            sensors = _config_value(agent_config, "sim_sensors")
            rgb_config = _config_value(sensors, "rgb_sensor")
            depth_config = _config_value(sensors, "depth_sensor")
            if rgb_config is None or depth_config is None:
                raise ValueError("RGB/depth simulator sensor config is unavailable")
            for config in (rgb_config, depth_config):
                if (int(_config_value(config, "height")) != height
                        or int(_config_value(config, "width")) != width):
                    raise ValueError("sensor config size differs from RGB-D sample")
                if str(_config_value(config, "sensor_subtype", "PINHOLE")) != "PINHOLE":
                    raise ValueError("only pinhole RGB-D sensors are supported")
            rgb_hfov = float(_config_value(rgb_config, "hfov"))
            depth_hfov = float(_config_value(depth_config, "hfov"))
            if (not 0.0 < rgb_hfov < 180.0
                    or not math.isclose(rgb_hfov, depth_hfov, abs_tol=1e-5)):
                raise ValueError("RGB/depth horizontal FOV is invalid or differs")
            depth_min = float(_config_value(depth_config, "min_depth"))
            depth_max = float(_config_value(depth_config, "max_depth"))
            if (not math.isfinite(depth_min) or not math.isfinite(depth_max)
                    or depth_min < 0.0 or depth_max <= depth_min):
                raise ValueError("depth range is invalid")
            normalized = _config_value(depth_config, "normalize_depth")
            if not isinstance(normalized, bool):
                raise ValueError("depth normalization setting is unavailable")

            sensor_states = state.sensor_states
            rgb_state = _config_value(sensor_states, rgb_key)
            depth_state = _config_value(sensor_states, depth_key)
            if rgb_state is None or depth_state is None:
                raise ValueError("current RGB/depth sensor poses are unavailable")
            agent_position = np.asarray(state.position, dtype=np.float64)
            rgb_position = np.asarray(rgb_state.position, dtype=np.float64)
            depth_position = np.asarray(depth_state.position, dtype=np.float64)
            if (agent_position.shape != (3,) or rgb_position.shape != (3,)
                    or depth_position.shape != (3,) or not np.isfinite(agent_position).all()
                    or not np.isfinite(rgb_position).all()
                    or not np.isfinite(depth_position).all()):
                raise ValueError("invalid agent or sensor position")
            agent_rotation = _quaternion_wxyz(state.rotation)
            rgb_rotation = _quaternion_wxyz(rgb_state.rotation)
            depth_rotation = _quaternion_wxyz(depth_state.rotation)
            if (np.linalg.norm(rgb_position - depth_position) > 1e-4
                    or abs(float(np.dot(rgb_rotation, depth_rotation))) < 1.0 - 1e-8):
                raise ValueError("RGB/depth sensor poses are not aligned")

            agent_inverse = agent_rotation * np.array([1.0, -1.0, -1.0, -1.0])
            camera_position = _quaternion_rotate(agent_inverse, rgb_position - agent_position)
            camera_rotation = _quaternion_product(agent_inverse, rgb_rotation)
            camera_rotation /= np.linalg.norm(camera_rotation)
            focal = (width / 2.0) / math.tan(math.radians(rgb_hfov) / 2.0)
            return {
                "status": "ok",
                "image_size_hw": [height, width],
                "K": [[focal, 0.0, width / 2.0],
                      [0.0, focal, height / 2.0],
                      [0.0, 0.0, 1.0]],
                "agent_pose": {
                    "position": agent_position.tolist(),
                    "rotation": agent_rotation.tolist(),
                },
                "camera_extrinsics": {
                    "position": camera_position.tolist(),
                    "rotation": camera_rotation.tolist(),
                },
                "depth_encoding": "normalized_0_1" if normalized else "meters",
                "min_depth_m": depth_min,
                "max_depth_m": depth_max,
                "camera_frame": "habitat_right_up_back",
                "aligned": True,
            }
        except Exception as exc:  # noqa: BLE001
            # A geometry consumer must reject this sample, never infer a camera
            # pose from the agent position or from a hard-coded sensor height.
            unavailable["reason"] = str(exc) or "required RGB-D metadata is unavailable"
            return unavailable

    def _compute_info(self, habitat_obs: Dict) -> Dict[str, Any]:
        """steps, episode_id, scene_id (+ raw_episode_id when a loader renumbered) and metrics."""
        info: Dict[str, Any] = {"steps": self._step_count}

        if hasattr(self._habitat_env, "current_episode"):
            episode = self._habitat_env.current_episode
            info["episode_id"] = getattr(episode, "episode_id", "unknown")
            info["scene_id"] = getattr(episode, "scene_id", "unknown")
            raw_episode_id = (getattr(episode, "info", None) or {}).get("raw_episode_id")
            if raw_episode_id is not None:
                info["raw_episode_id"] = raw_episode_id

        state = None
        try:
            state = self._habitat_env.sim.get_agent_state()
            position = state.position
            rotation = state.rotation
            info["agent_pose"] = {
                "position": [float(value) for value in position],
                "rotation": [
                    float(rotation.w), float(rotation.x),
                    float(rotation.y), float(rotation.z),
                ],
            }
        except (AttributeError, TypeError, ValueError):
            # Metadata only: custom test environments may not expose a simulator state.
            pass

        info["spatial_rgbd"] = self._spatial_rgbd_info(habitat_obs, state)

        if hasattr(self._habitat_env, "get_metrics"):
            info.update(self._habitat_env.get_metrics())

        return info

    # -- lifecycle -------------------------------------------------------------

    def initialize(self) -> None:
        """Eagerly create the Habitat environment (idempotent)."""
        if not self._initialized:
            logger.info("Creating Habitat environment...")
            self._habitat_env = self._create_habitat_env()
            self._initialized = True
            logger.info("Habitat environment created")

    def get_episode_manifest(self) -> Dict[str, Any]:
        """Expose loaded episode IDs to the evaluator, never to the policy."""
        self.initialize()
        episodes = []
        seen = set()
        for episode in self._habitat_env.episodes:
            scene_id = getattr(episode, "scene_id", None)
            episode_id = getattr(episode, "episode_id", None)
            if scene_id is None or episode_id is None or not str(scene_id) or not str(episode_id):
                raise ValueError("loaded Habitat episode lacks scene_id or episode_id")
            key = (str(scene_id), str(episode_id))
            if key in seen:
                raise ValueError(f"duplicate episode in loaded Habitat dataset: {key}")
            seen.add(key)
            episodes.append({"scene_id": key[0], "episode_id": key[1]})
        return {
            "split": str(getattr(self, "split", "")),
            "expected_count": len(episodes),
            "episodes": episodes,
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Start the next episode. ``seed`` / ``options`` are accepted for API compatibility."""
        self.initialize()
        self._step_count = 0

        habitat_obs = self._habitat_env.reset()
        obs = self._build_observation(habitat_obs)
        info = self._compute_info(habitat_obs)
        return obs, info

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one action; returns (obs, reward, terminated, truncated, info)."""
        self._step_count += 1

        habitat_action = self._convert_action(action)
        habitat_obs = self._habitat_env.step(habitat_action)
        obs = self._build_observation(habitat_obs)
        info = self._compute_info(habitat_obs)
        # Only the environment's real step creates this record. A reset or a
        # similarity between two poses is not evidence of a traversed edge.
        info["spatial_transition_record"] = True
        reward = self._compute_reward(obs, info)
        terminated = self._check_terminated(obs, info, action)
        truncated = self._step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._habitat_env is not None:
            self._habitat_env.close()
            self._habitat_env = None
            self._initialized = False
            logger.info("Habitat environment closed")
