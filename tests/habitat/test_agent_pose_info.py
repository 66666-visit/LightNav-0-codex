"""The Habitat info payload records the simulator pose when it is exposed."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_SERVER_ROOT = Path(__file__).resolve().parents[2] / "habitat_server"
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

from lightnav_habitat.base import BaseHabitatEnv  # noqa: E402
from lightnav_habitat.vlnce import VLNCEEnv  # noqa: E402


def test_compute_info_includes_agent_pose() -> None:
    rotation = SimpleNamespace(w=1.0, x=0.1, y=0.2, z=0.3)
    state = SimpleNamespace(position=[1.0, 2.0, 3.0], rotation=rotation)
    sim = SimpleNamespace(get_agent_state=lambda: state)
    episode = SimpleNamespace(scene_id="scene.glb", episode_id="007", info={})
    habitat_env = SimpleNamespace(
        sim=sim, current_episode=episode, get_metrics=lambda: {"distance_to_goal": 4.0},
    )
    wrapper = VLNCEEnv.__new__(VLNCEEnv)
    wrapper._habitat_env = habitat_env
    wrapper._step_count = 5

    info = BaseHabitatEnv._compute_info(wrapper, {})

    assert info["agent_pose"] == {
        "position": [1.0, 2.0, 3.0],
        "rotation": [1.0, 0.1, 0.2, 0.3],
    }
    assert info["distance_to_goal"] == 4.0


def test_compute_info_omits_pose_when_simulator_state_is_unavailable() -> None:
    wrapper = VLNCEEnv.__new__(VLNCEEnv)
    wrapper._habitat_env = SimpleNamespace()
    wrapper._step_count = 0

    assert "agent_pose" not in BaseHabitatEnv._compute_info(wrapper, {})
