"""CPU checks for exact single-Episode selection before Habitat iteration."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_SERVER_ROOT = Path(__file__).resolve().parents[2] / "habitat_server"
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

from lightnav_habitat.serve import main  # noqa: E402
from lightnav_habitat.vlnce import VLNCEEnv  # noqa: E402


class FakeHabitatEnv:
    def __init__(self, episodes: list[types.SimpleNamespace]) -> None:
        self._episodes = list(episodes)
        self.setter_calls = 0

    @property
    def episodes(self) -> list[types.SimpleNamespace]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: list[types.SimpleNamespace]) -> None:
        self._episodes = list(episodes)
        self.setter_calls += 1


def _selector(scene_id: str, episode_id: str, split: str = "val_unseen") -> VLNCEEnv:
    wrapper = VLNCEEnv.__new__(VLNCEEnv)
    wrapper.single_scene_id = scene_id
    wrapper.single_episode_id = episode_id
    wrapper.split = split
    return wrapper


def test_selects_one_episode_by_canonical_scene_and_exact_id(tmp_path: Path) -> None:
    actual = tmp_path / "actual" / "scene.glb"
    actual.parent.mkdir()
    actual.touch()
    link = tmp_path / "linked.glb"
    link.symlink_to(actual)
    episodes = [
        types.SimpleNamespace(scene_id=str(link), episode_id="001"),
        types.SimpleNamespace(scene_id=str(link), episode_id="1"),
    ]
    env = FakeHabitatEnv(episodes)

    _selector(str(actual), "001")._apply_single_episode_selection(env)

    assert env.episodes == episodes[:1]
    assert env.setter_calls == 1


@pytest.mark.parametrize("episode_id, matches", [("missing", 0), ("001", 2)])
def test_missing_or_ambiguous_episode_does_not_change_iterator(
    tmp_path: Path, episode_id: str, matches: int,
) -> None:
    scene = str(tmp_path / "scene.glb")
    episodes = [types.SimpleNamespace(scene_id=scene, episode_id="001")]
    if matches == 2:
        episodes *= 2
    env = FakeHabitatEnv(episodes)

    with pytest.raises(ValueError, match=f"matches={matches}"):
        _selector(scene, episode_id)._apply_single_episode_selection(env)

    assert env.episodes == episodes
    assert env.setter_calls == 0


def test_rejects_non_val_unseen_split(tmp_path: Path) -> None:
    scene = str(tmp_path / "scene.glb")
    env = FakeHabitatEnv([types.SimpleNamespace(scene_id=scene, episode_id="001")])
    with pytest.raises(ValueError, match="val_unseen"):
        _selector(scene, "001", split="train")._apply_single_episode_selection(env)
    assert env.setter_calls == 0


@pytest.mark.parametrize("extra", [
    ["--pilot-manifest", "pilot.json", "--strict-full-eval"],
    ["--strict-full-eval"],
    ["--split-id", "0", "--split-num", "2"],
    ["--split", "train"],
    ["--task", "objectnav"],
])
def test_cli_rejects_incompatible_modes(extra: list[str]) -> None:
    args = ["--config", "unused.yaml", "--single-scene-id", "scene.glb",
            "--single-episode-id", "001", *extra]
    with pytest.raises(SystemExit, match="2"):
        main(args)


@pytest.mark.parametrize("flag", ["--single-scene-id", "--single-episode-id"])
def test_cli_requires_both_flags(flag: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["--config", "unused.yaml", flag, "value"])
