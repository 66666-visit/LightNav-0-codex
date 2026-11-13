"""V4 configuration and result isolation without starting Habitat."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lightnav.cli.eval_habitat import build_parser, config_from_args
from lightnav.habitat.runner import (
    MemoryConfig,
    _guard_non_v3_v4_output_dir,
    _prepare_v3_output_dir,
    _prepare_v4_output_dir,
)
from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.v3_config import RGMV3Config
from lightnav.memory.v4_config import RGMV4Config

_REPO = Path(__file__).resolve().parents[2]
_LAUNCHER = _REPO / "scripts" / "eval_habitat.sh"


def _environment(root: Path, output: Path, client_args: str) -> dict[str, str]:
    model = root / "model"
    model.mkdir()
    fake_python = root / "fake_python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = '-c' ]; then exit 0; fi\n"
        f"if [ \"$1\" = '-' ]; then exec '{sys.executable}' \"$@\"; fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = dict(os.environ)
    env.update({
        "MODEL_PATH": str(model), "OUTPUT_ROOT": str(output),
        "CLIENT_ARGS": client_args, "CLIENT_PYTHON": str(fake_python),
        "HABITAT_PYTHON": str(fake_python), "GPU_IDS": "0",
        "READY_TIMEOUT_S": "0", "STRICT_FULL_EVAL": "0",
        "PILOT_MANIFEST_PATH": "", "EPISODE_MANIFEST_PATH": "",
    })
    return env


def _launch(root: Path, output: Path, args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_LAUNCHER)], cwd=_REPO,
        env=_environment(root, output, args), text=True,
        capture_output=True, check=False, timeout=30,
    )


def test_cli_v4_settings_do_not_enable_legacy_gate() -> None:
    args = build_parser().parse_args([
        "--model_path", "/not/loaded", "--output_dir", "/tmp/new_v4",
        "--memory_enabled", "--memory_strategy", "hierarchical",
        "--rgm_v4_enabled", "--no-rgm_v4_revisit_enabled",
        "--rgm_v4_prompt_max_records", "2",
    ])
    config = config_from_args(args)
    assert config.memory.rgm_v4.enabled
    assert not config.memory.rgm_v4.revisit_enabled
    assert config.memory.rgm_v4.prompt_max_records == 2
    assert not config.memory.rgm.enabled
    assert not config.memory.rgm_v3.enabled


@pytest.mark.parametrize("other", ["legacy", "v3"])
def test_v4_cannot_share_backend(other: str) -> None:
    kwargs = {"rgm_v4": RGMV4Config(enabled=True)}
    if other == "legacy":
        kwargs["rgm"] = RGMConfig(enabled=True)
    else:
        kwargs["rgm_v3"] = RGMV3Config(enabled=True)
    with pytest.raises(ValueError, match="cannot run together"):
        MemoryConfig(enabled=True, strategy="hierarchical", **kwargs)


def test_v4_protocol_marker_and_resume_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "rgm_v4_pilot_88"
    config = RGMV4Config(enabled=True)
    _prepare_v4_output_dir(str(path), config, resume=False)
    marker = json.loads((path / "rgm_v4_protocol.json").read_text())
    assert marker == {
        "protocol_version": "rgm_v4_v1", "scope": "episode_local",
        "prompt_max_records": 3, "prompt_max_chars": 400,
        "revisit_enabled": True,
    }
    _prepare_v4_output_dir(str(path), config, resume=True)
    with pytest.raises(FileExistsError):
        _prepare_v4_output_dir(str(path), config, resume=False)
    with pytest.raises(ValueError, match="differs"):
        _prepare_v4_output_dir(
            str(path), RGMV4Config(enabled=True, revisit_enabled=False), resume=True,
        )
    with pytest.raises(ValueError):
        _guard_non_v3_v4_output_dir(str(path))
    with pytest.raises(ValueError):
        _prepare_v3_output_dir(str(path), RGMV3Config(enabled=True), resume=False)


def test_v4_rejects_existing_files_and_reserved_v2_v3_paths(tmp_path: Path) -> None:
    config = RGMV4Config(enabled=True)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "old.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        _prepare_v4_output_dir(str(occupied), config, resume=False)
    assert (occupied / "old.txt").read_text() == "keep"
    for name in ("rgm_pilot_88_v2", "rgm_v3_pilot_88"):
        protected = tmp_path / name / "shard_0"
        with pytest.raises(ValueError, match="independent"):
            _prepare_v4_output_dir(str(protected), config, resume=False)
        assert not protected.exists()
    with pytest.raises(FileNotFoundError):
        _prepare_v4_output_dir(str(tmp_path / "missing"), config, resume=True)


def test_tab_separated_v4_flag_rejects_legacy_pilot_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "rgm_pilot_88_v2" / "shard_0"
    result = _launch(
        tmp_path, output,
        "--memory_enabled\t--rgm_v4_enabled\t--memory_strategy hierarchical",
    )
    assert result.returncode != 0
    assert "independent output directory" in result.stderr
    assert not output.exists()


def test_v4_snapshot_selects_independent_backend(tmp_path: Path) -> None:
    output = tmp_path / "rgm_v4_test"
    result = _launch(
        tmp_path, output,
        "--memory_enabled --memory_strategy hierarchical "
        "--memory_layers short,mid,long --rgm_v4_enabled "
        "--no-rgm_v4_revisit_enabled",
    )
    # The fake server exits before readiness. No Habitat evaluation is launched.
    assert result.returncode != 0
    snapshot = json.loads((output / "config_snapshot.yaml").read_text())
    assert snapshot["protocol_version"] == "rgm_v4_v1"
    assert snapshot["memory_effective"]["rgm_v4"]["enabled"]
    assert not snapshot["memory_effective"]["rgm_v4"]["revisit_enabled"]
    assert not snapshot["memory_effective"]["rgm"]["enabled"]
    assert "rgm_v3" not in snapshot["memory_effective"]
    assert "--rgm_v4_enabled" in snapshot["memory_args"]
