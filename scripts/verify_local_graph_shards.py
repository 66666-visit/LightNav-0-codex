#!/usr/bin/env python3
"""Validate Local Graph shard protocols against one launcher snapshot before merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lightnav.memory.local_graph_config import (
    LOCAL_GRAPH_NO_L2_PROTOCOL_VERSION,
    LOCAL_GRAPH_PROTOCOL_VERSION,
)

_MARKER_FIELDS = (
    "protocol_version", "scope", "prompt_max_chars", "max_landmarks",
    "reactivation_steps",
)


def verify_local_graph_shards(root: str | Path, gpu_ids: list[str]) -> None:
    """Reject missing, extra, or differently configured shards before merging."""
    root = Path(root)
    snapshot = json.loads((root / "config_snapshot.yaml").read_text(encoding="utf-8"))
    effective = snapshot.get("local_graph_effective")
    if (not isinstance(effective, dict) or effective.get("enabled") is not True
            or any(key not in effective for key in _MARKER_FIELDS)
            or snapshot.get("protocol_version") != effective.get("protocol_version")
            or snapshot.get("memory_effective", {}).get("local_graph_effective") != effective):
        raise ValueError("root snapshot lacks one enabled Local Graph protocol")
    mid_enabled = effective.get("mid_enabled", True)
    expected_version = (LOCAL_GRAPH_PROTOCOL_VERSION if mid_enabled is True
                        else LOCAL_GRAPH_NO_L2_PROTOCOL_VERSION)
    if type(mid_enabled) is not bool or effective["protocol_version"] != expected_version:
        raise ValueError("root snapshot has an invalid Local Graph L2 protocol")
    if not gpu_ids or snapshot.get("common", {}).get("gpu_ids") != gpu_ids:
        raise ValueError("root snapshot GPU order differs from launcher shards")
    expected_names = {f"shard_{index}" for index in range(len(gpu_ids))}
    shard_entries = {path.name: path for path in root.glob("shard_*")}
    if set(shard_entries) != expected_names or any(
        not path.is_dir() for path in shard_entries.values()
    ):
        raise ValueError(f"Local Graph shard set differs: {sorted(shard_entries)}")
    expected_marker = {key: effective[key] for key in _MARKER_FIELDS}
    if not mid_enabled:
        expected_marker["mid_enabled"] = False
    for name in sorted(expected_names):
        marker_path = root / name / "local_graph_protocol.json"
        if not marker_path.is_file():
            raise ValueError(f"Local Graph shard lacks protocol marker: {marker_path}")
        if json.loads(marker_path.read_text(encoding="utf-8")) != expected_marker:
            raise ValueError(f"Local Graph shard protocol differs from root snapshot: {name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="One Local Graph evaluation output root")
    parser.add_argument("gpu_ids", nargs="+", help="GPU IDs in launcher shard order")
    args = parser.parse_args(argv)
    verify_local_graph_shards(args.root, args.gpu_ids)


if __name__ == "__main__":
    main()
