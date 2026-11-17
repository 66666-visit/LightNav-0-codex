#!/usr/bin/env python3
"""Validate Local Graph shard protocols against one launcher snapshot before merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_MARKER_FIELDS = (
    "protocol_version", "scope", "prompt_max_chars", "max_landmarks",
    "reactivation_steps",
)
_PROMPT_MODES = frozenset({
    "legacy", "l1_original_only", "l1_organized_only", "l3_original_only",
    "l3_organized_only", "l1_l3_original_concat", "l1_l3_joint",
})
_LOCAL_GRAPH_PROTOCOL = "local_graph_v1"
_PROMPT_PROTOCOL = "l1_l3_prompt_ablation_v1"


def _root_protocol_ok(snapshot: dict, effective: dict) -> bool:
    """Require one unambiguous root protocol while accepting the old conflict shape.

    Older prompt-ablation runs overwrote ``protocol_version`` with the prompt
    protocol, leaving ``local_graph_effective.protocol_version`` unchanged. That
    made the two fields disagree. The current writer keeps
    ``protocol_version`` as the Local Graph protocol and stores the prompt
    protocol separately. For recovery, the old shape is accepted only when an
    explicit ``prompt_effective`` block records the same protocol version and a
    known mode. Nothing else is relaxed.
    """
    root_protocol = snapshot.get("protocol_version")
    if root_protocol == effective.get("protocol_version"):
        prompt_version = snapshot.get("prompt_protocol_version")
        prompt = snapshot.get("prompt_effective")
        if prompt_version is None:
            if prompt is None:
                return True
            return (
                isinstance(prompt, dict)
                and prompt.get("protocol_version") == _PROMPT_PROTOCOL
            )
        return (
            isinstance(prompt, dict)
            and prompt.get("protocol_version") == prompt_version
        )
    prompt = snapshot.get("prompt_effective")
    return bool(
        isinstance(prompt, dict)
        and prompt.get("protocol_version") == root_protocol
        and prompt.get("prompt_mode") in _PROMPT_MODES
    )


def _effective_marker_ok(effective: dict) -> bool:
    return (
        effective.get("protocol_version") == _LOCAL_GRAPH_PROTOCOL
        and effective.get("scope") in {"episode_local", "scene_persistent"}
        and all(
            type(effective.get(key)) is int and effective.get(key) >= 1
            for key in ("prompt_max_chars", "max_landmarks", "reactivation_steps")
        )
    )


def verify_local_graph_shards(root: str | Path, gpu_ids: list[str]) -> None:
    """Reject missing, extra, or differently configured shards before merging."""
    root = Path(root)
    snapshot = json.loads((root / "config_snapshot.yaml").read_text(encoding="utf-8"))
    effective = snapshot.get("local_graph_effective")
    if (not isinstance(effective, dict) or effective.get("enabled") is not True
            or any(key not in effective for key in _MARKER_FIELDS)
            or not _effective_marker_ok(effective)
            or not _root_protocol_ok(snapshot, effective)
            or snapshot.get("memory_effective", {}).get("local_graph_effective") != effective):
        raise ValueError("root snapshot lacks one enabled Local Graph protocol")
    if not gpu_ids or snapshot.get("common", {}).get("gpu_ids") != gpu_ids:
        raise ValueError("root snapshot GPU order differs from launcher shards")
    expected_names = {f"shard_{index}" for index in range(len(gpu_ids))}
    shard_entries = {path.name: path for path in root.glob("shard_*")}
    if set(shard_entries) != expected_names or any(
        not path.is_dir() for path in shard_entries.values()
    ):
        raise ValueError(f"Local Graph shard set differs: {sorted(shard_entries)}")
    expected_marker = {key: effective[key] for key in _MARKER_FIELDS}
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
