#!/usr/bin/env python3
"""Finish one L1/L3 group from complete shards without rerunning inference.

This is the safe recovery path for a run that finished inference but failed in
post-processing (for example the old ``protocol_version`` conflict between the
Local Graph and prompt protocols). It verifies shard completeness and protocol
metadata first, then merges results/diagnostics, runs the formal integrity gate
and writes the completion marker. It never deletes or rewrites shard results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from lightnav.habitat.experiment_integrity import (
    _key as _episode_identity,
)
from lightnav.habitat.experiment_integrity import (
    combine_shard_manifests,
    read_jsonl,
    read_manifest,
    verify_group,
)
from lightnav.habitat.merge import merge_results
from lightnav.habitat.prompt_artifacts import (
    PROMPT_DIAGNOSTICS_FILENAME,
    PROMPT_PROTOCOL_FILENAME,
    merge_prompt_artifacts,
    validate_group_prompt_artifacts,
)
from lightnav.memory.prompt_modes import (
    PROMPT_PROTOCOL_VERSION,
    effective_layer_activation,
    mode_plan,
)
from scripts.verify_local_graph_shards import verify_local_graph_shards

_MEMORY_DIAGNOSTICS = "memory_diagnostics.jsonl"
_RESULTS = "results.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _manifest_digest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_key(row: dict[str, Any]) -> tuple[str, str]:
    return _episode_identity(row, result=True)


def _same_manifest(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("split") == right.get("split")
        and left.get("expected_count") == right.get("expected_count")
        and [_episode_identity(row) for row in left.get("episodes", [])]
        == [_episode_identity(row) for row in right.get("episodes", [])]
    )


def _shard_dirs(group_dir: Path, gpu_ids: list[str]) -> list[Path]:
    expected = [f"shard_{index}" for index in range(len(gpu_ids))]
    actual = sorted(path.name for path in group_dir.glob("shard_*") if path.is_dir())
    if actual != sorted(expected):
        raise ValueError(
            f"shard set differs: expected {expected}, found {actual}"
        )
    return [group_dir / name for name in expected]


def _precheck(
    group_dir: Path, gpu_ids: list[str], expected_episodes: int,
    *, yolo_model: str | Path | None = None,
    seg_model: str | Path | None = None,
) -> tuple[dict[str, Any], list[Path], list[dict[str, Any]]]:
    if not group_dir.is_dir():
        raise FileNotFoundError(f"group directory is missing: {group_dir}")
    if (group_dir / "pilot_complete.json").is_file():
        raise FileExistsError(
            f"group already has a completion marker: {group_dir / 'pilot_complete.json'}"
        )
    snapshot_path = group_dir / "config_snapshot.yaml"
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"root config snapshot is missing: {snapshot_path}")
    snapshot = _read_json(snapshot_path)
    common = snapshot.get("common", {})
    if common.get("gpu_ids") != gpu_ids:
        raise ValueError("root snapshot GPU order differs from the recovery GPU list")
    provenance_path = group_dir.parent / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"run provenance is required for safe recovery: {provenance_path}"
        )
    provenance = _read_json(provenance_path)
    checks = {
        "model_weights_sha256": provenance.get("checkpoint_sha256"),
        "model_config_sha256": provenance.get("checkpoint_config_sha256"),
        "eval_config_sha256": provenance.get("checkpoint_eval_config_sha256"),
        "habitat_config_sha256": provenance.get("habitat_config_sha256"),
        "seed": provenance.get("seed"),
        "gpu_ids": provenance.get("gpu_ids"),
    }
    for key, expected in checks.items():
        if common.get(key) != expected:
            raise ValueError(
                f"shard config snapshot differs from run provenance at {key}"
            )
    shards = _shard_dirs(group_dir, gpu_ids)
    # Reuse the exact Local Graph protocol gate that failed in the original
    # post-processing run. The fixed verifier accepts both the new separated
    # fields and the old recovered conflict shape, without relaxing markers.
    verify_local_graph_shards(group_dir, gpu_ids)

    root_manifest_path = group_dir / "episode_manifest.json"
    if not root_manifest_path.is_file():
        raise FileNotFoundError(f"root episode manifest is missing: {root_manifest_path}")
    root_manifest = read_manifest(root_manifest_path)
    if root_manifest["expected_count"] != expected_episodes:
        raise ValueError(
            f"root manifest has {root_manifest['expected_count']} Episodes; "
            f"expected {expected_episodes}"
        )
    expected_manifest_digest = provenance.get("episode_manifest_canonical_sha256")
    if expected_manifest_digest and _manifest_digest(root_manifest) != expected_manifest_digest:
        raise ValueError("group manifest canonical digest differs from run provenance")
    for label, path, expected_key in (
        ("L1 YOLO", yolo_model, "l1_yolo_model_sha256"),
        ("L3 segmentation", seg_model, "l3_seg_model_sha256"),
    ):
        expected = provenance.get(expected_key)
        if path is not None and expected:
            if _sha256(Path(path)) != expected:
                raise ValueError(f"{label} weights differ from run provenance")
    shard_manifest_paths: list[Path] = []
    resolved_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for shard in shards:
        manifest_path = shard / "episode_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"shard manifest is missing: {manifest_path}")
        read_manifest(manifest_path)
        shard_manifest_paths.append(manifest_path)
        results_path = shard / _RESULTS
        if not results_path.is_file():
            raise FileNotFoundError(f"shard results are missing: {results_path}")
        rows = read_jsonl(results_path)
        for row in rows:
            key = _episode_key(row)
            if key in seen:
                raise ValueError(f"duplicate Episode across shards: {key}")
            seen.add(key)
            if row.get("forced_stop") is True:
                raise ValueError(f"shard result used a forced STOP: {key}")
            resolved_rows.append(row)
    combined = combine_shard_manifests(shard_manifest_paths)
    expected_keys = {
        _episode_identity(row) for row in combined["episodes"]
    }
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)[:5]
        extra = sorted(seen - expected_keys)[:5]
        raise ValueError(
            f"shard results do not match shard manifests: missing={missing}, extra={extra}"
        )
    if len(seen) != expected_episodes:
        raise ValueError(
            f"shards contain {len(seen)} unique Episodes; expected {expected_episodes}"
        )
    if not _same_manifest(combined, root_manifest):
        raise ValueError("combined shard manifests differ from the root manifest")
    prompt_effective = snapshot.get("prompt_effective")
    if prompt_effective is not None:
        mode = str(prompt_effective.get("prompt_mode"))
        expected_strategy = (
            "original_char_budget" if mode == "legacy" else "token_aware_context"
        )
        expected_budget = (
            prompt_effective.get("original_char_budget_chars")
        )
        legacy_activation = {
            "mode": "legacy", "l1_layer": "original",
            "l3_layer": "original_spatial_when_inject_else_l1",
            "combination": "legacy", "l1_prompt_enabled": True,
            "l3_prompt_enabled": "original_inject_mode",
            "joint_alignment_enabled": False,
        }
        expected_activation = (
            legacy_activation if mode == "legacy"
            else effective_layer_activation(mode_plan(mode))
        )
        reference_protocol: dict[str, Any] | None = None
        for shard in shards:
            for name in (PROMPT_PROTOCOL_FILENAME, PROMPT_DIAGNOSTICS_FILENAME):
                if not (shard / name).is_file():
                    raise FileNotFoundError(
                        f"prompt artifact is missing from {shard.name}: {name}"
                    )
            protocol = _read_json(shard / PROMPT_PROTOCOL_FILENAME)
            if reference_protocol is None:
                reference_protocol = protocol
            elif protocol != reference_protocol:
                raise ValueError(
                    f"shard prompt protocols differ within the group: {shard.name}"
                )
            budget = protocol.get("budget", {})
            if not isinstance(budget, dict) or budget.get("strategy") != expected_strategy:
                raise ValueError(
                    f"shard prompt protocol budget strategy differs: {shard.name}; "
                    "old character-budget shards cannot be mixed with the token-budget experiment"
                )
            if protocol.get("protocol_version") != PROMPT_PROTOCOL_VERSION:
                raise ValueError(f"shard prompt protocol version differs: {shard.name}")
            if protocol.get("mode") != mode:
                raise ValueError(
                    f"shard prompt protocol mode differs from root config: {shard.name}"
                )
            if protocol.get("layer_activation") != expected_activation:
                raise ValueError(f"shard prompt layer activation differs: {shard.name}")
            protocol_budget = budget.get("original_char_budget_chars")
            if expected_budget is not None and protocol_budget != expected_budget:
                raise ValueError(f"shard prompt character budget differs: {shard.name}")
            if mode == "legacy" and budget.get("chars") != protocol_budget:
                raise ValueError(f"legacy shard prompt character budget differs: {shard.name}")
    return snapshot, shards, resolved_rows


def _check_prompt_step_coverage(group_dir: Path) -> None:
    results = read_jsonl(group_dir / _RESULTS)
    diagnostics = read_jsonl(group_dir / PROMPT_DIAGNOSTICS_FILENAME)
    expected: dict[tuple[str, str], int] = {}
    for row in results:
        steps = row.get("steps")
        if type(steps) is not int or steps < 1:
            raise ValueError("merged result has no valid step count")
        expected[_episode_identity(row, result=True)] = steps - (
            1 if row.get("forced_stop") is True else 0
        )
    observed: dict[tuple[str, str], set[int]] = {key: set() for key in expected}
    for row in diagnostics:
        key = _episode_identity(row, result=True)
        if key not in observed:
            raise ValueError(f"prompt diagnostic has no completed result: {key}")
        step = row.get("step")
        if type(step) is not int or step in observed[key]:
            raise ValueError(f"duplicate or invalid prompt diagnostic step: {key}@{step}")
        observed[key].add(step)
    for key, steps in expected.items():
        if observed[key] != set(range(steps)):
            raise ValueError(
                f"prompt diagnostics incomplete for {key}: "
                f"expected {steps}, observed {len(observed[key])}"
            )


def _concatenate_memory_diagnostics(group_dir: Path, shards: list[Path]) -> int:
    temporary = group_dir / f".{_MEMORY_DIAGNOSTICS}.tmp"
    rows = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for shard in shards:
            source = shard / _MEMORY_DIAGNOSTICS
            if not source.is_file():
                raise FileNotFoundError(f"shard diagnostics are missing: {source}")
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"invalid diagnostic row in {source}")
                stream.write(line + "\n")
                rows += 1
    temporary.replace(group_dir / _MEMORY_DIAGNOSTICS)
    return rows


def finalize_group(
    group_dir: str | Path,
    *,
    gpu_ids: list[str],
    episode_manifest: str | Path,
    pilot_manifest: str | Path | None = None,
    yolo_model: str | Path | None = None,
    seg_model: str | Path | None = None,
    expected_episodes: int = 88,
    check_only: bool = False,
) -> dict[str, Any]:
    group_dir = Path(group_dir)
    snapshot, shards, rows = _precheck(
        group_dir, gpu_ids, expected_episodes,
        yolo_model=yolo_model, seg_model=seg_model,
    )
    shared_manifest = read_manifest(episode_manifest)
    if not _same_manifest(
        shared_manifest, read_manifest(group_dir / "episode_manifest.json")
    ):
        raise ValueError("group manifest differs from the shared 88-Episode manifest")
    if check_only:
        return {
            "status": "postprocess_ready",
            "group_dir": str(group_dir),
            "shards": [str(shard) for shard in shards],
            "episodes": len(rows),
        }

    # Merge only after every shard and every Episode was checked above.
    merge_results(shards, group_dir)
    diagnostics_rows = _concatenate_memory_diagnostics(group_dir, shards)
    prompt_effective = snapshot.get("prompt_effective")
    if prompt_effective is not None:
        merge_prompt_artifacts(group_dir, shards)
        # Validate before the completion marker is written, so a marker never
        # claims a group whose prompt artifacts are incomplete.
        validate_group_prompt_artifacts(
            group_dir, expected_mode=str(prompt_effective["prompt_mode"]),
        )
        _check_prompt_step_coverage(group_dir)
    verify_group(
        group_dir,
        shared=episode_manifest,
        pilot_selection=pilot_manifest,
    )
    provenance = {
        "status": "complete",
        "group_dir": str(group_dir),
        "gpu_ids": gpu_ids,
        "episodes": len(rows),
        "shards": len(shards),
        "memory_diagnostic_rows": diagnostics_rows,
        "config_snapshot_sha256": _sha256(group_dir / "config_snapshot.yaml"),
        "episode_manifest_sha256": _sha256(group_dir / "episode_manifest.json"),
        "shard_results_sha256": {
            shard.name: _sha256(shard / _RESULTS) for shard in shards
        },
        "completion_marker": (
            "pilot_complete.json" if pilot_manifest is not None
            else "formal_complete.json"
        ),
    }
    temporary = group_dir / ".postprocess_provenance.json.tmp"
    temporary.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    temporary.replace(group_dir / "postprocess_provenance.json")
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_dir", type=Path)
    parser.add_argument(
        "--gpu-ids", nargs="+", required=True,
        help="GPU IDs in the launcher shard order, e.g. 0 or 0 1",
    )
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--pilot-manifest", default=None)
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--seg-model", default=None)
    parser.add_argument("--expected-episodes", type=int, default=88)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = finalize_group(
            args.group_dir,
            gpu_ids=args.gpu_ids,
            episode_manifest=args.episode_manifest,
            pilot_manifest=args.pilot_manifest,
            yolo_model=args.yolo_model,
            seg_model=args.seg_model,
            expected_episodes=args.expected_episodes,
            check_only=args.check_only,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"group finalization failed: {exc}", file=__import__("sys").stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
