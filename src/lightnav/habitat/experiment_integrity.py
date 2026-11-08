"""Episode manifests and completeness gates for full Habitat evaluations.

This module contains only evaluation metadata. The manifest is never passed to
the navigation policy or MemoryManager.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from statistics import fmean
from typing import Any


def _key(row: dict[str, Any], *, result: bool = False) -> tuple[str, str]:
    scene = row.get("scene_id")
    episode = row.get("habitat_episode_id" if result else "episode_id")
    if scene is None or episode is None or not str(scene) or not str(episode):
        raise ValueError("episode record requires nonempty scene_id and episode_id")
    return str(scene), str(episode)


def validate_manifest(data: Any) -> dict[str, Any]:
    """Return a canonical manifest, rejecting duplicate or missing episode IDs."""
    if not isinstance(data, dict) or not isinstance(data.get("split"), str):
        raise ValueError("manifest requires a split string")
    rows = data.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("manifest requires an episodes list")
    expected = data.get("expected_count")
    if type(expected) is not int or expected != len(rows):
        raise ValueError("manifest expected_count differs from episode list length")
    seen: set[tuple[str, str]] = set()
    episodes: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("manifest episode must be an object")
        key = _key(row)
        if key in seen:
            raise ValueError(f"duplicate episode in manifest: {key}")
        seen.add(key)
        episodes.append({"scene_id": key[0], "episode_id": key[1]})
    return {"split": data["split"], "expected_count": expected, "episodes": episodes}


def read_manifest(path: str | Path) -> dict[str, Any]:
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def write_or_match_manifest(path: str | Path, data: dict[str, Any]) -> None:
    """Create a manifest once; existing manifests must match exactly."""
    target = Path(path)
    canonical = validate_manifest(data)
    if target.exists():
        if read_manifest(target) != canonical:
            raise ValueError(f"episode manifest differs from existing file: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(canonical, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def iter_jsonl(path: str | Path):
    """Iterate JSONL rows without retaining a full step diagnostic file."""
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row {number} in {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {number} in {path} is not an object")
            yield row


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a small result file; malformed rows fail instead of disappearing."""
    return list(iter_jsonl(path))


def repair_interrupted_jsonl_tail(path: str | Path) -> bool:
    """Archive only a malformed final row left by an interrupted append.

    Earlier malformed rows still fail. A valid final row without a newline is
    retained and terminated before further appends. This runs only on resume.
    """
    target = Path(path)
    if not target.is_file():
        return False
    with target.open("rb") as stream:
        number = 0
        while True:
            start = stream.tell()
            raw = stream.readline()
            if not raw:
                return False
            number += 1
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                if stream.read(1):
                    raise ValueError(f"invalid non-final JSONL row {number} in {path}") from exc
                archive = target.with_name(
                    f"{target.name}.interrupted_tail_{time.time_ns()}"
                )
                archive.write_bytes(raw)
                with target.open("r+b") as writable:
                    writable.truncate(start)
                    writable.flush()
                    os.fsync(writable.fileno())
                return True
            if not raw.endswith(b"\n"):
                with target.open("ab") as writable:
                    writable.write(b"\n")
                    writable.flush()
                    os.fsync(writable.fileno())
                return False


def archive_unfinished_diagnostics(
    path: str | Path, completed: set[tuple[str, str]],
) -> int:
    """On resume, preserve and remove step rows lacking a completed result.

    A process can end after writing an episode's diagnostics but before its
    result row. Those orphan rows must not be counted with the next attempt.
    They are copied to a separate archive, never silently discarded.
    """
    target = Path(path)
    if not target.is_file():
        return 0
    orphan_count = sum(
        _key(row, result=True) not in completed for row in iter_jsonl(target)
    )
    if not orphan_count:
        return 0
    archive = target.with_name(f"{target.name}.interrupted_{time.time_ns()}")
    temporary = target.with_name(f"{target.name}.repair_{time.time_ns()}.tmp")
    with archive.open("x", encoding="utf-8") as interrupted:
        with temporary.open("x", encoding="utf-8") as retained:
            for row in iter_jsonl(target):
                stream = retained if _key(row, result=True) in completed else interrupted
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            retained.flush()
            os.fsync(retained.fileno())
        interrupted.flush()
        os.fsync(interrupted.fileno())
    os.replace(temporary, target)
    return orphan_count


def check_coverage(
    rows: list[dict[str, Any]], manifest: dict[str, Any], *, complete: bool,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Reject duplicate or unknown results; optionally require the full split."""
    canonical = validate_manifest(manifest)
    expected = {_key(row) for row in canonical["episodes"]}
    actual: set[tuple[str, str]] = set()
    for row in rows:
        key = _key(row, result=True)
        if key in actual:
            raise ValueError(f"duplicate result episode: {key}")
        actual.add(key)
    extras = actual - expected
    if extras:
        raise ValueError(f"result contains episodes outside manifest: {sorted(extras)[:5]}")
    missing = expected - actual
    if complete and missing:
        raise ValueError(f"missing {len(missing)} episodes: {sorted(missing)[:5]}")
    return actual, missing


def combine_shard_manifests(paths: list[str | Path]) -> dict[str, Any]:
    """Combine server-ready manifests in shard order, rejecting overlap."""
    if not paths:
        raise ValueError("at least one shard manifest is required")
    parts = [read_manifest(path) for path in paths]
    split = parts[0]["split"]
    episodes: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for part in parts:
        if part["split"] != split:
            raise ValueError("shard manifests have different dataset splits")
        if not part["episodes"]:
            raise ValueError("an evaluation shard contains zero episodes")
        for row in part["episodes"]:
            key = _key(row)
            if key in seen:
                raise ValueError(f"duplicate episode across shards: {key}")
            seen.add(key)
            episodes.append(row)
    return {"split": split, "expected_count": len(episodes), "episodes": episodes}


def check_pilot_selection(manifest: dict[str, Any], pilot_selection: str | Path) -> None:
    """Require server-loaded IDs to match the fixed, evaluator-only Pilot sample."""
    canonical = validate_manifest(manifest)
    data = json.loads(Path(pilot_selection).read_text(encoding="utf-8"))
    rows = data.get("episodes") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict) or data.get("dataset_split") != "val_unseen"
        or data.get("expected_count") != 220 or data.get("per_scene") != 20
        or not isinstance(rows, list) or len(rows) != 220
        or canonical["split"] != "val_unseen" or canonical["expected_count"] != 220
    ):
        raise ValueError("Pilot selection requires exactly 220 val_unseen Episode IDs")
    selected = set()
    scene_counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Pilot selection contains a non-object Episode")
        scene = row.get("scene_id")
        episode = row.get("habitat_episode_id")
        if not isinstance(scene, str) or not scene or not isinstance(episode, str) or not episode:
            raise ValueError("Pilot selection contains an invalid scene or Episode ID")
        key = (scene, episode)
        if key in selected:
            raise ValueError(f"duplicate Episode in Pilot selection: {key}")
        selected.add(key)
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
    if len(scene_counts) != 11 or any(count != 20 for count in scene_counts.values()):
        raise ValueError("Pilot selection must contain 20 Episodes from each of 11 scenes")
    loaded = {_key(row) for row in canonical["episodes"]}
    if loaded != selected:
        raise ValueError(
            "loaded Episode IDs differ from fixed Pilot selection: "
            f"missing={sorted(selected - loaded)[:5]}, extra={sorted(loaded - selected)[:5]}"
        )


def collect_ready_manifests(
    ready_files: list[str | Path], output: str | Path, shared: str | Path | None = None,
    expected_count: int | None = None, pilot_selection: str | Path | None = None,
) -> dict[str, Any]:
    combined = combine_shard_manifests(ready_files)
    if expected_count is not None and combined["expected_count"] != expected_count:
        raise ValueError(
            f"loaded {combined['expected_count']} episodes; expected {expected_count} "
            "for this split"
        )
    if pilot_selection is not None:
        check_pilot_selection(combined, pilot_selection)
    write_or_match_manifest(output, combined)
    if shared is not None:
        write_or_match_manifest(shared, combined)
    return combined


def verify_group(
    root: str | Path, *, shared: str | Path | None = None,
    pilot_selection: str | Path | None = None,
) -> dict[str, Any]:
    """Require exact shard and merged result coverage before marking completion."""
    root = Path(root)
    manifest = read_manifest(root / "episode_manifest.json")
    if pilot_selection is not None:
        check_pilot_selection(manifest, pilot_selection)
    if shared is not None and read_manifest(shared) != manifest:
        raise ValueError("group episode manifest differs from shared manifest")
    shards = sorted(
        (path for path in root.glob("shard_*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("shard_")),
    )
    if not shards:
        raise ValueError("group has no shard directories")
    shard_manifests = [shard / "episode_manifest.json" for shard in shards]
    if combine_shard_manifests(shard_manifests) != manifest:
        raise ValueError("shard manifests differ from group manifest")
    shard_records: dict[tuple[str, str], dict[str, Any]] = {}
    for shard, shard_manifest_path in zip(shards, shard_manifests):
        rows = read_jsonl(shard / "results.jsonl")
        check_coverage(
            rows, read_manifest(shard_manifest_path), complete=True,
        )
        shard_records.update((_key(row, result=True), row) for row in rows)
    merged = read_jsonl(root / "results.jsonl")
    check_coverage(merged, manifest, complete=True)
    expected_steps: dict[tuple[str, str], int] = {}
    for row in merged:
        key = _key(row, result=True)
        without_shard = dict(row)
        without_shard.pop("shard", None)
        if without_shard != shard_records[key]:
            raise ValueError(f"merged Episode differs from its shard result: {key}")
        steps = row.get("steps")
        if type(steps) is not int or not 1 <= steps <= 500:
            raise ValueError(f"invalid formal episode step count for {key}: {steps}")
        if row.get("terminated") is not True and row.get("truncated") is not True:
            raise ValueError(f"formal episode lacks Habitat terminal signal: {key}")
        if row.get("forced_stop") is True:
            raise ValueError(f"formal episode used a forced STOP: {key}")
        for metric in ("success", "oracle_success", "spl", "ndtw", "final_distance"):
            value = row.get(metric)
            if type(value) not in (bool, int, float) or not math.isfinite(value):
                raise ValueError(f"formal episode lacks measured {metric}: {key}")
            if value < 0 or (metric != "final_distance" and value > 1):
                raise ValueError(f"formal episode has invalid {metric}: {key}")
        episode_time = row.get("episode_time_sec")
        if (
            not isinstance(episode_time, (int, float))
            or not math.isfinite(episode_time) or episode_time < 0
        ):
            raise ValueError(f"missing measured episode time for {key}")
        expected_steps[key] = steps
    observed_steps: dict[tuple[str, str], set[int]] = {
        key: set() for key in expected_steps
    }
    for row in iter_jsonl(root / "memory_diagnostics.jsonl"):
        key = _key(row, result=True)
        if key not in expected_steps:
            raise ValueError(f"diagnostic has no completed result episode: {key}")
        step = row.get("step")
        if type(step) is not int or step in observed_steps[key]:
            raise ValueError(f"duplicate or invalid diagnostic step for {key}: {step}")
        latency = row.get("inference_latency_ms")
        prompt_chars = row.get("prompt_chars")
        if (
            not isinstance(latency, (int, float)) or not math.isfinite(latency)
            or latency < 0 or type(prompt_chars) is not int or prompt_chars < 0
        ):
            raise ValueError(f"diagnostic lacks measured inference/prompt size for {key}")
        observed_steps[key].add(step)
    for key, steps in expected_steps.items():
        if observed_steps[key] != set(range(steps)):
            raise ValueError(f"diagnostic steps incomplete for {key}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("num_episodes") != manifest["expected_count"]:
        raise ValueError("summary episode count differs from loaded manifest")
    metrics = summary.get("metrics")
    required_metrics = (
        "SR_success_rate_pct", "OS_oracle_success_pct", "SPL_pct", "NDTW_pct",
        "NE_navigation_error_m",
    )
    if not isinstance(metrics, dict) or any(
        type(metrics.get(name)) not in (int, float)
        or not math.isfinite(metrics[name]) for name in required_metrics
    ):
        raise ValueError("summary benchmark metrics are absent")
    summary_episodes = summary.get("episodes")
    if (
        not isinstance(summary_episodes, list)
        or len(summary_episodes) != len(merged)
        or any(not isinstance(row, dict) for row in summary_episodes)
    ):
        raise ValueError("summary episode list differs from merged results")
    summary_keys = [_key(row, result=True) for row in summary_episodes]
    if len(set(summary_keys)) != len(summary_keys) or set(summary_keys) != set(expected_steps):
        raise ValueError("summary Episode IDs differ from merged results")
    benchmark_means = {
        "SR_success_rate_pct": (100.0 * fmean(float(row["success"]) for row in merged), 0.02),
        "OS_oracle_success_pct": (
            100.0 * fmean(float(row["oracle_success"]) for row in merged), 0.02,
        ),
        "SPL_pct": (100.0 * fmean(float(row["spl"]) for row in merged), 0.02),
        "NDTW_pct": (100.0 * fmean(float(row["ndtw"]) for row in merged), 0.02),
        "NE_navigation_error_m": (fmean(float(row["final_distance"]) for row in merged), 0.002),
    }
    for name, (measured, tolerance) in benchmark_means.items():
        if abs(float(metrics[name]) - measured) > tolerance:
            raise ValueError(f"summary {name} disagrees with Episode results")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    complete = {"expected_count": manifest["expected_count"], "manifest_sha256": digest}
    marker = root / ("pilot_complete.json" if pilot_selection is not None else "formal_complete.json")
    opposite = root / ("formal_complete.json" if pilot_selection is not None else "pilot_complete.json")
    if opposite.exists():
        raise ValueError(f"conflicting evaluation completion marker: {opposite}")
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != complete:
            raise ValueError("existing completion marker does not match this evaluation")
    else:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(complete, stream, indent=2)
            stream.write("\n")
    return complete


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="collect actual shard manifests before inference")
    collect.add_argument("--output", required=True)
    collect.add_argument("--shared")
    collect.add_argument("--expected-count", type=int)
    collect.add_argument("--pilot-selection", help="fixed Pilot sample; compare exact loaded IDs")
    collect.add_argument("ready_files", nargs="+")
    verify = sub.add_parser("verify", help="verify every loaded episode completed once")
    verify.add_argument("root")
    verify.add_argument("--shared")
    verify.add_argument("--pilot-selection", help="write pilot_complete.json after exact coverage")
    args = parser.parse_args(argv)
    if args.command == "collect":
        manifest = collect_ready_manifests(
            args.ready_files, args.output, args.shared, args.expected_count,
            args.pilot_selection,
        )
        print(f"Loaded {manifest['expected_count']} episodes from {len(args.ready_files)} shards")
    else:
        complete = verify_group(
            args.root, shared=args.shared, pilot_selection=args.pilot_selection,
        )
        label = "Pilot" if args.pilot_selection else "Formal"
        print(f"{label} evaluation complete: {complete['expected_count']} unique episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
