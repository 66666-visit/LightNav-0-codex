#!/usr/bin/env python3
"""Select a fixed, distance-stratified R2R val_unseen pilot manifest.

This is an evaluation-only tool. Route lengths and trajectory IDs are written
to the pilot manifest for audit and are never supplied to the navigation policy.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLIT = "val_unseen"
SCENE_COUNT = 11
PILOT_PER_SCENE = 20
SEED = 42
DISTANCE_GROUPS = ("short", "medium", "long")
SAMPLING_METHOD = "adaptive_scene_quota_geodesic_tertiles_trajectory_diverse_v2"
SCENE_PREFIX = "data/scene_datasets/"


def resolve_dataset_path(data_path: str | Path) -> Path:
    """Resolve DATA_PATH as a dataset file, template, or dataset root."""
    raw = str(data_path)
    if not raw:
        raise ValueError("DATA_PATH must identify the R2R val_unseen dataset")
    if "{split}" in raw:
        raw = raw.format(split=SPLIT)
    path = Path(raw).expanduser()
    if path.is_dir():
        candidates = [
            path / "R2R_VLNCE_v1-3_preprocessed" / SPLIT / f"{SPLIT}.json.gz",
            path / SPLIT / f"{SPLIT}.json.gz",
        ]
        existing = [candidate for candidate in candidates if candidate.is_file()]
        if len(existing) != 1:
            raise ValueError(
                f"DATA_PATH must resolve to exactly one {SPLIT}.json.gz: {path}"
            )
        path = existing[0]
    if not path.is_file() or path.name != f"{SPLIT}.json.gz":
        raise ValueError(f"R2R {SPLIT} dataset not found: {path}")
    return path.resolve()


def habitat_scene_id(scene_id: str, scenes_dir: str | Path | None) -> str:
    """Apply the same scene-path transformation as VLNCEDatasetV1.from_json."""
    if scenes_dir is None or str(scenes_dir) == "":
        return scene_id
    suffix = scene_id.removeprefix(SCENE_PREFIX)
    return os.path.join(str(scenes_dir), suffix)


def _route_length(row: dict[str, Any], episode_id: str) -> float:
    info = row.get("info")
    values = []
    if "geodesic_distance" in row:
        values.append(row["geodesic_distance"])
    if isinstance(info, dict) and "geodesic_distance" in info:
        values.append(info["geodesic_distance"])
    if not values:
        raise ValueError(
            f"Episode {episode_id} lacks geodesic_distance; cannot stratify by "
            "verified route length"
        )
    if len(values) == 2 and values[0] != values[1]:
        raise ValueError(f"Episode {episode_id} has conflicting geodesic distances")
    value = values[0]
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Episode {episode_id} has invalid geodesic_distance: {value!r}")
    return float(value)


def _seed_for(scene_id: str, group: str, seed: int) -> int:
    payload = f"{seed}\0{scene_id}\0{group}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _allocate_scene_quotas(
    full_scene_counts: dict[str, int], per_scene: int,
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """Fill scene deficits by distinct surplus scenes in deterministic rank order."""
    target = len(full_scene_counts) * per_scene
    if sum(full_scene_counts.values()) < target:
        raise ValueError(
            f"dataset has only {sum(full_scene_counts.values())} Episodes; "
            f"Pilot needs {target}"
        )
    quotas = {
        scene_id: min(count, per_scene)
        for scene_id, count in full_scene_counts.items()
    }
    shortfall = target - sum(quotas.values())
    # Every pass gives at most one extra seat to each scene. This avoids
    # concentrating two missing seats in the single largest scene. Because
    # each candidate starts at the same base quota, ranking by remaining
    # candidate count is equivalent to ranking by original scene size.
    recipients = sorted(
        (scene_id for scene_id in quotas if full_scene_counts[scene_id] > per_scene),
        key=lambda scene_id: (-(full_scene_counts[scene_id] - per_scene), scene_id),
    )
    while shortfall:
        assigned = 0
        for scene_id in recipients:
            if quotas[scene_id] >= full_scene_counts[scene_id]:
                continue
            quotas[scene_id] += 1
            shortfall -= 1
            assigned += 1
            if not shortfall:
                break
        if not assigned:
            raise ValueError(
                f"cannot redistribute {shortfall} missing Pilot seats; "
                "no scenes have spare Episodes"
            )
    allocation: dict[str, dict[str, Any]] = {}
    for scene_id, count in sorted(full_scene_counts.items()):
        quota = quotas[scene_id]
        if count < per_scene:
            reason = "all_available_below_base"
        elif quota > per_scene:
            reason = "redistributed_shortfall"
        else:
            reason = "base_quota"
        allocation[scene_id] = {
            "available_episodes": count,
            "base_quota": per_scene,
            "selected_quota": quota,
            "adjustment": quota - per_scene,
            "reason": reason,
        }
    return quotas, allocation


def _distance_quotas(quota: int) -> dict[str, int]:
    """Balance selected routes across short/medium/long, short first on ties."""
    base, remainder = divmod(quota, 3)
    return {
        group: base + (index < remainder)
        for index, group in enumerate(DISTANCE_GROUPS)
    }


def _select_scene(
    rows: list[dict[str, Any]], scene_id: str, seed: int, quota: int,
) -> list[dict[str, Any]]:
    """Select a scene's balanced route-length strata with trajectory diversity."""
    ordered = sorted(rows, key=lambda row: (row["route_length"], row["habitat_episode_id"]))
    if len(ordered) < quota:
        raise ValueError(
            f"scene {scene_id} has only {len(ordered)} Episodes; need {quota}"
        )
    # Rank tertiles keep all source Episodes eligible without arbitrary distance
    # thresholds or silently shrinking the sample at tied values. Put the
    # remainder into shorter strata so 20 candidates form 7/7/6, not 6/7/7.
    bucket_sizes = _distance_quotas(len(ordered))
    first = bucket_sizes["short"]
    second = first + bucket_sizes["medium"]
    buckets = {
        "short": ordered[:first],
        "medium": ordered[first:second],
        "long": ordered[second:],
    }
    quotas = _distance_quotas(quota)
    counts = Counter(row["trajectory_id"] for row in ordered)
    used_trajectories: set[str] = set()
    selected: list[dict[str, Any]] = []
    for group, group_quota in quotas.items():
        candidates = list(buckets[group])
        if len(candidates) < group_quota:
            raise ValueError(
                f"scene {scene_id} {group} stratum has {len(candidates)} "
                f"Episodes; need {group_quota}"
            )
        rng = random.Random(_seed_for(scene_id, group, seed))
        rng.shuffle(candidates)
        for _ in range(group_quota):
            # Prefer a new trajectory, then the rarer trajectory among all
            # source Episodes. The seeded shuffle breaks remaining ties.
            choice = min(
                candidates,
                key=lambda row: (
                    row["trajectory_id"] in used_trajectories,
                    counts[row["trajectory_id"]],
                ),
            )
            candidates.remove(choice)
            used_trajectories.add(choice["trajectory_id"])
            selected.append({**choice, "distance_group": group})
    return selected


def sample_pilot(
    dataset_path: str | Path,
    *,
    scenes_dir: str | Path | None = None,
    seed: int = SEED,
    per_scene: int = PILOT_PER_SCENE,
) -> dict[str, Any]:
    """Read source Episodes and return the immutable pilot selection record."""
    if per_scene != PILOT_PER_SCENE:
        raise ValueError("this pilot protocol requires base target 20 Episodes per scene")
    if seed != SEED:
        raise ValueError("this pilot protocol requires sample seed 42")
    source = resolve_dataset_path(dataset_path)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("episodes"), list):
        raise ValueError("dataset must contain an episodes list")
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[tuple[str, str]] = set()
    for index, episode in enumerate(data["episodes"]):
        if not isinstance(episode, dict):
            raise ValueError(f"dataset Episode {index} is not an object")
        raw_scene = episode.get("scene_id")
        raw_id = episode.get("episode_id")
        raw_trajectory = episode.get("trajectory_id")
        if not all(isinstance(value, (str, int)) and str(value) for value in (
            raw_scene, raw_id, raw_trajectory,
        )):
            raise ValueError(
                f"dataset Episode {index} lacks scene_id, episode_id, or trajectory_id"
            )
        scene_id = habitat_scene_id(str(raw_scene), scenes_dir)
        episode_id = str(raw_id)
        if (scene_id, episode_id) in seen_keys:
            raise ValueError(f"duplicate dataset Episode: {(scene_id, episode_id)}")
        seen_keys.add((scene_id, episode_id))
        by_scene[scene_id].append({
            "scene_id": scene_id,
            "dataset_scene_id": str(raw_scene),
            "habitat_episode_id": episode_id,
            "episode_id": episode_id,
            "trajectory_id": str(raw_trajectory),
            "route_length": _route_length(episode, episode_id),
        })
    if len(by_scene) != SCENE_COUNT:
        raise ValueError(
            f"R2R {SPLIT} must contain 11 scenes; found {len(by_scene)}"
        )
    full_scene_counts = {
        scene: len(rows) for scene, rows in sorted(by_scene.items())
    }
    scene_quotas, scene_allocation = _allocate_scene_quotas(
        full_scene_counts, per_scene,
    )
    selected: list[dict[str, Any]] = []
    for scene_id, rows in sorted(by_scene.items()):
        selected.extend(_select_scene(rows, scene_id, seed, scene_quotas[scene_id]))
    selected = sorted(
        selected,
        key=lambda row: (
            row["scene_id"],
            DISTANCE_GROUPS.index(row["distance_group"]),
            row["route_length"],
            row["habitat_episode_id"],
        ),
    )
    for row in selected:
        row.update({
            "sample_seed": seed,
            "dataset_split": SPLIT,
            "source_dataset": str(source),
            "sampling_method": SAMPLING_METHOD,
        })
    if len(selected) != SCENE_COUNT * PILOT_PER_SCENE:
        raise ValueError(f"pilot selected {len(selected)} Episodes; expected 220")
    keys = {(row["scene_id"], row["habitat_episode_id"]) for row in selected}
    if len(keys) != len(selected):
        raise ValueError("duplicate Episode in pilot selection")
    return {
        "dataset_split": SPLIT,
        "source_dataset": str(source),
        "source_sha256": source_digest,
        "sample_seed": seed,
        "sampling_method": SAMPLING_METHOD,
        "expected_count": len(selected),
        "per_scene": per_scene,
        "full_scene_counts": full_scene_counts,
        "scene_quotas": scene_quotas,
        "scene_allocation": scene_allocation,
        "episodes": selected,
    }


def sample_rgm_88(parent_manifest: str | Path, *, dataset_path: str | Path | None = None,
                  scenes_dir: str | Path | None = None, seed: int = SEED) -> dict[str, Any]:
    """Derive 8 per scene from the old 220 IDs without using model outcomes."""
    if seed != SEED:
        raise ValueError("RGM V1 Pilot requires seed 42")
    parent_path = Path(parent_manifest)
    parent_bytes = parent_path.read_bytes()
    parent = json.loads(parent_bytes)
    rows = parent.get("episodes")
    if (parent.get("dataset_split") != SPLIT or parent.get("sample_seed") != SEED
            or parent.get("expected_count") != 220 or not isinstance(rows, list)
            or len(rows) != 220):
        raise ValueError("parent must be the complete 220 Episode val_unseen Pilot")
    source_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if dataset_path is not None:
        source = resolve_dataset_path(dataset_path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if parent.get("source_sha256") and digest != parent["source_sha256"]:
            raise ValueError("R2R source hash differs from parent Pilot")
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            dataset = json.load(stream)
        for episode in dataset["episodes"]:
            key = (str(episode["scene_id"]), str(episode["episode_id"]))
            if key in source_rows:
                raise ValueError(f"duplicate R2R source Episode {key}")
            source_rows[key] = episode
            mapped_key = (habitat_scene_id(key[0], scenes_dir), key[1])
            if mapped_key in source_rows and source_rows[mapped_key] is not episode:
                raise ValueError(f"ambiguous mapped R2R Episode {mapped_key}")
            source_rows.setdefault(mapped_key, episode)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("parent Pilot has an invalid Episode row")
        scene = row.get("scene_id")
        episode = row.get("habitat_episode_id")
        if not isinstance(scene, str) or not scene or not isinstance(episode, str) or not episode:
            raise ValueError("parent Pilot has an invalid Episode key")
        key = (scene, episode)
        if key in seen:
            raise ValueError(f"duplicate parent Episode {key}")
        seen.add(key)
        enriched = dict(row)
        source_episode = source_rows.get((str(row.get("dataset_scene_id", scene)), episode))
        if source_episode is not None and "route_length" in enriched:
            measured_length = _route_length(source_episode, episode)
            if abs(float(enriched["route_length"]) - measured_length) > 1e-6:
                raise ValueError(f"parent route length differs from R2R source for {key}")
        if source_episode is not None and "trajectory_id" in enriched:
            if str(enriched["trajectory_id"]) != str(source_episode.get("trajectory_id")):
                raise ValueError(f"parent trajectory_id differs from R2R source for {key}")
        if "route_length" not in enriched:
            if source_episode is None:
                raise ValueError(f"route length missing for {key}; provide --data-path")
            enriched["route_length"] = _route_length(source_episode, episode)
        if "trajectory_id" not in enriched:
            if source_episode is None or "trajectory_id" not in source_episode:
                raise ValueError(f"trajectory_id missing for {key}; provide --data-path")
            enriched["trajectory_id"] = str(source_episode["trajectory_id"])
        length = enriched["route_length"]
        if type(length) not in (int, float) or not math.isfinite(length) or length <= 0:
            raise ValueError(f"invalid parent route length for {key}")
        by_scene[scene].append(enriched)
    if len(by_scene) != SCENE_COUNT or any(len(scene_rows) < 8 for scene_rows in by_scene.values()):
        raise ValueError("parent Pilot must have 11 scenes with at least 8 Episodes each")
    selected = []
    for scene, scene_rows in sorted(by_scene.items()):
        selected.extend(_select_scene(scene_rows, scene, seed, 8))
    selected.sort(key=lambda row: (
        row["scene_id"], DISTANCE_GROUPS.index(row["distance_group"]),
        row["route_length"], row["habitat_episode_id"],
    ))
    for row in selected:
        row["sample_seed"] = seed
        row["dataset_split"] = SPLIT
        row["sampling_method"] = "rgm_88_from_parent_220_distance_tertiles_v1"
    if len(selected) != 88 or len({(row["scene_id"], row["habitat_episode_id"])
                                   for row in selected}) != 88:
        raise ValueError("RGM Pilot must select 88 unique Episodes")
    return {
        "protocol_version": "rgm_pilot_88_v1", "dataset_split": SPLIT,
        "sample_seed": seed, "sampling_method": "rgm_88_from_parent_220_distance_tertiles_v1",
        "parent_manifest_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "source_dataset": parent.get("source_dataset"),
        "source_sha256": parent.get("source_sha256"),
        "expected_count": 88, "per_scene": 8,
        "full_scene_counts": parent.get("full_scene_counts"),
        "scene_quotas": {scene: 8 for scene in by_scene},
        "episodes": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=os.environ.get("DATA_PATH", ""))
    parser.add_argument(
        "--scenes-dir", default=os.environ.get("SCENES_DIR", SCENE_PREFIX)
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-scene", type=int, default=PILOT_PER_SCENE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--parent-manifest", type=Path,
                        help="derive the RGM 88 Episode Pilot from a legacy 220 Manifest")
    args = parser.parse_args(argv)
    manifest = (
        sample_rgm_88(
            args.parent_manifest, dataset_path=args.data_path or None,
            scenes_dir=args.scenes_dir, seed=args.seed,
        ) if args.parent_manifest else sample_pilot(
            args.data_path, scenes_dir=args.scenes_dir,
            per_scene=args.per_scene, seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    counts = Counter(row["scene_id"] for row in manifest["episodes"])
    print(f"Pilot selected {manifest['expected_count']} Episodes from {len(counts)} scenes")
    for scene_id, count in sorted(counts.items()):
        print(f"  {scene_id}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
