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
QUOTAS = {"short": 7, "medium": 7, "long": 6}
SAMPLING_METHOD = "per_scene_geodesic_tertiles_7_7_6_trajectory_diverse"
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


def _select_scene(rows: list[dict[str, Any]], scene_id: str, seed: int) -> list[dict[str, Any]]:
    """Choose 7/7/6 among scene-local distance tertiles with trajectory diversity."""
    ordered = sorted(rows, key=lambda row: (row["route_length"], row["habitat_episode_id"]))
    if len(ordered) < PILOT_PER_SCENE:
        raise ValueError(
            f"scene {scene_id} has only {len(ordered)} Episodes; need 20"
        )
    # Rank tertiles keep all source Episodes eligible without arbitrary distance
    # thresholds or silently shrinking the sample at tied values.
    first = len(ordered) // 3
    second = 2 * len(ordered) // 3
    buckets = {
        "short": ordered[:first],
        "medium": ordered[first:second],
        "long": ordered[second:],
    }
    counts = Counter(row["trajectory_id"] for row in ordered)
    used_trajectories: set[str] = set()
    selected: list[dict[str, Any]] = []
    for group, quota in QUOTAS.items():
        candidates = list(buckets[group])
        if len(candidates) < quota:
            raise ValueError(
                f"scene {scene_id} {group} stratum has {len(candidates)} "
                f"Episodes; need {quota}"
            )
        rng = random.Random(_seed_for(scene_id, group, seed))
        rng.shuffle(candidates)
        while sum(row["distance_group"] == group for row in selected) < quota:
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
        raise ValueError("this pilot protocol requires exactly 20 Episodes per scene")
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
    selected: list[dict[str, Any]] = []
    for scene_id, rows in sorted(by_scene.items()):
        selected.extend(_select_scene(rows, scene_id, seed))
    selected = sorted(
        selected,
        key=lambda row: (
            row["scene_id"],
            list(QUOTAS).index(row["distance_group"]),
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
    args = parser.parse_args(argv)
    manifest = sample_pilot(
        args.data_path,
        scenes_dir=args.scenes_dir,
        per_scene=args.per_scene,
        seed=args.seed,
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
