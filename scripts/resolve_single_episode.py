#!/usr/bin/env python3
"""Resolve a manifest ordinal to one exact R2R val_unseen Habitat Episode.

This reads evaluation metadata only. The destination and reference path are never
sent to a policy; use the printed scene/episode flags to restrict the server.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any

SPLIT = "val_unseen"
SCENE_PREFIX = "data/scene_datasets/"


def _scene_key(scene_id: str) -> str:
    if not isinstance(scene_id, str) or not scene_id.strip() or "\x00" in scene_id:
        raise ValueError("scene_id must be a nonempty valid path")
    return str(Path(scene_id).resolve())


def _dataset_scene_id(scene_id: str, scenes_dir: str) -> str:
    """Mirror VLNCEDatasetV1.from_json's scenes_dir path transformation."""
    if scene_id.startswith(SCENE_PREFIX):
        scene_id = scene_id[len(SCENE_PREFIX):]
    return os.path.join(scenes_dir, scene_id)


def _episode_id(row: dict[str, Any]) -> str:
    raw = row.get("habitat_episode_id", row.get("episode_id"))
    if isinstance(raw, bool) or not isinstance(raw, (str, int)) or not str(raw):
        raise ValueError("manifest Episode lacks a valid episode ID")
    episode_id = str(raw)
    if "habitat_episode_id" in row and "episode_id" in row:
        if str(row["episode_id"]) != episode_id:
            raise ValueError("manifest Episode has conflicting episode IDs")
    return episode_id


def _pose_vector(row: dict[str, Any], name: str, size: int) -> list[float]:
    values = row.get(name)
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"dataset Episode requires {name} with {size} numbers")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError(f"dataset Episode has invalid {name}")
    return [float(value) for value in values]


def _reference_path(episode: dict[str, Any]) -> list[list[float]] | None:
    path = episode.get("reference_path")
    if path is None:
        return None
    if not isinstance(path, list):
        raise ValueError("dataset Episode has invalid reference_path")
    return [_pose_vector({"point": point}, "point", 3) for point in path]


def _goal_position(episode: dict[str, Any]) -> list[float] | None:
    goals = episode.get("goals")
    if not isinstance(goals, list) or not goals:
        return None
    if not isinstance(goals[0], dict):
        raise ValueError("dataset Episode has invalid first goal")
    return _pose_vector(goals[0], "position", 3)


def resolve_episode(
    manifest_path: str | Path,
    dataset_path: str | Path,
    scenes_dir: str | Path,
    episode_number: int = 1,
) -> dict[str, Any]:
    """Return audited identity and start metadata for a one-based manifest ordinal."""
    if type(episode_number) is not int or episode_number < 1:
        raise ValueError("episode_number must be a positive, one-based manifest ordinal")
    scene_root = str(scenes_dir)
    if not scene_root or "\x00" in scene_root:
        raise ValueError("scenes_dir must be a valid path matching the Habitat server")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    split = manifest.get("dataset_split", manifest.get("split"))
    if split != SPLIT or ("dataset_split" in manifest and "split" in manifest
                          and manifest["split"] != split):
        raise ValueError("manifest must identify the val_unseen split")
    rows = manifest.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("manifest requires an episodes list")
    if "expected_count" in manifest and manifest["expected_count"] != len(rows):
        raise ValueError("manifest expected_count differs from episodes list")
    if episode_number > len(rows):
        raise ValueError(f"manifest contains only {len(rows)} Episodes")
    selected_row = rows[episode_number - 1]
    if not isinstance(selected_row, dict):
        raise ValueError("selected manifest Episode is not an object")
    manifest_scene = selected_row.get("scene_id")
    wanted = (_scene_key(manifest_scene), _episode_id(selected_row))

    dataset_file = Path(dataset_path)
    if dataset_file.name != f"{SPLIT}.json.gz":
        raise ValueError(f"dataset must be the R2R {SPLIT}.json.gz file")
    with gzip.open(dataset_file, "rt", encoding="utf-8") as stream:
        dataset = json.load(stream)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("episodes"), list):
        raise ValueError("dataset requires an episodes list")

    matches = []
    for index, episode in enumerate(dataset["episodes"]):
        if not isinstance(episode, dict):
            raise ValueError(f"dataset Episode {index} is not an object")
        raw_scene = episode.get("scene_id")
        raw_id = episode.get("episode_id")
        if not isinstance(raw_scene, str) or not raw_scene:
            raise ValueError(f"dataset Episode {index} lacks scene_id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)) or not str(raw_id):
            raise ValueError(f"dataset Episode {index} lacks episode_id")
        scene_id = _dataset_scene_id(raw_scene, scene_root)
        if (_scene_key(scene_id), str(raw_id)) == wanted:
            matches.append((index, episode, scene_id))
    if len(matches) != 1:
        raise ValueError(
            "selected manifest Episode must match exactly one source dataset Episode: "
            f"scene_id={manifest_scene!r}, episode_id={wanted[1]!r}, matches={len(matches)}"
        )

    index, episode, _ = matches[0]
    instruction = episode.get("instruction")
    if not isinstance(instruction, dict) or not isinstance(
        instruction.get("instruction_text"), str
    ) or not instruction["instruction_text"].strip():
        raise ValueError("dataset Episode lacks instruction.instruction_text")
    trajectory = episode.get("trajectory_id")
    if isinstance(trajectory, bool) or not isinstance(trajectory, (str, int)) or not str(trajectory):
        raise ValueError("dataset Episode lacks trajectory_id")
    return {
        "dataset_split": SPLIT,
        "manifest_ordinal": episode_number,
        "dataset_episode_index": index,
        "dataset_episode_number": index + 1,
        "dataset_scene_id": episode["scene_id"],
        "scene_id": manifest_scene,
        "habitat_episode_id": wanted[1],
        "trajectory_id": str(trajectory),
        "instruction_text": instruction["instruction_text"],
        "start_position": _pose_vector(episode, "start_position", 3),
        "start_rotation": _pose_vector(episode, "start_rotation", 4),
        "reference_path": _reference_path(episode),
        "goal_position": _goal_position(episode),
        "single_scene_id": manifest_scene,
        "single_episode_id": wanted[1],
        "server_selection_args": [
            "--single-scene-id", manifest_scene,
            "--single-episode-id", wanted[1],
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Pilot or evaluator episode_manifest JSON")
    parser.add_argument("--dataset", required=True, help="R2R val_unseen.json.gz")
    parser.add_argument("--scenes-dir", required=True, help="Same scenes_dir used by the Habitat server")
    parser.add_argument("--episode-number", type=int, default=1,
                        help="One-based ordinal in the manifest episodes list (default: 1)")
    args = parser.parse_args(argv)
    try:
        result = resolve_episode(args.manifest, args.dataset, args.scenes_dir,
                                 args.episode_number)
    except (OSError, ValueError, gzip.BadGzipFile, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
