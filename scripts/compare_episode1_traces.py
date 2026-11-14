#!/usr/bin/env python3
"""Compare three single-Episode traces without conflating actions and observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_trace(directory: Path) -> list[dict[str, Any]]:
    path = directory / "step_trace.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or any(row.get("step") != index for index, row in enumerate(rows)):
        raise ValueError(f"trace must have consecutive steps from zero: {path}")
    identity = tuple(rows[0].get(name) for name in (
        "scene_id", "habitat_episode_id", "instruction",
    ))
    if any(not isinstance(value, str) or not value for value in identity):
        raise ValueError(f"trace lacks an Episode identity: {path}")
    if any(tuple(row.get(name) for name in (
        "scene_id", "habitat_episode_id", "instruction",
    )) != identity for row in rows):
        raise ValueError(f"trace changed Episode identity: {path}")
    return rows


def _rgb_hash(directory: Path, row: dict[str, Any], *, use_raw: bool) -> str:
    relative = row.get("rgb")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("trace RGB path must be relative")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("trace RGB path escapes its result directory")
    saved = path.read_bytes()
    raw_hash = row.get("rgb_raw_sha256")
    if use_raw:
        if (not isinstance(raw_hash, str) or len(raw_hash) != 64
                or any(char not in "0123456789abcdef" for char in raw_hash)):
            raise ValueError("trace has an invalid raw RGB checksum")
        return raw_hash
    return hashlib.sha256(saved).hexdigest()


def _same_rgb(a_dir: Path, a: dict[str, Any], b_dir: Path, b: dict[str, Any]) -> bool:
    use_raw = a.get("rgb_raw_sha256") is not None and b.get("rgb_raw_sha256") is not None
    return (
        _rgb_hash(a_dir, a, use_raw=use_raw) == _rgb_hash(b_dir, b, use_raw=use_raw)
        and a.get("rgb_shape") == b.get("rgb_shape")
    )


def _action_signature(row: dict[str, Any]) -> str | None:
    """Compare commands actually sent to Habitat, not merely predicted RVQ IDs."""
    action = row.get("decoded_action")
    if action is None:
        return None
    return json.dumps(action, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _rvq_signature(row: dict[str, Any]) -> tuple[str, ...] | None:
    tokens = row.get("rvq_action_tokens")
    return tuple(str(token) for token in tokens) if isinstance(tokens, list) else None


def _pointing_signature(row: dict[str, Any]) -> tuple[str, ...] | None:
    tokens = row.get("pointing_tokens")
    return tuple(str(token) for token in tokens) if isinstance(tokens, list) else None


def compare_traces(baseline: str | Path, fix3: str | Path,
                   local_graph: str | Path) -> dict[str, Any]:
    """Return factual per-step divergence; route correctness needs route review."""
    paths = {name: Path(path) for name, path in (
        ("baseline_off", baseline), ("rgm_v4_fix3", fix3), ("local_graph", local_graph),
    )}
    traces = {name: _load_trace(path) for name, path in paths.items()}
    reference = traces["baseline_off"][0]
    identity = {name: reference[name] for name in (
        "scene_id", "habitat_episode_id", "instruction",
    )}
    first_raw = all(traces[name][0].get("rgb_raw_sha256") is not None for name in paths)
    first_hash = _rgb_hash(paths["baseline_off"], reference, use_raw=first_raw)
    for name in ("rgm_v4_fix3", "local_graph"):
        first = traces[name][0]
        if any(first.get(key) != value for key, value in identity.items()):
            raise ValueError(f"{name} does not have the same Episode and instruction")
        for key in (
            "model_path", "action_tokenizer_bundle", "decoder_method", "action_horizon",
            "backend", "max_new_tokens", "num_history_frames", "aspect_mode",
            "max_steps", "seed",
        ):
            if (key in reference or key in first) and first.get(key) != reference.get(key):
                raise ValueError(f"{name} differs in {key}")
        if not _same_rgb(paths["baseline_off"], reference, paths[name], first):
            raise ValueError(f"{name} has a different initial RGB observation")
        if first.get("rgb_shape") != reference.get("rgb_shape"):
            raise ValueError(f"{name} has a different initial RGB shape")
        if reference.get("pose_pre") is None or first.get("pose_pre") is None:
            raise ValueError("all traces need the initial pose to verify the same start")
        if first["pose_pre"] != reference["pose_pre"]:
            raise ValueError(f"{name} has a different initial pose")

    comparisons: dict[str, Any] = {}
    for name in ("rgm_v4_fix3", "local_graph"):
        paired = []
        first_action = None
        first_same_rgb_action = None
        first_rgb = None
        for step, (a, b) in enumerate(zip(traces["baseline_off"], traces[name])):
            same_rgb = _same_rgb(paths["baseline_off"], a, paths[name], b)
            action_a, action_b = _action_signature(a), _action_signature(b)
            action_diff = (action_a != action_b if action_a is not None and action_b is not None
                           else None)
            rvq_a, rvq_b = _rvq_signature(a), _rvq_signature(b)
            rvq_diff = (rvq_a != rvq_b if rvq_a is not None and rvq_b is not None else None)
            pointing_a, pointing_b = _pointing_signature(a), _pointing_signature(b)
            pointing_diff = (
                pointing_a != pointing_b
                if pointing_a is not None and pointing_b is not None else None
            )
            if first_rgb is None and not same_rgb:
                first_rgb = step
            if first_action is None and action_diff is True:
                first_action = step
            if first_same_rgb_action is None and same_rgb and action_diff is True:
                first_same_rgb_action = step
            paired.append({
                "step": step,
                "same_rgb": same_rgb,
                "action_divergence_vs_baseline": action_diff,
                "rvq_token_divergence_vs_baseline": rvq_diff,
                "pointing_token_divergence_vs_baseline": pointing_diff,
                "baseline_distance_post": a.get("distance_to_goal_post"),
                "condition_distance_post": b.get("distance_to_goal_post"),
                "invalid_exploration": None,
                "wrong_turn": None,
            })
        comparisons[name] = {
            "baseline_steps": len(traces["baseline_off"]),
            "condition_steps": len(traces[name]),
            "first_action_divergence_step": first_action,
            "first_same_rgb_action_divergence_step": first_same_rgb_action,
            "first_rgb_divergence_step": first_rgb,
            "paired_steps": paired,
        }
    return {
        "identity": identity,
        "initial_rgb_sha256": first_hash,
        "initial_rgb_hash_kind": "raw_pixels" if first_raw else "saved_png_bytes",
        "comparisons": comparisons,
        "first_correct_route_deviation_step": None,
        "route_review_status": "requires_reference_path_pose_and_rgb_review",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--fix3", required=True)
    parser.add_argument("--local-graph", required=True)
    parser.add_argument("--output", help="Write to a new JSON file; default prints to stdout")
    args = parser.parse_args(argv)
    report = compare_traces(args.baseline, args.fix3, args.local_graph)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
