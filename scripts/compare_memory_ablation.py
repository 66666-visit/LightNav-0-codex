"""Validate and compare selected, paired R2R memory ablations.

Reads measured episode results and step diagnostics. Missing measurements remain
null; the script never substitutes an estimate. A partial run is inspectable
with ``--allow-incomplete`` but cannot be labelled a formal completed result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

GROUPS = (
    "baseline_off",
    "flat_memory",
    "hierarchical_memory",
    "hierarchical_topology",
)
FORMAL_R2R_EPISODES = 1839
PILOT_SCENES = 11
PILOT_PER_SCENE = 20
PILOT_EPISODES = PILOT_SCENES * PILOT_PER_SCENE
METRICS = (
    "SR_pct", "OS_pct", "SPL_pct", "nDTW_pct", "NE_m",
    "avg_steps", "mean_inference_latency_ms",
)
EpisodeKey = tuple[str, str]


@dataclass
class GroupData:
    name: str
    records: dict[EpisodeKey, dict[str, Any]]
    diagnostics: dict[EpisodeKey, DiagnosticAggregate]
    summary: dict[str, Any] | None
    missing: list[EpisodeKey]
    diagnostic_gaps: dict[EpisodeKey, list[int]]
    orphan_diagnostic_rows: int
    timing: dict[str, Any] | None
    artifacts_complete: bool
    config: dict[str, Any] | None


@dataclass
class DiagnosticAggregate:
    """The measured fields needed from one episode, without retaining step rows."""

    step_bits: int = 0
    rows: int = 0
    latency_sum: float = 0.0
    latency_count: int = 0
    prompt_chars_sum: float = 0.0
    prompt_chars_count: int = 0
    injected_records_sum: float = 0.0
    injected_records_count: int = 0
    injected_memory_chars_sum: float = 0.0
    injected_memory_chars_count: int = 0
    yolo_fields_seen: bool = False
    yolo_calls: int = 0
    yolo_ms_sum: float = 0.0
    yolo_ms_count: int = 0
    dino_fields_seen: bool = False
    dino_calls: int = 0
    dino_ms_sum: float = 0.0
    dino_ms_count: int = 0
    topology_nodes_max: float | None = None
    topology_nodes_complete: bool = True
    topology_edges_max: float | None = None
    topology_edges_complete: bool = True
    place_revisits: int = 0
    place_revisits_complete: bool = True
    place_transitions: int = 0
    place_transitions_complete: bool = True
    missing_measurements: list[str] | None = None

    def add(self, row: dict[str, Any], *, group: str, key: EpisodeKey) -> None:
        step = row["step"]
        bit = 1 << step
        if self.step_bits & bit:
            raise ValueError(f"duplicate {group} diagnostic step {step} for {key}")
        self.step_bits |= bit
        self.rows += 1
        if self.missing_measurements is None:
            self.missing_measurements = []

        def add_number(name: str) -> None:
            value = _optional_number(row, name)
            if value is not None:
                setattr(self, f"{name}_sum", getattr(self, f"{name}_sum") + value)
                setattr(self, f"{name}_count", getattr(self, f"{name}_count") + 1)

        # The field names below match the aggregate members, except latency.
        latency = _optional_number(row, "inference_latency_ms")
        if latency is not None:
            self.latency_sum += latency
            self.latency_count += 1
        for name in ("prompt_chars", "injected_records", "injected_memory_chars"):
            add_number(name)

        self.yolo_fields_seen |= (
            "extraction_attempted" in row or "yolo_detection_ms" in row
        )
        if row.get("extraction_attempted") is True:
            self.yolo_calls += 1
            duration = _optional_number(row, "yolo_detection_ms")
            if duration is not None:
                self.yolo_ms_sum += duration
                self.yolo_ms_count += 1
        self.dino_fields_seen |= (
            "place_extraction_attempted" in row or "place_extraction_ms" in row
        )
        if row.get("place_extraction_attempted") is True:
            self.dino_calls += 1
            duration = _optional_number(row, "place_extraction_ms")
            if duration is not None:
                self.dino_ms_sum += duration
                self.dino_ms_count += 1
        for name in ("topology_nodes", "topology_edges"):
            value = _optional_number(row, name)
            if value is None:
                setattr(self, f"{name}_complete", False)
            else:
                previous = getattr(self, f"{name}_max")
                setattr(self, f"{name}_max", value if previous is None else max(previous, value))
        for name, count_name in (
            ("place_revisited", "place_revisits"),
            ("place_transition", "place_transitions"),
        ):
            if name not in row:
                setattr(self, f"{count_name}_complete", False)
            else:
                setattr(self, count_name, getattr(self, count_name) + bool(row[name]))

        location = f"{key}@{step}"
        if latency is None:
            self.missing_measurements.append(f"{location}: inference_latency_ms")
        if _optional_number(row, "prompt_chars") is None:
            self.missing_measurements.append(f"{location}: prompt_chars")
        if group == "baseline_off":
            return
        for name in ("extraction_attempted", "injected_records", "injected_memory_chars"):
            if name not in row or row[name] is None:
                self.missing_measurements.append(f"{location}: {name}")
        if row.get("extraction_attempted") is True and (
            _optional_number(row, "yolo_detection_ms") is None
        ):
            self.missing_measurements.append(f"{location}: yolo_detection_ms")
        if group == "hierarchical_topology":
            for name in (
                "place_extraction_attempted", "topology_nodes", "topology_edges",
                "place_revisited", "place_transition",
            ):
                if name not in row:
                    self.missing_measurements.append(f"{location}: {name}")
            if row.get("place_extraction_attempted") is True and (
                _optional_number(row, "place_extraction_ms") is None
            ):
                self.missing_measurements.append(f"{location}: place_extraction_ms")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _iter_jsonl(path: Path, *, required: bool):
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required file is missing: {path}")
        return
    with path.open(encoding="utf-8") as source:
        for number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{number}")
            yield value


def _episode_key(row: dict[str, Any], *, manifest: bool = False) -> EpisodeKey:
    if not isinstance(row, dict):
        raise ValueError(f"episode record must be an object, got {row!r}")
    scene = row.get("scene_id")
    episode = row.get("episode_id" if manifest else "habitat_episode_id")
    if episode is None and not manifest:
        episode = row.get("episode_id")
    if scene is None or episode is None or str(scene) == "" or str(episode) == "":
        raise ValueError(f"missing scene_id or Habitat episode ID in {row!r}")
    return str(scene), str(episode)


def _key_dict(key: EpisodeKey) -> dict[str, str]:
    return {"scene_id": key[0], "episode_id": key[1]}


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite measured number, got {value!r}")
    return float(value)


def _optional_number(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    return None if value is None else _number(value, name)


def load_manifest(root: Path) -> tuple[dict[str, Any], list[EpisodeKey]]:
    manifest = _read_json(root / "episode_manifest.json")
    if manifest.get("split") != "val_unseen":
        raise ValueError("formal memory ablation requires R2R split=val_unseen")
    rows = manifest.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("episode_manifest.json must contain an episodes list")
    keys = [_episode_key(row, manifest=True) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate Episode ID in episode_manifest.json")
    count = manifest.get("expected_count")
    if not isinstance(count, int) or count != len(keys) or count < 1:
        raise ValueError("manifest expected_count must equal its nonempty episodes list")
    return manifest, keys


def load_selected_groups(root: Path) -> tuple[str, ...]:
    """Use the launcher's immutable selection; old four-arm runs remain readable."""
    snapshot_path = root / "ablation_run.json"
    if not snapshot_path.is_file():
        return GROUPS
    snapshot = _read_json(snapshot_path)
    if "groups" not in snapshot:
        return GROUPS
    selected = snapshot["groups"]
    if (
        not isinstance(selected, list) or not selected
        or any(not isinstance(name, str) or name not in GROUPS for name in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("ablation_run.json groups must be a nonempty unique list of known groups")
    return tuple(selected)


def _pilot_quota_plan(full_scene_counts: dict[str, int], base: int) -> dict[str, int]:
    """Reconstruct the deterministic shortfall redistribution from source counts."""
    quotas = {scene: min(base, count) for scene, count in full_scene_counts.items()}
    seats = PILOT_EPISODES - sum(quotas.values())
    recipients = sorted(full_scene_counts, key=lambda scene: (-full_scene_counts[scene], scene))
    while seats:
        added = 0
        for scene in recipients:
            if seats == 0:
                break
            if quotas[scene] < full_scene_counts[scene]:
                quotas[scene] += 1
                seats -= 1
                added += 1
        if not added:
            raise ValueError("Pilot dataset cannot supply 220 distinct Episodes")
    return quotas


def _pilot_distance_quotas(size: int) -> dict[str, int]:
    third, remainder = divmod(size, 3)
    return {
        "short": third + (remainder >= 1),
        "medium": third + (remainder >= 2),
        "long": third,
    }


def _pilot_manifest(
    root: Path, expected: list[EpisodeKey], *, per_scene: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Validate that the fixed pilot selection matches the evaluation manifest."""
    pilot = _read_json(root / "pilot_manifest.json")
    if (
        pilot.get("dataset_split") != "val_unseen"
        or pilot.get("sample_seed") != 42
        or pilot.get("per_scene") != per_scene
        or pilot.get("expected_count") != len(expected)
    ):
        raise ValueError("pilot_manifest.json split, seed or counts disagree with pilot run")
    rows = pilot.get("episodes")
    counts = pilot.get("full_scene_counts")
    quotas = pilot.get("scene_quotas")
    allocation = pilot.get("scene_allocation")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("pilot_manifest.json must list every selected Episode")
    if not isinstance(counts, dict) or len(counts) != PILOT_SCENES or (
        any(type(count) is not int or count < 1 for count in counts.values())
        or sum(counts.values()) != FORMAL_R2R_EPISODES
    ):
        raise ValueError("pilot full_scene_counts must cover 11 scenes and sum to 1839")
    expected_quotas = _pilot_quota_plan(counts, per_scene)
    if (
        not isinstance(quotas, dict) or quotas != expected_quotas
        or not isinstance(allocation, dict) or set(allocation) != set(counts)
    ):
        raise ValueError("pilot scene quotas differ from deterministic shortfall allocation")
    for scene, full_count in counts.items():
        detail = allocation[scene]
        actual_quota = quotas[scene]
        reason = (
            "all_available_below_base" if full_count < per_scene else
            "redistributed_shortfall" if actual_quota > per_scene else "base_quota"
        )
        if not isinstance(detail, dict) or detail != {
            "available_episodes": full_count,
            "base_quota": per_scene,
            "selected_quota": actual_quota,
            "adjustment": actual_quota - per_scene,
            "reason": reason,
        }:
            raise ValueError(f"pilot scene {scene} has inconsistent allocation provenance")
    selected = []
    scene_groups: dict[str, dict[str, int]] = {}
    for row in rows:
        key = _episode_key(row)
        selected.append(key)
        if key[0] not in counts:
            raise ValueError(f"pilot scene {key[0]} absent from full_scene_counts")
        group = row.get("distance_group")
        if group not in ("short", "medium", "long"):
            raise ValueError(f"pilot Episode {key} has invalid distance_group")
        length = _number(row.get("route_length"), "pilot route_length")
        if length < 0:
            raise ValueError("pilot route_length must be nonnegative")
        if row.get("sample_seed") != 42 or row.get("dataset_split") != "val_unseen":
            raise ValueError(f"pilot Episode {key} has inconsistent seed or split")
        groups = scene_groups.setdefault(key[0], {"short": 0, "medium": 0, "long": 0})
        groups[group] += 1
    if len(set(selected)) != len(selected) or set(selected) != set(expected):
        raise ValueError("pilot_manifest.json Episode IDs differ from episode_manifest.json")
    if len(scene_groups) != PILOT_SCENES or any(
        scene_groups.get(scene) != _pilot_distance_quotas(quota)
        for scene, quota in expected_quotas.items()
    ):
        raise ValueError("pilot distance groups differ from each scene's balanced quota")
    actual_sha256 = hashlib.sha256((root / "pilot_manifest.json").read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("pilot_manifest.json differs from immutable ablation_run.json digest")
    return pilot


def _load_group(
    root: Path, name: str, manifest: dict[str, Any], expected: list[EpisodeKey],
    *, allow_incomplete: bool, pilot: bool = False,
) -> GroupData:
    directory = root / name
    if not directory.is_dir() and not allow_incomplete:
        raise FileNotFoundError(f"ablation group directory missing: {directory}")
    group_manifest = directory / "episode_manifest.json"
    if group_manifest.is_file():
        if _read_json(group_manifest) != manifest:
            raise ValueError(f"{name} episode manifest differs from the shared manifest")
    elif not allow_incomplete:
        raise FileNotFoundError(f"required file is missing: {group_manifest}")
    config_path = directory / "config_snapshot.yaml"
    config_present = config_path.is_file()
    if not config_present and not allow_incomplete:
        raise FileNotFoundError(f"required file is missing: {config_path}")
    config = _read_json(config_path) if config_present else None

    records: dict[EpisodeKey, dict[str, Any]] = {}
    for row in _iter_jsonl(directory / "results.jsonl", required=not allow_incomplete):
        key = _episode_key(row)
        if key in records:
            raise ValueError(f"duplicate Episode ID in {name}/results.jsonl: {key}")
        if row.get("forced_stop") is True:
            raise ValueError(f"{name} used a forced STOP for {key}")
        records[key] = row
    unexpected = set(records) - set(expected)
    if unexpected:
        raise ValueError(f"{name} contains Episode IDs outside manifest: {sorted(unexpected)}")
    missing = [key for key in expected if key not in records]
    if missing and not allow_incomplete:
        raise ValueError(f"{name} missing {len(missing)} manifest Episode IDs: {missing[:5]}")

    summary_path = directory / "summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else None
    if summary is None and not allow_incomplete:
        raise FileNotFoundError(f"required file is missing: {summary_path}")
    if summary is not None and summary.get("num_episodes") != len(records):
        raise ValueError(
            f"{name} summary count {summary.get('num_episodes')!r} "
            f"does not match {len(records)} result rows"
        )
    if summary is not None:
        _number(summary.get("total_time_sec"), f"{name}.total_time_sec")

    diagnostics: dict[EpisodeKey, DiagnosticAggregate] = {}
    orphan_diagnostic_rows = 0
    expected_set = set(expected)
    diagnostic_rows = _iter_jsonl(
        directory / "memory_diagnostics.jsonl", required=not allow_incomplete,
    )
    for row in diagnostic_rows:
        key = _episode_key(row)
        if key not in expected_set:
            raise ValueError(f"{name} diagnostic Episode ID outside manifest: {key}")
        step = row.get("step")
        if not isinstance(step, int) or not 0 <= step < 500:
            raise ValueError(f"{name} diagnostic step must be an integer in [0, 500)")
        if key not in records:
            # An interrupted episode has no committed result. Its partial
            # diagnostics must not enter completed-episode memory statistics.
            orphan_diagnostic_rows += 1
            continue
        diagnostics.setdefault(key, DiagnosticAggregate()).add(row, group=name, key=key)
    gaps: dict[EpisodeKey, list[int]] = {}
    for key, result in records.items():
        steps = _number(result.get("steps"), f"{name}.steps")
        if not steps.is_integer() or not 0 <= steps <= 500:
            raise ValueError(f"{name}.steps must be an integer in [0, 500]")
        wanted_bits = (1 << int(steps)) - 1
        aggregate = diagnostics.get(key)
        actual_bits = aggregate.step_bits if aggregate else 0
        if actual_bits != wanted_bits:
            gaps[key] = [step for step in range(int(steps)) if not actual_bits & (1 << step)]
            if not allow_incomplete:
                raise ValueError(
                    f"{name} lacks exact step diagnostics for {key}: "
                    f"expected {int(steps)}, got {aggregate.rows if aggregate else 0}"
                )
    timing_path = directory / "run_timing.json"
    timing = _read_json(timing_path) if timing_path.is_file() else None
    if timing is None and not allow_incomplete:
        raise FileNotFoundError(f"required file is missing: {timing_path}")
    attempts = timing.get("attempts") if timing is not None else None
    if timing is not None and (timing.get("group") != name or not isinstance(attempts, list)):
        raise ValueError(f"{name}/run_timing.json has the wrong group or attempts shape")
    if attempts is not None:
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise ValueError(f"{name}/run_timing.json attempts must be objects")
            duration = attempt.get("wall_time_sec")
            if duration is not None and _number(duration, "wall_time_sec") < 0:
                raise ValueError("wall_time_sec must be nonnegative")
    marker_path = directory / ("pilot_complete.json" if pilot else "formal_complete.json")
    opposite_marker = directory / ("formal_complete.json" if pilot else "pilot_complete.json")
    if opposite_marker.is_file():
        raise ValueError(f"{name} has a completion marker for the wrong evaluation mode")
    if marker_path.is_file():
        marker = _read_json(marker_path)
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if marker != {"expected_count": len(expected), "manifest_sha256": digest}:
            raise ValueError(f"{name} completion marker disagrees with manifest")
    artifacts_complete = bool(
        group_manifest.is_file() and config_present and summary_path.is_file()
        and (directory / "results.jsonl").is_file()
        and (directory / "memory_diagnostics.jsonl").is_file()
        and marker_path.is_file()
        and timing_path.is_file() and attempts
        and any(attempt.get("exit_code") == 0 for attempt in attempts)
    )
    return GroupData(name, records, diagnostics, summary, missing, gaps,
                     orphan_diagnostic_rows, timing, artifacts_complete, config)


def _validate_common_configs(
    groups: dict[str, GroupData], *, expected_count: int, pilot: bool = False,
) -> None:
    required = {
        "model_path", "model_config_sha256", "eval_config_sha256",
        "backend", "task", "split", "habitat_config",
        "habitat_config_sha256", "seed", "episodes", "max_steps",
        "data_path", "scenes_dir",
        "expected_episodes",
        "force_stop_at_max_steps", "gpu_ids", "action_tokenizer_bundle",
        "gpu_memory_utilization", "client_non_memory_args", "inference_env",
    }
    reference: tuple[dict[str, Any], Any] | None = None
    for name, group in groups.items():
        if group.config is None:
            continue
        common = group.config.get("common")
        if not isinstance(common, dict) or required - common.keys():
            raise ValueError(f"{name} config_snapshot.yaml lacks required common settings")
        if (
            common["task"] != "vlnce" or common["split"] != "val_unseen"
            or common["episodes"] != -1 or common["max_steps"] != 500
            or common["expected_episodes"] != expected_count
            or common["force_stop_at_max_steps"] is not False
        ):
            mode = "pilot" if pilot else "formal full-split"
            raise ValueError(f"{name} does not use {mode} VLN-CE settings")
        comparable = common, group.config.get("server_args")
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError(f"{name} common checkpoint, seed, server or inference settings differ")


def _episode_metrics(
    result: dict[str, Any], diagnostic: DiagnosticAggregate | None, *,
    require_complete: bool,
) -> dict[str, float | None]:
    success = _number(result.get("success"), "success")
    oracle = _number(result.get("oracle_success"), "oracle_success")
    spl = _number(result.get("spl"), "spl")
    ndtw = _number(result.get("ndtw"), "ndtw")
    ne = _number(result.get("final_distance"), "final_distance")
    steps = _number(result.get("steps"), "steps")
    if any(not 0.0 <= value <= 1.0 for value in (success, oracle, spl, ndtw)):
        raise ValueError("success, oracle_success, spl and ndtw must lie in [0, 1]")
    if ne < 0 or steps < 0:
        raise ValueError("navigation error and step count must be nonnegative")
    if diagnostic and diagnostic.rows and diagnostic.latency_count == diagnostic.rows:
        latency = diagnostic.latency_sum / diagnostic.latency_count
    else:
        latency = _optional_number(result, "inference_latency_ms_mean")
    duration = _optional_number(result, "episode_time_sec")
    if require_complete and (latency is None or duration is None):
        raise ValueError("complete run requires measured inference latency and episode duration")
    return {
        "SR_pct": success * 100.0,
        "OS_pct": oracle * 100.0,
        "SPL_pct": spl * 100.0,
        "nDTW_pct": ndtw * 100.0,
        "NE_m": ne,
        "avg_steps": steps,
        "mean_inference_latency_ms": latency,
        "episode_time_sec": duration,
    }


def _quantile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def bootstrap_ci(
    values: list[float], *, samples: int = 2000, seed: int = 0,
) -> list[float] | None:
    """95% percentile CI of the episode-level mean; seed makes reports repeatable."""
    if not values:
        return None
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap observations must be finite")
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return [_quantile(means, 0.025), _quantile(means, 0.975)]


def _latency_summary(
    diagnostic_groups: list[list[dict[str, Any]]], *, samples: int, seed: int,
) -> dict[str, Any]:
    """Average actual inference calls; bootstrap whole episodes as clusters."""
    clusters: list[tuple[float, int]] = []
    for rows in diagnostic_groups:
        values = [_optional_number(row, "inference_latency_ms") for row in rows]
        if values and all(value is not None for value in values):
            clusters.append((sum(float(value) for value in values), len(values)))
    return _latency_summary_from_pairs(clusters, samples=samples, seed=seed)


def _latency_summary_from_pairs(
    clusters: list[tuple[float, int]], *, samples: int, seed: int,
) -> dict[str, Any]:
    """Bootstrap episode clusters from sufficient sums and call counts."""
    if not clusters:
        return {"mean": None, "ci95": None, "n": 0, "inference_calls": 0}
    total = sum(cluster[0] for cluster in clusters)
    count = sum(cluster[1] for cluster in clusters)
    if len(clusters) == 1:
        confidence = [total / count, total / count]
    else:
        rng = random.Random(seed)
        n = len(clusters)
        resampled = []
        for _ in range(samples):
            picks = [clusters[rng.randrange(n)] for _ in range(n)]
            resampled.append(sum(pick[0] for pick in picks) / sum(pick[1] for pick in picks))
        resampled.sort()
        confidence = [_quantile(resampled, 0.025), _quantile(resampled, 0.975)]
    return {"mean": total / count, "ci95": confidence,
            "n": len(clusters), "inference_calls": count}


def _metric_summary(
    metric_rows: list[dict[str, float | None]], *, samples: int, seed: int,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for metric in (*METRICS, "episode_time_sec"):
        values = [float(row[metric]) for row in metric_rows if row.get(metric) is not None]
        output[metric] = {
            "mean": fmean(values) if values else None,
            "ci95": bootstrap_ci(values, samples=samples, seed=seed) if values else None,
            "n": len(values),
        }
    return output


def _memory_stats(group: GroupData) -> dict[str, Any]:
    aggregates = list(group.diagnostics.values())

    def mean_field(name: str) -> float | None:
        count = sum(getattr(item, f"{name}_count") for item in aggregates)
        return (
            sum(getattr(item, f"{name}_sum") for item in aggregates) / count
            if count else None
        )

    def timing_stats(prefix: str, seen_name: str) -> dict[str, Any] | None:
        if not aggregates or not any(getattr(item, seen_name) for item in aggregates):
            return None
        calls = sum(getattr(item, f"{prefix}_calls") for item in aggregates)
        measured = sum(getattr(item, f"{prefix}_ms_count") for item in aggregates)
        if measured != calls:
            return {"calls": calls, "total_ms": None, "mean_ms": None}
        total = sum(getattr(item, f"{prefix}_ms_sum") for item in aggregates)
        return {"calls": calls, "total_ms": total,
                "mean_ms": total / calls if calls else None}

    def final_graph_total(name: str) -> int | None:
        if not group.records:
            return None
        total = 0
        for key in group.records:
            item = group.diagnostics.get(key)
            if not item or not item.rows or not getattr(item, f"{name}_complete"):
                return None
            total += int(getattr(item, f"{name}_max"))
        return total

    def boolean_count(name: str) -> int | None:
        if not aggregates or any(not getattr(item, f"{name}_complete") for item in aggregates):
            return None
        return sum(getattr(item, name) for item in aggregates)

    return {
        "yolo_detection": timing_stats("yolo", "yolo_fields_seen"),
        "dinov2_extraction": timing_stats("dino", "dino_fields_seen"),
        "topology_nodes_total": final_graph_total("topology_nodes"),
        "topology_edges_total": final_graph_total("topology_edges"),
        "place_revisits": boolean_count("place_revisits"),
        "place_transitions": boolean_count("place_transitions"),
        "mean_injected_records_per_step": mean_field("injected_records"),
        "mean_prompt_chars_per_step": mean_field("prompt_chars"),
        "mean_injected_memory_chars_per_step": mean_field("injected_memory_chars"),
    }


def _measurement_gaps(group: GroupData) -> list[str]:
    """List absent measurements without treating disabled memory as zero work."""
    issues: list[str] = []
    for key, result in group.records.items():
        if _optional_number(result, "episode_time_sec") is None:
            issues.append(f"{key}: episode_time_sec")
        item = group.diagnostics.get(key)
        if item and item.missing_measurements:
            issues.extend(item.missing_measurements)
    return issues


def _runtime_stats(group: GroupData) -> dict[str, Any]:
    attempts = group.timing.get("attempts", []) if group.timing else []
    measured = [
        _number(attempt["wall_time_sec"], "wall_time_sec")
        for attempt in attempts if attempt.get("wall_time_sec") is not None
    ]
    incomplete_attempts = sum(
        attempt.get("wall_time_sec") is None for attempt in attempts
    )
    episode_durations = [
        _optional_number(row, "episode_time_sec") for row in group.records.values()
    ]
    return {
        # Unmeasured interrupted attempts prevent a truthful total wall time.
        "total_wall_time_sec": (
            sum(measured) if attempts and not incomplete_attempts else None
        ),
        "measured_wall_time_sec": sum(measured) if measured else None,
        "unmeasured_attempts": incomplete_attempts,
        "total_episode_time_sec": (
            sum(float(value) for value in episode_durations)
            if episode_durations and all(value is not None for value in episode_durations)
            else None
        ),
    }


def _check_summary_metrics(group: GroupData, metrics: dict[str, dict[str, Any]]) -> None:
    if group.summary is None:
        return
    published = group.summary.get("metrics", {})
    if not isinstance(published, dict):
        raise ValueError(f"{group.name} summary.metrics must be an object")
    aliases = {
        "SR_pct": ("SR_success_rate_pct", 0.02),
        "OS_pct": ("OS_oracle_success_pct", 0.02),
        "SPL_pct": ("SPL_pct", 0.02),
        "nDTW_pct": ("NDTW_pct", 0.02),
        "NE_m": ("NE_navigation_error_m", 0.002),
    }
    for metric, (published_name, tolerance) in aliases.items():
        if published_name not in published or metrics[metric]["mean"] is None:
            continue
        actual = _number(published[published_name], f"{group.name}.{published_name}")
        if abs(actual - metrics[metric]["mean"]) > tolerance:
            raise ValueError(
                f"{group.name} summary {published_name} disagrees with episode results"
            )


def _pilot_scene_statistics(
    metrics: dict[EpisodeKey, dict[str, float | None]],
    full_scene_counts: dict[str, int], scene_quotas: dict[str, int],
    *, samples: int, seed: int,
) -> dict[str, Any]:
    """Report measured scene means and clearly labelled full-split estimates."""
    scenes: dict[str, Any] = {}
    for scene in sorted(full_scene_counts):
        values = [row for key, row in metrics.items() if key[0] == scene]
        scenes[scene] = {
            "sampled_episodes": len(values),
            "target_episodes": scene_quotas[scene],
            "full_val_unseen_episodes": full_scene_counts[scene],
            "metrics": _metric_summary(values, samples=samples, seed=seed),
        }
    macro: dict[str, float | None] = {}
    weighted: dict[str, float | None] = {}
    total_full = sum(full_scene_counts.values())
    for metric in (*METRICS, "episode_time_sec"):
        scene_means = [scenes[scene]["metrics"][metric]["mean"] for scene in scenes]
        if any(value is None for value in scene_means) or any(
            scenes[scene]["sampled_episodes"] != scene_quotas[scene] for scene in scenes
        ):
            macro[metric] = None
            weighted[metric] = None
            continue
        macro[metric] = fmean(scene_means)
        weighted[metric] = sum(
            float(scenes[scene]["metrics"][metric]["mean"])
            * full_scene_counts[scene] / total_full
            for scene in scenes
        )
    return {
        "by_scene": scenes,
        "scene_macro_mean": macro,
        "full_val_scene_weighted_estimate": weighted,
    }


def compare(
    root: Path, *, allow_incomplete: bool = False, samples: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    snapshot_path = root / "ablation_run.json"
    if snapshot_path.is_file() and _read_json(snapshot_path).get("protocol_version") == "rgm_pilot_88_v1":
        return compare_rgm_88(root, samples=samples, seed=seed)
    selected_groups = load_selected_groups(root)
    manifest, expected = load_manifest(root)
    snapshot_path = root / "ablation_run.json"
    snapshot = _read_json(snapshot_path) if snapshot_path.is_file() else {}
    pilot_per_scene = snapshot.get("pilot_per_scene")
    if "pilot_per_scene" in snapshot and (
        type(pilot_per_scene) is not int or pilot_per_scene != PILOT_PER_SCENE
    ):
        raise ValueError("pilot_per_scene must be 20 for the 11-scene pilot")
    pilot = pilot_per_scene is not None
    if not pilot and (root / "pilot_manifest.json").is_file():
        raise ValueError("pilot_manifest.json exists in a formal ablation directory")
    if pilot and selected_groups != ("baseline_off", "hierarchical_topology"):
        raise ValueError("pilot requires baseline_off,hierarchical_topology in that order")
    pilot_metadata = None
    if pilot:
        if len(expected) != PILOT_EPISODES:
            raise ValueError(
                f"pilot requires {PILOT_EPISODES} Episode IDs; manifest has {len(expected)}"
            )
        digest = snapshot.get("pilot_manifest_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("pilot run lacks an immutable pilot_manifest_sha256")
        pilot_metadata = _pilot_manifest(
            root, expected, per_scene=pilot_per_scene, expected_sha256=digest,
        )
    elif "groups" in snapshot and len(expected) != FORMAL_R2R_EPISODES:
        raise ValueError(
            f"selected formal R2R groups require {FORMAL_R2R_EPISODES} Episode IDs; "
            f"manifest has {len(expected)}"
        )
    groups = {
        name: _load_group(root, name, manifest, expected,
                          allow_incomplete=allow_incomplete, pilot=pilot)
        for name in selected_groups
    }
    _validate_common_configs(groups, expected_count=len(expected), pilot=pilot)
    measurement_gaps = {name: _measurement_gaps(group) for name, group in groups.items()}
    complete = all(
        not group.missing and not group.diagnostic_gaps and group.artifacts_complete
        and not measurement_gaps[name]
        for name, group in groups.items()
    )
    if not complete and not allow_incomplete:
        raise ValueError("the selected-group ablation is incomplete")

    episode_metrics: dict[str, dict[EpisodeKey, dict[str, float | None]]] = {}
    group_report: dict[str, Any] = {}
    for name, group in groups.items():
        metrics = {
            key: _episode_metrics(
                result, group.diagnostics.get(key),
                require_complete=not allow_incomplete,
            )
            for key, result in group.records.items()
        }
        episode_metrics[name] = metrics
        summary = _metric_summary(list(metrics.values()), samples=samples, seed=seed)
        latency_clusters = [
            (item.latency_sum, item.latency_count)
            for key in group.records
            if (item := group.diagnostics.get(key)) is not None
            and item.rows and item.latency_count == item.rows
        ]
        summary["mean_inference_latency_ms"] = _latency_summary_from_pairs(
            latency_clusters,
            samples=samples, seed=seed,
        )
        _check_summary_metrics(group, summary)
        group_report[name] = {
            "completed_episodes": len(group.records),
            "success_count": sum(row["SR_pct"] == 100.0 for row in metrics.values()),
            "missing_episode_ids": [_key_dict(key) for key in group.missing],
            "diagnostic_gaps": [
                {**_key_dict(key), "missing_steps": steps}
                for key, steps in group.diagnostic_gaps.items()
            ],
            "orphan_diagnostic_rows": group.orphan_diagnostic_rows,
            "measurement_gaps": measurement_gaps[name],
            "metrics": summary,
            "runtime": _runtime_stats(group),
            "memory": _memory_stats(group),
        }
        if pilot_metadata is not None:
            group_report[name]["pilot_scenes"] = _pilot_scene_statistics(
                metrics, pilot_metadata["full_scene_counts"],
                pilot_metadata["scene_quotas"], samples=samples, seed=seed,
            )

    per_episode = []
    for key in expected:
        per_episode.append({
            **_key_dict(key),
            "groups": {name: episode_metrics[name].get(key) for name in selected_groups},
        })
    paired: dict[str, Any] = {}
    for first_index, first in enumerate(selected_groups):
        for second in selected_groups[first_index + 1:]:
            common = [
                key for key in expected
                if key in episode_metrics[first] and key in episode_metrics[second]
            ]
            differences = []
            for key in common:
                deltas = {
                    metric: (
                        episode_metrics[second][key][metric] - episode_metrics[first][key][metric]
                        if episode_metrics[second][key][metric] is not None
                        and episode_metrics[first][key][metric] is not None else None
                    )
                    for metric in (*METRICS, "episode_time_sec")
                }
                differences.append({**_key_dict(key), "delta": deltas})
            paired[f"{second}_minus_{first}"] = {
                "paired_count": len(common),
                "sr_flips": {
                    "success_to_failure": sum(
                        episode_metrics[first][key]["SR_pct"] == 100.0
                        and episode_metrics[second][key]["SR_pct"] == 0.0
                        for key in common
                    ),
                    "failure_to_success": sum(
                        episode_metrics[first][key]["SR_pct"] == 0.0
                        and episode_metrics[second][key]["SR_pct"] == 100.0
                        for key in common
                    ),
                },
                "metrics": _metric_summary(
                    [entry["delta"] for entry in differences], samples=samples, seed=seed,
                ),
                "per_episode": differences,
            }

    report = {
        "status": "complete" if complete else "incomplete",
        "evaluation_mode": "pilot" if pilot else "formal",
        "selected_groups": list(selected_groups),
        "manifest": manifest,
        "bootstrap": {"samples": samples, "seed": seed, "unit": "episode"},
        "groups": group_report,
        "per_episode": per_episode,
        "paired_differences": paired,
    }
    if pilot_metadata is not None:
        report["pilot"] = {
            "per_scene": pilot_per_scene,
            "sampled_scene_count": PILOT_SCENES,
            "full_scene_counts": pilot_metadata["full_scene_counts"],
            "scene_quotas": pilot_metadata["scene_quotas"],
            "scene_allocation": pilot_metadata["scene_allocation"],
            "estimate_note": (
                "Scene-weighted pilot estimates use measured pilot results and full "
                "val_unseen scene counts; they are not full-split benchmark scores. "
                "Episode bootstrap intervals condition on this 220-Episode selection "
                "and do not fully quantify between-scene uncertainty (11 scenes)."
            ),
        }
    return report


def compare_rgm_88(root: Path, *, samples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Compare exact paired A/C/D rows under the immutable RGM 88 protocol."""
    from dataclasses import asdict

    from lightnav.habitat.experiment_integrity import check_pilot_selection
    from lightnav.memory.rgm_config import RGMConfig

    launch = _read_json(root / "ablation_run.json")
    if launch.get("groups") != ["baseline_off", "rgm_gates", "rgm_full"]:
        raise ValueError("RGM Pilot groups must be A/C/D in order")
    manifest, expected = load_manifest(root)
    if len(expected) != 88:
        raise ValueError("RGM Pilot requires exactly 88 loaded Episodes")
    check_pilot_selection(manifest, root / "pilot_manifest.json")
    pilot_metadata = _read_json(root / "pilot_manifest.json")
    if pilot_metadata.get("parent_manifest_sha256") != launch.get("parent_manifest_sha256"):
        raise ValueError("RGM Pilot parent provenance differs from launcher snapshot")
    digest = hashlib.sha256((root / "pilot_manifest.json").read_bytes()).hexdigest()
    if digest != launch.get("pilot_manifest_sha256"):
        raise ValueError("RGM Pilot Manifest differs from immutable launcher snapshot")
    groups = {
        name: _load_group(root, name, manifest, expected, allow_incomplete=False, pilot=True)
        for name in ("rgm_gates", "rgm_full")
    }
    _validate_common_configs(groups, expected_count=88, pilot=True)
    for name, group in groups.items():
        if not group.artifacts_complete or group.missing or group.diagnostic_gaps:
            raise ValueError(f"{name} is not complete")
        if _measurement_gaps(group):
            raise ValueError(f"{name} has missing measured diagnostics")
        required_rgm = {
            "episode_id", "rgm_config", "yolo_detection_ms", "yolo_candidate_count",
            "yolo_temporal_support", "yolo_association_ms", "write_gate_decisions",
            "dino_global_topk", "dino_local_match_count", "dino_local_inlier_ratio",
            "place_decision", "topology_edge_status", "retrieval_candidates",
            "retrieval_gate_decisions", "fusion_selected_records", "prompt_gate_decision",
            "prompt_fallback_reason", "prompt_hash", "injected_memory_text",
            "injected_memory_chars", "inference_latency_ms",
        }
        for row in _iter_jsonl(root / name / "memory_diagnostics.jsonl", required=True):
            if required_rgm - row.keys():
                raise ValueError(f"{name} lacks RGM decision diagnostics")
    c_args = list(groups["rgm_gates"].config["memory_args"])
    d_args = list(groups["rgm_full"].config["memory_args"])
    required_flags = {
        "--memory_enabled", "--memory_place_enabled", "--rgm_enabled",
    }
    if not required_flags <= set(c_args):
        raise ValueError("C/D require memory, place recognition and RGM")
    for flag, value in (
        ("--memory_strategy", "hierarchical"),
        ("--memory_layers", "short,mid,long"),
        ("--memory_extractor", "yolo11"),
        ("--memory_extract_every_n_steps", "5"),
        ("--memory_place_extract_every_n_steps", "5"),
        ("--rgm_prompt_max_records", "3"),
        ("--rgm_prompt_max_chars", "400"),
    ):
        if flag not in c_args or c_args[c_args.index(flag) + 1] != value:
            raise ValueError(f"C/D protocol requires {flag} {value}")
    extra = {"--rgm_yolo_temporal_enabled", "--rgm_place_two_stage_enabled",
             "--rgm_topology_motion_validation_enabled"}
    if [arg for arg in d_args if arg not in extra] != c_args or not extra <= set(d_args):
        raise ValueError("C and D memory settings differ outside the three D extensions")
    if "--rgm_enabled" not in c_args:
        raise ValueError("C/D did not enable RGM")
    baseline_dir = root / "baseline_off"
    reused = (baseline_dir / "reuse_provenance.json").is_file()
    if reused:
        provenance = _read_json(baseline_dir / "reuse_provenance.json")
        if provenance.get("status") != "compatible" or provenance.get("reusable_count") != 88:
            raise ValueError("legacy A reuse lacks strict compatibility")
        if _read_json(baseline_dir / "episode_manifest.json") != manifest:
            raise ValueError("reused A manifest differs from C/D")
        marker = _read_json(baseline_dir / "pilot_complete.json")
        expected_marker = {
            "expected_count": 88,
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        }
        if marker != expected_marker:
            raise ValueError("reused A completion marker differs from loaded manifest")
        a_records = {}
        for row in _iter_jsonl(baseline_dir / "results.jsonl"):
            key = _episode_key(row)
            if key in a_records:
                raise ValueError(f"duplicate reused A Episode {key}")
            a_records[key] = row
        if set(a_records) != set(expected):
            raise ValueError("reused A does not cover the selected 88")
        a_summary = _read_json(baseline_dir / "summary.json")
        if a_summary.get("num_episodes") != 88:
            raise ValueError("reused A summary count differs from 88")
        a_config = _read_json(baseline_dir / "config_snapshot.yaml")
        if a_config.get("common") != groups["rgm_gates"].config.get("common"):
            raise ValueError("reused A common evaluation config differs from C/D")
        a_diagnostics = {}
        for row in _iter_jsonl(baseline_dir / "memory_diagnostics.jsonl"):
            key = _episode_key(row)
            if key not in a_records:
                raise ValueError("reused A has diagnostic outside selected 88")
            a_diagnostics.setdefault(key, DiagnosticAggregate()).add(
                row, group="baseline_off", key=key,
            )
        for key, record in a_records.items():
            aggregate = a_diagnostics.get(key)
            if (aggregate is None or aggregate.step_bits != (1 << int(record["steps"])) - 1):
                raise ValueError(f"reused A diagnostic coverage incomplete: {key}")
        groups["baseline_off"] = GroupData(
            "baseline_off", a_records, a_diagnostics, a_summary, [], {}, 0,
            None, True, a_config,
        )
    else:
        groups["baseline_off"] = _load_group(
            root, "baseline_off", manifest, expected, allow_incomplete=False, pilot=True,
        )
        if not groups["baseline_off"].artifacts_complete:
            raise ValueError("A rerun is not complete")
        _validate_common_configs(groups, expected_count=88, pilot=True)
    a_args = groups["baseline_off"].config.get("memory_args", [])
    if "--memory_enabled" in a_args or "--rgm_enabled" in a_args:
        raise ValueError("A must use the original memory-off policy")
    for name in ("rgm_gates", "rgm_full"):
        effective = groups[name].config.get("memory_effective")
        expected_rgm = asdict(RGMConfig(
            enabled=True,
            yolo_temporal_enabled=name == "rgm_full",
            place_two_stage_enabled=name == "rgm_full",
            topology_motion_validation_enabled=name == "rgm_full",
        ))
        if not isinstance(effective, dict) or effective.get("rgm") != expected_rgm:
            raise ValueError(f"{name} effective RGM settings differ from the fixed Pilot protocol")
        if effective.get("enabled") is not True:
            raise ValueError(f"{name} effective memory setting is disabled")
    ordered = ("baseline_off", "rgm_gates", "rgm_full")
    episode_metrics = {}
    group_report = {}
    for name in ordered:
        group = groups[name]
        metrics = {
            key: _episode_metrics(
                row, group.diagnostics.get(key),
                require_complete=not (reused and name == "baseline_off"),
            )
            for key, row in group.records.items()
        }
        _check_summary_metrics(
            group, _metric_summary(list(metrics.values()), samples=samples, seed=seed),
        )
        if reused and name == "baseline_off":
            for item in metrics.values():
                item["mean_inference_latency_ms"] = None
                item["episode_time_sec"] = None
        episode_metrics[name] = metrics
        summary = _metric_summary(list(metrics.values()), samples=samples, seed=seed)
        if reused and name == "baseline_off":
            summary["mean_inference_latency_ms"] = {
                "mean": None, "ci95": None, "n": 0,
            }
        group_report[name] = {
            "completed_episodes": 88, "metrics": summary,
            "legacy_reused": reused and name == "baseline_off",
        }
    pairs = {}
    for first, second in (("baseline_off", "rgm_gates"),
                          ("baseline_off", "rgm_full"), ("rgm_gates", "rgm_full")):
        per_episode = []
        for key in expected:
            differences = {}
            for metric in (*METRICS, "episode_time_sec"):
                left = episode_metrics[first][key][metric]
                right = episode_metrics[second][key][metric]
                differences[metric] = right - left if left is not None and right is not None else None
            per_episode.append({**_key_dict(key), "delta": differences})
        pairs[f"{second}_minus_{first}"] = {
            "paired_count": 88, "metrics": _metric_summary(
                [row["delta"] for row in per_episode], samples=samples, seed=seed,
            ),
            "per_episode": per_episode,
        }
    return {
        "status": "complete", "evaluation_mode": "rgm_pilot_88_v1",
        "selected_groups": list(ordered), "manifest": manifest,
        "bootstrap": {"samples": samples, "seed": seed, "unit": "episode"},
        "groups": group_report, "paired_differences": pairs,
        "legacy_a_reused": reused,
        "latency_note": "Legacy A latency excluded across hardware/environment" if reused else None,
    }


def write_report(root: Path, report: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "metric", "mean", "ci95_low", "ci95_high", "n"])
        for group, detail in report["groups"].items():
            for metric, value in detail["metrics"].items():
                ci = value["ci95"] or (None, None)
                writer.writerow([group, metric, value["mean"], *ci, value["n"]])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing the selected ablation groups")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="inspect interrupted runs; output status remains incomplete")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        report = compare(
            args.root, allow_incomplete=args.allow_incomplete,
            samples=args.bootstrap_samples, seed=args.seed,
        )
        write_report(args.root, report)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ablation comparison failed: {exc}", file=sys.stderr)
        return 1
    print(f"Ablation {report['status']}: {args.root / 'comparison.json'}")
    return 0 if report["status"] == "complete" or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
