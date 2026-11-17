"""Compare the six L1/L3 prompt-organization conditions and the legacy control.

Every group must share the same manifest, checkpoint, seed, Habitat config,
memory backends, detector/segmentation weights and prompt budget; only
``--memory_prompt_mode`` may differ. Missing measurements stay ``null``: this
script never substitutes an estimate. A partial run is inspectable with
``--allow-incomplete`` and is not reported as a completed comparison.

Outputs (under ``--root``):
  comparison.csv               group-level metrics with 95% bootstrap intervals
  comparison.json              complete machine-readable report
  paired_episode_deltas.csv    every Episode's paired delta for every pair
  final_report.md              human-readable summary and limitations
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

try:  # Imported when the repository root is on PYTHONPATH (the launcher does this).
    from lightnav.habitat.experiment_integrity import validate_manifest
    from lightnav.memory.prompt_modes import (
        ABLATION_PROMPT_MODES,
        PROMPT_PROTOCOL_VERSION,
        effective_layer_activation,
        mode_plan,
        resolve_prompt_budget,
    )
except ImportError:  # pragma: no cover - direct script execution without PYTHONPATH
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lightnav.habitat.experiment_integrity import validate_manifest
    from lightnav.memory.prompt_modes import (
        ABLATION_PROMPT_MODES,
        PROMPT_PROTOCOL_VERSION,
        effective_layer_activation,
        mode_plan,
        resolve_prompt_budget,
    )

GROUPS = (*ABLATION_PROMPT_MODES, "legacy")
_LEGACY_ACTIVATION = {
    "mode": "legacy", "l1_layer": "original",
    "l3_layer": "original_spatial_when_inject_else_l1",
    "combination": "legacy", "l1_prompt_enabled": True,
    "l3_prompt_enabled": "original_inject_mode",
    "joint_alignment_enabled": False,
}
EXPECTED_EPISODES = 88
METRICS = (
    "SR_pct", "SPL_pct", "NE_m", "nDTW_pct", "TL_m", "OSR_pct", "Latency_ms",
)
PROMPT_METRICS = (
    "mean_prompt_chars", "mean_injected_memory_chars", "mean_prompt_tokens",
    "mean_available_memory_tokens", "mean_memory_block_tokens",
    "mean_base_prompt_tokens", "mean_l1_facts", "mean_l1_items",
    "mean_l3_retrieved_nodes", "entity_match_rate", "conflict_total",
    "duplicates_total", "prompt_truncation_total",
)
EpisodeKey = tuple[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _iter_jsonl(path: Path, *, required: bool = True):
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"required file is missing: {path}")
        return
    with path.open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{number}")
            yield value


def _episode_key(row: dict[str, Any], *, manifest: bool = False) -> EpisodeKey:
    scene = row.get("scene_id")
    episode = row.get("episode_id" if manifest else "habitat_episode_id")
    if episode is None and not manifest:
        episode = row.get("episode_id")
    if not isinstance(scene, str) or not scene or not isinstance(episode, str) or not episode:
        raise ValueError(f"invalid episode identity in {row!r}")
    return scene, episode


def _key_dict(key: EpisodeKey) -> dict[str, str]:
    return {"scene_id": key[0], "episode_id": key[1]}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite measured number, got {value!r}")
    return float(value)


def _optional_number(value: Any, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _quantile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (
        position - lower
    )


def bootstrap_ci(
    values: list[float], *, samples: int = 2000, seed: int = 0,
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)
    )
    return [_quantile(means, 0.025), _quantile(means, 0.975)]


def paired_bootstrap_ci(
    deltas: list[float], *, samples: int = 2000, seed: int = 0,
) -> list[float] | None:
    return bootstrap_ci(deltas, samples=samples, seed=seed)


def mcnemar_exact(success_first: list[bool], success_second: list[bool]) -> dict[str, Any]:
    """Two-sided exact McNemar test for paired success/failure changes."""
    if len(success_first) != len(success_second):
        raise ValueError("paired success lists differ in length")
    first_only = sum(a and not b for a, b in zip(success_first, success_second))
    second_only = sum(b and not a for a, b in zip(success_first, success_second))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index) for index in range(min(first_only, second_only) + 1)
        )
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    return {
        "first_only_success": first_only,
        "second_only_success": second_only,
        "discordant_pairs": discordant,
        "p_value_exact_two_sided": p_value,
    }


def _metric_summary(
    rows: list[dict[str, float | None]], *, samples: int, seed: int,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        output[metric] = {
            "mean": fmean(values) if values else None,
            "ci95": bootstrap_ci(values, samples=samples, seed=seed) if values else None,
            "n": len(values),
        }
    return output


def _episode_metrics(result: dict[str, Any]) -> dict[str, float | None]:
    success = _number(result.get("success"), "success")
    oracle = _number(result.get("oracle_success"), "oracle_success")
    spl = _number(result.get("spl"), "spl")
    ndtw = _number(result.get("ndtw"), "ndtw")
    ne = _number(result.get("final_distance"), "final_distance")
    if any(not 0.0 <= value <= 1.0 for value in (success, oracle, spl, ndtw)):
        raise ValueError("success/oracle_success/spl/ndtw must lie in [0, 1]")
    if ne < 0:
        raise ValueError("final_distance must be nonnegative")
    path_length = _optional_number(result.get("path_length"), "path_length")
    if path_length is not None and path_length < 0:
        raise ValueError("path_length must be nonnegative")
    latency = _optional_number(
        result.get("inference_latency_ms_mean"), "inference_latency_ms_mean",
    )
    if latency is not None and latency < 0:
        raise ValueError("inference_latency_ms_mean must be nonnegative")
    return {
        "SR_pct": success * 100.0,
        "SPL_pct": spl * 100.0,
        "NE_m": ne,
        "nDTW_pct": ndtw * 100.0,
        "TL_m": path_length,
        "OSR_pct": oracle * 100.0,
        "Latency_ms": latency,
    }


def _manifest_identity(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    """Normalized ordered Episode identities, independent of scene path spelling."""
    rows = validate_manifest(manifest)["episodes"]
    return [
        (str(Path(row["scene_id"]).resolve(strict=False)), str(row["episode_id"]))
        for row in rows
    ]


def _same_manifest(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("split") == right.get("split")
        and left.get("expected_count") == right.get("expected_count")
        and _manifest_identity(left) == _manifest_identity(right)
    )


def _manifest_digest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load_manifest(root: Path) -> tuple[dict[str, Any], list[EpisodeKey]]:
    manifest = validate_manifest(_read_json(root / "episode_manifest.json"))
    if manifest["split"] != "val_unseen":
        raise ValueError("the ablation requires R2R split=val_unseen")
    if manifest["expected_count"] != EXPECTED_EPISODES:
        raise ValueError(
            f"the ablation requires exactly {EXPECTED_EPISODES} Episodes; "
            f"manifest has {manifest['expected_count']}"
        )
    keys = [_episode_key(row, manifest=True) for row in manifest["episodes"]]
    if len(set(keys)) != len(keys) or len(keys) != EXPECTED_EPISODES:
        raise ValueError("the manifest does not contain 88 unique Episodes")
    return manifest, keys


def _arguments_without_prompt_mode(arguments: list[str]) -> list[str]:
    """Remove only the experimental variable from a CLI token list."""
    output: list[str] = []
    skip_value = False
    for token in arguments:
        if skip_value:
            skip_value = False
            continue
        if token == "--memory_prompt_mode":
            skip_value = True
            continue
        if token.startswith("--memory_prompt_mode="):
            continue
        if token == "--prompt_measurement_enabled":
            continue
        output.append(token)
    return output


def _expected_activation(mode: str) -> dict[str, Any]:
    return _LEGACY_ACTIVATION if mode == "legacy" else effective_layer_activation(
        mode_plan(mode)
    )


def _validate_prompt_protocol(
    protocol: dict[str, Any], *, mode: str, budget_strategy: str,
    original_char_budget: int,
) -> None:
    if protocol.get("protocol_version") != PROMPT_PROTOCOL_VERSION:
        raise ValueError(f"{mode} prompt protocol version differs")
    if protocol.get("mode") != mode:
        raise ValueError(f"{mode} prompt protocol mode differs")
    budget = protocol.get("budget", {})
    if not isinstance(budget, dict) or budget.get("strategy") != budget_strategy:
        raise ValueError(f"{mode} prompt protocol budget strategy differs")
    if budget.get("original_char_budget_chars") != original_char_budget:
        raise ValueError(f"{mode} prompt protocol original character budget differs")
    if budget_strategy == "original_char_budget" and budget.get("chars") != original_char_budget:
        raise ValueError(f"{mode} legacy prompt protocol character budget differs")
    if protocol.get("layer_activation") != _expected_activation(mode):
        raise ValueError(f"{mode} prompt protocol layer activation differs")


def _load_prompt_rows(
    group_dir: Path, *, mode: str, records: dict[EpisodeKey, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[EpisodeKey, set[int]]]:
    path = group_dir / "prompt_diagnostics.jsonl"
    rows: list[dict[str, Any]] = []
    steps_by_episode: dict[EpisodeKey, set[int]] = {}
    for row in _iter_jsonl(path):
        if row.get("mode") != mode:
            raise ValueError(f"{group_dir.name} prompt diagnostic mode differs from {mode}")
        key = _episode_key(row)
        if key not in records:
            raise ValueError(f"{group_dir.name} prompt diagnostic outside the manifest: {key}")
        step = row.get("step")
        if type(step) is not int or step < 0:
            raise ValueError(f"{group_dir.name} prompt diagnostic has an invalid step")
        observed = steps_by_episode.setdefault(key, set())
        if step in observed:
            raise ValueError(f"{group_dir.name} has a duplicate prompt step {key}@{step}")
        observed.add(step)
        rows.append(row)
    return rows, steps_by_episode


def _check_step_coverage(
    group_dir: Path, records: dict[EpisodeKey, dict[str, Any]],
    steps_by_episode: dict[EpisodeKey, set[int]],
) -> None:
    for key, result in records.items():
        steps = int(_number(result.get("steps"), f"{group_dir.name}.steps"))
        expected_steps = steps - (1 if result.get("forced_stop") is True else 0)
        observed = steps_by_episode.get(key, set())
        if observed != set(range(expected_steps)):
            raise ValueError(
                f"{group_dir.name} lacks complete prompt diagnostics for {key}: "
                f"expected {expected_steps}, observed {len(observed)}"
            )


def _load_group(
    root: Path, mode: str, manifest: dict[str, Any], expected: list[EpisodeKey],
    *, allow_incomplete: bool, samples: int, seed: int,
) -> dict[str, Any] | None:
    group_dir = root / mode
    if not group_dir.is_dir():
        if allow_incomplete:
            return None
        raise FileNotFoundError(f"ablation group directory missing: {group_dir}")
    group_manifest = _read_json(group_dir / "episode_manifest.json")
    if not _same_manifest(group_manifest, manifest):
        raise ValueError(f"{mode} episode manifest differs from the shared manifest")
    config = _read_json(group_dir / "config_snapshot.yaml")
    protocol = _read_json(group_dir / "prompt_protocol.json")
    if not (group_dir / "pilot_complete.json").is_file() and not allow_incomplete:
        raise FileNotFoundError(f"{mode} has no pilot_complete.json")

    prompt_effective = config.get("prompt_effective")
    if not isinstance(prompt_effective, dict) or prompt_effective.get("prompt_mode") != mode:
        raise ValueError(f"{mode} config snapshot does not select its prompt mode")
    protocol_budget = protocol.get("budget", {})
    if not isinstance(protocol_budget, dict):
        protocol_budget = {}
    # The prompt protocol is the artifact that actually records the builder
    # budget; older snapshots may not duplicate these fields. Prefer the
    # snapshot, fall back to the protocol, never invent a value.
    budget_strategy = (
        prompt_effective.get("budget_strategy")
        or protocol_budget.get("strategy")
    )
    original_char_budget = (
        prompt_effective.get("original_char_budget_chars")
        or protocol_budget.get("original_char_budget_chars")
        or protocol_budget.get("chars")
    )
    if budget_strategy not in {"original_char_budget", "token_aware_context"}:
        raise ValueError(f"{mode} config snapshot lacks a valid budget strategy")
    if type(original_char_budget) is not int or original_char_budget < 1:
        raise ValueError(f"{mode} config snapshot lacks a valid original character budget")
    layer_activation = (
        prompt_effective.get("layer_activation")
        or protocol.get("layer_activation")
    )
    _validate_prompt_protocol(
        protocol, mode=mode, budget_strategy=budget_strategy,
        original_char_budget=original_char_budget,
    )

    records: dict[EpisodeKey, dict[str, Any]] = {}
    for row in _iter_jsonl(group_dir / "results.jsonl", required=not allow_incomplete):
        key = _episode_key(row)
        if key in records:
            raise ValueError(f"{mode} has duplicate result rows for {key}")
        if row.get("forced_stop") is True:
            raise ValueError(f"{mode} used a forced STOP for {key}")
        records[key] = row
    unexpected = set(records) - set(expected)
    if unexpected:
        raise ValueError(f"{mode} contains Episodes outside the manifest: {sorted(unexpected)[:5]}")
    missing = [key for key in expected if key not in records]
    if missing and not allow_incomplete:
        raise ValueError(f"{mode} is missing {len(missing)} manifest Episodes")

    summary = _read_json(group_dir / "summary.json") if (
        group_dir / "summary.json"
    ).is_file() else None
    if summary is None and not allow_incomplete:
        raise FileNotFoundError(f"required file is missing: {group_dir / 'summary.json'}")
    if summary is not None and summary.get("num_episodes") != len(records):
        raise ValueError(f"{mode} summary count differs from its result rows")

    prompt_rows, prompt_steps = _load_prompt_rows(group_dir, mode=mode, records=records)
    marked_complete = (group_dir / "pilot_complete.json").is_file()
    if marked_complete and missing:
        raise ValueError(f"{mode} is marked complete but misses {len(missing)} Episodes")
    if not allow_incomplete or marked_complete:
        _check_step_coverage(group_dir, records, prompt_steps)
    if not (group_dir / "per_episode.csv").is_file() and not allow_incomplete:
        raise FileNotFoundError(f"required file is missing: {group_dir / 'per_episode.csv'}")

    complete = (
        not missing and bool(prompt_rows)
        and (group_dir / "pilot_complete.json").is_file()
    )
    return {
        "mode": mode,
        "config": config,
        "protocol": protocol,
        "records": records,
        "summary": summary,
        "prompt_rows": prompt_rows,
        "missing": missing,
        "complete": complete,
        "budget_strategy": budget_strategy,
        "original_char_budget_chars": original_char_budget,
        "layer_activation": layer_activation,
    }


def _token_value(arguments: list[str], flag: str, default: int) -> int:
    for index, token in enumerate(arguments):
        if token == flag and index + 1 < len(arguments):
            return int(arguments[index + 1])
        if token.startswith(f"{flag}="):
            return int(token.split("=", 1)[1])
    return default


def _validate_group_identity(
    mode: str, config: dict[str, Any], provenance: dict[str, Any] | None,
) -> None:
    """Bind every complete group to the root run's checkpoint/benchmark identity."""
    if not provenance:
        return
    common = config.get("common", {})
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
                f"{mode} group identity differs from the run provenance at {key}"
            )
    if config.get("pilot_manifest_sha256") != provenance.get("pilot_manifest_sha256"):
        raise ValueError(f"{mode} pilot manifest differs from the run provenance")


def _validate_common_configuration(
    groups: dict[str, dict[str, Any]],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject any difference outside the single experimental variable."""
    if not groups:
        raise ValueError("no ablation groups to compare")
    reference_mode = next(iter(groups))
    reference = groups[reference_mode]["config"]
    reference_memory_tokens = _arguments_without_prompt_mode(
        reference.get("memory_args", [])
    )
    expected_budget = resolve_prompt_budget(
        max_context_chars=_token_value(
            reference_memory_tokens, "--memory_max_context_chars", 600,
        ),
        local_graph_prompt_max_chars=_token_value(
            reference_memory_tokens, "--local_graph_prompt_max_chars", 320,
        ),
    )
    common = reference.get("common")
    if not isinstance(common, dict):
        raise ValueError(f"{reference_mode} config snapshot lacks a common block")
    if common.get("task") != "vlnce" or common.get("split") != "val_unseen":
        raise ValueError("the ablation requires VLN-CE val_unseen settings")
    if common.get("expected_episodes") != EXPECTED_EPISODES:
        raise ValueError("every group must use the 88-Episode manifest")
    if common.get("max_steps") != 500 or common.get("force_stop_at_max_steps") is not False:
        raise ValueError("every group must use 500 steps without a forced STOP")
    if common.get("seed") != 42:
        raise ValueError("every group must use seed 42")
    reference_protocol = reference.get("prompt_effective", {}).get("protocol_version")
    if reference_protocol != PROMPT_PROTOCOL_VERSION:
        raise ValueError("the config snapshot lacks the prompt-ablation protocol version")
    for mode, group in groups.items():
        config = group["config"]
        _validate_group_identity(mode, config, provenance)
        if config.get("common") != common:
            raise ValueError(f"{mode} common checkpoint/seed/server settings differ")
        if config.get("server_args") != reference.get("server_args"):
            raise ValueError(f"{mode} server arguments differ")
        for field in ("pilot_manifest_path", "pilot_manifest_sha256"):
            if config.get(field) != reference.get(field):
                raise ValueError(f"{mode} {field} differs from the shared run")
        memory_args = _arguments_without_prompt_mode(config.get("memory_args", []))
        if memory_args != reference_memory_tokens:
            raise ValueError(f"{mode} differs outside --memory_prompt_mode")
        effective = config.get("memory_effective", {})
        if effective.get("enabled") is not True:
            raise ValueError(f"{mode} did not enable memory")
        if not config.get("local_graph_effective", {}).get("enabled"):
            raise ValueError(f"{mode} did not enable local graph memory")
        prompt_effective = config.get("prompt_effective")
        if not isinstance(prompt_effective, dict) or prompt_effective.get("prompt_mode") != mode:
            raise ValueError(f"{mode} config snapshot selects a different prompt mode")
        if group["layer_activation"] != _expected_activation(mode):
            raise ValueError(f"{mode} layer activation differs from its mode plan")
        strategy = group["budget_strategy"]
        expected_strategy = (
            "original_char_budget" if mode == "legacy" else "token_aware_context"
        )
        if strategy != expected_strategy:
            raise ValueError(f"{mode} budget strategy differs from its mode contract")
        if group["original_char_budget_chars"] != expected_budget:
            raise ValueError(f"{mode} original character budget differs")
        root_protocol = config.get("protocol_version")
        prompt_version = config.get("prompt_protocol_version")
        if root_protocol not in {PROMPT_PROTOCOL_VERSION, "local_graph_v1"}:
            raise ValueError(f"{mode} config snapshot root protocol version differs")
        if prompt_version is not None and prompt_version != PROMPT_PROTOCOL_VERSION:
            raise ValueError(f"{mode} config snapshot prompt protocol version differs")
        if root_protocol == PROMPT_PROTOCOL_VERSION and (
            prompt_effective.get("protocol_version") != PROMPT_PROTOCOL_VERSION
        ):
            # Recovery shape from an early prompt-ablation snapshot: the root
            # field was overwritten, so the explicit prompt block must agree.
            raise ValueError(f"{mode} old prompt snapshot lacks a matching prompt protocol")
        _validate_prompt_protocol(
            group["protocol"], mode=mode, budget_strategy=strategy,
            original_char_budget=group["original_char_budget_chars"],
        )
        group["budget_chars"] = expected_budget
    for mode in groups:
        expected = "original_char_budget" if mode == "legacy" else "token_aware_context"
        if groups[mode]["budget_strategy"] != expected:
            raise ValueError(f"{mode} uses the wrong budget strategy")
    return {
        "common": common,
        "original_char_budget_chars": expected_budget,
        "budget_strategies": {
            mode: groups[mode]["budget_strategy"] for mode in groups
        },
        "protocol_version": PROMPT_PROTOCOL_VERSION,
    }


def _prompt_statistics(
    mode: str, rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def mean_of(name: str, *, where=None) -> float | None:
        values = [
            float(row[name]) for row in rows
            if row.get(name) is not None and (where is None or where(row))
        ]
        return fmean(values) if values else None

    uses_l3 = mode_plan(mode).uses_l3 and mode != "legacy"
    attempted = sum(int(row.get("alignment_attempted", 0) or 0) for row in rows)
    confirmed = sum(int(row.get("alignment_confirmed", 0) or 0) for row in rows)
    injections = [
        row for row in rows if isinstance(row.get("actual_injection"), bool)
    ]
    budget_sources: dict[str, int] = {}
    for row in rows:
        source = row.get("budget_source") or "unknown"
        budget_sources[source] = budget_sources.get(source, 0) + 1
    return {
        "steps": len(rows),
        "episodes": len({_episode_key(row) for row in rows}),
        "mean_prompt_chars": mean_of("prompt_chars"),
        "mean_injected_memory_chars": mean_of("injected_memory_chars"),
        "mean_prompt_tokens": mean_of("prompt_tokens"),
        "mean_available_memory_tokens": mean_of("available_memory_tokens"),
        "mean_memory_block_tokens": mean_of("memory_block_tokens"),
        "mean_base_prompt_tokens": mean_of("base_prompt_tokens"),
        "mean_l1_facts": mean_of("l1_fact_count"),
        "mean_l1_items": mean_of("l1_item_count"),
        "mean_l3_retrieved_nodes": (
            mean_of("l3_candidate_count") if uses_l3 else None
        ),
        "entity_match_rate": (
            confirmed / attempted if mode_plan(mode).joint and attempted else None
        ),
        "entity_match_attempted": attempted if mode_plan(mode).joint else None,
        "entity_match_confirmed": confirmed if mode_plan(mode).joint else None,
        "conflict_total": (
            sum(int(row.get("conflict_count", 0) or 0) for row in rows)
            if mode_plan(mode).joint else None
        ),
        "duplicates_total": sum(
            int(row.get("duplicates_removed", 0) or 0) for row in rows
        ),
        "prompt_truncation_total": sum(
            int(row.get("processor_length_rejections", 0) or 0)
            + int(row.get("omitted_items", 0) or 0)
            + int(row.get("token_overflow_drops", 0) or 0)
            for row in rows
        ),
        "budget_sources": budget_sources,
        "token_budget_fallback_steps": (
            None if mode == "legacy" else sum(
                count for source, count in budget_sources.items()
                if source != "token_aware_context"
            )
        ),
        "injection_rate": (
            sum(row.get("actual_injection") is True for row in injections) / len(injections)
            if injections else None
        ),
        "budget_chars": rows[0].get("budget_chars") if rows else None,
    }


def _prompt_metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric in PROMPT_METRICS:
        value = summary.get(metric)
        rows.append({
            "metric": metric,
            "mean": value,
            "ci95_low": None,
            "ci95_high": None,
            "n": summary.get("steps") if value is not None else 0,
        })
    return rows


def _pair_report(
    first: str, second: str, expected: list[EpisodeKey],
    metrics: dict[str, dict[EpisodeKey, dict[str, float | None]]],
    *, samples: int, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deltas_by_metric: dict[str, list[float]] = {metric: [] for metric in METRICS}
    csv_rows: list[dict[str, Any]] = []
    success_first: list[bool] = []
    success_second: list[bool] = []
    paired = 0
    for key in expected:
        left = metrics[first].get(key)
        right = metrics[second].get(key)
        if left is None or right is None:
            continue
        paired += 1
        delta = {}
        for metric in METRICS:
            lvalue = left.get(metric)
            rvalue = right.get(metric)
            difference = (
                float(rvalue) - float(lvalue)
                if lvalue is not None and rvalue is not None else None
            )
            delta[metric] = difference
            if difference is not None:
                deltas_by_metric[metric].append(difference)
        success_first.append(bool(left["SR_pct"] >= 100.0))
        success_second.append(bool(right["SR_pct"] >= 100.0))
        csv_rows.append({
            "pair": f"{second}_minus_{first}",
            **_key_dict(key),
            "success_first": int(success_first[-1]),
            "success_second": int(success_second[-1]),
            **{f"delta_{metric}": delta[metric] for metric in METRICS},
        })
    summary = {
        metric: {
            "mean": fmean(values) if values else None,
            "ci95": paired_bootstrap_ci(values, samples=samples, seed=seed) if values else None,
            "n": len(values),
        }
        for metric, values in deltas_by_metric.items()
    }
    return {
        "paired_count": paired,
        "metrics": summary,
        "success_flips": {
            "first_only_success": sum(a and not b for a, b in zip(success_first, success_second)),
            "second_only_success": sum(b and not a for a, b in zip(success_first, success_second)),
        },
        "mcnemar": mcnemar_exact(success_first, success_second),
    }, csv_rows


def compare(
    root: Path, *, allow_incomplete: bool = False, samples: int = 2000, seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    manifest, expected = _load_manifest(root)
    provenance = _read_json(root / "provenance.json") if (
        root / "provenance.json"
    ).is_file() else None
    if provenance is not None:
        if provenance.get("episode_manifest_canonical_sha256") != _manifest_digest(manifest):
            raise ValueError("provenance manifest digest differs from episode_manifest.json")

    groups: dict[str, dict[str, Any]] = {}
    for mode in GROUPS:
        group = _load_group(
            root, mode, manifest, expected, allow_incomplete=allow_incomplete,
            samples=samples, seed=seed,
        )
        if group is not None:
            groups[mode] = group
    if not groups:
        raise ValueError("no ablation group is present")
    checks = _validate_common_configuration(groups, provenance)
    if set(groups) != set(GROUPS) and not allow_incomplete:
        missing = sorted(set(GROUPS) - set(groups))
        raise ValueError(f"ablation groups are missing: {missing}")

    episode_metrics: dict[str, dict[EpisodeKey, dict[str, float | None]]] = {}
    group_report: dict[str, Any] = {}
    for mode, group in groups.items():
        metrics: dict[EpisodeKey, dict[str, float | None]] = {}
        for key, result in group["records"].items():
            metrics[key] = _episode_metrics(result)
        episode_metrics[mode] = metrics
        summary = _metric_summary(
            list(metrics.values()), samples=samples, seed=seed,
        )
        prompt = _prompt_statistics(mode, group["prompt_rows"])
        group_report[mode] = {
            "completed_episodes": len(group["records"]),
            "missing_episode_ids": [_key_dict(key) for key in group["missing"]],
            "complete": bool(group["complete"] and not group["missing"]),
            "metrics": summary,
            "prompt": prompt,
            "budget_chars": group["budget_chars"],
            "budget_strategy": group["budget_strategy"],
            "prompt_protocol": group["protocol"],
            "measurement_gaps": (
                [] if summary["TL_m"]["n"] == len(metrics)
                else ["TL_m unavailable for some Episodes: no path_length"]
            ),
        }

    paired: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for index, first in enumerate(GROUPS):
        if first not in groups:
            continue
        for second in GROUPS[index + 1:]:
            if second not in groups:
                continue
            report, rows = _pair_report(
                first, second, expected, episode_metrics,
                samples=samples, seed=seed,
            )
            paired[f"{second}_minus_{first}"] = report
            paired_rows.extend(rows)

    navigation = _navigation_analysis(
        group_report, episode_metrics, expected, samples=samples, seed=seed,
    )
    complete = all(
        group_report[mode]["complete"] for mode in groups
    ) and set(groups) == set(GROUPS)
    report = {
        "status": "complete" if complete and not allow_incomplete else "incomplete",
        "protocol_version": PROMPT_PROTOCOL_VERSION,
        "evaluation_mode": "l1_l3_prompt_ablation_88",
        "manifest": manifest,
        "manifest_sha256": _manifest_digest(manifest),
        "provenance": provenance,
        "checks": checks,
        "bootstrap": {"samples": samples, "seed": seed, "unit": "episode"},
        "groups": group_report,
        "paired_differences": paired,
        "navigation_analysis": navigation,
        "notes": {
            "TL_m": "mean measured Habitat PathLength; null when the server omits it",
            "entity_match_rate": "confirmed trusted L3->L1 links / attempted alignments (joint only)",
            "conflict_total": "sum of preserved L1/L3 conflicts per step (joint only)",
            "duplicates_total": "sum of deduplicated L1 facts plus joint duplicate facts",
            "prompt_truncation_total": "whole-item omissions plus Processor length retries",
            "original_builder_truncation": (
                "The original L1/L3 builders do not expose how many candidate lines "
                "they dropped for the character budget; their reported truncation "
                "total only covers Processor retries and explicit organizer omissions."
            ),
        },
    }
    return report, paired_rows


def _navigation_analysis(
    group_report: dict[str, Any],
    episode_metrics: dict[str, dict[EpisodeKey, dict[str, float | None]]],
    expected: list[EpisodeKey], *, samples: int, seed: int,
) -> dict[str, Any]:
    """Paired navigation verdict against the unchanged legacy control.

    Only a paired SR improvement with a two-sided McNemar p < 0.05 is labelled
    statistically significant. Prompt length and diagnostic counts are
    deliberately not used as evidence of a navigation improvement.
    """
    if "legacy" not in episode_metrics:
        return {"status": "missing_legacy", "modes": {}}
    legacy = episode_metrics["legacy"]
    rows: dict[str, Any] = {}
    significant_up: list[str] = []
    significant_down: list[str] = []
    for mode in GROUPS:
        if mode == "legacy" or mode not in episode_metrics:
            continue
        current = episode_metrics[mode]
        deltas: dict[str, list[float]] = {
            "SR_pct": [], "SPL_pct": [], "NE_m": [], "Latency_ms": [],
        }
        legacy_success: list[bool] = []
        mode_success: list[bool] = []
        for key in expected:
            base = legacy.get(key)
            other = current.get(key)
            if base is None or other is None:
                continue
            for metric in deltas:
                left, right = base.get(metric), other.get(metric)
                if left is not None and right is not None:
                    deltas[metric].append(float(right) - float(left))
            legacy_success.append(bool(base["SR_pct"] >= 100.0))
            mode_success.append(bool(other["SR_pct"] >= 100.0))
        mcnemar = mcnemar_exact(legacy_success, mode_success)
        sr = deltas["SR_pct"]
        spl = deltas["SPL_pct"]
        ne = deltas["NE_m"]
        latency = deltas["Latency_ms"]
        sr_mean = fmean(sr) if sr else None
        if mcnemar["p_value_exact_two_sided"] < 0.05 and sr_mean is not None:
            if sr_mean > 0:
                significant_up.append(mode)
            elif sr_mean < 0:
                significant_down.append(mode)
        rows[mode] = {
            "paired_count": len(sr),
            "delta_SR_pct": {
                "mean": sr_mean,
                "ci95": paired_bootstrap_ci(sr, samples=samples, seed=seed) if sr else None,
            },
            "delta_SPL_pct": {
                "mean": fmean(spl) if spl else None,
                "ci95": paired_bootstrap_ci(spl, samples=samples, seed=seed) if spl else None,
            },
            "delta_NE_m": {
                "mean": fmean(ne) if ne else None,
                "ci95": paired_bootstrap_ci(ne, samples=samples, seed=seed) if ne else None,
            },
            "delta_Latency_ms": {
                "mean": fmean(latency) if latency else None,
                "ci95": (
                    paired_bootstrap_ci(latency, samples=samples, seed=seed)
                    if latency else None
                ),
            },
            "mcnemar": mcnemar,
            "verdict": (
                "significant_sr_improvement" if mode in significant_up else
                "significant_sr_degradation" if mode in significant_down else
                "not_demonstrated"
            ),
        }
    best_sr_mode = max(
        (mode for mode in group_report if mode != "legacy"),
        key=lambda mode: (
            group_report[mode]["metrics"]["SR_pct"]["mean"]
            if group_report[mode]["metrics"]["SR_pct"]["mean"] is not None else -1.0
        ),
        default=None,
    )
    all_complete = (
        set(group_report) == set(GROUPS)
        and all(group_report[mode]["complete"] for mode in GROUPS)
    )
    return {
        "status": "complete" if all_complete else "incomplete",
        "baseline": "legacy",
        "criterion": (
            "paired SR delta > 0 and two-sided exact McNemar p < 0.05; "
            "no claim from Prompt length or diagnostic counts"
        ),
        "modes": rows,
        "significant_sr_improvement": significant_up,
        "significant_sr_degradation": significant_down,
        "best_mean_sr_mode": best_sr_mode,
        "best_mean_sr_pct": (
            group_report[best_sr_mode]["metrics"]["SR_pct"]["mean"]
            if best_sr_mode is not None else None
        ),
    }


def write_reports(
    root: Path, report: dict[str, Any], paired_rows: list[dict[str, Any]],
) -> None:
    (root / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "category", "metric", "mean", "ci95_low", "ci95_high", "n"])
        for group, detail in report["groups"].items():
            for metric, value in detail["metrics"].items():
                ci = value.get("ci95") or (None, None)
                writer.writerow([group, "benchmark", metric, value.get("mean"), *ci, value.get("n")])
            for value in _prompt_metric_rows(detail["prompt"]):
                writer.writerow([
                    group, "prompt", value["metric"], value["mean"], None, None, value["n"],
                ])
        for pair, detail in report["paired_differences"].items():
            for metric, value in detail["metrics"].items():
                ci = value.get("ci95") or (None, None)
                writer.writerow([pair, "paired_delta", metric, value.get("mean"), *ci, value.get("n")])
    if paired_rows:
        fieldnames = list(paired_rows[0])
        with (root / "paired_episode_deltas.csv").open(
            "w", newline="", encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(paired_rows)
    else:
        (root / "paired_episode_deltas.csv").write_text("", encoding="utf-8")
    (root / "final_report.md").write_text(
        _markdown_report(report), encoding="utf-8",
    )


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# L1 + L3 Prompt 组织与融合消融（88 Episode 配对实验）",
        "",
        f"- 状态：**{report['status']}**",
        f"- 协议：`{report['protocol_version']}`",
        f"- Episode 数：{report['manifest']['expected_count']}",
        f"- Manifest SHA256：`{report['manifest_sha256']}`",
    ]
    if report.get("status") != "complete":
        lines.append("- **警告：报告不完整，缺失组或 Episode；以下统计仅供检查，不是最终结论。**")
    provenance = report.get("provenance") or {}
    if provenance:
        lines.append(f"- Checkpoint SHA256：`{provenance.get('checkpoint_sha256')}`")
        lines.append(f"- Seed：{provenance.get('seed')}")
    lines += [
        "",
        "## 1. 基准指标",
        "",
        "| 组 | SR (%) | SPL (%) | NE (m) | nDTW (%) | TL (m) | OSR (%) | Model latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, detail in report["groups"].items():
        metrics = detail["metrics"]
        lines.append(
            "| `{mode}` | {sr} | {spl} | {ne} | {ndt} | {tl} | {osr} | {lat} |".format(
                mode=mode,
                sr=_fmt(metrics["SR_pct"]["mean"]),
                spl=_fmt(metrics["SPL_pct"]["mean"]),
                ne=_fmt(metrics["NE_m"]["mean"], 3),
                ndt=_fmt(metrics["nDTW_pct"]["mean"]),
                tl=_fmt(metrics["TL_m"]["mean"]),
                osr=_fmt(metrics["OSR_pct"]["mean"]),
                lat=_fmt(metrics["Latency_ms"]["mean"], 1),
            )
        )
    lines += [
        "",
        "括号外的均值来自逐 Episode 实测；Model latency 为模型推理耗时均值，"
        "不含 Prompt 构造/Memory 测量；95% Bootstrap 区间（2000 次、按 Episode "
        "重采样）保存在 `comparison.json`。任一指标缺失时报告 `null`，不做插补。",
        "",
        "## 2. Prompt 组织统计",
        "",
        "| 组 | Prompt chars | 注入 chars | Prompt tokens | 可用 token/步 | 注入 token/步 | token 回退步数 | L1 事实/步 | L3 检索节点/步 | 实体匹配率 | 冲突数 | 去重数 | 截断次数 | 注入率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, detail in report["groups"].items():
        prompt = detail["prompt"]
        lines.append(
            "| `{mode}` | {pc} | {ic} | {pt} | {av} | {mt} | {fb} | {l1} | {l3} | {em} | {c} | {d} | {t} | {inj} |".format(
                mode=mode,
                pc=_fmt(prompt.get("mean_prompt_chars"), 1),
                ic=_fmt(prompt.get("mean_injected_memory_chars"), 1),
                pt=_fmt(prompt.get("mean_prompt_tokens"), 1),
                av=_fmt(prompt.get("mean_available_memory_tokens"), 1),
                mt=_fmt(prompt.get("mean_memory_block_tokens"), 1),
                fb=_fmt(prompt.get("token_budget_fallback_steps"), 0),
                l1=_fmt(prompt.get("mean_l1_facts"), 2),
                l3=_fmt(prompt.get("mean_l3_retrieved_nodes"), 2),
                em=_fmt(
                    None if prompt.get("entity_match_rate") is None
                    else prompt["entity_match_rate"] * 100.0, 1,
                ),
                c=_fmt(prompt.get("conflict_total"), 0),
                d=_fmt(prompt.get("duplicates_total"), 0),
                t=_fmt(prompt.get("prompt_truncation_total"), 0),
                inj=_fmt(
                    None if prompt.get("injection_rate") is None
                    else prompt["injection_rate"] * 100.0, 1,
                ),
            )
        )
    lines += [
        "",
        "实体匹配率只统计联合模式；它等于“由 L3 已记录的 `linked_l1_evidence_id` 确认的配对 / "
        "参与比较的 L3 事实数”。冲突数、L3 检索节点数在未启用对应层时为 `null`。"
        "Token 列为六个实验组的逐步模型派生预算；legacy 使用原 320 字符预算，"
        "其 token 列为 `null`，`token 回退步数` 不计 legacy。",
        "",
        "## 3. 配对差异与 McNemar 检验",
        "",
        "| 配对（后 − 前） | ΔSR | ΔSPL | ΔNE | ΔnDTW | ΔOSR | ΔLatency | b / c | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair, detail in report["paired_differences"].items():
        metrics = detail["metrics"]
        flips = detail["success_flips"]
        lines.append(
            "| `{pair}` | {dsr} | {dspl} | {dne} | {dndtw} | {dosr} | {dlat} | {b}/{c} | {p} |".format(
                pair=pair,
                dsr=_fmt(metrics["SR_pct"]["mean"]),
                dspl=_fmt(metrics["SPL_pct"]["mean"]),
                dne=_fmt(metrics["NE_m"]["mean"], 3),
                dndtw=_fmt(metrics["nDTW_pct"]["mean"]),
                dosr=_fmt(metrics["OSR_pct"]["mean"]),
                dlat=_fmt(metrics["Latency_ms"]["mean"], 1),
                b=flips["first_only_success"],
                c=flips["second_only_success"],
                p=_fmt(detail["mcnemar"]["p_value_exact_two_sided"], 4),
            )
        )
    lines += [
        "",
        "配对差值的 95% CI 与逐 Episode 明细分别在 `comparison.json` 和 "
        "`paired_episode_deltas.csv`。McNemar 采用双侧精确二项检验；b 为“前组成功、后组失败”，"
        "c 为“前组失败、后组成功”。",
        "",
        "## 4. 导航效果判定（legacy 对照）",
        "",
    ]
    navigation = report.get("navigation_analysis") or {}
    if navigation.get("status") == "missing_legacy":
        lines.append("Legacy 对照缺失，无法进行导航效果判定；不得以 concat 代替 legacy。")
    elif navigation.get("status") != "complete":
        lines.append(
            "本报告不完整（并非七组全部完成），下表的导航判定只是中间结果，"
            "不得作为最终结论；补齐七组后重新生成。"
        )
    else:
        lines += [
            "判定规则：只有**配对 SR 提升且双侧精确 McNemar p < 0.05**才标记为统计显著；"
            "Prompt 长度、Token 数、诊断计数不作为导航改善证据。",
            "",
            "| 组 vs legacy | ΔSR | McNemar p | ΔSPL | ΔNE | ΔLatency | 判定 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for mode, detail in navigation["modes"].items():
            lines.append(
                "| `{mode}` | {sr} | {p} | {spl} | {ne} | {lat} | {verdict} |".format(
                    mode=mode,
                    sr=_fmt(detail["delta_SR_pct"]["mean"]),
                    p=_fmt(detail["mcnemar"]["p_value_exact_two_sided"], 4),
                    spl=_fmt(detail["delta_SPL_pct"]["mean"]),
                    ne=_fmt(detail["delta_NE_m"]["mean"], 3),
                    lat=_fmt(detail["delta_Latency_ms"]["mean"], 1),
                    verdict=detail["verdict"],
                )
            )
        up = navigation.get("significant_sr_improvement") or []
        down = navigation.get("significant_sr_degradation") or []
        best = navigation.get("best_mean_sr_mode")
        lines += [
            "",
            (
                "统计显著 SR 改善的组：" + (", ".join(f"`{item}`" for item in up) if up else "无")
            ),
            (
                "统计显著 SR 下降的组：" + (", ".join(f"`{item}`" for item in down) if down else "无")
            ),
            (
                f"平均 SR 最高的组：`{best}`（{_fmt(navigation.get('best_mean_sr_pct'))}%）；"
                "这只是点估计排名，不构成显著性结论。"
                if best else "没有可比较的 SR 均值。"
            ),
            "",
            "结论必须同时看 SR/SPL/NE 与配对检验；若没有 McNemar 显著改善，则本数据不能"
            "证明任一 Prompt 组织方式真正改善导航效果，也不能用文本更短或诊断指标更好来替代。",
            "",
            "注意：这里有 6 个实验组分别与 legacy 比较，p<0.05 未做多重比较校正；"
            "McNemar 结果是探索性的，需结合 SPL/NE 的方向和效应大小解释，不能仅凭单个 p 值"
            "宣称某种 Prompt 组织方式真正更优。",
        ]
    lines += [
        "",
        "## 5. 一致性检查",
        "",
    ]
    checks = report.get("checks", {})
    lines += [
        f"- 共同 checkpoint / seed / server 设置：{'一致' if checks else '未验证'}",
        f"- Legacy 原始字符预算：{checks.get('original_char_budget_chars')}（仅 legacy 使用）",
        "- 六个实验组使用模型派生的 token 预算：上下文上限、基座 Prompt token 与"
        "生成长度逐 Step 实测；若无法测量则回退原 320 字符并记录原因。",
        "- 七组仅 `--memory_prompt_mode`（以及 legacy 的 `--prompt_measurement_enabled`）"
        "不同；其余 memory 参数逐 token 相同。",
        "- `l1_l3_original_concat` 不是 legacy：前者拼接两层原文，legacy 走原有"
        "L3/L1 二选一注入路径。",
        "",
        "## 6. 方法与限制",
        "",
        "1. 七组使用同一 88 Episode manifest、同一 checkpoint、seed=42、500 步、"
        "`NO_FORCE_STOP=1`、同一 Habitat 配置与动作解码。",
        "2. `legacy` 保留原 Prompt 文本、注入位置、原 320 字符模块预算与执行路径；"
        "六个实验组复用同一 `inject_memory_context` / `audit_prompt_injection`，"
        "并按同一步骤使用模型派生 token 预算（上下文上限 - 基座 Prompt token - "
        "生成长度 - 1 token 边界余量）。",
        "3. L1-only 组不会运行 L3 分割与建图；L3-only 组不会调用 L1 的 Local Graph "
        "Prompt 构建器。联合组只用 L3 写入时记录的已确认关联做实体对齐。",
        "4. Legacy 组默认只做测量（`--prompt_measurement_enabled`），不改变其 Prompt "
        "路径；它用于与六个实验组配对，不能由 `l1_l3_original_concat` 替代。",
        "5. 本报告只汇总实测导航结果。代码验收（Prompt 一致性、隔离、预算、实体对齐、去重、"
        "冲突保留、未知关系保护）由测试证明，不能替代真实导航效果。",
        "6. 88 Episode 配对检验的统计功效有限；McNemar 只检验成功/失败翻转，"
        "其余指标的区间是按 Episode 重采样的条件区间，不覆盖场景层面的不确定性。",
        "7. 若服务器未返回 `path_length`，TL 记为 `null`；不会用步数或距离替代。",
        "8. 原始 L1/L3 构建器不暴露字符预算丢弃计数；`l1_original_only` / "
        "`l3_original_only` 的截断总数只覆盖 Processor 重试，不能与组织模式的 "
        "`omitted_items` 直接等价。",
    ]
    return "\n".join(lines) + "\n"


def verify_only(root: Path, *, samples: int = 2000, seed: int = 0) -> int:
    """Validate manifest/artifacts/config of every group that already exists."""
    manifest, expected = _load_manifest(root)
    groups: dict[str, dict[str, Any]] = {}
    for mode in GROUPS:
        if not (root / mode).is_dir():
            continue
        group = _load_group(
            root, mode, manifest, expected, allow_incomplete=True,
            samples=samples, seed=seed,
        )
        if group is not None:
            groups[mode] = group
    if not groups:
        raise ValueError("no completed group is available to verify")
    provenance_path = root / "provenance.json"
    provenance = (
        _read_json(provenance_path) if provenance_path.is_file() else None
    )
    _validate_common_configuration(groups, provenance)
    print(f"verified {len(groups)} group(s): {', '.join(groups)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            return verify_only(
                args.root, samples=args.bootstrap_samples, seed=args.seed,
            )
        report, paired_rows = compare(
            args.root, allow_incomplete=args.allow_incomplete,
            samples=args.bootstrap_samples, seed=args.seed,
        )
        write_reports(args.root, report, paired_rows)
    except (OSError, ValueError, KeyError) as exc:
        print(f"prompt ablation comparison failed: {exc}", file=sys.stderr)
        return 1
    print(f"Prompt ablation {report['status']}: {args.root / 'comparison.json'}")
    return 0 if report["status"] == "complete" or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
