#!/usr/bin/env python3
"""Read-only compatibility audit and optional materialization of legacy A rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    from scripts import compare_memory_ablation as comparison
except ModuleNotFoundError:  # direct execution from scripts/
    import compare_memory_ablation as comparison


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("scene_id", "")), str(row.get("habitat_episode_id", row.get("episode_id", "")))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def inspect_legacy_a(parent_path: Path, selection_path: Path,
                     legacy_root: Path, target_config_path: Path) -> dict[str, Any]:
    """Reject uncertain reuse; never modifies the AutoDL artifacts."""
    reasons: list[str] = []
    gaps: list[str] = []
    selected = _load_json(selection_path)
    parent = _load_json(parent_path)
    if selected.get("protocol_version") != "rgm_pilot_88_v1":
        raise ValueError("new selection is not the RGM 88 protocol")
    parent_digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    if selected.get("parent_manifest_sha256") != parent_digest:
        reasons.append("parent_manifest_hash_mismatch")
    parent_rows = parent.get("episodes")
    selected_rows = selected.get("episodes")
    if not isinstance(parent_rows, list) or not isinstance(selected_rows, list):
        raise ValueError("both Pilot manifests require Episode lists")
    parent_keys = [_key(row) for row in parent_rows]
    selected_keys = [_key(row) for row in selected_rows]
    if len(parent_keys) != 220 or len(set(parent_keys)) != 220:
        reasons.append("parent_episode_coverage_invalid")
    if len(selected_keys) != 88 or len(set(selected_keys)) != 88:
        reasons.append("selected_episode_coverage_invalid")
    if set(selected_keys) - set(parent_keys):
        reasons.append("selected_episode_not_in_parent")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    legacy_config = None
    legacy_group = legacy_root / "baseline_off"
    try:
        old_manifest, old_keys = comparison.load_manifest(legacy_root)
        if len(old_keys) != 220 or set(old_keys) != set(parent_keys):
            reasons.append("old_a_manifest_differs_from_parent")
        group = comparison._load_group(
            legacy_root, "baseline_off", old_manifest, old_keys,
            allow_incomplete=False, pilot=True,
        )
        if not group.artifacts_complete or group.missing or group.diagnostic_gaps:
            reasons.append("old_a_not_complete")
        records = group.records
        legacy_config = group.config
        for row in comparison._iter_jsonl(
            legacy_group / "memory_diagnostics.jsonl", required=True,
        ):
            if _key(row) in set(selected_keys):
                diagnostics.append(row)
        metrics = {
            key: comparison._episode_metrics(row, group.diagnostics.get(key), require_complete=True)
            for key, row in records.items()
        }
        comparison._check_summary_metrics(
            group, comparison._metric_summary(list(metrics.values()), samples=10, seed=42),
        )
    except (OSError, ValueError, KeyError) as exc:
        reasons.append(f"old_a_artifact_validation_failed:{type(exc).__name__}:{exc}")
    missing = sorted(set(selected_keys) - set(records))
    if missing:
        reasons.append("old_a_missing_selected_results")
    try:
        target_config = _load_json(target_config_path)
    except (OSError, ValueError) as exc:
        reasons.append(f"target_config_unavailable:{exc}")
        target_config = None
    if legacy_config is not None and target_config is not None:
        old_common = legacy_config.get("common", {})
        new_common = target_config.get("common", {})
        required_equal = (
            "model_path", "model_config_sha256", "eval_config_sha256",
            "backend", "task", "split", "habitat_config_sha256", "seed",
            "max_steps", "force_stop_at_max_steps", "data_path", "scenes_dir",
            "action_tokenizer_bundle", "client_non_memory_args", "inference_env",
        )
        for name in required_equal:
            if old_common.get(name) is None or new_common.get(name) is None:
                reasons.append(f"missing_common_field:{name}")
            elif old_common[name] != new_common[name]:
                reasons.append(f"common_field_mismatch:{name}")
        if old_common.get("expected_episodes") != 220 or new_common.get("expected_episodes") != 88:
            reasons.append("wrong_expected_episode_protocol")
        if old_common.get("episodes") != -1 or new_common.get("episodes") != -1:
            reasons.append("changed_episode_limit")
        if old_common.get("max_steps") != 500 or old_common.get("force_stop_at_max_steps") is not False:
            reasons.append("changed_termination_protocol")
        for name in ("source_python_sha256", "model_weights_sha256", "action_bundle_sha256"):
            if not old_common.get(name) or not new_common.get(name):
                gaps.append(name)
            elif old_common[name] != new_common[name]:
                reasons.append(f"fingerprint_mismatch:{name}")
        if any(token in {"--memory_enabled", "--rgm_enabled"}
               for token in legacy_config.get("memory_args", [])):
            reasons.append("old_a_memory_not_off")
        if legacy_config.get("server_args") != target_config.get("server_args"):
            reasons.append("server_args_mismatch")
    if gaps:
        reasons.append("unresolved_provenance_gap")
    return {
        "status": "compatible" if not reasons else "rerun_a_required",
        "reasons": sorted(set(reasons)), "provenance_gaps": gaps,
        "selected_count": len(selected_keys), "reusable_count": 88 if not reasons else 0,
        "missing_episode_keys": [{"scene_id": scene, "habitat_episode_id": episode}
                                 for scene, episode in missing],
        "parent_manifest_sha256": parent_digest,
        "legacy_root": str(legacy_root),
        "navigation_metrics_reusable": not reasons,
        "runtime_latency_comparable": False,
        "_records": records, "_diagnostics": diagnostics,
        "_target_config": target_config, "_selected_keys": selected_keys,
    }


def materialize(plan: dict[str, Any], output_root: Path) -> None:
    """Create a new 88-row A view only after strict compatibility succeeds."""
    if plan["status"] != "compatible":
        raise ValueError("legacy A cannot be materialized without strict compatibility")
    group_dir = output_root / "baseline_off"
    if group_dir.exists():
        raise FileExistsError("baseline_off output already exists; preserve existing results")
    selection = _load_json(output_root / "pilot_manifest.json")
    manifest = {
        "split": "val_unseen", "expected_count": 88,
        "episodes": [{"scene_id": row["scene_id"],
                      "episode_id": row["habitat_episode_id"]}
                     for row in selection["episodes"]],
    }
    # Shared manifest comes from C's actual server selection. Its order must
    # agree as a set; preserve the server's exact serialized form and digest.
    shared = _load_json(output_root / "episode_manifest.json")
    if set(_key(row) for row in manifest["episodes"]) != set(
        _key(row) for row in shared["episodes"]
    ) or shared.get("expected_count") != 88:
        raise ValueError("C's loaded Episode manifest differs from selected 88")
    records = [plan["_records"][key] for key in plan["_selected_keys"]]
    diagnostics = [row for row in plan["_diagnostics"]
                   if _key(row) in set(plan["_selected_keys"])]
    metrics = {
        "SR_success_rate_pct": 100 * fmean(float(row["success"]) for row in records),
        "OS_oracle_success_pct": 100 * fmean(float(row["oracle_success"]) for row in records),
        "SPL_pct": 100 * fmean(float(row["spl"]) for row in records),
        "NDTW_pct": 100 * fmean(float(row["ndtw"]) for row in records),
        "NE_navigation_error_m": fmean(float(row["final_distance"]) for row in records),
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("legacy A metrics are not finite")
    config = dict(plan["_target_config"])
    config["memory_args"] = []
    config["client_args"] = list(config["common"]["client_non_memory_args"])
    from dataclasses import asdict

    from lightnav.memory.rgm_config import RGMConfig

    config["memory_effective"] = {"enabled": False, "rgm": asdict(RGMConfig())}
    config["legacy_a_source"] = plan["legacy_root"]
    marker = {
        "expected_count": 88,
        "manifest_sha256": hashlib.sha256(json.dumps(shared, sort_keys=True).encode()).hexdigest(),
    }
    files = {
        "episode_manifest.json": json.dumps(shared, indent=2) + "\n",
        "results.jsonl": "".join(json.dumps(row) + "\n" for row in records),
        "memory_diagnostics.jsonl": "".join(json.dumps(row) + "\n" for row in diagnostics),
        "summary.json": json.dumps({
            "num_episodes": 88, "metrics": metrics, "episodes": records,
            "total_time_sec": None,
            "runtime_latency_comparable": False,
        }, indent=2) + "\n",
        "config_snapshot.yaml": json.dumps(config, indent=2) + "\n",
        "pilot_complete.json": json.dumps(marker, indent=2) + "\n",
        "reuse_provenance.json": json.dumps({
            key: value for key, value in plan.items() if not key.startswith("_")
        }, indent=2) + "\n",
    }
    staging = Path(tempfile.mkdtemp(prefix=".baseline_off_staging_", dir=output_root))
    for name, content in files.items():
        (staging / name).write_text(content, encoding="utf-8")
    staging.rename(group_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--target-config", required=True, type=Path)
    parser.add_argument("--plan-output", required=True, type=Path)
    parser.add_argument("--materialize-root", type=Path)
    args = parser.parse_args(argv)
    plan = inspect_legacy_a(
        args.parent_manifest, args.selection, args.legacy_root, args.target_config,
    )
    report = {key: value for key, value in plan.items() if not key.startswith("_")}
    if args.plan_output.exists():
        if _load_json(args.plan_output) != report:
            raise ValueError("existing legacy reuse plan differs")
    else:
        args.plan_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.materialize_root and plan["status"] == "compatible":
        materialize(plan, args.materialize_root)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
