"""Per-run prompt protocol and per-step prompt diagnostics artifacts.

The protocol file is written once per output directory and is immutable for a
run. The diagnostics file has one row per inference step and records what the
prompt mode selected, how much of it entered the model input, and how many
items were dropped by the character or Processor budget. These files are only
written for the experimental prompt modes; ``legacy`` keeps its original
artifact set.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from lightnav.memory.prompt_modes import (
    PROMPT_PROTOCOL_VERSION,
    effective_layer_activation,
    mode_plan,
    resolve_prompt_budget,
)

PROMPT_PROTOCOL_FILENAME = "prompt_protocol.json"
PROMPT_DIAGNOSTICS_FILENAME = "prompt_diagnostics.jsonl"
PER_EPISODE_FILENAME = "per_episode.csv"

_PER_EPISODE_FIELDS = (
    "episode_id", "habitat_episode_id", "scene_id", "success", "oracle_success",
    "spl", "ndtw", "final_distance", "min_distance", "steps", "path_length",
    "termination_reason", "forced_stop",
)


def build_prompt_protocol(memory: Any) -> dict[str, Any]:
    """Describe one mode's layer activation, budget and injection contract."""
    plan = mode_plan(memory.prompt_mode)
    original_char_budget = resolve_prompt_budget(
        max_context_chars=memory.max_context_chars,
        local_graph_prompt_max_chars=memory.local_graph.prompt_max_chars,
    )
    if plan.mode == "legacy":
        layer_activation = {
            "mode": "legacy",
            "l1_layer": "original",
            "l3_layer": "original_spatial_when_inject_else_l1",
            "combination": "legacy",
            "l1_prompt_enabled": True,
            "l3_prompt_enabled": "original_inject_mode",
            "joint_alignment_enabled": False,
        }
        budget = {
            "strategy": "original_char_budget",
            "token_aware": False,
            "chars": original_char_budget,
            "original_char_budget_chars": original_char_budget,
            "basis": "min(memory.max_context_chars, local_graph.prompt_max_chars)",
            "note": "legacy prompt path is unchanged; this run only measures it",
            "max_landmarks": memory.local_graph.max_landmarks,
        }
    else:
        layer_activation = effective_layer_activation(plan)
        budget = {
            "strategy": "token_aware_context",
            "token_aware": True,
            # The 320-character value is reported only as the original module
            # allowance and as the explicit fallback; it is not the new cap.
            "original_char_budget_chars": original_char_budget,
            "original_char_budget_basis": (
                "min(memory.max_context_chars, local_graph.prompt_max_chars)"
            ),
            "context_limit_source": (
                "min(bundle.max_seq_len, processor.model_max_length, "
                "backend max_model_len)"
            ),
            "base_prompt_measurement": (
                "data_processor.process_sample(base_sample, add_generation_prompt=True)"
            ),
            "generation_reserve": "engine.max_new_tokens",
            "safety_margin_tokens": 1,
            "fallback": "original_char_budget_with_reported_reason",
            "whole_lines_only": True,
            "max_landmarks": memory.local_graph.max_landmarks,
        }
        if plan.mode == "l1_original_only":
            spatial_mode = getattr(memory.spatial_l3, "mode", "off")
            if spatial_mode == "on_demand":
                layer_activation["l3_layer"] = "on_demand_clip_when_l1_empty"
                layer_activation["l3_prompt_enabled"] = "on_demand_when_l1_empty"
                budget["on_demand_l3"] = {
                    "trigger": "l1 no_candidates or budget_excluded",
                    "budget": "same token-aware budget; original char budget fallback",
                    "retrieval": "l3_clip_query",
                    "raw_similarity_is_not_confidence": True,
                }
            elif spatial_mode == "geometry_on_demand":
                layer_activation["l3_layer"] = "geometry_on_demand_when_l1_empty"
                layer_activation["l3_prompt_enabled"] = (
                    "geometry_on_demand_when_l1_empty"
                )
                budget["on_demand_l3"] = {
                    "trigger": "l1 no_candidates or budget_excluded",
                    "budget": "same token-aware budget; original char budget fallback",
                    "retrieval": "spatial_graph_adapter",
                    "clip_weights_required": False,
                    "clip_similarity_used": False,
                }
                grounder_config = getattr(memory, "spatial_l3", None)
                if getattr(grounder_config, "instance_grounder_enabled", False):
                    selection_mode = getattr(
                        grounder_config, "grounder_selection_mode", "argmax",
                    )
                    budget["on_demand_l3"].update({
                        "retrieval": (
                            "spatial_graph_adapter+vil3dref_student"
                        ),
                        "grounder_selection_mode": selection_mode,
                        "proposal_object_evidence_mapping": (
                            selection_mode == "eligible"
                        ),
                        "eligible_proposal_selection": (
                            selection_mode == "eligible"
                        ),
                        "landmark_queries": bool(getattr(
                            grounder_config,
                            "grounder_landmark_queries", False,
                        )),
                        "geometry_fallback": (
                            "original SpatialGraphAdapter items"
                            if getattr(
                                grounder_config,
                                "grounder_geometry_fallback", False,
                            ) else False
                        ),
                        "ordered_landmark_memory": bool(getattr(
                            grounder_config,
                            "grounder_ordered_landmark_memory", False,
                        )),
                        "include_missing_landmark_placeholders": bool(getattr(
                            grounder_config,
                            "include_missing_landmark_placeholders", True,
                        )),
                        "preserve_original_spatial_fact": bool(getattr(
                            grounder_config,
                            "preserve_original_spatial_fact", False,
                        )),
                    })
    adaptive = getattr(memory, "adaptive_l3_query", None)
    if getattr(adaptive, "enabled", False):
        # Separate, opt-in arm: the original L1 layer is unchanged and L3 facts
        # are added only after the local controller asks for a query and the
        # local validator accepts every Evidence record.
        layer_activation["adaptive_l3"] = "controller_gated_verified_facts"
        budget["adaptive_l3"] = {
            "trigger": "local controller decision, not an L1-empty shortcut",
            "controller": "existing local LightNav inference engine",
            "query": "existing L3ClipQuery cosine Top-K; unchanged",
            "validation": "ObjectNode + complete historical Evidence set",
            "formatter": "deterministic; model text is never copied",
            "max_query_chars": adaptive.max_query_chars,
            "min_evidence_confidence": adaptive.min_evidence_confidence,
        }
    return {
        "protocol_version": PROMPT_PROTOCOL_VERSION,
        "mode": plan.mode,
        "layer_activation": layer_activation,
        "budget": budget,
        "injection": {
            "position": "immediately after </navigation_task> in the human message",
            "adapter": "lightnav.memory.memory_context_builder.inject_memory_context",
            "audit": "lightnav.memory.memory_context_builder.audit_prompt_injection",
            "whole_lines_only": True,
        },
        "builders": {
            "l1_original": "lightnav.memory.local_graph_fusion.LocalGraphFusion",
            "l1_organized": "lightnav.memory.l1_prompt_organizer.L1PromptOrganizer",
            "l3_original": "lightnav.memory.spatial_graph_adapter.SpatialGraphAdapter",
            "l3_organized": "lightnav.memory.l3_prompt_organizer.L3PromptOrganizer",
            "joint": "lightnav.memory.joint_prompt.JointPromptBuilder",
        },
        "invariants": {
            "original_prompt_text_preserved": True,
            "video_and_action_interface_unchanged": True,
            "model_weights_unchanged": True,
            "fixed_character_allowance_added": False,
        },
        **(
            {"adaptive_l3_query": adaptive.to_dict()}
            if getattr(adaptive, "enabled", False) else {}
        ),
    }


def write_or_match_prompt_protocol(output_dir: str | Path, protocol: dict[str, Any]) -> Path:
    """Create the immutable protocol; a resumed run must match it exactly."""
    path = Path(output_dir) / PROMPT_PROTOCOL_FILENAME
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError(f"existing prompt protocol differs from this run: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return path


def prompt_diagnostic_row(
    diagnostic: dict[str, Any], *, episode_id: str, habitat_episode_id: str,
    scene_id: str,
) -> dict[str, Any]:
    """Flatten one policy step's prompt-mode diagnostic into an audit row."""
    context = diagnostic.get("prompt_context")
    row: dict[str, Any] = {
        "episode_id": episode_id,
        "habitat_episode_id": habitat_episode_id,
        "scene_id": scene_id,
        "rollout_idx": diagnostic.get("rollout_idx", 0),
        "step": diagnostic.get("step"),
        "mode": diagnostic.get("prompt_mode"),
        "protocol_version": diagnostic.get("prompt_protocol_version"),
        "prompt_chars": diagnostic.get("prompt_chars"),
        "memory_backend": diagnostic.get("memory_backend"),
        "actual_injection": diagnostic.get("actual_injection"),
        "actual_historical_injection": diagnostic.get("actual_historical_injection"),
        "injected_evidence_ids": diagnostic.get("injected_evidence_ids", []),
        "injected_historical_evidence_ids": diagnostic.get(
            "injected_historical_evidence_ids", []
        ),
        "processor_length_rejections": diagnostic.get("processor_length_rejections", 0),
        "prompt_fallback_reason": diagnostic.get("prompt_fallback_reason"),
        "prompt_hash_before": diagnostic.get("prompt_hash_before"),
        "prompt_hash_after": diagnostic.get("prompt_hash_after"),
    }
    if not isinstance(context, dict):
        row["prompt_diagnostic_missing"] = True
        return row
    layers = context.get("layers", {})
    l1 = layers.get("l1", {}) if isinstance(layers, dict) else {}
    l3 = layers.get("l3", {}) if isinstance(layers, dict) else {}
    l3_on_demand = (
        layers.get("l3_on_demand", {}) if isinstance(layers, dict) else {}
    )
    joint = layers.get("joint", {}) if isinstance(layers, dict) else {}
    alignment = context.get("alignment", {}) or {}
    conflicts = context.get("conflicts", []) or []
    item_counts = context.get("layer_item_counts")
    if isinstance(item_counts, dict) and alignment:
        # Joint items carry both layer contributions; count what the model saw.
        l1_items = int(item_counts.get("l1", 0)) + int(item_counts.get("joint", 0))
        l3_items = int(item_counts.get("l3", 0)) + int(item_counts.get("joint", 0))
    else:
        l1_items = int(l1.get("item_count", 0) or 0)
        l3_items = int(l3.get("item_count", 0) or 0)
    l3_on_demand_items = int(l3_on_demand.get("item_count", 0) or 0)
    l3_items += l3_on_demand_items
    budget = context.get("budget", {})
    if not isinstance(budget, dict):
        budget = {}
    row.update({
        "source": context.get("source"),
        "budget_chars": context.get("budget_chars"),
        "budget_source": budget.get("budget_source"),
        "context_limit_tokens": budget.get("context_limit_tokens"),
        "base_prompt_tokens": budget.get("base_prompt_tokens"),
        "generation_reserve_tokens": budget.get("generation_reserve_tokens"),
        "safety_margin_tokens": budget.get("safety_margin_tokens"),
        "available_memory_tokens": budget.get("available_memory_tokens"),
        "memory_block_tokens": budget.get("memory_block_tokens"),
        "token_overflow_drops": budget.get("token_overflow_drops", 0),
        "token_budget_unavailable_reason": budget.get(
            "token_budget_unavailable_reason"
        ),
        "memory_tokens_injected": context.get("memory_tokens_injected"),
        "context_chars": context.get("context_chars"),
        "item_count": context.get("item_count"),
        "injected_memory_chars": context.get("injected_memory_chars"),
        "prompt_tokens": context.get("final_token_count"),
        "injected_memory_tokens": context.get("injected_memory_tokens"),
        "l1_item_count": l1_items,
        "l1_current_count": l1.get("current_count", 0),
        "l1_history_count": l1.get("history_count", 0),
        "l1_fact_count": l1.get("fact_count", l1.get("item_count", 0)),
        "l1_duplicates_removed": l1.get("duplicates_removed", 0),
        "l1_omitted_items": l1.get("omitted_items", 0),
        "l3_candidate_count": l3.get("candidate_count", 0),
        "l3_on_demand_item_count": l3_on_demand_items,
        "l3_on_demand_trigger": l3_on_demand.get("trigger"),
        "l3_item_count": l3_items,
        "l3_unverified_count": l3.get("unverified_count", 0),
        "l3_omitted_items": l3.get("omitted_items", 0),
        "joint_item_count": context.get("layer_item_counts", {}).get("joint", 0)
        if isinstance(context.get("layer_item_counts"), dict) else 0,
        "joint_omitted_items": joint.get("omitted_items", 0),
        "merged_entities": joint.get("merged_entities", 0),
        "duplicates_removed": context.get(
            "duplicates_removed", l1.get("duplicates_removed", 0),
        ),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "unknown_relation_count": context.get("unknown_relation_count", 0),
        "alignment_attempted": alignment.get("attempted", 0),
        "alignment_confirmed": alignment.get("confirmed", 0),
        "alignment_uncertain": alignment.get("uncertain", 0),
        "alignment_unmatched": alignment.get("unmatched", 0),
        "alignment_success_rate": alignment.get("success_rate"),
        "alignment": alignment,
        "omitted_items": context.get("omitted_items", 0),
        "state_only_injection": context.get("state_only_injection"),
        "prompt_roundtrip_ok": context.get("prompt_roundtrip_ok"),
        "prompt_fallback_reason": context.get(
            "prompt_fallback_reason", row["prompt_fallback_reason"],
        ),
        "processor_length_rejections": context.get(
            "processor_length_rejections", row["processor_length_rejections"],
        ),
    })
    if isinstance(context.get("spatial_l3"), dict):
        row["spatial_l3"] = context["spatial_l3"]
    spatial = context.get("spatial_l3")
    l3_query = spatial.get("l3_query") if isinstance(spatial, dict) else None
    if l3_on_demand.get("source") is not None:
        row["l3_on_demand_source"] = l3_on_demand.get("source")
    if isinstance(l3_query, dict):
        row.update({
            "l3_on_demand_retrieval": l3_query.get("retrieval"),
            "l3_on_demand_query_executed": l3_query.get("query_executed"),
            "l3_on_demand_skipped_reason": l3_query.get("skipped_reason"),
            "l3_on_demand_candidate_count": l3_query.get("candidate_count"),
            "l3_on_demand_injected_count": l3_query.get("injected_count"),
            "l3_on_demand_object_ids": l3_query.get("object_ids", []),
            "l3_on_demand_evidence_ids": l3_query.get("evidence_ids", []),
            "l3_on_demand_injected_object_ids": l3_query.get(
                "injected_object_ids", []
            ),
            "l3_on_demand_injected_evidence_ids": l3_query.get(
                "injected_evidence_ids", []
            ),
            "l3_grounding_query": l3_query.get("grounding_query"),
            "l3_grounding_query_source": l3_query.get("grounding_query_source"),
            "l3_instruction_landmarks": l3_query.get("instruction_landmarks"),
            "l3_landmark_role": l3_query.get("landmark_role"),
            "l3_landmark_order": l3_query.get("landmark_order"),
            "l3_landmark_phrase": l3_query.get("landmark_phrase"),
            "l3_grounder_proposal_count": l3_query.get(
                "grounder_proposal_count"
            ),
            "l3_grounder_scored_count": l3_query.get("grounder_scored_count"),
            "l3_eligible_candidate_count": l3_query.get(
                "eligible_candidate_count"
            ),
            "l3_geometry_fallback_attempted": l3_query.get(
                "geometry_fallback_attempted"
            ),
            "l3_selected_object_id": l3_query.get("selected_object_id"),
            "l3_evidence_ids": l3_query.get("evidence_ids", []),
            "l3_selection_source": l3_query.get("selection_source"),
            "l3_ordered_landmarks": l3_query.get("ordered_landmarks", []),
            "l3_ordered_memory_rendering": l3_query.get(
                "ordered_memory_rendering"
            ),
            "l3_formatable_proposal_count": l3_query.get(
                "formatable_proposal_count"
            ),
            "l3_selected_valid_proposals": l3_query.get(
                "selected_valid_proposals", []
            ),
            "l3_geometry_fallback_reason": l3_query.get(
                "geometry_fallback_reason"
            ),
            "l3_actual_injection": l3_query.get("actual_injection"),
            "l3_actual_injected_evidence_ids": l3_query.get(
                "actual_injected_evidence_ids", []
            ),
        })
    stats = context.get("l3_query_stats")
    if isinstance(stats, dict):
        row.update({
            "l3_queries_attempted": stats.get("queries_attempted"),
            "l3_queries_with_match": stats.get("queries_with_match"),
            "l3_queries_without_match": stats.get("queries_without_match"),
            "l3_queries_injected": stats.get("queries_injected"),
            "l3_prompt_budget_excluded": stats.get("prompt_budget_excluded"),
            "l3_query_trigger_rate": stats.get("query_trigger_rate"),
        })
    if isinstance(context.get("adaptive_l3"), dict):
        adaptive = context["adaptive_l3"]
        row["adaptive_l3"] = adaptive
        row["adaptive_l3_item_count"] = (
            len(adaptive.get("validated_object_ids", []))
            if adaptive.get("injected") else 0
        )
    return row


def legacy_prompt_diagnostic_row(
    diagnostic: dict[str, Any], *, engine_diagnostic: dict[str, Any] | None,
    episode_id: str, habitat_episode_id: str, scene_id: str,
    char_budget: int,
) -> dict[str, Any]:
    """Measurement-only row for the unchanged legacy prompt path.

    No layer/alignment statistics exist on this path; those fields are omitted so
    the comparison reports them as ``null`` instead of inventing values.
    """
    memory_text = str(diagnostic.get("memory_text") or "")
    model = engine_diagnostic if isinstance(engine_diagnostic, dict) else {}
    return {
        "episode_id": episode_id,
        "habitat_episode_id": habitat_episode_id,
        "scene_id": scene_id,
        "rollout_idx": diagnostic.get("rollout_idx", 0),
        "step": diagnostic.get("step"),
        "mode": "legacy",
        "protocol_version": "original_legacy",
        "prompt_chars": diagnostic.get("prompt_chars"),
        "memory_backend": diagnostic.get("memory_backend"),
        "source": diagnostic.get("memory_backend"),
        "actual_injection": diagnostic.get("actual_injection"),
        "actual_historical_injection": diagnostic.get("actual_historical_injection"),
        "injected_evidence_ids": diagnostic.get("injected_evidence_ids", []),
        "injected_historical_evidence_ids": diagnostic.get(
            "injected_historical_evidence_ids", []
        ),
        "processor_length_rejections": diagnostic.get(
            "processor_length_rejections", 0
        ),
        "prompt_fallback_reason": diagnostic.get("prompt_fallback_reason"),
        "budget_source": "original_char_budget",
        "budget_chars": char_budget,
        "injected_memory_chars": len(memory_text),
        "prompt_tokens": model.get("final_token_count"),
        "memory_tokens_injected": None,
        "token_overflow_drops": 0,
        "omitted_items": 0,
        "state_only_injection": False,
        "prompt_roundtrip_ok": True,
    }


def append_prompt_diagnostics(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def write_per_episode_csv(output_dir: str | Path, results: Iterable[dict[str, Any]]) -> Path:
    """Write the per-episode metric table used by the ablation comparison."""
    path = Path(output_dir) / PER_EPISODE_FILENAME
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_PER_EPISODE_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({
                field: result.get(field) for field in _PER_EPISODE_FIELDS
            })
    return path


def merge_prompt_artifacts(root: str | Path, shard_dirs: list[Path]) -> dict[str, Any]:
    """Validate shard protocols and merge prompt diagnostics under ``root``."""
    root = Path(root)
    protocols: list[dict[str, Any]] = []
    for shard in shard_dirs:
        path = shard / PROMPT_PROTOCOL_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"prompt protocol is missing: {path}")
        protocols.append(json.loads(path.read_text(encoding="utf-8")))
    if not protocols:
        raise ValueError("no prompt-protocol shard files were found")
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("shard prompt protocols differ within one evaluation")
    write_or_match_prompt_protocol(root, protocols[0])

    merged = root / PROMPT_DIAGNOSTICS_FILENAME
    rows = 0
    with merged.open("w", encoding="utf-8") as stream:
        for shard in shard_dirs:
            source = shard / PROMPT_DIAGNOSTICS_FILENAME
            if not source.is_file():
                raise FileNotFoundError(f"prompt diagnostics are missing: {source}")
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"invalid prompt diagnostic row in {source}")
                stream.write(line + "\n")
                rows += 1
    results = []
    results_path = root / "results.jsonl"
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                results.append(json.loads(line))
    write_per_episode_csv(root, results)
    return {
        "mode": protocols[0]["mode"],
        "rows": rows,
        "episodes": len(results),
    }


def validate_group_prompt_artifacts(
    group_dir: str | Path, *, expected_mode: str,
) -> dict[str, Any]:
    """Fail loudly when a completed group lacks its required prompt artifacts."""
    group_dir = Path(group_dir)
    protocol_path = group_dir / PROMPT_PROTOCOL_FILENAME
    diagnostics_path = group_dir / PROMPT_DIAGNOSTICS_FILENAME
    per_episode_path = group_dir / PER_EPISODE_FILENAME
    for path in (protocol_path, diagnostics_path, per_episode_path):
        if not path.is_file():
            raise FileNotFoundError(f"required prompt artifact missing: {path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != PROMPT_PROTOCOL_VERSION:
        raise ValueError(f"{group_dir} has an unsupported prompt protocol version")
    if protocol.get("mode") != expected_mode:
        raise ValueError(
            f"{group_dir} prompt mode is {protocol.get('mode')!r}, expected {expected_mode!r}"
        )
    rows = 0
    modes = set()
    for line in diagnostics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        modes.add(row.get("mode"))
        rows += 1
    if rows < 1:
        raise ValueError(f"{group_dir} has no prompt diagnostic rows")
    if modes != {expected_mode}:
        raise ValueError(f"{group_dir} diagnostic modes differ from its protocol")
    return {
        "mode": expected_mode,
        "protocol": protocol,
        "prompt_diagnostic_rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI used by ``scripts/eval_habitat.sh`` to merge shard prompt artifacts."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="lightnav-prompt-artifacts", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge", help="merge per-shard prompt artifacts under one root")
    merge.add_argument("--root", required=True)
    merge.add_argument("shards", nargs="+")
    verify = sub.add_parser("verify", help="require a group's prompt artifacts")
    verify.add_argument("--group", required=True)
    verify.add_argument("--mode", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "merge":
            summary = merge_prompt_artifacts(
                args.root, [Path(shard) for shard in args.shards],
            )
            print(f"merge: mode={summary['mode']} rows={summary['rows']}")
            return 0
        summary = validate_group_prompt_artifacts(args.group, expected_mode=args.mode)
        print(f"verify: mode={summary['mode']} rows={summary['prompt_diagnostic_rows']}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"prompt artifacts failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
