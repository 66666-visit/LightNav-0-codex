"""Task-conditioned L1/L3 selection; the external model never chooses robot actions."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from copy import deepcopy
from html import escape
from time import perf_counter
from typing import Any

from lightnav.memory.local_graph_fusion import _plain_label
from lightnav.memory.scene_query_config import SceneQueryConfig
from lightnav.memory.scene_query_provider import OpenAICompatibleQueryProvider, QueryProvider
from lightnav.memory.scene_query_tools import OPENAI_TOOLS, SceneQueryTools
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence
from lightnav.memory.spatial_graph_adapter import _relative_position
from lightnav.memory.v3_schemas import canonical_evidence_key

_HEADER = "[Navigation Memory]"
_DECISIONS = frozenset({"NO_MEMORY", "USE_L1", "USE_L3", "USE_BOTH"})
_SYSTEM = """You select existing navigation memory facts. You do not output a robot action.
Think before responding, then either call the read-only L3 tools or return Final.
Do not spend a response on a standalone THINK unless essential. First consider whether
L1 already suffices. Query only a named task-relevant landmark or object class; do not
enumerate generic categories or repeat a query whose result is already in this conversation.
Prefer find_objects with the exact class_name and history_eligible_only=true. Raw
Evidence IDs are not necessarily injectable: use only historical_eligible_evidence_ids
and historical_eligible_object_ids, and read each selected Evidence explicitly.
If no eligible L3 fact is available, select L1 or NO_MEMORY. A map object and an L1
detection with the same label are different identities unless a linked L1 Evidence ID
proves association.
Previous selections are context, never a cooldown: reuse a fact if still needed.
native_visual_source_steps identifies RGB observation steps already sampled in the
LightNav video input; add external memory from those steps only when it clarifies
the current task. recent_memory_selections summarizes prior IDs and injections.
Finish with exactly this JSON object (no prose or invented facts):
{"decision":"NO_MEMORY|USE_L1|USE_L3|USE_BOTH",
 "selected_l1_fact_ids":[],"selected_l3_object_ids":[],
 "selected_l3_evidence_ids":[],"reason_code":"SHORT_CODE"}.
To select L3, first discover the object with find_objects/read_node. Never select an
ID absent from tool results. Return Final promptly after sufficient results arrive.
Use NO_MEMORY when no verified fact adds value. Only select facts; local code renders them."""


class _NoFinalError(ValueError):
    pass


class _ToolRoundBudgetError(ValueError):
    pass


class _ToolCallBudgetError(ValueError):
    pass


class _EvidenceValidationError(ValueError):
    pass


class _GraphFusionError(ValueError):
    pass


def _safe_error(exc: Exception) -> str:
    """Never copy an API credential from a provider exception into diagnostics."""
    value = str(exc)
    key = os.getenv("MEMORY_QUERY_API_KEY", "")
    if key:
        value = value.replace(key, "[REDACTED]")
    return f"{type(exc).__name__}: {value[:400]}"


def _assistant_message(message: dict[str, Any], *, tool_calls: list[dict] | None = None) -> dict[str, Any]:
    """Preserve DeepSeek reasoning_content for the next tool-bearing API request."""
    result: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if "reasoning_content" in message:
        result["reasoning_content"] = message["reasoning_content"]
    if tool_calls is not None:
        result["tool_calls"] = tool_calls
    return result


def _graph_state_key(tools: SceneQueryTools) -> str:
    """Hash only query-visible map state and the historical eligibility boundary."""
    graph = tools.object_map
    nodes = tuple(sorted((key, node.object_id, node.class_name, tuple(node.center_world),
                          tuple(tuple(part) for part in node.bbox_3d),
                          node.observation_count, node.geometry_status,
                          tuple(node.evidence_ids)) for key, node in graph.objects.items()))
    evidence = tuple(sorted((key, repr(record)) for key, record in graph.evidence.items()))
    viewpoints = tuple(sorted((key, repr(node)) for key, node in graph.viewpoints.items()))
    # get_edges_for_node returns graph.edges in insertion order.
    edges = tuple(repr(edge) for edge in graph.edges)
    historical = tuple(sorted(key for key, record in graph.evidence.items()
                              if (record.episode_id, record.rollout_idx)
                              != (tools.episode_id, tools.rollout_idx)
                              or record.step_id < tools.step))
    linked_records = []
    if tools.store is not None:
        for record in graph.evidence.values():
            if record.linked_l1_evidence_id is None:
                continue
            linked = tools.store.get_evidence(record.linked_l1_evidence_id)
            linked_records.append((record.linked_l1_evidence_id, None if linked is None else (
                linked.scene_id, linked.episode_id, linked.rollout_idx, linked.step,
                linked.event_type, linked.semantic_label, linked.observation_status,
                linked.provenance_validity,
            )))
    snapshot = (graph._identity, nodes, evidence, viewpoints, edges, historical,
                tuple(sorted(linked_records)), tools.scope, tools.min_confidence)
    return hashlib.sha256(repr(snapshot).encode("utf-8")).hexdigest()


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate Final JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite Final JSON value: {value}")


def _ids(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must be a list of nonempty IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate IDs")
    return value


def _final(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Final must be a JSON object")
    allowed = {
        "decision", "selected_l1_fact_ids", "selected_l3_object_ids",
        "selected_l3_evidence_ids", "reason_code", "action",
    }
    if set(value) - allowed or allowed - {"action"} - set(value):
        raise ValueError("Final has unknown or missing fields")
    if value.get("action", "FINAL") != "FINAL" or value["decision"] not in _DECISIONS:
        raise ValueError("Final decision is invalid")
    reason = value["reason_code"]
    if not isinstance(reason, str) or not reason or len(reason) > 64 or not all(
        char.isupper() or char.isdigit() or char == "_" for char in reason
    ):
        raise ValueError("Final reason_code must be a short category")
    for field in ("selected_l1_fact_ids", "selected_l3_object_ids", "selected_l3_evidence_ids"):
        _ids(value[field], field)
    has_l1 = bool(value["selected_l1_fact_ids"])
    has_l3 = bool(value["selected_l3_object_ids"] and value["selected_l3_evidence_ids"])
    expected = "USE_BOTH" if has_l1 and has_l3 else "USE_L1" if has_l1 else "USE_L3" if has_l3 else "NO_MEMORY"
    if expected != value["decision"] or bool(value["selected_l3_object_ids"]) != bool(
        value["selected_l3_evidence_ids"]
    ):
        raise ValueError("Final decision and selected IDs disagree")
    return value


def read_l1_facts(manager: Any, step: int) -> list[dict[str, Any]]:
    """Copy actual short-term observations; validate their existing V3 provenance."""
    state = manager.working.state
    store = manager.v3_store
    if state is None or store is None or manager.v3_scene_id is None:
        return []
    facts: list[dict[str, Any]] = []
    for observation in manager.working.recent_observations:
        if observation.step_id > step:
            continue
        for ref in observation.detections:
            key = canonical_evidence_key(
                manager.v3_scene_id, state.episode_id, manager.v3_rollout_idx, ref.event_id,
            )
            record = store.get_evidence(key)
            if (record is None or record.observation_status != "observed"
                    or not record.provenance_validity or record.step != ref.step_id
                    or record.event_type not in {"object_observation", "landmark"}
                    or (record.semantic_label or "").casefold() != ref.label.casefold()):
                continue
            facts.append({
                "fact_id": key, "label": ref.label, "step": ref.step_id,
                "confidence": ref.confidence, "source": ref.source,
                "current": ref.step_id == step,
            })
    return facts


class SceneQueryController:
    """Finite Think/Query/Final loop with read-only tools and deterministic output."""

    def __init__(self, config: SceneQueryConfig, provider: QueryProvider | None = None) -> None:
        self.config = config
        self.provider = provider or OpenAICompatibleQueryProvider(config)
        self.reset_episode()

    def reset_episode(self) -> None:
        self._selection_history: list[dict[str, Any]] = []
        self._query_signatures: set[str] = set()
        self._tool_result_cache: dict[str, Any] = {}
        self._cached_graph_state_key: str | None = None
        self._injected_evidence: set[str] = set()
        self._stats: Counter[str] = Counter({
            name: 0 for name in (
                "step_count", "l1_selection_count", "l3_query_count",
                "l3_hit_count", "both_selection_count", "no_memory_count",
                "api_failure_count", "invalid_id_rejection_count",
                "api_no_final_count", "max_tool_rounds_count",
                "max_tool_calls_count", "evidence_validation_failure_count",
                "graph_fusion_failure_count", "api_request_count",
                "api_request_latency_ms_total", "duplicate_tool_call_count",
                "tool_calls_requested", "tool_calls_executed",
                "tool_calls_succeeded", "tool_calls_reused",
                "tool_calls_rejected_budget", "tool_unknown_call_count",
                "tool_find_objects_call_count", "tool_read_node_call_count",
                "tool_read_evidence_call_count", "tool_get_edges_for_node_call_count",
                "repeated_query_count", "repeated_evidence_injection_count",
                "repeated_fact_expression_count", "rendered_fact_count",
                "actual_injection_step_count", "injected_evidence_count",
                "historical_injection_step_count", "actual_prompt_memory_tokens",
                "actual_prompt_total_tokens", "token_limit_removed_item_count",
                "native_visual_overlap_item_count",
            )
        })

    def episode_statistics(self) -> dict[str, int | float]:
        counts = dict(self._stats)
        query_total = counts.get("l3_query_count", 0)
        injection_total = counts.get("injected_evidence_count", 0)
        expression_total = counts.get("rendered_fact_count", 0)
        step_total = counts.get("step_count", 0)
        counts.update({
            "repeated_query_rate": counts.get("repeated_query_count", 0) / query_total if query_total else 0.0,
            "repeated_evidence_injection_rate": counts.get("repeated_evidence_injection_count", 0) / injection_total if injection_total else 0.0,
            "repeated_fact_expression_rate": counts.get("repeated_fact_expression_count", 0) / expression_total if expression_total else 0.0,
            "effective_historical_injection_rate": counts.get("historical_injection_step_count", 0) / step_total if step_total else 0.0,
            "native_visual_source_overlap_rate": counts.get("native_visual_overlap_item_count", 0) / expression_total if expression_total else 0.0,
        })
        return counts

    def _remember(self, step: int, diagnostic: dict[str, Any]) -> None:
        self._selection_history.append({
            "step": step,
            "decision": diagnostic["decision"],
            "object_ids": list(diagnostic["selected_l3_object_ids"]),
            "evidence_ids": list(diagnostic["selected_l3_evidence_ids"]),
            "l1_fact_ids": list(diagnostic["selected_l1_fact_ids"]),
            "actual_injection": False,
            "injected_object_ids": [],
            "injected_evidence_ids": [],
            "injected_l1_fact_ids": [],
        })
        self._stats["step_count"] += 1
        self._count_decision(diagnostic["decision"], 1)
        category = diagnostic["failure_category"]
        category_keys = {
            "API_REQUEST_FAILED": "api_failure_count",
            "API_NO_FINAL": "api_no_final_count",
            "MAX_TOOL_ROUNDS": "max_tool_rounds_count",
            "MAX_TOOL_CALLS": "max_tool_calls_count",
            "EVIDENCE_VALIDATION_FAILED": "evidence_validation_failure_count",
            "GRAPH_FUSION_FAILED": "graph_fusion_failure_count",
        }
        if category in category_keys:
            self._stats[category_keys[category]] += 1
        self._stats["invalid_id_rejection_count"] += diagnostic["invalid_id_rejection_count"]
        self._stats["api_request_count"] += diagnostic["api_request_count"]
        self._stats["api_request_latency_ms_total"] += sum(diagnostic["request_latencies_ms"])
        self._stats["l3_query_count"] += len(diagnostic["query_calls"])
        self._stats["repeated_query_count"] += sum(
            bool(call.get("repeated")) for call in diagnostic["query_calls"]
        )
        for key in ("duplicate_tool_call_count", "tool_calls_requested",
                    "tool_calls_executed", "tool_calls_succeeded",
                    "tool_calls_reused", "tool_calls_rejected_budget"):
            self._stats[key] += diagnostic[key]
        for name, count in diagnostic["tool_call_counts"].items():
            key = f"tool_{name}_call_count" if name in {
                "find_objects", "read_node", "read_evidence", "get_edges_for_node",
            } else "tool_unknown_call_count"
            self._stats[key] += count

    def _count_decision(self, decision: str, delta: int) -> None:
        if decision in {"USE_L1", "USE_BOTH"}:
            self._stats["l1_selection_count"] += delta
        if decision in {"USE_L3", "USE_BOTH"}:
            self._stats["l3_hit_count"] += delta
        if decision == "USE_BOTH":
            self._stats["both_selection_count"] += delta
        if decision == "NO_MEMORY":
            self._stats["no_memory_count"] += delta

    def _selection_summary(self, available_l1_ids: set[str]) -> list[dict[str, Any]]:
        """One bounded-by-distinct-ID record per fact, with no step cooldown."""
        summary: dict[tuple[str, str], dict[str, Any]] = {}
        groups = (
            ("l3_object", "object_ids", "injected_object_ids"),
            ("l3_evidence", "evidence_ids", "injected_evidence_ids"),
            ("l1_fact", "l1_fact_ids", "injected_l1_fact_ids"),
        )
        for selection in self._selection_history:
            for kind, selected_field, injected_field in groups:
                for fact_id in selection[selected_field]:
                    if kind == "l1_fact" and fact_id not in available_l1_ids:
                        continue
                    key = (kind, fact_id)
                    item = summary.setdefault(key, {
                        "kind": kind, "id": fact_id,
                        "selection_count": 0, "last_selected_step": -1,
                        "injection_count": 0, "last_injected_step": None,
                    })
                    item["selection_count"] += 1
                    item["last_selected_step"] = selection["step"]
                for fact_id in selection[injected_field]:
                    if kind == "l1_fact" and fact_id not in available_l1_ids:
                        continue
                    item = summary.get((kind, fact_id))
                    if item is not None:
                        item["injection_count"] += 1
                        item["last_injected_step"] = selection["step"]
        return sorted(summary.values(), key=lambda item: (
            -item["last_selected_step"], item["kind"], item["id"],
        ))

    def record_injection(
        self, *, step: int, injected_ids: list[str], actual: bool,
        memory_tokens: int | None, final_token_count: int | None, removed_items: int,
        historical_ids: list[str],
        injected_object_ids: list[str] | None = None,
        injected_l1_fact_ids: list[str] | None = None,
        final_item_count: int = 0, duplicate_fact_expression_count: int = 0,
        native_visual_overlap_item_count: int = 0,
        final_decision: str | None = None,
    ) -> None:
        if self._selection_history and self._selection_history[-1]["step"] == step:
            selection = self._selection_history[-1]
            if final_decision is not None and final_decision != selection["decision"]:
                if final_decision not in _DECISIONS:
                    raise ValueError("final scene query decision is invalid")
                self._count_decision(selection["decision"], -1)
                self._count_decision(final_decision, 1)
            selection["final_decision"] = final_decision or selection["decision"]
            selection["actual_injection"] = actual
            selection["injected_object_ids"] = list(injected_object_ids or []) if actual else []
            selection["injected_evidence_ids"] = list(injected_ids) if actual else []
            selection["injected_l1_fact_ids"] = list(injected_l1_fact_ids or []) if actual else []
        if actual:
            self._stats["actual_injection_step_count"] += 1
            self._stats["injected_evidence_count"] += len(injected_ids)
            self._stats["rendered_fact_count"] += final_item_count
            self._stats["repeated_fact_expression_count"] += duplicate_fact_expression_count
            self._stats["native_visual_overlap_item_count"] += native_visual_overlap_item_count
            self._stats["repeated_evidence_injection_count"] += sum(
                key in self._injected_evidence for key in injected_ids
            )
            self._injected_evidence.update(injected_ids)
            if historical_ids:
                self._stats["historical_injection_step_count"] += 1
        if actual and isinstance(memory_tokens, int):
            self._stats["actual_prompt_memory_tokens"] += memory_tokens
        if isinstance(final_token_count, int):
            self._stats["actual_prompt_total_tokens"] += final_token_count
        self._stats["token_limit_removed_item_count"] += removed_items

    def run(
        self, manager: Any, *, instruction: str, step: int, agent_pose: Any,
        native_visual_source_steps: tuple[int, ...] = (),
    ) -> tuple[MemoryContext | None, dict[str, Any]]:
        state = manager.working.state
        assert state is not None and manager.spatial_map is not None and manager.v3_store is not None
        l1_facts = read_l1_facts(manager, step)
        l1_by_id = {item["fact_id"]: item for item in l1_facts}
        tools = SceneQueryTools(
            manager.spatial_map, manager.v3_store,
            scene_id=manager.v3_scene_id, episode_id=state.episode_id,
            rollout_idx=manager.v3_rollout_idx, step=step,
            scope=manager.local_graph_config.scope,
            min_confidence=manager.spatial_l3_config.min_injection_confidence,
        )
        counts = tools.candidate_counts()
        graph_state_key = _graph_state_key(tools)
        cache_invalidated = (self._cached_graph_state_key is not None
                             and self._cached_graph_state_key != graph_state_key)
        if self._cached_graph_state_key != graph_state_key:
            self._tool_result_cache.clear()
            self._cached_graph_state_key = graph_state_key
        l3_available = counts["l3_evidence_valid_count"] > 0
        diagnostic: dict[str, Any] = {
            "step": step, "memory_strategy": "scene_query",
            "l1_available_count": len(l1_facts), **counts,
            "query_rounds": 0, "api_rounds": 0, "query_calls": [],
            "query_returned_object_ids": [], "query_returned_evidence_ids": [],
            "query_returned_eligible_object_ids": [],
            "api_request_count": 0, "api_round_request_counts": [],
            "request_latencies_ms": [], "tool_call_counts": {},
            "duplicate_tool_call_count": 0, "tool_calls_requested": 0,
            "tool_calls_executed": 0, "tool_calls_succeeded": 0,
            "tool_calls_reused": 0, "tool_calls_rejected_budget": 0,
            "tool_call_budget": self.config.max_tool_calls,
            "query_cache_invalidated": cache_invalidated,
            "query_graph_state_key": graph_state_key,
            "l3_query_disabled_reason": None if l3_available else "NO_ELIGIBLE_HISTORY",
            "api_decision": None, "decision": "NO_MEMORY", "reason_code": None,
            "selected_l1_fact_ids": [], "selected_l3_object_ids": [],
            "selected_l3_evidence_ids": [], "deduplicated_fact_count": 0,
            "duplicate_fact_expression_count": 0, "conflict_abstention_count": 0,
            "repeated_selection_count": 0, "native_visual_overlap_count": 0,
            "memory_context_item_count": 0, "memory_context_chars": 0,
            "memory_context_tokens": None, "invalid_id_rejection_count": 0,
            "fallback_reason": None, "failure_category": None,
        }
        pose = None if agent_pose is None else {
            "position": list(agent_pose.position),
            "rotation_wxyz": list(agent_pose.rotation_wxyz),
        }
        user_payload = {
            "instruction": instruction, "current_step": step,
            "current_agent_pose": pose, "l1_facts": l1_facts,
            "native_visual_source_steps": list(native_visual_source_steps),
            "recent_memory_selections": self._selection_summary(set(l1_by_id)),
            "available_l3_query_tools": (
                [item["function"]["name"] for item in OPENAI_TOOLS] if l3_available else []
            ),
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": (
                _SYSTEM + f"\nExecution budget: at most {self.config.max_rounds} API responses "
                f"and {self.config.max_tool_calls} tool call requests. "
                "The last response must be Final; do not wait until it to query."
            )},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        returned_objects: set[str] = set()
        eligible_objects: set[str] = set()
        discovered_objects: set[str] = set()
        returned_evidence: set[str] = set()
        final: dict[str, Any] | None = None
        step_signatures: set[str] = set()
        tool_calls_requested = 0
        tool_budget_exhausted = False
        final_prompt_sent = False
        try:
            for round_idx in range(self.config.max_rounds):
                diagnostic["api_rounds"] = round_idx + 1
                final_only = (not l3_available or tool_budget_exhausted
                              or round_idx == self.config.max_rounds - 1)
                if final_only and round_idx > 0 and not final_prompt_sent:
                    messages.append({
                        "role": "user",
                        "content": "No more map tool calls are available. Return the structured Final "
                                   "using only already returned and verified IDs, or NO_MEMORY.",
                    })
                    final_prompt_sent = True
                started = perf_counter()
                try:
                    message = self.provider.complete(
                        messages, OPENAI_TOOLS if l3_available else [],
                        tool_choice="none" if final_only else "auto",
                    )
                finally:
                    diagnostic["api_request_count"] += 1
                    diagnostic["api_round_request_counts"].append(1)
                    diagnostic["request_latencies_ms"].append(
                        round((perf_counter() - started) * 1000.0, 3)
                    )
                if not isinstance(message, dict):
                    raise ValueError("API message must be an object")
                tool_calls = message.get("tool_calls") or []
                if not isinstance(tool_calls, list):
                    raise ValueError("API tool_calls must be a list")
                if tool_calls:
                    # Count API requests before checking the local round/call budgets.
                    diagnostic["tool_calls_requested"] += len(tool_calls)
                    counts_by_tool = diagnostic["tool_call_counts"]
                    for requested_call in tool_calls:
                        requested_function = (requested_call.get("function")
                                              if isinstance(requested_call, dict) else None)
                        requested_name = (requested_function.get("name")
                                          if isinstance(requested_function, dict) else None)
                        tool_name = (requested_name if isinstance(requested_name, str)
                                     and requested_name else "unknown")
                        counts_by_tool[tool_name] = counts_by_tool.get(tool_name, 0) + 1
                    if final_only:
                        if tool_budget_exhausted:
                            diagnostic["tool_calls_rejected_budget"] += len(tool_calls)
                            raise _ToolCallBudgetError("API requested tools after the tool-call budget")
                        if l3_available:
                            diagnostic["tool_calls_rejected_budget"] += len(tool_calls)
                            raise _ToolRoundBudgetError("API requested tools on the final-only round")
                        raise _NoFinalError("API requested L3 tools with no eligible history")
                    diagnostic["query_rounds"] += 1
                    messages.append(_assistant_message(message, tool_calls=tool_calls))
                    seen_call_ids: set[str] = set()
                    for call in tool_calls:
                        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                            raise ValueError("API tool call is invalid")
                        call_id = call.get("id")
                        if not isinstance(call_id, str) or not call_id or call_id in seen_call_ids:
                            raise ValueError("API tool call ID is missing or duplicated")
                        seen_call_ids.add(call_id)
                        function = call["function"]
                        name = function.get("name")
                        raw_args = function.get("arguments")
                        tool_name = name if isinstance(name, str) and name else "unknown"
                        tool_calls_requested += 1
                        if tool_calls_requested > self.config.max_tool_calls:
                            tool_budget_exhausted = True
                            diagnostic["tool_calls_rejected_budget"] += 1
                            diagnostic["query_calls"].append({
                                "name": tool_name, "round": round_idx + 1,
                                "error": "tool_call_budget_exhausted", "repeated": False,
                            })
                            content = {"error": "tool_call_budget_exhausted"}
                            messages.append({
                                "role": "tool", "tool_call_id": call_id,
                                "content": json.dumps(content),
                            })
                            continue
                        try:
                            arguments = json.loads(
                                raw_args, object_pairs_hook=_unique_json_pairs,
                                parse_constant=_reject_json_constant,
                            ) if isinstance(raw_args, str) else raw_args
                            signature = json.dumps(
                                [name, arguments], sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            repeated = signature in self._query_signatures
                            self._query_signatures.add(signature)
                            same_step_duplicate = signature in step_signatures
                            step_signatures.add(signature)
                            diagnostic["duplicate_tool_call_count"] += int(same_step_duplicate)
                            cache_hit = signature in self._tool_result_cache
                            if cache_hit:
                                result = deepcopy(self._tool_result_cache[signature])
                                diagnostic["tool_calls_reused"] += 1
                            else:
                                diagnostic["tool_calls_executed"] += 1
                                result = tools.call(name, raw_args)
                                diagnostic["tool_calls_succeeded"] += 1
                                self._tool_result_cache[signature] = deepcopy(result)
                            object_ids = (
                                [item["object_id"] for item in result] if name == "find_objects" else
                                [result["object_id"]] if name == "read_node" else
                                result.get("object_ids", []) if name == "read_evidence" else []
                            )
                            evidence_ids = [result["evidence_id"]] if name == "read_evidence" else []
                            eligible_ids = (
                                [item["object_id"] for item in result
                                 if item.get("historical_eligible_evidence_ids")]
                                if name == "find_objects" else
                                [result["object_id"]]
                                if name == "read_node" and result.get("historical_eligible_evidence_ids") else
                                result.get("historical_eligible_object_ids", [])
                                if name == "read_evidence" else []
                            )
                            returned_objects.update(object_ids)
                            eligible_objects.update(eligible_ids)
                            if name in {"find_objects", "read_node"}:
                                discovered_objects.update(object_ids)
                            returned_evidence.update(evidence_ids)
                            diagnostic["query_calls"].append({
                                "name": name, "arguments": arguments, "repeated": repeated,
                                "same_step_duplicate": same_step_duplicate,
                                "cache_hit": cache_hit, "round": round_idx + 1,
                                "returned_object_ids": object_ids,
                                "eligible_object_ids": eligible_ids,
                                "returned_evidence_ids": evidence_ids,
                            })
                            content = {"result": result, "cache_hit": cache_hit}
                        except (ValueError, TypeError, KeyError) as exc:
                            error = _safe_error(exc)
                            if "object ID" in error or "Evidence ID" in error:
                                diagnostic["invalid_id_rejection_count"] += 1
                            diagnostic["query_calls"].append({
                                "name": tool_name, "round": round_idx + 1,
                                "error": error, "repeated": False,
                            })
                            content = {"error": error}
                        messages.append({
                            "role": "tool", "tool_call_id": call_id,
                            "content": json.dumps(content, ensure_ascii=False),
                        })
                    if tool_calls_requested >= self.config.max_tool_calls:
                        tool_budget_exhausted = True
                    continue
                content = message.get("content")
                if content is None or content == "":
                    if (message.get("reasoning_content")
                            and round_idx < self.config.max_rounds - 1):
                        # DeepSeek can return a reasoning-only assistant turn.
                        # Preserve it for the next tool-bearing request without logging it.
                        messages.append(_assistant_message(message))
                        continue
                    raise _NoFinalError("API returned neither tool calls nor Final")
                try:
                    parsed = json.loads(
                        content, object_pairs_hook=_unique_json_pairs,
                        parse_constant=_reject_json_constant,
                    ) if isinstance(content, str) else content
                except (ValueError, TypeError) as exc:
                    raise _NoFinalError("API returned invalid Final JSON") from exc
                if isinstance(parsed, dict) and parsed.get("action") == "THINK":
                    if set(parsed) != {"action", "reason_code"}:
                        raise ValueError("Think has invalid fields")
                    messages.append(_assistant_message(message))
                    continue
                try:
                    final = _final(parsed)
                except ValueError as exc:
                    raise _NoFinalError(str(exc)) from exc
                break
            if final is None:
                if tool_budget_exhausted:
                    raise _ToolCallBudgetError("API did not return Final within tool-call budget")
                if diagnostic["query_rounds"] >= max(1, self.config.max_rounds - 1):
                    raise _ToolRoundBudgetError("API did not return Final within tool rounds")
                raise _NoFinalError("API did not return Final within max rounds")
            diagnostic["api_decision"] = final["decision"]
            diagnostic["reason_code"] = final["reason_code"]
            selected_l1 = final["selected_l1_fact_ids"]
            selected_objects = final["selected_l3_object_ids"]
            selected_evidence = final["selected_l3_evidence_ids"]
            if any(key not in l1_by_id for key in selected_l1):
                diagnostic["invalid_id_rejection_count"] += 1
                raise _EvidenceValidationError("Final selected an invalid L1 fact ID")
            if any(key not in discovered_objects for key in selected_objects):
                diagnostic["invalid_id_rejection_count"] += 1
                raise _EvidenceValidationError("Final selected an undiscovered object ID")
            if any(key not in returned_evidence for key in selected_evidence):
                diagnostic["invalid_id_rejection_count"] += 1
                raise _EvidenceValidationError("Final selected an unqueried Evidence ID")
            verified: dict[str, list[str]] = {object_id: [] for object_id in selected_objects}
            for evidence_id in selected_evidence:
                owners = [object_id for object_id in selected_objects
                          if evidence_id in manager.spatial_map.objects[object_id].evidence_ids]
                if len(owners) != 1:
                    diagnostic["invalid_id_rejection_count"] += 1
                    raise _EvidenceValidationError("Final Evidence ID has no unique selected object")
                try:
                    tools.validate_history(owners[0], evidence_id)
                except ValueError as exc:
                    if str(exc).startswith("fused Object has missing or weak Evidence"):
                        raise _GraphFusionError(str(exc)) from exc
                    raise _EvidenceValidationError(str(exc)) from exc
                verified[owners[0]].append(evidence_id)
            if any(not ids for ids in verified.values()):
                raise _EvidenceValidationError("Final selected an object without verified Evidence")
            try:
                context, kept_l1, kept_l3, dedup, conflicts = self._render(
                    manager, step, agent_pose, l1_by_id, selected_l1, verified,
                )
            except (KeyError, ValueError) as exc:
                raise _GraphFusionError(str(exc)) from exc
            diagnostic["selected_l1_fact_ids"] = kept_l1
            diagnostic["selected_l3_object_ids"] = list(kept_l3)
            diagnostic["selected_l3_evidence_ids"] = [
                evidence_id for ids in kept_l3.values() for evidence_id in ids
            ]
            diagnostic["deduplicated_fact_count"] = dedup
            diagnostic["conflict_abstention_count"] = conflicts
            diagnostic["decision"] = (
                "USE_BOTH" if kept_l1 and kept_l3 else "USE_L1" if kept_l1 else
                "USE_L3" if kept_l3 else "NO_MEMORY"
            )
            if context is not None:
                lines = [item.description for item in context.layer_items]
                diagnostic["memory_context_item_count"] = len(lines)
                diagnostic["memory_context_chars"] = len(context.text)
                diagnostic["duplicate_fact_expression_count"] = len(lines) - len(set(lines))
            prior_ids = {
                key for selection in self._selection_history
                for key in (*selection["object_ids"], *selection["evidence_ids"])
            }
            diagnostic["repeated_selection_count"] = sum(
                key in prior_ids for key in (
                    *diagnostic["selected_l3_object_ids"],
                    *diagnostic["selected_l3_evidence_ids"],
                )
            )
            diagnostic["native_visual_overlap_count"] = sum(
                l1_by_id[key]["current"] for key in kept_l1
            )
        except Exception as exc:  # noqa: BLE001
            # Fail closed: no old spatial-first or local-graph fallback is allowed.
            diagnostic["fallback_reason"] = _safe_error(exc)
            diagnostic["failure_category"] = (
                "MAX_TOOL_CALLS" if isinstance(exc, _ToolCallBudgetError) else
                "MAX_TOOL_ROUNDS" if isinstance(exc, _ToolRoundBudgetError) else
                "API_NO_FINAL" if isinstance(exc, _NoFinalError) else
                "EVIDENCE_VALIDATION_FAILED" if isinstance(exc, _EvidenceValidationError) else
                "GRAPH_FUSION_FAILED" if isinstance(exc, _GraphFusionError) else
                "API_REQUEST_FAILED"
            )
            diagnostic["reason_code"] = diagnostic["reason_code"] or diagnostic["failure_category"]
            context = None
        diagnostic["query_returned_object_ids"] = sorted(returned_objects)
        diagnostic["query_returned_eligible_object_ids"] = sorted(eligible_objects)
        diagnostic["query_returned_evidence_ids"] = sorted(returned_evidence)
        diagnostic["tool_call_budget_exhausted"] = tool_budget_exhausted
        diagnostic["api_queried_node_count"] = len(discovered_objects)
        diagnostic["final_selected_object_count"] = len(diagnostic["selected_l3_object_ids"])
        diagnostic["final_selected_evidence_count"] = len(diagnostic["selected_l3_evidence_ids"])
        self._remember(step, diagnostic)
        return context, diagnostic

    @staticmethod
    def _render(
        manager: Any, step: int, agent_pose: Any, l1_by_id: dict[str, dict[str, Any]],
        selected_l1: list[str], verified: dict[str, list[str]],
    ) -> tuple[MemoryContext | None, list[str], dict[str, list[str]], int, int]:
        object_map = manager.spatial_map
        assert object_map is not None
        selected_l1_set = set(selected_l1)
        consumed_l1: set[str] = set()
        kept_l3: dict[str, list[str]] = {}
        items: list[LayerMemoryItem] = []
        dedup = 0
        conflicts = 0
        for object_id, evidence_ids in verified.items():
            node = object_map.objects[object_id]
            label = _plain_label(node.class_name)
            if not label:
                continue
            linked: set[str] = set()
            for key in node.evidence_ids:
                spatial_ref = object_map.evidence.get(key)
                if spatial_ref is None or spatial_ref.linked_l1_evidence_id not in selected_l1_set:
                    continue
                l1_record = manager.v3_store.get_evidence(spatial_ref.linked_l1_evidence_id)
                if (l1_record is not None and l1_record.observation_status == "observed"
                        and l1_record.provenance_validity
                        and (l1_record.scene_id, l1_record.episode_id,
                             l1_record.rollout_idx, l1_record.step) ==
                        (spatial_ref.scene_id, spatial_ref.episode_id,
                         spatial_ref.rollout_idx, spatial_ref.step_id)
                        and (l1_record.semantic_label or "").casefold() ==
                        spatial_ref.class_name.casefold()):
                    linked.add(spatial_ref.linked_l1_evidence_id)
            paired_l1 = next((key for key in selected_l1 if key in linked), None)
            if paired_l1 is None and any(
                l1_by_id[key]["label"].casefold() == node.class_name.casefold()
                and l1_by_id[key]["current"] for key in selected_l1
            ):
                # A same-label current detection cannot identify a past map node.
                conflicts += 1
                continue
            record = object_map.evidence[evidence_ids[0]]
            center = tuple(node.center_world)
            center_text = ",".join(f"{value:.1f}" for value in center)
            line = f"Spatial history: {label} (Object ID {object_id}) mapped at world ({center_text}) m"
            relative = _relative_position(agent_pose, center) if agent_pose is not None else None
            if relative is not None:
                line += f"; now {relative[0]:.1f} m straight-line, bearing {relative[1]:+.0f} deg (left +)"
            unit = "observation" if node.observation_count == 1 else "observations"
            line += f"; {node.observation_count} {unit}; passage unknown."
            refs = [ObservationEvidence(
                event_id=key, step_id=object_map.evidence[key].step_id,
                label=label, confidence=object_map.evidence[key].confidence,
                source=object_map.evidence[key].source,
            ) for key in evidence_ids]
            provenance = [{
                "scene_id": object_map.evidence[key].scene_id,
                "episode_id": object_map.evidence[key].episode_id,
                "rollout_idx": object_map.evidence[key].rollout_idx,
                "step": object_map.evidence[key].step_id,
                "evidence_id": key,
                "linked_l1_evidence_id": object_map.evidence[key].linked_l1_evidence_id,
            } for key in evidence_ids]
            if paired_l1 is not None:
                fact = l1_by_id[paired_l1]
                prefix = "Current RGB" if fact["current"] else f"Recent RGB at step {fact['step']}"
                line = f"{prefix}: detected {label}; {line}"
                refs.insert(0, ObservationEvidence(
                    event_id=paired_l1, step_id=fact["step"], label=label,
                    confidence=fact["confidence"], source=fact["source"],
                ))
                provenance.insert(0, {"evidence_id": paired_l1, "step": fact["step"]})
                consumed_l1.add(paired_l1)
                dedup += 1
            item_ids = [ref.event_id for ref in refs]
            items.append(LayerMemoryItem(
                layer="long", kind="scene_query_spatial_object", description=line,
                step_id=record.step_id, confidence=record.confidence,
                source="scene_query", evidence_refs=refs, label=label,
                metadata={
                    "evidence_ids": item_ids, "serialized_text": escape(line, quote=False),
                    "provenance": provenance, "object_id": object_id,
                    "role": "current_and_spatial" if paired_l1 else "spatial_history",
                },
            ))
            kept_l3[object_id] = evidence_ids
        for key in selected_l1:
            if key in consumed_l1:
                continue
            fact = l1_by_id[key]
            label = _plain_label(fact["label"])
            if not label:
                continue
            line = (
                f"Current RGB: detected {label}; passage unknown."
                if fact["current"] else
                f"Recent RGB at step {fact['step']}: detected {label}; current visibility unknown."
            )
            items.insert(0, LayerMemoryItem(
                layer="short", kind="scene_query_l1_fact", description=line,
                step_id=fact["step"], confidence=fact["confidence"],
                source="scene_query", evidence_refs=[ObservationEvidence(
                    event_id=key, step_id=fact["step"], label=label,
                    confidence=fact["confidence"], source=fact["source"],
                )], label=label, metadata={
                    "evidence_ids": [key], "serialized_text": escape(line, quote=False),
                    "provenance": [{"evidence_id": key, "step": fact["step"]}],
                    "role": "current" if fact["current"] else "recent",
                },
            ))
        if not items:
            return None, [], {}, dedup, conflicts
        context = MemoryContext(
            text=_HEADER + "\n" + "\n".join(item.description for item in items),
            layer_items=items,
        )
        kept_l1 = [key for key in selected_l1 if any(
            key in item.metadata["evidence_ids"] for item in items
        )]
        return context, kept_l1, kept_l3, dedup, conflicts
