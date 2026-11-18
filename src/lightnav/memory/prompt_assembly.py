"""Assemble one bounded navigation-context block for a prompt mode.

This is the only place the experimental modes select data. The original
``LocalGraphFusion`` and ``SpatialGraphAdapter`` builders are called unchanged;
the organized builders are read-only formatters over the same records. The
supplemental concat/joint modes read the original L1 card and the organized L3
card; the joint path never runs ``L1PromptOrganizer``. The character budget is
the original Local Graph budget for every mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import Any

from lightnav.memory.joint_prompt import JointOrganization, JointPromptBuilder
from lightnav.memory.l1_prompt_organizer import L1Organization, L1PromptOrganizer
from lightnav.memory.l3_prompt_organizer import L3Organization, L3PromptOrganizer
from lightnav.memory.prompt_modes import PromptModePlan
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext
from lightnav.memory.token_budget import (
    UNBOUNDED_CHAR_LIMIT,
    MemoryTokenBudget,
    TokenBudgetFailure,
    select_context_lines,
)
from lightnav.memory.v3_store import V3MemoryStore


@dataclass
class PromptContextResult:
    """The selected context and the evidence that explains its selection."""

    plan: PromptModePlan
    context: MemoryContext | None
    source: str
    budget_chars: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
    local_graph_pending: bool = False
    l1_evidence_ids: tuple[str, ...] = ()
    budget: dict[str, Any] = field(default_factory=dict)
    block_tokens: int | None = None
    omitted_lines: int = 0
    omitted_items: int = 0
    injection_max_chars: int = 0

    @property
    def text(self) -> str:
        return self.context.text if self.context is not None else ""

    @property
    def layer_items(self) -> list[LayerMemoryItem]:
        return self.context.layer_items if self.context is not None else []


def build_prompt_context(
    plan: PromptModePlan,
    *,
    store: V3MemoryStore,
    object_map: Any | None,
    frame: Any | None,
    local_graph_fusion: Any,
    spatial_adapter: Any,
    l1_organizer: L1PromptOrganizer,
    l3_organizer: L3PromptOrganizer,
    joint_builder: JointPromptBuilder,
    scene_id: str,
    episode_id: str,
    rollout_idx: int,
    step: int,
    instruction: str,
    rgb_width: int,
    events: list[Any],
    current_place_id: str | None,
    scope: str,
    max_landmarks: int,
    min_confidence: float,
    budget_chars: int,
    char_limit: int | None = None,
    token_budget: MemoryTokenBudget | None = None,
    budget_failure: TokenBudgetFailure | None = None,
) -> PromptContextResult:
    """Build the mode's context without changing any underlying memory record."""
    if plan.require_legacy():
        raise ValueError("legacy is resolved on the original code path")
    if budget_chars < 1:
        raise ValueError("budget_chars must be positive")
    builder_chars = budget_chars if char_limit is None else char_limit
    if builder_chars < 1:
        raise ValueError("char_limit must be positive")

    l1_org: L1Organization | None = None
    l3_org: L3Organization | None = None
    l1_context: MemoryContext | None = None
    l3_context: MemoryContext | None = None
    local_graph_pending = False
    diagnostics: dict[str, Any] = {
        "mode": plan.mode,
        "budget_chars": budget_chars,
        "layers": {},
    }

    if plan.l1_source == "original":
        l1_context = local_graph_fusion.build(
            store, scene_id=scene_id, episode_id=episode_id, rollout_idx=rollout_idx,
            step=step, instruction=instruction, current_place_id=current_place_id,
            rgb_width=rgb_width, events=events,
        )
        local_graph_pending = l1_context is not None
        diagnostics["layers"]["l1"] = {
            "layer": "l1", "source": "local_graph_fusion",
            "item_count": len(l1_context.layer_items) if l1_context is not None else 0,
            "prompt_chars": len(l1_context.text) if l1_context is not None else 0,
            "empty_reason": None if l1_context is not None else "original_builder_empty",
        }
    elif plan.l1_source == "organized":
        l1_org = l1_organizer.build(
            store, scene_id=scene_id, episode_id=episode_id, rollout_idx=rollout_idx,
            step=step, instruction=instruction, rgb_width=rgb_width, events=events,
            max_chars=builder_chars,
        )
        l1_context = l1_org.context
        diagnostics["layers"]["l1"] = l1_org.diagnostics()

    if plan.uses_l3:
        if object_map is None:
            diagnostics["layers"]["l3"] = {
                "layer": "l3", "source": None, "candidate_count": 0,
                "item_count": 0, "prompt_chars": 0, "empty_reason": "l3_map_unavailable",
            }
        elif plan.l3_source == "original":
            l3_context = spatial_adapter.build(
                store, object_map, frame,
                scene_id=scene_id, episode_id=episode_id, rollout_idx=rollout_idx,
                step=step, instruction=instruction, scope=scope,
                max_landmarks=max_landmarks, prompt_max_chars=builder_chars,
                min_confidence=min_confidence,
            ) if frame is not None else None
            diagnostics["layers"]["l3"] = {
                "layer": "l3", "source": "spatial_graph_adapter",
                "candidate_count": (
                    len(l3_context.layer_items) if l3_context is not None else 0
                ),
                "item_count": len(l3_context.layer_items) if l3_context is not None else 0,
                "prompt_chars": len(l3_context.text) if l3_context is not None else 0,
                "empty_reason": (
                    None if l3_context is not None else
                    "no_frame" if frame is None else "no_trusted_spatial_candidate"
                ),
            }
        else:
            assert plan.l3_source == "organized"
            l3_org = l3_organizer.build(
                store, object_map, scene_id=scene_id, episode_id=episode_id,
                rollout_idx=rollout_idx, step=step, instruction=instruction,
                frame=frame, scope=scope, max_chars=builder_chars,
                min_confidence=min_confidence,
            )
            l3_context = l3_org.context
            diagnostics["layers"]["l3"] = l3_org.diagnostics()

    if plan.joint:
        if l3_org is None:
            l3_org = L3Organization(context=None, empty_reason="l3_map_unavailable")
        if plan.preserves_l1_verbatim:
            # The original Local Graph card is passed through verbatim. The
            # historical joint builder is not used here because it rebuilds L1
            # lines from L1Fact metadata, which would rewrite the original text.
            joint: JointOrganization = joint_builder.build_preserving_l1(
                l1_context, l3_org, step=step, max_chars=builder_chars,
            )
            diagnostics["layers"]["joint"] = joint.diagnostics()
            diagnostics["conflicts"] = list(joint.conflicts)
            diagnostics["alignment"] = joint.diagnostics()["alignment"]
            diagnostics["omitted_items"] = joint.omitted_items
            diagnostics["l1_omitted_items"] = 0
            diagnostics["l3_omitted_items"] = l3_org.omitted_items
            diagnostics["joint_omitted_items"] = joint.omitted_items
            diagnostics["duplicates_removed"] = joint.duplicates_removed
            diagnostics["unknown_relation_count"] = joint.unverified_relations
            diagnostics["layer_item_counts"] = {
                "l1": joint.l1_item_count, "l3": joint.l3_item_count,
                "joint": joint.joint_item_count,
                "aligned": joint.aligned_pairs,
            }
            l1_ids = tuple(dict.fromkeys(
                evidence_id
                for item in (joint.context.layer_items if joint.context else [])
                if item.metadata.get("layer_source") == "l1_original"
                for evidence_id in item.metadata.get("l1_evidence_ids", [])
            ))
            return _result(
                plan, joint.context, "joint", budget_chars, diagnostics,
                token_budget=token_budget, budget_failure=budget_failure,
                local_graph_pending=local_graph_pending and bool(l1_ids),
                l1_evidence_ids=l1_ids,
            )
        if l1_org is None:
            l1_org = L1PromptOrganizer().build(
                store, scene_id=scene_id, episode_id=episode_id,
                rollout_idx=rollout_idx, step=step, instruction=instruction,
                rgb_width=rgb_width, events=events, max_chars=budget_chars,
            )
        joint = joint_builder.build(
            l1_org, l3_org, step=step, max_chars=builder_chars,
        )
        diagnostics["layers"]["joint"] = joint.diagnostics()
        diagnostics["conflicts"] = list(joint.conflicts)
        diagnostics["alignment"] = joint.diagnostics()["alignment"]
        diagnostics["omitted_items"] = joint.omitted_items
        diagnostics["l1_omitted_items"] = l1_org.omitted_items
        diagnostics["l3_omitted_items"] = l3_org.omitted_items
        diagnostics["joint_omitted_items"] = joint.omitted_items
        diagnostics["duplicates_removed"] = (
            joint.duplicates_removed + l1_org.duplicates_removed
        )
        diagnostics["unknown_relation_count"] = joint.unverified_relations
        diagnostics["layer_item_counts"] = {
            "l1": joint.l1_item_count, "l3": joint.l3_item_count,
            "joint": joint.joint_item_count,
        }
        return _result(
            plan, joint.context, "joint", budget_chars, diagnostics,
            token_budget=token_budget, budget_failure=budget_failure,
            l1_evidence_ids=tuple(dict.fromkeys(
                evidence_id for item in (joint.context.layer_items if joint.context else [])
                if item.metadata.get("layer_contribution", {}).get("l1")
                for evidence_id in item.metadata.get("l1_evidence_ids", [])
            )),
        )
    if plan.concatenated:
        context, source, omitted, l1_included, l1_ids = _concat_contexts(
            l1_context, l3_context, builder_chars,
            l3_label=(
                "l3_organized" if plan.l3_source == "organized" else "l3_original"
            ),
        )
        diagnostics["omitted_items"] = omitted
        diagnostics["l1_omitted_items"] = 0
        diagnostics["l3_omitted_items"] = (
            l3_org.omitted_items if l3_org is not None else 0
        )
        diagnostics["joint_omitted_items"] = 0
        diagnostics["duplicates_removed"] = 0
        diagnostics["conflicts"] = []
        diagnostics["alignment"] = {
            "attempted": 0, "confirmed": 0, "uncertain": 0,
            "unmatched": 0, "success_rate": None, "items": [],
        }
        diagnostics["unknown_relation_count"] = 0
        diagnostics["layer_item_counts"] = {
            "l1": len(l1_context.layer_items) if l1_context is not None else 0,
            "l3": len(l3_context.layer_items) if l3_context is not None else 0,
            "joint": 0,
        }
        return _result(
            plan, context, source, budget_chars, diagnostics,
            token_budget=token_budget, budget_failure=budget_failure,
            local_graph_pending=local_graph_pending and l1_included,
            l1_evidence_ids=l1_ids,
        )
    if plan.l1_source is not None:
        context = l1_context
        source = "l1_original" if plan.l1_source == "original" else "l1_organized"
    else:
        context = l3_context
        source = "l3_original" if plan.l3_source == "original" else "l3_organized"
    diagnostics["omitted_items"] = (
        (l1_org.omitted_items if l1_org is not None else 0)
        + (l3_org.omitted_items if l3_org is not None else 0)
    )
    diagnostics["l1_omitted_items"] = l1_org.omitted_items if l1_org is not None else 0
    diagnostics["l3_omitted_items"] = l3_org.omitted_items if l3_org is not None else 0
    diagnostics["joint_omitted_items"] = 0
    l1_ids = (
        tuple(dict.fromkeys(
            ref.event_id for item in (context.layer_items if context is not None else [])
            for ref in item.evidence_refs
        ))
        if plan.uses_l1 else ()
    )
    return _result(
        plan, context, source, budget_chars, diagnostics,
        token_budget=token_budget, budget_failure=budget_failure,
        local_graph_pending=plan.l1_source == "original" and local_graph_pending,
        l1_evidence_ids=l1_ids,
    )


def _result(
    plan: PromptModePlan, context: MemoryContext | None, source: str,
    budget_chars: int, diagnostics: dict[str, Any], *,
    token_budget: MemoryTokenBudget | None = None,
    budget_failure: TokenBudgetFailure | None = None,
    local_graph_pending: bool = False,
    l1_evidence_ids: tuple[str, ...] = (),
) -> PromptContextResult:
    block_tokens: int | None = None
    omitted_lines = 0
    omitted_items = 0
    if token_budget is not None:
        diagnostics["budget"] = token_budget.diagnostics()
        if context is not None:
            if token_budget.available:
                selection = select_context_lines(
                    context, capacity_tokens=token_budget.available_tokens,
                    count_tokens=token_budget.counter,
                )
                context = selection.context
                block_tokens = selection.block_tokens
                omitted_lines = selection.omitted_lines
                omitted_items = selection.omitted_items
                diagnostics["omitted_items"] = (
                    int(diagnostics.get("omitted_items", 0) or 0) + omitted_lines
                )
                if context is None:
                    diagnostics["empty_reason"] = "token_selection_empty"
            else:
                context = None
                diagnostics["empty_reason"] = "no_token_headroom"
        diagnostics["budget"]["memory_block_tokens"] = block_tokens
        diagnostics["budget"]["omitted_lines_by_tokens"] = omitted_lines
    elif budget_failure is not None:
        diagnostics["budget"] = budget_failure.diagnostics()
    else:
        diagnostics["budget"] = {"budget_source": "char_fallback_320"}
    evidence_ids = list(dict.fromkeys(
        ref.event_id for item in (context.layer_items if context is not None else [])
        for ref in item.evidence_refs
    ))
    # Token/character selection can drop whole lines after the branch that
    # built the context. The supplemental original-L1 modes filter the L1
    # confirmation set to surviving lines; historical modes keep their old
    # behavior so completed groups remain reproducible.
    if plan.l1_source == "original" and plan.l3_source == "organized":
        surviving = set(evidence_ids)
        l1_evidence_ids = tuple(
            evidence_id for evidence_id in l1_evidence_ids
            if evidence_id in surviving
        )
    diagnostics.update({
        "source": source if context is not None else None,
        "context_chars": len(context.text) if context is not None else 0,
        "item_count": len(context.layer_items) if context is not None else 0,
        "evidence_ids": evidence_ids,
        "budget_chars": budget_chars if token_budget is None else None,
    })
    return PromptContextResult(
        plan=plan, context=context, source=source if context is not None else "none",
        budget_chars=budget_chars, diagnostics=diagnostics,
        local_graph_pending=(
            local_graph_pending and context is not None and bool(l1_evidence_ids)
        ),
        l1_evidence_ids=l1_evidence_ids if context is not None else (),
        budget=dict(diagnostics["budget"]), block_tokens=block_tokens,
        omitted_lines=omitted_lines, omitted_items=omitted_items,
        injection_max_chars=(
            UNBOUNDED_CHAR_LIMIT if token_budget is not None else budget_chars
        ),
    )


def _concat_contexts(
    l1: MemoryContext | None, l3: MemoryContext | None, budget_chars: int,
    *, l3_label: str,
) -> tuple[MemoryContext | None, str, int, bool, tuple[str, ...]]:
    """Concatenate the two selected blocks verbatim, then drop whole lines.

    The historical ``l1_l3_original_concat`` passes the original L1 and original
    L3 cards. The follow-up concat passes the same original L1 card and the
    organized L3 card; this function is unchanged apart from its generic name.
    """
    if l1 is None and l3 is None:
        return None, "none", 0, False, ()
    if l1 is None:
        return l3, l3_label, 0, False, ()
    if l3 is None:
        return l1, "l1_original", 0, True, tuple(dict.fromkeys(
            ref.event_id for item in l1.layer_items for ref in item.evidence_refs
        ))

    lines: list[str] = []
    used = 0
    included: list[str] = []
    omitted = 0
    all_lines = [*l1.text.splitlines(), *l3.text.splitlines()]
    for index, line in enumerate(all_lines):
        cost = len(line) + (1 if lines else 0)
        if used + cost > budget_chars:
            # The original cards are priority-ordered; once a line does not fit,
            # keep the prefix and drop every remaining line instead of leaving a
            # hole or a header without its facts.
            omitted += len(all_lines) - index
            break
        used += cost
        lines.append(line)
        included.append(line)
    while lines and lines[-1] in {"[Navigation Memory]"}:
        lines.pop()
        included.pop()
        omitted += 1
    if not included:
        return None, "none", omitted, False, ()
    included_set = {escape(line, quote=False) for line in included}
    items = [
        item for item in [*l1.layer_items, *l3.layer_items]
        if item.metadata.get("serialized_text") in included_set
    ]
    if not items:
        return None, "none", omitted, False, ()
    l1_items = [
        item for item in items
        if any(item is candidate for candidate in l1.layer_items)
    ]
    l1_ids = tuple(dict.fromkeys(
        ref.event_id for item in l1_items for ref in item.evidence_refs
    ))
    return (
        MemoryContext(text="\n".join(lines), layer_items=items),
        "concat", omitted, bool(l1_items), l1_ids,
    )
