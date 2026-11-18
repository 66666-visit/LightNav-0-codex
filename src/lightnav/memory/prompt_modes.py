"""Prompt-organization modes for the L1 + L3 navigation-context ablation.

The ten modes (legacy + eight historical conditions + the official-style L3
condition) select which existing memory layers contribute text and which form
of a layer is serialized. They never change model weights, action decoding, the
video input, or the original prompt text: every mode injects one bounded block
after ``</navigation_task>`` using the same adapter and the same budget policy
as the original Local Graph path.

``l3_official_style_only`` is the follow-up expression-only condition. It keeps
the L3 candidate retrieval of ``l3_organized_only`` but renders the selected
map facts as robot-relative statements in the language of the official Prompt
Guide. L1 prompt construction is disabled exactly as in the other L3-only mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

PROMPT_PROTOCOL_VERSION = "l1_l3_prompt_ablation_v1"

#: The six conditions of the original 88-Episode ablation.
CORE_ABLATION_PROMPT_MODES: tuple[str, ...] = (
    "l1_original_only",
    "l1_organized_only",
    "l3_original_only",
    "l3_organized_only",
    "l1_l3_original_concat",
    "l1_l3_joint",
)

#: The two follow-up conditions: original L1 verbatim plus organized L3.
SUPPLEMENTAL_ABLATION_PROMPT_MODES: tuple[str, ...] = (
    "l1_original_l3_organized_concat",
    "l1_original_l3_organized_joint",
)

#: All eight historical non-legacy conditions.
ABLATION_PROMPT_MODES: tuple[str, ...] = (
    *CORE_ABLATION_PROMPT_MODES,
    *SUPPLEMENTAL_ABLATION_PROMPT_MODES,
)

#: Follow-up expression-only condition (not part of the 88-Episode historical
#: seven). It reuses the organized-L3 candidate retrieval.
OFFICIAL_STYLE_PROMPT_MODE = "l3_official_style_only"

MEMORY_PROMPT_MODES: tuple[str, ...] = (
    "legacy", *ABLATION_PROMPT_MODES, OFFICIAL_STYLE_PROMPT_MODE,
)

_LAYER_SOURCES = {"original", "organized", "official"}


@dataclass(frozen=True)
class PromptModePlan:
    """Which layer sources feed one prompt and how they are combined."""

    mode: str
    l1_source: str | None
    l3_source: str | None
    combination: str  # legacy | single | concat | joint | none

    def __post_init__(self) -> None:
        if self.mode not in MEMORY_PROMPT_MODES:
            raise ValueError(f"unknown memory prompt mode: {self.mode}")
        for name in ("l1_source", "l3_source"):
            value = getattr(self, name)
            if value is not None and value not in _LAYER_SOURCES:
                raise ValueError(f"{name} must be original, organized or None")
        if self.combination not in {"legacy", "single", "concat", "joint", "none"}:
            raise ValueError("invalid prompt combination")
        if self.mode == "legacy" and self.combination != "legacy":
            raise ValueError("legacy must keep the original prompt path")
        if self.mode != "legacy" and self.combination == "legacy":
            raise ValueError("non-legacy modes require an explicit combination")
        if self.combination == "single" and (self.uses_l1 == self.uses_l3):
            raise ValueError("a single-layer mode must select exactly one layer")
        if self.combination == "joint" and (
            self.l1_source not in {"original", "organized"}
            or self.l3_source != "organized"
        ):
            raise ValueError("joint mode requires an L1 source and organized L3")
        if self.combination == "concat" and (
            self.l1_source != "original" or self.l3_source is None
        ):
            raise ValueError("concat mode requires original L1 and an L3 source")

    @property
    def uses_l1(self) -> bool:
        return self.l1_source is not None

    @property
    def uses_l3(self) -> bool:
        return self.l3_source is not None

    @property
    def l1_organized(self) -> bool:
        return self.l1_source == "organized"

    @property
    def l3_organized(self) -> bool:
        return self.l3_source == "organized"

    @property
    def l3_official(self) -> bool:
        return self.l3_source == "official"

    @property
    def joint(self) -> bool:
        return self.combination == "joint"

    @property
    def concatenated(self) -> bool:
        return self.combination == "concat"

    def require_legacy(self) -> bool:
        return self.combination == "legacy"

    @property
    def preserves_l1_verbatim(self) -> bool:
        """True when joint fusion must not rewrite the original L1 lines."""
        return self.joint and self.l1_source == "original"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_PLANS: dict[str, PromptModePlan] = {
    # Legacy is resolved on the original code path; this plan only documents it.
    "legacy": PromptModePlan("legacy", "original", None, "legacy"),
    "l1_original_only": PromptModePlan(
        "l1_original_only", "original", None, "single",
    ),
    "l1_organized_only": PromptModePlan(
        "l1_organized_only", "organized", None, "single",
    ),
    "l3_original_only": PromptModePlan(
        "l3_original_only", None, "original", "single",
    ),
    "l3_organized_only": PromptModePlan(
        "l3_organized_only", None, "organized", "single",
    ),
    "l1_l3_original_concat": PromptModePlan(
        "l1_l3_original_concat", "original", "original", "concat",
    ),
    "l1_l3_joint": PromptModePlan(
        "l1_l3_joint", "organized", "organized", "joint",
    ),
    "l1_original_l3_organized_concat": PromptModePlan(
        "l1_original_l3_organized_concat", "original", "organized", "concat",
    ),
    "l1_original_l3_organized_joint": PromptModePlan(
        "l1_original_l3_organized_joint", "original", "organized", "joint",
    ),
    OFFICIAL_STYLE_PROMPT_MODE: PromptModePlan(
        OFFICIAL_STYLE_PROMPT_MODE, None, "official", "single",
    ),
}


def mode_plan(mode: str) -> PromptModePlan:
    """Return the immutable plan for one ``--memory_prompt_mode`` value."""
    if not isinstance(mode, str):
        raise ValueError("memory prompt mode must be a string")
    try:
        return _PLANS[mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown memory prompt mode {mode!r}; expected one of {MEMORY_PROMPT_MODES}"
        ) from exc


def resolve_prompt_budget(*, max_context_chars: int, local_graph_prompt_max_chars: int) -> int:
    """Return the original memory-block character budget.

    The Local Graph and spatial-L3 paths already use
    ``min(local_graph.prompt_max_chars, memory.max_context_chars)``. Every
    experimental mode uses exactly that number, so no mode gains extra budget.
    """
    if type(max_context_chars) is not int or max_context_chars < 1:
        raise ValueError("max_context_chars must be a positive integer")
    if type(local_graph_prompt_max_chars) is not int or local_graph_prompt_max_chars < 1:
        raise ValueError("local_graph_prompt_max_chars must be a positive integer")
    return min(max_context_chars, local_graph_prompt_max_chars)


def validate_mode_backing(
    plan: PromptModePlan, *, local_graph_enabled: bool, spatial_l3_mode: str,
) -> None:
    """Reject modes whose layer sources are not enabled by the run config."""
    if not isinstance(plan, PromptModePlan):
        raise TypeError("plan must be a PromptModePlan")
    if plan.require_legacy():
        return
    if not local_graph_enabled:
        raise ValueError(f"{plan.mode} requires local graph memory")
    if plan.uses_l3 and spatial_l3_mode != "inject":
        raise ValueError(
            f"{plan.mode} requires spatial_l3.mode=inject (got {spatial_l3_mode!r})"
        )


def effective_layer_activation(plan: PromptModePlan) -> dict[str, object]:
    """Describe which data paths the mode actually runs (for protocol artifacts).

    Historical modes keep exactly the original activation dictionary so an
    already-completed protocol remains comparable. Only the follow-up joint
    mode adds the two lossless-L1 fields.
    """
    activation: dict[str, object] = {
        "mode": plan.mode,
        "l1_layer": plan.l1_source,
        "l3_layer": plan.l3_source,
        "combination": plan.combination,
        "l1_prompt_enabled": plan.uses_l1,
        "l3_prompt_enabled": plan.uses_l3,
        "joint_alignment_enabled": plan.joint,
    }
    if plan.preserves_l1_verbatim:
        activation["l1_preservation"] = "verbatim_original_builder_lines"
        activation["aligned_l3_output"] = "separate_map_line"
    return activation
