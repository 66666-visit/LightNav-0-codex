"""Model-derived token budget for the experimental navigation contexts.

The original memory paths use a character cap (Local Graph ``prompt_max_chars``,
320 by default). That cap is an information-selection budget of the old memory
module, not the model's context limit. The experimental prompt modes therefore
measure the actual base prompt and the actual serving context limit with the
checkpoint's existing Processor/Tokenizer, and select whole lines against the
remaining token headroom.

If the Processor/Tokenizer or the context limit is unavailable, the caller must
fall back to the original 320-character budget and report why. This module never
guesses a larger budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from html import escape
from typing import Any, Callable

from lightnav.memory.schemas import MemoryContext

logger = logging.getLogger(__name__)

#: Sentinel for "no character cap" when the real limit is applied in tokens.
#: It is not a prompt budget and must never be reported as one.
UNBOUNDED_CHAR_LIMIT = 2**31 - 1

#: One token of boundary reserve for the inserted newline; measured, not a
#: character allowance. Recorded in diagnostics.
BOUNDARY_RESERVE_TOKENS = 1


@dataclass(frozen=True)
class MemoryTokenBudget:
    """The measured per-step token allowance for one memory block."""

    available_tokens: int
    context_limit_tokens: int
    base_prompt_tokens: int
    generation_reserve_tokens: int
    safety_margin_tokens: int
    context_limit_source: str
    counter: Callable[[str], int]
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None and self.available_tokens > 0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "budget_source": "token_aware_context",
            "context_limit_tokens": self.context_limit_tokens,
            "base_prompt_tokens": self.base_prompt_tokens,
            "generation_reserve_tokens": self.generation_reserve_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "available_memory_tokens": self.available_tokens,
            "context_limit_source": self.context_limit_source,
            "token_budget_unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class TokenBudgetFailure:
    """Why a model-derived budget could not be measured."""

    reason: str
    context_limit_tokens: int | None = None
    base_prompt_tokens: int | None = None
    generation_reserve_tokens: int | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "budget_source": "char_fallback_320",
            "context_limit_tokens": self.context_limit_tokens,
            "base_prompt_tokens": self.base_prompt_tokens,
            "generation_reserve_tokens": self.generation_reserve_tokens,
            "available_memory_tokens": None,
            "token_budget_unavailable_reason": self.reason,
        }


@dataclass
class TokenSelection:
    """The whole-line context selected against a token capacity."""

    context: MemoryContext | None
    block_tokens: int
    omitted_lines: int
    omitted_items: int


def resolve_context_limit(engine: Any) -> tuple[int | None, str | None]:
    """Return the smallest known serving cap (Processor and backend), and its source."""
    bundle = getattr(engine, "bundle", None)
    data_processor = getattr(bundle, "data_processor", None)
    limits: list[tuple[int, str]] = []
    for value, name in (
        (getattr(bundle, "max_seq_len", None), "bundle.max_seq_len"),
        (getattr(data_processor, "model_max_length", None), "processor.model_max_length"),
    ):
        if isinstance(value, int) and value > 0:
            limits.append((value, name))
    backend = getattr(engine, "backend", None)
    if backend == "vllm_local":
        vllm = getattr(engine, "vllm_engine", None)
        model_config = getattr(vllm, "model_config", None)
        if model_config is None:
            model_config = getattr(getattr(vllm, "llm_engine", None), "model_config", None)
        backend_limit = getattr(model_config, "max_model_len", None)
        if not isinstance(backend_limit, int) or backend_limit <= 0:
            # The engine was created with this exact formula; use it only as a
            # fallback when the in-process config cannot be inspected.
            frames = getattr(bundle, "num_history_frames", None)
            if isinstance(frames, int) and frames > 0:
                backend_limit = max(2048, frames * 24 + 1024)
        if isinstance(backend_limit, int) and backend_limit > 0:
            limits.append((backend_limit, "vllm.max_model_len"))
    elif backend == "hf":
        model = getattr(bundle, "model", None)
        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", None)
        backend_limit = (
            getattr(text_config, "max_position_embeddings", None)
            if text_config is not None else None
        )
        if not isinstance(backend_limit, int) or backend_limit <= 0:
            backend_limit = getattr(config, "max_position_embeddings", None)
        if isinstance(backend_limit, int) and backend_limit > 0:
            limits.append((backend_limit, "hf.max_position_embeddings"))
    if not limits:
        return None, None
    value, name = min(limits)
    return value, name


def _prompt_length(processed: dict[str, Any]) -> int | None:
    ids = processed.get("input_ids")
    if ids is None:
        return None
    labels = processed.get("labels")
    try:
        if labels is not None:
            flat_labels = labels.reshape(-1)
            answer = (flat_labels != -100).nonzero(as_tuple=False)
            if answer.numel() > 0:
                return int(answer[0].item())
        return int(ids.reshape(-1).numel())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def measure_memory_token_budget(
    engine: Any, base_sample: dict[str, Any],
) -> MemoryTokenBudget | TokenBudgetFailure:
    """Measure available memory tokens from the real Processor/Tokenizer.

    ``base_sample`` is the unchanged sample for this step (no memory block).
    One Processor pass is used; no vision tower or generation is run.
    """
    bundle = getattr(engine, "bundle", None)
    data_processor = getattr(bundle, "data_processor", None)
    tokenizer = getattr(bundle, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(data_processor, "tokenizer", None)
    if data_processor is None or tokenizer is None:
        return TokenBudgetFailure(
            reason="processor_or_tokenizer_unavailable: the engine exposes no "
                   "data_processor/tokenizer for a model-derived budget",
        )
    context_limit, source = resolve_context_limit(engine)
    if context_limit is None:
        return TokenBudgetFailure(
            reason="context_limit_unavailable: neither the Processor max_seq_len nor "
                   "the backend max_model_len is known",
        )
    reserve = getattr(engine, "max_new_tokens", None)
    if not isinstance(reserve, int) or reserve < 0:
        return TokenBudgetFailure(
            reason="generation_reserve_unavailable: engine.max_new_tokens is not known",
            context_limit_tokens=context_limit,
        )
    try:
        processed = data_processor.process_sample(
            base_sample, add_generation_prompt=True, validate_video_shapes=False,
        )
    except Exception as exc:  # noqa: BLE001 - report, never guess
        return TokenBudgetFailure(
            reason=f"processor_measurement_failed: {type(exc).__name__}: {exc}",
            context_limit_tokens=context_limit,
            generation_reserve_tokens=reserve,
        )
    base_tokens = _prompt_length(processed)
    if base_tokens is None:
        return TokenBudgetFailure(
            reason="processor_measurement_missing_input_ids",
            context_limit_tokens=context_limit,
            generation_reserve_tokens=reserve,
        )
    available = (
        context_limit - base_tokens - reserve - BOUNDARY_RESERVE_TOKENS
    )
    counter = _make_counter(tokenizer)
    if counter is None:
        return TokenBudgetFailure(
            reason="tokenizer_encode_unavailable",
            context_limit_tokens=context_limit,
            base_prompt_tokens=base_tokens,
            generation_reserve_tokens=reserve,
        )
    return MemoryTokenBudget(
        available_tokens=max(0, available),
        context_limit_tokens=context_limit,
        base_prompt_tokens=base_tokens,
        generation_reserve_tokens=reserve,
        safety_margin_tokens=BOUNDARY_RESERVE_TOKENS,
        context_limit_source=source or "unknown",
        counter=counter,
    )


def _make_counter(tokenizer: Any) -> Callable[[str], int] | None:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return None

    def count(text: str) -> int:
        # Count exactly the escaped block that will be inserted plus the newline
        # in inject_memory_context. Special tokens are not part of the block.
        return len(encode(
            escape(text, quote=False) + "\n", add_special_tokens=False,
        ))

    return count


def select_context_lines(
    context: MemoryContext, *, capacity_tokens: int, count_tokens: Callable[[str], int],
) -> TokenSelection:
    """Keep the longest whole-line prefix that fits ``capacity_tokens``.

    Context lines are already priority-ordered by their builder. A line that
    does not fit ends the selection, so no half-line and no later lower-priority
    line can jump over it. Items whose serialized text is not included are
    dropped with their line.
    """
    if capacity_tokens <= 0 or not context.text.strip():
        return TokenSelection(
            context=None, block_tokens=0, omitted_lines=len(context.text.splitlines()),
            omitted_items=len(context.layer_items),
        )
    lines = context.text.splitlines()
    included: list[str] = []
    included_set: set[str] = set()
    omitted = 0
    block_tokens = 0
    for index, line in enumerate(lines):
        candidate = "\n".join([*included, line])
        tokens = count_tokens(candidate)
        if tokens > capacity_tokens:
            omitted = len(lines) - index
            break
        included.append(line)
        included_set.add(escape(line, quote=False))
        block_tokens = tokens
    if not included:
        return TokenSelection(
            context=None, block_tokens=0, omitted_lines=len(lines),
            omitted_items=len(context.layer_items),
        )
    kept_items = [
        item for item in context.layer_items
        if item.metadata.get("serialized_text") in included_set
    ]
    omitted_items = len(context.layer_items) - len(kept_items)
    if context.layer_items and not kept_items:
        return TokenSelection(
            context=None, block_tokens=block_tokens,
            omitted_lines=omitted, omitted_items=omitted_items,
        )
    return TokenSelection(
        context=MemoryContext(text="\n".join(included), layer_items=kept_items),
        block_tokens=block_tokens,
        omitted_lines=omitted,
        omitted_items=omitted_items,
    )


def trim_context_for_overflow(
    context: MemoryContext, *, drop_count: int = 1,
) -> tuple[MemoryContext | None, int]:
    """Drop the lowest-priority item lines after an exact token-delta overflow."""
    remaining_items = list(context.layer_items)
    lines = context.text.splitlines()
    dropped = 0
    for _ in range(drop_count):
        if not remaining_items:
            break
        victim = remaining_items[-1]
        dropped += 1
        remaining_items.pop()
        lines = [line for line in lines if line != victim.description]
    if not remaining_items:
        # Keep a state-only card only if it is short enough to be useful; the
        # caller decides based on the exact delta and may still drop it.
        return MemoryContext(text="\n".join(lines), layer_items=[]), dropped
    return MemoryContext(text="\n".join(lines), layer_items=remaining_items), dropped
