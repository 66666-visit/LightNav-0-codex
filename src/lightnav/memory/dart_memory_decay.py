"""Read-side DART-VLN memory decay for LightNav explicit memory.

DART-VLN reweights each memory slot during readout with a bounded product of
recency, repetition and novelty terms (see the paper and the official
``models/vilmodel.py::_compute_grid_memory_weights`` implementation):

``w = clip(exp(-lambda * age) * (1 - alpha * count / (count + 1))
           * (0.5 + 0.5 * novelty), w_min, w_max)``

LightNav does not expose a differentiable per-cell readout. Its prompt adapter
reads a bounded list of textual memory items, so this module applies the same
weight as a read-side ordering key: items are reordered by descending weight
(and, because the existing character budget drops trailing lines, the weight can
decide which complete item survives that budget). Stored Evidence, point clouds,
object geometry, semantic labels and validation rules are never changed here.

Novelty is only computed when a real, comparable feature vector exists. The
geometry-only L1/L3 records in this project do not carry a learned per-slot
feature, so their novelty value stays at the DART initialization prior (1.0) and
is reported as ``initial_prior`` rather than fabricated from class names, IoU or
geometric distance. V3 place nodes that have a real DINO descriptor are updated
with the exact cosine-EMA formula.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

from lightnav.memory.dart_decay_config import DartDecayConfig
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext


def _unit_feature(values: object, name: str) -> tuple[float, ...]:
    """Validate one real feature vector and return its unit-normalized form."""
    if isinstance(values, bool) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a numeric feature vector")
    try:
        vector = tuple(float(value) for value in values)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric feature vector") from exc
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must be finite and nonempty")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-12:
        raise ValueError(f"{name} must have nonzero norm")
    return tuple(value / norm for value in vector)


def cosine_novelty(old_feature: object, new_feature: object) -> float:
    """Return ``clip(1 - cosine(old, new), 0, 1)`` for comparable features."""
    old = _unit_feature(old_feature, "old feature")
    new = _unit_feature(new_feature, "new feature")
    if len(old) != len(new):
        raise ValueError("novelty features must have the same dimension")
    cosine = sum(left * right for left, right in zip(old, new))
    return float(min(1.0, max(0.0, 1.0 - min(1.0, max(-1.0, cosine)))))


def compute_decay_weight(
    *,
    age: int,
    visit_count: int,
    novelty: float,
    config: DartDecayConfig,
) -> float:
    """Compute the paper's bounded read-side weight without touching memory."""
    if isinstance(age, bool) or not isinstance(age, int) or age < 0:
        raise ValueError("age must be a nonnegative integer")
    if isinstance(visit_count, bool) or not isinstance(visit_count, int) or visit_count < 1:
        raise ValueError("visit_count must be a positive integer")
    if isinstance(novelty, bool) or not isinstance(novelty, (int, float)):
        raise ValueError("novelty must be numeric")
    if not math.isfinite(float(novelty)):
        raise ValueError("novelty must be finite")
    novelty_value = min(1.0, max(0.0, float(novelty)))
    recency = math.exp(-config.decay_lambda * age)
    repeat = 1.0 - config.alpha * (visit_count / (visit_count + 1.0))
    novelty_factor = 0.5 + 0.5 * novelty_value
    return float(min(config.w_max, max(config.w_min, recency * repeat * novelty_factor)))


def evidence_slot_id(evidence_key: str) -> str:
    return f"evidence:{evidence_key}"


def object_slot_id(object_id: str) -> str:
    return f"object:{object_id}"


def place_slot_id(node_id: str) -> str:
    return f"place:{node_id}"


@dataclass
class _SlotState:
    slot_id: str
    slot_type: str
    last_update_step: int
    visit_count: int
    novelty_ema: float = 1.0
    novelty_source: str = "initial_prior"
    feature: tuple[float, ...] | None = None
    novelty_updates: int = 0


@dataclass(frozen=True)
class DecayWeight:
    slot_id: str
    slot_type: str
    memory_age: int
    visit_count: int
    novelty: float
    novelty_source: str
    decay_weight: float


@dataclass(frozen=True)
class DecayReadout:
    context: MemoryContext | None
    diagnostic: dict[str, Any]


def _json_refs(item: LayerMemoryItem) -> list[str]:
    refs: list[str] = []
    metadata = item.metadata if isinstance(item.metadata, dict) else {}
    for key in ("evidence_ids", "supporting_evidence_ids"):
        values = metadata.get(key)
        if isinstance(values, (list, tuple)):
            refs.extend(str(value) for value in values if isinstance(value, str) and value)
    for reference in item.evidence_refs:
        if isinstance(reference.event_id, str) and reference.event_id:
            refs.append(reference.event_id)
    provenance = metadata.get("provenance")
    if isinstance(provenance, (list, tuple)):
        for source in provenance:
            if isinstance(source, dict):
                value = source.get("evidence_id")
                if isinstance(value, str) and value:
                    refs.append(value)
    return list(dict.fromkeys(refs))


def _item_label(item: LayerMemoryItem, index: int) -> str:
    metadata = item.metadata if isinstance(item.metadata, dict) else {}
    memory_id = metadata.get("memory_id")
    if isinstance(memory_id, str) and memory_id:
        return memory_id
    refs = _json_refs(item)
    if refs:
        return refs[0]
    return f"{item.layer}:{item.kind}:{index}"


def _place_node_from_memory_id(memory_id: object) -> str | None:
    if not isinstance(memory_id, str) or not memory_id.startswith("node:"):
        return None
    try:
        parts = json.loads(memory_id.removeprefix("node:"))
    except (TypeError, ValueError):
        return None
    if not isinstance(parts, list) or len(parts) != 2 or not isinstance(parts[1], str):
        return None
    return parts[1]


class DartMemoryDecay:
    """Episode-isolated slot metadata and read-side context reweighting."""

    def __init__(self, config: DartDecayConfig) -> None:
        if not isinstance(config, DartDecayConfig):
            raise ValueError("config must be a DartDecayConfig")
        self.config = config
        self._slots: dict[str, _SlotState] = {}
        self._episode: tuple[str, str, int] | None = None

    @property
    def episode(self) -> tuple[str, str, int] | None:
        return self._episode

    def reset_episode(self, scene_id: str, episode_id: str, rollout_idx: int = 0) -> None:
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("DART decay requires a nonempty Scene identity")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("DART decay requires a nonempty Episode identity")
        if isinstance(rollout_idx, bool) or not isinstance(rollout_idx, int) or rollout_idx < 0:
            raise ValueError("rollout_idx must be a nonnegative integer")
        self._slots.clear()
        self._episode = (scene_id, episode_id, rollout_idx)

    def close_episode(self) -> None:
        self._slots.clear()
        self._episode = None

    def _require_active(self) -> None:
        if self._episode is None:
            raise RuntimeError("DART decay episode is not active")

    def _update_slot(
        self, slot_id: str, slot_type: str, step: int, visit_count: int,
    ) -> _SlotState:
        if not isinstance(slot_id, str) or not slot_id:
            raise ValueError("slot_id must be a nonempty string")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if isinstance(visit_count, bool) or not isinstance(visit_count, int) or visit_count < 1:
            raise ValueError("visit_count must be a positive integer")
        slot = self._slots.get(slot_id)
        if slot is None:
            slot = _SlotState(
                slot_id=slot_id, slot_type=slot_type,
                last_update_step=step, visit_count=visit_count,
            )
            self._slots[slot_id] = slot
        else:
            slot.last_update_step = max(slot.last_update_step, step)
            slot.visit_count = max(slot.visit_count, visit_count)
        return slot

    def observe_evidence(self, slot_key: str, step: int) -> None:
        """Register an observed Evidence slot; its feature is not available."""
        self._require_active()
        self._update_slot(evidence_slot_id(slot_key), "evidence", step, 1)

    def observe_object(self, object_id: str, step: int, *, visit_count: int) -> None:
        """Refresh an L3 object slot without inventing a visual feature."""
        self._require_active()
        self._update_slot(object_slot_id(object_id), "object", step, visit_count)

    def observe_place(
        self, node_id: str, step: int, feature: object | None, *,
        old_feature: object | None = None,
    ) -> None:
        """Refresh a persistent place slot, updating novelty only from real features.

        The visit count is Episode-local: a scene-persistent node can carry a
        large ``observation_count`` from earlier Episodes, which must not
        suppress its first read in the current Episode. ``old_feature`` is the
        persistent node's descriptor before the current refresh (the DART slot
        feature); it is ignored for a new Episode-local slot.
        """
        self._require_active()
        normalized = _unit_feature(feature, "place feature") if feature is not None else None
        slot_id = place_slot_id(node_id)
        existing = self._slots.get(slot_id)
        prior = (
            _unit_feature(old_feature, "old place feature")
            if existing is not None and old_feature is not None else None
        )
        episode_count = 1 if existing is None else existing.visit_count + 1
        slot = self._update_slot(slot_id, "place", step, episode_count)
        if normalized is None:
            return
        if existing is None:
            # A new Episode gets a fresh DART novelty prior; a persistent node
            # from an earlier Episode must not seed this Episode's EMA.
            slot.feature = normalized
            return
        reference = prior if prior is not None else slot.feature
        if reference is None:
            slot.feature = normalized
            return
        novelty = cosine_novelty(reference, normalized)
        slot.novelty_ema = min(
            1.0, max(0.0, self.config.rho * slot.novelty_ema + (1.0 - self.config.rho) * novelty)
        )
        slot.novelty_source = "cosine_ema"
        slot.novelty_updates += 1
        slot.feature = normalized

    def weight_for(self, slot_id: str, step: int) -> DecayWeight | None:
        slot = self._slots.get(slot_id)
        if slot is None:
            return None
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        age = max(0, step - slot.last_update_step)
        novelty = min(1.0, max(0.0, slot.novelty_ema))
        return DecayWeight(
            slot_id=slot.slot_id, slot_type=slot.slot_type, memory_age=age,
            visit_count=max(1, slot.visit_count), novelty=novelty,
            novelty_source=slot.novelty_source,
            decay_weight=compute_decay_weight(
                age=age, visit_count=max(1, slot.visit_count), novelty=novelty,
                config=self.config,
            ),
        )

    def _weights_for_item(self, item: LayerMemoryItem, step: int) -> list[DecayWeight]:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        weights: list[DecayWeight] = []

        object_id = metadata.get("object_id")
        if isinstance(object_id, str):
            weight = self.weight_for(object_slot_id(object_id), step)
            if weight is not None:
                weights.append(weight)

        node_id = _place_node_from_memory_id(metadata.get("memory_id"))
        if node_id is not None:
            weight = self.weight_for(place_slot_id(node_id), step)
            if weight is not None:
                weights.append(weight)

        if not weights:
            for evidence_key in _json_refs(item):
                weight = self.weight_for(evidence_slot_id(evidence_key), step)
                if weight is not None:
                    weights.append(weight)
        return weights

    def _resolve_item(self, item: LayerMemoryItem, step: int) -> DecayWeight | None:
        weights = self._weights_for_item(item, step)
        if not weights:
            return None
        # A multi-evidence statement is only as strong as its weakest cited
        # slot. The paper's softmax operates on individual slots; the prompt
        # item is a whole claim, so use the minimum as a conservative proxy.
        return min(weights, key=lambda weight: (weight.decay_weight, weight.slot_id))

    def reweight_context(
        self, context: MemoryContext | None, step: int,
    ) -> DecayReadout:
        """Return a read-side reordered context and JSON-safe diagnostics."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if self.config.mode == "off":
            return DecayReadout(context, {})
        if context is None:
            return DecayReadout(None, self._diagnostic(step, [], [], {}))
        items = list(context.layer_items)
        if not items:
            return DecayReadout(context, self._diagnostic(step, [], [], {}))

        resolved = [self._resolve_item(item, step) for item in items]
        positions = [index for index, weight in enumerate(resolved) if weight is not None]
        if not positions:
            return DecayReadout(
                context,
                self._diagnostic(step, items, resolved, {"skip_reason": "no_slot_metadata"}),
            )

        lines = context.text.splitlines() if context.text else []
        one_line_per_item = len(lines) == len(items) + 1
        if lines and not one_line_per_item:
            # The weight cannot be applied without desynchronizing the prompt
            # text from ``layer_items``; report every item as skipped instead of
            # claiming a read-side effect that did not happen.
            return DecayReadout(
                context,
                self._diagnostic(
                    step, items, [None] * len(items),
                    {"skip_reason": "prompt_layout_is_not_one_line_per_item"},
                ),
            )

        ordered_positions = sorted(
            positions, key=lambda index: (-resolved[index].decay_weight, index),
        )
        permutation = list(range(len(items)))
        for destination, source in zip(positions, ordered_positions):
            permutation[destination] = source
        order_changed = permutation != list(range(len(items)))

        weighted_items = [items[index] for index in permutation]
        diagnostic = self._diagnostic(step, items, resolved, {
            "order_changed": order_changed,
            "text_reordered": bool(order_changed and one_line_per_item),
            "permutation": permutation,
        })
        if not order_changed:
            return DecayReadout(context, diagnostic)
        if not lines:
            return DecayReadout(
                replace(context, layer_items=weighted_items), diagnostic,
            )
        body = [lines[index + 1] for index in permutation]
        return DecayReadout(
            replace(
                context,
                text="\n".join([lines[0], *body]),
                layer_items=weighted_items,
            ),
            diagnostic,
        )

    def _diagnostic(
        self,
        step: int,
        items: Sequence[LayerMemoryItem],
        resolved: Sequence[DecayWeight | None],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        weights = [weight for weight in resolved if weight is not None]
        item_rows: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            weight = resolved[index] if index < len(resolved) else None
            row: dict[str, Any] = {
                "index": index,
                "item": _item_label(item, index),
                "evidence_ids": _json_refs(item),
                "applied": weight is not None,
                "memory_age": weight.memory_age if weight is not None else None,
                "visit_count": weight.visit_count if weight is not None else None,
                "novelty": weight.novelty if weight is not None else None,
                "novelty_source": weight.novelty_source if weight is not None else None,
                "slot_id": weight.slot_id if weight is not None else None,
                "slot_type": weight.slot_type if weight is not None else None,
                "decay_weight": weight.decay_weight if weight is not None else None,
                "skip_reason": (
                    None if weight is not None
                    else extra.get("skip_reason", "no_slot_metadata")
                ),
            }
            item_rows.append(row)
        before = [_item_label(item, index) for index, item in enumerate(items)]
        after = (
            [before[index] for index in extra["permutation"]]
            if "permutation" in extra else before
        )
        diagnostic = {
            "protocol_version": self.config.protocol_version,
            "mode": self.config.mode,
            "step": step,
            "config": asdict(self.config),
            "decay_applied_count": len(weights),
            "decay_skipped_count": len(items) - len(weights),
            "novelty_measured_count": sum(
                weight.novelty_source == "cosine_ema" for weight in weights
            ),
            "novelty_prior_count": sum(
                weight.novelty_source != "cosine_ema" for weight in weights
            ),
            "order_changed": bool(extra.get("order_changed", False)),
            "text_reordered": bool(extra.get("text_reordered", False)),
            "prompt_order_before": before,
            "prompt_order_after": after,
            "items": item_rows,
        }
        return diagnostic


def decay_weights_for_context(
    decay: DartMemoryDecay, context: MemoryContext | None, step: int,
) -> tuple[MemoryContext | None, dict[str, Any]]:
    """Convenience wrapper used by the policy tests and the manager."""
    readout = decay.reweight_context(context, step)
    return readout.context, readout.diagnostic


__all__ = [
    "DecayReadout",
    "DecayWeight",
    "DartMemoryDecay",
    "compute_decay_weight",
    "cosine_novelty",
    "decay_weights_for_context",
    "evidence_slot_id",
    "object_slot_id",
    "place_slot_id",
]
