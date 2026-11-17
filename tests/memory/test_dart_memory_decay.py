"""Unit tests for the DART-VLN read-side decay adaptation.

These are CPU code tests. They do not evaluate navigation quality and do not
replace a real Habitat/AutoDL run.
"""

from __future__ import annotations

import math
from copy import deepcopy

import pytest

from lightnav.memory.dart_decay_config import DartDecayConfig
from lightnav.memory.dart_memory_decay import (
    DartMemoryDecay,
    compute_decay_weight,
    cosine_novelty,
    evidence_slot_id,
    place_slot_id,
)
from lightnav.memory.memory_context_builder import build_context_text
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence
from lightnav.memory.v3_schemas import EvidenceRecord

_HEADER = "[Navigation Memory]"


def _item(
    name: str,
    *,
    step: int = 0,
    memory_id: str | None = None,
    object_id: str | None = None,
) -> LayerMemoryItem:
    metadata = {"evidence_ids": [name], "serialized_text": name}
    if memory_id is not None:
        metadata["memory_id"] = memory_id
    if object_id is not None:
        metadata["object_id"] = object_id
    return LayerMemoryItem(
        layer="short", kind="observation", description=name, step_id=step,
        confidence=0.9, source="test",
        evidence_refs=[ObservationEvidence(
            event_id=name, step_id=step, label=name, confidence=0.9, source="test",
        )],
        metadata=metadata,
    )


def _context(*lines: str) -> MemoryContext:
    return MemoryContext(
        text=_HEADER + "\n" + "\n".join(lines),
        layer_items=[_item(line) for line in lines],
    )


def _decay(**kwargs) -> DartMemoryDecay:
    config = DartDecayConfig(mode="decay_only", **kwargs)
    decay = DartMemoryDecay(config)
    decay.reset_episode("scene", "episode", 0)
    return decay


def test_paper_weight_formula_is_exact() -> None:
    config = DartDecayConfig(mode="decay_only")
    novelty = 0.4
    expected = (
        math.exp(-0.12 * 3.0)
        * (1.0 - 0.15 * (4.0 / 5.0))
        * (0.5 + 0.5 * novelty)
    )
    expected = min(config.w_max, max(config.w_min, expected))
    assert compute_decay_weight(
        age=3, visit_count=4, novelty=novelty, config=config,
    ) == pytest.approx(expected)


def test_age_monotonically_decreases_weight_until_floor() -> None:
    config = DartDecayConfig(mode="decay_only")
    weights = [
        compute_decay_weight(age=age, visit_count=1, novelty=1.0, config=config)
        for age in range(0, 20)
    ]
    assert all(left >= right for left, right in zip(weights, weights[1:]))
    assert weights[0] > weights[5] > weights[-1]
    assert weights[-1] == pytest.approx(config.w_min)


def test_visit_count_monotonically_decreases_weight() -> None:
    config = DartDecayConfig(mode="decay_only")
    weights = [
        compute_decay_weight(age=0, visit_count=count, novelty=1.0, config=config)
        for count in range(1, 12)
    ]
    assert all(left >= right for left, right in zip(weights, weights[1:]))
    assert weights[0] == pytest.approx(0.925)
    assert weights[-1] < weights[0]


def test_novelty_monotonically_increases_weight() -> None:
    config = DartDecayConfig(mode="decay_only")
    weights = [
        compute_decay_weight(age=0, visit_count=1, novelty=novelty, config=config)
        for novelty in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert all(left <= right for left, right in zip(weights, weights[1:]))
    assert weights[0] == pytest.approx(0.4625)
    assert weights[-1] == pytest.approx(0.925)


def test_weight_lower_and_upper_bounds_are_clamped() -> None:
    lower = DartDecayConfig(mode="decay_only", w_min=0.35, w_max=1.0)
    assert compute_decay_weight(
        age=10_000, visit_count=10_000, novelty=0.0, config=lower,
    ) == pytest.approx(0.35)
    upper = DartDecayConfig(mode="decay_only", w_min=0.35, w_max=0.8)
    assert compute_decay_weight(
        age=0, visit_count=1, novelty=1.0, config=upper,
    ) == pytest.approx(0.8)


def test_novelty_is_real_cosine_and_rejects_incomparable_features() -> None:
    assert cosine_novelty((1.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert cosine_novelty((1.0, 0.0), (2.0, 0.0)) == pytest.approx(0.0)
    assert cosine_novelty((1.0, 0.0), (0.0, 1.0)) == pytest.approx(1.0)
    assert cosine_novelty((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="same dimension"):
        cosine_novelty((1.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="nonzero norm"):
        cosine_novelty((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        cosine_novelty((float("nan"), 0.0), (1.0, 0.0))
    with pytest.raises(ValueError, match="numeric feature"):
        cosine_novelty("chair", (1.0, 0.0))


def test_missing_geometry_feature_keeps_dart_initial_prior_not_a_proxy() -> None:
    decay = _decay()
    decay.observe_evidence("object_evidence", step=0)
    weight = decay.weight_for(evidence_slot_id("object_evidence"), step=0)
    assert weight is not None
    assert weight.novelty == pytest.approx(1.0)
    assert weight.novelty_source == "initial_prior"
    # Class labels, IoU and geometric distance are not part of this interface.
    with pytest.raises(TypeError):
        decay.observe_evidence("object_evidence", step=0, iou=0.9)  # type: ignore[call-arg]


def test_place_novelty_ema_uses_real_feature_reobservation() -> None:
    decay = _decay(rho=0.5)
    decay.observe_place("place_0000", step=0, feature=(1.0, 0.0))
    first = decay.weight_for(place_slot_id("place_0000"), step=0)
    assert first is not None and first.novelty == pytest.approx(1.0)
    assert first.novelty_source == "initial_prior"

    decay.observe_place("place_0000", step=1, feature=(1.0, 0.0))
    second = decay.weight_for(place_slot_id("place_0000"), step=1)
    assert second is not None
    assert second.novelty == pytest.approx(0.5)
    assert second.novelty_source == "cosine_ema"
    assert second.visit_count == 2

    decay.observe_place("place_0000", step=2, feature=(1.0, 0.0))
    third = decay.weight_for(place_slot_id("place_0000"), step=2)
    assert third is not None
    assert third.novelty == pytest.approx(0.25)
    assert third.decay_weight < second.decay_weight < first.decay_weight


def test_context_is_reordered_by_decay_weight_without_mutating_source() -> None:
    decay = _decay()
    decay.observe_evidence("stale", step=0)
    decay.observe_evidence("fresh", step=10)
    context = _context("stale", "fresh")
    original_text = context.text
    original_items = list(context.layer_items)

    reweighted = decay.reweight_context(context, step=10)
    assert reweighted.context is not None
    assert [item.description for item in reweighted.context.layer_items] == ["fresh", "stale"]
    assert reweighted.context.text.splitlines()[1:] == ["fresh", "stale"]
    assert context.text == original_text
    assert context.layer_items == original_items
    diagnostic = reweighted.diagnostic
    assert diagnostic["decay_applied_count"] == 2
    assert diagnostic["decay_skipped_count"] == 0
    assert diagnostic["order_changed"] is True
    assert diagnostic["text_reordered"] is True
    rows = {row["item"]: row for row in diagnostic["items"]}
    assert rows["fresh"]["memory_age"] == 0
    assert rows["stale"]["memory_age"] == 10
    assert rows["fresh"]["decay_weight"] > rows["stale"]["decay_weight"]
    for row in diagnostic["items"]:
        assert {
            "memory_age", "visit_count", "novelty", "decay_weight",
        } <= set(row)
    assert {"decay_applied_count", "decay_skipped_count"} <= set(diagnostic)


def test_weight_order_changes_which_line_survives_prompt_budget() -> None:
    decay = _decay()
    decay.observe_evidence("stale_" + "s" * 120, step=0)
    decay.observe_evidence("fresh_" + "f" * 120, step=10)
    stale = "stale_" + "s" * 120
    fresh = "fresh_" + "f" * 120
    context = _context(stale, fresh)

    # The original order would inject the stale line first and drop the fresh
    # one under this budget. Decay reorders the same complete lines.
    budget = len(_HEADER) + 1 + len(fresh) + 1
    reweighted = decay.reweight_context(context, step=10).context
    assert reweighted is not None
    injected = build_context_text(reweighted, max_chars=budget)
    assert fresh in injected
    assert stale not in injected


def test_off_mode_is_an_exact_noop() -> None:
    decay = DartMemoryDecay(DartDecayConfig())
    context = _context("first", "second")
    readout = decay.reweight_context(context, step=3)
    assert readout.context is context
    assert readout.diagnostic == {}
    assert decay.episode is None


def test_episode_reset_isolates_slots() -> None:
    decay = _decay()
    decay.observe_evidence("old", step=0)
    assert decay.weight_for(evidence_slot_id("old"), step=0) is not None
    decay.reset_episode("scene", "episode_two", 0)
    assert decay.weight_for(evidence_slot_id("old"), step=0) is None
    context = _context("old")
    readout = decay.reweight_context(context, step=1)
    assert readout.context is context
    assert readout.diagnostic["decay_applied_count"] == 0
    assert readout.diagnostic["decay_skipped_count"] == 1
    assert readout.diagnostic["items"][0]["skip_reason"] == "no_slot_metadata"
    decay.close_episode()
    with pytest.raises(RuntimeError, match="not active"):
        decay.observe_evidence("after_close", step=0)


def test_non_one_to_one_prompt_layout_is_reported_as_skipped() -> None:
    decay = _decay()
    decay.observe_evidence("first", step=0)
    decay.observe_evidence("second", step=0)
    context = MemoryContext(
        text=_HEADER + "\nfirst and second on one line",
        layer_items=[_item("first"), _item("second")],
    )
    readout = decay.reweight_context(context, step=0)
    assert readout.context is context
    assert readout.diagnostic["decay_applied_count"] == 0
    assert readout.diagnostic["decay_skipped_count"] == 2
    assert all(
        row["skip_reason"] == "prompt_layout_is_not_one_line_per_item"
        for row in readout.diagnostic["items"]
    )


def test_multi_evidence_item_uses_the_weakest_slot() -> None:
    decay = _decay()
    decay.observe_evidence("fresh", step=9)
    decay.observe_evidence("stale", step=0)
    item = _item("combined")
    item.metadata["evidence_ids"] = ["fresh", "stale"]
    context = MemoryContext(text=_HEADER + "\ncombined", layer_items=[item])
    readout = decay.reweight_context(context, step=10)
    assert readout.context is not None
    row = readout.diagnostic["items"][0]
    assert row["memory_age"] == 10
    assert row["decay_weight"] == pytest.approx(0.35)


def test_diagnostics_are_json_safe_and_do_not_touch_item_metadata() -> None:
    import json

    decay = _decay()
    decay.observe_evidence("first", step=0)
    item = _item("first")
    before = deepcopy(item.to_dict())
    context = MemoryContext(text=_HEADER + "\nfirst", layer_items=[item])
    readout = decay.reweight_context(context, step=2)
    json.dumps(readout.diagnostic)
    assert item.to_dict() == before
    assert "decay_weight" not in item.metadata


@pytest.mark.parametrize("kwargs", [
    {"mode": "on"}, {"mode": "update_only"}, {"mode": "full"},
    {"rho": -0.1}, {"rho": 1.1}, {"decay_lambda": -1.0},
    {"alpha": -0.1}, {"alpha": 1.1}, {"w_min": 0.0},
    {"w_min": 0.9, "w_max": 0.5}, {"w_max": 1.1},
])
def test_invalid_decay_config_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DartDecayConfig(**kwargs)


def _evidence(
    event_id: str, *, step: int, label: str | None = None,
    kind: str = "object_observation",
    descriptor: tuple[float, ...] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        event_id, "scene", "episode", 0, step, kind,
        semantic_label=label,
        visual_descriptor=descriptor,
        raw_detector_confidence=0.9 if label else None,
        detector_name="yolo" if label else None,
        observation_status="observed",
    )


def test_local_graph_history_selection_uses_decay_weight() -> None:
    from lightnav.memory.local_graph_config import LocalGraphConfig
    from lightnav.memory.local_graph_fusion import LocalGraphFusion
    from lightnav.memory.v3_store import V3MemoryStore

    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the door or table")
    place_record = store.add(_evidence(
        "place", step=0, kind="place_observation", descriptor=(1.0, 0.0),
    ))
    place = store.add_place(place_record)
    door = store.add(_evidence("door", step=0, label="door"))
    door_node = store.add_landmark(door)
    table = store.add(_evidence("table", step=5, label="table"))
    table_node = store.add_landmark(table)
    store.add_relation(
        place.node_id, door_node.node_id, "inferred_relation",
        evidence_keys=(place_record.key, door.key),
    )
    store.add_relation(
        place.node_id, table_node.node_id, "inferred_relation",
        evidence_keys=(place_record.key, table.key),
    )

    fusion = LocalGraphFusion(LocalGraphConfig(enabled=True, max_landmarks=1))
    fusion.reset_episode("scene", "episode", 0)
    common = dict(
        store=store, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=6, instruction="find the door or table", current_place_id=place.node_id,
        rgb_width=300, events=[],
    )
    without_weights = fusion.build(**common)
    assert without_weights is not None
    assert without_weights.layer_items[0].label == "table"

    fusion.reset_episode("scene", "episode", 0)
    with_weights = fusion.build(
        **common,
        weight_lookup=lambda key, step: 0.9 if "door" in key else 0.4,
    )
    assert with_weights is not None
    assert with_weights.layer_items[0].label == "door"


def test_slot_metadata_age_and_visit_count_change_weight() -> None:
    decay = _decay()
    decay.observe_object("object-1", step=0, visit_count=1)
    fresh = decay.weight_for("object:object-1", step=0)
    older = decay.weight_for("object:object-1", step=5)
    assert fresh is not None and older is not None
    assert older.memory_age == 5
    assert older.decay_weight < fresh.decay_weight

    decay.observe_object("object-2", step=10, visit_count=1)
    once = decay.weight_for("object:object-2", step=10)
    repeated = decay.weight_for("object:object-2", step=10)
    decay.observe_object("object-2", step=10, visit_count=6)
    many = decay.weight_for("object:object-2", step=10)
    assert once is not None and repeated is not None and many is not None
    assert many.visit_count == 6
    assert many.decay_weight < repeated.decay_weight


def test_place_novelty_uses_persistent_node_feature_when_supplied() -> None:
    decay = _decay(rho=0.0)
    decay.observe_place("place_0000", step=0, feature=(0.0, 1.0))
    # The slot's last raw observation is (0, 1). The persistent node's stored
    # feature before this refresh is (1, 0), which must be the reference.
    decay.observe_place(
        "place_0000", step=1, feature=(0.0, 1.0),
        old_feature=(1.0, 0.0),
    )
    weight = decay.weight_for(place_slot_id("place_0000"), step=1)
    assert weight is not None
    assert weight.novelty == pytest.approx(1.0)
    assert weight.novelty_source == "cosine_ema"


def test_first_episode_observation_ignores_persistent_node_old_feature() -> None:
    decay = _decay(rho=0.0)
    decay.observe_place(
        "place_0000", step=0, feature=(0.0, 1.0),
        old_feature=(1.0, 0.0),
    )
    weight = decay.weight_for(place_slot_id("place_0000"), step=0)
    assert weight is not None
    assert weight.novelty == pytest.approx(1.0)
    assert weight.novelty_source == "initial_prior"


def test_mixed_context_reports_applied_and_skipped_counts() -> None:
    decay = _decay()
    decay.observe_evidence("known", step=0)
    context = _context("known", "unknown")
    readout = decay.reweight_context(context, step=0)
    assert readout.diagnostic["decay_applied_count"] == 1
    assert readout.diagnostic["decay_skipped_count"] == 1
    rows = {row["item"]: row for row in readout.diagnostic["items"]}
    assert rows["known"]["applied"] is True
    assert rows["unknown"]["applied"] is False
    assert rows["unknown"]["skip_reason"] == "no_slot_metadata"


def test_place_visit_count_restarts_in_a_new_episode() -> None:
    decay = _decay()
    decay.observe_place("place_0000", step=0, feature=(1.0, 0.0))
    decay.observe_place("place_0000", step=1, feature=(1.0, 0.0))
    before = decay.weight_for(place_slot_id("place_0000"), step=1)
    assert before is not None and before.visit_count == 2

    decay.reset_episode("scene", "episode_two", 0)
    decay.observe_place("place_0000", step=0, feature=(1.0, 0.0))
    after = decay.weight_for(place_slot_id("place_0000"), step=0)
    assert after is not None
    assert after.visit_count == 1
    assert after.novelty_source == "initial_prior"
