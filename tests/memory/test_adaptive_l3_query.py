"""CPU contracts for the Adaptive L3 Query sidecar (no model, no GPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lightnav.memory.adaptive_l3_query import (
    ABSTAIN,
    NO_QUERY,
    QUERY_L3,
    USE,
    AdaptiveL3QueryRunner,
    AdaptiveQueryDecision,
    LocalLightNavQueryController,
    MockAdaptiveQueryController,
    build_l1_snapshot,
    format_verified_facts,
    validate_l3_fact,
)
from lightnav.memory.adaptive_l3_query_config import AdaptiveL3QueryConfig
from lightnav.memory.conceptgraphs_lite import (
    ObjectMap,
    ObjectNode,
    ObservedFromEdge,
    Pose3D,
    SpatialEvidence,
    ViewpointNode,
)
from lightnav.memory.l3_clip_query import L3ClipQuery
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence
from lightnav.memory.v3_schemas import canonical_evidence_key

SCENE = "scene"
EPISODE = "episode"


class _TextEncoder:
    def __init__(self, feature=(1.0, 0.0)) -> None:
        self.feature = np.asarray(feature, dtype=np.float32)
        self.text_calls = 0

    def encode_image(self, rgb, boxes):
        raise AssertionError("fixtures attach image features directly")

    def encode_text(self, texts):
        self.text_calls += 1
        return np.asarray([self.feature], dtype=np.float32)


def _pose(x: float = 0.0) -> Pose3D:
    return Pose3D((x, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _add_node(
    object_map: ObjectMap, index: int, label: str, feature, *, steps=(1, 2),
    broken_evidence: bool = False, missing_evidence: bool = False,
) -> ObjectNode:
    object_id = json.dumps(
        [SCENE, EPISODE, 0, "object", index], separators=(",", ":"),
    )
    evidence_ids: list[str] = []
    for position, step in enumerate(steps):
        evidence_id = canonical_evidence_key(
            SCENE, EPISODE, 0, f"{label}_{index}_{position}",
        )
        evidence_ids.append(evidence_id)
        viewpoint_id = json.dumps(
            [SCENE, EPISODE, 0, step], separators=(",", ":"),
        )
        object_map.viewpoints[viewpoint_id] = ViewpointNode(
            viewpoint_id, SCENE, EPISODE, 0, step, _pose(), _pose(),
        )
        object_map.edges.append(ObservedFromEdge(
            object_id, viewpoint_id, evidence_id,
        ))
        if not missing_evidence:
            object_map.evidence[evidence_id] = SpatialEvidence(
                evidence_id, SCENE, EPISODE, 0, step, label, 0.9,
                "yolo11n-seg", 20, 8, "rejected_depth" if broken_evidence else "observed",
            )
    node = ObjectNode(
        object_id, label, (2.0, 0.0, -2.0),
        ((1.5, -0.5, -2.5), (2.5, 0.5, -1.5)),
        len(steps), evidence_ids, "observed", np.zeros((5, 3)),
    )
    object_map.objects[object_id] = node
    object_map.attach_clip_features({
        evidence_id: np.asarray(feature, dtype=np.float32)
        for evidence_id in evidence_ids
    })
    return node


def _map() -> ObjectMap:
    object_map = ObjectMap()
    object_map.reset_episode(SCENE, EPISODE, 0)
    _add_node(object_map, 1, "sofa", feature=(1.0, 0.0), steps=(1, 2))
    _add_node(object_map, 2, "table", feature=(0.6, 0.8), steps=(3, 4))
    return object_map


def _l1_context() -> MemoryContext:
    evidence_id = canonical_evidence_key(SCENE, EPISODE, 0, "l1_chair_0")
    line = "Current RGB: instruction landmark chair (detected; passage unknown)."
    return MemoryContext(
        text=f"[Navigation Memory]\n{line}",
        layer_items=[LayerMemoryItem(
            layer="short", kind="local_graph_current_landmark", description=line,
            step_id=0, confidence=0.9, source="local_graph_fusion",
            evidence_refs=[ObservationEvidence(
                event_id=evidence_id, step_id=0, label="chair", confidence=0.9,
                source="yolo11",
            )],
            label="chair",
            metadata={"evidence_ids": [evidence_id], "serialized_text": line},
        )],
    )


def _runner(controller, encoder=None, **config):
    encoder = encoder or _TextEncoder()
    return AdaptiveL3QueryRunner(
        config=AdaptiveL3QueryConfig(enabled=True, **config),
        controller=controller,
        encoder=encoder,
        top_k=3,
    )


def _query(controller, runner, object_map, *, step=10, l1_context=None):
    return runner.run(
        object_map=object_map, step=step, instruction="find the sofa",
        l1_status="built" if l1_context is not None else "no_candidates",
        l1_context=l1_context, agent_pose=_pose(), max_chars=320,
    )


def test_l1_sufficient_controller_says_no_query_and_clip_is_never_called() -> None:
    encoder = _TextEncoder()
    controller = MockAdaptiveQueryController(
        decide_script=[AdaptiveQueryDecision(NO_QUERY, reason_code="sufficient")],
    )
    runner = _runner(controller, encoder)
    l1_context = _l1_context()
    context = _query(controller, runner, _map(), l1_context=l1_context)

    assert context is None
    assert encoder.text_calls == 0
    assert controller.calls == [("decide", 10)]
    assert runner.last_diagnostic["skipped_reason"] == "controller_no_query"
    assert runner.last_diagnostic["query_executed"] is False


def test_l1_has_candidates_but_controller_can_still_query_l3() -> None:
    encoder = _TextEncoder()
    object_map = _map()
    sofa = object_map.objects[
        json.dumps([SCENE, EPISODE, 0, "object", 1], separators=(",", ":"))
    ]
    controller = MockAdaptiveQueryController(
        decide_script=[AdaptiveQueryDecision(QUERY_L3, query_text="sofa")],
        select_script=[AdaptiveQueryDecision(USE, object_ids=(sofa.object_id,))],
    )
    runner = _runner(controller, encoder)
    context = _query(controller, runner, object_map, l1_context=_l1_context())

    assert context is not None
    assert encoder.text_calls == 1
    assert runner.last_diagnostic["l1_status"] == "built"
    assert runner.last_diagnostic["query_executed"] is True
    assert runner.last_diagnostic["candidate_count"] == 2
    assert runner.last_diagnostic["validated_object_ids"] == [sofa.object_id]
    assert "Verified L3 history" in context.text


def test_no_valid_clip_candidate_abstains_without_touching_l1() -> None:
    encoder = _TextEncoder(feature=(-1.0, 0.0))
    object_map = _map()
    controller = MockAdaptiveQueryController(
        decide_script=[AdaptiveQueryDecision(QUERY_L3, query_text="sofa")],
    )
    runner = _runner(controller, encoder)
    l1_context = _l1_context()
    context = _query(controller, runner, object_map, l1_context=l1_context)

    assert context is None
    assert runner.last_diagnostic["candidate_count"] == 0
    assert runner.last_diagnostic["skipped_reason"] == "no_similarity_above_floor"
    assert controller.calls == [("decide", 10)]


def test_multiple_candidates_only_evidence_backed_objects_are_selected() -> None:
    object_map = _map()
    valid_id = json.dumps([SCENE, EPISODE, 0, "object", 1], separators=(",", ":"))
    invalid_id = json.dumps([SCENE, EPISODE, 0, "object", 2], separators=(",", ":"))
    # The table's Evidence records exist but its geometry was rejected, so its
    # fused centre cannot be presented as a verified historical fact.
    for evidence_id in object_map.objects[invalid_id].evidence_ids:
        object_map.evidence[evidence_id] = SpatialEvidence(
            evidence_id, SCENE, EPISODE, 0, 3, "table", 0.9,
            "yolo11n-seg", 20, 0, "rejected_depth",
        )
    controller = MockAdaptiveQueryController(
        decide_script=[AdaptiveQueryDecision(QUERY_L3, query_text="object")],
        select_script=[AdaptiveQueryDecision(
            USE, object_ids=(invalid_id, valid_id),
        )],
    )
    runner = _runner(controller)
    context = _query(controller, runner, object_map, l1_context=_l1_context())

    assert context is not None
    assert len(context.layer_items) == 1
    assert context.layer_items[0].metadata["object_id"] == valid_id
    assert "table" not in context.text
    assert runner.last_diagnostic["rejected"] == [{
        "object_id": invalid_id, "reason": "evidence_geometry_rejected_depth",
    }]


def test_same_class_different_object_ids_are_never_merged() -> None:
    object_map = ObjectMap()
    object_map.reset_episode(SCENE, EPISODE, 0)
    first = _add_node(object_map, 1, "sofa", feature=(1.0, 0.0), steps=(1, 2))
    second = _add_node(object_map, 2, "sofa", feature=(0.8, 0.6), steps=(3, 4))
    controller = MockAdaptiveQueryController(
        decide_script=[AdaptiveQueryDecision(QUERY_L3, query_text="sofa")],
        select_script=[AdaptiveQueryDecision(USE, object_ids=(second.object_id,))],
    )
    context = _query(controller, _runner(controller), object_map, l1_context=_l1_context())

    assert context is not None
    assert len(context.layer_items) == 1
    item = context.layer_items[0]
    assert item.metadata["object_id"] == second.object_id
    assert item.metadata["object_id"] != first.object_id
    assert item.metadata["evidence_ids"] == second.evidence_ids
    assert first.object_id not in context.text


def test_object_without_evidence_is_rejected_even_if_clip_matches() -> None:
    object_map = ObjectMap()
    object_map.reset_episode(SCENE, EPISODE, 0)
    node = _add_node(
        object_map, 1, "sofa", feature=(1.0, 0.0), steps=(1, 2),
        missing_evidence=True,
    )
    # Give the node a visual feature even though its Evidence records are absent.
    node.clip_feature = np.asarray([1.0, 0.0], dtype=np.float32)
    node.clip_feature_observations = 1
    controller = MockAdaptiveQueryController(
        decide_script=[AdaptiveQueryDecision(QUERY_L3, query_text="sofa")],
        select_script=[AdaptiveQueryDecision(USE, object_ids=(node.object_id,))],
    )
    runner = _runner(controller)
    context = _query(controller, runner, object_map, l1_context=_l1_context())

    assert context is None
    assert runner.last_diagnostic["skipped_reason"] == "no_verified_fact"
    assert runner.last_diagnostic["rejected"] == [{
        "object_id": node.object_id, "reason": "evidence_missing",
    }]


def test_query_failure_reports_an_error_and_injects_nothing() -> None:
    class _Broken(_TextEncoder):
        def encode_text(self, texts):
            raise RuntimeError("local CLIP failed")

    controller = MockAdaptiveQueryController(
        decide_script=[AdaptiveQueryDecision(QUERY_L3, query_text="sofa")],
    )
    l1_context = _l1_context()
    runner = _runner(controller, _Broken())
    context = _query(controller, runner, _map(), l1_context=l1_context)

    assert context is None
    assert runner.last_diagnostic["skipped_reason"] == "query_error"
    assert "local CLIP failed" in runner.last_diagnostic["error"]


def test_injected_prompt_exposes_real_object_and_evidence_ids() -> None:
    object_map = _map()
    node = object_map.objects[
        json.dumps([SCENE, EPISODE, 0, "object", 1], separators=(",", ":"))
    ]
    controller = MockAdaptiveQueryController(
        decide_script=[AdaptiveQueryDecision(QUERY_L3, query_text="sofa")],
        select_script=[AdaptiveQueryDecision(USE, object_ids=(node.object_id,))],
    )
    context = _query(controller, _runner(controller), object_map, l1_context=_l1_context())

    assert context is not None
    item = context.layer_items[0]
    assert item.metadata["object_id"] == node.object_id
    assert item.metadata["evidence_ids"] == node.evidence_ids
    assert [ref.event_id for ref in item.evidence_refs] == node.evidence_ids
    assert "object:1" in item.description
    assert "evidence:sofa_1_1" in item.description
    assert item.metadata["serialized_text"] in context.text
    assert len(context.text) <= 320


def test_episode_reset_clears_controller_and_runner_state() -> None:
    controller = MockAdaptiveQueryController(
        decide_script=[
            AdaptiveQueryDecision(QUERY_L3, query_text="sofa"),
            AdaptiveQueryDecision(QUERY_L3, query_text="sofa"),
        ],
        select_script=[AdaptiveQueryDecision(ABSTAIN, reason_code="controller_abstain")],
    )
    runner = _runner(controller)
    _query(controller, runner, _map())
    assert controller.calls
    assert runner.last_diagnostic

    runner.reset_episode(SCENE, EPISODE, 0)
    assert controller.calls == []
    assert runner.last_diagnostic == {}

    # The script restarts from its first decision after reset.
    _query(controller, runner, _map())
    assert controller.calls == [("decide", 10), ("select", 10)]
    assert runner.last_diagnostic["skipped_reason"] == "controller_abstain"


def test_unchanged_map_query_duplicate_is_not_sent_again() -> None:
    encoder = _TextEncoder()
    controller = MockAdaptiveQueryController(
        decide_script=[
            AdaptiveQueryDecision(QUERY_L3, query_text="sofa"),
            AdaptiveQueryDecision(QUERY_L3, query_text="sofa"),
        ],
        select_script=[AdaptiveQueryDecision(ABSTAIN), AdaptiveQueryDecision(ABSTAIN)],
    )
    runner = _runner(controller, encoder)
    object_map = _map()
    _query(controller, runner, object_map, step=10)
    _query(controller, runner, object_map, step=11)

    assert encoder.text_calls == 1
    assert runner.last_diagnostic["skipped_reason"] == (
        "duplicate_query_for_unchanged_map"
    )
    assert runner.last_diagnostic["steps_since_last_query"] == 1


def test_validator_rejects_current_step_and_missing_edge_evidence() -> None:
    object_map = _map()
    result = L3ClipQuery(object_map, _TextEncoder()).query("sofa", 1)
    match = result.matches[0]
    node = object_map.objects[match.object_id]

    fact, reason = validate_l3_fact(
        object_map, match, current_step=2, min_confidence=0.5,
    )
    assert fact is None
    assert reason == "evidence_not_historical"

    # Removing one observed_from edge makes the node's provenance incomplete.
    object_map.edges = [
        edge for edge in object_map.edges
        if not (edge.evidence_id == node.evidence_ids[-1])
    ]
    fact, reason = validate_l3_fact(
        object_map, match, current_step=10, min_confidence=0.5,
    )
    assert fact is None
    assert reason == "observed_from_edge_missing"


def test_formatter_is_deterministic_and_budgeted() -> None:
    object_map = _map()
    result = L3ClipQuery(object_map, _TextEncoder()).query("sofa", 1)
    fact, reason = validate_l3_fact(
        object_map, result.matches[0], current_step=10, min_confidence=0.5,
    )
    assert fact is not None and reason is None
    first = format_verified_facts([fact], agent_pose=_pose(), max_chars=320)
    second = format_verified_facts([fact], agent_pose=_pose(), max_chars=320)
    assert first is not None and second is not None
    assert first.text == second.text
    assert first.layer_items[0].metadata["serialized_text"] == (
        second.layer_items[0].metadata["serialized_text"]
    )
    assert format_verified_facts([fact], agent_pose=_pose(), max_chars=10) is None


def test_l1_snapshot_reads_only_existing_l1_items() -> None:
    context = _l1_context()
    snapshot = build_l1_snapshot(
        instruction="find the chair", step=4, l1_status="built", context=context,
    )
    assert snapshot.has_l1_items is True
    assert snapshot.labels == ("chair",)
    assert snapshot.evidence_ids == tuple(
        context.layer_items[0].metadata["evidence_ids"]
    )
    assert snapshot.l1_text == context.text


def test_default_config_is_off_and_rejects_invalid_bounds() -> None:
    assert AdaptiveL3QueryConfig().enabled is False
    with pytest.raises(ValueError, match="controller_max_new_tokens"):
        AdaptiveL3QueryConfig(controller_max_new_tokens=0)
    with pytest.raises(ValueError, match="min_evidence_confidence"):
        AdaptiveL3QueryConfig(min_evidence_confidence=1.5)


class _FakeEngine:
    def __init__(self, outputs: list[str]) -> None:
        from types import SimpleNamespace

        self.outputs = list(outputs)
        self.bundle = SimpleNamespace(
            video_size=(4, 4), video_fps=4, num_history_frames=2,
            pool_enable=False, pool_spatial=1, pool_mode="avg",
            slowfast_tiers=None, predict_horizon=1, action_method="flat",
        )
        self.prompts: list[str] = []
        self.samples: list[dict] = []

    def generate(self, sample, max_new_tokens=None):
        self.samples.append(sample)
        self.prompts.append(sample["conversations"][0]["value"])
        if not self.outputs:
            raise RuntimeError("no scripted output")
        return self.outputs.pop(0), 1.0


def _local_controller(outputs: list[str]) -> tuple[LocalLightNavQueryController, _FakeEngine]:
    import torch

    engine = _FakeEngine(outputs)
    video = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
    controller = LocalLightNavQueryController(
        engine, lambda: (video, [0, 1]), max_new_tokens=8,
    )
    return controller, engine


def test_local_controller_uses_the_existing_engine_and_parses_strictly() -> None:
    controller, engine = _local_controller([
        "QUERY_L3: sofa", "USE C1",
    ])
    snapshot = build_l1_snapshot(
        instruction="find the sofa", step=3, l1_status="built",
        context=_l1_context(),
    )
    decision = controller.decide(snapshot)
    assert decision.decision == QUERY_L3
    assert decision.query_text == "sofa"
    prompt = engine.prompts[0]
    assert "find the sofa" in prompt
    sample = engine.samples[0]
    assert prompt.count("<video>") == len(sample["video_segments"])

    object_map = _map()
    result = L3ClipQuery(object_map, _TextEncoder()).query("sofa", 2)
    selection = controller.select(snapshot, result.matches)
    assert selection.decision == USE
    assert selection.object_ids == (result.matches[0].object_id,)
    assert "C1" in engine.prompts[1]


def test_local_controller_fails_closed_on_garbage_and_errors() -> None:
    controller, _ = _local_controller(["<traj_1> nonsense", "garbage"])
    snapshot = build_l1_snapshot(
        instruction="find the sofa", step=3, l1_status="built", context=None,
    )
    decision = controller.decide(snapshot)
    assert decision.decision == NO_QUERY
    assert decision.reason_code == "unparseable_decision"

    object_map = _map()
    result = L3ClipQuery(object_map, _TextEncoder()).query("sofa", 2)
    selection = controller.select(snapshot, result.matches)
    assert selection.decision == ABSTAIN

    broken, _ = _local_controller([])
    decision = broken.decide(snapshot)
    assert decision.decision == NO_QUERY
    assert decision.reason_code == "controller_error"


def test_local_controller_refuses_trajectory_only_decoding(monkeypatch) -> None:
    import torch

    monkeypatch.setenv("VLN_EVAL_TRAJ_TOP1", "1")
    with pytest.raises(ValueError, match="VLN_EVAL_TRAJ_TOP1"):
        LocalLightNavQueryController(
            _FakeEngine([]), lambda: (torch.zeros((1, 3, 4, 4)), [0]),
        )


def test_budget_drop_is_not_reported_as_a_validated_fact() -> None:
    object_map = ObjectMap()
    object_map.reset_episode(SCENE, EPISODE, 0)
    first = _add_node(object_map, 1, "sofa", feature=(1.0, 0.0), steps=(1, 2))
    second = _add_node(object_map, 2, "table", feature=(0.8, 0.6), steps=(3, 4))
    controller = MockAdaptiveQueryController(
        decide_script=[AdaptiveQueryDecision(QUERY_L3, query_text="object")],
        select_script=[AdaptiveQueryDecision(
            USE, object_ids=(first.object_id, second.object_id),
        )],
    )
    runner = _runner(controller)
    context = runner.run(
        object_map=object_map, step=10, instruction="find the object",
        l1_status="built", l1_context=_l1_context(), agent_pose=_pose(),
        max_chars=230,
    )
    assert context is not None
    assert len(context.layer_items) == 1
    assert runner.last_diagnostic["validated_object_ids"] == [
        context.layer_items[0].metadata["object_id"]
    ]
    assert len(runner.last_diagnostic["validated_evidence_ids"]) == 2
