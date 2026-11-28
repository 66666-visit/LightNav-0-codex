"""C-arm CPU contracts: one-shot object-level historical memory admission.

The tests use the same fake segmentation extractor, fake ObjectMap and fake
ViL3DRel Student as the existing ordered-memory/instance-grounding suites.  No
model, CUDA device, Habitat process or network access is involved.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.habitat.policy import TrajVocabVLNCEPolicy
from lightnav.inference.policies import NavigationPolicy
from lightnav.memory.conceptgraphs_lite import (
    CameraIntrinsics,
    InstanceMask,
    ObjectNode,
    ObservedFromEdge,
    Pose3D,
    RGBDFrame,
    SpatialEvidence,
    ViewpointNode,
)
from lightnav.memory.history_admission import (
    AdmissionCandidate,
    HistoryAdmissionGate,
)
from lightnav.memory.instruction_landmarks import parse_instruction_landmarks
from lightnav.memory.l3_instance_grounder import (
    InstanceGroundingResult,
    ProposalScore,
)
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_context_builder import (
    audit_prompt_injection,
    inject_memory_context,
)
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.schemas import (
    DetectedObject,
    ObservationPacket,
    SceneExtraction,
)
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.v3_schemas import canonical_evidence_key

SCENE = "scene"
EPISODE = "episode"
INSTRUCTION = "Walk to the sofa and stop."
TWO_SOFA_INSTRUCTION = "Walk past the near sofa to the far sofa and stop."


class _SegExtractor:
    def reset_episode(self, episode_id, instruction):
        return None

    def close_episode(self):
        return None

    def extract(self, rgb, step_id):
        return []


class _ByIdScorer:
    """Pick one explicit ObjectNode per landmark query; labels stay identical."""

    def __init__(self, selection: dict[str, str]) -> None:
        self.selection = selection
        self.calls: list[str] = []

    def __call__(self, query: str, proposals) -> dict[str, float]:
        self.calls.append(query)
        chosen = self.selection[query]
        raw = {
            proposal.object_id: 0.9 if proposal.object_id == chosen else 0.01
            for proposal in proposals
        }
        total = sum(raw.values())
        return {
            object_id: value / total if total > 0 else 0.0
            for object_id, value in raw.items()
        }


class _StubGrounder:
    def __init__(self, scorer: _ByIdScorer) -> None:
        self.scorer = scorer
        self.calls: list[str] = []

    def select(self, *, instruction, object_map, sidecar, step, episode_id):
        proposals, _skipped = sidecar.proposals(object_map.objects)
        probabilities = self.scorer(instruction, proposals)
        scores = tuple(
            ProposalScore(
                object_id=proposal.object_id,
                class_name=proposal.class_name,
                point_count=proposal.point_count,
                center_vil=(0.0, 0.0, 0.0),
                size_vil=(1.0, 1.0, 1.0),
                logit=float(np.log(max(probabilities[proposal.object_id], 1e-12))),
                probability=probabilities[proposal.object_id],
                proposal_id=f"proposal-{index}",
                evidence_ids=tuple(proposal.evidence_ids),
            )
            for index, proposal in enumerate(proposals)
        )
        selected = (
            max(scores, key=lambda score: score.probability).object_id
            if scores else None
        )
        self.calls.append(instruction)
        return InstanceGroundingResult(
            selected_object_id=selected,
            selected_index=0 if scores else None,
            proposals=scores,
        )


class _UnusedGrounder:
    def select(self, **_kwargs):
        raise AssertionError("the grounder must not run when L1 is satisfied")


def _frame() -> RGBDFrame:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(16, dtype=np.uint8).reshape(4, 4) * 10
    intrinsics = CameraIntrinsics(2.0, 2.0, 1.5, 1.5, 4, 4)
    pose = Pose3D((0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    return RGBDFrame(rgb, np.full((4, 4), 2.0), intrinsics, pose, pose)


def _manager(
    grounder, *, admission_mode: str | None = "off",
    instruction: str = INSTRUCTION, include_placeholders: bool = False,
    local_graph: LocalGraphConfig | None = None,
    extractor=None, spatial_extractor=None,
) -> MemoryManager:
    config = {
        "mode": "geometry_on_demand",
        "seg_model_path": "/fake/seg.pt",
        "min_injection_confidence": 0.5,
        "rgb_sidecar_enabled": True,
        "instance_grounder_enabled": True,
        "grounder_checkpoint": "/fake/official_student.pt",
        "grounder_device": "cpu",
        "grounder_selection_mode": "eligible",
        "grounder_landmark_queries": True,
        "grounder_geometry_fallback": True,
        "grounder_ordered_landmark_memory": True,
        "include_missing_landmark_placeholders": include_placeholders,
        "preserve_original_spatial_fact": False,
    }
    if admission_mode is not None:
        config["history_admission_mode"] = admission_mode
    manager = MemoryManager(
        extractor=extractor,
        strategy="hierarchical",
        local_graph_config=local_graph or LocalGraphConfig(
            enabled=True, max_landmarks=2,
        ),
        spatial_l3_config=SpatialL3Config(**config),
        spatial_extractor=spatial_extractor or _SegExtractor(),
        clip_encoder=None,
        prompt_mode="l1_original_only",
        instance_grounder=grounder,
    )
    manager.reset_episode(EPISODE, instruction, scene_id=SCENE)
    return manager


def _fill_map(
    manager: MemoryManager, specs, *, prefix: str = "l3_seg_1",
    episode: str = EPISODE,
) -> None:
    object_map = manager.spatial_map
    assert object_map is not None
    frame = _frame()
    masks = []
    for index, (object_id, label, center, confidence) in enumerate(specs):
        evidence_id = canonical_evidence_key(
            SCENE, episode, 0, f"{prefix}_{index}",
        )
        viewpoint_id = json.dumps(
            [SCENE, episode, 0, f"{prefix}-view-{index}"], separators=(",", ":"),
        )
        object_map.viewpoints[viewpoint_id] = ViewpointNode(
            viewpoint_id, SCENE, episode, 0, 1, frame.agent_pose,
            frame.camera_pose,
        )
        object_map.evidence[evidence_id] = SpatialEvidence(
            evidence_id, SCENE, episode, 0, 1, label, confidence,
            "unit-test", 8, 8, "observed",
        )
        object_map.edges.append(ObservedFromEdge(
            object_id, viewpoint_id, evidence_id,
        ))
        object_map.objects[object_id] = ObjectNode(
            object_id, label, center,
            ((center[0] - 0.5, -0.5, center[2] - 0.5),
             (center[0] + 0.5, 0.5, center[2] + 0.5)),
            1, [evidence_id], "observed", np.zeros((4, 3)),
        )
        mask = np.zeros((4, 4), dtype=np.bool_)
        mask[:, index % 3:index % 3 + 2] = True
        masks.append(InstanceMask(
            label, 0.95, mask, evidence_id, "unit-test",
        ))
    assert manager.rgb_sidecar is not None
    manager.rgb_sidecar.record_frame(frame, masks, step_id=1)
    manager.rgb_sidecar.bind_objects(object_map.objects.values())
    manager.last_spatial_frame = frame


def _add_evidence(
    manager: MemoryManager, object_id: str, *, prefix: str, step: int = 2,
) -> str:
    """Add a second observed Evidence record to an existing ObjectNode."""
    object_map = manager.spatial_map
    assert object_map is not None and manager.rgb_sidecar is not None
    node = object_map.objects[object_id]
    evidence_id = canonical_evidence_key(SCENE, EPISODE, 0, prefix)
    viewpoint_id = json.dumps(
        [SCENE, EPISODE, 0, prefix], separators=(",", ":"),
    )
    frame = manager.last_spatial_frame
    assert frame is not None
    object_map.viewpoints[viewpoint_id] = ViewpointNode(
        viewpoint_id, SCENE, EPISODE, 0, step, frame.agent_pose,
        frame.camera_pose,
    )
    object_map.evidence[evidence_id] = SpatialEvidence(
        evidence_id, SCENE, EPISODE, 0, step, node.class_name, 0.85,
        "unit-test", 8, 8, "observed",
    )
    object_map.edges.append(ObservedFromEdge(
        object_id, viewpoint_id, evidence_id,
    ))
    node.evidence_ids.append(evidence_id)
    node.observation_count += 1
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[1:3, 1:3] = True
    manager.rgb_sidecar.record_frame(
        frame, [InstanceMask(node.class_name, 0.9, mask, evidence_id, "unit-test")],
        step_id=step,
    )
    manager.rgb_sidecar.bind_objects(object_map.objects.values())
    return evidence_id


def _query(manager: MemoryManager, *, step: int = 2, max_chars: int = 400,
           instruction: str = INSTRUCTION):
    return manager.get_geometry_on_demand_context(
        step=step, instruction=instruction, l1_status="no_candidates",
        max_chars=max_chars,
    )


def _item_evidence_ids(context) -> list[str]:
    return list(dict.fromkeys(
        evidence_id for item in context.layer_items
        for evidence_id in item.metadata.get("evidence_ids", [])
        if isinstance(evidence_id, str)
    ))


# --------------------------------------------------------------- 1. off == H1


def test_off_mode_is_byte_identical_to_the_h1_control() -> None:
    grounder = _StubGrounder(_ByIdScorer({"sofa": "object-sofa"}))
    control = _manager(grounder, admission_mode=None)
    _fill_map(control, [("object-sofa", "sofa", (1.0, 0.0, 1.0), 0.9)])
    control_context = _query(control)
    assert control_context is not None

    gated_off = _manager(
        _StubGrounder(_ByIdScorer({"sofa": "object-sofa"})),
        admission_mode="off",
    )
    _fill_map(gated_off, [("object-sofa", "sofa", (1.0, 0.0, 1.0), 0.9)])
    off_context = _query(gated_off)
    assert off_context is not None

    assert off_context.text == control_context.text
    assert [
        (item.description, item.metadata, item.evidence_refs)
        for item in off_context.layer_items
    ] == [
        (item.description, item.metadata, item.evidence_refs)
        for item in control_context.layer_items
    ]
    diagnostic = gated_off.last_l3_query_diagnostic
    assert diagnostic["history_admission_mode"] == "off"
    assert diagnostic["history_admission_adopted_object_ids"] == []
    assert diagnostic["history_admission_rejected_seen_object_ids"] == []


# ------------------------------------- 2. same object, changed Evidence ID off


def test_same_object_with_a_new_evidence_id_is_injected_only_once() -> None:
    grounder = _StubGrounder(_ByIdScorer({"sofa": "object-sofa"}))
    manager = _manager(grounder, admission_mode="once_per_object")
    _fill_map(manager, [("object-sofa", "sofa", (1.0, 0.0, 1.0), 0.9)])

    first = _query(manager, step=2)
    assert first is not None
    first_evidence = _item_evidence_ids(first)
    assert first_evidence
    first_diagnostic = manager.last_l3_query_diagnostic
    assert first_diagnostic["history_admission_accepted_object_ids"] == [
        "object-sofa"
    ]
    assert manager.history_admission_gate.pending_object_ids == frozenset(
        {"object-sofa"}
    )
    # Rule 5: pending is not adopted until the final Prompt audit confirms it.
    assert manager.history_admission_gate.adopted_object_ids == frozenset()
    manager.confirm_prompt_context(first_evidence, step=2)
    assert manager.history_admission_gate.adopted_object_ids == frozenset(
        {"object-sofa"}
    )

    second_evidence_id = _add_evidence(
        manager, "object-sofa", prefix="l3_seg_2",
    )
    second = _query(manager, step=3)
    assert second is None
    diagnostic = manager.last_l3_query_diagnostic
    assert diagnostic["history_admission_accepted_object_ids"] == []
    assert diagnostic["history_admission_rejected_seen_object_ids"] == [
        "object-sofa"
    ]
    assert diagnostic["skipped_reason"] == "history_admission_rejected_seen_object"
    assert second_evidence_id not in _item_evidence_ids(first)
    assert "injected_text" not in diagnostic or diagnostic["injected_text"] is None


# ------------------------------------------- 3/4. different instances, one class


def test_different_object_ids_of_the_same_class_are_both_admitted() -> None:
    instruction = TWO_SOFA_INSTRUCTION
    grounder = _StubGrounder(_ByIdScorer({
        "near sofa": "object-sofa-near",
        "far sofa": "object-sofa-far",
    }))
    manager = _manager(
        grounder, admission_mode="once_per_object", instruction=instruction,
    )
    _fill_map(manager, [
        ("object-sofa-near", "sofa", (1.0, 0.0, 1.0), 0.9),
        ("object-sofa-far", "sofa", (4.0, 0.0, 1.0), 0.9),
    ])
    context = _query(manager, step=2, instruction=instruction)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    assert diagnostic["history_admission_candidate_object_ids"] == [
        "object-sofa-near", "object-sofa-far",
    ]
    assert diagnostic["history_admission_accepted_object_ids"] == [
        "object-sofa-near", "object-sofa-far",
    ]
    assert diagnostic["history_admission_rejected_seen_object_ids"] == []
    # Same detector class, two distinct stable ObjectNode IDs: both lines stay.
    assert "near sofa" in context.text
    assert "far sofa" in context.text
    assert len(context.layer_items) == 2
    manager.confirm_prompt_context(_item_evidence_ids(context), step=2)
    assert manager.history_admission_gate.adopted_object_ids == frozenset(
        {"object-sofa-near", "object-sofa-far"}
    )


def test_gate_keys_on_full_object_id_not_class_name() -> None:
    gate = HistoryAdmissionGate("once_per_object")
    decision = gate.admit([
        AdmissionCandidate("object-sofa-a", ("e-a",)),
        AdmissionCandidate("object-sofa-b", ("e-b",)),
    ], step=1)
    assert decision.accepted_object_ids == ("object-sofa-a", "object-sofa-b")
    assert decision.rejected_seen_object_ids == ()
    gate.confirm(["e-a", "e-b"], step=1)
    again = gate.admit([
        AdmissionCandidate("object-sofa-a", ("e-a2",)),
        AdmissionCandidate("object-sofa-b", ("e-b2",)),
    ], step=2)
    assert again.accepted_object_ids == ()
    assert again.rejected_seen_object_ids == (
        "object-sofa-a", "object-sofa-b",
    )


# --------------------------------------- 5. not adopted before the final audit


def test_candidate_is_not_adopted_before_the_final_prompt_audit() -> None:
    gate = HistoryAdmissionGate("once_per_object")
    gate.begin_step(4)
    decision = gate.admit([
        AdmissionCandidate("object-bed", ("ev-bed",)),
    ], step=4)
    assert decision.accepted_object_ids == ("object-bed",)
    assert gate.pending_object_ids == frozenset({"object-bed"})
    assert gate.adopted_object_ids == frozenset()

    # An audit with unrelated Evidence proves nothing about this object.
    assert gate.confirm(["ev-other"], step=4) == ()
    assert gate.adopted_object_ids == frozenset()

    gate.begin_step(5)
    gate.admit([AdmissionCandidate("object-bed", ("ev-bed-2",))], step=5)
    assert gate.confirm(["ev-bed-2"], step=5) == ("object-bed",)
    assert gate.adopted_object_ids == frozenset({"object-bed"})
    # A stale Step number cannot commit a later pending proposal.
    gate.begin_step(6)
    gate.admit([AdmissionCandidate("object-chair", ("ev-chair",))], step=6)
    assert gate.confirm(["ev-chair"], step=5) == ()
    assert gate.adopted_object_ids == frozenset({"object-bed"})


# ------------------------------------------------- 6. missing object_id fail-open


def test_missing_object_id_is_fail_open_and_counted() -> None:
    gate = HistoryAdmissionGate("once_per_object")
    gate.begin_step(1)
    first = gate.admit([
        AdmissionCandidate(None, ("ev-none",)),
        AdmissionCandidate("", ("ev-empty",)),
    ], step=1)
    assert first.accepted_indices == (0, 1)
    assert first.rejected_seen_object_ids == ()
    assert first.missing_object_id_count == 2
    # Missing metadata cannot be adopted, but a repeat must not be refused.
    gate.confirm(["ev-none", "ev-empty"], step=1)
    assert gate.adopted_object_ids == frozenset()
    second = gate.admit([
        AdmissionCandidate(None, ("ev-none-2",)),
        AdmissionCandidate("", ("ev-empty-2",)),
    ], step=2)
    assert second.accepted_indices == (0, 1)
    assert second.missing_object_id_count == 2
    assert gate.diagnostics()["history_admission_missing_object_id_count"] == 2


# ---------------------------------------------------- 7. Episode boundary reset


def test_episode_reset_clears_adopted_objects() -> None:
    grounder = _StubGrounder(_ByIdScorer({"sofa": "object-sofa"}))
    manager = _manager(grounder, admission_mode="once_per_object")
    _fill_map(manager, [("object-sofa", "sofa", (1.0, 0.0, 1.0), 0.9)])
    context = _query(manager, step=2)
    assert context is not None
    manager.confirm_prompt_context(_item_evidence_ids(context), step=2)
    assert manager.history_admission_gate.adopted_object_ids == frozenset(
        {"object-sofa"}
    )

    manager.reset_episode("episode-2", INSTRUCTION, scene_id=SCENE)
    assert manager.history_admission_gate.adopted_object_ids == frozenset()
    assert manager.history_admission_gate.pending_object_ids == frozenset()
    _fill_map(
        manager, [("object-sofa", "sofa", (1.0, 0.0, 1.0), 0.9)],
        episode="episode-2",
    )
    second_episode = _query(manager, step=2)
    assert second_episode is not None
    assert manager.last_l3_query_diagnostic[
        "history_admission_accepted_object_ids"
    ] == ["object-sofa"]
    assert manager.history_admission_gate.adopted_object_ids == frozenset()


# ------------------------------------------------ 8. L1 current memory untouched


def _l1_manager(admission_mode: str) -> MemoryManager:
    detections = {
        step: SceneExtraction(
            objects=["chair"],
            detections=[DetectedObject(
                "chair", 0.95, (0.0, 0.0, 8.0, 8.0),
            )],
            confidence=0.95, source="fake_l1",
        )
        for step in (0, 1)
    }
    return _manager(
        _UnusedGrounder(), admission_mode=admission_mode,
        instruction="find the chair",
        local_graph=LocalGraphConfig(enabled=True, reactivation_steps=1),
        extractor=MockSceneExtractor(detections),
    )


def _l1_context(manager: MemoryManager):
    rgb = np.full((8, 8, 3), 20, dtype=np.uint8)
    events = manager.observe(ObservationPacket(
        step_id=0, rgb=rgb, instruction="find the chair",
    ))
    return manager.build_prompt_context(
        step=0, instruction="find the chair", events=events,
        rgb_width=8, frame=None,
    )


def test_l1_current_visual_memory_is_identical_with_the_gate_on() -> None:
    off = _l1_manager("off")
    once = _l1_manager("once_per_object")
    off_result = _l1_context(off)
    once_result = _l1_context(once)

    assert off_result.context is not None
    assert once_result.context is not None
    assert once_result.text == off_result.text
    assert [
        (item.description, item.kind, item.metadata)
        for item in once_result.layer_items
    ] == [
        (item.description, item.kind, item.metadata)
        for item in off_result.layer_items
    ]
    assert once_result.diagnostics["layers"]["l1"] == (
        off_result.diagnostics["layers"]["l1"]
    )
    # L1 never creates an L3 historical candidate.
    assert once.history_admission_gate.pending_object_ids == frozenset()
    assert once.history_admission_gate.adopted_object_ids == frozenset()


# ------------------------------ 9. prompt bytes, roles, action path, token budget


def test_prompt_contract_and_budget_are_unchanged_for_admitted_rows() -> None:
    grounder = _StubGrounder(_ByIdScorer({"sofa": "object-sofa"}))
    manager = _manager(grounder, admission_mode="once_per_object")
    _fill_map(manager, [("object-sofa", "sofa", (1.0, 0.0, 1.0), 0.9)])
    context = _query(manager, step=2)
    assert context is not None

    video = object()
    base_prompt = (
        "You are a navigation agent.\n"
        f"<navigation_task>{INSTRUCTION}</navigation_task>. "
        "Predict your future trajectory as tokens."
    )
    base = {
        "conversations": [
            {"from": "human", "value": base_prompt},
            {"from": "gpt", "value": "placeholder"},
        ],
        "video_segments": video,
    }
    augmented = inject_memory_context(base, context, max_chars=320)
    audit = audit_prompt_injection(base, augmented, context, max_chars=320)
    assert audit["roundtrip_ok"] is True
    assert augmented["video_segments"] is video
    assert [message["from"] for message in augmented["conversations"]] == [
        "human", "gpt",
    ]
    new_prompt = augmented["conversations"][0]["value"]
    assert new_prompt.index("[Navigation Memory]") > new_prompt.index(
        "</navigation_task>"
    )
    start = new_prompt.index("[Navigation Memory]")
    end = new_prompt.index("\nPredict your future trajectory", start)
    assert new_prompt[:start] + new_prompt[end + 1:] == base_prompt

    # The existing token/character contract is untouched by admission.
    assert manager.prompt_budget_chars == 320
    assert len(context.text) <= 320
    assert [item.layer for item in context.layer_items] == ["long"]
    assert {item.kind for item in context.layer_items} == {
        "ordered_instruction_landmark"
    }
    assert manager.history_admission_gate.adopted_object_ids == frozenset()


# --------------------------------------------------------------- configuration


def test_once_per_object_requires_the_h1_ordered_grounding_path() -> None:
    base = {
        "mode": "geometry_on_demand",
        "seg_model_path": "/fake/seg.pt",
        "rgb_sidecar_enabled": True,
        "instance_grounder_enabled": True,
        "grounder_checkpoint": "/fake/student.pt",
        "grounder_selection_mode": "eligible",
        "grounder_landmark_queries": True,
        "grounder_ordered_landmark_memory": True,
        "history_admission_mode": "once_per_object",
    }
    SpatialL3Config(**base)  # valid C configuration

    with pytest.raises(ValueError, match="geometry_on_demand"):
        SpatialL3Config(**{**base, "mode": "inject"})
    with pytest.raises(ValueError, match="instance_grounder_enabled"):
        SpatialL3Config(**{**base, "instance_grounder_enabled": False})
    with pytest.raises(ValueError, match="grounder_selection_mode=eligible"):
        SpatialL3Config(**{
            **base, "grounder_selection_mode": "argmax",
            "grounder_landmark_queries": False,
        })
    with pytest.raises(ValueError, match="grounder_ordered_landmark_memory"):
        SpatialL3Config(**{**base, "grounder_ordered_landmark_memory": False})
    with pytest.raises(ValueError, match="off or once_per_object"):
        SpatialL3Config(**{**base, "history_admission_mode": "always"})


def test_landmark_parser_keeps_two_same_class_instances_separate() -> None:
    parsed = parse_instruction_landmarks(TWO_SOFA_INSTRUCTION)
    assert [landmark.target_phrase for landmark in parsed.landmarks] == [
        "near sofa", "far sofa",
    ]


# ------------------------------------------- Fake-Engine end-to-end C contract


class _PolicyEngine:
    """CPU stand-in for the LightNav engine; no weights, no CUDA, no Habitat."""

    def __init__(self) -> None:
        self.bundle = SimpleNamespace(
            video_size=(4, 4), video_fps=4, num_history_frames=2,
            pool_enable=False, pool_spatial=1, pool_mode="avg",
            slowfast_tiers=None, predict_horizon=1, action_method="flat",
        )
        self.backend = "hf"
        self.samples: list[dict] = []
        self.last_input_diagnostic: dict | None = None
        self.last_prompt_chars = 0
        self.max_new_tokens = 8

    def reset_episode_state(self) -> None:
        return None

    def generate(self, sample: dict) -> tuple[str, float]:
        self.samples.append(sample)
        prompt = sample["conversations"][0]["value"]
        trace = sample.get("_rgm_trace", {})
        memory_text = trace.get("memory_text", "")
        visible = bool(memory_text and memory_text in prompt)
        self.last_input_diagnostic = {
            "final_token_ids": [ord(char) for char in prompt],
            "memory_text_present_in_final_tokens": visible,
            "model_visible_evidence_ids": [
                evidence_id
                for record in trace.get("accepted_records", [])
                for evidence_id in record["evidence_ids"]
            ] if visible else [],
        }
        self.last_prompt_chars = len(prompt)
        return "<traj_1>", 1.0


@pytest.fixture(autouse=True)
def _small_video(monkeypatch):
    monkeypatch.setattr(
        NavigationPolicy, "_convert_frame",
        lambda self, frame: torch.full((3, 4, 4), float(frame[0, 0, 0])),
    )


def _policy_obs(step: int) -> tuple[dict, dict]:
    rgb = np.full((8, 8, 3), step, dtype=np.uint8)
    obs = {"rgb": rgb, "depth": np.full((8, 8), 2.0, dtype=np.float32)}
    info = {
        "spatial_rgbd": {
            "status": "ok", "aligned": True, "image_size_hw": [8, 8],
            "K": [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]],
            "agent_pose": {
                "position": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            "camera_extrinsics": {
                "position": [0.0, 0.88, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            "camera_frame": "habitat_right_up_back",
            "depth_encoding": "meters", "min_depth_m": 0.1, "max_depth_m": 10.0,
        },
    }
    if step > 0:
        info["spatial_transition_record"] = True
    return obs, info


def _policy_with_prefilled_map():
    detections = {
        step: SceneExtraction(
            objects=["chair"],
            detections=[DetectedObject(
                "chair", 0.95, (0.0, 0.0, 8.0, 8.0),
            )],
            confidence=0.95, source="fake_l1",
        )
        for step in range(6)
    }
    manager = _manager(
        _StubGrounder(_ByIdScorer({"sofa": "object-sofa"})),
        admission_mode="once_per_object",
        local_graph=LocalGraphConfig(enabled=True, reactivation_steps=1),
        extractor=MockSceneExtractor(detections),
    )
    centroids = np.zeros((3, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    engine = _PolicyEngine()
    policy = TrajVocabVLNCEPolicy(
        engine, centroids, dt=1.0, lin_vel_range=(0.0, 0.3),
        ang_vel_range=(-30.0, 30.0), num_history_frames=2,
        memory_enabled=True, memory_manager=manager,
    )
    policy.reset(
        {"instruction": {"text": INSTRUCTION}},
        memory_episode_id=EPISODE, memory_scene_id=SCENE,
    )
    # The prefilled Evidence has Step 1, so it is historical from Step 2 on.
    _fill_map(manager, [("object-sofa", "sofa", (1.0, 0.0, 1.0), 0.9)])
    return policy, manager, engine


def _add_step2_evidence(manager: MemoryManager) -> str:
    """Change the same object's Evidence ID without changing its object_id."""
    object_map = manager.spatial_map
    assert object_map is not None and manager.rgb_sidecar is not None
    node = object_map.objects["object-sofa"]
    frame = manager.last_spatial_frame
    assert frame is not None
    evidence_id = canonical_evidence_key(SCENE, EPISODE, 0, "prefill_step2")
    viewpoint_id = json.dumps(
        [SCENE, EPISODE, 0, "prefill-step2"], separators=(",", ":"),
    )
    object_map.viewpoints[viewpoint_id] = ViewpointNode(
        viewpoint_id, SCENE, EPISODE, 0, 2, frame.agent_pose, frame.camera_pose,
    )
    object_map.evidence[evidence_id] = SpatialEvidence(
        evidence_id, SCENE, EPISODE, 0, 2, node.class_name, 0.85,
        "unit-test", 8, 8, "observed",
    )
    object_map.edges.append(ObservedFromEdge(
        "object-sofa", viewpoint_id, evidence_id,
    ))
    node.evidence_ids.append(evidence_id)
    node.observation_count += 1
    mask = np.ones(frame.depth_m.shape, dtype=np.bool_)
    manager.rgb_sidecar.record_frame(
        frame,
        [InstanceMask("sofa", 0.9, mask, evidence_id, "unit-test")],
        step_id=2,
    )
    manager.rgb_sidecar.bind_objects(object_map.objects.values())
    return evidence_id


def test_fake_engine_adopts_after_audit_then_refuses_same_object() -> None:
    policy, manager, engine = _policy_with_prefilled_map()
    prompts: dict[int, str] = {}
    diagnostics: dict[int, dict] = {}
    for step in range(5):
        if step == 3:
            _add_step2_evidence(manager)
        obs, info = _policy_obs(step)
        policy.act(obs, info)
        prompts[step] = policy.last_prompt_text or ""
        diagnostics[step] = dict(policy.last_memory_diagnostic)

    # One unchanged action inference per Step; memory never adds a forward pass.
    assert len(engine.samples) == 5
    assert "[Navigation Memory]" not in prompts[0]
    assert "[Navigation Memory]" not in prompts[1]

    injected = diagnostics[2]
    assert injected["actual_injection"] is True
    assert injected["prompt_context"]["spatial_l3"]["l3_query"][
        "history_admission_mode"
    ] == "once_per_object"
    assert injected["history_admission_accepted_object_ids"] == ["object-sofa"]
    assert injected["history_admission_rejected_seen_object_ids"] == []
    # The audit-visible Evidence confirms the object; pending becomes adopted.
    assert injected["history_admission_adopted_object_ids"] == ["object-sofa"]
    assert injected["history_admission_pending_object_ids"] == []
    assert "sofa" in prompts[2]

    rejected = diagnostics[3]
    assert rejected["actual_injection"] is False
    assert rejected["history_admission_accepted_object_ids"] == []
    assert rejected["history_admission_rejected_seen_object_ids"] == [
        "object-sofa"
    ]
    assert rejected["prompt_context"]["spatial_l3"]["l3_query"][
        "skipped_reason"
    ] == "history_admission_rejected_seen_object"
    assert "[Navigation Memory]" not in prompts[3]
    # The original instruction and the action interface are untouched.
    assert "<navigation_task>" in prompts[3]
    assert policy.get_info()["action_waypoint_index"] is not None
    assert manager.prompt_budget_chars == 320
    # A later Step with no L3 query must not repeat the previous Step's lists.
    no_query = diagnostics[4]
    assert no_query["history_admission_mode"] == "once_per_object"
    assert no_query["history_admission_accepted_object_ids"] == []
    assert no_query["history_admission_rejected_seen_object_ids"] == []
    assert no_query["history_admission_adopted_object_ids"] == ["object-sofa"]


def test_repeated_object_within_one_build_is_admitted_once() -> None:
    gate = HistoryAdmissionGate("once_per_object")
    decision = gate.admit([
        AdmissionCandidate("object-bed", ("ev-first",)),
        AdmissionCandidate("object-bed", ("ev-second",)),
    ], step=1)
    assert decision.accepted_indices == (0,)
    assert decision.accepted_object_ids == ("object-bed",)
    assert decision.rejected_seen_object_ids == ("object-bed",)
    assert gate.pending_object_ids == frozenset({"object-bed"})
    assert gate.confirm(["ev-first"], step=1) == ("object-bed",)
    assert gate.adopted_object_ids == frozenset({"object-bed"})
