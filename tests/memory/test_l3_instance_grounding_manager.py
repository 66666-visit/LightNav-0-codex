"""Manager-level contracts for the opt-in ViL3DRel instance grounder."""

from __future__ import annotations

import json

import numpy as np

from lightnav.memory.conceptgraphs_lite import (
    CameraIntrinsics,
    InstanceMask,
    ObjectMap,
    ObjectNode,
    ObservedFromEdge,
    Pose3D,
    RGBDFrame,
    SpatialEvidence,
    ViewpointNode,
)
from lightnav.memory.l3_instance_grounder import (
    InstanceGroundingResult,
    ProposalScore,
)
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.v3_schemas import canonical_evidence_key

SCENE = "scene"
EPISODE = "episode"


class _SegExtractor:
    def reset_episode(self, episode_id, instruction):
        return None

    def close_episode(self):
        return None

    def extract(self, rgb, step_id):
        return []


def _frame() -> RGBDFrame:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(16, dtype=np.uint8).reshape(4, 4) * 10
    intrinsics = CameraIntrinsics(2.0, 2.0, 1.5, 1.5, 4, 4)
    pose = Pose3D((0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    return RGBDFrame(rgb, np.full((4, 4), 2.0), intrinsics, pose, pose)


def _mask(evidence_id: str, start: int) -> InstanceMask:
    values = np.zeros((4, 4), dtype=np.bool_)
    values[:, start:start + 2] = True
    return InstanceMask(
        "sofa", 0.9, values, evidence_id, "unit-test",
    )


def _manager(*, sidecar: bool, grounder=None) -> MemoryManager:
    return MemoryManager(
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=SpatialL3Config(
            mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
            min_injection_confidence=0.5,
            rgb_sidecar_enabled=sidecar,
            instance_grounder_enabled=grounder is not None,
            grounder_checkpoint="/fake/official_student.pt",
            grounder_device="cpu",
            grounder_selection_mode=(
                "eligible" if grounder is not None else "argmax"
            ),
            grounder_landmark_queries=grounder is not None,
            grounder_geometry_fallback=grounder is not None,
        ),
        spatial_extractor=_SegExtractor(),
        clip_encoder=None,
        prompt_mode="l1_original_only",
        instance_grounder=grounder,
    )


class _FakeGrounder:
    """Deterministic official-score stub with the requested object ranked first."""

    def __init__(self, selected_object_id: str | None) -> None:
        self.selected_object_id = selected_object_id
        self.calls: list[dict] = []

    def select(self, *, instruction, object_map, sidecar, step, episode_id):
        proposals, skipped = sidecar.proposals(object_map.objects)
        self.calls.append({
            "instruction": instruction, "step": step,
            "proposal_ids": [proposal.object_id for proposal in proposals],
            "skipped": list(skipped),
        })
        ids = [proposal.object_id for proposal in proposals]
        ranked = (
            [self.selected_object_id] + [value for value in ids if value != self.selected_object_id]
            if self.selected_object_id in ids
            else ids
        )
        probability = {
            object_id: (len(ranked) - index) / len(ranked)
            for index, object_id in enumerate(ranked)
        }
        scores = tuple(
            ProposalScore(
                object_id=proposal.object_id,
                class_name=proposal.class_name,
                point_count=proposal.point_count,
                center_vil=(0.0, 0.0, 0.0),
                size_vil=(1.0, 1.0, 1.0),
                logit=float(probability.get(proposal.object_id, 0.0)),
                probability=float(probability.get(proposal.object_id, 0.0)),
                proposal_id=f"proposal-{index}",
                evidence_ids=tuple(proposal.evidence_ids),
            )
            for index, proposal in enumerate(proposals)
        )
        selected = self.selected_object_id
        if selected is not None and not any(
            proposal.object_id == selected for proposal in proposals
        ):
            selected = None
        selected_index = next(
            (index for index, proposal in enumerate(proposals)
             if proposal.object_id == selected),
            None,
        )
        return InstanceGroundingResult(
            selected_object_id=selected,
            selected_index=selected_index,
            proposals=scores,
            top1_top2_gap=0.1 if len(proposals) > 1 else None,
            latency_ms=1.0,
            checkpoint="fake",
            checkpoint_sha256="fake",
            skipped_objects=tuple(skipped),
            refusal_reason=None if selected is not None else "no_rgb_proposals",
        )


def _fill_map(manager: MemoryManager) -> tuple[ObjectMap, object, list[ObjectNode]]:
    object_map = manager.spatial_map
    assert object_map is not None
    frame = _frame()
    specs = [
        ("object-sofa-a", "sofa", (2.0, 0.0, -2.0), 0),
        ("object-sofa-b", "sofa", (2.0, 0.0, 2.0), 1),
        ("object-chair", "chair", (0.0, 0.0, -4.0), 2),
    ]
    nodes: list[ObjectNode] = []
    masks = []
    for index, (object_id, label, center, start) in enumerate(specs, start=1):
        evidence_id = canonical_evidence_key(
            SCENE, EPISODE, 0, f"l3_seg_1_{start}",
        )
        viewpoint_id = json.dumps(
            [SCENE, EPISODE, 0, index], separators=(",", ":"),
        )
        object_map.viewpoints[viewpoint_id] = ViewpointNode(
            viewpoint_id, SCENE, EPISODE, 0, 1, frame.agent_pose,
            frame.camera_pose,
        )
        object_map.evidence[evidence_id] = SpatialEvidence(
            evidence_id, SCENE, EPISODE, 0, 1, label, 0.9,
            "unit-test", 8, 8, "observed",
        )
        object_map.edges.append(ObservedFromEdge(
            object_id, viewpoint_id, evidence_id,
        ))
        node = ObjectNode(
            object_id, label, center,
            ((center[0] - 0.5, -0.5, center[2] - 0.5),
             (center[0] + 0.5, 0.5, center[2] + 0.5)),
            1, [evidence_id], "observed", np.zeros((4, 3)),
        )
        object_map.objects[object_id] = node
        nodes.append(node)
        masks.append(_mask(evidence_id, start=start if start < 2 else 0))
    assert manager.rgb_sidecar is not None
    manager.rgb_sidecar.record_frame(frame, masks, step_id=1)
    manager.rgb_sidecar.bind_objects(object_map.objects.values())
    manager.last_spatial_frame = frame
    return object_map, frame, nodes


def test_sidecar_is_off_by_default_and_records_when_enabled() -> None:
    default = _manager(sidecar=False)
    assert default.rgb_sidecar is None

    manager = _manager(sidecar=True)
    assert manager.rgb_sidecar is not None
    manager.reset_episode("episode", "find the far sofa", scene_id=SCENE)
    object_map, _frame_value, nodes = _fill_map(manager)
    assert manager.rgb_sidecar.summary()["records"] == 3
    for node in nodes:
        assert manager.rgb_sidecar.record_for(node.evidence_ids[0]).object_id == (
            node.object_id
        )


def test_grounder_injects_only_the_selected_existing_geometry_fact() -> None:
    grounder = _FakeGrounder("object-sofa-b")
    manager = _manager(sidecar=True, grounder=grounder)
    manager.reset_episode("episode", "find the far sofa", scene_id=SCENE)
    object_map, _frame_value, _nodes = _fill_map(manager)

    context = manager.get_geometry_on_demand_context(
        step=2, instruction="find the far sofa",
        l1_status="no_candidates", max_chars=320,
    )
    assert context is not None
    assert len(context.layer_items) == 1
    item = context.layer_items[0]
    assert item.metadata["object_id"] == "object-sofa-b"
    assert context.text.count("Spatial history:") == 1
    assert "object-sofa-a" not in context.text
    assert "object-chair" not in context.text

    diagnostic = manager.last_l3_query_diagnostic
    assert diagnostic["injected_count"] == 1
    assert diagnostic["object_ids"] == ["object-sofa-b"]
    assert diagnostic["candidate_count"] == 3
    assert len(diagnostic["instance_grounding"]["proposals"]) == 3
    assert diagnostic["instance_grounding"]["selected_object_id"] == "object-sofa-b"
    assert grounder.calls[0]["proposal_ids"] == [
        "object-chair", "object-sofa-a", "object-sofa-b",
    ]
    assert manager.l3_query_stats_summary()["queries_injected"] == 1
    json.dumps(manager.last_l3_query_diagnostic)


def test_l1_satisfied_never_calls_the_grounder_and_keeps_l1() -> None:
    grounder = _FakeGrounder("object-sofa-b")
    manager = _manager(sidecar=True, grounder=grounder)
    manager.reset_episode("episode", "find the far sofa", scene_id=SCENE)
    _fill_map(manager)
    context = manager.get_geometry_on_demand_context(
        step=2, instruction="find the far sofa",
        l1_status="built", max_chars=320,
    )
    assert context is None
    assert grounder.calls == []
    assert manager.last_l3_query_diagnostic["skipped_reason"] == "l1_satisfied"


class _PhantomGrounder(_FakeGrounder):
    """Official scores for an object ID that does not exist in the ObjectMap."""

    def select(self, *, instruction, object_map, sidecar, step, episode_id):
        self.calls.append({
            "instruction": instruction, "step": step,
            "proposal_ids": ["object-not-in-proposals"], "skipped": [],
        })
        score = ProposalScore(
            object_id="object-not-in-proposals",
            class_name="sofa",
            point_count=8,
            center_vil=(0.0, 0.0, 0.0),
            size_vil=(1.0, 1.0, 1.0),
            logit=2.0,
            probability=1.0,
            proposal_id="proposal-0",
            evidence_ids=(),
        )
        return InstanceGroundingResult(
            selected_object_id="object-not-in-proposals",
            selected_index=0,
            proposals=(score,),
            top1_top2_gap=None,
            latency_ms=1.0,
            checkpoint="fake",
            checkpoint_sha256="fake",
            refusal_reason=None,
        )


def test_unmapped_proposal_falls_back_to_the_unchanged_geometry_fact() -> None:
    grounder = _PhantomGrounder(None)
    manager = _manager(sidecar=True, grounder=grounder)
    manager.reset_episode("episode", "find the far sofa", scene_id=SCENE)
    _fill_map(manager)
    context = manager.get_geometry_on_demand_context(
        step=2, instruction="find the far sofa",
        l1_status="no_candidates", max_chars=320,
    )
    assert context is not None
    assert context.layer_items
    diagnostic = manager.last_l3_query_diagnostic
    assert diagnostic["injected_count"] == len(context.layer_items)
    assert diagnostic["selected_valid_proposals"] == []
    assert diagnostic["geometry_fallback_reason"]
    assert diagnostic["selection_sources"] == ["geometry_fallback"] * len(
        context.layer_items
    )
    assert diagnostic["actual_injection"] is True


class _OneMaskExtractor:
    def reset_episode(self, episode_id, instruction):
        return None

    def close_episode(self):
        return None

    def extract(self, rgb, step_id):
        from lightnav.memory.yolo_seg_extractor import SegmentationInstance

        mask = np.ones(rgb.shape[:2], dtype=np.bool_)
        return [SegmentationInstance(
            "sofa", 0.95, mask,
            (0.0, 0.0, float(rgb.shape[1]), float(rgb.shape[0])),
        )]


def _obs_info() -> tuple[dict, dict]:
    rgb = np.full((8, 8, 3), 40, dtype=np.uint8)
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    info = {"spatial_rgbd": {
        "status": "ok", "aligned": True, "image_size_hw": [8, 8],
        "K": [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]],
        "agent_pose": {"position": [0.0, 1.0, 0.0],
                       "rotation": [1.0, 0.0, 0.0, 0.0]},
        "camera_extrinsics": {"position": [0.0, 0.0, 0.0],
                              "rotation": [1.0, 0.0, 0.0, 0.0]},
        "camera_frame": "habitat_right_up_back",
        "depth_encoding": "meters", "min_depth_m": 0.1, "max_depth_m": 10.0,
    }}
    return {"rgb": rgb, "depth": depth}, info


def _observe_manager(*, sidecar: bool) -> MemoryManager:
    return MemoryManager(
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=SpatialL3Config(
            mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
            seg_every_n_steps=1, min_injection_confidence=0.5,
            rgb_sidecar_enabled=sidecar,
        ),
        spatial_extractor=_OneMaskExtractor(),
        clip_encoder=None,
        prompt_mode="l1_original_only",
    )


def test_observe_spatial_records_real_xyzrgb_without_changing_objectnode() -> None:
    control = _observe_manager(sidecar=False)
    control.reset_episode("episode", "find the sofa", scene_id=SCENE)
    obs, info = _obs_info()
    control.observe_spatial(obs, info, [], step=1)
    control_node = next(iter(control.spatial_map.objects.values()))
    control_points = control_node.points_world.copy()
    control_center = control_node.center_world

    manager = _observe_manager(sidecar=True)
    manager.reset_episode("episode", "find the sofa", scene_id=SCENE)
    diagnostic = manager.observe_spatial(obs, info, [], step=1)
    node = next(iter(manager.spatial_map.objects.values()))
    assert np.array_equal(node.points_world, control_points)
    assert node.center_world == control_center
    assert node.object_id == control_node.object_id
    assert diagnostic["rgb_sidecar"]["records_minted"] == 1
    record = manager.rgb_sidecar.record_for(node.evidence_ids[0])
    assert record is not None
    assert record.object_id == node.object_id
    assert record.colors_rgb.dtype == np.uint8
    assert np.array_equal(record.colors_rgb, obs["rgb"].reshape(-1, 3))


class _TwoMaskExtractor:
    def reset_episode(self, episode_id, instruction):
        return None

    def close_episode(self):
        return None

    def extract(self, rgb, step_id):
        from lightnav.memory.yolo_seg_extractor import SegmentationInstance

        height, width = rgb.shape[:2]
        left = np.zeros((height, width), dtype=np.bool_)
        left[:, : width // 2] = True
        right = np.zeros((height, width), dtype=np.bool_)
        right[:, width // 2:] = True
        return [
            SegmentationInstance("sofa", 0.95, left, (0.0, 0.0, width / 2, float(height))),
            SegmentationInstance("sofa", 0.94, right, (width / 2, 0.0, float(width), float(height))),
        ]


def _two_obs_info() -> tuple[dict, dict]:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :4, 0] = 20
    rgb[:, 4:, 0] = 80
    rgb[:, :, 1] = 10
    rgb[:, :, 2] = 5
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    info = {"spatial_rgbd": {
        "status": "ok", "aligned": True, "image_size_hw": [8, 8],
        "K": [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]],
        "agent_pose": {"position": [0.0, 1.0, 0.0],
                       "rotation": [1.0, 0.0, 0.0, 0.0]},
        "camera_extrinsics": {"position": [0.0, 0.0, 0.0],
                              "rotation": [1.0, 0.0, 0.0, 0.0]},
        "camera_frame": "habitat_right_up_back",
        "depth_encoding": "meters", "min_depth_m": 0.1, "max_depth_m": 10.0,
    }}
    return {"rgb": rgb, "depth": depth}, info


def test_two_same_class_instances_are_scored_then_only_one_is_injected() -> None:
    grounder = _FakeGrounder("object-sofa-b")
    manager = MemoryManager(
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=SpatialL3Config(
            mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
            seg_every_n_steps=1, min_injection_confidence=0.5,
            rgb_sidecar_enabled=True,
            instance_grounder_enabled=True,
            grounder_checkpoint="/fake/official_nr3d_student.pt",
            grounder_device="cpu",
        ),
        spatial_extractor=_TwoMaskExtractor(),
        clip_encoder=None,
        prompt_mode="l1_original_only",
        instance_grounder=grounder,
    )
    manager.reset_episode("episode", "find the far sofa", scene_id=SCENE)
    obs, info = _two_obs_info()
    diagnostic = manager.observe_spatial(obs, info, [], step=0)
    assert diagnostic["observed_objects"] == 2
    assert len(manager.spatial_map.objects) == 2
    assert manager.rgb_sidecar.summary()["records"] == 2
    ids = sorted(manager.spatial_map.objects)
    grounder.selected_object_id = ids[1]

    context = manager.get_geometry_on_demand_context(
        step=1, instruction="find the far sofa",
        l1_status="no_candidates", max_chars=320,
    )
    assert context is not None
    assert len(context.layer_items) == 1
    assert context.layer_items[0].metadata["object_id"] == ids[1]
    assert len(grounder.calls) == 1
    assert grounder.calls[0]["proposal_ids"] == ids
    l3 = manager.last_l3_query_diagnostic
    assert l3["proposal_count"] == 2
    assert len(l3["instance_grounding"]["proposals"]) == 2
    assert l3["injected_text"].count("Spatial history:") == 1


def test_sidecar_failure_never_breaks_geometry_or_l1(monkeypatch) -> None:
    manager = _observe_manager(sidecar=True)
    manager.reset_episode("episode", "find the sofa", scene_id=SCENE)
    assert manager.rgb_sidecar is not None
    monkeypatch.setattr(
        manager.rgb_sidecar, "record_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    obs, info = _obs_info()
    diagnostic = manager.observe_spatial(obs, info, [], step=1)
    assert diagnostic["observed_objects"] == 1
    assert len(manager.spatial_map.objects) == 1
    assert "rgb_sidecar_error" in diagnostic
    assert "boom" in diagnostic["rgb_sidecar_error"]


def test_no_historical_candidates_never_calls_the_grounder() -> None:
    grounder = _FakeGrounder("object-sofa-b")
    manager = _manager(sidecar=True, grounder=grounder)
    manager.reset_episode("episode", "find the far sofa", scene_id=SCENE)
    manager.last_spatial_frame = _frame()
    context = manager.get_geometry_on_demand_context(
        step=1, instruction="find the far sofa",
        l1_status="no_candidates", max_chars=320,
    )
    assert context is None
    assert grounder.calls == []
    assert manager.last_l3_query_diagnostic["injected_count"] == 0
    assert manager.last_l3_query_diagnostic["skipped_reason"] in {
        "empty_map", "no_historical_geometry_object",
    }


class _CountingRuntime:
    def __init__(self) -> None:
        self.calls = 0
        self.checkpoint = "fake"
        self.checkpoint_sha256 = "fake"

    def ground(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("grounder must not run on incomplete RGB")


def test_incomplete_rgb_refuses_the_whole_grounding_before_the_model() -> None:
    from types import SimpleNamespace

    from lightnav.memory.conceptgraphs_lite import GeometryConfig
    from lightnav.memory.l3_instance_grounder import ViL3DRelGrounder
    from lightnav.memory.rgb_sidecar import RGBSidecar

    frame = _frame()
    sidecar = RGBSidecar(GeometryConfig(min_points=1))
    sidecar.reset_episode(SCENE, EPISODE, 0)
    first = canonical_evidence_key(SCENE, EPISODE, 0, "l3_seg_1_0")
    second = canonical_evidence_key(SCENE, EPISODE, 0, "l3_seg_1_1")
    sidecar.record_frame(frame, [_mask(first, start=0)], step_id=1)
    node_a = ObjectNode(
        "object-a", "sofa", (0.0, 0.0, 0.0), ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        1, [first], "observed", np.zeros((4, 3)),
    )
    node_b = ObjectNode(
        "object-b", "sofa", (2.0, 0.0, 0.0), ((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
        1, [second], "observed", np.zeros((4, 3)),
    )
    object_map = SimpleNamespace(objects={
        node_a.object_id: node_a, node_b.object_id: node_b,
    })
    sidecar.bind_objects(object_map.objects.values())
    runtime = _CountingRuntime()
    grounder = ViL3DRelGrounder(runtime)
    result = grounder.select(
        instruction="find the far sofa", object_map=object_map,
        sidecar=sidecar, step=1, episode_id=EPISODE,
    )
    assert result.valid is False
    assert result.selected_object_id is None
    assert result.refusal_reason == "incomplete_rgb_sidecar"
    assert result.skipped_objects[0]["object_id"] == "object-b"
    assert runtime.calls == 0


def test_global_argmax_reference_falls_back_to_the_next_valid_candidate() -> None:
    grounder = _FakeGrounder("object-chair")
    manager = _manager(sidecar=True, grounder=grounder)
    manager.reset_episode("episode", "find the far sofa", scene_id=SCENE)
    _fill_map(manager)
    context = manager.get_geometry_on_demand_context(
        step=2, instruction="find the far sofa",
        l1_status="no_candidates", max_chars=320,
    )
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    injected = diagnostic["object_ids"]
    assert injected and injected[0] != "object-chair"
    assert set(injected) <= {"object-sofa-a", "object-sofa-b"}
    assert "object-sofa-a" in diagnostic["selected_valid_proposals"]
    assert "object-sofa-b" in diagnostic["selected_valid_proposals"]
    assert diagnostic["instance_grounding"]["selected_object_id"] == "object-chair"
    assert diagnostic["grounder_selected_object_id"] == "object-chair"
    assert diagnostic["injected_count"] == len(injected)
    assert diagnostic["actual_injection"] is True
    proposal = next(
        item for item in diagnostic["grounding_queries"][0]["proposals"]
        if item["persistent_object_id"] == "object-chair"
    )
    assert proposal["class_match"] is False
    assert proposal["formatable"] is False
    assert proposal["rejection_reason"] == "not_formatable"
