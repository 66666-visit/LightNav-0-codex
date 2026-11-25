"""Episode-level regressions for joint Grounding + Geometry instance selection.

The four Episodes below are the diagnostics that motivated this change:

* 487  -- no candidate: never inject, keep the original LightNav prompt;
* 168  -- many landmarks: keep several valid ObjectNodes, not one couch;
* 1080 -- "far couch, in front of the windows": if the global couch argmax has
           no formatable Evidence, use the next valid couch candidate;
* 1012 -- sink/bed/doorway/stairs: a globally highest sink score must not drop
           the valid bed memory, and an all-refusal grounding must fall back to
           the unchanged Geometry candidates.
"""

from __future__ import annotations

import json
import math

import numpy as np

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
from lightnav.memory.l3_evidence_selector import label_matches_target
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

INSTRUCTION_487 = (
    "Turn around and walk out of the room. Wait at the top of the stairs."
)
INSTRUCTION_1080 = (
    "Go forward toward the windows. Go toward the the far couch, "
    "Stop next to the couch, in front of the windows."
)
INSTRUCTION_1012 = (
    "walk away from sink, turn left, walk towards foot of bed, "
    "exit through doorway on left, turn right and step down "
    "two stairs and stop."
)
INSTRUCTION_168 = (
    "Walk through double doors into the house. "
    "Continue around the dining table and through the entry way "
    "to the next room. Walk up to the couch and armchairs "
    "surrounding a coffee table."
)


class _SegExtractor:
    def reset_episode(self, episode_id, instruction):
        return None

    def close_episode(self):
        return None

    def extract(self, rgb, step_id):
        return []


class _QueryScores:
    """Test-only official-score stub: one raw score per class for each query."""

    def __init__(self, scores: dict[str, dict[str, float]], default: float = 0.01):
        self.scores = scores
        self.default = default
        self.calls: list[str] = []

    def __call__(self, query: str, proposals) -> dict[str, float]:
        self.calls.append(query)
        raw = {
            proposal.object_id: float(
                self.scores.get(query, {}).get(
                    proposal.class_name, self.default,
                )
            )
            for proposal in proposals
        }
        total = sum(raw.values())
        return {
            object_id: value / total if total > 0 else 0.0
            for object_id, value in raw.items()
        }


class _StubGrounder:
    def __init__(self, scorer: _QueryScores) -> None:
        self.scorer = scorer
        self.calls: list[dict] = []

    def select(self, *, instruction, object_map, sidecar, step, episode_id):
        proposals, skipped = sidecar.proposals(object_map.objects)
        probabilities = self.scorer(instruction, proposals)
        scores = tuple(
            ProposalScore(
                object_id=proposal.object_id,
                class_name=proposal.class_name,
                point_count=proposal.point_count,
                center_vil=(0.0, 0.0, 0.0),
                size_vil=(1.0, 1.0, 1.0),
                logit=math.log(max(probabilities[proposal.object_id], 1e-12)),
                probability=probabilities[proposal.object_id],
                proposal_id=f"proposal-{index}",
                evidence_ids=tuple(proposal.evidence_ids),
            )
            for index, proposal in enumerate(proposals)
        )
        selected_index = max(
            range(len(scores)), key=lambda index: scores[index].probability,
        ) if scores else None
        self.calls.append({
            "query": instruction,
            "proposal_ids": [proposal.object_id for proposal in proposals],
            "selected_object_id": (
                scores[selected_index].object_id if selected_index is not None else None
            ),
        })
        return InstanceGroundingResult(
            selected_object_id=(
                scores[selected_index].object_id if selected_index is not None else None
            ),
            selected_index=selected_index,
            proposals=scores,
            top1_top2_gap=None,
            latency_ms=1.0,
            checkpoint="stub",
            checkpoint_sha256="stub",
            skipped_objects=tuple(skipped),
        )


class _FixedGrounder:
    """Test-only official-score stub that does NOT renormalize probabilities."""

    def __init__(self, probabilities: dict[str, float]) -> None:
        self.probabilities = probabilities
        self.calls: list[dict] = []

    def select(self, *, instruction, object_map, sidecar, step, episode_id):
        proposals, skipped = sidecar.proposals(object_map.objects)
        scores = tuple(
            ProposalScore(
                object_id=proposal.object_id,
                class_name=proposal.class_name,
                point_count=proposal.point_count,
                center_vil=(0.0, 0.0, 0.0),
                size_vil=(1.0, 1.0, 1.0),
                logit=math.log(max(
                    self.probabilities.get(proposal.object_id, 1e-12), 1e-12,
                )),
                probability=self.probabilities.get(proposal.object_id, 0.0),
                proposal_id=f"proposal-{index}",
                evidence_ids=tuple(proposal.evidence_ids),
            )
            for index, proposal in enumerate(proposals)
        )
        selected_index = max(
            range(len(scores)), key=lambda index: scores[index].probability,
        ) if scores else None
        self.calls.append({
            "query": instruction,
            "proposal_ids": [proposal.object_id for proposal in proposals],
            "selected_object_id": (
                scores[selected_index].object_id
                if selected_index is not None else None
            ),
        })
        return InstanceGroundingResult(
            selected_object_id=(
                scores[selected_index].object_id
                if selected_index is not None else None
            ),
            selected_index=selected_index,
            proposals=scores,
            top1_top2_gap=None,
            latency_ms=1.0,
            checkpoint="stub",
            checkpoint_sha256="stub",
            skipped_objects=tuple(skipped),
        )


class _RefusingGrounder:
    def __init__(self) -> None:
        self.calls = 0

    def select(self, *, instruction, object_map, sidecar, step, episode_id):
        self.calls += 1
        return InstanceGroundingResult(
            selected_object_id=None,
            selected_index=None,
            proposals=(),
            refusal_reason="grounding_error",
            error="stubbed refusal",
        )


def _frame() -> RGBDFrame:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(16, dtype=np.uint8).reshape(4, 4) * 10
    intrinsics = CameraIntrinsics(2.0, 2.0, 1.5, 1.5, 4, 4)
    pose = Pose3D((0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    return RGBDFrame(rgb, np.full((4, 4), 2.0), intrinsics, pose, pose)


def _manager(
    instruction: str, grounder, *, max_landmarks: int = 2,
    landmark_queries: bool = True, geometry_fallback: bool = True,
    selection_mode: str = "eligible",
    instruction_landmark_provider=None,
) -> MemoryManager:
    manager = MemoryManager(
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(
            enabled=True, max_landmarks=max_landmarks,
        ),
        spatial_l3_config=SpatialL3Config(
            mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
            min_injection_confidence=0.5, rgb_sidecar_enabled=True,
            instance_grounder_enabled=True,
            grounder_checkpoint="/fake/official_student.pt",
            grounder_device="cpu",
            grounder_selection_mode=selection_mode,
            grounder_landmark_queries=landmark_queries,
            grounder_geometry_fallback=geometry_fallback,
        ),
        spatial_extractor=_SegExtractor(), clip_encoder=None,
        prompt_mode="l1_original_only", instance_grounder=grounder,
        instruction_landmark_provider=instruction_landmark_provider,
    )
    manager.reset_episode(EPISODE, instruction, scene_id=SCENE)
    return manager


def _geometry_control_manager(
    instruction: str, *, max_landmarks: int = 2,
) -> MemoryManager:
    """Original Geometry On-demand arm: no sidecar, no Grounder."""
    manager = MemoryManager(
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(
            enabled=True, max_landmarks=max_landmarks,
        ),
        spatial_l3_config=SpatialL3Config(
            mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
            min_injection_confidence=0.5, rgb_sidecar_enabled=False,
            instance_grounder_enabled=False,
        ),
        spatial_extractor=_SegExtractor(), clip_encoder=None,
        prompt_mode="l1_original_only",
    )
    manager.reset_episode(EPISODE, instruction, scene_id=SCENE)
    return manager


def _fill_map(manager: MemoryManager, specs) -> None:
    object_map = manager.spatial_map
    assert object_map is not None
    frame = _frame()
    masks = []
    for index, (object_id, label, center, confidence) in enumerate(specs):
        evidence_id = canonical_evidence_key(
            SCENE, EPISODE, 0, f"l3_seg_1_{index}",
        )
        viewpoint_id = json.dumps(
            [SCENE, EPISODE, 0, index + 1], separators=(",", ":"),
        )
        object_map.viewpoints[viewpoint_id] = ViewpointNode(
            viewpoint_id, SCENE, EPISODE, 0, 1, frame.agent_pose,
            frame.camera_pose,
        )
        object_map.evidence[evidence_id] = SpatialEvidence(
            evidence_id, SCENE, EPISODE, 0, 1, label, confidence,
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
    if manager.rgb_sidecar is not None:
        manager.rgb_sidecar.record_frame(frame, masks, step_id=1)
        manager.rgb_sidecar.bind_objects(object_map.objects.values())
    manager.last_spatial_frame = frame


def _query(manager: MemoryManager, instruction: str, *, max_chars: int = 4000):
    return manager.get_geometry_on_demand_context(
        step=2, instruction=instruction, l1_status="no_candidates",
        max_chars=max_chars,
    )


def _specs_1080():
    return [
        ("object-couch-top", "couch", (2.0, 0.0, -2.0), 0.3),
        ("object-couch-valid", "couch", (2.0, 0.0, 2.0), 0.9),
        ("object-windows", "windows", (0.0, 0.0, -4.0), 0.9),
    ]


def _scores_1080() -> dict[str, dict[str, float]]:
    return {
        "windows": {"windows": 0.9, "couch": 0.01},
        "far couch, Stop next to the couch, in front of the windows": {
            "couch": 0.5, "windows": 0.05,
        },
    }


def test_episode_487_real_instruction_no_candidate() -> None:
    grounder = _StubGrounder(_QueryScores({}))
    manager = _manager(INSTRUCTION_487, grounder, geometry_fallback=False)
    manager.last_spatial_frame = _frame()

    context = _query(manager, INSTRUCTION_487, max_chars=320)
    assert context is None
    assert grounder.calls == []
    diagnostic = manager.last_l3_query_diagnostic
    assert diagnostic["injected_count"] == 0
    assert diagnostic["candidate_count"] == 0
    assert diagnostic["actual_injection"] is False
    assert diagnostic["skipped_reason"] in {
        "empty_map", "no_historical_geometry_object", "no_geometry_candidate",
    }
    targets = diagnostic["instruction_landmarks"]["target_objects"]
    assert targets == ["stairs"]


def test_episode_1080_real_instruction_couch_is_target_windows_reference() -> None:
    grounder = _StubGrounder(_QueryScores(_scores_1080()))
    manager = _manager(INSTRUCTION_1080, grounder)
    _fill_map(manager, _specs_1080())

    context = _query(manager, INSTRUCTION_1080, max_chars=4000)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    landmarks = diagnostic["instruction_landmarks"]
    assert landmarks["target_objects"] == ["couch"]
    assert "windows" in landmarks["reference_objects"]
    assert len(grounder.calls) == 3
    assert grounder.calls[0]["query"] == "windows"
    assert grounder.calls[1]["query"] == (
        "far couch, Stop next to the couch, in front of the windows"
    )
    # The target couch is injected before the window reference even though the
    # window query appears first in the instruction.
    assert diagnostic["object_ids"][0] == "object-couch-valid"
    assert "object-windows" in diagnostic["object_ids"]
    assert diagnostic["grounding_query"] == (
        "far couch, Stop next to the couch, in front of the windows"
    )
    assert diagnostic["grounding_query_source"] == "instruction_landmark"
    couch_query = next(
        item for item in diagnostic["grounding_queries"]
        if item["target_object"] == "couch"
    )
    assert couch_query["role"] == "target"
    proposals = {
        item["persistent_object_id"]: item
        for item in couch_query["proposals"]
    }
    assert proposals["object-couch-top"]["global_grounding_rank"] == 1
    assert proposals["object-couch-top"]["eligible_grounding_rank"] is None
    assert proposals["object-couch-top"]["rejection_reason"] == "not_formatable"
    assert proposals["object-couch-valid"]["global_grounding_rank"] == 2
    assert proposals["object-couch-valid"]["eligible_grounding_rank"] == 1
    assert couch_query["selected_is_global_top1"] is False
    assert couch_query["global_grounding_rank"] == 2
    assert couch_query["eligible_grounding_rank"] == 1
    assert diagnostic["selected_is_global_top1"] is False
    assert diagnostic["geometry_fallback_reason"] is None
    assert diagnostic["matches"][0]["selection_source"] == "vil3dref_grounding"


def test_episode_168_real_instruction_aliases_and_multi_object() -> None:
    specs = [
        ("object-doors", "double doors", (1.0, 0.0, 1.0), 0.9),
        ("object-dining", "dining table", (2.0, 0.0, 1.0), 0.9),
        ("object-entry", "entry way", (3.0, 0.0, 1.0), 0.9),
        ("object-couch", "couch", (4.0, 0.0, 1.0), 0.9),
        # Alias labels: chair ↔ armchair, table ↔ coffee table.
        ("object-armchairs", "chair", (5.0, 0.0, 1.0), 0.9),
        ("object-coffee", "table", (6.0, 0.0, 1.0), 0.9),
    ]
    scores = {
        "double doors": {"double doors": 0.9},
        "dining table": {"dining table": 0.9},
        "entry way": {"entry way": 0.9},
        "couch": {"couch": 0.9},
        "armchairs surrounding a coffee table": {"chair": 0.9, "table": 0.1},
        "coffee table": {"table": 0.9, "dining table": 0.01},
    }
    grounder = _StubGrounder(_QueryScores(scores))
    manager = _manager(INSTRUCTION_168, grounder)
    _fill_map(manager, specs)

    context = _query(manager, INSTRUCTION_168, max_chars=4000)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    assert len(grounder.calls) == 6
    assert set(diagnostic["object_ids"]) == {
        "object-doors", "object-dining", "object-entry", "object-couch",
        "object-armchairs", "object-coffee",
    }
    # Targets are rendered before the coffee-table reference.
    assert diagnostic["object_ids"][-1] == "object-coffee"
    landmarks = diagnostic["instruction_landmarks"]["landmarks"]
    armchairs = next(
        item for item in landmarks if item["target_object"] == "armchairs"
    )
    assert armchairs["reference_objects"] == ["coffee table"]
    assert armchairs["reference_phrase"] == "surrounding a coffee table"
    assert diagnostic["grounding_query"] == "double doors"
    assert diagnostic["geometry_fallback_reason"] is None


def test_episode_1012_real_instruction_sink_top_never_drops_targets() -> None:
    specs = [
        ("object-sink", "sink", (1.0, 0.0, 1.0), 0.9),
        ("object-bed", "bed", (2.0, 0.0, 1.0), 0.9),
        ("object-doorway", "doorway", (3.0, 0.0, 1.0), 0.9),
        ("object-stairs", "stairs", (4.0, 0.0, 1.0), 0.9),
    ]
    scores = {
        query: {"sink": 0.90, "bed": 0.05, "doorway": 0.03, "stairs": 0.02}
        for query in ("sink", "foot of bed", "doorway", "two stairs")
    }
    grounder = _StubGrounder(_QueryScores(scores))
    manager = _manager(INSTRUCTION_1012, grounder)
    _fill_map(manager, specs)

    context = _query(manager, INSTRUCTION_1012, max_chars=4000)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    assert len(grounder.calls) == 4
    assert set(diagnostic["object_ids"]) == {
        "object-sink", "object-bed", "object-doorway", "object-stairs",
    }
    # Every query had sink as the official global argmax.
    assert all(
        call["selected_object_id"] == "object-sink" for call in grounder.calls
    )
    # Targets are rendered first; sink is the source/reference anchor.
    assert diagnostic["object_ids"][0] == "object-bed"
    assert diagnostic["object_ids"][-1] == "object-sink"
    assert diagnostic["grounding_query"] == "foot of bed"
    landmarks = diagnostic["instruction_landmarks"]["landmarks"]
    assert [item["target_object"] for item in landmarks] == [
        "sink", "bed", "doorway", "stairs",
    ]
    by_object = {item["target_object"]: item for item in landmarks}
    assert by_object["sink"]["role"] == "reference"
    assert by_object["sink"]["action_phrase"] == "walk away from"
    assert by_object["bed"]["role"] == "target"
    assert by_object["bed"]["action_phrase"] == "walk towards"
    assert by_object["doorway"]["action_phrase"] == "exit through"
    assert by_object["stairs"]["action_phrase"] == "step down"
    assert "up" not in " ".join(
        item["action_phrase"] or "" for item in landmarks
    ).casefold().split()
    assert diagnostic["geometry_fallback_reason"] is None


def test_episode_1012_all_refusal_uses_geometry_fallback() -> None:
    specs = [
        ("object-sink", "sink", (1.0, 0.0, 1.0), 0.9),
        ("object-bed", "bed", (2.0, 0.0, 1.0), 0.9),
        ("object-doorway", "doorway", (3.0, 0.0, 1.0), 0.9),
        ("object-stairs", "stairs", (4.0, 0.0, 1.0), 0.9),
    ]
    grounder = _RefusingGrounder()
    manager = _manager(INSTRUCTION_1012, grounder)
    _fill_map(manager, specs)

    context = _query(manager, INSTRUCTION_1012, max_chars=4000)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    assert grounder.calls == 4
    assert diagnostic["actual_injection"] is True
    assert diagnostic["geometry_fallback_reason"]
    assert set(diagnostic["selection_sources"]) == {"geometry_fallback"}
    assert diagnostic["selected_valid_proposals"] == []
    assert len(context.layer_items) == 4
    assert all(
        item["selection_source"] == "geometry_fallback"
        for item in diagnostic["matches"]
    )


def test_geometry_fallback_equals_original_geometry_path() -> None:
    specs = [
        ("object-sink", "sink", (1.0, 0.0, 1.0), 0.9),
        ("object-bed", "bed", (2.0, 0.0, 1.0), 0.9),
        ("object-doorway", "doorway", (3.0, 0.0, 1.0), 0.9),
        ("object-stairs", "stairs", (4.0, 0.0, 1.0), 0.9),
    ]
    control = _geometry_control_manager(INSTRUCTION_1012, max_landmarks=2)
    _fill_map(control, specs)
    control_context = _query(control, INSTRUCTION_1012, max_chars=4000)
    assert control_context is not None

    grounder = _RefusingGrounder()
    fallback = _manager(
        INSTRUCTION_1012, grounder, max_landmarks=2,
        landmark_queries=False, geometry_fallback=True,
    )
    _fill_map(fallback, specs)
    fallback_context = _query(fallback, INSTRUCTION_1012, max_chars=4000)
    assert fallback_context is not None
    assert fallback_context.text == control_context.text
    assert [
        (item.metadata["object_id"], item.description)
        for item in fallback_context.layer_items
    ] == [
        (item.metadata["object_id"], item.description)
        for item in control_context.layer_items
    ]
    assert fallback.last_l3_query_diagnostic["geometry_fallback_reason"]
    assert fallback.last_l3_query_diagnostic["selection_mode"] == "eligible"


def test_unbounded_full_context_does_not_bypass_history_or_budget() -> None:
    # The highest-scoring node has no legal observed_from history; the
    # unbounded eligibility probe must still exclude it.
    specs = [
        ("object-couch-valid", "couch", (2.0, 0.0, 2.0), 0.9),
        ("object-couch-no-history", "couch", (2.0, 0.0, -2.0), 0.9),
    ]
    grounder = _StubGrounder(_QueryScores({
        "far couch, Stop next to the couch, in front of the windows": {
            "couch": 0.5,
        },
    }))
    manager = _manager(INSTRUCTION_1080, grounder)
    _fill_map(manager, specs)
    object_map = manager.spatial_map
    assert object_map is not None
    object_map.edges = [
        edge for edge in object_map.edges
        if edge.object_id != "object-couch-no-history"
    ]

    context = _query(manager, INSTRUCTION_1080, max_chars=4000)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    assert "object-couch-no-history" not in diagnostic["object_ids"]
    couch_query = next(
        item for item in diagnostic["grounding_queries"]
        if item["target_object"] == "couch"
    )
    no_history = next(
        item for item in couch_query["proposals"]
        if item["persistent_object_id"] == "object-couch-no-history"
    )
    assert no_history["object_node_valid"] is True
    assert no_history["formatable"] is False
    assert no_history["rejection_reason"] == "not_formatable"

    # Even the unbounded internal candidate set never bypasses the final
    # character budget.
    budget_manager = _manager(INSTRUCTION_168, _StubGrounder(_QueryScores({})))
    _fill_map(budget_manager, [
        ("object-couch", "couch", (4.0, 0.0, 1.0), 0.9),
    ])
    tiny = _query(budget_manager, INSTRUCTION_168, max_chars=40)
    assert tiny is None
    assert budget_manager.last_l3_query_diagnostic["skipped_reason"] == (
        "prompt_budget_excluded"
    )
    assert budget_manager.last_l3_query_diagnostic["actual_injection"] is False


def test_geometry_fallback_never_reintroduces_another_target_class() -> None:
    # Only the bed exists.  The sink/source query must not pick bed even with
    # fallback enabled.
    specs = [("object-bed", "bed", (2.0, 0.0, 1.0), 0.9)]
    grounder = _RefusingGrounder()
    manager = _manager(INSTRUCTION_1012, grounder)
    _fill_map(manager, specs)

    context = _query(manager, INSTRUCTION_1012, max_chars=4000)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    assert diagnostic["object_ids"] == ["object-bed"]
    sink_query = next(
        item for item in diagnostic["grounding_queries"]
        if item["target_object"] == "sink"
    )
    assert sink_query["selected_object_id"] is None
    assert sink_query["fallback_reason"]


def test_ablation_c_argmax_matches_original_single_fact_refusal() -> None:
    grounder = _StubGrounder(_QueryScores({
        INSTRUCTION_1080: {"couch": 0.5, "windows": 0.05},
    }))
    manager = _manager(
        INSTRUCTION_1080, grounder, selection_mode="argmax",
        landmark_queries=False, geometry_fallback=False,
    )
    _fill_map(manager, _specs_1080())
    context = _query(manager, INSTRUCTION_1080, max_chars=4000)
    assert context is None
    diagnostic = manager.last_l3_query_diagnostic
    assert len(grounder.calls) == 1
    assert grounder.calls[0]["query"] == INSTRUCTION_1080
    assert diagnostic["selection_mode"] == "argmax"
    # couch-top is the global argmax and is not formatable.
    assert diagnostic["skipped_reason"] == "selected_object_not_formatable"
    assert diagnostic["actual_injection"] is False


def test_ablation_d_eligible_without_landmarks_or_fallback() -> None:
    grounder = _StubGrounder(_QueryScores({
        INSTRUCTION_1080: {"couch": 0.5, "windows": 0.05},
    }))
    manager = _manager(
        INSTRUCTION_1080, grounder, selection_mode="eligible",
        landmark_queries=False, geometry_fallback=False,
    )
    _fill_map(manager, _specs_1080())
    context = _query(manager, INSTRUCTION_1080, max_chars=4000)
    assert context is not None
    assert len(grounder.calls) == 1
    assert context.layer_items[0].metadata["object_id"] == "object-couch-valid"
    assert manager.last_l3_query_diagnostic["selection_mode"] == "eligible"


def test_ablation_e_eligible_landmarks_without_fallback_refuses() -> None:
    grounder = _RefusingGrounder()
    manager = _manager(
        INSTRUCTION_1012, grounder, selection_mode="eligible",
        landmark_queries=True, geometry_fallback=False,
    )
    _fill_map(manager, [
        ("object-sink", "sink", (1.0, 0.0, 1.0), 0.9),
        ("object-bed", "bed", (2.0, 0.0, 1.0), 0.9),
    ])
    context = _query(manager, INSTRUCTION_1012, max_chars=4000)
    assert context is None
    diagnostic = manager.last_l3_query_diagnostic
    assert diagnostic["actual_injection"] is False
    assert diagnostic["rejection_reason"]


def test_ablation_f_eligible_landmarks_with_fallback() -> None:
    manager = _manager(
        INSTRUCTION_1012, _RefusingGrounder(), selection_mode="eligible",
        landmark_queries=True, geometry_fallback=True,
    )
    _fill_map(manager, [
        ("object-bed", "bed", (2.0, 0.0, 1.0), 0.9),
    ])
    context = _query(manager, INSTRUCTION_1012, max_chars=4000)
    assert context is not None
    assert context.layer_items[0].metadata["object_id"] == "object-bed"
    assert manager.last_l3_query_diagnostic["geometry_fallback_reason"]


def test_diagnostics_never_renormalize_probabilities() -> None:
    # These probabilities intentionally do not sum to one.
    grounder = _FixedGrounder({
        "object-couch-top": 0.6,
        "object-couch-valid": 0.2,
        "object-windows": 0.1,
    })
    manager = _manager(INSTRUCTION_1080, grounder)
    _fill_map(manager, _specs_1080())
    context = _query(manager, INSTRUCTION_1080, max_chars=4000)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    couch_query = next(
        item for item in diagnostic["grounding_queries"]
        if item["target_object"] == "couch"
    )
    assert couch_query["original_top1_probability"] == 0.6
    assert couch_query["selected_probability"] == 0.2
    assert couch_query["selected_is_global_top1"] is False
    assert couch_query["global_grounding_rank"] == 2
    assert couch_query["eligible_grounding_rank"] == 1
    assert diagnostic["original_top1_probability"] == 0.6
    assert diagnostic["selected_probability"] == 0.2


def test_instruction_landmarks_are_parsed_once_per_episode(monkeypatch) -> None:
    import lightnav.memory.manager as manager_module
    from lightnav.memory.instruction_landmarks import (
        parse_instruction_landmarks as real_parse,
    )

    calls: list[str] = []

    def counted(instruction, **kwargs):
        calls.append(instruction)
        return real_parse(instruction, **kwargs)

    monkeypatch.setattr(manager_module, "parse_instruction_landmarks", counted)
    manager = _manager(
        INSTRUCTION_1080, _StubGrounder(_QueryScores(_scores_1080())),
    )
    _fill_map(manager, _specs_1080())
    assert len(calls) == 1
    _query(manager, INSTRUCTION_1080, max_chars=4000)
    assert len(calls) == 1
    manager.reset_episode(EPISODE, INSTRUCTION_1080, scene_id=SCENE)
    assert len(calls) == 2


def test_full_instruction_fallback_when_landmarks_are_disabled() -> None:
    grounder = _StubGrounder(_QueryScores({
        INSTRUCTION_1080: {"couch": 0.9},
    }))
    manager = _manager(
        INSTRUCTION_1080, grounder, landmark_queries=False,
    )
    _fill_map(manager, [
        ("object-couch", "couch", (2.0, 0.0, 2.0), 0.9),
    ])
    context = _query(manager, INSTRUCTION_1080, max_chars=4000)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    assert len(grounder.calls) == 1
    assert grounder.calls[0]["query"] == INSTRUCTION_1080
    assert diagnostic["grounding_query"] == INSTRUCTION_1080
    assert diagnostic["grounding_query_source"] == "full_instruction_fallback"
    assert diagnostic["object_ids"] == ["object-couch"]


def test_label_identity_aliases_and_boundaries() -> None:
    assert label_matches_target("window", "windows")
    assert label_matches_target("windows", "window")
    assert label_matches_target("sofa", "couch")
    assert label_matches_target("couch", "sofa")
    assert label_matches_target("chair", "armchairs")
    assert label_matches_target("armchair", "chair")
    assert label_matches_target("table", "dining table")
    assert label_matches_target("table", "coffee table")
    assert not label_matches_target("dining table", "coffee table")
    assert not label_matches_target("coffee table", "dining table")
    assert not label_matches_target("couch", "chair")


def test_grounder_load_error_respects_fallback_switch(monkeypatch) -> None:
    manager = _manager(INSTRUCTION_1012, _StubGrounder(_QueryScores({})))
    _fill_map(manager, [("object-bed", "bed", (2.0, 0.0, 1.0), 0.9)])

    def _fail():
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(manager, "_get_instance_grounder", _fail)
    context = _query(manager, INSTRUCTION_1012, max_chars=4000)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    assert diagnostic["geometry_fallback_reason"] == "grounder_load_error"
    assert "checkpoint unavailable" in diagnostic["error"]
    assert diagnostic["actual_injection"] is True

    strict = _manager(
        INSTRUCTION_1012, _StubGrounder(_QueryScores({})),
        geometry_fallback=False,
    )
    _fill_map(strict, [("object-bed", "bed", (2.0, 0.0, 1.0), 0.9)])
    monkeypatch.setattr(strict, "_get_instance_grounder", _fail)
    strict_context = _query(strict, INSTRUCTION_1012, max_chars=4000)
    assert strict_context is None
    assert strict.last_l3_query_diagnostic["skipped_reason"] == (
        "grounder_load_error"
    )
    assert strict.last_l3_query_diagnostic["actual_injection"] is False


class _FlakyGrounder:
    def __init__(self, inner: _StubGrounder, failing_query: str) -> None:
        self.inner = inner
        self.failing_query = failing_query
        self.calls = inner.calls

    def select(self, *, instruction, object_map, sidecar, step, episode_id):
        if instruction == self.failing_query:
            self.calls.append({"query": instruction, "error": "boom"})
            raise RuntimeError("one query failed")
        return self.inner.select(
            instruction=instruction, object_map=object_map, sidecar=sidecar,
            step=step, episode_id=episode_id,
        )


def test_one_failed_landmark_query_never_drops_the_other_landmarks() -> None:
    scorer = _QueryScores({
        "sink": {"sink": 0.9},
        "foot of bed": {"bed": 0.9},
        "doorway": {"doorway": 0.9},
        "two stairs": {"stairs": 0.9},
    })
    grounder = _FlakyGrounder(_StubGrounder(scorer), "sink")
    manager = _manager(INSTRUCTION_1012, grounder)
    _fill_map(manager, [
        ("object-sink", "sink", (1.0, 0.0, 1.0), 0.9),
        ("object-bed", "bed", (2.0, 0.0, 1.0), 0.9),
        ("object-doorway", "doorway", (3.0, 0.0, 1.0), 0.9),
        ("object-stairs", "stairs", (4.0, 0.0, 1.0), 0.9),
    ])
    context = _query(manager, INSTRUCTION_1012, max_chars=4000)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    assert set(diagnostic["object_ids"]) == {
        "object-bed", "object-doorway", "object-stairs",
    }
    sink_query = next(
        item for item in diagnostic["grounding_queries"]
        if item["target_object"] == "sink"
    )
    assert sink_query["error"] == "RuntimeError: one query failed"
    assert diagnostic["grounding_query"] == "foot of bed"
    assert manager.l3_query_stats_summary()["queries_with_match"] == 1


class _InstructionProvider:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, instruction):
        self.calls += 1
        return [{"target_phrase": "far couch"}]


def test_existing_instruction_provider_is_reused_once_per_episode() -> None:
    provider = _InstructionProvider()
    grounder = _StubGrounder(_QueryScores({
        "far couch": {"couch": 0.9},
    }))
    manager = _manager(
        INSTRUCTION_1080, grounder, instruction_landmark_provider=provider,
    )
    _fill_map(manager, [
        ("object-couch", "couch", (2.0, 0.0, 2.0), 0.9),
    ])
    context = _query(manager, INSTRUCTION_1080, max_chars=4000)
    assert context is not None
    assert provider.calls == 1
    assert manager.instruction_landmarks.source == "provider"
    assert manager.last_l3_query_diagnostic["grounding_query"] == "far couch"
