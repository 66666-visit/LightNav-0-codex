"""CPU regressions for ordered landmark memory on Habitat Episode 1012.

The tests use the real 1012 instruction and synthetic ObjectMaps.  They never
load a checkpoint, CUDA, Habitat or a real detector.  The central regression is
the reported D/E/F pattern: with only a bed ObjectNode, D/E/F all inject the
same one-line bed fact (so a greedy model cannot act differently), while the
ordered plan keeps ``source sink`` / ``transition doorway`` / ``final two
stairs`` semantics even though those classes have no 3D evidence.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

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
from lightnav.memory.instruction_landmarks import parse_instruction_landmarks
from lightnav.memory.l3_evidence_selector import label_matches_target
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
from lightnav.memory.ordered_landmark_memory import (
    LandmarkMemoryRow,
    render_ordered_landmark_memory,
)
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.token_budget import (
    select_context_lines,
    trim_context_for_overflow,
)
from lightnav.memory.v3_schemas import canonical_evidence_key

SCENE = "scene"
EPISODE = "episode"
INSTRUCTION_1012 = (
    "walk away from sink, turn left, walk towards foot of bed, "
    "exit through doorway on left, turn right and step down "
    "two stairs and stop."
)
QUERIES_1012 = ("sink", "foot of bed", "doorway", "two stairs")
REQUIRED_ROW_FIELDS = {
    "landmark_role",
    "landmark_order",
    "landmark_phrase",
    "grounding_query",
    "grounder_proposal_count",
    "eligible_candidate_count",
    "selected_object_id",
    "evidence_ids",
    "geometry_fallback_attempted",
    "geometry_fallback_reason",
    "selection_source",
    "actual_injection",
    "actual_injected_evidence_ids",
}


class _SegExtractor:
    def reset_episode(self, episode_id, instruction):
        return None

    def close_episode(self):
        return None

    def extract(self, rgb, step_id):
        return []


class _QueryScores:
    """One raw official score per class for each landmark query."""

    def __init__(self, scores: dict[str, dict[str, float]], default: float = 0.01):
        self.scores = scores
        self.default = default
        self.calls: list[str] = []

    def __call__(self, query: str, proposals) -> dict[str, float]:
        self.calls.append(query)
        raw = {
            proposal.object_id: float(
                self.scores.get(query, {}).get(proposal.class_name, self.default)
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
        proposals, _skipped = sidecar.proposals(object_map.objects)
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
        selected = (
            max(scores, key=lambda score: score.probability).object_id
            if scores else None
        )
        self.calls.append({
            "query": instruction,
            "selected_object_id": selected,
            "proposal_count": len(scores),
        })
        return InstanceGroundingResult(
            selected_object_id=selected,
            selected_index=0 if scores else None,
            proposals=scores,
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
    grounder, *, ordered: bool = True, landmark_queries: bool = True,
    geometry_fallback: bool = True, selection_mode: str = "eligible",
    instruction: str = INSTRUCTION_1012,
) -> MemoryManager:
    manager = MemoryManager(
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True, max_landmarks=2),
        spatial_l3_config=SpatialL3Config(
            mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
            seg_confidence=0.25, seg_every_n_steps=5,
            min_injection_confidence=0.5, rgb_sidecar_enabled=True,
            instance_grounder_enabled=True,
            grounder_checkpoint="/fake/official_student.pt",
            grounder_device="cpu",
            grounder_selection_mode=selection_mode,
            grounder_landmark_queries=landmark_queries,
            grounder_geometry_fallback=geometry_fallback,
            grounder_ordered_landmark_memory=ordered,
        ),
        spatial_extractor=_SegExtractor(), clip_encoder=None,
        prompt_mode="l1_original_only", instance_grounder=grounder,
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


def _query(
    manager: MemoryManager, *, max_chars: int = 320,
    instruction: str = INSTRUCTION_1012,
):
    return manager.get_geometry_on_demand_context(
        step=2, instruction=instruction, l1_status="no_candidates",
        max_chars=max_chars,
    )


def _bed_only_scores() -> dict[str, dict[str, float]]:
    return {
        query: {"bed": 0.9}
        for query in QUERIES_1012
    }


def _no_coordinate(line: str) -> bool:
    return "world (" not in line and " line," not in line


# --------------------------------------------------------------------------- A


def test_1012_ordered_roles_are_verbatim_and_in_instruction_order() -> None:
    parsed = parse_instruction_landmarks(INSTRUCTION_1012)
    assert [item.navigation_role for item in parsed.landmarks] == [
        "source", "intermediate", "transition", "final",
    ]
    assert [item.landmark_order for item in parsed.landmarks] == [1, 2, 3, 4]
    assert [item.target_phrase for item in parsed.landmarks] == [
        "sink", "foot of bed", "doorway", "two stairs",
    ]
    assert [item.action_phrase for item in parsed.landmarks] == [
        "walk away from", "walk towards", "exit through", "step down",
    ]
    for item in parsed.landmarks:
        assert INSTRUCTION_1012[
            item.target_span[0]:item.target_span[1]
        ] == item.target_phrase
        assert INSTRUCTION_1012[
            item.action_span[0]:item.action_span[1]
        ] == item.action_phrase
    assert parsed.instruction == INSTRUCTION_1012


def test_roles_are_text_derived_not_episode_id_derived() -> None:
    instruction = (
        "walk towards the table, exit through the door, "
        "step down the stairs and stop."
    )
    parsed = parse_instruction_landmarks(instruction)
    assert [item.target_phrase for item in parsed.landmarks] == [
        "table", "door", "stairs",
    ]
    assert [item.navigation_role for item in parsed.landmarks] == [
        "intermediate", "transition", "final",
    ]
    # Swapping clause order changes the parsed order, not a lookup table.
    swapped = parse_instruction_landmarks(
        "exit through the door, walk towards the table, step down the stairs."
    )
    assert [item.navigation_role for item in swapped.landmarks] == [
        "transition", "intermediate", "final",
    ]


def test_structural_aliases_match_only_the_same_landmark_category() -> None:
    assert label_matches_target("door", "doorway")
    assert label_matches_target("doorway", "door")
    assert label_matches_target("staircase", "two stairs")
    assert label_matches_target("stairway", "stairs")
    assert not label_matches_target("door", "window")
    assert not label_matches_target("staircase", "doorway")
    assert not label_matches_target("coffee table", "dining table")


# --------------------------------------------------------------------------- B


def test_every_landmark_has_explicit_retrieval_diagnostics() -> None:
    grounder = _StubGrounder(_QueryScores(_bed_only_scores()))
    manager = _manager(grounder)
    _fill_map(manager, [("object-bed", "bed", (2.0, 0.0, 1.0), 0.9)])

    context = _query(manager)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    rows = diagnostic["ordered_landmarks"]
    assert len(rows) == 4
    assert [row["landmark_role"] for row in rows] == [
        "source", "intermediate", "transition", "final",
    ]
    assert [row["landmark_order"] for row in rows] == [1, 2, 3, 4]
    for row in rows:
        assert REQUIRED_ROW_FIELDS <= set(row)
        assert row["grounding_query"] in QUERIES_1012
        assert row["query_executed"] is True
    by_role = {row["landmark_role"]: row for row in rows}
    assert by_role["intermediate"]["selected_object_id"] == "object-bed"
    assert by_role["intermediate"]["selection_source"] == "vil3dref_grounding"
    assert by_role["intermediate"]["evidence_ids"]
    # Valid Grounding wins; fallback is not attempted for that landmark.
    assert by_role["intermediate"]["geometry_fallback_attempted"] is False
    assert by_role["intermediate"]["actual_injection"] is True
    for role in ("source", "transition", "final"):
        assert by_role[role]["selected_object_id"] is None
        assert by_role[role]["geometry_fallback_attempted"] is True
        assert by_role[role]["geometry_fallback_reason"] == (
            "no_same_category_geometry_item"
        )
        assert by_role[role]["actual_injection"] is False
        assert by_role[role]["actual_injected_evidence_ids"] == []
    # F's previously invisible fallback outcome is now explicit.
    assert diagnostic["geometry_fallback_attempted"] is True
    assert grounder.calls


def test_geometry_fallback_never_crosses_category_and_alias_is_used() -> None:
    scores = {
        "sink": {"bed": 0.9},
        "foot of bed": {"bed": 0.9},
        "doorway": {"door": 0.9, "bed": 0.1},
        "two stairs": {"stairs": 0.9, "bed": 0.1},
    }
    grounder = _StubGrounder(_QueryScores(scores))
    manager = _manager(grounder)
    _fill_map(manager, [
        ("object-bed", "bed", (2.0, 0.0, 1.0), 0.9),
        ("object-door", "door", (3.0, 0.0, 1.0), 0.9),
        ("object-stairs", "stairs", (4.0, 0.0, 1.0), 0.9),
    ])

    context = _query(manager)
    assert context is not None
    rows = {
        row["landmark_role"]: row
        for row in manager.last_l3_query_diagnostic["ordered_landmarks"]
    }
    # The sink query must not fall back to the bed; the doorway query may use
    # the structural door alias, and the final stairs query uses stairs.
    assert rows["source"]["selected_object_id"] is None
    assert rows["source"]["geometry_fallback_reason"] == (
        "no_same_category_geometry_item"
    )
    assert rows["transition"]["selected_object_id"] == "object-door"
    assert rows["final"]["selected_object_id"] == "object-stairs"


def test_unexecuted_map_unchanged_query_records_why_no_memory_appeared() -> None:
    grounder = _StubGrounder(_QueryScores(_bed_only_scores()))
    manager = _manager(grounder)
    _fill_map(manager, [("object-bed", "bed", (2.0, 0.0, 1.0), 0.9)])
    assert _query(manager) is not None
    calls = len(grounder.calls)

    second = _query(manager)
    assert second is None
    diagnostic = manager.last_l3_query_diagnostic
    assert diagnostic["query_executed"] is False
    assert diagnostic["skipped_reason"] == "map_unchanged_since_last_query"
    assert len(grounder.calls) == calls
    rows = diagnostic["ordered_landmarks"]
    assert len(rows) == 4
    assert all(row["query_executed"] is False for row in rows)
    assert all(
        row["query_skip_reason"] == "map_unchanged_since_last_query"
        for row in rows
    )


# --------------------------------------------------------------------------- C


def test_ordered_memory_keeps_all_landmarks_without_fake_coordinates() -> None:
    grounder = _StubGrounder(_QueryScores(_bed_only_scores()))
    manager = _manager(grounder)
    _fill_map(manager, [("object-bed", "bed", (2.0, 0.0, 1.0), 0.9)])

    context = _query(manager)
    assert context is not None
    assert len(context.text) <= 320
    lines = context.text.splitlines()
    assert lines[0] == "[Navigation Memory]"
    assert len(lines) == 5
    assert [line.split(" ", 1)[0] for line in lines[1:]] == ["1", "2", "3", "4"]
    assert '"sink"' in lines[1] and "source" in lines[1]
    assert '"foot of bed"' in lines[2] and "world (" in lines[2]
    assert '"doorway"' in lines[3] and "transition" in lines[3]
    assert '"two stairs"' in lines[4] and "final" in lines[4]
    for line in (lines[1], lines[3], lines[4]):
        assert _no_coordinate(line)
    # The bed fact is the only one with real coordinates.
    assert context.text.count("world (") == 1
    # The policy's prompt contract requires every accepted line to be present.
    for item in context.layer_items:
        assert item.metadata["serialized_text"] in context.text
        assert item.description in context.text


def test_final_landmark_fact_gets_budget_priority_over_first_target() -> None:
    scores = {
        "sink": {"bed": 0.9},
        "foot of bed": {"bed": 0.9},
        "doorway": {"door": 0.9, "bed": 0.1},
        "two stairs": {"stairs": 0.9, "bed": 0.1},
    }
    specs = [
        ("object-bed", "bed", (2.0, 0.0, 1.0), 0.9),
        ("object-door", "door", (3.0, 0.0, 1.0), 0.9),
        ("object-stairs", "stairs", (4.0, 0.0, 1.0), 0.9),
    ]
    old = _manager(_StubGrounder(_QueryScores(scores)), ordered=False)
    _fill_map(old, specs)
    old_context = _query(old)
    assert old_context is not None
    # The old renderer's first target consumes the whole 320-char budget.
    assert "bed observed" in old_context.text
    assert "door observed" not in old_context.text
    assert "stairs observed" not in old_context.text

    new = _manager(_StubGrounder(_QueryScores(scores)), ordered=True)
    _fill_map(new, specs)
    new_context = _query(new)
    assert new_context is not None
    assert len(new_context.text) <= 320
    assert "door at world" in new_context.text
    assert "stairs at world" in new_context.text
    assert "bed at world" not in new_context.text
    # The bed still keeps its instruction semantics; only its fact was dropped.
    assert '"foot of bed"' in new_context.text


def _geometry_control_manager() -> MemoryManager:
    """Original Geometry On-demand arm B: no sidecar, no Grounder."""
    manager = MemoryManager(
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True, max_landmarks=2),
        spatial_l3_config=SpatialL3Config(
            mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
            seg_confidence=0.25, seg_every_n_steps=5,
            min_injection_confidence=0.5, rgb_sidecar_enabled=False,
            instance_grounder_enabled=False,
        ),
        spatial_extractor=_SegExtractor(), clip_encoder=None,
        prompt_mode="l1_original_only",
    )
    manager.reset_episode(EPISODE, INSTRUCTION_1012, scene_id=SCENE)
    return manager


def test_original_geometry_can_inject_the_final_stairs_when_it_is_closest() -> None:
    # At the decisive step the final stairs can be the closest relevant object.
    # The original geometry path sorts by distance and renders it first; the
    # D/E/F path sorts targets by instruction order and spends the 320-char
    # budget on the bed first.  This is the code-level difference that can make
    # B's memory behavior-relevant while E/F's is not.
    specs = [
        ("object-stairs", "stairs", (0.5, 0.0, 1.0), 0.9),
        ("object-bed", "bed", (10.0, 0.0, 1.0), 0.9),
        ("object-sink", "sink", (10.0, 0.0, 2.0), 0.9),
    ]
    control = _geometry_control_manager()
    _fill_map(control, specs)
    control_context = _query(control)
    assert control_context is not None
    assert "stairs observed" in control_context.text
    assert "bed observed" not in control_context.text

    for ordered in (False, True):
        manager = _manager(
            _StubGrounder(_QueryScores({
                query: {"bed": 0.9}
                for query in QUERIES_1012
            })), ordered=ordered,
        )
        _fill_map(manager, specs)
        context = _query(manager)
        assert context is not None
        assert "bed" in context.text
        if not ordered:
            # The old renderer spends the whole budget on the first target.
            assert "stairs observed" not in context.text
            assert "stairs at world" not in context.text
        else:
            # The ordered renderer enriches the final landmark first.
            assert "stairs at world" in context.text
            assert '"two stairs"' in context.text


def test_d_e_f_inject_the_same_bed_fact_while_ordered_differs() -> None:
    specs = [("object-bed", "bed", (2.0, 0.0, 1.0), 0.9)]
    texts = {}
    for label, kwargs in {
        "D": {"ordered": False, "landmark_queries": False,
              "geometry_fallback": False},
        "E": {"ordered": False, "landmark_queries": True,
              "geometry_fallback": False},
        "F": {"ordered": False, "landmark_queries": True,
              "geometry_fallback": True},
        "G": {"ordered": True, "landmark_queries": True,
              "geometry_fallback": True},
    }.items():
        manager = _manager(
            _StubGrounder(_QueryScores(_bed_only_scores())), **kwargs,
        )
        _fill_map(manager, specs)
        context = _query(manager)
        assert context is not None, label
        texts[label] = context.text

    # D/E/F differ only in selection machinery; with only the bed eligible
    # they build byte-identical memory.  Greedy decoding therefore cannot
    # change action between D/E/F.
    assert texts["D"] == texts["E"] == texts["F"]
    assert hashlib.sha256(texts["D"].encode()).hexdigest() == hashlib.sha256(
        texts["F"].encode()
    ).hexdigest()
    assert texts["G"] != texts["D"]
    assert '"two stairs"' in texts["G"]


def test_ordered_memory_records_geometry_fallback_provenance() -> None:
    manager = _manager(_RefusingGrounder())
    _fill_map(manager, [
        ("object-bed", "bed", (2.0, 0.0, 1.0), 0.9),
        ("object-door", "door", (3.0, 0.0, 1.0), 0.9),
        ("object-stairs", "stairs", (4.0, 0.0, 1.0), 0.9),
    ])
    context = _query(manager)
    assert context is not None
    diagnostic = manager.last_l3_query_diagnostic
    assert diagnostic["geometry_fallback_attempted"] is True
    assert "geometry_fallback" in diagnostic["selection_sources"]
    assert any(
        item["source"] == "geometry_fallback"
        for item in diagnostic["injection_provenance"]
    )
    assert diagnostic["actual_injection"] is True
    assert diagnostic["actual_injected_evidence_ids"]
    # The final landmark is fact-enriched first under the 320-char budget.
    final = next(
        row for row in diagnostic["ordered_landmarks"]
        if row["landmark_role"] == "final"
    )
    assert final["selection_source"] == "geometry_fallback"
    assert final["spatial_fact_included"] is True
    assert final["actual_injected_evidence_ids"]


def test_ordered_mode_requires_the_full_eligible_landmark_contract() -> None:
    base = {
        "mode": "geometry_on_demand", "seg_model_path": "/fake/seg.pt",
        "rgb_sidecar_enabled": True, "instance_grounder_enabled": True,
        "grounder_checkpoint": "/fake/student.pt",
    }
    with pytest.raises(ValueError, match="grounder_selection_mode=eligible"):
        SpatialL3Config(**{**base, "grounder_ordered_landmark_memory": True})
    with pytest.raises(ValueError, match="grounder_landmark_queries"):
        SpatialL3Config(**{
            **base, "grounder_selection_mode": "eligible",
            "grounder_ordered_landmark_memory": True,
        })


def test_168_eligible_selection_is_retained_by_ordered_memory() -> None:
    instruction_168 = (
        "Walk through double doors into the house. Continue around the dining "
        "table and through the entry way to the next room. Walk up to the couch "
        "and armchairs surrounding a coffee table."
    )
    specs = [
        ("object-doors", "double doors", (1.0, 0.0, 1.0), 0.9),
        ("object-dining", "dining table", (2.0, 0.0, 1.0), 0.9),
        ("object-entry", "entry way", (3.0, 0.0, 1.0), 0.9),
        ("object-couch", "couch", (4.0, 0.0, 1.0), 0.9),
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
    manager = _manager(grounder, instruction=instruction_168)
    _fill_map(manager, specs)
    context = _query(manager, instruction=instruction_168)
    assert context is not None
    assert len(context.text) <= 320
    diagnostic = manager.last_l3_query_diagnostic
    assert len(grounder.calls) == 6
    rows = {row["landmark_phrase"]: row for row in diagnostic["ordered_landmarks"]}
    assert set(rows) == {
        "double doors", "dining table", "entry way", "couch",
        "armchairs", "coffee table",
    }
    assert rows["armchairs"]["selected_object_id"] == "object-armchairs"
    assert rows["coffee table"]["selected_object_id"] == "object-coffee"
    assert rows["double doors"]["selection_source"] == "vil3dref_grounding"
    # All six landmarks keep their ordered semantics in the compact plan line;
    # the remaining budget is spent on verified facts (final first).
    assert "Plan:" in context.text
    for phrase in (
        '"double doors"', '"dining table"', '"entry way"', '"couch"',
        '"armchairs"', '"coffee table"',
    ):
        assert phrase in context.text
    assert context.text.count("\n") >= 2


# ------------------------------------------------------------------ rendering


def _row(**overrides) -> LandmarkMemoryRow:
    values = {
        "landmark_order": 1,
        "landmark_role": "final",
        "landmark_phrase": "two stairs",
        "target_object": "stairs",
        "action_phrase": "step down",
        "grounding_query": "two stairs",
        "evidence_ids": ("ev-1",),
        "selected_object_id": "object-stairs",
        "selection_source": "vil3dref_grounding",
        "spatial_fact": "Spatial history: stairs observed at world (0.0,0.0,0.0) m; ...",
        "spatial_label": "stairs",
        "center_world": (0.0, 0.0, 0.0),
        "distance_m": 1.0,
        "bearing_deg": 0.0,
        "observation_count": 1,
        "evidence_step": 1,
        "evidence_confidence": 0.9,
        "evidence_source": "unit-test",
    }
    values.update(overrides)
    return LandmarkMemoryRow(**values)


def test_renderer_never_exceeds_budget_or_invents_coordinates() -> None:
    rows = [
        _row(
            landmark_order=1, landmark_role="source", landmark_phrase="sink",
            target_object="sink", evidence_ids=(), selected_object_id=None,
            selection_source=None, spatial_fact=None, spatial_label=None,
            center_world=None, distance_m=None, bearing_deg=None,
        ),
        _row(
            landmark_order=2, landmark_role="intermediate",
            landmark_phrase="foot of bed", target_object="bed",
            evidence_ids=("ev-bed",), selected_object_id="object-bed",
            spatial_label="bed", center_world=(2.0, 0.0, 1.0),
            distance_m=2.0, bearing_deg=-30.0,
        ),
        _row(
            landmark_order=3, landmark_role="transition",
            landmark_phrase="doorway", target_object="doorway",
            evidence_ids=(), selected_object_id=None, selection_source=None,
            spatial_fact=None, spatial_label=None, center_world=None,
            distance_m=None, bearing_deg=None,
        ),
        _row(landmark_order=4),
    ]
    for budget in (320, 300, 260):
        text, included = render_ordered_landmark_memory(rows, max_chars=budget)
        assert text is not None
        assert len(text) <= budget
        assert included
        for line in text.splitlines()[1:]:
            if '"sink"' in line or '"doorway"' in line:
                assert _no_coordinate(line)
    text, included = render_ordered_landmark_memory(rows, max_chars=60)
    assert text is None or len(text) <= 60
    assert all(len(row.skeleton_line()) > 0 for row in rows)


def test_existing_token_selector_never_exceeds_capacity() -> None:
    scores = {
        "sink": {"bed": 0.9},
        "foot of bed": {"bed": 0.9},
        "doorway": {"door": 0.9, "bed": 0.1},
        "two stairs": {"stairs": 0.9, "bed": 0.1},
    }
    manager = _manager(_StubGrounder(_QueryScores(scores)))
    _fill_map(manager, [
        ("object-bed", "bed", (2.0, 0.0, 1.0), 0.9),
        ("object-door", "door", (3.0, 0.0, 1.0), 0.9),
        ("object-stairs", "stairs", (4.0, 0.0, 1.0), 0.9),
    ])
    context = _query(manager)
    assert context is not None
    def counter(text: str) -> int:
        return max(1, len(text) // 4)

    for capacity in (120, 60, 30):
        selection = select_context_lines(
            context, capacity_tokens=capacity, count_tokens=counter,
        )
        assert selection.block_tokens <= capacity
        if selection.context is not None:
            for item in selection.context.layer_items:
                assert item.metadata["serialized_text"] in selection.context.text


def test_token_overflow_trim_keeps_the_final_landmark_longest() -> None:
    scores = {
        "sink": {"bed": 0.9},
        "foot of bed": {"bed": 0.9},
        "doorway": {"door": 0.9, "bed": 0.1},
        "two stairs": {"stairs": 0.9, "bed": 0.1},
    }
    manager = _manager(_StubGrounder(_QueryScores(scores)))
    _fill_map(manager, [
        ("object-bed", "bed", (2.0, 0.0, 1.0), 0.9),
        ("object-door", "door", (3.0, 0.0, 1.0), 0.9),
        ("object-stairs", "stairs", (4.0, 0.0, 1.0), 0.9),
    ])
    context = _query(manager)
    assert context is not None
    trimmed = context
    for _ in range(3):
        trimmed, dropped = trim_context_for_overflow(trimmed)
        assert dropped == 1
        assert trimmed is not None
    assert '"two stairs"' in trimmed.text


def test_prompt_roundtrip_preserves_instruction_and_template() -> None:
    grounder = _StubGrounder(_QueryScores(_bed_only_scores()))
    manager = _manager(grounder)
    _fill_map(manager, [("object-bed", "bed", (2.0, 0.0, 1.0), 0.9)])
    context = _query(manager)
    assert context is not None

    base_prompt = (
        "You are a navigation agent.\n"
        f"<navigation_task>{INSTRUCTION_1012}</navigation_task>. "
        "History follows."
    )
    base = {
        "conversations": [
            {"from": "human", "value": base_prompt},
            {"from": "gpt", "value": ""},
        ],
        "video_segments": object(),
    }
    augmented = inject_memory_context(base, context, max_chars=320)
    audit = audit_prompt_injection(base, augmented, context, max_chars=320)
    assert audit["roundtrip_ok"] is True
    assert audit["base_sha256"] == hashlib.sha256(
        base_prompt.encode()
    ).hexdigest()
    new_prompt = augmented["conversations"][0]["value"]
    assert INSTRUCTION_1012 in new_prompt
    assert new_prompt.index("[Navigation Memory]") > new_prompt.index(
        "</navigation_task>"
    )
    assert base_prompt not in new_prompt  # memory was inserted, not overwritten
    # Removing exactly the inserted block recovers the original prompt bytes.
    start = new_prompt.index("[Navigation Memory]")
    end = new_prompt.index("\nHistory follows.", start)
    assert new_prompt[:start] + new_prompt[end + 1:] == base_prompt
    assert augmented["video_segments"] is base["video_segments"]
