"""CPU contracts for Geometry-only L3 + the original on-demand gate."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lightnav.memory.conceptgraphs_lite import (
    ObjectMap,
    ObjectNode,
    ObservedFromEdge,
    Pose3D,
    SpatialEvidence,
    ViewpointNode,
)
from lightnav.memory.l3_on_demand import L3QueryController
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.v3_schemas import canonical_evidence_key

SCENE = "scene"
EPISODE = "episode"


class _TextEncoder:
    def __init__(self, feature=(1.0, 0.0)) -> None:
        self.feature = np.asarray(feature, dtype=np.float32)
        self.text_calls = 0

    def encode_image(self, rgb, boxes):
        raise AssertionError("geometry mode must not call CLIP image encoding")

    def encode_text(self, texts):
        self.text_calls += 1
        return np.asarray([self.feature], dtype=np.float32)


class _SegExtractor:
    def reset_episode(self, episode_id, instruction):
        pass

    def close_episode(self):
        pass

    def extract(self, rgb, step_id):
        return []


def _pose() -> Pose3D:
    return Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _map(*, with_clip: bool = True) -> ObjectMap:
    object_map = ObjectMap()
    object_map.reset_episode(SCENE, EPISODE, 0)
    object_id = json.dumps([SCENE, EPISODE, 0, "object", 1], separators=(",", ":"))
    evidence_id = canonical_evidence_key(SCENE, EPISODE, 0, "l3_seg_1_0")
    viewpoint_id = json.dumps([SCENE, EPISODE, 0, 1], separators=(",", ":"))
    object_map.viewpoints[viewpoint_id] = ViewpointNode(
        viewpoint_id, SCENE, EPISODE, 0, 1, _pose(), _pose(),
    )
    object_map.evidence[evidence_id] = SpatialEvidence(
        evidence_id, SCENE, EPISODE, 0, 1, "sofa", 0.9,
        "yolo11n-seg", 20, 8, "observed",
    )
    object_map.edges.append(ObservedFromEdge(object_id, viewpoint_id, evidence_id))
    node = ObjectNode(
        object_id, "sofa", (2.0, 0.0, -2.0),
        ((1.5, -0.5, -2.5), (2.5, 0.5, -1.5)),
        1, [evidence_id], "observed", np.zeros((5, 3)),
    )
    object_map.objects[object_id] = node
    if with_clip:
        object_map.attach_clip_features({evidence_id: [1.0, 0.0]})
    return object_map


def _geometry_gate() -> L3QueryController:
    return L3QueryController(None, top_k=1)


def test_geometry_gate_matches_clip_gate_for_l1_satisfied_and_empty() -> None:
    object_map = _map()
    clip = L3QueryController(_TextEncoder(), top_k=1)
    geometry = _geometry_gate()
    assert clip.maybe_query(
        object_map, step=2, instruction="find the sofa", l1_status="built",
    ) is None
    assert geometry.prepare_query(
        object_map, step=2, instruction="find the sofa", l1_status="built",
        available=lambda value: geometry.has_historical_geometry_candidates(value, 2),
        unavailable_reason="no_historical_geometry_object",
        retrieval="geometry_spatial_graph_adapter",
        signature_fn=geometry.geometry_map_signature,
    ) is None
    assert clip.last_diagnostic["skipped_reason"] == "l1_satisfied"
    assert geometry.last_diagnostic["skipped_reason"] == "l1_satisfied"
    assert clip.last_diagnostic["query_executed"] is False
    assert geometry.last_diagnostic["query_executed"] is False

    clip_result = clip.maybe_query(
        object_map, step=2, instruction="find the sofa", l1_status="no_candidates",
    )
    prepared = geometry.prepare_query(
        object_map, step=2, instruction="find the sofa", l1_status="no_candidates",
        available=lambda value: geometry.has_historical_geometry_candidates(value, 2),
        unavailable_reason="no_historical_geometry_object",
        retrieval="geometry_spatial_graph_adapter",
        signature_fn=geometry.geometry_map_signature,
    )
    assert clip_result is not None and prepared is not None
    assert clip.last_diagnostic["query_executed"] is True
    assert geometry.last_diagnostic["query_executed"] is True
    assert clip.last_diagnostic["query"] == geometry.last_diagnostic["query"]
    assert geometry.last_diagnostic["retrieval"] == "geometry_spatial_graph_adapter"
    assert "similarity_semantics" not in geometry.last_diagnostic
    assert "top_k" not in geometry.last_diagnostic


def test_geometry_gate_uses_the_same_unchanged_map_dedupe() -> None:
    object_map = _map()
    geometry = _geometry_gate()

    def prepare(step: int):
        return geometry.prepare_query(
            object_map, step=step, instruction="find the sofa",
            l1_status="no_candidates",
            available=lambda value: geometry.has_historical_geometry_candidates(
                value, step,
            ),
            unavailable_reason="no_historical_geometry_object",
            retrieval="geometry_spatial_graph_adapter",
            signature_fn=geometry.geometry_map_signature,
        )

    first = prepare(2)
    second = prepare(3)
    assert first is not None
    assert second is None
    assert geometry.last_diagnostic["skipped_reason"] == (
        "map_unchanged_since_last_query"
    )
    assert geometry.last_diagnostic["steps_since_last_query"] == 1

    # New geometric Evidence reopens the query.
    evidence_id = canonical_evidence_key(SCENE, EPISODE, 0, "l3_seg_3_0")
    viewpoint_id = json.dumps([SCENE, EPISODE, 0, 3], separators=(",", ":"))
    object_map.viewpoints[viewpoint_id] = ViewpointNode(
        viewpoint_id, SCENE, EPISODE, 0, 3, _pose(), _pose(),
    )
    object_map.evidence[evidence_id] = SpatialEvidence(
        evidence_id, SCENE, EPISODE, 0, 3, "sofa", 0.9,
        "yolo11n-seg", 20, 8, "observed",
    )
    node = next(iter(object_map.objects.values()))
    node.evidence_ids.append(evidence_id)
    node.observation_count += 1
    object_map.edges.append(ObservedFromEdge(
        node.object_id, viewpoint_id, evidence_id,
    ))
    assert prepare(4) is not None


def test_geometry_signature_is_not_the_empty_clip_signature() -> None:
    object_map = _map(with_clip=False)
    assert L3QueryController.map_signature(object_map) == ()
    assert L3QueryController.geometry_map_signature(object_map) != ()
    assert L3QueryController.has_clip_features(object_map) is False
    assert L3QueryController.has_historical_geometry_candidates(object_map, 2) is True
    assert L3QueryController.has_historical_geometry_candidates(object_map, 1) is False


def test_geometry_mode_rejects_a_clip_encoder() -> None:
    with pytest.raises(ValueError, match="must not receive a CLIP encoder"):
        MemoryManager(
            strategy="hierarchical",
            local_graph_config=LocalGraphConfig(enabled=True),
            spatial_l3_config=SpatialL3Config(
                mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
            ),
            spatial_extractor=_SegExtractor(),
            clip_encoder=_TextEncoder(),
            prompt_mode="l1_original_only",
        )


def test_geometry_mode_needs_no_clip_path_or_encoder() -> None:
    config = SpatialL3Config(
        mode="geometry_on_demand", seg_model_path="/fake/seg.pt",
        clip_model_path="",
    )
    assert config.clip_model_path == ""
    manager = MemoryManager(
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=config,
        spatial_extractor=_SegExtractor(),
        clip_encoder=None,
        prompt_mode="l1_original_only",
    )
    assert manager.clip_encoder is None
    assert manager.l3_query_controller is None
    assert manager.geometry_on_demand_controller is not None


@pytest.mark.parametrize(
    ("status", "executed", "reason"),
    [
        ("budget_excluded", True, None),
        ("reactivation_suppressed", False, "l1_reactivation_suppressed"),
    ],
)
def test_geometry_gate_keeps_the_original_status_semantics(
    status, executed, reason,
) -> None:
    object_map = _map()
    clip = L3QueryController(_TextEncoder(), top_k=1)
    geometry = _geometry_gate()
    clip.maybe_query(
        object_map, step=2, instruction="find the sofa", l1_status=status,
    )
    prepared = geometry.prepare_query(
        object_map, step=2, instruction="find the sofa", l1_status=status,
        available=lambda value: geometry.has_historical_geometry_candidates(value, 2),
        unavailable_reason="no_historical_geometry_object",
        retrieval="geometry_spatial_graph_adapter",
        signature_fn=geometry.geometry_map_signature,
    )
    assert (prepared is not None) is executed
    assert clip.last_diagnostic["query_executed"] is executed
    assert geometry.last_diagnostic["query_executed"] is executed
    assert geometry.last_diagnostic["skipped_reason"] == reason
    assert clip.last_diagnostic["skipped_reason"] == reason
