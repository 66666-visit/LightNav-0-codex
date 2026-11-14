"""Measured Habitat pose flows through MemoryManager and the spatial card."""

from __future__ import annotations

import numpy as np
import pytest

from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.place_recognition import MockPlaceDescriptorExtractor
from lightnav.memory.schemas import ObservationPacket, SceneExtraction
from lightnav.memory.spatial_geometry import AgentPose


def _manager() -> MemoryManager:
    return MemoryManager(
        extractor=MockSceneExtractor({
            0: SceneExtraction(objects=["couch"], confidence=0.91),
        }),
        strategy="hierarchical", place_enabled=True,
        place_extractor=MockPlaceDescriptorExtractor({0: (1, 0), 1: (1, 0)}),
        place_every_n_steps=1,
        local_graph_config=LocalGraphConfig(enabled=True, fusion_mode="spatial"),
    )


def _packet(step: int) -> ObservationPacket:
    return ObservationPacket(
        step_id=step, rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        instruction="find the couch",
    )


def test_habitat_pose_reaches_evidence_place_graph_and_formatted_prompt() -> None:
    manager = _manager()
    manager.reset_episode("episode", "find the couch", scene_id="scene")
    old_pose = AgentPose.from_habitat_info({
        "scene_id": "scene", "distance_to_goal": 0.01, "oracle_success": True,
        "agent_pose": {"position": [0, 1, -1], "rotation": [1, 0, 0, 0]},
    }, expected_scene_id="scene")
    manager.observe(_packet(0), agent_pose=old_pose)
    records = manager.v3_store.evidence_for_scene("scene")
    detection = next(item for item in records if item.event_type == "object_observation")
    old_place = next(item for item in records if item.event_type == "place_observation")
    assert detection.position == (0, 1, -1)
    assert old_place.position == (0, 1, -1)
    assert detection.heading == pytest.approx(0)
    assert detection.payload["position_semantics"] == "agent_observation_viewpoint"
    assert "distance_to_goal" not in detection.payload
    assert "oracle_success" not in detection.payload
    nodes = manager.v3_store.nodes_for_scene("scene")
    assert any(node.node_type == "place" and node.position == (0, 1, -1)
               for node in nodes)
    assert any(node.node_type == "landmark" and node.position == (0, 1, -1)
               for node in nodes)
    assert all(edge.relation_type == "inferred_relation"
               and edge.traversability_status == "unknown"
               for edge in manager.v3_store.edges_for_scene("scene"))

    current_pose = AgentPose("scene", (0, 1, 0), (1, 0, 0, 0))
    events = manager.observe(_packet(1), agent_pose=current_pose)
    context = manager.get_local_graph_context(step=1, rgb_width=8, events=events)
    assert context is not None
    assert "History: couch" in context.text
    assert "observed viewpoint" in context.text
    assert "relative viewpoint 1.0m front" in context.text
    assert "passage unknown" in context.text
    fields = context.layer_items[0].metadata["spatial_fields"]
    assert fields["source_position"] == old_pose.position
    assert fields["current_position"] == current_pose.position
    assert fields["relative_distance"] == pytest.approx(1.0)
    assert fields["relative_bearing"] == pytest.approx(0)
    assert fields["source_place_node_id"] is not None
    assert detection.key in fields["evidence_ids"]
    manager.close_episode()


def test_absent_pose_keeps_metric_unknown_and_wrong_scene_is_rejected() -> None:
    manager = _manager()
    manager.reset_episode("episode", "find the couch", scene_id="scene")
    manager.observe(_packet(0))
    with pytest.raises(ValueError, match="Scene/frame"):
        manager.observe(_packet(1), agent_pose=AgentPose(
            "other", (0, 0, 0), (1, 0, 0, 0),
        ))
    events = manager.observe(_packet(1), agent_pose=AgentPose(
        "scene", (0, 0, 0), (1, 0, 0, 0),
    ))
    context = manager.get_local_graph_context(step=1, rgb_width=8, events=events)
    assert context is not None
    assert "distance unknown, bearing unknown" in context.text
    assert context.layer_items[0].metadata["spatial_fields"]["relative_distance"] is None
    manager.close_episode()
