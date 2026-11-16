"""Cache regression tests for query-visible graph and eligibility changes."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from lightnav.memory.conceptgraphs_lite import (
    CameraIntrinsics,
    InstanceMask,
    Pose3D,
    RGBDFrame,
)
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.scene_query_config import SceneQueryConfig
from lightnav.memory.scene_query_controller import SceneQueryController
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.v3_schemas import EvidenceRecord, canonical_evidence_key


class _SegExtractor:
    def reset_episode(self, episode_id: str, instruction: str) -> None:
        pass

    def close_episode(self) -> None:
        pass


class _Provider:
    def __init__(self, name: str, arguments: dict[str, Any], runs: int) -> None:
        self.responses = deque(
            response
            for index in range(runs)
            for response in (
                {"content": None, "tool_calls": [{
                    "id": f"call_{index}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }]},
                {"content": json.dumps({
                    "decision": "NO_MEMORY", "selected_l1_fact_ids": [],
                    "selected_l3_object_ids": [], "selected_l3_evidence_ids": [],
                    "reason_code": "CACHE_TEST",
                })},
            )
        )
        self.calls: list[list[dict[str, Any]]] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        self.calls.append(json.loads(json.dumps(messages)))
        return self.responses.popleft()

    def tool_result(self, run_index: int) -> Any:
        replies = [message for message in self.calls[2 * run_index + 1]
                   if message["role"] == "tool"]
        assert len(replies) == 1
        return json.loads(replies[0]["content"])["result"]


def _frame() -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.full((8, 8), 2.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
        agent_pose=Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        camera_extrinsics=Pose3D((0.0, 0.88, 0.0), (1.0, 0.0, 0.0, 0.0)),
    )


def _manager() -> tuple[MemoryManager, str, str]:
    manager = MemoryManager(
        extractor=MockSceneExtractor({}), strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=SpatialL3Config(mode="inject", seg_model_path="/fake/seg.pt"),
        spatial_extractor=_SegExtractor(),
        scene_query_config=SceneQueryConfig(enabled=True),
    )
    manager.reset_episode("episode", "find the lamp", scene_id="scene")
    graph = manager.spatial_map
    assert graph is not None
    for step, name, left, right in ((0, "sofa", 1, 4), (1, "lamp", 5, 8)):
        mask = np.zeros((8, 8), dtype=np.bool_)
        mask[1:7, left:right] = True
        evidence_id = canonical_evidence_key("scene", "episode", 0, f"seg_{name}")
        node = graph.observe(_frame(), [InstanceMask(
            class_name=name, confidence=0.9, mask=mask, evidence_id=evidence_id,
        )], step_id=step)[0]
        if name == "lamp":
            lamp_id = node.object_id
            lamp_evidence_id = evidence_id
    return manager, lamp_id, lamp_evidence_id


def _run(
    controller: SceneQueryController, manager: MemoryManager, *, step: int,
    pose: Pose3D | None = None,
) -> dict[str, Any]:
    _, diagnostic = controller.run(
        manager, instruction="find the lamp", step=step, agent_pose=pose,
    )
    assert diagnostic["failure_category"] is None
    assert diagnostic["decision"] == "NO_MEMORY"
    return diagnostic


def test_find_objects_cache_invalidates_when_evidence_becomes_historical() -> None:
    manager, lamp_id, _ = _manager()
    provider = _Provider("find_objects", {
        "class_name": "lamp", "history_eligible_only": True,
    }, runs=2)
    controller = SceneQueryController(SceneQueryConfig(enabled=True), provider)

    first = _run(controller, manager, step=1)
    second = _run(controller, manager, step=2)

    assert provider.tool_result(0) == []
    assert [row["object_id"] for row in provider.tool_result(1)] == [lamp_id]
    assert first["tool_calls_executed"] == second["tool_calls_executed"] == 1
    assert second["query_cache_invalidated"] is True


def test_read_node_cache_invalidates_when_evidence_falls_below_floor() -> None:
    manager, lamp_id, evidence_id = _manager()
    provider = _Provider("read_node", {"object_id": lamp_id}, runs=2)
    controller = SceneQueryController(SceneQueryConfig(enabled=True), provider)
    first = _run(controller, manager, step=2)
    graph = manager.spatial_map
    assert graph is not None
    graph.evidence[evidence_id] = replace(graph.evidence[evidence_id], confidence=0.1)
    second = _run(controller, manager, step=2)

    assert provider.tool_result(0)["historical_eligible_evidence_ids"] == [evidence_id]
    assert provider.tool_result(1)["historical_eligible_evidence_ids"] == []
    assert first["tool_calls_executed"] == second["tool_calls_executed"] == 1
    assert second["query_cache_invalidated"] is True


def test_read_evidence_cache_invalidates_when_source_edge_disappears() -> None:
    manager, lamp_id, evidence_id = _manager()
    provider = _Provider("read_evidence", {"evidence_id": evidence_id}, runs=2)
    controller = SceneQueryController(SceneQueryConfig(enabled=True), provider)
    first = _run(controller, manager, step=2)
    graph = manager.spatial_map
    assert graph is not None
    graph.edges = [edge for edge in graph.edges if edge.evidence_id != evidence_id]
    second = _run(controller, manager, step=2)

    assert provider.tool_result(0)["historical_eligible_object_ids"] == [lamp_id]
    assert provider.tool_result(1)["historical_eligible_object_ids"] == []
    assert first["tool_calls_executed"] == second["tool_calls_executed"] == 1
    assert second["query_cache_invalidated"] is True


def test_read_evidence_cache_tracks_l1_link_validation_state() -> None:
    manager, _, evidence_id = _manager()
    graph = manager.spatial_map
    store = manager.v3_store
    assert graph is not None and store is not None
    l1_id = canonical_evidence_key("scene", "episode", 0, "l1_lamp")
    graph.evidence[evidence_id] = replace(
        graph.evidence[evidence_id], linked_l1_evidence_id=l1_id,
    )
    provider = _Provider("read_evidence", {"evidence_id": evidence_id}, runs=2)
    controller = SceneQueryController(SceneQueryConfig(enabled=True), provider)

    _run(controller, manager, step=2)
    store.add(EvidenceRecord(
        evidence_id="l1_lamp", scene_id="scene", episode_id="episode",
        rollout_idx=0, step=1, event_type="object_observation",
        semantic_label="lamp", observation_status="observed",
        provenance_validity=True,
    ))
    second = _run(controller, manager, step=2)

    assert provider.tool_result(0)["validated_linked_l1_evidence_id"] is None
    assert provider.tool_result(1)["validated_linked_l1_evidence_id"] == l1_id
    assert second["tool_calls_executed"] == 1
    assert second["query_cache_invalidated"] is True


def test_get_edges_cache_invalidates_when_provenance_edges_change() -> None:
    manager, lamp_id, evidence_id = _manager()
    provider = _Provider("get_edges_for_node", {"object_id": lamp_id}, runs=2)
    controller = SceneQueryController(SceneQueryConfig(enabled=True), provider)
    _run(controller, manager, step=2)
    graph = manager.spatial_map
    assert graph is not None
    graph.edges = [edge for edge in graph.edges if edge.evidence_id != evidence_id]
    second = _run(controller, manager, step=2)

    assert len(provider.tool_result(0)) == 1
    assert provider.tool_result(1) == []
    assert second["tool_calls_executed"] == 1
    assert second["query_cache_invalidated"] is True


def test_get_edges_cache_tracks_order_of_returned_edges() -> None:
    manager, lamp_id, _ = _manager()
    graph = manager.spatial_map
    assert graph is not None
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[1:7, 5:8] = True
    second_evidence_id = canonical_evidence_key("scene", "episode", 0, "seg_lamp_again")
    second_node = graph.observe(_frame(), [InstanceMask(
        class_name="lamp", confidence=0.9, mask=mask,
        evidence_id=second_evidence_id,
    )], step_id=2)[0]
    assert second_node.object_id == lamp_id
    provider = _Provider("get_edges_for_node", {"object_id": lamp_id}, runs=2)
    controller = SceneQueryController(SceneQueryConfig(enabled=True), provider)

    _run(controller, manager, step=3)
    graph.edges.reverse()
    second = _run(controller, manager, step=3)

    first_edges = provider.tool_result(0)
    assert len(first_edges) == 2
    assert provider.tool_result(1) == list(reversed(first_edges))
    assert second["tool_calls_executed"] == 1
    assert second["query_cache_invalidated"] is True


@pytest.mark.parametrize("name,arguments", [
    ("find_objects", {"class_name": "lamp"}),
    ("read_node", {"object_id": "lamp"}),
    ("read_evidence", {"evidence_id": "lamp"}),
    ("get_edges_for_node", {"object_id": "lamp"}),
])
def test_pose_change_reuses_pose_independent_tool_result(
    name: str, arguments: dict[str, str],
) -> None:
    manager, lamp_id, evidence_id = _manager()
    arguments = {key: (lamp_id if value == "lamp" and key == "object_id" else
                       evidence_id if value == "lamp" else value)
                 for key, value in arguments.items()}
    provider = _Provider(name, arguments, runs=2)
    controller = SceneQueryController(SceneQueryConfig(enabled=True), provider)
    first = _run(controller, manager, step=2, pose=_frame().agent_pose)
    moved_pose = Pose3D((5.0, 0.0, 2.0), (1.0, 0.0, 0.0, 0.0))
    second = _run(controller, manager, step=2, pose=moved_pose)

    assert provider.tool_result(0) == provider.tool_result(1)
    assert first["tool_calls_executed"] == 1
    assert second["tool_calls_executed"] == 0
    assert second["tool_calls_reused"] == 1
    assert second["query_cache_invalidated"] is False
