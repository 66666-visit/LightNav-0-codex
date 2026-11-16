"""CPU contracts for read-only structured queries against the real ObjectMap."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from lightnav.memory.conceptgraphs_lite import (
    CameraIntrinsics,
    InstanceMask,
    ObjectMap,
    Pose3D,
    RGBDFrame,
)
from lightnav.memory.scene_query_tools import OPENAI_TOOLS, TOOL_SCHEMAS, SceneQueryTools
from lightnav.memory.v3_schemas import canonical_evidence_key


def _frame(*, x: float = 0.0) -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.full((8, 8), 2.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
        agent_pose=Pose3D((x, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        camera_extrinsics=Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    )


def _instance(left: int, right: int, number: int) -> InstanceMask:
    mask = np.zeros((8, 8), dtype=bool)
    mask[:, left:right] = True
    return InstanceMask(
        class_name="chair", confidence=0.9, mask=mask,
        evidence_id=canonical_evidence_key("scene", "episode", 0, f"seg_{number}"),
    )


@pytest.fixture
def queried_map() -> tuple[ObjectMap, SceneQueryTools, list[str], list[str]]:
    graph = ObjectMap()
    graph.reset_episode("scene", "episode")
    instances = [_instance(0, 3, 0), _instance(5, 8, 1)]
    nodes = graph.observe(_frame(), instances, step_id=0)
    # The second viewpoint creates a real traversed_to edge. Object queries
    # must not reinterpret that viewpoint transition as object adjacency.
    graph.observe(_frame(x=0.2), [], step_id=1, transition_record=True)
    tools = SceneQueryTools(
        graph, None, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=1, scope="episode_local", min_confidence=0.5,
    )
    return graph, tools, [node.object_id for node in nodes], [i.evidence_id for i in instances]


def test_find_objects_reads_existing_nodes_and_unbudgeted_candidates(queried_map) -> None:
    graph, tools, object_ids, _ = queried_map
    found = tools.call("find_objects", {"class_name": "ChAiR"})
    assert {row["object_id"] for row in found} == set(object_ids)
    assert tools.call("find_objects", {"min_observation_count": 2}) == []
    assert tools.candidate_counts() == {
        "l3_object_node_count": len(graph.objects),
        "l3_raw_candidate_count": 2,
        "l3_evidence_valid_count": 2,
    }
    assert all(schema["additionalProperties"] is False for schema in TOOL_SCHEMAS.values())
    assert {tool["function"]["name"] for tool in OPENAI_TOOLS} == set(TOOL_SCHEMAS)


def test_read_node_returns_real_object_fields(queried_map) -> None:
    graph, tools, object_ids, evidence_ids = queried_map
    node = tools.call("read_node", {"object_id": object_ids[0]})
    original = graph.objects[object_ids[0]]
    assert node == {
        "object_id": original.object_id,
        "class_name": original.class_name,
        "center_world": list(original.center_world),
        "bbox_3d": [list(part) for part in original.bbox_3d],
        "observation_count": 1,
        "geometry_status": "observed",
        "evidence_ids": [evidence_ids[0]],
    }
    assert "points_world" not in node


def test_get_edges_for_node_returns_only_recorded_observation_edges(queried_map) -> None:
    graph, tools, object_ids, evidence_ids = queried_map
    assert any(edge.relation_type == "traversed_to" for edge in graph.edges)
    edges = tools.call("get_edges_for_node", {"object_id": object_ids[0]})
    assert len(edges) == 1
    assert edges[0] == {
        "object_id": object_ids[0],
        "viewpoint_id": next(
            edge.viewpoint_id for edge in graph.edges
            if edge.relation_type == "observed_from" and edge.object_id == object_ids[0]
        ),
        "evidence_id": evidence_ids[0],
        "relation_type": "observed_from",
    }
    assert not any("traversed_to" in str(edge) for edge in edges)


@pytest.mark.parametrize("name,args", [
    ("read_node", {"object_id": "missing"}),
    ("get_edges_for_node", {"object_id": "missing"}),
    ("find_objects", {"object_id": "missing"}),
])
def test_invalid_object_id_is_rejected(queried_map, name, args) -> None:
    _, tools, _, _ = queried_map
    with pytest.raises(ValueError, match="invalid Object ID"):
        tools.call(name, args)


def test_invalid_evidence_id_is_rejected(queried_map) -> None:
    _, tools, _, _ = queried_map
    with pytest.raises(ValueError, match="invalid Evidence ID"):
        tools.call("read_evidence", {"evidence_id": "missing"})


def test_unknown_query_tool_is_rejected(queried_map) -> None:
    _, tools, _, _ = queried_map
    with pytest.raises(ValueError, match="unknown L3 query tool"):
        tools.call("delete_objects", {})


def test_unknown_or_malformed_arguments_are_rejected(queried_map) -> None:
    _, tools, object_ids, _ = queried_map
    for arguments in (
        {"object_id": object_ids[0], "relation": "CLOSE"},
        '{"object_id":"x","object_id":"y"}',
        {"object_id": object_ids[0], "min_observation_count": True},
        {"object_id": object_ids[0], 9: "wrong key type"},
    ):
        with pytest.raises(ValueError):
            tools.call("read_node", arguments)
    with pytest.raises(ValueError, match="missing"):
        tools.call("read_node", {})


def test_queries_leave_objectmap_unchanged_and_return_detached_data(queried_map) -> None:
    graph, tools, object_ids, evidence_ids = queried_map
    before = pickle.dumps(graph)
    node = tools.call("read_node", {"object_id": object_ids[0]})
    evidence = tools.call("read_evidence", {"evidence_id": evidence_ids[0]})
    tools.call("find_objects", {})
    tools.call("get_edges_for_node", {"object_id": object_ids[0]})
    tools.validate_history(object_ids[0], evidence_ids[0])
    node["evidence_ids"].append("invented")
    evidence["object_ids"].append("invented")
    assert pickle.dumps(graph) == before
    assert graph.objects[object_ids[0]].evidence_ids == [evidence_ids[0]]


def test_same_label_nodes_keep_distinct_ids_and_evidence(queried_map) -> None:
    _, tools, object_ids, evidence_ids = queried_map
    assert len(set(object_ids)) == 2
    found = tools.call("find_objects", {"class_name": "chair"})
    assert {row["object_id"] for row in found} == set(object_ids)
    assert {tuple(row["evidence_ids"]) for row in found} == {
        (evidence_ids[0],), (evidence_ids[1],),
    }
    first = tools.call("read_evidence", {"evidence_id": evidence_ids[0]})
    assert first["object_ids"] == [object_ids[0]]
    assert first["step_id"] == 0
    assert first["confidence"] == pytest.approx(0.9)
    assert tools.validate_history(object_ids[0], evidence_ids[0])["object_id"] == object_ids[0]
    with pytest.raises(ValueError, match="does not belong"):
        tools.validate_history(object_ids[0], evidence_ids[1])
