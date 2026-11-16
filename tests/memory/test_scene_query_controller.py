"""CPU contracts for task-directed Think/Query/Final memory selection."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import numpy as np
import pytest

from lightnav.memory import scene_query_provider as provider_module
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
from lightnav.memory.scene_query_controller import SceneQueryController, read_l1_facts
from lightnav.memory.scene_query_provider import OpenAICompatibleQueryProvider
from lightnav.memory.schemas import DetectedObject, ObservationPacket, SceneExtraction
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.v3_schemas import canonical_evidence_key


class _SegExtractor:
    def reset_episode(self, episode_id: str, instruction: str) -> None:
        pass

    def close_episode(self) -> None:
        pass


class _MockProvider:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append({
            "messages": json.loads(json.dumps(messages)),
            "tool_names": [tool["function"]["name"] for tool in tools],
        })
        return self.responses.popleft()


def _manager(*, l1_by_step: dict[int, str] | None = None,
             episode_id: str = "episode") -> MemoryManager:
    detections = {
        step: SceneExtraction(
            objects=[label],
            detections=[DetectedObject(label, 0.95, (0.0, 0.0, 8.0, 8.0))],
            confidence=0.95,
        )
        for step, label in (l1_by_step or {}).items()
    }
    manager = MemoryManager(
        extractor=MockSceneExtractor(detections),
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=SpatialL3Config(mode="inject", seg_model_path="/fake/seg.pt"),
        spatial_extractor=_SegExtractor(),
        scene_query_config=SceneQueryConfig(enabled=True),
    )
    manager.reset_episode(episode_id, "find the landmark", scene_id="scene")
    return manager


def _observe_l1(manager: MemoryManager, step: int) -> None:
    manager.observe(ObservationPacket(
        step_id=step, rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        instruction="find the landmark",
    ))


def _frame() -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.full((8, 8), 2.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
        agent_pose=Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        camera_extrinsics=Pose3D((0.0, 0.88, 0.0), (1.0, 0.0, 0.0, 0.0)),
    )


def _map_objects(
    manager: MemoryManager, *entries: tuple[str, int, int, str | None],
) -> list[tuple[str, str]]:
    """Use the frozen ObjectMap writer to create real nodes and provenance edges."""
    instances = []
    state = manager.working.state
    assert state is not None
    for index, (label, left, right, linked_l1_id) in enumerate(entries):
        mask = np.zeros((8, 8), dtype=np.bool_)
        mask[1:7, left:right] = True
        instances.append(InstanceMask(
            class_name=label, confidence=0.9, mask=mask,
            evidence_id=canonical_evidence_key("scene", state.episode_id, 0, f"seg_{index}"),
            linked_l1_evidence_id=linked_l1_id,
        ))
    nodes = manager.spatial_map.observe(_frame(), instances, step_id=0)
    return [(node.object_id, instance.evidence_id)
            for node, instance in zip(nodes, instances)]


def _tool(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "id": f"call_{name}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _query(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"content": None, "tool_calls": list(calls)}


def _final(
    decision: str, *, l1: list[str] | None = None,
    objects: list[str] | None = None, evidence: list[str] | None = None,
    reason: str = "TASK_RELEVANT",
) -> dict[str, Any]:
    return {"content": json.dumps({
        "decision": decision,
        "selected_l1_fact_ids": l1 or [],
        "selected_l3_object_ids": objects or [],
        "selected_l3_evidence_ids": evidence or [],
        "reason_code": reason,
    })}


def _controller(*responses: dict[str, Any]) -> tuple[SceneQueryController, _MockProvider]:
    provider = _MockProvider(*responses)
    return SceneQueryController(SceneQueryConfig(enabled=True, max_rounds=4), provider), provider


def test_l1_suffices_without_a_map_query_or_l2_input() -> None:
    manager = _manager(l1_by_step={1: "sofa"})
    _observe_l1(manager, 0)
    _map_objects(manager, ("sofa", 1, 4, None))
    _observe_l1(manager, 1)
    l1_id = read_l1_facts(manager, 1)[0]["fact_id"]
    controller, provider = _controller(_final(
        "USE_L1", l1=[l1_id], reason="CURRENT_OBSERVATION_SUFFICIENT",
    ))

    context, diagnostic = controller.run(
        manager, instruction="go to the sofa", step=1, agent_pose=_frame().agent_pose,
    )

    assert diagnostic["decision"] == "USE_L1"
    assert diagnostic["l3_object_node_count"] == 1
    assert diagnostic["query_calls"] == []
    assert diagnostic["selected_l1_fact_ids"] == [l1_id]
    assert diagnostic["memory_context_tokens"] is None  # Token audit belongs to Policy.
    assert context is not None and len(context.layer_items) == 1
    assert "Current RGB: detected sofa" in context.text
    assert "Spatial history" not in context.text
    request = json.loads(provider.calls[0]["messages"][1]["content"])
    assert request["l1_facts"][0]["fact_id"] == l1_id
    assert request["available_l3_query_tools"] == provider.calls[0]["tool_names"]
    assert not {"l2_facts", "goal_position", "success", "future_observations"} & set(request)


def test_mock_selection_works_with_api_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMORY_QUERY_API_KEY", raising=False)
    manager = _manager(l1_by_step={0: "sofa"})
    _observe_l1(manager, 0)
    l1_id = read_l1_facts(manager, 0)[0]["fact_id"]
    controller, _ = _controller(_final("USE_L1", l1=[l1_id]))

    context, diagnostic = controller.run(
        manager, instruction="find the sofa", step=0, agent_pose=None,
    )

    assert context is not None
    assert diagnostic["fallback_reason"] is None
    assert diagnostic["decision"] == "USE_L1"


def test_disabled_query_ignores_query_only_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_QUERY_TIMEOUT", "malformed")
    monkeypatch.setenv("MEMORY_QUERY_MAX_ROUNDS", "malformed")
    assert SceneQueryConfig.from_env(enabled=False) == SceneQueryConfig()


def test_think_query_final_selects_verified_l3_history() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    object_id, evidence_id = _map_objects(manager, ("sofa", 1, 4, None))[0]
    _observe_l1(manager, 1)
    controller, provider = _controller(
        {"content": json.dumps({"action": "THINK", "reason_code": "NEED_HISTORY"})},
        _query(_tool("find_objects", class_name="sofa"),
               _tool("read_evidence", evidence_id=evidence_id)),
        _final("USE_L3", objects=[object_id], evidence=[evidence_id]),
    )

    context, diagnostic = controller.run(
        manager, instruction="turn after the sofa", step=1, agent_pose=_frame().agent_pose,
    )

    assert len(provider.calls) == 3
    assert diagnostic["api_rounds"] == 3
    assert diagnostic["query_rounds"] == 1
    assert diagnostic["decision"] == "USE_L3"
    assert diagnostic["query_returned_object_ids"] == [object_id]
    assert diagnostic["query_returned_evidence_ids"] == [evidence_id]
    assert diagnostic["selected_l3_evidence_ids"] == [evidence_id]
    assert context is not None and len(context.layer_items) == 1
    assert context.layer_items[0].metadata["object_id"] == object_id
    assert evidence_id in context.layer_items[0].metadata["evidence_ids"]
    assert "Spatial history: sofa" in context.text
    assert f"Object ID {object_id}" in context.text
    assert "mapped at world" in context.text
    assert "passage unknown" in context.text


def test_current_l1_and_distinct_historical_l3_are_both_selected() -> None:
    manager = _manager(l1_by_step={1: "lamp"})
    _observe_l1(manager, 0)
    object_id, evidence_id = _map_objects(manager, ("sofa", 1, 4, None))[0]
    _observe_l1(manager, 1)
    l1_id = read_l1_facts(manager, 1)[0]["fact_id"]
    controller, _ = _controller(
        _query(_tool("find_objects", class_name="sofa"),
               _tool("read_evidence", evidence_id=evidence_id)),
        _final("USE_BOTH", l1=[l1_id], objects=[object_id], evidence=[evidence_id]),
    )

    context, diagnostic = controller.run(
        manager, instruction="pass the sofa, then find the lamp", step=1,
        agent_pose=_frame().agent_pose,
    )

    assert diagnostic["decision"] == "USE_BOTH"
    assert diagnostic["selected_l1_fact_ids"] == [l1_id]
    assert diagnostic["selected_l3_object_ids"] == [object_id]
    assert context is not None and len(context.layer_items) == 2
    assert "Current RGB: detected lamp" in context.text
    assert "Spatial history: sofa" in context.text


def test_legacy_320_char_limit_does_not_filter_map_candidates() -> None:
    manager = _manager()
    manager.local_graph_config = LocalGraphConfig(enabled=True, prompt_max_chars=1)
    manager.max_context_chars = 1
    _observe_l1(manager, 0)
    objects = _map_objects(manager, ("sofa", 0, 3, None), ("lamp", 5, 8, None))
    _observe_l1(manager, 1)
    controller, _ = _controller(
        _query(
            _tool("find_objects"),
            *(_tool("read_evidence", evidence_id=evidence_id)
              for _, evidence_id in objects),
        ),
        _final(
            "USE_L3", objects=[object_id for object_id, _ in objects],
            evidence=[evidence_id for _, evidence_id in objects],
        ),
    )

    context, diagnostic = controller.run(
        manager, instruction="pass the sofa then find the lamp",
        step=1, agent_pose=_frame().agent_pose,
    )

    assert diagnostic["l3_object_node_count"] == 2
    assert diagnostic["l3_raw_candidate_count"] == 2
    assert diagnostic["l3_evidence_valid_count"] == 2
    assert diagnostic["final_selected_object_count"] == 2
    assert context is not None and len(context.layer_items) == 2
    assert len(context.text) > 320


def test_ambiguous_same_label_current_l1_abstains_from_l3_identity() -> None:
    manager = _manager(l1_by_step={1: "chair"})
    _observe_l1(manager, 0)
    objects = _map_objects(
        manager, ("chair", 0, 3, None), ("chair", 5, 8, None),
    )
    _observe_l1(manager, 1)
    l1_id = read_l1_facts(manager, 1)[0]["fact_id"]
    controller, _ = _controller(
        _query(_tool("find_objects", class_name="chair"),
               _tool("read_evidence", evidence_id=objects[0][1])),
        _final("USE_BOTH", l1=[l1_id], objects=[objects[0][0]],
               evidence=[objects[0][1]]),
    )

    context, diagnostic = controller.run(
        manager, instruction="find the chair", step=1, agent_pose=_frame().agent_pose,
    )

    assert len(manager.spatial_map.objects) == 2
    assert objects[0][0] != objects[1][0]
    assert diagnostic["api_decision"] == "USE_BOTH"
    assert diagnostic["decision"] == "USE_L1"
    assert diagnostic["conflict_abstention_count"] == 1
    assert diagnostic["selected_l3_object_ids"] == []
    assert context is not None and len(context.layer_items) == 1
    assert "Spatial history" not in context.text


def test_explicit_l1_evidence_link_collapses_duplicate_expression() -> None:
    manager = _manager(l1_by_step={0: "chair"})
    _observe_l1(manager, 0)
    l1_id = read_l1_facts(manager, 0)[0]["fact_id"]
    object_id, evidence_id = _map_objects(manager, ("chair", 1, 4, l1_id))[0]
    _observe_l1(manager, 1)
    controller, _ = _controller(
        _query(_tool("find_objects", class_name="chair"),
               _tool("read_evidence", evidence_id=evidence_id)),
        _final("USE_BOTH", l1=[l1_id], objects=[object_id], evidence=[evidence_id]),
    )

    context, diagnostic = controller.run(
        manager, instruction="remember the chair", step=1, agent_pose=_frame().agent_pose,
    )

    assert diagnostic["decision"] == "USE_BOTH"
    assert diagnostic["deduplicated_fact_count"] == 1
    assert context is not None and len(context.layer_items) == 1
    assert context.text.count("Spatial history: chair") == 1
    assert set(context.layer_items[0].metadata["evidence_ids"]) == {l1_id, evidence_id}


def test_unlinked_same_label_map_objects_keep_distinct_ids_and_facts() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    objects = _map_objects(
        manager, ("chair", 0, 3, None), ("chair", 5, 8, None),
    )
    _observe_l1(manager, 1)
    controller, _ = _controller(
        _query(_tool("find_objects", class_name="chair"),
               *(_tool("read_evidence", evidence_id=evidence_id)
                 for _, evidence_id in objects)),
        _final("USE_L3", objects=[object_id for object_id, _ in objects],
               evidence=[evidence_id for _, evidence_id in objects]),
    )

    context, diagnostic = controller.run(
        manager, instruction="inspect the two chairs", step=1, agent_pose=None,
    )

    assert diagnostic["decision"] == "USE_L3"
    assert diagnostic["selected_l3_object_ids"] == [item[0] for item in objects]
    assert context is not None and len(context.layer_items) == 2
    assert [item.metadata["object_id"] for item in context.layer_items] == [
        item[0] for item in objects
    ]
    assert all(f"Object ID {object_id}" in context.text for object_id, _ in objects)


def test_final_rejects_nonexistent_or_unread_evidence_id() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    object_id, evidence_id = _map_objects(manager, ("sofa", 1, 4, None))[0]
    _observe_l1(manager, 1)
    nonexistent = canonical_evidence_key("scene", "episode", 0, "missing")
    for selected_id, tool_calls in (
        (nonexistent, (_tool("find_objects", class_name="sofa"),
                       _tool("read_evidence", evidence_id=evidence_id))),
        (evidence_id, (_tool("find_objects", class_name="sofa"),)),
    ):
        controller, _ = _controller(
            _query(*tool_calls),
            _final("USE_L3", objects=[object_id], evidence=[selected_id]),
        )
        context, diagnostic = controller.run(
            manager, instruction="turn after sofa", step=1, agent_pose=None,
        )
        assert context is None
        assert diagnostic["decision"] == "NO_MEMORY"
        assert diagnostic["invalid_id_rejection_count"] == 1
        assert "unqueried Evidence ID" in diagnostic["fallback_reason"]


@pytest.mark.parametrize("bad_response", [
    {"content": "The sofa is at a made-up coordinate."},
    {"content": '{"decision":"NO_MEMORY","decision":"USE_L1",'
                '"selected_l1_fact_ids":[],"selected_l3_object_ids":[],'
                '"selected_l3_evidence_ids":[],"reason_code":"NO_FACT"}'},
    {"content": json.dumps({
        "decision": "NO_MEMORY", "selected_l1_fact_ids": [],
        "selected_l3_object_ids": [], "selected_l3_evidence_ids": [],
        "reason_code": "NO_FACT", "invented_coordinate": [1, 2, 3],
    })},
])
def test_free_text_and_extra_fact_fields_cannot_enter_memory(bad_response: dict) -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    controller, _ = _controller(bad_response)

    context, diagnostic = controller.run(
        manager, instruction="find sofa", step=0, agent_pose=None,
    )

    assert context is None
    assert diagnostic["decision"] == "NO_MEMORY"
    assert diagnostic["memory_context_item_count"] == 0
    assert diagnostic["fallback_reason"]


def test_repeating_needed_history_is_allowed_without_step_cooldown() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    object_id, evidence_id = _map_objects(manager, ("sofa", 1, 4, None))[0]
    _observe_l1(manager, 1)
    controller, provider = _controller(
        _query(_tool("find_objects", class_name="sofa"),
               _tool("read_evidence", evidence_id=evidence_id)),
        _final("USE_L3", objects=[object_id], evidence=[evidence_id]),
        _query(_tool("find_objects", class_name="sofa"),
               _tool("read_evidence", evidence_id=evidence_id)),
        _final("USE_L3", objects=[object_id], evidence=[evidence_id]),
    )
    first, first_diagnostic = controller.run(
        manager, instruction="turn after sofa", step=1, agent_pose=None,
    )
    controller.record_injection(
        step=1, injected_ids=[evidence_id], actual=True,
        memory_tokens=12, final_token_count=100, removed_items=0,
        historical_ids=[evidence_id],
    )
    _observe_l1(manager, 2)
    second, second_diagnostic = controller.run(
        manager, instruction="turn after sofa", step=2, agent_pose=None,
    )
    controller.record_injection(
        step=2, injected_ids=[evidence_id], actual=True,
        memory_tokens=12, final_token_count=100, removed_items=0,
        historical_ids=[evidence_id],
    )

    assert first is not None and second is not None
    assert first_diagnostic["selected_l3_object_ids"] == [object_id]
    assert second_diagnostic["selected_l3_object_ids"] == [object_id]
    assert second_diagnostic["repeated_selection_count"] == 2
    assert all(call["repeated"] for call in second_diagnostic["query_calls"])
    second_payload = json.loads(provider.calls[2]["messages"][1]["content"])
    evidence_history = next(item for item in second_payload["recent_memory_selections"]
                            if item["kind"] == "l3_evidence" and item["id"] == evidence_id)
    assert evidence_history["selection_count"] == 1
    assert evidence_history["injection_count"] == 1
    assert evidence_history["last_injected_step"] == 1
    stats = controller.episode_statistics()
    assert stats["repeated_evidence_injection_rate"] == 0.5
    assert stats["repeated_query_rate"] == 0.5


def test_uninjected_selection_is_not_counted_as_actual_prompt_memory() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    object_id, evidence_id = _map_objects(manager, ("sofa", 1, 4, None))[0]
    _observe_l1(manager, 1)
    controller, provider = _controller(
        _query(_tool("find_objects", class_name="sofa"),
               _tool("read_evidence", evidence_id=evidence_id)),
        _final("USE_L3", objects=[object_id], evidence=[evidence_id]),
        _final("NO_MEMORY"),
    )
    _, first = controller.run(manager, instruction="pass the sofa", step=1, agent_pose=None)
    assert first["decision"] == "USE_L3"
    controller.record_injection(
        step=1, injected_ids=[], actual=False, memory_tokens=42,
        final_token_count=100, removed_items=1, historical_ids=[],
        final_decision="NO_MEMORY",
    )
    _observe_l1(manager, 2)
    controller.run(manager, instruction="continue", step=2, agent_pose=None)

    stats = controller.episode_statistics()
    assert stats["l3_hit_count"] == 0
    assert stats["no_memory_count"] == 2
    assert stats["actual_prompt_memory_tokens"] == 0
    assert stats["token_limit_removed_item_count"] == 1
    history = json.loads(provider.calls[2]["messages"][1]["content"])[
        "recent_memory_selections"
    ]
    evidence_history = next(item for item in history if item["id"] == evidence_id)
    assert evidence_history["selection_count"] == 1
    assert evidence_history["injection_count"] == 0


def test_controller_state_is_isolated_across_two_consecutive_episodes() -> None:
    first_manager = _manager(episode_id="episode_one")
    _observe_l1(first_manager, 0)
    first_object, first_evidence = _map_objects(
        first_manager, ("sofa", 1, 4, None),
    )[0]
    _observe_l1(first_manager, 1)
    second_manager = _manager(episode_id="episode_two")
    _observe_l1(second_manager, 0)
    second_object, second_evidence = _map_objects(
        second_manager, ("sofa", 1, 4, None),
    )[0]
    _observe_l1(second_manager, 1)
    assert first_object != second_object
    assert first_evidence != second_evidence

    controller, provider = _controller(
        _query(_tool("find_objects", class_name="sofa"),
               _tool("read_evidence", evidence_id=first_evidence)),
        _final("USE_L3", objects=[first_object], evidence=[first_evidence]),
        _query(_tool("find_objects", class_name="sofa"),
               _tool("read_evidence", evidence_id=second_evidence)),
        _final("USE_L3", objects=[second_object], evidence=[second_evidence]),
    )
    first_context, first = controller.run(
        first_manager, instruction="turn after the sofa", step=1, agent_pose=None,
    )
    assert first_context is not None and first["decision"] == "USE_L3"
    controller.record_injection(
        step=1, injected_ids=[first_evidence], actual=True,
        memory_tokens=12, final_token_count=100, removed_items=0,
        historical_ids=[first_evidence], injected_object_ids=[first_object],
        final_item_count=1,
    )
    first_stats = controller.episode_statistics()
    assert first_stats["step_count"] == 1
    assert first_stats["l3_query_count"] == 2
    assert first_stats["actual_injection_step_count"] == 1
    assert first_stats["injected_evidence_count"] == 1
    assert first_stats["actual_prompt_memory_tokens"] == 12
    assert controller._selection_history
    assert controller._query_signatures
    assert controller._injected_evidence == {first_evidence}

    controller.reset_episode()
    assert controller._selection_history == []
    assert controller._query_signatures == set()
    assert controller._injected_evidence == set()
    assert all(value == 0 for value in controller.episode_statistics().values())

    second_context, second = controller.run(
        second_manager, instruction="turn after the sofa", step=1, agent_pose=None,
    )
    assert second_context is not None and second["decision"] == "USE_L3"
    assert all(not call["repeated"] for call in second["query_calls"])
    second_payload = json.loads(provider.calls[2]["messages"][1]["content"])
    assert second_payload["recent_memory_selections"] == []
    controller.record_injection(
        step=1, injected_ids=[second_evidence], actual=True,
        memory_tokens=15, final_token_count=103, removed_items=0,
        historical_ids=[second_evidence], injected_object_ids=[second_object],
        final_item_count=1,
    )
    second_stats = controller.episode_statistics()
    assert second_stats["step_count"] == 1
    assert second_stats["l3_query_count"] == 2
    assert second_stats["repeated_query_count"] == 0
    assert second_stats["repeated_evidence_injection_count"] == 0
    assert second_stats["injected_evidence_count"] == 1
    assert second_stats["actual_prompt_memory_tokens"] == 15
    assert second_stats["actual_prompt_total_tokens"] == 103
    assert first_evidence not in controller._injected_evidence


def test_changed_navigation_demand_selects_previously_unused_historical_object() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    sofa, lamp = _map_objects(
        manager, ("sofa", 0, 3, None), ("lamp", 5, 8, None),
    )
    _observe_l1(manager, 1)
    controller, provider = _controller(
        _query(_tool("find_objects", class_name="sofa"),
               _tool("read_evidence", evidence_id=sofa[1])),
        _final("USE_L3", objects=[sofa[0]], evidence=[sofa[1]]),
        _query(_tool("find_objects", class_name="lamp"),
               _tool("read_evidence", evidence_id=lamp[1])),
        _final("USE_L3", objects=[lamp[0]], evidence=[lamp[1]]),
    )
    _, first = controller.run(
        manager, instruction="go past the sofa", step=1, agent_pose=None,
    )
    _observe_l1(manager, 2)
    second_context, second = controller.run(
        manager, instruction="now go toward the lamp", step=2, agent_pose=None,
    )

    assert first["selected_l3_object_ids"] == [sofa[0]]
    assert second["selected_l3_object_ids"] == [lamp[0]]
    assert second["repeated_selection_count"] == 0
    assert second_context is not None and "Spatial history: lamp" in second_context.text
    second_payload = json.loads(provider.calls[2]["messages"][1]["content"])
    assert second_payload["instruction"] == "now go toward the lamp"
    assert any(
        item["kind"] == "l3_object" and item["id"] == sofa[0]
        for item in second_payload["recent_memory_selections"]
    )


@pytest.mark.parametrize("failure", ["timeout", "invalid_json"])
def test_real_provider_failure_fails_closed_and_never_logs_api_key(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    monkeypatch.setenv("MEMORY_QUERY_API_KEY", "secret_api_key_do_not_log")

    if failure == "timeout":
        def urlopen(_request: Any, *, timeout: float) -> Any:
            raise TimeoutError("simulated timeout")
    else:
        class Reply:
            def __enter__(self) -> Reply:
                return self

            def __exit__(self, *_args: Any) -> None:
                pass

            def read(self) -> bytes:
                return b"not valid JSON"

        def urlopen(_request: Any, *, timeout: float) -> Reply:
            return Reply()

    monkeypatch.setattr(provider_module, "urlopen", urlopen)
    config = SceneQueryConfig(
        enabled=True, base_url="https://example.invalid/v1", model="mock", max_rounds=2,
    )
    controller = SceneQueryController(config, OpenAICompatibleQueryProvider(config))
    context, diagnostic = controller.run(
        manager, instruction="find sofa", step=0, agent_pose=None,
    )

    assert context is None
    assert diagnostic["decision"] == "NO_MEMORY"
    assert diagnostic["fallback_reason"]
    assert "secret_api_key_do_not_log" not in json.dumps(diagnostic)
    assert controller.episode_statistics()["api_failure_count"] == 1
