"""CPU contracts for task-directed Think/Query/Final memory selection."""

from __future__ import annotations

import json
from collections import deque
from itertools import count
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

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        self.calls.append({
            "messages": json.loads(json.dumps(messages)),
            "tool_names": [tool["function"]["name"] for tool in tools],
            "tool_choice": tool_choice,
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


_TOOL_CALL_IDS = count()


def _tool(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "id": f"call_{name}_{next(_TOOL_CALL_IDS)}", "type": "function",
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


def _controller(
    *responses: dict[str, Any], max_rounds: int = 4, max_tool_calls: int = 12,
) -> tuple[SceneQueryController, _MockProvider]:
    provider = _MockProvider(*responses)
    return SceneQueryController(SceneQueryConfig(
        enabled=True, max_rounds=max_rounds, max_tool_calls=max_tool_calls,
    ), provider), provider


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


def test_multi_round_tool_feedback_preserves_reasoning_content_without_logging_it() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    object_id, evidence_id = _map_objects(manager, ("sofa", 1, 4, None))[0]
    _observe_l1(manager, 1)
    private_first = "PRIVATE_REASONING_FIRST"
    private_second = "PRIVATE_REASONING_SECOND"
    first_tool = _tool("find_objects", class_name="sofa")
    controller, provider = _controller(
        {"content": None, "reasoning_content": private_first,
         "tool_calls": [first_tool]},
        {"content": json.dumps({"action": "THINK", "reason_code": "READ_EVIDENCE"}),
         "reasoning_content": private_second},
        _query(_tool("read_evidence", evidence_id=evidence_id)),
        _final("USE_L3", objects=[object_id], evidence=[evidence_id]),
    )

    context, diagnostic = controller.run(
        manager, instruction="turn after the sofa", step=1, agent_pose=None,
    )

    assert context is not None and diagnostic["decision"] == "USE_L3"
    assert diagnostic["api_request_count"] == 4
    assert diagnostic["api_round_request_counts"] == [1, 1, 1, 1]
    assert len(diagnostic["request_latencies_ms"]) == 4
    assert provider.calls[-1]["tool_choice"] == "none"  # Reserved Final round.
    second_messages = provider.calls[1]["messages"]
    assert all(
        isinstance(message["content"], str) and message["content"] is not None
        for request in provider.calls for message in request["messages"]
        if message["role"] == "assistant"
    )
    assert second_messages[2]["content"] == ""
    assert second_messages[2]["reasoning_content"] == private_first
    assert second_messages[2]["tool_calls"] == [first_tool]
    assert second_messages[2]["tool_calls"][0]["id"] == second_messages[3]["tool_call_id"]
    assert json.loads(second_messages[3]["content"])["result"][0]["class_name"] == "sofa"
    assert any(message.get("reasoning_content") == private_second
               for message in provider.calls[2]["messages"])
    assert sum(message["role"] == "tool" for message in provider.calls[-1]["messages"]) == 2
    assert private_first not in json.dumps(diagnostic)
    assert private_second not in json.dumps(diagnostic)


def test_deepseek_thinking_http_tool_roundtrip_keeps_reasoning_and_verified_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    object_id, evidence_id = _map_objects(manager, ("sofa", 1, 4, None))[0]
    _observe_l1(manager, 1)
    monkeypatch.setenv("MEMORY_QUERY_API_KEY", "synthetic_secret_never_log")
    first_tool = _tool("find_objects", class_name="sofa", history_eligible_only=True)
    second_tool = _tool("read_evidence", evidence_id=evidence_id)
    responses = deque([
        {"role": "assistant", "content": None, "reasoning_content": "PRIVATE_ROUND_ONE",
         "tool_calls": [first_tool]},
        {"role": "assistant", "content": None, "reasoning_content": "PRIVATE_ROUND_TWO",
         "tool_calls": [second_tool]},
        {"role": "assistant", "content": _final(
            "USE_L3", objects=[object_id], evidence=[evidence_id],
        )["content"], "reasoning_content": "PRIVATE_ROUND_THREE"},
    ])
    requests: list[dict[str, Any]] = []

    class Reply:
        def __init__(self, message: dict[str, Any]) -> None:
            self.message = message

        def __enter__(self) -> Reply:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": self.message}]}).encode()

    def urlopen(request: Any, *, timeout: float) -> Reply:
        requests.append(json.loads(request.data))
        return Reply(responses.popleft())

    monkeypatch.setattr(provider_module, "urlopen", urlopen)
    controller = SceneQueryController(SceneQueryConfig(
        enabled=True, base_url="https://api.deepseek.com", model="deepseek-v4-pro",
        thinking="enabled", max_rounds=3,
    ))
    context, diagnostic = controller.run(
        manager, instruction="turn after sofa", step=1, agent_pose=None,
    )

    assert context is not None and "Spatial history: sofa" in context.text
    assert diagnostic["decision"] == "USE_L3"
    assert diagnostic["api_request_count"] == 3
    assert diagnostic["tool_call_counts"] == {"find_objects": 1, "read_evidence": 1}
    assert all(request["thinking"] == {"type": "enabled"} for request in requests)
    assert "tool_choice" not in requests[0] and "tool_choice" not in requests[1]
    assert requests[2]["tool_choice"] == "none"
    assert len(requests[2]["tools"]) == 4
    assert all(
        isinstance(message["content"], str) and message["content"] is not None
        for request in requests for message in request["messages"]
        if message["role"] == "assistant"
    )
    assert requests[1]["messages"][2]["content"] == ""
    assert [message["tool_calls"] for message in requests[1]["messages"]
            if message["role"] == "assistant" and message.get("tool_calls")] == [[first_tool]]
    assert [message["tool_calls"] for message in requests[2]["messages"]
            if message["role"] == "assistant" and message.get("tool_calls")] == [
                [first_tool], [second_tool],
            ]
    assert any(message.get("reasoning_content") == "PRIVATE_ROUND_ONE"
               for message in requests[1]["messages"])
    assert any(message.get("reasoning_content") == "PRIVATE_ROUND_TWO"
               for message in requests[2]["messages"])
    for request in requests[1:]:
        tool_calls = {
            call["id"] for message in request["messages"]
            if message["role"] == "assistant" for call in message.get("tool_calls", [])
        }
        assert all(
            message["tool_call_id"] in tool_calls and "result" in json.loads(message["content"])
            for message in request["messages"] if message["role"] == "tool"
        )
    assert sum(message["role"] == "tool" for message in requests[2]["messages"]) == 2
    assert "PRIVATE_ROUND_ONE" not in json.dumps(diagnostic)
    assert "synthetic_secret_never_log" not in json.dumps(diagnostic)


def test_same_step_duplicate_query_uses_one_dispatch_and_returns_both_tool_results() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    _map_objects(manager, ("sofa", 1, 4, None))
    _observe_l1(manager, 1)
    controller, provider = _controller(
        _query(_tool("find_objects", class_name="sofa"),
               _tool("find_objects", class_name="sofa")),
        _final("NO_MEMORY"),
    )

    context, diagnostic = controller.run(
        manager, instruction="find sofa", step=1, agent_pose=None,
    )

    assert context is None and diagnostic["decision"] == "NO_MEMORY"
    assert diagnostic["tool_call_counts"] == {"find_objects": 2}
    assert diagnostic["tool_calls_requested"] == 2
    assert diagnostic["tool_calls_executed"] == 1
    assert diagnostic["tool_calls_succeeded"] == 1
    assert diagnostic["tool_calls_reused"] == 1
    assert diagnostic["duplicate_tool_call_count"] == 1
    assert [call["cache_hit"] for call in diagnostic["query_calls"]] == [False, True]
    replies = [message for message in provider.calls[1]["messages"]
               if message["role"] == "tool"]
    assert len(replies) == 2 and replies[0]["tool_call_id"] != replies[1]["tool_call_id"]
    assert json.loads(replies[0]["content"])["result"] == json.loads(
        replies[1]["content"]
    )["result"]
    assert controller.episode_statistics()["l3_query_count"] == 2
    assert controller.episode_statistics()["repeated_query_count"] == 1


def test_cached_query_reused_for_unchanged_map_and_invalidated_after_graph_change() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    _map_objects(manager, ("sofa", 1, 4, None))
    _observe_l1(manager, 1)
    controller, _ = _controller(
        _query(_tool("find_objects", class_name="sofa", history_eligible_only=True)),
        _final("NO_MEMORY"),
        _query(_tool("find_objects", class_name="sofa", history_eligible_only=True)),
        _final("NO_MEMORY"),
        _query(_tool("find_objects", class_name="sofa", history_eligible_only=True)),
        _final("NO_MEMORY"),
    )
    _, first = controller.run(manager, instruction="find sofa", step=1, agent_pose=None)
    _observe_l1(manager, 2)  # No L3 map update; the query-visible graph is unchanged.
    _, second = controller.run(manager, instruction="find sofa", step=2, agent_pose=None)
    assert first["tool_calls_executed"] == 1
    assert second["query_cache_invalidated"] is False
    assert second["tool_calls_executed"] == 0
    assert second["tool_calls_reused"] == 1

    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[1:7, 5:8] = True
    manager.spatial_map.observe(_frame(), [InstanceMask(
        class_name="lamp", confidence=0.9, mask=mask,
        evidence_id=canonical_evidence_key("scene", "episode", 0, "seg_new"),
    )], step_id=3)
    _observe_l1(manager, 3)
    _observe_l1(manager, 4)
    _, third = controller.run(manager, instruction="find sofa", step=4, agent_pose=None)
    assert third["query_cache_invalidated"] is True
    assert third["tool_calls_executed"] == 1
    assert third["tool_calls_reused"] == 0
    stats = controller.episode_statistics()
    assert stats["tool_calls_requested"] == 3
    assert stats["tool_calls_executed"] == 2
    assert stats["tool_calls_succeeded"] == 2
    assert stats["tool_calls_reused"] == 1
    assert stats["tool_calls_rejected_budget"] == 0


def test_tool_call_budget_abstains_and_is_not_an_api_failure() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    _map_objects(manager, ("sofa", 1, 4, None))
    _observe_l1(manager, 1)
    controller, provider = _controller(
        _query(_tool("find_objects", class_name="sofa"),
               _tool("find_objects", class_name="sofa")),
        {"content": json.dumps({"action": "THINK", "reason_code": "STILL_QUERYING"})},
        {"content": json.dumps({"action": "THINK", "reason_code": "STILL_QUERYING"})},
        max_rounds=3, max_tool_calls=1,
    )

    context, diagnostic = controller.run(
        manager, instruction="find sofa", step=1, agent_pose=None,
    )

    assert context is None and diagnostic["decision"] == "NO_MEMORY"
    assert diagnostic["failure_category"] == "MAX_TOOL_CALLS"
    assert diagnostic["tool_call_budget_exhausted"] is True
    assert diagnostic["tool_calls_requested"] == 2
    assert diagnostic["tool_calls_executed"] == 1
    assert diagnostic["tool_calls_succeeded"] == 1
    assert diagnostic["tool_calls_rejected_budget"] == 1
    assert diagnostic["api_request_count"] == 3
    assert provider.calls[1]["tool_choice"] == provider.calls[2]["tool_choice"] == "none"
    stats = controller.episode_statistics()
    assert stats["max_tool_calls_count"] == 1
    assert stats["api_failure_count"] == 0
    assert stats["tool_calls_requested"] == 2
    assert stats["tool_calls_executed"] == 1
    assert stats["tool_calls_succeeded"] == 1
    assert stats["tool_calls_reused"] == 0
    assert stats["tool_calls_rejected_budget"] == 1


def test_tool_diagnostics_separate_requests_dispatch_success_cache_and_budget() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    _map_objects(manager, ("sofa", 1, 4, None))
    _observe_l1(manager, 1)
    controller, _ = _controller(
        _query(
            _tool("find_objects", class_name="sofa"),
            _tool("find_objects", class_name="sofa"),
            _tool("read_node"),  # Reaches the local dispatcher and fails validation.
            _tool("read_evidence", evidence_id="over_budget"),
        ),
        _final("NO_MEMORY"),
        max_rounds=3, max_tool_calls=3,
    )

    context, diagnostic = controller.run(
        manager, instruction="find sofa", step=1, agent_pose=None,
    )

    assert context is None and diagnostic["failure_category"] is None
    assert diagnostic["tool_call_counts"] == {
        "find_objects": 2, "read_node": 1, "read_evidence": 1,
    }
    expected = {
        "tool_calls_requested": 4,
        "tool_calls_executed": 2,
        "tool_calls_succeeded": 1,
        "tool_calls_reused": 1,
        "tool_calls_rejected_budget": 1,
    }
    for key, value in expected.items():
        assert diagnostic[key] == value
        assert controller.episode_statistics()[key] == value
    assert controller.episode_statistics()["tool_read_evidence_call_count"] == 1


def test_tool_call_budget_can_finalize_already_verified_history() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    object_id, evidence_id = _map_objects(manager, ("sofa", 1, 4, None))[0]
    _observe_l1(manager, 1)
    controller, provider = _controller(
        _query(_tool("find_objects", class_name="sofa", history_eligible_only=True),
               _tool("read_evidence", evidence_id=evidence_id)),
        _final("USE_L3", objects=[object_id], evidence=[evidence_id]),
        max_rounds=3, max_tool_calls=2,
    )
    context, diagnostic = controller.run(
        manager, instruction="turn after sofa", step=1, agent_pose=None,
    )
    assert context is not None and diagnostic["decision"] == "USE_L3"
    assert diagnostic["tool_call_budget_exhausted"] is True
    assert diagnostic["tool_calls_rejected_budget"] == 0
    assert diagnostic["failure_category"] is None
    assert provider.calls[-1]["tool_choice"] == "none"


def test_final_round_rejects_new_tool_call_and_records_round_budget() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    _map_objects(manager, ("sofa", 1, 4, None))
    _observe_l1(manager, 1)
    controller, provider = _controller(
        _query(_tool("find_objects", class_name="sofa")),
        _query(_tool("find_objects", class_name="sofa")),
        max_rounds=2,
    )
    context, diagnostic = controller.run(
        manager, instruction="find sofa", step=1, agent_pose=None,
    )
    assert context is None and diagnostic["failure_category"] == "MAX_TOOL_ROUNDS"
    assert diagnostic["decision"] == "NO_MEMORY"
    assert diagnostic["tool_calls_requested"] == 2
    assert diagnostic["tool_calls_executed"] == 1
    assert diagnostic["tool_calls_succeeded"] == 1
    assert diagnostic["tool_calls_rejected_budget"] == 1
    assert provider.calls[-1]["tool_choice"] == "none"
    assert controller.episode_statistics()["max_tool_rounds_count"] == 1
    assert controller.episode_statistics()["api_failure_count"] == 0


def test_api_never_returns_final_has_separate_failure_count() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    controller, provider = _controller(
        {"content": json.dumps({"action": "THINK", "reason_code": "WAIT"})},
        {"content": json.dumps({"action": "THINK", "reason_code": "WAIT"})},
        max_rounds=2,
    )
    context, diagnostic = controller.run(
        manager, instruction="find sofa", step=0, agent_pose=None,
    )
    assert context is None and diagnostic["failure_category"] == "API_NO_FINAL"
    assert diagnostic["query_calls"] == []
    assert all(not call["tool_names"] for call in provider.calls)
    assert controller.episode_statistics()["api_no_final_count"] == 1
    assert controller.episode_statistics()["api_failure_count"] == 0


def test_reasoning_only_turn_can_finish_without_tools_and_invalid_json_is_no_final() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    controller, provider = _controller(
        {"content": None, "reasoning_content": "PRIVATE_THINKING"},
        _final("NO_MEMORY"),
        max_rounds=2,
    )
    context, diagnostic = controller.run(
        manager, instruction="find sofa", step=0, agent_pose=None,
    )
    assert context is None and diagnostic["decision"] == "NO_MEMORY"
    assert diagnostic["failure_category"] is None
    assert provider.calls[1]["messages"][2]["reasoning_content"] == "PRIVATE_THINKING"
    assert "PRIVATE_THINKING" not in json.dumps(diagnostic)

    bad_controller, _ = _controller({"content": "invalid JSON"})
    bad_context, bad_diagnostic = bad_controller.run(
        manager, instruction="find sofa", step=0, agent_pose=None,
    )
    assert bad_context is None
    assert bad_diagnostic["failure_category"] == "API_NO_FINAL"
    assert bad_controller.episode_statistics()["api_no_final_count"] == 1
    assert bad_controller.episode_statistics()["api_failure_count"] == 0


def test_weak_fused_object_is_classified_separately_from_api_failure() -> None:
    manager = _manager()
    _observe_l1(manager, 0)
    sofa, _lamp = _map_objects(
        manager, ("sofa", 1, 4, None), ("lamp", 5, 8, None),
    )
    weak_mask = np.zeros((8, 8), dtype=np.bool_)
    weak_mask[1:7, 1:4] = True
    manager.spatial_map.observe(_frame(), [InstanceMask(
        class_name="sofa", confidence=0.3, mask=weak_mask,
        evidence_id=canonical_evidence_key("scene", "episode", 0, "seg_weak"),
    )], step_id=1)
    assert len(manager.spatial_map.objects[sofa[0]].evidence_ids) == 2
    _observe_l1(manager, 1)
    _observe_l1(manager, 2)
    controller, _ = _controller(
        _query(_tool("find_objects", class_name="sofa"),
               _tool("read_evidence", evidence_id=sofa[1])),
        _final("USE_L3", objects=[sofa[0]], evidence=[sofa[1]]),
    )
    context, diagnostic = controller.run(
        manager, instruction="turn after sofa", step=2, agent_pose=None,
    )
    assert context is None and diagnostic["api_decision"] == "USE_L3"
    assert diagnostic["decision"] == "NO_MEMORY"
    assert diagnostic["failure_category"] == "GRAPH_FUSION_FAILED"
    assert "confidence=0.3" in diagnostic["fallback_reason"]
    assert diagnostic["l3_evidence_valid_count"] == 1  # Strong lamp remains queryable.
    assert controller.episode_statistics()["graph_fusion_failure_count"] == 1
    assert controller.episode_statistics()["api_failure_count"] == 0


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
