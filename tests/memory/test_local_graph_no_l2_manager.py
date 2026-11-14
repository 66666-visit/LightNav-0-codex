"""A/B manager chain: L2 organization changes, L1/L3 and Local Graph do not."""

from __future__ import annotations

import numpy as np

from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.place_recognition import MockPlaceDescriptorExtractor
from lightnav.memory.schemas import MemoryQuery, ObservationPacket, SceneExtraction


def _manager(mid_enabled: bool) -> MemoryManager:
    return MemoryManager(
        extractor=MockSceneExtractor({
            0: SceneExtraction(objects=["archway"], confidence=0.88),
        }),
        strategy="hierarchical", place_enabled=True,
        place_extractor=MockPlaceDescriptorExtractor({
            0: (1, 0), 1: (1, 0), 2: (1, 0),
        }),
        place_every_n_steps=1,
        local_graph_config=LocalGraphConfig(
            enabled=True, scope="scene_persistent", mid_enabled=mid_enabled,
        ),
    )


def _action() -> dict:
    return {
        "action": "velocity_control",
        "action_args": {"linear_velocity": 0.2, "angular_velocity": 0.0},
    }


def test_l2_ablation_preserves_raw_events_l1_l3_and_local_graph_prompt() -> None:
    normal, ablated = _manager(True), _manager(False)
    managers = (normal, ablated)
    for manager in managers:
        manager.reset_episode("episode", "find the archway", scene_id="scene")
    prompt_pairs = []
    for step in range(3):
        contexts = []
        for manager in managers:
            if step:
                manager.record_action(_action(), prediction="<traj_1>")
            events = manager.observe(ObservationPacket(
                step_id=step, rgb=np.full((8, 8, 3), step, dtype=np.uint8),
                instruction="find the archway",
            ))
            contexts.append(manager.get_local_graph_context(
                step=step, rgb_width=8, events=events,
            ))
        prompt_pairs.append(tuple(contexts))
        assert normal.episodic.events == ablated.episodic.events
        assert normal.working.state.to_dict() == ablated.working.state.to_dict()
        assert normal.v3_store.current_short == ablated.v3_store.current_short
        assert normal.v3_store.evidence_for_scene("scene") == ablated.v3_store.evidence_for_scene(
            "scene"
        )
        assert normal.v3_store.nodes_for_scene("scene") == ablated.v3_store.nodes_for_scene(
            "scene"
        )
        assert normal.v3_store.edges_for_scene("scene") == ablated.v3_store.edges_for_scene(
            "scene"
        )
        assert normal.v3_store._journal == ablated.v3_store._journal
        assert contexts[0] == contexts[1]
        for field in ("short_memory_records", "long_memory_nodes", "long_memory_edges"):
            assert normal.last_v3_diagnostic[field] == ablated.last_v3_diagnostic[field]

    assert prompt_pairs[0][0] is not None
    assert prompt_pairs[1][0] is not None
    assert "Local history: archway" in prompt_pairs[1][0].text
    assert "Local history: archway" in prompt_pairs[2][0].text
    assert normal.episodic.index_enabled is True
    assert ablated.episodic.index_enabled is False
    assert normal.episodic.key_events and normal.episodic.segments
    assert ablated.episodic.key_events == [] and ablated.episodic.segments == []
    query = MemoryQuery("find the archway", current_step=2)
    assert normal.episodic.retrieve(query)
    assert ablated.episodic.retrieve(query) == []
    assert normal.retrieve_layers(query)["mid"]
    assert ablated.retrieve_layers(query)["mid"] == []
    assert normal.v3_store.active_segment(before_step=2) is not None
    assert ablated.v3_store.active_segment(before_step=2) is None
    assert normal.last_v3_diagnostic["l2_enabled"] is True
    assert ablated.last_v3_diagnostic["l2_enabled"] is False
    assert normal.last_v3_diagnostic["mid_memory_records"] > 0
    assert ablated.last_v3_diagnostic["mid_memory_records"] == 0
    assert normal.last_v3_diagnostic["mid_candidates"] > 0
    assert ablated.last_v3_diagnostic["mid_candidates"] == 0
    assert normal.last_v3_diagnostic["mid_prompt_records"] == 0
    assert ablated.last_v3_diagnostic["mid_prompt_records"] == 0

    for manager in managers:
        manager.close_episode()
    assert len(normal.v3_store.episodes_for_scene("scene")) == 1
    assert ablated.v3_store.episodes_for_scene("scene") == ()
    assert normal.v3_store.evidence_for_scene("scene") == ablated.v3_store.evidence_for_scene(
        "scene"
    )
    assert normal.v3_store.nodes_for_scene("scene") == ablated.v3_store.nodes_for_scene(
        "scene"
    )
    assert normal.v3_store.edges_for_scene("scene") == ablated.v3_store.edges_for_scene(
        "scene"
    )
