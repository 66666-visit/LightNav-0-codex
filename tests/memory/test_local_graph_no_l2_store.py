"""A real L2 store ablation retains the raw Evidence and topology banks."""

from __future__ import annotations

import pytest

from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.local_graph_fusion import LocalGraphFusion
from lightnav.memory.memory_context_builder import inject_memory_context
from lightnav.memory.v3_retrieval import V3Retriever
from lightnav.memory.v3_schemas import EvidenceRecord
from lightnav.memory.v3_store import V3MemoryStore


def _record(event_id: str, step: int, kind: str, **kwargs) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=event_id, scene_id="scene", episode_id="episode",
        rollout_idx=0, step=step, event_type=kind, **kwargs,
    )


def _store(mid_enabled: bool) -> V3MemoryStore:
    store = V3MemoryStore(scope="scene_persistent", mid_enabled=mid_enabled)
    store.begin_episode("scene", "episode", "find the couch")
    detection = store.add(_record(
        "couch", 0, "object_observation", semantic_label="couch",
        detector_name="yolo", raw_detector_confidence=0.91,
        observation_status="observed",
    ))
    landmark = store.add_landmark(detection)
    first = store.add(_record(
        "first_place", 0, "place_observation",
        visual_descriptor=(1, 0), observation_status="observed",
    ))
    first_node = store.add_place(first)
    store.add_relation(
        first_node.node_id, landmark.node_id, "inferred_relation",
        evidence_keys=(first.key, detection.key),
    )
    store.add(_record("command", 1, "action", payload={"description": "forward"}))
    second = store.add(_record(
        "second_place", 2, "place_observation",
        visual_descriptor=(0, 1), observation_status="observed",
    ))
    second_node = store.add_place(second)
    store.add_relation(
        first_node.node_id, second_node.node_id, "inferred_relation",
        evidence_keys=(first.key, second.key),
    )
    return store


def test_l2_disabled_skips_episode_organization_but_keeps_raw_graph() -> None:
    normal = _store(True)
    ablated = _store(False)
    assert normal._journal == ablated._journal
    assert normal.evidence_for_scene("scene") == ablated.evidence_for_scene("scene")
    assert normal.nodes_for_scene("scene") == ablated.nodes_for_scene("scene")
    assert normal.edges_for_scene("scene") == ablated.edges_for_scene("scene")
    assert normal.active_segment(before_step=2) is not None
    assert ablated.active_segment(before_step=2) is None

    def local_card(store: V3MemoryStore, mid_enabled: bool):
        fusion = LocalGraphFusion(LocalGraphConfig(
            enabled=True, scope="scene_persistent", mid_enabled=mid_enabled,
        ))
        fusion.reset_episode("scene", "episode", 0)
        first_place = next(node for node in store.nodes_for_scene("scene")
                           if node.node_type == "place" and node.first_seen_step == 0)
        return fusion.build(
            store, scene_id="scene", episode_id="episode", rollout_idx=0,
            step=3, instruction="find the couch", current_place_id=first_place.node_id,
            rgb_width=300, events=[],
        )

    normal_card = local_card(normal, True)
    ablated_card = local_card(ablated, False)
    assert normal_card is not None and ablated_card is not None
    assert "Local history: couch" in normal_card.text
    assert normal_card == ablated_card
    base = {"video_segments": object(), "conversations": [
        {"from": "human", "value": (
            "Visual observations <video>. <navigation_task>find the couch"
            "</navigation_task>. Predict trajectory."
        )},
        {"from": "gpt", "value": "placeholder"},
    ]}
    normal_prompt = inject_memory_context(base, normal_card, max_chars=320)
    ablated_prompt = inject_memory_context(base, ablated_card, max_chars=320)
    assert normal_prompt["conversations"] == ablated_prompt["conversations"]

    normal_result = V3Retriever(normal).retrieve(
        scene_id="scene", instruction="find couch", current_step=2,
        episode_id="episode", top_k=10,
    )
    ablated_result = V3Retriever(ablated).retrieve(
        scene_id="scene", instruction="find couch", current_step=2,
        episode_id="episode", top_k=10,
    )
    assert normal_result.channel_candidates["episodic"] > 0
    assert ablated_result.channel_candidates["episodic"] == 0

    assert normal.close_episode() is not None
    assert ablated.close_episode() is None
    assert len(normal.episodes_for_scene("scene")) == 1
    assert ablated.episodes_for_scene("scene") == ()
    assert normal.evidence_for_scene("scene") == ablated.evidence_for_scene("scene")
    assert normal.nodes_for_scene("scene") == ablated.nodes_for_scene("scene")
    assert normal.edges_for_scene("scene") == ablated.edges_for_scene("scene")
    with pytest.raises(ValueError, match="already been closed"):
        ablated.begin_episode("scene", "episode", "find couch")


def test_invalid_mid_enabled_is_rejected() -> None:
    with pytest.raises(ValueError, match="mid_enabled"):
        V3MemoryStore(mid_enabled=0)
