"""Frozen-policy local graph context uses only observed, local RGB evidence."""

from __future__ import annotations

import pytest

from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.local_graph_fusion import LocalGraphFusion
from lightnav.memory.memory_context_builder import audit_prompt_injection, inject_memory_context
from lightnav.memory.schemas import MemoryEvent
from lightnav.memory.v3_schemas import EvidenceRecord
from lightnav.memory.v3_store import V3MemoryStore


def _record(
    event_id: str, *, episode: str = "episode", scene: str = "scene",
    step: int = 0, kind: str = "object_observation",
    label: str | None = "door", confidence: float | None = 0.9,
    descriptor: tuple[float, ...] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        event_id, scene, episode, 0, step, kind, semantic_label=label,
        raw_detector_confidence=confidence if kind in {"object_observation", "landmark"} else None,
        detector_name="yolo" if kind in {"object_observation", "landmark"} else None,
        visual_descriptor=descriptor,
        observation_status="observed",
    )


def _event(record: EvidenceRecord, box: list[float] | None = None) -> MemoryEvent:
    return MemoryEvent(
        event_id=record.evidence_id, step_id=record.step,
        kind=record.event_type, description=f"Observed {record.semantic_label}",
        confidence=record.raw_detector_confidence or 0.0,
        source="yolo", episode_id=record.episode_id,
        metadata={"bbox_xyxy": box} if box else {},
    )


def _fusion(scope: str = "episode_local", **kwargs) -> LocalGraphFusion:
    return LocalGraphFusion(LocalGraphConfig(enabled=True, scope=scope, **kwargs))


def test_current_box_is_camera_relative_and_grounded_in_instruction() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the table")
    door = store.add(_record("table_0", label="dining table"))
    chair = store.add(_record("chair_0", label="chair"))
    fusion = _fusion()
    fusion.reset_episode("scene", "episode", 0)

    context = fusion.build(
        store, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=0, instruction="find the table", current_place_id=None,
        rgb_width=300, events=[
            _event(door, [10, 1, 90, 80]),
            _event(chair, [200, 1, 290, 80]),
        ],
    )

    assert context is not None
    assert "dining table in the left of the current view" in context.text
    assert "chair" not in context.text
    assert len(context.layer_items) == 1
    item = context.layer_items[0]
    assert item.metadata["evidence_ids"] == [door.key]
    assert item.metadata["provenance"][0]["evidence_id"] == door.key
    assert door.key not in context.text
    assert item.metadata["serialized_text"] == item.description


def test_local_card_keeps_original_instruction_and_video_sample() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the table")
    detection = store.add(_record("table_0", label="dining table"))
    fusion = _fusion()
    fusion.reset_episode("scene", "episode", 0)
    context = fusion.build(
        store, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=0, instruction="find the table", current_place_id=None,
        rgb_width=300, events=[_event(detection, [10, 1, 90, 80])],
    )
    video = object()
    base = {
        "video_segments": video,
        "conversations": [
            {"from": "human", "value": (
                "Visual observations <video>. Your assigned task is: "
                "<navigation_task>find the table</navigation_task>. "
                "Predict your future trajectory as tokens."
            )},
            {"from": "gpt", "value": "placeholder"},
        ],
    }
    injected = inject_memory_context(base, context, max_chars=320)
    audit = audit_prompt_injection(base, injected, context, max_chars=320)
    assert audit["roundtrip_ok"]
    assert injected["video_segments"] is video
    assert "<navigation_task>find the table</navigation_task>" in (
        injected["conversations"][0]["value"]
    )
    assert "dining table" in injected["conversations"][0]["value"]


def test_history_stays_on_current_dino_places_one_hop_landmark_edges() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the door")
    near = store.add(_record("near_door", step=0))
    near_landmark = store.add_landmark(near)
    near_place_record = store.add(_record(
        "near_place", step=0, kind="place_observation", label=None,
        confidence=None, descriptor=(1.0, 0.0),
    ))
    near_place = store.add_place(near_place_record)
    store.add_relation(
        near_place.node_id, near_landmark.node_id, "inferred_relation",
        evidence_keys=(near_place_record.key, near.key),
    )

    far = store.add(_record("far_door", step=0))
    far_landmark = store.add_landmark(far)
    far_place_record = store.add(_record(
        "far_place", step=0, kind="place_observation", label=None,
        confidence=None, descriptor=(0.0, 1.0),
    ))
    far_place = store.add_place(far_place_record)
    store.add_relation(
        far_place.node_id, far_landmark.node_id, "inferred_relation",
        evidence_keys=(far_place_record.key, far.key),
    )
    fusion = _fusion()
    fusion.reset_episode("scene", "episode", 0)

    context = fusion.build(
        store, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=1, instruction="find the door", current_place_id=near_place.node_id,
        rgb_width=300, events=[],
    )

    assert context is not None
    assert "possible revisit" in context.text
    assert "current direction and passage unknown" in context.text
    assert context.layer_items[0].metadata["evidence_ids"] == [near.key]
    assert far.key not in context.layer_items[0].metadata["evidence_ids"]
    assert fusion.build(
        store, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=1, instruction="find the door", current_place_id=None,
        rgb_width=300, events=[],
    ) is None


def test_same_state_is_suppressed_then_reactivated_after_confirmed_injection() -> None:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the door")
    fusion = _fusion(reactivation_steps=3)
    fusion.reset_episode("scene", "episode", 0)

    def build_at(step: int):
        record = store.add(_record(f"door_{step}", step=step))
        return fusion.build(
            store, scene_id="scene", episode_id="episode", rollout_idx=0,
            step=step, instruction="find the door", current_place_id=None,
            rgb_width=300, events=[_event(record, [110, 0, 190, 80])],
        ), record

    initial, first = build_at(0)
    assert initial is not None
    fusion.confirm_injected([first.key], 0)
    suppressed, _ = build_at(1)
    assert suppressed is None
    reactivated, third = build_at(3)
    assert reactivated is not None
    fusion.discard_selection()
    retry = fusion.build(
        store, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=3, instruction="find the door", current_place_id=None,
        rgb_width=300, events=[_event(third, [110, 0, 190, 80])],
    )
    assert retry is not None
    fusion.confirm_injected([third.key], 3)
    assert fusion.build(
        store, scene_id="scene", episode_id="episode", rollout_idx=0,
        step=3, instruction="find the door", current_place_id=None,
        rgb_width=300, events=[_event(third, [110, 0, 190, 80])],
    ) is None


def test_persistent_local_history_keeps_episode_provenance() -> None:
    store = V3MemoryStore(scope="scene_persistent")
    store.begin_episode("scene", "old", "find door")
    old = store.add(_record("old_door", episode="old"))
    landmark = store.add_landmark(old)
    old_place = store.add(_record(
        "old_place", episode="old", kind="place_observation",
        label=None, confidence=None, descriptor=(1.0, 0.0),
    ))
    place = store.add_place(old_place)
    store.add_relation(
        place.node_id, landmark.node_id, "inferred_relation",
        evidence_keys=(old_place.key, old.key),
    )
    store.close_episode()
    store.begin_episode("scene", "new", "find door")
    new_place = store.add(_record(
        "new_place", episode="new", kind="place_observation",
        label=None, confidence=None, descriptor=(1.0, 0.0),
    ))
    matched = store.add_place(new_place)
    assert matched.node_id == place.node_id
    fusion = _fusion("scene_persistent")
    fusion.reset_episode("scene", "new", 0)

    context = fusion.build(
        store, scene_id="scene", episode_id="new", rollout_idx=0,
        step=0, instruction="find door", current_place_id=matched.node_id,
        rgb_width=300, events=[],
    )

    assert context is not None
    assert context.layer_items[0].metadata["evidence_ids"] == [old.key]
    assert context.layer_items[0].metadata["provenance"][0]["episode_id"] == "old"
    assert "possible revisit" in context.text


@pytest.mark.parametrize("kwargs", [
    {"scope": "global"}, {"prompt_max_chars": 0},
    {"max_landmarks": True}, {"reactivation_steps": -1},
    {"protocol_version": "rgm_v4_v1"},
])
def test_invalid_config_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        LocalGraphConfig(**kwargs)
