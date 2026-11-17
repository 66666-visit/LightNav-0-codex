"""CPU integration checks for the opt-in DART decay readout.

The fake segmentation/L1 models make this a code-path test, not a navigation
evaluation. It does, however, use the real L3 ObjectMap, SpatialGraphAdapter,
prompt builder and policy wiring to show that decay can change which complete
memory fact reaches the frozen model input.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.habitat.policy import TrajVocabVLNCEPolicy
from lightnav.inference.policies import NavigationPolicy
from lightnav.memory.dart_decay_config import DartDecayConfig
from lightnav.memory.dart_memory_decay import object_slot_id
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.schemas import DetectedObject, SceneExtraction
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.yolo_seg_extractor import SegmentationInstance


class _Engine:
    def __init__(self) -> None:
        self.bundle = SimpleNamespace(
            video_size=(4, 4), video_fps=4, num_history_frames=2,
            pool_enable=False, pool_spatial=1, pool_mode="avg",
            slowfast_tiers=None, predict_horizon=1, action_method="flat",
        )
        self.backend = "hf"
        self.samples: list[dict] = []
        self.last_input_diagnostic: dict | None = None
        self.last_prompt_chars = 0

    def reset_episode_state(self) -> None:
        pass

    def generate(self, sample: dict) -> tuple[str, float]:
        self.samples.append(sample)
        prompt = sample["conversations"][0]["value"]
        trace = sample.get("_rgm_trace", {})
        memory_text = trace.get("memory_text", "")
        visible = memory_text in prompt if memory_text else False
        ids = [ord(char) for char in prompt]
        self.last_input_diagnostic = {
            "final_token_ids_sha256": sha256(
                np.asarray(ids, dtype=np.int32).tobytes()
            ).hexdigest(),
            "final_token_ids": ids,
            "memory_text_present_in_final_tokens": visible,
            "model_visible_evidence_ids": [
                evidence_id for record in trace.get("accepted_records", [])
                for evidence_id in record["evidence_ids"]
            ] if visible else [],
        }
        self.last_prompt_chars = len(prompt)
        return "<traj_1>", 1.0


class _ScriptedSegExtractor:
    """One full-frame instance on scripted steps; depth comes from the step obs."""

    def __init__(self, by_step: dict[int, str]) -> None:
        self.by_step = by_step

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        pass

    def close_episode(self) -> None:
        pass

    def extract(self, rgb: np.ndarray, step_id: int) -> list[SegmentationInstance]:
        label = self.by_step.get(step_id)
        if label is None:
            return []
        mask = np.ones(rgb.shape[:2], dtype=np.bool_)
        return [SegmentationInstance(
            label, 0.95, mask, (0.0, 0.0, float(rgb.shape[1]), float(rgb.shape[0])),
        )]


@pytest.fixture(autouse=True)
def _small_video(monkeypatch):
    monkeypatch.setattr(
        NavigationPolicy, "_convert_frame",
        lambda self, frame: torch.full((3, 4, 4), float(frame[0, 0, 0])),
    )


def _observation(step: int, *, depth_m: float = 2.0) -> tuple[dict, dict]:
    rgb = np.full((8, 8, 3), step, dtype=np.uint8)
    obs = {"rgb": rgb, "depth": np.full((8, 8), depth_m, dtype=np.float32)}
    info = {
        "spatial_rgbd": {
            "status": "ok", "aligned": True, "image_size_hw": [8, 8],
            "K": [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]],
            "agent_pose": {"position": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
            "camera_extrinsics": {
                "position": [0.0, 0.88, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0],
            },
            "camera_frame": "habitat_right_up_back",
            "depth_encoding": "meters", "min_depth_m": 0.1, "max_depth_m": 10.0,
        },
    }
    if step > 0:
        info["spatial_transition_record"] = True
    return obs, info


def _policy(
    mode: str, *, pass_config: bool = True,
) -> tuple[TrajVocabVLNCEPolicy, MemoryManager, _Engine]:
    detections = {
        0: SceneExtraction(
            objects=["chair"], detections=[DetectedObject(
                "chair", 0.95, (0.0, 0.0, 8.0, 8.0),
            )], confidence=0.95, source="fake_l1",
        ),
        2: SceneExtraction(
            objects=["table"], detections=[DetectedObject(
                "table", 0.95, (0.0, 0.0, 8.0, 8.0),
            )], confidence=0.95, source="fake_l1",
        ),
    }
    manager = MemoryManager(
        extractor=MockSceneExtractor(detections), strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True, max_landmarks=1),
        spatial_l3_config=SpatialL3Config(
            mode="inject", seg_model_path="/fake/yolo11n-seg.pt",
            seg_every_n_steps=1,
        ),
        spatial_extractor=_ScriptedSegExtractor({0: "chair", 2: "table"}),
        **(
            {"dart_decay_config": DartDecayConfig(mode=mode)}
            if pass_config else {}
        ),
    )
    centroids = np.zeros((3, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    centroids[2, :, 2] = 0.1
    engine = _Engine()
    policy = TrajVocabVLNCEPolicy(
        engine, centroids, dt=1.0, lin_vel_range=(0.0, 0.3),
        ang_vel_range=(-30.0, 30.0), num_history_frames=2,
        memory_enabled=True, memory_manager=manager,
    )
    policy.reset(
        {"instruction": {"text": "find the chair or table"}},
        memory_episode_id="episode", memory_scene_id="scene",
    )
    return policy, manager, engine


def _act(policy: TrajVocabVLNCEPolicy, step: int, *, depth_m: float) -> None:
    obs, info = _observation(step, depth_m=depth_m)
    policy.act(obs, info)


def test_decay_changes_which_complete_spatial_fact_enters_prompt() -> None:
    off, off_manager, off_engine = _policy("off")
    decay, decay_manager, decay_engine = _policy("decay_only")

    # chair is nearer but older; table is slightly farther but fresher.
    for policy in (off, decay):
        _act(policy, 0, depth_m=0.8)
        _act(policy, 1, depth_m=0.8)
        _act(policy, 2, depth_m=1.0)
        _act(policy, 3, depth_m=1.0)

    off_prompt = off_engine.samples[-1]["conversations"][0]["value"]
    decay_prompt = decay_engine.samples[-1]["conversations"][0]["value"]
    assert "Spatial history: chair" in off_prompt
    assert "Spatial history: table" not in off_prompt
    assert "Spatial history: table" in decay_prompt
    assert "Spatial history: chair" not in decay_prompt

    off_diagnostic = off.last_memory_diagnostic
    decay_diagnostic = decay.last_memory_diagnostic
    assert off_diagnostic["dart_decay"] is None
    decay_rows = decay_diagnostic["dart_decay"]
    assert decay_rows["mode"] == "decay_only"
    assert decay_rows["step"] == 3
    assert decay_rows["decay_applied_count"] == 1
    assert decay_rows["decay_skipped_count"] == 0
    row = decay_rows["items"][0]
    assert row["memory_age"] == 1
    assert row["visit_count"] == 1
    assert row["novelty"] == pytest.approx(1.0)
    assert row["novelty_source"] == "initial_prior"
    assert row["decay_weight"] == pytest.approx(0.8204, abs=1e-3)
    assert decay_diagnostic["spatial_l3"]["context_source"] == "spatial_l3"

    objects = {
        node.class_name: object_id
        for object_id, node in decay_manager.spatial_map.objects.items()
    }
    selection = decay_rows["candidate_selection"]
    assert selection["backend"] == "spatial_l3"
    assert selection["selected_object_ids"] == [objects["table"]]
    ranked = {row["object_id"]: row for row in selection["candidates"]}
    assert ranked[objects["table"]]["rank"] == 1
    assert ranked[objects["chair"]]["rank"] == 2
    assert ranked[objects["chair"]]["decay_weight"] < ranked[objects["table"]]["decay_weight"]

    # The decay run still used the same L3 map and stored Evidence.
    assert set(decay_manager.spatial_map.objects) == set(off_manager.spatial_map.objects)


def test_decay_does_not_mutate_stored_evidence_or_object_geometry() -> None:
    policy, manager, _ = _policy("decay_only")
    _act(policy, 0, depth_m=0.8)
    _act(policy, 1, depth_m=0.8)
    _act(policy, 2, depth_m=1.0)
    assert manager.v3_store is not None and manager.spatial_map is not None

    evidence_before = {
        record.key: deepcopy(record.to_dict())
        for record in manager.v3_store.evidence_for_scene("scene")
    }
    objects_before = {
        object_id: (
            deepcopy(node.center_world), deepcopy(node.bbox_3d),
            deepcopy(node.points_world), node.observation_count,
            list(node.evidence_ids), node.geometry_status,
        )
        for object_id, node in manager.spatial_map.objects.items()
    }

    _act(policy, 3, depth_m=1.0)

    evidence_after = {
        record.key: record.to_dict()
        for record in manager.v3_store.evidence_for_scene("scene")
    }
    objects_after = {
        object_id: (
            deepcopy(node.center_world), deepcopy(node.bbox_3d),
            deepcopy(node.points_world), node.observation_count,
            list(node.evidence_ids), node.geometry_status,
        )
        for object_id, node in manager.spatial_map.objects.items()
    }
    for key, value in evidence_before.items():
        assert evidence_after[key] == value
    assert set(objects_after) == set(objects_before)
    for object_id, before in objects_before.items():
        after = objects_after[object_id]
        assert after[0] == before[0]
        assert after[1] == before[1]
        assert np.array_equal(after[2], before[2])
        assert after[3:] == before[3:]


def test_episode_reset_clears_decay_slots_and_old_objects() -> None:
    policy, manager, engine = _policy("decay_only")
    _act(policy, 0, depth_m=0.8)
    _act(policy, 1, depth_m=0.8)
    _act(policy, 2, depth_m=1.0)
    assert manager.dart_decay is not None
    old_objects = list(manager.spatial_map.objects)
    assert old_objects

    policy.reset(
        {"instruction": {"text": "find the chair or table"}},
        memory_episode_id="episode_two", memory_scene_id="scene",
    )
    assert manager.dart_decay.episode == ("scene", "episode_two", 0)
    assert manager.spatial_map.objects == {}
    for object_id in old_objects:
        assert manager.dart_decay.weight_for(object_slot_id(object_id), step=0) is None

    _act(policy, 0, depth_m=0.8)
    diagnostic = policy.last_memory_diagnostic["dart_decay"]
    assert diagnostic["step"] == 0
    assert diagnostic["decay_skipped_count"] == 0
    assert all(row["visit_count"] == 1 for row in diagnostic["items"])
    assert engine.samples


def test_off_matches_default_policy_prompts_and_tokens() -> None:
    explicit_off, _, explicit_engine = _policy("off")
    default, _, default_engine = _policy("off", pass_config=False)
    for policy in (explicit_off, default):
        _act(policy, 0, depth_m=0.8)
        _act(policy, 1, depth_m=0.8)
        _act(policy, 2, depth_m=1.0)
        _act(policy, 3, depth_m=1.0)
    assert [sample["conversations"] for sample in explicit_engine.samples] == [
        sample["conversations"] for sample in default_engine.samples
    ]
    assert explicit_engine.last_input_diagnostic == default_engine.last_input_diagnostic
    assert explicit_off.last_memory_diagnostic["dart_decay"] is None


def test_dino_place_reobservation_updates_novelty_ema_in_manager() -> None:
    from lightnav.memory.dart_memory_decay import place_slot_id
    from lightnav.memory.place_recognition import MockPlaceDescriptorExtractor
    from lightnav.memory.schemas import ObservationPacket
    from lightnav.memory.v3_config import RGMV3Config

    descriptor = MockPlaceDescriptorExtractor({0: (1.0, 0.0), 1: (1.0, 0.0)})
    manager = MemoryManager(
        extractor=MockSceneExtractor(), strategy="hierarchical",
        place_enabled=True, place_extractor=descriptor, place_every_n_steps=1,
        v3_config=RGMV3Config(enabled=True),
        dart_decay_config=DartDecayConfig(mode="decay_only"),
    )
    manager.reset_episode("episode", "find the chair", scene_id="scene")
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    manager.observe(ObservationPacket(step_id=0, rgb=rgb, instruction="find the chair"))
    assert manager.dart_decay is not None
    nodes = manager.v3_store.nodes_for_scene("scene")
    place = next(node for node in nodes if node.node_type == "place")
    first = manager.dart_decay.weight_for(place_slot_id(place.node_id), step=0)
    assert first is not None
    assert first.novelty == pytest.approx(1.0)
    assert first.novelty_source == "initial_prior"

    manager.observe(ObservationPacket(step_id=1, rgb=rgb, instruction="find the chair"))
    second = manager.dart_decay.weight_for(place_slot_id(place.node_id), step=1)
    assert second is not None
    assert second.novelty == pytest.approx(0.5)
    assert second.novelty_source == "cosine_ema"
    assert second.visit_count == 2
    assert second.decay_weight < first.decay_weight
