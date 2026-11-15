"""CPU integration checks for ConceptGraphs-Lite shadow and prompt selection."""

from __future__ import annotations

import random
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.habitat.policy import TrajVocabVLNCEPolicy
from lightnav.inference.policies import NavigationPolicy
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.schemas import DetectedObject, SceneExtraction
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.yolo_seg_extractor import SegmentationInstance


class _Engine:
    def __init__(self, *, reject_spatial: bool = False) -> None:
        self.bundle = SimpleNamespace(
            video_size=(4, 4), video_fps=4, num_history_frames=2,
            pool_enable=False, pool_spatial=1, pool_mode="avg",
            slowfast_tiers=None, predict_horizon=1, action_method="flat",
        )
        self.backend = "hf"
        self.reject_spatial = reject_spatial
        self.samples: list[dict] = []
        self.draws: list[float] = []
        self.last_input_diagnostic: dict | None = None
        self.last_prompt_chars = 0

    def reset_episode_state(self) -> None:
        pass

    def generate(self, sample: dict) -> tuple[str, float]:
        self.samples.append(sample)
        prompt = sample["conversations"][0]["value"]
        if self.reject_spatial and "Spatial history:" in prompt:
            raise ValueError("processor input length exceeded")
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
        draw = random.random() + float(np.random.random()) + float(torch.rand(()))
        self.draws.append(draw)
        return ("<traj_1>" if draw < 1.5 else "<traj_2>"), 1.0


class _SegExtractor:
    def __init__(self, *, confidence: float = 0.95) -> None:
        self.confidence = confidence

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        pass

    def close_episode(self) -> None:
        pass

    def extract(self, rgb: np.ndarray, step_id: int) -> list[SegmentationInstance]:
        # Deliberate global RNG consumption catches a shadow-mode action leak.
        random.random()
        np.random.random()
        torch.rand(())
        mask = np.ones(rgb.shape[:2], dtype=np.bool_)
        return [SegmentationInstance(
            "chair", self.confidence, mask,
            (0.0, 0.0, float(rgb.shape[1]), float(rgb.shape[0])),
        )]


@pytest.fixture(autouse=True)
def _small_video(monkeypatch):
    monkeypatch.setattr(
        NavigationPolicy, "_convert_frame",
        lambda self, frame: torch.full((3, 4, 4), float(frame[0, 0, 0])),
    )


def _spatial_obs(step: int, *, geometry: bool = True) -> tuple[dict, dict]:
    rgb = np.full((8, 8, 3), step, dtype=np.uint8)
    obs = {"rgb": rgb, "depth": np.full((8, 8), 2.0, dtype=np.float32)}
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
    if not geometry:
        info["spatial_rgbd"]["status"] = "unavailable"
    if step > 0:
        info["spatial_transition_record"] = True
    return obs, info


def _policy(mode: str, *, low_confidence: bool = False, reject_spatial: bool = False):
    detections = {
        step: SceneExtraction(
            objects=["chair"], detections=[DetectedObject(
                "chair", 0.95, (0.0, 0.0, 8.0, 8.0),
            )], confidence=0.95, source="fake_l1",
        ) for step in (0, 1)
    }
    spatial = SpatialL3Config(
        mode=mode, seg_model_path="/fake/yolo11n-seg.pt",
        seg_every_n_steps=1,
    )
    manager = MemoryManager(
        extractor=MockSceneExtractor(detections), strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=spatial,
        spatial_extractor=_SegExtractor(confidence=0.3 if low_confidence else 0.95)
        if mode != "off" else None,
    )
    centroids = np.zeros((3, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    centroids[2, :, 2] = 0.1
    engine = _Engine(reject_spatial=reject_spatial)
    policy = TrajVocabVLNCEPolicy(
        engine, centroids, dt=1.0, lin_vel_range=(0.0, 0.3),
        ang_vel_range=(-30.0, 30.0), num_history_frames=2,
        memory_enabled=True, memory_manager=manager,
    )
    policy.reset(
        {"instruction": {"text": "find the chair"}},
        memory_episode_id="episode", memory_scene_id="scene",
    )
    return policy, manager, engine


def _act(policy, step: int, *, geometry: bool = True):
    obs, info = _spatial_obs(step, geometry=geometry)
    return policy.act(obs, info)


def test_shadow_keeps_exact_baseline_prompt_tokens_and_action() -> None:
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    baseline, baseline_manager, baseline_engine = _policy("off")
    baseline_actions = [_act(baseline, step) for step in (0, 1)]
    baseline_samples = list(baseline_engine.samples)
    baseline_draws = list(baseline_engine.draws)
    baseline_ids = baseline_engine.last_input_diagnostic["final_token_ids"]
    baseline_hash = baseline_engine.last_input_diagnostic["final_token_ids_sha256"]
    baseline_store_count = len(baseline_manager.v3_store.evidence_for_scene("scene"))

    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    shadow, shadow_manager, shadow_engine = _policy("shadow")
    shadow_actions = [_act(shadow, step) for step in (0, 1)]

    assert shadow_actions == baseline_actions
    assert shadow_engine.draws == baseline_draws
    assert [item["conversations"] for item in shadow_engine.samples] == [
        item["conversations"] for item in baseline_samples
    ]
    assert shadow_engine.last_input_diagnostic["final_token_ids"] == baseline_ids
    assert shadow_engine.last_input_diagnostic["final_token_ids_sha256"] == baseline_hash
    assert len(shadow_manager.v3_store.evidence_for_scene("scene")) == baseline_store_count
    assert len(shadow_manager.spatial_map.objects) == 1
    assert shadow.last_memory_diagnostic["spatial_l3"]["candidate_count"] == 1
    assert shadow.last_memory_diagnostic["spatial_l3"]["shadow_no_spatial_injection"]
    assert "Spatial history:" not in shadow_engine.samples[-1]["conversations"][0]["value"]


def test_shadow_extractor_failure_keeps_original_model_path() -> None:
    policy, manager, engine = _policy("shadow")

    def fail(rgb, step_id):
        raise AttributeError("broken fake extractor")

    manager.spatial_extractor.extract = fail
    _act(policy, 0)

    assert len(engine.samples) == 1
    assert policy.last_memory_diagnostic["spatial_l3"]["rejection_reason"].startswith(
        "AttributeError:"
    )
    assert "Spatial history:" not in engine.samples[0]["conversations"][0]["value"]


def test_injection_uses_historical_l3_evidence_and_token_audit() -> None:
    policy, manager, engine = _policy("inject")
    _act(policy, 0)
    _act(policy, 1)

    prompt = engine.samples[-1]["conversations"][0]["value"]
    diagnostic = policy.last_memory_diagnostic
    assert "Spatial history: chair" in prompt
    assert "passage unknown" in prompt
    assert diagnostic["spatial_l3"]["context_source"] == "spatial_l3"
    assert diagnostic["actual_injection"] is True
    assert diagnostic["actual_historical_injection"] is True
    assert diagnostic["prompt_hash_before"] != diagnostic["prompt_hash_after"]
    assert diagnostic["injected_evidence_ids"] == diagnostic["injected_historical_evidence_ids"]
    assert diagnostic["injected_evidence_ids"][0] in manager.spatial_map.evidence
    assert manager.v3_store.get_evidence(diagnostic["injected_evidence_ids"][0]) is None
    spatial_context = manager.get_spatial_context(step=1)
    assert spatial_context.layer_items[0].metadata["history_viewpoint_world"] == (
        0.0, 0.88, 0.0,
    )


@pytest.mark.parametrize("geometry,low_confidence", [(False, False), (True, True)])
def test_injection_falls_back_to_original_local_graph(geometry, low_confidence) -> None:
    policy, manager, engine = _policy("inject", low_confidence=low_confidence)
    _act(policy, 0)
    _act(policy, 1, geometry=geometry)

    diagnostic = policy.last_memory_diagnostic
    assert diagnostic["spatial_l3"]["context_source"] == "local_graph"
    assert diagnostic["spatial_l3"]["fallback_reason"] == "no_trusted_spatial_candidate"
    assert "Spatial history:" not in engine.samples[-1]["conversations"][0]["value"]
    assert manager.v3_store is not None


def test_low_confidence_update_cannot_move_an_injected_object_center() -> None:
    policy, manager, engine = _policy("inject")
    _act(policy, 0)
    manager.spatial_extractor.confidence = 0.3
    _act(policy, 1)

    assert manager.spatial_map.objects
    assert policy.last_memory_diagnostic["spatial_l3"]["context_source"] == "local_graph"
    assert "Spatial history:" not in engine.samples[-1]["conversations"][0]["value"]


def test_spatial_processor_overflow_retries_local_graph() -> None:
    policy, _, engine = _policy("inject", reject_spatial=True)
    _act(policy, 0)
    _act(policy, 1)

    diagnostic = policy.last_memory_diagnostic
    assert diagnostic["processor_length_rejections"] == 1
    assert diagnostic["spatial_l3"]["context_source"] == "local_graph"
    assert "Spatial history:" not in engine.samples[-1]["conversations"][0]["value"]
