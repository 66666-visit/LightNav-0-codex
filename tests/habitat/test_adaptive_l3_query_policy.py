"""Policy integration: Adaptive L3 Query on the original L1 path (CPU only)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.habitat.policy import TrajVocabVLNCEPolicy
from lightnav.inference.policies import NavigationPolicy
from lightnav.memory.adaptive_l3_query import (
    NO_QUERY,
    QUERY_L3,
    USE,
    AdaptiveQueryDecision,
    MockAdaptiveQueryController,
)
from lightnav.memory.adaptive_l3_query_config import AdaptiveL3QueryConfig
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
        self.max_new_tokens = 8

    def reset_episode_state(self) -> None:
        pass

    def generate(self, sample: dict) -> tuple[str, float]:
        self.samples.append(sample)
        prompt = sample["conversations"][0]["value"]
        trace = sample.get("_rgm_trace", {})
        memory_text = trace.get("memory_text", "")
        visible = bool(memory_text and memory_text in prompt)
        ids = [ord(char) for char in prompt]
        self.last_input_diagnostic = {
            "final_token_ids": ids,
            "memory_text_present_in_final_tokens": visible,
            "model_visible_evidence_ids": [
                evidence_id for record in trace.get("accepted_records", [])
                for evidence_id in record["evidence_ids"]
            ] if visible else [],
        }
        self.last_prompt_chars = len(prompt)
        provider = getattr(self, "controller_outputs", None)
        if provider:
            return provider.pop(0), 1.0
        return "<traj_1>", 1.0


class _SegExtractor:
    def __init__(self, label: str = "sofa", *, every_n_steps: int = 2) -> None:
        self.label = label
        self.every_n_steps = every_n_steps

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        pass

    def close_episode(self) -> None:
        pass

    def extract(self, rgb: np.ndarray, step_id: int) -> list[SegmentationInstance]:
        if step_id % self.every_n_steps != 0:
            return []
        mask = np.ones(rgb.shape[:2], dtype=np.bool_)
        return [SegmentationInstance(
            self.label, 0.95, mask,
            (0.0, 0.0, float(rgb.shape[1]), float(rgb.shape[0])),
        )]


class _ClipEncoder:
    feature_dim = 2

    def __init__(
        self, image_feature=(1.0, 0.0), text_feature=(1.0, 0.0), *,
        broken_text: bool = False,
    ) -> None:
        self.image_feature = np.asarray(image_feature, dtype=np.float32)
        self.text_feature = np.asarray(text_feature, dtype=np.float32)
        self.broken_text = broken_text
        self.text_calls = 0

    def encode_image(self, rgb, boxes):
        return np.tile(self.image_feature, (len(boxes), 1))

    def encode_text(self, texts):
        self.text_calls += 1
        if self.broken_text:
            raise RuntimeError("local CLIP encoder failed")
        return np.asarray([self.text_feature], dtype=np.float32)


@pytest.fixture(autouse=True)
def _small_video(monkeypatch):
    monkeypatch.setattr(
        NavigationPolicy, "_convert_frame",
        lambda self, frame: torch.full((3, 4, 4), float(frame[0, 0, 0])),
    )


def _obs(step: int) -> tuple[dict, dict]:
    rgb = np.full((8, 8, 3), step, dtype=np.uint8)
    obs = {"rgb": rgb, "depth": np.full((8, 8), 2.0, dtype=np.float32)}
    info = {
        "spatial_rgbd": {
            "status": "ok", "aligned": True, "image_size_hw": [8, 8],
            "K": [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]],
            "agent_pose": {
                "position": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            },
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
    *, adaptive: bool, instruction: str = "find the chair",
    l1_label: str = "chair", seg_label: str = "sofa",
    decide_script=(), select_script=(), text_feature=(1.0, 0.0),
    prompt_mode: str = "legacy", broken_text: bool = False,
):
    detections = {
        step: SceneExtraction(
            objects=[l1_label], detections=[DetectedObject(
                l1_label, 0.95, (0.0, 0.0, 8.0, 8.0),
            )], confidence=0.95, source="fake_l1",
        )
        for step in (0, 1, 2)
    }
    encoder = _ClipEncoder(text_feature=text_feature, broken_text=broken_text)
    manager = MemoryManager(
        extractor=MockSceneExtractor(detections), strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True, reactivation_steps=1),
        spatial_l3_config=SpatialL3Config(
            mode="shadow", seg_model_path="/fake/yolo11n-seg.pt",
            seg_every_n_steps=2, clip_model_path="/fake/open_clip.pt",
            clip_top_k=3,
        ),
        spatial_extractor=_SegExtractor(seg_label, every_n_steps=2),
        clip_encoder=encoder,
        prompt_mode=prompt_mode,
        adaptive_l3_query_config=AdaptiveL3QueryConfig(enabled=adaptive),
    )
    centroids = np.zeros((3, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    engine = _Engine()
    policy = TrajVocabVLNCEPolicy(
        engine, centroids, dt=1.0, lin_vel_range=(0.0, 0.3),
        ang_vel_range=(-30.0, 30.0), num_history_frames=2,
        memory_enabled=True, memory_manager=manager,
        memory_prompt_mode=prompt_mode,
    )
    policy.reset(
        {"instruction": {"text": instruction}},
        memory_episode_id="episode", memory_scene_id="scene",
    )
    mock = None
    if adaptive:
        mock = MockAdaptiveQueryController(
            decide_script=decide_script, select_script=select_script,
        )
        policy.adaptive_l3_runner.controller = mock
    return policy, manager, engine, encoder, mock


def _act(policy, step: int):
    obs, info = _obs(step)
    return policy.act(obs, info)


def _prompt(engine: _Engine) -> str:
    return engine.samples[-1]["conversations"][0]["value"]


def _sofa_id(manager: MemoryManager) -> str:
    return next(
        object_id for object_id, node in manager.spatial_map.objects.items()
        if node.class_name == "sofa"
    )


def _sofa_evidence(manager: MemoryManager) -> tuple[str, ...]:
    return tuple(manager.spatial_map.objects[_sofa_id(manager)].evidence_ids)


def test_no_query_prompt_is_byte_identical_to_l1_original() -> None:
    baseline, _, baseline_engine, _, _ = _policy(adaptive=False)
    _act(baseline, 0)
    _act(baseline, 1)
    baseline_prompt = _prompt(baseline_engine)

    policy, manager, engine, encoder, mock = _policy(
        adaptive=True,
        decide_script=[
            AdaptiveQueryDecision(NO_QUERY, reason_code="sufficient"),
            AdaptiveQueryDecision(NO_QUERY, reason_code="sufficient"),
        ],
    )
    _act(policy, 0)
    _act(policy, 1)

    assert _prompt(engine) == baseline_prompt
    assert encoder.text_calls == 0
    assert mock.calls == [("decide", 0), ("decide", 1)]
    diagnostic = policy.last_memory_diagnostic["adaptive_l3"]
    assert diagnostic["skipped_reason"] == "controller_no_query"
    assert diagnostic["query_executed"] is False
    assert manager.last_local_graph_status == "built"


def test_l1_has_candidates_but_adaptive_query_adds_verified_fact() -> None:
    policy, manager, engine, encoder, mock = _policy(
        adaptive=True,
        instruction="walk past the table and stop near the sofa", l1_label="table",
        decide_script=[
            AdaptiveQueryDecision(NO_QUERY),
            AdaptiveQueryDecision(QUERY_L3, query_text="sofa"),
        ],
    )
    # The real Object ID exists only after the first observation; fill the
    # scripted selection once the map contains it.
    _act(policy, 0)
    object_id = _sofa_id(manager)
    mock._select_script = (AdaptiveQueryDecision(USE, object_ids=(object_id,)),)
    _act(policy, 1)

    prompt = _prompt(engine)
    diagnostic = policy.last_memory_diagnostic["adaptive_l3"]
    assert encoder.text_calls == 1
    assert manager.last_local_graph_status == "built"
    assert diagnostic["l1_status"] == "built"
    assert diagnostic["query_executed"] is True
    assert diagnostic["validated_object_ids"] == [object_id]
    assert diagnostic["validated_evidence_ids"] == list(_sofa_evidence(manager))
    assert diagnostic["context_source"] == "l1_plus_adaptive_l3"
    assert diagnostic["prompt_verified"] is True
    assert "Current RGB: instruction landmark table" in prompt
    assert "Verified L3 history" in prompt
    assert object_id in diagnostic["validated_object_ids"]
    assert _sofa_evidence(manager)[0] in diagnostic["validated_evidence_ids"]
    assert "object:" in prompt and "evidence:" in prompt
    assert prompt.count("</navigation_task>.") == 1

    baseline, _, baseline_engine, _, _ = _policy(
        adaptive=False,
        instruction="walk past the table and stop near the sofa", l1_label="table",
    )
    _act(baseline, 0)
    _act(baseline, 1)
    base_prompt = _prompt(baseline_engine)
    prefix = base_prompt.split("</navigation_task>.", 1)[0]
    assert prompt.startswith(prefix)
    # Removing exactly the verified L3 block recovers the baseline Prompt.
    l3_start = prompt.index("[Verified L3 history")
    l3_end = prompt.index("\nYou may be", l3_start) + 1
    assert prompt[:l3_start] + prompt[l3_end:] == base_prompt


def test_no_clip_match_matches_the_baseline_prompt_exactly() -> None:
    baseline, _, baseline_engine, _, _ = _policy(adaptive=False)
    _act(baseline, 0)
    _act(baseline, 1)

    policy, _, engine, encoder, _ = _policy(
        adaptive=True,
        decide_script=[
            AdaptiveQueryDecision(NO_QUERY),
            AdaptiveQueryDecision(QUERY_L3, query_text="sofa"),
        ],
        text_feature=(-1.0, 0.0),
    )
    _act(policy, 0)
    _act(policy, 1)

    assert encoder.text_calls == 1
    assert _prompt(engine) == _prompt(baseline_engine)
    diagnostic = policy.last_memory_diagnostic["adaptive_l3"]
    assert diagnostic["skipped_reason"] == "no_similarity_above_floor"
    assert "Verified L3 history" not in policy.last_memory_diagnostic["memory_text"]


def test_l1_empty_still_allows_one_adaptive_query() -> None:
    policy, manager, engine, _, mock = _policy(
        adaptive=True, instruction="find the sofa", l1_label="chair",
        decide_script=[
            AdaptiveQueryDecision(NO_QUERY),
            AdaptiveQueryDecision(QUERY_L3, query_text="sofa"),
        ],
    )
    _act(policy, 0)
    object_id = _sofa_id(manager)
    mock._select_script = (AdaptiveQueryDecision(USE, object_ids=(object_id,)),)
    _act(policy, 1)

    prompt = _prompt(engine)
    diagnostic = policy.last_memory_diagnostic["adaptive_l3"]
    assert manager.last_local_graph_status == "no_candidates"
    assert diagnostic["context_source"] == "adaptive_l3"
    assert diagnostic["l1_included"] is False
    assert diagnostic["validated_object_ids"] == [object_id]
    assert "Verified L3 history" in prompt


def test_episode_reset_clears_controller_calls_and_diagnostics() -> None:
    policy, manager, _, _, mock = _policy(
        adaptive=True,
        decide_script=[
            AdaptiveQueryDecision(NO_QUERY),
            AdaptiveQueryDecision(QUERY_L3, query_text="sofa"),
        ],
    )
    _act(policy, 0)
    object_id = _sofa_id(manager)
    mock._select_script = (AdaptiveQueryDecision(USE, object_ids=(object_id,)),)
    _act(policy, 1)
    assert mock.calls == [("decide", 0), ("decide", 1), ("select", 1)]

    policy.reset(
        {"instruction": {"text": "find the chair"}},
        memory_episode_id="episode-2", memory_scene_id="scene",
    )
    assert mock.calls == []
    assert policy.adaptive_l3_runner.last_diagnostic == {}
    assert manager.last_adaptive_l3_diagnostic == {}
    assert manager.spatial_map.objects == {}

    _act(policy, 0)
    assert mock.calls == [("decide", 0)]
    assert policy.last_memory_diagnostic["adaptive_l3"]["skipped_reason"] == (
        "controller_no_query"
    )


def test_l1_original_only_no_query_keeps_the_baseline_prompt() -> None:
    baseline, _, baseline_engine, _, _ = _policy(
        adaptive=False, prompt_mode="l1_original_only",
    )
    _act(baseline, 0)
    _act(baseline, 1)

    policy, _, engine, encoder, _ = _policy(
        adaptive=True, prompt_mode="l1_original_only",
        decide_script=[AdaptiveQueryDecision(NO_QUERY), AdaptiveQueryDecision(NO_QUERY)],
    )
    _act(policy, 0)
    _act(policy, 1)

    assert _prompt(engine) == _prompt(baseline_engine)
    assert encoder.text_calls == 0
    context = policy.last_memory_diagnostic["prompt_context"]
    assert context["adaptive_l3"]["skipped_reason"] == "controller_no_query"


def test_l1_original_only_injects_a_verified_fact_after_the_l1_card() -> None:
    policy, manager, engine, _, mock = _policy(
        adaptive=True, prompt_mode="l1_original_only",
        decide_script=[
            AdaptiveQueryDecision(NO_QUERY),
            AdaptiveQueryDecision(QUERY_L3, query_text="sofa"),
        ],
    )
    _act(policy, 0)
    object_id = _sofa_id(manager)
    mock._select_script = (AdaptiveQueryDecision(USE, object_ids=(object_id,)),)
    _act(policy, 1)

    prompt = _prompt(engine)
    context = policy.last_memory_diagnostic["prompt_context"]
    assert context["source"] == "l1_plus_adaptive_l3"
    assert context["adaptive_l3"]["validated_object_ids"] == [object_id]
    assert context["adaptive_l3"]["prompt_verified"] is True
    assert "Verified L3 history" in prompt
    assert "Current RGB: instruction landmark chair" in prompt
    assert policy.last_memory_diagnostic["actual_injection"] is True


def test_adaptive_requires_shadow_mode_in_the_manager() -> None:
    with pytest.raises(ValueError, match="shadow"):
        MemoryManager(
            strategy="hierarchical",
            local_graph_config=LocalGraphConfig(enabled=True, reactivation_steps=1),
            spatial_l3_config=SpatialL3Config(
                mode="inject", seg_model_path="/fake/seg.pt",
                clip_model_path="/fake/clip.pt",
            ),
            spatial_extractor=_SegExtractor(),
            clip_encoder=_ClipEncoder(),
            adaptive_l3_query_config=AdaptiveL3QueryConfig(enabled=True),
        )


def test_adaptive_default_off_does_not_construct_a_runner() -> None:
    policy, manager, _, _, _ = _policy(adaptive=False)
    assert manager.adaptive_l3_query_config.enabled is False
    assert policy.adaptive_l3_runner is None


def test_local_clip_error_keeps_the_baseline_prompt_byte_identical() -> None:
    baseline, _, baseline_engine, _, _ = _policy(adaptive=False)
    _act(baseline, 0)
    _act(baseline, 1)

    policy, _, engine, encoder, _ = _policy(
        adaptive=True, broken_text=True,
        decide_script=[
            AdaptiveQueryDecision(NO_QUERY),
            AdaptiveQueryDecision(QUERY_L3, query_text="sofa"),
        ],
    )
    _act(policy, 0)
    _act(policy, 1)

    assert encoder.text_calls == 1
    assert _prompt(engine) == _prompt(baseline_engine)
    diagnostic = policy.last_memory_diagnostic["adaptive_l3"]
    assert diagnostic["skipped_reason"] == "query_error"
    assert "local CLIP encoder failed" in diagnostic["error"]
    assert "Verified L3 history" not in _prompt(engine)
