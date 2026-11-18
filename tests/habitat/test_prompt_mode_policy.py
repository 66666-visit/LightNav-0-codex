"""Prompt-mode policy integration: legacy equality, layer isolation and budgets."""

from __future__ import annotations

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

MODES = (
    "legacy", "l1_original_only", "l1_organized_only", "l3_original_only",
    "l3_organized_only", "l1_l3_original_concat", "l1_l3_joint",
    "l1_original_l3_organized_concat", "l1_original_l3_organized_joint",
)


class Engine:
    def __init__(self, *, reject_memory_once: bool = False) -> None:
        self.bundle = SimpleNamespace(
            video_size=(4, 4), video_fps=4, num_history_frames=2,
            pool_enable=False, pool_spatial=1, pool_mode="avg",
            slowfast_tiers=None, predict_horizon=1, action_method="flat",
        )
        self.backend = "hf"
        self.samples: list[dict] = []
        self.last_input_diagnostic: dict | None = None
        self.last_prompt_chars = 0
        self.reject_memory_once = reject_memory_once
        self.rejections = 0

    def reset_episode_state(self) -> None:
        pass

    def generate(self, sample: dict) -> tuple[str, float]:
        prompt = sample["conversations"][0]["value"]
        trace = sample.get("_rgm_trace", {})
        memory_text = trace.get("memory_text", "")
        if self.reject_memory_once and memory_text and not self.rejections:
            self.rejections += 1
            raise ValueError("processor input length exceeded")
        self.samples.append(sample)
        visible = bool(memory_text and memory_text in prompt)
        ids = [ord(char) for char in prompt]
        self.last_input_diagnostic = {
            "final_token_ids_sha256": "x",
            "final_token_ids": ids,
            "memory_text_present_in_final_tokens": visible,
            "final_token_count": len(ids),
            "model_visible_evidence_ids": [
                evidence_id for record in trace.get("accepted_records", [])
                for evidence_id in record["evidence_ids"]
            ] if visible else [],
            "injected_memory_tokens": None,
        }
        self.last_prompt_chars = len(prompt)
        return "<traj_1>", 1.0


class SegExtractor:
    def __init__(self, *, explode: bool = False) -> None:
        self.calls = 0
        self.explode = explode

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        pass

    def close_episode(self) -> None:
        pass

    def extract(self, rgb: np.ndarray, step_id: int) -> list[SegmentationInstance]:
        self.calls += 1
        if self.explode:
            raise AssertionError("L3 segmentation must not run in an L1-only mode")
        mask = np.ones(rgb.shape[:2], dtype=bool)
        return [SegmentationInstance(
            "chair", 0.95, mask,
            (0.0, 0.0, float(rgb.shape[1]), float(rgb.shape[0])),
        )]


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
    mode: str, *, spatial_mode: str = "inject", labels: tuple[str, ...] = ("chair",),
    seg_explode: bool = False, reject_memory_once: bool = False,
    reactivation_steps: int = 8,
):
    detections = {
        step: SceneExtraction(
            objects=list(labels),
            detections=[
                DetectedObject(label, 0.95, (0.0, 0.0, 8.0, 8.0)) for label in labels
            ],
            confidence=0.95, source="fake_l1",
        )
        for step in (0, 1, 2)
    }
    manager = MemoryManager(
        extractor=MockSceneExtractor(detections), strategy="hierarchical",
        local_graph_config=LocalGraphConfig(
            enabled=True, reactivation_steps=reactivation_steps,
        ),
        spatial_l3_config=SpatialL3Config(
            mode=spatial_mode, seg_model_path="/fake/yolo11n-seg.pt",
            seg_every_n_steps=1,
        ),
        spatial_extractor=SegExtractor(explode=seg_explode)
        if spatial_mode != "off" else None,
        prompt_mode=mode,
    )
    centroids = np.zeros((3, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    engine = Engine(reject_memory_once=reject_memory_once)
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


def _run(policy, steps: int) -> list[str]:
    prompts = []
    for step in range(steps):
        obs, info = _obs(step)
        policy.act(obs, info)
        prompts.append(policy.last_prompt_text or "")
    return prompts


def test_legacy_and_l1_original_only_are_byte_identical_when_l3_is_off() -> None:
    legacy, _, legacy_engine = _policy("legacy", spatial_mode="off")
    legacy_prompts = _run(legacy, 2)
    l1, _, l1_engine = _policy("l1_original_only", spatial_mode="off")
    l1_prompts = _run(l1, 2)

    assert legacy_prompts == l1_prompts
    assert [sample["conversations"] for sample in legacy_engine.samples] == [
        sample["conversations"] for sample in l1_engine.samples
    ]
    assert legacy.last_memory_diagnostic["memory_backend"] == "local_graph"
    assert l1.last_memory_diagnostic["prompt_mode"] == "l1_original_only"


def test_l1_only_modes_never_run_the_l3_sidecar_or_inject_map_facts() -> None:
    for mode in ("l1_original_only", "l1_organized_only"):
        policy, manager, engine = _policy(mode, seg_explode=True)
        _run(policy, 2)
        assert manager.spatial_extractor.calls == 0
        assert manager.spatial_map is not None
        assert manager.spatial_map.objects == {}
        prompt = engine.samples[-1]["conversations"][0]["value"]
        assert "Spatial history:" not in prompt
        assert "[Navigation Context - L3]" not in prompt
        assert policy.last_memory_diagnostic["prompt_context"]["layers"].get("l3") is None


def test_l3_only_mode_never_calls_the_local_graph_builder_or_injects_l1_text() -> None:
    policy, manager, engine = _policy("l3_organized_only")

    def forbidden(*args, **kwargs):
        raise AssertionError("local graph prompt building must not run in an L3-only mode")

    manager.local_graph_fusion.build = forbidden
    _run(policy, 2)
    prompt = engine.samples[-1]["conversations"][0]["value"]
    assert "Current RGB:" not in prompt
    assert "[Navigation Context - L1" not in prompt
    assert "Spatial history:" in prompt or "[Navigation Context - L3]" in prompt
    assert "l1" not in policy.last_memory_diagnostic["prompt_context"]["layers"]


def test_original_l1_organized_l3_concat_keeps_the_original_l1_card() -> None:
    policy, manager, engine = _policy(
        "l1_original_l3_organized_concat", reactivation_steps=1,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("the new concat mode must not run the L1 organizer")

    manager.l1_prompt_organizer.build = forbidden
    _run(policy, 3)

    prompt = engine.samples[-1]["conversations"][0]["value"]
    assert "[Navigation Memory]" in prompt
    assert "Current RGB:" in prompt
    assert "[Navigation Context - L3]" in prompt
    assert "not a path" in prompt
    diagnostic = policy.last_memory_diagnostic["prompt_context"]
    assert diagnostic["source"] == "concat"
    assert diagnostic["layers"]["l1"]["source"] == "local_graph_fusion"
    assert diagnostic["layers"]["l3"]["source"] == "l3_prompt_organizer"


def test_lossless_joint_keeps_original_l1_and_does_not_run_the_l1_organizer() -> None:
    policy, manager, engine = _policy(
        "l1_original_l3_organized_joint", reactivation_steps=1,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("lossless joint mode must not run the L1 organizer")

    manager.l1_prompt_organizer.build = forbidden
    _run(policy, 3)

    prompt = engine.samples[-1]["conversations"][0]["value"]
    assert "[Navigation Context - L1+L3]" in prompt
    assert "Current RGB:" in prompt
    assert "Now:" not in prompt
    assert "not a path" in prompt
    diagnostic = policy.last_memory_diagnostic["prompt_context"]
    assert diagnostic["source"] == "joint"
    assert diagnostic["layers"]["l1"]["source"] == "local_graph_fusion"
    assert diagnostic["layers"]["joint"]["source"] == "joint_prompt_builder"
    assert diagnostic["layer_item_counts"]["l1"] >= 1
    assert diagnostic["layer_item_counts"]["l3"] >= 1


def test_lossless_joint_keeps_the_exact_l1_lines_from_l1_original_only() -> None:
    l1_policy, l1_manager, _ = _policy(
        "l1_original_only", reactivation_steps=1,
    )
    joint_policy, joint_manager, _ = _policy(
        "l1_original_l3_organized_joint", reactivation_steps=1,
    )
    for step in range(3):
        obs, info = _obs(step)
        l1_policy.act(obs, info)
        joint_policy.act(obs, info)
        l1_context = l1_manager.last_prompt_context
        joint_context = joint_manager.last_prompt_context
        assert l1_context is not None and joint_context is not None
        l1_lines = [
            item.metadata["serialized_text"]
            for item in (l1_context.context.layer_items
                         if l1_context.context is not None else [])
        ]
        joint_lines = [
            item.metadata["serialized_text"]
            for item in (joint_context.context.layer_items
                         if joint_context.context is not None else [])
            if item.metadata.get("l1_prompt_preserved") is True
        ]
        assert joint_lines == l1_lines
        assert l1_context.diagnostics["layers"]["l1"]["prompt_chars"] == (
            joint_context.diagnostics["layers"]["l1"]["prompt_chars"]
        )


def test_joint_mode_aligns_and_records_layer_contributions() -> None:
    policy, _, engine = _policy("l1_l3_joint")
    _run(policy, 2)

    prompt = engine.samples[-1]["conversations"][0]["value"]
    assert "now visible" in prompt
    assert "map" in prompt and "passage unknown" in prompt
    diagnostic = policy.last_memory_diagnostic["prompt_context"]
    assert diagnostic["layers"]["joint"]["alignment"]["confirmed"] >= 1
    assert diagnostic["layer_item_counts"]["joint"] >= 1
    assert diagnostic["layers"]["l1"]["item_count"] >= 1
    assert diagnostic["budget_chars"] == 320


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_respects_the_original_character_budget(mode: str) -> None:
    policy, manager, _ = _policy(mode)
    _run(policy, 2)
    diagnostic = policy.last_memory_diagnostic
    if mode == "legacy":
        assert diagnostic["memory_backend"] == "local_graph"
        return
    assert len(policy.last_memory_text) <= 320
    assert diagnostic["prompt_context"]["budget_chars"] == 320
    assert manager.prompt_budget_chars == 320


def test_processor_overflow_retries_without_memory_and_keeps_original_prompt() -> None:
    policy, _, engine = _policy("l1_organized_only", reject_memory_once=True)
    obs, info = _obs(0)
    policy.act(obs, info)

    assert engine.rejections == 1
    diagnostic = policy.last_memory_diagnostic
    assert diagnostic["processor_length_rejections"] == 1
    assert diagnostic["prompt_context"]["prompt_fallback_reason"] == (
        "all prompt-mode items exceeded the Processor budget"
    )
    assert policy.last_memory_text == ""
    assert "<navigation_task>find the chair</navigation_task>" in (
        engine.samples[-1]["conversations"][0]["value"]
    )


def test_episode_state_isolation_resets_context_between_episodes() -> None:
    policy, _, engine = _policy("l1_organized_only", labels=("chair",))
    _run(policy, 2)
    assert "chair" in engine.samples[-1]["conversations"][0]["value"]

    policy.reset(
        {"instruction": {"text": "find the table"}},
        memory_episode_id="episode_2", memory_scene_id="scene",
    )
    # The same extractor fixture has no table at the new episode; stale chair
    # evidence must not leak into the new instruction's context.
    _run(policy, 1)
    prompt = engine.samples[-1]["conversations"][0]["value"]
    assert "chair" not in prompt
    assert policy.last_memory_diagnostic["prompt_context"]["layers"]["l1"]["item_count"] == 0


def test_legacy_keeps_the_original_prompt_text_and_diagnostic_shape() -> None:
    policy, _, engine = _policy("legacy", spatial_mode="off")
    _run(policy, 2)
    diagnostic = policy.last_memory_diagnostic
    assert diagnostic["memory_backend"] == "local_graph"
    assert "prompt_context" not in diagnostic
    assert "prompt_mode" not in diagnostic
    base = engine.samples[-1]["conversations"][0]["value"]
    assert "<navigation_task>find the chair</navigation_task>" in base
    # The legacy block is the original `[Navigation Memory]` card.
    if policy.last_memory_text:
        assert policy.last_memory_text.startswith("[Navigation Memory]")


def test_concat_confirmation_only_forwards_l1_evidence_to_local_graph() -> None:
    _, manager, _ = _policy("l1_l3_original_concat")
    calls: list[list[str]] = []
    manager._active_prompt_fusion = manager.prompt_local_graph_fusion
    manager.prompt_local_graph_fusion.confirm_injected = (
        lambda evidence_ids, step: calls.append(list(evidence_ids))
    )
    manager._prompt_context_pending = True
    manager._prompt_context_pending_ids = frozenset({"l1_chair"})
    manager.confirm_prompt_context(["l1_chair", "l3_seg_0"], step=2)
    assert calls == [["l1_chair"]]
    assert manager._prompt_context_pending is False


def test_joint_episode_reset_removes_previous_l3_objects_and_l1_facts() -> None:
    policy, manager, engine = _policy("l1_l3_joint", labels=("chair",))
    _run(policy, 2)
    assert "chair" in engine.samples[-1]["conversations"][0]["value"]
    assert manager.spatial_map is not None and manager.spatial_map.objects

    manager.extractor.by_step = {
        0: SceneExtraction(
            objects=["table"],
            detections=[DetectedObject("table", 0.95, (0.0, 0.0, 8.0, 8.0))],
            confidence=0.95, source="fake_l1",
        ),
    }
    policy.reset(
        {"instruction": {"text": "find the table"}},
        memory_episode_id="episode_2", memory_scene_id="scene",
    )
    assert manager.spatial_map is not None and manager.spatial_map.objects == {}
    _run(policy, 1)
    prompt = engine.samples[-1]["conversations"][0]["value"]
    assert "chair" not in prompt
    assert "table" in prompt
    assert policy.last_memory_diagnostic["prompt_context"]["layers"]["l1"]["item_count"] == 1


def test_concat_prompt_with_mixed_l1_and_l3_evidence_does_not_break_action() -> None:
    policy, manager, engine = _policy("l1_l3_original_concat")
    # Use a larger budget here only to force both original blocks into one
    # prompt, so the confirmation path sees mixed L1 and L3 evidence ids.
    manager.prompt_budget_chars = 600
    policy.reset(
        {"instruction": {"text": "find the chair and the door"}},
        memory_episode_id="episode", memory_scene_id="scene",
    )
    manager.extractor.by_step = {
        0: SceneExtraction(
            objects=["chair"],
            detections=[DetectedObject("chair", 0.95, (0.0, 0.0, 8.0, 8.0))],
            confidence=0.95, source="fake_l1",
        ),
        1: SceneExtraction(
            objects=["door"],
            detections=[DetectedObject("door", 0.95, (0.0, 0.0, 8.0, 8.0))],
            confidence=0.95, source="fake_l1",
        ),
    }
    _run(policy, 2)
    prompt = engine.samples[-1]["conversations"][0]["value"]
    assert "Local history" in prompt or "Current RGB" in prompt
    assert "Spatial history:" in prompt
    diagnostic = policy.last_memory_diagnostic
    assert set(diagnostic["injected_evidence_ids"]) >= set(
        diagnostic["prompt_context"]["injected_evidence_ids"]
    )
    assert diagnostic["actual_injection"] is True


def test_l3_original_only_matches_the_legacy_spatial_block_when_available() -> None:
    legacy, _, legacy_engine = _policy("legacy", spatial_mode="inject")
    legacy_prompts = _run(legacy, 2)
    l3, _, l3_engine = _policy("l3_original_only", spatial_mode="inject")
    l3_prompts = _run(l3, 2)

    # Step 0 has no historical L3 candidate: legacy falls back to Local Graph,
    # while L3-only must not inject L1 text. Step 1 has the same L3 block.
    assert "Spatial history:" not in l3_prompts[0]
    assert "Spatial history: chair" in legacy_prompts[1]
    assert "Spatial history: chair" in l3_prompts[1]
    assert legacy_prompts[1] == l3_prompts[1]


def test_prompt_builder_failure_falls_back_to_the_original_sample() -> None:
    policy, manager, engine = _policy("l1_l3_joint")

    def fail(**kwargs):
        raise RuntimeError("organizer exploded")

    manager.build_prompt_context = fail
    obs, info = _obs(0)
    action = policy.act(obs, info)

    assert action["action"] == "velocity_control"
    prompt = engine.samples[-1]["conversations"][0]["value"]
    assert "<navigation_task>find the chair</navigation_task>" in prompt
    assert "[Navigation Context" not in prompt
    diagnostic = policy.last_memory_diagnostic
    assert diagnostic["memory_text"] == ""
    assert diagnostic["prompt_fallback_reason"].startswith(
        "prompt_context: RuntimeError"
    )
    assert diagnostic["prompt_context"]["item_count"] == 0
