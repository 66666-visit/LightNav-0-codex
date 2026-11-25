"""Policy integration: L1 Original + Geometry-only L3 + original on-demand gate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.habitat.policy import TrajVocabVLNCEPolicy
from lightnav.inference.policies import NavigationPolicy
from lightnav.memory.l3_instance_grounder import (
    InstanceGroundingResult,
    ProposalScore,
)
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

    def generate_from_frames(
        self, video, instruction: str, *, predict_horizon: int = 1,
        frame_ids=None, task_type: str = "vlnce_traj",
    ) -> tuple[str, float]:
        from lightnav.inference.samples import build_vln_traj_sample

        return self.generate(build_vln_traj_sample(
            video, instruction, list(frame_ids) if frame_ids else None,
            self.bundle,
        ))

    def generate(self, sample: dict) -> tuple[str, float]:
        self.samples.append(sample)
        prompt = sample["conversations"][0]["value"]
        trace = sample.get("_rgm_trace", {})
        memory_text = trace.get("memory_text", "")
        visible = bool(memory_text and memory_text in prompt)
        self.last_input_diagnostic = {
            "final_token_ids": [ord(char) for char in prompt],
            "memory_text_present_in_final_tokens": visible,
            "model_visible_evidence_ids": [
                evidence_id for record in trace.get("accepted_records", [])
                for evidence_id in record["evidence_ids"]
            ] if visible else [],
        }
        self.last_prompt_chars = len(prompt)
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
    mode: str = "geometry_on_demand", *, instruction: str = "find the sofa",
    l1_label: str = "chair", seg_label: str = "sofa",
    prompt_mode: str = "l1_original_only",
    rgb_sidecar: bool = False, instance_grounder=None,
    memory_enabled: bool = True,
):
    detections = {
        step: SceneExtraction(
            objects=[l1_label], detections=[DetectedObject(
                l1_label, 0.95, (0.0, 0.0, 8.0, 8.0),
            )], confidence=0.95, source="fake_l1",
        )
        for step in (0, 1, 2)
    }
    manager = None
    if memory_enabled:
        manager = MemoryManager(
            extractor=MockSceneExtractor(detections), strategy="hierarchical",
            local_graph_config=LocalGraphConfig(
                enabled=True, reactivation_steps=1,
            ),
            spatial_l3_config=SpatialL3Config(
                mode=mode, seg_model_path="/fake/yolo11n-seg.pt",
                seg_every_n_steps=2, min_injection_confidence=0.5,
                rgb_sidecar_enabled=rgb_sidecar,
                instance_grounder_enabled=instance_grounder is not None,
                grounder_checkpoint="/fake/official_nr3d_student.pt",
                grounder_device="cpu",
                grounder_selection_mode=(
                    "eligible" if instance_grounder is not None else "argmax"
                ),
                grounder_landmark_queries=instance_grounder is not None,
                grounder_geometry_fallback=instance_grounder is not None,
            ),
            spatial_extractor=_SegExtractor(seg_label, every_n_steps=2),
            clip_encoder=None,
            prompt_mode=prompt_mode,
            instance_grounder=instance_grounder,
        )
    centroids = np.zeros((3, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    engine = _Engine()
    policy = TrajVocabVLNCEPolicy(
        engine, centroids, dt=1.0, lin_vel_range=(0.0, 0.3),
        ang_vel_range=(-30.0, 30.0), num_history_frames=2,
        memory_enabled=memory_enabled,
        memory_manager=manager if memory_enabled else None,
        memory_prompt_mode=prompt_mode if memory_enabled else "legacy",
    )
    policy.reset(
        {"instruction": {"text": instruction}},
        memory_episode_id="episode", memory_scene_id="scene",
    )
    return policy, manager, engine


def _act(policy, step: int):
    obs, info = _obs(step)
    return policy.act(obs, info)


def _prompt(engine: _Engine) -> str:
    return engine.samples[-1]["conversations"][0]["value"]


def test_geometry_arm_uses_no_clip_encoder_and_no_controller() -> None:
    policy, manager, _ = _policy()
    assert manager.clip_encoder is None
    assert manager.l3_query_controller is None
    assert manager.geometry_on_demand_controller is not None
    assert policy.adaptive_l3_runner is None
    assert manager.spatial_l3_config.clip_model_path == ""


def test_l1_satisfied_geometry_never_queries_and_prompt_is_identical() -> None:
    baseline, _, baseline_engine = _policy(
        "off", instruction="find the chair", l1_label="chair",
    )
    _act(baseline, 0)
    _act(baseline, 1)

    policy, manager, engine = _policy(
        instruction="find the chair", l1_label="chair",
    )
    _act(policy, 0)
    _act(policy, 1)

    assert _prompt(engine) == _prompt(baseline_engine)
    assert manager.l3_query_stats["queries_attempted"] == 0
    assert policy.last_memory_diagnostic["prompt_context"]["spatial_l3"][
        "l3_query"
    ]["skipped_reason"] == "l1_satisfied"


def test_l1_empty_geometry_injects_existing_spatial_history_not_clip() -> None:
    policy, manager, engine = _policy(
        instruction="find the sofa", l1_label="chair", seg_label="sofa",
    )
    baseline, _, baseline_engine = _policy(
        "off", instruction="find the sofa", l1_label="chair",
    )
    _act(baseline, 0)
    _act(baseline, 1)
    assert len(engine.samples) == 0
    _act(policy, 0)
    assert len(engine.samples) == 1  # only the frozen navigation inference
    assert manager.l3_query_stats["queries_attempted"] == 0
    step0 = policy.last_memory_diagnostic["prompt_context"]["spatial_l3"]["l3_query"]
    assert step0["skipped_reason"] == "no_historical_geometry_object"
    assert step0["query_executed"] is False
    _act(policy, 1)

    prompt = _prompt(engine)
    diagnostic = policy.last_memory_diagnostic["prompt_context"]
    l3 = diagnostic["spatial_l3"]["l3_query"]
    assert len(engine.samples) == 2  # still one action inference per step
    assert "Spatial history: sofa" in prompt
    assert "L3 CLIP" not in prompt
    assert "cos " not in prompt
    assert diagnostic["source"] == "l3_geometry_on_demand"
    assert diagnostic["spatial_l3"]["context_source"] == "l3_geometry_on_demand"
    assert l3["retrieval"] == "geometry_spatial_graph_adapter"
    assert l3["query_executed"] is True
    assert l3["candidate_count"] == 1
    assert l3["injected_count"] == 1
    assert l3["object_ids"]
    assert l3["evidence_ids"]
    assert manager.l3_query_stats_summary()["queries_with_match"] == 1
    assert manager.l3_query_stats_summary()["queries_injected"] == 1

    base_prompt = _prompt(baseline_engine)
    assert prompt.startswith(base_prompt.split("</navigation_task>.", 1)[0])
    start = prompt.index("[Navigation Memory]")
    end = prompt.index("\nYou may be", start) + 1
    assert prompt[:start] + prompt[end:] == base_prompt


def test_no_geometry_candidate_keeps_the_l1_prompt_byte_identical() -> None:
    baseline, _, baseline_engine = _policy(
        "off", instruction="find the sofa", l1_label="chair",
    )
    _act(baseline, 0)
    _act(baseline, 1)

    policy, manager, engine = _policy(
        instruction="find the sofa", l1_label="chair", seg_label="lamp",
    )
    _act(policy, 0)
    _act(policy, 1)

    assert _prompt(engine) == _prompt(baseline_engine)
    summary = manager.l3_query_stats_summary()
    assert summary["queries_attempted"] == 1
    assert summary["queries_without_match"] == 1
    assert summary["queries_injected"] == 0
    l3 = policy.last_memory_diagnostic["prompt_context"]["spatial_l3"]["l3_query"]
    assert l3["skipped_reason"] == "no_geometry_candidate"
    assert l3["candidate_count"] == 0
    assert l3["injected_count"] == 0


def test_geometry_episode_reset_clears_gate_and_stats() -> None:
    policy, manager, engine = _policy()
    _act(policy, 0)
    _act(policy, 1)
    assert manager.l3_query_stats["queries_attempted"] == 1
    assert manager.geometry_on_demand_controller.last_diagnostic is not None

    policy.reset(
        {"instruction": {"text": "find the sofa"}},
        memory_episode_id="episode-2", memory_scene_id="scene",
    )
    assert manager.geometry_on_demand_controller.last_diagnostic is None
    assert manager.l3_query_stats["queries_attempted"] == 0
    assert manager.spatial_map.objects == {}

    _act(policy, 0)
    assert manager.l3_query_stats["queries_attempted"] == 0
    _act(policy, 1)
    assert manager.l3_query_stats["queries_attempted"] == 1
    assert "Spatial history: sofa" in _prompt(engine)


def test_geometry_mode_works_on_the_legacy_l1_path_too() -> None:
    policy, manager, engine = _policy(prompt_mode="legacy")
    _act(policy, 0)
    _act(policy, 1)
    assert policy.adaptive_l3_runner is None
    assert "Spatial history: sofa" in _prompt(engine)
    assert policy.last_memory_diagnostic["spatial_l3"]["context_source"] == (
        "l3_geometry_on_demand"
    )
    assert manager.l3_query_stats["queries_attempted"] == 1


def test_geometry_path_never_touches_clip_query_or_encoder(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("geometry_on_demand must not use CLIP")

    monkeypatch.setattr("lightnav.memory.manager.L3ClipQuery.query", fail)
    monkeypatch.setattr("lightnav.memory.manager.L3ClipQuery.__init__", fail)
    monkeypatch.setattr("lightnav.memory.clip_features.LocalClipEncoder", fail)
    policy, manager, engine = _policy()
    _act(policy, 0)
    _act(policy, 1)
    assert "Spatial history: sofa" in _prompt(engine)
    assert manager.clip_encoder is None


class _FakeGrounder:
    def __init__(self) -> None:
        self.calls = 0

    def select(self, *, instruction, object_map, sidecar, step, episode_id):
        self.calls += 1
        proposals, skipped = sidecar.proposals(object_map.objects)
        scores = tuple(
            ProposalScore(
                object_id=proposal.object_id,
                class_name=proposal.class_name,
                point_count=proposal.point_count,
                center_vil=(0.0, 0.0, 0.0),
                size_vil=(1.0, 1.0, 1.0),
                logit=1.0 - index,
                probability=1.0 if index == 0 else 0.0,
            )
            for index, proposal in enumerate(proposals)
        )
        selected = proposals[-1].object_id if proposals else None
        return InstanceGroundingResult(
            selected_object_id=selected,
            selected_index=len(proposals) - 1 if proposals else None,
            proposals=scores,
            top1_top2_gap=None,
            latency_ms=1.0,
            checkpoint="fake",
            checkpoint_sha256="fake",
            skipped_objects=tuple(skipped),
            refusal_reason=None if selected else "no_rgb_proposals",
        )


def test_grounded_geometry_injects_one_fact_and_keeps_l1_prefix() -> None:
    baseline, _, baseline_engine = _policy(
        "off", instruction="find the sofa", l1_label="chair",
    )
    _act(baseline, 0)
    _act(baseline, 1)

    grounder = _FakeGrounder()
    policy, manager, engine = _policy(
        instruction="find the sofa", l1_label="chair", seg_label="sofa",
        rgb_sidecar=True, instance_grounder=grounder,
    )
    _act(policy, 0)
    _act(policy, 1)

    assert grounder.calls == 1
    assert manager.rgb_sidecar is not None
    assert manager.rgb_sidecar.summary()["records"] >= 1
    prompt = _prompt(engine)
    assert prompt.count("Spatial history:") == 1
    diagnostic = policy.last_memory_diagnostic["prompt_context"]
    l3 = diagnostic["spatial_l3"]["l3_query"]
    assert diagnostic["source"] == "l3_geometry_on_demand"
    assert l3["injected_count"] == 1
    assert l3["instance_grounding"]["selected_object_id"] == l3["object_ids"][0]
    assert len(l3["instance_grounding"]["proposals"]) == 1
    # The new joint Grounding/Geometry diagnostics are present and consistent.
    assert l3["grounding_query"] == "sofa"
    assert l3["grounding_query_source"] == "instruction_landmark"
    assert l3["instruction_landmarks"]["landmarks"][0]["target_object"] == "sofa"
    assert l3["grounder_proposal_count"] == 1
    assert l3["grounder_scored_count"] == 1
    assert l3["formatable_proposal_count"] == 1
    assert l3["selected_valid_proposals"] == l3["object_ids"]
    assert l3["geometry_fallback_reason"] is None
    assert l3["actual_injection"] is True
    assert l3["actual_injected_evidence_ids"] == l3["evidence_ids"]
    assert l3["prompt_hash_before"] != l3["prompt_hash_after"]
    assert diagnostic["spatial_l3"]["actual_injection"] is True
    # The L1 prefix outside the memory block is byte-identical to baseline.
    base_prompt = _prompt(baseline_engine)
    assert prompt.startswith(base_prompt.split("</navigation_task>.", 1)[0])
    start = prompt.index("[Navigation Memory]")
    end = prompt.index("\nYou may be", start) + 1
    assert prompt[:start] + prompt[end:] == base_prompt


# -- Prompt invariance ---------------------------------------------------------

_FROZEN_INSTRUCTION = (
    "Go forward toward the windows. Go toward the the far couch, "
    "Stop next to the couch, in front of the windows."
)


def test_lightnav_original_prompt_hash_is_frozen_before_the_change() -> None:
    """The checkpoint prompt is a frozen contract; this hash predates this work."""
    from hashlib import sha256

    from lightnav.inference.samples import build_vln_traj_sample

    bundle = SimpleNamespace(
        pool_enable=True, pool_spatial=2, pool_mode="avg", video_fps=4,
        num_history_frames=4, predict_horizon=1, slowfast_tiers=None,
    )
    sample = build_vln_traj_sample(
        torch.zeros(5, 3, 4, 4), _FROZEN_INSTRUCTION,
        [0, 1, 2, 3, 4], bundle,
    )
    prompt = sample["conversations"][0]["value"]
    assert sha256(prompt.encode("utf-8")).hexdigest() == (
        "ec014d7d9799597312e036345b19561e297f8c2d44d337e237597478e14dacfb"
    )
    assert sample["conversations"][1] == {"from": "gpt", "value": "placeholder"}
    assert len(sample["video_segments"]) == 2
    assert prompt.count("<video>") == 2


def test_memory_disabled_policy_prompt_is_the_unmodified_lightnav_template() -> None:
    from hashlib import sha256

    from lightnav.prompts import VLN_TRAJ_PROMPT_TEMPLATE

    policy, manager, engine = _policy(
        "off", instruction=_FROZEN_INSTRUCTION, memory_enabled=False,
    )
    assert manager is None
    _act(policy, 0)
    _act(policy, 1)
    prompt = _prompt(engine)
    assert prompt == VLN_TRAJ_PROMPT_TEMPLATE.format(task=_FROZEN_INSTRUCTION)
    assert prompt == engine.samples[0]["conversations"][0]["value"]
    assert "[Navigation Memory]" not in prompt
    assert sha256(prompt.encode("utf-8")).hexdigest() == (
        "d4538cd84b1bdcaf5fca1019c4035ceca88cd2a31ac49c6d4469c80391681b90"
    )


def test_instance_grounder_with_no_candidate_changes_no_prompt_byte() -> None:
    from hashlib import sha256

    baseline, _, baseline_engine = _policy(
        "off", instruction="find the sofa", l1_label="chair",
    )
    _act(baseline, 0)
    _act(baseline, 1)
    baseline_prompt = _prompt(baseline_engine)

    policy, manager, engine = _policy(
        "geometry_on_demand", instruction="find the sofa", l1_label="chair",
        seg_label="lamp", rgb_sidecar=True, instance_grounder=_FakeGrounder(),
    )
    _act(policy, 0)
    _act(policy, 1)
    prompt = _prompt(engine)
    diagnostic = policy.last_memory_diagnostic["prompt_context"]
    l3 = diagnostic["spatial_l3"]["l3_query"]

    assert prompt == baseline_prompt
    assert "[Navigation Memory]" not in prompt
    assert l3["skipped_reason"] == "no_geometry_candidate"
    assert l3["actual_injection"] is False
    assert l3["actual_injected_evidence_ids"] == []
    assert l3["prompt_hash_before"] == l3["prompt_hash_after"]
    assert l3["prompt_hash_before"] == sha256(prompt.encode("utf-8")).hexdigest()
    assert diagnostic["spatial_l3"]["actual_injection"] is False
    assert manager.l3_query_stats_summary()["queries_injected"] == 0
