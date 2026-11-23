"""Policy integration: L1 Original with L3 CLIP evidence only after L1 is empty."""

from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.habitat.policy import TrajVocabVLNCEPolicy
from lightnav.inference.policies import NavigationPolicy
from lightnav.memory.l3_clip_query import L3QueryMatch
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.schemas import DetectedObject, SceneExtraction
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.yolo_seg_extractor import SegmentationInstance


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class _Processor:
    def __init__(self, tokenizer, context_limit: int = 4096) -> None:
        self.tokenizer = tokenizer
        self.model_max_length = context_limit

    def process_sample(self, sample, **kwargs):
        return {
            "input_ids": torch.zeros(1000, dtype=torch.long),
            "labels": None,
        }


class _Engine:
    def __init__(self, *, token_aware: bool = False, context_limit: int = 4096) -> None:
        self.bundle = SimpleNamespace(
            video_size=(4, 4), video_fps=4, num_history_frames=2,
            pool_enable=False, pool_spatial=1, pool_mode="avg",
            slowfast_tiers=None, predict_horizon=1, action_method="flat",
            max_seq_len=context_limit if token_aware else None,
        )
        if token_aware:
            tokenizer = _Tokenizer()
            self.bundle.tokenizer = tokenizer
            self.bundle.data_processor = _Processor(tokenizer, context_limit)
            self.max_new_tokens = 10
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
        visible = bool(memory_text and memory_text in prompt)
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


class _SegExtractor:
    def __init__(self, label: str = "sofa", *, every_n_steps: int = 1) -> None:
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

    def __init__(self, *, image_feature=(1.0, 0.0), text_feature=(1.0, 0.0)) -> None:
        self.image_feature = np.asarray(image_feature, dtype=np.float32)
        self.text_feature = np.asarray(text_feature, dtype=np.float32)
        self.text_calls = 0
        self.image_calls = 0

    def encode_image(self, rgb, boxes):
        self.image_calls += 1
        return np.tile(self.image_feature, (len(boxes), 1))

    def encode_text(self, texts):
        self.text_calls += 1
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
    mode: str = "on_demand", *, instruction: str = "find the sofa",
    l1_label: str = "chair", seg_label: str = "sofa",
    text_feature=(1.0, 0.0), every_n_steps: int = 1,
    clip_top_k: int = 3, prompt_mode: str = "legacy", token_aware: bool = False,
    context_limit: int = 4096,
):
    detections = {
        step: SceneExtraction(
            objects=[l1_label], detections=[DetectedObject(
                l1_label, 0.95, (0.0, 0.0, 8.0, 8.0),
            )], confidence=0.95, source="fake_l1",
        ) for step in (0, 1, 2)
    }
    spatial = SpatialL3Config(
        mode=mode, seg_model_path="/fake/yolo11n-seg.pt",
        seg_every_n_steps=every_n_steps,
        clip_model_path="/fake/open_clip.pt" if mode == "on_demand" else "",
        clip_top_k=clip_top_k,
    )
    encoder = _ClipEncoder(text_feature=text_feature)
    manager = MemoryManager(
        extractor=MockSceneExtractor(detections), strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=spatial,
        spatial_extractor=(
            _SegExtractor(seg_label, every_n_steps=every_n_steps)
            if mode != "off" else None
        ),
        clip_encoder=encoder if mode == "on_demand" else None,
        prompt_mode=prompt_mode,
    )
    centroids = np.zeros((3, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    engine = _Engine(token_aware=token_aware, context_limit=context_limit)
    policy = TrajVocabVLNCEPolicy(
        engine, centroids, dt=1.0, lin_vel_range=(0.0, 0.3),
        ang_vel_range=(-30.0, 30.0), num_history_frames=2,
        memory_enabled=True, memory_manager=manager,
    )
    policy.reset(
        {"instruction": {"text": instruction}},
        memory_episode_id="episode", memory_scene_id="scene",
    )
    return policy, manager, engine, encoder


def _act(policy, step: int):
    obs, info = _obs(step)
    return policy.act(obs, info)


def _prompt(engine: _Engine) -> str:
    return engine.samples[-1]["conversations"][0]["value"]


def test_l1_satisfied_keeps_the_exact_original_prompt_and_skips_clip_text() -> None:
    baseline, _, baseline_engine, _ = _policy(
        "off", instruction="find the chair", l1_label="chair",
    )
    _act(baseline, 0)
    baseline_prompt = _prompt(baseline_engine)

    policy, manager, engine, encoder = _policy(
        "on_demand", instruction="find the chair", l1_label="chair",
    )
    _act(policy, 0)
    assert _prompt(engine) == baseline_prompt
    assert encoder.text_calls == 0
    assert manager.last_local_graph_status == "built"
    assert policy.last_memory_diagnostic["spatial_l3"]["l1_status"] == "built"
    assert policy.last_memory_diagnostic["spatial_l3"]["context_source"] == "local_graph"
    assert "L3 CLIP cos" not in _prompt(engine)


def test_l1_empty_triggers_l3_and_injects_only_spatial_evidence() -> None:
    policy, manager, engine, encoder = _policy(
        "on_demand", instruction="find the sofa", l1_label="chair",
    )
    baseline, _, baseline_engine, _ = _policy(
        "off", instruction="find the sofa", l1_label="chair",
    )
    _act(baseline, 0)
    _act(policy, 0)

    prompt = _prompt(engine)
    diagnostic = policy.last_memory_diagnostic
    assert encoder.text_calls == 1
    assert manager.last_local_graph_status == "no_candidates"
    assert diagnostic["spatial_l3"]["l1_status"] == "no_candidates"
    assert diagnostic["spatial_l3"]["l3_query"]["trigger"] == "no_candidates"
    assert diagnostic["spatial_l3"]["l3_query"]["candidate_count"] == 1
    assert diagnostic["spatial_l3"]["context_source"] == "l3_on_demand"
    assert diagnostic["actual_injection"] is True
    assert "L3 CLIP cos" in prompt
    assert "sofa" in prompt
    assert "world (" in prompt
    assert "passage unknown" in prompt
    assert prompt.count("</navigation_task>.") == 1
    # The L3 block is appended; the original prompt suffix is untouched.
    assert prompt.startswith(
        baseline_engine.samples[0]["conversations"][0]["value"].split(
            "</navigation_task>.", 1,
        )[0],
    )
    assert prompt.endswith(
        baseline_engine.samples[0]["conversations"][0]["value"].split(
            "</navigation_task>.", 1,
        )[1].lstrip(" "),
    )
    assert diagnostic["memory_text"].count("L3 CLIP cos") == 1
    assert len(diagnostic["memory_text"]) <= manager.prompt_budget_chars


def test_l3_no_match_does_not_inject_and_matches_baseline_prompt() -> None:
    baseline, _, baseline_engine, _ = _policy(
        "off", instruction="find the sofa", l1_label="chair",
    )
    _act(baseline, 0)
    policy, _, engine, encoder = _policy(
        "on_demand", instruction="find the sofa", l1_label="chair",
        text_feature=(-1.0, 0.0),
    )
    _act(policy, 0)

    assert encoder.text_calls == 1
    assert _prompt(engine) == _prompt(baseline_engine)
    assert policy.last_memory_diagnostic["memory_text"] == ""
    assert policy.last_memory_diagnostic["actual_injection"] is False
    assert policy.last_memory_diagnostic["spatial_l3"]["l3_query"]["skipped_reason"] == (
        "no_similarity_above_floor"
    )
    assert "L3 CLIP cos" not in _prompt(engine)


def test_unchanged_l3_map_is_not_queried_on_every_step() -> None:
    policy, _, engine, encoder = _policy(
        "on_demand", instruction="find the sofa", l1_label="chair",
        every_n_steps=2,
    )
    _act(policy, 0)
    assert encoder.text_calls == 1
    _act(policy, 1)
    assert encoder.text_calls == 1
    assert policy.last_memory_diagnostic["spatial_l3"]["l3_query"]["skipped_reason"] == (
        "map_unchanged_since_last_query"
    )
    assert "L3 CLIP cos" not in policy.last_memory_diagnostic["memory_text"]


def test_local_query_interface_returns_existing_object_and_geometry() -> None:
    policy, manager, _, _ = _policy(
        "on_demand", instruction="find the sofa", l1_label="chair",
    )
    _act(policy, 0)
    _act(policy, 1)

    result = manager.query_l3_objects("sofa", 2)
    assert result.valid
    assert result.query == "sofa"
    assert len(result.matches) == 1
    match: L3QueryMatch = result.matches[0]
    assert match.object_id in manager.spatial_map.objects
    assert match.class_name == "sofa"
    assert match.similarity == pytest.approx(1.0)
    assert len(match.center_world) == 3
    assert match.bbox_3d[0] != match.bbox_3d[1]
    assert match.observation_count >= 1
    assert match.evidence_ids
    assert result.to_dict()["matches"][0]["object_id"] == match.object_id


def test_l1_original_only_with_on_demand_keeps_the_exact_l1_prompt() -> None:
    baseline, _, baseline_engine, _ = _policy(
        "off", instruction="find the chair", l1_label="chair",
        prompt_mode="l1_original_only",
    )
    _act(baseline, 0)
    baseline_prompt = _prompt(baseline_engine)

    policy, manager, engine, encoder = _policy(
        "on_demand", instruction="find the chair", l1_label="chair",
        prompt_mode="l1_original_only",
    )
    _act(policy, 0)

    assert _prompt(engine) == baseline_prompt
    assert encoder.text_calls == 0
    assert manager.plan.mode == "l1_original_only"
    assert manager.l3_query_stats["l1_built"] == 1
    assert manager.l3_query_stats["queries_attempted"] == 0
    spatial = policy.last_memory_diagnostic["prompt_context"]["spatial_l3"]
    assert spatial["context_source"] == "l1_original"
    assert spatial["l3_query"]["trigger"] == "built"
    assert spatial["l3_query"]["skipped_reason"] == "l1_satisfied"


def test_l1_original_only_falls_back_to_on_demand_when_l1_is_empty() -> None:
    policy, manager, engine, encoder = _policy(
        "on_demand", instruction="find the sofa", l1_label="chair",
        prompt_mode="l1_original_only",
    )
    baseline, _, baseline_engine, _ = _policy(
        "off", instruction="find the sofa", l1_label="chair",
        prompt_mode="l1_original_only",
    )
    _act(baseline, 0)
    _act(policy, 0)

    prompt = _prompt(engine)
    diagnostic = policy.last_memory_diagnostic
    spatial = diagnostic["prompt_context"]["spatial_l3"]
    assert encoder.text_calls == 1
    assert manager.plan.mode == "l1_original_only"
    assert manager.l3_query_stats["l1_no_candidates"] == 1
    assert manager.l3_query_stats["queries_attempted"] == 1
    assert manager.l3_query_stats["queries_with_match"] == 1
    assert manager.l3_query_stats_summary()["query_trigger_rate"] == pytest.approx(1.0)
    assert spatial["context_source"] == "l3_on_demand"
    assert spatial["l3_query"]["query_source"] == "instruction"
    assert spatial["l3_query"]["candidates"][0]["object_id"]
    assert spatial["l3_query"]["candidates"][0]["similarity"] == pytest.approx(1.0)
    assert spatial["l3_query"]["candidates"][0]["decision"] == "matched"
    assert "L3 CLIP cos" in prompt
    assert "passage unknown" in prompt
    assert diagnostic["actual_injection"] is True
    assert prompt.count("</navigation_task>.") == 1
    base_prompt = baseline_engine.samples[0]["conversations"][0]["value"]
    assert prompt.startswith(base_prompt.split("</navigation_task>.", 1)[0])
    assert prompt.endswith(
        base_prompt.split("</navigation_task>.", 1)[1].lstrip(" ")
    )
    assert len(diagnostic["memory_text"]) <= 320


def test_l1_original_only_on_demand_does_not_inject_without_valid_match() -> None:
    baseline, _, baseline_engine, _ = _policy(
        "off", instruction="find the sofa", l1_label="chair",
        prompt_mode="l1_original_only",
    )
    _act(baseline, 0)
    policy, manager, engine, encoder = _policy(
        "on_demand", instruction="find the sofa", l1_label="chair",
        text_feature=(-1.0, 0.0), prompt_mode="l1_original_only",
    )
    _act(policy, 0)

    assert encoder.text_calls == 1
    assert _prompt(engine) == _prompt(baseline_engine)
    assert policy.last_memory_diagnostic["memory_text"] == ""
    assert manager.l3_query_stats["queries_without_match"] == 1
    spatial = policy.last_memory_diagnostic["prompt_context"]["spatial_l3"]
    assert spatial["l3_query"]["skipped_reason"] == "no_similarity_above_floor"
    assert spatial["l3_query"]["candidates"][0]["decision"] == (
        "below_positive_cosine"
    )
    assert spatial["l3_query_stats"]["queries_with_match"] == 0


def test_l1_original_only_on_demand_uses_the_token_aware_budget() -> None:
    policy, manager, engine, encoder = _policy(
        "on_demand", instruction="find the sofa", l1_label="chair",
        prompt_mode="l1_original_only", token_aware=True,
    )
    _act(policy, 0)
    prompt = _prompt(engine)
    context = policy.last_memory_diagnostic["prompt_context"]
    assert encoder.text_calls == 1
    assert context["spatial_l3"]["context_source"] == "l3_on_demand"
    assert context["budget"]["budget_source"] == "token_aware_context"
    assert context["budget"]["available_memory_tokens"] > 0
    assert context["budget"]["memory_block_tokens"] > 0
    assert "L3 CLIP cos" in prompt
    assert manager.l3_query_stats_summary()["queries_with_match"] == 1


def test_real_r2r_instruction_is_passed_as_the_default_clip_query() -> None:
    instruction = (
        "Walk past the chair and turn left at the wooden table, "
        "then stop near the sofa."
    )
    policy, manager, engine, encoder = _policy(
        "on_demand", instruction=instruction, l1_label="lamp",
        seg_label="sofa", prompt_mode="l1_original_only",
    )
    _act(policy, 0)
    spatial = policy.last_memory_diagnostic["prompt_context"]["spatial_l3"]
    assert spatial["l3_query"]["query"] == instruction
    assert spatial["l3_query"]["query_source"] == "instruction"
    assert spatial["l3_query"]["query_executed"] is True
    assert "L3 CLIP cos" in _prompt(engine)
    assert manager.l3_query_stats_summary()["l1_empty_steps"] == 1


def test_token_budget_can_exclude_the_l3_block_without_injecting_it() -> None:
    policy, manager, engine, encoder = _policy(
        "on_demand", instruction="find the sofa", l1_label="chair",
        prompt_mode="l1_original_only", token_aware=True, context_limit=1100,
    )
    _act(policy, 0)
    diagnostic = policy.last_memory_diagnostic
    context = diagnostic["prompt_context"]
    assert encoder.text_calls == 1
    assert context["spatial_l3"]["l3_query"]["candidate_count"] == 1
    assert context["spatial_l3"]["l3_query"]["skipped_reason"] == (
        "prompt_budget_excluded"
    )
    assert context["l3_query_stats"]["prompt_budget_excluded"] == 1
    assert diagnostic["memory_text"] == ""
    assert "L3 CLIP cos" not in _prompt(engine)
