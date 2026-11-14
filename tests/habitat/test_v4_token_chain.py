"""CPU-only V4 checks from RGB history to actual fake-vLLM prompt IDs."""

from __future__ import annotations

import sys
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lightnav.habitat.policy as policy_module
from lightnav.data_processor import Qwen3VLDataProcessor
from lightnav.habitat.policy import TrajVocabVLNCEPolicy
from lightnav.inference.engine import VLNInferenceEngine
from lightnav.inference.policies import NavigationPolicy
from lightnav.inference.token_diagnostics import token_ids_sha256
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.schemas import SceneExtraction
from lightnav.memory.v4_config import RGMV4Config
from lightnav.memory.v4_navigation_state import COMPLETION_UNKNOWN
from lightnav.slowfast import validate_slowfast_tiers


class _Tokenizer:
    _special = {
        "<|im_start|>assistant\n": [6, 5, 7],
        "<|im_start|>user\n": [6, 8, 7],
        "<|im_end|>": [3],
    }

    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 1, "<|video_pad|>": 2, "<|im_end|>": 3}.get(token, 4)

    def encode(self, value, *, add_special_tokens=False):
        if value == "assistant":
            return [5]
        ids = []
        offset = 0
        while offset < len(value):
            special = next((s for s in self._special if value.startswith(s, offset)), None)
            if special is None:
                ids.append(ord(value[offset]) + 1000)
                offset += 1
            else:
                ids.extend(self._special[special])
                offset += len(special)
        return ids

    def decode(self, ids, *, skip_special_tokens=False):
        special = {3: "<|im_end|>", 5: "assistant", 6: "<|im_start|>", 7: "\n", 8: "user"}
        return "".join(
            chr(int(token_id) - 1000) if int(token_id) >= 1000
            else special.get(int(token_id), "") for token_id in ids
        )


class _QwenProcessor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.video_processor = SimpleNamespace(
            fps=4, temporal_patch_size=2, do_sample_frames=False,
            do_resize=False, do_rescale=False, do_normalize=False,
        )
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize=False):
        assert tokenize is False
        turns = []
        for message in messages:
            parts = ["V" * 7 if part["type"] == "video" else part["text"]
                     for part in message["content"]]
            turns.append(f"<|im_start|>{message['role']}\n" + "".join(parts) + "<|im_end|>")
        return "\n".join(turns)

    def __call__(self, *, text, videos, **kwargs):
        assert videos
        metadata = kwargs.get("video_metadata")
        self.calls.append({
            "text": text,
            "videos": [video.clone() for video in videos],
            "frame_indices": None if metadata is None else [
                list(item.frames_indices) for item in metadata
            ],
        })
        return {"input_ids": torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)}


class _VLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompts, params, *, use_tqdm=False):
        self.prompts.extend(prompts)
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=(), text="<traj_1>")])
                for _ in prompts]


def _position_ids(**kwargs):
    length = kwargs["input_ids"].shape[-1]
    return {"position_ids": torch.arange(length).repeat(3, 1).unsqueeze(1)}


def _engine(monkeypatch, *, max_seq_len=4096, slowfast_tiers=None):
    monkeypatch.setenv("VLN_EVAL_TRAJ_TOP1", "0")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    processor = _QwenProcessor()
    data_processor = Qwen3VLDataProcessor(
        processor=processor, model_max_length=max_seq_len,
        video_fps=4, video_size=(4, 4), position_id_func=_position_ids,
    )
    bundle = SimpleNamespace(
        video_size=(4, 4), video_fps=4, num_history_frames=2,
        pool_enable=False, pool_spatial=1, pool_mode="avg", pool_stage="pre_vit",
        slowfast_tiers=slowfast_tiers, predict_horizon=1, action_method="flat",
        data_processor=data_processor, processor=processor, tokenizer=processor.tokenizer,
        device=torch.device("cpu"),
    )
    vllm = _VLLM()
    engine = VLNInferenceEngine(
        bundle, backend="vllm_local", vllm_engine=vllm, enable_vit_cache=False,
    )
    return engine, processor, vllm


def _policy(engine, *, memory_enabled, detections=None, max_records=3):
    centroids = np.zeros((2, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    manager = None
    if memory_enabled:
        manager = MemoryManager(
            extractor=MockSceneExtractor(detections or {}), strategy="hierarchical",
            v4_config=RGMV4Config(enabled=True, prompt_max_records=max_records),
        )
    policy = TrajVocabVLNCEPolicy(
        engine, centroids, dt=1.0,
        lin_vel_range=(0.0, 0.3), ang_vel_range=(-30.0, 30.0),
        num_history_frames=2, memory_enabled=memory_enabled, memory_manager=manager,
    )
    return policy, manager


def _reset(policy, instruction="Find the archway", episode="episode_v4"):
    policy.reset(
        {"instruction": {"text": instruction}},
        memory_episode_id=episode, memory_scene_id="scene_a",
    )


def _run_steps(policy, count=3, *, info=None):
    for step in range(count):
        frame = np.full((8, 8, 3), step, dtype=np.uint8)
        policy.act({"rgb": frame}, info)


def _ids(vllm):
    return vllm.prompts[-1]["prompt_token_ids"]


@pytest.fixture(autouse=True)
def _distinct_model_frames(monkeypatch):
    monkeypatch.setattr(
        NavigationPolicy, "_convert_frame",
        lambda self, frame: torch.full((3, 4, 4), float(frame[0, 0, 0])),
    )


def test_none_uses_original_sample_and_exact_baseline_tokens(monkeypatch):
    baseline_engine, baseline_processor, baseline_vllm = _engine(monkeypatch)
    baseline, _ = _policy(baseline_engine, memory_enabled=False)
    _reset(baseline)
    _run_steps(baseline, 1)

    engine, processor, vllm = _engine(monkeypatch)
    policy, manager = _policy(engine, memory_enabled=True)
    built_samples = []
    generated_samples = []
    build_original = policy_module.build_vln_traj_sample
    generate_original = engine.generate

    def spy_build(*args, **kwargs):
        sample = build_original(*args, **kwargs)
        built_samples.append(sample)
        return sample

    def spy_generate(sample, *args, **kwargs):
        generated_samples.append(sample)
        return generate_original(sample, *args, **kwargs)

    monkeypatch.setattr(policy_module, "build_vln_traj_sample", spy_build)
    monkeypatch.setattr(engine, "generate", spy_generate)
    _reset(policy)
    _run_steps(policy, 1, info={"success": True, "goal_distance": 0.0, "spl": 1.0})

    assert generated_samples[0] is built_samples[0]
    assert "_rgm_trace" not in generated_samples[0]
    assert _ids(vllm) == _ids(baseline_vllm)
    assert token_ids_sha256(_ids(vllm)) == token_ids_sha256(_ids(baseline_vllm))
    assert processor.calls[-1]["text"] == baseline_processor.calls[-1]["text"]
    assert torch.equal(processor.calls[-1]["videos"][0],
                       baseline_processor.calls[-1]["videos"][0])
    assert manager.last_v4_diagnostic["memory_use_mode"] == "NONE"
    assert policy.last_memory_diagnostic["v4_diagnostic"]["actual_injection"] is False
    assert policy.last_memory_diagnostic["v4_diagnostic"]["injected_memory_tokens"] == 0
    assert manager.v4_state_updater.state.progress_status == COMPLETION_UNKNOWN
    assert manager.write_gate is None
    assert manager.retrieval_gate.decisions == []
    assert manager.prompt_gate.last_selection.accepted_records == ()
    baseline.close_episode()
    policy.close_episode()


def test_old_frame_injects_history_without_changing_rgb_or_answer_prefix(monkeypatch):
    baseline_engine, baseline_processor, baseline_vllm = _engine(monkeypatch)
    baseline, _ = _policy(baseline_engine, memory_enabled=False)
    _reset(baseline)
    _run_steps(baseline)

    engine, processor, vllm = _engine(monkeypatch)
    policy, manager = _policy(
        engine, memory_enabled=True,
        detections={0: SceneExtraction(objects=["archway"], confidence=0.88)},
    )
    built_samples = []
    generated_samples = []
    build_original = policy_module.build_vln_traj_sample
    generate_original = engine.generate

    def spy_build(*args, **kwargs):
        sample = build_original(*args, **kwargs)
        built_samples.append(sample)
        return sample

    def spy_generate(sample, *args, **kwargs):
        generated_samples.append(sample)
        return generate_original(sample, *args, **kwargs)

    monkeypatch.setattr(policy_module, "build_vln_traj_sample", spy_build)
    monkeypatch.setattr(engine, "generate", spy_generate)
    _reset(policy)
    _run_steps(policy)

    record = next(item for item in manager.v3_store.evidence_for_scene("scene_a")
                  if item.semantic_label == "archway")
    decoded = processor.tokenizer.decode(_ids(vllm))
    diagnostic = policy.last_memory_diagnostic["v4_diagnostic"]
    input_diagnostic = engine.last_input_diagnostic
    assert manager.last_v4_diagnostic["visible_frame_ids"] == [1, 2]
    assert manager.v4_frame_ids_by_step[("scene_a", "episode_v4", 0, 0)] == 0
    assert diagnostic["memory_use_mode"] == "HISTORY_REFERENCE"
    assert diagnostic["actual_injection"] is True
    assert record.key in diagnostic["injected_evidence_ids"]
    assert "[Navigation Memory]" in decoded
    assert f"Ref={sha256(record.key.encode()).hexdigest()[:12]}" in decoded
    assert record.key not in decoded
    assert record.key in input_diagnostic["model_visible_evidence_ids"]
    assert 'Episode="episode_v4"' in decoded and "Step=0" in decoded
    assert "<navigation_task>Find the archway</navigation_task>" in decoded
    assert decoded.endswith("<|im_start|>assistant\n")
    assert processor.tokenizer.decode(_ids(baseline_vllm)).endswith(
        "<|im_start|>assistant\n"
    )
    assert _ids(vllm) != _ids(baseline_vllm)
    assert input_diagnostic["memory_text_present_in_final_tokens"] is True
    assert input_diagnostic["memory_text_present_in_vllm_prompt"] is True
    assert input_diagnostic["vllm_prompt_matches_final_tokens"] is True
    assert input_diagnostic["final_token_ids_sha256"] == token_ids_sha256(_ids(vllm))
    assert input_diagnostic["vllm_prompt_token_ids_sha256"] == token_ids_sha256(_ids(vllm))
    assert input_diagnostic["baseline_final_token_ids_sha256"] == token_ids_sha256(
        _ids(baseline_vllm)
    )
    assert input_diagnostic["original_prompt_sha256"] == diagnostic["prompt_hash_before"]
    assert input_diagnostic["injected_prompt_sha256"] == diagnostic["prompt_hash_after"]
    assert diagnostic["prompt_hash_before"] != diagnostic["prompt_hash_after"]
    assert input_diagnostic["injected_memory_tokens"] == (
        len(_ids(vllm)) - len(_ids(baseline_vllm))
    )
    assert input_diagnostic["injected_memory_tokens"] > 0
    assert input_diagnostic["injected_memory_chars"] == diagnostic["injected_memory_chars"]
    assert any(source["evidence_id"] == record.key
               and source["episode_id"] == "episode_v4" and source["step"] == 0
               for source in input_diagnostic["evidence_provenance"])
    assert generated_samples[-1]["video_segments"] is built_samples[-1]["video_segments"]
    assert generated_samples[-1]["video_segments"][0]["video"] is (
        built_samples[-1]["video_segments"][0]["video"]
    )
    assert generated_samples[-1]["video_segments"][0]["frame_indices"] == [1, 2]
    assert baseline_processor.calls[-1]["videos"][0].shape[0] == 2
    assert torch.equal(processor.calls[-1]["videos"][0],
                       baseline_processor.calls[-1]["videos"][0])
    assert torch.equal(processor.calls[-1]["videos"][0][:, 0, 0, 0],
                       torch.tensor([1.0, 2.0]))
    baseline.close_episode()
    policy.close_episode()


def test_slowfast_anchor_keeps_old_rgb_and_suppresses_duplicate_text(monkeypatch):
    tiers = validate_slowfast_tiers([
        {"name": "current", "age_lo": 0, "age_hi": 1, "mode": "dense", "pool_spatial": 1},
        {"name": "anchor", "age_lo": 0, "age_hi": 0, "mode": "anchor",
         "num_frames": 1, "pool_spatial": 1},
    ])
    engine, processor, vllm = _engine(monkeypatch, slowfast_tiers=tiers)
    policy, manager = _policy(
        engine, memory_enabled=True,
        detections={0: SceneExtraction(objects=["archway"], confidence=0.88)},
    )
    _reset(policy)
    _run_steps(policy, 4)
    assert 0 in manager.last_v4_diagnostic["visible_frame_ids"]
    assert manager.last_v4_diagnostic["memory_use_mode"] == "NONE"
    assert policy.last_memory_diagnostic["v4_diagnostic"]["actual_injection"] is False
    assert "[Navigation Memory]" not in processor.tokenizer.decode(_ids(vllm))
    assert processor.calls[-1]["frame_indices"][0] == [0, 0]
    policy.close_episode()


def test_new_evidence_does_not_reinject_an_older_confirmed_fact(monkeypatch):
    engine, processor, vllm = _engine(monkeypatch)
    policy, manager = _policy(
        engine, memory_enabled=True,
        detections={
            0: SceneExtraction(objects=["archway"], confidence=0.88),
            1: SceneExtraction(objects=["archway"], confidence=0.88),
        },
    )
    _reset(policy)
    _run_steps(policy, 5)
    records = sorted(
        (item for item in manager.v3_store.evidence_for_scene("scene_a")
         if item.semantic_label == "archway"),
        key=lambda item: item.step,
    )
    assert len(records) == 2
    for record in records:
        marker = f"Ref={sha256(record.key.encode()).hexdigest()[:12]}"
        assert sum(marker in processor.tokenizer.decode(
            prompt["prompt_token_ids"]
        ) for prompt in vllm.prompts) == 1
    assert policy.last_memory_diagnostic["v4_diagnostic"]["actual_injection"] is False
    policy.close_episode()


def test_processor_overflow_retries_whole_items_and_keeps_baseline(monkeypatch):
    baseline_engine, _, baseline_vllm = _engine(monkeypatch)
    baseline, _ = _policy(baseline_engine, memory_enabled=False)
    _reset(baseline)
    _run_steps(baseline)
    baseline_ids = list(_ids(baseline_vllm))

    # The Processor measures length before the training answer placeholder is
    # removed for generation; budget against that full length, not prompt IDs.
    baseline_full_len = baseline_engine.bundle.data_processor.last_process_diagnostic[
        "tokens_before_truncation"
    ]
    engine, processor, vllm = _engine(monkeypatch, max_seq_len=baseline_full_len + 8)
    policy, manager = _policy(
        engine, memory_enabled=True,
        detections={0: SceneExtraction(objects=["archway"], confidence=0.88)},
    )
    _reset(policy)
    _run_steps(policy)

    assert _ids(vllm) == baseline_ids
    assert processor.tokenizer.decode(_ids(vllm)).endswith("<|im_start|>assistant\n")
    assert manager.last_v4_diagnostic["memory_use_mode"] == "NONE"
    assert policy.last_memory_diagnostic["v4_diagnostic"]["actual_injection"] is False
    assert any("budget" in item["reason"] or "token" in item["reason"]
               for item in policy.last_memory_diagnostic["v4_diagnostic"]["dropped_candidates"])
    assert engine.bundle.data_processor.last_process_diagnostic["truncated"] is False
    baseline.close_episode()
    policy.close_episode()


def test_local_graph_card_reaches_vllm_with_original_video_and_action_decoder(monkeypatch):
    baseline_engine, baseline_processor, baseline_vllm = _engine(monkeypatch)
    baseline, _ = _policy(baseline_engine, memory_enabled=False)
    _reset(baseline)
    _run_steps(baseline, 1)

    engine, processor, vllm = _engine(monkeypatch)
    manager = MemoryManager(
        extractor=MockSceneExtractor({
            0: SceneExtraction(objects=["archway"], confidence=0.88),
        }),
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
    )
    centroids = np.zeros((2, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    policy = TrajVocabVLNCEPolicy(
        engine, centroids, dt=1.0,
        lin_vel_range=(0.0, 0.3), ang_vel_range=(-30.0, 30.0),
        num_history_frames=2, memory_enabled=True, memory_manager=manager,
    )
    _reset(policy)
    _run_steps(policy, 1)

    diagnostic = policy.last_memory_diagnostic
    model_input = engine.last_input_diagnostic
    assert diagnostic["memory_backend"] == "local_graph"
    assert diagnostic["actual_injection"] is True
    assert model_input["memory_text_present_in_final_tokens"] is True
    assert model_input["memory_text_present_in_vllm_prompt"] is True
    assert model_input["vllm_prompt_matches_final_tokens"] is True
    assert model_input["injected_memory_tokens"] > 0
    assert "Current RGB: instruction landmark archway" in processor.tokenizer.decode(_ids(vllm))
    assert "<navigation_task>Find the archway</navigation_task>" in policy.last_prompt_text
    assert torch.equal(processor.calls[-1]["videos"][0], baseline_processor.calls[-1]["videos"][0])
    assert policy.get_info()["raw_text"] == baseline.get_info()["raw_text"]
    assert _ids(vllm) != _ids(baseline_vllm)
    baseline.close_episode()
    policy.close_episode()


def test_local_graph_processor_overflow_falls_back_to_exact_baseline_tokens(monkeypatch):
    baseline_engine, _, baseline_vllm = _engine(monkeypatch)
    baseline, _ = _policy(baseline_engine, memory_enabled=False)
    _reset(baseline)
    _run_steps(baseline, 1)
    baseline_ids = list(_ids(baseline_vllm))
    baseline_full_len = baseline_engine.bundle.data_processor.last_process_diagnostic[
        "tokens_before_truncation"
    ]

    engine, _, vllm = _engine(monkeypatch, max_seq_len=baseline_full_len + 8)
    manager = MemoryManager(
        extractor=MockSceneExtractor({
            0: SceneExtraction(objects=["archway"], confidence=0.88),
        }),
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
    )
    centroids = np.zeros((2, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    policy = TrajVocabVLNCEPolicy(
        engine, centroids, dt=1.0,
        lin_vel_range=(0.0, 0.3), ang_vel_range=(-30.0, 30.0),
        num_history_frames=2, memory_enabled=True, memory_manager=manager,
    )
    _reset(policy)
    _run_steps(policy, 1)

    assert _ids(vllm) == baseline_ids
    assert policy.last_memory_diagnostic["processor_length_rejections"] == 1
    assert policy.last_memory_diagnostic["actual_injection"] is False
    assert policy.last_memory_diagnostic["memory_text"] == ""
    assert engine.bundle.data_processor.last_process_diagnostic["truncated"] is False
    baseline.close_episode()
    policy.close_episode()
