"""CPU-only RGM tests through the final vLLM prompt token IDs.

The Qwen processor and vLLM are small fakes, but the Policy, memory stores,
DataProcessor, and Engine entry points are real. A reversible tokenizer lets
the test inspect the exact IDs handed to the fake vLLM engine.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.data_processor import Qwen3VLDataProcessor
from lightnav.habitat.policy import TrajVocabVLNCEPolicy
from lightnav.inference.engine import VLNInferenceEngine
from lightnav.inference.policies import NavigationPolicy
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.schemas import SceneExtraction, VisualPlaceFeatures


class _CharTokenizer:
    """Encode every rendered character separately so token presence is unambiguous."""

    _special = {
        "<|im_start|>assistant\n": [6, 5, 7],
        "<|im_start|>user\n": [6, 8, 7],
        "<|im_end|>": [3],
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|image_pad|>": 1, "<|video_pad|>": 2, "<|im_end|>": 3}.get(token, 4)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if text == "assistant":
            return [5]
        ids = []
        offset = 0
        while offset < len(text):
            special = next((s for s in self._special if text.startswith(s, offset)), None)
            if special is not None:
                ids.extend(self._special[special])
                offset += len(special)
            else:
                ids.append(ord(text[offset]) + 1000)
                offset += 1
        return ids

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        special = {3: "<|im_end|>", 5: "assistant", 6: "<|im_start|>", 7: "\n", 8: "user"}
        return "".join(
            chr(int(token_id) - 1000) if int(token_id) >= 1000
            else special.get(int(token_id), "")
            for token_id in ids
        )


class _QwenProcessorSpy:
    def __init__(self, *, video_expansion: int = 7) -> None:
        self.tokenizer = _CharTokenizer()
        self.video_processor = SimpleNamespace(
            fps=4, temporal_patch_size=2, do_sample_frames=False,
            do_resize=False, do_rescale=False, do_normalize=False,
        )
        self.video_expansion = video_expansion
        self.last_rendered_text = ""
        self.last_tokenizer_text = ""
        self.last_video_shapes = []

    def apply_chat_template(self, messages, *, tokenize: bool = False) -> str:
        assert tokenize is False
        turns = []
        for message in messages:
            parts = []
            for part in message["content"]:
                if part["type"] == "video":
                    parts.append("V" * self.video_expansion)
                else:
                    parts.append(part["text"])
            turns.append(f"<|im_start|>{message['role']}\n" + "".join(parts) + "<|im_end|>")
        self.last_rendered_text = "\n".join(turns)
        return self.last_rendered_text

    def __call__(self, *, text, videos, **kwargs):
        assert videos
        self.last_tokenizer_text = text
        self.last_video_shapes = [tuple(video.shape) for video in videos]
        return {"input_ids": torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)}


class _VLLMSpy:
    def __init__(self) -> None:
        self.prompts = []

    def generate(self, prompts, params, *, use_tqdm=False):
        self.prompts.extend(prompts)
        return [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=(), text="<traj_1>")])
            for _ in prompts
        ]


def _position_ids(**kwargs):
    length = kwargs["input_ids"].shape[-1]
    return {"position_ids": torch.arange(length).repeat(3, 1).unsqueeze(1)}


def _make_engine(monkeypatch, *, max_seq_len=4096, video_expansion=7):
    monkeypatch.setenv("VLN_EVAL_TRAJ_TOP1", "0")
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
    )
    processor = _QwenProcessorSpy(video_expansion=video_expansion)
    data_processor = Qwen3VLDataProcessor(
        processor=processor, model_max_length=max_seq_len,
        video_fps=4, video_size=(4, 4), position_id_func=_position_ids,
    )
    bundle = SimpleNamespace(
        video_size=(4, 4), video_fps=4, num_history_frames=4,
        pool_enable=False, pool_spatial=1, pool_mode="avg", pool_stage="pre_vit",
        slowfast_tiers=None, predict_horizon=1, action_method="flat",
        data_processor=data_processor, processor=processor, tokenizer=processor.tokenizer,
        device=torch.device("cpu"),
    )
    vllm = _VLLMSpy()
    engine = VLNInferenceEngine(
        bundle, backend="vllm_local", vllm_engine=vllm, enable_vit_cache=False,
    )
    return engine, processor, vllm


def _make_policy(engine, *, config=None, detections=None, memory_enabled=True,
                 place_extractor=None):
    centroids = np.zeros((2, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    manager = None
    if memory_enabled:
        manager = MemoryManager(
            extractor=MockSceneExtractor(detections or {}), strategy="hierarchical",
            rgm_config=config or RGMConfig(enabled=True),
            place_enabled=place_extractor is not None, place_extractor=place_extractor,
        )
    policy = TrajVocabVLNCEPolicy(
        engine, centroids, dt=1.0,
        lin_vel_range=(0.0, 0.3), ang_vel_range=(-30.0, 30.0),
        num_history_frames=4, memory_enabled=memory_enabled, memory_manager=manager,
    )
    return policy, manager


@pytest.fixture(autouse=True)
def _tiny_frame(monkeypatch):
    monkeypatch.setattr(
        NavigationPolicy, "_convert_frame", lambda self, frame: torch.zeros(3, 4, 4),
    )


def _act(policy, *, instruction="Find TV", steps=1):
    policy.reset({"instruction": {"text": instruction}}, memory_episode_id="episode_probe")
    for _ in range(steps):
        policy.act({"rgb": np.zeros((8, 8, 3), dtype=np.uint8)})


def _prompt_ids(vllm):
    return vllm.prompts[-1]["prompt_token_ids"]


def test_accepted_memory_reaches_final_vllm_tokens_and_tracks_evidence(monkeypatch):
    engine, processor, vllm = _make_engine(monkeypatch)
    policy, manager = _make_policy(
        engine, detections={0: SceneExtraction(objects=["uniquememorymarker"], confidence=0.88)},
    )
    _act(policy, instruction="Find uniquememorymarker")
    evidence_id = next(
        event.event_id for event in manager.episodic.events
        if event.kind == "object_observation"
    )
    decoded = processor.tokenizer.decode(_prompt_ids(vllm))
    diagnostic = engine.last_input_diagnostic
    assert "instruction landmark uniquememorymarker" in processor.last_rendered_text
    assert "instruction landmark uniquememorymarker" in processor.last_tokenizer_text
    assert "instruction landmark uniquememorymarker" in decoded
    assert "<navigation_task>Find uniquememorymarker</navigation_task>" in decoded
    assert decoded.endswith("<|im_start|>assistant\n")
    assert "placeholder" not in decoded  # Engine cuts at the first assistant answer boundary.
    assert processor.last_video_shapes == [(1, 3, 4, 4)]
    assert policy.get_info()["raw_text"] == "<traj_1>"
    assert policy._last_executed_action["action"] == "velocity_control"
    assert diagnostic["episode_id"] == "episode_probe"
    assert diagnostic["step_id"] == 0
    assert len(diagnostic["prompt_sha256"]) == 64
    assert diagnostic["memory_text_present_in_final_tokens"] is True
    assert diagnostic["memory_text_present_in_vllm_prompt"] is True
    assert diagnostic["original_instruction_present_in_final_tokens"] is True
    assert diagnostic["original_instruction_present_in_vllm_prompt"] is True
    assert diagnostic["assistant_generation_boundary_present_in_final_tokens"] is True
    assert diagnostic["assistant_generation_boundary_present_in_vllm_prompt"] is True
    assert diagnostic["vllm_prompt_matches_final_tokens"] is True
    assert evidence_id in diagnostic["model_visible_evidence_ids"]
    assert diagnostic["accepted_records"][0]["evidence_ids"] == [evidence_id]
    assert diagnostic["accepted_records"][0]["present_in_final_tokens"] is True
    assert diagnostic["accepted_records"][0]["present_in_vllm_prompt"] is True
    assert diagnostic["processor_tokens_before_truncation"] == (
        diagnostic["processor_tokens_after_truncation"]
    )
    assert diagnostic["processor_truncated"] is False
    assert policy.last_memory_diagnostic["injected_evidence_ids"] == [evidence_id]
    assert policy.last_memory_diagnostic["injected_records"] == 1
    assert len(policy.last_memory_diagnostic["prompt_gate_decision"]["accepted_records"]) == 1
    assert policy.last_memory_diagnostic["model_input_diagnostic"] == diagnostic
    assert diagnostic["prompt_sha256"] == (
        policy.last_memory_diagnostic["prompt_hash"]["augmented_sha256"]
    )
    assert diagnostic["vllm_prompt_token_count"] == len(_prompt_ids(vllm))
    assert len(diagnostic["vllm_prompt_token_ids_sha256"]) == 64
    policy.close_episode()


def test_empty_and_rejected_memory_keep_exact_original_tokens(monkeypatch):
    base_engine, base_processor, base_vllm = _make_engine(monkeypatch)
    baseline, _ = _make_policy(base_engine, memory_enabled=False)
    _act(baseline, instruction="Find TV")

    empty_engine, empty_processor, empty_vllm = _make_engine(monkeypatch)
    empty_policy, _ = _make_policy(empty_engine)
    _act(empty_policy, instruction="Find TV")

    rejected_engine, processor, rejected_vllm = _make_engine(monkeypatch)
    rejected_policy, rejected_manager = _make_policy(
        rejected_engine,
        detections={0: SceneExtraction(objects=["chair"], confidence=0.88)},
    )
    _act(rejected_policy, instruction="Find TV")
    assert _prompt_ids(base_vllm) == _prompt_ids(empty_vllm) == _prompt_ids(rejected_vllm)
    assert (base_processor.last_video_shapes == empty_processor.last_video_shapes
            == processor.last_video_shapes)
    rejected_id = next(
        event.event_id for event in rejected_manager.episodic.events
        if event.kind == "object_observation"
    )
    assert "chair" not in processor.tokenizer.decode(_prompt_ids(rejected_vllm))
    assert rejected_id not in rejected_engine.last_input_diagnostic["model_visible_evidence_ids"]
    assert any(
        decision["reason"] == "write_gate_rejected"
        and rejected_id in decision["evidence_ids"]
        for decision in rejected_policy.last_memory_diagnostic["retrieval_gate_decisions"]
    )
    assert rejected_policy.last_memory_diagnostic["injected_records"] == 0
    baseline.close_episode()
    empty_policy.close_episode()
    rejected_policy.close_episode()


def test_legacy_video_overflow_warns_and_keeps_compatible_truncation(monkeypatch, caplog):
    engine, _, vllm = _make_engine(monkeypatch, max_seq_len=120, video_expansion=300)
    policy, _ = _make_policy(
        engine, detections={0: SceneExtraction(objects=["television"], confidence=0.88)},
    )
    _act(policy)
    assert len(vllm.prompts) == 1
    processor_trace = engine.bundle.data_processor.last_process_diagnostic
    assert processor_trace["tokens_before_truncation"] > 120
    assert processor_trace["tokens_after_truncation"] == 120
    assert processor_trace["truncated"] is True
    assert processor_trace["length_rejected"] is False
    assert "processor truncated prompt" in caplog.text
    assert policy.last_memory_diagnostic["injected_memory_text"]
    assert engine.last_input_diagnostic["processor_truncated"] is True
    assert engine.last_input_diagnostic["final_token_count"] == 120
    assert engine.last_input_diagnostic["vllm_prompt_matches_final_tokens"] is True
    policy.close_episode()


def _temporal_observations():
    return {
        0: SceneExtraction(objects=["couch"], confidence=0.4),
        5: SceneExtraction(objects=["couch"], confidence=0.4),
    }


def test_c_without_temporal_support_keeps_original_final_input(monkeypatch):
    baseline_engine, _, baseline_vllm = _make_engine(monkeypatch)
    baseline, _ = _make_policy(baseline_engine, memory_enabled=False)
    _act(baseline, instruction="Find sofa", steps=6)
    engine, _, vllm = _make_engine(monkeypatch)
    policy, _ = _make_policy(
        engine, config=RGMConfig(enabled=True, yolo_temporal_enabled=False),
        detections=_temporal_observations(),
    )
    _act(policy, instruction="Find sofa", steps=6)
    assert engine.last_input_diagnostic["step_id"] == 5
    assert engine.last_input_diagnostic["memory_text_present_in_final_tokens"] is False
    assert policy.last_memory_diagnostic["injected_records"] == 0
    assert _prompt_ids(vllm) == _prompt_ids(baseline_vllm)
    baseline.close_episode()
    policy.close_episode()


def test_d_temporal_support_reaches_final_model_input(monkeypatch):
    engine, processor, vllm = _make_engine(monkeypatch)
    policy, _ = _make_policy(
        engine, config=RGMConfig(enabled=True, yolo_temporal_enabled=True),
        detections=_temporal_observations(),
    )
    _act(policy, instruction="Find sofa", steps=6)
    assert engine.last_input_diagnostic["step_id"] == 5
    assert engine.last_input_diagnostic["memory_text_present_in_final_tokens"] is True
    assert engine.last_input_diagnostic["memory_text_present_in_vllm_prompt"] is True
    assert engine.last_input_diagnostic["vllm_prompt_matches_final_tokens"] is True
    assert "instruction landmark couch" in processor.tokenizer.decode(_prompt_ids(vllm))
    assert any(row["reason"] == "temporally_supported_category" and row["accepted"]
               for row in policy.last_memory_diagnostic["write_gate_decisions"])
    policy.close_episode()


def test_d_temporal_evidence_changes_final_model_input(monkeypatch):
    c_engine, _, c_vllm = _make_engine(monkeypatch)
    c_policy, _ = _make_policy(
        c_engine, config=RGMConfig(enabled=True, yolo_temporal_enabled=False),
        detections=_temporal_observations(),
    )
    _act(c_policy, instruction="Find sofa", steps=6)

    d_engine, _, d_vllm = _make_engine(monkeypatch)
    d_policy, _ = _make_policy(
        d_engine, config=RGMConfig(enabled=True, yolo_temporal_enabled=True),
        detections=_temporal_observations(),
    )
    _act(d_policy, instruction="Find sofa", steps=6)

    assert _prompt_ids(c_vllm) != _prompt_ids(d_vllm)
    assert (c_engine.last_input_diagnostic["final_token_ids_sha256"]
            != d_engine.last_input_diagnostic["final_token_ids_sha256"])
    c_policy.close_episode()
    d_policy.close_episode()


class _PlaceExtractorSpy:
    source = "mock_dino"

    def reset_episode(self, episode_id, instruction):
        pass

    def close_episode(self):
        pass

    def extract(self, packet):
        return (1.0, 0.0) if packet.step_id == 0 else None

    def extract_features(self, packet):
        if packet.step_id != 0:
            return None
        return VisualPlaceFeatures(
            global_descriptor=(1.0, 0.0),
            patch_descriptors=np.zeros((0, 2), dtype=np.float16),
            patch_grid=(1, 1),
            preprocessing_metadata={"image_quality": 1.0},
        )


def test_d_verified_visual_place_changes_final_model_input(monkeypatch):
    c_engine, _, c_vllm = _make_engine(monkeypatch)
    c_policy, _ = _make_policy(
        c_engine, config=RGMConfig(enabled=True),
        place_extractor=_PlaceExtractorSpy(),
    )
    _act(c_policy, steps=2)

    d_engine, processor, d_vllm = _make_engine(monkeypatch)
    d_policy, _ = _make_policy(
        d_engine, config=RGMConfig(
            enabled=True, yolo_temporal_enabled=True,
            place_two_stage_enabled=True, topology_motion_validation_enabled=True,
        ),
        place_extractor=_PlaceExtractorSpy(),
    )
    _act(d_policy, steps=2)

    assert c_engine.last_input_diagnostic["memory_text_present_in_final_tokens"] is False
    assert d_engine.last_input_diagnostic["memory_text_present_in_vllm_prompt"] is True
    assert "place_observation_0" in d_engine.last_input_diagnostic["model_visible_evidence_ids"]
    assert "visual place" in processor.tokenizer.decode(_prompt_ids(d_vllm))
    assert "current place unverified" in processor.tokenizer.decode(_prompt_ids(d_vllm))
    assert _prompt_ids(c_vllm) != _prompt_ids(d_vllm)
    c_policy.close_episode()
    d_policy.close_episode()


def test_c_step0_object_reaches_final_tokens_without_place_claim(monkeypatch):
    detections = {0: SceneExtraction(objects=["couch"], confidence=0.88)}
    c_engine, c_processor, c_vllm = _make_engine(monkeypatch)
    c_policy, _ = _make_policy(
        c_engine, config=RGMConfig(enabled=True), detections=detections,
        place_extractor=_PlaceExtractorSpy(),
    )
    _act(c_policy, instruction="Find the sofa")
    c_text = c_processor.tokenizer.decode(_prompt_ids(c_vllm))
    assert "instruction landmark couch" in c_text
    assert "visual place hypothesis" not in c_text
    assert c_engine.last_input_diagnostic["memory_text_present_in_vllm_prompt"] is True
    c_policy.close_episode()


def test_d_step0_place_object_relation_reaches_final_tokens(monkeypatch):
    detections = {0: SceneExtraction(objects=["couch"], confidence=0.88)}
    d_engine, d_processor, d_vllm = _make_engine(monkeypatch)
    d_policy, _ = _make_policy(
        d_engine, config=RGMConfig(
            enabled=True, yolo_temporal_enabled=True,
            place_two_stage_enabled=True, topology_motion_validation_enabled=True,
        ), detections=detections, place_extractor=_PlaceExtractorSpy(),
    )
    _act(d_policy, instruction="Find the sofa")
    d_text = d_processor.tokenizer.decode(_prompt_ids(d_vllm))
    assert "instruction landmark couch while visual place hypothesis" in d_text
    assert d_engine.last_input_diagnostic["memory_text_present_in_vllm_prompt"] is True
    assert d_engine.last_input_diagnostic["vllm_prompt_matches_final_tokens"] is True
    d_policy.close_episode()


def test_c_d_step0_place_object_inputs_differ(monkeypatch):
    detections = {0: SceneExtraction(objects=["couch"], confidence=0.88)}
    c_engine, _, c_vllm = _make_engine(monkeypatch)
    c_policy, _ = _make_policy(
        c_engine, config=RGMConfig(enabled=True), detections=detections,
        place_extractor=_PlaceExtractorSpy(),
    )
    _act(c_policy, instruction="Find the sofa")
    d_engine, _, d_vllm = _make_engine(monkeypatch)
    d_policy, _ = _make_policy(
        d_engine, config=RGMConfig(
            enabled=True, yolo_temporal_enabled=True,
            place_two_stage_enabled=True, topology_motion_validation_enabled=True,
        ), detections=detections, place_extractor=_PlaceExtractorSpy(),
    )
    _act(d_policy, instruction="Find the sofa")
    assert _prompt_ids(c_vllm) != _prompt_ids(d_vllm)
    assert c_engine.last_input_diagnostic["vllm_prompt_token_ids_sha256"] != (
        d_engine.last_input_diagnostic["vllm_prompt_token_ids_sha256"]
    )
    c_policy.close_episode()
    d_policy.close_episode()
