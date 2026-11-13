"""CPU-only V3 checks from an observation through the final vLLM prompt IDs.

The Qwen processor, tokenizer and vLLM decoder are deterministic fakes. The
Policy, MemoryManager, writer, V3 storage/retrieval/formatter, DataProcessor,
and Engine entry points remain real.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.data_processor import Qwen3VLDataProcessor
from lightnav.habitat.policy import TrajVocabVLNCEPolicy
from lightnav.habitat.runner import MemoryConfig, _prepare_v3_output_dir
from lightnav.inference.engine import VLNInferenceEngine
from lightnav.inference.policies import NavigationPolicy
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.schemas import SceneExtraction
from lightnav.memory.v3_config import RGMV3Config


class _Tokenizer:
    _special = {
        "<|im_start|>assistant\n": [6, 5, 7],
        "<|im_start|>user\n": [6, 8, 7],
        "<|im_end|>": [3],
    }

    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 1, "<|video_pad|>": 2, "<|im_end|>": 3}.get(token, 4)

    def encode(self, text, *, add_special_tokens=False):
        if text == "assistant":
            return [5]
        ids = []
        offset = 0
        while offset < len(text):
            special = next((s for s in self._special if text.startswith(s, offset)), None)
            if special is None:
                ids.append(ord(text[offset]) + 1000)
                offset += 1
            else:
                ids.extend(self._special[special])
                offset += len(special)
        return ids

    def decode(self, ids, *, skip_special_tokens=False):
        special = {3: "<|im_end|>", 5: "assistant", 6: "<|im_start|>", 7: "\n", 8: "user"}
        return "".join(
            chr(int(token_id) - 1000) if int(token_id) >= 1000
            else special.get(int(token_id), "")
            for token_id in ids
        )


class _QwenProcessor:
    def __init__(self, *, video_expansion=7):
        self.tokenizer = _Tokenizer()
        self.video_expansion = video_expansion
        self.video_processor = SimpleNamespace(
            fps=4, temporal_patch_size=2, do_sample_frames=False,
            do_resize=False, do_rescale=False, do_normalize=False,
        )
        self.rendered = ""
        self.tokenizer_text = ""
        self.video_shapes = []

    def apply_chat_template(self, messages, *, tokenize=False):
        assert tokenize is False
        turns = []
        for message in messages:
            parts = [
                "V" * self.video_expansion if part["type"] == "video" else part["text"]
                for part in message["content"]
            ]
            turns.append(f"<|im_start|>{message['role']}\n" + "".join(parts) + "<|im_end|>")
        self.rendered = "\n".join(turns)
        return self.rendered

    def __call__(self, *, text, videos, **kwargs):
        assert videos
        self.tokenizer_text = text
        self.video_shapes = [tuple(video.shape) for video in videos]
        return {"input_ids": torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)}


class _VLLM:
    def __init__(self):
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


def _engine(monkeypatch, *, max_seq_len=4096, video_expansion=7):
    monkeypatch.setenv("VLN_EVAL_TRAJ_TOP1", "0")
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
    )
    processor = _QwenProcessor(video_expansion=video_expansion)
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
    vllm = _VLLM()
    engine = VLNInferenceEngine(
        bundle, backend="vllm_local", vllm_engine=vllm, enable_vit_cache=False,
    )
    return engine, processor, vllm


def _policy(engine, *, scope="episode_local", detections=None,
            memory_enabled=True, max_records=3, max_chars=400):
    centroids = np.zeros((2, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    manager = None
    if memory_enabled:
        manager = MemoryManager(
            extractor=MockSceneExtractor(detections or {}), strategy="hierarchical",
            v3_config=RGMV3Config(
                enabled=True, scope=scope, prompt_max_records=max_records,
                prompt_max_chars=max_chars,
            ),
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


def _reset(policy, episode, scene, instruction):
    policy.reset(
        {"instruction": {"text": instruction}},
        memory_episode_id=episode, memory_scene_id=scene,
    )


def _act(policy, times=1):
    for _ in range(times):
        policy.act({"rgb": np.zeros((8, 8, 3), dtype=np.uint8)})


def _ids(vllm):
    return vllm.prompts[-1]["prompt_token_ids"]


def test_current_observation_reaches_real_processor_and_final_tokens(monkeypatch):
    engine, processor, vllm = _engine(monkeypatch)
    policy, manager = _policy(
        engine,
        detections={0: SceneExtraction(objects=["uniquev3marker"], confidence=0.88)},
    )
    _reset(policy, "episode_one", "scene_a", "Walk toward uniquev3marker")
    _act(policy)

    records = manager.v3_store.evidence_for_scene("scene_a")
    assert manager.write_gate is None
    assert manager.retrieval_gate.decisions == []
    assert manager.prompt_gate.last_selection.accepted_records == ()
    marker = next(record for record in records if record.semantic_label == "uniquev3marker")
    decoded = processor.tokenizer.decode(_ids(vllm))
    diagnostic = engine.last_input_diagnostic
    assert "uniquev3marker" in processor.rendered
    assert "uniquev3marker" in processor.tokenizer_text
    assert processor.video_shapes == [(1, 3, 4, 4)]
    assert policy._last_executed_action["action"] == "velocity_control"
    assert "uniquev3marker" in decoded
    assert "<navigation_task>Walk toward uniquev3marker</navigation_task>" in decoded
    assert decoded.endswith("<|im_start|>assistant\n")
    assert diagnostic["memory_text_present_in_final_tokens"] is True
    assert diagnostic["memory_text_present_in_vllm_prompt"] is True
    assert diagnostic["original_instruction_present_in_final_tokens"] is True
    assert diagnostic["vllm_prompt_matches_final_tokens"] is True
    assert marker.key in diagnostic["model_visible_evidence_ids"]
    assert marker.key in diagnostic["literal_model_visible_evidence_ids"]
    assert diagnostic["accepted_records"][0]["present_in_vllm_prompt"] is True
    assert marker.key in diagnostic["accepted_records"][0][
        "literal_evidence_ids_in_vllm_prompt"
    ]
    assert policy.last_memory_diagnostic["v3_diagnostic"]["actual_injection"] is True
    assert policy.last_memory_diagnostic["v3_diagnostic"]["retrieval_nonempty"] is True
    assert policy.last_memory_diagnostic["v3_diagnostic"]["fusion_nonempty"] is True
    assert policy.last_memory_diagnostic["v3_diagnostic"]["formatter_nonempty"] is True
    assert policy.last_memory_diagnostic["v3_diagnostic"][
        "actual_historical_injection"
    ] is False
    assert policy.last_memory_diagnostic["v3_diagnostic"]["prompt_token_delta"] > 0
    policy.close_episode()


def test_current_episode_mid_segment_reaches_final_prompt_tokens(monkeypatch):
    engine, processor, vllm = _engine(monkeypatch)
    policy, manager = _policy(
        engine, detections={
            0: SceneExtraction(objects=["archway"], confidence=0.88),
            2: SceneExtraction(objects=["archway"], confidence=0.88),
        },
    )
    _reset(policy, "123", "scene_a", "Walk past the archway and turn left")
    _act(policy, times=3)
    decoded = processor.tokenizer.decode(_ids(vllm))
    diag = policy.last_memory_diagnostic["v3_diagnostic"]
    assert manager.v3_store.active_segment(before_step=2) is not None
    assert diag["episodic_historical_candidates"] >= 1
    assert diag["prompt_historical_records_serialized"] >= 1
    assert diag["actual_historical_injection"] is True
    assert "[ACTION HISTORY] Earlier segment" in decoded
    assert "<navigation_task>Walk past the archway and turn left</navigation_task>" in decoded
    assert engine.last_input_diagnostic["memory_text_present_in_vllm_prompt"] is True
    policy.close_episode()


def test_empty_v3_memory_preserves_exact_original_model_input(monkeypatch):
    base_engine, base_processor, base_vllm = _engine(monkeypatch)
    baseline, _ = _policy(base_engine, memory_enabled=False)
    _reset(baseline, "base", "scene_a", "Find the door")
    _act(baseline)

    v3_engine, v3_processor, v3_vllm = _engine(monkeypatch)
    policy, _ = _policy(v3_engine)
    _reset(policy, "empty", "scene_a", "Find the door")
    _act(policy)

    assert _ids(v3_vllm) == _ids(base_vllm)
    assert v3_processor.video_shapes == base_processor.video_shapes
    assert v3_engine.last_input_diagnostic["memory_text_present_in_final_tokens"] is False
    assert policy.last_memory_diagnostic["v3_diagnostic"]["prompt_records_serialized"] == 0
    assert policy.last_memory_diagnostic["v3_diagnostic"]["retrieval_nonempty"] is False
    assert policy.last_memory_diagnostic["v3_diagnostic"]["fusion_nonempty"] is False
    assert policy.last_memory_diagnostic["v3_diagnostic"]["formatter_nonempty"] is False
    assert policy.last_memory_diagnostic["v3_diagnostic"]["actual_injection"] is False
    baseline.close_episode()
    policy.close_episode()


def test_episode_scope_and_scene_identity_control_historical_retrieval(monkeypatch):
    instruction = "Find the historicalarchway"
    base_engine, _, base_vllm = _engine(monkeypatch)
    baseline, _ = _policy(base_engine, memory_enabled=False)
    _reset(baseline, "base", "scene_a", instruction)
    _act(baseline)

    for scope, second_scene, expect_history in (
        ("episode_local", "scene_a", False),
        ("scene_persistent", "scene_b", False),
        ("scene_persistent", "scene_a", True),
    ):
        engine, processor, vllm = _engine(monkeypatch)
        policy, manager = _policy(
            engine, scope=scope,
            detections={0: SceneExtraction(objects=["historicalarchway"], confidence=0.81)},
        )
        _reset(policy, "historical_episode", "scene_a", instruction)
        _act(policy, times=2)  # The second observation records the preceding command.
        policy.close_episode(outcome="failure")
        manager.extractor.by_step = {}

        _reset(policy, "current_episode", second_scene, instruction)
        _act(policy)
        decoded = processor.tokenizer.decode(_ids(vllm))
        assert ("[Navigation memory;" in decoded) is expect_history
        assert (_ids(vllm) != _ids(base_vllm)) is expect_history
        if expect_history:
            assert "historicalarchway" in decoded
            assert engine.last_input_diagnostic["memory_text_present_in_vllm_prompt"] is True
            assert engine.last_input_diagnostic["model_visible_evidence_ids"]
            assert policy.last_memory_diagnostic["v3_diagnostic"][
                "actual_historical_injection"
            ] is True
            assert any("historical_episode" in evidence_id for evidence_id in
                       engine.last_input_diagnostic["model_visible_evidence_ids"])
        else:
            assert engine.last_input_diagnostic["memory_text_present_in_final_tokens"] is False
        policy.close_episode()
    baseline.close_episode()


def test_record_budget_keeps_whole_memory_lines_and_instruction(monkeypatch):
    engine, processor, vllm = _engine(monkeypatch)
    policy, _ = _policy(
        engine, max_records=1,
        detections={0: SceneExtraction(
            objects=["firstmarker", "secondmarker"], confidence=0.85,
        )},
    )
    _reset(policy, "budget_episode", "scene_a", "Find firstmarker and secondmarker")
    _act(policy)
    decoded = processor.tokenizer.decode(_ids(vllm))
    diag = policy.last_memory_diagnostic["v3_diagnostic"]
    assert diag["prompt_records_serialized"] == 1
    assert policy.last_memory_diagnostic["injected_records"] == 1
    assert len(diag["injected_evidence_ids"]) == 1
    assert len(engine.last_input_diagnostic["accepted_records"]) == 1
    assert len(engine.last_input_diagnostic["model_visible_evidence_ids"]) >= 1
    assert "<navigation_task>Find firstmarker and secondmarker</navigation_task>" in decoded
    assert decoded.endswith("<|im_start|>assistant\n")
    policy.close_episode()


def test_v3_video_overflow_never_silently_truncates_task_or_memory(monkeypatch):
    engine, _, vllm = _engine(monkeypatch, max_seq_len=120, video_expansion=300)
    policy, manager = _policy(
        engine,
        detections={0: SceneExtraction(objects=["uniquememorymarker"], confidence=0.88)},
    )
    _reset(policy, "overlong", "scene_a", "Find uniquememorymarker")
    with pytest.raises(ValueError, match="length|Length|token|Token"):
        _act(policy)
    assert vllm.prompts == []
    assert engine.bundle.data_processor.last_process_diagnostic[
        "tokens_before_truncation"
    ] > 120
    assert engine.bundle.data_processor.last_process_diagnostic["length_rejected"] is True
    assert engine.bundle.data_processor.last_process_diagnostic[
        "tokens_after_truncation"
    ] is None
    assert manager.last_v3_diagnostic["formatter_nonempty"] is True
    policy.close_episode()


def test_past_behavior_experience_is_visible_with_failure_label(monkeypatch):
    engine, processor, vllm = _engine(monkeypatch, max_seq_len=8192)
    policy, manager = _policy(
        engine, scope="scene_persistent", max_records=5, max_chars=1000,
        detections={0: SceneExtraction(objects=["archway"], confidence=0.9)},
    )
    _reset(policy, "past", "scene_a", "Find the archway")
    _act(policy, times=2)
    policy.close_episode(outcome="failure")  # Synthetic trusted outcome, not Habitat info.
    manager.extractor.by_step = {}
    _reset(policy, "now", "scene_a", "Find the archway")
    _act(policy)
    decoded = processor.tokenizer.decode(_ids(vllm))
    self_diag = policy.last_memory_diagnostic["v3_diagnostic"]
    assert "[EPISODIC EXPERIENCE]" in decoded
    assert "Outcome: failure" in decoded
    assert "actions=1" in decoded
    assert "Issued velocity control" in decoded
    assert self_diag["episodic_candidates"] >= 1
    assert self_diag["actual_injection"] is True
    assert engine.last_input_diagnostic["memory_text_present_in_vllm_prompt"] is True
    policy.close_episode()


def test_v3_protocol_marker_blocks_scope_mix_and_legacy_pilot_path(tmp_path):
    path = tmp_path / "rgm_v3_v1_episode_local"
    local = RGMV3Config(enabled=True, scope="episode_local")
    _prepare_v3_output_dir(str(path), local, resume=False)
    assert (path / "rgm_v3_protocol.json").is_file()
    _prepare_v3_output_dir(str(path), local, resume=True)
    with pytest.raises(ValueError, match="differs"):
        _prepare_v3_output_dir(
            str(path), RGMV3Config(enabled=True, scope="scene_persistent"), resume=True,
        )
    with pytest.raises(ValueError, match="legacy"):
        _prepare_v3_output_dir(
            str(tmp_path / "rgm_pilot_88_v2" / "shard_0"), local, resume=False,
        )


def test_legacy_and_v3_backends_cannot_share_one_run():
    with pytest.raises(ValueError, match="cannot run together"):
        MemoryConfig(
            enabled=True, strategy="hierarchical",
            rgm=RGMConfig(enabled=True), rgm_v3=RGMV3Config(enabled=True),
        )
