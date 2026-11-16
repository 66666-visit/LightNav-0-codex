"""CPU Mock API checks for the Scene Query path through Habitat Policy."""

from __future__ import annotations

import json
import sys
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.data_processor import Qwen3VLDataProcessor
from lightnav.habitat.policy import TrajVocabVLNCEPolicy
from lightnav.inference.engine import VLNInferenceEngine
from lightnav.inference.policies import NavigationPolicy
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.scene_query_config import SceneQueryConfig
from lightnav.memory.schemas import DetectedObject, SceneExtraction
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.yolo_seg_extractor import SegmentationInstance


class _Engine:
    def __init__(self, *, reject_memory: bool = False) -> None:
        self.bundle = SimpleNamespace(
            video_size=(4, 4), video_fps=4, num_history_frames=2,
            pool_enable=False, pool_spatial=1, pool_mode="avg",
            slowfast_tiers=None, predict_horizon=1, action_method="flat",
        )
        self.backend = "hf"
        self.reject_memory = reject_memory
        self.samples: list[dict] = []
        self.last_input_diagnostic: dict | None = None
        self.last_prompt_chars = 0

    def reset_episode_state(self) -> None:
        pass

    def generate(self, sample: dict) -> tuple[str, float]:
        self.samples.append(sample)
        prompt = sample["conversations"][0]["value"]
        if self.reject_memory and "[Navigation Memory]" in prompt:
            raise ValueError("processor input length exceeded")
        trace = sample.get("_rgm_trace", {})
        memory_text = trace.get("memory_text", "")
        visible = bool(memory_text and memory_text in prompt)
        token_ids = [ord(char) for char in prompt]
        self.last_input_diagnostic = {
            "final_token_ids": token_ids,
            "final_token_ids_sha256": sha256(
                np.asarray(token_ids, dtype=np.int32).tobytes()
            ).hexdigest(),
            "final_token_count": len(token_ids),
            "injected_memory_tokens": len(memory_text) if visible else 0,
            "memory_text_present_in_final_tokens": visible,
            "model_visible_evidence_ids": [
                evidence_id for record in trace.get("accepted_records", [])
                for evidence_id in record["evidence_ids"]
            ] if visible else [],
        }
        self.last_prompt_chars = len(prompt)
        return "<traj_1>", 1.0


class _SegExtractor:
    def reset_episode(self, episode_id: str, instruction: str) -> None:
        pass

    def close_episode(self) -> None:
        pass

    def extract(self, rgb: np.ndarray, step_id: int) -> list[SegmentationInstance]:
        return [SegmentationInstance(
            "chair", 0.95, np.ones(rgb.shape[:2], dtype=np.bool_),
            (0.0, 0.0, float(rgb.shape[1]), float(rgb.shape[0])),
        )]


def _final(decision: str, *, l1: list[str] | None = None,
           objects: list[str] | None = None, evidence: list[str] | None = None) -> dict:
    return {"content": json.dumps({
        "decision": decision,
        "selected_l1_fact_ids": l1 or [],
        "selected_l3_object_ids": objects or [],
        "selected_l3_evidence_ids": evidence or [],
        "reason_code": "TEST_SELECTION",
    })}


def _tool(name: str, arguments: dict, number: int) -> dict:
    return {"tool_calls": [{
        "id": f"query_{number}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}


class _Provider:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.user_payloads: list[dict] = []
        self.calls = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        self.calls += 1
        payload = json.loads(messages[1]["content"])
        self.user_payloads.append(payload)
        if self.mode == "timeout":
            raise TimeoutError("synthetic API timeout")
        if self.mode == "invalid_json":
            return {"content": "not a JSON object"}
        if self.mode == "no_memory" or payload["current_step"] == 0:
            return _final("NO_MEMORY")
        if self.mode == "l1":
            current = next(item for item in payload["l1_facts"] if item["current"])
            return _final("USE_L1", l1=[current["fact_id"]])
        tool_results = [message for message in messages if message["role"] == "tool"]
        if not tool_results:
            return _tool("find_objects", {"class_name": "chair"}, 1)
        first = json.loads(tool_results[0]["content"])["result"][0]
        if len(tool_results) == 1:
            return _tool("read_evidence", {"evidence_id": first["evidence_ids"][0]}, 2)
        evidence = json.loads(tool_results[1]["content"])["result"]
        l1 = []
        if self.mode == "both":
            linked = evidence["linked_l1_evidence_id"]
            assert linked in {item["fact_id"] for item in payload["l1_facts"]}
            l1 = [linked]
        return _final(
            "USE_BOTH" if l1 else "USE_L3", l1=l1,
            objects=[first["object_id"]], evidence=[evidence["evidence_id"]],
        )


@pytest.fixture(autouse=True)
def _small_video(monkeypatch) -> None:
    monkeypatch.setattr(
        NavigationPolicy, "_convert_frame",
        lambda self, frame: torch.full((3, 4, 4), float(frame[0, 0, 0])),
    )


def _policy(provider: _Provider, *, enabled: bool | None = True,
            reject_memory: bool = False, engine=None):
    detections = {step: SceneExtraction(
        objects=["chair"], detections=[DetectedObject(
            "chair", 0.95, (0.0, 0.0, 8.0, 8.0),
        )], confidence=0.95, source="fake_l1",
    ) for step in (0, 1)}
    manager = MemoryManager(
        extractor=MockSceneExtractor(detections), strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=SpatialL3Config(
            mode="inject", seg_model_path="/fake/yolo11n-seg.pt", seg_every_n_steps=1,
        ),
        spatial_extractor=_SegExtractor(),
        scene_query_config=(SceneQueryConfig(enabled=enabled, max_rounds=4)
                            if enabled is not None else None),
        scene_query_provider=provider,
    )
    centroids = np.zeros((3, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    centroids[2, :, 2] = 0.1
    engine = engine or _Engine(reject_memory=reject_memory)
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


def _act(policy: TrajVocabVLNCEPolicy, step: int) -> None:
    rgb = np.full((8, 8, 3), step, dtype=np.uint8)
    obs = {"rgb": rgb, "depth": np.full((8, 8), 2.0, dtype=np.float32)}
    info = {"spatial_rgbd": {
        "status": "ok", "aligned": True, "image_size_hw": [8, 8],
        "K": [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]],
        "agent_pose": {"position": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
        "camera_extrinsics": {
            "position": [0.0, 0.88, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0],
        },
        "camera_frame": "habitat_right_up_back",
        "depth_encoding": "meters", "min_depth_m": 0.1, "max_depth_m": 10.0,
    }}
    if step > 0:
        info["spatial_transition_record"] = True
    policy.act(obs, info)


@pytest.mark.parametrize(("mode", "decision", "actual"), [
    ("no_memory", "NO_MEMORY", False),
    ("l1", "USE_L1", True),
    ("l3", "USE_L3", True),
    ("both", "USE_BOTH", True),
])
def test_four_decisions_match_actual_memory_and_sampled_rgb_sources(
    mode: str, decision: str, actual: bool,
) -> None:
    policy, _, engine = _policy(_Provider(mode))
    _act(policy, 0)
    _act(policy, 1)
    diagnostic = policy.last_memory_diagnostic
    assert diagnostic is not None
    assert diagnostic["api_decision"] == diagnostic["decision"] == decision
    assert diagnostic["actual_injection"] is actual
    sampled_frame_ids = {
        int(frame_id) for segment in engine.samples[-1]["video_segments"]
        for frame_id in (segment.get("frame_indices") or ())
    }
    expected_source_steps = sorted(
        step for step, frame_id in policy._scene_query_frame_ids_by_step.items()
        if frame_id in sampled_frame_ids
    )
    assert diagnostic["native_visual_source_steps"] == expected_source_steps
    assert bool(diagnostic["selected_l1_fact_ids"]) == (decision in {"USE_L1", "USE_BOTH"})
    assert bool(diagnostic["selected_l3_evidence_ids"]) == (decision in {"USE_L3", "USE_BOTH"})
    if decision == "NO_MEMORY":
        assert "[Navigation Memory]" not in engine.samples[-1]["conversations"][0]["value"]
        assert diagnostic["injected_evidence_ids"] == []
        assert diagnostic["memory_context_tokens"] == 0
    else:
        assert "[Navigation Memory]" in engine.samples[-1]["conversations"][0]["value"]
        assert diagnostic["memory_context_tokens"] == diagnostic[
            "model_input_diagnostic"
        ]["injected_memory_tokens"]
        assert set(diagnostic["injected_evidence_ids"]) == {
            *diagnostic["selected_l1_fact_ids"],
            *diagnostic["selected_l3_evidence_ids"],
        }
    if decision == "USE_L1":
        assert 1 in expected_source_steps
        assert diagnostic["current_l1_visual_proxy_count"] == 1
        assert diagnostic["native_visual_overlap_count"] == 1
        assert diagnostic["native_visual_overlap_evidence_ids"] == diagnostic[
            "selected_l1_fact_ids"
        ]


def test_policy_resets_controller_history_and_injection_stats_between_episodes() -> None:
    provider = _Provider("both")
    policy, _, engine = _policy(provider)
    _act(policy, 0)
    _act(policy, 1)
    first = policy.last_memory_diagnostic
    assert first is not None and first["decision"] == "USE_BOTH"
    first_ids = set(first["injected_evidence_ids"])
    first_stats = policy.scene_query_episode_stats
    assert first_stats is not None
    assert first_stats["step_count"] == 2
    assert first_stats["l3_query_count"] == 2
    assert first_stats["actual_injection_step_count"] == 1
    assert first_stats["injected_evidence_count"] == len(first_ids)
    calls_before_reset = provider.calls

    policy.reset(
        {"instruction": {"text": "find the chair"}},
        memory_episode_id="episode_two", memory_scene_id="scene",
    )
    controller = policy._scene_query_controller
    assert controller is not None
    assert controller._selection_history == []
    assert controller._query_signatures == set()
    assert controller._injected_evidence == set()
    assert policy._scene_query_frame_ids_by_step == {}
    assert all(value == 0 for value in policy.scene_query_episode_stats.values())

    _act(policy, 0)
    assert provider.user_payloads[calls_before_reset]["recent_memory_selections"] == []
    _act(policy, 1)
    second = policy.last_memory_diagnostic
    assert second is not None and second["decision"] == "USE_BOTH"
    assert second["episode_id"] == "episode_two"
    assert set(second["injected_evidence_ids"]).isdisjoint(first_ids)
    assert all(not call["repeated"] for call in second["query_calls"])
    second_stats = policy.scene_query_episode_stats
    assert second_stats is not None
    assert second_stats["step_count"] == 2
    assert second_stats["l3_query_count"] == 2
    assert second_stats["actual_injection_step_count"] == 1
    assert second_stats["repeated_query_count"] == 0
    assert second_stats["repeated_evidence_injection_count"] == 0
    assert second_stats["injected_evidence_count"] == len(second["injected_evidence_ids"])
    assert second_stats["actual_prompt_memory_tokens"] == second["memory_context_tokens"]
    assert second_stats["actual_prompt_total_tokens"] == sum(
        len(sample["conversations"][0]["value"]) for sample in engine.samples[-2:]
    )


def test_scene_query_builds_one_context_and_preserves_original_prompt(monkeypatch) -> None:
    provider = _Provider("both")
    policy, manager, engine = _policy(provider)

    def old_spatial_path(*args, **kwargs):
        raise AssertionError("legacy spatial-first path ran in scene_query mode")

    monkeypatch.setattr(manager, "get_spatial_context", old_spatial_path)
    _act(policy, 0)
    _act(policy, 1)
    sample = engine.samples[-1]
    trace = sample["_rgm_trace"]
    prompt = sample["conversations"][0]["value"]
    diagnostic = policy.last_memory_diagnostic
    assert trace["memory_use_mode"] == "scene_query"
    assert len(trace["accepted_records"]) == 1
    assert prompt.count("[Navigation Memory]") == 1
    assert prompt.count("Spatial history:") == 1
    assert "Recent RGB at step 0: detected chair" in prompt
    assert diagnostic["prompt_roundtrip_ok"] is True
    assert diagnostic["prompt_hash_before"] == sha256(
        trace["_base_sample"]["conversations"][0]["value"].encode()
    ).hexdigest()
    assert trace["_base_sample"]["video_segments"] is sample["video_segments"]
    assert diagnostic["decision"] == "USE_BOTH"
    assert diagnostic["memory_context_item_count"] == 1
    assert diagnostic["actual_injection"] is True
    assert diagnostic["actual_historical_injection"] is True
    assert set(diagnostic["injected_evidence_ids"]) == {
        *diagnostic["selected_l1_fact_ids"],
        *diagnostic["selected_l3_evidence_ids"],
    }
    assert diagnostic["selected_l3_object_ids"][0] in manager.spatial_map.objects
    assert diagnostic["selected_l3_evidence_ids"][0] in manager.spatial_map.evidence
    assert diagnostic["deduplicated_fact_count"] == 1
    stats = policy.scene_query_episode_stats
    assert stats is not None
    assert stats["step_count"] == 2
    assert stats["both_selection_count"] == 1
    assert stats["l3_query_count"] == 2
    assert stats["actual_injection_step_count"] == 1
    assert diagnostic["scene_query_episode_stats"] == stats
    assert diagnostic["native_visual_source_steps"]
    assert diagnostic["native_visual_overlap_count"] == 1
    assert set(diagnostic["native_visual_overlap_evidence_ids"]) == set(
        diagnostic["injected_evidence_ids"]
    )


def test_scene_query_overflow_drops_whole_fact_and_restores_baseline() -> None:
    provider = _Provider("both")
    policy, _, engine = _policy(provider, reject_memory=True)
    _act(policy, 0)
    _act(policy, 1)
    diagnostic = policy.last_memory_diagnostic
    attempted = engine.samples[-2]
    final = engine.samples[-1]
    assert "[Navigation Memory]" in attempted["conversations"][0]["value"]
    assert final["conversations"] == attempted["_rgm_trace"]["_base_sample"]["conversations"]
    assert "_rgm_trace" not in final
    assert diagnostic["processor_length_rejections"] == 1
    assert len(diagnostic["removed_due_token_limit"]) == 1
    assert set(diagnostic["removed_due_token_limit"][0]["evidence_ids"]) == {
        *diagnostic["pre_budget_selected_l1_fact_ids"],
        *diagnostic["pre_budget_selected_l3_evidence_ids"],
    }
    assert diagnostic["selected_l1_fact_ids"] == []
    assert diagnostic["selected_l3_evidence_ids"] == []
    assert diagnostic["injected_evidence_ids"] == []
    assert diagnostic["actual_injection"] is False
    assert diagnostic["prompt_hash_before"] == diagnostic["prompt_hash_after"]
    assert diagnostic["deduplicated_fact_count"] == 0
    assert diagnostic["native_visual_overlap_count"] == 0
    assert diagnostic["scene_query_episode_stats"]["both_selection_count"] == 0
    assert diagnostic["scene_query_episode_stats"]["no_memory_count"] == 2
    assert diagnostic["scene_query_episode_stats"]["rendered_fact_count"] == 0
    assert diagnostic["scene_query_episode_stats"]["actual_prompt_memory_tokens"] == 0


@pytest.mark.parametrize("failure", ["timeout", "invalid_json"])
def test_api_failure_uses_exact_original_lightnav_sample(failure: str) -> None:
    baseline, _, baseline_engine = _policy(_Provider("no_memory"))
    _act(baseline, 0)
    failed, _, failed_engine = _policy(_Provider(failure))
    _act(failed, 0)
    sample = failed_engine.samples[-1]
    diagnostic = failed.last_memory_diagnostic
    assert sample["conversations"] == baseline_engine.samples[-1]["conversations"]
    baseline_segments = baseline_engine.samples[-1]["video_segments"]
    assert len(sample["video_segments"]) == len(baseline_segments)
    for actual, expected in zip(sample["video_segments"], baseline_segments):
        assert actual.keys() == expected.keys()
        for key in actual:
            if isinstance(actual[key], torch.Tensor):
                assert torch.equal(actual[key], expected[key])
            else:
                assert actual[key] == expected[key]
    assert "_rgm_trace" not in sample
    assert diagnostic["fallback_reason"]
    assert diagnostic["selected_l3_evidence_ids"] == []
    assert diagnostic["actual_injection"] is False
    assert "Spatial history:" not in sample["conversations"][0]["value"]


def test_disabled_scene_query_keeps_existing_spatial_l3_mode() -> None:
    provider = _Provider("timeout")  # Must never be called when disabled.
    policy, _, engine = _policy(provider, enabled=False)
    _act(policy, 0)
    _act(policy, 1)
    prompt = engine.samples[-1]["conversations"][0]["value"]
    assert provider.calls == 0
    assert "Spatial history: chair" in prompt
    assert policy.last_memory_diagnostic["spatial_l3"]["context_source"] == "spatial_l3"
    assert policy.last_memory_diagnostic["actual_injection"] is True


def test_explicitly_disabled_scene_query_matches_existing_policy_input() -> None:
    baseline_provider = _Provider("timeout")
    baseline, _, baseline_engine = _policy(baseline_provider, enabled=None)
    disabled_provider = _Provider("timeout")
    disabled, _, disabled_engine = _policy(disabled_provider, enabled=False)
    for step in (0, 1):
        _act(baseline, step)
        _act(disabled, step)

    assert baseline_provider.calls == disabled_provider.calls == 0
    assert [sample["conversations"] for sample in baseline_engine.samples] == [
        sample["conversations"] for sample in disabled_engine.samples
    ]
    for first, second in zip(baseline_engine.samples, disabled_engine.samples):
        assert len(first["video_segments"]) == len(second["video_segments"])
        for original, candidate in zip(first["video_segments"], second["video_segments"]):
            assert original.keys() == candidate.keys()
            for key in original:
                if isinstance(original[key], torch.Tensor):
                    assert torch.equal(original[key], candidate[key])
                else:
                    assert original[key] == candidate[key]


def test_controller_payload_has_only_l1_l3_and_no_oracle() -> None:
    provider = _Provider("l1")
    policy, _, _ = _policy(provider)
    _act(policy, 0)
    _act(policy, 1)
    payload = provider.user_payloads[-1]
    assert set(payload) == {
        "instruction", "current_step", "current_agent_pose", "l1_facts",
        "native_visual_source_steps", "recent_memory_selections",
        "available_l3_query_tools",
    }
    assert payload["instruction"] == "find the chair"
    assert payload["current_step"] == 1
    assert payload["current_agent_pose"]["position"] == [0.0, 0.0, 0.0]
    assert payload["native_visual_source_steps"]
    assert payload["l1_facts"]
    assert set(payload["available_l3_query_tools"]) == {
        "find_objects", "read_node", "read_evidence", "get_edges_for_node",
    }
    assert "l2" not in json.dumps(payload).casefold()
    assert "success" not in json.dumps(payload).casefold()


class _CharTokenizer:
    _special = {
        "<|im_start|>assistant\n": [6, 5, 7],
        "<|im_start|>user\n": [6, 8, 7],
        "<|im_end|>": [3],
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|image_pad|>": 1, "<|video_pad|>": 2, "<|im_end|>": 3}.get(token, 4)

    def encode(self, value: str, *, add_special_tokens: bool = False) -> list[int]:
        if value == "assistant":
            return [5]
        ids: list[int] = []
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

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        special = {3: "<|im_end|>", 5: "assistant", 6: "<|im_start|>", 7: "\n", 8: "user"}
        return "".join(
            chr(int(token_id) - 1000) if int(token_id) >= 1000
            else special.get(int(token_id), "") for token_id in ids
        )


class _QwenProcessor:
    def __init__(self) -> None:
        self.tokenizer = _CharTokenizer()
        self.video_processor = SimpleNamespace(
            fps=4, temporal_patch_size=2, do_sample_frames=False,
            do_resize=False, do_rescale=False, do_normalize=False,
        )

    def apply_chat_template(self, messages, *, tokenize: bool = False) -> str:
        assert tokenize is False
        turns = []
        for message in messages:
            parts = ["V" * 7 if part["type"] == "video" else part["text"]
                     for part in message["content"]]
            turns.append(f"<|im_start|>{message['role']}\n" + "".join(parts) + "<|im_end|>")
        return "\n".join(turns)

    def __call__(self, *, text, videos, **kwargs):
        assert videos
        return {"input_ids": torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)}


class _VLLM:
    def __init__(self) -> None:
        self.prompts: list[dict] = []

    def generate(self, prompts, params, *, use_tqdm: bool = False):
        self.prompts.extend(prompts)
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=(), text="<traj_1>")])
                for _ in prompts]


def _position_ids(**kwargs):
    length = kwargs["input_ids"].shape[-1]
    return {"position_ids": torch.arange(length).repeat(3, 1).unsqueeze(1)}


def _real_engine(monkeypatch, *, max_seq_len: int):
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
        slowfast_tiers=None, predict_horizon=1, action_method="flat",
        data_processor=data_processor, processor=processor, tokenizer=processor.tokenizer,
        device=torch.device("cpu"),
    )
    vllm = _VLLM()
    engine = VLNInferenceEngine(
        bundle, backend="vllm_local", vllm_engine=vllm, enable_vit_cache=False,
    )
    return engine, processor, vllm


def test_real_processor_token_audit_and_capacity_fallback(monkeypatch) -> None:
    baseline_engine, processor, baseline_vllm = _real_engine(monkeypatch, max_seq_len=4096)
    baseline, _, _ = _policy(_Provider("no_memory"), engine=baseline_engine)
    _act(baseline, 0)
    _act(baseline, 1)
    baseline_ids = baseline_vllm.prompts[-1]["prompt_token_ids"]

    full_engine, _, full_vllm = _real_engine(monkeypatch, max_seq_len=4096)
    full, _, _ = _policy(_Provider("both"), engine=full_engine)
    _act(full, 0)
    _act(full, 1)
    full_ids = full_vllm.prompts[-1]["prompt_token_ids"]
    decoded = processor.tokenizer.decode(full_ids)
    assert "<navigation_task>find the chair</navigation_task>" in decoded
    assert "[Navigation Memory]" in decoded
    assert "Spatial history: chair" in decoded
    assert full.last_memory_diagnostic["actual_injection"] is True
    assert full.last_memory_diagnostic["model_input_diagnostic"][
        "vllm_prompt_matches_final_tokens"
    ] is True
    assert set(full.last_memory_diagnostic["injected_evidence_ids"]) == {
        *full.last_memory_diagnostic["selected_l1_fact_ids"],
        *full.last_memory_diagnostic["selected_l3_evidence_ids"],
    }
    assert len(full_ids) > len(baseline_ids)
    full_diagnostic = full.last_memory_diagnostic
    audited = full_diagnostic["model_input_diagnostic"]
    memory_tokens = len(full_ids) - len(baseline_ids)
    assert audited["baseline_final_token_count"] == len(baseline_ids)
    assert audited["injected_memory_tokens"] == memory_tokens
    assert full_diagnostic["memory_context_tokens"] == memory_tokens
    assert full_diagnostic["scene_query_episode_stats"]["actual_prompt_memory_tokens"] == memory_tokens
    assert full_diagnostic["scene_query_episode_stats"]["actual_prompt_total_tokens"] == sum(
        len(prompt["prompt_token_ids"]) for prompt in full_vllm.prompts
    )
    assert full_diagnostic["scene_query_episode_stats"]["injected_evidence_count"] == len(
        full_diagnostic["injected_evidence_ids"]
    )

    limited_engine, _, limited_vllm = _real_engine(
        monkeypatch, max_seq_len=len(baseline_ids),
    )
    limited, _, _ = _policy(_Provider("both"), engine=limited_engine)
    _act(limited, 0)
    _act(limited, 1)
    assert limited_vllm.prompts[-1]["prompt_token_ids"] == baseline_ids
    assert limited.last_memory_diagnostic["processor_length_rejections"] == 1
    assert len(limited.last_memory_diagnostic["removed_due_token_limit"]) == 1
    assert limited.last_memory_diagnostic["selected_l3_evidence_ids"] == []
    assert limited.last_memory_diagnostic["actual_injection"] is False
    assert limited.last_memory_diagnostic["memory_context_tokens"] == 0
    assert limited.last_memory_diagnostic["scene_query_episode_stats"][
        "actual_prompt_memory_tokens"
    ] == 0
    assert limited.last_memory_diagnostic["scene_query_episode_stats"][
        "actual_prompt_total_tokens"
    ] == sum(len(prompt["prompt_token_ids"]) for prompt in limited_vllm.prompts)
    assert limited.last_memory_diagnostic["scene_query_episode_stats"][
        "token_limit_removed_item_count"
    ] == 1
