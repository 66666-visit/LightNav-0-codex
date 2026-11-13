"""Opt-in local-checkpoint Processor acceptance; no model weights or Habitat run."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from lightnav.inference.engine import VLNInferenceEngine, VitResult
from lightnav.inference.model import (
    _build_data_processor, _make_bundle, _position_id_func_from_config,
    _resolve_processing_params, resolve_model_paths,
)
from lightnav.inference.samples import build_vln_traj_sample
from lightnav.memory.memory_context_builder import (
    audit_prompt_injection, build_context_text, inject_memory_context,
)
from lightnav.memory.schemas import MemoryContext
from lightnav.memory.v3_formatter import V3PromptFormatter
from lightnav.memory.v3_retrieval import V3Retriever
from lightnav.memory.v3_schemas import EvidenceRecord
from lightnav.memory.v3_store import V3MemoryStore
from lightnav.processing import VLNQwen3VLProcessor


class _TextOnlyEngine(VLNInferenceEngine):
    """Exercise the real Processor and Engine prompt path without running ViT."""

    def _vit_forward_from_inputs(self, inputs):
        self.vision_inputs = {
            key: value.clone() for key in ("pixel_values_videos", "video_grid_thw")
            if isinstance((value := inputs.get(key)), torch.Tensor)
        }
        prepared = self._prepare_inputs(inputs)
        video_grid = self.vision_inputs.get("video_grid_thw")
        return VitResult(
            prompt_ids=prepared["input_ids"][0].tolist(),
            # A deterministic stand-in keeps the actual Engine -> vLLM
            # multimodal payload path observable without loading ViT weights.
            video_embeds=torch.zeros((1, 4)) if video_grid is not None else None,
            video_grid_thw=video_grid,
            process_diagnostic=self.bundle.data_processor.last_process_diagnostic,
        )


class _FakeVLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompts, params, *, use_tqdm=False):
        self.prompts.extend(prompts)
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=(), text="<traj_1>")])
                for _ in prompts]


def test_local_checkpoint_processor_keeps_memory_and_vision_inputs(monkeypatch):
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        pytest.skip("set MODEL_PATH to an existing local LightNav checkpoint")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("VLN_EVAL_TRAJ_TOP1", "0")
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
    )

    _, _, processor_dir = resolve_model_paths(model_path)
    processor = VLNQwen3VLProcessor.from_pretrained(
        processor_dir, padding_side="left", local_files_only=True,
    )
    params = _resolve_processing_params(model_path, task_type="vlnce")
    position_id_func = _position_id_func_from_config(model_path)
    data_processor = _build_data_processor(processor, params, position_id_func)
    bundle = _make_bundle(
        None, processor, data_processor, torch.device("cpu"), position_id_func, params,
    )
    h, w = bundle.video_size
    video = torch.zeros((2, 3, h, w), dtype=torch.float32)
    instruction = "Find the uniquev3marker archway"
    base = build_vln_traj_sample(video, instruction, [0, 1], bundle)

    store = V3MemoryStore(scope="episode_local")
    store.begin_episode("synthetic_scene", "synthetic_episode", instruction)
    record = store.add(EvidenceRecord(
        evidence_id="event_0", scene_id="synthetic_scene",
        episode_id="synthetic_episode", rollout_idx=0, step=0,
        event_type="object_observation", semantic_label="uniquev3marker",
        payload={"description": "Observed uniquev3marker archway"},
        raw_detector_confidence=0.88, observation_status="observed",
    ))
    retrieved = V3Retriever(store).retrieve(
        scene_id="synthetic_scene", instruction=instruction,
        current_step=1, episode_id="synthetic_episode", top_k=3,
    )
    assert retrieved.counts["historical_same_episode_prior_step"] >= 1
    context = V3PromptFormatter(max_records=3, max_chars=400).format(
        retrieved.records, current_step=1,
    )
    assert context.layer_items
    augmented = inject_memory_context(base, context, max_chars=400)
    assert audit_prompt_injection(base, augmented, context, max_chars=400)["roundtrip_ok"]
    augmented["_rgm_trace"] = {
        "episode_id": "synthetic_episode", "scene_id": "synthetic_scene",
        "rollout_idx": 0, "memory_backend": "rgm_v3", "step_id": 1,
        "original_instruction": instruction,
        "memory_text": build_context_text(context, max_chars=400),
        "accepted_records": [{
            "evidence_ids": [record.key],
            "text": context.layer_items[0].metadata["serialized_text"],
        }],
    }

    fake_vllm = _FakeVLLM()
    engine = _TextOnlyEngine(
        bundle, backend="vllm_local", vllm_engine=fake_vllm,
        enable_vit_cache=False,
    )
    base_text, _ = engine.generate(base)
    base_ids = fake_vllm.prompts[-1]["prompt_token_ids"]
    base_multimodal = fake_vllm.prompts[-1].get("multi_modal_data")
    base_vision = engine.vision_inputs
    empty_text, _ = engine.generate(inject_memory_context(
        base, MemoryContext(text="", layer_items=[]), max_chars=400,
    ))
    assert empty_text == base_text
    assert fake_vllm.prompts[-1]["prompt_token_ids"] == base_ids
    assert all(torch.equal(base_vision[key], engine.vision_inputs[key])
               for key in base_vision)
    augmented_text, _ = engine.generate(augmented)
    augmented_ids = fake_vllm.prompts[-1]["prompt_token_ids"]
    augmented_multimodal = fake_vllm.prompts[-1].get("multi_modal_data")
    augmented_vision = engine.vision_inputs
    diagnostic = engine.last_input_diagnostic

    assert base_text == augmented_text == "<traj_1>"
    assert base_ids != augmented_ids
    assert diagnostic["original_instruction_present_in_final_tokens"] is True
    assert diagnostic["assistant_generation_boundary_present_in_final_tokens"] is True
    assert diagnostic["memory_text_present_in_final_tokens"] is True
    assert diagnostic["memory_text_present_in_vllm_prompt"] is True
    assert diagnostic["vllm_prompt_matches_final_tokens"] is True
    assert record.key in diagnostic["literal_model_visible_evidence_ids"]
    assert base_vision and base_vision.keys() == augmented_vision.keys()
    assert all(torch.equal(base_vision[key], augmented_vision[key]) for key in base_vision)
    assert base_multimodal is not None and augmented_multimodal is not None
    assert all(torch.equal(base_multimodal["video"][key], augmented_multimodal["video"][key])
               for key in ("video_embeds", "video_grid_thw"))
    store.close_episode()
