"""Opt-in local LightNav Processor acceptance for V4; no model weights run."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from lightnav.inference.engine import VitResult, VLNInferenceEngine
from lightnav.inference.model import (
    _build_data_processor,
    _make_bundle,
    _position_id_func_from_config,
    _resolve_processing_params,
    resolve_model_paths,
)
from lightnav.inference.samples import build_vln_traj_sample
from lightnav.inference.token_diagnostics import token_ids_sha256
from lightnav.memory.memory_context_builder import (
    audit_prompt_injection,
    build_context_text,
    inject_memory_context,
)
from lightnav.memory.v3_schemas import canonical_evidence_key
from lightnav.memory.v4_context import NavigationFact, NavigationMemoryContext
from lightnav.memory.v4_formatter import V4PromptFormatter
from lightnav.memory.v4_navigation_state import EvidenceProvenance
from lightnav.memory.v4_utilization import HISTORY_REFERENCE
from lightnav.processing import VLNQwen3VLProcessor


class _TextOnlyEngine(VLNInferenceEngine):
    def _vit_forward_from_inputs(self, inputs):
        self.vision_inputs = {
            key: value.clone() for key in ("pixel_values_videos", "video_grid_thw")
            if isinstance((value := inputs.get(key)), torch.Tensor)
        }
        prepared = self._prepare_inputs(inputs)
        grid = self.vision_inputs.get("video_grid_thw")
        return VitResult(
            prompt_ids=prepared["input_ids"][0].tolist(),
            video_embeds=torch.zeros((1, 4)) if grid is not None else None,
            video_grid_thw=grid,
            process_diagnostic=self.bundle.data_processor.last_process_diagnostic,
        )


class _FakeVLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompts, params, *, use_tqdm=False):
        self.prompts.extend(prompts)
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=(), text="<traj_1>")])
                for _ in prompts]


def test_local_checkpoint_processor_keeps_v4_text_and_original_vision(monkeypatch):
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        pytest.skip("set MODEL_PATH to an existing local LightNav checkpoint")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("VLN_EVAL_TRAJ_TOP1", "0")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        SamplingParams=lambda **kwargs: kwargs,
    ))

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
    instruction = "Find the uniquev4marker archway"
    base = build_vln_traj_sample(
        torch.zeros((2, 3, h, w), dtype=torch.float32), instruction, [0, 1], bundle,
    )
    evidence_id = canonical_evidence_key(
        "synthetic_scene", "synthetic_episode", 0, "observed_archway",
    )
    source = EvidenceProvenance(
        evidence_id=evidence_id, scene_id="synthetic_scene",
        episode_id="synthetic_episode", rollout_idx=0, step=0,
    )
    fact = NavigationFact(
        kind="object_observation", semantic_label="uniquev4marker",
        description="RGB detector reported uniquev4marker",
        evidence_ids=(evidence_id,), provenance=(source,),
        source_memory_id="synthetic_observation", temporal_role="HISTORICAL",
        observation_status="observed", visit_status="UNKNOWN",
        spatial_status="unknown", traversability_status="unknown",
        instruction_relevant=True, source="synthetic_rgb",
    )
    context = V4PromptFormatter().format(NavigationMemoryContext(
        mode=HISTORY_REFERENCE, selected_facts=(fact,),
    ))
    assert context is not None
    augmented = inject_memory_context(base, context, max_chars=400)
    audit = audit_prompt_injection(base, augmented, context, max_chars=400)
    assert audit["roundtrip_ok"]
    memory_text = build_context_text(context, max_chars=400)
    augmented["_rgm_trace"] = {
        "memory_backend": "rgm_v4", "memory_use_mode": HISTORY_REFERENCE,
        "episode_id": "synthetic_episode", "scene_id": "synthetic_scene",
        "rollout_idx": 0, "step_id": 1,
        "original_instruction": instruction, "memory_text": memory_text,
        "accepted_records": [{
            "evidence_ids": [evidence_id],
            "text": context.layer_items[0].metadata["serialized_text"],
        }],
        "original_prompt_sha256": audit["base_sha256"],
        "injected_prompt_sha256": audit["augmented_sha256"],
        "injected_memory_chars": len(memory_text),
        "evidence_provenance": context.layer_items[0].metadata["provenance"],
        "_v4_base_sample": base,
    }

    fake_vllm = _FakeVLLM()
    engine = _TextOnlyEngine(
        bundle, backend="vllm_local", vllm_engine=fake_vllm,
        enable_vit_cache=False,
    )
    engine.generate(base)
    baseline_ids = fake_vllm.prompts[-1]["prompt_token_ids"]
    baseline_vision = engine.vision_inputs
    baseline_payload = fake_vllm.prompts[-1]["multi_modal_data"]["video"]
    engine.generate(augmented)
    augmented_ids = fake_vllm.prompts[-1]["prompt_token_ids"]
    diagnostic = engine.last_input_diagnostic

    assert baseline_ids != augmented_ids
    assert diagnostic["baseline_final_token_ids_sha256"] == token_ids_sha256(baseline_ids)
    assert diagnostic["final_token_ids_sha256"] == token_ids_sha256(augmented_ids)
    assert diagnostic["vllm_prompt_token_ids_sha256"] == token_ids_sha256(augmented_ids)
    assert diagnostic["vllm_prompt_matches_final_tokens"] is True
    assert diagnostic["memory_text_present_in_final_tokens"] is True
    assert diagnostic["memory_text_present_in_vllm_prompt"] is True
    assert diagnostic["original_instruction_present_in_final_tokens"] is True
    assert diagnostic["assistant_generation_boundary_present_in_final_tokens"] is True
    assert diagnostic["model_visible_evidence_ids"] == [evidence_id]
    assert diagnostic["injected_memory_tokens"] > 0
    assert baseline_vision and baseline_vision.keys() == engine.vision_inputs.keys()
    assert all(torch.equal(baseline_vision[key], engine.vision_inputs[key])
               for key in baseline_vision)
    augmented_payload = fake_vllm.prompts[-1]["multi_modal_data"]["video"]
    assert all(torch.equal(baseline_payload[key], augmented_payload[key])
               for key in ("video_embeds", "video_grid_thw"))
