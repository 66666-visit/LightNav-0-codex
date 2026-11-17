"""Model-derived token budget and whole-line selection."""

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
from lightnav.memory.schemas import DetectedObject, MemoryContext, SceneExtraction
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.token_budget import (
    TokenBudgetFailure,
    measure_memory_token_budget,
    select_context_lines,
)
from lightnav.memory.v3_schemas import EvidenceRecord, canonical_evidence_key


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return text.split()


class _DataProcessor:
    def __init__(self, base_tokens: int = 20) -> None:
        self.base_tokens = base_tokens
        self.model_max_length = 64
        self.tokenizer = _Tokenizer()
        self.calls: list[dict] = []

    def process_sample(self, sample, add_generation_prompt=False, validate_video_shapes=False):
        self.calls.append(dict(sample))
        ids = torch.zeros(self.base_tokens, dtype=torch.long)
        labels = torch.full((self.base_tokens,), -100, dtype=torch.long)
        return {"input_ids": ids, "labels": labels}


def test_measure_token_budget_uses_context_limit_base_prompt_and_generation() -> None:
    bundle = SimpleNamespace(
        max_seq_len=64, data_processor=_DataProcessor(20),
        tokenizer=_Tokenizer(), model=None, num_history_frames=4,
    )
    engine = SimpleNamespace(
        bundle=bundle, backend="hf", max_new_tokens=4,
    )
    budget = measure_memory_token_budget(engine, {"conversations": []})
    assert not isinstance(budget, TokenBudgetFailure)
    assert budget.context_limit_tokens == 64
    assert budget.base_prompt_tokens == 20
    assert budget.generation_reserve_tokens == 4
    assert budget.safety_margin_tokens == 1
    assert budget.available_tokens == 39
    assert budget.context_limit_source == "bundle.max_seq_len"
    assert budget.counter("a b c") == 3  # fake tokenizer splits on whitespace


def test_missing_processor_or_context_limit_is_reported_not_guessed() -> None:
    no_processor = measure_memory_token_budget(
        SimpleNamespace(bundle=SimpleNamespace(max_seq_len=64), max_new_tokens=4),
        {},
    )
    assert isinstance(no_processor, TokenBudgetFailure)
    assert no_processor.reason.startswith("processor_or_tokenizer_unavailable")

    data_processor = _DataProcessor()
    data_processor.model_max_length = None
    no_limit = measure_memory_token_budget(
        SimpleNamespace(
            bundle=SimpleNamespace(
                max_seq_len=None, data_processor=data_processor,
                tokenizer=_Tokenizer(), model=None,
            ),
            backend="hf", max_new_tokens=4,
        ),
        {},
    )
    assert isinstance(no_limit, TokenBudgetFailure)
    assert no_limit.reason.startswith("context_limit_unavailable")

    no_reserve = measure_memory_token_budget(
        SimpleNamespace(
            bundle=SimpleNamespace(
                max_seq_len=64, data_processor=_DataProcessor(),
                tokenizer=_Tokenizer(), model=None,
            ),
            backend="hf",
        ),
        {},
    )
    assert isinstance(no_reserve, TokenBudgetFailure)
    assert no_reserve.reason.startswith("generation_reserve_unavailable")


def _context() -> MemoryContext:
    lines = [
        "[Navigation Context - Test]",
        "Step 0; state line.",
        "Now: chair, conf 0.90, 1 sighting, last step 0.",
        "Earlier: table, last step 0, 1 sighting.",
    ]
    items = [
        _item(line, f"ev_{index}")
        for index, line in enumerate(lines[2:], start=1)
    ]
    return MemoryContext(text="\n".join(lines), layer_items=items)


def _item(line: str, evidence_id: str):
    from lightnav.memory.schemas import LayerMemoryItem, ObservationEvidence

    return LayerMemoryItem(
        layer="short", kind="test", description=line, step_id=0, confidence=0.9,
        source="test",
        evidence_refs=[ObservationEvidence(
            event_id=evidence_id, step_id=0, label="x", confidence=0.9, source="test",
        )],
        metadata={"serialized_text": line, "evidence_ids": [evidence_id]},
    )


def test_select_context_lines_keeps_only_a_whole_line_prefix() -> None:
    def count(text: str) -> int:
        return len(text.split())
    context = _context()
    selection = select_context_lines(context, capacity_tokens=20, count_tokens=count)
    assert selection.context is not None
    assert selection.context.text.startswith("[Navigation Context - Test]")
    assert selection.omitted_lines >= 1
    assert selection.omitted_items >= 1
    assert selection.block_tokens <= 20
    for item in selection.context.layer_items:
        assert item.metadata["serialized_text"] in selection.context.text

    none = select_context_lines(context, capacity_tokens=1, count_tokens=count)
    assert none.context is None
    assert none.omitted_items == len(context.layer_items)


# ---------------------------------------------------------------------------
# Policy integration with a model-derived budget
# ---------------------------------------------------------------------------


class _TokenEngine:
    def __init__(self) -> None:
        self.data_processor = _DataProcessor(20)
        self.bundle = SimpleNamespace(
            video_size=(4, 4), video_fps=4, num_history_frames=2,
            pool_enable=False, pool_spatial=1, pool_mode="avg",
            slowfast_tiers=None, predict_horizon=1, action_method="flat",
            max_seq_len=64, data_processor=self.data_processor,
            tokenizer=_Tokenizer(), model=None,
            device=torch.device("cpu"),
        )
        self.backend = "hf"
        self.max_new_tokens = 4
        self.samples: list[dict] = []
        self.last_input_diagnostic: dict | None = None
        self.last_prompt_chars = 0

    def reset_episode_state(self) -> None:
        pass

    def generate(self, sample: dict):
        self.samples.append(sample)
        prompt = sample["conversations"][0]["value"]
        trace = sample.get("_rgm_trace", {})
        memory_text = trace.get("memory_text", "")
        self.last_input_diagnostic = {
            "final_token_count": 20 + len(memory_text.split()),
            "memory_text_present_in_final_tokens": bool(
                memory_text and memory_text in prompt
            ),
            "model_visible_evidence_ids": [
                evidence_id for record in trace.get("accepted_records", [])
                for evidence_id in record["evidence_ids"]
            ] if memory_text else [],
        }
        self.last_prompt_chars = len(prompt)
        return "<traj_1>", 1.0


@pytest.fixture(autouse=True)
def _small_video(monkeypatch):
    monkeypatch.setattr(
        NavigationPolicy, "_convert_frame",
        lambda self, frame: torch.full((3, 4, 4), float(frame[0, 0, 0])),
    )


def _record(label: str, step: int = 0) -> EvidenceRecord:
    return EvidenceRecord(
        f"{label}_{step}", "scene", "episode", 0, step, "object_observation",
        semantic_label=label, raw_detector_confidence=0.95,
        detector_name="yolo", observation_status="observed",
    )


def test_policy_uses_token_budget_instead_of_the_fixed_320_char_cap() -> None:
    store_seed = _record("chair")
    assert canonical_evidence_key(
        "scene", "episode", 0, store_seed.evidence_id,
    )
    manager = MemoryManager(
        extractor=MockSceneExtractor({
            0: SceneExtraction(
                objects=["chair"],
                detections=[DetectedObject("chair", 0.95, (0.0, 0.0, 4.0, 4.0))],
                confidence=0.95, source="fake_l1",
            ),
        }),
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=SpatialL3Config(mode="off"),
        prompt_mode="l1_organized_only",
    )
    engine = _TokenEngine()
    centroids = np.zeros((3, 10, 3), dtype=np.float32)
    centroids[1, :, 0] = 0.25
    policy = TrajVocabVLNCEPolicy(
        engine, centroids, dt=1.0, lin_vel_range=(0.0, 0.3),
        ang_vel_range=(-30.0, 30.0), num_history_frames=2,
        memory_enabled=True, memory_manager=manager,
    )
    policy.reset(
        {"instruction": {"text": "find the chair"}},
        memory_episode_id="episode", memory_scene_id="scene",
    )
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    policy.act({"rgb": rgb}, {})
    diagnostic = policy.last_memory_diagnostic["prompt_context"]
    budget = diagnostic["budget"]
    assert engine.data_processor.calls, "the base prompt was measured"
    assert budget["budget_source"] == "token_aware_context"
    assert budget["context_limit_tokens"] == 64
    assert budget["base_prompt_tokens"] == 20
    assert budget["generation_reserve_tokens"] == 4
    assert budget["available_memory_tokens"] == 39
    assert diagnostic["budget_chars"] is None
    assert diagnostic["memory_tokens_injected"] is not None
    assert diagnostic["memory_tokens_injected"] <= 39
    assert len(policy.last_memory_text) > 0
