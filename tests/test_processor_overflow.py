"""Inference must reject a token limit that would silently cut the task suffix."""

from __future__ import annotations

import pytest
import torch

from lightnav.data_processor import Qwen3VLDataProcessor


def test_overlong_processor_input_is_rejected_before_mutation() -> None:
    processor = object.__new__(Qwen3VLDataProcessor)
    processor.model_max_length = 4
    ids = torch.tensor([11, 12, 13, 14, 15, 16])
    data = {
        "input_ids": ids.clone(),
        "attention_mask": torch.ones_like(ids),
        "labels": torch.full_like(ids, -100),
        "position_ids": torch.zeros(3, len(ids), dtype=torch.long),
    }

    with pytest.raises(ValueError, match="input length 6 exceeds model_max_length 4"):
        processor._apply_max_length(data, strict_prompt=True)

    assert torch.equal(data["input_ids"], ids)
    assert data["position_ids"].shape == (3, 6)


def test_processor_input_at_limit_is_unchanged() -> None:
    processor = object.__new__(Qwen3VLDataProcessor)
    processor.model_max_length = 4
    ids = torch.tensor([11, 12, 13, 14])
    data = {"input_ids": ids}

    assert processor._apply_max_length(data) is data
    assert torch.equal(data["input_ids"], ids)


def test_untraced_baseline_keeps_legacy_truncation_with_diagnostic(caplog) -> None:
    processor = object.__new__(Qwen3VLDataProcessor)
    processor.model_max_length = 4
    processor.image_token_id = 1
    processor.video_token_id = 2
    processor.last_process_diagnostic = {"truncated": False, "length_rejected": False}
    ids = torch.tensor([11, 12, 13, 14, 15, 16])
    data = {
        "input_ids": ids.clone(),
        "attention_mask": torch.ones_like(ids),
        "labels": torch.full_like(ids, -100),
        "position_ids": torch.zeros(3, len(ids), dtype=torch.long),
    }
    output = processor._apply_max_length(data, strict_prompt=False)
    assert output["input_ids"].tolist() == [11, 12, 13, 14]
    assert output["position_ids"].shape == (3, 4)
    assert processor.last_process_diagnostic["truncated"] is True
    assert "processor truncated prompt" in caplog.text
