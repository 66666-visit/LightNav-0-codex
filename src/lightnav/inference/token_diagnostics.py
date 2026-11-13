"""Inspect final prompt token IDs without storing decoded navigation prompts."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any


def token_ids_sha256(token_ids: list[int]) -> str:
    """Hash a stable, unambiguous serialization of the final token sequence."""
    payload = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return sha256(payload).hexdigest()


def memory_token_checks(tokenizer: Any, token_ids: list[int], trace: dict) -> dict:
    """Check the actual token sequence without retaining the decoded prompt."""
    memory_text = trace.get("memory_text") or ""
    records = trace.get("accepted_records") or []
    instruction = trace.get("original_instruction")
    decoded = None
    try:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        pass
    # The memory must be at the injection point after the navigation task.
    # A copy of the same words elsewhere in the user instruction is not proof
    # that the Processor kept the injected block.
    block_present = bool(
        memory_text and decoded is not None and any(
            f"</navigation_task>{separator}{memory_text}\n" in decoded
            for separator in (". ", " ")
        )
    )
    try:
        assistant_prefix = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False,
        )
        answer_boundary = bool(assistant_prefix) and token_ids[-len(assistant_prefix):] == assistant_prefix
    except Exception:
        answer_boundary = None
    checked_records = []
    visible_ids: list[str] = []
    literal_visible_ids: list[str] = []
    for record in records:
        record_text = record.get("text") or ""
        present = bool(block_present and record_text and record_text in memory_text)
        evidence_ids = [str(value) for value in record.get("evidence_ids", [])]
        checked_records.append({
            "evidence_ids": evidence_ids,
            "text_sha256": sha256(record_text.encode("utf-8")).hexdigest(),
            "present_in_final_tokens": present if decoded is not None else None,
            "literal_evidence_ids_in_final_tokens": [
                evidence_id for evidence_id in evidence_ids
                if present and evidence_id in record_text
            ],
        })
        if present:
            visible_ids.extend(evidence_ids)
            literal_visible_ids.extend(
                evidence_id for evidence_id in evidence_ids if evidence_id in record_text
            )
    return {
        "instruction_text_present": (
            f"<navigation_task>{instruction}</navigation_task>" in decoded
            if isinstance(instruction, str) and decoded is not None else None
        ),
        "assistant_generation_boundary_present": answer_boundary,
        "memory_text_present": (
            block_present if memory_text and decoded is not None
            else False if not memory_text else None
        ),
        "accepted_records": checked_records,
        "model_visible_evidence_ids": list(dict.fromkeys(visible_ids)),
        "literal_model_visible_evidence_ids": list(dict.fromkeys(literal_visible_ids)),
    }
