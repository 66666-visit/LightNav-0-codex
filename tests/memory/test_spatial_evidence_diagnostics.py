"""Read-only diagnostics for optional L1 links on independent L3 masks."""

from __future__ import annotations

import numpy as np
import pytest

from lightnav.memory.schemas import MemoryEvent
from lightnav.memory.spatial_graph_adapter import make_spatial_evidence
from lightnav.memory.v3_schemas import EvidenceRecord, canonical_evidence_key
from lightnav.memory.v3_store import V3MemoryStore
from lightnav.memory.yolo_seg_extractor import SegmentationInstance

_REASONS = (
    "unmatched_no_same_class",
    "unmatched_invalid_l1_bbox",
    "unmatched_low_iou",
    "unmatched_low_mask_coverage",
    "unmatched_invalid_l1_evidence",
)


def _make(*, label: str = "chair", bbox: object = (0, 0, 8, 8),
          persist_l1: bool = False) -> tuple[list, dict[str, int]]:
    store = V3MemoryStore()
    store.begin_episode("scene", "episode", "find the chair")
    if persist_l1:
        store.add(EvidenceRecord(
            evidence_id="l1_0", scene_id="scene", episode_id="episode",
            rollout_idx=0, step=0, event_type="object_observation",
            semantic_label=label, observation_status="observed",
        ))
    event = MemoryEvent(
        event_id="l1_0", step_id=0, kind="object_observation",
        description=f"Observed {label}", confidence=0.9, source="mock_l1",
        metadata={"bbox_xyxy": bbox} if bbox is not None else {},
        episode_id="episode",
    )
    segment = SegmentationInstance(
        "chair", 0.9, np.ones((8, 8), dtype=np.bool_),
        (0.0, 0.0, 8.0, 8.0),
    )
    return make_spatial_evidence(
        [segment], [event], store,
        scene_id="scene", episode_id="episode", rollout_idx=0, step=0,
        source="mock_l3", min_iou=0.5, ambiguity_margin=0.1,
    )


@pytest.mark.parametrize(("label", "bbox", "expected"), [
    ("sofa", (0, 0, 8, 8), "unmatched_no_same_class"),
    ("chair", None, "unmatched_invalid_l1_bbox"),
    ("chair", (0, 0, 2, 2), "unmatched_low_iou"),
    ("chair", (0, 0, 8, 5), "unmatched_low_mask_coverage"),
    ("chair", (0, 0, 8, 8), "unmatched_invalid_l1_evidence"),
])
def test_one_unmatched_mask_has_one_reason_without_changing_minting(
    label: str, bbox: object, expected: str,
) -> None:
    masks, counts = _make(label=label, bbox=bbox)
    assert counts["minted"] == counts["unmatched"] == 1
    assert counts["linked"] == counts["ambiguous"] == 0
    assert sum(counts[reason] for reason in _REASONS) == counts["unmatched"]
    assert counts[expected] == 1
    assert masks[0].linked_l1_evidence_id is None


def test_valid_l1_link_preserves_existing_evidence_identity() -> None:
    masks, counts = _make(persist_l1=True)
    assert counts["minted"] == counts["linked"] == 1
    assert counts["unmatched"] == 0
    assert sum(counts[reason] for reason in _REASONS) == 0
    assert masks[0].linked_l1_evidence_id == canonical_evidence_key(
        "scene", "episode", 0, "l1_0",
    )

