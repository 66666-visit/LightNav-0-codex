#!/usr/bin/env python3
"""Print the old/new L3 prompt comparison on a deterministic synthetic map.

This is a CPU-only demonstration of the serialization difference:
``SpatialGraphAdapter`` (original L3), ``L3PromptOrganizer`` (organized L3) and
``L3OfficialPromptOrganizer`` (official-style L3) read the same synthetic
``ObjectMap`` and the same robot pose.  It never runs Habitat or a model.

    PYTHONPATH=src python scripts/show_l3_prompt_comparison.py
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from lightnav.memory.conceptgraphs_lite import (
    CameraIntrinsics,
    ObjectMap,
    ObjectNode,
    ObservedFromEdge,
    Pose3D,
    RGBDFrame,
    SpatialEvidence,
    ViewpointNode,
)
from lightnav.memory.l3_official_prompt import L3OfficialPromptOrganizer
from lightnav.memory.l3_prompt_organizer import L3PromptOrganizer
from lightnav.memory.spatial_graph_adapter import SpatialGraphAdapter
from lightnav.memory.v3_schemas import canonical_evidence_key
from lightnav.memory.v3_store import V3MemoryStore

_IDENTITY = Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
_NO_FRAME = object()

CHAIR_LEFT = {
    "label": "chair", "center": (-1.0, 0.5, -2.0),
    "bbox": ((-1.3, 0.0, -2.3), (-0.7, 1.0, -1.7)),
}
CHAIR_RIGHT = {
    "label": "chair", "center": (3.5, 0.5, -3.0),
    "bbox": ((3.2, 0.0, -3.3), (3.8, 1.0, -2.7)),
}
CHAIR_MIDDLE = {
    "label": "chair", "center": (0.0, 0.5, -6.0),
    "bbox": ((-0.3, 0.0, -6.3), (0.3, 1.0, -5.7)),
}
DINING_TABLE = {
    "label": "dining table", "center": (0.5, 0.4, -2.5),
    "bbox": ((-0.2, 0.35, -3.2), (1.2, 0.75, -1.8)),
}


def _frame() -> RGBDFrame:
    return RGBDFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.full((8, 8), 2.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(8.0, 8.0, 3.5, 3.5, 8, 8),
        agent_pose=_IDENTITY, camera_extrinsics=_IDENTITY,
    )


def _build_map(objects: list[dict[str, Any]]) -> tuple[V3MemoryStore, ObjectMap]:
    scene, episode, rollout = "scene", "episode", 0
    store = V3MemoryStore()
    store.begin_episode(scene, episode, "find the chair")
    object_map = ObjectMap()
    object_map.reset_episode(scene, episode, rollout)
    for index, spec in enumerate(objects):
        step = int(spec.get("step", 7))
        label = str(spec["label"])
        center = tuple(float(value) for value in spec["center"])
        bbox = tuple(tuple(float(value) for value in point) for point in spec["bbox"])
        viewpoint_id = json.dumps(
            [scene, episode, rollout, step], separators=(",", ":"),
        )
        object_map.viewpoints.setdefault(viewpoint_id, ViewpointNode(
            viewpoint_id, scene, episode, rollout, step, _IDENTITY, _IDENTITY,
        ))
        evidence_id = canonical_evidence_key(
            scene, episode, rollout, f"l3_seg_{step}_{index}",
        )
        object_map.evidence[evidence_id] = SpatialEvidence(
            evidence_id, scene, episode, rollout, step, label,
            float(spec.get("confidence", 0.9)), "yolo11n-seg", 100, 100,
            "observed", None,
        )
        object_id = f"object_{index}"
        object_map.objects[object_id] = ObjectNode(
            object_id, label, center, bbox, 2, [evidence_id], "observed",
            np.empty((0, 3)),
        )
        object_map.edges.append(ObservedFromEdge(object_id, viewpoint_id, evidence_id))
    return store, object_map


def _cards(
    objects: list[dict[str, Any]], *, instruction: str, max_items: int = 2,
    step: int = 10, frame: Any = _NO_FRAME,
):
    store, object_map = _build_map(objects)
    actual_frame = _frame() if frame is _NO_FRAME else frame
    common = dict(
        scene_id="scene", episode_id="episode", rollout_idx=0, step=step,
        instruction=instruction, frame=actual_frame, scope="episode_local",
        min_confidence=0.5,
    )
    original = SpatialGraphAdapter().build(
        store, object_map, actual_frame, scene_id="scene", episode_id="episode",
        rollout_idx=0, step=step, instruction=instruction, scope="episode_local",
        max_landmarks=2, prompt_max_chars=320, min_confidence=0.5,
    ) if actual_frame is not None else None
    organized = L3PromptOrganizer(max_items=max_items).build(
        store, object_map, max_chars=2000, **common,
    )
    official = L3OfficialPromptOrganizer(max_items=max_items).build(
        store, object_map, max_chars=2000, **common,
    )
    return original, organized, official


def _show(title: str, cards: tuple[Any, Any, Any]) -> None:
    original, organized, official = cards
    print(f"### {title}\n")
    print("Original (`l3_original_only`):\n")
    print("```text")
    print(original.text if original else "<empty>")
    print("```\n")
    print("Organized (`l3_organized_only`):\n")
    print("```text")
    print(organized.context.text if organized.context else "<empty>")
    print("```\n")
    print("Official style (`l3_official_style_only`):\n")
    print("```text")
    print(official.context.text if official.context else "<empty>")
    print("```\n")
    print("Expression bookkeeping:\n")
    print("```json")
    print(json.dumps(official.expression, ensure_ascii=False, indent=2))
    print("```\n")


def main() -> int:
    _show(
        "A. Two remembered chairs; one is adjacent to a large dining table",
        _cards([CHAIR_LEFT, CHAIR_RIGHT, DINING_TABLE], instruction="find the chair"),
    )
    _show(
        "B. Three remembered chairs; stable left-to-right order",
        _cards(
            [CHAIR_LEFT, CHAIR_MIDDLE, CHAIR_RIGHT], instruction="find the chair",
            max_items=3,
        ),
    )
    _show(
        "C. A remembered chair with no robot pose for this step",
        _cards([CHAIR_LEFT], instruction="find the chair", frame=None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
