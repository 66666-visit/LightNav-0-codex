"""Show a scripted RGB place loop through the real MemoryManager interfaces.

This uses synthetic RGB, MockSceneExtractor and MockPlaceDescriptorExtractor.
It never loads DINOv2, YOLO, a LightNav checkpoint, or Habitat data.

Run: PYTHONPATH=src python3 scripts/demo_topological_memory.py
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_context_builder import build_context_text
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.place_recognition import MockPlaceDescriptorExtractor
from lightnav.memory.schemas import DetectedObject, ObservationPacket, SceneExtraction


def main() -> None:
    instruction = "Return to the chair, then find the couch"
    descriptors = MockPlaceDescriptorExtractor({
        0: (1.0, 0.0, 0.0),
        5: (0.0, 1.0, 0.0),
        10: (0.0, 0.0, 1.0),
        20: (1.0, 0.0, 0.0),
    })
    objects = MockSceneExtractor({
        0: SceneExtraction(detections=[
            DetectedObject("chair", 0.81), DetectedObject("chair", 0.69),
        ], confidence=0.81),
        5: SceneExtraction(detections=[DetectedObject("table", 0.78)], confidence=0.78),
        20: SceneExtraction(detections=[DetectedObject("couch", 0.88)], confidence=0.88),
    })
    manager = MemoryManager(
        extractor=objects, strategy="hierarchical",
        place_enabled=True, place_extractor=descriptors,
        place_every_n_steps=5, place_similarity_threshold=0.85,
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    manager.reset_episode("mock_visual_loop", instruction)
    try:
        for step in range(21):
            if step:
                manager.record_action({
                    "action": "velocity_control",
                    "action_args": {"linear_velocity": 0.2, "angular_velocity": 0.0},
                })
            manager.observe(ObservationPacket(step_id=step, rgb=frame, instruction=instruction))
            if step in {0, 5, 10, 20}:
                fused = manager.get_context()
                print(json.dumps({
                    "step": step,
                    "place_result": (
                        asdict(manager.last_place_result) if manager.last_place_result else None
                    ),
                    "place_nodes": [node.to_dict() for node in manager.topological_memory.nodes],
                    "place_edges": [edge.to_dict() for edge in manager.topological_memory.edges],
                    "layers": {
                        layer: [item.to_dict() for item in items]
                        for layer, items in manager.last_layer_results.items()
                    },
                    "fused_text": build_context_text(fused, max_chars=600),
                }, ensure_ascii=False, indent=2))
    finally:
        manager.close_episode()


if __name__ == "__main__":
    main()
