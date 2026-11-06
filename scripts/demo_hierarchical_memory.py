"""Print a CPU Mock episode's three memory layers and fused navigation text.

Run in the inference environment with ``PYTHONPATH=src python3
scripts/demo_hierarchical_memory.py``. No checkpoint, Habitat data, or RGB files
are used. The scripted detections are tests of memory plumbing, not YOLO output.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from lightnav.memory.manager import MemoryManager
from lightnav.memory.memory_writer import MockSceneExtractor
from lightnav.memory.schemas import DetectedObject, ObservationPacket, SceneExtraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers", default="short,mid,long",
        help="Comma-separated hierarchy layers, e.g. mid,long or short",
    )
    args = parser.parse_args()
    layers = () if args.layers.strip().casefold() == "none" else tuple(
        part.strip() for part in args.layers.split(",")
    )
    instruction = "Find the couch seen earlier, then continue down the hall"
    extractor = MockSceneExtractor({
        5: SceneExtraction(
            detections=[DetectedObject("couch", 0.82, (2, 3, 12, 18))],
            confidence=0.82, source="scripted_mock",
        ),
        10: SceneExtraction(
            detections=[DetectedObject("chair", 0.61, (18, 2, 28, 17))],
            confidence=0.61, source="scripted_mock",
        ),
    })
    manager = MemoryManager(extractor=extractor, strategy="hierarchical", layers=layers)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    manager.reset_episode("mock_hierarchy", instruction)
    try:
        for step in range(21):
            if step:
                manager.record_action({
                    "action": "velocity_control",
                    "action_args": {"linear_velocity": 0.2, "angular_velocity": 0.0},
                })
            manager.observe(ObservationPacket(step_id=step, rgb=frame, instruction=instruction))
            if step in {5, 20}:
                fused = manager.get_context()
                print(json.dumps({
                    "step": step,
                    "layers": {
                        layer: [item.to_dict() for item in items]
                        for layer, items in manager.last_layer_results.items()
                    },
                    "fused_text": fused.text,
                    "fused_records": [item.to_dict() for item in fused.layer_items],
                }, ensure_ascii=False, indent=2))
    finally:
        manager.close_episode()


if __name__ == "__main__":
    main()
