"""RGM Manager abstains from unsupported topology updates in a fake RGB episode."""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(_HAS_NUMPY, "NumPy is required")
class RGMManagerTests(unittest.TestCase):
    def test_rotation_and_stop_are_not_rgm_forward_motion(self) -> None:
        from lightnav.memory.memory_writer import MemoryWriter

        def command(linear: float, angular: float) -> dict:
            return {"action": "velocity_control", "action_args": {
                "linear_velocity": linear, "angular_velocity": angular,
            }}

        self.assertFalse(MemoryWriter._has_positive_forward_command(command(0.0, 0.5)))
        self.assertFalse(MemoryWriter._has_positive_forward_command(command(-1.0, 0.0)))
        self.assertTrue(MemoryWriter._has_positive_forward_command(command(0.2, 0.0)))

    def test_unknown_place_keeps_one_reference_and_no_transition(self) -> None:
        import numpy as np

        from lightnav.memory.manager import MemoryManager
        from lightnav.memory.rgm_config import RGMConfig
        from lightnav.memory.schemas import ObservationPacket, VisualPlaceFeatures

        class FakeDINO:
            source = "fake_dino"

            def __init__(self) -> None:
                self.calls = []

            def reset_episode(self, episode_id, instruction) -> None:
                self.calls.clear()

            def close_episode(self) -> None:
                pass

            def extract_features(self, packet):
                self.calls.append(packet.step_id)
                vector = (1.0, 0.0) if packet.step_id == 0 else (0.0, 1.0)
                return VisualPlaceFeatures(
                    vector, np.eye(8, dtype=np.float16), (2, 4),
                    tuple(range(8)), {"image_quality": 1.0},
                )

        dino = FakeDINO()
        manager = MemoryManager(
            strategy="hierarchical", place_enabled=True, place_extractor=dino,
            rgm_config=RGMConfig(
                enabled=True, yolo_temporal_enabled=True,
                place_two_stage_enabled=True, topology_motion_validation_enabled=True,
            ),
        )
        manager.reset_episode("episode_1", "Find sofa")
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        for step in range(6):
            action = ({"action": "velocity_control",
                       "action_args": {"linear_velocity": 0.5}}
                      if step == 5 else None)
            manager.observe(ObservationPacket(step, rgb, "Find sofa", action))
        self.assertEqual(dino.calls, [0, 5])
        self.assertEqual(manager.last_place_decision.status, "UNKNOWN")
        self.assertEqual(len(manager.topological_memory.nodes), 1)
        self.assertEqual(len(manager.topological_memory.edges), 0)
        self.assertEqual(manager.last_topology_edge_status, "NONE")
        manager.close_episode()
        self.assertEqual(manager.topological_memory.nodes, [])
        self.assertEqual(manager.place_verifier.last_topk, [])
        self.assertIsNone(manager.write_gate)
        self.assertEqual(manager.prompt_gate.last_selection.injected_text, "")

    def test_rotation_and_rgb_shift_do_not_create_rgm_topology_edge(self) -> None:
        import numpy as np

        from lightnav.memory.manager import MemoryManager
        from lightnav.memory.place_verification import LocalCheck
        from lightnav.memory.rgm_config import RGMConfig
        from lightnav.memory.schemas import ObservationPacket, VisualPlaceFeatures

        class FakeDINO:
            source = "fake_dino"

            def reset_episode(self, episode_id, instruction) -> None:
                pass

            def close_episode(self) -> None:
                pass

            def extract_features(self, packet):
                vector = (1.0, 0.0) if packet.step_id == 0 else (0.0, 1.0)
                return VisualPlaceFeatures(
                    vector, np.eye(8, dtype=np.float16), (2, 4),
                    tuple(range(8)), {"image_quality": 1.0},
                )

        manager = MemoryManager(
            strategy="hierarchical", place_enabled=True, place_extractor=FakeDINO(),
            rgm_config=RGMConfig(enabled=True, place_two_stage_enabled=True,
                                 topology_motion_validation_enabled=True),
        )
        manager.reset_episode("episode_rotation", "Find sofa")
        with patch("lightnav.memory.place_verification.local_check",
                   return_value=LocalCheck(8, 0.9, "local_consistent")):
            for step in range(11):
                rgb = np.full((8, 8, 3), 255 if step == 10 else 0, dtype=np.uint8)
                action = ({"action": "velocity_control", "action_args": {
                    "linear_velocity": 0.0, "angular_velocity": 0.5,
                }} if step == 10 else None)
                manager.observe(ObservationPacket(step, rgb, "Find sofa", action))
        self.assertEqual(manager.last_place_decision.status, "NEW")
        self.assertEqual(len(manager.topological_memory.nodes), 2)
        self.assertEqual(manager.topological_memory.edges, [])
        self.assertEqual(manager.last_topology_edge_status, "VISUAL_SHIFT_CANDIDATE")
        manager.close_episode()


if __name__ == "__main__":
    unittest.main()
