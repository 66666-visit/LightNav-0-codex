"""Two-stage place decisions use local evidence and retain UNKNOWN."""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.schemas import VisualPlaceFeatures
from lightnav.memory.topological_memory import TopologicalMemory

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_CV2 = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(_HAS_NUMPY, "NumPy is required")
class RGMPlaceTests(unittest.TestCase):
    def test_initial_reference_and_global_ambiguity(self) -> None:
        import numpy as np

        from lightnav.memory.place_verification import TwoStagePlaceVerifier

        cfg = RGMConfig(enabled=True, place_two_stage_enabled=True)
        verifier = TwoStagePlaceVerifier(cfg)
        features = VisualPlaceFeatures((1.0, 0.0), np.eye(8, dtype=np.float16),
                                       (2, 4), tuple(range(8)), {"image_quality": 1.0})
        topology = TopologicalMemory()
        first = verifier.decide(0, features, topology)
        self.assertEqual(first.status, "NEW")
        node = topology.observe(0, features.global_descriptor, [],
                                descriptor_source="mock_dino", observation_event_id="place_0")
        verifier.accept(node.place_id, features)
        topology.observe(5, features.global_descriptor, [],
                         descriptor_source="mock_dino", observation_event_id="place_5",
                         force_new=True)
        ambiguous = verifier.decide(10, features, topology)
        self.assertEqual(ambiguous.status, "UNKNOWN")
        self.assertEqual(ambiguous.reason, "unresolved_competing_candidate")
        self.assertEqual(len(topology.nodes), 2)

    def test_top_k_local_check_can_rescue_second_candidate(self) -> None:
        import numpy as np

        from lightnav.memory.place_verification import LocalCheck, TwoStagePlaceVerifier

        verifier = TwoStagePlaceVerifier(RGMConfig(enabled=True, place_two_stage_enabled=True))
        topology = TopologicalMemory()
        frames = [VisualPlaceFeatures(vector, np.eye(8, dtype=np.float16),
                                      (2, 4), tuple(range(8)), {"image_quality": 1.0})
                  for vector in ((1.0, 0.0), (0.8, 0.6), (0.98, 0.2))]
        first = topology.observe(0, frames[0].global_descriptor, [],
                                 descriptor_source="mock_dino", observation_event_id="place_0")
        second = topology.observe(5, frames[1].global_descriptor, [],
                                  descriptor_source="mock_dino", observation_event_id="place_5",
                                  force_new=True)
        verifier.accept(first.place_id, frames[0])
        verifier.accept(second.place_id, frames[1])
        with patch("lightnav.memory.place_verification.local_check", side_effect=[
            LocalCheck(8, 0.10, "local_geometry_inconsistent"),
            LocalCheck(9, 0.80, "local_consistent"),
        ]) as mocked:
            decision = verifier.decide(10, frames[2], topology)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(decision.status, "MATCH")
        self.assertEqual(decision.candidate_place_id, second.place_id)
        self.assertEqual(verifier.last_topk[0]["local_reason"], "local_geometry_inconsistent")
        self.assertEqual(verifier.last_topk[1]["local_reason"], "local_consistent")

    def test_two_locally_supported_candidates_are_ambiguous(self) -> None:
        import numpy as np

        from lightnav.memory.place_verification import LocalCheck, TwoStagePlaceVerifier

        verifier = TwoStagePlaceVerifier(RGMConfig(enabled=True, place_two_stage_enabled=True))
        features = VisualPlaceFeatures((1.0, 0.0), np.eye(8, dtype=np.float16),
                                       (2, 4), tuple(range(8)), {"image_quality": 1.0})
        topology = TopologicalMemory()
        for step in (0, 5):
            node = topology.observe(step, features.global_descriptor, [],
                                    descriptor_source="mock_dino",
                                    observation_event_id=f"place_{step}", force_new=step > 0)
            verifier.accept(node.place_id, features)
        with patch("lightnav.memory.place_verification.local_check",
                   return_value=LocalCheck(8, 0.75, "local_consistent")):
            decision = verifier.decide(10, features, topology)
        self.assertEqual(decision.status, "UNKNOWN")
        self.assertEqual(decision.reason, "ambiguous_local_candidates")

    def test_global_match_needs_local_consistency(self) -> None:
        import numpy as np

        from lightnav.memory.place_verification import LocalCheck, TwoStagePlaceVerifier

        verifier = TwoStagePlaceVerifier(RGMConfig(enabled=True, place_two_stage_enabled=True))
        features = VisualPlaceFeatures((1.0, 0.0), np.eye(8, dtype=np.float16),
                                       (2, 4), tuple(range(8)), {"image_quality": 1.0})
        topology = TopologicalMemory()
        node = topology.observe(0, features.global_descriptor, [],
                                descriptor_source="mock_dino", observation_event_id="place_0")
        verifier.accept(node.place_id, features)
        with patch("lightnav.memory.place_verification.local_check",
                   return_value=LocalCheck(8, None, "opencv_unavailable")):
            self.assertEqual(verifier.decide(5, features, topology).status, "UNKNOWN")
        with patch("lightnav.memory.place_verification.local_check",
                   return_value=LocalCheck(8, 0.75, "local_consistent")):
            decision = verifier.decide(5, features, topology)
        self.assertEqual(decision.status, "MATCH")
        self.assertEqual(decision.candidate_place_id, node.place_id)

    @unittest.skipUnless(_HAS_CV2, "OpenCV is required for geometry consistency")
    def test_local_mutual_matches_have_geometric_inliers(self) -> None:
        import numpy as np

        from lightnav.memory.place_verification import local_check

        features = VisualPlaceFeatures(
            (1.0, 0.0), np.eye(16, dtype=np.float16),
            (4, 4), tuple(range(16)), {"image_quality": 1.0},
        )
        check = local_check(features, features, RGMConfig())
        self.assertEqual(check.match_count, 16)
        self.assertIsNotNone(check.inlier_ratio)
        self.assertGreaterEqual(check.inlier_ratio, 0.9)

    def test_keyframes_remain_bounded_and_keep_initial_reference(self) -> None:
        import numpy as np

        from lightnav.memory.place_verification import TwoStagePlaceVerifier

        verifier = TwoStagePlaceVerifier(RGMConfig(place_keyframes_max=2))
        frames = [VisualPlaceFeatures(
            vector, np.eye(8, dtype=np.float16), (2, 4),
            tuple(range(8)), {"image_quality": 1.0},
        ) for vector in ((1.0, 0.0), (0.9, 0.1), (0.0, 1.0))]
        for frame in frames:
            verifier.accept("place_000", frame)
        self.assertEqual(len(verifier._keyframes["place_000"]), 2)
        self.assertIs(verifier._keyframes["place_000"][0], frames[0])
        self.assertIs(verifier._keyframes["place_000"][1], frames[2])


if __name__ == "__main__":
    unittest.main()
