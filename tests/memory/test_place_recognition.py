"""Local DINOv2 descriptor contract without downloading or loading real weights."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lightnav.memory.place_recognition import (
    DINOv2PlaceDescriptorExtractor,
    MockPlaceDescriptorExtractor,
    normalize_descriptor,
)
from lightnav.memory.schemas import ObservationPacket

_HAS_VISION_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "PIL", "torch")
)


def _fake_local_directory(path: Path) -> None:
    (path / "config.json").write_text(json.dumps({"model_type": "dinov2"}))
    (path / "preprocessor_config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"fake weights: injected factory never reads this")


class PlaceRecognitionTest(unittest.TestCase):
    def test_normalization_and_rejection_of_unusable_vectors(self) -> None:
        self.assertEqual(normalize_descriptor((3.0, 4.0)), (0.6, 0.8))
        for values in ((), (0.0, 0.0), (float("nan"), 1.0), (float("inf"), 1.0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_descriptor(values)

    def test_mock_returns_step_specific_normalized_descriptor_or_none(self) -> None:
        extractor = MockPlaceDescriptorExtractor({0: (3.0, 4.0), 5: (0.0, 2.0)})
        extractor.reset_episode("ep", "find couch")
        self.assertEqual(extractor.extract(ObservationPacket(step_id=0)), (0.6, 0.8))
        self.assertIsNone(extractor.extract(ObservationPacket(step_id=1)))
        self.assertEqual(extractor.extract(ObservationPacket(step_id=5)), (0.0, 1.0))
        self.assertEqual(extractor.calls, [0, 1, 5])
        extractor.close_episode()
        self.assertEqual(extractor.calls, [])

    def test_missing_local_model_or_weights_fails_before_model_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "directory not found"):
                DINOv2PlaceDescriptorExtractor(str(root / "missing"))
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                DINOv2PlaceDescriptorExtractor(str(root))
            (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "preprocessor_config.json"):
                DINOv2PlaceDescriptorExtractor(str(root))
            (root / "preprocessor_config.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "weights not found"):
                DINOv2PlaceDescriptorExtractor(str(root))

    def test_device_is_checked_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_local_directory(root)
            for device in (None, "cuda:-1", "other"):
                with self.subTest(device=device), self.assertRaisesRegex(ValueError, "device"):
                    DINOv2PlaceDescriptorExtractor(str(root), device=device)

    @unittest.skipUnless(_HAS_VISION_DEPS, "NumPy, Pillow and torch are required")
    def test_rgb_patch_pooling_dimension_normalization_lazy_load_and_local_only(self) -> None:
        import numpy as np
        import torch

        class FakeProcessor:
            def __init__(self) -> None:
                self.pixels: list[tuple[int, int, int]] = []

            def __call__(self, *, images, return_tensors):
                self.pixels.append(images.getpixel((0, 0)))
                self.assert_return_tensors = return_tensors
                return {"pixel_values": torch.zeros((1, 3, 8, 8))}

        class FakeModel:
            config = SimpleNamespace(model_type="dinov2", hidden_size=384, num_register_tokens=0)

            def __init__(self) -> None:
                self.call_count = 0
                self.inference_mode: list[bool] = []
                self.device: str | None = None
                self.eval_called = False

            def to(self, device):
                self.device = device
                return self

            def eval(self):
                self.eval_called = True
                return self

            def __call__(self, **inputs):
                self.call_count += 1
                self.inference_mode.append(torch.is_inference_mode_enabled())
                self.assert_pixel_values = inputs["pixel_values"]
                tokens = torch.zeros((1, 3, 384))
                tokens[0, 0, 0] = 100.0  # CLS must not contribute to the patch pool.
                tokens[0, 1, 0:2] = torch.tensor((3.0, 4.0))
                tokens[0, 2, 0:2] = torch.tensor((9.0, 12.0))
                return SimpleNamespace(last_hidden_state=tokens)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_local_directory(root)
            processor, model = FakeProcessor(), FakeModel()
            loads: list[tuple[str, str, bool]] = []

            def processor_factory(path: str, *, local_files_only: bool):
                loads.append(("processor", path, local_files_only))
                return processor

            def model_factory(path: str, *, local_files_only: bool):
                loads.append(("model", path, local_files_only))
                return model

            extractor = DINOv2PlaceDescriptorExtractor(
                str(root), processor_factory=processor_factory, model_factory=model_factory,
            )
            self.assertFalse(extractor.loaded)
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            rgb[0, 0] = (220, 11, 30)
            packet = ObservationPacket(step_id=0, rgb=rgb)
            first = extractor.extract(packet)
            second = extractor.extract(packet)

            self.assertTrue(extractor.loaded)
            self.assertEqual(len(first), 384)
            self.assertTrue(math.isclose(math.sqrt(sum(x * x for x in first)), 1.0))
            self.assertAlmostEqual(first[0], 0.6)
            self.assertAlmostEqual(first[1], 0.8)
            self.assertTrue(all(value == 0.0 for value in first[2:]))
            self.assertEqual(first, second)
            self.assertEqual(processor.pixels, [(220, 11, 30), (220, 11, 30)])
            self.assertEqual(processor.assert_return_tensors, "pt")
            self.assertEqual(loads, [
                ("processor", str(root), True), ("model", str(root), True),
            ])
            self.assertEqual(model.call_count, 2)
            self.assertEqual(model.inference_mode, [True, True])
            self.assertEqual(model.device, "cpu")
            self.assertTrue(model.eval_called)

    @unittest.skipUnless(_HAS_VISION_DEPS, "NumPy, Pillow and torch are required")
    def test_two_stage_features_use_one_forward_and_bounded_patch_grid(self) -> None:
        import numpy as np
        import torch

        class FakeProcessor:
            def __call__(self, *, images, return_tensors):
                return {"pixel_values": torch.zeros((1, 3, 28, 28))}

        class FakeModel:
            config = SimpleNamespace(
                model_type="dinov2", hidden_size=384, num_register_tokens=0,
                patch_size=14,
            )

            def __init__(self):
                self.calls = 0

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, **inputs):
                self.calls += 1
                tokens = torch.ones((1, 5, 384))
                tokens[:, 0, :] = 100.0
                return SimpleNamespace(last_hidden_state=tokens)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_local_directory(root)
            model = FakeModel()
            extractor = DINOv2PlaceDescriptorExtractor(
                str(root), processor_factory=lambda *args, **kwargs: FakeProcessor(),
                model_factory=lambda *args, **kwargs: model,
            )
            features = extractor.extract_features(ObservationPacket(
                step_id=0, rgb=np.arange(28 * 28 * 3, dtype=np.uint32).reshape(28, 28, 3)
                .astype(np.uint8),
            ))
            self.assertEqual(model.calls, 1)
            self.assertEqual(features.patch_grid, (2, 2))
            self.assertEqual(features.patch_descriptors.shape, (4, 384))
            self.assertEqual(features.patch_indices, (0, 1, 2, 3))
            self.assertEqual(len(features.global_descriptor), 384)

    @unittest.skipUnless(_HAS_VISION_DEPS, "NumPy, Pillow and torch are required")
    def test_bad_rgb_rejected_without_loading_model(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fake_local_directory(root)
            extractor = DINOv2PlaceDescriptorExtractor(str(root))
            for rgb in (
                np.zeros((8, 8), dtype=np.uint8),
                np.zeros((8, 8, 3), dtype=np.float32),
                np.zeros((0, 8, 3), dtype=np.uint8),
            ):
                with self.subTest(shape=rgb.shape), self.assertRaises(ValueError):
                    extractor.extract(ObservationPacket(step_id=0, rgb=rgb))
            self.assertFalse(extractor.loaded)


if __name__ == "__main__":
    unittest.main()
