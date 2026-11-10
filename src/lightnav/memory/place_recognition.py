"""Local DINOv2-Small RGB place descriptors, separate from object detection.

The descriptor is a normalized mean of patch tokens. It describes an observed
image, not a room label, position, traversable route, or physical identity.
Imports and model loading are deferred until an enabled caller requests a real
descriptor; no model or image is downloaded or saved by this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from lightnav.memory.schemas import ObservationPacket, VisualPlaceFeatures


def normalize_descriptor(values: Sequence[float]) -> tuple[float, ...]:
    """Return a finite, unit-length descriptor, rejecting missing evidence."""
    if len(values) == 0:
        raise ValueError("place descriptor must have at least one component")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("place descriptor components must be finite")
    length = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("place descriptor must have nonzero finite norm")
    return tuple(value / length for value in vector)


class PlaceDescriptorExtractor(Protocol):
    """Episode-aware adapter that extracts one descriptor from the current RGB."""

    source: str

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        ...

    def extract(self, packet: ObservationPacket) -> tuple[float, ...] | None:
        ...

    def close_episode(self) -> None:
        ...


class MockPlaceDescriptorExtractor:
    """Scripted descriptors for tests and diagnostics; no visual inference."""

    source = "mock_place_descriptor"

    def __init__(self, by_step: Mapping[int, Sequence[float]] | None = None) -> None:
        self.by_step = dict(by_step or {})
        self.calls: list[int] = []

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        self.calls.clear()

    def extract(self, packet: ObservationPacket) -> tuple[float, ...] | None:
        self.calls.append(packet.step_id)
        values = self.by_step.get(packet.step_id)
        return normalize_descriptor(values) if values is not None else None

    def close_episode(self) -> None:
        self.calls.clear()


class DINOv2PlaceDescriptorExtractor:
    """Read a local Hugging Face DINOv2-Small directory on first inference."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        *,
        processor_factory: Callable[..., Any] | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"local DINOv2 model directory not found: {path}")
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"local DINOv2 config.json not found: {path}")
        if not (path / "preprocessor_config.json").is_file():
            raise FileNotFoundError(f"local DINOv2 preprocessor_config.json not found: {path}")
        if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
            raise FileNotFoundError(f"local DINOv2 model weights not found: {path}")
        valid_device = isinstance(device, str) and (
            device in {"cpu", "cuda"}
            or (device.startswith("cuda:") and device[5:].isdigit())
        )
        if not valid_device:
            raise ValueError("DINOv2 device must be 'cpu', 'cuda', or 'cuda:N'")

        self.model_path = path
        self.device = device
        self.source = f"dinov2:{path.name}"
        self._processor_factory = processor_factory
        self._model_factory = model_factory
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        # The model is immutable and can be reused. All episode state is held by
        # the caller's TopologicalMemory, not by this descriptor extractor.
        pass

    def close_episode(self) -> None:
        pass

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("DINOv2 place descriptors require torch") from exc
        if self._processor_factory is None or self._model_factory is None:
            try:
                from transformers import AutoImageProcessor, AutoModel
            except ImportError as exc:
                raise RuntimeError("DINOv2 place descriptors require transformers") from exc
        else:
            AutoImageProcessor = AutoModel = None
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"DINOv2 device {self.device!r} requires available CUDA")
        processor_factory = self._processor_factory or AutoImageProcessor.from_pretrained
        model_factory = self._model_factory or AutoModel.from_pretrained
        try:
            processor = processor_factory(str(self.model_path), local_files_only=True)
            model = model_factory(str(self.model_path), local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load local DINOv2-Small processor/model from "
                f"{self.model_path}: {exc}"
            ) from exc
        config = getattr(model, "config", None)
        model_type = getattr(config, "model_type", None)
        hidden_size = getattr(config, "hidden_size", None)
        if model_type not in {"dinov2", "dinov2_with_registers"} or hidden_size != 384:
            raise ValueError(
                "local place descriptor model must be DINOv2-Small "
                f"(model_type=dinov2, hidden_size=384), got {model_type!r}, {hidden_size!r}"
            )
        model.to(self.device)
        model.eval()
        self._processor = processor
        self._model = model
        self._torch = torch

    def _extract_features(self, packet: ObservationPacket, *, include_patches: bool) -> VisualPlaceFeatures:
        """Return global evidence and optionally local evidence from one forward."""
        import numpy as np
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("DINOv2 place descriptors require Pillow") from exc

        rgb = np.asarray(packet.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("DINOv2 input must be HWC uint8 RGB")
        if rgb.shape[0] == 0 or rgb.shape[1] == 0:
            raise ValueError("DINOv2 RGB frame must have nonzero height and width")
        image_quality = min(1.0, float(np.std(rgb.astype(np.float32).mean(axis=2))) / 64.0)
        self._load()
        image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
        batch = self._processor(images=image, return_tensors="pt")
        if not isinstance(batch, Mapping) or "pixel_values" not in batch:
            raise ValueError("DINOv2 image processor did not return pixel_values")
        inputs = {key: value.to(self.device) for key, value in batch.items()}
        with self._torch.inference_mode():
            output = self._model(**inputs)
            tokens = getattr(output, "last_hidden_state", None)
            register_count = int(getattr(self._model.config, "num_register_tokens", 0) or 0)
            first_patch = 1 + register_count
            if (
                tokens is None or tokens.ndim != 3 or tokens.shape[0] != 1
                or tokens.shape[1] <= first_patch or tokens.shape[2] != 384
            ):
                raise ValueError("DINOv2 did not return a batch of 384-D patch tokens")
            patches = tokens[:, first_patch:, :].squeeze(0)
            pooled = patches.mean(dim=0)
        pixel_values = batch["pixel_values"]
        global_descriptor = normalize_descriptor(pooled.detach().cpu().tolist())
        if not include_patches:
            return VisualPlaceFeatures(global_descriptor, (), (0, 0))
        patch_size = int(getattr(self._model.config, "patch_size", 14))
        height = int(pixel_values.shape[-2]) // patch_size
        width = int(pixel_values.shape[-1]) // patch_size
        if height * width != int(patches.shape[0]):
            raise ValueError("DINO patch count disagrees with processed image grid")
        # A bounded regular grid keeps geometric correspondence while avoiding
        # quadratic matching over every DINO Patch at larger processor sizes.
        import math

        stride = max(1, math.ceil(math.sqrt(int(patches.shape[0]) / 256)))
        indices = tuple(row * width + column for row in range(0, height, stride)
                        for column in range(0, width, stride))
        sampled = patches.detach().cpu().numpy()[list(indices)].astype(np.float32)
        norms = np.linalg.norm(sampled, axis=1, keepdims=True)
        if not np.all(np.isfinite(sampled)) or np.any(norms <= 0):
            raise ValueError("DINO patch descriptors must be finite and nonzero")
        patch_rows = (sampled / norms).astype(np.float16)
        return VisualPlaceFeatures(
            global_descriptor=global_descriptor,
            patch_descriptors=patch_rows,
            patch_grid=(height, width),
            patch_indices=indices,
            preprocessing_metadata={
                "processed_height": int(pixel_values.shape[-2]),
                "processed_width": int(pixel_values.shape[-1]),
                "patch_size": patch_size,
                "rgb_height": int(rgb.shape[0]), "rgb_width": int(rgb.shape[1]),
                "roi_mapping": "unavailable_without_processor_crop_metadata",
                "patch_stride": stride,
                "image_quality": image_quality,
            },
        )

    def extract_features(self, packet: ObservationPacket) -> VisualPlaceFeatures:
        """Global plus Patch features for RGM D, with one model forward."""
        return self._extract_features(packet, include_patches=True)

    def extract(self, packet: ObservationPacket) -> tuple[float, ...]:
        """Legacy global-only descriptor API; no Patch transfer to the CPU."""
        return self._extract_features(packet, include_patches=False).global_descriptor
