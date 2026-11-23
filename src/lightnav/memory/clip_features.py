"""Local CLIP object features for the ConceptGraphs-Lite L3 map.

This module follows the official ConceptGraphs feature path instead of defining
a new visual encoder:

* ``conceptgraph/utils/model_utils.py::compute_clip_features_batched`` crops
  each detection's bounding box with a 20-pixel pad, runs the CLIP image
  encoder, and L2-normalizes every crop feature.
* ``conceptgraph/slam/utils.py::merge_obj2_into_obj1`` fuses a new observation
  into an object as a detection-count weighted average of normalized features,
  followed by another L2 normalization.
* ``conceptgraph/slam/slam_classes.py::MapObjectList.compute_similarities`` and
  ``scripts/visualize_cfslam_results.py::color_by_clip_sim`` use normalized
  text/image features and cosine similarity for text queries.

The encoder is imported lazily so CPU tests and the default L1 path never
require torch or an OpenCLIP checkpoint.  Weights are loaded from a local file
only; ``LocalClipEncoder`` never asks a model hub to download anything.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

#: Bounding-box pad copied from ConceptGraphs
#: ``utils/model_utils.py::compute_clip_features(_batched)``.
CLIP_CROP_PADDING_PX = 20


@runtime_checkable
class ClipFeatureEncoder(Protocol):
    """Minimal encoder contract used by feature extraction and text queries."""

    @property
    def feature_dim(self) -> int:
        ...

    def encode_image(
        self, rgb: np.ndarray, boxes: Sequence[tuple[int, int, int, int]],
    ) -> np.ndarray:
        ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        ...


def clip_crop_boxes(
    masks: Sequence[np.ndarray], image_hw: tuple[int, int], *,
    padding: int = CLIP_CROP_PADDING_PX,
) -> list[tuple[int, int, int, int] | None]:
    """Return padded bounding-box crops for boolean instance masks.

    ConceptGraphs crops ``detections.xyxy`` with a symmetric 20-pixel pad and
    clips the pad at the image border.  A segmentation mask's tight bounding
    box is the equivalent box here; the crop itself still contains background,
    exactly as in the official implementation.  Masks without pixels map to
    ``None`` so callers can preserve one-to-one mask/feature ordering.
    """
    if type(padding) is not int or padding < 0:
        raise ValueError("CLIP crop padding must be a nonnegative integer")
    height, width = image_hw
    if type(height) is not int or type(width) is not int or height < 1 or width < 1:
        raise ValueError("image_hw must contain positive integers")
    boxes: list[tuple[int, int, int, int] | None] = []
    for mask in masks:
        values = np.asarray(mask)
        if values.ndim != 2 or values.dtype != np.bool_ or values.shape != (height, width):
            raise ValueError("instance masks must be boolean and aligned to the RGB image")
        rows = np.flatnonzero(values.any(axis=1))
        cols = np.flatnonzero(values.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            boxes.append(None)
            continue
        x_min, x_max = int(cols[0]), int(cols[-1]) + 1
        y_min, y_max = int(rows[0]), int(rows[-1]) + 1
        left = min(padding, x_min)
        top = min(padding, y_min)
        right = min(padding, width - x_max)
        bottom = min(padding, height - y_max)
        boxes.append((x_min - left, y_min - top, x_max + right, y_max + bottom))
    return boxes


def normalize_feature(feature: Any, *, expected_dim: int | None = None) -> np.ndarray:
    """Validate a 1-D feature and return its L2-normalized float32 vector."""
    values = np.asarray(feature, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("CLIP feature must be a nonempty one-dimensional vector")
    if expected_dim is not None and values.size != expected_dim:
        raise ValueError(
            f"CLIP feature dimension {values.size} does not match {expected_dim}"
        )
    if not np.isfinite(values).all():
        raise ValueError("CLIP feature must contain only finite values")
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("CLIP feature must have a finite positive norm")
    return (values / norm).astype(np.float32)


def fuse_clip_feature(
    previous: np.ndarray | None, previous_count: int, observation: Any,
) -> tuple[np.ndarray, int]:
    """Fuse one normalized observation exactly like ConceptGraphs.

    ``merge_obj2_into_obj1`` computes
    ``(old * n_old + new * n_new) / (n_old + n_new)`` and normalizes the
    result.  A detection contributes one observation (``n_new == 1``), so the
    running update is a detection-count weighted mean followed by L2
    normalization.
    """
    if type(previous_count) is not int or previous_count < 0:
        raise ValueError("previous feature count must be a nonnegative integer")
    current = normalize_feature(observation)
    if previous is None or previous_count == 0:
        return current, 1
    prior = normalize_feature(previous, expected_dim=current.size)
    fused = (prior * previous_count + current) / float(previous_count + 1)
    return normalize_feature(fused, expected_dim=current.size), previous_count + 1


def cosine_similarity(query: Any, features: Any) -> np.ndarray:
    """Return cosine similarity between one query and an ``(N, D)`` matrix."""
    query_values = normalize_feature(query)
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != query_values.size or matrix.shape[0] == 0:
        raise ValueError("features must be a nonempty (N, D) matrix matching the query")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError("features must have finite positive norms")
    return ((matrix @ query_values) / norms).astype(np.float32)


class LocalClipEncoder:
    """Load the official OpenCLIP model from a local checkpoint.

    ConceptGraphs uses ``open_clip.create_model_and_transforms("ViT-H-14",
    "laion2b_s32b_b79k")``.  The model name defaults to the same architecture;
    ``model_path`` must be an already-downloaded local OpenCLIP checkpoint.
    Nothing is fetched from the network and no API is called.
    """

    def __init__(
        self, model_path: str | Path, *, device: str = "cpu",
        model_name: str = "ViT-H-14", precision: str = "fp32",
        batch_size: int = 8,
    ) -> None:
        path = Path(model_path).expanduser()
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("CLIP model name must be nonempty")
        if not path.is_file():
            raise FileNotFoundError(f"local CLIP checkpoint not found: {path}")
        if not isinstance(device, str) or not device:
            raise ValueError("CLIP device must be nonempty")
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("CLIP precision must be fp32, fp16 or bf16")
        if device.strip().lower() == "cpu" and precision != "fp32":
            raise ValueError("CLIP fp16/bf16 requires a CUDA device")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("CLIP batch size must be a positive integer")
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "on-demand L3 CLIP requires open_clip_torch; install the 'clip' extra"
            ) from exc
        try:
            # The file-existence check above means OpenCLIP resolves
            # ``pretrained`` through its local-file branch; it never asks a
            # model hub to download weights.  This is the same
            # create_model_and_transforms call ConceptGraphs uses.
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=str(path), device=device,
                precision=precision,
            )
            tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:  # noqa: BLE001 - report the exact local path
            raise RuntimeError(
                f"failed to load local CLIP checkpoint {path} as {model_name}"
            ) from exc
        self._torch = torch
        self._model = model.eval()
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._device = device
        self._precision = precision
        self._input_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision]
        self._batch_size = batch_size
        self._feature_dim = int(getattr(model.visual, "output_dim"))
        self.model_path = path
        self.model_name = model_name

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def encode_image(
        self, rgb: np.ndarray, boxes: Sequence[tuple[int, int, int, int]],
    ) -> np.ndarray:
        values = np.asarray(rgb)
        if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
            raise ValueError("CLIP image input must be HWC uint8 RGB")
        height, width = values.shape[:2]
        if len(boxes) == 0:
            return np.empty((0, self._feature_dim), dtype=np.float32)
        if min(height, width) < 1:
            raise ValueError("CLIP image input must be nonempty")
        from PIL import Image

        crops = []
        for box in boxes:
            if len(box) != 4:
                raise ValueError("CLIP crop box must have four coordinates")
            x1, y1, x2, y2 = (int(value) for value in box)
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("CLIP crop box must lie inside the RGB image")
            crops.append(self._preprocess(Image.fromarray(values[y1:y2, x1:x2])))
        features = []
        with self._torch.no_grad():
            for start in range(0, len(crops), self._batch_size):
                batch = self._torch.stack(
                    crops[start:start + self._batch_size],
                ).to(device=self._device, dtype=self._input_dtype)
                encoded = self._model.encode_image(batch).float()
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                features.append(encoded.cpu())
        return self._torch.cat(features, dim=0).numpy().astype(np.float32)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("CLIP text input must be a nonempty sequence of strings")
        with self._torch.no_grad():
            tokens = self._tokenizer(list(texts)).to(self._device)
            encoded = self._model.encode_text(tokens).float()
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        return encoded.cpu().numpy().astype(np.float32)
