"""Independent local YOLO11n-seg instance masks for ConceptGraphs-Lite.

The L1 detection model is never reused. NumPy inputs are BGR for Ultralytics,
so RGB is converted only for the prediction call. Masks must be native-sized;
boxes alone cannot stand in for segmentation.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentationInstance:
    class_name: str
    confidence: float
    mask: np.ndarray
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("segmentation class name must be nonempty")
        if not math.isfinite(self.confidence) or not 0 < self.confidence <= 1:
            raise ValueError("segmentation confidence must be in (0, 1]")
        values = np.asarray(self.mask)
        if values.ndim != 2 or values.dtype != np.bool_ or not values.any():
            raise ValueError("segmentation mask must be a nonempty boolean image")
        if len(self.bbox_xyxy) != 4 or not all(math.isfinite(v) for v in self.bbox_xyxy):
            raise ValueError("segmentation bbox must have four finite coordinates")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not (0 <= x1 < x2 <= values.shape[1] and 0 <= y1 < y2 <= values.shape[0]):
            raise ValueError("segmentation bbox must lie within the native image")
        object.__setattr__(self, "mask", values)


class YOLOSegExtractor:
    """Load an explicitly configured local segmentation model and parse masks."""

    def __init__(
        self, model_path: str, device: str = "cpu", confidence: float = 0.25,
        every_n_steps: int = 5, *, model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("segmentation model path must be nonempty")
        path = Path(model_path).expanduser()
        if path.suffix.lower() != ".pt":
            raise ValueError(f"segmentation model must be a local .pt file: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"local segmentation model file not found: {path}")
        if not isinstance(device, str) or not device:
            raise ValueError("segmentation device must be nonempty")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError("segmentation confidence must be in (0, 1]")
        if type(every_n_steps) is not int or every_n_steps < 1:
            raise ValueError("segmentation every_n_steps must be positive")
        self.model_path = path
        self.device = device
        self.confidence = float(confidence)
        self.every_n_steps = every_n_steps
        self.last_attempt_step: int | None = None
        self.last_error: str | None = None
        self.last_detection_ms: float | None = None
        if model_factory is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("YOLOSegExtractor requires ultralytics") from exc
            model_factory = YOLO
        try:
            self._model = model_factory(str(path))
        except Exception as exc:
            raise RuntimeError(f"failed to load local segmentation model: {path}") from exc
        if getattr(self._model, "task", None) != "segment":
            raise RuntimeError(f"local YOLO model must have segment task: {path}")

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        self.last_attempt_step = None
        self.last_error = None
        self.last_detection_ms = None

    def close_episode(self) -> None:
        self.reset_episode("", "")

    def should_extract(self, step_id: int) -> bool:
        return type(step_id) is int and step_id >= 0 and step_id % self.every_n_steps == 0

    def extract(self, rgb: np.ndarray, step_id: int) -> list[SegmentationInstance]:
        self.last_detection_ms = None
        if not self.should_extract(step_id):
            return []
        self.last_attempt_step = step_id
        self.last_error = None
        started = perf_counter()
        try:
            image = np.asarray(rgb)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("segmentation input must be HWC uint8 RGB")
            if min(image.shape[:2]) < 1:
                raise ValueError("segmentation image must be nonempty")
            bgr = np.ascontiguousarray(image[..., ::-1])
            results = self._model.predict(
                source=bgr, conf=self.confidence, device=self.device,
                verbose=False, save=False, retina_masks=True,
            )
            return self._parse_results(results, image.shape[:2])
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("YOLO segmentation failed at step=%s model=%s: %s",
                           step_id, self.model_path, exc, exc_info=True)
            return []
        finally:
            self.last_detection_ms = (perf_counter() - started) * 1000.0

    def _parse_results(
        self, results: Any, image_hw: tuple[int, int],
    ) -> list[SegmentationInstance]:
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None:
            return []
        classes = self._array(boxes.cls).reshape(-1)
        scores = self._array(boxes.conf).reshape(-1)
        coordinates = self._array(boxes.xyxy)
        if len(classes) != len(scores) or coordinates.shape != (len(classes), 4):
            raise ValueError("segmentation boxes/classes/confidences have different counts")
        if len(classes) == 0:
            return []
        if masks is None or getattr(masks, "data", None) is None:
            raise ValueError("YOLO segment result has boxes but no instance masks")
        mask_data = self._array(masks.data)
        if mask_data.shape != (len(classes), *image_hw):
            raise ValueError("YOLO instance masks are not aligned to native RGB size")
        names = getattr(result, "names", None)
        if names is None:
            names = getattr(self._model, "names", None)
        instances: list[SegmentationInstance] = []
        for index, (raw_class, raw_score) in enumerate(zip(classes, scores)):
            number, score = float(raw_class), float(raw_score)
            if not math.isfinite(number) or not number.is_integer() or not math.isfinite(score):
                continue
            if not self.confidence <= score <= 1:
                continue
            class_id = int(number)
            if isinstance(names, Mapping):
                label = names.get(class_id)
            elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                label = names[class_id]
            else:
                label = None
            if not isinstance(label, str) or not label.strip():
                continue
            mask_values = mask_data[index]
            if not np.isfinite(mask_values).all():
                continue
            mask = mask_values >= 0.5
            if not mask.any():
                continue
            bbox = tuple(float(value) for value in coordinates[index])
            try:
                instances.append(SegmentationInstance(label.strip(), score, mask, bbox))
            except ValueError:
                continue
        return instances

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if value is None:
            raise ValueError("YOLO segmentation output is missing a tensor")
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)
