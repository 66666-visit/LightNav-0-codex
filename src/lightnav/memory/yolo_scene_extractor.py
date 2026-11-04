"""Optional local YOLO11n detector for robot-observed RGB frames.

Ultralytics treats NumPy input as BGR, whereas ObservationPacket carries RGB.
The conversion here is transient; neither image pixels nor model boxes are saved
outside a compact detection record. The model is loaded only when explicitly
constructed with a local weights file.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lightnav.memory.schemas import DetectedObject, ObservationPacket, SceneExtraction

logger = logging.getLogger(__name__)


class YOLOSceneExtractor:
    """Detect COCO-style object classes; make no place or topology claims."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence: float = 0.25,
        every_n_steps: int = 5,
        *,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0.0 < confidence <= 1.0 or not math.isfinite(confidence):
            raise ValueError("confidence must be finite and in (0, 1]")
        if not isinstance(every_n_steps, int) or every_n_steps < 1:
            raise ValueError("every_n_steps must be a positive integer")
        if not device or not isinstance(device, str):
            raise ValueError("device must be a nonempty string")
        path = Path(model_path).expanduser()
        if path.suffix.lower() != ".pt":
            raise ValueError(f"YOLO model must be a local .pt file: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"local YOLO model file not found: {path}")

        self.model_path = path
        self.device = device
        self.confidence = float(confidence)
        self.every_n_steps = every_n_steps
        # The file name is recorded; architecture is supplied by the operator
        # and cannot be inferred safely from an arbitrary renamed .pt file.
        self.source = f"ultralytics_yolo:{path.name}"
        self.last_attempt_step: int | None = None
        self.last_error: str | None = None

        if model_factory is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "YOLOSceneExtractor requires the optional ultralytics package"
                ) from exc
            model_factory = YOLO
        try:
            self._model = model_factory(str(path))
        except Exception as exc:
            raise RuntimeError(f"failed to load local YOLO model: {path}") from exc
        task = getattr(self._model, "task", "detect")
        if task != "detect":
            raise RuntimeError(f"local YOLO model must have detection task, got {task!r}: {path}")

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        self.last_attempt_step = None
        self.last_error = None

    def should_extract(self, step_id: int) -> bool:
        """Run on the first frame (step 0) and every N observations thereafter."""
        return step_id >= 0 and step_id % self.every_n_steps == 0

    def extract(self, packet: ObservationPacket) -> SceneExtraction:
        if not self.should_extract(packet.step_id):
            return SceneExtraction()
        self.last_attempt_step = packet.step_id
        self.last_error = None
        try:
            rgb = np.asarray(packet.rgb)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError("YOLO input must be HWC uint8 RGB")
            if rgb.shape[0] == 0 or rgb.shape[1] == 0:
                raise ValueError("YOLO input must have nonzero height and width")
            # Ultralytics interprets NumPy arrays as BGR, then converts them to
            # RGB internally. Make a contiguous BGR view for that API.
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            results = self._model.predict(
                source=bgr, conf=self.confidence, device=self.device,
                verbose=False, save=False,
            )
            detections = self._parse_results(results)
            if not detections:
                return SceneExtraction()
            return SceneExtraction(
                objects=[detection.label for detection in detections],
                detections=detections,
                confidence=max(detection.confidence for detection in detections),
                source=self.source,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "YOLO extraction failed at step=%s model=%s device=%s: %s",
                packet.step_id, self.model_path, self.device, exc, exc_info=True,
            )
            return SceneExtraction()

    def close_episode(self) -> None:
        self.last_attempt_step = None
        self.last_error = None

    def _parse_results(self, results: Any) -> list[DetectedObject]:
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        classes = self._as_list(boxes.cls)
        confidences = self._as_list(boxes.conf)
        if len(classes) != len(confidences):
            raise ValueError("YOLO result has mismatched class and confidence counts")
        raw_xyxy = getattr(boxes, "xyxy", None)
        if raw_xyxy is None:
            raise ValueError("YOLO result has no detection boxes")
        coordinates = self._as_list(raw_xyxy)
        if len(coordinates) != len(classes):
            raise ValueError("YOLO result has mismatched box counts")
        names = getattr(result, "names", None)
        if names is None:
            names = getattr(self._model, "names", None)

        detections: list[DetectedObject] = []
        for index, (class_value, confidence_value) in enumerate(zip(classes, confidences)):
            class_number = float(class_value)
            score = float(confidence_value)
            if (
                not math.isfinite(class_number) or not class_number.is_integer()
                or not math.isfinite(score) or not self.confidence <= score <= 1.0
            ):
                continue
            class_id = int(class_number)
            if isinstance(names, Mapping):
                raw_label = names.get(class_id)
            elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                raw_label = names[class_id]
            else:
                raw_label = None
            if not isinstance(raw_label, str) or not raw_label.strip():
                continue
            values = coordinates[index]
            if len(values) != 4:
                continue
            coords = tuple(float(value) for value in values)
            if not all(math.isfinite(value) for value in coords):
                continue
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                continue
            detections.append(
                DetectedObject(label=raw_label.strip(), confidence=score, bbox_xyxy=coords)
            )
        return detections

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value)
