"""Independent category sightings; optional instance matching needs stronger evidence."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence

from lightnav.memory.schemas import MemoryEvent


@dataclass(frozen=True)
class TemporalSupport:
    label: str
    event_ids: tuple[str, ...]
    steps: tuple[int, ...]
    supported: bool


class ObjectAssociation:
    """Keep category evidence separate from physical instance identity."""

    def __init__(self, *, window: int = 3, min_support: int = 2) -> None:
        self.window = window
        self.min_support = min_support
        self._extraction_steps: deque[int] = deque(maxlen=window)
        self._sightings: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
        self._seen_ids: set[str] = set()
        self._last_step: int | None = None

    def observe(self, event: MemoryEvent) -> TemporalSupport:
        label = event.description.removeprefix("Observed ").strip().casefold()
        if event.event_id in self._seen_ids:
            raise ValueError(f"duplicate evidence ID: {event.event_id}")
        if self._last_step is not None and event.step_id < self._last_step:
            raise ValueError("temporal evidence steps must increase")
        self.advance_extraction(event.step_id)
        self._seen_ids.add(event.event_id)
        sightings = self._sightings[label]
        if not sightings or sightings[-1][0] != event.step_id:
            sightings.append((event.step_id, event.event_id))
        ids = tuple(evidence_id for _, evidence_id in sightings)
        steps = tuple(step for step, _ in sightings)
        return TemporalSupport(label, ids, steps, len(steps) >= self.min_support)

    def advance_extraction(self, step: int) -> None:
        """Count a successful YOLO call even when it returned no detections."""
        if self._last_step is not None and step < self._last_step:
            raise ValueError("temporal extraction steps must increase")
        if step not in self._extraction_steps:
            self._extraction_steps.append(step)
        self._last_step = step
        valid = set(self._extraction_steps)
        for entries in self._sightings.values():
            while entries and entries[0][0] not in valid:
                entries.popleft()

    @staticmethod
    def instance_match(
        previous_label: str, current_label: str,
        previous_bbox: Sequence[float] | None, current_bbox: Sequence[float] | None,
        previous_roi: Sequence[float] | None, current_roi: Sequence[float] | None,
        *, visual_motion_correspondence: bool,
    ) -> bool:
        """Return an instance hypothesis only with independent appearance and motion support."""
        if (
            previous_label.casefold() != current_label.casefold()
            or previous_bbox is None or current_bbox is None
            or previous_roi is None or current_roi is None
            or not visual_motion_correspondence
        ):
            return False
        if len(previous_roi) != len(current_roi) or not previous_roi:
            return False
        dot = sum(float(a) * float(b) for a, b in zip(previous_roi, current_roi))
        norm_a = sqrt(sum(float(a) ** 2 for a in previous_roi))
        norm_b = sqrt(sum(float(b) ** 2 for b in current_roi))
        if not all(isfinite(value) for value in (*previous_bbox, *current_bbox, dot, norm_a, norm_b)):
            return False
        if norm_a <= 0 or norm_b <= 0 or dot / (norm_a * norm_b) < 0.85:
            return False
        x1 = max(previous_bbox[0], current_bbox[0])
        y1 = max(previous_bbox[1], current_bbox[1])
        x2 = min(previous_bbox[2], current_bbox[2])
        y2 = min(previous_bbox[3], current_bbox[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = (previous_bbox[2] - previous_bbox[0]) * (previous_bbox[3] - previous_bbox[1])
        area_b = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
        union = area_a + area_b - intersection
        return union > 0 and intersection / union >= 0.20

    def clear(self) -> None:
        self._extraction_steps.clear()
        self._sightings.clear()
        self._seen_ids.clear()
        self._last_step = None
