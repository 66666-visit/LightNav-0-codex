"""Convert allowed episode observations into structured memory records.

There is no production semantic RGB extractor in LightNav. The default extractor
returns no semantic facts; the mock is injectable for tests. A compact RGB color
histogram supports only an appearance-change signal and is never saved as an image.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

import numpy as np

from lightnav.memory.episodic_memory import EpisodicMemory
from lightnav.memory.scene_graph import SceneGraph
from lightnav.memory.schemas import MemoryEvent, ObservationPacket, SceneExtraction
from lightnav.memory.working_memory import WorkingMemoryStore

_CURRENT_PLACE = object()


class SceneExtractor(Protocol):
    def reset_episode(self, episode_id: str, instruction: str) -> None:
        """Drop any extractor-local state from the previous episode."""

    def extract(self, packet: ObservationPacket) -> SceneExtraction:
        """Extract supported semantic facts from the packet's RGB frame."""

    def close_episode(self) -> None:
        """Release episode-local extractor state."""


class NullSceneExtractor:
    """Production default until a validated local RGB semantic model is connected."""

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        pass

    def extract(self, packet: ObservationPacket) -> SceneExtraction:
        return SceneExtraction()

    def close_episode(self) -> None:
        pass


class MockSceneExtractor:
    """Step-scripted semantic facts for CPU tests; never used automatically."""

    def __init__(self, by_step: dict[int, SceneExtraction] | None = None) -> None:
        self.by_step = by_step or {}

    def reset_episode(self, episode_id: str, instruction: str) -> None:
        pass

    def extract(self, packet: ObservationPacket) -> SceneExtraction:
        scripted = self.by_step.get(packet.step_id)
        if scripted is None:
            return SceneExtraction()
        return SceneExtraction(
            place_label=scripted.place_label,
            objects=list(scripted.objects),
            landmarks=list(scripted.landmarks),
            confidence=scripted.confidence,
            source="mock_scene_extractor",
        )

    def close_episode(self) -> None:
        pass


class MemoryWriter:
    def __init__(
        self,
        working: WorkingMemoryStore,
        episodic: EpisodicMemory,
        graph: SceneGraph,
        extractor: SceneExtractor,
    ) -> None:
        self.working = working
        self.episodic = episodic
        self.graph = graph
        self.extractor = extractor
        self._event_seq = 0
        self._last_rgb_histogram: tuple[float, ...] | None = None

    def observe(self, packet: ObservationPacket) -> list[MemoryEvent]:
        """Write one observation, excluding any evaluator-only ``info`` fields."""
        rgb_histogram = self._rgb_histogram(packet.rgb)
        if self.working.state is None:
            raise RuntimeError("episode is not active; call reset_episode first")
        if packet.instruction and packet.instruction != self.working.state.instruction:
            raise ValueError("instruction changed within an episode")
        if packet.step_id <= self.working.state.step_id:
            raise ValueError(f"step_id must increase: {packet.step_id} <= {self.working.state.step_id}")
        if packet.last_action is not None and not isinstance(packet.last_action, dict):
            raise TypeError("last_action must be a dict or None")

        previous_place = self.working.state.place_id
        appearance_delta = self._appearance_delta(rgb_histogram)
        # Extraction and validation may fail. Do both before advancing the step
        # or writing any event, so the same observation can be retried.
        extraction = self.extractor.extract(packet)
        self._validate_extraction(extraction)
        self.working.advance(
            packet.step_id, last_action=packet.last_action, prediction=packet.prediction,
        )
        new_events: list[MemoryEvent] = []
        forward_command = self._has_forward_command(packet.last_action)

        if packet.last_action is not None:
            new_events.append(self._write_action(packet))
        if appearance_delta is not None and appearance_delta >= 0.45:
            new_events.append(self._event(
                packet.step_id, "visual_change", "RGB appearance changed since the previous observation",
                confidence=min(0.9, 0.45 + appearance_delta / 3),
                source="rgb_color_histogram", importance=0.55,
                place_id=None if forward_command else previous_place,
                metadata={"histogram_delta": round(appearance_delta, 3)},
            ))

        current_place = self.graph.get_node(previous_place) if previous_place else None
        semantic_label = extraction.place_label.strip() if extraction.place_label else ""
        label_changed = bool(
            semantic_label and current_place
            and semantic_label.casefold() != current_place.label.casefold()
        )
        strong_visual_change = appearance_delta is not None and appearance_delta >= 0.65
        semantic_transition = bool(
            current_place and forward_command and label_changed
            and extraction.confidence >= 0.65
        )

        if semantic_label and extraction.confidence > 0.0 and (current_place is None or semantic_transition):
            evidence = f"RGB extractor reported place label at step {packet.step_id}"
            place = self.graph.add_place(
                semantic_label, confidence=extraction.confidence, source=extraction.source,
                step_id=packet.step_id, evidence=evidence,
            )
            self.working.set_place(place.node_id, extraction.confidence, extraction.source)
            new_events.append(self._event(
                packet.step_id, "place_visit", f"Extractor reported {semantic_label} (place hypothesis)",
                confidence=extraction.confidence, source=extraction.source,
                place_id=place.node_id, importance=0.7,
            ))
            if semantic_transition:
                graph_evidence = (
                    "forward velocity command followed by a different extracted place label "
                    f"at step {packet.step_id}; traversal unverified"
                )
                self.graph.add_edge(
                    current_place.node_id, place.node_id, "possible_transition",
                    confidence=min(0.6, extraction.confidence),
                    source="issued_command_and_rgb_semantics",
                    step_id=packet.step_id, evidence=graph_evidence,
                )
        elif current_place is not None:
            if semantic_label and not label_changed and not forward_command and extraction.confidence > 0.0:
                self.graph.update_node(
                    current_place.node_id, confidence=extraction.confidence,
                    source=extraction.source,
                    evidence=f"same extracted place label without a forward command at step {packet.step_id}",
                    visited=True, step_id=packet.step_id,
                )
            if label_changed:
                new_events.append(self._event(
                    packet.step_id, "place_hypothesis",
                    f"RGB extractor described {semantic_label}; place identity unresolved",
                    confidence=min(0.5, extraction.confidence), source=extraction.source,
                    place_id=None, importance=0.4,
                ))

        if current_place is not None and not semantic_transition and (forward_command or label_changed):
            # The old place hypothesis cannot be carried into later frames
            # after unresolved motion or a conflicting extracted label.
            self.working.clear_place()

        if forward_command and strong_visual_change and not semantic_transition:
            new_events.append(self._event(
                packet.step_id, "possible_transition",
                "RGB appearance changed after a forward command; place change unverified",
                confidence=0.35, source="issued_command_and_rgb_color_histogram",
                place_id=None, importance=0.4,
                metadata={"histogram_delta": round(appearance_delta, 3)},
            ))

        place_id = self.working.state.place_id
        if extraction.source != "none" and extraction.confidence > 0.0:
            detections = (("object_observation", extraction.objects), ("landmark", extraction.landmarks))
            for kind, labels in detections:
                for label in labels:
                    label = label.strip()
                    if not label:
                        continue
                    node = None
                    if place_id:
                        node = self.graph.add_object(
                            label, place_id=place_id, confidence=extraction.confidence,
                            source=extraction.source, step_id=packet.step_id,
                            evidence=f"extractor output for RGB observation {packet.step_id}",
                        )
                    metadata = {"node_id": node.node_id} if node is not None else {}
                    new_events.append(self._event(
                        packet.step_id, kind, f"Observed {label}",
                        confidence=extraction.confidence, source=extraction.source,
                        place_id=place_id, importance=0.8 if kind == "landmark" else 0.6,
                        metadata=metadata,
                    ))
                    if self._instruction_relevant(label, self.working.state.instruction):
                        self.working.record_instruction_match(label)
                        new_events.append(self._event(
                            packet.step_id, "instruction_match",
                            f"Observed instruction-related {label}",
                            confidence=extraction.confidence, source=extraction.source,
                            place_id=place_id, importance=0.95,
                            metadata=metadata,
                        ))

        self._last_rgb_histogram = rgb_histogram
        return new_events

    def _write_action(self, packet: ObservationPacket) -> MemoryEvent:
        action = packet.last_action or {}
        name = str(action.get("action", "unknown"))
        args = action.get("action_args") if isinstance(action.get("action_args"), dict) else {}
        angular = args.get("angular_velocity")
        if name == "velocity_control" and isinstance(angular, (int, float)) and abs(angular) > 0.05:
            kind = "turn"
            description = f"Issued velocity control with angular command {angular:.2f} before this view"
            importance = 0.7
        else:
            kind = "action"
            description = f"Issued {name} before this view"
            importance = 0.4
        return self._event(
            packet.step_id, kind, description, confidence=1.0,
            source="issued_command_and_next_observation",
            importance=importance,
            metadata={"action": action, "prediction": packet.prediction or ""},
        )

    def _event(
        self,
        step_id: int,
        kind: str,
        description: str,
        *,
        confidence: float,
        source: str,
        place_id: str | None | object = _CURRENT_PLACE,
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> MemoryEvent:
        event = MemoryEvent(
            event_id=f"event_{self._event_seq}", step_id=step_id, kind=kind,
            description=description, confidence=confidence, source=source,
            place_id=self.working.state.place_id if place_id is _CURRENT_PLACE else place_id,
            importance=importance, metadata=metadata or {},
        )
        self._event_seq += 1
        self.episodic.add(event)
        self.working.record_event(event)
        return event

    @staticmethod
    def _rgb_histogram(rgb: object) -> tuple[float, ...]:
        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("ObservationPacket.rgb must be HWC uint8 RGB")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("RGB frame must have nonzero height and width")
        count = array.shape[0] * array.shape[1]
        bins: list[float] = []
        for channel in range(3):
            histogram = np.bincount((array[..., channel].astype(np.uint16) // 32).ravel(), minlength=8)
            bins.extend(float(value) / count for value in histogram[:8])
        return tuple(bins)

    def _appearance_delta(self, histogram: tuple[float, ...]) -> float | None:
        if self._last_rgb_histogram is None:
            return None
        return sum(abs(a - b) for a, b in zip(histogram, self._last_rgb_histogram)) / 3.0

    @staticmethod
    def _has_forward_command(action: dict | None) -> bool:
        if not isinstance(action, dict) or action.get("action") != "velocity_control":
            return False
        args = action.get("action_args")
        if not isinstance(args, dict):
            return False
        raw = args.get("linear_velocity")
        return isinstance(raw, (int, float)) and math.isfinite(raw) and raw > -0.95

    @staticmethod
    def _instruction_relevant(label: str, instruction: str) -> bool:
        label_words = {word for word in re.findall(r"\w+", label.casefold()) if len(word) >= 3}
        instruction_words = set(re.findall(r"\w+", instruction.casefold()))
        return bool(label_words and label_words <= instruction_words)

    @staticmethod
    def _validate_extraction(extraction: SceneExtraction) -> None:
        if not isinstance(extraction, SceneExtraction):
            raise TypeError("SceneExtractor.extract must return SceneExtraction")
        if (
            not isinstance(extraction.confidence, (int, float))
            or not math.isfinite(extraction.confidence)
            or not 0.0 <= extraction.confidence <= 1.0
        ):
            raise ValueError("extractor confidence must be in [0, 1]")
        if extraction.place_label is not None and not isinstance(extraction.place_label, str):
            raise TypeError("extractor place_label must be a string or None")
        for labels in (extraction.objects, extraction.landmarks):
            if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
                raise TypeError("extractor objects and landmarks must be lists of strings")
        if not isinstance(extraction.source, str) or not extraction.source.strip():
            raise ValueError("extractor source must be a nonempty string")
        if (
            extraction.place_label or extraction.objects or extraction.landmarks
        ) and extraction.source.strip().casefold() == "none":
            raise ValueError("semantic facts require an extractor source")
