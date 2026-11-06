"""Bounded, mutable state and recent semantic observations for one episode."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

from lightnav.memory.schemas import (
    LayerMemoryItem,
    MemoryEvent,
    MemoryQuery,
    ObservationEvidence,
    WorkingMemory,
)


@dataclass(frozen=True)
class ShortMotion:
    """One observed command; kept separately from the visual detections."""

    event_id: str
    step_id: int
    kind: str
    description: str
    source: str
    confidence: float
    action: dict | None


@dataclass(frozen=True)
class ShortObservation:
    """One step's evidence; an empty detection tuple means no detection this step."""

    step_id: int
    detections: tuple[ObservationEvidence, ...]
    motions: tuple[ShortMotion, ...]
    place_id: str | None
    place_confidence: float
    place_source: str
    place_event_id: str | None = None
    place_evidence: ObservationEvidence | None = None
    place_event_description: str | None = None


class WorkingMemoryStore:
    def __init__(self, recent_limit: int = 8) -> None:
        if recent_limit < 1:
            raise ValueError("recent_limit must be positive")
        self.recent_limit = recent_limit
        self.state: WorkingMemory | None = None
        self._observations: deque[ShortObservation] = deque(maxlen=recent_limit)

    @property
    def recent_observations(self) -> tuple[ShortObservation, ...]:
        """Structured steps only; RGB frames are never retained here."""
        return tuple(self._observations)

    def reset(self, episode_id: str, instruction: str) -> WorkingMemory:
        self._observations.clear()
        self.state = WorkingMemory(episode_id=episode_id, instruction=instruction)
        return self.state

    def advance(
        self,
        step_id: int,
        *,
        place_id: str | None = None,
        place_confidence: float | None = None,
        place_source: str | None = None,
        last_action: dict | None = None,
        prediction: str | None = None,
    ) -> WorkingMemory:
        state = self._require_state()
        if step_id <= state.step_id:
            raise ValueError(f"step_id must increase: {step_id} <= {state.step_id}")
        state.step_id = step_id
        state.task_progress = "in_progress"
        if place_id is not None:
            state.place_id = place_id
        if place_confidence is not None:
            state.place_confidence = place_confidence
        if place_source is not None:
            state.place_source = place_source
        if last_action is not None:
            state.last_action = last_action.copy()
        if prediction is not None:
            state.last_prediction = prediction
        return state

    def set_place(self, place_id: str, confidence: float, source: str) -> None:
        state = self._require_state()
        state.place_id = place_id
        state.place_confidence = confidence
        state.place_source = source

    def clear_place(self) -> None:
        """Discard a place hypothesis after an unverified move or conflicting view."""
        state = self._require_state()
        state.place_id = None
        state.place_confidence = 0.0
        state.place_source = "none"

    def record_event(self, event: MemoryEvent) -> None:
        state = self._require_state()
        state.recent_event_ids.append(event.event_id)
        del state.recent_event_ids[:-self.recent_limit]

    def record_instruction_match(self, label: str) -> None:
        state = self._require_state()
        if label not in state.instruction_matches:
            state.instruction_matches.append(label)

    def record_observation(self, step_id: int, events: list[MemoryEvent]) -> ShortObservation:
        """Index the already-written events once, after MemoryWriter succeeds.

        A step without detections is still stored, so a prior sighting cannot be
        reported as if the current frame contained that object.
        """
        state = self._require_state()
        if step_id != state.step_id:
            raise ValueError(f"observation step {step_id} differs from working step {state.step_id}")
        if self._observations and step_id <= self._observations[-1].step_id:
            raise ValueError("short observation steps must increase")
        if any(event.step_id != step_id for event in events):
            raise ValueError("all events must belong to the observation step")

        detections: list[ObservationEvidence] = []
        motions: list[ShortMotion] = []
        place_evidence: ObservationEvidence | None = None
        place_event_description: str | None = None
        for event in events:
            if event.kind in {"object_observation", "landmark"}:
                label = event.description.removeprefix("Observed ").strip()
                node_id = event.metadata.get("node_id")
                detections.append(ObservationEvidence(
                    event_id=event.event_id,
                    step_id=event.step_id,
                    label=label,
                    confidence=event.confidence,
                    source=event.source,
                    node_id=node_id if isinstance(node_id, str) else None,
                    evidence=f"episodic_event:{event.event_id}",
                ))
            elif event.kind in {"action", "turn"}:
                action = event.metadata.get("action")
                motions.append(ShortMotion(
                    event_id=event.event_id,
                    step_id=event.step_id,
                    kind=event.kind,
                    description=event.description,
                    source=event.source,
                    confidence=event.confidence,
                    action=deepcopy(action) if isinstance(action, dict) else None,
                ))
            elif (
                event.kind == "place_visit" and state.place_id is not None
                and event.place_id == state.place_id
            ):
                place_evidence = ObservationEvidence(
                    event_id=event.event_id,
                    step_id=event.step_id,
                    label=event.place_id,
                    confidence=event.confidence,
                    source=event.source,
                    evidence=f"episodic_event:{event.event_id}",
                )
                place_event_description = event.description

        if state.place_id is not None and place_evidence is None and self._observations:
            previous = self._observations[-1]
            if previous.place_id == state.place_id:
                place_evidence = deepcopy(previous.place_evidence)
                place_event_description = previous.place_event_description

        observation = ShortObservation(
            step_id=step_id,
            detections=tuple(detections),
            motions=tuple(motions),
            place_id=state.place_id,
            place_confidence=state.place_confidence if state.place_id else 0.0,
            place_source=state.place_source if state.place_id else "none",
            place_event_id=place_evidence.event_id if place_evidence is not None else None,
            place_evidence=place_evidence,
            place_event_description=place_event_description,
        )
        self._observations.append(observation)
        return observation

    def retrieve(self, query: MemoryQuery) -> list[LayerMemoryItem]:
        """Return local state; a carried sighting is explicitly marked as past."""
        state = self._require_state()
        if query.top_k <= 0:
            return []
        available = [obs for obs in self._observations if obs.step_id <= query.current_step]
        if not available:
            return []
        current = available[-1]
        is_current_step = current.step_id == query.current_step
        visual_step = current if is_current_step and current.detections else next(
            (obs for obs in reversed(available) if obs.detections), None,
        )

        results: list[LayerMemoryItem] = []
        if visual_step is not None:
            current_detection = is_current_step and visual_step.step_id == query.current_step
            kind = "current_observation" if current_detection else "recent_observation"
            query_words = set(query.instruction.casefold().split())
            ordered = sorted(
                visual_step.detections,
                key=lambda ref: (
                    -int(ref.label.casefold() in query_words), -ref.confidence, ref.event_id,
                ),
            )
            for ref in ordered:
                description = (
                    f"Current RGB view detected {ref.label}"
                    if current_detection else
                    f"Last detected {ref.label} at step {ref.step_id}; "
                    f"no current detection at step {query.current_step}"
                )
                results.append(LayerMemoryItem(
                    layer="short", kind=kind, description=description,
                    step_id=ref.step_id, confidence=ref.confidence, source=ref.source,
                    evidence_refs=[ref], importance=0.8 if current_detection else 0.6,
                    label=ref.label,
                ))

        if (
            is_current_step and current.place_id is not None
            and current.place_id == state.place_id and current.place_evidence is not None
        ):
            evidence = current.place_evidence
            carried = evidence.step_id < query.current_step
            original_description = current.place_event_description or "place visit recorded"
            description = (
                f"Carried place hypothesis {current.place_id} (uncertain; "
                f"last place evidence at step {evidence.step_id}: {original_description}; "
                "no new place evidence this step)"
                if carried else
                f"Current place hypothesis {current.place_id} (uncertain; "
                f"evidence: {original_description})"
            )
            results.append(LayerMemoryItem(
                layer="short", kind="place_hypothesis",
                description=description,
                step_id=evidence.step_id, confidence=evidence.confidence,
                source=evidence.source, evidence_refs=[evidence], importance=0.5,
                label=current.place_id,
            ))

        latest_motion = next(
            (motion for obs in reversed(available) for motion in reversed(obs.motions)), None,
        )
        if latest_motion is not None:
            evidence = ObservationEvidence(
                event_id=latest_motion.event_id, step_id=latest_motion.step_id,
                label=latest_motion.kind, confidence=latest_motion.confidence,
                source=latest_motion.source,
                evidence=f"episodic_event:{latest_motion.event_id}",
            )
            results.append(LayerMemoryItem(
                layer="short",
                kind="recent_turn" if latest_motion.kind == "turn" else "recent_action",
                description=f"Recent command at step {latest_motion.step_id}: "
                            f"{latest_motion.description}",
                step_id=latest_motion.step_id, confidence=latest_motion.confidence,
                source=latest_motion.source, evidence_refs=[evidence],
                importance=0.7 if latest_motion.kind == "turn" else 0.4,
            ))
        return results[:query.top_k]

    def get_context(self, query: MemoryQuery | None = None) -> list[LayerMemoryItem]:
        state = self._require_state()
        if query is None:
            query = MemoryQuery(
                instruction=state.instruction, current_step=state.step_id,
                place_id=state.place_id, top_k=self.recent_limit,
            )
        return self.retrieve(query)

    def close(self) -> None:
        self.state = None
        self._observations.clear()

    def _require_state(self) -> WorkingMemory:
        if self.state is None:
            raise RuntimeError("episode is not active; call reset_episode first")
        return self.state
