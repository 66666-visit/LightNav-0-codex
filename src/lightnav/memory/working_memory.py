"""Bounded, mutable state for the current navigation episode."""

from __future__ import annotations

from lightnav.memory.schemas import MemoryEvent, WorkingMemory


class WorkingMemoryStore:
    def __init__(self, recent_limit: int = 8) -> None:
        if recent_limit < 1:
            raise ValueError("recent_limit must be positive")
        self.recent_limit = recent_limit
        self.state: WorkingMemory | None = None

    def reset(self, episode_id: str, instruction: str) -> WorkingMemory:
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

    def close(self) -> None:
        self.state = None

    def _require_state(self) -> WorkingMemory:
        if self.state is None:
            raise RuntimeError("episode is not active; call reset_episode first")
        return self.state
