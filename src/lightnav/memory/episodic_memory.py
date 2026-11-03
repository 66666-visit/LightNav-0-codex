"""Episode-local event journal with structured queries."""

from __future__ import annotations

from lightnav.memory.schemas import MemoryEvent


class EpisodicMemory:
    def __init__(self) -> None:
        self._events: list[MemoryEvent] = []
        self._ids: set[str] = set()

    @property
    def events(self) -> list[MemoryEvent]:
        return list(self._events)

    def add(self, event: MemoryEvent) -> None:
        if event.event_id in self._ids:
            raise ValueError(f"duplicate event_id: {event.event_id}")
        if not 0.0 <= event.confidence <= 1.0:
            raise ValueError("event confidence must be in [0, 1]")
        if not 0.0 <= event.importance <= 1.0:
            raise ValueError("event importance must be in [0, 1]")
        if not event.source:
            raise ValueError("event source is required")
        self._events.append(event)
        self._ids.add(event.event_id)

    def query(
        self,
        *,
        kind: str | None = None,
        place_id: str | None = None,
        since_step: int | None = None,
        min_confidence: float = 0.0,
    ) -> list[MemoryEvent]:
        return [
            event for event in self._events
            if (kind is None or event.kind == kind)
            and (place_id is None or event.place_id == place_id)
            and (since_step is None or event.step_id >= since_step)
            and event.confidence >= min_confidence
        ]

    def clear(self) -> None:
        self._events.clear()
        self._ids.clear()
