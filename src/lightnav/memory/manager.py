"""Coordinate working, episodic, and graph memory for one R2R episode at a time."""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from lightnav.memory.episodic_memory import EpisodicMemory
from lightnav.memory.memory_writer import MemoryWriter, NullSceneExtractor, SceneExtractor
from lightnav.memory.retriever import MemoryRetriever
from lightnav.memory.scene_graph import SceneGraph
from lightnav.memory.schemas import MemoryContext, MemoryEvent, MemoryQuery, ObservationPacket
from lightnav.memory.working_memory import WorkingMemoryStore


class MemoryManager:
    """Episode-local memory. No records survive ``close_episode`` or a new reset."""

    def __init__(
        self,
        extractor: SceneExtractor | None = None,
        *,
        top_k: int = 5,
        max_context_chars: int = 600,
    ) -> None:
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if max_context_chars < 0:
            raise ValueError("max_context_chars must be nonnegative")
        self.extractor = extractor if extractor is not None else NullSceneExtractor()
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.working = WorkingMemoryStore()
        self.episodic = EpisodicMemory()
        self.scene_graph = SceneGraph()
        self.retriever = MemoryRetriever()
        self._writer: MemoryWriter | None = None
        self._pending_action: dict | None = None
        self._pending_prediction: str | None = None

    @property
    def active(self) -> bool:
        return self._writer is not None

    def reset_episode(self, episode_id: str | None = None, instruction: str = "") -> None:
        """Begin a new episode, discarding all previous episode data."""
        self.close_episode()
        episode_id = episode_id or uuid4().hex
        self.extractor.reset_episode(episode_id, instruction)
        self.working.reset(episode_id, instruction)
        self._writer = MemoryWriter(
            self.working, self.episodic, self.scene_graph, self.extractor,
        )

    def observe(self, packet: ObservationPacket) -> list[MemoryEvent]:
        """Consume one real RGB observation and the action that preceded it.

        ``record_action`` records a command chosen by the policy. It is only
        converted into an executed-action event when the next observation arrives.
        This keeps commands from a failed ``env.step`` out of episodic memory.
        """
        writer = self._require_writer()
        action = packet.last_action if packet.last_action is not None else self._pending_action
        prediction = packet.prediction if packet.prediction is not None else self._pending_prediction
        safe_packet = ObservationPacket(
            step_id=packet.step_id, rgb=packet.rgb, instruction=packet.instruction,
            last_action=deepcopy(action), prediction=prediction,
        )
        events = writer.observe(safe_packet)
        self._pending_action = None
        self._pending_prediction = None
        return events

    def record_action(self, action: dict, *, prediction: str | None = None) -> None:
        """Stage the selected command; it becomes history after a later observation."""
        self._require_writer()
        if not isinstance(action, dict):
            raise TypeError("action must be a dict")
        self._pending_action = deepcopy(action)
        self._pending_prediction = prediction

    def retrieve(self, query: MemoryQuery | None = None) -> MemoryContext:
        state = self.working._require_state()
        if query is None:
            query = MemoryQuery(
                instruction=state.instruction, current_step=state.step_id,
                place_id=state.place_id, top_k=self.top_k,
                max_chars=self.max_context_chars,
            )
        return self.retriever.retrieve(
            query, self.episodic.events, self.scene_graph.nodes, self.scene_graph.edges,
        )

    def get_context(self, query: MemoryQuery | None = None) -> MemoryContext:
        return self.retrieve(query)

    def close_episode(self) -> None:
        """Clear all episode-local records, including staged actions and RGB summary."""
        try:
            if self._writer is not None:
                self.extractor.close_episode()
        finally:
            self._writer = None
            self._pending_action = None
            self._pending_prediction = None
            self.working.close()
            self.episodic.clear()
            self.scene_graph.clear()

    def _require_writer(self) -> MemoryWriter:
        if self._writer is None:
            raise RuntimeError("episode is not active; call reset_episode first")
        return self._writer
