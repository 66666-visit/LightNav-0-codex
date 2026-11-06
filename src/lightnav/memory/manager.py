"""Coordinate working, episodic, and graph memory for one R2R episode at a time."""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from lightnav.memory.episodic_memory import EpisodicMemory
from lightnav.memory.memory_fusion import MemoryFusion
from lightnav.memory.memory_writer import MemoryWriter, NullSceneExtractor, SceneExtractor
from lightnav.memory.retriever import MemoryRetriever
from lightnav.memory.scene_graph import SceneGraph
from lightnav.memory.schemas import (
    LayerMemoryItem,
    MemoryContext,
    MemoryEvent,
    MemoryQuery,
    ObservationPacket,
)
from lightnav.memory.semantic_index import SemanticIndex
from lightnav.memory.working_memory import WorkingMemoryStore

_LAYERS = frozenset({"short", "mid", "long"})


class MemoryManager:
    """Episode-local memory. No records survive ``close_episode`` or a new reset."""

    def __init__(
        self,
        extractor: SceneExtractor | None = None,
        *,
        top_k: int = 5,
        max_context_chars: int = 600,
        strategy: str = "flat",
        layers: tuple[str, ...] = ("short", "mid", "long"),
    ) -> None:
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if max_context_chars < 0:
            raise ValueError("max_context_chars must be nonnegative")
        if strategy not in {"flat", "hierarchical"}:
            raise ValueError("memory strategy must be 'flat' or 'hierarchical'")
        if len(set(layers)) != len(layers) or any(layer not in _LAYERS for layer in layers):
            raise ValueError("memory layers must be a unique subset of short, mid, long")
        self.extractor = extractor if extractor is not None else NullSceneExtractor()
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.strategy = strategy
        self.layers = tuple(layers)
        self.working = WorkingMemoryStore()
        self.episodic = EpisodicMemory(
            index_enabled=strategy == "hierarchical" and "mid" in self.layers,
        )
        self.scene_graph = SceneGraph()
        self.semantic_index = SemanticIndex()
        self.retriever = MemoryRetriever()
        self.fusion = MemoryFusion()
        self.last_layer_results: dict[str, list[LayerMemoryItem]] = {}
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
        self.last_layer_results = {}
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
        if self.strategy == "hierarchical":
            if "short" in self.layers:
                self.working.record_observation(packet.step_id, events)
            if "long" in self.layers:
                self.semantic_index.update(events, self.scene_graph, self.working.state.instruction)
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
        self.working._require_state()
        query = query or self._default_query()
        if self.strategy == "flat":
            self.last_layer_results = {}
            return self.retriever.retrieve(
                query, self.episodic.events, self.scene_graph.nodes, self.scene_graph.edges,
            )
        layers = self.retrieve_layers(query)
        return self.fusion.build(query, layers["short"], layers["mid"], layers["long"])

    def retrieve_layers(self, query: MemoryQuery | None = None) -> dict[str, list[LayerMemoryItem]]:
        """Run each enabled hierarchy's own query before cross-layer fusion."""
        if self.strategy != "hierarchical":
            raise RuntimeError("layer retrieval requires hierarchical memory strategy")
        self.working._require_state()
        query = query or self._default_query()
        results = {
            "short": self.working.retrieve(query) if "short" in self.layers else [],
            "mid": self.episodic.retrieve(query) if "mid" in self.layers else [],
            "long": self.semantic_index.retrieve(query) if "long" in self.layers else [],
        }
        self.last_layer_results = results
        return results

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
            self.semantic_index.clear()
            self.last_layer_results = {}

    def _default_query(self) -> MemoryQuery:
        state = self.working._require_state()
        return MemoryQuery(
            instruction=state.instruction, current_step=state.step_id,
            place_id=state.place_id, top_k=self.top_k,
            max_chars=self.max_context_chars,
        )

    def _require_writer(self) -> MemoryWriter:
        if self._writer is None:
            raise RuntimeError("episode is not active; call reset_episode first")
        return self._writer
