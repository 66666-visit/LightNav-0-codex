"""Coordinate working, episodic, and graph memory for one R2R episode at a time."""

from __future__ import annotations

import logging
from copy import deepcopy
from time import perf_counter
from uuid import uuid4

from lightnav.memory.episodic_memory import EpisodicMemory
from lightnav.memory.memory_fusion import MemoryFusion
from lightnav.memory.memory_writer import MemoryWriter, NullSceneExtractor, SceneExtractor
from lightnav.memory.place_recognition import PlaceDescriptorExtractor
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
from lightnav.memory.topological_memory import PlaceObservationResult, TopologicalMemory
from lightnav.memory.working_memory import WorkingMemoryStore

_LAYERS = frozenset({"short", "mid", "long"})
logger = logging.getLogger(__name__)


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
        place_enabled: bool = False,
        place_extractor: PlaceDescriptorExtractor | None = None,
        place_every_n_steps: int = 5,
        place_similarity_threshold: float = 0.85,
    ) -> None:
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if max_context_chars < 0:
            raise ValueError("max_context_chars must be nonnegative")
        if strategy not in {"flat", "hierarchical"}:
            raise ValueError("memory strategy must be 'flat' or 'hierarchical'")
        if len(set(layers)) != len(layers) or any(layer not in _LAYERS for layer in layers):
            raise ValueError("memory layers must be a unique subset of short, mid, long")
        if not isinstance(place_every_n_steps, int) or place_every_n_steps < 1:
            raise ValueError("place_every_n_steps must be a positive integer")
        if not 0.0 < place_similarity_threshold <= 1.0:
            raise ValueError("place_similarity_threshold must be in (0, 1]")
        if (
            place_enabled and place_extractor is None
            and strategy == "hierarchical" and "long" in layers
        ):
            raise ValueError("enabled topological place memory requires a place_extractor")
        self.extractor = extractor if extractor is not None else NullSceneExtractor()
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.strategy = strategy
        self.layers = tuple(layers)
        self.place_enabled = bool(place_enabled)
        self.place_extractor = place_extractor
        self.place_every_n_steps = place_every_n_steps
        self.topology_active = (
            self.place_enabled and strategy == "hierarchical" and "long" in layers
        )
        self.working = WorkingMemoryStore()
        self.episodic = EpisodicMemory(
            index_enabled=strategy == "hierarchical" and "mid" in self.layers,
        )
        self.scene_graph = SceneGraph()
        self.semantic_index = SemanticIndex()
        self.topological_memory = TopologicalMemory(similarity_threshold=place_similarity_threshold)
        self.retriever = MemoryRetriever()
        self.fusion = MemoryFusion()
        self.last_layer_results: dict[str, list[LayerMemoryItem]] = {}
        self.last_place_result: PlaceObservationResult | None = None
        self.last_place_error: str | None = None
        self.last_place_extraction_attempted = False
        self.last_place_extraction_ms: float | None = None
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
        if self.topology_active:
            assert self.place_extractor is not None
            self.place_extractor.reset_episode(episode_id, instruction)
        self.working.reset(episode_id, instruction)
        self.last_layer_results = {}
        self.last_place_result = None
        self.last_place_error = None
        self.last_place_extraction_attempted = False
        self.last_place_extraction_ms = None
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
        self.last_place_result = None
        self.last_place_error = None
        self.last_place_extraction_attempted = False
        self.last_place_extraction_ms = None
        events = writer.observe(safe_packet)
        if self.strategy == "hierarchical":
            if "short" in self.layers:
                self.working.record_observation(packet.step_id, events)
            if "long" in self.layers:
                self.semantic_index.update(events, self.scene_graph, self.working.state.instruction)
        if self.topology_active and packet.step_id % self.place_every_n_steps == 0:
            self._observe_place(safe_packet, events)
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
            "long": (
                self.topological_memory.retrieve(query) if self.topology_active
                else self.semantic_index.retrieve(query)
            ) if "long" in self.layers else [],
        }
        self.last_layer_results = results
        return results

    def get_context(self, query: MemoryQuery | None = None) -> MemoryContext:
        return self.retrieve(query)

    def close_episode(self) -> None:
        """Clear all episode-local records, including staged actions and RGB summary."""
        try:
            if self._writer is not None:
                try:
                    self.extractor.close_episode()
                finally:
                    if self.topology_active:
                        assert self.place_extractor is not None
                        self.place_extractor.close_episode()
        finally:
            self._writer = None
            self._pending_action = None
            self._pending_prediction = None
            self.working.close()
            self.episodic.clear()
            self.scene_graph.clear()
            self.semantic_index.clear()
            self.topological_memory.clear()
            self.last_layer_results = {}
            self.last_place_result = None
            self.last_place_error = None
            self.last_place_extraction_attempted = False
            self.last_place_extraction_ms = None

    def _observe_place(self, packet: ObservationPacket, events: list[MemoryEvent]) -> None:
        """Index one current RGB descriptor; never use Habitat evaluator information."""
        assert self.place_extractor is not None
        try:
            self.last_place_extraction_attempted = True
            started = perf_counter()
            try:
                descriptor = self.place_extractor.extract(packet)
            finally:
                self.last_place_extraction_ms = (perf_counter() - started) * 1000.0
            if descriptor is None:
                return
            prior_step = self.topological_memory.last_observation_step
            motion_ids = [
                event.event_id for event in self.episodic.events
                if event.kind in {"action", "turn"}
                and (prior_step is None or prior_step < event.step_id <= packet.step_id)
                and event.source == "issued_command_and_next_observation"
                and MemoryWriter._has_forward_command(event.metadata.get("action"))
            ]
            evidence_id = f"place_observation_{packet.step_id}"
            result = self.topological_memory.observe(
                packet.step_id, descriptor, events,
                descriptor_source=self.place_extractor.source,
                observation_event_id=evidence_id,
                transition_evidence_ids=motion_ids,
            )
            observation_event = MemoryEvent(
                event_id=evidence_id,
                step_id=packet.step_id,
                kind="place_observation",
                description="Observed RGB visual place descriptor",
                confidence=1.0,
                source=self.place_extractor.source,
                importance=0.6,
                metadata={
                    "topological_place_id": result.place_id,
                    "descriptor_dim": len(descriptor),
                    "top1_similarity": result.top1_similarity,
                    "place_matched": result.matched,
                    "place_created": result.created,
                    "place_revisited": result.revisited,
                },
            )
            self.episodic.add(observation_event)
            self.working.record_event(observation_event)
            events.append(observation_event)
            self.last_place_result = result
        except Exception as exc:
            self.last_place_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "place descriptor/update failed at step=%s source=%s: %s",
                packet.step_id, self.place_extractor.source, exc, exc_info=True,
            )

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
