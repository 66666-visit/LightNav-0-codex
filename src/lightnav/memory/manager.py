"""Coordinate legacy, V3, or selective V4 explicit memory backends."""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import asdict, replace
from time import perf_counter
from uuid import uuid4

from lightnav.memory.episodic_memory import EpisodicMemory
from lightnav.memory.memory_fusion import MemoryFusion
from lightnav.memory.memory_prompt_gate import MemoryPromptGate
from lightnav.memory.memory_retrieval_gate import MemoryRetrievalGate
from lightnav.memory.memory_write_gate import MemoryWriteGate
from lightnav.memory.memory_writer import MemoryWriter, NullSceneExtractor, SceneExtractor
from lightnav.memory.place_recognition import PlaceDescriptorExtractor
from lightnav.memory.place_verification import TwoStagePlaceVerifier
from lightnav.memory.retriever import MemoryRetriever
from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.scene_graph import SceneGraph
from lightnav.memory.schemas import (
    LayerMemoryItem,
    MemoryContext,
    MemoryEvent,
    MemoryQuery,
    MemorySelection,
    ObservationPacket,
    PlaceDecision,
)
from lightnav.memory.semantic_index import SemanticIndex
from lightnav.memory.topological_memory import PlaceObservationResult, TopologicalMemory
from lightnav.memory.v3_config import RGMV3Config
from lightnav.memory.v3_formatter import V3PromptFormatter
from lightnav.memory.v3_retrieval import V3Retriever
from lightnav.memory.v3_schemas import EvidenceRecord
from lightnav.memory.v3_store import V3MemoryStore
from lightnav.memory.v4_config import RGMV4Config
from lightnav.memory.v4_context import (
    DroppedCandidate,
    NavigationMemoryContext,
    build_navigation_memory_context,
)
from lightnav.memory.v4_formatter import V4PromptFormatter
from lightnav.memory.v4_navigation_state import NavigationStateUpdater
from lightnav.memory.v4_utilization import REVISIT_AUDIT, FrameKey, MemoryUtilizationManager
from lightnav.memory.working_memory import WorkingMemoryStore

_LAYERS = frozenset({"short", "mid", "long"})
logger = logging.getLogger(__name__)


class MemoryManager:
    """Select one memory backend; V3/V4 scopes control Scene retention."""

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
        rgm_config: RGMConfig | None = None,
        v3_config: RGMV3Config | None = None,
        v4_config: RGMV4Config | None = None,
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
        self.rgm_config = rgm_config or RGMConfig()
        self.v3_config = v3_config or RGMV3Config()
        self.v4_config = v4_config or RGMV4Config()
        if sum((self.rgm_config.enabled, self.v3_config.enabled, self.v4_config.enabled)) > 1:
            raise ValueError("legacy RGM, RGM V3 and RGM V4 cannot run together")
        if self.rgm_config.enabled and strategy != "hierarchical":
            raise ValueError("RGM requires hierarchical memory")
        if (self.v3_config.enabled or self.v4_config.enabled) and strategy != "hierarchical":
            raise ValueError("RGM V3/V4 requires hierarchical memory")
        if (self.v3_config.enabled or self.v4_config.enabled) and set(self.layers) != _LAYERS:
            raise ValueError("RGM V3/V4 requires short, mid and long memory layers")
        if self.rgm_config.enabled and self.rgm_config.place_two_stage_enabled and not place_enabled:
            raise ValueError("RGM two-stage place recognition requires place memory")
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
        self.v3_store = (
            V3MemoryStore(scope=(self.v4_config.scope if self.v4_config.enabled
                                 else self.v3_config.scope),
                          place_similarity_threshold=place_similarity_threshold)
            if self.v3_config.enabled or self.v4_config.enabled else None
        )
        self.v3_retriever = V3Retriever(self.v3_store) if self.v3_store else None
        self.v3_formatter = (
            V3PromptFormatter(max_records=self.v3_config.prompt_max_records,
                              max_chars=min(self.v3_config.prompt_max_chars,
                                            self.max_context_chars))
            if self.v3_config.enabled else None
        )
        self.v4_state_updater = (
            NavigationStateUpdater(self.v3_store, revisit_enabled=self.v4_config.revisit_enabled)
            if self.v4_config.enabled else None
        )
        self.v4_utilization = MemoryUtilizationManager() if self.v4_config.enabled else None
        self.v4_formatter = (
            V4PromptFormatter(max_records=self.v4_config.prompt_max_records,
                              max_chars=min(self.v4_config.prompt_max_chars,
                                            self.max_context_chars))
            if self.v4_config.enabled else None
        )
        self.v4_frame_ids_by_step: dict[FrameKey, int] = {}
        self.last_v4_context: NavigationMemoryContext | None = None
        self._v4_structured_context: NavigationMemoryContext | None = None
        self._v4_visible_frame_ids: tuple[int, ...] = ()
        self._v4_dropped_candidates: list[dict] = []
        self.last_v4_diagnostic: dict = {}
        self.v3_scene_id: str | None = None
        self.v3_rollout_idx = 0
        self.v3_place_id: str | None = None
        self.v3_descriptor: tuple[float, ...] | None = None
        self._v3_prior_place_id: str | None = None
        self._v3_pending_descriptors: dict[str, tuple[float, ...]] = {}
        self.last_v3_diagnostic: dict = {}
        self.last_layer_results: dict[str, list[LayerMemoryItem]] = {}
        self.last_place_result: PlaceObservationResult | None = None
        self.last_place_error: str | None = None
        self.last_place_extraction_attempted = False
        self.last_place_extraction_ms: float | None = None
        self.last_place_decision: PlaceDecision | None = None
        self.last_topology_edge_status: str | None = None
        self.write_gate: MemoryWriteGate | None = None
        self.retrieval_gate = MemoryRetrievalGate(enabled=self.rgm_config.retrieval_gate_enabled)
        self.prompt_gate = MemoryPromptGate(
            enabled=self.rgm_config.prompt_gate_enabled,
            max_records=self.rgm_config.prompt_max_records,
            max_chars=self.rgm_config.prompt_max_chars,
            step0_conditional=self.rgm_config.step0_conditional,
            min_current_confidence=self.rgm_config.min_instruction_landmark_confidence,
            temporal_min_support=self.rgm_config.temporal_min_support,
        )
        self.place_verifier = TwoStagePlaceVerifier(self.rgm_config)
        self._writer: MemoryWriter | None = None
        self._pending_action: dict | None = None
        self._pending_prediction: str | None = None

    @property
    def active(self) -> bool:
        return self._writer is not None

    def reset_episode(self, episode_id: str | None = None, instruction: str = "",
                      *, scene_id: str | None = None, rollout_idx: int = 0) -> None:
        """Begin an Episode; V3 retains prior Scene history only when configured."""
        self.close_episode()
        episode_id = episode_id or uuid4().hex
        if self.v3_store is not None:
            # Callers without a scene identity must remain isolated by Episode.
            self.v3_scene_id = scene_id or episode_id
            self.v3_rollout_idx = rollout_idx
            self.v3_place_id = None
            self.v3_descriptor = None
            self._v3_prior_place_id = None
            self._v3_pending_descriptors = {}
            self.v3_store.begin_episode(
                self.v3_scene_id, episode_id, instruction, rollout_idx=rollout_idx,
            )
        if self.v4_state_updater is not None:
            self.v4_state_updater.reset_episode(
                self.v3_scene_id, episode_id, instruction, rollout_idx=rollout_idx,
            )
            assert self.v4_utilization is not None
            self.v4_utilization.reset_episode(self.v3_scene_id, episode_id, rollout_idx)
            self.v4_frame_ids_by_step = {}
            self.last_v4_context = None
            self._v4_structured_context = None
            self._v4_visible_frame_ids = ()
            self._v4_dropped_candidates = []
            self.last_v4_diagnostic = {}
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
        self.last_place_decision = None
        self.last_topology_edge_status = None
        self.last_v3_diagnostic = {}
        self.write_gate = (
            MemoryWriteGate(episode_id, instruction, self.rgm_config)
            if self.rgm_config.enabled else None
        )
        self._writer = MemoryWriter(
            self.working, self.episodic, self.scene_graph, self.extractor,
            event_gate=self.write_gate.decide if self.write_gate else None,
            extraction_observer=self.write_gate.record_extraction if self.write_gate else None,
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
        self.last_place_decision = None
        self.last_topology_edge_status = None
        self.place_verifier.last_topk = []
        if self.v3_store is not None:
            self.v3_descriptor = None
            if MemoryWriter._has_positive_forward_command(action):
                # The command is not proof of travel. Drop the carried current
                # place until a new visual descriptor is observed.
                self.v3_place_id = None
        events = writer.observe(safe_packet)
        if self.strategy == "hierarchical":
            if "short" in self.layers:
                self.working.record_observation(packet.step_id, events)
            if "long" in self.layers and self.v3_store is None:
                adopted = (
                    [event for event in events if event.metadata.get("rgm_adopted", False)]
                    if self.rgm_config.enabled else events
                )
                self.semantic_index.update(adopted, self.scene_graph, self.working.state.instruction)
        if self.topology_active and packet.step_id % self.place_every_n_steps == 0:
            if self.v3_store is not None:
                self._observe_place_v3(safe_packet, events)
            elif self.rgm_config.enabled and self.rgm_config.place_two_stage_enabled:
                self._observe_place_rgm(safe_packet, events)
            else:
                self._observe_place(
                    safe_packet,
                    [event for event in events if event.metadata.get("rgm_adopted", False)]
                    if self.rgm_config.enabled else events,
                )
        if self.v3_store is not None:
            self._persist_v3_events(events)
        if self.v4_state_updater is not None:
            self.v4_state_updater.update(packet.step_id)
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

    def record_v4_frame_id(self, step: int, frame_id: int) -> None:
        """Bind one memory Step to the TrackingAgent frame actually observed."""
        if self.v4_state_updater is None or self.v4_state_updater.state is None:
            raise RuntimeError("V4 navigation state is not active")
        state = self.v4_state_updater.state
        if (type(step) is not int or step != state.last_update_step
                or type(frame_id) is not int or frame_id < 0):
            raise ValueError("V4 frame mapping requires the current step and a frame ID")
        key = (state.scene_id, state.episode_id, state.rollout_idx, step)
        previous = self.v4_frame_ids_by_step.get(key)
        if previous is not None and previous != frame_id:
            raise ValueError("memory step cannot be rebound to another RGB frame")
        self.v4_frame_ids_by_step[key] = frame_id

    def _record_v4_drops(self, *groups: tuple[DroppedCandidate, ...] | list[DroppedCandidate]) -> None:
        for group in groups:
            for item in group:
                record = asdict(item)
                if record not in self._v4_dropped_candidates:
                    self._v4_dropped_candidates.append(record)

    def _v4_history_fallback(
        self, selected: NavigationMemoryContext, formatted: MemoryContext | None, *,
        excluded_evidence_ids: frozenset[str] = frozenset(),
    ) -> tuple[NavigationMemoryContext, MemoryContext | None]:
        """Try relevant history if a whole revisit statement cannot fit."""
        if formatted is not None or selected.mode != REVISIT_AUDIT:
            return selected, formatted
        assert self._v4_structured_context is not None
        assert self.v4_utilization is not None and self.v4_formatter is not None
        assert self.v4_state_updater is not None and self.v4_state_updater.state is not None
        self.v4_utilization.discard_uninjected_selection()
        history_only = replace(self._v4_structured_context, revisit_evidence=())
        fallback = self.v4_utilization.select(
            history_only, self.v4_state_updater.state,
            visible_frame_ids=self._v4_visible_frame_ids,
            frame_ids_by_step=self.v4_frame_ids_by_step,
        )
        formatted = self.v4_formatter.format(
            fallback, excluded_evidence_ids=excluded_evidence_ids,
        )
        self._record_v4_drops(fallback.dropped_candidates, self.v4_formatter.last_dropped)
        return fallback, formatted

    def retrieve(self, query: MemoryQuery | None = None, *,
                 visible_frame_ids: tuple[int, ...] = ()) -> MemoryContext | None:
        self.working._require_state()
        query = query or self._default_query()
        if self.v4_config.enabled:
            assert self.v3_store is not None and self.v3_retriever is not None
            assert self.v4_state_updater is not None and self.v4_state_updater.state is not None
            assert self.v4_utilization is not None and self.v4_formatter is not None
            result = self.v3_retriever.retrieve(
                scene_id=self.v3_scene_id, instruction=query.instruction,
                current_step=query.current_step, visual_descriptor=self.v3_descriptor,
                place_id=self.v3_place_id, episode_id=self.working.state.episode_id,
                rollout_idx=self.v3_rollout_idx,
                top_k=max(query.top_k, self.v4_config.prompt_max_records * 3),
            )
            structured = build_navigation_memory_context(
                result, self.v3_store, self.v4_state_updater.state,
            )
            self._v4_structured_context = structured
            self._v4_visible_frame_ids = tuple(visible_frame_ids)
            self._v4_dropped_candidates = []
            selected = self.v4_utilization.select(
                structured, self.v4_state_updater.state,
                visible_frame_ids=visible_frame_ids,
                frame_ids_by_step=self.v4_frame_ids_by_step,
            )
            formatted = self.v4_formatter.format(selected)
            self._record_v4_drops(selected.dropped_candidates, self.v4_formatter.last_dropped)
            attempted_mode = selected.mode
            selected, formatted = self._v4_history_fallback(selected, formatted)
            self.last_v4_context = selected
            if formatted is None and selected.mode != "NONE":
                self.v4_utilization.discard_uninjected_selection()
            self.last_v4_diagnostic = {
                **result.counts,
                "navigation_progress_status": self.v4_state_updater.state.progress_status,
                "navigation_uncertainty": self.v4_state_updater.state.uncertainty,
                "memory_use_mode": selected.mode if formatted is not None else "NONE",
                "attempted_memory_use_mode": attempted_mode,
                "retrieval_candidate_memory_ids": list(selected.candidate_memory_ids),
                "visible_frame_ids": list(visible_frame_ids),
                "known_step_frame_mappings": len(self.v4_frame_ids_by_step),
                "selected_evidence_ids": list(dict.fromkeys(
                    ref.event_id for item in formatted.layer_items for ref in item.evidence_refs
                )) if formatted is not None else [],
                "prompt_records_serialized": self.v4_formatter.last_serialized,
                "prompt_chars": len(formatted.text) if formatted is not None else 0,
                "dropped_candidates": list(self._v4_dropped_candidates),
            }
            return formatted
        if self.v3_store is not None:
            assert self.v3_retriever is not None and self.v3_formatter is not None
            result = self.v3_retriever.retrieve(
                scene_id=self.v3_scene_id, instruction=query.instruction,
                current_step=query.current_step, visual_descriptor=self.v3_descriptor,
                place_id=self.v3_place_id, episode_id=self.working.state.episode_id,
                rollout_idx=self.v3_rollout_idx, top_k=query.top_k,
            )
            context = self.v3_formatter.format(result.records, current_step=query.current_step)
            active_identity = (self.working.state.episode_id, self.v3_rollout_idx)

            def is_historical_ref(ref: str) -> bool:
                record = self.v3_store.get_evidence(ref)
                return bool(record is not None and (
                    (record.episode_id, record.rollout_idx) != active_identity
                    or record.step < query.current_step
                ))

            historical_prompt_records = sum(
                any(is_historical_ref(ref.event_id) for ref in item.evidence_refs)
                for item in context.layer_items
            )
            historical_supported_records = sum(
                any(is_historical_ref(ref) for ref in
                    item.metadata.get("supporting_evidence_ids", ()))
                for item in context.layer_items
            )
            self.last_v3_diagnostic.update(result.counts)
            self.last_v3_diagnostic.update({
                "retrieval_nonempty": result.merged_candidates > 0,
                "fusion_nonempty": bool(result.records),
                "formatter_nonempty": bool(context.text and context.layer_items),
                "prompt_historical_records_serialized": historical_prompt_records,
                "prompt_historical_supported_records": historical_supported_records,
                "prompt_current_step_only_records_serialized": (
                    len(context.layer_items) - historical_prompt_records
                ),
                "prompt_records_selected": self.v3_formatter.last_selected,
                "prompt_records_considered": len(
                    self.v3_formatter.last_candidate_line_chars
                ),
                "prompt_records_serialized": self.v3_formatter.last_serialized,
                "prompt_chars": len(context.text),
                "prompt_candidate_line_chars": list(
                    self.v3_formatter.last_candidate_line_chars
                ),
                "prompt_serialized_line_chars": list(
                    self.v3_formatter.last_serialized_line_chars
                ),
                "effective_prompt_max_chars": self.v3_formatter.max_chars,
                "selected_evidence_ids": list(dict.fromkeys(
                    ref.event_id for item in context.layer_items for ref in item.evidence_refs
                )),
            })
            return context
        if self.strategy == "flat":
            self.last_layer_results = {}
            return self.retriever.retrieve(
                query, self.episodic.events, self.scene_graph.nodes, self.scene_graph.edges,
            )
        layers = self.retrieve_layers(query)
        if not self.rgm_config.enabled:
            return self.fusion.build(query, layers["short"], layers["mid"], layers["long"])
        accepted_ids = {
            event.event_id for event in self.episodic.events
            if event.metadata.get("rgm_adopted", False)
        }
        filtered = self.retrieval_gate.filter(query, layers, accepted_ids)
        fusion_query = MemoryQuery(
            instruction=query.instruction, current_step=query.current_step,
            place_id=query.place_id, top_k=max(query.top_k, self.rgm_config.prompt_max_records),
            max_chars=max(query.max_chars, 800),
        )
        fused = self.fusion.build(
            fusion_query, filtered["short"], filtered["mid"], filtered["long"],
        )
        support_by_event_id = {
            event.event_id: len(set(event.metadata.get("rgm_temporal_support", {}).get("steps", [])))
            for event in self.episodic.events
        }
        return self.prompt_gate.select(query, fused, accepted_ids, support_by_event_id)

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

    def get_context(self, query: MemoryQuery | None = None, *,
                    visible_frame_ids: tuple[int, ...] = ()) -> MemoryContext | None:
        return self.retrieve(query, visible_frame_ids=visible_frame_ids)

    def reformat_v4_context(self, excluded_evidence_ids: frozenset[str]) -> MemoryContext | None:
        """Retry a V4 token overflow with a whole evidence item removed."""
        if self.v4_formatter is None or self.last_v4_context is None:
            raise RuntimeError("no V4 selection available for reformatting")
        formatted = self.v4_formatter.format(
            self.last_v4_context, excluded_evidence_ids=excluded_evidence_ids,
        )
        self._record_v4_drops(
            self.last_v4_context.dropped_candidates, self.v4_formatter.last_dropped,
        )
        selected, formatted = self._v4_history_fallback(
            self.last_v4_context, formatted,
            excluded_evidence_ids=excluded_evidence_ids,
        )
        self.last_v4_context = selected
        self.last_v4_diagnostic.update({
            "memory_use_mode": selected.mode if formatted is not None else "NONE",
            "selected_evidence_ids": list(dict.fromkeys(
                ref.event_id for item in formatted.layer_items for ref in item.evidence_refs
            )) if formatted is not None else [],
            "prompt_records_serialized": self.v4_formatter.last_serialized,
            "prompt_chars": len(formatted.text) if formatted is not None else 0,
            "dropped_candidates": list(self._v4_dropped_candidates),
        })
        if formatted is None:
            assert self.v4_utilization is not None
            self.v4_utilization.discard_uninjected_selection()
        return formatted

    def close_episode(self, *, outcome: str = "unknown") -> None:
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
            if self.v3_store is not None:
                self.v3_store.close_episode(outcome=outcome)
                self.v3_scene_id = None
                self.v3_place_id = None
                self.v3_descriptor = None
                self._v3_prior_place_id = None
                self._v3_pending_descriptors = {}
            if self.v4_state_updater is not None:
                self.v4_state_updater.state = None
                self.v4_frame_ids_by_step = {}
                self.last_v4_context = None
                self._v4_structured_context = None
                self._v4_visible_frame_ids = ()
                self._v4_dropped_candidates = []
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
            self.last_place_decision = None
            self.last_topology_edge_status = None
            if self.write_gate is not None:
                self.write_gate.clear()
            self.write_gate = None
            self.place_verifier.clear()
            self.retrieval_gate.decisions = []
            self.prompt_gate.last_selection = MemorySelection((), (), (), "")
            self.last_v3_diagnostic = {}

    def _observe_place_v3(self, packet: ObservationPacket, events: list[MemoryEvent]) -> None:
        """Capture one DINO observation without claiming physical traversal."""
        assert self.place_extractor is not None
        self.last_place_extraction_attempted = True
        started = perf_counter()
        try:
            feature_method = getattr(self.place_extractor, "extract_features", None)
            features = (feature_method(packet) if feature_method is not None
                        else self.place_extractor.extract(packet))
            if features is None:
                return
            values = getattr(features, "global_descriptor", features)
            descriptor = tuple(float(value) for value in values)
            if not descriptor:
                return
            event_id = f"place_observation_{packet.step_id}"
            self._v3_pending_descriptors[event_id] = descriptor
            event = MemoryEvent(
                event_id=event_id, step_id=packet.step_id,
                kind="place_observation",
                description="Observed RGB visual place descriptor",
                confidence=1.0, source=self.place_extractor.source,
                importance=0.6, episode_id=self.working.state.episode_id,
                metadata={"descriptor_dim": len(descriptor)},
            )
            self.episodic.add(event)
            self.working.record_event(event)
            events.append(event)
            self.v3_descriptor = descriptor
        except Exception as exc:
            self.last_place_error = f"{type(exc).__name__}: {exc}"
            logger.warning("V3 place extraction failed at step=%s: %s",
                           packet.step_id, exc, exc_info=True)
        finally:
            self.last_place_extraction_ms = (perf_counter() - started) * 1000.0

    def _persist_v3_events(self, events: list[MemoryEvent]) -> None:
        """Write every valid observed event with a Scene/Episode-qualified key."""
        assert self.v3_store is not None and self.v3_scene_id is not None
        instruction = self.working.state.instruction
        instruction_words = set(re.findall(r"[a-z0-9]+", instruction.casefold()))
        direct_by_label: dict[str, str] = {}
        persisted = 0
        low_confidence = 0
        repeated = 0
        visible_landmarks: list[tuple[str, str]] = []
        for event in events:
            is_detection = event.kind in {"object_observation", "landmark"}
            label = (event.description.removeprefix("Observed ").strip()
                     if is_detection else None)
            payload = {"description": event.description}
            if event.kind in {"action", "turn"}:
                action = event.metadata.get("action")
                if isinstance(action, dict):
                    payload["action"] = deepcopy(action)
            source_refs: tuple[str, ...] = ()
            if event.kind == "instruction_match":
                matched_label = event.description.removeprefix(
                    "Observed instruction-related ",
                ).strip().casefold()
                if matched_label in direct_by_label:
                    source_refs = (direct_by_label[matched_label],)
            terms = tuple(sorted(instruction_words & set(
                re.findall(r"[a-z0-9]+", (label or event.description).casefold())
            )))
            record = EvidenceRecord(
                evidence_id=event.event_id, scene_id=self.v3_scene_id,
                episode_id=event.episode_id, rollout_idx=self.v3_rollout_idx,
                step=event.step_id, event_type=event.kind, semantic_label=label,
                payload=payload, visual_descriptor=self._v3_pending_descriptors.pop(
                    event.event_id, None,
                ), detector_name=event.source if is_detection else None,
                raw_detector_confidence=event.confidence if is_detection else None,
                instruction_terms=terms, source_evidence_ids=source_refs,
                observation_status=("observed" if is_detection or event.kind in {
                    "place_observation", "visual_change", "place_visit",
                } else "unknown"),
                spatial_status="observed" if event.kind == "place_observation" else "unknown",
                traversability_status="unknown",
                provenance_validity=(event.kind != "instruction_match" or bool(source_refs)),
            )
            stored = self.v3_store.add(record)
            persisted += 1
            if (stored.raw_detector_confidence is not None
                    and stored.raw_detector_confidence < self.v3_config.low_confidence_threshold):
                low_confidence += 1
            if stored.support_count > 1:
                repeated += 1
            if is_detection:
                direct_by_label[label.casefold()] = stored.key
                if "long" in self.layers:
                    landmark = self.v3_store.add_landmark(stored)
                    visible_landmarks.append((landmark.node_id, stored.key))
            elif event.kind == "place_observation" and "long" in self.layers:
                node = self.v3_store.add_place(stored)
                for landmark_id, landmark_key in visible_landmarks:
                    self.v3_store.add_relation(
                        node.node_id, landmark_id, "inferred_relation",
                        evidence_keys=(stored.key, landmark_key),
                    )
                prior = self._v3_prior_place_id
                if prior is not None and prior != node.node_id:
                    prior_node = next((item for item in self.v3_store.nodes_for_scene(
                        self.v3_scene_id,
                    ) if item.node_id == prior), None)
                    if prior_node is not None and prior_node.evidence_ids:
                        self.v3_store.add_relation(
                            prior, node.node_id, "inferred_relation",
                            evidence_keys=(prior_node.evidence_ids[-1], stored.key),
                        )
                self.v3_place_id = node.node_id
                self._v3_prior_place_id = node.node_id
        self.last_v3_diagnostic = {
            "events_observed": len(events), "events_parsed": len(events),
            "events_persisted": persisted,
            "short_memory_records": len(self.v3_store.current_short),
            "mid_memory_records": (
                len(self.v3_store.episodes_for_scene(self.v3_scene_id))
                + int(self.v3_store.active_segment(
                    before_step=self.working.state.step_id,
                ) is not None)
            ),
            "long_memory_nodes": len(self.v3_store.nodes_for_scene(self.v3_scene_id)),
            "long_memory_edges": len(self.v3_store.edges_for_scene(self.v3_scene_id)),
            "promoted_records": self.v3_store.promoted_records,
            "updated_records": self.v3_store.updated_records,
            "low_confidence_records": low_confidence,
            "repeated_observations": repeated,
            "spatial_unknown_records": sum(
                edge.traversability_status == "unknown"
                for edge in self.v3_store.edges_for_scene(self.v3_scene_id)
            ),
            "verified_transitions": sum(
                edge.traversability_status in {"traversed", "verified"}
                for edge in self.v3_store.edges_for_scene(self.v3_scene_id)
            ),
            "failed_experiences": sum(
                episode.outcome == "failure"
                for episode in self.v3_store.episodes_for_scene(self.v3_scene_id)
            ),
        }

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
                episode_id=self.working.state.episode_id,
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

    def _observe_place_rgm(self, packet: ObservationPacket, events: list[MemoryEvent]) -> None:
        """Use one DINO forward and abstain from uncertain spatial updates."""
        assert self.place_extractor is not None
        self.last_place_extraction_attempted = True
        started = perf_counter()
        try:
            feature_method = getattr(self.place_extractor, "extract_features", None)
            if feature_method is None:
                self.last_place_decision = PlaceDecision(
                    "UNKNOWN", None, None, None, 0, None, "patch_extractor_unavailable",
                )
                return
            features = feature_method(packet)
            if features is None:
                self.last_place_decision = PlaceDecision(
                    "UNKNOWN", None, None, None, 0, None, "no_visual_features",
                )
                return
            decision = self.place_verifier.decide(packet.step_id, features, self.topological_memory)
            self.last_place_decision = decision
            evidence_id = f"place_observation_{packet.step_id}"
            visual_change = self._writer.last_appearance_delta or 0.0
            prior_step = self.topological_memory.last_observation_step
            motion_ids = [
                event.event_id for event in self.episodic.events
                if event.kind in {"action", "turn"}
                and (prior_step is None or prior_step < event.step_id <= packet.step_id)
                and event.source == "issued_command_and_next_observation"
                and MemoryWriter._has_positive_forward_command(event.metadata.get("action"))
            ]
            motion_valid = bool(motion_ids) and (
                not self.rgm_config.topology_motion_validation_enabled
                or visual_change >= self.rgm_config.topology_visual_delta_min
            )
            result = None
            if decision.status in {"MATCH", "NEW"}:
                result = self.topological_memory.observe(
                    packet.step_id, features.global_descriptor,
                    [event for event in events if event.metadata.get("rgm_adopted", False)],
                    descriptor_source=self.place_extractor.source,
                    observation_event_id=evidence_id,
                    transition_evidence_ids=motion_ids if motion_valid else (),
                    force_match_place_id=decision.candidate_place_id
                    if decision.status == "MATCH" else None,
                    force_new=decision.status == "NEW",
                )
                self.place_verifier.accept(result.place_id, features)
                self.last_place_result = result
            self.last_topology_edge_status = (
                "OBSERVED_TRANSITION" if result and result.transition else
                "VISUAL_SHIFT_CANDIDATE" if visual_change >= self.rgm_config.topology_visual_delta_min
                else "NONE"
            )
            event = MemoryEvent(
                event_id=evidence_id, step_id=packet.step_id, kind="place_observation",
                description="Observed RGB visual place descriptor", confidence=1.0,
                source=self.place_extractor.source, importance=0.6,
                episode_id=self.working.state.episode_id,
                metadata={
                    "rgm_adopted": decision.status != "UNKNOWN", "place_decision": decision.status,
                    "reason": decision.reason, "topological_place_id": result.place_id if result else None,
                    "topology_edge_status": self.last_topology_edge_status,
                    "verified_traversal": False, "pose": "unavailable", "collision": "unavailable",
                },
            )
            self.episodic.add(event)
            self.working.record_event(event)
            events.append(event)
        except Exception as exc:
            self.last_place_error = f"{type(exc).__name__}: {exc}"
            self.last_place_decision = PlaceDecision(
                "UNKNOWN", None, None, None, 0, None, "place_verification_error",
            )
            logger.warning("RGM place verification failed at step=%s: %s",
                           packet.step_id, exc, exc_info=True)
        finally:
            self.last_place_extraction_ms = (perf_counter() - started) * 1000.0

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
