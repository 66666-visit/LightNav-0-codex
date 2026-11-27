"""Coordinate legacy, V3, or selective V4 explicit memory backends."""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import asdict, replace
from html import escape
from time import perf_counter
from uuid import uuid4

import numpy as np

from lightnav.memory.adaptive_l3_query import AdaptiveL3QueryRunner
from lightnav.memory.adaptive_l3_query_config import AdaptiveL3QueryConfig
from lightnav.memory.clip_features import clip_crop_boxes
from lightnav.memory.conceptgraphs_lite import ObjectMap, RGBDFrame
from lightnav.memory.episodic_memory import EpisodicMemory
from lightnav.memory.instruction_landmarks import (
    InstructionLandmarks,
    parse_instruction_landmarks,
)
from lightnav.memory.joint_prompt import JointPromptBuilder
from lightnav.memory.l1_prompt_organizer import L1PromptOrganizer
from lightnav.memory.l3_clip_query import L3ClipQuery, L3QueryResult
from lightnav.memory.l3_evidence_selector import (
    build_proposal_eligibility,
    choose_first_eligible,
    eligible_object_ids,
    render_context,
    resolve_geometry_fallback,
)
from lightnav.memory.l3_on_demand import L3QueryController, build_l3_query_context
from lightnav.memory.l3_prompt_organizer import L3PromptOrganizer
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.local_graph_fusion import LocalGraphFusion
from lightnav.memory.memory_fusion import MemoryFusion
from lightnav.memory.memory_prompt_gate import MemoryPromptGate
from lightnav.memory.memory_retrieval_gate import MemoryRetrievalGate
from lightnav.memory.memory_write_gate import MemoryWriteGate
from lightnav.memory.memory_writer import MemoryWriter, NullSceneExtractor, SceneExtractor
from lightnav.memory.ordered_landmark_memory import (
    LandmarkMemoryRow,
    build_ordered_memory_context,
    injected_evidence_ids,
    render_ordered_landmark_memory,
)
from lightnav.memory.place_recognition import PlaceDescriptorExtractor
from lightnav.memory.place_verification import TwoStagePlaceVerifier
from lightnav.memory.prompt_assembly import PromptContextResult, build_prompt_context
from lightnav.memory.prompt_modes import (
    PromptModePlan,
    mode_plan,
    resolve_prompt_budget,
    validate_mode_backing,
)
from lightnav.memory.retriever import MemoryRetriever
from lightnav.memory.rgb_sidecar import RGBSidecar
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
from lightnav.memory.spatial_graph_adapter import SpatialGraphAdapter, make_spatial_evidence
from lightnav.memory.spatial_l3_config import SpatialL3Config
from lightnav.memory.token_budget import (
    UNBOUNDED_CHAR_LIMIT,
    MemoryTokenBudget,
    TokenBudgetFailure,
)
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
_NAVIGATION_MEMORY_HEADER = "[Navigation Memory]"
logger = logging.getLogger(__name__)


def _spatial_match_payload(
    item: LayerMemoryItem, *, selection_source: str | None = None,
) -> dict:
    """One existing geometry fact, reported without changing its text."""
    metadata = item.metadata
    return {
        "object_id": metadata.get("object_id"),
        "label": item.label,
        "evidence_ids": list(metadata.get("evidence_ids", [])),
        "confidence": item.confidence,
        "observation_count": metadata.get("observation_count"),
        "center_world": metadata.get("center_world"),
        "distance_m": metadata.get("current_relative_distance_m"),
        "bearing_deg": metadata.get("current_relative_bearing_deg"),
        "selection_source": selection_source,
    }


def _render_original_spatial_fact(item: LayerMemoryItem) -> str | None:
    """Return F's exact ``render_context`` sentence for one existing item.

    This deliberately calls the original formatter instead of copying its
    ``Spatial history: ...`` template.  Only the single fact body is returned;
    the ordered renderer keeps G's role/phrase prefix around it.
    """
    text, included = render_context(
        [item], header=_NAVIGATION_MEMORY_HEADER,
        max_chars=UNBOUNDED_CHAR_LIMIT,
    )
    if text is None or not included:
        return None
    _, separator, line = text.partition("\n")
    return line if separator else None


def _grounding_failure_reason(eligibility: list, result: object) -> str:
    """Name the concrete reason no eligible proposal remained."""
    refusal = getattr(result, "refusal_reason", None)
    error = getattr(result, "error", None)
    if error is not None:
        return refusal or "grounding_error"
    if not eligibility:
        return refusal or "no_scored_proposal"
    if not any(proposal.formatable for proposal in eligibility):
        return "no_formatable_grounding_candidate"
    if not any(proposal.class_match for proposal in eligibility):
        return "no_class_matching_proposal"
    if any(
        not proposal.object_node_valid or not proposal.evidence_mapping_valid
        for proposal in eligibility
    ):
        return "invalid_evidence_mapping"
    return refusal or "no_eligible_proposal"


def _landmark_identity(landmark: object | None) -> dict:
    if landmark is None:
        return {
            "landmark_order": 0,
            "landmark_role": None,
            "landmark_phrase": None,
            "target_object": None,
            "action_phrase": None,
            "grounding_query": None,
        }
    return {
        "landmark_order": landmark.landmark_order,
        "landmark_role": landmark.navigation_role,
        "landmark_phrase": landmark.target_phrase,
        "target_object": landmark.target_object,
        "action_phrase": landmark.action_phrase,
        "grounding_query": landmark.grounding_query,
    }


def _unexecuted_landmark_rows(
    landmarks: InstructionLandmarks | None, reason: str,
) -> list[LandmarkMemoryRow]:
    """One explainable row per landmark when no grounder query ran this step."""
    if landmarks is None or not landmarks.landmarks:
        return []
    rows: list[LandmarkMemoryRow] = []
    for landmark in landmarks.landmarks:
        rows.append(LandmarkMemoryRow(
            landmark_order=landmark.landmark_order,
            landmark_role=landmark.navigation_role,
            landmark_phrase=landmark.target_phrase,
            target_object=landmark.target_object,
            action_phrase=landmark.action_phrase,
            grounding_query=landmark.grounding_query,
            query_executed=False,
            query_skip_reason=reason,
            geometry_fallback_reason=reason,
        ))
    return rows


def _empty_l3_query_stats() -> dict:
    return {
        "steps": 0,
        "l1_built": 0,
        "l1_no_candidates": 0,
        "l1_budget_excluded": 0,
        "l1_reactivation_suppressed": 0,
        "l1_other": 0,
        "queries_attempted": 0,
        "queries_with_match": 0,
        "queries_without_match": 0,
        "query_errors": 0,
        "prompt_budget_excluded": 0,
        "skipped_reasons": {},
    }


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
        local_graph_config: LocalGraphConfig | None = None,
        spatial_l3_config: SpatialL3Config | None = None,
        spatial_extractor: object | None = None,
        clip_encoder: object | None = None,
        prompt_mode: str = "legacy",
        adaptive_l3_query_config: AdaptiveL3QueryConfig | None = None,
        instance_grounder: object | None = None,
        instruction_landmark_provider: object | None = None,
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
        self.local_graph_config = local_graph_config or LocalGraphConfig()
        self.spatial_l3_config = spatial_l3_config or SpatialL3Config()
        self.plan: PromptModePlan = mode_plan(prompt_mode)
        validate_mode_backing(
            self.plan,
            local_graph_enabled=self.local_graph_config.enabled,
            spatial_l3_mode=self.spatial_l3_config.mode,
        )
        self.prompt_budget_chars = resolve_prompt_budget(
            max_context_chars=self.max_context_chars,
            local_graph_prompt_max_chars=self.local_graph_config.prompt_max_chars,
        )
        self.l1_prompt_organizer = L1PromptOrganizer(
            max_items=self.local_graph_config.max_landmarks,
        )
        self.l3_prompt_organizer = L3PromptOrganizer(
            max_items=self.local_graph_config.max_landmarks,
        )
        self.joint_prompt_builder = JointPromptBuilder(
            max_items=self.local_graph_config.max_landmarks + 1,
        )
        self._prompt_context_pending = False
        self._prompt_context_pending_ids: frozenset[str] = frozenset()
        self.last_prompt_context: PromptContextResult | None = None
        if self.spatial_l3_config.mode != "off" and not self.local_graph_config.enabled:
            raise ValueError("spatial L3 requires Local Graph to be enabled")
        if self.spatial_l3_config.mode != "off" and spatial_extractor is None:
            raise ValueError("spatial L3 requires an independent segmentation extractor")
        if self.spatial_l3_config.mode in {
            "on_demand", "geometry_on_demand",
        } and self.plan.mode not in ("legacy", "l1_original_only"):
            raise ValueError(
                "on-demand L3 requires legacy or l1_original_only L1 prompt path"
            )
        if self.spatial_l3_config.mode == "on_demand" and clip_encoder is None:
            raise ValueError("on_demand L3 requires a local CLIP encoder")
        if clip_encoder is not None:
            for method in ("encode_image", "encode_text"):
                if not callable(getattr(clip_encoder, method, None)):
                    raise ValueError(f"CLIP encoder is missing {method}()")
        self.spatial_extractor = spatial_extractor
        self.clip_encoder = clip_encoder
        self.spatial_map = (
            ObjectMap(self.spatial_l3_config.geometry)
            if self.spatial_l3_config.mode != "off" else None
        )
        self.spatial_adapter = (
            SpatialGraphAdapter() if self.spatial_map is not None else None
        )
        # Read-only XYZRGB recorder for the opt-in ViL3DRel grounder.  It is
        # created only when explicitly enabled; it never mutates ObjectNode.
        self.rgb_sidecar = (
            RGBSidecar(self.spatial_l3_config.geometry)
            if self.spatial_l3_config.rgb_sidecar_enabled else None
        )
        if self.spatial_l3_config.instance_grounder_enabled and self.rgb_sidecar is None:
            raise ValueError("instance grounder requires the RGB sidecar")
        self.instance_grounder = instance_grounder
        self._instance_grounder_error: str | None = None
        # Parsed once per Episode (never per query) when the grounder is on.
        # The provider hook lets an existing DeepSeek/OpenAI-compatible parser
        # be reused; no provider is bundled and failure always falls back.
        if instruction_landmark_provider is not None and not callable(
            getattr(instruction_landmark_provider, "extract", None)
        ):
            raise TypeError("instruction_landmark_provider must provide extract()")
        self.instruction_landmark_provider = instruction_landmark_provider
        self.instruction_landmarks: InstructionLandmarks | None = None
        if self.instance_grounder is not None and not callable(
            getattr(self.instance_grounder, "select", None)
        ):
            raise TypeError("instance_grounder must provide select()")
        if (
            self.spatial_l3_config.mode == "geometry_on_demand"
            and clip_encoder is not None
        ):
            raise ValueError(
                "geometry_on_demand must not receive a CLIP encoder; it uses "
                "the existing geometry-only spatial retrieval"
            )
        self.l3_query_controller = (
            L3QueryController(
                clip_encoder,
                top_k=self.spatial_l3_config.clip_top_k,
                min_similarity=self.spatial_l3_config.clip_min_similarity,
            )
            if self.spatial_l3_config.mode == "on_demand" else None
        )
        # Decision-only instance of the original on-demand controller.  It owns
        # the L1-empty gate and dedupe state; ``maybe_query`` is never called, so
        # no CLIP encoder, weight or similarity computation is involved.
        self.geometry_on_demand_controller = (
            L3QueryController(None, top_k=1)
            if self.spatial_l3_config.mode == "geometry_on_demand" else None
        )
        self.last_local_graph_status = "not_attempted"
        self.last_l3_query_diagnostic: dict = {}
        self.adaptive_l3_query_config = (
            adaptive_l3_query_config or AdaptiveL3QueryConfig()
        )
        if not isinstance(self.adaptive_l3_query_config, AdaptiveL3QueryConfig):
            raise ValueError(
                "adaptive_l3_query_config must be an AdaptiveL3QueryConfig"
            )
        if self.adaptive_l3_query_config.enabled:
            if not self.local_graph_config.enabled:
                raise ValueError("adaptive L3 query requires Local Graph memory")
            if self.spatial_l3_config.mode != "shadow":
                # shadow builds the read-only map without injecting spatial
                # candidates; inject/on_demand would bypass or duplicate L1.
                raise ValueError(
                    "adaptive L3 query requires spatial_l3.mode=shadow so the "
                    "original L1 path stays primary"
                )
            if clip_encoder is None:
                raise ValueError("adaptive L3 query requires a local CLIP encoder")
            if self.plan.mode not in {"legacy", "l1_original_only"}:
                raise ValueError(
                    "adaptive L3 query requires the legacy or l1_original_only "
                    "L1 prompt path"
                )
        self.adaptive_l3_runner: AdaptiveL3QueryRunner | None = None
        self.last_adaptive_l3_diagnostic: dict = {}
        self.l3_query_stats = _empty_l3_query_stats()
        self.last_episode_l3_query_stats: dict = {}
        self._l3_stats_step: int | None = None
        self.last_spatial_frame: RGBDFrame | None = None
        self.last_spatial_diagnostic: dict = {}
        if sum((self.rgm_config.enabled, self.v3_config.enabled,
                self.v4_config.enabled, self.local_graph_config.enabled)) > 1:
            raise ValueError("legacy RGM, RGM V3, RGM V4 and local graph cannot run together")
        if self.plan.mode != "legacy" and (
            self.rgm_config.enabled or self.v3_config.enabled or self.v4_config.enabled
        ):
            raise ValueError("memory prompt modes require the Local Graph backend only")
        if self.rgm_config.enabled and strategy != "hierarchical":
            raise ValueError("RGM requires hierarchical memory")
        if (self.v3_config.enabled or self.v4_config.enabled
                or self.local_graph_config.enabled) and strategy != "hierarchical":
            raise ValueError("RGM V3/V4/local graph requires hierarchical memory")
        if (self.v3_config.enabled or self.v4_config.enabled
                or self.local_graph_config.enabled) and set(self.layers) != _LAYERS:
            raise ValueError("RGM V3/V4/local graph requires short, mid and long memory layers")
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
            V3MemoryStore(scope=(self.local_graph_config.scope if self.local_graph_config.enabled
                                 else self.v4_config.scope if self.v4_config.enabled
                                 else self.v3_config.scope),
                          place_similarity_threshold=place_similarity_threshold)
            if (self.v3_config.enabled or self.v4_config.enabled
                or self.local_graph_config.enabled) else None
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
        self.local_graph_fusion = (
            LocalGraphFusion(self.local_graph_config)
            if self.local_graph_config.enabled else None
        )
        # Experimental modes apply the model-derived token limit instead of the
        # original 320-character module budget. The original builder code is
        # reused; only its internal character ceiling is lifted.
        self.prompt_local_graph_fusion = (
            LocalGraphFusion(replace(
                self.local_graph_config, prompt_max_chars=UNBOUNDED_CHAR_LIMIT,
            ))
            if self.local_graph_config.enabled and self.plan.mode != "legacy"
            else self.local_graph_fusion
        )
        self._active_prompt_fusion = self.prompt_local_graph_fusion
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
        if self.local_graph_fusion is not None:
            self.local_graph_fusion.reset_episode(
                self.v3_scene_id, episode_id, rollout_idx,
            )
        if (self.prompt_local_graph_fusion is not None
                and self.prompt_local_graph_fusion is not self.local_graph_fusion):
            self.prompt_local_graph_fusion.reset_episode(
                self.v3_scene_id, episode_id, rollout_idx,
            )
        if self.spatial_map is not None:
            self.spatial_map.reset_episode(self.v3_scene_id, episode_id, rollout_idx)
            self.last_spatial_frame = None
            self.last_spatial_diagnostic = {}
            self.spatial_extractor.reset_episode(episode_id, instruction)
        if self.l3_query_controller is not None:
            self.l3_query_controller.reset_episode(
                self.v3_scene_id, episode_id, rollout_idx,
            )
        if self.geometry_on_demand_controller is not None:
            self.geometry_on_demand_controller.reset_episode(
                self.v3_scene_id, episode_id, rollout_idx,
            )
        if self.rgb_sidecar is not None:
            self.rgb_sidecar.reset_episode(
                self.v3_scene_id, episode_id, rollout_idx,
            )
        # Parse the original instruction once at Episode initialization and
        # cache it.  A provider is optional; the local extractor is used when
        # none is configured, and every phrase is checked against the source.
        self.instruction_landmarks = None
        if (
            self.spatial_l3_config.instance_grounder_enabled
            and self.spatial_l3_config.grounder_landmark_queries
            and instruction.strip()
        ):
            try:
                self.instruction_landmarks = parse_instruction_landmarks(
                    instruction, provider=self.instruction_landmark_provider,
                )
            except Exception as exc:  # noqa: BLE001 - parsing must never fail nav
                logger.warning(
                    "instruction landmark parsing failed; using the full "
                    "instruction for grounding: %s", exc,
                )
                self.instruction_landmarks = InstructionLandmarks(
                    instruction=instruction, landmarks=(),
                    source="parser_error_fallback",
                )
        if self.adaptive_l3_runner is not None:
            self.adaptive_l3_runner.reset_episode(
                self.v3_scene_id, episode_id, rollout_idx,
            )
        self.last_adaptive_l3_diagnostic = {}
        self.last_local_graph_status = "not_attempted"
        self.last_l3_query_diagnostic = {}
        self.l3_query_stats = _empty_l3_query_stats()
        self.last_episode_l3_query_stats = {}
        self._l3_stats_step = None
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
        self._prompt_context_pending = False
        self._prompt_context_pending_ids = frozenset()
        self._active_prompt_fusion = self.prompt_local_graph_fusion
        self.last_prompt_context = None
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

    def observe_spatial(
        self, obs: dict, info: dict | None, events: list[MemoryEvent], *, step: int,
    ) -> dict:
        """Build the independent 3D graph after L1 has persisted this step's Evidence."""
        self._require_writer()
        if self.spatial_map is None or self.spatial_extractor is None:
            return {}
        diagnostic = {
            "mode": self.spatial_l3_config.mode, "step": step,
            "frame_valid": False, "segmentation_count": 0,
            "evidence_minted": 0, "evidence_linked": 0, "evidence_unmatched": 0,
            "evidence_ambiguous": 0, "observed_objects": 0,
            "object_nodes": len(self.spatial_map.objects),
            "viewpoints": len(self.spatial_map.viewpoints),
            "observed_from_edges": sum(
                edge.relation_type == "observed_from" for edge in self.spatial_map.edges
            ),
            "traversed_to_edges": sum(
                edge.relation_type == "traversed_to" for edge in self.spatial_map.edges
            ),
            "clip_feature_observations": 0,
            "clip_feature_ms": None,
            "clip_feature_error": None,
            "rejection_reason": None,
        }
        self.last_spatial_frame = None
        try:
            frame = RGBDFrame.from_habitat(obs, info)
            diagnostic["frame_valid"] = True
            self.last_spatial_frame = frame
            instances = self.spatial_extractor.extract(frame.rgb, step)
            diagnostic["segmentation_count"] = len(instances)
            masks, counts = make_spatial_evidence(
                instances, events, self.v3_store,
                scene_id=self.v3_scene_id,
                episode_id=self.working.state.episode_id,
                rollout_idx=self.v3_rollout_idx,
                step=step,
                source=f"yolo11n-seg:{self.spatial_l3_config.seg_model_path.rsplit('/', 1)[-1]}",
                min_iou=self.spatial_l3_config.evidence_min_iou,
                ambiguity_margin=self.spatial_l3_config.evidence_ambiguity_margin,
            )
            diagnostic["evidence_minted"] = counts["minted"]
            diagnostic["evidence_linked"] = counts["linked"]
            diagnostic["evidence_unmatched"] = counts["unmatched"]
            diagnostic["evidence_ambiguous"] = counts["ambiguous"]
            observed = self.spatial_map.observe(
                frame, masks, step_id=step,
                transition_record=bool((info or {}).get("spatial_transition_record") is True),
            )
            diagnostic["observed_objects"] = len(observed)
            diagnostic["object_nodes"] = len(self.spatial_map.objects)
            if self.rgb_sidecar is not None:
                # Read-only: the sidecar stores the exact per-mask pixels after
                # ObjectMap.observe has already assigned object identities.  A
                # sidecar failure must never break L1 or the geometry map.
                try:
                    recorded = self.rgb_sidecar.record_frame(
                        frame, masks, step_id=step,
                    )
                    bound = self.rgb_sidecar.bind_objects(
                        self.spatial_map.objects.values(),
                    )
                    diagnostic["rgb_sidecar"] = {
                        **recorded,
                        "bound": bound,
                        "summary": self.rgb_sidecar.summary(),
                    }
                except Exception as exc:  # noqa: BLE001
                    diagnostic["rgb_sidecar_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    logger.warning(
                        "RGB sidecar failed step=%s: %s", step, exc,
                    )
            self._attach_clip_features(frame, masks, observed, diagnostic)
            diagnostic["viewpoints"] = len(self.spatial_map.viewpoints)
            diagnostic["observed_from_edges"] = sum(
                edge.relation_type == "observed_from" for edge in self.spatial_map.edges
            )
            diagnostic["traversed_to_edges"] = sum(
                edge.relation_type == "traversed_to" for edge in self.spatial_map.edges
            )
        except Exception as exc:  # noqa: BLE001
            # L3 is an optional sidecar. A segmentation or geometry failure must
            # never prevent the frozen policy from using its original path.
            diagnostic["rejection_reason"] = f"{type(exc).__name__}: {exc}"
            self.last_spatial_frame = None
            logger.warning("spatial L3 rejected step=%s: %s", step, exc)
        self.last_spatial_diagnostic = diagnostic
        return dict(diagnostic)

    def _attach_clip_features(
        self, frame: RGBDFrame, masks: list, observed: list, diagnostic: dict,
    ) -> None:
        """Extract one local CLIP feature per accepted mask, after geometry.

        This never participates in object association: the map is already
        updated, and only Evidence IDs that belong to a returned node receive a
        feature.  Failure is recorded and cannot break the L3 geometry pass or
        the frozen policy.
        """
        if (self.clip_encoder is None or self.spatial_map is None
                or not (
                    self.spatial_l3_config.mode == "on_demand"
                    or self.adaptive_l3_query_config.enabled
                )):
            return
        if not masks:
            return
        current_ids = {mask.evidence_id for mask in masks}
        linked_ids = {
            evidence_id
            for node in observed
            for evidence_id in node.evidence_ids
            if evidence_id in current_ids
        }
        selected = [mask for mask in masks if mask.evidence_id in linked_ids]
        if not selected:
            return
        started = perf_counter()
        try:
            boxes = clip_crop_boxes(
                [mask.mask for mask in selected], frame.rgb.shape[:2],
            )
            pairs = [
                (mask, box) for mask, box in zip(selected, boxes)
                if box is not None
            ]
            if not pairs:
                return
            features = self.clip_encoder.encode_image(
                frame.rgb, [box for _, box in pairs],
            )
            values = np.asarray(features, dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != len(pairs) or not np.isfinite(values).all():
                raise ValueError("CLIP encoder returned features with an invalid shape")
            attached = self.spatial_map.attach_clip_features({
                mask.evidence_id: values[index]
                for index, (mask, _) in enumerate(pairs)
            })
            diagnostic["clip_feature_observations"] = attached
        except Exception as exc:  # noqa: BLE001 - optional visual sidecar
            diagnostic["clip_feature_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("L3 CLIP feature extraction failed step=%s: %s",
                           diagnostic.get("step"), exc)
        finally:
            diagnostic["clip_feature_ms"] = (perf_counter() - started) * 1000.0

    def get_spatial_context(self, *, step: int) -> MemoryContext | None:
        """Adapt reliable historical 3D objects to the existing prompt contract."""
        if self.spatial_map is None or self.spatial_adapter is None:
            return None
        if self.last_spatial_frame is None or self.v3_store is None:
            return None
        return self.spatial_adapter.build(
            self.v3_store, self.spatial_map, self.last_spatial_frame,
            scene_id=self.v3_scene_id, episode_id=self.working.state.episode_id,
            rollout_idx=self.v3_rollout_idx, step=step,
            instruction=self.working.state.instruction,
            scope=self.local_graph_config.scope,
            max_landmarks=self.local_graph_config.max_landmarks,
            prompt_max_chars=min(
                self.local_graph_config.prompt_max_chars, self.max_context_chars,
            ),
            min_confidence=self.spatial_l3_config.min_injection_confidence,
        )

    def build_prompt_context(
        self, *, step: int, instruction: str, events: list[MemoryEvent],
        rgb_width: int, frame: object | None,
        token_budget: MemoryTokenBudget | None = None,
        budget_failure: TokenBudgetFailure | None = None,
    ) -> PromptContextResult:
        """Select this step's context for the configured prompt mode."""
        if self.plan.require_legacy():
            raise RuntimeError("legacy prompts use the original Local Graph path")
        if self.v3_store is None:
            raise RuntimeError("memory prompt modes require a V3 evidence store")
        token_limited = token_budget is not None and token_budget.available
        fusion = (
            self.prompt_local_graph_fusion
            if self.plan.l1_source == "original" and token_limited
            else self.local_graph_fusion
        )
        self._active_prompt_fusion = fusion
        char_limit = (
            UNBOUNDED_CHAR_LIMIT if token_limited else self.prompt_budget_chars
        )
        l3_diag: dict = {}
        on_demand_builder = None
        on_demand_source = "l3_clip_query"
        on_demand_empty_reason = "no_valid_on_demand_match"
        on_demand_context_source = "l3_on_demand"
        geometry_on_demand = (
            self.plan.mode == "l1_original_only"
            and self.spatial_l3_config.mode == "geometry_on_demand"
            and self.geometry_on_demand_controller is not None
        )
        clip_on_demand = (
            self.plan.mode == "l1_original_only"
            and self.spatial_l3_config.mode == "on_demand"
            and self.l3_query_controller is not None
        )
        if geometry_on_demand or clip_on_demand:
            def _on_demand(l1_status: str) -> MemoryContext | None:
                nonlocal l3_diag
                if geometry_on_demand:
                    context = self.get_geometry_on_demand_context(
                        step=step, instruction=instruction,
                        l1_status=l1_status, max_chars=char_limit,
                    )
                else:
                    pose = (
                        getattr(frame, "agent_pose", None)
                        if frame is not None else None
                    )
                    context = self.get_on_demand_l3_context(
                        step=step, instruction=instruction, agent_pose=pose,
                        l1_status=l1_status, max_chars=char_limit,
                    )
                l3_diag = dict(self.last_l3_query_diagnostic)
                return context

            if geometry_on_demand:
                on_demand_source = "spatial_graph_adapter"
                on_demand_empty_reason = "no_geometry_candidate"
                on_demand_context_source = "l3_geometry_on_demand"
            on_demand_builder = _on_demand
        result = build_prompt_context(
            self.plan,
            store=self.v3_store,
            object_map=self.spatial_map,
            frame=frame,
            local_graph_fusion=fusion,
            spatial_adapter=self.spatial_adapter,
            l1_organizer=self.l1_prompt_organizer,
            l3_organizer=self.l3_prompt_organizer,
            joint_builder=self.joint_prompt_builder,
            scene_id=self.v3_scene_id,
            episode_id=self.working.state.episode_id,
            rollout_idx=self.v3_rollout_idx,
            step=step,
            instruction=instruction,
            rgb_width=rgb_width,
            events=events,
            current_place_id=self.v3_place_id,
            scope=self.local_graph_config.scope,
            max_landmarks=self.local_graph_config.max_landmarks,
            min_confidence=self.spatial_l3_config.min_injection_confidence,
            budget_chars=self.prompt_budget_chars,
            char_limit=char_limit,
            token_budget=token_budget,
            budget_failure=budget_failure,
            on_demand_l3=on_demand_builder,
            on_demand_source=on_demand_source,
            on_demand_empty_reason=on_demand_empty_reason,
            on_demand_context_source=on_demand_context_source,
        )
        l1_status = (
            result.diagnostics.get("layers", {}).get("l1", {}).get("build_status")
        )
        if isinstance(l1_status, str):
            self.last_local_graph_status = l1_status
            self._record_l1_status(l1_status, step)
        if on_demand_builder is not None:
            if not l3_diag:
                l3_diag = {
                    "trigger": l1_status or "not_attempted",
                    "query_executed": False,
                    "candidate_count": 0,
                    "skipped_reason": (
                        "l1_satisfied" if result.context is not None else "not_invoked"
                    ),
                }
            if (
                int(l3_diag.get("candidate_count", 0) or 0) > 0
                and result.context is None
                and not l3_diag.get("prompt_budget_excluded")
            ):
                l3_diag["skipped_reason"] = "prompt_budget_excluded"
                l3_diag["prompt_budget_excluded"] = True
                self.l3_query_stats["prompt_budget_excluded"] += 1
            if result.context is not None:
                injected_items = list(result.context.layer_items)
                l3_diag["injected_count"] = len(injected_items)
                l3_diag["injected_object_ids"] = [
                    item.metadata.get("object_id") for item in injected_items
                    if isinstance(item.metadata.get("object_id"), str)
                ]
                l3_diag["injected_evidence_ids"] = list(dict.fromkeys(
                    evidence_id for item in injected_items
                    for evidence_id in item.metadata.get("evidence_ids", [])
                    if isinstance(evidence_id, str)
                ))
            else:
                l3_diag["injected_count"] = 0
                l3_diag["injected_object_ids"] = []
                l3_diag["injected_evidence_ids"] = []
            result.diagnostics["on_demand_l3"] = l3_diag
            result.diagnostics["l3_query_stats"] = self.l3_query_stats_summary()
        if self.plan.l1_source == "original" and not result.local_graph_pending:
            # The original builder created a pending selection that the mode
            # did not serialize (for example its block was dropped by budget).
            self._active_prompt_fusion.discard_selection()
        self.last_prompt_context = result
        self._prompt_context_pending = result.local_graph_pending
        self._prompt_context_pending_ids = frozenset(result.l1_evidence_ids)
        return result

    def confirm_prompt_context(self, evidence_ids: list[str], *, step: int) -> None:
        """Record an injected Local Graph selection only after the prompt proves it."""
        if not self._prompt_context_pending:
            return
        self._prompt_context_pending = False
        # Only the L1 evidence of this context may confirm its Local Graph
        # selection; a joint/concat context also carries L3 ids.
        confirmed = [
            evidence_id for evidence_id in evidence_ids
            if evidence_id in self._prompt_context_pending_ids
        ]
        self._prompt_context_pending_ids = frozenset()
        if confirmed:
            self._active_prompt_fusion.confirm_injected(confirmed, step)
        else:
            self._active_prompt_fusion.discard_selection()

    def discard_prompt_context(self) -> None:
        """Release a prompt-mode selection rejected by the prompt/token contract."""
        if not self._prompt_context_pending:
            return
        self._prompt_context_pending = False
        self._prompt_context_pending_ids = frozenset()
        self._active_prompt_fusion.discard_selection()

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

    def get_local_graph_context(self, *, step: int, rgb_width: int,
                                events: list[MemoryEvent]) -> MemoryContext | None:
        """Build a local navigation card from current RGB and scene evidence."""
        if self.local_graph_fusion is None or self.v3_store is None:
            raise RuntimeError("local graph memory is not active")
        context = self.local_graph_fusion.build(
            self.v3_store, scene_id=self.v3_scene_id,
            episode_id=self.working.state.episode_id,
            rollout_idx=self.v3_rollout_idx, step=step,
            instruction=self.working.state.instruction,
            current_place_id=self.v3_place_id,
            rgb_width=rgb_width, events=events,
        )
        self.last_local_graph_status = self.local_graph_fusion.last_build_status
        self._record_l1_status(self.last_local_graph_status, step)
        return context

    def _record_l1_status(self, status: str, step: int) -> None:
        """Count one L1 retrieval outcome per step for the on-demand audit."""
        if self.spatial_l3_config.mode not in {"on_demand", "geometry_on_demand"}:
            return
        if self._l3_stats_step == step:
            return
        self._l3_stats_step = step
        self.l3_query_stats["steps"] += 1
        if status == "built":
            self.l3_query_stats["l1_built"] += 1
        elif status == "no_candidates":
            self.l3_query_stats["l1_no_candidates"] += 1
        elif status == "budget_excluded":
            self.l3_query_stats["l1_budget_excluded"] += 1
        elif status == "reactivation_suppressed":
            self.l3_query_stats["l1_reactivation_suppressed"] += 1
        else:
            self.l3_query_stats["l1_other"] += 1

    def _record_l3_query(self, diagnostic: dict | None) -> None:
        if (
            self.spatial_l3_config.mode not in {"on_demand", "geometry_on_demand"}
            or not diagnostic
        ):
            return
        if diagnostic.get("query_executed"):
            self.l3_query_stats["queries_attempted"] += 1
            if int(diagnostic.get("candidate_count", 0) or 0) > 0:
                self.l3_query_stats["queries_with_match"] += 1
            else:
                self.l3_query_stats["queries_without_match"] += 1
        if diagnostic.get("prompt_budget_excluded"):
            self.l3_query_stats["prompt_budget_excluded"] += 1
        reason = diagnostic.get("skipped_reason")
        if reason == "query_error":
            self.l3_query_stats["query_errors"] += 1
        if reason and not diagnostic.get("query_executed"):
            reasons = self.l3_query_stats["skipped_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1

    def l3_query_stats_summary(self) -> dict:
        """Return cumulative L1/query/injection counts and explicit rates."""
        stats = {
            **self.l3_query_stats,
            "skipped_reasons": dict(self.l3_query_stats["skipped_reasons"]),
        }
        empty_steps = stats["l1_no_candidates"] + stats["l1_budget_excluded"]
        attempted = stats["queries_attempted"]
        matched = stats["queries_with_match"]
        injected = max(0, matched - stats["prompt_budget_excluded"])
        stats["l1_empty_steps"] = empty_steps
        stats["queries_injected"] = injected
        stats["query_trigger_rate"] = (
            attempted / empty_steps if empty_steps else None
        )
        stats["match_rate_of_queries"] = (
            matched / attempted if attempted else None
        )
        stats["injection_rate_of_l1_empty"] = (
            injected / empty_steps if empty_steps else None
        )
        return stats

    def get_on_demand_l3_context(
        self, *, step: int, instruction: str, agent_pose: object | None,
        l1_status: str, max_chars: int, query_text: str | None = None,
    ) -> MemoryContext | None:
        """Query L3 only after L1 reported an empty candidate set.

        The controller suppresses repeat queries for an unchanged object map,
        so a failed or empty query never becomes a per-step CLIP call.
        """
        if self.l3_query_controller is None or self.spatial_map is None:
            return None
        if agent_pose is None:
            self.last_l3_query_diagnostic = {
                "trigger": l1_status, "step": step,
                "skipped_reason": "no_current_pose", "candidate_count": 0,
                "query_executed": False,
            }
            self._record_l3_query(self.last_l3_query_diagnostic)
            return None
        result = self.l3_query_controller.maybe_query(
            self.spatial_map, step=step, instruction=instruction,
            l1_status=l1_status, query_text=query_text,
        )
        self.last_l3_query_diagnostic = dict(
            self.l3_query_controller.last_diagnostic or {},
        )
        context = None
        if result is not None:
            context = build_l3_query_context(
                result, self.spatial_map, agent_pose=agent_pose, max_chars=max_chars,
            )
            if context is None:
                self.last_l3_query_diagnostic["skipped_reason"] = "prompt_budget_excluded"
                self.last_l3_query_diagnostic["prompt_budget_excluded"] = True
        self._record_l3_query(self.last_l3_query_diagnostic)
        return context

    def set_instance_grounder(self, grounder: object | None) -> None:
        """Attach a grounder (used by tests and by a policy-owned runtime)."""
        if grounder is not None and not callable(getattr(grounder, "select", None)):
            raise TypeError("instance_grounder must provide select()")
        self.instance_grounder = grounder
        self._instance_grounder_error = None

    def _get_instance_grounder(self) -> object:
        """Load the official Student once; never silently disable it."""
        if self.instance_grounder is not None:
            return self.instance_grounder
        # Do not cache a load failure permanently: a missing/misconfigured
        # dependency can be fixed while the manager lives (e.g. a transient
        # CUDA context/OOM error during startup).  The last error is retained
        # for diagnostics and the next call retries.
        config = self.spatial_l3_config
        try:
            from lightnav.memory.l3_instance_grounder import (
                OfficialStudentRuntime,
                ViL3DRelGrounder,
            )

            runtime = OfficialStudentRuntime(
                config.grounder_checkpoint,
                config_path=config.grounder_config or None,
                device=config.grounder_device,
                bert_vocab_path=config.grounder_bert_vocab_path or None,
            )
            grounder = ViL3DRelGrounder(runtime, seed=config.grounder_seed)
        except Exception as exc:  # noqa: BLE001 - reported to the audit
            self._instance_grounder_error = f"{type(exc).__name__}: {exc}"
            raise
        self._instance_grounder_error = None
        self.instance_grounder = grounder
        return grounder

    def _grounded_argmax_geometry_context(
        self, *, step: int, instruction: str, l1_status: str, max_chars: int,
    ) -> MemoryContext | None:
        """Original d659e57 path: one full query, official global argmax only.

        This is the explicit ablation C.  It is byte-for-byte the same
        selection rule as the frozen baseline: no landmark parser, no
        eligibility re-ranking and no geometry fallback.  Extra diagnostics are
        observational only.
        """
        controller = self.geometry_on_demand_controller
        if controller is None or self.spatial_map is None:
            return None
        if (
            self.last_spatial_frame is None
            or self.spatial_adapter is None
            or self.v3_store is None
        ):
            diagnostic = {
                "trigger": l1_status, "step": step,
                "skipped_reason": "no_current_frame", "query_executed": False,
                "candidate_count": 0, "injected_count": 0,
                "retrieval": "geometry_spatial_graph_adapter+vil3dref_student",
                "grounding_query": instruction,
                "grounding_query_source": "full_instruction",
                "selection_mode": "argmax",
                "actual_injection": False,
                "actual_injected_evidence_ids": [],
            }
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        prepared = controller.prepare_query(
            self.spatial_map, step=step, instruction=instruction,
            l1_status=l1_status,
            available=lambda object_map: (
                controller.has_historical_geometry_candidates(object_map, step)
            ),
            unavailable_reason="no_historical_geometry_object",
            retrieval="geometry_spatial_graph_adapter+vil3dref_student",
            signature_fn=controller.geometry_map_signature,
        )
        diagnostic = dict(controller.last_diagnostic or {})
        diagnostic.update({
            "injected_count": 0,
            "retrieval": "geometry_spatial_graph_adapter+vil3dref_student",
            "grounding_query": instruction,
            "grounding_query_source": "full_instruction",
            "selection_mode": "argmax",
            "actual_injection": False,
            "actual_injected_evidence_ids": [],
            "geometry_fallback_reason": None,
            "prompt_hash_before": None,
            "prompt_hash_after": None,
        })
        if self.instruction_landmarks is not None:
            diagnostic["instruction_landmarks"] = (
                self.instruction_landmarks.to_dict()
            )
        if prepared is None:
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        _query, _ = prepared

        def _build(context_chars: int) -> MemoryContext | None:
            return self.spatial_adapter.build(
                self.v3_store, self.spatial_map, self.last_spatial_frame,
                scene_id=self.v3_scene_id,
                episode_id=self.working.state.episode_id,
                rollout_idx=self.v3_rollout_idx, step=step,
                instruction=instruction, scope=self.local_graph_config.scope,
                max_landmarks=max(1, len(self.spatial_map.objects)),
                prompt_max_chars=context_chars,
                min_confidence=self.spatial_l3_config.min_injection_confidence,
            )

        started = perf_counter()
        try:
            full_context = _build(UNBOUNDED_CHAR_LIMIT)
        except Exception as exc:  # noqa: BLE001
            diagnostic["skipped_reason"] = "query_error"
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            diagnostic["query_ms"] = (perf_counter() - started) * 1000.0
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        if full_context is None:
            diagnostic.update({
                "candidate_count": 0,
                "formatable_candidate_count": 0,
                "skipped_reason": "no_geometry_candidate",
                "query_ms": (perf_counter() - started) * 1000.0,
            })
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        formatable_items = {
            item.metadata.get("object_id"): item
            for item in full_context.layer_items
            if isinstance(item.metadata.get("object_id"), str)
        }
        if self.rgb_sidecar is None:
            diagnostic.update({
                "candidate_count": 0,
                "skipped_reason": "rgb_sidecar_unavailable",
                "query_ms": (perf_counter() - started) * 1000.0,
            })
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        try:
            grounder = self._get_instance_grounder()
        except Exception as exc:  # noqa: BLE001
            diagnostic.update({
                "candidate_count": 0,
                "skipped_reason": "grounder_load_error",
                "error": f"{type(exc).__name__}: {exc}",
                "query_ms": (perf_counter() - started) * 1000.0,
            })
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        try:
            result = grounder.select(
                instruction=instruction,
                object_map=self.spatial_map,
                sidecar=self.rgb_sidecar,
                step=step,
                episode_id=self.working.state.episode_id,
            )
        except Exception as exc:  # noqa: BLE001 - a sidecar must fail closed
            diagnostic.update({
                "candidate_count": 0,
                "skipped_reason": "grounding_error",
                "error": f"{type(exc).__name__}: {exc}",
                "query_ms": (perf_counter() - started) * 1000.0,
            })
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        diagnostic["instance_grounding"] = result.to_dict()
        diagnostic["grounding_ms"] = result.latency_ms
        diagnostic["candidate_count"] = (
            len(result.proposals) if result.valid else 0
        )
        diagnostic["formatable_candidate_count"] = len(full_context.layer_items)
        diagnostic["formatable_proposal_count"] = len(full_context.layer_items)
        diagnostic["proposal_count"] = (
            len(result.proposals) if result.proposals
            else len(result.offered_proposals)
        )
        diagnostic["grounder_proposal_count"] = diagnostic["proposal_count"]
        diagnostic["grounder_scored_count"] = len(result.proposals)
        ranked = sorted(
            result.proposals,
            key=lambda score: (
                -float(score.probability), -float(score.logit), score.object_id,
            ),
        )
        diagnostic["original_top1_probability"] = (
            float(ranked[0].probability) if ranked else None
        )
        formatable_ids = [
            score.object_id for score in ranked
            if score.object_id in formatable_items
        ]
        diagnostic["selected_valid_proposals"] = formatable_ids
        selected_score = next(
            (
                (rank, score) for rank, score in enumerate(ranked, start=1)
                if score.object_id == result.selected_object_id
            ),
            None,
        )
        diagnostic["global_grounding_rank"] = (
            selected_score[0] if selected_score is not None else None
        )
        diagnostic["selected_probability"] = (
            float(selected_score[1].probability)
            if selected_score is not None else None
        )
        diagnostic["selected_is_global_top1"] = bool(
            selected_score is not None and selected_score[0] == 1
        )
        diagnostic["eligible_grounding_rank"] = (
            formatable_ids.index(result.selected_object_id) + 1
            if result.selected_object_id in formatable_ids else None
        )
        diagnostic["query_ms"] = (perf_counter() - started) * 1000.0
        if not result.valid or result.selected_object_id is None:
            diagnostic["skipped_reason"] = (
                result.refusal_reason or "grounding_refused"
            )
            diagnostic["rejection_reason"] = diagnostic["skipped_reason"]
            if result.error is not None:
                diagnostic["error"] = result.error
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        selected_id = result.selected_object_id
        selected_items = [
            item for item in full_context.layer_items
            if item.metadata.get("object_id") == selected_id
        ]
        if not selected_items:
            diagnostic["candidate_count"] = 0
            diagnostic["skipped_reason"] = "selected_object_not_formatable"
            diagnostic["rejection_reason"] = "selected_object_not_formatable"
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        item = selected_items[0]
        text = "[Navigation Memory]\n" + item.description
        if len(text) > max_chars:
            diagnostic["candidate_count"] = 0
            diagnostic["skipped_reason"] = "prompt_budget_excluded"
            diagnostic["prompt_budget_excluded"] = True
            diagnostic["rejection_reason"] = "prompt_budget_excluded"
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        metadata = item.metadata
        evidence_ids = list(dict.fromkeys(
            value for value in metadata.get("evidence_ids", [])
            if isinstance(value, str)
        ))
        diagnostic.update({
            "injected_count": 1,
            "selected_object_id": selected_id,
            "selected_object_ids": [selected_id],
            "object_ids": [selected_id],
            "evidence_ids": evidence_ids,
            "matches": [{
                "object_id": selected_id,
                "label": item.label,
                "evidence_ids": list(metadata.get("evidence_ids", [])),
                "confidence": item.confidence,
                "observation_count": metadata.get("observation_count"),
                "center_world": metadata.get("center_world"),
                "distance_m": metadata.get("current_relative_distance_m"),
                "bearing_deg": metadata.get("current_relative_bearing_deg"),
                "selection_source": "vil3dref_grounding",
            }],
            "selection_sources": ["vil3dref_grounding"],
            "injection_provenance": [{
                "object_id": selected_id,
                "evidence_ids": evidence_ids,
                "source": "vil3dref_grounding",
                "grounding_query": instruction,
            }],
            "geometry_context_chars": len(text),
            "skipped_reason": None,
            "rejection_reason": None,
            "actual_injection": True,
            "actual_injected_evidence_ids": evidence_ids,
        })
        diagnostic["injected_text"] = text
        context = MemoryContext(text=text, layer_items=[item])
        self.last_l3_query_diagnostic = diagnostic
        self._record_l3_query(diagnostic)
        return context

    def _grounded_geometry_context(
        self, *, step: int, instruction: str, l1_status: str, max_chars: int,
    ) -> MemoryContext | None:
        """Query -> official Student scores -> eligible sources -> Prompt facts.

        Every call passes *all* complete ObjectNodes to the Student.  The
        official logits/probabilities are kept unchanged; the manager then
        selects from the eligible proposal set, not from the global argmax.
        Geometry fallback is an explicit opt-in: with landmark queries only the
        same-category item for that landmark is reused; a full-instruction
        query uses the original bounded adapter output.
        """
        controller = self.geometry_on_demand_controller
        if controller is None or self.spatial_map is None:
            return None
        if (
            self.last_spatial_frame is None
            or self.spatial_adapter is None
            or self.v3_store is None
        ):
            diagnostic = {
                "trigger": l1_status, "step": step,
                "skipped_reason": "no_current_frame", "query_executed": False,
                "candidate_count": 0, "injected_count": 0,
                "retrieval": "geometry_spatial_graph_adapter+vil3dref_student",
                "actual_injection": False, "actual_injected_evidence_ids": [],
                "ordered_memory_enabled": bool(
                    self.spatial_l3_config.grounder_ordered_landmark_memory
                ),
                "ordered_landmarks": [
                    row.to_dict() for row in _unexecuted_landmark_rows(
                        self.instruction_landmarks, "no_current_frame",
                    )
                ],
            }
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        prepared = controller.prepare_query(
            self.spatial_map, step=step, instruction=instruction,
            l1_status=l1_status,
            available=lambda object_map: (
                controller.has_historical_geometry_candidates(object_map, step)
            ),
            unavailable_reason="no_historical_geometry_object",
            retrieval="geometry_spatial_graph_adapter+vil3dref_student",
            signature_fn=controller.geometry_map_signature,
        )
        diagnostic = dict(controller.last_diagnostic or {})
        diagnostic.update({
            "injected_count": 0,
            "retrieval": "geometry_spatial_graph_adapter+vil3dref_student",
            "selection_mode": "eligible",
            "actual_injection": False,
            "actual_injected_evidence_ids": [],
            "grounding_queries": [],
            "geometry_fallback_reason": None,
            "geometry_fallback_attempted": False,
            "eligible_candidate_count": 0,
            "ordered_memory_enabled": bool(
                self.spatial_l3_config.grounder_ordered_landmark_memory
            ),
            "ordered_landmarks": [],
            "prompt_hash_before": None,
            "prompt_hash_after": None,
        })
        landmarks = self.instruction_landmarks
        if landmarks is not None:
            diagnostic["instruction_landmarks"] = landmarks.to_dict()
        if prepared is None:
            rows = _unexecuted_landmark_rows(
                landmarks, str(diagnostic.get("skipped_reason") or "query_not_executed"),
            )
            diagnostic["ordered_landmarks"] = [row.to_dict() for row in rows]
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        _query, _ = prepared

        objects_total = len(self.spatial_map.objects)

        def _build(context_chars: int, max_landmarks: int) -> MemoryContext | None:
            return self.spatial_adapter.build(
                self.v3_store, self.spatial_map, self.last_spatial_frame,
                scene_id=self.v3_scene_id,
                episode_id=self.working.state.episode_id,
                rollout_idx=self.v3_rollout_idx, step=step,
                instruction=instruction, scope=self.local_graph_config.scope,
                max_landmarks=max_landmarks,
                prompt_max_chars=context_chars,
                min_confidence=self.spatial_l3_config.min_injection_confidence,
                alias_aware=True,
            )

        started = perf_counter()
        try:
            full_context = _build(UNBOUNDED_CHAR_LIMIT, max(1, objects_total))
        except Exception as exc:  # noqa: BLE001
            diagnostic["skipped_reason"] = "query_error"
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            diagnostic["ordered_landmarks"] = [
                row.to_dict() for row in _unexecuted_landmark_rows(
                    landmarks, "query_error",
                )
            ]
            diagnostic["query_ms"] = (perf_counter() - started) * 1000.0
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        diagnostic["objects_total"] = objects_total
        if full_context is None:
            diagnostic.update({
                "candidate_count": 0,
                "formatable_candidate_count": 0,
                "rejected_object_count": objects_total,
                "skipped_reason": "no_geometry_candidate",
                "ordered_landmarks": [
                    row.to_dict() for row in _unexecuted_landmark_rows(
                        landmarks, "no_geometry_candidate",
                    )
                ],
                "query_ms": (perf_counter() - started) * 1000.0,
            })
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        items_by_object: dict[str, LayerMemoryItem] = {}
        for item in full_context.layer_items:
            object_id = item.metadata.get("object_id")
            if isinstance(object_id, str) and object_id not in items_by_object:
                items_by_object[object_id] = item
        diagnostic["formatable_candidate_count"] = len(items_by_object)
        diagnostic["formatable_proposal_count"] = len(items_by_object)

        def _geometry_fallback(reason: str) -> MemoryContext | None:
            """Use the unchanged adapter output when Grounding cannot map."""
            diagnostic["geometry_fallback_reason"] = reason
            try:
                fallback_context = _build(
                    max_chars, self.local_graph_config.max_landmarks,
                )
            except Exception as exc:  # noqa: BLE001
                diagnostic["skipped_reason"] = "query_error"
                diagnostic["error"] = f"{type(exc).__name__}: {exc}"
                diagnostic["query_ms"] = (perf_counter() - started) * 1000.0
                self.last_l3_query_diagnostic = diagnostic
                self._record_l3_query(diagnostic)
                return None
            if fallback_context is None:
                probe = None
                try:
                    probe = _build(
                        UNBOUNDED_CHAR_LIMIT,
                        self.local_graph_config.max_landmarks,
                    )
                except Exception:  # noqa: BLE001 - diagnostic probe only
                    probe = None
                if probe is not None:
                    diagnostic.update({
                        "candidate_count": len(probe.layer_items),
                        "prompt_budget_excluded": True,
                        "skipped_reason": "prompt_budget_excluded",
                        "actual_injection": False,
                        "actual_injected_evidence_ids": [],
                    })
                else:
                    diagnostic.update({
                        "candidate_count": 0,
                        "skipped_reason": "no_geometry_candidate",
                        "actual_injection": False,
                        "actual_injected_evidence_ids": [],
                    })
                diagnostic["query_ms"] = (perf_counter() - started) * 1000.0
                self.last_l3_query_diagnostic = diagnostic
                self._record_l3_query(diagnostic)
                return None
            fallback_items = fallback_context.layer_items
            fallback_evidence_ids = list(dict.fromkeys(
                evidence_id for item in fallback_items
                for evidence_id in item.metadata.get("evidence_ids", [])
                if isinstance(evidence_id, str)
            ))
            diagnostic.update({
                "candidate_count": len(fallback_items),
                "injected_count": len(fallback_items),
                "selected_object_id": None,
                "object_ids": [
                    item.metadata.get("object_id") for item in fallback_items
                ],
                "evidence_ids": fallback_evidence_ids,
                "matches": [
                    _spatial_match_payload(item) for item in fallback_items
                ],
                "proposal_object_id": [],
                "persistent_object_id": [],
                "grounding_logit": [],
                "grounding_probability": [],
                "grounding_rank": [],
                "selection_sources": ["geometry_fallback"] * len(fallback_items),
                "injection_provenance": [
                    {
                        "object_id": item.metadata.get("object_id"),
                        "evidence_ids": list(
                            item.metadata.get("evidence_ids", [])
                        ),
                        "source": "geometry_fallback",
                        "grounding_query": None,
                    }
                    for item in fallback_items
                ],
                "geometry_context_chars": len(fallback_context.text),
                "skipped_reason": None,
                "actual_injection": True,
                "actual_injected_evidence_ids": fallback_evidence_ids,
            })
            diagnostic["query_ms"] = (perf_counter() - started) * 1000.0
            diagnostic["injected_text"] = fallback_context.text
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return fallback_context

        def _record_refusal(reason: str, *, error: str | None = None) -> None:
            diagnostic.update({
                "candidate_count": 0,
                "skipped_reason": reason,
                "rejection_reason": reason,
                "actual_injection": False,
                "actual_injected_evidence_ids": [],
                "query_ms": (perf_counter() - started) * 1000.0,
            })
            if error is not None:
                diagnostic["error"] = error
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)

        if self.rgb_sidecar is None:
            if self.spatial_l3_config.grounder_geometry_fallback:
                return _geometry_fallback("rgb_sidecar_unavailable")
            _record_refusal(
                "rgb_sidecar_unavailable", error="RGB sidecar is not configured",
            )
            return None
        try:
            grounder = self._get_instance_grounder()
        except Exception as exc:  # noqa: BLE001
            if self.spatial_l3_config.grounder_geometry_fallback:
                diagnostic["error"] = f"{type(exc).__name__}: {exc}"
                return _geometry_fallback("grounder_load_error")
            _record_refusal(
                "grounder_load_error", error=f"{type(exc).__name__}: {exc}",
            )
            return None

        # One official forward per cached landmark phrase.  The complete
        # instruction still goes to LightNav; only the grounder query changes.
        if landmarks is not None and landmarks.landmarks:
            queries = [
                (landmark.grounding_query, "instruction_landmark", landmark)
                for landmark in landmarks.grounding_queries
            ]
        else:
            queries = [(instruction, "full_instruction_fallback", None)]
        if "instruction_landmarks" not in diagnostic:
            diagnostic["instruction_landmarks"] = {
                "instruction": instruction,
                "source": "full_instruction_fallback",
                "landmarks": [],
                "target_objects": [],
                "reference_objects": [],
            }

        selected_records: list[dict] = []
        selected_ids: set[str] = set()
        selected_items_by_order: dict[int, LayerMemoryItem] = {}
        landmark_rows: list[LandmarkMemoryRow] = []
        query_diagnostics: list[dict] = []
        aggregate_valid: list[str] = []
        aggregate_reasons: list[str] = []
        primary_result = None
        primary_diagnostic: dict = {}
        grounding_ms = 0.0

        def _row_for_query(
            *, landmark: object | None, query: str, query_index: int,
            result: object | None = None, chosen: object | None = None,
            chosen_item: LayerMemoryItem | None = None,
            eligibility: list | None = None,
            fallback_result: object | None = None,
            query_executed: bool = True,
            skip_reason: str | None = None,
        ) -> LandmarkMemoryRow:
            identity = _landmark_identity(landmark)
            metadata = chosen_item.metadata if chosen_item is not None else {}
            center = metadata.get("center_world")
            center_world = (
                tuple(float(value) for value in center)
                if isinstance(center, (list, tuple)) and len(center) == 3
                else None
            )
            evidence_ids = tuple(
                value for value in metadata.get("evidence_ids", [])
                if isinstance(value, str)
            )
            refs = (
                list(getattr(chosen_item, "evidence_refs", []) or [])
                if chosen_item is not None else []
            )
            distance = metadata.get("current_relative_distance_m")
            bearing = metadata.get("current_relative_bearing_deg")
            observations = metadata.get("observation_count")
            selection_source = None
            if chosen is not None:
                selection_source = "vil3dref_grounding"
            elif chosen_item is not None:
                selection_source = "geometry_fallback"
            return LandmarkMemoryRow(
                landmark_order=identity["landmark_order"] or (query_index + 1),
                landmark_role=identity["landmark_role"] or "final",
                landmark_phrase=identity["landmark_phrase"] or query,
                target_object=identity["target_object"] or "",
                action_phrase=identity["action_phrase"],
                grounding_query=query,
                query_executed=query_executed,
                query_skip_reason=skip_reason,
                proposal_count=(
                    (len(result.proposals) if result.proposals
                     else len(result.offered_proposals))
                    if result is not None else 0
                ),
                eligible_candidate_count=(
                    sum(1 for proposal in eligibility if proposal.eligible)
                    if eligibility is not None else 0
                ),
                selected_object_id=(
                    chosen_item.metadata.get("object_id")
                    if chosen_item is not None else None
                ),
                evidence_ids=evidence_ids,
                selection_source=selection_source,
                spatial_fact=(
                    chosen_item.description
                    if chosen_item is not None else None
                ),
                original_spatial_fact=(
                    _render_original_spatial_fact(chosen_item)
                    if chosen_item is not None else None
                ),
                spatial_label=(
                    chosen_item.label if chosen_item is not None else None
                ),
                center_world=center_world,
                distance_m=(
                    float(distance) if distance is not None else None
                ),
                bearing_deg=(
                    float(bearing) if bearing is not None else None
                ),
                observation_count=(
                    int(observations) if observations is not None else None
                ),
                evidence_step=(
                    int(chosen_item.step_id)
                    if chosen_item is not None else None
                ),
                evidence_confidence=(
                    float(chosen_item.confidence)
                    if chosen_item is not None else 0.0
                ),
                evidence_source=(
                    str(refs[0].source) if refs else None
                ),
                geometry_fallback_attempted=(
                    bool(getattr(fallback_result, "attempted", False))
                    if fallback_result is not None else False
                ),
                geometry_fallback_reason=(
                    getattr(fallback_result, "reason", None)
                    if fallback_result is not None else skip_reason
                ),
            )

        for query_index, (query, query_source, landmark) in enumerate(queries):
            target_object = (
                landmark.target_object if landmark is not None else None
            )
            query_started = perf_counter()
            try:
                result = grounder.select(
                    instruction=query,
                    object_map=self.spatial_map,
                    sidecar=self.rgb_sidecar,
                    step=step,
                    episode_id=self.working.state.episode_id,
                )
            except Exception as exc:  # noqa: BLE001 - a sidecar must fail closed
                query_diagnostic = {
                    "index": query_index,
                    "grounding_query": query,
                    "grounding_query_source": query_source,
                    "role": landmark.role if landmark is not None else None,
                    "landmark_role": (
                        landmark.navigation_role if landmark is not None else None
                    ),
                    "landmark_order": (
                        landmark.landmark_order if landmark is not None else None
                    ),
                    "landmark_phrase": (
                        landmark.target_phrase if landmark is not None else None
                    ),
                    "target_phrase": (
                        landmark.target_phrase if landmark is not None else None
                    ),
                    "target_object": target_object,
                    "reference_phrase": (
                        landmark.reference_phrase if landmark is not None else None
                    ),
                    "reference_objects": (
                        list(landmark.reference_objects)
                        if landmark is not None else []
                    ),
                    "grounder_proposal_count": 0,
                    "grounder_scored_count": 0,
                    "formatable_proposal_count": 0,
                    "eligible_candidate_count": 0,
                    "eligible_object_count": 0,
                    "proposals": [],
                    "selected_valid_proposals": [],
                    "selected_object_id": None,
                    "selected_for_injection": False,
                    "evidence_ids": [],
                    "selection_source": None,
                    "fallback_reason": "grounding_error",
                    "rejection_reason": "grounding_error",
                    "geometry_fallback_attempted": False,
                    "geometry_fallback_reason": "grounding_error",
                    "actual_injection": False,
                    "actual_injected_evidence_ids": [],
                    "original_top1_probability": None,
                    "selected_probability": None,
                    "selected_is_global_top1": False,
                    "global_grounding_rank": None,
                    "eligible_grounding_rank": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "query_ms": (perf_counter() - query_started) * 1000.0,
                }
                query_diagnostics.append(query_diagnostic)
                aggregate_reasons.append("grounding_error")
                landmark_rows.append(_row_for_query(
                    landmark=landmark, query=query, query_index=query_index,
                    query_executed=False, skip_reason="grounding_error",
                ))
                continue
            if primary_result is None:
                primary_result = result
            grounding_ms += float(getattr(result, "latency_ms", 0.0) or 0.0)
            eligibility = build_proposal_eligibility(
                result,
                object_map=self.spatial_map,
                sidecar=self.rgb_sidecar,
                items_by_object=items_by_object,
                target_object=target_object,
                selected_object_ids=selected_ids,
            )
            ranked_scores = sorted(
                result.proposals,
                key=lambda score: (
                    -float(score.probability), -float(score.logit),
                    score.object_id,
                ),
            )
            original_top1_probability = (
                float(ranked_scores[0].probability)
                if ranked_scores else None
            )
            valid_ids = eligible_object_ids(eligibility)
            for object_id in valid_ids:
                if object_id not in aggregate_valid:
                    aggregate_valid.append(object_id)
            chosen = choose_first_eligible(
                eligibility, selected_object_ids=selected_ids,
            )
            fallback_reason = None
            fallback_result = None
            chosen_item = None
            selection_source = None
            if chosen is not None:
                chosen_item = items_by_object.get(chosen.persistent_object_id)
                selection_source = "vil3dref_grounding"
            else:
                fallback_reason = _grounding_failure_reason(eligibility, result)
                aggregate_reasons.append(fallback_reason)
                # Reuse the original SpatialGraphAdapter item for this
                # landmark only when geometry fallback is explicitly enabled.
                # A full-instruction query has no landmark identity, so it
                # falls through to the original bounded geometry fallback.
                fallback_result = resolve_geometry_fallback(
                    full_context.layer_items,
                    target_object=target_object,
                    selected_object_ids=selected_ids,
                    enabled=(
                        self.spatial_l3_config.grounder_geometry_fallback
                        and target_object is not None
                    ),
                )
                if fallback_result.item is not None:
                    chosen_item = fallback_result.item
                    selection_source = "geometry_fallback"
            if chosen_item is not None:
                object_id = chosen_item.metadata.get("object_id")
                if isinstance(object_id, str) and object_id not in selected_ids:
                    selected_ids.add(object_id)
                    if landmark is not None:
                        selected_items_by_order[landmark.landmark_order] = chosen_item
                    selected_records.append({
                        "item": chosen_item,
                        "selection_source": selection_source,
                        "proposal": chosen,
                        "query": query,
                        "query_source": query_source,
                        "landmark": landmark,
                    })
            if chosen_item is not None:
                object_id = chosen_item.metadata.get("object_id")
                if isinstance(object_id, str) and object_id not in selected_ids:
                    selected_ids.add(object_id)
                    selected_records.append({
                        "item": chosen_item,
                        "selection_source": selection_source,
                        "proposal": chosen,
                        "query": query,
                        "query_source": query_source,
                        "landmark": landmark,
                    })
            query_diagnostic = {
                "index": query_index,
                "grounding_query": query,
                "grounding_query_source": query_source,
                "role": landmark.role if landmark is not None else None,
                "landmark_role": (
                    landmark.navigation_role if landmark is not None else None
                ),
                "landmark_order": (
                    landmark.landmark_order if landmark is not None else None
                ),
                "landmark_phrase": (
                    landmark.target_phrase if landmark is not None else None
                ),
                "target_phrase": (
                    landmark.target_phrase if landmark is not None else None
                ),
                "target_object": target_object,
                "reference_phrase": (
                    landmark.reference_phrase if landmark is not None else None
                ),
                "reference_objects": (
                    list(landmark.reference_objects)
                    if landmark is not None else []
                ),
                "grounder_proposal_count": (
                    len(result.proposals) if result.proposals
                    else len(result.offered_proposals)
                ),
                "grounder_scored_count": len(result.proposals),
                "formatable_proposal_count": sum(
                    1 for proposal in eligibility if proposal.formatable
                ),
                "eligible_candidate_count": sum(
                    1 for proposal in eligibility if proposal.eligible
                ),
                "eligible_object_count": len(valid_ids),
                "proposals": [
                    proposal.to_dict() for proposal in eligibility
                ],
                "selected_valid_proposals": valid_ids,
                "grounder_selected_object_id": result.selected_object_id,
                "selected_object_id": (
                    chosen_item.metadata.get("object_id")
                    if chosen_item is not None else None
                ),
                "selected_for_injection": chosen_item is not None,
                "evidence_ids": (
                    list(chosen_item.metadata.get("evidence_ids", []))
                    if chosen_item is not None else []
                ),
                "selection_source": selection_source,
                "fallback_reason": fallback_reason,
                "rejection_reason": (
                    fallback_reason if chosen is None and fallback_reason else None
                ),
                "geometry_fallback_attempted": (
                    bool(fallback_result.attempted)
                    if fallback_result is not None else False
                ),
                "geometry_fallback_reason": (
                    fallback_result.reason
                    if fallback_result is not None else None
                ),
                "actual_injection": False,
                "actual_injected_evidence_ids": [],
                "original_top1_probability": original_top1_probability,
                "selected_probability": (
                    chosen.grounding_probability
                    if chosen is not None else None
                ),
                "selected_is_global_top1": (
                    chosen.global_grounding_rank == 1
                    if chosen is not None else False
                ),
                "global_grounding_rank": (
                    chosen.global_grounding_rank
                    if chosen is not None else None
                ),
                "eligible_grounding_rank": (
                    chosen.eligible_grounding_rank
                    if chosen is not None else None
                ),
                "error": result.error,
                "query_ms": (perf_counter() - query_started) * 1000.0,
            }
            query_diagnostics.append(query_diagnostic)
            landmark_rows.append(_row_for_query(
                landmark=landmark, query=query, query_index=query_index,
                result=result, chosen=chosen, chosen_item=chosen_item,
                eligibility=eligibility, fallback_result=fallback_result,
            ))
            if not primary_diagnostic:
                primary_diagnostic = query_diagnostic

        with_proposals = [
            item for item in query_diagnostics
            if item.get("grounder_proposal_count")
        ]
        if with_proposals:
            # Report the first real target query as primary, so a windows
            # reference cannot become the headline grounding query.  Fall back
            # to the first query that produced proposals.
            primary_diagnostic = next(
                (
                    item for item in with_proposals
                    if item.get("role") == "target"
                ),
                with_proposals[0],
            )
        diagnostic["grounding_queries"] = query_diagnostics
        diagnostic["grounding_query"] = primary_diagnostic.get("grounding_query")
        diagnostic["grounding_query_source"] = primary_diagnostic.get(
            "grounding_query_source",
        )
        diagnostic["grounder_proposal_count"] = primary_diagnostic.get(
            "grounder_proposal_count", 0,
        )
        diagnostic["grounder_scored_count"] = primary_diagnostic.get(
            "grounder_scored_count", 0,
        )
        diagnostic["grounder_selected_object_id"] = primary_diagnostic.get(
            "grounder_selected_object_id",
        )
        diagnostic["selected_valid_proposals"] = list(aggregate_valid)
        diagnostic["selection_mode"] = "eligible"
        diagnostic["original_top1_probability"] = primary_diagnostic.get(
            "original_top1_probability",
        )
        diagnostic["selected_probability"] = primary_diagnostic.get(
            "selected_probability",
        )
        diagnostic["selected_is_global_top1"] = primary_diagnostic.get(
            "selected_is_global_top1",
        )
        diagnostic["global_grounding_rank"] = primary_diagnostic.get(
            "global_grounding_rank",
        )
        diagnostic["eligible_grounding_rank"] = primary_diagnostic.get(
            "eligible_grounding_rank",
        )
        diagnostic["rejection_reason"] = primary_diagnostic.get(
            "rejection_reason",
        )
        diagnostic["grounding_ms"] = grounding_ms
        if primary_result is not None:
            diagnostic["instance_grounding"] = primary_result.to_dict()
            diagnostic["proposal_count"] = diagnostic["grounder_proposal_count"]
        diagnostic["query_ms"] = (perf_counter() - started) * 1000.0
        diagnostic["landmark_role"] = primary_diagnostic.get("landmark_role")
        diagnostic["landmark_order"] = primary_diagnostic.get("landmark_order")
        diagnostic["landmark_phrase"] = primary_diagnostic.get("landmark_phrase")
        diagnostic["eligible_candidate_count"] = sum(
            int(item.get("eligible_candidate_count", 0) or 0)
            for item in query_diagnostics
        )
        diagnostic["geometry_fallback_attempted"] = any(
            bool(item.get("geometry_fallback_attempted")) for item in query_diagnostics
        )
        rows_in_order = sorted(
            landmark_rows, key=lambda row: row.landmark_order,
        )
        diagnostic["landmark_rows"] = [row.to_dict() for row in rows_in_order]

        if self.spatial_l3_config.grounder_ordered_landmark_memory:
            diagnostic["ordered_memory_rendering"] = {
                "include_missing_landmark_placeholders": bool(
                    self.spatial_l3_config.include_missing_landmark_placeholders
                ),
                "preserve_original_spatial_fact": bool(
                    self.spatial_l3_config.preserve_original_spatial_fact
                ),
                "line_count": 0,
                "fact_line_count": 0,
                "placeholder_line_count": 0,
            }
            text, included = render_ordered_landmark_memory(
                rows_in_order, max_chars=max_chars,
                include_missing_landmark_placeholders=(
                    self.spatial_l3_config.include_missing_landmark_placeholders
                ),
                preserve_original_spatial_fact=(
                    self.spatial_l3_config.preserve_original_spatial_fact
                ),
            )
            included_by_order = {
                row.landmark_order: row for row in included
            }
            all_rows = [
                included_by_order.get(row.landmark_order, row)
                for row in rows_in_order
            ]
            diagnostic["ordered_landmarks"] = [
                row.to_dict() for row in all_rows
            ]
            if text is None:
                diagnostic.update({
                    "candidate_count": 0,
                    "skipped_reason": "ordered_memory_budget_excluded",
                    "rejection_reason": "ordered_memory_budget_excluded",
                    "actual_injection": False,
                    "actual_injected_evidence_ids": [],
                })
                self.last_l3_query_diagnostic = diagnostic
                self._record_l3_query(diagnostic)
                return None
            evidence_rows = [
                row for row in included if row.spatial_fact_included
            ]
            if not evidence_rows:
                diagnostic.update({
                    "candidate_count": 0,
                    "skipped_reason": "no_evidence_fact_within_budget",
                    "rejection_reason": "no_evidence_fact_within_budget",
                    "actual_injection": False,
                    "actual_injected_evidence_ids": [],
                })
                self.last_l3_query_diagnostic = diagnostic
                self._record_l3_query(diagnostic)
                return None
            injected_ids = injected_evidence_ids(included)
            records_by_order = {
                record["landmark"].landmark_order: record
                for record in selected_records
                if record["landmark"] is not None
            }
            evidence_pairs = [
                (row, selected_items_by_order.get(row.landmark_order))
                for row in evidence_rows
            ]
            evidence_records = [
                records_by_order.get(row.landmark_order) for row in evidence_rows
            ]
            fallback_reasons = [
                row.geometry_fallback_reason for row in evidence_rows
                if (
                    row.selection_source == "geometry_fallback"
                    and row.geometry_fallback_reason
                )
            ]
            rendered_line_count = max(0, len(text.splitlines()) - 1)
            diagnostic["ordered_memory_rendering"].update({
                "line_count": rendered_line_count,
                "fact_line_count": len(evidence_rows),
                "placeholder_line_count": max(
                    0, rendered_line_count - len(evidence_rows),
                ),
            })
            diagnostic.update({
                "candidate_count": max(
                    diagnostic["grounder_proposal_count"], len(included),
                ),
                "injected_count": len(included),
                "injected_landmark_count": len(included),
                "injected_evidence_landmark_count": len(evidence_rows),
                "selected_object_id": evidence_rows[0].selected_object_id,
                "selected_object_ids": [
                    row.selected_object_id for row in evidence_rows
                ],
                "object_ids": [
                    row.selected_object_id for row in evidence_rows
                ],
                "evidence_ids": injected_ids,
                "matches": [
                    _spatial_match_payload(
                        item, selection_source=row.selection_source,
                    )
                    for row, item in evidence_pairs
                    if item is not None
                ],
                "proposal_object_id": [
                    record["proposal"].proposal_object_id
                    if record is not None and record["proposal"] is not None
                    else None
                    for record in evidence_records
                ],
                "persistent_object_id": [
                    row.selected_object_id for row in evidence_rows
                ],
                "grounding_logit": [
                    record["proposal"].grounding_logit
                    if record is not None and record["proposal"] is not None
                    else None
                    for record in evidence_records
                ],
                "grounding_probability": [
                    record["proposal"].grounding_probability
                    if record is not None and record["proposal"] is not None
                    else None
                    for record in evidence_records
                ],
                "grounding_rank": [
                    record["proposal"].grounding_rank
                    if record is not None and record["proposal"] is not None
                    else None
                    for record in evidence_records
                ],
                "selected_global_grounding_ranks": [
                    record["proposal"].global_grounding_rank
                    if record is not None and record["proposal"] is not None
                    else None
                    for record in evidence_records
                ],
                "selected_eligible_grounding_ranks": [
                    record["proposal"].eligible_grounding_rank
                    if record is not None and record["proposal"] is not None
                    else None
                    for record in evidence_records
                ],
                "selected_probabilities": [
                    record["proposal"].grounding_probability
                    if record is not None and record["proposal"] is not None
                    else None
                    for record in evidence_records
                ],
                "selected_is_global_top1_flags": [
                    (
                        record["proposal"].global_grounding_rank == 1
                        if record is not None and record["proposal"] is not None
                        else None
                    )
                    for record in evidence_records
                ],
                "selection_sources": [
                    row.selection_source for row in evidence_rows
                ],
                "injection_provenance": [
                    {
                        "object_id": row.selected_object_id,
                        "evidence_ids": list(row.evidence_ids),
                        "source": row.selection_source,
                        "grounding_query": row.grounding_query,
                        "landmark_role": row.landmark_role,
                        "landmark_order": row.landmark_order,
                        "landmark_phrase": row.landmark_phrase,
                    }
                    for row in evidence_rows
                ],
                "geometry_context_chars": len(text),
                "geometry_fallback_reason": (
                    "landmark_geometry_fallback: " + "; ".join(
                        dict.fromkeys(fallback_reasons)
                    )
                    if fallback_reasons else None
                ),
                "geometry_fallback_attempted": any(
                    row.geometry_fallback_attempted for row in included
                ),
                "skipped_reason": None,
                "rejection_reason": None,
                "actual_injection": True,
                "actual_injected_evidence_ids": injected_ids,
            })
            diagnostic["injected_text"] = text
            context = build_ordered_memory_context(included, text=text)
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return context

        if selected_records:
            # Budget priority: final targets first, reference/source anchors
            # afterwards.  Within a role the original instruction order is
            # retained.  Diagnostics still list queries in instruction order.
            render_records = sorted(
                selected_records,
                key=lambda record: (
                    0 if (
                        record["landmark"] is None
                        or record["landmark"].role == "target"
                    ) else 1,
                    (
                        record["landmark"].index
                        if record["landmark"] is not None else 0
                    ),
                ),
            )
            selected_items = [
                record["item"] for record in render_records
            ]
            text, included = render_context(
                selected_items, header=_NAVIGATION_MEMORY_HEADER,
                max_chars=max_chars,
            )
            if text is not None:
                included_identity = {id(item) for item in included}
                included_records = [
                    record for record in render_records
                    if id(record["item"]) in included_identity
                ]
                selected_object_ids = [
                    item.metadata.get("object_id") for item in included
                ]
                selected_evidence_ids = list(dict.fromkeys(
                    evidence_id for item in included
                    for evidence_id in item.metadata.get("evidence_ids", [])
                    if isinstance(evidence_id, str)
                ))
                diagnostic.update({
                    "candidate_count": max(
                        diagnostic["grounder_proposal_count"], len(included),
                    ),
                    "injected_count": len(included),
                    "selected_object_id": selected_object_ids[0],
                    "object_ids": selected_object_ids,
                    "evidence_ids": selected_evidence_ids,
                    "matches": [
                        _spatial_match_payload(
                            item,
                            selection_source=record["selection_source"],
                        )
                        for item, record in zip(included, included_records)
                    ],
                    "proposal_object_id": [
                        item.metadata.get("object_id") for item in included
                    ],
                    "persistent_object_id": [
                        item.metadata.get("object_id") for item in included
                    ],
                    "grounding_logit": [
                        record["proposal"].grounding_logit
                        if record["proposal"] is not None else None
                        for record in included_records
                    ],
                    "grounding_probability": [
                        record["proposal"].grounding_probability
                        if record["proposal"] is not None else None
                        for record in included_records
                    ],
                    "grounding_rank": [
                        record["proposal"].grounding_rank
                        if record["proposal"] is not None else None
                        for record in included_records
                    ],
                    "selected_global_grounding_ranks": [
                        record["proposal"].global_grounding_rank
                        if record["proposal"] is not None else None
                        for record in included_records
                    ],
                    "selected_eligible_grounding_ranks": [
                        record["proposal"].eligible_grounding_rank
                        if record["proposal"] is not None else None
                        for record in included_records
                    ],
                    "selected_probabilities": [
                        record["proposal"].grounding_probability
                        if record["proposal"] is not None else None
                        for record in included_records
                    ],
                    "selected_is_global_top1_flags": [
                        (
                            record["proposal"].global_grounding_rank == 1
                            if record["proposal"] is not None else None
                        )
                        for record in included_records
                    ],
                    "selection_sources": [
                        record["selection_source"] for record in included_records
                    ],
                    "injection_provenance": [
                        {
                            "object_id": item.metadata.get("object_id"),
                            "evidence_ids": list(
                                item.metadata.get("evidence_ids", [])
                            ),
                            "source": record["selection_source"],
                            "grounding_query": record["query"],
                        }
                        for item, record in zip(included, included_records)
                    ],
                    "geometry_context_chars": len(text),
                    "skipped_reason": None,
                    "actual_injection": True,
                    "actual_injected_evidence_ids": selected_evidence_ids,
                    "proposal_object_count": len(items_by_object),
                })
                included_object_ids = {
                    object_id for object_id in selected_object_ids
                    if isinstance(object_id, str)
                }
                updated_rows = [
                    replace(
                        row,
                        included_in_prompt=(
                            row.selected_object_id in included_object_ids
                        ),
                        spatial_fact_included=(
                            row.selected_object_id in included_object_ids
                        ),
                        actual_injection=(
                            row.selected_object_id in included_object_ids
                        ),
                        actual_injected_evidence_ids=(
                            row.evidence_ids
                            if row.selected_object_id in included_object_ids
                            else ()
                        ),
                    )
                    for row in rows_in_order
                ]
                diagnostic["ordered_landmarks"] = [
                    row.to_dict() for row in updated_rows
                ]
                for item in query_diagnostics:
                    selected = item.get("selected_object_id")
                    if selected in included_object_ids:
                        item["actual_injection"] = True
                        item["actual_injected_evidence_ids"] = list(
                            item.get("evidence_ids", [])
                        )
                fallback_sources = [
                    record["selection_source"] for record in included_records
                ]
                if any(
                    source == "geometry_fallback"
                    for source in fallback_sources
                ):
                    diagnostic["geometry_fallback_reason"] = (
                        "landmark_geometry_fallback: " + "; ".join(
                            dict.fromkeys(aggregate_reasons)
                        )
                    )
                else:
                    diagnostic["geometry_fallback_reason"] = None
                diagnostic["injected_text"] = text
                context = MemoryContext(text=text, layer_items=list(included))
                self.last_l3_query_diagnostic = diagnostic
                self._record_l3_query(diagnostic)
                return context
            diagnostic.update({
                "candidate_count": 0,
                "skipped_reason": "prompt_budget_excluded",
                "prompt_budget_excluded": True,
                "actual_injection": False,
                "actual_injected_evidence_ids": [],
            })
            diagnostic["ordered_landmarks"] = [
                row.to_dict() for row in rows_in_order
            ]
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None

        # No landmark produced an eligible Grounding fact.  With multi-landmark
        # queries, only per-landmark same-category fallback is allowed, so a
        # sink query can never reintroduce a bed/doorway item.  The global
        # bounded geometry fallback is used only for a full-instruction query.
        fallback_reason = "; ".join(dict.fromkeys(aggregate_reasons)) or (
            "no_eligible_proposal"
        )
        has_landmark_queries = bool(
            landmarks is not None
            and landmarks.landmarks
            and self.spatial_l3_config.grounder_landmark_queries
        )
        if (
            self.spatial_l3_config.grounder_geometry_fallback
            and not has_landmark_queries
        ):
            return _geometry_fallback(fallback_reason)
        diagnostic.update({
            "candidate_count": 0,
            "skipped_reason": fallback_reason,
            "rejection_reason": fallback_reason,
            "actual_injection": False,
            "actual_injected_evidence_ids": [],
            "ordered_landmarks": [row.to_dict() for row in rows_in_order],
            "query_ms": (perf_counter() - started) * 1000.0,
        })
        self.last_l3_query_diagnostic = diagnostic
        self._record_l3_query(diagnostic)
        return None

    def get_geometry_on_demand_context(
        self, *, step: int, instruction: str, l1_status: str, max_chars: int,
    ) -> MemoryContext | None:
        """Original on-demand gate + 59d09ba geometry-only candidate retrieval.

        The L1-empty trigger and the ``(map signature, query)`` dedupe are
        delegated to :class:`L3QueryController.prepare_query`, the same code the
        CLIP path uses.  Retrieval is the unchanged
        :class:`SpatialGraphAdapter.build`; no CLIP encoder, weight, embedding or
        cosine score exists on this path.
        """
        if self.spatial_l3_config.instance_grounder_enabled:
            if self.spatial_l3_config.grounder_selection_mode == "argmax":
                return self._grounded_argmax_geometry_context(
                    step=step, instruction=instruction, l1_status=l1_status,
                    max_chars=max_chars,
                )
            return self._grounded_geometry_context(
                step=step, instruction=instruction, l1_status=l1_status,
                max_chars=max_chars,
            )
        controller = self.geometry_on_demand_controller
        if controller is None or self.spatial_map is None:
            return None
        if (
            self.last_spatial_frame is None
            or self.spatial_adapter is None
            or self.v3_store is None
        ):
            diagnostic = {
                "trigger": l1_status, "step": step,
                "skipped_reason": "no_current_frame", "query_executed": False,
                "candidate_count": 0, "injected_count": 0,
                "retrieval": "geometry_spatial_graph_adapter",
            }
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        prepared = controller.prepare_query(
            self.spatial_map, step=step, instruction=instruction,
            l1_status=l1_status,
            available=lambda object_map: (
                controller.has_historical_geometry_candidates(object_map, step)
            ),
            unavailable_reason="no_historical_geometry_object",
            retrieval="geometry_spatial_graph_adapter",
            signature_fn=controller.geometry_map_signature,
        )
        diagnostic = dict(controller.last_diagnostic or {})
        diagnostic["injected_count"] = 0
        if prepared is None:
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        _query, _ = prepared

        def _build(context_chars: int) -> MemoryContext | None:
            return self.spatial_adapter.build(
                self.v3_store, self.spatial_map, self.last_spatial_frame,
                scene_id=self.v3_scene_id,
                episode_id=self.working.state.episode_id,
                rollout_idx=self.v3_rollout_idx, step=step,
                instruction=instruction, scope=self.local_graph_config.scope,
                max_landmarks=self.local_graph_config.max_landmarks,
                prompt_max_chars=context_chars,
                min_confidence=self.spatial_l3_config.min_injection_confidence,
            )

        started = perf_counter()
        try:
            context = _build(max_chars)
            diagnostic["query_ms"] = (perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001 - a sidecar must not break L1
            diagnostic["skipped_reason"] = "query_error"
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            diagnostic["query_ms"] = (perf_counter() - started) * 1000.0
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None
        objects_total = len(self.spatial_map.objects)
        diagnostic["objects_total"] = objects_total
        if context is None:
            # Distinguish "no geometric candidate" from "candidate dropped by
            # the on-demand Prompt budget" without inventing a score.
            probe = None
            try:
                probe = _build(UNBOUNDED_CHAR_LIMIT)
            except Exception:  # noqa: BLE001 - diagnostic probe only
                probe = None
            if probe is not None:
                diagnostic["candidate_count"] = len(probe.layer_items)
                diagnostic["prompt_budget_excluded"] = True
                diagnostic["skipped_reason"] = "prompt_budget_excluded"
            else:
                diagnostic["candidate_count"] = 0
                diagnostic["skipped_reason"] = "no_geometry_candidate"
            diagnostic["rejected_object_count"] = objects_total
            self.last_l3_query_diagnostic = diagnostic
            self._record_l3_query(diagnostic)
            return None

        items = context.layer_items
        object_ids: list[str] = []
        evidence_ids: list[str] = []
        matches: list[dict[str, object]] = []
        for item in items:
            object_id = item.metadata.get("object_id")
            item_evidence = list(item.metadata.get("evidence_ids", []))
            if isinstance(object_id, str):
                object_ids.append(object_id)
            evidence_ids.extend(
                value for value in item_evidence if isinstance(value, str)
            )
            matches.append({
                "object_id": object_id,
                "label": item.label,
                "evidence_ids": item_evidence,
                "confidence": item.confidence,
                "observation_count": item.metadata.get("observation_count"),
                "center_world": item.metadata.get("center_world"),
                "distance_m": item.metadata.get("current_relative_distance_m"),
                "bearing_deg": item.metadata.get("current_relative_bearing_deg"),
            })
        diagnostic.update({
            "candidate_count": len(items),
            "injected_count": len(items),
            "rejected_object_count": max(0, objects_total - len(items)),
            "object_ids": object_ids,
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "matches": matches,
            "geometry_context_chars": len(context.text),
            "skipped_reason": None,
        })
        self.last_l3_query_diagnostic = diagnostic
        self._record_l3_query(diagnostic)
        return context

    def query_l3_objects(self, text: str, top_k: int) -> L3QueryResult:
        """Local, non-mutating text query over the current L3 object map."""
        if self.spatial_map is None or self.clip_encoder is None:
            raise RuntimeError("on-demand L3 CLIP query is not active")
        return L3ClipQuery(self.spatial_map, self.clip_encoder).query(
            text, top_k,
            min_similarity=self.spatial_l3_config.clip_min_similarity,
        )

    def set_adaptive_l3_runner(self, runner: AdaptiveL3QueryRunner | None) -> None:
        """Attach the policy-owned runner (it needs the already-loaded engine)."""
        if not self.adaptive_l3_query_config.enabled:
            raise ValueError("adaptive L3 query is disabled")
        if runner is not None and not isinstance(runner, AdaptiveL3QueryRunner):
            raise TypeError("runner must be an AdaptiveL3QueryRunner")
        self.adaptive_l3_runner = runner

    def build_adaptive_l3_context(
        self, *, step: int, instruction: str, l1_context: MemoryContext | None,
        l1_status: str, agent_pose: object | None, max_chars: int,
    ) -> MemoryContext | None:
        """Ask the controller whether to query, then combine only verified facts.

        The L1 card is never rewritten.  If the runner abstains, fails, or returns
        nothing that fits the remaining budget, the exact object it was given is
        returned.  When L3 facts are added, the L1 text stays the prefix of the
        combined text and the L1 items stay first.
        """
        if not self.adaptive_l3_query_config.enabled:
            raise RuntimeError("adaptive L3 query is disabled")
        if type(max_chars) is not int or max_chars < 1:
            raise ValueError("adaptive L3 prompt budget must be a positive integer")
        if self.adaptive_l3_runner is None:
            self.last_adaptive_l3_diagnostic = {
                "step": step, "skipped_reason": "controller_not_attached",
                "injected": False, "query_executed": False,
            }
            return l1_context
        if self.spatial_map is None:
            self.last_adaptive_l3_diagnostic = {
                "step": step, "skipped_reason": "l3_map_unavailable",
                "injected": False, "query_executed": False,
            }
            return l1_context
        l1_chars = len(l1_context.text) if l1_context is not None else 0
        reserve = 1 if l1_context is not None else 0
        remaining = max_chars - l1_chars - reserve
        if remaining < len("[Verified L3 history]"):
            self.last_adaptive_l3_diagnostic = {
                "step": step, "skipped_reason": "l1_exhausts_prompt_budget",
                "injected": False, "query_executed": False,
                "l1_chars": l1_chars, "budget_chars": max_chars,
            }
            return l1_context
        context = self.adaptive_l3_runner.run(
            object_map=self.spatial_map, step=step, instruction=instruction,
            l1_status=l1_status, l1_context=l1_context, agent_pose=agent_pose,
            max_chars=remaining,
        )
        diagnostic = dict(self.adaptive_l3_runner.last_diagnostic)
        diagnostic["l1_chars"] = l1_chars
        diagnostic["budget_chars"] = max_chars
        self.last_adaptive_l3_diagnostic = diagnostic
        if context is None:
            return l1_context
        if l1_context is None:
            diagnostic["l1_included"] = False
            return context
        combined_text = l1_context.text.rstrip("\n") + "\n" + context.text
        # ``build_context_text`` escapes the whole block before injection, so
        # check the escaped length to guarantee the L3 lines are never cut off.
        if len(escape(combined_text, quote=False)) > max_chars:
            diagnostic["injected"] = False
            diagnostic["skipped_reason"] = "combined_prompt_budget_excluded"
            return l1_context
        diagnostic["l1_included"] = True
        diagnostic["combined_chars"] = len(combined_text)
        return MemoryContext(
            events=l1_context.events,
            nodes=l1_context.nodes,
            edges=l1_context.edges,
            text=combined_text,
            layer_items=[*l1_context.layer_items, *context.layer_items],
        )

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
            if self.local_graph_fusion is not None:
                self.local_graph_fusion.close_episode()
            if (self.prompt_local_graph_fusion is not None
                    and self.prompt_local_graph_fusion is not self.local_graph_fusion):
                self.prompt_local_graph_fusion.close_episode()
            if self.spatial_map is not None:
                self.spatial_extractor.close_episode()
                self.spatial_map.objects.clear()
                self.spatial_map.viewpoints.clear()
                self.spatial_map.edges.clear()
                self.spatial_map.evidence.clear()
                self.last_spatial_frame = None
                self.last_spatial_diagnostic = {}
            if self.l3_query_controller is not None:
                self.l3_query_controller.close_episode()
            if self.geometry_on_demand_controller is not None:
                self.geometry_on_demand_controller.close_episode()
            if self.rgb_sidecar is not None:
                self.rgb_sidecar.close_episode()
            self.instruction_landmarks = None
            if self.adaptive_l3_runner is not None:
                self.adaptive_l3_runner.close_episode()
            self.last_adaptive_l3_diagnostic = {}
            self.last_local_graph_status = "not_attempted"
            self.last_l3_query_diagnostic = {}
            self.last_episode_l3_query_stats = (
                self.l3_query_stats_summary()
                if self.spatial_l3_config.mode in {
                    "on_demand", "geometry_on_demand",
                } else {}
            )
            self.l3_query_stats = _empty_l3_query_stats()
            self._l3_stats_step = None
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
            self._prompt_context_pending = False
            self._prompt_context_pending_ids = frozenset()
            self.last_prompt_context = None

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
