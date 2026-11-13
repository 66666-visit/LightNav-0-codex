"""VLN-CE velocity policy: model trajectory tokens -> Habitat ``velocity_control`` actions.

On every ``act`` the latest RGB frame is pushed into the frame buffer, one
inference step is run with the VLN-CE trajectory prompt, the emitted token(s)
are decoded to ``(H, 3)`` robot-local waypoints ``[forward_m, lateral_m,
yaw_rad]`` and the first velocity-relevant waypoint is mapped to a normalized
``velocity_control`` action dict.

A decode failure (no trajectory token, out-of-range id, wrong RVQ level count)
falls back to a zero-velocity command. Habitat's ``velocity_control`` treats a
command below its minimum speeds as a stop, so that fallback ends the episode.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from lightnav.inference.samples import build_vln_traj_sample
from lightnav.tracking import TrackingAgent
from lightnav.velocity import first_waypoint_to_velocity_cmd
from lightnav.vln_utils import parse_rvq_action_tokens, parse_traj_token

if TYPE_CHECKING:
    from lightnav.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


def extract_instruction(obs: dict[str, Any] | None) -> str:
    """Return the instruction text of an observation (``str`` or ``{"text": str}``), else ``""``."""
    if obs is None:
        return ""
    instruction_raw = obs.get("instruction", None)
    if instruction_raw is None:
        return ""
    if isinstance(instruction_raw, dict):
        return instruction_raw.get("text", "")
    return str(instruction_raw)


def select_action_waypoint(waypoints: np.ndarray, atol: float = 1e-6) -> tuple[int, np.ndarray]:
    """Return the first waypoint that can drive unicycle velocity, skipping leading origin rows."""
    for idx, waypoint in enumerate(waypoints):
        forward_or_yaw = np.asarray([waypoint[0], waypoint[2]], dtype=np.float32)
        if np.any(np.abs(forward_or_yaw) > atol):
            return idx, waypoint
    return 0, waypoints[0]


_select_action_waypoint = select_action_waypoint  # historical private name


class TrajVocabVLNCEPolicy:
    """Habitat VLN-CE policy composed over ``TrackingAgent``.

    Exactly one of ``centroids`` (flat ``<traj_k>`` vocabulary, ``(K, H, 3)``
    array or ``.npy`` path) or ``rvq_bundle`` (``<act_l*>`` RVQ decoding) must be
    given, and it must agree with the checkpoint's ``action_method`` when the
    engine exposes one.

    ``dt``, ``lin_vel_range`` (m/s) and ``ang_vel_range`` (deg/s) are the
    environment's ``velocity_control`` settings, reported by the Habitat server
    in ``info`` on reset.
    """

    def __init__(
        self,
        engine: Any,
        centroids: np.ndarray | str | Path | None = None,
        *,
        dt: float,
        lin_vel_range: tuple[float, float],
        ang_vel_range: tuple[float, float],
        num_history_frames: int = 64,
        rvq_bundle: Any = None,
        memory_enabled: bool = False,
        memory_manager: MemoryManager | None = None,
        memory_verbose: bool = False,
    ) -> None:
        if (centroids is None) == (rvq_bundle is None):
            raise ValueError("pass exactly one of `centroids` (flat) or `rvq_bundle` (rvq).")
        self.method: str = "rvq" if rvq_bundle is not None else "flat"

        centroids_arr: np.ndarray | None = None
        if self.method == "flat":
            if isinstance(centroids, (str, Path)):
                arr = np.load(Path(centroids))
            else:
                arr = np.asarray(centroids)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"centroids must have shape (K, H, 3), got {arr.shape}")
            if arr.shape[0] < 2 or arr.shape[1] < 1:
                raise ValueError(
                    f"centroids must have K>=2 and H>=1, got K={arr.shape[0]}, H={arr.shape[1]}"
                )
            centroids_arr = np.ascontiguousarray(arr, dtype=np.float32)

        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        # The sample builder swaps in the RVQ output sentence when the checkpoint's
        # action_method is "rvq", so the decoder must agree with it.
        bundle_method = getattr(getattr(engine, "bundle", None), "action_method", None)
        if bundle_method is not None and (bundle_method == "rvq") != (self.method == "rvq"):
            given = "an RVQ bundle" if self.method == "rvq" else "flat centroids"
            raise ValueError(
                f"checkpoint action_tokenizer method is {bundle_method!r} but the policy was "
                f"given {given}; pass --action_tokenizer_bundle for rvq checkpoints or "
                f"--traj_vocab_path for flat ones."
            )

        self.engine = engine
        self.dt: float = float(dt)
        self.lin_vel_range: tuple[float, float] = (float(lin_vel_range[0]), float(lin_vel_range[1]))
        self.ang_vel_range: tuple[float, float] = (float(ang_vel_range[0]), float(ang_vel_range[1]))
        self.num_history_frames: int = int(num_history_frames)

        self.agent = TrackingAgent(
            engine,
            centroids=centroids_arr,
            num_history_frames=self.num_history_frames,
            rvq_bundle=rvq_bundle,
        )
        self.centroids: np.ndarray | None = self.agent.centroids
        self.rvq = self.agent.rvq
        self.K: int | None = self.agent.K
        self.H: int = self.agent.H
        self.instruction: str = ""

        self.memory_enabled = bool(memory_enabled)
        self.memory_verbose = bool(memory_verbose)
        if self.memory_enabled:
            if memory_manager is None:
                from lightnav.memory.manager import MemoryManager

                memory_manager = MemoryManager()
            self.memory_manager = memory_manager
        else:
            self.memory_manager = None
        self._memory_episode_index = -1
        self._memory_episode_open = False
        self._memory_step = 0
        self._last_executed_action: dict | None = None
        self._last_prediction: str | None = None
        self.last_memory_diagnostic: dict[str, Any] | None = None
        self.last_model_inference_ms: float | None = None
        self.last_prompt_chars: int | None = None

        self._last_cluster_id: int | None = None
        self._last_rvq_codes: list[int] | None = None
        self._last_waypoints: np.ndarray | None = None
        self._last_action_waypoint_index: int | None = None
        self._last_action_waypoint: np.ndarray | None = None
        self._last_raw_text: str = ""

    def _clear_last(self) -> None:
        self._last_cluster_id = None
        self._last_rvq_codes = None
        self._last_waypoints = None
        self._last_action_waypoint_index = None
        self._last_action_waypoint = None
        self._last_raw_text = ""

    def reset(self, obs: dict[str, Any] | None = None, *,
              memory_episode_id: str | None = None,
              memory_scene_id: str | None = None,
              memory_rollout_idx: int = 0) -> None:
        """Reset the frame buffer and seed the instruction from ``obs`` if given."""
        self.close_episode()
        self.instruction = extract_instruction(obs) if obs is not None else ""
        self.agent.reset(self.instruction)
        self._clear_last()
        if self.memory_manager is not None:
            self._memory_episode_index += 1
            reset_kwargs = {
                "episode_id": memory_episode_id or f"policy_episode_{self._memory_episode_index}",
                "instruction": self.instruction,
            }
            if memory_scene_id is not None:
                reset_kwargs.update(scene_id=memory_scene_id, rollout_idx=memory_rollout_idx)
            self.memory_manager.reset_episode(**reset_kwargs)
            self._memory_episode_open = True
            self._memory_step = 0
            self._last_executed_action = None
            self._last_prediction = None

    def close_episode(self, *, outcome: str | None = None) -> None:
        """Release the explicit memory of the current episode, if enabled."""
        self.last_memory_diagnostic = None
        self.last_model_inference_ms = None
        self.last_prompt_chars = None
        if self._memory_episode_open:
            if outcome is None:
                self.memory_manager.close_episode()
            else:
                self.memory_manager.close_episode(outcome=outcome)
            self._memory_episode_open = False
            self._last_executed_action = None
            self._last_prediction = None

    def observe(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        """Push the RGB frame into the history buffer without running inference."""
        self.agent.observe(obs["rgb"])

    def stop_action(self) -> dict:
        """The explicit STOP: a zero velocity_control command (sets ``is_stop_called``)."""
        self.last_memory_diagnostic = None
        self.last_model_inference_ms = None
        self.last_prompt_chars = None
        action = first_waypoint_to_velocity_cmd(
            np.zeros(3, dtype=np.float32), self.dt, self.lin_vel_range, self.ang_vel_range
        )
        if self._memory_episode_open:
            self.memory_manager.record_action(action, prediction=None)
            self._last_executed_action = action
        return action

    def _generate_with_v4_memory(
        self, obs: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Build one unchanged LightNav sample, then selectively add V4 facts."""
        from lightnav.memory.memory_context_builder import (
            audit_prompt_injection,
            build_context_text,
            inject_memory_context,
        )
        from lightnav.memory.schemas import ObservationPacket

        manager = self.memory_manager
        step = self._memory_step
        events = manager.observe(ObservationPacket(
            step_id=step, rgb=obs["rgb"], instruction=self.instruction,
            last_action=self._last_executed_action, prediction=self._last_prediction,
        ))
        self._memory_step += 1
        # TrackingAgent assigns this ID to the RGB just observed. Keeping the
        # explicit mapping matters when policy.observe() preloaded other frames.
        manager.record_v4_frame_id(step, self.agent._history_frame_ids[-1])
        base_sample = build_vln_traj_sample(
            self.agent._get_video_tensor(), self.agent.instruction,
            list(self.agent._history_frame_ids), self.engine.bundle,
        )
        visible_frame_ids = tuple(
            int(frame_id)
            for segment in base_sample["video_segments"]
            for frame_id in (segment.get("frame_indices") or ())
        )
        context = manager.get_context(visible_frame_ids=visible_frame_ids)
        budget = min(manager.v4_config.prompt_max_chars, manager.max_context_chars)
        base_prompt = next(
            turn["value"] for turn in base_sample["conversations"]
            if turn.get("from") == "human"
        )
        base_hash = sha256(base_prompt.encode("utf-8")).hexdigest()
        excluded: set[str] = set()
        overflow_drops: list[dict[str, Any]] = []
        fallback_reason: str | None = None
        prompt_audit: dict[str, Any] | None = None
        memory_text = ""
        accepted_records: list[dict[str, Any]] = []
        sample = base_sample
        started = perf_counter()
        while True:
            if context is not None:
                try:
                    augmented = inject_memory_context(base_sample, context, max_chars=budget)
                    prompt_audit = audit_prompt_injection(
                        base_sample, augmented, context, max_chars=budget,
                    )
                    memory_text = build_context_text(context, max_chars=budget)
                    if not memory_text:
                        raise ValueError("V4 formatter returned no prompt-safe text")
                    accepted_records = []
                    for item in context.layer_items:
                        line = item.metadata.get("serialized_text")
                        if not isinstance(line, str) or line not in memory_text:
                            raise ValueError("V4 evidence text did not match the injected block")
                        accepted_records.append({
                            "evidence_ids": list(item.metadata["evidence_ids"]),
                            "text": line,
                        })
                    sample = {
                        **augmented,
                        "_rgm_trace": {
                            "episode_id": manager.working.state.episode_id,
                            "scene_id": manager.v3_scene_id,
                            "rollout_idx": manager.v3_rollout_idx,
                            "memory_backend": "rgm_v4",
                            "memory_use_mode": manager.last_v4_diagnostic["memory_use_mode"],
                            "step_id": step,
                            "original_instruction": self.instruction,
                            "memory_text": memory_text,
                            "accepted_records": accepted_records,
                            "original_prompt_sha256": prompt_audit["base_sha256"],
                            "injected_prompt_sha256": prompt_audit["augmented_sha256"],
                            "injected_memory_chars": len(memory_text),
                            "dropped_candidates": [
                                *manager.last_v4_diagnostic["dropped_candidates"],
                                *overflow_drops,
                            ],
                            "evidence_provenance": [
                                source for item in context.layer_items
                                for source in item.metadata["provenance"]
                            ],
                            "_v4_base_sample": base_sample,
                        },
                    }
                except ValueError as exc:
                    fallback_reason = f"prompt_contract: {exc}"
                    manager.v4_utilization.discard_uninjected_selection()
                    context = None
                    sample = base_sample
                    prompt_audit = None
                    memory_text = ""
                    accepted_records = []
            try:
                text, _ = self.engine.generate(sample)
                break
            except ValueError as exc:
                if context is None or "processor input length" not in str(exc):
                    raise
                # Drop the longest complete item, then let the formatter try a
                # later candidate. Never truncate instruction or visual tokens.
                victim = max(
                    context.layer_items,
                    key=lambda item: len(item.metadata["serialized_text"]),
                )
                victim_ids = tuple(victim.metadata["evidence_ids"])
                excluded.update(victim_ids)
                overflow_drops.append({
                    "memory_id": victim.metadata["memory_id"],
                    "evidence_ids": list(victim_ids),
                    "reason": "processor_token_budget",
                })
                context = manager.reformat_v4_context(frozenset(excluded))
                sample = base_sample
                prompt_audit = None
                memory_text = ""
                accepted_records = []
                if context is None:
                    fallback_reason = "all V4 memory items exceeded the Processor token budget"
        self.last_model_inference_ms = (perf_counter() - started) * 1000.0

        model_input = getattr(self.engine, "last_input_diagnostic", None)
        checked = isinstance(model_input, dict)
        block_visible = bool(
            context is not None and checked
            and model_input.get("memory_text_present_in_final_tokens") is True
        )
        if block_visible and getattr(self.engine, "backend", None) == "vllm_local":
            block_visible = bool(
                model_input.get("memory_text_present_in_vllm_prompt") is True
                and model_input.get("vllm_prompt_matches_final_tokens") is True
            )
        visible_ids = list(model_input.get("model_visible_evidence_ids", [])) if block_visible else []
        active_identity = (manager.working.state.episode_id, manager.v3_rollout_idx)
        historical_visible_ids = []
        for evidence_id in visible_ids:
            record = manager.v3_store.get_evidence(evidence_id)
            if record is not None and (
                (record.episode_id, record.rollout_idx) != active_identity or record.step < step
            ):
                historical_visible_ids.append(evidence_id)
        actual_injection = bool(context is not None and block_visible and visible_ids)
        if actual_injection:
            manager.v4_utilization.confirm_injected_evidence(visible_ids)
        elif context is not None:
            manager.v4_utilization.discard_uninjected_selection()
        v4_diagnostic = dict(manager.last_v4_diagnostic)
        v4_diagnostic.update({
            "prompt_hash_before": base_hash,
            "prompt_hash_after": (
                prompt_audit["augmented_sha256"] if prompt_audit is not None else base_hash
            ),
            "prompt_roundtrip_ok": prompt_audit["roundtrip_ok"] if prompt_audit else True,
            "sample_identity_is_baseline": sample is base_sample,
            "memory_use_mode": (
                manager.last_v4_diagnostic["memory_use_mode"] if context is not None else "NONE"
            ),
            "injected_memory_chars": len(memory_text),
            "injected_memory_tokens": (
                0 if context is None else
                model_input.get("injected_memory_tokens") if checked else None
            ),
            "final_token_ids_sha256": model_input.get("final_token_ids_sha256") if checked else None,
            "vllm_prompt_token_ids_sha256": (
                model_input.get("vllm_prompt_token_ids_sha256") if checked else None
            ),
            "actual_injection": actual_injection if checked or context is None else None,
            "actual_historical_injection": bool(historical_visible_ids)
            if checked or context is None else None,
            "injected_evidence_ids": visible_ids,
            "injected_historical_evidence_ids": historical_visible_ids,
            "dropped_candidates": [*v4_diagnostic["dropped_candidates"], *overflow_drops],
            "processor_length_rejections": len(overflow_drops),
            "prompt_fallback_reason": fallback_reason,
        })
        diagnostic = {
            "episode_id": manager.working.state.episode_id,
            "memory_backend": "rgm_v4",
            "rgm_v4_config": asdict(manager.v4_config),
            "step": step,
            "events_written": len(events),
            "v4_diagnostic": v4_diagnostic,
            "model_input_diagnostic": dict(model_input) if checked else None,
        }
        return text, sample, diagnostic

    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict:
        """Run one inference step and return a Habitat ``velocity_control`` action dict."""
        self.last_memory_diagnostic = None
        self.last_model_inference_ms = None
        self.last_prompt_chars = None
        diagnostic: dict[str, Any] | None = None
        self.agent.observe(obs["rgb"])
        v4_enabled = bool(
            self._memory_episode_open and getattr(
                getattr(self.memory_manager, "v4_config", None), "enabled", False,
            )
        )
        if v4_enabled:
            text, sample, diagnostic = self._generate_with_v4_memory(obs)
            rgm_enabled = False  # V4 diagnostics are finalized in the helper.
        elif self._memory_episode_open:
            from lightnav.memory.memory_context_builder import (
                audit_prompt_injection,
                build_context_text,
                inject_memory_context,
            )
            from lightnav.memory.schemas import ObservationPacket

            # Only robot-visible data enters memory. Habitat ``info`` contains
            # evaluator-only state such as distance_to_goal and oracle_success.
            events = self.memory_manager.observe(
                ObservationPacket(
                    step_id=self._memory_step,
                    rgb=obs["rgb"],
                    instruction=self.instruction,
                    last_action=self._last_executed_action,
                    prediction=self._last_prediction,
                )
            )
            self._memory_step += 1
            sample = build_vln_traj_sample(
                self.agent._get_video_tensor(),
                self.agent.instruction,
                list(self.agent._history_frame_ids),
                self.engine.bundle,
            )
            context = self.memory_manager.get_context()
            base_sample = sample
            prompt_fallback_reason = None
            legacy_rgm_enabled = getattr(
                getattr(self.memory_manager, "rgm_config", None), "enabled", False,
            )
            v3_enabled = getattr(
                getattr(self.memory_manager, "v3_config", None), "enabled", False,
            )
            rgm_enabled = legacy_rgm_enabled or v3_enabled
            try:
                if rgm_enabled and not build_context_text(
                    context, max_chars=getattr(self.memory_manager, "max_context_chars", 600),
                ):
                    sample = base_sample
                else:
                    sample = inject_memory_context(
                        base_sample, context,
                        max_chars=getattr(self.memory_manager, "max_context_chars", 600),
                    )
                prompt_audit = audit_prompt_injection(
                    base_sample, sample, context,
                    max_chars=getattr(self.memory_manager, "max_context_chars", 600),
                ) if rgm_enabled else None
            except ValueError as exc:
                if not rgm_enabled:
                    raise
                sample = base_sample
                prompt_audit = None
                prompt_fallback_reason = str(exc)
            memory_text = build_context_text(
                context, max_chars=getattr(self.memory_manager, "max_context_chars", 600),
            )
            if prompt_fallback_reason is not None:
                memory_text = ""
            injected_lines = memory_text.splitlines()
            if injected_lines and injected_lines[0] in {
                "Episode observations (may be uncertain):",
                "[Navigation memory; uncertain RGB observations]",
                "[Navigation memory; evidence-linked observations]",
            }:
                injected_lines = injected_lines[1:]
            if rgm_enabled:
                if v3_enabled:
                    accepted_records = []
                    if memory_text and sample is not base_sample:
                        for item in context.layer_items:
                            line = item.metadata.get("serialized_text")
                            if not isinstance(line, str) or line not in memory_text:
                                raise ValueError("V3 formatted memory/evidence alignment failed")
                            accepted_records.append({
                                "evidence_ids": list(dict.fromkeys(
                                    ref.event_id for ref in item.evidence_refs
                                )),
                                "text": line,
                            })
                    if len(accepted_records) != len(injected_lines):
                        raise ValueError("V3 serialized record count differs from prompt lines")
                else:
                    accepted_records = [
                        {
                            "evidence_ids": list(dict.fromkeys(
                                ref.event_id for ref in item.evidence_refs
                            )),
                            "text": line,
                        }
                        for item, line in zip(context.layer_items, injected_lines)
                    ]
                sample = {
                    **sample,
                    "_rgm_trace": {
                        "episode_id": self.memory_manager.working.state.episode_id,
                        "scene_id": self.memory_manager.v3_scene_id if v3_enabled else None,
                        "rollout_idx": self.memory_manager.v3_rollout_idx if v3_enabled else 0,
                        "memory_backend": "rgm_v3" if v3_enabled else "legacy_rgm",
                        "step_id": self._memory_step - 1,
                        "original_instruction": self.instruction,
                        "memory_text": memory_text,
                        "accepted_records": accepted_records,
                    },
                }
            if self.memory_verbose or hasattr(self.memory_manager, "scene_graph"):
                extractor = self.memory_manager.extractor
                attempted = getattr(extractor, "last_attempt_step", None) == self._memory_step - 1
                diagnostic = {
                    "step": self._memory_step - 1,
                    "extractor": type(extractor).__name__,
                    "extraction_attempted": attempted,
                    "extraction_error": getattr(extractor, "last_error", None) if attempted else None,
                    "yolo_detection_ms": (
                        getattr(extractor, "last_detection_ms", None) if attempted else None
                    ),
                    "objects": [
                        {"label": event.description.removeprefix("Observed "),
                         "confidence": float(event.confidence)}
                        for event in events if event.kind == "object_observation"
                    ],
                    "events_written": len(events),
                    "graph_nodes": len(self.memory_manager.scene_graph.nodes),
                    "graph_edges": len(self.memory_manager.scene_graph.edges),
                    "retrieved_records": (
                        len(context.events) + len(context.nodes) + len(context.edges)
                        + len(context.layer_items)
                    ) if not v3_enabled else None,
                    "injected_records": len(accepted_records) if v3_enabled else len(injected_lines),
                    "injected_memory_chars": len(memory_text),
                    "injected_memory_text": memory_text,
                }
                if legacy_rgm_enabled:
                    from dataclasses import asdict

                    gate = self.memory_manager.write_gate
                    current_event_ids = {event.event_id for event in events}
                    diagnostic.update({
                        "episode_id": self.memory_manager.working.state.episode_id,
                        "rgm_config": asdict(self.memory_manager.rgm_config),
                        "yolo_candidate_count": len([event for event in events
                                                     if event.kind == "object_observation"]),
                        "yolo_temporal_support": [
                            event.metadata.get("rgm_temporal_support") for event in events
                            if event.kind == "object_observation"
                        ],
                        "yolo_association_ms": (
                            sum(float(event.metadata.get("rgm_association_ms", 0.0))
                                for event in events
                                if event.kind in {"object_observation", "landmark"})
                            if attempted else None
                        ),
                        "write_gate_decisions": [
                            asdict(item) for item in gate.decisions
                            if item.evidence_ids and item.evidence_ids[-1] in current_event_ids
                        ]
                        if gate is not None else [],
                        "retrieval_gate_decisions": list(self.memory_manager.retrieval_gate.decisions),
                        "retrieval_candidates": [
                            {"layer": layer, "kind": item.kind, "step": item.step_id,
                             "evidence_ids": [ref.event_id for ref in item.evidence_refs]}
                            for layer, items in self.memory_manager.last_layer_results.items()
                            for item in items
                        ],
                        "fusion_selected_records": [item.kind for item in context.layer_items],
                        "injected_evidence_ids": list(dict.fromkeys(
                            evidence_id for record in accepted_records
                            for evidence_id in record["evidence_ids"]
                        )),
                        "prompt_gate_decision": asdict(
                            self.memory_manager.prompt_gate.last_selection,
                        ),
                        "prompt_fallback_reason": prompt_fallback_reason,
                        "prompt_hash": prompt_audit,
                        "dino_global_topk": list(self.memory_manager.place_verifier.last_topk)
                        if self.memory_manager.last_place_extraction_attempted else [],
                        "place_decision": asdict(self.memory_manager.last_place_decision)
                        if self.memory_manager.last_place_decision else None,
                        "dino_local_match_count": (
                            self.memory_manager.last_place_decision.local_match_count
                            if self.memory_manager.last_place_decision else None
                        ),
                        "dino_local_inlier_ratio": (
                            self.memory_manager.last_place_decision.local_inlier_ratio
                            if self.memory_manager.last_place_decision else None
                        ),
                        "topology_edge_status": self.memory_manager.last_topology_edge_status,
                        "last_confirmed_place_step": (
                            self.memory_manager.topological_memory.last_observation_step
                        ),
                    })
                if v3_enabled:
                    from dataclasses import asdict

                    diagnostic.pop("retrieved_records", None)
                    tokenizer = getattr(self.engine.bundle, "tokenizer", None)
                    token_delta = None
                    if tokenizer is not None:
                        try:
                            base_human = next(
                                turn["value"] for turn in base_sample["conversations"]
                                if turn.get("from") == "human"
                            )
                            augmented_human = next(
                                turn["value"] for turn in sample["conversations"]
                                if turn.get("from") == "human"
                            )
                            token_delta = (
                                len(tokenizer.encode(augmented_human, add_special_tokens=False))
                                - len(tokenizer.encode(base_human, add_special_tokens=False))
                            )
                        except (KeyError, StopIteration, TypeError, ValueError):
                            token_delta = None
                    v3_diagnostic = dict(self.memory_manager.last_v3_diagnostic)
                    v3_diagnostic.update({
                        "prompt_chars": len(memory_text),
                        "prompt_token_delta": token_delta,
                        "prompt_token_delta_basis": "human_text_tokenizer_encode"
                        if token_delta is not None else "unverified",
                    })
                    diagnostic.update({
                        "episode_id": self.memory_manager.working.state.episode_id,
                        "memory_backend": "rgm_v3",
                        "rgm_v3_config": asdict(self.memory_manager.v3_config),
                        "v3_diagnostic": v3_diagnostic,
                        "prompt_hash": prompt_audit,
                        "injected_evidence_ids": [],
                    })
                if getattr(self.memory_manager, "strategy", "flat") == "hierarchical":
                    diagnostic["memory_strategy"] = "hierarchical"
                    diagnostic["layer_contexts"] = {
                        layer: [item.to_dict() for item in items]
                        for layer, items in self.memory_manager.last_layer_results.items()
                    }
                    if getattr(self.memory_manager, "topology_active", False):
                        topology = self.memory_manager.topological_memory
                        place_result = self.memory_manager.last_place_result
                        place = (
                            topology.get_node(topology.current_place_id)
                            if topology.current_place_id is not None else None
                        )
                        diagnostic.update({
                            "scene_graph_nodes": diagnostic["graph_nodes"],
                            "scene_graph_edges": diagnostic["graph_edges"],
                            "topology_nodes": len(topology.nodes),
                            "topology_edges": len(topology.edges),
                            "memory_episode_id": self.memory_manager.working.state.episode_id,
                            "place_id": topology.current_place_id,
                            "place_extraction_attempted": (
                                self.memory_manager.last_place_extraction_attempted
                            ),
                            "place_extraction_ms": (
                                self.memory_manager.last_place_extraction_ms
                            ),
                            "place_similarity": (
                                place_result.top1_similarity if place_result else None
                            ),
                            "place_created": bool(place_result and place_result.created),
                            "place_matched": bool(place_result and place_result.matched),
                            "place_revisited": bool(place_result and place_result.revisited),
                            "place_transition": (
                                place_result.transition.to_dict()
                                if place_result and place_result.transition else None
                            ),
                            "place_observation_count": place.observation_count if place else 0,
                            "place_visit_count": place.visit_count if place else 0,
                            "long_memory_retrieved": len(
                                self.memory_manager.last_layer_results.get("long", [])
                            ),
                            "place_extraction_error": self.memory_manager.last_place_error,
                        })
            started = perf_counter()
            try:
                text, _ = self.engine.generate(sample)
            finally:
                self.last_model_inference_ms = (perf_counter() - started) * 1000.0
        else:
            started = perf_counter()
            try:
                text, _ = self.engine.generate_from_frames(
                    self.agent._get_video_tensor(),
                    self.agent.instruction,
                    predict_horizon=1,
                    frame_ids=list(self.agent._history_frame_ids),
                    task_type="vlnce_traj",
                )
            finally:
                self.last_model_inference_ms = (perf_counter() - started) * 1000.0
        prompt_chars = getattr(self.engine, "last_prompt_chars", None)
        if isinstance(prompt_chars, int) and prompt_chars >= 0:
            self.last_prompt_chars = prompt_chars
        elif self._memory_episode_open:
            self.last_prompt_chars = len(sample["conversations"][0]["value"])
        if diagnostic is not None:
            diagnostic["prompt_chars"] = self.last_prompt_chars
            if rgm_enabled:
                model_input = getattr(self.engine, "last_input_diagnostic", None)
                diagnostic["model_input_diagnostic"] = (
                    dict(model_input) if isinstance(model_input, dict) else None
                )
                if v3_enabled:
                    v3_diag = diagnostic["v3_diagnostic"]
                    verified = isinstance(model_input, dict)
                    block_visible = (
                        model_input.get("memory_text_present_in_final_tokens") is True
                        if verified else None
                    )
                    if verified and getattr(self.engine, "backend", None) == "vllm_local":
                        block_visible = (
                            block_visible and
                            model_input.get("memory_text_present_in_vllm_prompt") is True and
                            model_input.get("vllm_prompt_matches_final_tokens") is True
                        )
                    visible_ids = (
                        list(model_input.get("literal_model_visible_evidence_ids", []))
                        if block_visible else []
                    )
                    active_identity = (
                        self.memory_manager.working.state.episode_id,
                        self.memory_manager.v3_rollout_idx,
                    )
                    historical_visible_ids = []
                    for evidence_id in visible_ids:
                        record = self.memory_manager.v3_store.get_evidence(evidence_id)
                        if record is not None and (
                            (record.episode_id, record.rollout_idx) != active_identity
                            or record.step < self._memory_step - 1
                        ):
                            historical_visible_ids.append(evidence_id)
                    v3_diag.update({
                        "prompt_hash_before": prompt_audit["base_sha256"]
                        if prompt_audit else None,
                        "prompt_hash_after": prompt_audit["augmented_sha256"]
                        if prompt_audit else None,
                        "actual_injection": bool(block_visible and visible_ids)
                        if verified or not memory_text else None,
                        "actual_historical_injection": bool(
                            block_visible and historical_visible_ids
                        ) if verified or not memory_text else None,
                        "injected_evidence_ids": visible_ids,
                        "injected_historical_evidence_ids": historical_visible_ids,
                        "final_token_count": model_input.get("final_token_count")
                        if verified else None,
                    })
            self.last_memory_diagnostic = diagnostic
            if self.memory_verbose:
                print("[memory] " + json.dumps(diagnostic, ensure_ascii=False), flush=True)
        self._last_raw_text = text

        try:
            waypoints, _ = self.agent.decode_waypoints(text)
        except ValueError as exc:
            logger.warning("%s decode failure (%s); zero-velocity fallback", self.method, exc)
            self._last_cluster_id = -1
            self._last_rvq_codes = None
            self._last_waypoints = None
            self._last_action_waypoint_index = None
            self._last_action_waypoint = None
            action = first_waypoint_to_velocity_cmd(
                np.zeros(3, dtype=np.float32), self.dt, self.lin_vel_range, self.ang_vel_range
            )
            if self._memory_episode_open:
                self.memory_manager.record_action(action, prediction=text)
                self._last_executed_action = action
                self._last_prediction = text
            return action

        # decode_waypoints already validated the tokens; re-parse only to expose the ids.
        if self.method == "rvq":
            codes = parse_rvq_action_tokens(text)
            self._last_cluster_id = int(codes[0])
            self._last_rvq_codes = [int(c) for c in codes]
        else:
            self._last_cluster_id = int(parse_traj_token(text))
            self._last_rvq_codes = None

        self._last_waypoints = np.array(waypoints, dtype=np.float32, copy=True)
        waypoint_index, waypoint = select_action_waypoint(self._last_waypoints)
        self._last_action_waypoint_index = int(waypoint_index)
        self._last_action_waypoint = waypoint.copy()
        action = first_waypoint_to_velocity_cmd(
            waypoint, self.dt, self.lin_vel_range, self.ang_vel_range
        )
        if self._memory_episode_open:
            self.memory_manager.record_action(action, prediction=text)
            self._last_executed_action = action
            self._last_prediction = text
        return action

    def get_info(self) -> dict[str, Any]:
        """Return the last predicted trajectory and ids for logging; minimal dict before any act.

        After a failed decode there is no ``predicted_traj``, but ``raw_text`` still
        carries what the model said so logs and rendered videos can show it.
        """
        if self._last_waypoints is None or self._last_cluster_id is None:
            return {"num_history_frames": self.num_history_frames, "raw_text": self._last_raw_text}
        return {
            "predicted_traj": self._last_waypoints,
            "cluster_id": self._last_cluster_id,
            "rvq_codes": self._last_rvq_codes,
            "raw_text": self._last_raw_text,
            "action_waypoint_index": self._last_action_waypoint_index,
            "action_waypoint": self._last_action_waypoint,
            "num_history_frames": self.num_history_frames,
        }
