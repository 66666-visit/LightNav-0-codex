"""Adaptive L3 Query: local-model Think -> Query -> verified facts -> USE/ABSTAIN.

The mechanism is the active structured-query loop of Choi et al., *Memory-
Environment Dual-Grounding for Embodied Agents* (Findings of EMNLP 2026,
DOI 10.18653/2026.findings-emnlp.16), reduced to the one decision LightNav
needs.  Upstream ``ReActStrQ.run`` dispatches ``Think`` / ``Act`` / ``Query`` and
appends the ``Info:`` result of a read-only scene-graph interface call to the
planning context (``core/planner/procthor/reactstrq.py``); the interface is a
fixed whitelist executed against the live graph
(``core/retriever/interface_procthor.py``).  LightNav keeps the Query half only:

* robot motion remains owned by the unchanged policy and action decoder;
* the controller reads the existing L1 Original card and decides whether one
  L3 query is warranted;
* the query is the existing :class:`L3ClipQuery`, whose cosine ranking is not
  modified or re-tuned;
* facts are read from the existing ``ObjectNode`` / ``SpatialEvidence`` records
  and re-validated locally before any text is rendered;
* a deterministic formatter is the only writer of Prompt text;
* unless at least one fact validates, the caller keeps the original L1 Prompt
  byte-for-byte.

The production controller is :class:`LocalLightNavQueryController`.  It calls
the already-loaded ``VLNInferenceEngine`` through its public ``generate(sample)``
entry point, so no external API, second weight set or duplicate model load is
introduced.  ``MockAdaptiveQueryController`` is the CPU/no-model implementation
used by tests and demos; it is not a runtime default.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from html import escape
from time import perf_counter
from typing import Any, Protocol, Sequence

import numpy as np

from lightnav.memory.adaptive_l3_query_config import AdaptiveL3QueryConfig
from lightnav.memory.clip_features import ClipFeatureEncoder
from lightnav.memory.conceptgraphs_lite import ObjectMap, ObjectNode, ObservedFromEdge
from lightnav.memory.l3_clip_query import L3ClipQuery, L3QueryMatch
from lightnav.memory.local_graph_fusion import _plain_label
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence
from lightnav.memory.spatial_graph_adapter import _relative_position
from lightnav.memory.v3_schemas import parse_evidence_key

NO_QUERY = "NO_QUERY"
QUERY_L3 = "QUERY_L3"
USE = "USE"
ABSTAIN = "ABSTAIN"
DECISIONS = frozenset({NO_QUERY, QUERY_L3, USE, ABSTAIN})

_VERIFIED_HEADER = (
    "[Verified L3 history; not current view; straight-line range; passage unknown]"
)
_FORBIDDEN_DECODING_ENV = "VLN_EVAL_TRAJ_TOP1"
_SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")


def _compact_ref(value: str, prefix: str) -> str:
    """Return the stable leaf of a canonical Object/Evidence JSON key."""
    try:
        parts = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        parts = None
    if isinstance(parts, list) and parts:
        return f"{prefix}:{parts[-1]}"
    text = str(value)
    return f"{prefix}:{text[-24:]}" if text else f"{prefix}:unknown"


def _clean_output(text: object) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = _SPECIAL_TOKEN.sub("", text)
    return " ".join(cleaned.split())


@dataclass(frozen=True)
class L1Snapshot:
    """The exact L1 Original information the controller is allowed to read."""

    step: int
    instruction: str
    l1_status: str
    l1_text: str
    labels: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    item_count: int = 0

    @property
    def has_l1_items(self) -> bool:
        return self.item_count > 0


def build_l1_snapshot(
    *, instruction: str, step: int, l1_status: str, context: MemoryContext | None,
) -> L1Snapshot:
    """Read, never rewrite, the original L1 card that the policy already built."""
    if not isinstance(instruction, str):
        raise TypeError("adaptive L3 instruction must be a string")
    if type(step) is not int or step < 0:
        raise ValueError("adaptive L3 step must be a nonnegative integer")
    items = context.layer_items if context is not None else []
    labels = tuple(dict.fromkeys(
        item.label for item in items if isinstance(item.label, str) and item.label
    ))
    evidence_ids = tuple(dict.fromkeys(
        evidence_id for item in items
        for evidence_id in item.metadata.get("evidence_ids", [])
        if isinstance(evidence_id, str) and evidence_id
    ))
    return L1Snapshot(
        step=step,
        instruction=instruction,
        l1_status=l1_status if isinstance(l1_status, str) else "not_attempted",
        l1_text=context.text if context is not None else "",
        labels=labels,
        evidence_ids=evidence_ids,
        item_count=len(items),
    )


@dataclass(frozen=True)
class AdaptiveQueryDecision:
    """One bounded controller decision plus its raw local-model output."""

    decision: str
    query_text: str | None = None
    object_ids: tuple[str, ...] = ()
    reason_code: str = ""
    raw_output: str | None = None
    latency_ms: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"unknown adaptive L3 decision: {self.decision!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "query_text": self.query_text,
            "object_ids": list(self.object_ids),
            "reason_code": self.reason_code,
            "raw_output": self.raw_output,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class AdaptiveQueryController(Protocol):
    """Two-phase controller: decide to query, then accept real candidates."""

    def decide(self, snapshot: L1Snapshot) -> AdaptiveQueryDecision:
        """Return NO_QUERY or QUERY_L3 (with the text to retrieve)."""

    def select(
        self, snapshot: L1Snapshot, candidates: Sequence[L3QueryMatch],
    ) -> AdaptiveQueryDecision:
        """Return USE (real Object IDs) or ABSTAIN after reading candidate facts."""

    def reset_episode(self, scene_id: str, episode_id: str, rollout_idx: int) -> None:
        """Clear every per-Episode decision."""

    def close_episode(self) -> None:
        """Release all per-Episode state."""


class MockAdaptiveQueryController:
    """Deterministic scripted controller for CPU tests and demos.

    It exercises the same decision contract as the local-model controller but
    never loads or calls a model.  It is not selected by the runtime CLI.
    """

    def __init__(
        self,
        *,
        decide_script: Sequence[AdaptiveQueryDecision] = (),
        select_script: Sequence[AdaptiveQueryDecision] = (),
    ) -> None:
        self._decide_script = tuple(decide_script)
        self._select_script = tuple(select_script)
        self._decide_index = 0
        self._select_index = 0
        self.calls: list[tuple[str, int]] = []
        self.snapshots: list[L1Snapshot] = []

    def reset_episode(self, scene_id: str, episode_id: str, rollout_idx: int) -> None:
        self._decide_index = 0
        self._select_index = 0
        self.calls.clear()
        self.snapshots.clear()

    def close_episode(self) -> None:
        self.reset_episode("", "", 0)

    def decide(self, snapshot: L1Snapshot) -> AdaptiveQueryDecision:
        self.calls.append(("decide", snapshot.step))
        self.snapshots.append(snapshot)
        if self._decide_index >= len(self._decide_script):
            return AdaptiveQueryDecision(NO_QUERY, reason_code="mock_exhausted")
        decision = self._decide_script[self._decide_index]
        self._decide_index += 1
        return decision

    def select(
        self, snapshot: L1Snapshot, candidates: Sequence[L3QueryMatch],
    ) -> AdaptiveQueryDecision:
        self.calls.append(("select", snapshot.step))
        if self._select_index >= len(self._select_script):
            return AdaptiveQueryDecision(ABSTAIN, reason_code="mock_exhausted")
        decision = self._select_script[self._select_index]
        self._select_index += 1
        return decision


def _candidate_alias(rank: int) -> str:
    return f"c{rank}"


def _candidate_line(rank: int, match: L3QueryMatch) -> str:
    refs = ", ".join(_compact_ref(value, "evidence") for value in match.evidence_ids[:2])
    if len(match.evidence_ids) > 2:
        refs += f", +{len(match.evidence_ids) - 2}"
    center = ",".join(f"{value:.1f}" for value in match.center_world)
    low = ",".join(f"{value:.1f}" for value in match.bbox_3d[0])
    high = ",".join(f"{value:.1f}" for value in match.bbox_3d[1])
    return (
        f"{_candidate_alias(rank).upper()}: class={_plain_label(match.class_name)}, "
        f"cosine={match.similarity:.3f}, center=({center})m, "
        f"bbox=({low})..({high})m, evidence={len(match.evidence_ids)} "
        f"({refs}), latest_step={match.latest_step}, "
        f"observations={match.observation_count}, "
        f"detector_conf={match.detector_confidence:.2f}"
    )


class LocalLightNavQueryController:
    """Controller backed by the already-loaded local LightNav inference engine.

    The engine is passed in by the policy; this class never constructs a model,
    opens a socket or reads an API key.  It builds a bounded text prompt with the
    current video segment layout produced by the existing sample builder (so the
    processor contract and ViT cache are reused), asks for one decision line and
    parses it strictly.  Any unparseable output, exception or missing input is
    converted to NO_QUERY / ABSTAIN by the caller.
    """

    def __init__(
        self,
        engine: Any,
        video_provider: Any,
        *,
        max_new_tokens: int = 24,
        max_query_chars: int = 256,
        max_candidates: int = 8,
    ) -> None:
        if not callable(getattr(engine, "generate", None)):
            raise ValueError("adaptive L3 controller requires the local inference engine")
        if not callable(video_provider):
            raise ValueError("adaptive L3 controller requires a current-video provider")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("adaptive L3 controller max_new_tokens must be positive")
        if type(max_query_chars) is not int or max_query_chars < 1:
            raise ValueError("adaptive L3 controller max_query_chars must be positive")
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError("adaptive L3 controller max_candidates must be positive")
        if os.environ.get(_FORBIDDEN_DECODING_ENV, "0").lower() in {"1", "true", "yes"}:
            # This env restriction masks the logits to trajectory ids, so a text
            # decision cannot be produced.  Refuse instead of silently NO_QUERY.
            raise ValueError(
                f"{_FORBIDDEN_DECODING_ENV}=1 restricts decoding to action tokens; "
                "the adaptive L3 controller needs unrestricted local decoding"
            )
        self.engine = engine
        self.video_provider = video_provider
        self.max_new_tokens = max_new_tokens
        self.max_query_chars = max_query_chars
        self.max_candidates = max_candidates

    def reset_episode(self, scene_id: str, episode_id: str, rollout_idx: int) -> None:
        return None

    def close_episode(self) -> None:
        return None

    def _generate(self, prompt: str) -> tuple[str, float]:
        from lightnav.inference.samples import build_vln_traj_sample

        video, frame_ids = self.video_provider()
        if video is None:
            raise ValueError("current video is unavailable")
        bundle = getattr(self.engine, "bundle", None)
        template = build_vln_traj_sample(
            video, "", list(frame_ids) if frame_ids is not None else [], bundle,
        )
        segments = template["video_segments"]
        placeholders = " ".join(["<video>"] * len(segments))
        sample = {
            "video_segments": segments,
            "conversations": [
                {"from": "human", "value": f"{placeholders}\n{prompt}"},
                {"from": "gpt", "value": "placeholder"},
            ],
            "video_fps": getattr(bundle, "video_fps", 4),
            "_allow_vit_cache": True,
            "_skip_normalize": True,
        }
        started = perf_counter()
        text, _ = self.engine.generate(sample, self.max_new_tokens)
        return _clean_output(text), (perf_counter() - started) * 1000.0

    def decide(self, snapshot: L1Snapshot) -> AdaptiveQueryDecision:
        l1_text = snapshot.l1_text.strip() or "(no L1 landmark was retrieved)"
        prompt = (
            "You are the read-only memory query controller of a navigation robot. "
            "You never choose a robot action. The observation video is the current "
            "state.\n"
            f"Navigation instruction: {snapshot.instruction}\n"
            f"Step: {snapshot.step}\n"
            f"L1 memory status: {snapshot.l1_status}\n"
            f"L1 memory card:\n{l1_text}\n\n"
            "Decide whether this L1 information is already sufficient to follow the "
            "instruction. Reply with exactly one line:\n"
            "NO_QUERY\n"
            "or\n"
            "QUERY_L3: <short object phrase to look up in the historical 3D map>\n"
            "Query only when a task-relevant landmark is missing, or a historical "
            "spatial fact must be checked. A class label is not an instance identity."
        )
        try:
            raw, latency = self._generate(prompt)
        except Exception as exc:  # noqa: BLE001 - controller failure must fail closed
            return AdaptiveQueryDecision(
                NO_QUERY, reason_code="controller_error", error=f"{type(exc).__name__}: {exc}",
            )
        match = re.match(r"\s*(NO_QUERY|QUERY_L3)\b[:\s]*(.*)", raw, re.IGNORECASE | re.DOTALL)
        if match is None:
            return AdaptiveQueryDecision(
                NO_QUERY, reason_code="unparseable_decision", raw_output=raw,
                latency_ms=latency,
            )
        decision = match.group(1).upper()
        if decision == NO_QUERY:
            return AdaptiveQueryDecision(
                NO_QUERY, reason_code="controller_no_query", raw_output=raw,
                latency_ms=latency,
            )
        query = match.group(2).strip()
        if not query:
            query = " ".join(snapshot.instruction.split())
        query = query[: self.max_query_chars]
        return AdaptiveQueryDecision(
            QUERY_L3, query_text=query, reason_code="controller_query",
            raw_output=raw, latency_ms=latency,
        )

    def select(
        self, snapshot: L1Snapshot, candidates: Sequence[L3QueryMatch],
    ) -> AdaptiveQueryDecision:
        visible = list(candidates)[: self.max_candidates]
        if not visible:
            return AdaptiveQueryDecision(ABSTAIN, reason_code="no_candidates")
        lines = "\n".join(_candidate_line(index, match)
                          for index, match in enumerate(visible, start=1))
        l1_text = snapshot.l1_text.strip() or "(no L1 landmark was retrieved)"
        prompt = (
            "You are the read-only memory query controller of a navigation robot. "
            "You never choose a robot action.\n"
            f"Navigation instruction: {snapshot.instruction}\n"
            f"L1 memory card:\n{l1_text}\n"
            f"Historical 3D-map query candidates:\n{lines}\n\n"
            "Select only aliases backed by the listed historical Evidence. Same-class "
            "objects are different instances and must not be merged. Reply with exactly "
            "one line:\nABSTAIN\nor\nUSE C1 [C2 ...]\n"
            "Use only the aliases printed above."
        )
        try:
            raw, latency = self._generate(prompt)
        except Exception as exc:  # noqa: BLE001
            return AdaptiveQueryDecision(
                ABSTAIN, reason_code="controller_error", error=f"{type(exc).__name__}: {exc}",
            )
        match = re.match(r"\s*(ABSTAIN|USE)\b(.*)", raw, re.IGNORECASE | re.DOTALL)
        if match is None:
            return AdaptiveQueryDecision(
                ABSTAIN, reason_code="unparseable_selection", raw_output=raw,
                latency_ms=latency,
            )
        if match.group(1).upper() == ABSTAIN:
            return AdaptiveQueryDecision(
                ABSTAIN, reason_code="controller_abstain", raw_output=raw,
                latency_ms=latency,
            )
        aliases = re.findall(r"\bc(\d+)\b", match.group(2), re.IGNORECASE)
        if not aliases:
            return AdaptiveQueryDecision(
                ABSTAIN, reason_code="empty_selection", raw_output=raw, latency_ms=latency,
            )
        indexes = sorted({int(value) for value in aliases})
        if any(index < 1 or index > len(visible) for index in indexes):
            return AdaptiveQueryDecision(
                ABSTAIN, reason_code="unknown_candidate_alias", raw_output=raw,
                latency_ms=latency,
            )
        object_ids = tuple(visible[index - 1].object_id for index in indexes)
        return AdaptiveQueryDecision(
            USE, object_ids=object_ids, reason_code="controller_use",
            raw_output=raw, latency_ms=latency,
        )


@dataclass(frozen=True)
class VerifiedL3Fact:
    """One ObjectNode whose identity, geometry and full Evidence passed validation."""

    object_id: str
    class_name: str
    center_world: tuple[float, float, float]
    bbox_3d: tuple[tuple[float, float, float], tuple[float, float, float]]
    observation_count: int
    evidence_ids: tuple[str, ...]
    evidence_steps: tuple[int, ...]
    evidence_confidence: float
    source: str
    latest_step: int
    latest_evidence_id: str
    geometry_status: str
    similarity: float
    rank: int
    provenance: tuple[dict[str, Any], ...]

    @property
    def compact_ref(self) -> str:
        return _compact_ref(self.object_id, "object")


def _observed_from_index(object_map: ObjectMap) -> dict[str, list[ObservedFromEdge]]:
    index: dict[str, list[ObservedFromEdge]] = {}
    for edge in object_map.edges:
        if isinstance(edge, ObservedFromEdge):
            index.setdefault(edge.evidence_id, []).append(edge)
    return index


def validate_l3_fact(
    object_map: ObjectMap,
    match: L3QueryMatch,
    *,
    current_step: int,
    min_confidence: float,
) -> tuple[VerifiedL3Fact | None, str | None]:
    """Re-read one matched ObjectNode and its Evidence; never derive new geometry.

    A fact is eligible only when every Evidence record of the node exists, belongs
    to the canonical key, names the same class, has ``geometry_status=observed``,
    meets the confidence floor, is strictly older than ``current_step``, and has
    exactly one matching ``observed_from`` viewpoint edge.  A node with a partial
    or current-step Evidence set is rejected as a whole because its fused centre
    and bounding box use all of its observations.
    """
    if not isinstance(object_map, ObjectMap):
        raise TypeError("object_map must be an ObjectMap")
    if type(current_step) is not int or current_step < 0:
        raise ValueError("current_step must be nonnegative")
    node = object_map.objects.get(match.object_id)
    if node is None:
        return None, "object_missing"
    if not isinstance(node, ObjectNode):
        return None, "object_missing"
    if not node.evidence_ids:
        return None, "no_evidence"
    if node.observation_count != len(node.evidence_ids):
        return None, "observation_count_mismatch"
    if tuple(node.evidence_ids) != tuple(match.evidence_ids):
        return None, "evidence_set_changed"
    if node.geometry_status != "observed":
        return None, f"geometry_{node.geometry_status}"
    center = np.asarray(node.center_world, dtype=np.float64)
    bbox = np.asarray(node.bbox_3d, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        return None, "invalid_centre"
    if bbox.shape != (2, 3) or not np.isfinite(bbox).all() or np.any(bbox[1] < bbox[0]):
        return None, "invalid_bbox"

    edge_index = _observed_from_index(object_map)
    provenance: list[dict[str, Any]] = []
    steps: list[int] = []
    confidences: list[float] = []
    source: str | None = None
    for evidence_id in node.evidence_ids:
        record = object_map.evidence.get(evidence_id)
        if record is None:
            return None, "evidence_missing"
        try:
            scene_id, episode_id, rollout_idx, _ = parse_evidence_key(evidence_id)
        except ValueError:
            return None, "evidence_key_invalid"
        if (record.scene_id, record.episode_id, record.rollout_idx) != (
            scene_id, episode_id, rollout_idx,
        ):
            return None, "evidence_identity_mismatch"
        if record.class_name.casefold() != node.class_name.casefold():
            return None, "evidence_class_mismatch"
        if record.geometry_status != "observed":
            return None, f"evidence_geometry_{record.geometry_status}"
        if float(record.confidence) < float(min_confidence):
            return None, "evidence_below_confidence"
        if record.step_id >= current_step:
            return None, "evidence_not_historical"
        edges = edge_index.get(evidence_id, [])
        if len(edges) != 1:
            return None, "observed_from_edge_missing"
        edge = edges[0]
        if edge.object_id != node.object_id:
            return None, "observed_from_edge_mismatch"
        viewpoint = object_map.viewpoints.get(edge.viewpoint_id)
        if viewpoint is None:
            return None, "viewpoint_missing"
        if (viewpoint.scene_id, viewpoint.episode_id, viewpoint.rollout_idx,
                viewpoint.step_id) != (scene_id, episode_id, rollout_idx, record.step_id):
            return None, "viewpoint_identity_mismatch"
        steps.append(int(record.step_id))
        confidences.append(float(record.confidence))
        source = record.source
        provenance.append({
            "scene_id": record.scene_id,
            "episode_id": record.episode_id,
            "rollout_idx": record.rollout_idx,
            "step": record.step_id,
            "evidence_id": evidence_id,
            "source": record.source,
            "confidence": record.confidence,
            "geometry_status": record.geometry_status,
        })
    latest_index = max(range(len(steps)), key=lambda index: steps[index])
    return VerifiedL3Fact(
        object_id=node.object_id,
        class_name=node.class_name,
        center_world=tuple(float(value) for value in node.center_world),
        bbox_3d=tuple(
            tuple(float(value) for value in corner) for corner in node.bbox_3d
        ),
        observation_count=int(node.observation_count),
        evidence_ids=tuple(node.evidence_ids),
        evidence_steps=tuple(steps),
        evidence_confidence=min(confidences),
        source=source or "l3_segmentation",
        latest_step=steps[latest_index],
        latest_evidence_id=node.evidence_ids[latest_index],
        geometry_status=node.geometry_status,
        similarity=float(match.similarity),
        rank=int(match.rank),
        provenance=tuple(provenance),
    ), None


def _fact_line(fact: VerifiedL3Fact, agent_pose: Any) -> str:
    label = _plain_label(fact.class_name) or "object"
    n = len(fact.evidence_ids)
    latest = _compact_ref(fact.latest_evidence_id, "evidence")
    if n == 1:
        evidence = latest
    else:
        evidence = f"{n} records, latest {latest}"
    relative = _relative_position(agent_pose, fact.center_world)
    if relative is None:
        geometry = "current relative geometry unavailable"
    else:
        distance, bearing = relative
        degrees = 0.0 if abs(bearing) < 0.5 else bearing
        geometry = f"{distance:.1f} m, bearing {degrees:+.0f} deg (left +)"
    return (
        f"- {label} ({fact.compact_ref}; {evidence}; step {fact.latest_step}; "
        f"{fact.observation_count} obs; conf {fact.evidence_confidence:.2f}); "
        f"{geometry}; passage unknown."
    )


def format_verified_facts(
    facts: Sequence[VerifiedL3Fact], *, agent_pose: Any, max_chars: int,
) -> MemoryContext | None:
    """Render only validated facts; the model's free text is never copied."""
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("adaptive L3 prompt budget must be a positive integer")
    lines: list[str] = []
    items: list[LayerMemoryItem] = []
    used = len(_VERIFIED_HEADER)
    for fact in facts:
        line = _fact_line(fact, agent_pose)
        safe_line = escape(line, quote=False)
        if used + 1 + len(safe_line) > max_chars:
            continue
        used += 1 + len(safe_line)
        lines.append(line)
        items.append(LayerMemoryItem(
            layer="long",
            kind="adaptive_l3_verified_fact",
            description=line,
            step_id=fact.latest_step,
            confidence=fact.evidence_confidence,
            source="l3_clip_query",
            evidence_refs=[
                ObservationEvidence(
                    event_id=evidence_id,
                    step_id=step,
                    label=fact.class_name,
                    confidence=fact.evidence_confidence,
                    source=fact.source,
                    node_id=fact.object_id,
                )
                for evidence_id, step in zip(fact.evidence_ids, fact.evidence_steps)
            ],
            label=_plain_label(fact.class_name) or None,
            metadata={
                "evidence_ids": list(fact.evidence_ids),
                "serialized_text": safe_line,
                "provenance": [dict(record) for record in fact.provenance],
                "role": "adaptive_l3_fact",
                "object_id": fact.object_id,
                "object_ref": fact.compact_ref,
                "class_name": fact.class_name,
                "similarity": fact.similarity,
                "similarity_semantics": "raw_cosine_not_confidence",
                "rank": fact.rank,
                "observation_count": fact.observation_count,
                "center_world": list(fact.center_world),
                "bbox_3d": [list(corner) for corner in fact.bbox_3d],
                "geometry_status": fact.geometry_status,
                "traversability_status": "unknown",
            },
        ))
    if not items:
        return None
    return MemoryContext(text=_VERIFIED_HEADER + "\n" + "\n".join(lines), layer_items=items)


def _map_signature(object_map: ObjectMap) -> tuple[tuple[Any, ...], ...]:
    return tuple(sorted(
        (
            node.object_id,
            len(node.evidence_ids),
            int(node.clip_feature_observations),
        )
        for node in object_map.objects.values()
    ))


class AdaptiveL3QueryRunner:
    """Orchestrate one bounded controller query over the live L3 map.

    A query is attempted only when the controller asks for one.  The runner
    never mutates the map, never recomputes geometry, never changes CLIP
    scores, and returns ``None`` unless a formatter produced a context from
    fully validated records.
    """

    def __init__(
        self,
        *,
        config: AdaptiveL3QueryConfig,
        controller: AdaptiveQueryController,
        encoder: ClipFeatureEncoder,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> None:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("adaptive L3 top_k must be a positive integer")
        if (
            not isinstance(min_similarity, (int, float))
            or not math.isfinite(float(min_similarity))
            or not -1.0 <= float(min_similarity) <= 1.0
        ):
            raise ValueError("adaptive L3 min_similarity must be finite and in [-1, 1]")
        self.config = config
        self.controller = controller
        self.encoder = encoder
        self.top_k = top_k
        self.min_similarity = float(min_similarity)
        self._identity: tuple[str, str, int] | None = None
        self._last_query: tuple[tuple[tuple[Any, ...], ...], str] | None = None
        self._last_query_step: int | None = None
        self.last_diagnostic: dict[str, Any] = {}
        self.stats: dict[str, Any] = self._empty_stats()
        self.last_episode_stats: dict[str, Any] = {}

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "steps": 0,
            "controller_calls": 0,
            "queries_attempted": 0,
            "queries_with_match": 0,
            "queries_without_match": 0,
            "injected": 0,
            "abstained": 0,
            "errors": 0,
            "skipped_reasons": {},
        }

    def reset_episode(self, scene_id: str, episode_id: str, rollout_idx: int = 0) -> None:
        self._identity = (scene_id, episode_id, rollout_idx)
        self._last_query = None
        self._last_query_step = None
        self.last_diagnostic = {}
        self.stats = self._empty_stats()
        self.controller.reset_episode(scene_id, episode_id, rollout_idx)

    def close_episode(self) -> None:
        self.last_episode_stats = {
            **self.stats,
            "skipped_reasons": dict(self.stats["skipped_reasons"]),
        }
        self._identity = None
        self._last_query = None
        self._last_query_step = None
        self.last_diagnostic = {}
        self.controller.close_episode()

    def _diagnostic(self, *, step: int, instruction: str, status: str) -> dict[str, Any]:
        return {
            "step": step,
            "scene_id": self._identity[0] if self._identity is not None else None,
            "episode_id": self._identity[1] if self._identity is not None else None,
            "rollout_idx": self._identity[2] if self._identity is not None else None,
            "instruction": instruction,
            "l1_status": status,
            "decisions": [],
            "query": None,
            "query_executed": False,
            "candidate_count": 0,
            "candidates": [],
            "selected_object_ids": [],
            "validated_object_ids": [],
            "validated_evidence_ids": [],
            "rejected": [],
            "injected": False,
            "prompt_chars": 0,
            "skipped_reason": None,
            "error": None,
            "controller_ms": 0.0,
            "query_ms": 0.0,
        }

    def _finish(self, diagnostic: dict[str, Any], reason: str | None) -> None:
        if reason is not None:
            diagnostic["skipped_reason"] = reason
            reasons = self.stats["skipped_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1
            self.stats["abstained"] += 1
        diagnostic["stats"] = self.summary()
        self.last_diagnostic = diagnostic

    def run(
        self,
        *,
        object_map: ObjectMap | None,
        step: int,
        instruction: str,
        l1_status: str,
        l1_context: MemoryContext | None,
        agent_pose: Any,
        max_chars: int,
    ) -> MemoryContext | None:
        if not self.config.enabled:
            return None
        if object_map is None or self.encoder is None:
            self.last_diagnostic = {
                "step": step, "skipped_reason": "l3_map_or_encoder_unavailable",
                "query_executed": False, "candidate_count": 0, "injected": False,
            }
            return None
        if agent_pose is None:
            self.last_diagnostic = {
                "step": step, "skipped_reason": "no_current_pose",
                "query_executed": False, "candidate_count": 0, "injected": False,
            }
            return None
        diagnostic = self._diagnostic(step=step, instruction=instruction, status=l1_status)
        self.last_diagnostic = diagnostic
        self.stats["steps"] += 1
        snapshot = build_l1_snapshot(
            instruction=instruction, step=step, l1_status=l1_status, context=l1_context,
        )
        try:
            decision = self.controller.decide(snapshot)
        except Exception as exc:  # noqa: BLE001 - a sidecar must not break L1
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            self.stats["errors"] += 1
            self._finish(diagnostic, "controller_error")
            return None
        self.stats["controller_calls"] += 1
        diagnostic["decisions"].append({"phase": "decide", **decision.to_dict()})
        diagnostic["controller_ms"] += decision.latency_ms
        if decision.decision != QUERY_L3:
            self._finish(diagnostic, "controller_no_query")
            return None
        query = " ".join((decision.query_text or instruction).split())
        if not query:
            self._finish(diagnostic, "empty_query")
            return None
        diagnostic["query"] = query
        signature = _map_signature(object_map)
        if self._last_query == (signature, query):
            diagnostic["steps_since_last_query"] = (
                step - self._last_query_step if self._last_query_step is not None else None
            )
            self._finish(diagnostic, "duplicate_query_for_unchanged_map")
            return None
        self._last_query = (signature, query)
        self._last_query_step = step
        diagnostic["query_executed"] = True
        self.stats["queries_attempted"] += 1
        try:
            result = L3ClipQuery(object_map, self.encoder).query(
                query, self.top_k, min_similarity=self.min_similarity,
            )
        except Exception as exc:  # noqa: BLE001
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            self.stats["errors"] += 1
            self._finish(diagnostic, "query_error")
            return None
        diagnostic["query_ms"] = result.encode_ms
        diagnostic["candidate_count"] = len(result.matches)
        diagnostic["candidates"] = [match.to_dict() for match in result.matches]
        if not result.matches:
            self.stats["queries_without_match"] += 1
            self._finish(diagnostic, result.skipped_reason or "no_match")
            return None
        self.stats["queries_with_match"] += 1
        try:
            selection = self.controller.select(snapshot, result.matches)
        except Exception as exc:  # noqa: BLE001
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            self.stats["errors"] += 1
            self._finish(diagnostic, "controller_error")
            return None
        diagnostic["decisions"].append({"phase": "select", **selection.to_dict()})
        diagnostic["controller_ms"] += selection.latency_ms
        if selection.decision != USE:
            self._finish(diagnostic, selection.reason_code or "controller_abstain")
            return None
        wanted = set(selection.object_ids)
        selected = [match for match in result.matches if match.object_id in wanted]
        diagnostic["selected_object_ids"] = [match.object_id for match in selected]
        if not selected:
            self._finish(diagnostic, "selection_not_in_candidates")
            return None
        facts: list[VerifiedL3Fact] = []
        for match in selected:
            fact, reason = validate_l3_fact(
                object_map, match, current_step=step,
                min_confidence=self.config.min_evidence_confidence,
            )
            if fact is None:
                diagnostic["rejected"].append({
                    "object_id": match.object_id, "reason": reason,
                })
                continue
            facts.append(fact)
        if not facts:
            self._finish(diagnostic, "no_verified_fact")
            return None
        context = format_verified_facts(facts, agent_pose=agent_pose, max_chars=max_chars)
        if context is None:
            self._finish(diagnostic, "prompt_budget_excluded")
            return None
        # Report only facts the formatter actually serialized; a fact dropped
        # for budget must not appear as validated/injected in the audit.
        included = {
            item.metadata["object_id"] for item in context.layer_items
            if item.metadata.get("role") == "adaptive_l3_fact"
        }
        rendered = [fact for fact in facts if fact.object_id in included]
        diagnostic["validated_object_ids"] = [fact.object_id for fact in rendered]
        diagnostic["validated_evidence_ids"] = [
            evidence_id for fact in rendered for evidence_id in fact.evidence_ids
        ]
        diagnostic["injected"] = True
        diagnostic["prompt_chars"] = len(context.text)
        self.stats["injected"] += 1
        diagnostic["stats"] = self.summary()
        self.last_diagnostic = diagnostic
        return context

    def summary(self) -> dict[str, Any]:
        return {**self.stats, "skipped_reasons": dict(self.stats["skipped_reasons"])}
