"""Instruction-grounded local graph evidence for a frozen LightNav policy.

The graph restricts historical retrieval to landmarks linked to the current
DINO visual-place hypothesis. A visual match and its inferred edges do not
prove a physical revisit or traversability. This module only supplies bounded
observations to the existing prompt adapter; it never chooses an action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from typing import Iterable

from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, MemoryEvent, ObservationEvidence
from lightnav.memory.v3_retrieval import _terms
from lightnav.memory.v3_schemas import EvidenceRecord, canonical_evidence_key
from lightnav.memory.v3_store import V3MemoryStore

_HEADER = "[Navigation Memory]"
_DETECTION_KINDS = frozenset({"object_observation", "landmark"})
_MIN_CONFIDENCE = 0.5


def _plain_label(value: str) -> str:
    """Keep detector labels short and unable to introduce prompt markers."""
    words = [part for part in value.split() if part]
    cleaned = " ".join("".join(char for char in word if char.isalnum() or char == "-")
                       for word in words)
    return cleaned[:48].strip()


def _camera_sector(box: object, width: int) -> str | None:
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1 or x1 < 0 or x2 > width:
        return None
    center = (x1 + x2) / (2 * width)
    return "left" if center < 1 / 3 else "right" if center > 2 / 3 else "center"


def _provenance(record: EvidenceRecord) -> dict[str, object]:
    return {
        "scene_id": record.scene_id,
        "episode_id": record.episode_id,
        "rollout_idx": record.rollout_idx,
        "step": record.step,
        "evidence_id": record.key,
    }


@dataclass(frozen=True)
class _Candidate:
    record: EvidenceRecord
    label: str
    role: str
    sector: str | None = None

    def line(self) -> str:
        if self.role == "current":
            direction = f" in the {self.sector} of the current view" if self.sector else ""
            return f"Current RGB: instruction landmark {self.label}{direction} (detected; passage unknown)."
        return (
            f"Local history: {self.label} was seen near a visually similar area "
            "(possible revisit; current direction and passage unknown)."
        )

    def signature(self) -> tuple[str, str, str | None]:
        return self.role, self.label.casefold(), self.sector


class LocalGraphFusion:
    """Build one compact, provenance-linked context from current and 1-hop facts."""

    def __init__(self, config: LocalGraphConfig) -> None:
        self.config = config
        self._episode: tuple[str, str, int] | None = None
        self._last_signature: tuple[object, ...] | None = None
        self._last_injected_step: int | None = None
        self._pending: tuple[int, tuple[object, ...], frozenset[str]] | None = None

    def reset_episode(self, scene_id: str, episode_id: str, rollout_idx: int) -> None:
        self._episode = (scene_id, episode_id, rollout_idx)
        self._last_signature = None
        self._last_injected_step = None
        self._pending = None

    def close_episode(self) -> None:
        self._episode = None
        self._last_signature = None
        self._last_injected_step = None
        self._pending = None

    def discard_selection(self) -> None:
        """Release a context that was rejected by prompt or token budgeting."""
        self._pending = None

    def confirm_injected(self, evidence_ids: Iterable[str], step: int) -> None:
        """Count a selection only after the prompt actually contains all its facts."""
        pending = self._pending
        self._pending = None
        if pending is None:
            raise ValueError("local graph has no pending selection to confirm")
        pending_step, signature, selected_ids = pending
        if step != pending_step:
            raise ValueError("local graph injection step differs from selection")
        confirmed = set(evidence_ids)
        if not confirmed or not confirmed <= selected_ids:
            raise ValueError("confirmed evidence differs from local graph selection")
        if confirmed == selected_ids:
            self._last_signature = signature
            self._last_injected_step = step

    def build(
        self, store: V3MemoryStore, *, scene_id: str, episode_id: str,
        rollout_idx: int, step: int, instruction: str,
        current_place_id: str | None, rgb_width: int,
        events: list[MemoryEvent],
    ) -> MemoryContext | None:
        self._pending = None
        if not self.config.enabled:
            return None
        identity = (scene_id, episode_id, rollout_idx)
        if self._episode != identity or store.active_episode != identity:
            raise ValueError("local graph selection requires the active Scene/Episode")
        if type(step) is not int or step < 0 or type(rgb_width) is not int or rgb_width < 1:
            raise ValueError("step must be nonnegative and RGB width must be positive")
        if store.scope != self.config.scope:
            raise ValueError("local graph and evidence store scopes differ")

        current = self._current_candidates(
            store, scene_id, episode_id, rollout_idx, step,
            instruction, rgb_width, events,
        )
        history = self._historical_candidates(
            store, scene_id, episode_id, rollout_idx, step,
            instruction, current_place_id,
        )
        # Preserve one historical anchor when both channels have evidence.
        selected: list[_Candidate] = current[:self.config.max_landmarks]
        if history and selected:
            candidate = next((item for item in history if item.label.casefold() not in {
                fact.label.casefold() for fact in selected
            }), history[0])
            if len(selected) >= self.config.max_landmarks and self.config.max_landmarks > 1:
                selected[-1] = candidate
            elif len(selected) < self.config.max_landmarks:
                selected.append(candidate)
        elif history:
            selected = history[:self.config.max_landmarks]
        if not selected:
            return None

        signature: tuple[object, ...] = (
            current_place_id, tuple(candidate.signature() for candidate in selected),
        )
        if (signature == self._last_signature and self._last_injected_step is not None
                and step - self._last_injected_step < self.config.reactivation_steps):
            return None

        lines: list[str] = []
        items: list[LayerMemoryItem] = []
        used = len(_HEADER)
        for candidate in selected:
            line = candidate.line()
            safe_line = escape(line, quote=False)
            if used + 1 + len(safe_line) > self.config.prompt_max_chars:
                continue
            record = candidate.record
            used += 1 + len(safe_line)
            lines.append(line)
            items.append(LayerMemoryItem(
                layer="short" if candidate.role == "current" else "long",
                kind=f"local_graph_{candidate.role}_landmark", description=line,
                step_id=record.step,
                confidence=record.raw_detector_confidence or 0.0,
                source="local_graph_fusion",
                evidence_refs=[ObservationEvidence(
                    event_id=record.key, step_id=record.step, label=candidate.label,
                    confidence=record.raw_detector_confidence or 0.0,
                    source=record.detector_name or "rgb_detector",
                )],
                label=candidate.label,
                metadata={
                    "evidence_ids": [record.key],
                    "serialized_text": safe_line,
                    "provenance": [_provenance(record)],
                    "role": candidate.role,
                    "current_view_sector": candidate.sector,
                    "place_id": current_place_id if candidate.role == "history" else None,
                    "place_identity": "unverified" if candidate.role == "history" else None,
                },
            ))
        if not items:
            return None
        self._pending = (
            step, signature,
            frozenset(ref.event_id for item in items for ref in item.evidence_refs),
        )
        return MemoryContext(text=_HEADER + "\n" + "\n".join(lines), layer_items=items)

    @staticmethod
    def _current_candidates(
        store: V3MemoryStore, scene_id: str, episode_id: str,
        rollout_idx: int, step: int, instruction: str,
        rgb_width: int, events: list[MemoryEvent],
    ) -> list[_Candidate]:
        by_label: dict[str, _Candidate] = {}
        for event in events:
            if (event.kind not in _DETECTION_KINDS or event.step_id != step
                    or event.episode_id != episode_id
                    or event.metadata.get("rgm_adopted", True) is False):
                continue
            label = event.description.removeprefix("Observed ").strip()
            plain = _plain_label(label)
            if not plain or not (_terms(label) & _terms(instruction)):
                continue
            key = canonical_evidence_key(scene_id, episode_id, rollout_idx, event.event_id)
            record = store.get_evidence(key)
            if (record is None or record.event_type not in _DETECTION_KINDS
                    or record.observation_status != "observed"
                    or not record.provenance_validity
                    or (record.raw_detector_confidence or 0.0) < _MIN_CONFIDENCE):
                continue
            sector = _camera_sector(event.metadata.get("bbox_xyxy"), rgb_width)
            candidate = _Candidate(record, plain, "current", sector)
            prior = by_label.get(plain.casefold())
            if prior is None or (record.raw_detector_confidence or 0.0) > (
                prior.record.raw_detector_confidence or 0.0
            ):
                by_label[plain.casefold()] = candidate
        return sorted(by_label.values(), key=lambda item: (
            -(item.record.raw_detector_confidence or 0.0), item.label.casefold(),
        ))

    def _historical_candidates(
        self, store: V3MemoryStore, scene_id: str,
        episode_id: str, rollout_idx: int, step: int,
        instruction: str, current_place_id: str | None,
    ) -> list[_Candidate]:
        if current_place_id is None:
            return []
        nodes = {node.node_id: node for node in store.nodes_for_scene(scene_id)}
        current_node = nodes.get(current_place_id)
        if current_node is None or current_node.node_type != "place":
            return []
        by_label: dict[str, _Candidate] = {}
        for edge in store.edges_for_scene(scene_id):
            if edge.relation_type not in {"inferred_relation", "spatial_relation"}:
                continue
            if edge.source_node == current_place_id:
                landmark_id = edge.target_node
            elif edge.target_node == current_place_id:
                landmark_id = edge.source_node
            else:
                continue
            landmark = nodes.get(landmark_id)
            if landmark is None or landmark.node_type != "landmark":
                continue
            for key in set(landmark.evidence_ids) & set(edge.evidence_ids):
                record = store.get_evidence(key)
                if (record is None or record.event_type not in _DETECTION_KINDS
                        or record.observation_status != "observed"
                        or not record.provenance_validity
                        or (record.raw_detector_confidence or 0.0) < _MIN_CONFIDENCE):
                    continue
                if (record.episode_id, record.rollout_idx) == (episode_id, rollout_idx):
                    if record.step >= step:
                        continue
                elif self.config.scope != "scene_persistent":
                    continue
                label = record.semantic_label or ""
                plain = _plain_label(label)
                if not plain or not (_terms(label) & _terms(instruction)):
                    continue
                candidate = _Candidate(record, plain, "history")
                prior = by_label.get(plain.casefold())
                if prior is None or (
                    record.episode_id == episode_id,
                    record.rollout_idx == rollout_idx,
                    record.step,
                    record.raw_detector_confidence or 0.0,
                ) > (
                    prior.record.episode_id == episode_id,
                    prior.record.rollout_idx == rollout_idx,
                    prior.record.step,
                    prior.record.raw_detector_confidence or 0.0,
                ):
                    by_label[plain.casefold()] = candidate
        return sorted(by_label.values(), key=lambda item: (
            (item.record.episode_id, item.record.rollout_idx) != (episode_id, rollout_idx),
            -item.record.step, -(item.record.raw_detector_confidence or 0.0),
            item.label.casefold(),
        ))
