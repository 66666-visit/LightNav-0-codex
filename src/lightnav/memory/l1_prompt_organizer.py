"""Organize L1 short-term detector memory into a compact navigation card.

L1 is the existing short-term observation memory: one YOLO11n detection record
per step in :class:`~lightnav.memory.v3_store.V3MemoryStore`. This module only
re-reads those records and the events of the current step. It never writes,
merges or rewrites memory, never runs a model, and never claims a metric
position or a traversable passage that L1 does not record.

The card deliberately does not describe the RGB image: the frozen model already
receives the video. It adds only detector-side facts the pixels do not state
(the class label, its instruction relevance, confidence, image sector and the
observation history depth).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from html import escape
from typing import Any

from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, MemoryEvent, ObservationEvidence
from lightnav.memory.v3_retrieval import _terms
from lightnav.memory.v3_schemas import EvidenceRecord, canonical_evidence_key
from lightnav.memory.v3_store import V3MemoryStore

HEADER = "[Navigation Context - L1]"
_DETECTION_KINDS = frozenset({"object_observation", "landmark"})
_MIN_CONFIDENCE = 0.5
_STATE_TEMPLATE = "Step {step}; RGB is the current view; L1 adds detector metadata only."
_NO_FACT_LINE = (
    "No instruction-related landmark in the current view and none in L1 history."
)
_UNKNOWN_LINE = (
    "L1 has no metric position or passage evidence; labels are detector classes, "
    "not identities."
)


def _plain_label(value: str) -> str:
    """Keep detector labels short and unable to introduce prompt markers."""
    words = [part for part in value.split() if part]
    cleaned = " ".join(
        "".join(char for char in word if char.isalnum() or char == "-") for word in words
    )
    return cleaned[:48].strip()


def _camera_sector(box: object, width: int) -> str | None:
    """Return the horizontal image sector of one detection box, or None."""
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        return None
    try:
        x1, _y1, x2, _y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, x2)):
        return None
    if x2 <= x1 or x1 < 0 or x2 > width:
        return None
    center = (x1 + x2) / (2 * width)
    return "left" if center < 1 / 3 else "right" if center > 2 / 3 else "center"


def _relevance(label: str, instruction: str) -> float:
    """Instruction grounding score of one detector label; 0 means irrelevant."""
    label_terms = _terms(label)
    if not label_terms:
        return 0.0
    overlap = label_terms & _terms(instruction)
    if not overlap:
        return 0.0
    score = len(overlap) / len(label_terms)
    if label.casefold() in instruction.casefold():
        score += 0.5
    return score


def _provenance_step(fact: L1Fact, evidence_id: str) -> int:
    for entry in fact.provenance:
        if entry.get("evidence_id") == evidence_id and isinstance(entry.get("step"), int):
            return int(entry["step"])
    return fact.last_step


def _provenance(record: EvidenceRecord) -> dict[str, object]:
    return {
        "scene_id": record.scene_id,
        "episode_id": record.episode_id,
        "rollout_idx": record.rollout_idx,
        "step": record.step,
        "evidence_id": record.key,
        "source": record.detector_name or "rgb_detector",
    }


@dataclass(frozen=True)
class L1Fact:
    """One deduplicated L1 observation fact with its source records."""

    label: str
    role: str  # current | history
    sector: str | None
    confidence: float
    last_step: int
    first_step: int
    sightings: int
    source: str
    evidence_ids: tuple[str, ...]
    provenance: tuple[dict[str, object], ...]
    match_score: float
    past_episode: bool = False

    def line(self) -> str:
        if self.role == "current":
            sector = f" ({self.sector})" if self.sector else ""
            sightings = f"{self.sightings} sighting" + (
                "" if self.sightings == 1 else "s"
            )
            return (
                f"Now: {self.label}{sector}, conf {self.confidence:.2f}, "
                f"{sightings} since step {self.first_step}, last step {self.last_step}."
            )
        episode_note = " in an earlier episode" if self.past_episode else ""
        sightings = f"{self.sightings} sighting" + (
            "" if self.sightings == 1 else "s"
        )
        return (
            f"Earlier: {self.label}{episode_note}, last step {self.last_step}, "
            f"{sightings}, conf {self.confidence:.2f}; "
            "not confirmed visible now."
        )

    def signature(self) -> tuple[str, str]:
        return self.role, self.label.casefold()

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label, "role": self.role, "sector": self.sector,
            "confidence": self.confidence, "last_step": self.last_step,
            "first_step": self.first_step, "sightings": self.sightings,
            "source": self.source, "evidence_ids": list(self.evidence_ids),
            "match_score": self.match_score, "past_episode": self.past_episode,
        }


@dataclass
class L1Organization:
    """The serialized card plus the measurable selection bookkeeping."""

    context: MemoryContext | None
    facts: list[L1Fact] = field(default_factory=list)
    selected_facts: list[L1Fact] = field(default_factory=list)
    current_count: int = 0
    history_count: int = 0
    duplicates_removed: int = 0
    omitted_items: int = 0
    prompt_chars: int = 0
    empty_reason: str | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "layer": "l1",
            "source": "l1_prompt_organizer" if self.context is not None else None,
            "item_count": len(self.selected_facts),
            "fact_count": len(self.facts),
            "current_count": self.current_count,
            "history_count": self.history_count,
            "duplicates_removed": self.duplicates_removed,
            "omitted_items": self.omitted_items,
            "prompt_chars": self.prompt_chars,
            "items": [fact.to_dict() for fact in self.selected_facts],
            "empty_reason": self.empty_reason,
        }


class L1PromptOrganizer:
    """Select and serialize current and historical L1 detections within one budget."""

    source = "l1_prompt_organizer"

    def __init__(self, *, max_items: int = 3) -> None:
        if type(max_items) is not int or max_items < 1:
            raise ValueError("L1 max_items must be a positive integer")
        self.max_items = max_items

    def build(
        self,
        store: V3MemoryStore,
        *,
        scene_id: str,
        episode_id: str,
        rollout_idx: int,
        step: int,
        instruction: str,
        rgb_width: int,
        events: list[MemoryEvent],
        max_chars: int,
    ) -> L1Organization:
        if store.active_episode != (scene_id, episode_id, rollout_idx):
            raise ValueError("L1 organization requires the active Scene/Episode")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if type(rgb_width) is not int or rgb_width < 1:
            raise ValueError("rgb_width must be a positive integer")
        if type(max_chars) is not int or max_chars < 1:
            raise ValueError("max_chars must be a positive integer")

        current, current_duplicates = self._current_facts(
            store, scene_id, episode_id, rollout_idx, step, instruction, rgb_width, events,
        )
        history, history_duplicates = self._history_facts(
            store, scene_id, episode_id, rollout_idx, step, instruction, current,
        )
        facts = [*current, *history]
        facts.sort(key=lambda fact: (
            0 if fact.role == "current" else 1,
            -fact.match_score,
            -fact.confidence,
            -fact.last_step,
            fact.label.casefold(),
        ))
        organization = L1Organization(
            context=None,
            facts=facts,
            current_count=len(current),
            history_count=len(history),
            duplicates_removed=current_duplicates + history_duplicates,
        )
        if not facts:
            organization.empty_reason = "no_instruction_related_l1_fact"
            # A state-only card is still useful: it states that the detector saw
            # no instruction landmark in this view without inventing one.
            self._serialize(organization, step=step, max_chars=max_chars, facts=[])
            return organization
        self._serialize(organization, step=step, max_chars=max_chars, facts=facts)
        return organization

    def _serialize(
        self, organization: L1Organization, *, step: int, max_chars: int,
        facts: list[L1Fact],
    ) -> None:
        lines = [HEADER]
        used = len(HEADER)
        items: list[LayerMemoryItem] = []
        selected: list[L1Fact] = []

        def add_line(line: str, *, fact: L1Fact | None = None) -> bool:
            nonlocal used
            safe_line = escape(line, quote=False)
            if used + 1 + len(safe_line) > max_chars:
                return False
            used += 1 + len(safe_line)
            lines.append(line)
            if fact is not None:
                selected.append(fact)
                items.append(self._item(fact, safe_line))
            return True

        state = _STATE_TEMPLATE.format(step=step)
        state_added = add_line(state)
        if not state_added:
            organization.empty_reason = "budget_too_small_for_state"
            return

        if facts:
            for fact in facts:
                if len(selected) >= self.max_items:
                    organization.omitted_items += 1
                    continue
                if not add_line(fact.line(), fact=fact):
                    organization.omitted_items += 1
            if not selected:
                add_line(_NO_FACT_LINE)
            elif not add_line(_UNKNOWN_LINE):
                pass
        else:
            add_line(_NO_FACT_LINE)

        if len(items) == 0 and facts:
            organization.empty_reason = "budget_too_small_for_fact"
            return
        organization.selected_facts = selected
        organization.prompt_chars = used
        organization.context = MemoryContext(text="\n".join(lines), layer_items=items)

    def _item(self, fact: L1Fact, safe_line: str) -> LayerMemoryItem:
        return LayerMemoryItem(
            layer="short" if fact.role == "current" else "mid",
            kind=f"l1_{fact.role}_detection",
            description=fact.line(),
            step_id=fact.last_step,
            confidence=fact.confidence,
            source=self.source,
            evidence_refs=[
                ObservationEvidence(
                    event_id=evidence_id,
                    step_id=_provenance_step(fact, evidence_id),
                    label=fact.label,
                    confidence=fact.confidence,
                    source=fact.source,
                )
                for evidence_id in fact.evidence_ids
            ],
            label=fact.label,
            metadata={
                "layer_source": "l1",
                "evidence_ids": list(fact.evidence_ids),
                "serialized_text": safe_line,
                "provenance": list(fact.provenance),
                "role": fact.role,
                "sector": fact.sector,
                "sightings": fact.sightings,
                "first_step": fact.first_step,
                "last_step": fact.last_step,
                "match_score": fact.match_score,
                "current_view_sector": fact.sector,
                "place_identity": "unverified",
                "traversability_status": "unknown",
            },
        )

    def _current_facts(
        self, store: V3MemoryStore, scene_id: str, episode_id: str, rollout_idx: int,
        step: int, instruction: str, rgb_width: int, events: list[MemoryEvent],
    ) -> tuple[list[L1Fact], int]:
        by_label: dict[str, L1Fact] = {}
        seen = 0
        for event in events:
            if (event.kind not in _DETECTION_KINDS or event.step_id != step
                    or event.episode_id != episode_id
                    or event.metadata.get("rgm_adopted", True) is False):
                continue
            label = _plain_label(event.description.removeprefix("Observed ").strip())
            if not label or not _relevance(label, instruction):
                continue
            key = canonical_evidence_key(scene_id, episode_id, rollout_idx, event.event_id)
            record = store.get_evidence(key)
            if (record is None or record.event_type not in _DETECTION_KINDS
                    or record.observation_status != "observed"
                    or not record.provenance_validity
                    or (record.raw_detector_confidence or 0.0) < _MIN_CONFIDENCE):
                continue
            seen += 1
            sector = _camera_sector(event.metadata.get("bbox_xyxy"), rgb_width)
            candidate = L1Fact(
                label=label,
                role="current",
                sector=sector,
                confidence=float(record.raw_detector_confidence or event.confidence),
                last_step=step,
                first_step=step,
                sightings=1,
                source=record.detector_name or event.source,
                evidence_ids=(record.key,),
                provenance=(_provenance(record),),
                match_score=_relevance(label, instruction),
            )
            prior = by_label.get(label.casefold())
            if prior is None:
                by_label[label.casefold()] = candidate
            elif candidate.confidence > prior.confidence:
                by_label[label.casefold()] = candidate
        return list(by_label.values()), max(0, seen - len(by_label))

    def _history_facts(
        self, store: V3MemoryStore, scene_id: str, episode_id: str, rollout_idx: int,
        step: int, instruction: str, current: list[L1Fact],
    ) -> tuple[list[L1Fact], int]:
        by_label: dict[str, list[EvidenceRecord]] = {}
        seen = 0
        for record in store.evidence_for_scene(scene_id):
            if (record.event_type not in _DETECTION_KINDS
                    or record.observation_status != "observed"
                    or not record.provenance_validity
                    or (record.raw_detector_confidence or 0.0) < _MIN_CONFIDENCE):
                continue
            same_episode = (record.episode_id, record.rollout_idx) == (episode_id, rollout_idx)
            if same_episode:
                if record.step >= step:
                    continue
            elif store.scope != "scene_persistent":
                continue
            label = _plain_label(record.semantic_label or "")
            if not label or not _relevance(label, instruction):
                continue
            seen += 1
            by_label.setdefault(label.casefold(), []).append(record)

        current_by_label = {fact.label.casefold(): fact for fact in current}
        facts: list[L1Fact] = []
        folded = 0
        for label_key, records in by_label.items():
            records.sort(key=lambda record: (
                (record.episode_id, record.rollout_idx) == (episode_id, rollout_idx),
                record.rollout_idx, record.step, record.raw_detector_confidence or 0.0,
            ), reverse=True)
            latest = records[0]
            evidence_ids = tuple(dict.fromkeys(record.key for record in records[:3]))
            provenance = tuple(_provenance(record) for record in records[:3])
            confidence = max(float(record.raw_detector_confidence or 0.0) for record in records)
            label = _plain_label(latest.semantic_label or "")
            past_episode = (latest.episode_id, latest.rollout_idx) != (episode_id, rollout_idx)
            prior = current_by_label.get(label_key)
            if prior is not None:
                # A current detection already states the visible fact. Fold the
                # historical depth into that single line instead of repeating it.
                # Step numbers restart every Episode, so "since step" only uses
                # sightings of the current Episode; prior-Episode support stays
                # visible through the observation count.
                same_episode_steps = [
                    record.step for record in records
                    if (record.episode_id, record.rollout_idx) == (episode_id, rollout_idx)
                ]
                current_by_label[label_key] = L1Fact(
                    label=prior.label, role="current", sector=prior.sector,
                    confidence=prior.confidence, last_step=prior.last_step,
                    first_step=(
                        min(prior.first_step, *same_episode_steps)
                        if same_episode_steps else prior.first_step
                    ),
                    sightings=prior.sightings + len(records),
                    source=prior.source,
                    evidence_ids=tuple(dict.fromkeys((*prior.evidence_ids, *evidence_ids))),
                    provenance=(*prior.provenance, *provenance),
                    match_score=max(
                        prior.match_score,
                        _relevance(label, instruction),
                    ),
                    past_episode=False,
                )
                folded += 1
                continue
            facts.append(L1Fact(
                label=label, role="history", sector=None, confidence=confidence,
                last_step=latest.step,
                first_step=min(record.step for record in records),
                sightings=len(records), source=latest.detector_name or "rgb_detector",
                evidence_ids=evidence_ids, provenance=provenance,
                match_score=_relevance(label, instruction), past_episode=past_episode,
            ))
        if current:
            # Replace the folded current facts with the enriched copies.
            current[:] = [current_by_label[fact.label.casefold()] for fact in current]
        serialized_history = len(facts) + folded
        return facts, max(0, seen - serialized_history)
