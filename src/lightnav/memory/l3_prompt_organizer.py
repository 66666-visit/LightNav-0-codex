"""Organize the L3 geometry-only semantic map into a navigation card.

L3 is the existing ConceptGraphs-Lite ``ObjectMap``: observed 3D instances,
real viewpoints, ``observed_from`` and ``traversed_to`` edges, and per-mask
Evidence. This module reads that map and the linked L1 Evidence references that
L3 already recorded at write time. It never edits the map, never invents a
Place/Edge/Evidence record, and never turns a straight-line range into a path.

Relation classes are kept explicit:

* ``observed`` -- the object's Evidence came from recorded viewpoints;
* ``spatial``  -- the current straight-line range/bearing to its stored centre;
* ``traversal``-- a real viewpoint-to-viewpoint ``traversed_to`` record exists;
* ``unknown``  -- objects passability is never proven by observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import Any

from lightnav.memory.local_graph_fusion import _plain_label
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence
from lightnav.memory.spatial_graph_adapter import _relative_position
from lightnav.memory.v3_retrieval import _terms
from lightnav.memory.v3_store import V3MemoryStore

HEADER = "[Navigation Context - L3]"
_STATE_TEMPLATE = "Step {step}; map facts are history, not current visibility."
_UNKNOWN_TEMPLATE = (
    "Unverified evidence: {count} instruction object(s); no path is proven."
)
_DETECTION_KINDS = frozenset({"object_observation", "landmark"})


def _range_text(distance_m: float, bearing_deg: float | None) -> str:
    if bearing_deg is None or abs(bearing_deg) < 0.5:
        bearing = "0deg"
    else:
        bearing = f"{bearing_deg:+.0f}deg"
    return f"{distance_m:.1f}m line {bearing} (not a path)"


def _provenance_step(fact: L3Fact, evidence_id: str) -> int:
    for entry in fact.provenance:
        if entry.get("evidence_id") == evidence_id and isinstance(entry.get("step"), int):
            return int(entry["step"])
    return fact.last_step


def _relevance(label: str, instruction: str) -> float:
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


@dataclass(frozen=True)
class L3Fact:
    """One object-map fact with the exact relation classes it supports."""

    object_id: str
    label: str
    evidence_ids: tuple[str, ...]
    linked_l1_evidence_ids: tuple[str, ...]
    provenance: tuple[dict[str, object], ...]
    last_step: int
    first_step: int
    observation_count: int
    viewpoints: int
    traversal_links: int
    confidence: float
    source: str
    distance_m: float | None
    bearing_deg: float | None
    geometry_verified: bool
    past_episode: bool
    match_score: float
    traversal_steps: tuple[int, ...] = ()

    def line(self) -> str:
        if not self.geometry_verified:
            return (
                f"{self.label}: map evidence unverified; position and passage unknown."
            )
        views = f"{self.viewpoints} view" + ("" if self.viewpoints == 1 else "s")
        parts = [f"{self.label}: last step {self.last_step}, {views}"]
        if self.distance_m is not None:
            parts.append(_range_text(self.distance_m, self.bearing_deg))
        else:
            parts.append("position unknown")
        if self.traversal_links:
            parts.append("traversal recorded between observed viewpoints")
        parts.append(f"seg conf {self.confidence:.2f}")
        parts.append("passage unknown")
        if self.past_episode:
            parts.append("earlier episode")
        return "; ".join(parts) + "."

    def signature(self) -> tuple[str, str]:
        return "l3", self.object_id

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id, "label": self.label,
            "evidence_ids": list(self.evidence_ids),
            "linked_l1_evidence_ids": list(self.linked_l1_evidence_ids),
            "last_step": self.last_step, "first_step": self.first_step,
            "observation_count": self.observation_count,
            "viewpoints": self.viewpoints, "traversal_links": self.traversal_links,
            "confidence": self.confidence, "source": self.source,
            "distance_m": self.distance_m, "bearing_deg": self.bearing_deg,
            "geometry_verified": self.geometry_verified,
            "past_episode": self.past_episode, "match_score": self.match_score,
            "traversal_steps": list(self.traversal_steps),
        }


@dataclass
class L3Organization:
    """Selected map facts plus retrieval/relation bookkeeping."""

    context: MemoryContext | None
    facts: list[L3Fact] = field(default_factory=list)
    selected_facts: list[L3Fact] = field(default_factory=list)
    candidate_count: int = 0
    verified_count: int = 0
    unverified_count: int = 0
    current_only_skipped: int = 0
    omitted_items: int = 0
    prompt_chars: int = 0
    empty_reason: str | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "layer": "l3",
            "source": "l3_prompt_organizer" if self.context is not None else None,
            "candidate_count": self.candidate_count,
            "verified_count": self.verified_count,
            "unverified_count": self.unverified_count,
            "current_only_skipped": self.current_only_skipped,
            "item_count": len(self.selected_facts),
            "omitted_items": self.omitted_items,
            "prompt_chars": self.prompt_chars,
            "items": [fact.to_dict() for fact in self.selected_facts],
            "empty_reason": self.empty_reason,
        }


class L3PromptOrganizer:
    """Select instruction-relevant map objects and serialize their relations."""

    source = "l3_prompt_organizer"

    def __init__(self, *, max_items: int = 2) -> None:
        if type(max_items) is not int or max_items < 1:
            raise ValueError("L3 max_items must be a positive integer")
        self.max_items = max_items

    def build(
        self,
        store: V3MemoryStore,
        object_map: Any,
        *,
        scene_id: str,
        episode_id: str,
        rollout_idx: int,
        step: int,
        instruction: str,
        frame: Any | None,
        scope: str,
        max_chars: int,
        min_confidence: float,
    ) -> L3Organization:
        if store.active_episode != (scene_id, episode_id, rollout_idx):
            raise ValueError("L3 organization requires the active Scene/Episode")
        if store.scope != scope:
            raise ValueError("L3 organization and evidence store scopes differ")
        if scope not in {"episode_local", "scene_persistent"}:
            raise ValueError("invalid spatial scope")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if type(max_chars) is not int or max_chars < 1:
            raise ValueError("max_chars must be a positive integer")
        if not isinstance(min_confidence, (int, float)) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must lie in [0, 1]")

        organization = L3Organization(context=None)
        observed_edges = [
            edge for edge in object_map.edges
            if edge.relation_type == "observed_from"
        ]
        traversal_edges = [
            edge for edge in object_map.edges
            if edge.relation_type == "traversed_to"
        ]
        by_object: dict[str, dict[str, Any]] = {}
        for obj in object_map.objects.values():
            label = _plain_label(obj.class_name)
            score = _relevance(label, instruction) if label else 0.0
            if not label or not score:
                continue
            organization.candidate_count += 1
            evidence = [
                object_map.evidence.get(evidence_id) for evidence_id in obj.evidence_ids
            ]
            verified_geometry = bool(evidence) and all(
                record is not None
                and record.geometry_status == "observed"
                and record.confidence >= min_confidence
                for record in evidence
            )
            history = self._valid_history(
                store, obj, observed_edges, object_map, scene_id, episode_id,
                rollout_idx, step, scope, min_confidence,
            )
            if not history:
                if any(
                    record is not None
                    and record.episode_id == episode_id
                    and record.rollout_idx == rollout_idx
                    and record.step_id >= step
                    for record in evidence
                ):
                    organization.current_only_skipped += 1
                    continue
                if verified_geometry:
                    continue
                organization.unverified_count += 1
                usable = [record for record in evidence if record is not None]
                usable.sort(key=lambda record: (
                    (record.episode_id, record.rollout_idx) == (episode_id, rollout_idx),
                    record.rollout_idx, record.step_id,
                ), reverse=True)
                newest = usable[0] if usable else None
                by_object[obj.object_id] = {
                    "obj": obj, "label": label, "score": score, "history": [],
                    "verified": False,
                    "past_episode": bool(
                        newest is not None and (newest.episode_id, newest.rollout_idx)
                        != (episode_id, rollout_idx)
                    ),
                    "linked": tuple(dict.fromkeys(
                        record.linked_l1_evidence_id for record in usable
                        if record.linked_l1_evidence_id is not None
                    )),
                    "traversal": (), "distance": None, "bearing": None,
                    "fallback_step": newest.step_id if newest is not None else 0,
                    "fallback_confidence": float(newest.confidence)
                    if newest is not None else 0.0,
                    "fallback_provenance": {
                        "scene_id": scene_id, "episode_id": (
                            newest.episode_id if newest is not None else episode_id
                        ),
                        "rollout_idx": (
                            newest.rollout_idx if newest is not None else rollout_idx
                        ),
                        "step": newest.step_id if newest is not None else step,
                        "evidence_id": (
                            newest.evidence_id if newest is not None else obj.object_id
                        ),
                        "source": "yolo11n-seg",
                        "geometry_status": (
                            newest.geometry_status if newest is not None else "unknown"
                        ),
                    },
                }
                continue
            latest = history[-1]
            past_episode = (
                latest["record"].episode_id, latest["record"].rollout_idx
            ) != (episode_id, rollout_idx)
            if not verified_geometry:
                organization.unverified_count += 1
            else:
                organization.verified_count += 1
            viewpoint_ids = {entry["viewpoint_id"] for entry in history}
            traversal = tuple(sorted({
                edge.transition_step for edge in traversal_edges
                if edge.source_viewpoint_id in viewpoint_ids
                or edge.target_viewpoint_id in viewpoint_ids
            }))
            relative = (
                _relative_position(frame.agent_pose, obj.center_world)
                if frame is not None and verified_geometry else None
            )
            by_object[obj.object_id] = {
                "obj": obj, "label": label, "score": score, "history": history,
                "verified": verified_geometry, "past_episode": past_episode,
                "linked": tuple(dict.fromkeys(
                    entry["l1_id"] for entry in history if entry["l1_id"] is not None
                )),
                "traversal": traversal,
                "distance": relative[0] if relative is not None else None,
                "bearing": relative[1] if relative is not None else None,
            }

        facts = [self._fact(payload) for payload in by_object.values()]
        facts.sort(key=lambda fact: (
            0 if fact.geometry_verified else 1,
            -fact.match_score,
            -fact.last_step,
            -fact.observation_count,
            fact.distance_m if fact.distance_m is not None else float("inf"),
            fact.label.casefold(),
            fact.object_id,
        ))
        organization.facts = facts
        self._serialize(organization, step=step, max_chars=max_chars)
        return organization

    def _valid_history(
        self, store: V3MemoryStore, obj: Any, observed_edges: list[Any], object_map: Any,
        scene_id: str, episode_id: str, rollout_idx: int, step: int, scope: str,
        min_confidence: float,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for edge in observed_edges:
            if edge.object_id != obj.object_id or edge.evidence_id not in obj.evidence_ids:
                continue
            viewpoint = object_map.viewpoints.get(edge.viewpoint_id)
            record = object_map.evidence.get(edge.evidence_id)
            if viewpoint is None or record is None:
                continue
            if (record.scene_id, record.episode_id, record.rollout_idx, record.step_id) != (
                viewpoint.scene_id, viewpoint.episode_id, viewpoint.rollout_idx,
                viewpoint.step_id,
            ):
                continue
            if record.scene_id != scene_id:
                continue
            same_episode = (record.episode_id, record.rollout_idx) == (episode_id, rollout_idx)
            if same_episode and record.step_id >= step:
                continue
            if not same_episode and scope != "scene_persistent":
                continue
            if (record.geometry_status != "observed"
                    or record.class_name.casefold() != obj.class_name.casefold()
                    or record.confidence < min_confidence):
                continue
            l1_id = record.linked_l1_evidence_id
            if l1_id is not None:
                l1 = store.get_evidence(l1_id)
                if (l1 is None or l1.scene_id != scene_id
                        or l1.episode_id != record.episode_id
                        or l1.rollout_idx != record.rollout_idx
                        or l1.step != record.step_id
                        or l1.observation_status != "observed"
                        or not l1.provenance_validity
                        or (l1.semantic_label or "").casefold() != obj.class_name.casefold()):
                    l1_id = None
            history.append({
                "record": record, "viewpoint_id": edge.viewpoint_id, "l1_id": l1_id,
            })
        history.sort(key=lambda entry: (
            (entry["record"].episode_id, entry["record"].rollout_idx)
            == (episode_id, rollout_idx),
            entry["record"].rollout_idx, entry["record"].step_id,
        ))
        return history

    def _fact(self, payload: dict[str, Any]) -> L3Fact:
        obj = payload["obj"]
        history = payload["history"]
        evidence_ids = tuple(dict.fromkeys(entry["record"].evidence_id for entry in history))
        steps = [entry["record"].step_id for entry in history]
        latest = history[-1]["record"] if history else None
        provenance = tuple({
            "scene_id": entry["record"].scene_id,
            "episode_id": entry["record"].episode_id,
            "rollout_idx": entry["record"].rollout_idx,
            "step": entry["record"].step_id,
            "evidence_id": entry["record"].evidence_id,
            "source": entry["record"].source,
            "linked_l1_evidence_id": entry["l1_id"],
            "geometry_status": entry["record"].geometry_status,
        } for entry in history[:3])
        if latest is None:
            fallback = payload["fallback_provenance"]
            evidence_ids = (fallback["evidence_id"],)
            steps = [payload["fallback_step"]]
            provenance = (fallback,)
            latest_confidence = payload["fallback_confidence"]
        else:
            latest_confidence = float(latest.confidence)
        return L3Fact(
            object_id=obj.object_id,
            label=payload["label"],
            evidence_ids=evidence_ids,
            linked_l1_evidence_ids=payload["linked"],
            provenance=provenance,
            last_step=max(steps) if steps else 0,
            first_step=min(steps) if steps else 0,
            observation_count=obj.observation_count,
            viewpoints=len({entry["viewpoint_id"] for entry in history}),
            traversal_links=len(payload["traversal"]),
            confidence=latest_confidence,
            source=latest.source if latest is not None else "yolo11n-seg",
            distance_m=payload["distance"],
            bearing_deg=payload["bearing"],
            geometry_verified=payload["verified"],
            past_episode=payload["past_episode"],
            match_score=payload["score"],
            traversal_steps=payload["traversal"],
        )

    def _serialize(self, organization: L3Organization, *, step: int, max_chars: int) -> None:
        lines = [HEADER]
        used = len(HEADER)
        items: list[LayerMemoryItem] = []
        selected: list[L3Fact] = []
        unverified_selected = 0

        def add_line(line: str, *, fact: L3Fact | None = None) -> bool:
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

        if not add_line(_STATE_TEMPLATE.format(step=step)):
            organization.empty_reason = "budget_too_small_for_state"
            return
        for fact in organization.facts:
            if len(selected) >= self.max_items:
                organization.omitted_items += 1
                continue
            if not fact.geometry_verified and unverified_selected >= 1:
                organization.omitted_items += 1
                continue
            if not add_line(fact.line(), fact=fact):
                organization.omitted_items += 1
                continue
            if not fact.geometry_verified:
                unverified_selected += 1
        if not selected:
            organization.empty_reason = "budget_too_small_for_any_l3_fact"
            return
        unverified = sum(not fact.geometry_verified for fact in selected)
        if unverified:
            add_line(_UNKNOWN_TEMPLATE.format(count=unverified))
        organization.selected_facts = selected
        organization.prompt_chars = used
        organization.context = MemoryContext(text="\n".join(lines), layer_items=items)

    def _item(self, fact: L3Fact, safe_line: str) -> LayerMemoryItem:
        return LayerMemoryItem(
            layer="long",
            kind="l3_organized_spatial_object",
            description=fact.line(),
            step_id=fact.last_step,
            confidence=fact.confidence,
            source=self.source,
            evidence_refs=[
                ObservationEvidence(
                    event_id=evidence_id,
                    step_id=_provenance_step(fact, evidence_id),
                    label=fact.label,
                    confidence=fact.confidence, source=fact.source,
                )
                for evidence_id in fact.evidence_ids
            ],
            label=fact.label,
            metadata={
                "layer_source": "l3",
                "evidence_ids": list(fact.evidence_ids),
                "linked_l1_evidence_ids": list(fact.linked_l1_evidence_ids),
                "serialized_text": safe_line,
                "provenance": list(fact.provenance),
                "role": "spatial_history",
                "object_id": fact.object_id,
                "geometry_verified": fact.geometry_verified,
                "current_relative_distance_m": fact.distance_m,
                "current_relative_bearing_deg": fact.bearing_deg,
                "observation_count": fact.observation_count,
                "viewpoints": fact.viewpoints,
                "traversal_links": fact.traversal_links,
                "traversal_steps": list(fact.traversal_steps),
                "traversability_status": "unknown",
                "match_score": fact.match_score,
            },
        )
