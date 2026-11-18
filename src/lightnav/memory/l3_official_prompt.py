"""Express the shared L3 map facts in the language of LightNav's Prompt Guide.

The official guide (``README.md``, *Prompt Guide*) ranks four ways to single out
an object: **direction + object**, a **relational anchor** (``next to`` / ``on``
/ ``behind``), **extremes** (``leftmost`` / ``rightmost``), then **ordinals**
(``first`` / ``second`` / ``third``, weakest).  It requires directions relative
to the robot, an unambiguous object phrase, one disambiguation strategy at a
time, large salient anchors, and extremes before ordinals.

That guide describes a navigation *command*; L3 is historical map memory.  This
module therefore adapts the language without pretending the map is a current
observation:

* every line keeps the robot-relative direction and straight-line range, which
  are the only navigationally usable part of a remembered object;
* the observed 3D box is read from the same ``ObjectMap`` object the candidate
  fact points at, so the shared ``L3Fact`` record does not change;
* ``next to`` / ``on`` are computed only from two observed 3D bounding boxes and
  are used only to disambiguate a label the map recorded more than once;
* ``leftmost`` / ``rightmost`` / ordinals need a stable left-to-right bearing
  order in the forward half-plane and are counted over recorded objects only;
* the header is one atomic line so budget truncation can never remove the
  "history, not current view", "straight-line, not a path" or "passage
  unverified" statements while keeping a fact line.

Candidate retrieval is not reimplemented here.  ``L3PromptOrganizer.collect``
produces the ordered fact list and its Evidence; this module only rewrites the
selected facts.  A fact whose geometry is unverified is dropped rather than
dressed up as a directional statement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from typing import Any

from lightnav.memory.l3_prompt_organizer import (
    L3Fact,
    L3Organization,
    L3PromptOrganizer,
    _plain_label,
)
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence
from lightnav.memory.spatial_graph_adapter import _relative_position
from lightnav.memory.v3_store import V3MemoryStore

EXPRESSION_VERSION = "l3_official_style_v1"
SOURCE = "l3_official_prompt"
HEADER = (
    "[Navigation Map Memory - L3; history, not current view; directions relative "
    "to you; straight-line ranges, not paths; passage unverified]"
)

#: Eight robot-relative sectors; the boundaries are geometric conventions, not
#: model-tuned values.
_DIRECTION_SECTOR_DEG = 22.5
#: Extremes/ordinals are only defined for a left-to-right order in front.
_FRONT_HALF_PLANE_DEG = 90.0
#: Adjacent bearings closer than this cannot support a stable ordering claim.
_MIN_ORDER_SEPARATION_DEG = 15.0
#: The guide's ordinal examples never go beyond "third".
_MAX_ORDINAL = 3
#: Horizontal gap between two observed boxes that still counts as "next to".
_NEXT_TO_MAX_GAP_M = 0.5
#: "next to" also needs the two observed extents to be at roughly the same
#: height.  A wall-mounted object two metres above a table is not next to it.
_NEXT_TO_MAX_VERTICAL_GAP_M = 0.75
#: Vertical tolerances for "on".  Observed surfaces are partial, but a visible
#: gap larger than 10 cm is not evidence of support, so "on" is not claimed.
_ON_MAX_ABOVE_M = 0.10
_ON_MAX_BELOW_M = 0.05
#: The target's footprint must overlap the anchor's at all.
_ON_MIN_FOOTPRINT_OVERLAP = 0.35
#: The target must lie mostly over the anchor, not hang off it.
_ON_MIN_TARGET_CONTAINMENT = 0.8
#: A "large, salient" anchor must be at least this fraction of the object's
#: observed footprint.
_ANCHOR_MIN_FOOTPRINT_RATIO = 0.75

_ORDINALS = {1: "second", 2: "third"}
_STRATEGIES = (
    "direction", "relational_anchor", "extreme", "ordinal", "direction_without_pose",
)


def direction_phrase(bearing_deg: float | None) -> str | None:
    """Return the robot-relative phrase for a left-positive bearing.

    ``_relative_position`` returns ``+`` for left, ``-`` for right and
    ``-Z`` (camera/agent forward) as zero.  The wording follows the guide's
    indoor examples (``front-left``, ``in front``, ``behind you on the
    right``).
    """
    if bearing_deg is None or not math.isfinite(bearing_deg):
        return None
    bearing = ((float(bearing_deg) + 180.0) % 360.0) - 180.0
    magnitude = abs(bearing)
    if magnitude <= _DIRECTION_SECTOR_DEG:
        return "in front of you"
    side = "left" if bearing > 0 else "right"
    if magnitude <= 3 * _DIRECTION_SECTOR_DEG:
        return f"to your front-{side}"
    if magnitude <= 5 * _DIRECTION_SECTOR_DEG:
        return f"to your {side}"
    if magnitude <= 7 * _DIRECTION_SECTOR_DEG:
        return f"behind you on the {side}"
    return "behind you"


def bbox_or_none(
    value: object,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Return a finite, ordered-corner-ready box or ``None``.

    The L3 map stores ``bbox_3d`` as two world-space corners of the observed
    point cloud.  A malformed or non-finite box must disable geometry-based
    wording instead of producing a guessed relation.
    """
    try:
        lower, upper = value
        bbox = (
            tuple(float(item) for item in lower),
            tuple(float(item) for item in upper),
        )
    except (TypeError, ValueError):
        return None
    if any(len(point) != 3 for point in bbox):
        return None
    if not all(math.isfinite(item) for point in bbox for item in point):
        return None
    return bbox


def _bbox_corners(
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lower, upper = bbox
    return (
        tuple(min(lower[index], upper[index]) for index in range(3)),
        tuple(max(lower[index], upper[index]) for index in range(3)),
    )


def _footprint_area(bbox: tuple[tuple[float, float, float], tuple[float, float, float]]) -> float:
    lower, upper = bbox
    return max(0.0, upper[0] - lower[0]) * max(0.0, upper[2] - lower[2])


def _footprint_overlap_area(
    first: tuple[tuple[float, float, float], tuple[float, float, float]],
    second: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    first_lower, first_upper = first
    second_lower, second_upper = second
    overlap_x = min(first_upper[0], second_upper[0]) - max(first_lower[0], second_lower[0])
    overlap_z = min(first_upper[2], second_upper[2]) - max(first_lower[2], second_lower[2])
    return max(0.0, overlap_x) * max(0.0, overlap_z)


def _footprint_overlap_ratio(
    first: tuple[tuple[float, float, float], tuple[float, float, float]],
    second: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    overlap = _footprint_overlap_area(first, second)
    if overlap <= 0.0:
        return 0.0
    smaller = min(_footprint_area(first), _footprint_area(second))
    if smaller <= 0.0:
        return 0.0
    return float(overlap / smaller)


def _vertical_gap(
    first: tuple[tuple[float, float, float], tuple[float, float, float]],
    second: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    first_lower, first_upper = first
    second_lower, second_upper = second
    return max(
        0.0,
        max(first_lower[1], second_lower[1]) - min(first_upper[1], second_upper[1]),
    )


def _horizontal_gap(
    first: tuple[tuple[float, float, float], tuple[float, float, float]],
    second: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> float:
    first_lower, first_upper = first
    second_lower, second_upper = second
    gap_x = max(
        0.0, second_lower[0] - first_upper[0], first_lower[0] - second_upper[0],
    )
    gap_z = max(
        0.0, second_lower[2] - first_upper[2], first_lower[2] - second_upper[2],
    )
    return math.hypot(gap_x, gap_z)


def support_relation(
    target: tuple[tuple[float, float, float], tuple[float, float, float]],
    anchor: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> str | None:
    """Return ``"on"`` / ``"next to"`` only when the observed boxes support it.

    ``on`` requires the target's observed bottom to be within 10 cm of the
    anchor's observed top (or at most 5 cm below it), the target to sit mostly
    over the anchor's footprint, and the target footprint to be no larger than
    the anchor's.  ``next to`` requires no footprint overlap, a horizontal gap
    of at most 0.5 m and a vertical gap of at most 0.75 m.  A partially
    interpenetrating, stacked or wall-mounted object returns ``None`` instead of
    being described as side-by-side.
    """
    first = _bbox_corners(target)
    second = _bbox_corners(anchor)
    overlap = _footprint_overlap_ratio(first, second)
    if overlap > 0.0:
        if overlap < _ON_MIN_FOOTPRINT_OVERLAP:
            # The boxes partially interpenetrate: neither "on" nor "next to"
            # can be asserted from this geometry.
            return None
        delta = first[0][1] - second[1][1]
        target_area = _footprint_area(first)
        containment = (
            _footprint_overlap_area(first, second) / target_area
            if target_area > 0.0 else 0.0
        )
        if (
            -_ON_MAX_BELOW_M <= delta <= _ON_MAX_ABOVE_M
            and first[1][1] > second[1][1]
            and target_area <= _footprint_area(second)
            and containment >= _ON_MIN_TARGET_CONTAINMENT
        ):
            return "on"
        return None
    if _horizontal_gap(first, second) > _NEXT_TO_MAX_GAP_M:
        return None
    if _vertical_gap(first, second) > _NEXT_TO_MAX_VERTICAL_GAP_M:
        return None
    return "next to"


@dataclass(frozen=True)
class _MapObject:
    """One observed map object that may serve as anchor or comparison member."""

    object_id: str
    label: str
    evidence_ids: tuple[str, ...]
    bbox_3d: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    observation_count: int
    footprint_m2: float
    volume_m3: float
    in_selected: bool
    bearing_deg: float | None


@dataclass
class _Expression:
    """The noun phrase and strategy chosen for one selected fact."""

    phrase: str
    strategy: str
    label_group_size: int
    anchor: _MapObject | None = None
    relation: str | None = None
    extreme: str | None = None
    ordinal: int | None = None

    def metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "expression_strategy": self.strategy,
            "label_group_size": self.label_group_size,
        }
        if self.anchor is not None:
            data["anchor_object_id"] = self.anchor.object_id
            data["anchor_label"] = self.anchor.label
            data["anchor_evidence_ids"] = list(self.anchor.evidence_ids)
            data["anchor_bbox_3d"] = (
                [list(point) for point in self.anchor.bbox_3d]
                if self.anchor.bbox_3d is not None else None
            )
            data["relation"] = self.relation
        if self.extreme is not None:
            data["extreme"] = self.extreme
            data["counting_basis"] = "left_to_right_in_front"
        if self.ordinal is not None:
            data["ordinal"] = self.ordinal
            data["counting_direction"] = "from the left in front"
        return data


class L3OfficialPromptOrganizer:
    """Rewrite the shared L3 facts as robot-relative map-memory statements."""

    source = SOURCE

    def __init__(
        self, *, max_items: int = 2, collector: L3PromptOrganizer | None = None,
    ) -> None:
        if type(max_items) is not int or max_items < 1:
            raise ValueError("L3 max_items must be a positive integer")
        self.max_items = max_items
        self.collector = collector or L3PromptOrganizer(max_items=max_items)

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
        if type(max_chars) is not int or max_chars < 1:
            raise ValueError("max_chars must be a positive integer")
        organization = self.collector.collect(
            store, object_map, scene_id=scene_id, episode_id=episode_id,
            rollout_idx=rollout_idx, step=step, instruction=instruction,
            frame=frame, scope=scope, min_confidence=min_confidence,
        )
        organization.source = self.source
        self._serialize(
            organization, store=store, object_map=object_map, step=step,
            scene_id=scene_id, episode_id=episode_id, rollout_idx=rollout_idx,
            frame=frame, scope=scope, max_chars=max_chars,
            min_confidence=min_confidence,
        )
        return organization

    # -- expression selection -------------------------------------------------

    def _map_objects(
        self,
        *,
        store: V3MemoryStore,
        object_map: Any,
        scene_id: str,
        episode_id: str,
        rollout_idx: int,
        step: int,
        scope: str,
        frame: Any | None,
        selected_ids: frozenset[str],
        min_confidence: float,
    ) -> dict[str, list[_MapObject]]:
        """Return history-verified map objects grouped by plain label.

        The same verification and history rules as ``collect`` are applied; the
        object's instruction relevance is not required because an anchor is not
        a selected fact.  A current-step-only observation never becomes an
        anchor, exactly as it never becomes a selected fact.
        """
        edges_by_object: dict[str, list[Any]] = {}
        for edge in object_map.edges:
            if edge.relation_type == "observed_from":
                edges_by_object.setdefault(edge.object_id, []).append(edge)
        grouped: dict[str, list[_MapObject]] = {}
        for obj in object_map.objects.values():
            label = _plain_label(obj.class_name)
            if not label:
                continue
            evidence = [
                object_map.evidence.get(evidence_id) for evidence_id in obj.evidence_ids
            ]
            if not evidence or not all(
                record is not None
                and record.geometry_status == "observed"
                and record.confidence >= min_confidence
                for record in evidence
            ):
                continue
            history = self.collector.valid_history(
                store, obj, edges_by_object.get(obj.object_id, []), object_map,
                scene_id, episode_id, rollout_idx, step, scope, min_confidence,
            )
            if not history:
                continue
            bbox = bbox_or_none(obj.bbox_3d)
            center = None
            if obj.center_world is not None and len(obj.center_world) == 3:
                try:
                    values = tuple(float(value) for value in obj.center_world)
                except (TypeError, ValueError):
                    values = ()
                if len(values) == 3 and all(math.isfinite(value) for value in values):
                    center = values
            bearing = None
            if frame is not None and center is not None:
                relative = _relative_position(frame.agent_pose, center)
                if relative is not None:
                    bearing = relative[1]
            footprint = _footprint_area(_bbox_corners(bbox)) if bbox is not None else 0.0
            volume = 0.0
            if bbox is not None:
                lower, upper = _bbox_corners(bbox)
                volume = max(0.0, upper[0] - lower[0]) * max(
                    0.0, upper[1] - lower[1],
                ) * max(0.0, upper[2] - lower[2])
            grouped.setdefault(label.casefold(), []).append(_MapObject(
                object_id=obj.object_id, label=label,
                evidence_ids=tuple(obj.evidence_ids), bbox_3d=bbox,
                observation_count=int(obj.observation_count),
                footprint_m2=footprint, volume_m3=volume,
                in_selected=obj.object_id in selected_ids, bearing_deg=bearing,
            ))
        return grouped

    @staticmethod
    def _relation_disambiguates(
        fact: L3Fact, group: list[_MapObject], anchor: _MapObject, relation: str,
    ) -> bool:
        """A relation is usable only if no other recorded same-label object shares it.

        If a second recorded chair is also "next to the dining table", the phrase
        does not single out either chair.  When another group member has no
        observed box the uniqueness cannot be checked, so the relation is not
        used.  The caller then falls back to a stable extreme/ordinal order or
        to direction + straight-line range.
        """
        if anchor.bbox_3d is None:
            return False
        anchor_bbox = _bbox_corners(anchor.bbox_3d)
        for member in group:
            if member.object_id == fact.object_id:
                continue
            if member.bbox_3d is None:
                return False
            if support_relation(_bbox_corners(member.bbox_3d), anchor_bbox) == relation:
                return False
        return True

    def _relational_anchor(
        self, fact: L3Fact, group: list[_MapObject], pool: list[_MapObject],
        target_box: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    ) -> tuple[_MapObject, str] | None:
        if target_box is None:
            return None
        target_bbox = _bbox_corners(target_box)
        target_footprint = _footprint_area(target_bbox)
        if target_footprint <= 0.0:
            return None
        label = fact.label.casefold()
        anchor_label_counts: dict[str, int] = {}
        for member in pool:
            key = member.label.casefold()
            anchor_label_counts[key] = anchor_label_counts.get(key, 0) + 1
        best: tuple[tuple[Any, ...], _MapObject, str] | None = None
        for candidate in pool:
            if candidate.object_id == fact.object_id:
                continue
            if candidate.label.casefold() == label or candidate.bbox_3d is None:
                continue
            if anchor_label_counts.get(candidate.label.casefold(), 0) > 1:
                # "the dining table" is ambiguous when the map recorded two.
                continue
            if candidate.footprint_m2 < target_footprint * _ANCHOR_MIN_FOOTPRINT_RATIO:
                continue
            relation = support_relation(target_bbox, _bbox_corners(candidate.bbox_3d))
            if relation is None:
                continue
            if not self._relation_disambiguates(fact, group, candidate, relation):
                continue
            key = (
                1 if relation == "on" else 0,
                candidate.footprint_m2,
                candidate.volume_m3,
                candidate.observation_count,
                1 if candidate.in_selected else 0,
                candidate.label.casefold(),
                candidate.object_id,
            )
            if best is None or key > best[0]:
                best = (key, candidate, relation)
        return (best[1], best[2]) if best is not None else None

    def _group_order(
        self, fact: L3Fact, group: list[_MapObject],
    ) -> tuple[int, int] | None:
        """Return ``(index_from_left, forward_group_size)`` or ``None``."""
        if any(member.bearing_deg is None for member in group):
            # A member without a direction cannot be placed in the left-to-right
            # order, so the count "of the N recorded in front" would be a lie.
            return None
        forward = [
            member for member in group
            if abs(float(member.bearing_deg)) <= _FRONT_HALF_PLANE_DEG
        ]
        if len(forward) < 2:
            return None
        if not any(member.object_id == fact.object_id for member in forward):
            return None
        forward.sort(key=lambda member: (-float(member.bearing_deg), member.object_id))
        for first, second in zip(forward, forward[1:]):
            if (
                float(first.bearing_deg) - float(second.bearing_deg)
                < _MIN_ORDER_SEPARATION_DEG
            ):
                return None
        index = next(
            position for position, member in enumerate(forward)
            if member.object_id == fact.object_id
        )
        return index, len(forward)

    def _expression(
        self, fact: L3Fact, grouped: dict[str, list[_MapObject]],
        object_bboxes: dict[
            str, tuple[tuple[float, float, float], tuple[float, float, float]] | None
        ],
    ) -> _Expression:
        label = fact.label.casefold()
        group = grouped.get(label, [])
        group_size = len(group)
        if group_size >= 2:
            pool = [member for members in grouped.values() for member in members]
            anchor = self._relational_anchor(
                fact, group, pool, object_bboxes.get(fact.object_id),
            )
            if anchor is not None:
                candidate, relation = anchor
                return _Expression(
                    phrase=f"{fact.label} {relation} the {candidate.label}",
                    strategy="relational_anchor", label_group_size=group_size,
                    anchor=candidate, relation=relation,
                )
            order = self._group_order(fact, group)
            if order is not None:
                index, forward_size = order
                if index == 0:
                    return _Expression(
                        phrase=f"leftmost {fact.label} of the {forward_size} recorded in front",
                        strategy="extreme", label_group_size=group_size,
                        extreme="leftmost",
                    )
                if index == forward_size - 1:
                    return _Expression(
                        phrase=f"rightmost {fact.label} of the {forward_size} recorded in front",
                        strategy="extreme", label_group_size=group_size,
                        extreme="rightmost",
                    )
                if index < _MAX_ORDINAL and forward_size >= 3:
                    ordinal = index + 1
                    return _Expression(
                        phrase=(
                            f"{_ORDINALS[index]} {fact.label} from the left among the "
                            f"{forward_size} recorded in front"
                        ),
                        strategy="ordinal", label_group_size=group_size,
                        ordinal=ordinal,
                    )
        return _Expression(
            phrase=fact.label, strategy="direction", label_group_size=group_size,
        )

    # -- serialization --------------------------------------------------------

    def _line(self, fact: L3Fact, expression: _Expression) -> str:
        record = self._record_text(fact)
        direction = direction_phrase(fact.bearing_deg)
        if direction is None or fact.distance_m is None:
            expression.strategy = "direction_without_pose"
            return (
                f"Map record {record}: the {expression.phrase} was recorded, but "
                "its direction from you is unavailable; passage unverified."
            )
        return (
            f"Map record {record}: the {expression.phrase} is {direction}, "
            f"{fact.distance_m:.1f} m straight-line."
        )

    @staticmethod
    def _record_text(fact: L3Fact) -> str:
        views = f"{fact.viewpoints} view" + ("" if fact.viewpoints == 1 else "s")
        text = f"(step {fact.latest_step}, {views}"
        if fact.past_episode:
            text += ", earlier episode"
        return text + ")"

    def _serialize(
        self,
        organization: L3Organization,
        *,
        store: V3MemoryStore,
        object_map: Any,
        step: int,
        scene_id: str,
        episode_id: str,
        rollout_idx: int,
        frame: Any | None,
        scope: str,
        max_chars: int,
        min_confidence: float,
    ) -> None:
        strategy_counts = {name: 0 for name in _STRATEGIES}
        expression: dict[str, Any] = {
            "expression_version": EXPRESSION_VERSION,
            "strategy_counts": strategy_counts,
            "unverified_omitted": sum(
                not fact.geometry_verified for fact in organization.facts
            ),
            "duplicates_removed": 0,
            "ambiguous_labels": {},
            "anchors_used": [],
            "candidate_object_ids": [
                fact.object_id for fact in organization.facts
            ],
            "selected_object_ids": [],
        }
        organization.expression = expression

        # ``collect`` sorts verified facts before unverified ones.  Taking the
        # verified prefix of ``max_items`` therefore cannot select a fact the
        # organized mode ranked below an unverified fact: it is exactly the
        # organized selection minus the (at most one) unverified line, which
        # this serializer cannot express without inventing a position.
        candidates = [fact for fact in organization.facts if fact.geometry_verified]
        label_totals: dict[str, int] = {}
        for obj in object_map.objects.values():
            label = _plain_label(obj.class_name).casefold()
            if label:
                label_totals[label] = label_totals.get(label, 0) + 1
        needs_pool = any(
            label_totals.get(fact.label.casefold(), 0) > 1 for fact in candidates
        )
        object_bboxes = {
            object_id: bbox_or_none(obj.bbox_3d)
            for object_id, obj in object_map.objects.items()
        }
        grouped: dict[str, list[_MapObject]] = {}
        if needs_pool:
            selected_ids = frozenset(fact.object_id for fact in candidates)
            grouped = self._map_objects(
                store=store, object_map=object_map, scene_id=scene_id,
                episode_id=episode_id, rollout_idx=rollout_idx, step=step,
                scope=scope, frame=frame, selected_ids=selected_ids,
                min_confidence=min_confidence,
            )
            expression["ambiguous_labels"] = {
                label: len(members) for label, members in grouped.items()
                if len(members) > 1
            }

        safe_header = escape(HEADER, quote=False)
        used = len(safe_header)
        if used > max_chars:
            organization.empty_reason = "budget_too_small_for_header"
            return

        lines = [HEADER]
        items: list[LayerMemoryItem] = []
        selected: list[L3Fact] = []
        seen: set[str] = set()
        for fact in candidates:
            if len(selected) >= self.max_items:
                organization.omitted_items += 1
                continue
            chosen = self._expression(fact, grouped, object_bboxes)
            line = self._line(fact, chosen)
            normalized = " ".join(line.casefold().split())
            if normalized in seen:
                expression["duplicates_removed"] += 1
                organization.omitted_items += 1
                continue
            safe_line = escape(line, quote=False)
            if used + 1 + len(safe_line) > max_chars:
                organization.omitted_items += 1
                continue
            used += 1 + len(safe_line)
            lines.append(line)
            seen.add(normalized)
            selected.append(fact)
            items.append(self._item(fact, chosen, line, safe_line))
            strategy_counts[chosen.strategy] = (
                strategy_counts.get(chosen.strategy, 0) + 1
            )
            if chosen.anchor is not None:
                expression["anchors_used"].append({
                    "object_id": chosen.anchor.object_id,
                    "label": chosen.anchor.label,
                    "relation": chosen.relation,
                    "evidence_ids": list(chosen.anchor.evidence_ids),
                })

        if not selected:
            organization.empty_reason = "no_expressible_l3_fact"
            return
        organization.selected_facts = selected
        organization.prompt_chars = used
        organization.expression["selected_object_ids"] = [
            fact.object_id for fact in selected
        ]
        organization.context = MemoryContext(
            text="\n".join(lines), layer_items=items,
        )

    def _item(
        self, fact: L3Fact, expression: _Expression, line: str, safe_line: str,
    ) -> LayerMemoryItem:
        metadata: dict[str, Any] = {
            "layer_source": "l3",
            "evidence_ids": list(fact.evidence_ids),
            "linked_l1_evidence_ids": list(fact.linked_l1_evidence_ids),
            "serialized_text": safe_line,
            "provenance": list(fact.provenance),
            "role": "spatial_history",
            "object_id": fact.object_id,
            "geometry_verified": True,
            "current_relative_distance_m": fact.distance_m,
            "current_relative_bearing_deg": fact.bearing_deg,
            "observation_count": fact.observation_count,
            "viewpoints": fact.viewpoints,
            "traversal_links": fact.traversal_links,
            "traversal_steps": list(fact.traversal_steps),
            "traversability_status": "unknown",
            "match_score": fact.match_score,
            "range_kind": "straight_line_not_path",
            "history_kind": "map_record_not_current_view",
            "expression_version": EXPRESSION_VERSION,
        }
        metadata.update(expression.metadata())
        return LayerMemoryItem(
            layer="long",
            kind="l3_official_map_fact",
            description=line,
            step_id=fact.latest_step,
            confidence=fact.confidence,
            source=self.source,
            evidence_refs=[
                ObservationEvidence(
                    event_id=evidence_id,
                    step_id=self._provenance_step(fact, evidence_id),
                    label=fact.label,
                    confidence=fact.confidence,
                    source=fact.source,
                )
                for evidence_id in fact.evidence_ids
            ],
            label=fact.label,
            metadata=metadata,
        )

    @staticmethod
    def _provenance_step(fact: L3Fact, evidence_id: str) -> int:
        for entry in fact.provenance:
            if entry.get("evidence_id") == evidence_id and isinstance(
                entry.get("step"), int,
            ):
                return int(entry["step"])
        return fact.last_step
