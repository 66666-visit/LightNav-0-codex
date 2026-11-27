"""Ordered instruction-landmark Navigation Memory.

The previous instance-grounding path rendered only the landmarks for which a
formatable geometry fact existed.  That silently removed the final landmark
from the Prompt whenever its detector class was absent (for example
``doorway``/``stairs`` under a COCO detector), and the character budget could
also spend the whole allowance on the first fact (the bed) before the final
landmark was reached.

This module keeps the *ordered instruction plan* in the Prompt and adds real
spatial facts only where an existing ``SpatialGraphAdapter`` item was selected.
It never invents a coordinate or a relation:

* every landmark line uses a verbatim instruction phrase and its parsed role /
  order;
* a line gets ``world (...)`` / range / bearing only from that selected item's
  recorded ``LayerMemoryItem`` metadata;
* a landmark without a selected item keeps only its instruction semantics and
  explicitly says that no 3D evidence exists.

Rendering is whole-line and character-bounded, like the existing
``render_context``.  The compact semantic skeleton is reserved first, so an
absent structural landmark is still represented even when its evidence fact is
unavailable.  Remaining characters enrich landmarks with real facts in
``final > transition > intermediate > source > reference`` order; the output
line order itself always follows the original instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from html import escape
from typing import Any, Iterable, Sequence

from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, ObservationEvidence

HEADER = "[Navigation Memory]"

_ROLE_PRIORITY = {
    "reference": 0,
    "source": 1,
    "intermediate": 2,
    "transition": 3,
    "final": 4,
}


@dataclass(frozen=True)
class LandmarkMemoryRow:
    """One ordered instruction landmark and its *verified* fact, if any."""

    landmark_order: int
    landmark_role: str
    landmark_phrase: str
    target_object: str
    action_phrase: str | None
    grounding_query: str
    query_executed: bool = True
    query_skip_reason: str | None = None
    proposal_count: int = 0
    eligible_candidate_count: int = 0
    selected_object_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    selection_source: str | None = None
    #: Exact ``LayerMemoryItem.description`` from SpatialGraphAdapter.  It is
    #: copied verbatim when the compact metadata form is unavailable.
    spatial_fact: str | None = None
    #: The exact sentence emitted by F's original ``render_context`` formatter
    #: for this item.  It is populated by the manager so the H2/H3 rendering
    #: arms can reuse F's string without copying or re-implementing its
    #: ``Spatial history: ...`` template.
    original_spatial_fact: str | None = None
    spatial_label: str | None = None
    center_world: tuple[float, float, float] | None = None
    distance_m: float | None = None
    bearing_deg: float | None = None
    observation_count: int | None = None
    evidence_step: int | None = None
    evidence_confidence: float = 0.0
    evidence_source: str | None = None
    geometry_fallback_attempted: bool = False
    geometry_fallback_reason: str | None = None
    #: Filled only by :func:`render_ordered_landmark_memory`.
    included_in_prompt: bool = False
    spatial_fact_included: bool = False
    actual_injection: bool = False
    actual_injected_evidence_ids: tuple[str, ...] = ()
    rendered_text: str | None = None

    @property
    def has_verified_fact(self) -> bool:
        """True only when the row has both a selected item and its Evidence."""
        return bool(
            self.selected_object_id
            and self.evidence_ids
            and self.spatial_fact
        )

    def _prefix(self) -> str:
        # Quotes are presentation only; the exact phrase is kept in diagnostics.
        phrase = self.landmark_phrase.replace('"', "'")
        return f'{self.landmark_order} {self.landmark_role} "{phrase}"'

    def skeleton_line(self) -> str:
        """Instruction-only line; contains no coordinate and no range."""
        return f"{self._prefix()}: no 3D evidence."

    def fact_line(
        self, *, preserve_original_spatial_fact: bool = False,
    ) -> str:
        """Line with a real recorded fact, or the skeleton when none exists.

        ``preserve_original_spatial_fact`` swaps only the fact sentence: the
        exact string produced by F's original formatter is appended after the
        same role/phrase prefix.  Coordinates, bearings, observed counts,
        source and ``passage unknown`` are never re-derived here.
        """
        if not self.has_verified_fact:
            return self.skeleton_line()
        if preserve_original_spatial_fact:
            fact = self.original_spatial_fact or self.spatial_fact
            if not fact:
                return self.skeleton_line()
            return f"{self._prefix()}: {fact}"
        label = self.spatial_label or self.target_object
        parts: list[str] = []
        if self.center_world is not None:
            center = ",".join(f"{value:.1f}" for value in self.center_world)
            parts.append(f"{label} at world ({center})m")
        elif self.spatial_fact:
            # Exact adapter sentence; it contains no invented numbers.
            parts.append(self.spatial_fact.rstrip("."))
        else:
            return self.skeleton_line()
        if self.distance_m is not None and self.bearing_deg is not None:
            parts.append(
                f"now {self.distance_m:.1f}m line,{self.bearing_deg:+.0f}deg"
            )
        if self.observation_count is not None:
            parts.append(f"{self.observation_count} obs")
        if self.selection_source:
            parts.append(self.selection_source)
        return f"{self._prefix()}: " + "; ".join(parts) + "."

    def to_dict(self) -> dict[str, Any]:
        return {
            "landmark_order": self.landmark_order,
            "landmark_role": self.landmark_role,
            "landmark_phrase": self.landmark_phrase,
            "target_object": self.target_object,
            "action_phrase": self.action_phrase,
            "grounding_query": self.grounding_query,
            "query_executed": self.query_executed,
            "query_skip_reason": self.query_skip_reason,
            "grounder_proposal_count": self.proposal_count,
            "eligible_candidate_count": self.eligible_candidate_count,
            "selected_object_id": self.selected_object_id,
            "evidence_ids": list(self.evidence_ids),
            "selection_source": self.selection_source,
            "spatial_fact": self.spatial_fact,
            "original_spatial_fact": self.original_spatial_fact,
            "spatial_label": self.spatial_label,
            "center_world": (
                list(self.center_world)
                if self.center_world is not None else None
            ),
            "distance_m": self.distance_m,
            "bearing_deg": self.bearing_deg,
            "observation_count": self.observation_count,
            "evidence_step": self.evidence_step,
            "geometry_fallback_attempted": self.geometry_fallback_attempted,
            "geometry_fallback_reason": self.geometry_fallback_reason,
            "included_in_prompt": self.included_in_prompt,
            "spatial_fact_included": self.spatial_fact_included,
            "actual_injection": self.actual_injection,
            "actual_injected_evidence_ids": list(
                self.actual_injected_evidence_ids
            ),
        }


def _line_length(lines: Sequence[str], *, header: str) -> int:
    return len(header) + sum(len(line) + 1 for line in lines)


def _retain_rows(
    rows: Sequence[LandmarkMemoryRow], *, max_chars: int, header: str,
) -> list[LandmarkMemoryRow]:
    """Drop the least behavior-critical *semantic* lines only if forced.

    The final landmark is the one that controls STOP, so it is retained last;
    locative references are dropped first.  At the normal 320-character budget
    all four 1012 lines fit, so this path is a safety net for tiny budgets.
    """
    retained = list(rows)
    while retained:
        lines = [row.skeleton_line() for row in retained]
        if _line_length(lines, header=header) <= max_chars:
            return retained
        victim = min(
            retained,
            key=lambda row: (
                _ROLE_PRIORITY.get(row.landmark_role, 0),
                -row.landmark_order,
            ),
        )
        retained.remove(victim)
    return retained


def _enrichment_order(rows: Iterable[LandmarkMemoryRow]) -> list[LandmarkMemoryRow]:
    return sorted(
        (row for row in rows if row.has_verified_fact),
        key=lambda row: (
            -_ROLE_PRIORITY.get(row.landmark_role, 0),
            row.landmark_order,
        ),
    )


def _plan_line(rows: Sequence[LandmarkMemoryRow]) -> str:
    parts = [
        f'{row.landmark_order} {row.landmark_role} '
        f'"{row.landmark_phrase.replace(chr(34), chr(39))}"'
        for row in rows
    ]
    return "Plan: " + "; ".join(parts) + "."


def _render_compact_plan(
    rows: Sequence[LandmarkMemoryRow], *, max_chars: int, header: str,
    preserve_original_spatial_fact: bool,
) -> tuple[str | None, tuple[LandmarkMemoryRow, ...]]:
    """Fallback for many landmarks: one ordered plan line plus fact lines.

    Six or more per-landmark skeleton lines can consume the whole 320-character
    budget before any real fact fits.  This form keeps every landmark's role,
    phrase and order in one whole line, then spends the rest on verified facts
    (final first).  It still contains no coordinate for a landmark without a
    selected item.
    """
    plan = _plan_line(rows)
    used = len(header) + 1 + len(plan)
    if used > max_chars:
        return None, ()
    remaining = max_chars - used
    fact_rows: list[LandmarkMemoryRow] = []
    for row in _enrichment_order(rows):
        fact = row.fact_line(
            preserve_original_spatial_fact=preserve_original_spatial_fact,
        )
        if len(fact) + 1 <= remaining:
            fact_rows.append(row)
            remaining -= len(fact) + 1
    if not fact_rows:
        return None, ()
    fact_by_order = {row.landmark_order: row for row in fact_rows}
    included: list[LandmarkMemoryRow] = []
    for row in rows:
        if row.landmark_order in fact_by_order:
            line = row.fact_line(
                preserve_original_spatial_fact=preserve_original_spatial_fact,
            )
            included.append(replace(
                row,
                included_in_prompt=True,
                spatial_fact_included=True,
                actual_injection=True,
                actual_injected_evidence_ids=row.evidence_ids,
                rendered_text=line,
            ))
        else:
            included.append(replace(
                row,
                included_in_prompt=True,
                spatial_fact_included=False,
                actual_injection=False,
                actual_injected_evidence_ids=(),
                rendered_text=plan,
            ))
    fact_lines = [
        fact_by_order[row.landmark_order].fact_line(
            preserve_original_spatial_fact=preserve_original_spatial_fact,
        )
        for row in rows if row.landmark_order in fact_by_order
    ]
    text = header + "\n" + "\n".join([plan, *fact_lines])
    if len(text) > max_chars:
        return None, ()
    return text, tuple(included)


def _render_fact_lines_only(
    rows: Sequence[LandmarkMemoryRow], *, max_chars: int, header: str,
    preserve_original_spatial_fact: bool,
) -> tuple[str | None, tuple[LandmarkMemoryRow, ...]]:
    """Render verified fact lines only, without any no-Evidence placeholder.

    This is the H1/H3 branch.  The same enrichment priority as G is used to
    decide which whole fact lines fit, but no character budget is reserved for
    a skeleton or a compact ``Plan:`` line.  The emitted line order is still the
    original instruction order.  A row without a verified fact contributes
    nothing, so no coordinate-free placeholder is injected.
    """
    if not any(row.has_verified_fact for row in rows):
        return None, ()
    remaining = max_chars - len(header)
    selected_orders: set[int] = set()
    for row in _enrichment_order(rows):
        line = row.fact_line(
            preserve_original_spatial_fact=preserve_original_spatial_fact,
        )
        if len(line) + 1 <= remaining:
            selected_orders.add(row.landmark_order)
            remaining -= len(line) + 1
    if not selected_orders:
        return None, ()
    lines: list[str] = []
    included: list[LandmarkMemoryRow] = []
    for row in rows:
        if row.landmark_order not in selected_orders:
            continue
        line = row.fact_line(
            preserve_original_spatial_fact=preserve_original_spatial_fact,
        )
        lines.append(line)
        included.append(replace(
            row,
            included_in_prompt=True,
            spatial_fact_included=True,
            actual_injection=True,
            actual_injected_evidence_ids=row.evidence_ids,
            rendered_text=line,
        ))
    text = header + "\n" + "\n".join(lines)
    if len(text) > max_chars:
        return None, ()
    return text, tuple(included)


def render_ordered_landmark_memory(
    rows: Sequence[LandmarkMemoryRow],
    *,
    max_chars: int,
    header: str = HEADER,
    include_missing_landmark_placeholders: bool = True,
    preserve_original_spatial_fact: bool = False,
) -> tuple[str | None, tuple[LandmarkMemoryRow, ...]]:
    """Render the ordered plan and as many real facts as the budget allows.

    The two keyword switches are the H1/H2/H3 rendering ablation:

    * ``include_missing_landmark_placeholders=True`` (default) keeps G's
      skeleton/``Plan:`` lines for landmarks without a verified fact.  When it
      is false, only verified fact lines are rendered and the freed budget is
      available to real facts.
    * ``preserve_original_spatial_fact=True`` keeps G's role/phrase prefix but
      reuses F's exact formatter sentence for the fact body.  It does not
      re-template coordinates, bearing, observed counts or source.

    With both switches at their defaults this function is byte-identical to the
    original G renderer.  Returns ``(text, included_rows)``.  ``included_rows``
    carry the exact rendered line and whether their real fact entered the text.
    No line ever exceeds ``max_chars``.
    """
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("ordered landmark budget must be a positive integer")
    ordered = sorted(rows, key=lambda row: row.landmark_order)
    if not ordered or max_chars <= len(header) + 1:
        return None, ()
    if not include_missing_landmark_placeholders:
        return _render_fact_lines_only(
            ordered, max_chars=max_chars, header=header,
            preserve_original_spatial_fact=preserve_original_spatial_fact,
        )
    retained = _retain_rows(ordered, max_chars=max_chars, header=header)
    if not retained:
        return None, ()

    skeleton_lines = {row.landmark_order: row.skeleton_line() for row in retained}
    used = _line_length(
        [skeleton_lines[row.landmark_order] for row in retained],
        header=header,
    )
    remaining = max_chars - used
    rendered = dict(skeleton_lines)
    fact_included: set[int] = set()
    for row in _enrichment_order(retained):
        fact = row.fact_line(
            preserve_original_spatial_fact=preserve_original_spatial_fact,
        )
        skeleton = skeleton_lines[row.landmark_order]
        delta = len(fact) - len(skeleton)
        if delta <= remaining:
            rendered[row.landmark_order] = fact
            fact_included.add(row.landmark_order)
            remaining -= delta

    if fact_included:
        lines: list[str] = []
        included: list[LandmarkMemoryRow] = []
        for row in retained:
            line = rendered[row.landmark_order]
            lines.append(line)
            has_fact = row.landmark_order in fact_included
            included.append(replace(
                row,
                included_in_prompt=True,
                spatial_fact_included=has_fact,
                actual_injection=has_fact,
                actual_injected_evidence_ids=(
                    row.evidence_ids if has_fact else ()
                ),
                rendered_text=line,
            ))
        text = header + "\n" + "\n".join(lines)
        if len(text) > max_chars:
            return None, ()
        return text, tuple(included)

    # Many landmarks: per-landmark skeletons leave no room for a fact.  Use one
    # ordered plan line and then the verified fact lines.
    return _render_compact_plan(
        ordered, max_chars=max_chars, header=header,
        preserve_original_spatial_fact=preserve_original_spatial_fact,
    )


def build_ordered_memory_context(
    included_rows: Sequence[LandmarkMemoryRow],
    *,
    text: str,
) -> MemoryContext:
    """Convert rendered rows into the existing Prompt item/context format.

    One item is created per unique rendered line.  In per-landmark mode that is
    one item per landmark; in compact-plan mode the single plan line carries
    all no-fact landmarks, and each verified fact gets its own item.  This is
    required by the policy's whole-line Prompt contract.
    """
    rows_by_line: dict[str, list[LandmarkMemoryRow]] = {}
    for row in included_rows:
        line = row.rendered_text or row.skeleton_line()
        rows_by_line.setdefault(line, []).append(row)
    lines = text.splitlines()[1:] if text.startswith(HEADER) else text.splitlines()
    items: list[LayerMemoryItem] = []
    for line in lines:
        group = rows_by_line.get(line)
        if not group:
            continue
        evidence_ids = list(dict.fromkeys(
            evidence_id for row in group
            for evidence_id in (
                row.evidence_ids if row.spatial_fact_included else ()
            )
        ))
        first = group[0]
        refs = [
            ObservationEvidence(
                event_id=evidence_id,
                step_id=max(
                    (row.evidence_step or 0 for row in group
                     if row.spatial_fact_included),
                    default=0,
                ),
                label=first.spatial_label or first.target_object,
                confidence=max(
                    (float(row.evidence_confidence) for row in group),
                    default=0.0,
                ),
                source=(
                    first.evidence_source or first.selection_source
                    or "instruction"
                ),
            )
            for evidence_id in evidence_ids
        ]
        group_priority = max(
            (_ROLE_PRIORITY.get(row.landmark_role, 0) for row in group),
            default=0,
        )
        if not evidence_ids and line.startswith("Plan:"):
            # The combined semantic plan line must survive token trimming;
            # individual no-fact lines keep their own role priority.
            group_priority = max(group_priority, 5)
        items.append(LayerMemoryItem(
            layer="long",
            kind="ordered_instruction_landmark",
            description=line,
            step_id=max((row.evidence_step or 0 for row in group), default=0),
            confidence=max(
                (float(row.evidence_confidence) for row in group), default=0.0,
            ),
            source=first.selection_source or "instruction_landmark",
            evidence_refs=refs,
            label=first.spatial_label or first.target_object,
            metadata={
                "evidence_ids": evidence_ids,
                "serialized_text": escape(line, quote=False),
                "role": "ordered_landmark",
                "object_id": first.selected_object_id,
                "landmark_order": first.landmark_order,
                "landmark_role": first.landmark_role,
                "landmark_phrase": first.landmark_phrase,
                "grounding_query": first.grounding_query,
                "selection_source": first.selection_source,
                "spatial_fact_included": any(
                    row.spatial_fact_included for row in group
                ),
                "actual_injection": any(
                    row.spatial_fact_included for row in group
                ),
                "landmark_priority": group_priority,
                "landmarks": [
                    {
                        "landmark_order": row.landmark_order,
                        "landmark_role": row.landmark_role,
                        "landmark_phrase": row.landmark_phrase,
                        "selected_object_id": row.selected_object_id,
                        "evidence_ids": list(row.evidence_ids),
                        "spatial_fact_included": row.spatial_fact_included,
                    }
                    for row in group
                ],
                "provenance": [
                    {
                        "object_id": row.selected_object_id,
                        "evidence_ids": (
                            list(row.evidence_ids)
                            if row.spatial_fact_included else []
                        ),
                        "source": (
                            row.selection_source or "instruction_landmark"
                        ),
                        "landmark_phrase": row.landmark_phrase,
                        "landmark_role": row.landmark_role,
                        "landmark_order": row.landmark_order,
                        "grounding_query": row.grounding_query,
                    }
                    for row in group
                ],
            },
        ))
    # The Prompt text keeps instruction order.  Item order is a separate
    # overflow-priority channel: the existing trim helper drops from the end,
    # so final/transition facts and the semantic plan line are kept longest.
    items.sort(
        key=lambda item: (
            -int(item.metadata.get("landmark_priority", 0)),
            item.description,
        )
    )
    return MemoryContext(text=text, layer_items=items)


def injected_evidence_ids(
    included_rows: Sequence[LandmarkMemoryRow],
) -> list[str]:
    """Evidence IDs whose real fact line entered the Prompt, in row order."""
    return list(dict.fromkeys(
        evidence_id
        for row in included_rows if row.spatial_fact_included
        for evidence_id in row.evidence_ids
    ))


__all__ = [
    "HEADER",
    "LandmarkMemoryRow",
    "build_ordered_memory_context",
    "injected_evidence_ids",
    "render_ordered_landmark_memory",
]
