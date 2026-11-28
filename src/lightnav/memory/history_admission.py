"""Episode-local one-shot admission for the final L3 historical memory.

The C arm is H1 plus exactly one switch.  It does not score, rank, threshold,
decay or re-order anything: it only remembers which stable ``ObjectNode``
``object_id`` values have already entered the Prompt during the current
Episode.  A later Step that re-selects the same object is refused, even when
the selected Evidence ID changed, because the repeated object is the only
question this ablation asks.

The gate is deliberately small and fail-open:

* a candidate with a reliable, nonempty ``object_id`` is admitted once per
  Episode and refused on every later appearance (``once_per_object``);
* candidates without a reliable ``object_id`` are always admitted and counted,
  so missing metadata can never silently drop memory that the original L3 path
  would have injected;
* different ``object_id`` values are independent, including two instances of
  the same class;
* admission is only *pending* until the caller confirms that the final Prompt
  audit actually exposed the candidate's Evidence.  The adopted set is updated
  only by :meth:`HistoryAdmissionGate.confirm`;
* :meth:`HistoryAdmissionGate.reset_episode` clears every piece of state, so no
  object can leak across Episodes or rollouts.

``off`` is the default and makes :meth:`HistoryAdmissionGate.admit` a pure
pass-through: no pending entry is created and no candidate is ever refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

HISTORY_ADMISSION_MODES = ("off", "once_per_object")


@dataclass(frozen=True)
class AdmissionCandidate:
    """One historical object that is about to enter the Prompt."""

    object_id: str | None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissionDecision:
    """The gate's decision for one Prompt build, in candidate order."""

    mode: str
    candidates: tuple[AdmissionCandidate, ...]
    accepted_indices: tuple[int, ...]
    rejected_seen_object_ids: tuple[str, ...]
    missing_object_id_count: int
    step: int | None = None

    @property
    def accepted_candidates(self) -> tuple[AdmissionCandidate, ...]:
        return tuple(self.candidates[index] for index in self.accepted_indices)

    @property
    def candidate_object_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            candidate.object_id
            for candidate in self.candidates
            if candidate.object_id
        ))

    @property
    def accepted_object_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            candidate.object_id
            for candidate in self.accepted_candidates
            if candidate.object_id
        ))

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_seen_object_ids)


def _normalize_evidence_ids(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    return tuple(
        dict.fromkeys(
            value for value in values if isinstance(value, str) and value
        )
    )


def _normalize_candidate(
    candidate: AdmissionCandidate | tuple[Any, Iterable[str] | None] | Any,
) -> AdmissionCandidate:
    if isinstance(candidate, AdmissionCandidate):
        object_id = candidate.object_id
        evidence_ids = candidate.evidence_ids
    elif isinstance(candidate, tuple) and len(candidate) == 2:
        object_id, evidence_ids = candidate
    else:
        object_id, evidence_ids = candidate, ()
    if not isinstance(object_id, str) or not object_id.strip():
        object_id = None
    return AdmissionCandidate(
        object_id=object_id,
        evidence_ids=_normalize_evidence_ids(evidence_ids),
    )


class HistoryAdmissionGate:
    """Episode-local ``once_per_object`` admission policy for L3 history."""

    def __init__(self, mode: str = "off") -> None:
        if mode not in HISTORY_ADMISSION_MODES:
            raise ValueError(
                "history admission mode must be off or once_per_object"
            )
        self.mode = mode
        self._adopted: set[str] = set()
        self._pending: list[AdmissionCandidate] = []
        self._pending_object_ids: set[str] = set()
        self._pending_step: int | None = None
        self._last_decision: AdmissionDecision | None = None
        self._last_confirmed: tuple[str, ...] = ()
        self.scene_id: str | None = None
        self.episode_id: str | None = None
        self.rollout_idx = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def adopted_object_ids(self) -> frozenset[str]:
        """Objects confirmed as actually injected in this Episode."""
        return frozenset(self._adopted)

    @property
    def pending_object_ids(self) -> frozenset[str]:
        return frozenset(self._pending_object_ids)

    def reset_episode(
        self,
        scene_id: str | None = None,
        episode_id: str | None = None,
        rollout_idx: int = 0,
    ) -> None:
        """Clear all Episode-local state; never carry an object across Episodes."""
        self._adopted.clear()
        self._pending.clear()
        self._pending_object_ids.clear()
        self._pending_step = None
        self._last_decision = None
        self._last_confirmed = ()
        self.scene_id = scene_id
        self.episode_id = episode_id
        self.rollout_idx = rollout_idx

    def begin_step(self, step: int) -> None:
        """Drop a previous Step's unconfirmed proposal before a new Prompt build."""
        self._pending.clear()
        self._pending_object_ids.clear()
        self._pending_step = step
        self._last_decision = None
        self._last_confirmed = ()

    def admit(
        self,
        candidates: Sequence[
            AdmissionCandidate | tuple[Any, Iterable[str] | None]
        ],
        *,
        step: int | None = None,
    ) -> AdmissionDecision:
        """Decide which candidates may enter this Prompt build.

        ``off`` admits every candidate and tracks nothing.  With
        ``once_per_object`` the first candidate for a stable object is admitted
        (pending audit), a repeated object is refused, and a candidate without
        a stable object ID is admitted fail-open and counted.
        """
        normalized = tuple(_normalize_candidate(value) for value in candidates)
        missing_count = sum(
            1 for candidate in normalized if candidate.object_id is None
        )
        if self.mode == "off":
            decision = AdmissionDecision(
                mode=self.mode,
                candidates=normalized,
                accepted_indices=tuple(range(len(normalized))),
                rejected_seen_object_ids=(),
                missing_object_id_count=missing_count,
                step=step,
            )
            self._last_decision = decision
            self._last_confirmed = ()
            return decision

        accepted: list[int] = []
        rejected: list[str] = []
        for index, candidate in enumerate(normalized):
            object_id = candidate.object_id
            if object_id is None:
                # Fail-open: missing metadata must never suppress the original
                # L3 injection.  It cannot be adopted or deduplicated.
                accepted.append(index)
                continue
            if object_id in self._adopted or object_id in self._pending_object_ids:
                if object_id not in rejected:
                    rejected.append(object_id)
                continue
            accepted.append(index)
            self._pending.append(candidate)
            self._pending_object_ids.add(object_id)
        if step is not None:
            self._pending_step = step
        decision = AdmissionDecision(
            mode=self.mode,
            candidates=normalized,
            accepted_indices=tuple(accepted),
            rejected_seen_object_ids=tuple(rejected),
            missing_object_id_count=missing_count,
            step=step,
        )
        self._last_decision = decision
        self._last_confirmed = ()
        return decision

    def confirm(
        self, visible_evidence_ids: Iterable[str], *, step: int | None = None,
    ) -> tuple[str, ...]:
        """Commit pending objects only after the final Prompt audit succeeded.

        ``visible_evidence_ids`` is the exact Evidence list the model input
        checker recovered from the final Prompt tokens.  A pending object is
        adopted only when at least one of its own Evidence IDs is visible.  A
        stale confirmation from another Step is ignored.
        """
        if self.mode == "off":
            return ()
        if step is not None and self._pending_step is not None and (
            self._pending_step != step
        ):
            return ()
        visible = {
            value for value in visible_evidence_ids
            if isinstance(value, str) and value
        }
        confirmed: list[str] = []
        for candidate in self._pending:
            object_id = candidate.object_id
            if object_id is None or not candidate.evidence_ids:
                continue
            if visible.intersection(candidate.evidence_ids):
                self._adopted.add(object_id)
                if object_id not in confirmed:
                    confirmed.append(object_id)
        self._pending.clear()
        self._pending_object_ids.clear()
        self._pending_step = None
        self._last_confirmed = tuple(confirmed)
        return self._last_confirmed

    def discard_pending(self) -> None:
        """Forget the unconfirmed proposal of a Prompt that was not injected."""
        self._pending.clear()
        self._pending_object_ids.clear()
        self._pending_step = None
        self._last_confirmed = ()

    def diagnostics(self) -> dict[str, Any]:
        """Stable fields shared by memory_diagnostics and prompt_diagnostics."""
        decision = self._last_decision
        return {
            "history_admission_mode": self.mode,
            "history_admission_step": (
                decision.step if decision is not None else None
            ),
            "history_admission_candidate_object_ids": list(
                decision.candidate_object_ids if decision is not None else ()
            ),
            "history_admission_accepted_object_ids": list(
                decision.accepted_object_ids if decision is not None else ()
            ),
            "history_admission_rejected_seen_object_ids": list(
                decision.rejected_seen_object_ids
                if decision is not None else ()
            ),
            "history_admission_missing_object_id_count": (
                decision.missing_object_id_count
                if decision is not None else 0
            ),
            "history_admission_confirmed_object_ids": list(
                self._last_confirmed
            ),
            "history_admission_adopted_object_ids": sorted(self._adopted),
            "history_admission_pending_object_ids": sorted(
                self._pending_object_ids
            ),
        }


__all__ = [
    "AdmissionCandidate",
    "AdmissionDecision",
    "HISTORY_ADMISSION_MODES",
    "HistoryAdmissionGate",
]
