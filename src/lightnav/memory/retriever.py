"""Small, deterministic retrieval over one episode's explicit memory.

No embeddings or model calls are made here.  A query ranks recorded observations
using instruction words, the current place hypothesis, age and record quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lightnav.memory.schemas import MemoryContext, MemoryEvent, MemoryQuery, SceneEdge, SceneNode

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "at", "by", "for", "from", "in", "into", "is", "it",
        "of", "on", "or", "the", "then", "there", "through", "to", "until", "you",
        "your", "go", "walk", "move", "head", "continue", "take",
    }
)
_HEADER = "Episode observations (may be uncertain):\n"
_MAX_LINE_CHARS = 200
_RELATION_STATUS = {
    "possible_transition": "unverified_hypothesis",
    "visible_from": "visual_observation",
}


def _terms(value: str) -> set[str]:
    words = set()
    for match in _WORDS.finditer(value.casefold()):
        token = match.group()
        if token not in _STOP_WORDS:
            words.add(token)
            # Simple plural matching; the original token is retained too.
            if len(token) > 4 and token.endswith("s"):
                words.add(token[:-1])
    return words


def _clean(value: object) -> str:
    """Keep one line per record and leave special-token escaping to the builder."""
    return " ".join(str(value).split())


def _quality(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _relation_status(edge: SceneEdge) -> str:
    if edge.relation == "connected_to":
        return "verified_traversal" if edge.source == "verified_traversal" else "unverified_claim"
    if edge.relation == "contains":
        return "verified_containment" if edge.source == "verified_containment" else "unverified_claim"
    return _RELATION_STATUS.get(edge.relation, "reported")


@dataclass(frozen=True)
class _Candidate:
    score: float
    step: int
    key: str
    kind: str
    item: MemoryEvent | SceneNode | SceneEdge
    line: str


class MemoryRetriever:
    """Rank event and graph records and return a bounded prompt context."""

    def retrieve(
        self,
        query: MemoryQuery,
        events: list[MemoryEvent],
        nodes: list[SceneNode],
        edges: list[SceneEdge],
    ) -> MemoryContext:
        if query.top_k <= 0 or query.max_chars <= len(_HEADER):
            return MemoryContext(events=[], nodes=[], edges=[], text="")

        query_terms = _terms(query.instruction)
        node_by_id = {node.node_id: node for node in nodes}
        candidates: list[_Candidate] = []

        def score(text: str, step: int, confidence: float, importance: float, place: bool) -> float:
            matched = len(query_terms & _terms(text))
            lexical = 4.0 * matched / len(query_terms) if query_terms else 0.0
            age = max(0, query.current_step - step)
            recency = 1.0 / (1.0 + age / 12.0)
            return (
                lexical
                + (2.0 if place else 0.0)
                + recency
                + _quality(confidence)
                + 0.7 * _quality(importance)
            )

        for event in events:
            if event.step_id > query.current_step:
                continue
            description = _clean(event.description)
            place = bool(query.place_id and event.place_id == query.place_id)
            line = (
                f"[event:{_clean(event.kind)} step={event.step_id} "
                f"confidence={event.confidence:.2f} source={_clean(event.source)}] "
                f"{description}"
            )
            candidates.append(
                _Candidate(
                    score(description, event.step_id, event.confidence, event.importance, place),
                    event.step_id,
                    event.event_id,
                    "event",
                    event,
                    line,
                )
            )

        for node in nodes:
            if node.first_step > query.current_step:
                continue
            label = _clean(node.label)
            place = bool(query.place_id and node.node_id == query.place_id)
            if node.kind == "place":
                status = f"identity {_clean(node.status)}; {'visited' if node.visited else 'unvisited'}"
                label = f"{node.node_id} ({label})"
            else:
                status = _clean(node.status)
            line = (
                f"[{_clean(node.kind)} step={node.last_step} "
                f"confidence={node.confidence:.2f} source={_clean(node.source)}] "
                f"{label} ({status})"
            )
            candidates.append(
                _Candidate(
                    score(label, node.last_step, node.confidence, 0.5, place),
                    node.last_step,
                    node.node_id,
                    "node",
                    node,
                    line,
                )
            )

        for edge in edges:
            if edge.step_id > query.current_step:
                continue
            source_node = node_by_id.get(edge.source_id)
            target_node = node_by_id.get(edge.target_id)
            source_label = (
                f"{source_node.node_id} ({_clean(source_node.label)})"
                if source_node else edge.source_id
            )
            target_label = (
                f"{target_node.node_id} ({_clean(target_node.label)})"
                if target_node else edge.target_id
            )
            evidence = _clean(edge.evidence)
            status = _relation_status(edge)
            if edge.relation == "possible_transition":
                description = f"{source_label} may lead to {target_label}; evidence: {evidence}"
            elif edge.relation == "visible_from":
                description = f"{target_label} was visible from {source_label}; evidence: {evidence}"
            elif status == "unverified_claim":
                verb = "connect to" if edge.relation == "connected_to" else "contain"
                description = f"{source_label} may {verb} {target_label}; evidence: {evidence}"
            else:
                description = f"{source_label} {_clean(edge.relation)} {target_label}; evidence: {evidence}"
            place = bool(
                query.place_id and query.place_id in (edge.source_id, edge.target_id)
            )
            line = (
                f"[relation:{_clean(edge.relation)} "
                f"status={status} "
                f"step={edge.step_id} confidence={edge.confidence:.2f} "
                f"source={_clean(edge.source)}] {description}"
            )
            candidates.append(
                _Candidate(
                    score(description, edge.step_id, edge.confidence, 0.5, place),
                    edge.step_id,
                    edge.edge_id,
                    "edge",
                    edge,
                    line,
                )
            )

        candidates.sort(key=lambda item: (-item.score, -item.step, item.kind, item.key))
        selected_events: list[MemoryEvent] = []
        selected_nodes: list[SceneNode] = []
        selected_edges: list[SceneEdge] = []
        lines: list[str] = []
        used = len(_HEADER)
        for candidate in candidates:
            if len(lines) >= query.top_k:
                break
            line = candidate.line
            if len(line) > _MAX_LINE_CHARS:
                line = line[: _MAX_LINE_CHARS - 1].rstrip() + "…"
            remaining = query.max_chars - used - (1 if lines else 0)
            if remaining <= 0:
                break
            truncated_for_budget = False
            if len(line) > remaining:
                # Truncation preserves the confidence/source prefix when the
                # budget allows it; never include a partial marker alone.
                marker_end = line.find("]")
                if marker_end < 0 or marker_end + 3 > remaining:
                    break
                line = line[: remaining - 1].rstrip() + "…"
                truncated_for_budget = True
            lines.append(line)
            used += len(line) + (1 if len(lines) > 1 else 0)
            if candidate.kind == "event":
                selected_events.append(candidate.item)
            elif candidate.kind == "node":
                selected_nodes.append(candidate.item)
            else:
                selected_edges.append(candidate.item)
            if truncated_for_budget:
                break

        return MemoryContext(
            events=selected_events,
            nodes=selected_nodes,
            edges=selected_edges,
            text=_HEADER + "\n".join(lines) if lines else "",
        )
