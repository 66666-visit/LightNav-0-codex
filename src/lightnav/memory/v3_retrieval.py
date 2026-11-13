"""Scene-scoped, evidence-grounded retrieval for the V3 navigation memory.

Each channel ranks independently. Reciprocal rank fusion combines ranks rather
than adding DINO cosine values to lexical or graph scores with unlike scales.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Iterable

from lightnav.memory.v3_schemas import (
    EpisodeMemory, EvidenceRecord, RetrievedMemory, SpatialEdgeV3, SpatialNodeV3,
    parse_evidence_key,
)

if TYPE_CHECKING:
    from lightnav.memory.v3_store import V3MemoryStore


_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_SYNONYMS = {
    "sofa": "couch", "sofas": "couch", "couches": "couch",
    "tv": "television", "tvs": "television", "televisions": "television",
}
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "at", "be", "by", "for", "in", "is", "of",
    "on", "or", "the", "this", "to", "was", "were", "with",
    "find", "go", "head", "look", "move", "navigate", "proceed", "search",
    "travel", "toward", "towards", "walk",
})
_CHANNELS = ("visual", "semantic", "spatial", "episodic")
_HISTORY_COUNT_KEYS = (
    "historical_same_episode_prior_step", "historical_cross_episode",
    "historical_final_top_k", "historical_merged_candidates",
    "source_step_lt_current_step", "current_step_only_final_top_k",
    *(f"{channel}_historical_candidates" for channel in _CHANNELS),
)
_VISUAL_EVENTS = frozenset({
    "object_observation", "landmark", "place_observation", "place_visit",
})
_BEHAVIOR_EVENTS = frozenset({"action", "turn", "issued_command", "prediction"})


def _terms(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (tuple, list)):
        return set().union(*(_terms(part) for part in value)) if value else set()
    return {
        _SYNONYMS.get(word, word)
        for word in _WORDS.findall(str(value).casefold())
        if word not in _STOPWORDS
    }


def _cosine(a: Iterable[float] | None, b: Iterable[float] | None) -> float | None:
    if a is None or b is None:
        return None
    left, right = tuple(a), tuple(b)
    if not left or len(left) != len(right):
        return None
    if any(not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (*left, *right)):
        return None
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left == 0 or norm_right == 0:
        return None
    return sum(x * y for x, y in zip(left, right)) / (norm_left * norm_right)


def _short(value: str, limit: int = 90) -> str:
    clean = " ".join(value.split())
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"


def _record_description(record: EvidenceRecord) -> str:
    label = record.semantic_label or record.event_type
    detail = record.payload.get("description") if isinstance(record.payload, dict) else None
    words = f"{label}; {_short(detail, 50)}" if isinstance(detail, str) and detail else label
    confidence = (
        f"; detector confidence={record.raw_detector_confidence:.2f}"
        if record.raw_detector_confidence is not None else ""
    )
    return _short(f"{words}; observed step={record.step}; support={record.support_count}{confidence}", 150)


def _node_description(node: SpatialNodeV3, similarity: float | None = None) -> str:
    label = f"; label={node.semantic_label}" if node.semantic_label else ""
    match = f"; visual cosine={similarity:.3f}" if similarity is not None else ""
    return _short(
        f"Visual {node.node_type} {node.node_id}{label}; views={node.observation_count}"
        f"{match}; identity uncertain", 150,
    )


def _edge_status(edge: SpatialEdgeV3) -> str:
    # A failure-only traversal can be used as an avoidance experience but
    # never as evidence of a successful passage.
    if edge.successful_traversals == 0 and edge.failed_traversals > 0:
        return "failed"
    if edge.traversability_status in {"verified", "traversed"} and edge.successful_traversals == 0:
        return "unknown"
    return edge.traversability_status


def _edge_description(edge: SpatialEdgeV3, status: str,
                      scene_nodes: dict[str, SpatialNodeV3]) -> str:
    def endpoint(node_id: str) -> str:
        node = scene_nodes.get(node_id)
        label = f" ({_short(node.semantic_label, 32)})" if node and node.semantic_label else ""
        return node_id + label

    return _short(
        f"{endpoint(edge.source_node)} → {endpoint(edge.target_node)}; "
        f"relation={edge.relation_type}; "
        f"successful traversals={edge.successful_traversals}; "
        f"failed traversals={edge.failed_traversals}; traversability={status}", 160,
    )


def _episode_description(episode: EpisodeMemory, store: V3MemoryStore,
                         *, active_segment: bool = False) -> str:
    actions = [store.get_evidence(ref) for ref in episode.actions]
    last_action = next((record for record in reversed(actions) if record is not None), None)
    action_text = last_action.payload.get("description") if last_action is not None else None
    if not isinstance(action_text, str) or not action_text.strip():
        action_text = last_action.event_type if last_action is not None else "none"
    labels = [store.get_evidence(ref) for ref in episode.landmarks]
    landmark_text = ", ".join(dict.fromkeys(
        record.semantic_label for record in labels
        if record is not None and record.semantic_label
    )) or "none"
    if active_segment:
        # The current instruction is already in the base prompt, so repeating
        # it spends scarce memory budget without adding historical evidence.
        return (
            f"Earlier segment: action={_short(action_text, 46)}; "
            f"saw={_short(landmark_text, 24)}; ongoing"
        )
    return (
        f"Past episode {_short(episode.episode_id, 20)}: "
        f"instruction={_short(episode.instruction, 30)}; "
        f"actions={len(episode.actions)} (last: {_short(action_text, 42)}); "
        f"landmarks={_short(landmark_text, 24)}; outcome={episode.outcome}"
    )


@dataclass(frozen=True)
class RetrievalResult:
    records: tuple[RetrievedMemory, ...]
    channel_candidates: dict[str, int]
    merged_candidates: int
    final_top_k: int
    history_counts: dict[str, int] = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        return {
            **{f"{name}_candidates": self.channel_candidates[name] for name in _CHANNELS},
            "merged_candidates": self.merged_candidates,
            "final_top_k": self.final_top_k,
            **{name: self.history_counts.get(name, 0) for name in _HISTORY_COUNT_KEYS},
        }


def reciprocal_rank_fusion(
    channels: dict[str, list[RetrievedMemory]], *, top_k: int, k0: int = 60,
) -> tuple[RetrievedMemory, ...]:
    """Merge channel ranks with deterministic identity deduplication."""
    if top_k < 0 or k0 < 1:
        raise ValueError("top_k must be nonnegative and k0 must be positive")
    combined: dict[str, RetrievedMemory] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for channel in _CHANNELS:
        seen_in_channel: set[str] = set()
        for rank, memory in enumerate(channels.get(channel, []), 1):
            if memory.memory_id in seen_in_channel:
                continue
            seen_in_channel.add(memory.memory_id)
            combined.setdefault(memory.memory_id, memory)
            scores[memory.memory_id] = scores.get(memory.memory_id, 0.0) + 1.0 / (k0 + rank)
            ranks.setdefault(memory.memory_id, {})[channel] = rank
    # RRF remains the primary rank. At an exact score tie, a direct observation
    # from this step must not lose to historical experience by lexical ID order.
    ordered = sorted(combined, key=lambda key: (
        -scores[key], -int(combined[key].current_observation), key,
    ))[:top_k]
    return tuple(replace(combined[key], score=scores[key], channel_ranks=ranks[key])
                 for key in ordered)


class V3Retriever:
    def __init__(self, store: V3MemoryStore, *, rrf_k0: int = 60) -> None:
        if rrf_k0 < 1:
            raise ValueError("rrf_k0 must be positive")
        self.store = store
        self.rrf_k0 = rrf_k0

    def _valid_refs(self, refs: Iterable[str], scene_id: str) -> tuple[str, ...] | None:
        unique = tuple(dict.fromkeys(refs))
        if not unique:
            return None
        for ref in unique:
            try:
                parsed = parse_evidence_key(ref)
            except (TypeError, ValueError):
                return None
            stored = self.store.get_evidence(ref)
            if parsed[0] != scene_id or stored is None or not stored.provenance_validity:
                return None
        return unique

    def _record_candidate(self, record: EvidenceRecord, *,
                          current_observation: bool = False) -> RetrievedMemory:
        return RetrievedMemory(
            memory_id=f"evidence:{record.key}",
            kind="action_history" if record.event_type in _BEHAVIOR_EVENTS else "observation",
            scene_id=record.scene_id, description=_record_description(record),
            evidence_ids=(record.key,), observation_status=record.observation_status,
            spatial_status=record.spatial_status,
            traversability_status=record.traversability_status,
            outcome=record.outcome, source_step=record.step,
            current_observation=current_observation,
        )

    def _node_candidate(self, node: SpatialNodeV3, refs: tuple[str, ...],
                        similarity: float | None = None) -> RetrievedMemory:
        key = json.dumps([node.scene_id, node.node_id], separators=(",", ":"))
        return RetrievedMemory(
            memory_id=f"node:{key}", kind="place", scene_id=node.scene_id,
            description=_node_description(node, similarity), evidence_ids=refs,
            observation_status="observed", spatial_status=node.spatial_status,
            source_step=node.last_seen_step,
        )

    def _edge_candidate(self, edge: SpatialEdgeV3, refs: tuple[str, ...],
                        scene_nodes: dict[str, SpatialNodeV3]) -> RetrievedMemory:
        key = json.dumps([edge.scene_id, edge.edge_id], separators=(",", ":"))
        status = _edge_status(edge)
        kind = edge.relation_type
        if kind in {"spatial_relation", "viewpoint_connection"}:
            kind = "inferred_relation"
        return RetrievedMemory(
            memory_id=f"edge:{key}", kind=kind,
            scene_id=edge.scene_id,
            description=_edge_description(edge, status, scene_nodes),
            evidence_ids=refs, observation_status="observed",
            spatial_status=edge.spatial_status, traversability_status=status,
            source_step=edge.last_seen_step,
        )

    def _episode_candidate(self, episode: EpisodeMemory,
                           refs: tuple[str, ...], *,
                           active_segment: bool = False) -> RetrievedMemory:
        components = [episode.scene_id, episode.episode_id, episode.rollout_idx]
        if active_segment:
            components.append(episode.end_step)
        key = json.dumps(components, separators=(",", ":"))
        return RetrievedMemory(
            memory_id=f"{'segment' if active_segment else 'episode'}:{key}",
            kind="action_history" if active_segment else "episodic_experience",
            scene_id=episode.scene_id,
            description=_episode_description(
                episode, self.store, active_segment=active_segment,
            ),
            evidence_ids=refs, outcome=episode.outcome,
            source_step=episode.end_step,
        )

    def retrieve(
        self, *, scene_id: str, instruction: str, current_step: int,
        visual_descriptor: Iterable[float] | None = None,
        place_id: str | None = None, episode_id: str | None = None,
        rollout_idx: int = 0, top_k: int = 3,
    ) -> RetrievalResult:
        """Retrieve from four channels; low lexical overlap only changes rank."""
        if not scene_id or current_step < 0 or top_k < 0:
            raise ValueError("scene_id, nonnegative current_step and top_k are required")
        if top_k == 0:
            return RetrievalResult((), {name: 0 for name in _CHANNELS}, 0, 0)
        evidence = [record for record in self.store.evidence_for_scene(scene_id)
                    if record.scene_id == scene_id and record.provenance_validity
                    and self._valid_refs((record.key,), scene_id)
                    and (not record.source_evidence_ids
                         or self._valid_refs(record.source_evidence_ids, scene_id))
                    and record.event_type != "instruction_match"
                    and not (record.episode_id == episode_id
                             and record.rollout_idx == rollout_idx
                             and record.step > current_step)]
        evidence_order = {record.key: index for index, record in enumerate(evidence)}
        nodes = [(node, refs) for node in self.store.nodes_for_scene(scene_id)
                 if node.scene_id == scene_id
                 if (refs := self._valid_refs(node.evidence_ids, scene_id)) is not None]
        scene_nodes = {node.node_id: node for node, _ in nodes}
        edges = [(edge, refs) for edge in self.store.edges_for_scene(scene_id)
                 if edge.scene_id == scene_id
                 if (refs := self._valid_refs(edge.evidence_ids, scene_id)) is not None]
        episodes = [(episode, refs, False)
                    for episode in self.store.episodes_for_scene(scene_id)
                    if episode.scene_id == scene_id
                    and not (episode.episode_id == episode_id
                             and episode.rollout_idx == rollout_idx)
                    if (refs := self._valid_refs(episode.evidence_ids, scene_id)) is not None]
        active_segment = self.store.active_segment(before_step=current_step)
        if (
            active_segment is not None and active_segment.scene_id == scene_id
            and active_segment.episode_id == episode_id
            and active_segment.rollout_idx == rollout_idx
        ):
            refs = self._valid_refs(active_segment.evidence_ids, scene_id)
            if refs is not None:
                episodes.append((active_segment, refs, True))

        visual_ranked: list[tuple[float, RetrievedMemory]] = []
        if visual_descriptor is not None:
            for node, refs in nodes:
                similarity = _cosine(visual_descriptor, node.visual_descriptor)
                if similarity is not None and similarity >= self.store.place_similarity_threshold:
                    visual_ranked.append((similarity, self._node_candidate(node, refs, similarity)))
        visual_ranked.sort(key=lambda entry: (-entry[0], entry[1].memory_id))
        visual = [memory for _, memory in visual_ranked[:top_k]]

        query_terms = _terms(instruction)
        short_keys = {record.key for record in self.store.current_short}
        semantic_ranked: list[tuple[int, int, int, int, int, RetrievedMemory]] = []
        for record in evidence:
            searchable = _terms((record.semantic_label, record.event_type,
                                 record.instruction_terms, record.payload.get("description")))
            overlap = len(query_terms & searchable)
            current_visual = (
                record.episode_id == episode_id and record.rollout_idx == rollout_idx
                and record.step == current_step and record.event_type in _VISUAL_EVENTS
                and (bool(record.semantic_label) or overlap > 0)
            )
            if overlap == 0 and not current_visual:
                continue
            # Direct RGB observations outrank derived instruction matches and
            # action notes. A current observation may be useful even when its
            # label is absent from the instruction; unrelated *history* needs
            # a visual or spatial match to enter the candidate pool.
            direct = int(record.event_type in _VISUAL_EVENTS)
            candidate = self._record_candidate(
                record, current_observation=current_visual,
            )
            semantic_ranked.append((int(current_visual), int(record.key in short_keys),
                                    direct, overlap, evidence_order[record.key], candidate))
        for node, refs in nodes:
            overlap = len(query_terms & _terms((node.semantic_label, node.node_type)))
            if overlap:
                recent = max((evidence_order.get(ref, -1) for ref in refs), default=-1)
                semantic_ranked.append((0, 0, 1, overlap, recent,
                                        self._node_candidate(node, refs)))
        for edge, refs in edges:
            endpoint_labels = (
                scene_nodes[edge.source_node].semantic_label,
                scene_nodes[edge.target_node].semantic_label,
            ) if edge.source_node in scene_nodes and edge.target_node in scene_nodes else ()
            overlap = len(query_terms & _terms((edge.relation_type, endpoint_labels)))
            if overlap:
                recent = max((evidence_order.get(ref, -1) for ref in refs), default=-1)
                semantic_ranked.append((0, 0, 0, overlap, recent,
                                        self._edge_candidate(edge, refs, scene_nodes)))
        semantic_ranked.sort(key=lambda entry: (
            -entry[0], -entry[1], -entry[2], -entry[3], -entry[4], entry[5].memory_id,
        ))
        semantic = [memory for *_, memory in semantic_ranked[:top_k]]

        spatial_ranked: list[tuple[int, RetrievedMemory]] = []
        if place_id is not None:
            for node, refs in nodes:
                if node.node_id == place_id:
                    spatial_ranked.append((3, self._node_candidate(node, refs)))
            for edge, refs in edges:
                if place_id in (edge.source_node, edge.target_node):
                    spatial_ranked.append((2, self._edge_candidate(edge, refs, scene_nodes)))
                    other = edge.target_node if edge.source_node == place_id else edge.source_node
                    for node, node_refs in nodes:
                        if node.node_id == other:
                            spatial_ranked.append((1, self._node_candidate(node, node_refs)))
        spatial_ranked.sort(key=lambda entry: (-entry[0], entry[1].memory_id))
        spatial = [memory for _, memory in spatial_ranked[:top_k]]

        episodic_ranked: list[tuple[float, int, RetrievedMemory]] = []
        for index, (episode, refs, is_active_segment) in enumerate(episodes):
            # The active Episode has the *same* instruction by definition; that
            # tautology is not relevance evidence for its earlier segment.
            matched = 0 if is_active_segment else len(query_terms & _terms(episode.instruction))
            landmark_terms = set()
            for ref in episode.landmarks:
                event = self.store.get_evidence(ref)
                if event is not None:
                    landmark_terms.update(_terms(event.semantic_label))
            matched += 0.5 * len(query_terms & landmark_terms)
            if matched > 0:
                episodic_ranked.append((matched, index, self._episode_candidate(
                    episode, refs, active_segment=is_active_segment,
                )))
        episodic_ranked.sort(key=lambda entry: (-entry[0], -entry[1], entry[2].memory_id))
        episodic = [memory for _, _, memory in episodic_ranked[:top_k]]

        channels = {
            "visual": visual, "semantic": semantic,
            "spatial": spatial, "episodic": episodic,
        }
        selected = reciprocal_rank_fusion(channels, top_k=top_k, k0=self.rrf_k0)
        history_cache: dict[str, tuple[bool, bool, bool]] = {}

        def history_flags(memory: RetrievedMemory) -> tuple[bool, bool, bool]:
            if memory.memory_id not in history_cache:
                same_prior = False
                cross_episode = False
                current = False
                for ref in memory.evidence_ids:
                    record = self.store.get_evidence(ref)
                    if record is None or episode_id is None:
                        continue
                    if (record.episode_id, record.rollout_idx) == (episode_id, rollout_idx):
                        same_prior |= record.step < current_step
                        current |= record.step == current_step
                    else:
                        cross_episode = True
                history_cache[memory.memory_id] = (same_prior, cross_episode, current)
            return history_cache[memory.memory_id]

        merged = {memory.memory_id: memory for group in channels.values() for memory in group}
        history_counts = {
            "historical_same_episode_prior_step": sum(
                history_flags(memory)[0] for memory in selected
            ),
            "historical_cross_episode": sum(
                history_flags(memory)[1] for memory in selected
            ),
            "historical_final_top_k": sum(
                any(history_flags(memory)[:2]) for memory in selected
            ),
            "historical_merged_candidates": sum(
                any(history_flags(memory)[:2]) for memory in merged.values()
            ),
            "source_step_lt_current_step": sum(
                memory.source_step is not None and memory.source_step < current_step
                for memory in selected
            ),
            "current_step_only_final_top_k": sum(
                history_flags(memory)[2] and not any(history_flags(memory)[:2])
                for memory in selected
            ),
            **{
                f"{channel}_historical_candidates": sum(
                    any(history_flags(memory)[:2]) for memory in candidates
                ) for channel, candidates in channels.items()
            },
        }
        for memory in selected:
            for ref in memory.evidence_ids:
                self.store.touch_short(ref)
        return RetrievalResult(
            records=selected,
            channel_candidates={
                "visual": len(visual_ranked),
                "semantic": len(semantic_ranked),
                "spatial": len(spatial_ranked),
                "episodic": len(episodic_ranked),
            },
            merged_candidates=len(merged),
            final_top_k=len(selected),
            history_counts=history_counts,
        )
