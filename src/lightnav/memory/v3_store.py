"""Scene-isolated, evidence-preserving three-layer RGM V3 memory storage.

The short cache is bounded. Episode-local scope discards the online bank at an
Episode boundary; scene-persistent scope retains observations in this process.
RGB arrays and evaluator state are never stored here. A command plus two RGB
views cannot prove that the agent physically traversed an edge.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from typing import Iterable

from lightnav.memory.v3_schemas import (
    EpisodeMemory,
    EvidenceRecord,
    SpatialEdgeV3,
    SpatialNodeV3,
)

_NODE_TYPES = frozenset({"place", "landmark", "viewpoint"})
_RELATIONS = frozenset({
    "observed_transition", "inferred_relation", "verified_traversal",
    "spatial_relation", "viewpoint_connection",
})
_ACTION_TYPES = frozenset({"action", "turn", "issued_command"})
_TRANSITION_TYPES = frozenset({
    "observed_transition", "inferred_relation", "verified_traversal",
    "possible_transition", "visual_change", "place_shift",
    "observed_motion", "motion_observation", "position_change",
})
_LANDMARK_TYPES = frozenset({"landmark", "object_observation"})
_SEGMENT_OBSERVATION_TYPES = frozenset({
    "object_observation", "landmark", "place_observation", "place_visit",
})
_OBSERVED_MOTION_TYPES = frozenset({"observed_motion", "motion_observation", "position_change"})
_OUTCOMES = frozenset({"success", "failure", "unknown", "partial", "truncated", "aborted"})
_SCOPES = frozenset({"episode_local", "scene_persistent"})
_MEASUREMENT_SOURCES = frozenset({
    "odometry", "wheel_odometry", "visual_odometry", "robot_localization",
    "proprioceptive_pose",
})


def _unit_vector(values: tuple[float, ...] | list[float] | None) -> tuple[float, ...] | None:
    if values is None:
        return None
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("visual descriptor must be finite and nonempty")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        raise ValueError("visual descriptor must have nonzero norm")
    return tuple(value / norm for value in vector)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("visual descriptor dimensions differ")
    return sum(a * b for a, b in zip(left, right))


def _position_distance(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    try:
        a = tuple(float(value) for value in left)
        b = tuple(float(value) for value in right)
    except (TypeError, ValueError):
        return None
    if len(a) != len(b) or not a or not all(math.isfinite(v) for v in (*a, *b)):
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _has_observed_displacement(record: EvidenceRecord) -> bool:
    """Require a measured displacement; an issued velocity command is insufficient."""
    if record.event_type not in _OBSERVED_MOTION_TYPES or record.observation_status != "observed":
        return False
    if record.payload.get("measurement_source") not in _MEASUREMENT_SOURCES:
        return False
    displacement = record.payload.get("displacement_m")
    if isinstance(displacement, (int, float)) and not isinstance(displacement, bool):
        if math.isfinite(displacement) and displacement > 0.01:
            return True
    before = record.payload.get("position_before")
    after = record.payload.get("position_after")
    measured = _position_distance(before, after)
    return measured is not None and measured > 0.01


class V3MemoryStore:
    """Three-layer bank with explicit Episode or Scene retention semantics."""

    def __init__(self, *, short_capacity: int = 8,
                 place_similarity_threshold: float = 0.85,
                 ambiguity_margin: float = 0.03,
                 scope: str = "episode_local") -> None:
        if not isinstance(short_capacity, int) or short_capacity < 1:
            raise ValueError("short_capacity must be positive")
        if not -1.0 <= place_similarity_threshold <= 1.0:
            raise ValueError("place_similarity_threshold must be in [-1, 1]")
        if not 0.0 <= ambiguity_margin <= 2.0:
            raise ValueError("ambiguity_margin must be in [0, 2]")
        if scope not in _SCOPES:
            raise ValueError(f"scope must be one of {sorted(_SCOPES)}")
        self.scope = scope
        self.short_capacity = short_capacity
        self.place_similarity_threshold = place_similarity_threshold
        self.ambiguity_margin = ambiguity_margin
        self._evidence: dict[str, EvidenceRecord] = {}
        self._scene_evidence: dict[str, list[str]] = {}
        self._episodes: dict[tuple[str, str, int], EpisodeMemory] = {}
        self._scene_nodes: dict[str, dict[str, SpatialNodeV3]] = {}
        self._scene_edges: dict[str, dict[tuple[str, str, str], SpatialEdgeV3]] = {}
        self._node_episode_refs: dict[tuple[str, str], set[tuple[str, str, int]]] = {}
        self._category_support: dict[tuple[str, str], int] = {}
        self._category_steps: dict[tuple[str, str, str, int], set[int]] = {}
        self._category_episodes: dict[tuple[str, str], set[tuple[str, int]]] = {}
        self._short: dict[str, tuple[int, int]] = {}  # canonical key -> (frequency, last touch)
        self._touch_clock = 0
        self._node_sequence: dict[str, int] = {}
        self._edge_sequence: dict[str, int] = {}
        self._traversal_motion_keys: set[tuple[str, str]] = set()
        self._active: tuple[str, str, int] | None = None
        self._instruction = ""
        self._journal: list[str] = []
        self.promoted_records = 0
        self.updated_records = 0

    @property
    def active_episode(self) -> tuple[str, str, int] | None:
        return self._active

    @property
    def current_short(self) -> tuple[EvidenceRecord, ...]:
        """Recent cache entries, newest first; reading does not count as a hit."""
        keys = sorted(self._short, key=lambda key: -self._short[key][1])
        return tuple(deepcopy(self._evidence[key]) for key in keys)

    def begin_episode(self, scene_id: str, episode_id: str, instruction: str,
                      rollout_idx: int = 0) -> None:
        if self._active is not None:
            raise RuntimeError("close the active Episode before beginning another")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id is required")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id is required")
        if not isinstance(rollout_idx, int) or rollout_idx < 0:
            raise ValueError("rollout_idx must be a nonnegative integer")
        key = (scene_id, episode_id, rollout_idx)
        if key in self._episodes:
            raise ValueError("Episode has already been closed in this store")
        self._active = key
        self._instruction = instruction
        self._journal = []
        self._short.clear()

    def add(self, record: EvidenceRecord) -> EvidenceRecord:
        """Persist every valid event, including low-confidence and unmatched ones."""
        if self._active is None:
            raise RuntimeError("begin_episode must be called first")
        if (record.scene_id, record.episode_id, record.rollout_idx) != self._active:
            raise ValueError("evidence belongs to another Scene or Episode")
        if record.key in self._evidence:
            raise ValueError(f"duplicate evidence key: {record.key}")
        for source_key in record.source_evidence_ids:
            source = self._evidence.get(source_key)
            if source is None or source.scene_id != record.scene_id:
                raise ValueError("source evidence must exist in the same Scene")
        label = (record.semantic_label or "").strip().casefold()
        if label and record.event_type in _LANDMARK_TYPES:
            support_key = (record.scene_id, label)
            count = self._category_support.get(support_key, 0) + 1
            self._category_support[support_key] = count
            step_key = (record.scene_id, label, record.episode_id, record.rollout_idx)
            steps = self._category_steps.setdefault(step_key, set())
            steps.add(record.step)
            episodes = self._category_episodes.setdefault(support_key, set())
            episodes.add((record.episode_id, record.rollout_idx))
            record = replace(
                record, support_count=max(record.support_count, count),
                temporal_support=max(record.temporal_support, len(steps)),
                episode_support=max(record.episode_support, len(episodes)),
            )
            if count > 1:
                self.updated_records += 1
        record = deepcopy(record)
        self._evidence[record.key] = record
        self._scene_evidence.setdefault(record.scene_id, []).append(record.key)
        self._journal.append(record.key)
        self._touch_clock += 1
        self._short[record.key] = (1, self._touch_clock)
        if len(self._short) > self.short_capacity:
            victim = min(self._short, key=lambda key: self._short[key])
            del self._short[victim]
        return deepcopy(record)

    def touch_short(self, evidence_key: str) -> bool:
        """Count an actual cache hit for recency and frequency eviction."""
        entry = self._short.get(evidence_key)
        if entry is None:
            return False
        self._touch_clock += 1
        self._short[evidence_key] = (entry[0] + 1, self._touch_clock)
        return True

    def get_evidence(self, evidence_key: str) -> EvidenceRecord | None:
        record = self._evidence.get(evidence_key)
        return deepcopy(record) if record is not None else None

    def evidence_for_scene(self, scene_id: str) -> tuple[EvidenceRecord, ...]:
        return tuple(deepcopy(self._evidence[key]) for key in self._scene_evidence.get(scene_id, ()))

    def episodes_for_scene(self, scene_id: str) -> tuple[EpisodeMemory, ...]:
        return tuple(deepcopy(episode) for key, episode in self._episodes.items() if key[0] == scene_id)

    def active_segment(self, *, before_step: int) -> EpisodeMemory | None:
        """Read-only Mid snapshot of earlier observations and completed actions.

        The current step is excluded. An action and a visual observation are
        required so an empty command history cannot become a memory prompt.
        """
        if self._active is None:
            return None
        if not isinstance(before_step, int) or before_step < 0:
            raise ValueError("before_step must be a nonnegative integer")
        records = [
            self._evidence[key] for key in self._journal
            if self._evidence[key].step < before_step
        ]
        if not any(record.event_type in _ACTION_TYPES for record in records):
            return None
        if not any(record.event_type in _SEGMENT_OBSERVATION_TYPES for record in records):
            return None
        scene_id, episode_id, rollout_idx = self._active
        steps = [record.step for record in records]
        return EpisodeMemory(
            scene_id=scene_id, episode_id=episode_id,
            instruction=self._instruction, rollout_idx=rollout_idx,
            observations=[record.key for record in records
                          if record.event_type not in _ACTION_TYPES],
            actions=[record.key for record in records
                     if record.event_type in _ACTION_TYPES],
            transitions=[record.key for record in records
                         if record.event_type in _TRANSITION_TYPES],
            landmarks=[record.key for record in records
                       if record.event_type in _LANDMARK_TYPES],
            outcome="unknown", start_step=min(steps), end_step=max(steps),
            evidence_ids=[record.key for record in records],
        )

    def nodes_for_scene(self, scene_id: str) -> tuple[SpatialNodeV3, ...]:
        return tuple(deepcopy(node) for node in self._scene_nodes.get(scene_id, {}).values())

    def edges_for_scene(self, scene_id: str) -> tuple[SpatialEdgeV3, ...]:
        return tuple(deepcopy(edge) for edge in self._scene_edges.get(scene_id, {}).values())

    def add_place(self, record: EvidenceRecord, *, node_id: str | None = None,
                  node_type: str = "place") -> SpatialNodeV3:
        """Increment one same-scene visual node; ambiguous matches create new nodes."""
        if node_type not in _NODE_TYPES:
            raise ValueError(f"unsupported node_type: {node_type}")
        stored = self._require_persisted(record)
        if not stored.provenance_validity:
            raise ValueError("invalid provenance cannot update spatial memory")
        descriptor = _unit_vector(stored.visual_descriptor)
        scene_nodes = self._scene_nodes.setdefault(stored.scene_id, {})
        node = scene_nodes.get(node_id) if node_id is not None else None
        if node_id is not None and (node is None or node.node_type != node_type):
            raise ValueError("spatial node does not exist in this Scene with the requested type")
        if node is None and descriptor is not None:
            candidates = []
            for candidate in scene_nodes.values():
                if candidate.node_type != node_type or candidate.visual_descriptor is None:
                    continue
                candidate_vector = _unit_vector(candidate.visual_descriptor)
                if candidate_vector is None or len(candidate_vector) != len(descriptor):
                    continue
                distance = _position_distance(candidate.position, stored.position)
                if distance is not None and distance > 2.0:
                    continue
                similarity = _cosine(candidate_vector, descriptor)
                if similarity >= self.place_similarity_threshold:
                    candidates.append((similarity, candidate.node_id))
            candidates.sort(key=lambda pair: (-pair[0], pair[1]))
            if candidates and (
                len(candidates) == 1
                or candidates[0][0] - candidates[1][0] >= self.ambiguity_margin
            ):
                node = scene_nodes[candidates[0][1]]
        if node is None:
            sequence = self._node_sequence.get(stored.scene_id, 0)
            self._node_sequence[stored.scene_id] = sequence + 1
            node = SpatialNodeV3(
                node_id=f"{node_type}_{sequence:04d}", scene_id=stored.scene_id,
                node_type=node_type, visual_descriptor=descriptor,
                semantic_label=stored.semantic_label, position=stored.position,
                detector_confidence=stored.raw_detector_confidence,
                observation_count=0, support_episode_count=0,
                evidence_ids=[], first_seen_step=stored.step,
                last_seen_step=stored.step, spatial_status="unknown",
            )
            scene_nodes[node.node_id] = node
        old_count = node.observation_count
        if old_count and descriptor is not None and node.visual_descriptor is not None:
            previous = _unit_vector(node.visual_descriptor)
            assert previous is not None
            if len(previous) != len(descriptor):
                raise ValueError("visual descriptor dimensions differ")
            node.visual_descriptor = _unit_vector(tuple(
                (old_count * old + new) / (old_count + 1)
                for old, new in zip(previous, descriptor)
            ))
        elif descriptor is not None:
            node.visual_descriptor = descriptor
        node.observation_count += 1
        episode_refs = self._node_episode_refs.setdefault((stored.scene_id, node.node_id), set())
        active = self._require_active()
        if active not in episode_refs:
            episode_refs.add(active)
            node.support_episode_count += 1
        node.evidence_ids.append(stored.key)
        # Steps restart at zero in each Episode; these are bounds on local step
        # values, while support_episode_count records cross-Episode sightings.
        node.first_seen_step = min(node.first_seen_step, stored.step)
        node.last_seen_step = max(node.last_seen_step, stored.step)
        if node.semantic_label is None and stored.semantic_label is not None:
            node.semantic_label = stored.semantic_label
        if node.position is None and stored.position is not None:
            node.position = stored.position
        if stored.raw_detector_confidence is not None:
            if node.detector_confidence is None:
                node.detector_confidence = stored.raw_detector_confidence
            else:
                node.detector_confidence = max(
                    node.detector_confidence, stored.raw_detector_confidence,
                )
        self.promoted_records += 1
        if old_count:
            self.updated_records += 1
        return deepcopy(node)

    def add_landmark(self, record: EvidenceRecord, *, node_id: str | None = None) -> SpatialNodeV3:
        """A label alone cannot merge physical landmarks; use an explicit ID or descriptor."""
        return self.add_place(record, node_id=node_id, node_type="landmark")

    def add_viewpoint(self, record: EvidenceRecord, *, node_id: str | None = None) -> SpatialNodeV3:
        return self.add_place(record, node_id=node_id, node_type="viewpoint")

    def add_relation(
        self, source_node: str, target_node: str, relation_type: str,
        *, evidence_keys: Iterable[str], motion_evidence_key: str | None = None,
    ) -> SpatialEdgeV3:
        """Keep visual transitions separate from measured physical traversal."""
        active = self._require_active()
        scene_id = active[0]
        if relation_type not in _RELATIONS:
            raise ValueError(f"unsupported relation_type: {relation_type}")
        if source_node == target_node:
            raise ValueError("a spatial edge requires two different nodes")
        scene_nodes = self._scene_nodes.get(scene_id, {})
        if source_node not in scene_nodes or target_node not in scene_nodes:
            raise ValueError("both spatial nodes must exist in the active Scene")
        keys = tuple(dict.fromkeys(evidence_keys))
        if not keys:
            raise ValueError("spatial relation requires evidence")
        records = []
        for key in keys:
            record = self._evidence.get(key)
            if record is None or record.scene_id != scene_id or not record.provenance_validity:
                raise ValueError("spatial relation evidence must be valid in the active Scene")
            records.append(record)
        node_keys = [set(scene_nodes[node_id].evidence_ids) for node_id in (source_node, target_node)]
        if any(not refs.intersection(keys) for refs in node_keys):
            raise ValueError("spatial relation requires evidence for both endpoint nodes")
        if relation_type in {"observed_transition", "verified_traversal"}:
            if any(not any(
                key in refs and (
                    self._evidence[key].scene_id,
                    self._evidence[key].episode_id,
                    self._evidence[key].rollout_idx,
                ) == active
                for key in keys
            ) for refs in node_keys):
                raise ValueError("an observed transition needs active-Episode endpoint observations")
        if relation_type == "verified_traversal":
            if motion_evidence_key is None or motion_evidence_key not in keys:
                raise ValueError("verified traversal requires independent motion evidence")
            motion = self._evidence[motion_evidence_key]
            if (motion.scene_id, motion.episode_id, motion.rollout_idx) != active:
                raise ValueError("verified traversal requires current-Episode motion evidence")
            if not _has_observed_displacement(motion):
                raise ValueError("an issued command or RGB shift is not measured traversal")
            source_steps = [
                record.step for record in records
                if record.key in node_keys[0]
            ]
            target_steps = [
                record.step for record in records
                if record.key in node_keys[1]
            ]
            if not any(start <= motion.step <= end
                       for start in source_steps for end in target_steps):
                raise ValueError("measured motion must occur between endpoint observations")
        elif motion_evidence_key is not None:
            raise ValueError("motion evidence is only used for verified_traversal")
        edge_key = (source_node, target_node, relation_type)
        scene_edges = self._scene_edges.setdefault(scene_id, {})
        edge = scene_edges.get(edge_key)
        if relation_type == "verified_traversal" and any(
            known_motion == motion_evidence_key
            and (edge is None or known_edge != edge.edge_id)
            for known_edge, known_motion in self._traversal_motion_keys
        ):
            raise ValueError("one measured motion cannot verify multiple spatial edges")
        if edge is None:
            sequence = self._edge_sequence.get(scene_id, 0)
            self._edge_sequence[scene_id] = sequence + 1
            measured = relation_type == "verified_traversal"
            edge = SpatialEdgeV3(
                edge_id=f"edge_{sequence:04d}", scene_id=scene_id,
                source_node=source_node, target_node=target_node,
                relation_type=relation_type, evidence_ids=[],
                traversal_count=int(measured), successful_traversals=int(measured),
                first_seen_step=min(record.step for record in records),
                last_seen_step=max(record.step for record in records),
                spatial_status=("observed" if relation_type == "observed_transition"
                                else "traversed" if relation_type == "verified_traversal"
                                else "inferred"),
                traversability_status="traversed" if measured else "unknown",
            )
            scene_edges[edge_key] = edge
            if measured:
                assert motion_evidence_key is not None
                self._traversal_motion_keys.add((edge.edge_id, motion_evidence_key))
        else:
            self.updated_records += 1
        for key in keys:
            if key not in edge.evidence_ids:
                edge.evidence_ids.append(key)
        edge.first_seen_step = min(edge.first_seen_step, *(record.step for record in records))
        edge.last_seen_step = max(edge.last_seen_step, *(record.step for record in records))
        confidences = [
            self._evidence[key].raw_detector_confidence for key in edge.evidence_ids
            if self._evidence[key].raw_detector_confidence is not None
        ]
        if confidences:
            edge.confidence_statistics = {
                "count": len(confidences), "min": min(confidences),
                "max": max(confidences), "mean": sum(confidences) / len(confidences),
            }
        if relation_type == "verified_traversal":
            assert motion_evidence_key is not None
            traversal_key = (edge.edge_id, motion_evidence_key)
            if traversal_key not in self._traversal_motion_keys:
                self._traversal_motion_keys.add(traversal_key)
                edge.traversal_count += 1
                edge.successful_traversals += 1
            edge.spatial_status = "traversed"
            edge.traversability_status = "traversed"
        self.promoted_records += 1
        return deepcopy(edge)

    def close_episode(self, *, outcome: str = "unknown") -> EpisodeMemory | None:
        """Promote the journal, then apply the configured memory retention scope."""
        if self._active is None:
            return None
        if outcome not in _OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(_OUTCOMES)}")
        scene_id, episode_id, rollout_idx = self._active
        records = [self._evidence[key] for key in self._journal]
        steps = [record.step for record in records]
        episode = EpisodeMemory(
            scene_id=scene_id, episode_id=episode_id, instruction=self._instruction,
            rollout_idx=rollout_idx,
            observations=[record.key for record in records if record.event_type not in _ACTION_TYPES],
            actions=[record.key for record in records if record.event_type in _ACTION_TYPES],
            transitions=[record.key for record in records if record.event_type in _TRANSITION_TYPES],
            landmarks=[record.key for record in records if record.event_type in _LANDMARK_TYPES],
            outcome=outcome,
            start_step=min(steps, default=0), end_step=max(steps, default=0),
            evidence_ids=[record.key for record in records],
        )
        self._episodes[self._active] = episode
        self.promoted_records += len(records)
        self._active = None
        self._instruction = ""
        self._journal = []
        self._short.clear()
        if self.scope == "episode_local":
            self._clear_banks()
        return deepcopy(episode)

    def clear(self) -> None:
        """Explicitly drop online state, including any active Episode journal."""
        self._active = None
        self._instruction = ""
        self._journal = []
        self._clear_banks()
        self.promoted_records = 0
        self.updated_records = 0

    def _clear_banks(self) -> None:
        self._evidence.clear()
        self._scene_evidence.clear()
        self._episodes.clear()
        self._scene_nodes.clear()
        self._scene_edges.clear()
        self._node_episode_refs.clear()
        self._category_support.clear()
        self._category_steps.clear()
        self._category_episodes.clear()
        self._short.clear()
        self._touch_clock = 0
        self._node_sequence.clear()
        self._edge_sequence.clear()
        self._traversal_motion_keys.clear()

    def _require_active(self) -> tuple[str, str, int]:
        if self._active is None:
            raise RuntimeError("begin_episode must be called first")
        return self._active

    def _require_persisted(self, record: EvidenceRecord) -> EvidenceRecord:
        active = self._require_active()
        stored = self._evidence.get(record.key)
        if stored is None or (stored.scene_id, stored.episode_id, stored.rollout_idx) != active:
            raise ValueError("spatial update requires persisted active-Episode evidence")
        return stored
