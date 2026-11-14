"""Pose grounded local graph facts for the frozen LightNav prompt interface.

Only the observation viewpoint has a measured position. A straight-line
bearing never implies an unobstructed route or a next action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape

from lightnav.memory.local_graph_fusion import (
    _HEADER,
    _MIN_CONFIDENCE,
    LocalGraphFusion,
    _plain_label,
    _provenance,
)
from lightnav.memory.schemas import LayerMemoryItem, MemoryContext, MemoryEvent, ObservationEvidence
from lightnav.memory.spatial_geometry import AgentPose, relative_direction, relative_viewpoint
from lightnav.memory.v3_retrieval import _terms
from lightnav.memory.v3_schemas import EvidenceRecord, SpatialEdgeV3, SpatialNodeV3
from lightnav.memory.v3_store import V3MemoryStore

_DETECTION_KINDS = frozenset({"object_observation", "landmark"})


def _plain_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)[:40]


@dataclass(frozen=True)
class SpatialLandmarkCandidate:
    """One historical detection and its measured observation viewpoint."""

    semantic_label: str
    landmark_node_id: str
    source_place_node_id: str | None
    source_episode_id: str
    source_step: int
    source_position: tuple[float, float, float] | None
    current_position: tuple[float, float, float] | None
    relative_distance: float | None
    relative_bearing: float | None
    graph_relation: str
    traversability_status: str
    evidence_confidence: float
    evidence_ids: tuple[str, ...]
    source_frame_id: str | None
    graph_edge_ids: tuple[str, ...]
    record: EvidenceRecord

    @property
    def spatial_known(self) -> bool:
        return self.relative_distance is not None

    def line(self) -> str:
        place = self.source_place_node_id or "unlinked"
        if self.relative_distance is None:
            relative = "distance unknown, bearing unknown"
        elif self.relative_bearing is None:
            relative = f"{self.relative_distance:.1f}m, at viewpoint"
        else:
            degrees = math.degrees(self.relative_bearing)
            bearing = (f"{abs(degrees):.0f}deg "
                       f"{'left' if degrees > 0 else 'right'}") if abs(degrees) >= 0.5 else "0deg"
            relative = (f"{self.relative_distance:.1f}m "
                        f"{relative_direction(self.relative_bearing, self.relative_distance)} "
                        f"({bearing})")
        return (
            f"History: {self.semantic_label}; observed viewpoint {_plain_id(place)} "
            f"[{_plain_id(self.source_episode_id)}:{self.source_step}]; relative viewpoint {relative}; "
            f"graph {self.graph_relation}; passage {self.traversability_status}; "
            f"detector {self.evidence_confidence:.2f}."
        )

    def signature(self) -> tuple[object, ...]:
        distance_bin = (round(self.relative_distance * 2) if self.relative_distance is not None
                        else None)
        return (
            self.landmark_node_id, self.source_place_node_id,
            distance_bin, relative_direction(self.relative_bearing, self.relative_distance),
            self.graph_relation, self.traversability_status,
        )


def _verified(edge: SpatialEdgeV3) -> bool:
    return (
        edge.relation_type == "verified_traversal"
        and edge.traversal_count > 0 and edge.successful_traversals > 0
        and edge.traversability_status in {"traversed", "verified"}
    )


def _graph_relation(
    current_place_id: str | None, source_place_id: str | None,
    edges: tuple[SpatialEdgeV3, ...], nodes: dict[str, SpatialNodeV3],
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    if current_place_id is None or source_place_id is None:
        return "unknown", "unknown", (), ()
    if current_place_id == source_place_id:
        return "same_visual_place", "unknown", (), ()
    if current_place_id not in nodes or source_place_id not in nodes:
        return "unknown", "unknown", (), ()
    if nodes[current_place_id].node_type != "place" or nodes[source_place_id].node_type != "place":
        return "unknown", "unknown", (), ()
    direct = [edge for edge in edges if edge.relation_type != "verified_traversal" and {
        edge.source_node, edge.target_node,
    } == {current_place_id, source_place_id}]
    verified_edges = [edge for edge in edges if _verified(edge) and (
        nodes.get(edge.source_node) is not None
        and nodes.get(edge.target_node) is not None
        and nodes[edge.source_node].node_type == nodes[edge.target_node].node_type == "place"
    )]
    frontier = [(current_place_id, (), ())]
    seen = {current_place_id}
    while frontier:
        place_id, path_ids, path_refs = frontier.pop(0)
        for edge in verified_edges:
            if place_id not in {edge.source_node, edge.target_node}:
                continue
            neighbor = edge.target_node if edge.source_node == place_id else edge.source_node
            if neighbor in seen:
                continue
            next_ids = (*path_ids, edge.edge_id)
            next_refs = (*path_refs, *edge.evidence_ids)
            if neighbor == source_place_id:
                relation = "verified_traversal" if len(next_ids) == 1 else "verified_path"
                return relation, "traversed", next_ids, tuple(dict.fromkeys(next_refs))
            seen.add(neighbor)
            frontier.append((neighbor, next_ids, next_refs))
    if direct:
        priority = {"observed_transition": 0, "inferred_relation": 1,
                    "spatial_relation": 2, "viewpoint_connection": 3}
        edge = min(direct, key=lambda item: priority.get(item.relation_type, 4))
        return edge.relation_type, "unknown", (edge.edge_id,), tuple(edge.evidence_ids)
    return "unknown", "unknown", (), ()


def _source_place(
    record: EvidenceRecord, landmark_id: str, edges: tuple[SpatialEdgeV3, ...],
    nodes: dict[str, SpatialNodeV3], store: V3MemoryStore,
) -> tuple[str | None, EvidenceRecord | None]:
    """Use only a Place→Landmark edge supported by this exact detection."""
    for edge in edges:
        if edge.relation_type not in {"inferred_relation", "spatial_relation"}:
            continue
        if landmark_id not in {edge.source_node, edge.target_node} or record.key not in edge.evidence_ids:
            continue
        place_id = edge.target_node if edge.source_node == landmark_id else edge.source_node
        place = nodes.get(place_id)
        if place is None or place.node_type != "place":
            continue
        for key in edge.evidence_ids:
            source = store.get_evidence(key)
            if (source is not None and key in place.evidence_ids
                    and source.event_type == "place_observation"
                    and (source.episode_id, source.rollout_idx, source.step) == (
                        record.episode_id, record.rollout_idx, record.step,
                    )):
                return place_id, source
    return None, None


class SpatialLocalGraphFusion(LocalGraphFusion):
    """Local metric search with explicit evidence and graph trust semantics."""

    def build(
        self, store: V3MemoryStore, *, scene_id: str, episode_id: str,
        rollout_idx: int, step: int, instruction: str,
        current_place_id: str | None, rgb_width: int,
        events: list[MemoryEvent], current_pose: AgentPose | None = None,
    ) -> MemoryContext | None:
        self._pending = None
        if not self.config.enabled or self.config.fusion_mode != "spatial":
            return None
        identity = (scene_id, episode_id, rollout_idx)
        if self._episode != identity or store.active_episode != identity:
            raise ValueError("spatial local graph requires the active Scene/Episode")
        if store.scope != self.config.scope:
            raise ValueError("spatial local graph and store scopes differ")
        if type(step) is not int or step < 0 or type(rgb_width) is not int or rgb_width < 1:
            raise ValueError("step must be nonnegative and RGB width positive")
        if current_pose is not None and current_pose.scene_id != scene_id:
            raise ValueError("current pose differs from active memory Scene")

        current = self._current_candidates(
            store, scene_id, episode_id, rollout_idx, step,
            instruction, rgb_width, events,
        )
        history = self._historical_spatial_candidates(
            store, scene_id=scene_id, episode_id=episode_id,
            rollout_idx=rollout_idx, step=step, instruction=instruction,
            current_place_id=current_place_id, current_pose=current_pose,
        )
        selected: list[tuple[str, object]] = [
            ("current", item) for item in current[:self.config.max_landmarks]
        ]
        if history and selected:
            if len(selected) >= self.config.max_landmarks and self.config.max_landmarks > 1:
                selected[-1] = ("history", history[0])
            elif len(selected) < self.config.max_landmarks:
                selected.append(("history", history[0]))
        elif history:
            selected = [("history", item) for item in history[:self.config.max_landmarks]]
        if not selected:
            return None
        # A raw current RGB frame is already in the video. If both complete
        # lines cannot fit, keep one grounded historical fact first.
        history_selected = [pair for pair in selected if pair[0] == "history"]
        if history_selected and (len(_HEADER) + sum(
            1 + len(escape(item.line(), quote=False)) for _, item in selected
        ) > self.config.prompt_max_chars):
            selected = history_selected + [pair for pair in selected if pair[0] == "current"]
        signature: tuple[object, ...] = (
            current_place_id,
            tuple((role, item.signature()) for role, item in selected),
        )
        if (signature == self._last_signature and self._last_injected_step is not None
                and step - self._last_injected_step < self.config.reactivation_steps):
            return None

        lines: list[str] = []
        items: list[LayerMemoryItem] = []
        used = len(_HEADER)
        for role, item in selected:
            line = item.line()
            safe = escape(line, quote=False)
            if used + 1 + len(safe) > self.config.prompt_max_chars:
                continue
            used += 1 + len(safe)
            lines.append(line)
            record = item.record
            if role == "history":
                evidence_ids = list(item.evidence_ids)
                spatial_fields = {
                    "semantic_label": item.semantic_label,
                    "landmark_node_id": item.landmark_node_id,
                    "source_place_node_id": item.source_place_node_id,
                    "source_episode_id": item.source_episode_id,
                    "source_step": item.source_step,
                    "source_position": item.source_position,
                    "current_position": item.current_position,
                    "relative_distance": item.relative_distance,
                    "relative_bearing": item.relative_bearing,
                    "relative_direction": relative_direction(
                        item.relative_bearing, item.relative_distance,
                    ),
                    "graph_relation": item.graph_relation,
                    "traversability_status": item.traversability_status,
                    "evidence_confidence": item.evidence_confidence,
                    "evidence_ids": evidence_ids,
                    "source_frame_id": item.source_frame_id,
                    "graph_edge_ids": list(item.graph_edge_ids),
                }
            else:
                evidence_ids = [record.key]
                spatial_fields = None
            provenance = []
            refs = []
            for key in evidence_ids:
                source = store.get_evidence(key)
                if source is None:
                    raise ValueError("spatial candidate refers to missing evidence")
                provenance.append(_provenance(source))
                refs.append(ObservationEvidence(
                    event_id=key, step_id=source.step,
                    label=item.label if role == "current" else item.semantic_label,
                    confidence=source.raw_detector_confidence or 0.0,
                    source=source.detector_name or "habitat_pose",
                ))
            items.append(LayerMemoryItem(
                layer="short" if role == "current" else "long",
                kind=f"local_graph_spatial_{role}_landmark", description=line,
                step_id=record.step, confidence=record.raw_detector_confidence or 0.0,
                source="local_graph_spatial", evidence_refs=refs,
                label=item.label if role == "current" else item.semantic_label,
                metadata={
                    "evidence_ids": evidence_ids, "serialized_text": safe,
                    "provenance": provenance, "role": role,
                    "spatial_fields": spatial_fields,
                    "current_view_sector": item.sector if role == "current" else None,
                },
            ))
        if not items:
            return None
        self._pending = (
            step, signature,
            frozenset(ref.event_id for item in items for ref in item.evidence_refs),
        )
        return MemoryContext(text=_HEADER + "\n" + "\n".join(lines), layer_items=items)

    def _historical_spatial_candidates(
        self, store: V3MemoryStore, *, scene_id: str, episode_id: str,
        rollout_idx: int, step: int, instruction: str,
        current_place_id: str | None, current_pose: AgentPose | None,
    ) -> list[SpatialLandmarkCandidate]:
        nodes = {node.node_id: node for node in store.nodes_for_scene(scene_id)}
        edges = store.edges_for_scene(scene_id)
        candidates: list[SpatialLandmarkCandidate] = []
        for landmark in nodes.values():
            if landmark.node_type != "landmark":
                continue
            for key in landmark.evidence_ids:
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
                label = _plain_label(record.semantic_label or "")
                if not label or not (_terms(label) & _terms(instruction)):
                    continue
                source_place_id, place_record = _source_place(
                    record, landmark.node_id, edges, nodes, store,
                )
                source_position = record.position or (
                    place_record.position if place_record is not None else None
                )
                frame_id = record.payload.get("pose_frame") or (
                    place_record.payload.get("pose_frame") if place_record is not None else None
                )
                distance = bearing = None
                if current_pose is not None and source_position is not None and isinstance(frame_id, str):
                    try:
                        distance, bearing = relative_viewpoint(
                            current_pose, source_position,
                            target_scene_id=record.scene_id, target_frame_id=frame_id,
                        )
                    except ValueError:
                        pass
                if distance is not None and distance > self.config.neighbor_radius_m:
                    continue
                relation, traversability, edge_ids, edge_refs = _graph_relation(
                    current_place_id, source_place_id, edges, nodes,
                )
                # Geometry permits local retrieval without a DINO place match. With
                # unknown geometry, require at least a local visual graph relation.
                if distance is None and relation == "unknown":
                    continue
                evidence_ids = tuple(dict.fromkeys((
                    record.key,
                    *((place_record.key,) if place_record is not None else ()),
                    *edge_refs,
                )))
                candidates.append(SpatialLandmarkCandidate(
                    semantic_label=label, landmark_node_id=landmark.node_id,
                    source_place_node_id=source_place_id,
                    source_episode_id=record.episode_id, source_step=record.step,
                    source_position=source_position, current_position=(
                        current_pose.position if current_pose is not None else None
                    ),
                    relative_distance=distance, relative_bearing=bearing,
                    graph_relation=relation, traversability_status=traversability,
                    evidence_confidence=record.raw_detector_confidence or 0.0,
                    evidence_ids=evidence_ids, source_frame_id=(
                        frame_id if isinstance(frame_id, str) else None
                    ),
                    graph_edge_ids=edge_ids, record=record,
                ))
        # Keep distinct landmark nodes even when they share a semantic label.
        return sorted(candidates, key=lambda item: (
            0 if item.traversability_status == "traversed" else 1,
            0 if item.spatial_known else 1,
            item.relative_distance if item.relative_distance is not None else float("inf"),
            -(item.evidence_confidence), -item.source_step, item.landmark_node_id,
        ))
