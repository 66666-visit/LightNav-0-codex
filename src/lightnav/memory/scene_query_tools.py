"""Strict, read-only structured queries over the existing ConceptGraphs-Lite map.

The query API exposes only fields already recorded by ``ObjectMap``. In
particular, ``observed_from`` is provenance, not spatial adjacency. No query
touches ObjectMap, the V3 short cache, or the geometry association code.
"""

from __future__ import annotations

import json
import math
from typing import Any

from lightnav.memory.conceptgraphs_lite import ObjectMap
from lightnav.memory.v3_schemas import parse_evidence_key
from lightnav.memory.v3_store import V3MemoryStore

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "find_objects": {
        "type": "object",
        "properties": {
            "class_name": {"type": "string", "minLength": 1},
            "object_id": {"type": "string", "minLength": 1},
            "min_observation_count": {"type": "integer", "minimum": 1},
        },
        "required": [],
        "additionalProperties": False,
    },
    "read_node": {
        "type": "object",
        "properties": {"object_id": {"type": "string", "minLength": 1}},
        "required": ["object_id"],
        "additionalProperties": False,
    },
    "read_evidence": {
        "type": "object",
        "properties": {"evidence_id": {"type": "string", "minLength": 1}},
        "required": ["evidence_id"],
        "additionalProperties": False,
    },
    "get_edges_for_node": {
        "type": "object",
        "properties": {"object_id": {"type": "string", "minLength": 1}},
        "required": ["object_id"],
        "additionalProperties": False,
    },
}

_DESCRIPTIONS = {
    "find_objects": "Find recorded 3D object nodes by exact class, ID, or observation count.",
    "read_node": "Read one existing ObjectMap node and its Evidence IDs.",
    "read_evidence": "Read one recorded spatial Evidence and its Object IDs.",
    "get_edges_for_node": "Read actual observed_from provenance edges of one object.",
}

OPENAI_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": name, "description": _DESCRIPTIONS[name], "parameters": schema,
    }}
    for name, schema in TOOL_SCHEMAS.items()
]


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON argument: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON argument: {value}")


def _validate_arguments(name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
    if name not in TOOL_SCHEMAS:
        raise ValueError(f"unknown L3 query tool: {name}")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(
                arguments, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments must be valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")
    schema = TOOL_SCHEMAS[name]
    if any(not isinstance(key, str) for key in arguments):
        raise ValueError("tool argument names must be strings")
    unknown = set(arguments) - set(schema["properties"])
    if unknown:
        raise ValueError(f"unknown {name} argument(s): {', '.join(sorted(unknown))}")
    missing = set(schema["required"]) - set(arguments)
    if missing:
        raise ValueError(f"missing {name} argument(s): {', '.join(sorted(missing))}")
    for key, value in arguments.items():
        field = schema["properties"][key]
        if field["type"] == "string":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name}.{key} must be a nonempty string")
        elif field["type"] == "integer":
            if type(value) is not int or value < field["minimum"]:
                raise ValueError(f"{name}.{key} must be an integer >= {field['minimum']}")
    return arguments


def _finite_vector(values: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain {length} coordinates")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must have finite coordinates")
    return tuple(float(value) for value in values)


class SceneQueryTools:
    """Snapshot-like read adapter; callers keep the map stable during each step."""

    def __init__(
        self, object_map: ObjectMap, store: V3MemoryStore | None, *,
        scene_id: str, episode_id: str, rollout_idx: int, step: int,
        scope: str, min_confidence: float,
    ) -> None:
        if not scene_id or not episode_id or type(rollout_idx) is not int or rollout_idx < 0:
            raise ValueError("SceneQueryTools requires a Scene/Episode/rollout identity")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if scope not in {"episode_local", "scene_persistent"}:
            raise ValueError("invalid spatial scope")
        if (type(min_confidence) not in (int, float) or not math.isfinite(min_confidence)
                or not 0 <= min_confidence <= 1):
            raise ValueError("min_confidence must be finite and in [0, 1]")
        if object_map._identity != (scene_id, episode_id, rollout_idx):
            raise ValueError("ObjectMap does not match the active Scene/Episode")
        if store is not None and (
            store.active_episode != (scene_id, episode_id, rollout_idx)
            or store.scope != scope
        ):
            raise ValueError("Evidence store does not match the active scope or Episode")
        self.object_map = object_map
        self.store = store
        self.scene_id = scene_id
        self.episode_id = episode_id
        self.rollout_idx = rollout_idx
        self.step = step
        self.scope = scope
        self.min_confidence = float(min_confidence)

    def call(self, name: str, arguments: dict[str, Any] | str) -> list[dict] | dict:
        """Dispatch a strictly validated query, with no dynamic method lookup."""
        args = _validate_arguments(name, arguments)
        if name == "find_objects":
            return self.find_objects(**args)
        if name == "read_node":
            return self.read_node(**args)
        if name == "read_evidence":
            return self.read_evidence(**args)
        if name == "get_edges_for_node":
            return self.get_edges_for_node(**args)
        raise AssertionError("validated query was not dispatched")

    def find_objects(
        self, *, class_name: str | None = None, object_id: str | None = None,
        min_observation_count: int | None = None,
    ) -> list[dict]:
        if object_id is not None and object_id not in self.object_map.objects:
            raise ValueError(f"invalid Object ID: {object_id}")
        result = []
        for node in self.object_map.objects.values():
            if object_id is not None and node.object_id != object_id:
                continue
            if class_name is not None and node.class_name.casefold() != class_name.casefold():
                continue
            if (min_observation_count is not None
                    and node.observation_count < min_observation_count):
                continue
            result.append(self._node_record(node.object_id))
        return sorted(result, key=lambda row: row["object_id"])

    def read_node(self, *, object_id: str) -> dict:
        if object_id not in self.object_map.objects:
            raise ValueError(f"invalid Object ID: {object_id}")
        return self._node_record(object_id)

    def _node_record(self, object_id: str) -> dict:
        node = self.object_map.objects[object_id]
        center = _finite_vector(node.center_world, 3, "object center")
        if not isinstance(node.bbox_3d, (tuple, list)) or len(node.bbox_3d) != 2:
            raise ValueError("object bounding box is invalid")
        box = [_finite_vector(part, 3, "object bounding box") for part in node.bbox_3d]
        return {
            "object_id": node.object_id,
            "class_name": node.class_name,
            "center_world": list(center),
            "bbox_3d": [list(part) for part in box],
            "observation_count": node.observation_count,
            "geometry_status": node.geometry_status,
            "evidence_ids": list(node.evidence_ids),
        }

    def read_evidence(self, *, evidence_id: str) -> dict:
        record = self.object_map.evidence.get(evidence_id)
        if record is None:
            raise ValueError(f"invalid Evidence ID: {evidence_id}")
        object_ids = sorted(
            node.object_id for node in self.object_map.objects.values()
            if evidence_id in node.evidence_ids
        )
        return {
            "evidence_id": record.evidence_id,
            "object_ids": object_ids,
            "scene_id": record.scene_id,
            "episode_id": record.episode_id,
            "rollout_idx": record.rollout_idx,
            "step_id": record.step_id,
            "class_name": record.class_name,
            "confidence": record.confidence,
            "source": record.source,
            "mask_pixel_count": record.mask_pixel_count,
            "depth_point_count": record.depth_point_count,
            "geometry_status": record.geometry_status,
            "linked_l1_evidence_id": record.linked_l1_evidence_id,
        }

    def get_edges_for_node(self, *, object_id: str) -> list[dict]:
        if object_id not in self.object_map.objects:
            raise ValueError(f"invalid Object ID: {object_id}")
        # A TraversedToEdge connects viewpoints, never two ObjectNodes.
        return [
            {
                "object_id": edge.object_id,
                "viewpoint_id": edge.viewpoint_id,
                "evidence_id": edge.evidence_id,
                "relation_type": edge.relation_type,
            }
            for edge in self.object_map.edges
            if edge.relation_type == "observed_from" and edge.object_id == object_id
        ]

    def candidate_counts(self) -> dict[str, int]:
        """Report map size, raw geometry candidates, and proven historical nodes."""
        raw_ids = [
            node.object_id for node in self.object_map.objects.values()
            if node.geometry_status == "observed"
        ]
        valid = 0
        for object_id in raw_ids:
            node = self.object_map.objects[object_id]
            if any(self._history_or_none(object_id, eid) is not None
                   for eid in node.evidence_ids):
                valid += 1
        return {
            "l3_object_node_count": len(self.object_map.objects),
            "l3_raw_candidate_count": len(raw_ids),
            "l3_evidence_valid_count": valid,
        }

    def _history_or_none(self, object_id: str, evidence_id: str) -> dict | None:
        try:
            return self.validate_history(object_id, evidence_id)
        except ValueError:
            return None

    def validate_history(self, object_id: str, evidence_id: str) -> dict:
        """Verify one old observation and the complete fused-node provenance.

        The returned dictionary is a detached, JSON-safe fact snapshot for the
        deterministic renderer. It never equates two same-label Object IDs.
        """
        node = self.object_map.objects.get(object_id)
        if node is None:
            raise ValueError(f"invalid Object ID: {object_id}")
        if evidence_id not in self.object_map.evidence:
            raise ValueError(f"invalid Evidence ID: {evidence_id}")
        if evidence_id not in node.evidence_ids:
            raise ValueError("Evidence does not belong to the selected Object ID")
        if node.geometry_status != "observed":
            raise ValueError("Object geometry is not observed")
        # The fused center/box uses all masks. A weak mask can move it, so all
        # contributing Evidence must satisfy the existing injection floor.
        if not node.evidence_ids:
            raise ValueError("Object has no Evidence")
        for ref in node.evidence_ids:
            contributor = self.object_map.evidence.get(ref)
            if (contributor is None or contributor.geometry_status != "observed"
                    or contributor.class_name.casefold() != node.class_name.casefold()
                    or not math.isfinite(contributor.confidence)
                    or contributor.confidence < self.min_confidence):
                raise ValueError("fused Object has missing or weak Evidence")
        record = self.object_map.evidence[evidence_id]
        try:
            ref_identity = parse_evidence_key(record.evidence_id)[:3]
        except ValueError as exc:
            raise ValueError("Evidence ID is not canonical") from exc
        if ref_identity != (record.scene_id, record.episode_id, record.rollout_idx):
            raise ValueError("Evidence ID and provenance identity disagree")
        if record.scene_id != self.scene_id:
            raise ValueError("Evidence crosses Scene identity")
        current_episode = (
            record.episode_id, record.rollout_idx
        ) == (self.episode_id, self.rollout_idx)
        if self.scope == "episode_local" and not current_episode:
            raise ValueError("Evidence crosses Episode-local scope")
        if current_episode and record.step_id >= self.step:
            raise ValueError("Evidence is not historical at this step")
        viewpoints = self.object_map.viewpoints
        matching = [
            edge for edge in self.object_map.edges
            if edge.relation_type == "observed_from"
            and edge.object_id == object_id and edge.evidence_id == evidence_id
        ]
        if len(matching) != 1:
            raise ValueError("Evidence requires one observed_from edge")
        viewpoint = viewpoints.get(matching[0].viewpoint_id)
        if viewpoint is None or (
            record.scene_id, record.episode_id, record.rollout_idx, record.step_id
        ) != (
            viewpoint.scene_id, viewpoint.episode_id, viewpoint.rollout_idx,
            viewpoint.step_id,
        ):
            raise ValueError("Evidence viewpoint provenance disagrees")
        center = _finite_vector(node.center_world, 3, "object center")
        if not isinstance(node.bbox_3d, (tuple, list)) or len(node.bbox_3d) != 2:
            raise ValueError("object bounding box is invalid")
        box = [_finite_vector(part, 3, "object bounding box") for part in node.bbox_3d]
        viewpoint_position = _finite_vector(
            viewpoint.camera_pose.position, 3, "viewpoint camera position",
        )
        linked_l1_id = record.linked_l1_evidence_id
        if linked_l1_id is not None:
            l1 = self.store.get_evidence(linked_l1_id) if self.store is not None else None
            if (l1 is None or l1.scene_id != record.scene_id
                    or l1.episode_id != record.episode_id
                    or l1.rollout_idx != record.rollout_idx
                    or l1.step != record.step_id
                    or l1.observation_status != "observed"):
                linked_l1_id = None
        provenance = {
            "scene_id": record.scene_id,
            "episode_id": record.episode_id,
            "rollout_idx": record.rollout_idx,
            "step": record.step_id,
            "evidence_id": record.evidence_id,
            "source": record.source,
            "linked_l1_evidence_id": linked_l1_id,
            "geometry_status": record.geometry_status,
            "viewpoint_id": viewpoint.viewpoint_id,
        }
        return {
            "object_id": node.object_id,
            "label": node.class_name,
            "center_world": list(center),
            "bbox_3d": [list(part) for part in box],
            "observation_count": node.observation_count,
            "evidence_id": record.evidence_id,
            "evidence_step": record.step_id,
            "confidence": record.confidence,
            "source": record.source,
            "linked_l1_evidence_id": linked_l1_id,
            "viewpoint_position_world": list(viewpoint_position),
            "provenance": provenance,
        }
