"""Explicit episode scene graph with conservative place and object identity.

Matching labels alone never merge instances. A visible object is linked to an
evidenced place with ``visible_from``; otherwise it remains unanchored.
Traversable connections require verified evidence.
"""

from __future__ import annotations

from typing import Any

from lightnav.memory.schemas import SceneEdge, SceneNode


class SceneGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, SceneNode] = {}
        self._edges: dict[str, SceneEdge] = {}
        self._next_node = 0
        self._next_edge = 0

    @property
    def nodes(self) -> list[SceneNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[SceneEdge]:
        return list(self._edges.values())

    def get_node(self, node_id: str) -> SceneNode | None:
        return self._nodes.get(node_id)

    def add_place(
        self,
        label: str,
        *,
        confidence: float,
        source: str,
        step_id: int,
        evidence: str,
    ) -> SceneNode:
        """Create a distinct visited place hypothesis, even if its label repeats."""
        self._validate_observation(confidence, source, evidence)
        node = SceneNode(
            node_id=self._node_id("place"), kind="place", label=label,
            confidence=confidence, source=source, first_step=step_id,
            last_step=step_id, visited=True, status="uncertain",
            attributes={"evidence": evidence},
        )
        self._nodes[node.node_id] = node
        return node

    def add_object(
        self,
        label: str,
        *,
        place_id: str | None,
        confidence: float,
        source: str,
        step_id: int,
        evidence: str,
        instance_key: str | None = None,
    ) -> SceneNode:
        """Record a visual object, optionally linked to an evidenced place.

        A name is a description, not an instance identifier. In particular, two
        chairs visible from the same place must not collapse into one node. An
        object with no place hypothesis remains an unanchored graph node.
        """
        self._validate_observation(confidence, source, evidence)
        if place_id is not None:
            self._require_place(place_id)
        if not label.strip():
            raise ValueError("object label is required")
        if instance_key is not None:
            if not isinstance(instance_key, str) or not instance_key.strip():
                raise ValueError("instance_key must be a nonempty string")
        if instance_key is not None and place_id is not None:
            for edge in self.query_edges(relation="visible_from", source_id=place_id):
                existing = self._nodes[edge.target_id]
                if existing.attributes.get("instance_key") == instance_key:
                    replace_best = confidence >= existing.confidence
                    self.update_node(
                        existing.node_id, confidence=confidence, source=source,
                        evidence=evidence, step_id=step_id,
                    )
                    if replace_best:
                        existing.label = label
                    self.add_edge(
                        place_id, existing.node_id, "visible_from", confidence=confidence,
                        source=source, step_id=step_id, evidence=evidence,
                    )
                    return existing
        attributes = {"evidence": evidence}
        if instance_key is not None:
            attributes["instance_key"] = instance_key
        node = SceneNode(
            node_id=self._node_id("object"), kind="object", label=label,
            confidence=confidence, source=source, first_step=step_id,
            last_step=step_id, status="observed", attributes=attributes,
        )
        self._nodes[node.node_id] = node
        if place_id is not None:
            self.add_edge(
                place_id, node.node_id, "visible_from", confidence=confidence,
                source=source, step_id=step_id, evidence=evidence,
            )
        return node

    def update_node(
        self,
        node_id: str,
        *,
        confidence: float | None = None,
        source: str | None = None,
        evidence: str | None = None,
        visited: bool | None = None,
        status: str | None = None,
        step_id: int | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> SceneNode:
        node = self._nodes[node_id]
        # Validate the complete update before mutating any field.
        if confidence is not None:
            if source is None or evidence is None:
                raise ValueError("confidence updates require source and evidence")
            self._validate_observation(confidence, source, evidence)
        elif source is not None or evidence is not None:
            raise ValueError("source and evidence require confidence")
        if status is not None and status not in {"uncertain", "observed"}:
            raise ValueError(f"invalid status: {status}")
        if attributes and ("evidence" in attributes or "instance_key" in attributes):
            raise ValueError("evidence and instance_key require an observation update")
        if confidence is not None and confidence >= node.confidence:
            node.confidence = confidence
            node.source = source
            node.attributes["evidence"] = evidence
        if visited is not None:
            node.visited = node.visited or visited
        if status is not None:
            node.status = status
        if step_id is not None:
            node.last_step = max(node.last_step, step_id)
        if attributes:
            node.attributes.update(attributes)
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        *,
        confidence: float,
        source: str,
        step_id: int,
        evidence: str,
    ) -> SceneEdge:
        self._validate_observation(confidence, source, evidence)
        if source_id not in self._nodes or target_id not in self._nodes:
            raise KeyError("both edge endpoints must exist")
        source_node = self._nodes[source_id]
        target_node = self._nodes[target_id]
        if relation in {"visible_from", "contains"}:
            if source_node.kind != "place" or target_node.kind != "object":
                raise ValueError(f"{relation} must link a place to an object")
            if relation == "contains" and source != "verified_containment":
                raise ValueError("contains requires verified containment evidence")
        elif relation in {"possible_transition", "connected_to"}:
            if source_node.kind != "place" or target_node.kind != "place":
                raise ValueError(f"{relation} must link places")
            if source_id == target_id:
                raise ValueError("a place cannot transition to itself")
            if relation == "connected_to" and source != "verified_traversal":
                raise ValueError("connected_to requires verified traversal evidence")
        else:
            raise ValueError(f"unsupported relation: {relation}")
        for edge in self._edges.values():
            if (edge.source_id, edge.target_id, edge.relation) == (source_id, target_id, relation):
                # These fields describe one observation and must stay paired.
                if confidence >= edge.confidence:
                    edge.confidence = confidence
                    edge.source = source
                    edge.step_id = step_id
                    edge.evidence = evidence
                return edge
        edge = SceneEdge(
            edge_id=self._edge_id(), source_id=source_id, target_id=target_id,
            relation=relation, confidence=confidence, source=source,
            step_id=step_id, evidence=evidence,
        )
        self._edges[edge.edge_id] = edge
        return edge

    def query_nodes(
        self,
        *,
        kind: str | None = None,
        label: str | None = None,
        visited: bool | None = None,
        place_id: str | None = None,
    ) -> list[SceneNode]:
        visible = None
        if place_id is not None:
            visible = {
                edge.target_id for edge in self._edges.values()
                if edge.source_id == place_id and edge.relation in {"visible_from", "contains"}
            }
        return [
            node for node in self._nodes.values()
            if (kind is None or node.kind == kind)
            and (label is None or node.label.casefold() == label.casefold())
            and (visited is None or node.visited == visited)
            and (visible is None or node.node_id in visible)
        ]

    def query_edges(
        self,
        *,
        relation: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
    ) -> list[SceneEdge]:
        return [
            edge for edge in self._edges.values()
            if (relation is None or edge.relation == relation)
            and (source_id is None or edge.source_id == source_id)
            and (target_id is None or edge.target_id == target_id)
        ]

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._next_node = 0
        self._next_edge = 0

    def _node_id(self, kind: str) -> str:
        value = f"{kind}_{self._next_node}"
        self._next_node += 1
        return value

    def _edge_id(self) -> str:
        value = f"edge_{self._next_edge}"
        self._next_edge += 1
        return value

    def _require_place(self, place_id: str) -> SceneNode:
        place = self._nodes.get(place_id)
        if place is None or place.kind != "place":
            raise KeyError(f"unknown place: {place_id}")
        return place

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @classmethod
    def _validate_observation(cls, confidence: float, source: str, evidence: str) -> None:
        cls._validate_confidence(confidence)
        if not source or not evidence:
            raise ValueError("source and evidence are required")
