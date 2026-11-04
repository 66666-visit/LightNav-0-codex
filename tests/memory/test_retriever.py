"""CPU-only retrieval tests using records from one synthetic episode."""

from __future__ import annotations

import unittest

from lightnav.memory.retriever import MemoryRetriever
from lightnav.memory.schemas import MemoryEvent, MemoryQuery, SceneEdge, SceneNode


def event(
    event_id: str,
    description: str,
    *,
    step: int,
    place: str | None = None,
    confidence: float = 0.8,
    importance: float = 0.5,
    kind: str = "landmark",
    source: str = "mock_observation",
    metadata: dict | None = None,
) -> MemoryEvent:
    return MemoryEvent(
        event_id=event_id,
        step_id=step,
        kind=kind,
        description=description,
        confidence=confidence,
        source=source,
        place_id=place,
        importance=importance,
        metadata=metadata or {},
    )


def object_node(node_id: str, label: str, *, step: int, confidence: float) -> SceneNode:
    return SceneNode(
        node_id=node_id,
        kind="object",
        label=label,
        confidence=confidence,
        source="ultralytics_yolo:yolo11n.pt",
        first_step=step,
        last_step=step,
        status="observed",
    )


def object_event(
    event_id: str,
    node_id: str,
    label: str,
    *,
    step: int,
    confidence: float,
) -> MemoryEvent:
    return event(
        event_id,
        f"Observed {label}",
        step=step,
        confidence=confidence,
        importance=0.6,
        kind="object_observation",
        source="ultralytics_yolo:yolo11n.pt",
        metadata={"node_id": node_id},
    )


class MemoryRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = MemoryRetriever()

    def test_instruction_words_outweigh_an_unrelated_recent_event(self) -> None:
        events = [
            event("sofa", "A red sofa was visible by the doorway", step=4),
            event("motion", "The robot moved forward", step=20),
        ]
        context = self.retriever.retrieve(
            MemoryQuery("find the red sofa", current_step=20, top_k=1, max_chars=300),
            events,
            [],
            [],
        )
        self.assertEqual([item.event_id for item in context.events], ["sofa"])
        self.assertIn("source=mock_observation", context.text)
        self.assertIn("confidence=0.80", context.text)

    def test_place_recency_confidence_and_importance_affect_order(self) -> None:
        events = [
            event("old", "Doorway", step=1, confidence=0.3, importance=0.1),
            event("recent", "Doorway", step=19, confidence=0.9, importance=0.9),
            event("place", "Doorway", step=2, place="place-1", confidence=0.3),
        ]
        query = MemoryQuery("doorway", current_step=20, place_id="place-1", top_k=3)
        context = self.retriever.retrieve(query, events, [], [])
        self.assertEqual([item.event_id for item in context.events], ["place", "recent", "old"])

    def test_top_k_and_character_budget_limit_returned_records(self) -> None:
        events = [event(str(i), f"Observed blue door number {i}", step=i) for i in range(6)]
        context = self.retriever.retrieve(
            MemoryQuery("blue door", current_step=6, top_k=2, max_chars=145),
            events,
            [],
            [],
        )
        self.assertLessEqual(len(context.events), 2)
        self.assertLessEqual(len(context.text), 145)
        self.assertGreater(len(context.events), 0)
        self.assertEqual(context.text.count("[event:"), len(context.events))

    def test_graph_nodes_and_supported_edges_are_retrievable(self) -> None:
        hall = SceneNode(
            node_id="hall",
            kind="place",
            label="hallway",
            confidence=0.8,
            source="mock_observation",
            first_step=1,
            last_step=4,
            visited=True,
        )
        kitchen = SceneNode(
            node_id="kitchen",
            kind="place",
            label="kitchen",
            confidence=0.9,
            source="mock_observation",
            first_step=5,
            last_step=8,
            visited=False,
        )
        edge = SceneEdge(
            edge_id="hall-kitchen",
            source_id="hall",
            target_id="kitchen",
            relation="connected_to",
            confidence=0.7,
            source="action_observation",
            step_id=8,
            evidence="robot passed through a doorway",
        )
        context = self.retriever.retrieve(
            MemoryQuery("go to the kitchen", current_step=9, top_k=3, max_chars=600),
            [],
            [hall, kitchen],
            [edge],
        )
        self.assertEqual(len(context.nodes), 2)
        self.assertEqual(len(context.edges), 1)
        self.assertIn("connected_to", context.text)
        self.assertIn("robot passed through a doorway", context.text)

    def test_zero_limits_and_future_records_return_empty_context(self) -> None:
        future = event("future", "sofa", step=11)
        context = self.retriever.retrieve(
            MemoryQuery("sofa", current_step=10, top_k=5), [future], [], []
        )
        self.assertEqual(context.text, "")
        self.assertEqual(context.events, [])
        empty = self.retriever.retrieve(
            MemoryQuery("sofa", current_step=10, top_k=0), [future], [], []
        )
        self.assertEqual(empty.events, [])
        no_records = self.retriever.retrieve(
            MemoryQuery("sofa", current_step=10), [], [], []
        )
        self.assertEqual(no_records.text, "")
        self.assertEqual(no_records.events, [])
        self.assertEqual(no_records.nodes, [])
        self.assertEqual(no_records.edges, [])

    def test_step_five_visual_observations_survive_action_history_with_turn(self) -> None:
        action_source = "issued_command_and_next_observation"
        actions = [
            event(
                f"action_{step}", "Issued velocity_control before this view",
                step=step, confidence=1.0, importance=0.4,
                kind="action", source=action_source,
            )
            for step in range(1, 5)
        ]
        turn = event(
            "turn_5", "Issued velocity control with angular command 0.30 before this view",
            step=5, confidence=1.0, importance=0.7,
            kind="turn", source=action_source,
        )
        detections = [
            ("object_0", "chair", 0.493),
            ("object_1", "chair", 0.330),
            ("object_2", "couch", 0.298),
        ]
        observations = [
            object_event(f"seen_{index}", node_id, label, step=5, confidence=confidence)
            for index, (node_id, label, confidence) in enumerate(detections)
        ]
        nodes = [
            object_node(node_id, label, step=5, confidence=confidence)
            for node_id, label, confidence in detections
        ]

        context = self.retriever.retrieve(
            MemoryQuery("continue along the hallway", current_step=5, top_k=5, max_chars=600),
            actions + [turn] + observations,
            nodes,
            [],
        )

        self.assertEqual(
            {item.metadata["node_id"] for item in context.events
             if item.kind == "object_observation"},
            {"object_0", "object_1", "object_2"},
        )
        self.assertEqual(context.text.count("Observed chair"), 2)
        self.assertEqual(context.text.count("Observed couch"), 1)
        self.assertIn(turn, context.events)
        self.assertLessEqual(sum(item.kind == "action" for item in context.events), 1)
        self.assertLessEqual(len(context.text), 600)

    def test_step_fifteen_retrieves_relevant_step_five_couch_after_actions(self) -> None:
        couch = object_event("seen_couch", "object_couch", "couch", step=5, confidence=0.72)
        actions = [
            event(
                f"action_{step}", "Issued velocity_control before this view",
                step=step, confidence=1.0, importance=0.4,
                kind="action", source="issued_command_and_next_observation",
            )
            for step in range(6, 15)
        ]
        context = self.retriever.retrieve(
            MemoryQuery("find the couch", current_step=15, top_k=5, max_chars=600),
            [couch] + actions,
            [object_node("object_couch", "couch", step=5, confidence=0.72)],
            [],
        )

        self.assertIn(couch, context.events)
        self.assertIn("object_observation step=5", context.text)
        self.assertLessEqual(sum(item.kind == "action" for item in context.events), 1)

    def test_one_detection_has_one_prompt_representation(self) -> None:
        observation = object_event("seen_chair", "object_0", "chair", step=5, confidence=0.8)
        instruction_match = event(
            "matched_chair", "Observed instruction-related chair",
            step=5, confidence=0.8, importance=0.95,
            kind="instruction_match", source="ultralytics_yolo:yolo11n.pt",
            metadata={"node_id": "object_0"},
        )
        node = object_node("object_0", "chair", step=5, confidence=0.8)
        events = [observation, instruction_match]
        nodes = [node]

        context = self.retriever.retrieve(
            MemoryQuery("find the chair", current_step=5, top_k=5, max_chars=600),
            events, nodes, [],
        )

        self.assertEqual(context.events, [observation])
        self.assertEqual(context.nodes, [])
        self.assertEqual(context.text.count("chair"), 1)
        self.assertEqual(events, [observation, instruction_match])
        self.assertEqual(nodes, [node])

    def test_same_name_with_different_node_ids_remains_two_detections(self) -> None:
        chairs = [
            object_event("seen_a", "object_a", "chair", step=5, confidence=0.8),
            object_event("seen_b", "object_b", "chair", step=5, confidence=0.7),
        ]
        nodes = [
            object_node("object_a", "chair", step=5, confidence=0.8),
            object_node("object_b", "chair", step=5, confidence=0.7),
        ]
        context = self.retriever.retrieve(
            MemoryQuery("find chairs", current_step=5, top_k=5, max_chars=600),
            chairs, nodes, [],
        )

        self.assertEqual(
            {item.metadata["node_id"] for item in context.events},
            {"object_a", "object_b"},
        )
        self.assertEqual(context.nodes, [])
        self.assertEqual(context.text.count("Observed chair"), 2)

    def test_character_limit_keeps_selected_records_complete(self) -> None:
        observations = [
            event(
                f"seen_{step}",
                "Observed chair near a doorway with a complete description",
                step=step,
            )
            for step in (9, 10)
        ]
        context = self.retriever.retrieve(
            MemoryQuery("chair doorway", current_step=10, top_k=2, max_chars=250),
            observations, [], [],
        )
        lines = context.text.splitlines()[1:]
        self.assertEqual(len(lines), len(context.events))
        self.assertTrue(lines)
        self.assertTrue(all(line.endswith("complete description") for line in lines))
        self.assertLessEqual(len(context.text), 250)


if __name__ == "__main__":
    unittest.main()
