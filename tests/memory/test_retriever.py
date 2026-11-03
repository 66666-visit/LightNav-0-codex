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
) -> MemoryEvent:
    return MemoryEvent(
        event_id=event_id,
        step_id=step,
        kind="landmark",
        description=description,
        confidence=confidence,
        source="mock_observation",
        place_id=place,
        importance=importance,
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


if __name__ == "__main__":
    unittest.main()
