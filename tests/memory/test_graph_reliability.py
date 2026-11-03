"""Regression checks for conservative scene-graph evidence handling."""

import unittest

from lightnav.memory.scene_graph import SceneGraph


class SceneGraphReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = SceneGraph()
        self.hall = self.graph.add_place(
            "hall", confidence=0.8, source="rgb_model", step_id=0,
            evidence="RGB place description at step 0",
        )
        self.room = self.graph.add_place(
            "room", confidence=0.8, source="rgb_model", step_id=3,
            evidence="RGB place description at step 3",
        )

    def test_same_name_does_not_identify_an_object_instance(self) -> None:
        first = self.graph.add_object(
            "chair", place_id=self.hall.node_id, confidence=0.7,
            source="rgb_model_a", step_id=1, evidence="first RGB detection",
        )
        second = self.graph.add_object(
            "chair", place_id=self.hall.node_id, confidence=0.9,
            source="rgb_model_b", step_id=2, evidence="second RGB detection",
        )
        self.assertNotEqual(first.node_id, second.node_id)
        self.assertEqual(self.graph.query_nodes(kind="object", place_id=self.hall.node_id), [first, second])
        self.assertEqual(len(self.graph.query_edges(relation="visible_from")), 2)
        self.assertEqual(self.graph.query_edges(relation="contains"), [])

    def test_stable_instance_key_is_scoped_to_one_place(self) -> None:
        first = self.graph.add_object(
            "chair", place_id=self.hall.node_id, confidence=0.6,
            source="rgb_model_a", step_id=1, evidence="first detection",
            instance_key="tracker-7",
        )
        repeated = self.graph.add_object(
            "red chair", place_id=self.hall.node_id, confidence=0.9,
            source="rgb_model_b", step_id=2, evidence="tracked second detection",
            instance_key="tracker-7",
        )
        elsewhere = self.graph.add_object(
            "chair", place_id=self.room.node_id, confidence=0.7,
            source="rgb_model_a", step_id=3, evidence="detection in room",
            instance_key="tracker-7",
        )
        self.assertIs(first, repeated)
        self.assertNotEqual(first.node_id, elsewhere.node_id)
        self.assertEqual(first.label, "red chair")
        self.assertEqual(first.source, "rgb_model_b")
        self.assertEqual(first.attributes["evidence"], "tracked second detection")
        self.assertEqual(len(self.graph.query_edges(relation="visible_from")), 2)

    def test_visual_evidence_is_not_containment(self) -> None:
        obj = self.graph.add_object(
            "chair", place_id=self.hall.node_id, confidence=0.8,
            source="rgb_model", step_id=1, evidence="visible in RGB",
        )
        with self.assertRaisesRegex(ValueError, "verified containment"):
            self.graph.add_edge(
                self.hall.node_id, obj.node_id, "contains", confidence=0.8,
                source="rgb_model", step_id=1, evidence="visible in RGB",
            )
        self.assertEqual(self.graph.query_edges(relation="contains"), [])

    def test_unverified_movement_cannot_create_connected_to(self) -> None:
        for source in ("issued_command_and_rgb_semantics", "executed_action_and_observation"):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "verified traversal"):
                    self.graph.add_edge(
                        self.hall.node_id, self.room.node_id, "connected_to",
                        confidence=0.7, source=source, step_id=3,
                        evidence="forward command followed by changed RGB histogram",
                    )
        self.assertEqual(self.graph.query_edges(relation="connected_to"), [])

    def test_possible_transition_is_distinct_from_verified_connection(self) -> None:
        uncertain = self.graph.add_edge(
            self.hall.node_id, self.room.node_id, "possible_transition",
            confidence=0.5, source="issued_command_and_rgb_semantics", step_id=3,
            evidence="forward command and changed RGB place description",
        )
        self.assertEqual(self.graph.query_edges(relation="possible_transition"), [uncertain])
        self.assertEqual(self.graph.query_edges(relation="connected_to"), [])
        verified = self.graph.add_edge(
            self.hall.node_id, self.room.node_id, "connected_to",
            confidence=0.9, source="verified_traversal", step_id=4,
            evidence="independent traversal verifier confirmed arrival",
        )
        self.assertEqual(self.graph.query_edges(relation="connected_to"), [verified])
        self.assertEqual(self.graph.query_edges(relation="possible_transition"), [uncertain])

    def test_node_confidence_keeps_its_source_and_evidence(self) -> None:
        obj = self.graph.add_object(
            "chair", place_id=self.hall.node_id, confidence=0.6,
            source="extractor_a", step_id=1, evidence="first detection",
            instance_key="stable-1",
        )
        self.graph.update_node(
            obj.node_id, confidence=0.9, source="extractor_b",
            evidence="stronger detection", step_id=2,
        )
        self.graph.update_node(
            obj.node_id, confidence=0.5, source="extractor_c",
            evidence="weaker detection", step_id=3,
        )
        self.assertEqual((obj.confidence, obj.source, obj.attributes["evidence"]),
                         (0.9, "extractor_b", "stronger detection"))
        self.assertEqual(obj.last_step, 3)
        with self.assertRaisesRegex(ValueError, "source and evidence"):
            self.graph.update_node(obj.node_id, confidence=1.0, visited=True)
        self.assertEqual((obj.confidence, obj.source, obj.visited), (0.9, "extractor_b", False))

    def test_edge_confidence_keeps_its_source_evidence_and_step(self) -> None:
        edge = self.graph.add_edge(
            self.hall.node_id, self.room.node_id, "possible_transition",
            confidence=0.5, source="extractor_a", step_id=1,
            evidence="first place description",
        )
        self.graph.add_edge(
            self.hall.node_id, self.room.node_id, "possible_transition",
            confidence=0.8, source="extractor_b", step_id=2,
            evidence="stronger place description",
        )
        self.graph.add_edge(
            self.hall.node_id, self.room.node_id, "possible_transition",
            confidence=0.4, source="extractor_c", step_id=3,
            evidence="weaker place description",
        )
        self.assertEqual(
            (edge.confidence, edge.source, edge.evidence, edge.step_id),
            (0.8, "extractor_b", "stronger place description", 2),
        )
        self.assertEqual(len(self.graph.query_edges(relation="possible_transition")), 1)


if __name__ == "__main__":
    unittest.main()
