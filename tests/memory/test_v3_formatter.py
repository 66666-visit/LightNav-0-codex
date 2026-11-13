"""V3 text formatting keeps complete evidence-linked notes within the prompt budget."""

from __future__ import annotations

import json
import unittest

from lightnav.inference.token_diagnostics import memory_token_checks
from lightnav.memory.memory_context_builder import (
    audit_prompt_injection, build_context_text, inject_memory_context,
)
from lightnav.memory.v3_formatter import V3PromptFormatter
from lightnav.memory.v3_retrieval import V3Retriever
from lightnav.memory.v3_schemas import EvidenceRecord, RetrievedMemory, canonical_evidence_key
from lightnav.memory.v3_store import V3MemoryStore


def _record(evidence_id: str, description: str, *, kind: str = "observation",
            outcome: str = "unknown", traversability: str = "unknown") -> RetrievedMemory:
    ref = canonical_evidence_key("scene_a", "episode_a", 0, evidence_id)
    return RetrievedMemory(
        memory_id=ref, kind=kind, scene_id="scene_a", description=description,
        evidence_ids=(ref,), outcome=outcome, traversability_status=traversability,
        observation_status="observed", score=1 / 61, channel_ranks={"semantic": 1},
    )


class V3FormatterTests(unittest.TestCase):
    def test_past_experience_humanizes_action_without_changing_evidence(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene_a", "past", "Find the archway")
        store.add(EvidenceRecord(
            evidence_id="event_0", scene_id="scene_a", episode_id="past", rollout_idx=0,
            step=0, event_type="object_observation", semantic_label="archway",
            payload={"description": "Observed archway"}, observation_status="observed",
        ))
        raw_action = {"action": "velocity_control", "action_args": {
            "linear_velocity": 0.2, "angular_velocity": 0.0,
        }}
        original = "Issued velocity_control before this view"
        action = store.add(EvidenceRecord(
            evidence_id="event_1", scene_id="scene_a", episode_id="past", rollout_idx=0,
            step=1, event_type="action",
            payload={"description": original, "action": raw_action},
        ))
        store.close_episode(outcome="failure")
        store.begin_episode("scene_a", "now", "Find the archway")
        result = V3Retriever(store).retrieve(
            scene_id="scene_a", instruction="Find the archway", current_step=0,
            episode_id="now", top_k=5,
        )
        experience = next(record for record in result.records
                          if record.kind == "episodic_experience")
        self.assertIn("actions=1 (last: Issued velocity control before this view)",
                      experience.description)
        self.assertIn(action.key, experience.evidence_ids)
        stored = store.get_evidence(action.key)
        self.assertEqual(stored.payload["description"], original)
        self.assertEqual(stored.payload["action"], raw_action)
        context = V3PromptFormatter(max_chars=400).format([experience], current_step=0)
        self.assertIn("Issued velocity control before this view", context.text)
        self.assertIn("Outcome: failure", context.text)
        self.assertEqual(context.layer_items[0].metadata["supporting_evidence_ids"],
                         list(experience.evidence_ids))
        store.close_episode()

    def test_long_past_action_is_complete_or_omitted_at_budget_boundary(self) -> None:
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode("scene_a", "past", "Find the archway")
        store.add(EvidenceRecord(
            evidence_id="event_0", scene_id="scene_a", episode_id="past", rollout_idx=0,
            step=0, event_type="object_observation", semantic_label="archway",
            observation_status="observed",
        ))
        original = "Issued velocity_control before this view; " + "visual checkpoint " * 25
        action = store.add(EvidenceRecord(
            evidence_id="event_1", scene_id="scene_a", episode_id="past", rollout_idx=0,
            step=1, event_type="action", payload={"description": original},
        ))
        store.close_episode(outcome="unknown")
        store.begin_episode("scene_a", "now", "Find the archway")
        result = V3Retriever(store).retrieve(
            scene_id="scene_a", instruction="Find the archway", current_step=0,
            episode_id="now", top_k=5,
        )
        experience = next(record for record in result.records
                          if record.kind == "episodic_experience")
        rendered_action = " ".join(original.replace("velocity_control", "velocity control").split())
        self.assertIn(rendered_action, experience.description)
        self.assertNotIn("…", experience.description)
        self.assertEqual(store.get_evidence(action.key).payload["description"], original)

        wide = V3PromptFormatter(max_chars=1200).format([experience], current_step=0)
        exact_limit = len(build_context_text(wide, max_chars=1200))
        exact = V3PromptFormatter(max_chars=exact_limit).format([experience], current_step=0)
        too_small = V3PromptFormatter(max_chars=exact_limit - 1).format(
            [experience], current_step=0,
        )
        self.assertIn(rendered_action, build_context_text(exact, max_chars=exact_limit))
        self.assertEqual(too_small.text, "")
        self.assertEqual(too_small.layer_items, [])

        later = _record("event_later", "Observed nearby archway")
        default = V3PromptFormatter(max_chars=400).format(
            [experience, later], current_step=0,
        )
        self.assertEqual([item.metadata["memory_id"] for item in default.layer_items],
                         [later.memory_id])
        self.assertNotIn("visual checkpoint", default.text)
        store.close_episode()

    def test_store_to_retrieval_to_prompt_to_token_and_local_fallback(self) -> None:
        store = V3MemoryStore(scope="episode_local")
        store.begin_episode("scene_a", "first", "Find red archway")
        stored = store.add(EvidenceRecord(
            evidence_id="event_0", scene_id="scene_a", episode_id="first",
            rollout_idx=0, step=0, event_type="object_observation",
            semantic_label="archway", payload={"description": "Observed red archway"},
            raw_detector_confidence=0.24, observation_status="observed",
        ))
        retrieved = V3Retriever(store).retrieve(
            scene_id="scene_a", instruction="Find red archway", current_step=0,
            episode_id="first", top_k=3,
        )
        context = V3PromptFormatter().format(retrieved.records, current_step=0)
        base = {"video_segments": [], "conversations": [
            {"from": "human", "value": (
                "Task <navigation_task>Find red archway</navigation_task>. Continue."
            )},
            {"from": "gpt", "value": "placeholder"},
        ]}
        augmented = inject_memory_context(base, context)
        line = context.layer_items[0].metadata["serialized_text"]

        class Tokenizer:
            @staticmethod
            def encode(value, *, add_special_tokens=False):
                return [ord(char) for char in value]

            @staticmethod
            def decode(ids, *, skip_special_tokens=False):
                return "".join(chr(item) for item in ids)

        checks = memory_token_checks(
            Tokenizer(), Tokenizer.encode(
                augmented["conversations"][0]["value"] + "<|im_start|>assistant\n"
            ),
            {"memory_text": build_context_text(context),
             "original_instruction": "Find red archway",
             "accepted_records": [{"text": line, "evidence_ids": [stored.key]}]},
        )
        self.assertEqual(stored.raw_detector_confidence, 0.24)
        self.assertEqual(retrieved.final_top_k, 1)
        self.assertTrue(checks["memory_text_present"])
        self.assertTrue(checks["assistant_generation_boundary_present"])
        self.assertEqual(checks["literal_model_visible_evidence_ids"], [stored.key])
        self.assertNotEqual(augmented["conversations"], base["conversations"])
        store.close_episode()
        store.begin_episode("scene_a", "second", "Find red archway")
        empty = V3Retriever(store).retrieve(
            scene_id="scene_a", instruction="Find red archway", current_step=0,
            episode_id="second", top_k=3,
        )
        no_context = V3PromptFormatter().format(empty.records, current_step=0)
        self.assertEqual(inject_memory_context(base, no_context), base)
        store.close_episode()

    def test_real_context_builder_injection_keeps_instruction_and_provenance(self) -> None:
        record = _record("event_7", "Observed a low-confidence red door")
        context = V3PromptFormatter().format([record], current_step=5)
        sample = {
            "video_segments": [],
            "conversations": [
                {"from": "human", "value": (
                    "Video <video>. Task <navigation_task>Find the red door"
                    "</navigation_task>. Continue navigation."
                )},
                {"from": "gpt", "value": "placeholder"},
            ],
        }
        changed = inject_memory_context(sample, context, max_chars=400)
        audit = audit_prompt_injection(sample, changed, context, max_chars=400)
        self.assertNotEqual(audit["base_sha256"], audit["augmented_sha256"])
        self.assertIn("<navigation_task>Find the red door</navigation_task>",
                      changed["conversations"][0]["value"])
        self.assertIn(record.evidence_ids[0], changed["conversations"][0]["value"])
        self.assertEqual(context.layer_items[0].evidence_refs[0].event_id,
                         record.evidence_ids[0])
        self.assertIn(context.layer_items[0].metadata["serialized_text"],
                      changed["conversations"][0]["value"])

    def test_budget_drops_lower_ranked_whole_note(self) -> None:
        first = _record("event_1", "Observed door")
        second = _record("event_2", "Observed table")
        first_context = V3PromptFormatter().format([first], current_step=0)
        formatter = V3PromptFormatter(max_chars=len(first_context.text) + 1)
        context = formatter.format([first, second], current_step=0)
        self.assertEqual(context.text, first_context.text)
        self.assertEqual(len(context.layer_items), 1)
        self.assertEqual(formatter.last_serialized, 1)

    def test_realistic_composite_keys_stay_within_400_chars_and_three_records(self) -> None:
        scene = "mp3d/2t7WUuJeko7/2t7WUuJeko7.glb"
        episode = json.dumps([scene, "123"], separators=(",", ":"))
        store = V3MemoryStore(scope="scene_persistent")
        store.begin_episode(scene, episode, "Walk past the red archway and turn left")
        store.add(EvidenceRecord(
            evidence_id="event_0", scene_id=scene, episode_id=episode,
            rollout_idx=0, step=0, event_type="object_observation",
            semantic_label="archway", payload={"description": "Observed archway"},
            observation_status="observed", raw_detector_confidence=0.88,
        ))
        store.add(EvidenceRecord(
            evidence_id="event_1", scene_id=scene, episode_id=episode,
            rollout_idx=0, step=1, event_type="action",
            payload={"description": "Issued velocity control with angular command 0.30"},
        ))
        store.close_episode(outcome="failure")
        store.begin_episode(scene, "456", "Walk past the red archway and turn left")
        retrieved = V3Retriever(store).retrieve(
            scene_id=scene, instruction="Walk past the red archway and turn left",
            current_step=0, episode_id="456", top_k=5,
        )
        formatter = V3PromptFormatter(max_records=3, max_chars=400)
        context = formatter.format(retrieved.records, current_step=0)
        self.assertGreater(retrieved.final_top_k, 0)
        self.assertGreater(len(context.layer_items), 0)
        self.assertLessEqual(len(context.layer_items), 3)
        self.assertLessEqual(len(build_context_text(context, max_chars=400)), 400)
        self.assertEqual(len(formatter.last_serialized_line_chars), len(context.layer_items))
        self.assertTrue(all(length > 0 for length in formatter.last_candidate_line_chars))
        store.close_episode()

    def test_oversized_composite_key_skips_to_later_candidate(self) -> None:
        huge_key = canonical_evidence_key("scene_a", "x" * 420, 0, "event_0")
        short_key = canonical_evidence_key("scene_a", "episode_a", 0, "event_1")
        large = RetrievedMemory(
            memory_id="large", kind="observation", scene_id="scene_a",
            description="Observed archway", evidence_ids=(huge_key,),
            observation_status="observed",
        )
        short = RetrievedMemory(
            memory_id="short", kind="observation", scene_id="scene_a",
            description="Observed door", evidence_ids=(short_key,),
            observation_status="observed",
        )
        formatter = V3PromptFormatter(max_records=3, max_chars=400)
        context = formatter.format([large, short], current_step=0)
        self.assertEqual(len(context.layer_items), 1)
        self.assertEqual(context.layer_items[0].metadata["memory_id"], "short")
        self.assertEqual(len(formatter.last_candidate_line_chars), 2)
        self.assertGreater(formatter.last_candidate_line_chars[0], 400)
        self.assertLessEqual(len(build_context_text(context, max_chars=400)), 400)

    def test_current_observation_and_relevant_mid_fit_default_budget(self) -> None:
        scene = "mp3d/2t7WUuJeko7/2t7WUuJeko7.glb"
        store = V3MemoryStore(scope="episode_local")
        store.begin_episode(scene, "123", "Walk past the red archway and turn left")
        for event_id, step, kind, label, description in (
            ("event_0", 0, "object_observation", "archway", "Observed archway"),
            ("event_1", 1, "action", None,
             "Issued velocity control with angular command 0.30 before this view"),
            ("event_2", 2, "object_observation", "archway", "Observed archway"),
        ):
            store.add(EvidenceRecord(
                evidence_id=event_id, scene_id=scene, episode_id="123", rollout_idx=0,
                step=step, event_type=kind, semantic_label=label,
                payload={"description": description}, observation_status="observed",
                raw_detector_confidence=0.88 if label else None,
            ))
        result = V3Retriever(store).retrieve(
            scene_id=scene, instruction="Walk past the red archway and turn left",
            current_step=2, episode_id="123", top_k=5,
        )
        formatter = V3PromptFormatter(max_records=3, max_chars=400)
        context = formatter.format(result.records, current_step=2)
        self.assertTrue(any(item.kind == "v3_observation" for item in context.layer_items))
        self.assertTrue(any(item.kind == "v3_action_history" for item in context.layer_items))
        self.assertLessEqual(len(build_context_text(context, max_chars=400)), 400)
        self.assertEqual(len(context.layer_items), 2)
        store.close_episode()

    def test_failed_experience_and_unknown_shift_are_not_success_paths(self) -> None:
        failure = _record("event_3", "Left turn made no progress",
                          kind="episodic_experience", outcome="failure")
        shift = _record("event_4", "Visual shift after a command",
                        kind="observed_transition", traversability="unknown")
        context = V3PromptFormatter().format([failure, shift], current_step=10)
        self.assertIn("[EPISODIC EXPERIENCE]", context.text)
        self.assertIn("Outcome: failure", context.text)
        self.assertIn("Traversability: unknown", context.text)
        self.assertNotIn("[TRAVERSED]", context.text)

    def test_oversized_top_record_does_not_hide_shorter_valid_record(self) -> None:
        huge = _record("event_1", "x" * 400)
        short = _record("event_2", "Door observed")
        one = V3PromptFormatter().format([short], current_step=0)
        formatter = V3PromptFormatter(max_chars=len(one.text) + 1)
        result = formatter.format([huge, short], current_step=0)
        self.assertEqual(result.text, one.text)
        self.assertEqual(result.layer_items[0].evidence_refs[0].event_id,
                         short.evidence_ids[0])

    def test_only_printed_evidence_id_is_marked_model_visible(self) -> None:
        first = canonical_evidence_key("scene_a", "episode_a", 0, "event_1")
        second = canonical_evidence_key("scene_a", "episode_a", 0, "event_2")
        record = RetrievedMemory(
            memory_id="place:1", kind="place", scene_id="scene_a",
            description="Observed same visual place twice", evidence_ids=(first, second),
            observation_status="observed",
        )
        context = V3PromptFormatter().format([record], current_step=2)
        self.assertIn(second, context.text)
        self.assertEqual([ref.event_id for ref in context.layer_items[0].evidence_refs],
                         [second])
        self.assertEqual(context.layer_items[0].metadata["supporting_evidence_ids"],
                         [first, second])

    def test_injected_text_and_evidence_survive_a_reversible_tokenizer(self) -> None:
        record = _record("event_9", "Observed red archway")
        context = V3PromptFormatter().format([record], current_step=2)
        base = {"video_segments": [], "conversations": [
            {"from": "human", "value": (
                "Task <navigation_task>Find red archway</navigation_task>. Continue."
            )},
            {"from": "gpt", "value": "placeholder"},
        ]}
        changed = inject_memory_context(base, context, max_chars=400)
        text = changed["conversations"][0]["value"]

        class Tokenizer:
            @staticmethod
            def encode(value, *, add_special_tokens=False):
                return [ord(char) for char in value]

            @staticmethod
            def decode(ids, *, skip_special_tokens=False):
                return "".join(chr(item) for item in ids)

        line = context.layer_items[0].metadata["serialized_text"]
        checks = memory_token_checks(Tokenizer(), Tokenizer.encode(text), {
            "memory_text": build_context_text(context, max_chars=400),
            "original_instruction": "Find red archway",
            "accepted_records": [{"text": line, "evidence_ids": [record.evidence_ids[0]]}],
        })
        self.assertTrue(checks["memory_text_present"])
        self.assertTrue(checks["instruction_text_present"])
        self.assertEqual(checks["model_visible_evidence_ids"], [record.evidence_ids[0]])
        self.assertEqual(checks["literal_model_visible_evidence_ids"],
                         [record.evidence_ids[0]])


if __name__ == "__main__":
    unittest.main()
