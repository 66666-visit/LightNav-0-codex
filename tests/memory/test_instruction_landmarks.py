"""CPU contracts for the extractive instruction-landmark parser.

All four instructions below are the real Episode texts.  No Episode ID is
hard-coded in the parser; the tests only prove the general span/role rules on
the real wording.
"""

from __future__ import annotations

import pytest

from lightnav.memory.instruction_landmarks import parse_instruction_landmarks

INSTRUCTIONS = {
    487: (
        "Turn around and walk out of the room. "
        "Wait at the top of the stairs."
    ),
    1080: (
        "Go forward toward the windows. Go toward the the far couch, "
        "Stop next to the couch, in front of the windows."
    ),
    1012: (
        "walk away from sink, turn left, walk towards foot of bed, "
        "exit through doorway on left, turn right and step down "
        "two stairs and stop."
    ),
    168: (
        "Walk through double doors into the house. "
        "Continue around the dining table and through the entry way "
        "to the next room. Walk up to the couch and armchairs "
        "surrounding a coffee table."
    ),
}


def _landmarks(episode_id: int):
    instruction = INSTRUCTIONS[episode_id]
    parsed = parse_instruction_landmarks(instruction)
    assert parsed.instruction == instruction
    for landmark in parsed.landmarks:
        assert instruction[
            landmark.target_span[0]:landmark.target_span[1]
        ] == landmark.target_phrase
        assert instruction[
            landmark.phrase_span[0]:landmark.phrase_span[1]
        ] == landmark.grounding_query
        assert landmark.target_object in landmark.target_phrase
        if landmark.target_object_span is not None:
            assert instruction[
                landmark.target_object_span[0]:landmark.target_object_span[1]
            ] == landmark.target_object
        if landmark.action_phrase is not None:
            assert instruction[
                landmark.action_span[0]:landmark.action_span[1]
            ] == landmark.action_phrase
            assert landmark.action_phrase in instruction
        if landmark.reference_phrase is not None:
            assert landmark.reference_phrase in instruction
            assert instruction[
                landmark.reference_span[0]:landmark.reference_span[1]
            ] == landmark.reference_phrase
        for phrase, span in zip(
            landmark.reference_phrases, landmark.reference_spans,
        ):
            assert instruction[span[0]:span[1]] == phrase
    return parsed


def test_episode_487_keeps_the_stairs_and_drops_generic_room() -> None:
    parsed = _landmarks(487)
    assert [landmark.target_object for landmark in parsed.landmarks] == [
        "stairs",
    ]
    stairs = parsed.landmarks[0]
    assert stairs.role == "target"
    assert stairs.action_phrase is not None
    assert "Wait" in stairs.action_phrase
    assert "room" not in parsed.to_dict()["target_objects"]
    assert "out" not in parsed.to_dict()["target_objects"]


def test_episode_1080_keeps_couch_as_target_and_windows_as_reference() -> None:
    parsed = _landmarks(1080)
    assert parsed.to_dict()["target_objects"] == ["couch"]
    couch = next(
        landmark for landmark in parsed.landmarks
        if landmark.role == "target"
    )
    assert couch.target_object == "couch"
    assert couch.target_phrase == "far couch"
    assert couch.action_phrase == "Go toward"
    assert "windows" in couch.reference_objects
    assert "in front of the windows" in couch.reference_phrases
    assert "next to the couch" in couch.reference_phrases
    assert couch.grounding_query == (
        "far couch, Stop next to the couch, in front of the windows"
    )
    # windows is a waypoint/reference, never the couch query target.
    assert "windows" not in parsed.to_dict()["target_objects"]
    window = next(
        landmark for landmark in parsed.landmarks
        if landmark.role == "reference"
        and landmark.target_object == "windows"
    )
    assert window.relation_kind in {"waypoint_reference", "locative"}
    assert "couch" not in window.target_object


def test_episode_1012_preserves_away_towards_through_and_down() -> None:
    parsed = _landmarks(1012)
    assert [landmark.target_object for landmark in parsed.landmarks] == [
        "sink", "bed", "doorway", "stairs",
    ]
    sink, bed, doorway, stairs = parsed.landmarks
    assert sink.role == "reference"
    assert sink.relation_kind == "source"
    assert sink.action_phrase == "walk away from"
    assert sink.reference_phrase == "away from sink"
    assert bed.role == "target"
    assert bed.relation_kind == "destination_descriptor"
    assert bed.target_phrase == "foot of bed"
    assert bed.target_object == "bed"
    assert bed.action_phrase == "walk towards"
    assert doorway.target_object == "doorway"
    assert doorway.action_phrase == "exit through"
    assert stairs.target_object == "stairs"
    assert stairs.target_phrase == "two stairs"
    assert stairs.action_phrase == "step down"
    assert all(
        "up" not in (landmark.action_phrase or "").casefold().split()
        for landmark in parsed.landmarks
    )


def test_episode_168_keeps_all_six_objects_and_real_relations() -> None:
    parsed = _landmarks(168)
    assert parsed.to_dict()["target_objects"] == [
        "double doors", "dining table", "entry way", "couch", "armchairs",
    ]
    all_objects = [
        landmark.target_object for landmark in parsed.landmarks
    ]
    assert all_objects == [
        "double doors", "dining table", "entry way", "couch", "armchairs",
        "coffee table",
    ]
    armchairs = next(
        landmark for landmark in parsed.landmarks
        if landmark.target_object == "armchairs"
    )
    assert "coffee table" in armchairs.reference_objects
    assert armchairs.reference_phrase == "surrounding a coffee table"
    coffee = next(
        landmark for landmark in parsed.landmarks
        if landmark.target_object == "coffee table"
    )
    assert coffee.role == "reference"
    assert coffee.host_index == armchairs.index
    assert "by the dining table" not in INSTRUCTIONS[168]
    assert all(
        "by the dining table" not in (landmark.reference_phrase or "")
        for landmark in parsed.landmarks
    )


@pytest.mark.parametrize("episode_id", sorted(INSTRUCTIONS))
def test_every_extracted_phrase_is_verbatim(episode_id: int) -> None:
    instruction = INSTRUCTIONS[episode_id]
    parsed = parse_instruction_landmarks(instruction)
    for landmark in parsed.landmarks:
        assert landmark.target_phrase in instruction
        assert landmark.grounding_query in instruction
        assert landmark.target_object in landmark.target_phrase
        for reference in landmark.reference_objects:
            assert reference in instruction
            assert any(reference in phrase for phrase in landmark.reference_phrases)


def test_no_landmark_hardcoding_by_episode_id() -> None:
    # The parser has no Episode argument.  Reversing two independent clauses
    # changes order but not the relation rules.
    first = parse_instruction_landmarks(
        "Walk through double doors into the house. "
        "Walk up to the couch and armchairs surrounding a coffee table."
    )
    second = parse_instruction_landmarks(
        "Walk up to the couch and armchairs surrounding a coffee table. "
        "Walk through double doors into the house."
    )
    assert [item.target_object for item in first.landmarks] == [
        "double doors", "couch", "armchairs", "coffee table",
    ]
    assert [item.target_object for item in second.landmarks] == [
        "couch", "armchairs", "coffee table", "double doors",
    ]


@pytest.mark.parametrize(
    "instruction",
    ["find it", "go there", "continue", "", "   "],
)
def test_no_invented_landmark_when_the_instruction_has_no_object(
    instruction: str,
) -> None:
    parsed = parse_instruction_landmarks(instruction)
    assert parsed.landmarks == ()
    assert parsed.source == "no_landmark_fallback"
    assert parsed.target_landmarks == ()
    assert parsed.grounding_queries == ()


class _Provider:
    def __init__(self, items):
        self.items = items
        self.calls = 0

    def extract(self, instruction):
        self.calls += 1
        return self.items


def test_valid_provider_output_is_accepted_verbatim() -> None:
    instruction = INSTRUCTIONS[1080]
    provider = _Provider([
        {
            "target_phrase": "far couch",
            "reference_phrase": "in front of the windows",
        },
    ])
    parsed = parse_instruction_landmarks(instruction, provider=provider)
    assert provider.calls == 1
    assert parsed.source == "provider"
    assert [landmark.target_object for landmark in parsed.landmarks] == ["couch"]
    assert parsed.landmarks[0].reference_objects == ("windows",)
    query = parsed.landmarks[0].grounding_query
    assert query.startswith("far couch")
    assert query.endswith("in front of the windows")
    assert query in instruction


def test_invented_provider_phrase_falls_back_to_extraction() -> None:
    instruction = INSTRUCTIONS[1080]
    provider = _Provider([{"target_phrase": "green armchair"}])
    parsed = parse_instruction_landmarks(instruction, provider=provider)
    assert parsed.source == "provider_invalid_fallback"
    assert "green armchair" not in instruction
    assert parsed.to_dict()["target_objects"] == ["couch"]


class _RaisingProvider:
    def extract(self, instruction):
        raise RuntimeError("provider unavailable")


def test_provider_failure_falls_back_without_failing_the_episode() -> None:
    parsed = parse_instruction_landmarks(
        INSTRUCTIONS[1080], provider=_RaisingProvider(),
    )
    assert parsed.source == "provider_invalid_fallback"
    assert parsed.to_dict()["target_objects"] == ["couch"]
