"""Extractive, span-preserving navigation-instruction landmark parser.

LightNav-0 does not ship an instruction parser or a SceneQuery/DeepSeek/OpenAI-
compatible query provider (the repository has no such module).  This module is
therefore the *safe local fallback* requested for the ViL3DRel adapter:

* it only extracts text that already exists in the original instruction
  (every phrase is returned with an exact ``[start, end)`` span and is checked
  against the source string);
* it never invents an object, a label or a phrase;
* it is deterministic and has no external API, dependency, model or threshold;
* a caller may inject a provider (for example a future OpenAI-compatible
  instruction parser) through ``provider=``.  The provider output is validated
  against the original instruction; any error or invalid span falls back to the
  deterministic extractor instead of failing the navigation Episode.

The original navigation instruction is never rewritten.  Callers use the
extracted target phrase as the ViL3DRel query and still pass the complete
original instruction to LightNav.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, Sequence

_TOKEN_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*|[.,;:!?]", re.UNICODE)

#: Closed-class words.  They delimit noun chunks; they are not object names.
_DETERMINERS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "another", "each", "every", "both", "all", "its", "their", "his", "her",
    "our", "your", "my", "no",
    # Pronouns are not landmarks; "find it" must not become an object query.
    "i", "me", "you", "he", "she", "it", "we", "us", "they", "them",
    "something", "anything", "someone", "anyone", "somewhere", "here",
    "there",
})
_CONJUNCTIONS = frozenset({
    "and", "or", "then", "while", "when", "after", "before", "until", "once",
    "also", "but", "so", "as", "if", "because",
})
_VERBS = frozenset({
    "go", "goes", "going", "went", "walk", "walks", "walking", "head", "heads",
    "heading", "move", "moves", "moving", "turn", "turns", "turning", "proceed",
    "proceeds", "travel", "travels", "navigate", "navigates", "enter", "enters",
    "entering", "leave", "leaves", "leaving", "exit", "exits", "exiting",
    "approach", "approaches", "approaching", "reach", "reaches", "reaching",
    "stop", "stops", "stopping", "find", "finds", "finding", "locate",
    "locates", "locating", "get", "gets", "getting", "continue", "continues",
    "follow", "follows", "following", "pass", "passes", "passing", "cross",
    "crosses", "crossing", "climb", "climbs", "climbing", "take", "takes",
    "taking", "make", "makes", "making", "look", "looks", "looking", "face",
    "faces", "facing", "stand", "stands", "standing", "wait", "waits",
    "waiting", "arrive", "arrives", "arriving", "start", "starts", "starting",
    "run", "runs", "running", "drive", "drives", "driving", "step", "steps",
    "stepping", "keep", "keeps", "keeping", "come", "comes", "coming", "bring",
    "brings", "lead", "leads", "leading", "veer", "veers", "veering", "hike",
    "hikes", "hiking", "ascend", "ascends", "ascending", "descend", "descends",
    "descending", "push", "pushes", "pushing", "pull", "pulls", "pulling",
    "open", "opens", "opening", "close", "closes", "closing", "use", "uses",
    "using", "scoot", "scoots", "scooting",
})
#: Verbs that state/stop/observe rather than move toward a new destination.
#: They must not turn a locative reference into a second target.
_NON_MOTION_VERBS = frozenset({
    "stop", "stops", "stopping", "wait", "waits", "waiting", "find",
    "finds", "finding", "locate", "locates", "locating", "look", "looks",
    "looking", "face", "faces", "facing", "stand", "stands", "standing",
})
_MOTION_VERBS = _VERBS - _NON_MOTION_VERBS
#: Relation verbs ("armchairs surrounding a coffee table").
_RELATION_VERBS = frozenset({
    "surrounding", "surrounded", "surround", "surrounds",
})
#: Relation words introduce a reference object ("in front of the windows").
_RELATION_WORDS = frozenset({
    "in", "on", "at", "by", "near", "beside", "behind", "between", "above",
    "below", "under", "inside", "outside", "across", "with",
})
#: Source relations are references too, but their direction is away/out.
_SOURCE_PREPOSITIONS = frozenset({"from", "away"})
#: Path/destination words.  They are action context, not a reference.
_PATH_PREPOSITIONS = frozenset({
    "to", "toward", "towards", "into", "onto", "through", "past", "along",
    "around", "over", "up", "down", "off", "for", "of",
})
_PREPOSITIONS = _RELATION_WORDS | _PATH_PREPOSITIONS | _SOURCE_PREPOSITIONS
#: Spatial nouns/particles are part of a relation or direction, not an object.
_SPATIAL_NOUNS = frozenset({
    "left", "right", "front", "back", "top", "bottom", "side", "edge",
    "corner", "middle", "center", "centre", "end", "foot", "inside",
    "outside", "upstairs", "downstairs", "forward", "backward", "backwards",
    "ahead", "straight", "out", "away",
})
#: Generic regions are not object anchors; "next room" must not become "room".
_NON_OBJECT_NOUNS = frozenset({"room", "house", "home"})
#: Multi-word spatial relations, longest first at match time.  ``source``
#: phrases preserve away/out directions instead of reversing them.
_RELATION_PHRASES = (
    (("in", "the", "middle", "of"), "relation"),
    (("in", "the", "center", "of"), "relation"),
    (("in", "the", "centre", "of"), "relation"),
    (("on", "the", "left", "side", "of"), "relation"),
    (("on", "the", "right", "side", "of"), "relation"),
    (("to", "the", "left", "of"), "relation"),
    (("to", "the", "right", "of"), "relation"),
    (("at", "the", "foot", "of"), "relation"),
    (("at", "the", "top", "of"), "relation"),
    (("at", "the", "end", "of"), "relation"),
    (("in", "front", "of"), "relation"),
    (("in", "back", "of"), "relation"),
    (("on", "top", "of"), "relation"),
    (("next", "to"), "relation"),
    (("close", "to"), "relation"),
    (("across", "from"), "relation"),
    (("outside", "of"), "relation"),
    (("inside", "of"), "relation"),
    (("left", "of"), "relation"),
    (("right", "of"), "relation"),
    (("foot", "of"), "relation"),
    (("top", "of"), "relation"),
    (("side", "of"), "relation"),
    (("edge", "of"), "relation"),
    (("corner", "of"), "relation"),
    (("middle", "of"), "relation"),
    (("center", "of"), "relation"),
    (("centre", "of"), "relation"),
    (("end", "of"), "relation"),
    (("back", "of"), "relation"),
    (("front", "of"), "relation"),
    (("away", "from"), "source"),
    (("out", "of"), "source"),
)
#: Direction/distance and number modifiers are not part of the object name.
_OBJECT_MODIFIERS = frozenset({
    "far", "near", "nearby", "distant", "left", "right", "front", "back",
    "rear", "top", "bottom", "upper", "lower", "middle", "center", "central",
    "adjacent", "opposite", "closest", "nearest", "farther", "furthest",
    "first", "second", "third", "fourth", "fifth", "next", "last", "same",
    "other", "another", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
})
_SENTENCE_END = frozenset({".", ";", "!", "?"})


class LandmarkQueryProvider(Protocol):
    """Optional instruction-landmark provider.

    The method must return mappings with ``target_phrase`` and optionally
    ``reference_phrase``.  Every phrase must be an exact substring of the
    original instruction; otherwise the deterministic extractor is used.
    """

    def extract(self, instruction: str) -> Sequence[Mapping[str, Any]]:
        """Return ordered instruction landmarks."""


@dataclass(frozen=True)
class InstructionLandmark:
    """One instruction landmark, all phrases exact substrings of the source."""

    index: int
    role: str  # "target" | "reference"
    target_phrase: str
    target_object: str
    target_span: tuple[int, int]
    action_phrase: str | None
    action_span: tuple[int, int] | None
    reference_phrase: str | None
    reference_objects: tuple[str, ...]
    reference_span: tuple[int, int] | None
    phrase_span: tuple[int, int]
    grounding_query: str
    host_index: int | None = None
    relation_kind: str | None = None
    target_object_span: tuple[int, int] | None = None
    grounding_query_span: tuple[int, int] | None = None
    reference_phrases: tuple[str, ...] = ()
    reference_spans: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "role": self.role,
            "target_phrase": self.target_phrase,
            "target_object": self.target_object,
            "target_span": list(self.target_span),
            "target_object_span": (
                list(self.target_object_span) if self.target_object_span else None
            ),
            "action_phrase": self.action_phrase,
            "action_span": list(self.action_span) if self.action_span else None,
            "reference_phrase": self.reference_phrase,
            "reference_phrases": list(self.reference_phrases),
            "reference_objects": list(self.reference_objects),
            "reference_span": (
                list(self.reference_span) if self.reference_span else None
            ),
            "reference_spans": [list(span) for span in self.reference_spans],
            "phrase_span": list(self.phrase_span),
            "grounding_query": self.grounding_query,
            "grounding_query_span": (
                list(self.grounding_query_span)
                if self.grounding_query_span else None
            ),
            "host_index": self.host_index,
            "relation_kind": self.relation_kind,
        }


@dataclass(frozen=True)
class InstructionLandmarks:
    """Cached parse result for one Episode instruction."""

    instruction: str
    landmarks: tuple[InstructionLandmark, ...]
    source: str
    parser: str = "lightnav.memory.instruction_landmarks.extractive_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "source": self.source,
            "parser": self.parser,
            "landmarks": [landmark.to_dict() for landmark in self.landmarks],
            "target_objects": [
                landmark.target_object
                for landmark in self.landmarks
                if landmark.role == "target"
            ],
            "reference_objects": list(dict.fromkeys(
                value
                for landmark in self.landmarks
                for value in landmark.reference_objects
            )),
        }

    @property
    def target_landmarks(self) -> tuple[InstructionLandmark, ...]:
        return tuple(
            landmark for landmark in self.landmarks if landmark.role == "target"
        )

    @property
    def grounding_queries(self) -> tuple[InstructionLandmark, ...]:
        """Every extracted landmark is queried once, in original instruction order."""
        return self.landmarks


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int
    kind: str  # content | determiner | conjunction | verb | preposition | relation | spatial | punct


_PHRASES_LONGEST_FIRST = tuple(sorted(
    _RELATION_PHRASES, key=lambda item: len(item[0]), reverse=True,
))


def _classify(token: str) -> str:
    value = token.casefold()
    if token in _SENTENCE_END or token in {",", ":"}:
        return "punct"
    if value in _DETERMINERS:
        return "determiner"
    if value in _CONJUNCTIONS:
        return "conjunction"
    if value in _VERBS:
        return "verb"
    if value in _RELATION_VERBS:
        return "relation"
    if value in _SOURCE_PREPOSITIONS:
        return "source"
    if value in _RELATION_WORDS:
        return "relation"
    if value in _PREPOSITIONS:
        return "preposition"
    if value in _SPATIAL_NOUNS:
        return "spatial"
    return "content"


def _raw_tokens(instruction: str) -> list[_Token]:
    return [
        _Token(match.group(0), match.start(), match.end(), _classify(match.group(0)))
        for match in _TOKEN_RE.finditer(instruction)
    ]


def _merge_phrases(instruction: str, raw: list[_Token]) -> list[_Token]:
    """Merge multi-word relations ("in front of") into one relation token."""
    merged: list[_Token] = []
    index = 0
    while index < len(raw):
        match: tuple[tuple[str, ...], str] | None = None
        for words, kind in _PHRASES_LONGEST_FIRST:
            if index + len(words) > len(raw):
                continue
            if all(
                raw[index + offset].text.casefold() == word
                for offset, word in enumerate(words)
            ):
                match = (words, kind)
                break
        if match is None:
            merged.append(raw[index])
            index += 1
            continue
        words, kind = match
        first = raw[index]
        last = raw[index + len(words) - 1]
        merged.append(_Token(
            instruction[first.start:last.end], first.start, last.end, kind,
        ))
        index += len(words)
    return merged


def _content_runs(tokens: list[_Token]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(tokens):
        if tokens[index].kind != "content":
            index += 1
            continue
        end = index
        while end + 1 < len(tokens) and tokens[end + 1].kind == "content":
            end += 1
        runs.append((index, end))
        index = end + 1
    return runs


def _object_from_run(
    instruction: str, tokens: list[_Token], run: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    """Drop only leading direction/distance/number modifiers from the phrase."""
    first, last = run
    start = first
    while start < last and (
        tokens[start].text.casefold() in _OBJECT_MODIFIERS
        or tokens[start].text.isdigit()
    ):
        start += 1
    if start >= last and tokens[start].text.casefold() in _OBJECT_MODIFIERS:
        start = first
    return (
        instruction[tokens[start].start:tokens[last].end],
        (tokens[start].start, tokens[last].end),
    )


def _clause_break_between(tokens: list[_Token], start: int, end: int) -> bool:
    for index in range(start, end):
        token = tokens[index]
        if token.kind == "punct" and token.text in _SENTENCE_END:
            return True
    return False


def _action_phrase(
    instruction: str, tokens: list[_Token], run_start: int, boundary: int,
) -> tuple[str, tuple[int, int]] | None:
    # ``boundary`` is a token index.  A new sentence resets the action span so
    # a verb from the previous clause is never attached to this landmark.
    for index in range(boundary, run_start):
        if tokens[index].kind == "punct" and tokens[index].text in _SENTENCE_END:
            boundary = index + 1
    candidates = [
        index for index in range(boundary, run_start)
        if tokens[index].kind in {"verb", "preposition", "relation"}
    ]
    if not candidates:
        return None
    # The action is the last motion word before the landmark ("... on the left,
    # and up the two stairs" -> "up"), not every function word in the gap.
    motion = [
        index for index in candidates
        if tokens[index].kind == "preposition"
        or (
            tokens[index].kind == "verb"
            and tokens[index].text.casefold() in _MOTION_VERBS
        )
    ]
    chosen = motion[-1] if motion else candidates[-1]
    # Keep the motion verb even when a direction particle sits between it and
    # the final preposition ("Go forward toward the windows").
    start = chosen
    for index in range(chosen - 1, boundary - 1, -1):
        if (
            tokens[index].kind == "verb"
            and tokens[index].text.casefold() in _MOTION_VERBS
        ):
            start = index
            break
    else:
        while (
            start - 1 >= boundary
            and tokens[start - 1].kind in {"verb", "preposition"}
        ):
            start -= 1
    phrase = instruction[tokens[start].start:tokens[chosen].end].strip()
    if not phrase:
        return None
    lead = len(instruction[tokens[start].start:tokens[chosen].end]) - len(
        instruction[tokens[start].start:tokens[chosen].end].lstrip()
    )
    return phrase, (tokens[start].start + lead, tokens[chosen].end)


def _governing(
    tokens: list[_Token], run_start: int,
) -> tuple[int | None, _Token | None, bool]:
    """Return (token index, governing token, sentence break) before a run."""
    for index in range(run_start - 1, -1, -1):
        token = tokens[index]
        if token.kind == "determiner":
            continue
        if token.kind == "punct":
            if token.text in _SENTENCE_END:
                return None, None, True
            continue
        if token.kind in {"content", "spatial"}:
            return None, None, False
        return index, token, False
    return None, None, False


def _is_relation(token: _Token | None) -> bool:
    return bool(
        token is not None
        and (
            token.kind in {"relation", "source"}
            or (
                token.kind == "preposition"
                and token.text.casefold() in _RELATION_WORDS
            )
        )
    )


def _starts_new_destination(tokens: list[_Token], relation_index: int) -> bool:
    """True when a motion phrase (walk towards ...) introduces this relation."""
    for index in range(relation_index - 1, -1, -1):
        token = tokens[index]
        if token.kind == "determiner":
            continue
        if token.kind == "punct":
            if token.text in _SENTENCE_END:
                return False
            continue
        if token.kind == "verb":
            return token.text.casefold() in _MOTION_VERBS
        if token.kind == "preposition":
            return token.text.casefold() in _PATH_PREPOSITIONS
        return False
    return False


def _source_action_phrase(
    instruction: str, tokens: list[_Token], relation_index: int, boundary: int,
) -> tuple[str, tuple[int, int]] | None:
    """Keep the source direction verbatim: walk away from / walk out of."""
    motion = None
    for index in range(relation_index - 1, boundary - 1, -1):
        if (
            tokens[index].kind == "verb"
            and tokens[index].text.casefold() in _MOTION_VERBS
        ):
            motion = index
            break
    if motion is None:
        return None
    start = motion
    while start - 1 >= boundary and tokens[start - 1].kind in {"verb", "preposition"}:
        start -= 1
    phrase = instruction[tokens[start].start:tokens[relation_index].end].strip()
    if not phrase:
        return None
    return phrase, (tokens[start].start, tokens[relation_index].end)


def _host_for_reference(
    tokens: list[_Token], records: list[dict[str, Any]], relation_start: int,
) -> int | None:
    for candidate in reversed(records):
        candidate_end = candidate["target_span"][1]
        if candidate_end > relation_start:
            continue
        if _clause_break_between(
            tokens,
            _index_at_or_after(tokens, candidate_end),
            _index_at_or_before(tokens, relation_start),
        ):
            continue
        if (
            candidate["role"] == "reference"
            and candidate.get("host_index") is not None
        ):
            return candidate["host_index"]
        return candidate["index"]
    return None


def _make_record(
    *, index: int, role: str, target_phrase: str, target_span: tuple[int, int],
    target_object: str, target_object_span: tuple[int, int],
    action: tuple[str, tuple[int, int]] | None,
    reference_phrase: str | None, reference_span: tuple[int, int] | None,
    reference_objects: tuple[str, ...], host_index: int | None,
    relation_kind: str | None,
) -> dict[str, Any]:
    return {
        "index": index,
        "role": role,
        "target_phrase": target_phrase,
        "target_object": target_object,
        "target_span": target_span,
        "target_object_span": target_object_span,
        "action_phrase": action[0] if action else None,
        "action_span": action[1] if action else None,
        "reference_phrase": reference_phrase,
        "reference_phrases": (
            (reference_phrase,) if reference_phrase is not None else ()
        ),
        "reference_objects": reference_objects,
        "reference_span": reference_span,
        "reference_spans": (reference_span,) if reference_span else (),
        "phrase_span": target_span,
        "grounding_query": target_phrase,
        "grounding_query_span": target_span,
        "host_index": host_index,
        "relation_kind": relation_kind,
    }


def _build_landmarks(instruction: str, source: str) -> InstructionLandmarks:
    tokens = _merge_phrases(instruction, _raw_tokens(instruction))
    runs = _content_runs(tokens)
    if not runs:
        return InstructionLandmarks(instruction, (), "no_landmark_fallback")

    records: list[dict[str, Any]] = []
    previous_run_token_end = 0
    for run in runs:
        run_start, run_end = run
        run_phrase = instruction[tokens[run_start].start:tokens[run_end].end]
        target_object, object_span = _object_from_run(instruction, tokens, run)
        if target_object.casefold() in _NON_OBJECT_NOUNS:
            previous_run_token_end = run_end + 1
            continue
        governing_index, governing, broken = _governing(tokens, run_start)
        action = _action_phrase(
            instruction, tokens, run_start, previous_run_token_end,
        )
        role = "target"
        reference_phrase = None
        reference_span = None
        reference_objects: tuple[str, ...] = ()
        host_index = None
        relation_kind = None
        target_phrase = run_phrase
        target_span = (tokens[run_start].start, tokens[run_end].end)

        if not broken and _is_relation(governing):
            assert governing_index is not None and governing is not None
            if governing.kind == "source":
                # "walk away from sink": sink is a reference, never reversed
                # into a destination, and keeps the away/out wording.
                role = "reference"
                relation_kind = "source"
                reference_phrase = instruction[
                    governing.start:tokens[run_end].end
                ]
                reference_span = (
                    governing.start, tokens[run_end].end,
                )
                reference_objects = (target_object,)
                action = _source_action_phrase(
                    instruction, tokens, governing_index, previous_run_token_end,
                )
            elif _starts_new_destination(tokens, governing_index):
                # "walk towards foot of bed": the relation phrase is the
                # destination descriptor, not a reference of the previous
                # landmark.  Keep "foot of" inside the target phrase.
                target_phrase = instruction[
                    governing.start:tokens[run_end].end
                ]
                target_span = (governing.start, tokens[run_end].end)
                relation_kind = "destination_descriptor"
                action = _action_phrase(
                    instruction, tokens, governing_index,
                    previous_run_token_end,
                )
            else:
                host_index = _host_for_reference(
                    tokens, records, governing.start,
                )
                if host_index is not None:
                    role = "reference"
                    relation_kind = "locative"
                    reference_phrase = instruction[
                        governing.start:tokens[run_end].end
                    ]
                    reference_span = (
                        governing.start, tokens[run_end].end,
                    )
                    reference_objects = (target_object,)
                    action = None
                    # A co-referring mention ("next to the couch" after "far
                    # couch") is context for the host, not a second instance
                    # query.  Different reference objects still get a query.
                    host = records[host_index]
                    if (
                        host["target_object"].casefold()
                        == target_object.casefold()
                    ):
                        host.setdefault("_references", []).append({
                            "reference_phrase": reference_phrase,
                            "reference_span": reference_span,
                            "reference_objects": reference_objects,
                        })
                        previous_run_token_end = run_end + 1
                        continue

        record = _make_record(
            index=len(records), role=role, target_phrase=target_phrase,
            target_span=target_span, target_object=target_object,
            target_object_span=object_span, action=action,
            reference_phrase=reference_phrase, reference_span=reference_span,
            reference_objects=reference_objects, host_index=host_index,
            relation_kind=relation_kind,
        )
        records.append(record)
        if role == "reference" and host_index is not None:
            records[host_index].setdefault("_references", []).append({
                "reference_phrase": record["reference_phrase"],
                "reference_span": record["reference_span"],
                "reference_objects": record["reference_objects"],
            })
        previous_run_token_end = run_end + 1

    for record in records:
        references = record.pop("_references", [])
        if not references:
            continue
        record["reference_phrases"] = tuple(
            reference["reference_phrase"] for reference in references
        )
        record["reference_spans"] = tuple(
            reference["reference_span"] for reference in references
        )
        record["reference_objects"] = tuple(
            value
            for reference in references
            for value in reference["reference_objects"]
        )
        record["reference_phrase"] = references[0]["reference_phrase"]
        record["reference_span"] = references[0]["reference_span"]
        # Keep the complete original span from the target through the last
        # reference, including punctuation and any intervening clause.
        end = max(reference["reference_span"][1] for reference in references)
        start = record["target_span"][0]
        record["phrase_span"] = (start, end)
        record["grounding_query_span"] = (start, end)
        record["grounding_query"] = instruction[start:end]

    # A destination that a later target explicitly references is a waypoint
    # anchor, not the final target: "Go toward the windows ... far couch, in
    # front of the windows" must keep couch as the target object.  This uses
    # only exact co-reference, never a new relation.
    for record in records:
        if record["role"] != "target":
            continue
        for other in records:
            if (
                other["role"] != "target"
                or other["index"] <= record["index"]
                or other["target_object"].casefold()
                == record["target_object"].casefold()
            ):
                continue
            match = next(
                (
                    (phrase, span, object_id)
                    for phrase, span, object_id in zip(
                        other["reference_phrases"],
                        other["reference_spans"],
                        other["reference_objects"],
                    )
                    if object_id.casefold()
                    == record["target_object"].casefold()
                ),
                None,
            )
            if match is None:
                continue
            record["role"] = "reference"
            record["relation_kind"] = "waypoint_reference"
            record["host_index"] = other["index"]
            record["reference_phrase"] = match[0]
            record["reference_span"] = match[1]
            record["reference_phrases"] = (match[0],)
            record["reference_spans"] = (match[1],)
            record["reference_objects"] = (match[2],)
            break

    landmarks = tuple(
        InstructionLandmark(**{
            key: value for key, value in record.items()
            if not key.startswith("_")
        })
        for record in records
    )
    return InstructionLandmarks(instruction, landmarks, source)


def _index_at_or_after(tokens: list[_Token], position: int) -> int:
    for index, token in enumerate(tokens):
        if token.start >= position:
            return index
    return len(tokens)


def _index_at_or_before(tokens: list[_Token], position: int) -> int:
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index].end <= position:
            return index + 1
    return 0


def _provider_landmarks(
    instruction: str, provider: LandmarkQueryProvider,
) -> InstructionLandmarks | None:
    """Validate a provider result; return None when any phrase is not verbatim."""
    try:
        items = provider.extract(instruction)
    except Exception:  # noqa: BLE001 - a provider must never break navigation
        return None
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return None
    cursor = 0
    landmarks: list[InstructionLandmark] = []
    for item in items:
        if not isinstance(item, Mapping):
            return None
        target_phrase = item.get("target_phrase")
        if not isinstance(target_phrase, str) or not target_phrase.strip():
            return None
        target_start = instruction.find(target_phrase, cursor)
        if target_start < 0:
            return None
        target_end = target_start + len(target_phrase)
        reference_phrase = item.get("reference_phrase")
        phrase_end = target_end
        reference_span = None
        reference_objects: tuple[str, ...] = ()
        if reference_phrase is not None:
            if (
                not isinstance(reference_phrase, str)
                or not reference_phrase.strip()
                or instruction.find(reference_phrase, target_end) < 0
            ):
                return None
            reference_start = instruction.find(reference_phrase, target_end)
            phrase_end = reference_start + len(reference_phrase)
            reference_span = (reference_start, phrase_end)
            reference_tokens = _raw_tokens(reference_phrase)
            content = [
                token for token in reference_tokens if token.kind == "content"
            ]
            if content:
                reference_objects = (
                    " ".join(token.text for token in content),
                )
        target_tokens = _raw_tokens(target_phrase)
        object_tokens = [
            token for token in target_tokens if token.kind == "content"
        ]
        target_object = (
            " ".join(token.text for token in object_tokens)
            if object_tokens else target_phrase
        )
        merged_target = _merge_phrases(target_phrase, target_tokens)
        target_runs = _content_runs(merged_target)
        if target_runs:
            target_object, _ = _object_from_run(
                target_phrase, merged_target, target_runs[0],
            )
        landmarks.append(InstructionLandmark(
            index=len(landmarks),
            role="target",
            target_phrase=target_phrase,
            target_object=target_object,
            target_span=(target_start, target_end),
            action_phrase=None,
            action_span=None,
            reference_phrase=reference_phrase if isinstance(reference_phrase, str) else None,
            reference_objects=reference_objects,
            reference_span=reference_span,
            phrase_span=(target_start, phrase_end),
            grounding_query=instruction[target_start:phrase_end],
            host_index=None,
        ))
        cursor = phrase_end
    if not landmarks:
        return None
    return InstructionLandmarks(instruction, tuple(landmarks), "provider")


def parse_instruction_landmarks(
    instruction: str, *, provider: LandmarkQueryProvider | None = None,
) -> InstructionLandmarks:
    """Parse one instruction into ordered, verbatim instruction landmarks.

    The deterministic extractor is always a safe fallback.  This function never
    raises for an arbitrary instruction string; a caller can therefore cache the
    result once at Episode initialization without risking the navigation run.
    """
    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    if not instruction.strip():
        return InstructionLandmarks(instruction, (), "no_landmark_fallback")
    if provider is not None:
        provided = _provider_landmarks(instruction, provider)
        if provided is not None:
            return provided
        parsed = _build_landmarks(instruction, "provider_invalid_fallback")
    else:
        parsed = _build_landmarks(instruction, "extractive")
    if not parsed.landmarks:
        return replace(parsed, source="no_landmark_fallback")
    return parsed


__all__ = [
    "InstructionLandmark",
    "InstructionLandmarks",
    "LandmarkQueryProvider",
    "parse_instruction_landmarks",
]
