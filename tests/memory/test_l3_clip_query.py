"""CPU contracts for the local L3 CLIP Top-K query (mock features only)."""

from __future__ import annotations

import numpy as np
import pytest

from lightnav.memory.conceptgraphs_lite import ObjectMap, ObjectNode, SpatialEvidence
from lightnav.memory.l3_clip_query import L3ClipQuery, query_objects


class _TextEncoder:
    feature_dim = 2

    def __init__(self, feature=(1.0, 0.0)) -> None:
        self.feature = np.asarray(feature, dtype=np.float32)
        self.text_calls = 0

    def encode_image(self, rgb, boxes):
        raise AssertionError("query must not re-encode object images")

    def encode_text(self, texts):
        self.text_calls += 1
        return np.asarray([self.feature], dtype=np.float32)


def _add_object(
    object_map: ObjectMap, object_id: str, label: str, feature, *,
    center=(1.0, 0.5, -2.0), step: int = 0, count: int = 2,
) -> ObjectNode:
    evidence_id = f"ev-{object_id}"
    node = ObjectNode(
        object_id, label, center,
        ((center[0] - 0.5, center[1] - 0.5, center[2] - 0.5),
         (center[0] + 0.5, center[1] + 0.5, center[2] + 0.5)),
        count, [evidence_id], "observed",
        np.zeros((4, 3), dtype=np.float64),
    )
    object_map.objects[object_id] = node
    object_map.evidence[evidence_id] = SpatialEvidence(
        evidence_id, "scene", "episode", 0, step, label, 0.9,
        "yolo11n-seg-clip-test", 12, 4, "observed",
    )
    if feature is not None:
        object_map.attach_clip_features({evidence_id: feature})
    return node


def _three_objects() -> ObjectMap:
    object_map = ObjectMap()
    _add_object(object_map, "a-chair", "chair", [1.0, 0.0])
    _add_object(object_map, "b-sofa", "sofa", [0.6, 0.8], center=(3.0, 0.5, -4.0))
    _add_object(object_map, "c-table", "table", [0.0, 1.0], center=(-1.0, 0.5, -1.0))
    return object_map


def test_top_k_orders_by_cosine_similarity() -> None:
    result = query_objects(
        _three_objects(), "seat", 2, encoder=_TextEncoder([1.0, 0.0]),
    )
    assert [match.object_id for match in result.matches] == ["a-chair", "b-sofa"]
    assert [round(match.similarity, 3) for match in result.matches] == [1.0, 0.6]
    assert result.skipped_reason is None
    assert result.objects_total == 3
    assert result.objects_with_features == 3


def test_top_k_is_capped_by_available_objects() -> None:
    result = query_objects(
        _three_objects(), "seat", 99, encoder=_TextEncoder([1.0, 0.2]),
    )
    assert len(result.matches) == 3
    assert result.top_k == 99


def test_equal_similarities_break_ties_by_object_id() -> None:
    object_map = ObjectMap()
    _add_object(object_map, "z-object", "chair", [1.0, 0.0])
    _add_object(object_map, "a-object", "chair", [1.0, 0.0])
    result = L3ClipQuery(object_map, _TextEncoder([1.0, 0.0])).query("chair", 2)
    assert [match.object_id for match in result.matches] == ["a-object", "z-object"]


def test_min_similarity_filters_weak_matches() -> None:
    result = L3ClipQuery(
        _three_objects(), _TextEncoder([0.0, 1.0]),
    ).query("table", 3, min_similarity=0.9)
    assert [match.object_id for match in result.matches] == ["c-table"]


def test_empty_query_returns_reason_without_encoding() -> None:
    encoder = _TextEncoder()
    result = L3ClipQuery(_three_objects(), encoder).query("   ", 3)
    assert result.skipped_reason == "empty_query"
    assert result.matches == ()
    assert encoder.text_calls == 0


def test_empty_map_returns_reason_without_encoding() -> None:
    encoder = _TextEncoder()
    result = L3ClipQuery(ObjectMap(), encoder).query("chair", 3)
    assert result.skipped_reason == "empty_map"
    assert result.objects_total == 0
    assert encoder.text_calls == 0


def test_objects_without_features_return_reason_without_encoding() -> None:
    object_map = ObjectMap()
    _add_object(object_map, "a-chair", "chair", None)
    encoder = _TextEncoder()
    result = L3ClipQuery(object_map, encoder).query("chair", 1)
    assert result.skipped_reason == "no_object_features"
    assert result.objects_total == 1
    assert result.objects_with_features == 0
    assert encoder.text_calls == 0


def test_no_similarity_above_floor_returns_no_match() -> None:
    object_map = ObjectMap()
    _add_object(object_map, "a-chair", "chair", [1.0, 0.0])
    _add_object(object_map, "b-sofa", "sofa", [0.6, 0.8])
    _add_object(object_map, "c-table", "table", [0.2, 0.98])
    result = query_objects(
        object_map, "chair", 3, encoder=_TextEncoder([-1.0, 0.0]),
    )
    assert result.matches == ()
    assert result.valid is False
    assert result.skipped_reason == "no_similarity_above_floor"


def test_feature_dimension_mismatch_is_reported() -> None:
    result = query_objects(
        _three_objects(), "chair", 3, encoder=_TextEncoder([1.0, 0.0, 0.0]),
    )
    assert result.skipped_reason == "feature_dimension_mismatch"
    assert result.matches == ()


def test_invalid_stored_feature_is_skipped_not_scored() -> None:
    object_map = _three_objects()
    object_map.objects["a-chair"].clip_feature = np.asarray(
        [float("nan"), 0.0], dtype=np.float32,
    )
    object_map.objects["a-chair"].clip_feature_observations = 1
    result = query_objects(
        object_map, "chair", 3, encoder=_TextEncoder([1.0, 0.2]),
    )
    assert "a-chair" not in {match.object_id for match in result.matches}
    assert {match.object_id for match in result.matches} == {"b-sofa", "c-table"}


def test_orthogonal_match_is_not_injected() -> None:
    result = query_objects(
        _three_objects(), "chair", 3, encoder=_TextEncoder([1.0, 0.0]),
    )
    assert [match.object_id for match in result.matches] == ["a-chair", "b-sofa"]
    assert all(match.similarity > 0.0 for match in result.matches)


def test_result_preserves_existing_identity_and_geometry() -> None:
    object_map = _three_objects()
    before = {
        object_id: (
            node.center_world, node.bbox_3d, node.observation_count,
            tuple(node.evidence_ids), node.points_world.copy(),
        )
        for object_id, node in object_map.objects.items()
    }
    result = query_objects(
        object_map, "sofa", 1, encoder=_TextEncoder([0.6, 0.8]),
    )
    match = result.matches[0]
    assert match.object_id == "b-sofa"
    assert match.class_name == "sofa"
    assert match.center_world == (3.0, 0.5, -4.0)
    assert match.bbox_3d == ((2.5, 0.0, -4.5), (3.5, 1.0, -3.5))
    assert match.observation_count == 2
    assert match.evidence_ids == ("ev-b-sofa",)
    assert match.latest_step == 0
    assert match.geometry_status == "observed"
    assert match.point_count == 4
    assert match.to_dict()["object_id"] == "b-sofa"
    for object_id, node in object_map.objects.items():
        center, bbox, count, evidence_ids, points = before[object_id]
        assert (node.center_world, node.bbox_3d, node.observation_count,
                tuple(node.evidence_ids)) == (center, bbox, count, evidence_ids)
        assert np.array_equal(node.points_world, points)


def test_query_rejects_bad_top_k_and_feature_matrix() -> None:
    encoder = _TextEncoder()
    with pytest.raises(ValueError):
        L3ClipQuery(_three_objects(), encoder).query("chair", 0)
    with pytest.raises(TypeError):
        L3ClipQuery(_three_objects(), encoder).query(None, 1)
    with pytest.raises(ValueError):
        L3ClipQuery(_three_objects(), encoder).query("chair", 1, min_similarity=2.0)


def test_attach_clip_features_fuses_without_touching_geometry() -> None:
    object_map = ObjectMap()
    node = _add_object(object_map, "a-chair", "chair", [1.0, 0.0])
    geometry = (
        node.center_world, node.bbox_3d, node.observation_count,
        tuple(node.evidence_ids), node.points_world.copy(),
    )
    node.evidence_ids.append("ev-second")
    object_map.evidence["ev-second"] = SpatialEvidence(
        "ev-second", "scene", "episode", 0, 4, "chair", 0.8,
        "yolo11n-seg", 9, 4, "observed",
    )
    assert object_map.attach_clip_features({"ev-second": [0.0, 1.0]}) == 1
    expected = np.asarray([1.0, 1.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert node.clip_feature.tolist() == pytest.approx(expected.tolist())
    assert node.clip_feature_observations == 2
    assert node.clip_feature_evidence_ids == ["ev-a-chair", "ev-second"]
    center, bbox, count, evidence_ids, points = geometry
    assert (node.center_world, node.bbox_3d, node.observation_count) == (center, bbox, count)
    assert tuple(node.evidence_ids) == evidence_ids + ("ev-second",)
    assert np.array_equal(node.points_world, points)
    # Re-attaching the same Evidence must not double-count the observation.
    assert object_map.attach_clip_features({"ev-second": [0.0, 1.0]}) == 0
    assert node.clip_feature_observations == 2
    assert object_map.attach_clip_features({"unknown": [1.0, 0.0]}) == 0


class _MappingEncoder:
    feature_dim = 2

    def __init__(self, features: dict[str, tuple[float, float]]) -> None:
        self.features = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in features.items()
        }
        self.texts: list[str] = []

    def encode_image(self, rgb, boxes):
        raise AssertionError("query must not re-encode object images")

    def encode_text(self, texts):
        self.texts.extend(texts)
        return np.asarray([self.features[texts[0]]], dtype=np.float32)


def test_candidates_record_rank_evidence_and_rejection_reason() -> None:
    object_map = ObjectMap()
    _add_object(object_map, "a-chair", "chair", [1.0, 0.0])
    _add_object(object_map, "b-sofa", "sofa", [0.0, 1.0])
    _add_object(object_map, "c-table", "table", [0.5, 0.5])
    result = L3ClipQuery(
        object_map, _TextEncoder([1.0, 0.0]),
    ).query("chair", 1)
    assert [match.object_id for match in result.matches] == ["a-chair"]
    assert result.matches[0].rank == 1
    assert result.matches[0].detector_confidence == pytest.approx(0.9)
    assert [(c.rank, c.object_id, c.decision) for c in result.candidates] == [
        (1, "a-chair", "matched"),
        (2, "c-table", "ranked_out_by_top_k"),
        (3, "b-sofa", "below_positive_cosine"),
    ]
    assert all(candidate.evidence_ids for candidate in result.candidates)
    assert result.to_dict()["candidates"][2]["decision"] == "below_positive_cosine"


def test_real_r2r_instruction_and_landmark_phrase_keep_exact_query_text() -> None:
    # This Mock only proves plumbing: the controller/query pass the caller's
    # text verbatim to the official CLIP text-query path.  It does not claim a
    # real CLIP model would rank these features the same way.
    r2r = (
        "Walk past the sofa and turn right at the wooden table, "
        "then stop near the window."
    )
    object_map = ObjectMap()
    _add_object(object_map, "a-sofa", "sofa", [1.0, 0.0])
    _add_object(object_map, "b-table", "table", [0.0, 1.0])
    _add_object(object_map, "c-window", "window", [0.8, 0.6])
    encoder = _MappingEncoder({
        r2r: (0.0, 1.0),
        "window": (0.8, 0.6),
    })
    full = query_objects(object_map, r2r, 1, encoder=encoder)
    short = query_objects(object_map, "window", 1, encoder=encoder)
    assert encoder.texts == [r2r, "window"]
    assert full.matches[0].object_id == "b-table"
    assert short.matches[0].object_id == "c-window"
    assert full.query == r2r
    assert short.query == "window"


def test_positive_cosine_is_stored_as_raw_similarity_not_confidence() -> None:
    result = query_objects(
        _three_objects(), "chair", 1, encoder=_TextEncoder([1.0, 0.0]),
    )
    match = result.matches[0]
    payload = match.to_dict()
    assert payload["similarity"] == match.similarity
    assert "confidence" not in payload
    assert payload["detector_confidence"] == pytest.approx(0.9)
    assert result.candidates[0].decision == "matched"
    assert set(result.to_dict()) >= {"query", "matches", "candidates"}


def test_attach_clip_features_matches_official_three_observation_merge() -> None:
    object_map = ObjectMap()
    node = _add_object(object_map, "a-chair", "chair", [1.0, 0.0])
    for evidence_id, feature in (
        ("ev-second", [0.0, 1.0]),
        ("ev-third", [0.5, 0.5]),
    ):
        node.evidence_ids.append(evidence_id)
        object_map.evidence[evidence_id] = SpatialEvidence(
            evidence_id, "scene", "episode", 0, 3, "chair", 0.8,
            "yolo11n-seg", 9, 4, "observed",
        )
        object_map.attach_clip_features({evidence_id: feature})
    expected = np.asarray([1.0, 0.0], dtype=np.float32)
    count = 1
    for feature in ([0.0, 1.0], [0.5, 0.5]):
        merged = np.asarray(feature, dtype=np.float32)
        merged /= np.linalg.norm(merged)
        expected = (expected * count + merged) / (count + 1)
        expected /= np.linalg.norm(expected)
        count += 1
    assert node.clip_feature_observations == 3
    assert np.allclose(node.clip_feature, expected, atol=1e-6)


def test_weak_positive_cosine_is_kept_as_raw_score_not_confidence() -> None:
    object_map = ObjectMap()
    _add_object(object_map, "a-chair", "chair", [1.0, 0.0])
    _add_object(object_map, "b-sofa", "sofa", [0.1, 0.995])
    result = query_objects(
        object_map, "chair", 2, encoder=_TextEncoder([1.0, 0.0]),
    )
    weak = next(match for match in result.matches if match.object_id == "b-sofa")
    assert weak.similarity == pytest.approx(0.1, abs=0.01)
    assert "confidence" not in weak.to_dict()
    assert weak.detector_confidence == pytest.approx(0.9)
    candidate = next(
        item for item in result.candidates if item.object_id == "b-sofa"
    )
    assert candidate.decision == "matched"
