"""CPU contracts for the local CLIP feature helpers (mock features, no weights)."""

from __future__ import annotations

import numpy as np
import pytest

from lightnav.memory.clip_features import (
    CLIP_CROP_PADDING_PX,
    ClipFeatureEncoder,
    clip_crop_boxes,
    cosine_similarity,
    fuse_clip_feature,
    normalize_feature,
)


class _MockEncoder:
    feature_dim = 4

    def encode_image(self, rgb, boxes):
        return np.ones((len(boxes), 4), dtype=np.float32)

    def encode_text(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def test_clip_crop_boxes_pads_and_clips_at_borders() -> None:
    mask = np.zeros((10, 12), dtype=np.bool_)
    mask[3:6, 4:8] = True

    assert clip_crop_boxes([mask], (10, 12)) == [(0, 0, 12, 10)]
    assert clip_crop_boxes([mask], (10, 12), padding=1) == [(3, 2, 9, 7)]
    assert CLIP_CROP_PADDING_PX == 20


def test_clip_crop_boxes_preserve_mask_order_and_empty_masks() -> None:
    first = np.zeros((8, 8), dtype=np.bool_)
    first[1:3, 2:5] = True
    second = np.zeros((8, 8), dtype=np.bool_)
    second[5:7, 0:2] = True
    boxes = clip_crop_boxes([first, np.zeros((8, 8), dtype=np.bool_), second], (8, 8))
    assert boxes[0] == (0, 0, 8, 8)
    assert boxes[1] is None
    assert boxes[2] == (0, 0, 8, 8)


@pytest.mark.parametrize("mask", [
    np.zeros((8, 8), dtype=np.uint8),
    np.zeros((7, 8), dtype=np.bool_),
])
def test_clip_crop_boxes_reject_unaligned_or_non_boolean_masks(mask) -> None:
    with pytest.raises(ValueError):
        clip_crop_boxes([mask], (8, 8))


def test_normalize_feature_is_unit_length_float32() -> None:
    feature = normalize_feature([3.0, 4.0])
    assert feature.dtype == np.float32
    assert feature.shape == (2,)
    assert float(np.linalg.norm(feature)) == pytest.approx(1.0)
    assert feature.tolist() == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize("feature", [
    [0.0, 0.0],
    [1.0, float("nan")],
    [[1.0, 0.0]],
    [],
])
def test_normalize_feature_rejects_invalid_vectors(feature) -> None:
    with pytest.raises(ValueError):
        normalize_feature(feature)


def test_normalize_feature_checks_expected_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        normalize_feature([1.0, 0.0], expected_dim=3)


def test_fuse_clip_feature_matches_official_detection_weighted_mean() -> None:
    # ConceptGraphs merge_obj2_into_obj1:
    # ((old * 2) + (new * 1)) / 3, then L2 normalize.
    fused, count = fuse_clip_feature(
        np.asarray([1.0, 0.0], dtype=np.float32), 2,
        np.asarray([0.0, 1.0], dtype=np.float32),
    )
    assert count == 3
    expected = np.asarray([2.0, 1.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert fused.tolist() == pytest.approx(expected.tolist())
    assert float(np.linalg.norm(fused)) == pytest.approx(1.0)


def test_fuse_clip_feature_first_observation_and_dimension_guard() -> None:
    fused, count = fuse_clip_feature(None, 0, [0.0, 5.0])
    assert count == 1
    assert fused.tolist() == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError, match="dimension"):
        fuse_clip_feature(np.asarray([1.0, 0.0]), 1, [1.0, 0.0, 0.0])


def test_cosine_similarity_uses_normalized_dot_product() -> None:
    sims = cosine_similarity([1.0, 0.0], [[2.0, 0.0], [0.0, -3.0], [1.0, 1.0]])
    assert sims.tolist() == pytest.approx([1.0, 0.0, 1.0 / np.sqrt(2.0)])


def test_cosine_similarity_rejects_zero_norm_features() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [[0.0, 0.0]])


def test_mock_encoder_satisfies_protocol() -> None:
    assert isinstance(_MockEncoder(), ClipFeatureEncoder)


def test_repeated_fusion_is_equivalent_to_sequential_official_merge() -> None:
    # Official ConceptGraphs merge_obj2_into_obj1 is pairwise and normalizes
    # after every merge.  Reproduce that loop independently, then compare it to
    # the persistent ObjectNode update used by LightNav.
    observations = [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([2.0, 1.0], dtype=np.float32),
        np.asarray([0.2, -0.4], dtype=np.float32),
    ]

    def _official_sequential(values):
        merged = normalize_feature(values[0])
        count = 1
        for value in values[1:]:
            merged = normalize_feature((merged * count + normalize_feature(value)) / (count + 1))
            count += 1
        return merged, count

    expected, expected_count = _official_sequential(observations)
    actual = None
    count = 0
    for observation in observations:
        actual, count = fuse_clip_feature(actual, count, observation)
    assert count == expected_count == len(observations)
    assert np.allclose(actual, expected, atol=1e-6)
    assert float(np.linalg.norm(actual)) == pytest.approx(1.0)
