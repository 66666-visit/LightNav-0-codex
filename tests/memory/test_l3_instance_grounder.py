"""CPU contracts for official ViL3DRel Student XYZRGB preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from lightnav.memory.l3_instance_grounder import (
    OfficialStudentRuntime,
    build_official_batch,
    habitat_to_vil3dref,
    prepare_official_object,
)
from lightnav.memory.rgb_sidecar import XYZRGBProposal


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [101, *[ord(char) for char in text], 102]


def _proposal(object_id: str = "object-A", points: int = 1400) -> XYZRGBProposal:
    rng = np.random.RandomState(42)
    xyz = rng.uniform(-2.0, 2.0, size=(points, 3)).astype(np.float32)
    rgb = rng.randint(0, 256, size=(points, 3)).astype(np.uint8)
    return XYZRGBProposal(
        object_id=object_id,
        class_name="couch",
        points_world_rgb=np.concatenate((xyz, rgb.astype(np.float32)), axis=1),
        evidence_ids=("evidence-1",),
        observation_count=1,
    )


def test_coordinate_conversion_maps_habitat_y_up_to_vil_z_up() -> None:
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-1.0, 0.5, -4.0],
    ], dtype=np.float32)
    converted = habitat_to_vil3dref(points)
    assert np.allclose(converted, [[0, 0, 0], [1, -3, 2], [-1, 4, 0.5]])
    # Determinant of the rotation is +1, i.e. no reflection.
    matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_preprocessing_matches_the_official_nr3d_dataset() -> None:
    """Compare against the official ``_get_obj_inputs`` on identical z-up XYZRGB."""
    pytest.importorskip("jsonlines")
    pytest.importorskip("einops")
    reference_src = (
        Path(__file__).resolve().parents[2]
        / "references" / "vil3dref" / "og3d_src"
    )
    if not (reference_src / "data" / "gtlabelpcd_dataset.py").is_file():
        pytest.skip("official ViL3DRel reference checkout is not available")
    sys.path.insert(0, str(reference_src))
    from data.gtlabelpcd_dataset import GTLabelPcdDataset

    rng = np.random.RandomState(7)
    xyz_habitat = rng.uniform(-3.0, 3.0, size=(1500, 3)).astype(np.float32)
    colors = rng.randint(0, 256, size=(1500, 3)).astype(np.uint8)
    xyz_vil = habitat_to_vil3dref(xyz_habitat)
    official_pcd = np.concatenate(
        (xyz_vil.astype(np.float64), colors.astype(np.float64) / 127.5 - 1.0),
        axis=1,
    )

    dataset = GTLabelPcdDataset.__new__(GTLabelPcdDataset)
    dataset.num_points = 1024
    dataset.max_obj_len = None
    np.random.seed(0)
    official_fts, official_locs, _colors, _labels, _ids, _target = (
        dataset._get_obj_inputs(
            [official_pcd.copy()], [np.zeros((3, 4), dtype=np.float32)],
            ["couch"], ["0"], 0, theta=None,
        )
    )

    proposal = XYZRGBProposal(
        object_id="0", class_name="couch",
        points_world_rgb=np.concatenate(
            (xyz_habitat, colors.astype(np.float32)), axis=1,
        ),
        evidence_ids=("evidence-1",), observation_count=1,
    )
    our_fts, our_loc, center, size = prepare_official_object(
        proposal, rng=np.random.RandomState(0),
    )
    assert our_fts.shape == (1024, 6)
    assert np.allclose(our_fts, official_fts[0], atol=1e-5)
    assert np.allclose(our_loc, official_locs[0], atol=1e-5)
    assert np.allclose(center, official_locs[0][:3], atol=1e-5)
    assert np.allclose(size, official_locs[0][3:], atol=1e-5)
    # The official sample is centred in a unit ball and RGB lives in [-1, 1].
    assert np.max(np.linalg.norm(our_fts[:, :3], axis=1)) == pytest.approx(1.0, abs=1e-5)
    assert our_fts[:, 3:].min() >= -1.0 and our_fts[:, 3:].max() <= 1.0


def test_small_object_uses_replacement_and_preserves_xyz_rgb_pairing() -> None:
    points = np.array([
        [0.0, 0.0, 0.0, 10.0, 20.0, 30.0],
        [1.0, 0.0, 0.0, 40.0, 50.0, 60.0],
        [0.0, 1.0, 0.0, 70.0, 80.0, 90.0],
    ], dtype=np.float32)
    proposal = XYZRGBProposal(
        object_id="small", class_name="lamp",
        points_world_rgb=points, evidence_ids=("e",), observation_count=1,
    )
    features, loc, center, size = prepare_official_object(
        proposal, rng=np.random.RandomState(3),
    )
    assert features.shape == (1024, 6)
    # Reconstruct the exact sampled indices with the same seed.  Both the XYZ
    # and RGB rows must correspond to those indices; a post-sample permutation
    # of either channel would fail here (checking only "colour is in the source
    # set" would not catch it).
    expected_idx = np.random.RandomState(3).choice(
        len(points), size=1024, replace=True,
    )
    xyz_vil = points[:, :3].copy()
    xyz_vil[:, 1] = -points[:, 2]
    xyz_vil[:, 2] = points[:, 1]
    sampled_xyz = xyz_vil[expected_idx].astype(np.float64)
    sampled_xyz -= sampled_xyz.mean(axis=0)
    max_dist = float(np.max(np.sqrt(np.sum(sampled_xyz ** 2, axis=1))))
    if max_dist < 1e-6:
        max_dist = 1.0
    sampled_xyz /= max_dist
    sampled_rgb = points[expected_idx, 3:] / 127.5 - 1.0
    assert np.allclose(features[:, :3], sampled_xyz, atol=1e-5)
    assert np.allclose(features[:, 3:], sampled_rgb, atol=1e-5)
    assert loc.shape == (6,)
    assert np.all(size >= 0.0)


def test_build_batch_keeps_object_order_and_returns_one_1024_point_tensor() -> None:
    proposals = [_proposal("object-A"), _proposal("object-B", points=800)]
    batch, metadata = build_official_batch(
        "go toward the far couch", proposals, _Tokenizer(), seed=5,
    )
    assert batch["obj_fts"].shape == (1, 2, 1024, 6)
    assert batch["obj_locs"].shape == (1, 2, 6)
    assert batch["obj_masks"].shape == (1, 2)
    assert batch["txt_ids"].shape[0] == 1
    assert batch["txt_masks"].all()
    assert [item["object_id"] for item in metadata] == ["object-A", "object-B"]
    assert [item["point_count"] for item in metadata] == [1400, 800]
    assert [item["proposal_id"] for item in metadata] == [
        "proposal-0", "proposal-1",
    ]
    assert [item["evidence_ids"] for item in metadata] == [
        ["evidence-1"], ["evidence-1"],
    ]


def test_build_batch_rejects_empty_instruction_and_empty_proposals() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        build_official_batch("", [_proposal()], _Tokenizer())
    with pytest.raises(ValueError, match="at least one proposal"):
        build_official_batch("find the couch", [], _Tokenizer())


def test_bert_from_config_restores_the_classmethod_descriptor() -> None:
    import inspect

    from transformers import BertModel

    from lightnav.memory.l3_instance_grounder import _BertFromConfig

    before = inspect.getattr_static(BertModel, "from_pretrained")
    with _BertFromConfig():
        assert BertModel.from_pretrained is not before
    after = inspect.getattr_static(BertModel, "from_pretrained")
    assert after is before
    assert isinstance(after, classmethod)


class _FakeStudentModel:
    def __init__(self, logits=None, error: Exception | None = None) -> None:
        self.logits = logits
        self.error = error

    def __call__(self, batch):
        if self.error is not None:
            raise self.error
        return {"og3d_logits": torch.tensor([self.logits], dtype=torch.float32)}


def _bare_runtime(model) -> OfficialStudentRuntime:
    runtime = object.__new__(OfficialStudentRuntime)
    runtime.checkpoint = Path("fake_student.pt")
    runtime.checkpoint_sha256 = "fake-sha"
    runtime.device_name = "cpu"
    runtime.tokenizer = _Tokenizer()
    runtime.model = model
    runtime.last_input_metadata = []
    runtime.compat_ignored_keys = []
    return runtime


def test_ground_uses_official_argmax_and_reports_all_scores() -> None:
    runtime = _bare_runtime(_FakeStudentModel([0.1, 2.0]))
    proposals = [_proposal("object-A"), _proposal("object-B", points=800)]
    result = runtime.ground(
        "find the far couch", proposals, step=3, episode_id="episode",
    )
    assert result.valid is True
    assert result.selected_object_id == "object-B"
    assert result.selected_index == 1
    assert [item.object_id for item in result.proposals] == [
        "object-A", "object-B",
    ]
    assert result.proposals[1].probability > result.proposals[0].probability
    assert result.top1_top2_gap is not None and result.top1_top2_gap > 0
    assert result.error is None
    assert len(result.offered_proposals) == 2
    assert [item.proposal_id for item in result.proposals] == [
        "proposal-0", "proposal-1",
    ]
    assert [item.evidence_ids for item in result.proposals] == [
        ("evidence-1",), ("evidence-1",),
    ]
    payload = result.proposals[1].to_dict()
    assert payload["persistent_object_id"] == "object-B"
    assert payload["proposal_object_id"] == "object-B"
    assert payload["grounding_probability"] == payload["probability"]


def test_ground_error_keeps_offered_proposals_for_diagnostics() -> None:
    runtime = _bare_runtime(_FakeStudentModel(error=RuntimeError("CUDA OOM")))
    proposals = [_proposal("object-A"), _proposal("object-B", points=800)]
    result = runtime.ground(
        "find the far couch", proposals, step=3, episode_id="episode",
    )
    assert result.valid is False
    assert result.refusal_reason == "grounding_error"
    assert "CUDA OOM" in (result.error or "")
    assert result.proposals == ()
    assert [item["object_id"] for item in result.offered_proposals] == [
        "object-A", "object-B",
    ]
