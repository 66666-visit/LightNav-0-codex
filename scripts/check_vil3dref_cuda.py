"""C/D gate: official pointnet2 CUDA ops + full Nr3D Student forward.

Run only in a CUDA-enabled venv where the official extension has been built:

    PYTHONPATH=src python scripts/check_vil3dref_cuda.py \
        --checkpoint .../gtlabelpcd_mix/nr3d/ckpts/model_epoch_96.pt \
        --bert_vocab .../vocab.txt --out .../cuda_forward.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
_POINTNET2_SRC = (
    _REPO / "references" / "vil3dref" / "og3d_src"
    / "model" / "external_tools" / "pointnet2"
)
if str(_POINTNET2_SRC) not in sys.path:
    sys.path.insert(0, str(_POINTNET2_SRC))


class _SyntheticTokenizer:
    def encode(self, text: str) -> list[int]:
        # Deterministic BERT-shaped ids; this check is about the CUDA graph,
        # not language quality.  The official tokenizer is exercised by the
        # episode path.
        return [101, *[1 + (ord(char) % 1000) for char in text[:32]], 102]


def check_pointnet2(device: torch.device) -> dict:
    import pointnet2._ext as ext

    torch.manual_seed(0)
    xyz = torch.rand(1, 1024, 3, device=device, dtype=torch.float32)
    features = torch.rand(1, 3, 1024, device=device, dtype=torch.float32)
    idx = pointnet2_fps(xyz, 32)
    new_xyz = gather_points(xyz.transpose(1, 2).contiguous(), idx).transpose(1, 2)
    grouped_idx = pointnet2_ball(0.3, 16, xyz, new_xyz)
    grouped = pointnet2_group(features, grouped_idx)
    unknown = new_xyz.contiguous()
    known = xyz.contiguous()
    dist2, nn_idx = pointnet2_three_nn(unknown, known)
    interpolated = pointnet2_interpolate(features, nn_idx, 1.0 / (dist2 + 1e-8))
    return {
        "extension": ext.__file__,
        "fps_shape": list(idx.shape),
        "ball_query_shape": list(grouped_idx.shape),
        "group_points_shape": list(grouped.shape),
        "three_nn_shapes": [list(dist2.shape), list(nn_idx.shape)],
        "three_interpolate_shape": list(interpolated.shape),
        "all_finite": bool(
            torch.isfinite(grouped).all()
            and torch.isfinite(interpolated).all()
        ),
    }


def pointnet2_fps(xyz, npoint):
    import pointnet2_utils

    return pointnet2_utils.furthest_point_sample(xyz, npoint)


def gather_points(xyz, idx):
    import pointnet2_utils

    return pointnet2_utils.gather_operation(xyz, idx)


def pointnet2_ball(radius, nsample, xyz, new_xyz):
    import pointnet2_utils

    return pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz)


def pointnet2_group(features, idx):
    import pointnet2_utils

    return pointnet2_utils.grouping_operation(features, idx)


def pointnet2_three_nn(unknown, known):
    import pointnet2_utils

    return pointnet2_utils.three_nn(unknown, known)


def pointnet2_interpolate(features, idx, weight):
    import pointnet2_utils

    return pointnet2_utils.three_interpolate(features, idx, weight)


def check_full_student(args: argparse.Namespace, device: torch.device) -> dict:
    from lightnav.memory.l3_instance_grounder import (
        OfficialStudentRuntime,
        build_official_batch,
    )
    from lightnav.memory.rgb_sidecar import XYZRGBProposal

    runtime = OfficialStudentRuntime(
        args.checkpoint, config_path=args.config, device=str(device),
        bert_vocab_path=args.bert_vocab,
    )
    rng = np.random.RandomState(0)
    proposals = []
    for index, count in enumerate((1500, 800, 1200)):
        xyz = rng.uniform(-2.0, 2.0, size=(count, 3)).astype(np.float32)
        rgb = rng.randint(0, 256, size=(count, 3)).astype(np.float32)
        proposals.append(XYZRGBProposal(
            object_id=f"object-{index}", class_name="couch",
            points_world_rgb=np.concatenate((xyz, rgb), axis=1),
            evidence_ids=(f"evidence-{index}",), observation_count=1,
        ))
    batch, metadata = build_official_batch(
        "go toward the far couch in front of the windows",
        proposals, _SyntheticTokenizer(), seed=0,
    )
    moved = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        output = runtime.model(moved)
    logits = output["og3d_logits"][0].detach().float().cpu()
    probs = torch.softmax(logits, dim=0)
    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "batch": {key: list(value.shape) for key, value in batch.items()},
        "metadata": metadata,
        "logits": [float(value) for value in logits],
        "probabilities": [float(value) for value in probs],
        "argmax": int(torch.argmax(logits)),
        "all_finite": bool(torch.isfinite(logits).all() and torch.isfinite(probs).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bert_vocab", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this process")
    device = torch.device("cuda")
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "pointnet2": check_pointnet2(device),
        "student_forward": check_full_student(args, device),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0 if (
        report["pointnet2"]["all_finite"]
        and report["student_forward"]["all_finite"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
