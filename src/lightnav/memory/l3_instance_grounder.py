"""Language-conditioned instance grounding for Geometry L3 candidates.

The module is an isolated adapter between the existing Geometry on-demand
query and the existing Prompt formatter.  It does **not** re-rank candidates
with hand-written rules: it loads the official ViL3DRel Nr3D Student checkpoint
and runs the official text encoder + spatial transformer + grounding head.  It
returns the official softmax scores and argmax for every proposal; no
confidence threshold is added.  Selecting an *eligible* proposal from those
official scores (one that maps to a formatable geometry fact) is done by
:mod:`lightnav.memory.l3_evidence_selector`, never by changing the scores.

Inputs come from :mod:`lightnav.memory.rgb_sidecar`, which records the real
XYZRGB pixels of every L3 Evidence mask.  The official preprocessing is
reproduced exactly:

* convert Habitat ``+Y`` up to the ScanNet/ViL3DRel ``+Z`` up frame;
* centre and size from the object points;
* sample 1024 points (with replacement when there are fewer);
* centre the sampled XYZ and divide by the maximum radius;
* normalize RGB from ``[0, 255]`` to ``[-1, 1]``.

If the sidecar is incomplete, the model cannot be loaded, or argmax does not
map back to a real ObjectNode, the result is an explicit refusal.  The caller
must keep the original L1 Prompt; this module never returns all same-class
candidates as a fallback.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from lightnav.memory.rgb_sidecar import RGBSidecar, XYZRGBProposal

_REFERENCE_ROOT = (
    Path(__file__).resolve().parents[3] / "references" / "vil3dref" / "og3d_src"
)
_DEFAULT_CONFIG = _REFERENCE_ROOT / "configs" / "nr3d_gtlabelpcd_mix_model.yaml"
_DEFAULT_NUM_POINTS = 1024
class ProposalTokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        """Return one BERT token id sequence, including ``[CLS]``/``[SEP]``."""


@dataclass(frozen=True)
class ProposalScore:
    """One proposal's official logit and softmax probability.

    ``proposal_id`` is the per-forward proposal row; ``object_id`` is the
    persistent ``ObjectNode`` ID it came from, and ``evidence_ids`` is the
    node's Evidence set.  The mapping is recorded explicitly so a caller never
    has to infer identity from list position.
    """

    object_id: str
    class_name: str
    point_count: int
    center_vil: tuple[float, float, float]
    size_vil: tuple[float, float, float]
    logit: float
    probability: float
    proposal_id: str = ""
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_object_id": self.object_id,
            "persistent_object_id": self.object_id,
            "object_id": self.object_id,
            "class_name": self.class_name,
            "evidence_ids": list(self.evidence_ids),
            "point_count": self.point_count,
            "center_vil": list(self.center_vil),
            "size_vil": list(self.size_vil),
            "grounding_logit": self.logit,
            "grounding_probability": self.probability,
            "logit": self.logit,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class InstanceGroundingResult:
    """One official Student forward plus an explicit selected/refused outcome."""

    selected_object_id: str | None
    selected_index: int | None
    proposals: tuple[ProposalScore, ...] = ()
    top1_top2_gap: float | None = None
    latency_ms: float = 0.0
    checkpoint: str = ""
    checkpoint_sha256: str = ""
    skipped_objects: tuple[dict[str, Any], ...] = ()
    offered_proposals: tuple[dict[str, Any], ...] = ()
    refusal_reason: str | None = None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.selected_object_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_object_id": self.selected_object_id,
            "selected_index": self.selected_index,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "top1_top2_gap": self.top1_top2_gap,
            "latency_ms": self.latency_ms,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "skipped_objects": [dict(item) for item in self.skipped_objects],
            "offered_proposals": [dict(item) for item in self.offered_proposals],
            "refusal_reason": self.refusal_reason,
            "error": self.error,
        }


def habitat_to_vil3dref(points_xyz: np.ndarray) -> np.ndarray:
    """Rotate Habitat ``(x, y, z)`` world points to ViL3DRel ``(x, -z, y)``.

    The matrix has determinant +1 and maps Habitat's +Y-up axis onto the
    ScanNet/ViL3DRel +Z-up axis used by ``CMT.calc_pairwise_locs``.
    """
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite (N, 3)")
    result = np.empty_like(points)
    result[:, 0] = points[:, 0]
    result[:, 1] = -points[:, 2]
    result[:, 2] = points[:, 1]
    return result


def _sample_indices(
    count: int, num_points: int, rng: np.random.RandomState,
) -> np.ndarray:
    if count < 1:
        raise ValueError("cannot sample an empty point cloud")
    return rng.choice(count, size=num_points, replace=count < num_points)


def prepare_official_object(
    proposal: XYZRGBProposal,
    *,
    num_points: int = _DEFAULT_NUM_POINTS,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(obj_fts[1024,6], obj_loc[6], center_vil, size_vil)``.

    ``obj_fts`` is exactly the official Student input: 1024 XYZRGB points,
    XYZ in a unit ball and RGB in ``[-1, 1]``.  ``obj_loc`` is the official
    ``[center, size]`` of the complete proposal (before the 1024-point sample),
    in the z-up frame.
    """
    if type(num_points) is not int or num_points < 1:
        raise ValueError("num_points must be a positive integer")
    if not isinstance(proposal, XYZRGBProposal):
        raise TypeError("proposal must be an XYZRGBProposal")
    raw = np.asarray(proposal.points_world_rgb, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 6 or raw.shape[0] < 1:
        raise ValueError("proposal must be a nonempty (N, 6) XYZRGB array")
    if not np.isfinite(raw).all():
        raise ValueError("proposal contains non-finite values")
    xyz_vil = habitat_to_vil3dref(raw[:, :3])
    colors = raw[:, 3:6]
    if np.any(colors < 0.0) or np.any(colors > 255.0):
        raise ValueError("proposal RGB must be real pixels in [0, 255]")
    center = xyz_vil.mean(axis=0, dtype=np.float64)
    size = xyz_vil.max(axis=0) - xyz_vil.min(axis=0)
    rng = rng or np.random
    indices = _sample_indices(len(xyz_vil), num_points, rng)
    sampled_xyz = np.array(xyz_vil[indices], dtype=np.float32, copy=True)
    sampled_rgb = np.array(colors[indices], dtype=np.float32, copy=True)
    sampled_xyz -= sampled_xyz.mean(axis=0)
    max_dist = float(np.max(np.sqrt(np.sum(sampled_xyz ** 2, axis=1))))
    if max_dist < 1e-6:
        max_dist = 1.0
    sampled_xyz /= max_dist
    sampled_rgb = sampled_rgb / 127.5 - 1.0
    obj_fts = np.concatenate((sampled_xyz, sampled_rgb), axis=1).astype(np.float32)
    obj_loc = np.concatenate((center, size), axis=0).astype(np.float32)
    return obj_fts, obj_loc, center.astype(np.float32), size.astype(np.float32)


def build_official_batch(
    instruction: str,
    proposals: Sequence[XYZRGBProposal],
    tokenizer: ProposalTokenizer,
    *,
    num_points: int = _DEFAULT_NUM_POINTS,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the official Student batch plus per-proposal metadata.

    The sampling RNG is seeded per object (``seed + index``) so one query is
    reproducible without mutating NumPy's global RNG.
    """
    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    if not instruction.strip():
        raise ValueError("instruction must be nonempty")
    if not proposals:
        raise ValueError("at least one proposal is required")
    if not callable(getattr(tokenizer, "encode", None)):
        raise TypeError("tokenizer must provide encode(text)")
    token_ids = tokenizer.encode(instruction)
    if not isinstance(token_ids, list) or not token_ids or not all(
        type(token) is int for token in token_ids
    ):
        raise ValueError("tokenizer.encode must return a nonempty list[int]")
    obj_fts: list[np.ndarray] = []
    obj_locs: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals):
        rng = np.random.RandomState(seed + index)
        features, loc, center, size = prepare_official_object(
            proposal, num_points=num_points, rng=rng,
        )
        obj_fts.append(features)
        obj_locs.append(loc)
        metadata.append({
            "proposal_id": f"proposal-{index}",
            "object_id": proposal.object_id,
            "class_name": proposal.class_name,
            "point_count": proposal.point_count,
            "evidence_ids": list(proposal.evidence_ids),
            "center_vil": [float(value) for value in center],
            "size_vil": [float(value) for value in size],
        })
    import torch

    txt_ids = torch.tensor([token_ids], dtype=torch.long)
    txt_masks = torch.ones((1, len(token_ids)), dtype=torch.bool)
    return {
        "txt_ids": txt_ids,
        "txt_masks": txt_masks,
        "obj_fts": torch.from_numpy(np.stack(obj_fts, axis=0)[None, ...]),
        "obj_locs": torch.from_numpy(np.stack(obj_locs, axis=0)[None, ...]),
        "obj_masks": torch.ones((1, len(proposals)), dtype=torch.bool),
    }, metadata


class _BertFromConfig:
    """Context manager replacing the official network BERT download.

    ``ReferIt3DNet.__init__`` calls
    ``BertModel.from_pretrained('bert-base-uncased', config=...)``.  Every BERT
    tensor is present in the official checkpoint, and the strict state-dict
    load below replaces all of them, so constructing the exact 3-layer
    architecture from the config is equivalent and keeps inference offline.
    """

    def __init__(self) -> None:
        self._original: Any = None

    def __enter__(self) -> None:
        import inspect

        from transformers import BertModel

        # Keep the exact classmethod descriptor.  ``BertModel.from_pretrained``
        # is a bound method; assigning it back would leave a process-wide
        # instance-style override instead of restoring the classmethod.
        self._original = inspect.getattr_static(BertModel, "from_pretrained")
        BertModel.from_pretrained = classmethod(  # type: ignore[assignment]
            lambda cls, _name, config=None, **_kwargs: cls(config)
        )

    def __exit__(self, *_exc: object) -> None:
        from transformers import BertModel

        if self._original is None:
            try:
                delattr(BertModel, "from_pretrained")
            except AttributeError:
                pass
        else:
            BertModel.from_pretrained = self._original  # type: ignore[assignment]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OfficialStudentRuntime:
    """Official Nr3D Student checkpoint loaded once in the current process."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        config_path: str | Path | None = None,
        device: str = "cuda",
        bert_vocab_path: str | Path | None = None,
        source_root: str | Path | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"official ViL3DRel checkpoint not found: {self.checkpoint}")
        self.config_path = Path(config_path).expanduser() if config_path else (
            source_root / "configs" / "nr3d_gtlabelpcd_mix_model.yaml"
            if source_root is not None else _DEFAULT_CONFIG
        )
        if not self.config_path.is_file():
            raise FileNotFoundError(f"official ViL3DRel config not found: {self.config_path}")
        self.device_name = device
        self.bert_vocab_path = (
            Path(bert_vocab_path).expanduser() if bert_vocab_path else None
        )
        if self.bert_vocab_path is not None and not self.bert_vocab_path.is_file():
            raise FileNotFoundError(f"BERT vocab not found: {self.bert_vocab_path}")
        self.checkpoint_sha256 = _sha256(self.checkpoint)
        self.compat_ignored_keys: list[str] = []
        self.checkpoint_format = ""
        self.model = self._load_model(source_root)
        self.tokenizer = self._load_tokenizer()
        self.last_input_metadata: list[dict[str, Any]] = []

    def _source_root(self, source_root: str | Path | None) -> Path:
        if source_root is not None:
            root = Path(source_root).expanduser()
        else:
            root = _REFERENCE_ROOT
        if not (root / "model" / "referit3d_net.py").is_file():
            raise FileNotFoundError(f"official ViL3DRel source not found: {root}")
        return root

    def _load_model(self, source_root: str | Path | None) -> Any:
        import torch

        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "the official ViL3DRel Student requires a CUDA device and a "
                "compiled pointnet2 extension; CUDA is not available in this process"
            )
        root = self._source_root(source_root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        pointnet2_root = root / "model" / "external_tools" / "pointnet2"
        if str(pointnet2_root) not in sys.path:
            sys.path.insert(0, str(pointnet2_root))
        try:
            importlib.import_module("pointnet2._ext")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "official pointnet2 CUDA extension is not importable; build it with "
                "scripts/build_vil3dref_pointnet2.sh before enabling the grounder"
            ) from exc
        import yaml
        from easydict import EasyDict

        config = EasyDict(yaml.safe_load(self.config_path.read_text()))["model"]
        config.model_type = "gtpcd"
        config.obj_encoder.use_color_enc = False
        with _BertFromConfig():
            module = importlib.import_module("model.referit3d_net")
            model = module.ReferIt3DNet(config, torch.device("cpu"))
        checkpoint = torch.load(
            self.checkpoint, map_location="cpu", weights_only=False,
        )
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError("official checkpoint is not a state dict")
        student = {
            key.removeprefix("student_model."): value
            for key, value in checkpoint.items()
            if key.startswith("student_model.")
        }
        checkpoint_format = "mix_student_prefix"
        if not student:
            # train_mix.py also accepts a student-only checkpoint without the
            # wrapper prefix.  Keep the official teacher weights out if a mix
            # checkpoint is passed without the expected prefix.
            student = {
                key: value for key, value in checkpoint.items()
                if not key.startswith("teacher_model.")
            }
            checkpoint_format = "student_only"
        if not student:
            raise ValueError("official checkpoint has no student weights")
        self.checkpoint_format = checkpoint_format
        loaded = model.load_state_dict(student, strict=False)
        # transformers >=5 no longer registers the fixed BERT position_ids
        # buffer.  It is not used by the forward pass and the official
        # checkpoint stores it as a non-parameter buffer, so ignoring exactly
        # this key preserves every learned weight.
        ignored = {"txt_encoder.embeddings.position_ids"}
        missing = [
            key for key in loaded.missing_keys if key not in ignored
        ]
        unexpected = [
            key for key in loaded.unexpected_keys if key not in ignored
        ]
        self.compat_ignored_keys = sorted(
            (set(loaded.unexpected_keys) | set(loaded.missing_keys)) & ignored
        )
        if missing or unexpected:
            raise ValueError(
                "official Student state dict is not strict: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"first_missing={missing[:3]} first_unexpected={unexpected[:3]}"
            )
        first_conv = (
            "obj_encoder.pcd_net.encoder.0.mlps.0.layer0.conv.weight"
        )
        weight = model.state_dict().get(first_conv)
        if weight is None or tuple(weight.shape) != (64, 6, 1, 1):
            raise ValueError(
                "official Student PointNet++ does not expect 6-channel XYZRGB"
            )
        target_device = torch.device(self.device_name)
        model.to(target_device)
        # ReferIt3DNet.prepare_batch moves every input batch to the device it
        # was *constructed* with; model.to() does not update that attribute.
        # Keeping the constructor's CPU device would move CUDA inputs back to
        # CPU while the weights are on CUDA.
        model.device = target_device
        model.eval()
        return model

    def _load_tokenizer(self) -> Any:
        from transformers import BertTokenizer

        if self.bert_vocab_path is not None:
            # transformers 4.x names the argument ``vocab_file``; 5.x names it
            # ``vocab``.  Both load the same local WordPiece file.
            try:
                return BertTokenizer(vocab=str(self.bert_vocab_path))
            except (TypeError, ValueError):
                return BertTokenizer(vocab_file=str(self.bert_vocab_path))
        return BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=True,
        )

    @property
    def device(self) -> Any:
        import torch

        return torch.device(self.device_name)

    def ground(
        self,
        instruction: str,
        proposals: Sequence[XYZRGBProposal],
        *,
        step: int,
        episode_id: str = "",
        seed: int = 0,
    ) -> InstanceGroundingResult:
        if not proposals:
            return InstanceGroundingResult(
                None, None, checkpoint=str(self.checkpoint),
                checkpoint_sha256=self.checkpoint_sha256,
                refusal_reason="no_rgb_proposals",
            )
        if not instruction.strip():
            return InstanceGroundingResult(
                None, None, checkpoint=str(self.checkpoint),
                checkpoint_sha256=self.checkpoint_sha256,
                refusal_reason="empty_instruction",
            )
        started = time.perf_counter()
        offered: tuple[dict[str, Any], ...] = tuple({
            "proposal_id": f"proposal-{index}",
            "object_id": proposal.object_id,
            "class_name": proposal.class_name,
            "point_count": proposal.point_count,
            "evidence_ids": list(proposal.evidence_ids),
        } for index, proposal in enumerate(proposals))
        try:
            import torch

            batch, metadata = build_official_batch(
                instruction, proposals, self.tokenizer, seed=seed,
            )
            self.last_input_metadata = metadata
            offered = tuple(dict(item) for item in metadata)
            moved = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            with torch.no_grad():
                output = self.model(moved)
            logits = output["og3d_logits"][0].detach().float().cpu().numpy()
            if logits.shape != (len(proposals),):
                raise ValueError(
                    f"official Student returned {logits.shape} logits for "
                    f"{len(proposals)} proposals"
                )
            if not np.isfinite(logits).all():
                raise ValueError("official Student returned non-finite logits")
            probabilities = np.asarray(
                torch.softmax(torch.from_numpy(logits), dim=0).numpy(),
                dtype=np.float64,
            )
            if not np.isfinite(probabilities).all():
                raise ValueError("official Student returned non-finite probabilities")
            selected_index = int(np.argmax(logits))
            scores = tuple(
                ProposalScore(
                    object_id=proposal.object_id,
                    class_name=proposal.class_name,
                    point_count=proposal.point_count,
                    center_vil=tuple(float(value) for value in meta["center_vil"]),
                    size_vil=tuple(float(value) for value in meta["size_vil"]),
                    logit=float(logit),
                    probability=float(probability),
                    proposal_id=meta["proposal_id"],
                    evidence_ids=tuple(meta["evidence_ids"]),
                )
                for proposal, meta, logit, probability in zip(
                    proposals, metadata, logits, probabilities,
                )
            )
            ordered = sorted(probabilities, reverse=True)
            gap = (
                float(ordered[0] - ordered[1]) if len(ordered) > 1 else None
            )
            return InstanceGroundingResult(
                selected_object_id=scores[selected_index].object_id,
                selected_index=selected_index,
                proposals=scores,
                top1_top2_gap=gap,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                checkpoint=str(self.checkpoint),
                checkpoint_sha256=self.checkpoint_sha256,
                offered_proposals=offered,
            )
        except Exception as exc:  # noqa: BLE001 - a sidecar must fail closed
            return InstanceGroundingResult(
                None, None, latency_ms=(time.perf_counter() - started) * 1000.0,
                checkpoint=str(self.checkpoint),
                checkpoint_sha256=self.checkpoint_sha256,
                offered_proposals=offered,
                refusal_reason="grounding_error",
                error=f"{type(exc).__name__}: {exc}",
            )


class ViL3DRelGrounder:
    """Thin manager-facing wrapper: proposals in, one Object ID or refusal out."""

    def __init__(self, runtime: OfficialStudentRuntime, *, seed: int = 0) -> None:
        if not callable(getattr(runtime, "ground", None)):
            raise TypeError("runtime must provide ground()")
        if type(seed) is not int or seed < 0:
            raise ValueError("grounder seed must be a nonnegative integer")
        self.runtime = runtime
        self.seed = seed
        self.last_result: InstanceGroundingResult | None = None

    def select(
        self,
        *,
        instruction: str,
        object_map: Any,
        sidecar: RGBSidecar,
        step: int,
        episode_id: str = "",
    ) -> InstanceGroundingResult:
        """Score every complete proposal for one grounding query.

        ``instruction`` is either the original navigation instruction or one
        extracted instruction-landmark phrase.  In both cases the sidecar's
        complete proposal list is passed to the official Student unchanged, so
        spatial-relation reasoning still sees every candidate.  This method
        returns the official argmax and all scores; choosing an eligible
        proposal from those scores is the manager's responsibility.
        """
        if not isinstance(sidecar, RGBSidecar):
            raise TypeError("sidecar must be an RGBSidecar")
        proposals, skipped = sidecar.proposals(object_map.objects)
        if skipped:
            # Any ObjectNode with missing RGB makes the scene proposal set
            # incomplete; running on the remaining subset would silently change
            # the model's spatial context.  Refuse the whole query and keep the
            # original L1 card.
            result = InstanceGroundingResult(
                selected_object_id=None,
                selected_index=None,
                proposals=(),
                checkpoint=str(getattr(self.runtime, "checkpoint", "")),
                checkpoint_sha256=str(
                    getattr(self.runtime, "checkpoint_sha256", "")
                ),
                skipped_objects=tuple(skipped),
                refusal_reason="incomplete_rgb_sidecar",
            )
            self.last_result = result
            return result
        result = self.runtime.ground(
            instruction, proposals, step=step, episode_id=episode_id, seed=self.seed,
        )
        self.last_result = result
        return result

    def proposal_scores(self) -> list[dict[str, Any]]:
        if self.last_result is None:
            return []
        return [proposal.to_dict() for proposal in self.last_result.proposals]


__all__ = [
    "InstanceGroundingResult",
    "OfficialStudentRuntime",
    "ProposalScore",
    "ViL3DRelGrounder",
    "build_official_batch",
    "habitat_to_vil3dref",
    "prepare_official_object",
]
