"""Habitat evaluation runner: drive a remote Habitat env with the VLN-CE velocity policy.

Flow: build the inference engine -> connect to the env server -> loop over
episodes (``reset`` -> per-step ``policy.act`` -> ``env.step``) -> append one
result dict per episode to ``<output_dir>/results.jsonl`` -> print/write the
VLN-CE or ObjectNav summary. Decoding is greedy; no sampling knobs are set.
With ``save_video`` / ``record_dir`` each step's frame is also rendered with the
prediction (``lightnav.viz``) into ``<output_dir>/videos/`` and/or recorded raw.

The env, policy and engine constructors can be injected so the loop can be
exercised without a GPU or model.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from lightnav.eval_config import get_task_params, load_eval_config, resolve_asset_path
from lightnav.habitat.experiment_integrity import (
    _key as episode_key,
    archive_unfinished_diagnostics,
    check_coverage,
    read_jsonl,
    repair_interrupted_jsonl_tail,
    validate_manifest,
    write_or_match_manifest,
)
from lightnav.habitat.policy import TrajVocabVLNCEPolicy, extract_instruction
from lightnav.habitat.progress import ProgressReporter
from lightnav.habitat.results import (
    make_json_safe,
    print_objectnav_summary,
    print_vlnce_summary,
)
from lightnav.memory.rgm_config import RGMConfig
from lightnav.serving.protocol import pointing_payload
from lightnav.serving.token_budget import decode_token_budget, probe_grounding_tokens
from lightnav.tracking import load_centroids
from lightnav.traj_vocab import load_rvq_bundle
from lightnav.vln_utils import DEFAULT_TRAJ_HORIZON, DEFAULT_TRAJ_K

logger = logging.getLogger(__name__)

_CENTROIDS_FILENAME = "centroids_whole_chunk_K{K}_h{H}.npy"


@dataclass
class MemoryConfig:
    """Habitat policy's episode-local explicit memory settings."""

    enabled: bool = False
    top_k: int = 5
    max_context_chars: int = 600
    strategy: str = "flat"
    layers: tuple[str, ...] = ("short", "mid", "long")
    extractor: str = "null"
    extractor_model_path: str = ""
    extractor_device: str = "cpu"
    extractor_confidence: float = 0.25
    extract_every_n_steps: int = 5
    place_enabled: bool = False
    place_model_path: str = ""
    place_extract_every_n_steps: int = 5
    place_similarity_threshold: float = 0.85
    place_device: str = "cpu"
    rgm: RGMConfig = field(default_factory=RGMConfig)

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.max_context_chars < 1:
            raise ValueError("memory.top_k and memory.max_context_chars must be positive")
        if self.strategy not in {"flat", "hierarchical"}:
            raise ValueError("memory.strategy must be 'flat' or 'hierarchical'")
        if len(set(self.layers)) != len(self.layers) or any(
            layer not in {"short", "mid", "long"} for layer in self.layers
        ):
            raise ValueError("memory.layers must be a unique subset of short, mid, long")
        if self.extractor not in {"null", "yolo11"}:
            raise ValueError("memory.extractor must be 'null' or 'yolo11'")
        if self.enabled and self.extractor == "yolo11" and not self.extractor_model_path:
            raise ValueError("memory.extractor_model_path is required for yolo11")
        if not self.extractor_device:
            raise ValueError("memory.extractor_device must be nonempty")
        if not 0.0 < self.extractor_confidence <= 1.0:
            raise ValueError("memory.extractor_confidence must be in (0, 1]")
        if not isinstance(self.extract_every_n_steps, int) or self.extract_every_n_steps < 1:
            raise ValueError("memory.extract_every_n_steps must be a positive integer")
        if (
            self.enabled and self.place_enabled and self.strategy == "hierarchical"
            and "long" in self.layers and not self.place_model_path
        ):
            raise ValueError("memory.place_model_path is required for enabled place memory")
        if (
            not isinstance(self.place_extract_every_n_steps, int)
            or self.place_extract_every_n_steps < 1
        ):
            raise ValueError("memory.place_extract_every_n_steps must be positive")
        if not 0.0 < self.place_similarity_threshold <= 1.0:
            raise ValueError("memory.place_similarity_threshold must be in (0, 1]")
        if not isinstance(self.place_device, str) or not self.place_device.strip():
            raise ValueError("memory.place_device must be nonempty")
        if self.rgm.enabled and not self.enabled:
            raise ValueError("RGM requires memory_enabled")


@dataclass
class HabitatEvalConfig:
    """Settings for one evaluation run (mirrored by the ``lightnav-eval-habitat`` CLI)."""

    model_path: str
    server: str = "tcp://localhost:5555"
    backend: str = "vllm_local"
    episodes: int = -1  # <= 0: full split, stop at the first repeated episode key
    max_steps: int = 500
    # Simulation-only: when the step budget is exhausted without the policy having
    # stopped, the LAST action of the budget is an explicit STOP (zero velocity) so the
    # episode ends by ``agent_stop`` and Success/SPL are measured at the final pose
    # instead of being zeroed by the step-limit truncation. Never applies to the
    # WebSocket serving path.
    force_stop_at_max_steps: bool = True
    output_dir: str = "output/habitat_eval"
    languages: list[str] | None = None  # RxR only, e.g. ["en-US", "en-IN"]
    traj_vocab_path: str | None = None  # flat vocab dir or centroids .npy
    K: int | None = None
    horizon: int | None = None
    action_tokenizer_bundle: str | None = None  # RVQ bundle dir
    gpu_memory_utilization: float = 0.65
    max_num_seqs: int = 1
    num_history_frames: int | None = None
    aspect_mode: str = "stretch"  # stretch | keep (see InferenceConfig)
    # Per-step decode cap. Greedy decoding stops at eos, so this only truncates; 64 is
    # the reference value and leaves room for any grounding prefix + action tokens.
    max_new_tokens: int = 64
    zmq_timeout_ms: int = 600000
    verbose: bool = False
    # Visualisation (docs/VISUALIZATION.md). `save_video` needs the `video` extra and writes
    # <output_dir>/videos/<episode_id>.mp4 with one frame per policy step; `record_dir`
    # additionally records the raw episodes in the layout `lightnav-render` reads.
    save_video: bool = False
    video_fps: int = 10
    hfov_deg: float = 120.0  # agent camera, as in habitat_server/configs/*.yaml
    cam_height: float = 0.88  # metres above the floor, as in the yamls
    waypoint_dt_s: float = 0.1  # seconds per waypoint row assumed by the HUD readout only
    record_dir: str = ""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    seed: int | None = None
    strict_full_eval: bool = False
    resume: bool = False
    progress_shard_index: int = 0
    progress_shard_count: int = 1
    progress_tty: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.progress_shard_index) is not int
            or type(self.progress_shard_count) is not int
            or self.progress_shard_count < 1
            or not 0 <= self.progress_shard_index < self.progress_shard_count
        ):
            raise ValueError("progress shard index/count are invalid")


# ---------------------------------------------------------------------------
# Action decoder resolution
# ---------------------------------------------------------------------------


def _manifest_horizon(bundle_dir: Path) -> int | None:
    manifest = bundle_dir / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        horizon = json.loads(manifest.read_text()).get("horizon")
    except (OSError, ValueError):
        return None
    return int(horizon) if horizon else None


def _load_bundle(bundle_dir: Path, horizon: int | None):
    if horizon is None:
        horizon = _manifest_horizon(bundle_dir) or DEFAULT_TRAJ_HORIZON
    return load_rvq_bundle(bundle_dir, int(horizon), num_frames=0, load_cluster_ids=False)


def resolve_action_decoder(cfg: HabitatEvalConfig, bundle: Any) -> tuple[np.ndarray | None, Any]:
    """Return ``(centroids, rvq_bundle)`` with exactly one of them set.

    Resolution order:
      1. explicit ``cfg.action_tokenizer_bundle`` or ``cfg.traj_vocab_path`` (+ ``K``/``horizon``,
         defaulting to the checkpoint's ``eval_config.json`` ``tasks.vlnce`` values);
      2. the ``eval_config.json`` snapshot: a task with ``action_tokenizer.method == "rvq"``
         whose ``bundle_path`` exists, else a task whose ``traj_vocab_path`` holds the
         matching centroids file;
      3. sibling directories ``<model_path>/action_tokenizer`` / ``<model_path>/traj_vocab``.
    """
    saved = load_eval_config(cfg.model_path) or {}
    vlnce_params = get_task_params(saved, "vlnce") if saved else {}
    K = int(cfg.K) if cfg.K else int(vlnce_params.get("traj_vocab_K") or DEFAULT_TRAJ_K)
    horizon: int | None = int(cfg.horizon) if cfg.horizon else None
    if horizon is None and vlnce_params.get("predict_horizon"):
        horizon = int(vlnce_params["predict_horizon"])

    # 1. explicit
    if cfg.action_tokenizer_bundle and cfg.traj_vocab_path:
        raise ValueError("pass only one of --action_tokenizer_bundle and --traj_vocab_path")
    if cfg.action_tokenizer_bundle:
        bundle_dir = Path(cfg.action_tokenizer_bundle)
        print(f"[eval] RVQ bundle: {bundle_dir}", flush=True)
        return None, _load_bundle(bundle_dir, horizon)
    if cfg.traj_vocab_path:
        h = horizon or DEFAULT_TRAJ_HORIZON
        print(f"[eval] traj vocab centroids: {cfg.traj_vocab_path} (K={K}, H={h})", flush=True)
        return load_centroids(cfg.traj_vocab_path, K, h), None

    # 2. eval_config.json snapshot. Only the family the checkpoint declares is
    # considered (an RVQ checkpoint never falls back to flat centroids and vice
    # versa), so a mismatch surfaces here with the tried paths instead of after
    # the first env.reset().
    is_rvq = getattr(bundle, "action_method", "flat") == "rvq"
    tried: list[str] = []
    tasks = [t for t in (saved.get("tasks") or {}).values() if isinstance(t, dict)]
    for task in tasks if is_rvq else []:
        at = task.get("action_tokenizer") or {}
        if at.get("method") != "rvq":
            continue
        bundle_path = at.get("bundle_path")
        task_horizon = int(task.get("predict_horizon") or 0)
        if not bundle_path or not task_horizon:
            continue
        bundle_dir = resolve_asset_path(cfg.model_path, bundle_path)
        if (bundle_dir / "manifest.json").is_file():
            print(f"[eval] RVQ bundle from eval_config.json snapshot: {bundle_dir}", flush=True)
            return None, load_rvq_bundle(
                bundle_dir, task_horizon, num_frames=0, load_cluster_ids=False
            )
        tried.append(str(bundle_dir / "manifest.json"))
    for task in tasks if not is_rvq else []:
        vocab_path = task.get("traj_vocab_path")
        if not vocab_path:
            continue
        k = int(task.get("traj_vocab_K") or 0)
        h = int(task.get("predict_horizon") or 0)
        if not k or not h:
            continue
        candidate = resolve_asset_path(cfg.model_path, vocab_path) / _CENTROIDS_FILENAME.format(K=k, H=h)
        if candidate.is_file():
            print(
                f"[eval] traj vocab centroids from eval_config.json snapshot: {candidate}",
                flush=True,
            )
            return load_centroids(candidate, k, h), None
        tried.append(str(candidate))

    # 3. sibling dirs next to the checkpoint (same family rule)
    model_dir = Path(cfg.model_path)
    if is_rvq:
        for sibling_bundle in (model_dir / "action_tokenizer" / "vlnce", model_dir / "action_tokenizer"):
            if (sibling_bundle / "manifest.json").is_file():
                print(f"[eval] RVQ bundle: {sibling_bundle}", flush=True)
                return None, _load_bundle(sibling_bundle, horizon)
            tried.append(str(sibling_bundle / "manifest.json"))
    else:
        h = horizon or DEFAULT_TRAJ_HORIZON
        sibling_centroids = model_dir / "traj_vocab" / _CENTROIDS_FILENAME.format(K=K, H=h)
        if sibling_centroids.is_file():
            print(f"[eval] traj vocab centroids: {sibling_centroids}", flush=True)
            return load_centroids(sibling_centroids, K, h), None
        tried.append(str(sibling_centroids))

    method = getattr(bundle, "action_method", None)
    hint = (
        "--action_tokenizer_bundle <dir>"
        if method == "rvq"
        else f"--traj_vocab_path <dir-or-.npy> [--K {K} --horizon {h}]"
    )
    raise FileNotFoundError(
        "Could not resolve the action decoder for this checkpoint"
        + (f" (action_method={method!r})" if method else "")
        + f". Tried: {tried}. Pass {hint}."
    )


# ---------------------------------------------------------------------------
# Default factories (real engine / env / policy)
# ---------------------------------------------------------------------------


def _default_engine_factory(cfg: HabitatEvalConfig) -> tuple[Any, Any]:
    from lightnav.inference import InferenceConfig, build_engine

    inf_cfg = InferenceConfig(
        model_path=cfg.model_path,
        backend=cfg.backend,
        max_new_tokens=cfg.max_new_tokens,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_num_seqs=cfg.max_num_seqs,
        num_history_frames=cfg.num_history_frames,
        aspect_mode=cfg.aspect_mode,
    )
    return build_engine(inf_cfg, task_type="vlnce", max_new_tokens=cfg.max_new_tokens)


def _default_env_factory(cfg: HabitatEvalConfig) -> Any:
    from lightnav.habitat.remote_env import RemoteEnvClient

    return RemoteEnvClient(cfg.server, timeout_ms=cfg.zmq_timeout_ms)


def _apply_decode_budget(engine: Any, bundle: Any, rvq_bundle: Any) -> None:
    """Raise ``engine.max_new_tokens`` to the checkpoint's per-step token budget if needed.

    The cap only truncates (greedy decoding stops at eos), so raising it never changes
    the decoded action; a cap below grounding prefix + action tokens would cut the last
    action token and turn every step into a decode failure. Engines without a
    tokenizer (test doubles) are left untouched.
    """
    tokenizer = getattr(bundle, "tokenizer", None)
    current = getattr(engine, "max_new_tokens", None)
    if tokenizer is None or current is None:
        return
    try:
        budget = decode_token_budget(probe_grounding_tokens(tokenizer), rvq_bundle)
    except Exception as exc:  # pragma: no cover - defensive; keep the configured cap
        logger.warning("could not probe the decode token budget (%s); keeping %s", exc, current)
        return
    if budget > int(current):
        print(
            f"[eval] raising max_new_tokens {current} -> {budget} (grounding prefix + action tokens)",
            flush=True,
        )
        engine.max_new_tokens = int(budget)


def build_velocity_policy(
    cfg: HabitatEvalConfig, engine: Any, bundle: Any, info: dict[str, Any]
) -> TrajVocabVLNCEPolicy:
    """Build the policy from the velocity-control settings the env reports on reset."""
    missing = [k for k in ("habitat_time_step", "lin_vel_range", "ang_vel_range") if k not in info]
    if missing:
        raise KeyError(
            f"Habitat env did not expose velocity-control config keys: {missing}. "
            "The env server must report habitat_time_step / lin_vel_range / ang_vel_range in info."
        )
    centroids, rvq_bundle = resolve_action_decoder(cfg, bundle)
    _apply_decode_budget(engine, bundle, rvq_bundle)
    num_history_frames = (
        int(cfg.num_history_frames)
        if cfg.num_history_frames is not None
        else int(bundle.num_history_frames)
    )
    memory_manager = None
    if cfg.memory.enabled:
        from lightnav.memory.manager import MemoryManager

        extractor = None
        place_extractor = None
        if cfg.memory.extractor == "yolo11":
            from lightnav.memory.yolo_scene_extractor import YOLOSceneExtractor

            extractor = YOLOSceneExtractor(
                model_path=cfg.memory.extractor_model_path,
                device=cfg.memory.extractor_device,
                confidence=cfg.memory.extractor_confidence,
                every_n_steps=cfg.memory.extract_every_n_steps,
            )
        if (
            cfg.memory.place_enabled and cfg.memory.strategy == "hierarchical"
            and "long" in cfg.memory.layers
        ):
            from lightnav.memory.place_recognition import DINOv2PlaceDescriptorExtractor

            place_extractor = DINOv2PlaceDescriptorExtractor(
                model_path=cfg.memory.place_model_path,
                device=cfg.memory.place_device,
            )
        memory_manager = MemoryManager(
            extractor=extractor,
            top_k=cfg.memory.top_k,
            max_context_chars=cfg.memory.max_context_chars,
            strategy=cfg.memory.strategy,
            layers=cfg.memory.layers,
            place_enabled=cfg.memory.place_enabled,
            place_extractor=place_extractor,
            place_every_n_steps=cfg.memory.place_extract_every_n_steps,
            place_similarity_threshold=cfg.memory.place_similarity_threshold,
            rgm_config=cfg.memory.rgm,
        )
    return TrajVocabVLNCEPolicy(
        engine,
        centroids,
        dt=float(info["habitat_time_step"]),
        lin_vel_range=tuple(float(x) for x in info["lin_vel_range"]),
        ang_vel_range=tuple(float(x) for x in info["ang_vel_range"]),
        num_history_frames=num_history_frames,
        rvq_bundle=rvq_bundle,
        memory_enabled=cfg.memory.enabled,
        memory_manager=memory_manager,
        memory_verbose=cfg.verbose,
    )


# ---------------------------------------------------------------------------
# Visualisation: per-episode overlay videos + optional raw episode recording
# ---------------------------------------------------------------------------

_VIDEO_MODULES = ("cv2", "imageio", "imageio_ffmpeg")
_VIDEOS_SUBDIR = "videos"


def _require_video_deps() -> None:
    """Fail fast (before the engine loads) when ``save_video`` lacks the ``video`` extra."""
    missing = []
    for name in _VIDEO_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise ImportError(
            f"--save_video needs {', '.join(missing)} (missing); install the video extra: "
            "pip install 'lightnav[video]'"
        )


class _EpisodeVideo:
    """One per-episode mp4: opened on the first frame, renamed into place on ``close``.

    Frames are written under a temporary name so a crash never leaves a truncated
    file that looks finished. Every frame is padded to even dimensions (yuv420p).
    """

    def __init__(self, path: Path, fps: int) -> None:
        self.path = path
        self.fps = int(fps)
        self._partial = path.with_name(path.stem + ".partial" + path.suffix)
        self._writer: Any = None
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        from lightnav.viz import open_video_writer, pad_to_even_dimensions

        frame, _pad_right, _pad_bottom = pad_to_even_dimensions(np.ascontiguousarray(frame))
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = open_video_writer(self._partial, self.fps)
        self._writer.append_data(frame)
        self.frames += 1

    def abort(self) -> None:
        """Discard the partial file after an encoding failure."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001 - the writer is already broken
                pass
            self._writer = None
        self._partial.unlink(missing_ok=True)

    def close(self) -> Path | None:
        """Finish the file; return its final path, or None when nothing was written."""
        if self._writer is None:
            return None
        try:
            self._writer.close()
        finally:
            self._writer = None
        if self.frames == 0:
            self._partial.unlink(missing_ok=True)
            return None
        os.replace(self._partial, self.path)
        return self.path


class _EvalVisualizer:
    """Renders the overlay for the observation the policy acted on, and records raw steps.

    Built only when ``cfg.save_video`` or ``cfg.record_dir`` is set, so ``lightnav.viz``
    is imported only then. The video timeline is ``per_step`` (one frame per policy
    step); the recorder gets the same frames JPEG-encoded plus the per-step record, in
    the layout ``lightnav-render`` reads.
    """

    def __init__(self, cfg: HabitatEvalConfig, output_dir: str) -> None:
        self.cfg = cfg
        self.videos_dir = Path(output_dir) / _VIDEOS_SUBDIR
        self._video: _EpisodeVideo | None = None
        self._video_failed = False
        self._last_waypoints: np.ndarray | None = None
        self._recorder: Any = None
        self._conn: Any = None
        if cfg.save_video:
            _require_video_deps()
        from lightnav.viz import encode_jpeg_bytes, render_frame

        self._render_frame = render_frame
        self._encode_jpeg_bytes = encode_jpeg_bytes
        if cfg.record_dir:
            from lightnav.viz import EpisodeRecorder

            self._recorder = EpisodeRecorder(
                Path(cfg.record_dir),
                task="habitat",
                model_path=cfg.model_path,
                hfov_deg=float(cfg.hfov_deg),
                cam_height=float(cfg.cam_height),
                video_fps=int(cfg.video_fps),
                timeline="per_step",
                waypoint_dt_s=float(cfg.waypoint_dt_s),
            )
            self._conn = self._recorder.begin_connection(label="eval")

    @property
    def record_run_dir(self) -> Path | None:
        return getattr(self._recorder, "run_dir", None) if self._recorder is not None else None

    def begin_episode(self, episode_id: str) -> None:
        self._last_waypoints = None
        self._video_failed = False
        if self.cfg.save_video:
            self._video = _EpisodeVideo(self.videos_dir / f"{episode_id}.mp4", self.cfg.video_fps)
        if self._conn is not None:
            self._conn.begin_episode()

    def _render(
        self,
        rgb: np.ndarray,
        *,
        waypoints: np.ndarray | None,
        instruction: str,
        step: int,
        stop: bool,
        pointing: dict | None,
    ) -> np.ndarray:
        return self._render_frame(
            rgb,
            waypoints=waypoints,
            instruction=instruction,
            step=step,
            step_fps=None,
            stop=stop,
            pointing=pointing,
            hfov_deg=self.cfg.hfov_deg,
            cam_height=self.cfg.cam_height,
            dt_s=self.cfg.waypoint_dt_s,
        )

    def _write(self, frame: np.ndarray) -> None:
        if self._video is None or self._video_failed:
            return
        try:
            self._video.write(frame)
        except Exception as exc:
            # Diagnostics must not cost the evaluation: drop this episode's video and go on.
            self._video_failed = True
            logger.warning(
                "video write failed for %s (%s: %s); the episode has no video",
                self._video.path, type(exc).__name__, exc,
            )

    def step(
        self,
        rgb: np.ndarray,
        *,
        step: int,
        instruction: str,
        policy_info: dict[str, Any],
        latency_ms: float,
        **extra: Any,
    ) -> None:
        """Overlay + record one policy step; ``rgb`` is the frame the policy acted on."""
        waypoints = policy_info.get("predicted_traj")  # None after a failed decode
        if waypoints is not None:
            waypoints = np.asarray(waypoints, dtype=np.float32)
        raw_text = str(policy_info.get("raw_text", "") or "")
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        pointing = pointing_payload(raw_text, width=width, height=height)
        stop = waypoints is not None and not np.any(np.abs(waypoints) > 1e-6)
        self._last_waypoints = waypoints

        if self._video is not None:
            frame = self._render(
                rgb, waypoints=waypoints, instruction=instruction, step=step, stop=stop,
                pointing=pointing,
            )
            self._write(frame)

        if self._conn is not None:
            try:
                self._conn.record_step(
                    step=step,
                    seq=step,
                    image=self._encode_jpeg_bytes(rgb),
                    instruction=instruction,
                    waypoints=None if waypoints is None else waypoints.tolist(),
                    stop=bool(stop),
                    visible=None,
                    raw_text=raw_text,
                    latency_ms=float(latency_ms),
                    pointing=pointing,
                    **extra,
                )
            except Exception as exc:
                logger.warning("episode recording failed at step %d: %s: %s", step, type(exc).__name__, exc)

    def final_frame(self, rgb: np.ndarray, *, step: int, instruction: str) -> None:
        """Append the terminal observation (with the last prediction) so the end is visible."""
        if self._video is None or self._video.frames == 0:
            return
        waypoints = self._last_waypoints
        stop = waypoints is not None and not np.any(np.abs(waypoints) > 1e-6)
        frame = self._render(
            rgb, waypoints=waypoints, instruction=instruction, step=step, stop=stop, pointing=None
        )
        self._write(frame)

    def end_episode(self) -> str | None:
        """Close the episode; return the video path relative to ``output_dir`` if written."""
        video_rel: str | None = None
        if self._video is not None:
            written = None
            if self._video_failed:
                self._video.abort()
            else:
                try:
                    written = self._video.close()
                except Exception as exc:
                    logger.warning(
                        "could not finish %s: %s: %s", self._video.path, type(exc).__name__, exc
                    )
                    self._video.abort()
            if written is not None:
                video_rel = f"{_VIDEOS_SUBDIR}/{written.name}"
            self._video = None
        if self._conn is not None:
            try:
                self._conn.end_episode()
            except Exception as exc:
                logger.warning("could not finish the episode recording: %s: %s", type(exc).__name__, exc)
        return video_rel

    def close(self) -> None:
        self.end_episode()
        if self._recorder is not None:
            try:
                self._recorder.close()
            except Exception as exc:
                logger.warning("recorder shutdown failed: %s: %s", type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------


def _safe_scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else float(default)
    try:
        return float(value[0])
    except Exception:
        try:
            return float(value)
        except Exception:
            return float(default)


def _format_action(action: Any) -> str:
    if isinstance(action, dict):
        args = action.get("action_args", {}) or {}
        lin = float(args.get("linear_velocity", float("nan")))
        ang = float(args.get("angular_velocity", float("nan")))
        return f"VEL(lin={lin:+.2f},ang={ang:+.2f})"
    return str(action)


def run_habitat_eval(
    cfg: HabitatEvalConfig,
    *,
    env_factory: Callable[[HabitatEvalConfig], Any] | None = None,
    policy_factory: Callable[[HabitatEvalConfig, Any, Any, dict], Any] | None = None,
    engine_factory: Callable[[HabitatEvalConfig], tuple[Any, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the evaluation and return the per-episode result dicts.

    ``engine_factory(cfg) -> (engine, bundle)``, ``env_factory(cfg) -> env`` with
    ``reset()/step()/close()``, ``policy_factory(cfg, engine, bundle, info) -> policy``
    with ``reset(obs)/act(obs, info)/get_info()``. Each defaults to the real one.
    """
    engine_factory = engine_factory or _default_engine_factory
    env_factory = env_factory or _default_env_factory
    policy_factory = policy_factory or build_velocity_policy

    if cfg.strict_full_eval and (
        cfg.episodes != -1 or cfg.max_steps != 500 or cfg.force_stop_at_max_steps
        or cfg.languages
    ):
        raise ValueError(
            "formal R2R evaluation requires the full split, 500 steps, no forced STOP "
            "and no client language filter"
        )
    if cfg.resume and not cfg.strict_full_eval:
        raise ValueError("resume requires strict_full_eval and a loaded episode manifest")
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.manual_seed(cfg.seed)

    output_dir = cfg.output_dir or "output/habitat_eval"
    os.makedirs(output_dir, exist_ok=True)
    results_jsonl = os.path.join(output_dir, "results.jsonl")
    diagnostics_jsonl = os.path.join(output_dir, "memory_diagnostics.jsonl")
    if cfg.strict_full_eval and not cfg.resume and os.path.exists(results_jsonl):
        raise FileExistsError(
            f"formal output already contains results: {results_jsonl}; use --resume"
        )

    # Before the engine: a missing video dependency should fail in seconds, not after
    # the weights are loaded.
    viz = _EvalVisualizer(cfg, output_dir) if (cfg.save_video or cfg.record_dir) else None

    engine, bundle = engine_factory(cfg)
    env = env_factory(cfg)

    progress: ProgressReporter | None = None
    try:
        manifest = None
        manifest_index: dict[tuple[str, str], int] = {}
        if hasattr(env, "get_episode_manifest"):
            manifest = validate_manifest(env.get_episode_manifest())
            write_or_match_manifest(os.path.join(output_dir, "episode_manifest.json"), manifest)
            manifest_index = {
                episode_key(row): index
                for index, row in enumerate(manifest["episodes"])
            }
            print(f"Loaded {manifest['expected_count']} episodes on this shard", flush=True)
        if cfg.strict_full_eval:
            if manifest is None or manifest["split"] != "val_unseen" or not manifest["episodes"]:
                raise ValueError("formal evaluation requires a nonempty val_unseen server manifest")
        if cfg.resume:
            for path in (results_jsonl, diagnostics_jsonl):
                if repair_interrupted_jsonl_tail(path):
                    print(f"Archived an incomplete final JSONL row from {path}", flush=True)
        existing_results = (
            read_jsonl(results_jsonl)
            if cfg.resume and os.path.exists(results_jsonl) else []
        )
        if manifest is not None and existing_results:
            check_coverage(existing_results, manifest, complete=False)
        completed_keys = {
            episode_key(row, result=True)
            for row in existing_results
        }
        if cfg.resume:
            archived = archive_unfinished_diagnostics(diagnostics_jsonl, completed_keys)
            if archived:
                print(f"Archived {archived} unfinished diagnostic steps before resume")
        Path(diagnostics_jsonl).touch(exist_ok=True)
        if cfg.languages:
            logger.info(
                "Skipping numeric Episode progress: client language filter has no "
                "known post-filter manifest count"
            )
        elif manifest is not None and manifest["expected_count"] > 0:
            total = manifest["expected_count"]
            if cfg.episodes > 0:
                total = min(total, cfg.episodes)
            group_label = (
                Path(output_dir).parent.name
                if Path(output_dir).name.startswith("shard_") else "LightNav"
            )
            try:
                progress = ProgressReporter(
                    total, shard_index=cfg.progress_shard_index,
                    shard_count=cfg.progress_shard_count,
                    completed_results=existing_results, group_label=group_label,
                    tty_override=True if cfg.progress_tty else None,
                    auto_heartbeat=True,
                )
            except Exception:
                logger.exception("Progress reporting failed to initialize; evaluation continues")
    except BaseException:
        try:
            if viz is not None:
                viz.close()
        finally:
            env.close()
        raise

    sep = "=" * 60
    print(f"\n{sep}")
    print("Habitat Evaluation")
    print(sep)
    print(f"  Model:    {cfg.model_path}")
    print(f"  Server:   {cfg.server}")
    print(f"  Backend:  {cfg.backend}")
    print(f"  Episodes: {cfg.episodes if cfg.episodes > 0 else 'full split'}")
    print(f"  Output:   {output_dir}")
    lang_filter = set(cfg.languages) if cfg.languages else None
    if lang_filter:
        print(f"  Language filter: {sorted(lang_filter)}")
    if cfg.save_video:
        print(f"  Videos:   {os.path.join(output_dir, _VIDEOS_SUBDIR)} ({cfg.video_fps} fps, one frame per step)")
    if viz is not None and viz.record_run_dir is not None:
        print(f"  Record:   {viz.record_run_dir}")
    print(f"{sep}\n", flush=True)

    results: list[dict[str, Any]] = list(existing_results)
    skipped_lang = 0
    seen: set[tuple[Any, Any]] = set()
    n_run = 0
    policy = None
    start_time = time.time()

    def report_progress(method: str, *args: Any, **kwargs: Any) -> None:
        """Keep optional terminal reporting from changing evaluation results."""
        nonlocal progress
        if progress is None:
            return
        try:
            getattr(progress, method)(*args, **kwargs)
        except Exception:
            logger.exception("Progress reporting failed; evaluation continues")
            progress = None

    try:
        while True:
            obs, info = env.reset()

            # (scene_id, episode_id): episode ids are scene-local in HM3D ObjectNav, so
            # a repeat of the pair (not the id alone) marks the env cycling back to the start.
            habitat_ep_id = str(info.get("episode_id", ""))
            habitat_scene_id = str(info.get("scene_id", ""))
            ep_key = episode_key({
                "scene_id": habitat_scene_id, "episode_id": habitat_ep_id,
            })
            if ep_key in seen:
                print(f"Full split completed: {len(seen)} unique episodes seen.", flush=True)
                if cfg.strict_full_eval and len(seen) != manifest["expected_count"]:
                    raise ValueError(
                        f"episode iterator wrapped after {len(seen)} of "
                        f"{manifest['expected_count']} loaded episodes"
                    )
                break
            if cfg.strict_full_eval and ep_key not in manifest_index:
                raise ValueError(f"episode iterator returned an unknown episode: {ep_key}")
            seen.add(ep_key)

            # Language-filtered episodes are skipped without stepping and do not
            # count toward `episodes` or the episode counter.
            if lang_filter and info.get("language", "") not in lang_filter:
                skipped_lang += 1
                continue

            if ep_key in completed_keys:
                print(f"Skipping already completed episode {ep_key}", flush=True)
                continue

            episode_number = manifest_index[ep_key] if cfg.strict_full_eval else n_run
            episode_id = f"episode_{episode_number:03d}"
            print(
                f"Episode {n_run + 1} ({episode_id}) [{habitat_scene_id} / {habitat_ep_id}]...",
                flush=True,
            )

            report_progress("start", habitat_scene_id, habitat_ep_id,
                            max_steps=cfg.max_steps)
            episode_start = time.monotonic()
            instruction = extract_instruction(obs)
            if policy is None:
                policy = policy_factory(cfg, engine, bundle, info)
            if cfg.memory.rgm.enabled and isinstance(policy, TrajVocabVLNCEPolicy):
                policy.reset(
                    obs, memory_episode_id=json.dumps(
                        [habitat_scene_id, habitat_ep_id], separators=(",", ":"),
                    ),
                )
            else:
                policy.reset(obs)
            if viz is not None:
                viz.begin_episode(episode_id)

            final_success: Any = False
            forced_stop_used = False
            terminated = False
            truncated = False
            min_distance_to_goal = float("inf")
            step = -1
            video_rel: str | None = None
            episode_diagnostics: list[dict[str, Any]] = []
            inference_latencies: list[float] = []
            try:
                for step in range(cfg.max_steps):
                    if obs.get("rgb") is None:
                        break
                    t_act = time.monotonic()
                    forced_stop = bool(
                        cfg.force_stop_at_max_steps
                        and step == cfg.max_steps - 1
                        and hasattr(policy, "stop_action")
                    )
                    if forced_stop:
                        # Step budget exhausted: end the episode with an explicit STOP.
                        action = policy.stop_action()
                        forced_stop_used = True
                    else:
                        action = policy.act(obs, info)
                    act_ms = (time.monotonic() - t_act) * 1000.0
                    policy_info: dict[str, Any] = {}
                    if cfg.verbose or viz is not None:
                        policy_info = policy.get_info() if hasattr(policy, "get_info") else {}
                    if cfg.verbose:
                        print(
                            f"    Step {step}: {_format_action(action)}"
                            f" raw={policy_info.get('raw_text', '')!r}"
                            f" traj_id={policy_info.get('cluster_id')}"
                            f" wp_idx={policy_info.get('action_waypoint_index')}",
                            flush=True,
                        )
                    if viz is not None:
                        # The overlay belongs to the frame the policy acted on, so it is
                        # rendered before env.step replaces the observation.
                        viz.step(
                            obs["rgb"],
                            step=step,
                            instruction=instruction,
                            policy_info=policy_info,
                            latency_ms=act_ms,
                            episode_id=episode_id,
                            habitat_episode_id=str(habitat_ep_id),
                            scene_id=str(habitat_scene_id),
                        )
                    obs, _reward, terminated, truncated, info = env.step(action)

                    model_ms = getattr(policy, "last_model_inference_ms", None)
                    if model_ms is not None:
                        inference_latencies.append(float(model_ms))
                    diagnostic = getattr(policy, "last_memory_diagnostic", None)
                    row = dict(diagnostic) if isinstance(diagnostic, dict) else {}
                    row.update({
                        "episode_key": {
                            "scene_id": habitat_scene_id,
                            "episode_id": habitat_ep_id,
                        },
                        "scene_id": habitat_scene_id,
                        "habitat_episode_id": habitat_ep_id,
                        "rollout_idx": 0,
                        "step": step,
                        "act_latency_ms": act_ms,
                        "inference_latency_ms": model_ms,
                        "prompt_chars": getattr(policy, "last_prompt_chars", None),
                        "memory_enabled": cfg.memory.enabled,
                    })
                    episode_diagnostics.append(row)

                    dtg = _safe_scalar(info.get("distance_to_goal"), default=float("inf"))
                    if dtg < min_distance_to_goal:
                        min_distance_to_goal = dtg

                    if not (terminated or truncated):
                        report_progress("update_step", step + 1, max_steps=cfg.max_steps)

                    if terminated or truncated:
                        final_success = info.get("success", False)
                        break
                if viz is not None and obs.get("rgb") is not None:
                    viz.final_frame(obs["rgb"], step=step + 1, instruction=instruction)
            finally:
                try:
                    if viz is not None:
                        video_rel = viz.end_episode()
                finally:
                    if hasattr(policy, "close_episode"):
                        policy.close_episode()

            if cfg.strict_full_eval:
                if not (terminated or truncated):
                    raise ValueError(f"formal episode ended without Habitat termination: {ep_key}")
                required_metrics = (
                    "success", "oracle_success", "spl", "ndtw", "distance_to_goal",
                )
                for name in required_metrics:
                    if name not in info or not math.isfinite(
                        _safe_scalar(info[name], default=float("nan"))
                    ):
                        raise ValueError(f"formal episode {ep_key} lacks measured {name}")

            oracle_success_metric = info.get("oracle_success")
            if oracle_success_metric is None:
                fallback_success_distance = 0.1 if info.get("object_category") else 3.0
                oracle_success = min_distance_to_goal < fallback_success_distance
            else:
                oracle_success = _safe_scalar(oracle_success_metric) > 0.0

            result = {
                "episode_id": episode_id,
                "habitat_episode_id": habitat_ep_id,
                "raw_episode_id": str(info.get("raw_episode_id", "")),
                "scene_id": habitat_scene_id,
                "rollout_idx": 0,
                "success": final_success,
                "oracle_success": oracle_success,
                "spl": info.get("spl", 0.0),
                "ndtw": info.get("ndtw", 0.0),
                "soft_spl": float(info.get("soft_spl", 0.0)),
                "object_category": str(info.get("object_category", "")),
                "steps": step + 1,
                "final_distance": info.get("distance_to_goal", float("inf")),
                "min_distance": min_distance_to_goal,
                "instruction": instruction,
                "termination_reason": info.get("termination_reason", "unknown"),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "forced_stop": forced_stop_used,
                "termination_details": info.get("termination_details", {}),
                "episode_time_sec": time.monotonic() - episode_start,
                "inference_latency_ms_mean": (
                    sum(inference_latencies) / len(inference_latencies)
                    if inference_latencies else None
                ),
            }
            if video_rel is not None:
                result["video"] = video_rel
            if cfg.strict_full_eval and result["steps"] < 1:
                raise ValueError(f"formal episode has no completed step: {ep_key}")
            with open(diagnostics_jsonl, "a", encoding="utf-8") as f:
                for row in episode_diagnostics:
                    f.write(json.dumps(make_json_safe(row)) + "\n")
                f.flush()
                os.fsync(f.fileno())
            with open(results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(make_json_safe(result)) + "\n")
                f.flush()
                os.fsync(f.fileno())
            results.append(result)
            completed_keys.add(ep_key)
            report_progress("complete", result)

            status = "OK" if result["success"] else "FAIL"
            print(
                f"  [{status}] success={bool(result['success'])}, "
                f"oracle={'True' if oracle_success else 'False'}, "
                f"min_dist={min_distance_to_goal:.2f}m, "
                f"steps={result['steps']}, "
                f"dist={float(result['final_distance']):.2f}m, "
                f"SPL={float(result['spl']):.3f}, "
                f"NDTW={float(result['ndtw']):.3f}",
                flush=True,
            )

            n_run += 1
            if cfg.episodes > 0 and n_run >= cfg.episodes:
                print(f"Reached --episodes {cfg.episodes}.", flush=True)
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        report_progress("abort")
        try:
            if policy is not None and hasattr(policy, "close_episode"):
                policy.close_episode()
        finally:
            try:
                if viz is not None:
                    viz.close()
            finally:
                env.close()

    elapsed = time.time() - start_time
    if cfg.strict_full_eval:
        check_coverage(results, manifest, complete=True)
    report_progress("finish")
    if skipped_lang > 0:
        print(f"\nSkipped {skipped_lang} episodes due to language filter.")

    extra_info = {
        "model": cfg.model_path,
        "backend": cfg.backend,
        "total_episode_time_sec": sum(
            float(row["episode_time_sec"])
            for row in results if row.get("episode_time_sec") is not None
        ),
        "resumed": cfg.resume,
    }
    if any(r.get("object_category") for r in results):
        print_objectnav_summary(results, elapsed, output_dir, extra_info=extra_info)
    else:
        print_vlnce_summary(results, elapsed, output_dir, extra_info=extra_info)
    return results
