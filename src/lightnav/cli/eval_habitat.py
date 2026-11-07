"""``lightnav-eval-habitat``: evaluate a checkpoint against a running Habitat env server.

Example::

    lightnav-eval-habitat --model_path /path/to/checkpoint \
        --server tcp://localhost:5555 --episodes -1 --output_dir output/r2r

See ``docs/EVAL_HABITAT.md`` for the full guide.
"""

from __future__ import annotations

import argparse
import os
import sys

from lightnav.habitat.runner import HabitatEvalConfig, MemoryConfig, run_habitat_eval

# Argparse defaults come from the dataclass so the CLI and the Python API cannot drift.
_DEFAULTS = HabitatEvalConfig(model_path="")


def _memory_layers(value: str) -> tuple[str, ...]:
    """Parse a subset of the three independently enabled hierarchy layers."""
    if value.strip().casefold() == "none":
        return ()
    layers = tuple(part.strip().casefold() for part in value.split(","))
    if not layers or len(set(layers)) != len(layers) or any(
        layer not in {"short", "mid", "long"} for layer in layers
    ):
        raise argparse.ArgumentTypeError("memory layers must be short,mid,long or a subset")
    return layers


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lightnav-eval-habitat",
        description="Run VLN-CE (R2R/RxR) or ObjectNav (HM3D/OVON) evaluation against a "
        "Habitat env server over ZMQ.",
    )
    p.add_argument("--model_path", required=True, help="Checkpoint directory.")
    p.add_argument("--server", default=_DEFAULTS.server, help="Habitat env server address.")
    p.add_argument("--backend", choices=["hf", "vllm_local"], default=_DEFAULTS.backend)
    p.add_argument(
        "--episodes",
        type=int,
        default=_DEFAULTS.episodes,
        help="Number of episodes to run; <= 0 runs the full split (until the env wraps).",
    )
    p.add_argument(
        "--max_steps", type=int, default=_DEFAULTS.max_steps, help="Per-episode step cap."
    )
    p.add_argument(
        "--no_force_stop",
        action="store_true",
        help="Do not send an explicit STOP as the last action of the step budget; let the "
        "env truncate instead (default: force STOP at --max_steps).",
    )
    p.add_argument("--output_dir", default=_DEFAULTS.output_dir)
    p.add_argument(
        "--seed", type=int, default=_DEFAULTS.seed,
        help="Set the client inference random seed; server seed is set separately.",
    )
    p.add_argument(
        "--strict_full_eval", action="store_true",
        help="Require all loaded val_unseen episodes once with the standard 500-step cap.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip completed episode IDs already recorded in this output directory.",
    )
    p.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="RxR only: keep episodes whose info['language'] is in this list, e.g. en-US en-IN.",
    )
    p.add_argument(
        "--traj_vocab_path",
        default=None,
        help="Flat trajectory vocabulary: directory holding centroids_whole_chunk_K{K}_h{H}.npy "
        "or the .npy file itself.",
    )
    p.add_argument("--K", type=int, default=None, help="Vocabulary size for --traj_vocab_path.")
    p.add_argument("--horizon", type=int, default=None, help="Trajectory horizon H.")
    p.add_argument(
        "--action_tokenizer_bundle",
        default=None,
        help="RVQ action tokenizer bundle directory (env ACTION_TOKENIZER_BUNDLE is used when "
        "neither this flag nor --traj_vocab_path is given).",
    )
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=_DEFAULTS.gpu_memory_utilization,
        help="vLLM GPU memory fraction (vllm_local backend).",
    )
    p.add_argument("--max_num_seqs", type=int, default=_DEFAULTS.max_num_seqs)
    p.add_argument(
        "--num_history_frames",
        type=int,
        default=None,
        help="Override the checkpoint's history window (default: from eval_config.json).",
    )
    p.add_argument(
        "--aspect_mode",
        choices=["stretch", "keep"],
        default=_DEFAULTS.aspect_mode,
        help="stretch frames to the checkpoint's video_size (default) or keep the aspect ratio.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=_DEFAULTS.max_new_tokens,
        help="Per-step decode cap; raised automatically if the checkpoint's grounding prefix "
        "plus action tokens need more.",
    )
    p.add_argument("--zmq_timeout_ms", type=int, default=_DEFAULTS.zmq_timeout_ms)
    p.add_argument("--verbose", action="store_true", help="Print one line per step.")
    p.add_argument(
        "--memory_enabled",
        action="store_true",
        help="Enable episode-local explicit memory in the VLN policy (default: off).",
    )
    p.add_argument(
        "--memory_top_k", type=int, default=_DEFAULTS.memory.top_k,
        help="Maximum number of retrieved memory records per inference step.",
    )
    p.add_argument(
        "--memory_max_context_chars", type=int, default=_DEFAULTS.memory.max_context_chars,
        help="Character cap for the memory text inserted into the VLN prompt.",
    )
    p.add_argument(
        "--memory_strategy", choices=["flat", "hierarchical"],
        default=_DEFAULTS.memory.strategy,
        help="Memory retrieval strategy (default: flat for Retriever V2 reproducibility).",
    )
    p.add_argument(
        "--memory_layers", type=_memory_layers,
        default=_DEFAULTS.memory.layers,
        help="Comma-separated hierarchical layers, e.g. short,mid,long or short,long; "
        "use none to disable all (default: short,mid,long).",
    )
    p.add_argument(
        "--memory_extractor", choices=["null", "yolo11"], default=_DEFAULTS.memory.extractor,
        help="RGB semantic extractor (default: null; yolo11 loads a local model).",
    )
    p.add_argument(
        "--memory_extractor_model_path", default=_DEFAULTS.memory.extractor_model_path,
        help="Local YOLO11n weights file; required when memory and yolo11 are enabled.",
    )
    p.add_argument(
        "--memory_extractor_device", default=_DEFAULTS.memory.extractor_device,
        help="Ultralytics inference device, e.g. cpu or cuda:0.",
    )
    p.add_argument(
        "--memory_extractor_confidence", type=float,
        default=_DEFAULTS.memory.extractor_confidence,
        help="Minimum YOLO object detection confidence.",
    )
    p.add_argument(
        "--memory_extract_every_n_steps", type=int,
        default=_DEFAULTS.memory.extract_every_n_steps,
        help="Run detection at step 0 and every N steps thereafter (default: 5).",
    )
    p.add_argument(
        "--memory_place_enabled", action="store_true",
        help="Enable DINOv2 visual place graph in hierarchical long memory (default: off).",
    )
    p.add_argument(
        "--memory_place_model_path", default=_DEFAULTS.memory.place_model_path,
        help="Existing local Hugging Face DINOv2-Small model directory; never downloaded.",
    )
    p.add_argument(
        "--memory_place_extract_every_n_steps", type=int,
        default=_DEFAULTS.memory.place_extract_every_n_steps,
        help="Extract a place descriptor at step 0 and every N steps (default: 5).",
    )
    p.add_argument(
        "--memory_place_similarity_threshold", type=float,
        default=_DEFAULTS.memory.place_similarity_threshold,
        help="Cosine match threshold (initial 0.85; requires Habitat calibration).",
    )
    p.add_argument(
        "--memory_place_device", default=_DEFAULTS.memory.place_device,
        help="DINOv2 inference device, e.g. cpu or cuda:0 (default: cpu).",
    )

    # Visualisation (docs/VISUALIZATION.md); needs `pip install 'lightnav[video]'`.
    viz = p.add_argument_group("visualisation")
    viz.add_argument(
        "--save_video",
        action="store_true",
        help="Write <output_dir>/videos/<episode_id>.mp4 per episode: the agent's RGB frames "
        "with the predicted trajectory, pointing markers and a HUD (one frame per step).",
    )
    viz.add_argument(
        "--video_fps",
        type=int,
        default=_DEFAULTS.video_fps,
        help="Playback frame rate of the saved videos (one policy step per frame).",
    )
    viz.add_argument(
        "--hfov_deg",
        type=float,
        default=_DEFAULTS.hfov_deg,
        help="Horizontal FOV of the agent camera, for the trajectory overlay (the shipped "
        "server yamls use 120).",
    )
    viz.add_argument(
        "--cam_height",
        type=float,
        default=_DEFAULTS.cam_height,
        help="Camera height above the floor in metres, for the trajectory overlay (0.88 in "
        "the shipped yamls).",
    )
    viz.add_argument(
        "--waypoint_dt_s",
        type=float,
        default=_DEFAULTS.waypoint_dt_s,
        help="Seconds per waypoint row assumed by the HUD velocity readout.",
    )
    viz.add_argument(
        "--record_dir",
        default=_DEFAULTS.record_dir,
        help="Also record the raw episodes (frames + per-step JSON) under this directory, in "
        "the layout `lightnav-render` reads. Empty (default) = off.",
    )
    return p


def config_from_args(args: argparse.Namespace) -> HabitatEvalConfig:
    bundle = args.action_tokenizer_bundle
    if bundle is None and not args.traj_vocab_path:
        bundle = os.environ.get("ACTION_TOKENIZER_BUNDLE") or None
    args.action_tokenizer_bundle = bundle
    return HabitatEvalConfig(
        model_path=args.model_path,
        server=args.server,
        backend=args.backend,
        episodes=args.episodes,
        max_steps=args.max_steps,
        force_stop_at_max_steps=not args.no_force_stop,
        output_dir=args.output_dir,
        languages=args.languages,
        traj_vocab_path=args.traj_vocab_path,
        K=args.K,
        horizon=args.horizon,
        action_tokenizer_bundle=args.action_tokenizer_bundle,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        num_history_frames=args.num_history_frames,
        aspect_mode=args.aspect_mode,
        max_new_tokens=args.max_new_tokens,
        zmq_timeout_ms=args.zmq_timeout_ms,
        verbose=args.verbose,
        memory=MemoryConfig(
            enabled=args.memory_enabled,
            top_k=args.memory_top_k,
            max_context_chars=args.memory_max_context_chars,
            strategy=args.memory_strategy,
            layers=args.memory_layers,
            extractor=args.memory_extractor,
            extractor_model_path=args.memory_extractor_model_path,
            extractor_device=args.memory_extractor_device,
            extractor_confidence=args.memory_extractor_confidence,
            extract_every_n_steps=args.memory_extract_every_n_steps,
            place_enabled=args.memory_place_enabled,
            place_model_path=args.memory_place_model_path,
            place_extract_every_n_steps=args.memory_place_extract_every_n_steps,
            place_similarity_threshold=args.memory_place_similarity_threshold,
            place_device=args.memory_place_device,
        ),
        save_video=args.save_video,
        video_fps=args.video_fps,
        hfov_deg=args.hfov_deg,
        cam_height=args.cam_height,
        waypoint_dt_s=args.waypoint_dt_s,
        record_dir=args.record_dir,
        seed=args.seed,
        strict_full_eval=args.strict_full_eval,
        resume=args.resume,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_habitat_eval(config_from_args(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
