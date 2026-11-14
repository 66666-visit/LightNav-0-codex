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
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.v3_config import RGMV3Config
from lightnav.memory.v4_config import RGMV4Config

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
        "--trace_steps", action="store_true",
        help="For --episodes 1, save each RGB frame and an evaluator-side step_trace.jsonl "
        "with the human prompt text, injected memory, model tokens and decoded action; "
        "requires a fresh explicit output directory.",
    )
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
        "--progress_shard_index", type=int, default=_DEFAULTS.progress_shard_index,
        help="Zero-based shard index shown in live Episode progress (default: 0).",
    )
    p.add_argument(
        "--progress_shard_count", type=int, default=_DEFAULTS.progress_shard_count,
        help="Number of shards shown in live Episode progress (default: 1).",
    )
    p.add_argument(
        "--progress_tty", action="store_true",
        help="Use an in-place progress display when stderr is forwarded from a TTY.",
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
    p.add_argument("--rgm_enabled", action="store_true",
                   help="Enable Reliability-Gated Memory within hierarchical memory.")
    p.add_argument("--rgm_v3_enabled", action="store_true",
                   help="Enable the independent evidence-driven RGM V3 backend.")
    p.add_argument("--rgm_v3_scope", choices=("episode_local", "scene_persistent"),
                   default="episode_local",
                   help="V3 online memory scope; default episode_local.")
    p.add_argument("--rgm_v3_prompt_max_records", type=int, default=3,
                   help="Maximum complete V3 memory statements in the prompt.")
    p.add_argument("--rgm_v3_prompt_max_chars", type=int, default=400,
                   help="Maximum characters in the V3 memory block.")
    p.add_argument("--rgm_v4_enabled", action="store_true",
                   help="Enable selective RGM V4 fusion over the frozen LightNav policy.")
    p.add_argument("--rgm_v4_scope", choices=("episode_local", "scene_persistent"),
                   default="episode_local", help="V4 online memory scope.")
    p.add_argument("--rgm_v4_prompt_max_records", type=int, default=3,
                   help="Maximum complete V4 memory facts in the prompt.")
    p.add_argument("--rgm_v4_prompt_max_chars", type=int, default=400,
                   help="Maximum characters in the V4 memory block.")
    p.add_argument("--rgm_v4_revisit_enabled", action=argparse.BooleanOptionalAction,
                   default=True, help="Audit DINO visual revisit candidates in V4.")
    p.add_argument("--local_graph_enabled", action="store_true",
                   help="Enable independent local graph navigation context.")
    p.add_argument("--local_graph_scope", choices=("episode_local", "scene_persistent"),
                   default="episode_local", help="Local graph online memory scope.")
    p.add_argument("--local_graph_prompt_max_chars", type=int, default=320,
                   help="Maximum characters in a local graph navigation card.")
    p.add_argument("--local_graph_max_landmarks", type=int, default=2,
                   help="Maximum instruction-related landmarks in one local card.")
    p.add_argument("--local_graph_reactivation_steps", type=int, default=8,
                   help="Earliest step interval for reactivating an unchanged local card.")
    for name, default in (
        ("step0_conditional", True), ("write_gate_enabled", True),
        ("retrieval_gate_enabled", True), ("prompt_gate_enabled", True),
        ("yolo_temporal_enabled", False), ("place_two_stage_enabled", False),
        ("topology_motion_validation_enabled", False),
    ):
        p.add_argument(f"--rgm_{name}", action=argparse.BooleanOptionalAction,
                       default=default, help=f"RGM {name.replace('_', ' ')} switch.")
    p.add_argument("--rgm_adaptive_extraction_enabled", action="store_true",
                   help="Reserved; V1 rejects adaptive extraction.")
    for name, value in (
        ("min_candidate_confidence", 0.25),
        ("min_instruction_landmark_confidence", 0.60),
        ("place_global_candidate_min", 0.75), ("place_global_new_max", 0.65),
        ("place_local_cosine_min", 0.70), ("place_local_inlier_min", 0.50),
        ("place_min_image_quality", 0.20),
        ("place_global_margin_min", 0.03),
        ("place_new_temporal_similarity_min", 0.85),
        ("place_ransac_reprojection_threshold", 0.08),
        ("topology_visual_delta_min", 0.35),
    ):
        p.add_argument(f"--rgm_{name}", type=float, default=value,
                       help=f"Uncalibrated RGM {name.replace('_', ' ')} threshold.")
    for name, value in (
        ("temporal_window", 3), ("temporal_min_support", 2),
        ("place_top_k", 3), ("place_local_min_matches", 8),
        ("place_keyframes_max", 3), ("prompt_max_records", 3),
        ("prompt_max_chars", 400),
    ):
        p.add_argument(f"--rgm_{name}", type=int, default=value,
                       help=f"RGM {name.replace('_', ' ')} setting.")

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
        trace_steps=args.trace_steps,
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
            rgm=RGMConfig(
                enabled=args.rgm_enabled,
                step0_conditional=args.rgm_step0_conditional,
                write_gate_enabled=args.rgm_write_gate_enabled,
                retrieval_gate_enabled=args.rgm_retrieval_gate_enabled,
                prompt_gate_enabled=args.rgm_prompt_gate_enabled,
                yolo_temporal_enabled=args.rgm_yolo_temporal_enabled,
                place_two_stage_enabled=args.rgm_place_two_stage_enabled,
                topology_motion_validation_enabled=args.rgm_topology_motion_validation_enabled,
                adaptive_extraction_enabled=args.rgm_adaptive_extraction_enabled,
                min_candidate_confidence=args.rgm_min_candidate_confidence,
                min_instruction_landmark_confidence=args.rgm_min_instruction_landmark_confidence,
                temporal_window=args.rgm_temporal_window,
                temporal_min_support=args.rgm_temporal_min_support,
                place_top_k=args.rgm_place_top_k,
                place_global_candidate_min=args.rgm_place_global_candidate_min,
                place_global_new_max=args.rgm_place_global_new_max,
                place_local_cosine_min=args.rgm_place_local_cosine_min,
                place_local_min_matches=args.rgm_place_local_min_matches,
                place_local_inlier_min=args.rgm_place_local_inlier_min,
                place_min_image_quality=args.rgm_place_min_image_quality,
                place_global_margin_min=args.rgm_place_global_margin_min,
                place_new_temporal_similarity_min=args.rgm_place_new_temporal_similarity_min,
                place_ransac_reprojection_threshold=args.rgm_place_ransac_reprojection_threshold,
                place_keyframes_max=args.rgm_place_keyframes_max,
                topology_visual_delta_min=args.rgm_topology_visual_delta_min,
                prompt_max_records=args.rgm_prompt_max_records,
                prompt_max_chars=args.rgm_prompt_max_chars,
            ),
            rgm_v3=RGMV3Config(
                enabled=args.rgm_v3_enabled, scope=args.rgm_v3_scope,
                prompt_max_records=args.rgm_v3_prompt_max_records,
                prompt_max_chars=args.rgm_v3_prompt_max_chars,
            ),
            rgm_v4=RGMV4Config(
                enabled=args.rgm_v4_enabled, scope=args.rgm_v4_scope,
                prompt_max_records=args.rgm_v4_prompt_max_records,
                prompt_max_chars=args.rgm_v4_prompt_max_chars,
                revisit_enabled=args.rgm_v4_revisit_enabled,
            ),
            local_graph=LocalGraphConfig(
                enabled=args.local_graph_enabled, scope=args.local_graph_scope,
                prompt_max_chars=args.local_graph_prompt_max_chars,
                max_landmarks=args.local_graph_max_landmarks,
                reactivation_steps=args.local_graph_reactivation_steps,
            ),
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
        progress_shard_index=args.progress_shard_index,
        progress_shard_count=args.progress_shard_count,
        progress_tty=args.progress_tty,
    )


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(raw_args)
    if (args.rgm_v3_enabled or args.rgm_v4_enabled or args.local_graph_enabled) and not any(
        token == "--output_dir" or token.startswith("--output_dir=")
        for token in raw_args
    ):
        parser.error("RGM V3/V4/local graph requires an explicit --output_dir")
    run_habitat_eval(config_from_args(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
