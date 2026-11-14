# Model assets and parameters

Checkpoint layout, the action decoder, `eval_config.json` and every server / CLI parameter.

## Checkpoint

A Hugging Face directory (or a directory containing `hf_ckpt/`) with `config.json`,
`model*.safetensors`, `tokenizer*`, `processor_config.json`, and ideally
`eval_config.json`. The architecture must be stock `Qwen3VLForConditionalGeneration`; the
trajectory / action tokens are ordinary rows of the embedding table.

## Action decoder

An **RVQ action-tokenizer bundle**: a directory with `manifest.json`, the per-level
codebook `.npy` files and the `jacobian_weights` file it names. The released checkpoints
ship it as `action_tokenizer/`, referenced from `eval_config.json`, so it is found without
flags; `--action_tokenizer_bundle` overrides it.

## `eval_config.json` (processing parameters)

If present next to or above the checkpoint (the search walks up four parents, so
`hf_ckpt/`, the `global_step_*` directory and the run root are all covered), it supplies the
processing parameters that must match training and cannot be guessed:

```json
{
  "version": 1,
  "common": {"video_size": [224, 320], "pool_enable": true, "pool_spatial": 2,
             "pool_mode": "avg", "pool_stage": "pre_vit"},
  "tasks": {
    "trackvla": {"num_history_frames": 64, "predict_horizon": 10, "video_fps": 4,
                 "slowfast_tiers": null,
                 "action_tokenizer": {"method": "rvq", "bundle_path": "/path/to/action_tokenizer"}},
    "vlnce":    {"num_history_frames": 64, "predict_horizon": 10, "video_fps": 4,
                 "action_tokenizer": {"method": "rvq", "bundle_path": "action_tokenizer"}}
  }
}
```

| Key | Meaning |
|---|---|
| `common.video_size` | model input frame size `[H, W]`; every frame is resized to it |
| `common.pool_*` | spatial pooling of vision tokens (`pool_stage` `pre_vit` or `post_vit`) |
| `tasks.<task>.num_history_frames` | history window (frames) fed to the model per step |
| `tasks.<task>.predict_horizon` | trajectory horizon `H` (rows per chunk) |
| `tasks.<task>.video_fps` | frame rate the checkpoint was trained at (send frames at this rate or faster; waypoint rows carry no time base of their own, see [DEPLOYMENT.md](DEPLOYMENT.md)) |
| `tasks.<task>.slowfast_tiers` | optional multi-rate history layout (SlowFast checkpoints keep the whole episode) |
| `tasks.<task>.action_tokenizer` | decoder snapshot: `{"method": "rvq", "bundle_path"}`; relative paths resolve against the config's directory |

`tasks` is keyed by task: `trackvla` for tracking (`--task tracking`), `vlnce` for
navigation (`--task vln`, and the Habitat evaluation). Values are read automatically; the CLIs'
`--num_history_frames` (and `--pool_spatial`, where offered) override them. Without the file the code
falls back to conservative defaults (`num_history_frames=16`, no pooling,
`video_size=(224, 320)`) which will not match a trained checkpoint — treat a missing
`eval_config.json` as a warning sign. Full schema: `src/lightnav/eval_config.py`.


### Shipping the action decoder with the checkpoint

`tasks.<task>.action_tokenizer.bundle_path` may be a **relative path**; it is resolved
against the directory holding `eval_config.json` (falling back to the checkpoint
directory). The released checkpoints ship their decoder this way:

```
hf_ckpt/
  config.json, model-*.safetensors, tokenizer*, processor_config.json
  eval_config.json          # bundle_path: "action_tokenizer" (both tasks)
  action_tokenizer/         # manifest.json + codebook_l*.npy + jacobian_weights.npy (+ alpha)
```

With such a layout `lightnav-serve`, `lightnav-predict`, `lightnav-eval-habitat` and
`build_tracking_agent()` need no `--action_tokenizer_bundle`: the
decoder is taken from the task entry that matches the server task (`--task tracking` →
`trackvla`, `--task vln` → `vlnce`), then from the other tasks, then from the sibling
directories `action_tokenizer/<task>` or `action_tokenizer/`. The explicit flag always
wins.

## Server / CLI parameters

`lightnav-eval-habitat` accepts `--memory_enabled` (default off), exposed as
`HabitatEvalConfig.memory.enabled`. `--memory_top_k` (default 5) and
`--memory_max_context_chars` (default 600) bound the retrieved text. The original
instruction, video segments, and trajectory/RVQ output contract stay intact.
With memory off, the original inference call is used. The extractor defaults to
`--memory_extractor null`, which produces no RGB semantic labels. To detect local
YOLO11n object classes, install the optional `yolo11` extra and pass
`--memory_extractor yolo11 --memory_extractor_model_path /local/yolo11n.pt`.
`--memory_extractor_device` defaults to `cpu`, `--memory_extractor_confidence` to
`0.25`, and `--memory_extract_every_n_steps` to `5` (steps 0, 5, 10, ...).
`--memory_strategy` defaults to `flat`, preserving the existing Retriever V2
path. Set `--memory_strategy hierarchical` to query the independent short,
mid, and long semantic stores before fusion. `--memory_layers` defaults to
`short,mid,long`; any nonempty subset is valid (`short`, `mid`, `long`,
`short,mid`, `short,long`, `mid,long`, or `short,mid,long`). `none` returns an
empty hierarchical context. These layer flags have no effect while memory is
off or the strategy is `flat`.
`--memory_place_enabled` selects an optional DINOv2-Small visual place graph
for the hierarchical `long` layer; it is off by default and has no effect when
memory, hierarchical retrieval, or `long` is off. Supply an existing local Hugging
Face directory with `--memory_place_model_path`; no model is downloaded. The
descriptor is extracted at step 0 and every `--memory_place_extract_every_n_steps`
(default 5). `--memory_place_similarity_threshold` defaults to **0.85**, an
initial cosine threshold that needs calibration on Habitat RGB observations.
`--memory_place_device` defaults to `cpu` and accepts `cuda` or `cuda:N` when
available. The graph records visual place hypotheses and observed transitions,
not verified rooms, physical object identities, or traversable connections.
`--verbose` prints a `[memory]` JSON line per policy step with extraction,
event, graph, retrieval, per-layer, and injected-prompt diagnostics. See
[MEMORY_MVP.md](MEMORY_MVP.md) for evidence rules and a one-episode command.

For a single-Episode audit, pass `--episodes 1 --trace_steps --output_dir <fresh-dir>`.
This writes `<output_dir>/step_trace.jsonl` and the corresponding unmodified
`<output_dir>/rgb/step_0000.png` frames. Each step records the policy's human
prompt text (reconstructed with the same sample builder in baseline mode),
memory text in that prompt, confirmed model-visible memory
when the token audit can establish it, raw pointing and RVQ token strings,
decoded waypoints and velocity action, raw RGB pixel checksum, plus evaluator-only pre/post distance
and pose when the Habitat server supplies one. The bundled R2R server includes
`agent_pose` (position and quaternion) when simulator state is available;
otherwise the trace records `null`.
`action_divergence_vs_baseline`, `invalid_exploration`, and `wrong_turn` remain `null` until matched
traces are compared with each other and the reference route. The trace never
feeds evaluator distance or pose into the policy. `--trace_steps` requires
`--episodes 1`, an explicit output directory, and a fresh trace/results path;
it cannot be combined with `--resume`.

For formal R2R validation, use [MEMORY_ABLATION.md](MEMORY_ABLATION.md).
`scripts/run_memory_ablation.sh` requires `MODEL_PATH`, `YOLO_MODEL_PATH`, and
`DINO_MODEL_PATH` for the four-arm or topology A/B experiment. `--output-root DIR`
creates a new run. `--groups baseline_off,hierarchical_topology` selects only
the formal A/B pair; omitting it runs all four arms. `--resume DIR` restores
the recorded group selection and rejects changes to it. The launcher fixes
`EPISODES=-1`, `MAX_STEPS=500`, `NO_FORCE_STOP=1`, and `SPLIT=val_unseen`.
`SEED` defaults to 42 and `GPU_IDS` to 0. The runner checks the actual loaded
Episode manifest against exactly `EXPECTED_EPISODES=1839`, then verifies
exact coverage before creating `formal_complete.json`.

For the fixed R2R Pilot A/B comparison, set `DATA_PATH` to the existing R2R
dataset root containing `val_unseen/val_unseen.json.gz`, then pass
`--groups baseline_off,hierarchical_topology --pilot-per-scene 20` to
`scripts/run_memory_ablation.sh`. The launcher uses sampling seed 42 and
targets 20 Episodes in each of 11 scenes. A scene with fewer than 20
contributes all available Episodes; missing seats go to the scenes with the
most remaining candidates, with `scene_id` breaking ties. For the confirmed
val_unseen distribution, quotas are 18/21/21 and 20 in the other eight scenes;
short/medium/long quotas are 6/6/6, 7/7/7, and 7/7/6 respectively.
It writes `pilot_manifest.json` once, serves only those 220 Episode IDs, sets
`EXPECTED_EPISODES=220`, and requires each arm's `pilot_complete.json` after
exact coverage checks. `--resume DIR` reuses the saved manifest. This mode
does not write `formal_complete.json`; omitting `--pilot-per-scene` retains the
full 1,839-Episode evaluation. `SCENES_DIR` should match the scene path used
by the Habitat dataset loader. See [MEMORY_ABLATION.md](MEMORY_ABLATION.md).

### RGM V1 88-Episode Pilot

`--rgm_enabled` activates the three evidence gates inside the existing hierarchical
memory path. It requires `--memory_enabled --memory_strategy hierarchical`.
`--rgm_write_gate_enabled`, `--rgm_retrieval_gate_enabled`, and
`--rgm_prompt_gate_enabled` default to on when RGM is selected; each supports
the corresponding `--no-rgm_*` form. `--rgm_step0_conditional` controls the
instruction-related current-frame exception at step 0. D adds
`--rgm_yolo_temporal_enabled`, `--rgm_place_two_stage_enabled`, and
`--rgm_topology_motion_validation_enabled`. Each can be disabled separately.
`--rgm_adaptive_extraction_enabled` is reserved and rejected in V1; YOLO and
DINO retain their fixed 5-step cadence.

The uncalibrated numeric flags and defaults are:

| Flag suffix after `--rgm_` | Default | Meaning |
| --- | ---: | --- |
| `min_candidate_confidence` | 0.25 | Minimum raw detection confidence |
| `min_instruction_landmark_confidence` | 0.60 | Direct instruction landmark acceptance |
| `temporal_window` / `temporal_min_support` | 3 / 2 | Independent extraction events for category evidence |
| `place_top_k` | 3 | Global DINO place candidates |
| `place_global_candidate_min` / `place_global_new_max` | 0.75 / 0.65 | Global match and new-candidate bounds |
| `place_local_cosine_min` / `place_local_min_matches` | 0.70 / 8 | Mutual Patch correspondence thresholds |
| `place_local_inlier_min` / `place_global_margin_min` | 0.50 / 0.03 | Local geometry and global ambiguity bounds |
| `place_new_temporal_similarity_min` | 0.85 | Similarity needed between independent novel-place observations |
| `place_ransac_reprojection_threshold` | 0.08 | Normalized image-coordinate RANSAC tolerance |
| `place_min_image_quality` | 0.20 | Minimum grayscale contrast proxy before place identity decisions |
| `place_keyframes_max` | 3 | Episode-local Patch keyframes per visual place |
| `topology_visual_delta_min` | 0.35 | Minimum RGB change for an observed transition |
| `prompt_max_records` / `prompt_max_chars` | 3 / 400 | Final injected memory budget |

All RGM flags appear in `config_snapshot.yaml` under `memory_args`. Scores are
engineering thresholds, not calibrated probabilities. No current server field
supplies Pose, collision feedback or odometry; RGM never marks a traversal as
physically verified.

For the new A/C/D Pilot, use
`bash scripts/run_memory_ablation.sh --rgm-88 --parent-manifest OLD/pilot_manifest.json`
with `MODEL_PATH`, `YOLO_MODEL_PATH`, `DINO_MODEL_PATH` and `DATA_PATH` set.
Optional `--legacy-root` points to `output/r2r_pilot_220_v2`, `--output-root`
defaults to `output/rgm_pilot_88_v1`, and `--resume` preserves the fixed selection.
The launcher derives 88 IDs from the legacy 220 Manifest (8 per scene, 3/3/2
distance strata), runs C=`rgm_gates` and D=`rgm_full`, then strictly checks
legacy A=`baseline_off`. Missing source or weight fingerprints reject automatic
A reuse and cause the launcher to run A on the same 88 IDs. B is absent from
the RGM protocol. The legacy 220 and full 1839 launch modes above remain
available separately.

### RGM V3 memory backend

RGM V3 is selected independently of the RGM V1 `--rgm_enabled` switch. Use
`--rgm_v3_enabled` with `--memory_enabled --memory_strategy hierarchical`
and `--memory_layers short,mid,long`.
Do not combine V1 and V3 switches in one run. The V3 flags are:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--rgm_v3_enabled` | off | Run the V3 evidence store, multichannel retrieval and prompt formatter |
| `--rgm_v3_scope` | `episode_local` | `episode_local` clears online memory at episode end; `scene_persistent` retains autonomous observations within the same scene |
| `--rgm_v3_prompt_max_records` | `3` | Maximum complete memory statements in the prompt |
| `--rgm_v3_prompt_max_chars` | `400` | Maximum characters in the formatted memory block |

An enabled V3 run requires an explicit `--output_dir` pointing to a new, empty
directory. The runner writes `rgm_v3_protocol.json` with protocol version
`rgm_v3_v1` and the selected scope. Do not reuse the V1 A/C/D output directory
or its Pilot Manifest. For the first comparison with unmodified LightNav, select
`episode_local`; `scene_persistent` is a separate experimental setting and must
be reported separately. The two scopes must never be combined into one metric.

V3 uses scene, episode, rollout and local event ID together as a provenance key.
An issued action or RGB appearance change records an unverified spatial cue;
only an independently measured displacement can verify a physical traversal.
The available Habitat client does not provide such a measurement to the online
memory, so its traversability status remains `unknown`. Evaluator metrics such
as `info.success` and oracle distance are excluded from online memory.
See [RGM_V3.md](RGM_V3.md) for the memory and token-level diagnostic contract.

### RGM V4 selective memory fusion

RGM V4 uses the V3 evidence store and retriever, then updates an explicit
navigation state and selects only history that can help the current decision.
It leaves the checkpoint, instruction, RGB frame sequence, processor chat
template, generation settings, and action decoder unchanged. Enable it with
`--memory_enabled --memory_strategy hierarchical --memory_layers short,mid,long
--rgm_v4_enabled`. The Legacy RGM, V3, and V4 switches are mutually exclusive.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--rgm_v4_enabled` | off | Select the independent V4 fusion backend |
| `--rgm_v4_scope` | `episode_local` | Clear memory after each Episode, or retain observations within a scene using `scene_persistent` |
| `--rgm_v4_prompt_max_records` | `3` | Maximum complete evidence statements |
| `--rgm_v4_prompt_max_chars` | `400` | Maximum characters in the whole memory block |
| `--rgm_v4_revisit_enabled` / `--no-rgm_v4_revisit_enabled` | on | Include or exclude uncertain visual revisit candidates |

Online revisit candidates also require the existing DINO place extractor,
configured with `--memory_place_enabled --memory_place_model_path <local_path>`.
Without descriptors, V4 still provides history and progress audits and makes
no revisit claim.

Pass an explicit, new `--output_dir`, for example
`output/rgm_v4_pilot_88/`. V4 creates `rgm_v4_protocol.json` with protocol
`rgm_v4_v1` and rejects a nonempty directory or a mismatched resume. The runner
and launcher protect the V2 and V3 Pilot directories from reuse. A V4 run does
not start by setting these flags alone; launch Habitat separately when the
evaluation environment and selected episodes are ready.

The four use modes are `NONE`, `HISTORY_REFERENCE`, `REVISIT_AUDIT`, and
`PROGRESS_AUDIT`. `NONE` passes the original Policy sample unchanged. The other
modes add at most the configured number of source-qualified facts. A visual
match remains a possible revisit, an issued action remains unverified, and
unknown traversal is reported as unknown. Diagnostics record the original and
injected prompt hashes, final Processor token hash, vLLM token hash when used,
Evidence IDs and Episode/Step provenance, dropped candidates, exact token
delta, and whether memory actually entered model input. See
[RGM_V4.md](RGM_V4.md) for the implementation and source mapping.

### Local graph navigation context (independent experiment)

`--local_graph_enabled` selects the frozen-policy, current-view/local-DINO-graph
memory path described in [MEMORY_DESIGN.md](research/MEMORY_DESIGN.md). It requires
`--memory_enabled --memory_strategy hierarchical --memory_layers short,mid,long`
and is mutually exclusive with `--rgm_enabled`, `--rgm_v3_enabled` and
`--rgm_v4_enabled`. Supply the same YOLO/DINO model paths as the V4 comparison.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--local_graph_enabled` | off | Enable the independent local graph context |
| `--local_graph_scope` | `episode_local` | Clear evidence each Episode, or retain it in the same Scene with `scene_persistent` |
| `--local_graph_prompt_max_chars` | `320` | Entire navigation card character budget |
| `--local_graph_max_landmarks` | `2` | Maximum current/local historical landmark lines |
| `--local_graph_reactivation_steps` | `8` | Minimum interval before reusing an unchanged card |

The runner requires a new, empty explicit output directory and writes
`local_graph_protocol.json` (`local_graph_v1`). Current YOLO box position is a
camera-sector observation; a DINO-linked earlier landmark is labelled as a
possible revisit with unknown current direction and traversability. The module
does not infer a route or change the action decoder.

The underlying `scripts/eval_habitat.sh` also accepts `STRICT_FULL_EVAL=1`
for the full-split gate, `RESUME=1` to skip existing completed Episodes,
`EPISODE_MANIFEST_PATH` to create or compare a shared manifest, `SEED` for
server and client, and `NO_FORCE_STOP=1` to let the standard Habitat step cap
truncate. Its matching client flags are `--strict_full_eval`, `--resume`,
`--seed N`, and `--no_force_stop`; the Habitat server receives `--seed N` and
`--strict-full-eval`. For Pilot runs, `PILOT_MANIFEST_PATH` supplies the fixed
selection to the Habitat server as `--pilot-manifest`; the integrity gate
checks exact selected IDs and writes `pilot_complete.json`. The ordinary
defaults remain available for smoke tests.

For one R2R Episode, the Habitat server accepts `--single-scene-id <path>` and
`--single-episode-id <id>` together. They select exactly one loaded `val_unseen`
Episode by canonical scene path and full ID before iteration, and reject pilot,
strict full evaluation or sharding. The server also reports `agent_pose`
(`position` and quaternion) in `info` when simulator state is available; this
is evaluator-side trace data and is not passed into the memory packet. Resolve
the two IDs from a manifest and the dataset with
`scripts/resolve_single_episode.py`; see
[EPISODE1_EXPERIMENT.md](research/EPISODE1_EXPERIMENT.md).

`lightnav-eval-habitat` reports Episode progress from the server's loaded
manifest. `--progress_shard_index` and `--progress_shard_count` label direct
client runs (defaults: `0` and `1`); `scripts/eval_habitat.sh` supplies them
automatically. In multi-GPU runs, progress and ETA are **per shard**, using
that shard's actual loaded Episode count. Completed results are included on
resume. Episode start and completion print immediately; non-TTY runs emit a
timer-driven heartbeat about every 12 seconds, including during a slow model
call. Single-shard TTY runs use an in-place
display (`--progress_tty` when stderr is forwarded through the launcher);
multi-shard runs use labelled heartbeat lines. Progress goes to stderr so the shell
launcher streams it live and writes `logs/client_progress_gpu*_shard*.log`
alongside its existing client logs. ETA is approximate and initially reads
`CALIBRATING` until completed Episode durations are available. A client-side
`--languages` filter has no known post-filter Episode count in the server
manifest, so numeric progress is suppressed for that run.

`lightnav-serve` flags (environment names are what `scripts/start_servers.sh` and
`docker/entrypoint.sh` read):

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--model_path` | `MODEL_PATH` | required | checkpoint directory |
| `--action_tokenizer_bundle` | `ACTION_TOKENIZER_BUNDLE` | from the checkpoint | RVQ bundle dir; overrides the decoder the checkpoint ships |
| `--task` | `TASK` | `tracking` | `tracking` (tracking prompt, `tasks.trackvla`) or `vln` (VLN prompt, `tasks.vlnce`) |
| `--backend` | `BACKEND` | `vllm_local` | `vllm_local` or `hf` |
| `--gpu_memory_utilization` | `GPU_MEM_UTIL` | `0.85` | fraction of GPU memory handed to the vLLM engine |
| `--max_batch_size` | `MAX_BATCH_SIZE` | `8` | max sessions per scheduler tick; also sizes vLLM `max_num_seqs`. `1` = strictly serial |
| `--max_wait_ms` | `MAX_WAIT_MS` | `8` | max wait to fill a batch; a lone request is flushed after 2 ms |
| `--quantization` | `VLLM_QUANT` | none (bf16) | `fp8_llm_only`: fp8 LLM, bf16 ViT (~1.5× on Jetson Thor). On SM 11.0 also set `VLLM_DISABLED_KERNELS`; `scripts/serve_thor.sh` does both. See [JETSON_THOR.md](JETSON_THOR.md) |
| `--vit_cache_entries` | `VLN_VIT_CACHE_ENTRIES` | auto (512 w/ SlowFast) | ViT tubelet LRU capacity (`vllm_local`; speed only, never output). Long robot sessions want `1024` |
| `--num_history_frames` | `NUM_HISTORY_FRAMES` | checkpoint config | history-window override (normally leave unset) |
| `--pool_spatial` | `POOL_SPATIAL` | checkpoint config | spatial-pooling override |
| `--aspect_mode` | `ASPECT_MODE` | `stretch` | `stretch`: resize every frame to `video_size` (training behaviour); `keep`: per-session size with the camera's aspect ratio at the same pixel budget (4:3 → 288×384 for a 256×448 checkpoint) |
| `--max_new_tokens` | `MAX_NEW_TOKENS` | `8` | lower bound for the per-step decode cap (see below) |
| `--host` / `--port` | `HOST` / `PORT` | `0.0.0.0` / `8050` | bind address |
| `--ready_file` | – | none | file touched once the port is bound (used by the launchers) |
| `--no_warmup` | – | off | skip the synthetic warm-up inference before binding the port |
| `--record_dir` | `RECORD_DIR` | off | record every connection's episodes (frames + per-step records) for `lightnav-render` (see [DEPLOYMENT.md](DEPLOYMENT.md)) |
| `--record_fps`, `--record_timeline`, `--no_record_images` | `RECORD_FPS`, `RECORD_TIMELINE`, `RECORD_IMAGES` | `10`, `realtime`, images on | recording defaults written to the manifest / frame storage |
| `--cam_hfov_deg`, `--cam_height`, `--traj_forward_offset`, `--waypoint_dt_s` | `CAM_HFOV_DEG`, `CAM_HEIGHT`, `TRAJ_FORWARD_OFFSET`, `WAYPOINT_DT_S` | `90`, `0.5`, auto, `0.1` | client-camera geometry and HUD convention used only by the rendered overlay |

### MuJoCo demo parameters

`mujoco_demo` is a separate `uv` project. Run `./run.sh` for its bundled
TurtleBot, or use `uv run --extra microduck vln-mujoco ...` when selecting the
optional MicroDuck backend. MicroDuck needs two external files, the MJCF from
`pollen-robotics/microduck_rl` and an ONNX policy from
`pollen-robotics/microduck-policies`; the download steps are in
[mujoco_demo/README.md](../mujoco_demo/README.md#microduck-optional).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `8088` | web-console bind address |
| `--vln-server` | empty | default LightNav WebSocket URL shown in the console |
| `--robot` | `turtlebot` | robot backend: `turtlebot` or `microduck` |
| `--robot-model` | none | external MicroDuck MJCF; required only with `--robot microduck` |
| `--walking-policy` | none | external MicroDuck ONNX policy; required only with `--robot microduck` |

### SlowFast visual-token budget diagnostic

From the repository root, run
`python scripts/inspect_slowfast_budget.py --checkpoint checkpoints/LightNav-0`
in the inference environment. The script
reads the local checkpoint's `eval_config.json`, tokenizer and Qwen3-VL processor;
it uses synthetic RGB frames on CPU and prints JSON without loading weights,
downloading files or writing an output file. The checkpoint must be present locally.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | `checkpoints/LightNav-0` | local checkpoint directory containing `eval_config.json` and processor files |
| `--instruction` | `walk forward through the corridor and stop` | fixed prompt text for measuring the complete generation input length |

For episode lengths 2, 10, 34, 90, 120 and 256, the report separates ViT input
patch rows, ViT output tokens before post-ViT pooling, LLM visual tokens after
pooling, expanded video placeholders, and the full generation prompt length.
Post-ViT pooling lowers the LLM visual-token count but does not proportionally
lower ViT work. `model_input_tokens` is `null` when the checkpoint's `max_seq_len`
would cut the generation prompt; the untruncated processor measurement is shown
separately.

### Independent MuJoCo Baseline evaluator

`scripts/eval_mujoco_baseline.py` controls the existing MuJoCo demo through its
WebSocket, relays model messages through an observation proxy, and reads
simulator poses. It independently checks the recorded TurtleBot trajectory
against the bundled scene's collision geoms. It runs in `mujoco_demo`'s `uv`
environment; the AutoDL example uses LightNav `ws://127.0.0.1:8051` and MuJoCo
port `8088`;
see [EVAL_MUJOCO_BASELINE.md](EVAL_MUJOCO_BASELINE.md) for target verification,
commands, output and metric definitions. The task JSON and checkpoint processor
diagnostic must be supplied before running a scored evaluation. Start the model
service with default `--aspect_mode stretch` and the same SlowFast/processor
settings used for the diagnostic. `--model-id` is a user-provided label, not a
weight identity check.

| Flag | Meaning |
| --- | --- |
| `--demo-url` | MuJoCo demo HTTP base URL, for example `http://127.0.0.1:8088` |
| `--vln-server` | LightNav inference WebSocket URL that the demo should use |
| `--tasks` | fixed JSON task set with scene, robot, expected spawn XY/yaw, instruction, target circles and `goal_verified` flags |
| `--budget-diagnostic` | JSON report from `inspect_slowfast_budget.py`, used to calibrate derived per-step visual-token budgets |
| `--model-id` | operator-provided checkpoint/version label recorded with every run |
| `--repeat` | number of repetitions of each fixed task |
| `--timeout-s` | wall-clock limit for one task run |
| `--max-steps` | completed model-prediction limit for one task run |
| `--output` | new, exclusive directory for per-run `results.jsonl` and aggregate `summary.json`; existing runs are not overwritten |

Engine-level environment variables (no flag):

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLN_KV_CACHE_GIB` | auto | vLLM KV-cache size in GiB (auto-scales with `max_num_seqs`, floor 2 GiB) |
| `VLN_VLLM_ENFORCE_EAGER` | `0` | `1` disables CUDA-graph capture (faster start, slower steps) |
| `LIGHTNAV_ATTN` | `sdpa` | attention implementation for the `hf` backend (e.g. `flash_attention_2`) |
| `VLN_EVAL_TEMPERATURE` / `TOP_P` / `TOP_K` | greedy | sampling knobs for experiments; benchmark numbers assume they are unset |

**Decode token budget.** A step emits a *grounding prefix* followed by the action token(s):
one `<act_l{d}_*>` per RVQ level. The server
probes the tokenizer for the grounding families it knows and caps generation at
`max(--max_new_tokens, prefix + action_tokens)`, which saves the decode step otherwise spent
reaching `eos` without ever truncating the action.

| checkpoint family | grounding prefix | tokens |
|---|---|---|
| tracking, legacy | `<tpos_k>` | 1 |
| tracking + grid pointing | `<opos_k>` | 1 |
| navigation + dual pointing | `<apos_A><opos_O>` | 2 |

**GPU sizing.** The 4B checkpoint needs ~10 GB for weights in bf16 plus the vLLM KV cache;
one 24 GB GPU comfortably serves one engine with `--gpu_memory_utilization 0.85`. When
several servers share a GPU, `scripts/start_servers.sh` divides the utilisation by
`SERVERS_PER_GPU`.

Note what `--gpu_memory_utilization` does *not* do here. The engine passes an explicit
`kv_cache_memory_bytes` (`VLN_KV_CACHE_GIB`, floor 2 GiB) and therefore skips vLLM's
profile run, so real usage is roughly weights + that KV cache + CUDA graphs (~15 GB for
the released 4B checkpoint) whatever the fraction says. The fraction still acts as the
ceiling vLLM checks against free memory at start-up, so lower it when the GPU is shared
with another job, and raise `VLN_KV_CACHE_GIB` — not the fraction — when you want a
bigger KV cache.

---

## Input aspect ratio (`--aspect_mode`)

The checkpoints are trained on frames resized to a fixed `common.video_size` (256×448 for
the released checkpoint, ≈16:9). By default (`stretch`) every client frame is resized to
that size regardless of its aspect ratio, exactly as in training — a 4:3 camera therefore
arrives horizontally squeezed.

`--aspect_mode keep` (env `ASPECT_MODE=keep`; also `lightnav-predict --aspect_mode`,
`lightnav-eval-habitat --aspect_mode`, `build_tracking_agent(aspect_mode=)`) instead picks,
per session from its first frame, the size with the **source aspect ratio** whose area is
closest to `video_size` and whose sides are multiples of 32 (patch × merge; and of the pre-ViT
pooling factors when the checkpoint pools before the ViT). The vision-token count per frame
stays at the training budget, so history length and sequence length are unchanged:

| camera | `stretch` | `keep` |
|---|---|---|
| 16:9 (480×270, 1920×1080) | 256×448 | 256×448 (identical) |
| 4:3 (640×480) | 256×448 (squeezed) | 288×384 |
| 1:1 | 256×448 | 352×352 |

Frames within one session must share one size (the first frame decides; `reset` starts
over). `keep` avoids geometric distortion but changes the token grid the model sees, which
the released checkpoint was not trained on — validate on your robot before relying on it.
Full native resolution (feeding the camera's own pixel count) is not supported.
