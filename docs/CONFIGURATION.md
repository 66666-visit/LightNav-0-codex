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
otherwise the trace records `null`. Each row also carries top-level
`rollout_idx`, `position`, `rotation`, `action`, `decoded_action_label`,
`collision` (server field when supplied), `memory_text`, `actual_injection`,
`actual_injected_evidence_ids`, `prompt_hash_before`, `prompt_hash_after`,
`grounding_query`, `selected_object_ids` and `geometry_fallback_reason`.
These fields are written after `policy.act`; none is forwarded to the policy,
Grounder or memory selection. Goal distance/reference path remain
evaluator-only offline fields.
`action_divergence_vs_baseline`, `invalid_exploration`, and `wrong_turn` remain `null` until matched
traces are compared with each other and the reference route. The legacy
memory paths never feed evaluator distance or pose into the policy; the
explicit ConceptGraphs-Lite L3 modes separately consume the server's
`spatial_rgbd` pose and calibration, never evaluator goal distance. `--trace_steps` requires
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
engineering thresholds, not calibrated probabilities. RGM V1 does not consume
pose, collision feedback or odometry, so it never marks a traversal as
physically verified. The new `spatial_rgbd` metadata is consumed only by the
explicit ConceptGraphs-Lite L3 path described below.

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

### ConceptGraphs-Lite L3 (Habitat Local Graph extension)

`--spatial_l3_mode {off,shadow,inject,on_demand,geometry_on_demand}` adds the
independent three-dimensional object map described in
[CONCEPTGRAPHS_LITE_L3.md](research/CONCEPTGRAPHS_LITE_L3.md). `on_demand` is
documented in [L3_ON_DEMAND_CLIP_QUERY.md](research/L3_ON_DEMAND_CLIP_QUERY.md):
it keeps the L1 Original path primary and runs a local CLIP text query only after
L1 returns an empty candidate set. `geometry_on_demand` uses the same L1-empty
trigger and dedupe but the existing geometry-only candidate retrieval
(`SpatialGraphAdapter`); it never loads CLIP weights or runs OpenCLIP. See
[GEOMETRY_ON_DEMAND.md](research/GEOMETRY_ON_DEMAND.md). The opt-in ViL3DRel
instance grounder extends it with the joint proposal/Evidence selection and
geometry fallback documented in
[INSTANCE_DISAMBIGUATION.md](research/INSTANCE_DISAMBIGUATION.md).
It requires `--memory_enabled --memory_strategy hierarchical --memory_layers
short,mid,long --local_graph_enabled`. The default `off` preserves the existing
Local Graph behavior. It runs only in the Habitat evaluation client when that
client receives aligned RGB-D and calibration metadata from the Habitat server;
the WebSocket inference server has no RGB-D map interface.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spatial_l3_mode` | `off` | `shadow` builds/searches the graph for diagnostics without modifying the original model input; `inject` allows audited, trustworthy spatial candidates; `on_demand` keeps L1 primary and runs the CLIP query only after an empty L1 result; `geometry_on_demand` keeps the same trigger but retrieves the existing geometry-only spatial candidates (no CLIP) |
| `--spatial_l3_seg_model_path FILE` | empty | Existing local `yolo11n-seg.pt`; required for `shadow`, `inject` or `on_demand`; never downloaded or overwritten |
| `--spatial_l3_seg_device` | `cpu` | Independent segmentation inference device |
| `--spatial_l3_seg_confidence` | `0.25` | Raw YOLO segmentation score floor, not a calibrated probability |
| `--spatial_l3_seg_every_n_steps` | `5` | L3 Mask extraction cadence (step 0, 5, 10, …), independent of L1 |
| `--spatial_l3_evidence_min_iou` | `0.5` | Minimum same-class segmentation-box/L1-box IoU for an optional cross-reference |
| `--spatial_l3_evidence_ambiguity_margin` | `0.1` | Reject an L1 cross-reference when the best and second-best IoU differ by less |
| `--spatial_l3_min_injection_confidence` | `0.5` | Minimum raw L3 segmentation confidence for a historical object to enter Prompt |
| `--spatial_l3_clip_model_path FILE` | empty | Local OpenCLIP checkpoint for `on_demand`; required by that mode, never downloaded |
| `--spatial_l3_clip_model_name` | `ViT-H-14` | OpenCLIP architecture name; matches the official ConceptGraphs encoder |
| `--spatial_l3_clip_device` | `cpu` | Local CLIP image/text inference device |
| `--spatial_l3_clip_precision` | `fp32` | OpenCLIP precision (`fp32`, `fp16`, `bf16`); `fp16` requires a CUDA `clip_device` and is the recommended 24 GB-GPU setting |
| `--spatial_l3_clip_batch_size` | `8` | Maximum image crops per CLIP forward pass; bounds activation memory |
| `--spatial_l3_clip_top_k` | `3` | Maximum matched objects appended to an L1-empty Prompt |
| `--spatial_l3_clip_min_similarity` | `0.0` | Cosine floor for a valid match; `0.0` is the sign boundary, not a tuned confidence threshold |
| `--spatial_l3_rgb_sidecar` | off | Opt-in read-only XYZRGB sidecar: records the exact per-mask pixels of every L3 Evidence so the official ViL3DRel Student can consume real XYZRGB. It does not change ObjectNode, `observe()` or geometry association |
| `--spatial_l3_instance_grounder` | off | Ground same-class instances with the official ViL3DRel Nr3D Student between the geometry query and the existing Prompt formatter. Requires the sidecar and a compiled official pointnet2 CUDA extension |
| `--spatial_l3_grounder_checkpoint FILE` | empty | Local official Nr3D Student checkpoint (`gtlabelpcd_mix/nr3d/ckpts/model_epoch_96.pt`); never downloaded at inference |
| `--spatial_l3_grounder_config FILE` | empty | Official model YAML; empty uses the `nr3d_gtlabelpcd_mix_model.yaml` provided under `references/vil3dref` |
| `--spatial_l3_grounder_device` | `cuda` | Torch device for the official Student; no silent CPU fallback |
| `--spatial_l3_grounder_bert_vocab FILE` | empty | Local `bert-base-uncased` `vocab.txt`; empty uses the local Hugging Face cache with `local_files_only=True` |
| `--spatial_l3_grounder_seed` | `0` | Deterministic seed for the official 1024-point sampling |
| `--spatial_l3_grounder_selection` | `argmax` | `argmax` is the original d659e57 one-full-instruction/global-argmax path. `eligible` enables joint proposal → ObjectNode → Evidence → formatable-fact filtering (ablation D/E/F) |
| `--spatial_l3_grounder_landmark_queries` | off | Parse the original instruction once per Episode and send each verbatim target/reference phrase to ViL3DRel. Requires `--spatial_l3_grounder_selection eligible` (ablation E/F). The complete original instruction is still passed to LightNav unchanged |
| `--spatial_l3_grounder_geometry_fallback` | off | When eligible Grounding cannot map a landmark, reuse the unchanged `SpatialGraphAdapter` item for that same landmark. Requires `eligible` (ablation F). A full-instruction query uses the original bounded geometry fallback |
| `--spatial_l3_grounder_ordered_landmarks` | off | Organize the existing `[Navigation Memory]` block as the ordered instruction plan. Every landmark keeps its verbatim phrase, role and instruction order; a real spatial fact is added only for a landmark with a selected `SpatialGraphAdapter` item. Requires `--spatial_l3_grounder_selection eligible`, `--spatial_l3_grounder_landmark_queries` and `--spatial_l3_instance_grounder` (ablation G for 1012) |
| `--spatial_l3_grounder_include_missing_landmark_placeholders` | on | H1/H3 rendering switch. With `--spatial_l3_grounder_ordered_landmarks`, keep (default) or omit the coordinate-free `no 3D evidence` skeleton/`Plan:` lines for landmarks without a verified fact. Omitting them releases that budget to real fact lines; `ordered_memory_rendering` in the L3 diagnostic records the resulting line/fact/placeholder counts. Requires ordered landmarks |
| `--spatial_l3_grounder_preserve_original_spatial_fact` | off | H2/H3 rendering switch. With `--spatial_l3_grounder_ordered_landmarks`, keep G's role/phrase prefix but render the fact body with F's original `render_context` sentence (coordinates, straight-line range, bearing sign, observation count, source, `passage unknown`) instead of G's compact sentence. Requires ordered landmarks |
| `--spatial_l3_history_admission_mode` | `off` | C arm object-level admission at the final L3 history/Prompt boundary. `off` is a pure pass-through and leaves every existing arm unchanged. `once_per_object` admits a stable `ObjectNode.object_id` once per Episode and refuses a later re-selection of the same object even when its Evidence ID changed; different object IDs are independent (including same-class instances) and a candidate without a reliable `object_id` stays fail-open and is counted. Adoption is committed only after the final Prompt token audit proves the candidate's Evidence was injected, and the state is cleared at every Episode boundary. Requires `geometry_on_demand`, `l1_original_only`, eligible selection, the instance grounder and ordered landmarks |
| `--adaptive_l3_query` | off | Opt-in Adaptive L3 Query arm: a local controller decides whether to query L3 on any step, not only when L1 is empty |
| `--adaptive_l3_query_max_new_tokens` | `24` | Maximum decoded tokens for one local controller decision |
| `--adaptive_l3_query_max_query_chars` | `256` | Maximum characters of a controller-produced CLIP query |
| `--adaptive_l3_query_min_evidence_confidence` | `0.5` | Minimum raw Evidence confidence accepted by the local fact validator |

`on_demand` is valid with `--memory_prompt_mode legacy` or
`--memory_prompt_mode l1_original_only`; the latter is the preserved 75 % SR
L1 Original arm. It is rejected with the other prompt-ablation modes. It also
requires `--memory_extractor yolo11` and the optional `clip` extra
(`pip install -e ".[clip]"`) for `open_clip_torch`. The L3 CLIP checkpoint is
loaded from the local file path passed to OpenCLIP; LightNav never downloads
weights.

`geometry_on_demand` has the same L1-empty trigger, dedupe and Prompt insertion
point, but retrieves the existing geometry-only candidate set from
`SpatialGraphAdapter.build`. It requires no `--spatial_l3_clip_model_path`, does
not import OpenCLIP at runtime and never computes a cosine similarity. The same
`legacy` / `l1_original_only` restriction applies.

The opt-in instance grounder is a separate gate on top of
`geometry_on_demand` and is off by default. The default grounder configuration
is the original d659e57 path: one full-instruction query, the official global
argmax, one existing fact, no geometry fallback (ablation C). The joint filter
(D/E/F) is an explicit opt-in and never silently changes the old path.

`--spatial_l3_rgb_sidecar` records the exact pixels of each Evidence mask (same
`backproject_world` sampling and intrinsics/pose as the geometry map), so
`--spatial_l3_instance_grounder` can build the official Student's `(1024, 6)`
XYZRGB proposals without bbox synthesis. It feeds all complete ObjectNodes
(including reference objects) to the official ViL3DRel Nr3D Student and keeps
every original logit and softmax probability. No probability is renormalized.
With `--spatial_l3_grounder_selection eligible`, each proposal is mapped
explicitly to its persistent ObjectNode and Evidence set; the manager selects
the highest-ranked proposal with a valid ObjectNode, observed Evidence mapping,
formatable geometry fact and matching target category. If the global argmax
fails those checks, the next valid candidate is used. Category identity is
alias/plural aware for `couch/sofa`, `chair/armchair` and generic
`table/dining table/coffee table`; dining table and coffee table remain
distinct. With `--spatial_l3_grounder_landmark_queries`, the original
instruction is parsed once per Episode into verbatim target/reference phrases
and each phrase is sent to the Student separately; LightNav still receives the
complete original instruction. Multiple valid ObjectNodes can be injected under
the existing Prompt character/Token budget. With
`--spatial_l3_grounder_geometry_fallback`, a failed landmark reuses its own
existing `SpatialGraphAdapter` item; a full-instruction query uses the original
bounded geometry fallback. Without it, a failed query keeps the original L1
Prompt. With `--spatial_l3_grounder_ordered_landmarks`, those selected facts are
organized as an ordered instruction plan at the existing Prompt insertion point:
a compact semantic line is reserved for every landmark, so a structural
landmark whose detector class is absent (for example `doorway`/`stairs` under
COCO) is still represented without invented coordinates; remaining characters
enrich `final > transition > intermediate > source > reference` landmarks with
their real stored facts. This is ablation G, used only for the 1012 diagnostic.

The two rendering switches isolate the three changes G introduced. The default
pair (placeholders on, G facts) is byte-identical to G.
`--no-spatial_l3_grounder_include_missing_landmark_placeholders` is H1: no
landmark without a selected fact contributes a skeleton or `Plan:` line, so the
freed characters can carry additional real facts.
`--spatial_l3_grounder_preserve_original_spatial_fact` is H2: placeholders,
roles, order and the selection pipeline stay unchanged, but each fact body is
F's exact `render_context` sentence rather than G's re-templated sentence.
Enabling both is H3. The L3 diagnostic records
`ordered_memory_rendering` with the effective switches and the rendered
fact/placeholder line counts, so a budget-selection difference is never silent.
Candidate filtering, Evidence IDs, fact enrichment order, map updates, query
frequency and Grounding are not changed by either switch.

The C arm is H1 plus `--spatial_l3_history_admission_mode once_per_object`.
It adds one episode-local gate at that same final L3 history boundary: the
first Prompt that actually injects a stable `object_id` adopts it, and a later
Step that re-selects the same object (even through a new Evidence ID) is
refused. The gate never scores, thresholds, decays, re-ranks or merges
categories; it only keys on the complete stable `object_id`. A candidate
without a reliable `object_id` is admitted fail-open so missing metadata cannot
suppress the original L3 injection. Pending admission is written to
`memory_diagnostics.jsonl` and `prompt_diagnostics.jsonl` as
`history_admission_mode`, `history_admission_candidate_object_ids`,
`history_admission_accepted_object_ids`,
`history_admission_rejected_seen_object_ids`,
`history_admission_missing_object_id_count` (plus the audit-confirmed and
cumulative adopted/pending lists). `--spatial_l3_history_admission_mode off`
is the default and changes no Prompt byte.

The checkpoint, BERT vocab and compiled extension must be local;
inference never downloads anything. Install the small official-model dependencies with
`pip install -e ".[vil3dref]"` (the CUDA-enabled torch is provided by the
existing inference environment). Build the extension with the same Python/torch
as the inference process (or a venv with a matching Python/torch ABI); a Python
extension is not portable across Python or torch builds. Build/check on a GPU
host:

```bash
CUDA_HOME=/usr/local/cuda PYTHON=/path/to/cuda-venv/bin/python \
  bash scripts/build_vil3dref_pointnet2.sh

PYTHONPATH=src python scripts/check_vil3dref_cuda.py \
  --checkpoint /path/model_epoch_96.pt \
  --bert_vocab /path/bert-base-uncased/vocab.txt
```

#### Adaptive L3 Query (opt-in)

`--adaptive_l3_query` keeps the original L1 card as the primary memory and adds
one bounded sidecar: the already-loaded local LightNav inference engine answers
a `NO_QUERY` / `QUERY_L3` question, the existing `L3ClipQuery` retrieves
candidates with unchanged cosine Top-K, the engine then answers
`USE C1 …` / `ABSTAIN` over candidate aliases, and a deterministic validator
re-reads the real `ObjectNode` and **all** of its `SpatialEvidence` records.
Only facts that pass are rendered by a deterministic formatter and appended
after the L1 card. No controller output text reaches the Prompt.

It requires `--memory_enabled --memory_strategy hierarchical --memory_layers
short,mid,long --local_graph_enabled --memory_extractor yolo11
--spatial_l3_mode shadow --spatial_l3_seg_model_path FILE
--spatial_l3_clip_model_path FILE` and `--memory_prompt_mode legacy` or
`l1_original_only`. `shadow` only builds the read-only map; it never injects
spatial candidates, so the L1 path stays primary. The mode is rejected with
`inject`, `on_demand`, other prompt modes, or a missing `clip_model_path`.
It reuses the local inference engine; it does not call an external API and does
not load a second model or download weights. `VLN_EVAL_TRAJ_TOP1=1` is rejected
because that restriction masks text decoding to trajectory ids.

The controller is asked once per step even when L1 has candidates; that is the
point of the arm. A query is never repeated for the exact same query text and
unchanged object map within one Episode (no fixed cooldown is involved). Every
failure path — missing engine, parse failure, CLIP error, no candidate, invalid
or current-step Evidence, missing `observed_from` edge, prompt budget — keeps the
original L1 Prompt byte for byte. Diagnostics appear under `adaptive_l3` in the
step diagnostic and `prompt_context` (`validated_object_ids`,
`validated_evidence_ids`, `rejected`, `skipped_reason`, `prompt_verified`).

The prompt-mode path uses the existing token-aware budget (base prompt and
Processor/tokenizer measurement) and falls back to the original 320-character
allowance only when measurement is unavailable. The legacy path keeps its
original 320-character allowance. Three compact on-demand lines fit inside 320
characters; the L3 block is still whole-line truncated by the unchanged budget
logic. `L3 CLIP cos` in the prompt is the raw cosine score, not a probability.
Per-step `memory_diagnostics.jsonl`/`prompt_diagnostics.jsonl` records the raw
similarity, rank, Object ID, Evidence IDs, rejection decision, query trigger
reason and cumulative `l3_query_stats` (trigger/match/injection rates).

The original L1 `--memory_extractor yolo11 --memory_extractor_model_path
/local/yolo11n.pt` still runs separately. The L3 model must produce a real
instance Mask; no detection box is rasterized into one. Both weights are local,
independently configured, and untouched by the runner. Each valid L3 Mask
receives its own source-qualified, Scene/Episode/rollout/Step-specific Evidence
record. A same-class, high-overlap L1 box may be kept as an **optional**
cross-reference. A missing or ambiguous L1 match never changes the L3 Mask or
blocks a geometrically valid object. The original L1 Evidence contract remains
unchanged. Use separate local paths for the two model weights.

Each non-`off` L3 run also writes `conceptgraphs_l3_protocol.json` with its
mode and geometry/CLIP settings beside the unchanged `local_graph_protocol.json`.
For `l1_original_only` + `on_demand`, `prompt_protocol.json` records the
`on_demand_when_l1_empty` fallback and the unchanged L1 token budget.
The output directory must be fresh, so these settings cannot overwrite an
earlier Local Graph result.

Habitat may send Depth normalized after clipping to its configured sensor
range. The server declares `depth_encoding`, sensor min/max, K, wxyz Agent
Pose, and current Sensor Extrinsics under `info.spatial_rgbd` for the same
observation. The client converts depth to metres and rejects censored range
endpoints, invalid pixels, unaligned images, and missing/ambiguous metadata.
Its L3 object map is Episode-local in V1.1 even when the old Local Graph
scope is `scene_persistent`.

All geometry flags have the prefix `--spatial_l3_` followed by the field below.
They can be passed directly to `lightnav-eval-habitat`; distance fields use
metres, while IoU and ratio fields are dimensionless and point limits are integers.

| Suffix | Default | Meaning |
| --- | ---: | --- |
| `min_depth_m` / `max_depth_m` | `0.1` / `10.0` | Map filter; the Habitat server's actual sensor min/max range is applied first |
| `min_points` | `8` | Minimum valid Mask depth pixels for geometry |
| `max_points_per_instance` / `max_points_per_object` | `256` / `2048` | Bounded online point-cloud samples |
| `voxel_size_m` | `0.03` | Voxel downsampling size and strict nearest-neighbor radius for `n_ratio` |
| `max_center_distance_m` | `0.75` | Existing-object candidate search radius |
| `min_overlap_ratio` | `0.35` | Minimum one-way `n_ratio`: fraction of new detection points within `voxel_size_m` of a historical object point |
| `min_bbox_iou` | `0.01` | Minimum padded 3D axis-aligned box IoU pre-gate; not the match score |
| `ambiguity_margin_ratio` | `0.12` | Minimum dimensionless best-vs-second `n_ratio` advantage; otherwise create a new node |
| `min_travel_distance_m` | `0.01` | Minimum measured pose displacement for a recorded `traversed_to` edge |

The earlier experimental flags `--spatial_l3_max_surface_distance_m` and
`--spatial_l3_ambiguity_margin_m` have been removed. Use `voxel_size_m` for
the `n_ratio` neighbor radius and `ambiguity_margin_ratio` for its score gap.

The matching rule requires the same class, center distance within its maximum,
padded 3D box IoU above its floor, and `n_ratio` above its floor. The original
ConceptGraphs `overlap` mode defines `n_ratio` as the one-way fraction of new
detection points whose nearest historical-object point is strictly closer than
the voxel radius. L3 uses only this geometry score: it computes ambiguity
against all historical candidates before greedy, one-to-one assignment within
each frame. A losing contender creates a new object rather than falling back
to its runner-up. The class, center, padded-box, ambiguity and one-to-one gates
are this project's conservative Geometry-only adaptations. These thresholds
are engineering starting points, not values measured on this project's R2R scenes.
Adjust only against real RGB-D data and report the selected values with the
result. A recorded environment step with valid pre/post pose is required for
`traversed_to`; an issued velocity alone is insufficient.

For a one-Episode AutoDL smoke run, after starting the Habitat server in its
separate Python 3.9 environment and preparing both existing local weights:

```bash
lightnav-eval-habitat \
    --model_path /path/to/LightNav-0 --server tcp://127.0.0.1:5555 \
    --backend vllm_local --episodes 1 --trace_steps \
    --output_dir output/conceptgraphs_l3_shadow_smoke \
    --memory_enabled --memory_strategy hierarchical \
    --memory_layers short,mid,long --local_graph_enabled \
    --memory_extractor yolo11 \
    --memory_extractor_model_path /path/to/yolo11n.pt \
    --spatial_l3_mode shadow \
    --spatial_l3_seg_model_path /path/to/yolo11n-seg.pt
```

Use a fresh output directory for each run. Change only `shadow` to `inject`
and use a separate directory for the injection smoke run. In `shadow`, compare
the actual final Prompt/Token ID hashes and decoded action to a matched
baseline run on the same observation; the graph and candidates are diagnostic.
In `inject`, only trusted candidates pass the existing Prompt gate, budget and
Token audit. A missing geometric candidate falls back to the original Local
Graph candidate path; an empty/failed final prompt falls back to the original
LightNav sample.

### L1 + L3 prompt organization (`--memory_prompt_mode`)

`--memory_prompt_mode` selects how the existing L1 short-term detector memory
and the L3 3D semantic map are serialized into the one bounded block after
`</navigation_task>`. It never changes the model weights, video input, action
decoder, injection position or the legacy character budget. `legacy` (the
default) keeps the original Local Graph/L3 execution path byte for byte.

| Mode | Injected context |
| --- | --- |
| `legacy` | Original path: L3 spatial candidate when `inject`, otherwise the original Local Graph card |
| `l1_original_only` | Original L1 Local Graph card; L3 is off by default, an on-demand CLIP fallback with `--spatial_l3_mode on_demand`, an on-demand geometry-only fallback with `--spatial_l3_mode geometry_on_demand`, or a controller-gated verified block with `--adaptive_l3_query` |
| `l1_organized_only` | Organized L1 card: current vs earlier detections, confidence, image sector, sightings; L3 does not run |
| `l3_original_only` | Original L3 `Spatial history` line; no L1 prompt builder and no Local Graph fallback |
| `l3_organized_only` | Organized L3 card: instruction-relevant objects, observation count, straight-line range/bearing marked "not a path", traversal evidence, unknown passage |
| `l1_l3_original_concat` | Original L1 card followed by the original L3 card under the same total budget |
| `l1_l3_joint` | Organized L1 + L3 fusion with trusted entity alignment, deduplication, preserved conflicts and explicit uncertainty |

Every non-legacy mode requires `--memory_enabled`, `--memory_strategy
hierarchical --memory_layers short,mid,long`, `--local_graph_enabled` and an
explicit `--output_dir`; the six ablation L3 modes additionally require
`--spatial_l3_mode inject` and `--spatial_l3_seg_model_path`. The separate
`l1_original_only` + `on_demand` arm uses the same L1 builder and budget but is
not one of the six ablation conditions. `RESUME=1` is rejected for these modes.

**Budget policy.** `min(--memory_max_context_chars,
--local_graph_prompt_max_chars)` (320 with the defaults) is the original Local
Graph module's character allowance, not the Qwen3-VL context limit. It therefore
applies unchanged only to `legacy`. The six experimental modes select whole
lines against a per-step, model-derived token budget:

- context limit = the smallest known serving cap among
  `bundle.max_seq_len`, `processor.model_max_length` and the backend
  (`vllm.max_model_len`, or the HF model's `max_position_embeddings`);
- occupied base tokens = one Processor pass over the unchanged sample
  (`data_processor.process_sample(..., add_generation_prompt=True)`);
- reserve = `engine.max_new_tokens`;
- one token of boundary reserve for the inserted newline;
- available memory tokens = limit − base − reserve − 1.

The budget is measured with the checkpoint's existing Processor/Tokenizer; no
external API, weight change or new fixed character allowance is used. If the
Processor/Tokenizer, context limit or generation reserve is unavailable, the
mode falls back to the original 320-character allowance and records
`budget_source=char_fallback_320` plus the concrete reason in
`prompt_diagnostics.jsonl`; it never guesses a larger budget. The measurement
adds one CPU Processor pass per experimental inference step.

`--prompt_measurement_enabled` writes the same `prompt_protocol.json`,
`prompt_diagnostics.jsonl` and `per_episode.csv` for a `legacy` control run
without changing its prompt path, injection point or 320-character budget. It
is used by the seventh legacy arm of the ablation.

The run writes `prompt_protocol.json` once and `prompt_diagnostics.jsonl` per
inference step, plus `per_episode.csv`; `scripts/eval_habitat.sh` merges shard
diagnostics into the group root. The root `config_snapshot.yaml` keeps
`protocol_version` for the Local Graph backend and stores the prompt/experiment
protocol separately as `prompt_protocol_version`; this prevents one field from
being overwritten by two protocols. `scripts/run_l1_l3_prompt_ablation.sh` runs
the six experimental modes plus the unchanged `legacy` control on one fixed
88-Episode manifest and `scripts/compare_l1_l3_prompt_ablation.py` writes the
paired comparison. See
[L1_L3_DUAL_MEMORY_PROMPT_ABLATION.md](research/L1_L3_DUAL_MEMORY_PROMPT_ABLATION.md).

If inference finished but post-processing failed, use the same root and GPU
order with `--resume`:

```bash
GPU_IDS='0 1' MODEL_PATH=... YOLO_MODEL_PATH=... SEG_MODEL_PATH=... \
EPISODE_MANIFEST_PATH=... PILOT_MANIFEST_PATH=... \
  bash scripts/run_l1_l3_prompt_ablation.sh \
    --output-root /existing/root --resume
```

A completed group is verified and skipped; a group with complete shards but no
completion marker is finalized by `scripts/finalize_l1_l3_group.py` without
rerunning inference; a group with missing/partial shards stops the launcher and
is preserved. The resume path checks the provenance manifest/checkpoint/L1/L3/
Habitat/pilot SHA256, seed and GPU order before touching any group.

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
is evaluator-side trace data for legacy modes. Explicit ConceptGraphs-Lite L3
uses the separate same-step `spatial_rgbd` calibration and pose record. Resolve
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
