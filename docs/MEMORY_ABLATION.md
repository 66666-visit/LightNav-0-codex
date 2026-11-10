# Full R2R val_unseen memory ablation and A/B comparison

This is the formal evaluation path. It runs all four conditions by default, or
only the conditions named by `--groups`. The earlier one-episode
commands in [MEMORY_MVP.md](MEMORY_MVP.md) are smoke tests, not formal results.
The shipped R2R `val_unseen` split contains **1,839 episodes**. The formal
launcher reads the actual server dataset manifests before starting inference,
prints the loaded count, requires exactly 1,839, and checks that every
loaded `(scene_id, episode_id)` appears exactly once in each condition.

## Prerequisites and run

Use the already working AutoDL Habitat Python 3.9 environment and LightNav
Python 3.11 inference environment. The checkpoint, R2R dataset, local YOLO11n
`.pt` file, and local Hugging Face DINOv2-Small directory must already exist.
The DINO directory needs `config.json`, `preprocessor_config.json`, and local
weights. This script installs nothing and downloads nothing.

From the repository root, set paths to those existing assets:

```bash
export MODEL_PATH=/path/to/LightNav-0
export YOLO_MODEL_PATH=/path/to/local/yolo11n.pt
export DINO_MODEL_PATH=/path/to/local/dinov2-small
export GPU_IDS=0
export SEED=42
bash scripts/run_memory_ablation.sh --output-root output/memory_ablation_001
```

For the original LightNav versus the complete hierarchical topology agent:

```bash
bash scripts/run_memory_ablation.sh \
  --output-root output/memory_ab_001 \
  --groups baseline_off,hierarchical_topology
```

This A/B command runs only those two groups. Omit `--groups` to retain the
original four-condition experiment. Group names must be unique and selected
from the table below; the order in `--groups` is the comparison direction.

`--output-root` must name a new directory. If the R2R files are outside the
paths in `habitat_server/configs/vlnce_r2r.yaml`, set `DATA_PATH` and
`SCENES_DIR` to the same existing dataset and scene roots used for the
verified baseline. `HABITAT_CONFIG` can point to that same baseline R2R config.
Set `ACTION_TOKENIZER_BUNDLE` only if the checkpoint does not contain its own
RVQ bundle. `CLIENT_PYTHON` and `HABITAT_PYTHON` can select the existing
environments when they differ from the launcher defaults.

The launcher runs the arms sequentially, using identical model, R2R split,
Habitat config, seed, GPU shard layout, action decoder, backend, and inference
settings. It rejects `EPISODES` other than `-1`, `MAX_STEPS` other than `500`,
language filters, custom server arguments, and settings that would force a
budget-end STOP. Both client and server use the VLN-CE **500-step** limit.
The model's own STOP prediction and the environment's normal termination still
apply; reaching the standard limit is recorded as truncation.

| Condition | Memory | RGB object detector | L1/L2/L3 | Visual place graph |
| --- | --- | --- | --- | --- |
| `baseline_off` | off | off | off | off |
| `flat_memory` | flat Retriever V2 | YOLO11n every 5 steps | flat path | off |
| `hierarchical_memory` | hierarchical | YOLO11n every 5 steps | short, mid, long | off |
| `hierarchical_topology` | hierarchical | YOLO11n every 5 steps | short, mid, long | DINOv2-Small every 5 steps; cosine threshold 0.85 |

The `0.85` threshold is an initial setting. It has not been calibrated on
full R2R RGB observations. No arm changes SlowFast sampling, model weights,
RVQ decoding, or the memory text's existing 600-character limit.

## Resume an interrupted experiment

Keep the original output directory and use the **same environment variables**:

```bash
bash scripts/run_memory_ablation.sh --resume output/memory_ablation_001
```

For the A/B example, use `--resume output/memory_ab_001`. The saved group list
is used automatically; supplying `--groups` again is optional and must match
the original selection exactly.

The immutable `ablation_run.json` rejects changed checkpoint, R2R config
contents, paths, seed, shard layout, group selection, or other common settings. An arm with
`formal_complete.json` is rechecked against its manifest and skipped. An
unfinished arm resumes its shard results, skipping already completed Episode
IDs. Completed JSONL rows remain on disk; duplicate Episode IDs or malformed
rows cause an error rather than being silently counted twice. Launcher wall
time is measured per attempt; an interrupted attempt without a measured end
time is retained as unmeasured, not estimated.
On resume, an incomplete final JSONL row is archived separately before the
completed rows are read. Diagnostics for an Episode without a committed result
are also archived; previously completed Episode rows remain in place.

## Files and completion criteria

```text
output/memory_ablation_001/
  ablation_run.json                 immutable common run settings and selected groups
  episode_manifest.json            shared loaded R2R Episode IDs
  comparison.json                  selected-arm means, intervals, paired deltas
  comparison.csv                   compact metric table
  <one directory per selected group>/
    config_snapshot.yaml           JSON-formatted YAML of actual CLI/server settings
    episode_manifest.json          same IDs as the shared manifest
    results.jsonl                  one merged row per completed Episode
    summary.json                   benchmark aggregate metrics
    memory_diagnostics.jsonl        measured per-step latency, prompt and memory data
    run_timing.json                measured launcher attempts
    formal_complete.json           written only after exact coverage validation
    shard_*/                        per-shard results, manifest and diagnostics
    logs/                           server/client logs
```

Each `summary.json` reports SR, OS, SPL, nDTW, NE, mean steps, and elapsed
time. `memory_diagnostics.jsonl` supplies actual per-step inference latency,
YOLO/DINO extraction duration when run, topology counts, revisit and transition
events, injected record count, and prompt character count. The baseline has
per-step inference and prompt diagnostics without memory extraction. Missing
measurements remain missing; the comparison tool does not invent values.

The formal completion gate requires all selected conditions to use the same
manifest, every shard and merged result to cover its assigned IDs once,
complete per-step diagnostics, and a summary count equal to the loaded count.
Formal results also require Habitat termination or truncation and measured SR,
OS, SPL, nDTW, and navigation error for every Episode.
It writes `formal_complete.json` only after these checks pass. Output from a
failed or interrupted arm is retained but is **not** a formal completed result.

The launcher calls the comparison tool after all selected arms finish. To inspect
an incomplete run without labeling it complete:

```bash
python3 scripts/compare_memory_ablation.py output/memory_ablation_001 --allow-incomplete
```

For a completed run, `comparison.json` includes 95% episode-bootstrap
intervals and paired per-Episode differences for SR, OS, SPL, nDTW, NE,
steps, and inference latency, plus measured run time and memory diagnostics.
For A/B, the paired report also counts SR success-to-failure and
failure-to-success Episodes; SR, SPL, nDTW, and NE paired differences each
include a 95% bootstrap interval.
The bootstrap RNG seed is separate from the Habitat evaluation seed.
The comparison tool streams step diagnostics so a full validation set does not
need to keep every Prompt text and layer record in memory at once.

## Fixed 11-scene Pilot A/B experiment

The Pilot runs **220 selected Episodes** from R2R `val_unseen`. It targets 20
per scene. A scene with fewer than 20 contributes every available Episode;
missing seats go one per scene per pass to other scenes, ranked by remaining
candidate count (descending), then `scene_id` (ascending). The confirmed
1,839-Episode distribution has one scene, `pLe4wQe7qrG`, with 18 Episodes.
The resulting quotas are **18 in that scene, 21 in two scenes, and 20 in each
of the other eight**. The two receiving scenes are determined from the actual
dataset counts and recorded in the manifest.

Within each scene, source `geodesic_distance` ranks routes into short, medium,
and long thirds. Sampling seed 42 favors distinct `trajectory_id` values. The
distance quotas are 6/6/6 for 18, 7/7/6 for 20, and 7/7/7 for 21; other
quotas split as evenly as possible with extra seats assigned short, then
medium. The sampler fails if a required route-length field, scene count, or
distance stratum is unavailable. It does not use either agent's scores.
Route lengths, trajectories, and the fixed selection are evaluation metadata
and never enter the navigation policy.

Set `DATA_PATH` to the existing **R2R dataset root** containing
`val_unseen/val_unseen.json.gz` (for example,
`/path/to/R2R_VLNCE_v1-3_preprocessed`). Set `SCENES_DIR` when the Habitat
configuration uses a scene root other than `data/scene_datasets/`; the sampler
must produce the same scene IDs as the Habitat loader. With the checkpoint and
local YOLO/DINO assets set as above, launch only the Pilot A/B pair:

```bash
export DATA_PATH=/path/to/R2R_VLNCE_v1-3_preprocessed
export SCENES_DIR=/path/to/scene_datasets
bash scripts/run_memory_ablation.sh \
  --groups baseline_off,hierarchical_topology \
  --pilot-per-scene 20 \
  --output-root output/r2r_pilot_220
```

The launcher writes `pilot_manifest.json` once in the new output directory,
including source dataset SHA-256, original scene counts, actual scene quotas,
adjustment reasons, sampled IDs, route-length groups, and trajectory IDs.
Each arm's Habitat server filters to those exact
IDs **before sharding or stepping**; `EPISODES=-1` then iterates its selected
shard, rather than taking the first 220 of the full split. Both arms use the
same fixed manifest, checkpoint, seed, 500-step limit, normal STOP/truncation,
SlowFast, and RVQ path. The loaded shard manifests must exactly match the
selection before inference starts. Pilot mode accepts only the A/B pair; omit
`--pilot-per-scene` for the unchanged full 1,839-Episode, four-arm default.

To continue an interrupted Pilot with the same environment variables:

```bash
bash scripts/run_memory_ablation.sh --resume output/r2r_pilot_220
```

Resume reuses `pilot_manifest.json` without drawing another sample. It rejects
changed settings, source dataset bytes, or sample manifest bytes, and skips
already completed Episode IDs without counting them twice. An arm is complete
only after its shard results, merged results, summary, and per-step diagnostics
pass exact coverage checks. Pilot completion writes **`pilot_complete.json`**
in each arm; it never writes `formal_complete.json` or labels Pilot metrics as
the full `val_unseen` score. A partial run retains its JSONL results and has
no completion marker.

The Pilot output has the same per-arm logs, `config_snapshot.yaml`,
`episode_manifest.json`, `results.jsonl`, `summary.json`,
`memory_diagnostics.jsonl`, and `run_timing.json` described above. The root
also contains `ablation_run.json`, the shared `episode_manifest.json`, fixed
`pilot_manifest.json`, `comparison.json`, and `comparison.csv` after both arms
complete. The comparison reports measured A/B SR, OS, SPL, nDTW, NE, success
counts and paired changes, per-Episode paired differences and bootstrap 95%
intervals, per-scene results, a scene macro mean, and an estimate weighted by
each scene's **full** validation-set Episode count. The weighted estimate is a
Pilot extrapolation, not an evaluated full-set result. Runtime and YOLO,
DINOv2, and topology diagnostics come from recorded logs. With only 11 scenes,
Pilot confidence intervals have substantial scene-sampling uncertainty.

## Verification before AutoDL execution

In the Python 3.11 inference environment:

```bash
PYTHONPATH=src python -m pytest -q tests/memory tests/habitat/test_policy.py tests/habitat/test_runner.py tests/habitat/test_memory_ablation_launcher.py tests/habitat/test_experiment_integrity.py tests/habitat/test_pilot_server_selection.py tests/test_sample_r2r_pilot.py tests/test_compare_memory_ablation.py
python -B -m compileall -q src tests scripts habitat_server
git diff --check
```

The current local WSL environment has Python 3.10 and lacks NumPy, pytest,
Ruff, GPU, Habitat data, and local model weights. It can run only dependency-free
unit checks; no full R2R result or real DINOv2 performance claim comes from it.

## RGM V1: new 88-Episode A/C/D Pilot

This is a separate protocol from the 220-Episode experiment above. A is the
unchanged `baseline_off` policy, C is `rgm_gates` (Write, Retrieval and Prompt
Gates with existing fixed-rate YOLO and global DINO), and D is `rgm_full`
(the same gates plus category-level YOLO temporal support, two-stage DINO
matching and conservative topology motion evidence). B is not run. C and D
share the same 88 Episode IDs, checkpoint, visual history, action decoder,
500-step limit and 3-record/400-character memory budget.

On AutoDL, point to the old 220 selection and artifacts. No local copy is
required during development:

```bash
MODEL_PATH=/path/to/LightNav-0 \
YOLO_MODEL_PATH=/path/to/yolo11n.pt \
DINO_MODEL_PATH=/path/to/dinov2-small \
DATA_PATH=/path/to/R2R_VLNCE_v1-3_preprocessed \
SCENES_DIR=/path/to/scene_datasets \
bash scripts/run_memory_ablation.sh --rgm-88 \
  --parent-manifest output/r2r_pilot_220_v2/pilot_manifest.json \
  --legacy-root output/r2r_pilot_220_v2 \
  --output-root output/rgm_pilot_88_v1
```

Resume with the same paths and `--resume`. The new sampler selects 8 of the
parent's Episodes per scene, with 3/3/2 short/medium/long strata, Seed 42 and
trajectory diversity. It never looks at old model outcomes. The A reuse audit
reads the old results, diagnostics, config snapshot and completion marker.
It rejects missing result rows, incompatible model/evaluation settings and
missing source or weight fingerprints; the launcher then runs A on the 88 IDs.
Old artifacts are never modified. If reuse succeeds, old navigation metrics
are paired with C/D; old latency is excluded from cross-hardware comparisons.

Each completed group writes exact coverage artifacts and a Pilot marker.
`comparison.json` contains all three group summaries, paired Episode deltas
and seeded Episode-bootstrap 95% intervals. A missing marker, diagnostic step
or incompatible C/D configuration prevents a complete comparison. RGB and
unverified traversal claims are not retained as validated physical state.
