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

## Verification before AutoDL execution

In the Python 3.11 inference environment:

```bash
PYTHONPATH=src python -m pytest -q tests/memory tests/habitat/test_policy.py tests/habitat/test_runner.py tests/habitat/test_memory_ablation_launcher.py tests/test_compare_memory_ablation.py
python -B -m compileall -q src tests scripts
git diff --check
```

The current local WSL environment has Python 3.10 and lacks NumPy, pytest,
Ruff, GPU, Habitat data, and local model weights. It can run only dependency-free
unit checks; no full R2R result or real DINOv2 performance claim comes from it.
