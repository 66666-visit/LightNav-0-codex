#!/usr/bin/env bash
# C arm: H1 + One-shot Historical Memory Admission (object level).
#
# C is deliberately not part of ABLATION_GROUP D/E/F/G/H1/H2/H3: that launcher
# defines a different rendering ablation.  This script runs exactly one
# controlled Episode (scene 2azQ1b91cZZ, Habitat Episode 168, rollout 0).
#
# The only algorithmic difference from H1 is:
#   --spatial_l3_history_admission_mode once_per_object
# which admits a stable ObjectNode object_id once per Episode at the final L3
# history/Prompt boundary and refuses a later re-selection of the same object,
# even when its Evidence ID changed.  Every other model, data, Habitat, YOLO,
# Grounder, memory and Prompt flag is the uploaded H1 configuration.
#
# DRY_RUN defaults to 1: it validates paths/flags on CPU, writes
# resolved_config.json, prints the exact command and starts no Habitat, GPU or
# server process.  DRY_RUN=0 starts the single Episode.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/LightNav-0-codex-ablation/checkpoints/LightNav-0}
YOLO_MODEL_PATH=${YOLO_MODEL_PATH:-/root/autodl-tmp/models/yolo11/yolo11n.pt}
SEG_MODEL_PATH=${SEG_MODEL_PATH:-/root/autodl-tmp/models/yolo11/yolo11n-seg.pt}
GROUNDER_CHECKPOINT=${GROUNDER_CHECKPOINT:-/root/autodl-tmp/models/vil3dref/model_epoch_96.pt}
GROUNDER_BERT_VOCAB=${GROUNDER_BERT_VOCAB:-/root/autodl-tmp/models/vil3dref/vocab.txt}
DATA_PATH=${DATA_PATH:-/root/autodl-tmp/LightNav-0-codex/data/datasets/R2R_VLNCE_v1-3_preprocessed}
SCENES_DIR=${SCENES_DIR:-/root/autodl-tmp/LightNav-0-codex/data/scene_datasets}

# H1's exact scene asset, including the H1-only -fast scene_datasets root.
SCENE_ID=${SCENE_ID:-/root/autodl-tmp/LightNav-0-fast/data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb}
EPISODE_ID=168
ROLLOUT_IDX=0
SEED=42
MAX_STEPS=500
GPU_IDS=${GPU_IDS:-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}

GROUNDER_CONFIG=${GROUNDER_CONFIG:-$REPO_ROOT/references/vil3dref/og3d_src/configs/nr3d_gtlabelpcd_mix_model.yaml}
HABITAT_CONFIG=${HABITAT_CONFIG:-$REPO_ROOT/habitat_server/configs/vlnce_r2r.yaml}
HABITAT_CONDA_ENV=${HABITAT_CONDA_ENV:-habitat}
CLIENT_PYTHON=${CLIENT_PYTHON:-$REPO_ROOT/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/output/memory_admission_c_168_$TIMESTAMP}
while [[ "$OUTPUT_ROOT" == */ && "$OUTPUT_ROOT" != / ]]; do
    OUTPUT_ROOT="${OUTPUT_ROOT%/}"
done
# eval_habitat requires its own root to be empty; it is a child output so the
# base directory can also hold the resolved experiment config without mixing
# the two contracts.
RUN_OUTPUT="$OUTPUT_ROOT/run"
RESOLVED_CONFIG="$OUTPUT_ROOT/resolved_config.json"

if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
    echo "[memory_admission_c_168] ERROR: DRY_RUN must be 0 or 1 (got '$DRY_RUN')" >&2
    exit 2
fi

for path in "$MODEL_PATH" "$YOLO_MODEL_PATH" "$SEG_MODEL_PATH" \
    "$GROUNDER_CHECKPOINT" "$GROUNDER_CONFIG" "$GROUNDER_BERT_VOCAB" \
    "$HABITAT_CONFIG" "$SCENE_ID" "$CLIENT_PYTHON"; do
    if [ ! -e "$path" ]; then
        echo "[memory_admission_c_168] ERROR: missing required path: $path" >&2
        exit 1
    fi
done
for path in "$DATA_PATH" "$SCENES_DIR" "$MODEL_PATH"; do
    if [ ! -d "$path" ]; then
        echo "[memory_admission_c_168] ERROR: missing directory: $path" >&2
        exit 1
    fi
done
if [ ! -x "$CLIENT_PYTHON" ]; then
    echo "[memory_admission_c_168] ERROR: CLIENT_PYTHON is not executable: $CLIENT_PYTHON" >&2
    exit 1
fi
if [ -e "$OUTPUT_ROOT" ] && [ ! -d "$OUTPUT_ROOT" ]; then
    echo "[memory_admission_c_168] ERROR: output base is not a directory: $OUTPUT_ROOT" >&2
    exit 1
fi
if [ -e "$RUN_OUTPUT" ] && [ ! -d "$RUN_OUTPUT" ]; then
    echo "[memory_admission_c_168] ERROR: run output is not a directory: $RUN_OUTPUT" >&2
    exit 1
fi
if [ -d "$RUN_OUTPUT" ] \
    && [ -n "$(find "$RUN_OUTPUT" -mindepth 1 -print -quit)" ]; then
    echo "[memory_admission_c_168] ERROR: run output must be new or empty: $RUN_OUTPUT" >&2
    exit 1
fi

# H1's exact frozen client contract plus the one C switch.  The rendering
# switches are the H1 pair: no missing-Evidence placeholders, G fact sentences.
CLIENT_ARGS="--trace_steps --no_force_stop --verbose \
  --memory_enabled --memory_strategy hierarchical \
  --memory_layers short,mid,long --memory_top_k 5 --memory_max_context_chars 600 \
  --memory_extractor yolo11 --memory_extractor_model_path $YOLO_MODEL_PATH \
  --memory_extractor_device cpu --memory_extractor_confidence 0.25 \
  --memory_extract_every_n_steps 5 \
  --local_graph_enabled --local_graph_scope episode_local \
  --local_graph_prompt_max_chars 320 --local_graph_max_landmarks 2 \
  --local_graph_reactivation_steps 8 \
  --memory_prompt_mode l1_original_only \
  --spatial_l3_mode geometry_on_demand \
  --spatial_l3_seg_model_path $SEG_MODEL_PATH --spatial_l3_seg_device cpu \
  --spatial_l3_seg_confidence 0.25 --spatial_l3_seg_every_n_steps 5 \
  --spatial_l3_evidence_min_iou 0.5 --spatial_l3_evidence_ambiguity_margin 0.1 \
  --spatial_l3_min_injection_confidence 0.5 \
  --spatial_l3_rgb_sidecar \
  --spatial_l3_instance_grounder \
  --spatial_l3_grounder_checkpoint $GROUNDER_CHECKPOINT \
  --spatial_l3_grounder_config $GROUNDER_CONFIG \
  --spatial_l3_grounder_device cuda \
  --spatial_l3_grounder_bert_vocab $GROUNDER_BERT_VOCAB \
  --spatial_l3_grounder_seed 0 \
  --spatial_l3_grounder_selection eligible \
  --spatial_l3_grounder_landmark_queries \
  --spatial_l3_grounder_geometry_fallback \
  --spatial_l3_grounder_ordered_landmarks \
  --no-spatial_l3_grounder_include_missing_landmark_placeholders \
  --no-spatial_l3_grounder_preserve_original_spatial_fact \
  --spatial_l3_history_admission_mode once_per_object \
  --prompt_measurement_enabled"

mkdir -p "$OUTPUT_ROOT"
python3 - "$RESOLVED_CONFIG" "$SCENE_ID" "$EPISODE_ID" "$ROLLOUT_IDX" \
    "$RUN_OUTPUT" "$CLIENT_ARGS" "$DRY_RUN" "$MODEL_PATH" "$YOLO_MODEL_PATH" \
    "$SEG_MODEL_PATH" "$GROUNDER_CHECKPOINT" "$GROUNDER_CONFIG" \
    "$GROUNDER_BERT_VOCAB" "$DATA_PATH" "$SCENES_DIR" "$CLIENT_PYTHON" \
    "$GPU_IDS" "$GPU_MEM_UTIL" <<'PY'
import json
import shlex
import sys
from pathlib import Path

(
    resolved_path, scene_id, episode_id, rollout_idx, output_root,
    client_args, dry_run, model_path, yolo_path, seg_path, grounder_checkpoint,
    grounder_config, grounder_vocab, data_path, scenes_dir, client_python,
    gpu_ids, gpu_mem_util,
) = sys.argv[1:19]
tokens = shlex.split(client_args)
if tokens[tokens.index("--spatial_l3_history_admission_mode") + 1] != (
    "once_per_object"
):
    raise SystemExit("C arm must pass once_per_object")
if "--no-spatial_l3_grounder_include_missing_landmark_placeholders" not in tokens:
    raise SystemExit("C arm must keep H1's placeholder switch off")
if "--no-spatial_l3_grounder_preserve_original_spatial_fact" not in tokens:
    raise SystemExit("C arm must keep H1's G fact sentence")
payload = {
    "protocol": "memory_admission_c_168_v1",
    "ablation_group": "C",
    "semantics": (
        "H1 + episode-local object-level one-shot historical memory "
        "admission at the final L3 Prompt boundary"
    ),
    "only_change_from_h1": {
        "spatial_l3_history_admission_mode": "once_per_object",
    },
    "h1_rendering": {
        "include_missing_landmark_placeholders": False,
        "preserve_original_spatial_fact": False,
    },
    "episode_key": {
        "scene_id": scene_id,
        "habitat_episode_id": str(episode_id),
        "rollout_idx": int(rollout_idx),
    },
    "episode_count": 1,
    "output_root": output_root,
    "resolved_config_path": resolved_path,
    "dry_run": str(dry_run) == "1",
    "client_args": tokens,
    "assets": {
        "model_path": model_path,
        "yolo_model_path": yolo_path,
        "seg_model_path": seg_path,
        "grounder_checkpoint": grounder_checkpoint,
        "grounder_config": grounder_config,
        "grounder_bert_vocab": grounder_vocab,
        "data_path": data_path,
        "scenes_dir": scenes_dir,
        "client_python": client_python,
        "scene_id": scene_id,
    },
    "frozen": {
        "seed": 42,
        "max_steps": 500,
        "episodes_per_run": 1,
        "force_stop": False,
        "gpu_ids": gpu_ids,
        "gpu_memory_utilization": float(gpu_mem_util),
    },
}
Path(resolved_path).write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY

# eval_habitat.sh is a child process: everything it reads must be exported.
export MODEL_PATH YOLO_MODEL_PATH SEG_MODEL_PATH DATA_PATH SCENES_DIR
export HABITAT_CONFIG CLIENT_PYTHON HABITAT_CONDA_ENV
export BACKEND=${BACKEND:-vllm_local}
export TASK=${TASK:-vlnce}
export SPLIT=${SPLIT:-val_unseen}
export GPU_IDS GPU_MEM_UTIL
export STRICT_FULL_EVAL=0

bash -n scripts/eval_habitat.sh
if [ "$DRY_RUN" = "1" ]; then
    # Hiding CUDA devices keeps DRY_RUN unambiguously CPU-only.
    CUDA_VISIBLE_DEVICES= PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$CLIENT_PYTHON" -c "import lightnav.cli.eval_habitat" >/dev/null
    echo "[memory_admission_c_168] DRY_RUN: no server, GPU or Habitat process started."
    echo "[memory_admission_c_168] ablation group: C (H1 + once_per_object)"
    echo "[memory_admission_c_168] episode: $EPISODE_ID rollout=$ROLLOUT_IDX scene=$SCENE_ID"
    echo "[memory_admission_c_168] resolved config: $RESOLVED_CONFIG"
    echo "  SERVER_ARGS='--single-scene-id $SCENE_ID --single-episode-id $EPISODE_ID' \\"
    echo "  EPISODES=1 MAX_STEPS=500 NO_FORCE_STOP=1 SEED=42 \\"
    echo "  GPU_IDS='$GPU_IDS' GPU_MEM_UTIL='$GPU_MEM_UTIL' OUTPUT_ROOT='$RUN_OUTPUT' \\"
    echo "  CLIENT_ARGS='$CLIENT_ARGS' bash scripts/eval_habitat.sh"
    exit 0
fi

SERVER_ARGS="--single-scene-id $SCENE_ID --single-episode-id $EPISODE_ID" \
EPISODES=1 MAX_STEPS=500 NO_FORCE_STOP=1 SEED=42 \
GPU_IDS="$GPU_IDS" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
OUTPUT_ROOT="$RUN_OUTPUT" \
CLIENT_ARGS="$CLIENT_ARGS" \
    bash scripts/eval_habitat.sh

echo "[memory_admission_c_168] evaluation complete: $OUTPUT_ROOT"
