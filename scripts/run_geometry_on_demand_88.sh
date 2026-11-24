#!/usr/bin/env bash
# L1 Original + Geometry-only L3 + original On-demand Controller (88 Episode).
#
# This is the geometry-retrieval arm of the 660320f CLIP on-demand experiment.
# It uses the same checkpoint, manifests, seed, 500-step cap, YOLO cadence,
# Prompt budget and L1 flags; only the L3 candidate source changes:
#   CLIP L3ClipQuery  ->  59d09ba SpatialGraphAdapter.build
# No CLIP weight, OpenCLIP import, CLIP embedding or cosine score is needed.
#
# Real AutoDL paths are taken from docs/research/L3_ON_DEMAND_AUTODL.md and can
# be overridden explicitly.  Run this script on the GPU host only; it prepares
# and starts the existing evaluation launcher, it does not contain a CPU smoke.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/LightNav-0}
YOLO_MODEL_PATH=${YOLO_MODEL_PATH:-/root/autodl-tmp/weights/yolo11n.pt}
SEG_MODEL_PATH=${SEG_MODEL_PATH:-/root/autodl-tmp/weights/yolo11n-seg.pt}
EPISODE_MANIFEST_PATH=${EPISODE_MANIFEST_PATH:-/root/autodl-tmp/manifests/88_episode_manifest.json}
PILOT_MANIFEST_PATH=${PILOT_MANIFEST_PATH:-/root/autodl-tmp/manifests/88_episode_pilot.json}
DATA_PATH=${DATA_PATH:-/root/autodl-tmp/data}
SCENES_DIR=${SCENES_DIR:-/root/autodl-tmp/scenes}
HABITAT_CONFIG=${HABITAT_CONFIG:-$REPO_ROOT/habitat_server/configs/vlnce_r2r.yaml}
CLIENT_PYTHON=${CLIENT_PYTHON:-$REPO_ROOT/.venv311/bin/python}
HABITAT_CONDA_ENV=${HABITAT_CONDA_ENV:-habitat}

for path in \
    "$MODEL_PATH" "$YOLO_MODEL_PATH" "$SEG_MODEL_PATH" \
    "$EPISODE_MANIFEST_PATH" "$PILOT_MANIFEST_PATH" \
    "$HABITAT_CONFIG" "$CLIENT_PYTHON"
do
    if [ ! -e "$path" ]; then
        echo "[geometry_on_demand] ERROR: missing required path: $path" >&2
        exit 1
    fi
done
for path in "$DATA_PATH" "$SCENES_DIR"; do
    if [ ! -d "$path" ]; then
        echo "[geometry_on_demand] ERROR: missing data directory: $path" >&2
        exit 1
    fi
done

export MODEL_PATH
export YOLO_MODEL_PATH
export SEG_MODEL_PATH
export EPISODE_MANIFEST_PATH
export PILOT_MANIFEST_PATH
export DATA_PATH
export SCENES_DIR
export HABITAT_CONFIG
export CLIENT_PYTHON
export HABITAT_CONDA_ENV

export BACKEND=${BACKEND:-vllm_local}
export TASK=${TASK:-vlnce}
export SPLIT=${SPLIT:-val_unseen}
export GPU_IDS=${GPU_IDS:-0}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.65}
export EPISODES=${EPISODES:--1}
export MAX_STEPS=${MAX_STEPS:-500}
export STRICT_FULL_EVAL=${STRICT_FULL_EVAL:-1}
export EXPECTED_EPISODES=${EXPECTED_EPISODES:-88}
export NO_FORCE_STOP=${NO_FORCE_STOP:-1}
export SEED=${SEED:-42}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/output/geometry_on_demand_l1_original_88_$(date +%Y%m%d_%H%M%S)}

# Frozen L1 Original flags and the original on-demand trigger/config values.
# Deliberately no --spatial_l3_clip_* flag: this arm must not need CLIP.
export CLIENT_ARGS="--no_force_stop --verbose \
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
  --prompt_measurement_enabled"

echo "[geometry_on_demand] output: $OUTPUT_ROOT"
echo "[geometry_on_demand] geometry retrieval: SpatialGraphAdapter.build (no CLIP)"
exec bash scripts/eval_habitat.sh
