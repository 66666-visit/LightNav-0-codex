#!/usr/bin/env bash
# 1012-only evaluation for the ordered landmark-memory arm.
#
# This is deliberately not the four-Episode launcher: it resolves exactly one
# Habitat Episode ID (1012) from the supplied manifest, runs one server/client
# pair, and never starts the 88-Episode evaluation.
#
# Arm G (ordered) keeps the F contract and adds:
#   --spatial_l3_grounder_ordered_landmarks
# Eligible selection, per-landmark queries and same-category geometry fallback
# remain enabled, so all previously proven 168 behaviour is preserved.
#
# DRY_RUN=1 performs CPU-only validation of assets/manifest/flags and prints
# the exact command; it starts no server, GPU or Habitat process.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL_PATH:?set MODEL_PATH to the LightNav-0 checkpoint directory}"
: "${YOLO_MODEL_PATH:?set YOLO_MODEL_PATH to the local yolo11n.pt}"
: "${SEG_MODEL_PATH:?set SEG_MODEL_PATH to the local yolo11n-seg.pt}"
: "${GROUNDER_CHECKPOINT:?set GROUNDER_CHECKPOINT to the official Nr3D Student checkpoint}"
: "${GROUNDER_BERT_VOCAB:?set GROUNDER_BERT_VOCAB to the local bert vocab.txt}"
: "${EPISODE_MANIFEST_PATH:?set EPISODE_MANIFEST_PATH to a manifest containing 1012}"
: "${DATA_PATH:?set DATA_PATH to the R2R data root}"
: "${SCENES_DIR:?set SCENES_DIR to the scenes root}"

GROUNDER_CONFIG=${GROUNDER_CONFIG:-$REPO_ROOT/references/vil3dref/og3d_src/configs/nr3d_gtlabelpcd_mix_model.yaml}
HABITAT_CONFIG=${HABITAT_CONFIG:-$REPO_ROOT/habitat_server/configs/vlnce_r2r.yaml}
HABITAT_CONDA_ENV=${HABITAT_CONDA_ENV:-habitat}
CLIENT_PYTHON=${CLIENT_PYTHON:-$REPO_ROOT/.venv/bin/python}
DRY_RUN=${DRY_RUN:-0}
EPISODE_ID=1012
BASE_OUTPUT=${OUTPUT_ROOT:-$REPO_ROOT/output/instance_disambiguation_1012_$(date +%Y%m%d_%H%M%S)}
while [[ "$BASE_OUTPUT" == */ && "$BASE_OUTPUT" != / ]]; do
    BASE_OUTPUT="${BASE_OUTPUT%/}"
done
RUN_OUTPUT="$BASE_OUTPUT/ordered_landmarks"
RESOLVED_CONFIG="$BASE_OUTPUT/resolved_config.json"

for path in "$MODEL_PATH" "$YOLO_MODEL_PATH" "$SEG_MODEL_PATH" \
    "$GROUNDER_CHECKPOINT" "$GROUNDER_CONFIG" "$GROUNDER_BERT_VOCAB" \
    "$HABITAT_CONFIG" "$EPISODE_MANIFEST_PATH"; do
    if [ ! -e "$path" ]; then
        echo "[instance_disambiguation_1012] ERROR: missing required path: $path" >&2
        exit 1
    fi
done
for path in "$DATA_PATH" "$SCENES_DIR"; do
    if [ ! -d "$path" ]; then
        echo "[instance_disambiguation_1012] ERROR: missing data directory: $path" >&2
        exit 1
    fi
done
if [ ! -x "$CLIENT_PYTHON" ]; then
    echo "[instance_disambiguation_1012] ERROR: CLIENT_PYTHON is not executable: $CLIENT_PYTHON" >&2
    exit 1
fi

mapfile -t MATCHES < <(python3 - "$EPISODE_MANIFEST_PATH" "$EPISODE_ID" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, wanted = sys.argv[1], str(sys.argv[2])
payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
rows = payload.get("episodes", payload if isinstance(payload, list) else [])
for row in rows:
    if not isinstance(row, dict):
        continue
    episode_id = row.get("habitat_episode_id", row.get("episode_id"))
    if isinstance(episode_id, bool) or not isinstance(episode_id, (str, int)):
        continue
    if str(episode_id) != wanted:
        continue
    rollout = row.get("rollout_idx", row.get("rollout_index", 0))
    scene_id = row.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id:
        continue
    if isinstance(rollout, bool) or not isinstance(rollout, int) or rollout < 0:
        continue
    print(f"{scene_id}\t{rollout}")
PY
)
if [ "${#MATCHES[@]}" -ne 1 ]; then
    echo "[instance_disambiguation_1012] ERROR: expected exactly one manifest row for Episode $EPISODE_ID, matches=${#MATCHES[@]}" >&2
    exit 1
fi
IFS=$'\t' read -r SCENE_ID ROLLOUT_IDX <<< "${MATCHES[0]}"

# Exact same frozen client contract as D/E/F, plus the ordered-memory flag.
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
  --prompt_measurement_enabled"

python3 - "$RESOLVED_CONFIG" "$SCENE_ID" "$EPISODE_ID" "$ROLLOUT_IDX" \
    "$RUN_OUTPUT" "$CLIENT_ARGS" "$DRY_RUN" <<'PY'
import json
import shlex
import sys
from pathlib import Path

(
    resolved_path, scene_id, episode_id, rollout_idx, run_output,
    client_args, dry_run,
) = sys.argv[1:8]
payload = {
    "protocol": "instance_disambiguation_1012_ordered_landmarks_v1",
    "ablation_group": "G",
    "semantics": (
        "joint eligible selection + multi-landmark queries + "
        "same-category geometry fallback + ordered landmark memory"
    ),
    "episode_id": int(episode_id),
    "episode_key": {
        "scene_id": scene_id,
        "habitat_episode_id": episode_id,
        "rollout_idx": int(rollout_idx),
    },
    "output_root": run_output,
    "dry_run": dry_run == "1",
    "client_args": shlex.split(client_args),
    "frozen": {"seed": 42, "max_steps": 500, "episodes_per_run": 1},
}
Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
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
export GPU_IDS=${GPU_IDS:-0}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.65}
export STRICT_FULL_EVAL=0

if [ "$DRY_RUN" = "1" ]; then
    CUDA_VISIBLE_DEVICES= PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$CLIENT_PYTHON" -c "import lightnav.cli.eval_habitat" >/dev/null
    echo "[instance_disambiguation_1012] DRY_RUN: no server or GPU started."
    echo "[instance_disambiguation_1012] ablation group: G (ordered landmark memory)"
    echo "[instance_disambiguation_1012] episode: $EPISODE_ID rollout=$ROLLOUT_IDX scene=$SCENE_ID"
    echo "[instance_disambiguation_1012] resolved config: $RESOLVED_CONFIG"
    echo "  SERVER_ARGS='--single-scene-id $SCENE_ID --single-episode-id $EPISODE_ID' \\"
    echo "  EPISODES=1 MAX_STEPS=500 NO_FORCE_STOP=1 OUTPUT_ROOT='$RUN_OUTPUT' \\"
    echo "  CLIENT_ARGS='$CLIENT_ARGS' bash scripts/eval_habitat.sh"
    exit 0
fi

SERVER_ARGS="--single-scene-id $SCENE_ID --single-episode-id $EPISODE_ID" \
EPISODES=1 MAX_STEPS=500 NO_FORCE_STOP=1 SEED=42 \
OUTPUT_ROOT="$RUN_OUTPUT" \
CLIENT_ARGS="$CLIENT_ARGS" \
    bash scripts/eval_habitat.sh

echo "[instance_disambiguation_1012] evaluation complete: $RUN_OUTPUT"
