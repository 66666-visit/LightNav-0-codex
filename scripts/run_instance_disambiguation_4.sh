#!/usr/bin/env bash
# LightNav-0 instance-disambiguation arm: FOUR diagnostic Episodes only.
#
# Episodes are aligned by (scene basename, exact Habitat episode ID, rollout_idx)
# against an existing evaluator/Pilot manifest.  IDs are compared as strings:
# "episode_001" is never interpreted as Habitat ID 1.  No resampling, no ordinal
# fallback.  The four selected Episodes are run one at a time through the
# existing Habitat server single-Episode selection; no 88-Episode run starts.
#
# Ablation F (joint eligible selection + multi-landmark + geometry fallback) is
# explicit in BASE_CLIENT_ARGS.  The original d659e57 path remains available by
# changing the three grounder flags; this launcher never silently changes it.
#
# DRY_RUN=1 performs CPU-only validation and prints the resolved commands; it
# does not start a server, client or GPU process.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Required inputs: fail instead of guessing AutoDL paths.
: "${MODEL_PATH:?set MODEL_PATH to the LightNav-0 checkpoint directory}"
: "${YOLO_MODEL_PATH:?set YOLO_MODEL_PATH to the local yolo11n.pt}"
: "${SEG_MODEL_PATH:?set SEG_MODEL_PATH to the local yolo11n-seg.pt}"
: "${GROUNDER_CHECKPOINT:?set GROUNDER_CHECKPOINT to the official Nr3D Student checkpoint}"
: "${CLIENT_PYTHON:?set CLIENT_PYTHON to the inference venv python}"
EPISODE_MANIFEST_PATH=${EPISODE_MANIFEST_PATH:-}
: "${DATA_PATH:?set DATA_PATH to the R2R data root}"
: "${SCENES_DIR:?set SCENES_DIR to the scenes root}"

GROUNDER_CONFIG=${GROUNDER_CONFIG:-$REPO_ROOT/references/vil3dref/og3d_src/configs/nr3d_gtlabelpcd_mix_model.yaml}
GROUNDER_BERT_VOCAB=${GROUNDER_BERT_VOCAB:?set GROUNDER_BERT_VOCAB to the local bert vocab.txt}
HABITAT_CONFIG=${HABITAT_CONFIG:-$REPO_ROOT/habitat_server/configs/vlnce_r2r.yaml}
HABITAT_CONDA_ENV=${HABITAT_CONDA_ENV:-habitat}
EPISODE_IDS=${EPISODE_IDS:-"487 168 1080 1012"}
EPISODE_SELECTION=${EPISODE_SELECTION:-}
DRY_RUN=${DRY_RUN:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/output/instance_disambiguation_4_$(date +%Y%m%d_%H%M%S)}

for path in "$MODEL_PATH" "$YOLO_MODEL_PATH" "$SEG_MODEL_PATH" \
    "$GROUNDER_CHECKPOINT" "$GROUNDER_CONFIG" "$GROUNDER_BERT_VOCAB" \
    "$HABITAT_CONFIG" "$CLIENT_PYTHON"; do
    if [ ! -e "$path" ]; then
        echo "[instance_disambiguation] ERROR: missing required path: $path" >&2
        exit 1
    fi
done
for path in "$DATA_PATH" "$SCENES_DIR"; do
    if [ ! -d "$path" ]; then
        echo "[instance_disambiguation] ERROR: missing data directory: $path" >&2
        exit 1
    fi
done
if [ -z "$EPISODE_SELECTION" ] && [ -z "$EPISODE_MANIFEST_PATH" ]; then
    echo "[instance_disambiguation] ERROR: set EPISODE_MANIFEST_PATH or EPISODE_SELECTION" >&2
    exit 1
fi
if [ -z "$EPISODE_SELECTION" ] && [ ! -e "$EPISODE_MANIFEST_PATH" ]; then
    echo "[instance_disambiguation] ERROR: missing manifest: $EPISODE_MANIFEST_PATH" >&2
    exit 1
fi
if [ -e "$OUTPUT_ROOT" ] && [ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit)" ]; then
    echo "[instance_disambiguation] ERROR: output root must be new or empty: $OUTPUT_ROOT" >&2
    exit 1
fi
mkdir -p "$OUTPUT_ROOT"

SELECTION_PATH="$OUTPUT_ROOT/four_episode_selection.json"
RESOLVED_PATH="$OUTPUT_ROOT/resolved_config.json"
python3 - "$EPISODE_SELECTION" "$EPISODE_MANIFEST_PATH" "$EPISODE_IDS" \
    "$SELECTION_PATH" "$RESOLVED_PATH" "$MODEL_PATH" "$YOLO_MODEL_PATH" \
    "$SEG_MODEL_PATH" "$GROUNDER_CHECKPOINT" "$GROUNDER_CONFIG" \
    "$GROUNDER_BERT_VOCAB" "$DATA_PATH" "$SCENES_DIR" "$CLIENT_PYTHON" \
    "$OUTPUT_ROOT" "$DRY_RUN" <<'PY'
import json
import os
import sys
from pathlib import Path

(
    selection_path_raw,
    manifest_path,
    wanted_raw,
    out_selection,
    out_resolved,
    model_path,
    yolo_path,
    seg_path,
    grounder_checkpoint,
    grounder_config,
    grounder_vocab,
    data_path,
    scenes_dir,
    client_python,
    output_root,
    dry_run,
) = sys.argv[1:17]


def _scene_basename(scene_id: str) -> str:
    return os.path.basename(scene_id.rstrip("/"))


def _row_key(row: dict) -> tuple[str, str, int] | None:
    scene_id = row.get("scene_id")
    episode_id = row.get("habitat_episode_id", row.get("episode_id"))
    if not isinstance(scene_id, str) or not scene_id:
        return None
    if isinstance(episode_id, bool) or not isinstance(episode_id, (str, int)):
        return None
    rollout = row.get("rollout_idx", row.get("rollout_index", 0))
    if isinstance(rollout, bool) or not isinstance(rollout, int) or rollout < 0:
        return None
    return _scene_basename(scene_id), str(episode_id), rollout


def _selected_row(row: dict) -> dict:
    scene_id = row["scene_id"]
    episode_id = row.get("habitat_episode_id", row.get("episode_id"))
    rollout = row.get("rollout_idx", row.get("rollout_index", 0))
    return {
        "scene_id": scene_id,
        "scene_basename": _scene_basename(scene_id),
        "habitat_episode_id": str(episode_id),
        "rollout_idx": int(rollout),
    }


if selection_path_raw:
    selection_path = Path(selection_path_raw)
    if not selection_path.is_file():
        raise SystemExit(f"selection file not found: {selection_path}")
    data = json.loads(selection_path.read_text(encoding="utf-8"))
    rows = data.get("episodes") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise SystemExit("selection must contain an episodes list")
    selected = []
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("selection row is not an object")
        key = _row_key(row)
        if key is None:
            raise SystemExit(f"selection row lacks scene_id/ID/rollout: {row}")
        selected.append(_selected_row(row))
    selection_source = "explicit_file"
else:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = manifest.get("episodes")
    if not isinstance(rows, list):
        raise SystemExit("manifest requires an episodes list")
    wanted = [value for value in wanted_raw.split() if value]
    if len(set(wanted)) != len(wanted):
        raise SystemExit("EPISODE_IDS contains duplicates")
    by_id: dict[str, list[dict]] = {value: [] for value in wanted}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _row_key(row)
        if key is None:
            continue
        _, episode_id, _ = key
        if episode_id in by_id:
            by_id[episode_id].append(row)
    selected = []
    for episode_id in wanted:
        matches = by_id.get(episode_id, [])
        if len(matches) != 1:
            raise SystemExit(
                "Episode ID must match exactly one manifest row "
                f"(no ordinal fallback): {episode_id!r} matches={len(matches)}"
            )
        selected.append(_selected_row(matches[0]))
    selection_source = "manifest_exact_id"

if len(selected) != 4:
    raise SystemExit(f"exactly four Episodes are required, got {len(selected)}")
keys = [
    (row["scene_basename"], row["habitat_episode_id"], row["rollout_idx"])
    for row in selected
]
if len(set(keys)) != 4:
    raise SystemExit(f"selection contains duplicate scene/ID/rollout keys: {keys}")
if len({row["habitat_episode_id"] for row in selected}) != 4:
    raise SystemExit("selection contains duplicate Habitat episode IDs")
if any(row["rollout_idx"] != 0 for row in selected):
    raise SystemExit(
        "this R2R launcher supports rollout_idx=0 only; the Habitat client "
        "runs one deterministic rollout per selected Episode"
    )

Path(out_selection).write_text(
    json.dumps({"episodes": selected}, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
resolved = {
    "dry_run": str(dry_run) == "1",
    "selection_source": selection_source,
    "episodes": selected,
    "ablation": {
        "id": "F",
        "semantics": (
            "joint eligible selection + multi-landmark queries + "
            "same-category geometry fallback"
        ),
        "grounder_selection_mode": "eligible",
        "grounder_landmark_queries": True,
        "grounder_geometry_fallback": True,
    },
    "frozen": {
        "model_path": model_path,
        "backend": os.environ.get("BACKEND", "vllm_local"),
        "task": os.environ.get("TASK", "vlnce"),
        "split": os.environ.get("SPLIT", "val_unseen"),
        "episodes_per_run": 1,
        "max_steps": int(os.environ.get("MAX_STEPS", "500")),
        "seed": int(os.environ.get("SEED", "42")),
        "gpu_ids": os.environ.get("GPU_IDS", "0"),
        "gpu_memory_utilization": float(
            os.environ.get("GPU_MEM_UTIL", "0.65")
        ),
    },
    "assets": {
        "yolo_model_path": yolo_path,
        "seg_model_path": seg_path,
        "grounder_checkpoint": grounder_checkpoint,
        "grounder_config": grounder_config,
        "grounder_bert_vocab": grounder_vocab,
        "data_path": data_path,
        "scenes_dir": scenes_dir,
        "client_python": client_python,
    },
    "output_root": output_root,
}
Path(out_resolved).write_text(
    json.dumps(resolved, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(resolved, ensure_ascii=False, indent=2))
PY

# CPU-only preflight: the inference venv can import the client module, and the
# launcher shell parses.  This does not create a CUDA context or start a server.
bash -n scripts/eval_habitat.sh
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$CLIENT_PYTHON" -c "import lightnav.cli.eval_habitat" >/dev/null

export MODEL_PATH YOLO_MODEL_PATH SEG_MODEL_PATH DATA_PATH SCENES_DIR
export HABITAT_CONFIG CLIENT_PYTHON HABITAT_CONDA_ENV
export BACKEND=${BACKEND:-vllm_local}
export TASK=${TASK:-vlnce}
export SPLIT=${SPLIT:-val_unseen}
export GPU_IDS=${GPU_IDS:-0}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.65}
export MAX_STEPS=${MAX_STEPS:-500}
export STRICT_FULL_EVAL=0
export NO_FORCE_STOP=1
export SEED=${SEED:-42}

BASE_CLIENT_ARGS="--trace_steps --no_force_stop --verbose \
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
  --prompt_measurement_enabled"

if [ "$DRY_RUN" = "1" ]; then
    echo "[instance_disambiguation] DRY_RUN: no server or GPU started."
    echo "[instance_disambiguation] resolved config: $RESOLVED_PATH"
    echo "[instance_disambiguation] selection: $SELECTION_PATH"
    while IFS=$'\t' read -r scene_id episode_id rollout_idx; do
        echo "[instance_disambiguation] episode $episode_id rollout=$rollout_idx scene=$scene_id"
        echo "  SERVER_ARGS='--single-scene-id $scene_id --single-episode-id $episode_id' \\"
        echo "  EPISODES=1 OUTPUT_ROOT='$OUTPUT_ROOT/episode_${episode_id}_r${rollout_idx}' \\"
        echo "  CLIENT_ARGS='$BASE_CLIENT_ARGS' bash scripts/eval_habitat.sh"
    done < <(python3 - "$SELECTION_PATH" <<'PY'
import json
import sys
from pathlib import Path
for row in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["episodes"]:
    print(f"{row['scene_id']}\t{row['habitat_episode_id']}\t{row['rollout_idx']}")
PY
)
    exit 0
fi

mapfile -t EPISODE_ROWS < <(python3 - "$SELECTION_PATH" <<'PY'
import json
import sys
from pathlib import Path
for row in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["episodes"]:
    print(f"{row['scene_id']}\t{row['habitat_episode_id']}\t{row['rollout_idx']}")
PY
)
if [ "${#EPISODE_ROWS[@]}" -ne 4 ]; then
    echo "[instance_disambiguation] ERROR: expected 4 Episodes, got ${#EPISODE_ROWS[@]}" >&2
    exit 1
fi

for row in "${EPISODE_ROWS[@]}"; do
    IFS=$'\t' read -r scene_id episode_id rollout_idx <<< "$row"
    episode_out="$OUTPUT_ROOT/episode_${episode_id}_r${rollout_idx}"
    echo "[instance_disambiguation] Episode $episode_id rollout=$rollout_idx ($scene_id)"
    echo "[instance_disambiguation] output: $episode_out"
    SERVER_ARGS="--single-scene-id $scene_id --single-episode-id $episode_id" \
    EPISODES=1 \
    OUTPUT_ROOT="$episode_out" \
    CLIENT_ARGS="$BASE_CLIENT_ARGS" \
        bash scripts/eval_habitat.sh
done

echo "[instance_disambiguation] four-Episode run complete: $OUTPUT_ROOT"
