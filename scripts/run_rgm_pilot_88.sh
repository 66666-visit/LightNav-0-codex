#!/usr/bin/env bash
# RGM V1 A/C/D Pilot. All actual evaluation occurs only when explicitly invoked.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="$REPO_ROOT/output/rgm_pilot_88_v1"
legacy_root="$REPO_ROOT/output/r2r_pilot_220_v2"
parent_manifest=""
resume=0

die() { echo "[rgm_88] ERROR: $*" >&2; exit 1; }
usage() {
    echo 'Usage: MODEL_PATH=... YOLO_MODEL_PATH=... DINO_MODEL_PATH=... DATA_PATH=... bash scripts/run_memory_ablation.sh --rgm-88 --parent-manifest OLD_220_MANIFEST [--legacy-root OLD_RUN] [--output-root NEW_RUN] [--resume]'
}
while (($#)); do
    case "$1" in
        --parent-manifest) [ "$#" -ge 2 ] || die 'missing parent Manifest'; parent_manifest=$2; shift 2 ;;
        --legacy-root) [ "$#" -ge 2 ] || die 'missing legacy root'; legacy_root=$2; shift 2 ;;
        --output-root) [ "$#" -ge 2 ] || die 'missing output root'; output_root=$2; shift 2 ;;
        --resume) resume=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
[ -n "$parent_manifest" ] || die '--parent-manifest is required'
[ -f "$parent_manifest" ] || die "old Pilot Manifest missing: $parent_manifest"
parent_manifest="$(cd "$(dirname "$parent_manifest")" && pwd -P)/$(basename "$parent_manifest")"
output_root="$(python3 - "$output_root" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PY
)"
legacy_root="$(python3 - "$legacy_root" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PY
)"
[ -n "${MODEL_PATH:-}" ] && [ -d "$MODEL_PATH" ] || die 'MODEL_PATH must be a local checkpoint'
[ -n "${YOLO_MODEL_PATH:-}" ] && [ -f "$YOLO_MODEL_PATH" ] || die 'YOLO_MODEL_PATH must be a local .pt file'
[ -n "${DINO_MODEL_PATH:-}" ] && [ -d "$DINO_MODEL_PATH" ] || die 'DINO_MODEL_PATH must be a local model directory'
[ -n "${DATA_PATH:-}" ] && [ -f "$DATA_PATH/val_unseen/val_unseen.json.gz" ] || die 'DATA_PATH must contain R2R val_unseen'
[ "${SEED:-42}" = 42 ] || die 'RGM Pilot seed is fixed at 42'
[ "${BACKEND:-vllm_local}" = vllm_local ] || die 'RGM Pilot backend is fixed at vllm_local'
[ -z "${CLIENT_ARGS:-}" ] || die 'CLIENT_ARGS is controlled by the RGM protocol'
[ -z "${SERVER_ARGS:-}" ] || die 'SERVER_ARGS is controlled by the RGM protocol'
[ -z "${LANGUAGES:-}" ] || die 'RGM Pilot uses unfiltered R2R val_unseen'
for knob in VLN_EVAL_TEMPERATURE VLN_EVAL_TOP_P VLN_EVAL_TOP_K VLN_EVAL_TRAJ_TOP1; do
    [ -z "${!knob:-}" ] || die "$knob changes the frozen inference protocol"
done
for path in "$MODEL_PATH" "$YOLO_MODEL_PATH" "$DINO_MODEL_PATH" "$DATA_PATH"; do
    [[ "$path" != *[[:space:]]* ]] || die "path contains whitespace unsupported by CLIENT_ARGS: $path"
done
inference_python="${CLIENT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[ -x "$inference_python" ] || inference_python=python3
pilot_manifest="$output_root/pilot_manifest.json"
shared_manifest="$output_root/episode_manifest.json"
if [ "$resume" -eq 0 ]; then
    [ ! -e "$pilot_manifest" ] || die 'existing Pilot Manifest; use --resume'
    mkdir -p "$output_root"
    sample_args=(--parent-manifest "$parent_manifest" --data-path "$DATA_PATH"
                 --output "$pilot_manifest" --seed 42)
    [ -n "${SCENES_DIR:-}" ] && sample_args+=(--scenes-dir "$SCENES_DIR")
    python3 "$REPO_ROOT/scripts/sample_r2r_pilot.py" "${sample_args[@]}"
else
    [ -f "$pilot_manifest" ] || die 'resume requires existing Pilot Manifest'
fi
pilot_sha="$(sha256sum "$pilot_manifest" | cut -d' ' -f1)"
parent_sha="$(sha256sum "$parent_manifest" | cut -d' ' -f1)"
HABITAT_CONFIG="${HABITAT_CONFIG:-$REPO_ROOT/habitat_server/configs/vlnce_r2r.yaml}"
export MODEL_PATH YOLO_MODEL_PATH DINO_MODEL_PATH DATA_PATH HABITAT_CONFIG
export RGM_LEGACY_ROOT="$legacy_root"
export SEED=42 BACKEND=vllm_local
export GPU_IDS="${GPU_IDS:-0}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"
export YOLO_DEVICE="${YOLO_DEVICE:-cpu}" DINO_DEVICE="${DINO_DEVICE:-cpu}"
export PILOT_MANIFEST_PATH="$pilot_manifest" EPISODE_MANIFEST_PATH="$shared_manifest"
export EXPECTED_EPISODES=88
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - \
    "$output_root/ablation_run.json" "$pilot_sha" "$parent_sha" "$resume" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
snapshot = {
    "protocol_version": "rgm_pilot_88_v1",
    "groups": ["baseline_off", "rgm_gates", "rgm_full"],
    "pilot_per_scene": 8,
    "pilot_manifest_sha256": sys.argv[2],
    "parent_manifest_sha256": sys.argv[3],
    "model_path": str(Path(os.environ["MODEL_PATH"]).resolve()),
    "yolo_model_path": str(Path(os.environ["YOLO_MODEL_PATH"]).resolve()),
    "dino_model_path": str(Path(os.environ["DINO_MODEL_PATH"]).resolve()),
    "habitat_config": str(Path(os.environ["HABITAT_CONFIG"]).resolve()),
    "habitat_config_sha256": hashlib.sha256(Path(os.environ["HABITAT_CONFIG"]).read_bytes()).hexdigest(),
    "data_path": str(Path(os.environ["DATA_PATH"]).resolve()),
    "scenes_dir": os.environ.get("SCENES_DIR", ""),
    "action_tokenizer_bundle": os.environ.get("ACTION_TOKENIZER_BUNDLE", ""),
    "legacy_root": str(Path(os.environ["RGM_LEGACY_ROOT"]).resolve()),
    "gpu_ids": os.environ["GPU_IDS"].split(),
    "gpu_memory_utilization": os.environ["GPU_MEM_UTIL"],
    "yolo_device": os.environ["YOLO_DEVICE"],
    "dino_device": os.environ["DINO_DEVICE"],
    "seed": 42, "max_steps": 500,
}
if sys.argv[4] == "1":
    if not path.is_file() or json.loads(path.read_text()) != snapshot:
        raise SystemExit("resume settings differ from immutable RGM run snapshot")
else:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, indent=2)
        stream.write("\n")
PY

common=(--no_force_stop --verbose --memory_top_k 5 --memory_max_context_chars 600)
rgm_common=(
    --memory_enabled --memory_extractor yolo11
    --memory_extractor_model_path "$YOLO_MODEL_PATH"
    --memory_extractor_device "$YOLO_DEVICE"
    --memory_extractor_confidence 0.25 --memory_extract_every_n_steps 5
    --memory_strategy hierarchical --memory_layers short,mid,long
    --memory_place_enabled --memory_place_model_path "$DINO_MODEL_PATH"
    --memory_place_extract_every_n_steps 5 --memory_place_similarity_threshold 0.85
    --memory_place_device "$DINO_DEVICE"
    --rgm_enabled --rgm_prompt_max_records 3 --rgm_prompt_max_chars 400
)

run_group() {
    local group=$1
    shift
    local group_dir="$output_root/$group"
    if [ -f "$group_dir/pilot_complete.json" ]; then
        echo "[rgm_88] $group already complete"
        return
    fi
    local group_resume=0
    [ -d "$group_dir" ] && group_resume=1
    local args=("${common[@]}" "$@")
    local client_args
    printf -v client_args '%s ' "${args[@]}"
    local started=$SECONDS
    MODEL_PATH="$MODEL_PATH" HABITAT_CONFIG="$HABITAT_CONFIG" TASK=vlnce \
        SPLIT=val_unseen EPISODES=-1 MAX_STEPS=500 NO_FORCE_STOP=1 \
        STRICT_FULL_EVAL=1 RESUME="$group_resume" \
        EXPECTED_EPISODES=88 PILOT_MANIFEST_PATH="$pilot_manifest" \
        EPISODE_MANIFEST_PATH="$shared_manifest" OUTPUT_ROOT="$group_dir" \
        SEED=42 GPU_IDS="$GPU_IDS" BACKEND=vllm_local \
        GPU_MEM_UTIL="$GPU_MEM_UTIL" CLIENT_ARGS="$client_args" \
        bash "$REPO_ROOT/scripts/eval_habitat.sh"
    [ -f "$group_dir/pilot_complete.json" ] || die "$group has no completion marker"
    python3 - "$group_dir/run_timing.json" "$group" "$((SECONDS - started))" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
prior = json.loads(path.read_text()) if path.exists() else {"group": sys.argv[2], "attempts": []}
prior["attempts"].append({"exit_code": 0, "wall_time_sec": float(sys.argv[3])})
path.write_text(json.dumps(prior, indent=2) + "\n")
PY
}

run_group rgm_gates "${rgm_common[@]}"
run_group rgm_full "${rgm_common[@]}" \
    --rgm_yolo_temporal_enabled --rgm_place_two_stage_enabled \
    --rgm_topology_motion_validation_enabled

if [ -d "$output_root/baseline_off" ] && \
   [ ! -f "$output_root/baseline_off/pilot_complete.json" ]; then
    echo '[rgm_88] resuming existing incomplete A group'
    run_group baseline_off
elif [ ! -f "$output_root/baseline_off/pilot_complete.json" ]; then
    PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$inference_python" -m scripts.reuse_legacy_a \
        --parent-manifest "$parent_manifest" --selection "$pilot_manifest" \
        --legacy-root "$legacy_root" \
        --target-config "$output_root/rgm_gates/config_snapshot.yaml" \
        --plan-output "$output_root/legacy_a_reuse_plan.json" \
        --materialize-root "$output_root"
    reuse_status="$(python3 - "$output_root/legacy_a_reuse_plan.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["status"])
PY
)"
    if [ "$reuse_status" != compatible ]; then
        echo '[rgm_88] Legacy A cannot be strictly reused; running the 88 Episode A group'
        run_group baseline_off
    fi
fi
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$inference_python" -m scripts.compare_memory_ablation "$output_root"
echo "[rgm_88] complete: $output_root/comparison.json"
