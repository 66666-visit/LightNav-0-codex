#!/usr/bin/env bash
# Run selected R2R val_unseen memory conditions with one immutable common setup.
# Results remain in separate directories; an interrupted condition can be resumed.
set -euo pipefail

TAG='[memory_ablation]'
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  MODEL_PATH=... YOLO_MODEL_PATH=... DINO_MODEL_PATH=... \
    bash scripts/run_memory_ablation.sh [--output-root DIR] [--groups NAME,NAME] \
      [--pilot-per-scene 20]
  MODEL_PATH=... YOLO_MODEL_PATH=... DINO_MODEL_PATH=... \
    bash scripts/run_memory_ablation.sh --resume DIR [--groups NAME,NAME] \
      [--pilot-per-scene 20]

Defaults to all four conditions. --groups selects from baseline_off, flat_memory,
hierarchical_memory, hierarchical_topology; the two-arm A/B selection is
baseline_off,hierarchical_topology. Runs the complete R2R val_unseen split with 500 steps,
no budget-forced STOP, one common seed, and the same checkpoint/config/shards.
The first command creates a fresh output directory. --resume preserves prior
episode results and skips conditions already verified as formally complete.
Without --groups, --resume uses the selection recorded in the output directory.
Pilot mode requires the A/B groups and DATA_PATH. It targets 20 Episodes per
scene, redistributes any shortage deterministically, and selects 220 total.
It writes pilot_complete.json (never formal_complete.json).
On resume, the fixed pilot_manifest.json is reused without resampling.

Optional common settings: GPU_IDS (default 0), SEED (default 42), BACKEND,
GPU_MEM_UTIL, HABITAT_CONFIG, DATA_PATH, SCENES_DIR, ACTION_TOKENIZER_BUNDLE,
CLIENT_PYTHON, HABITAT_PYTHON, INFER_VENV, HABITAT_CONDA_ENV, BASE_PORT.
EOF
}

die() {
    echo "$TAG ERROR: $*" >&2
    exit 1
}

mode=new
ablation_root="${ABLATION_ROOT:-$REPO_ROOT/output/memory_ablation_$(date +%Y%m%d_%H%M%S)}"
all_groups=(baseline_off flat_memory hierarchical_memory hierarchical_topology)
groups_csv=""
groups_set=0
pilot_per_scene=""
pilot_set=0
while (($#)); do
    case "$1" in
        --output-root)
            [ "$mode" = new ] && [ "$#" -ge 2 ] || die '--output-root needs a directory'
            ablation_root=$2
            shift 2
            ;;
        --resume)
            [ "$mode" = new ] && [ "$#" -ge 2 ] || die '--resume needs a directory'
            mode=resume
            ablation_root=$2
            shift 2
            ;;
        --groups)
            [ "$groups_set" -eq 0 ] && [ "$#" -ge 2 ] || die '--groups needs one comma-separated list'
            groups_csv=$2
            groups_set=1
            shift 2
            ;;
        --pilot-per-scene)
            [ "$pilot_set" -eq 0 ] && [ "$#" -ge 2 ] || die '--pilot-per-scene needs one value'
            pilot_per_scene=$2
            pilot_set=1
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) die "unknown argument: $1" ;;
    esac
done
if [ "$pilot_set" -eq 1 ] && [ -z "$pilot_per_scene" ]; then
    die '--pilot-per-scene requires 20'
fi

if [ "$mode" = resume ] && [ "$pilot_set" -eq 0 ]; then
    [ -f "$ablation_root/ablation_run.json" ] || die "missing resume snapshot: $ablation_root/ablation_run.json"
    pilot_per_scene=$(python3 - "$ablation_root/ablation_run.json" <<'PY'
import json
import sys
from pathlib import Path

saved = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = saved.get("pilot_per_scene", "")
print(value if isinstance(value, int) and not isinstance(value, bool) else "")
PY
    )
fi
if [ -n "$pilot_per_scene" ]; then
    [ "$pilot_per_scene" = 20 ] || die '--pilot-per-scene must be 20 for this Pilot protocol'
fi

if [ "$mode" = resume ] && [ "$groups_set" -eq 0 ]; then
    [ -f "$ablation_root/ablation_run.json" ] || die "missing resume snapshot: $ablation_root/ablation_run.json"
    groups_csv=$(python3 - "$ablation_root/ablation_run.json" <<'PY'
import json
import sys
from pathlib import Path

saved = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
groups = saved.get("groups", [
    "baseline_off", "flat_memory", "hierarchical_memory", "hierarchical_topology",
])
if not isinstance(groups, list) or not all(isinstance(name, str) for name in groups):
    raise SystemExit("invalid groups in ablation_run.json")
print(",".join(groups))
PY
    )
elif [ "$groups_set" -eq 0 ]; then
    groups_csv=$(IFS=,; echo "${all_groups[*]}")
fi
if [ -z "$groups_csv" ] || [[ "$groups_csv" == ,* || "$groups_csv" == *, || "$groups_csv" == *,,* || "$groups_csv" == *$'\n'* ]]; then
    die '--groups requires a nonempty comma-separated list without empty entries'
fi
IFS=, read -ra selected_groups <<< "$groups_csv"
seen_groups=()
for group in "${selected_groups[@]}"; do
    case "$group" in
        baseline_off|flat_memory|hierarchical_memory|hierarchical_topology) ;;
        *) die "invalid --groups entry: $group" ;;
    esac
    for prior in "${seen_groups[@]}"; do
        [ "$prior" != "$group" ] || die "duplicate --groups entry: $group"
    done
    seen_groups+=("$group")
done
if [ -n "$pilot_per_scene" ]; then
    [ "${#selected_groups[@]}" -eq 2 ] \
        && [ "${selected_groups[0]}" = baseline_off ] \
        && [ "${selected_groups[1]}" = hierarchical_topology ] \
        || die 'Pilot requires --groups baseline_off,hierarchical_topology'
fi

# Reject settings that would silently turn a formal full-split comparison into
# a small smoke run, change the benchmark, or make selected arms incomparable.
[ "${TASK:-vlnce}" = vlnce ] || die 'TASK must be vlnce'
[ "${SPLIT:-val_unseen}" = val_unseen ] || die 'SPLIT must be val_unseen'
[ "${EPISODES:--1}" = -1 ] || die 'EPISODES must be -1 (the complete split)'
[ "${MAX_STEPS:-500}" = 500 ] || die 'MAX_STEPS must be 500'
[ "${NO_FORCE_STOP:-1}" = 1 ] || die 'NO_FORCE_STOP must be 1'
[ -z "${CLIENT_ARGS:-}" ] || die 'set common settings through this launcher, not CLIENT_ARGS'
[ -z "${SERVER_ARGS:-}" ] || die 'SERVER_ARGS could change benchmark termination'
[ -z "${LANGUAGES:-}" ] || die 'LANGUAGES would filter R2R episodes'
[ -z "${SUCCESS_DISTANCE:-}" ] || die 'SUCCESS_DISTANCE must use the R2R config'
[ -z "${NAVMESH_CELL_HEIGHT:-}" ] || die 'NAVMESH_CELL_HEIGHT does not apply to R2R'
[ -z "${LOG_DIR:-}" ] || die 'LOG_DIR would mix logs from different conditions'
if [ -n "$pilot_per_scene" ]; then
    [ "${EXPECTED_EPISODES:-220}" = 220 ] || die 'Pilot requires EXPECTED_EPISODES=220'
    EXPECTED_EPISODES=220
    [ -n "${DATA_PATH:-}" ] || die 'Pilot requires DATA_PATH to the R2R dataset root'
    [ -f "$DATA_PATH/val_unseen/val_unseen.json.gz" ] ||
        die "Pilot DATA_PATH must contain val_unseen/val_unseen.json.gz: $DATA_PATH"
    DATA_PATH="$(cd "$DATA_PATH" && pwd -P)"
else
    [ "${EXPECTED_EPISODES:-1839}" = 1839 ] || die 'EXPECTED_EPISODES must be 1839 for full R2R val_unseen'
    EXPECTED_EPISODES=1839
fi

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the LightNav checkpoint directory}
YOLO_MODEL_PATH=${YOLO_MODEL_PATH:-}
DINO_MODEL_PATH=${DINO_MODEL_PATH:-}
HABITAT_CONFIG=${HABITAT_CONFIG:-$REPO_ROOT/habitat_server/configs/vlnce_r2r.yaml}
GPU_IDS=${GPU_IDS:-0}
SEED=${SEED:-42}
BACKEND=${BACKEND:-vllm_local}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.65}
YOLO_DEVICE=${YOLO_DEVICE:-cpu}
DINO_DEVICE=${DINO_DEVICE:-cpu}

[ -d "$MODEL_PATH" ] || die "checkpoint directory not found: $MODEL_PATH"
need_yolo=0
need_dino=0
for group in "${selected_groups[@]}"; do
    [ "$group" = baseline_off ] || need_yolo=1
    [ "$group" = hierarchical_topology ] && need_dino=1
done
[ "$need_yolo" -eq 1 ] || YOLO_MODEL_PATH=""
[ "$need_dino" -eq 1 ] || DINO_MODEL_PATH=""
if [ "$need_yolo" -eq 1 ]; then
    [ -s "$YOLO_MODEL_PATH" ] || die "YOLO weights not found or empty: $YOLO_MODEL_PATH"
fi
if [ "$need_dino" -eq 1 ]; then
    [ -d "$DINO_MODEL_PATH" ] || die "DINOv2 directory not found: $DINO_MODEL_PATH"
    [ -f "$DINO_MODEL_PATH/config.json" ] || die 'DINOv2 config.json is missing'
    [ -f "$DINO_MODEL_PATH/preprocessor_config.json" ] || die 'DINOv2 preprocessor_config.json is missing'
    if ! compgen -G "$DINO_MODEL_PATH/*.safetensors" >/dev/null &&
       ! compgen -G "$DINO_MODEL_PATH/pytorch_model*.bin" >/dev/null; then
        die 'DINOv2 local weight files are missing'
    fi
fi
[ -f "$HABITAT_CONFIG" ] || die "Habitat config not found: $HABITAT_CONFIG"
[[ "$SEED" =~ ^[0-9]+$ ]] || die 'SEED must be a nonnegative integer'
[[ "$GPU_IDS" =~ ^[0-9]+(\ [0-9]+)*$ ]] || die 'GPU_IDS must be space-separated GPU indices'
for path in "$YOLO_MODEL_PATH" "$DINO_MODEL_PATH"; do
    [[ "$path" != *[[:space:]]* ]] || die 'YOLO/DINO paths cannot contain spaces in CLIENT_ARGS'
done

MODEL_PATH="$(cd "$MODEL_PATH" && pwd -P)"
[ -n "$YOLO_MODEL_PATH" ] && YOLO_MODEL_PATH="$(cd "$(dirname "$YOLO_MODEL_PATH")" && pwd -P)/$(basename "$YOLO_MODEL_PATH")"
[ -n "$DINO_MODEL_PATH" ] && DINO_MODEL_PATH="$(cd "$DINO_MODEL_PATH" && pwd -P)"
HABITAT_CONFIG="$(cd "$(dirname "$HABITAT_CONFIG")" && pwd -P)/$(basename "$HABITAT_CONFIG")"
INFERENCE_PYTHON=${CLIENT_PYTHON:-}
if [ -z "$INFERENCE_PYTHON" ]; then
    INFER_VENV=${INFER_VENV:-$REPO_ROOT/.venv}
    if [ -x "$INFER_VENV/bin/python" ]; then
        INFERENCE_PYTHON="$INFER_VENV/bin/python"
    else
        INFERENCE_PYTHON=python
    fi
fi

if [ "$mode" = new ]; then
    [ ! -e "$ablation_root" ] || die "output directory already exists: $ablation_root; use --resume"
    mkdir -p "$(dirname "$ablation_root")"
    mkdir "$ablation_root"
else
    [ -d "$ablation_root" ] || die "resume directory not found: $ablation_root"
fi
ablation_root="$(cd "$ablation_root" && pwd -P)"
export MODEL_PATH YOLO_MODEL_PATH DINO_MODEL_PATH HABITAT_CONFIG GPU_IDS SEED
export BACKEND GPU_MEM_UTIL YOLO_DEVICE DINO_DEVICE

pilot_manifest=""
pilot_manifest_sha256=""
if [ -n "$pilot_per_scene" ]; then
    pilot_manifest="$ablation_root/pilot_manifest.json"
    if [ "$mode" = new ]; then
        sample_args=(--data-path "$DATA_PATH" --output "$pilot_manifest" --per-scene 20 --seed 42)
        [ -n "${SCENES_DIR:-}" ] && sample_args+=(--scenes-dir "$SCENES_DIR")
        python3 "$REPO_ROOT/scripts/sample_r2r_pilot.py" "${sample_args[@]}"
    else
        [ -f "$pilot_manifest" ] || die "missing fixed Pilot manifest: $pilot_manifest"
    fi
    # A resumed Pilot must use the original sample and identical source bytes.
    pilot_manifest_sha256=$(python3 - "$pilot_manifest" "$DATA_PATH/val_unseen/val_unseen.json.gz" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
source = Path(sys.argv[2]).resolve()
data = json.loads(manifest.read_text(encoding="utf-8"))
digest = hashlib.sha256(source.read_bytes()).hexdigest()
if data.get("source_dataset") != str(source) or data.get("source_sha256") != digest:
    raise SystemExit("Pilot source dataset differs from fixed pilot_manifest.json")
if data.get("sample_seed") != 42 or data.get("expected_count") != 220:
    raise SystemExit("Pilot manifest has an incompatible sampling protocol")
print(hashlib.sha256(manifest.read_bytes()).hexdigest())
PY
    )
fi
export DATA_PATH
export PILOT_PER_SCENE="$pilot_per_scene" PILOT_MANIFEST_SHA256="$pilot_manifest_sha256"

# An immutable root record prevents a resumed group from silently using a
# different checkpoint, dataset/config, shard layout, seed, or inference setup.
python3 - "$ablation_root/ablation_run.json" "$mode" "${selected_groups[@]}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2]
config = Path(os.environ["HABITAT_CONFIG"])
snapshot = {
    "groups": sys.argv[3:],
    "launch_cwd": os.getcwd(),
    "model_path": os.environ["MODEL_PATH"],
    "yolo_model_path": os.environ["YOLO_MODEL_PATH"],
    "dino_model_path": os.environ["DINO_MODEL_PATH"],
    "habitat_config": str(config),
    "habitat_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "data_path": os.environ.get("DATA_PATH", ""),
    "scenes_dir": os.environ.get("SCENES_DIR", ""),
    "action_tokenizer_bundle": os.environ.get("ACTION_TOKENIZER_BUNDLE", ""),
    "backend": os.environ["BACKEND"],
    "gpu_memory_utilization": os.environ["GPU_MEM_UTIL"],
    "gpu_ids": os.environ["GPU_IDS"].split(),
    "base_port": os.environ.get("BASE_PORT", "5555"),
    "client_python": os.environ.get("CLIENT_PYTHON", ""),
    "habitat_python": os.environ.get("HABITAT_PYTHON", ""),
    "infer_venv": os.environ.get("INFER_VENV", ""),
    "habitat_conda_env": os.environ.get("HABITAT_CONDA_ENV", "habitat"),
    "seed": int(os.environ["SEED"]),
    "task": "vlnce",
    "split": "val_unseen",
    "episodes": -1,
    "max_steps": 500,
    "force_stop_at_max_steps": False,
    "yolo_device": os.environ["YOLO_DEVICE"],
    "dino_device": os.environ["DINO_DEVICE"],
    "yolo_every_n_steps": 5,
    "dino_every_n_steps": 5,
    "place_similarity_threshold": 0.85,
}
if os.environ["PILOT_PER_SCENE"]:
    snapshot["pilot_per_scene"] = int(os.environ["PILOT_PER_SCENE"])
    snapshot["pilot_manifest_sha256"] = os.environ["PILOT_MANIFEST_SHA256"]
if mode == "new":
    with path.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, indent=2, sort_keys=True)
        stream.write("\n")
else:
    if not path.is_file():
        raise SystemExit(f"missing resume snapshot: {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if "groups" not in saved and snapshot["groups"] == [
        "baseline_off", "flat_memory", "hierarchical_memory", "hierarchical_topology",
    ]:
        # A four-arm run started before --groups existed remains resumable.
        snapshot.pop("groups")
    if saved != snapshot:
        raise SystemExit("resume settings differ from ablation_run.json; refusing to mix runs")
PY

manifest="$ablation_root/episode_manifest.json"
common_args=(
    --no_force_stop --verbose
    --memory_top_k 5 --memory_max_context_chars 600
)
common_yolo=(
    --memory_enabled --memory_extractor yolo11
    --memory_extractor_model_path "$YOLO_MODEL_PATH"
    --memory_extractor_device "$YOLO_DEVICE"
    --memory_extractor_confidence 0.25
    --memory_extract_every_n_steps 5
)
hierarchical=(--memory_strategy hierarchical --memory_layers short,mid,long)

# eval_habitat.sh currently accepts one CLIENT_ARGS string and splits it into
# words. Paths above are checked for whitespace rather than silently mangled.
for group in "${selected_groups[@]}"; do
    group_dir="$ablation_root/$group"
    complete_marker=formal_complete.json
    [ -n "$pilot_per_scene" ] && complete_marker=pilot_complete.json
    if [ -f "$group_dir/$complete_marker" ]; then
        [ -f "$manifest" ] || die "shared manifest missing for completed $group"
        [ -f "$group_dir/episode_manifest.json" ] || die "group manifest missing for $group"
        cmp -s "$manifest" "$group_dir/episode_manifest.json" ||
            die "completed $group has a different episode manifest"
        verify_args=(verify "$group_dir" --shared "$manifest")
        [ -n "$pilot_manifest" ] && verify_args+=(--pilot-selection "$pilot_manifest")
        PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
            "$INFERENCE_PYTHON" -m lightnav.habitat.experiment_integrity \
                "${verify_args[@]}" ||
            die "completed $group failed the full episode coverage check"
        echo "$TAG $group already complete ($complete_marker); skipping"
        continue
    fi

    group_resume=0
    [ -d "$group_dir" ] && group_resume=1
    case "$group" in
        baseline_off)
            arm_args=()
            ;;
        flat_memory)
            arm_args=("${common_yolo[@]}" --memory_strategy flat)
            ;;
        hierarchical_memory)
            arm_args=("${common_yolo[@]}" "${hierarchical[@]}")
            ;;
        hierarchical_topology)
            arm_args=(
                "${common_yolo[@]}" "${hierarchical[@]}"
                --memory_place_enabled
                --memory_place_model_path "$DINO_MODEL_PATH"
                --memory_place_extract_every_n_steps 5
                --memory_place_similarity_threshold 0.85
                --memory_place_device "$DINO_DEVICE"
            )
            ;;
    esac
    all_args=("${common_args[@]}" "${arm_args[@]}")
    printf -v group_client_args '%s ' "${all_args[@]}"
    run_label="full val_unseen"
    [ -n "$pilot_per_scene" ] && run_label="Pilot val_unseen, 220 Episodes"
    echo "$TAG running $group ($run_label, 500 steps, seed=$SEED, resume=$group_resume)"
    mkdir -p "$group_dir"
    timing_file="$group_dir/run_timing.json"
    timing_attempt="${group}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    timing_started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    timing_start_ns="$(date +%s%N)"
    python3 - "$timing_file" "$group" "$timing_attempt" "$timing_started" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
group, attempt_id, started_at = sys.argv[2:]
data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
    "group": group, "attempts": [],
}
if data.get("group") != group or not isinstance(data.get("attempts"), list):
    raise SystemExit(f"invalid prior launcher timing file: {path}")
data["attempts"].append({
    "attempt_id": attempt_id, "started_at_utc": started_at,
    "ended_at_utc": None, "wall_time_sec": None, "exit_code": None,
})
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
    set +e
    MODEL_PATH="$MODEL_PATH" HABITAT_CONFIG="$HABITAT_CONFIG" TASK=vlnce \
        SPLIT=val_unseen EPISODES=-1 MAX_STEPS=500 NO_FORCE_STOP=1 \
        STRICT_FULL_EVAL=1 RESUME="$group_resume" \
        EXPECTED_EPISODES="$EXPECTED_EPISODES" PILOT_MANIFEST_PATH="$pilot_manifest" \
        EPISODE_MANIFEST_PATH="$manifest" OUTPUT_ROOT="$group_dir" \
        SEED="$SEED" GPU_IDS="$GPU_IDS" BACKEND="$BACKEND" \
        GPU_MEM_UTIL="$GPU_MEM_UTIL" CLIENT_ARGS="$group_client_args" \
        bash "$REPO_ROOT/scripts/eval_habitat.sh"
    eval_status=$?
    set -e
    timing_end_ns="$(date +%s%N)"
    timing_ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    python3 - "$timing_file" "$timing_attempt" "$timing_ended" \
        "$timing_start_ns" "$timing_end_ns" "$eval_status" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
attempt_id, ended_at, started_ns, ended_ns, exit_code = sys.argv[2:]
data = json.loads(path.read_text(encoding="utf-8"))
attempt = next(
    (item for item in data["attempts"] if item["attempt_id"] == attempt_id), None
)
if attempt is None:
    raise SystemExit(f"missing launcher timing attempt: {attempt_id}")
attempt["ended_at_utc"] = ended_at
attempt["wall_time_sec"] = max(0.0, (int(ended_ns) - int(started_ns)) / 1e9)
attempt["exit_code"] = int(exit_code)
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
    [ "$eval_status" -eq 0 ] || die "$group failed (exit=$eval_status); resume with --resume $ablation_root"
    [ -f "$group_dir/$complete_marker" ] ||
        die "$group finished without $complete_marker; preserve results and resume later"
    [ -f "$manifest" ] || die 'shared episode manifest was not created'
    cmp -s "$manifest" "$group_dir/episode_manifest.json" ||
        die "$group episode manifest differs from the shared manifest"
done

"$INFERENCE_PYTHON" "$REPO_ROOT/scripts/compare_memory_ablation.py" "$ablation_root"
echo "$TAG all ${#selected_groups[@]} selected groups complete: $ablation_root"
