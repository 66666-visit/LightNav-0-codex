#!/usr/bin/env bash
# Run the six L1/L3 prompt-organization conditions + legacy control on one fixed 88-Episode set.
#
# Only the prompt mode differs (legacy additionally enables measurement). The checkpoint, seed,
# Habitat config, episode manifest, memory backends, detector/segmentation
# weights, budgets and action/evaluation settings are shared and checked.
#
# Required environment:
#   MODEL_PATH              LightNav checkpoint directory
#   YOLO_MODEL_PATH         existing yolo11n.pt (L1 detector)
#   SEG_MODEL_PATH          existing yolo11n-seg.pt (independent L3 segmenter)
#   EPISODE_MANIFEST_PATH   existing 88-Episode evaluation manifest
#   PILOT_MANIFEST_PATH     existing 88-Episode server selection manifest
# Optional:
#   HABITAT_CONFIG, DATA_PATH, SCENES_DIR, GPU_IDS, GPU_MEM_UTIL, BASE_PORT,
#   CLIENT_PYTHON, HABITAT_PYTHON, HABITAT_CONDA_ENV, MANIFEST_SEARCH_ROOT
#
# This script never creates an episode manifest and never runs a condition when
# two different historical 88-Episode manifests are present.
set -euo pipefail

TAG='[l1_l3_prompt_ablation]'
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODES=(l1_original_only l1_organized_only l3_original_only l3_organized_only l1_l3_original_concat l1_l3_joint legacy)
ABLATION_ROOT="${ABLATION_ROOT:-$REPO_ROOT/output/l1_l3_prompt_ablation_$(date +%Y%m%d_%H%M%S)}"
PREFLIGHT_ONLY=0
DRY_RUN=0

die() { echo "$TAG ERROR: $*" >&2; exit 1; }
usage() {
    cat <<'EOF'
Usage:
  MODEL_PATH=... YOLO_MODEL_PATH=... SEG_MODEL_PATH=... \
  EPISODE_MANIFEST_PATH=... PILOT_MANIFEST_PATH=... \
    bash scripts/run_l1_l3_prompt_ablation.sh [--output-root DIR] \
      [--preflight-only] [--dry-run]

Always runs the six experimental conditions and the unchanged legacy control:
  l1_original_only l1_organized_only l3_original_only l3_organized_only
  l1_l3_original_concat l1_l3_joint legacy

--preflight-only validates the manifest/checkpoint/config, writes the run
snapshot and exits. --dry-run performs the same read-only checks and prints the
seven commands without writing anything or starting Habitat.
The launcher does not resume: keep a failed group for inspection, then either
fix the cause and use a new --output-root or reuse a completed group only when
its pilot_complete.json already exists.
EOF
}

while (($#)); do
    case "$1" in
        --output-root) [ "$#" -ge 2 ] || die '--output-root needs a directory'; ABLATION_ROOT=$2; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[ "$DRY_RUN" -eq 0 ] || [ "$PREFLIGHT_ONLY" -eq 0 ] || \
    die 'choose --dry-run or --preflight-only, not both'
[ "${TASK:-vlnce}" = vlnce ] || die 'TASK must be vlnce'
[ "${SPLIT:-val_unseen}" = val_unseen ] || die 'SPLIT must be val_unseen'
[ "${EPISODES:--1}" = -1 ] || die 'EPISODES is controlled by this protocol'
[ "${MAX_STEPS:-500}" = 500 ] || die 'MAX_STEPS must be 500'
[ "${NO_FORCE_STOP:-1}" = 1 ] || die 'NO_FORCE_STOP must be 1'
[ "${STRICT_FULL_EVAL:-1}" = 1 ] || die 'STRICT_FULL_EVAL must be 1'
[ "${EXPECTED_EPISODES:-88}" = 88 ] || die 'EXPECTED_EPISODES must be 88'
[ "${SEED:-42}" = 42 ] || die 'SEED is fixed at 42'
[ "${BACKEND:-vllm_local}" = vllm_local ] || die 'BACKEND is fixed at vllm_local'
[ -z "${CLIENT_ARGS:-}" ] || die 'CLIENT_ARGS is controlled by this protocol'
[ -z "${SERVER_ARGS:-}" ] || die 'SERVER_ARGS is controlled by this protocol'
[ -z "${LANGUAGES:-}" ] || die 'LANGUAGES would filter the shared Episode set'
[ -z "${SUCCESS_DISTANCE:-}" ] || die 'SUCCESS_DISTANCE must use the R2R config'
for knob in VLN_EVAL_TEMPERATURE VLN_EVAL_TOP_P VLN_EVAL_TOP_K VLN_EVAL_TRAJ_TOP1; do
    [ -z "${!knob:-}" ] || die "$knob changes the frozen inference protocol"
done

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the LightNav checkpoint directory}
YOLO_MODEL_PATH=${YOLO_MODEL_PATH:?set YOLO_MODEL_PATH to a local yolo11n.pt}
SEG_MODEL_PATH=${SEG_MODEL_PATH:?set SEG_MODEL_PATH to a local yolo11n-seg.pt}
EPISODE_MANIFEST_PATH=${EPISODE_MANIFEST_PATH:?set EPISODE_MANIFEST_PATH to the existing 88-Episode manifest}
PILOT_MANIFEST_PATH=${PILOT_MANIFEST_PATH:?set PILOT_MANIFEST_PATH to the existing 88-Episode server selection}
[ -d "$MODEL_PATH" ] || die "checkpoint directory not found: $MODEL_PATH"
[ -s "$YOLO_MODEL_PATH" ] || die "L1 YOLO weights not found: $YOLO_MODEL_PATH"
[ -s "$SEG_MODEL_PATH" ] || die "L3 segmentation weights not found: $SEG_MODEL_PATH"
[ -f "$EPISODE_MANIFEST_PATH" ] || die "episode manifest not found: $EPISODE_MANIFEST_PATH"
[ -f "$PILOT_MANIFEST_PATH" ] || die "pilot manifest not found: $PILOT_MANIFEST_PATH"
[ "$(sha256sum "$YOLO_MODEL_PATH" | cut -d' ' -f1)" != "$(sha256sum "$SEG_MODEL_PATH" | cut -d' ' -f1)" ] || \
    die 'L1 and L3 must use different model files'
HABITAT_CONFIG=${HABITAT_CONFIG:-$REPO_ROOT/habitat_server/configs/vlnce_r2r.yaml}
[ -f "$HABITAT_CONFIG" ] || die "Habitat config not found: $HABITAT_CONFIG"
GPU_IDS=${GPU_IDS:-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.65}
BASE_PORT=${BASE_PORT:-5555}
MANIFEST_SEARCH_ROOT=${MANIFEST_SEARCH_ROOT:-$REPO_ROOT/output}
INFERENCE_PYTHON=${CLIENT_PYTHON:-}
if [ -z "$INFERENCE_PYTHON" ]; then
    INFER_VENV=${INFER_VENV:-$REPO_ROOT/.venv311}
    if [ -x "$INFER_VENV/bin/python" ]; then
        INFERENCE_PYTHON="$INFER_VENV/bin/python"
    else
        INFERENCE_PYTHON=python3
    fi
fi
for path in "$MODEL_PATH" "$YOLO_MODEL_PATH" "$SEG_MODEL_PATH" "$EPISODE_MANIFEST_PATH" \
            "$PILOT_MANIFEST_PATH" "$HABITAT_CONFIG"; do
    [[ "$path" != *[[:space:]]* ]] || die "path contains whitespace unsupported by CLIENT_ARGS: $path"
done

MODEL_PATH="$(cd "$MODEL_PATH" && pwd -P)"
YOLO_MODEL_PATH="$(cd "$(dirname "$YOLO_MODEL_PATH")" && pwd -P)/$(basename "$YOLO_MODEL_PATH")"
SEG_MODEL_PATH="$(cd "$(dirname "$SEG_MODEL_PATH")" && pwd -P)/$(basename "$SEG_MODEL_PATH")"
EPISODE_MANIFEST_PATH="$(cd "$(dirname "$EPISODE_MANIFEST_PATH")" && pwd -P)/$(basename "$EPISODE_MANIFEST_PATH")"
PILOT_MANIFEST_PATH="$(cd "$(dirname "$PILOT_MANIFEST_PATH")" && pwd -P)/$(basename "$PILOT_MANIFEST_PATH")"
HABITAT_CONFIG="$(cd "$(dirname "$HABITAT_CONFIG")" && pwd -P)/$(basename "$HABITAT_CONFIG")"
[ -d "$MANIFEST_SEARCH_ROOT" ] || MANIFEST_SEARCH_ROOT="$(dirname "$EPISODE_MANIFEST_PATH")"

if [ "$DRY_RUN" -eq 0 ] && [ "$PREFLIGHT_ONLY" -eq 0 ]; then
    [ ! -e "$ABLATION_ROOT" ] || die "output directory already exists: $ABLATION_ROOT; keep it or choose a new --output-root"
    mkdir -p "$ABLATION_ROOT"
fi
if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    [ ! -e "$ABLATION_ROOT" ] || die "preflight output directory already exists: $ABLATION_ROOT"
    mkdir -p "$ABLATION_ROOT"
fi
if [ "$DRY_RUN" -eq 0 ] || [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    ABLATION_ROOT="$(cd "$ABLATION_ROOT" && pwd -P)"
fi

export MODEL_PATH YOLO_MODEL_PATH SEG_MODEL_PATH EPISODE_MANIFEST_PATH \
       PILOT_MANIFEST_PATH HABITAT_CONFIG ABLATION_ROOT MANIFEST_SEARCH_ROOT
export CLIENT_PYTHON="$INFERENCE_PYTHON"
export SEED=42 BACKEND=vllm_local EXPECTED_EPISODES=88
export GPU_IDS GPU_MEM_UTIL BASE_PORT
export DRY_RUN PREFLIGHT_ONLY
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Preflight: validate 88 unique Episodes, checkpoint identity and every
# historical 88-Episode manifest; reject two disagreeing histories.
"$INFERENCE_PYTHON" - <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from lightnav.habitat.experiment_integrity import (
    check_pilot_selection,
    validate_manifest,
)

root = Path(os.environ["ABLATION_ROOT"])
manifest_path = Path(os.environ["EPISODE_MANIFEST_PATH"])
pilot_path = Path(os.environ["PILOT_MANIFEST_PATH"])
model_path = Path(os.environ["MODEL_PATH"])
habitat_config = Path(os.environ["HABITAT_CONFIG"])
dry_run = os.environ["DRY_RUN"] == "1"
preflight_only = os.environ["PREFLIGHT_ONLY"] == "1"


def die(message):
    print(f"[l1_l3_prompt_ablation] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def canonical_identity(manifest):
    return [
        _normalise(row["scene_id"]) + "\x00" + str(row["episode_id"])
        for row in manifest["episodes"]
    ]


def _normalise(value):
    return str(Path(str(value)).resolve(strict=False))


def manifest_digest(manifest):
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def tree_digest(path, suffixes):
    files = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix in suffixes
    )
    if not files:
        return None
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


try:
    manifest = validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    check_pilot_selection(manifest, pilot_path)
except (OSError, ValueError, KeyError) as exc:
    die(f"episode/pilot manifest rejected: {exc}")
if manifest["split"] != "val_unseen":
    die("the shared manifest must use R2R val_unseen")
if manifest["expected_count"] != 88 or len(manifest["episodes"]) != 88:
    die(f"the shared manifest must contain exactly 88 Episodes; got {manifest['expected_count']}")
identity = canonical_identity(manifest)
if len(set(identity)) != 88:
    die("the shared manifest does not contain 88 unique Episodes")
manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
canonical_sha256 = manifest_digest(manifest)

# Find every historical evaluation/pilot manifest with 88 Episodes. Two
# different histories make the comparison undefined, so stop instead of
# choosing one or creating a new manifest.
search_root = Path(os.environ["MANIFEST_SEARCH_ROOT"])
history_manifests = {}
if search_root.is_dir():
    candidates = list(search_root.glob("*/episode_manifest.json"))
    candidates += list(search_root.glob("*/*/episode_manifest.json"))
    candidates += list(search_root.glob("*/*/*/episode_manifest.json"))
    for candidate in candidates:
        try:
            other = validate_manifest(json.loads(candidate.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            continue
        if other["expected_count"] != 88:
            continue
        history_manifests[str(candidate.resolve())] = (
            manifest_digest(other), canonical_identity(other),
        )
history_manifests[str(manifest_path.resolve())] = (canonical_sha256, identity)
distinct = {}
for path, (digest, other_identity) in history_manifests.items():
    distinct.setdefault(tuple(other_identity), []).append(path)
if len(distinct) > 1:
    groups = []
    for paths in distinct.values():
        groups.append(", ".join(sorted(paths)))
    die(
        "multiple inconsistent 88-Episode manifests exist; refusing to choose one:\n  "
        + "\n  ".join(groups)
    )
pilot_history = {}
if search_root.is_dir():
    candidates = list(search_root.glob("*/pilot_manifest.json"))
    candidates += list(search_root.glob("*/*/pilot_manifest.json"))
    candidates += list(search_root.glob("*/*/*/pilot_manifest.json"))
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("expected_count") != 88:
            continue
        rows = data.get("episodes")
        if not isinstance(rows, list) or len(rows) != 88:
            continue
        keys = tuple(sorted(
            _normalise(row.get("scene_id")) + "\x00" + str(row.get("habitat_episode_id"))
            for row in rows if isinstance(row, dict)
        ))
        pilot_history.setdefault(keys, []).append(str(candidate.resolve()))
pilot_history.setdefault(tuple(sorted(
    _normalise(row["scene_id"]) + "\x00" + str(row["episode_id"])
    for row in manifest["episodes"]
)), []).append(str(pilot_path.resolve()))
if len(pilot_history) > 1:
    groups = ["; ".join(sorted(paths)) for paths in pilot_history.values()]
    die(
        "multiple inconsistent 88-Episode pilot manifests exist; refusing to choose one:\n  "
        + "\n  ".join(groups)
    )

weights_sha256 = tree_digest(model_path, {".safetensors", ".bin"})
if not weights_sha256:
    die(f"checkpoint has no .safetensors/.bin weights: {model_path}")

# Task-2 preflight: look for an already complete legacy 88-Episode result. A
# candidate is only reported; the launcher conservatively runs the seventh
# legacy group unless a human confirms an exact match. No candidate is created
# or modified.
legacy_candidates = []
if search_root.is_dir():
    candidate_dirs = {path.parent for path in search_root.glob("*/config_snapshot.yaml")}
    candidate_dirs |= {path.parent for path in search_root.glob("*/*/config_snapshot.yaml")}
    candidate_dirs |= {path.parent for path in search_root.glob("*/*/*/config_snapshot.yaml")}
    for candidate in sorted(candidate_dirs):
        reasons = []
        required = [
            candidate / "results.jsonl", candidate / "summary.json",
            candidate / "per_episode.csv", candidate / "episode_manifest.json",
            candidate / "config_snapshot.yaml",
        ]
        if any(not item.is_file() for item in required):
            reasons.append("missing required result artifact")
        results = []
        try:
            results = [
                json.loads(line) for line in
                (candidate / "results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError) as exc:
            reasons.append(f"unreadable results.jsonl: {exc}")
        if len(results) != 88:
            reasons.append(f"result rows={len(results)}, expected 88")
        try:
            candidate_manifest = validate_manifest(
                json.loads((candidate / "episode_manifest.json").read_text(encoding="utf-8"))
            )
            if canonical_identity(candidate_manifest) != identity:
                reasons.append("episode manifest differs from the shared manifest")
        except (OSError, ValueError, KeyError) as exc:
            reasons.append(f"manifest check failed: {exc}")
        config = None
        try:
            config = json.loads(
                (candidate / "config_snapshot.yaml").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            reasons.append(f"config snapshot unreadable: {exc}")
        if isinstance(config, dict):
            common = config.get("common", {})
            if common.get("model_weights_sha256") != weights_sha256:
                reasons.append("checkpoint weight SHA256 differs")
            if common.get("eval_config_sha256") != (
                provenance_eval_sha := hashlib.sha256(
                    (model_path / "eval_config.json").read_bytes()
                ).hexdigest() if (model_path / "eval_config.json").is_file() else None
            ):
                reasons.append("checkpoint eval_config.json SHA256 differs")
            if common.get("habitat_config_sha256") != hashlib.sha256(
                habitat_config.read_bytes()
            ).hexdigest():
                reasons.append("Habitat config SHA256 differs")
            if common.get("seed") != 42 or common.get("task") != "vlnce" or (
                common.get("split") != "val_unseen"
            ):
                reasons.append("seed/task/split differs")
            if common.get("max_steps") != 500 or common.get("force_stop_at_max_steps") is not False:
                reasons.append("step cap/forced-stop differs")
            if common.get("expected_episodes") != 88:
                reasons.append("expected episode count differs")
            effective = config.get("prompt_effective", {})
            if not isinstance(effective, dict) or (
                effective.get("prompt_mode") != "legacy"
                or effective.get("budget_strategy") != "original_char_budget"
            ):
                reasons.append("not a legacy prompt-measurement result")
            memory_args = config.get("memory_args", [])
            for flag, value in (
                ("--memory_enabled", None), ("--local_graph_enabled", None),
                ("--spatial_l3_mode", "inject"),
            ):
                if flag not in memory_args:
                    reasons.append(f"missing {flag}")
                elif value is not None and memory_args.index(flag) + 1 < len(memory_args) and (
                    memory_args[memory_args.index(flag) + 1] != value
                ):
                    reasons.append(f"{flag} differs")
        try:
            summary = json.loads((candidate / "summary.json").read_text(encoding="utf-8"))
            if summary.get("num_episodes") != 88:
                reasons.append("summary episode count differs")
        except (OSError, ValueError) as exc:
            reasons.append(f"summary unreadable: {exc}")
        try:
            per_episode_rows = sum(
                1 for line in (candidate / "per_episode.csv").read_text(
                    encoding="utf-8",
                ).splitlines() if line.strip()
            ) - 1
            if per_episode_rows != 88:
                reasons.append(f"per_episode.csv rows={per_episode_rows}, expected 88")
        except OSError as exc:
            reasons.append(f"per_episode.csv unreadable: {exc}")
        legacy_candidates.append({
            "path": str(candidate.resolve()),
            "status": "compatible_candidate" if not reasons else "incompatible",
            "reasons": reasons,
        })

provenance = {
    "protocol_version": "l1_l3_prompt_ablation_v1",
    "modes": [
        "l1_original_only", "l1_organized_only", "l3_original_only",
        "l3_organized_only", "l1_l3_original_concat", "l1_l3_joint",
    ],
    "episode_manifest_path": str(manifest_path),
    "episode_manifest_sha256": manifest_sha256,
    "episode_manifest_canonical_sha256": canonical_sha256,
    "episode_manifest_expected_count": manifest["expected_count"],
    "pilot_manifest_path": str(pilot_path),
    "pilot_manifest_sha256": hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
    "historical_88_manifests_checked": sorted(history_manifests),
    "checkpoint_path": str(model_path),
    "checkpoint_sha256": weights_sha256,
    "checkpoint_config_sha256": hashlib.sha256(
        (model_path / "config.json").read_bytes()
    ).hexdigest() if (model_path / "config.json").is_file() else None,
    "checkpoint_eval_config_sha256": hashlib.sha256(
        (model_path / "eval_config.json").read_bytes()
    ).hexdigest() if (model_path / "eval_config.json").is_file() else None,
    "l1_yolo_model_sha256": hashlib.sha256(Path(os.environ["YOLO_MODEL_PATH"]).read_bytes()).hexdigest(),
    "l3_seg_model_sha256": hashlib.sha256(Path(os.environ["SEG_MODEL_PATH"]).read_bytes()).hexdigest(),
    "habitat_config_path": str(habitat_config),
    "habitat_config_sha256": hashlib.sha256(habitat_config.read_bytes()).hexdigest(),
    "seed": int(os.environ["SEED"]),
    "max_steps": 500,
    "gpu_ids": os.environ["GPU_IDS"].split(),
    "gpu_memory_utilization": os.environ["GPU_MEM_UTIL"],
    "base_port": int(os.environ["BASE_PORT"]),
    "legacy_history_candidates": legacy_candidates,
    "legacy_reuse": "not_applied_by_launcher",
    "legacy_group_required": True,
    "legacy_reuse_note": (
        "The launcher runs the seventh legacy group. A historical candidate is "
        "reported only; exact reuse requires manual confirmation so a mismatched "
        "control can never enter the paired comparison."
        if legacy_candidates else
        "No historical 88-Episode result was found; the seventh legacy group must run."
    ),
}
print(json.dumps(provenance, indent=2, ensure_ascii=False))

if dry_run:
    raise SystemExit(0)

target = root / "provenance.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(provenance, stream, indent=2, ensure_ascii=False)
    stream.write("\n")
(root / "manifest.sha256").write_text(
    manifest_sha256 + "  " + str(manifest_path) + "\n", encoding="utf-8",
)
(root / "checkpoint.sha256").write_text(
    weights_sha256 + "  " + str(model_path) + "\n", encoding="utf-8",
)
(root / "episode_manifest.json").write_bytes(manifest_path.read_bytes())
PY

if [ "$DRY_RUN" -eq 1 ]; then
    echo "$TAG dry-run: read-only preflight passed; no files were written"
    for mode in "${MODES[@]}"; do
        echo "$TAG   $mode -> $ABLATION_ROOT/$mode"
        echo "$TAG     CLIENT_ARGS=<shared> --memory_prompt_mode $mode"
    done
    exit 0
fi
if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    echo "$TAG preflight complete: $ABLATION_ROOT/provenance.json"
    exit 0
fi

# The seven arms share every setting except --memory_prompt_mode (and the
# legacy measurement switch).
common_args=(
    --no_force_stop --verbose
    --memory_enabled --memory_extractor yolo11
    --memory_extractor_model_path "$YOLO_MODEL_PATH"
    --memory_extractor_device "${YOLO_DEVICE:-cpu}"
    --memory_extractor_confidence 0.25 --memory_extract_every_n_steps 5
    --memory_strategy hierarchical --memory_layers short,mid,long
    --memory_top_k 5 --memory_max_context_chars 600
    --local_graph_enabled --local_graph_scope episode_local
    --local_graph_prompt_max_chars 320 --local_graph_max_landmarks 2
    --local_graph_reactivation_steps 8
    --spatial_l3_mode inject
    --spatial_l3_seg_model_path "$SEG_MODEL_PATH"
    --spatial_l3_seg_device "${SEG_DEVICE:-cpu}"
    --spatial_l3_seg_confidence 0.25 --spatial_l3_seg_every_n_steps 5
    --spatial_l3_min_injection_confidence 0.5
    --prompt_measurement_enabled
)
printf -v shared_client_args '%s ' "${common_args[@]}"
echo "$TAG preflight passed; starting six experimental conditions + legacy"

for mode in "${MODES[@]}"; do
    group_dir="$ABLATION_ROOT/$mode"
    if [ -f "$group_dir/pilot_complete.json" ]; then
        echo "$TAG $mode already complete; verifying artifacts"
        PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
            "$INFERENCE_PYTHON" -m lightnav.habitat.prompt_artifacts verify \
                --group "$group_dir" --mode "$mode"
        continue
    fi
    [ ! -e "$group_dir" ] || die "$mode output exists without pilot_complete.json; inspect it before rerunning"
    echo "$TAG running $mode"
    started=$SECONDS
    MODEL_PATH="$MODEL_PATH" HABITAT_CONFIG="$HABITAT_CONFIG" TASK=vlnce \
        SPLIT=val_unseen EPISODES=-1 MAX_STEPS=500 NO_FORCE_STOP=1 \
        STRICT_FULL_EVAL=1 RESUME=0 EXPECTED_EPISODES=88 \
        PILOT_MANIFEST_PATH="$PILOT_MANIFEST_PATH" \
        EPISODE_MANIFEST_PATH="$ABLATION_ROOT/episode_manifest.json" \
        OUTPUT_ROOT="$group_dir" SEED=42 GPU_IDS="$GPU_IDS" BACKEND=vllm_local \
        GPU_MEM_UTIL="$GPU_MEM_UTIL" BASE_PORT="$BASE_PORT" \
        CLIENT_ARGS="$shared_client_args--memory_prompt_mode $mode" \
        bash "$REPO_ROOT/scripts/eval_habitat.sh"
    [ -f "$group_dir/pilot_complete.json" ] || die "$mode has no pilot_complete.json"
    # The comparison verifies the Episode identities after normalizing scene
    # paths, so a different textual spelling is not a false failure.
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$INFERENCE_PYTHON" -m lightnav.habitat.prompt_artifacts verify \
            --group "$group_dir" --mode "$mode"
    "$INFERENCE_PYTHON" "$REPO_ROOT/scripts/compare_l1_l3_prompt_ablation.py" \
        --verify-only --root "$ABLATION_ROOT" >/dev/null
    python3 - "$group_dir/run_timing.json" "$mode" "$((SECONDS - started))" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
prior = json.loads(path.read_text()) if path.exists() else {"group": sys.argv[2], "attempts": []}
prior["attempts"].append({"exit_code": 0, "wall_time_sec": float(sys.argv[3])})
path.write_text(json.dumps(prior, indent=2) + "\n")
PY
done

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$INFERENCE_PYTHON" "$REPO_ROOT/scripts/compare_l1_l3_prompt_ablation.py" \
        --root "$ABLATION_ROOT"
echo "$TAG complete: $ABLATION_ROOT/final_report.md"
