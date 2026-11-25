#!/usr/bin/env bash
# LightNav-0 instance-disambiguation arm: FOUR diagnostic Episodes only.
#
# Episodes are aligned by (scene basename, exact Habitat episode ID, rollout_idx)
# against an existing evaluator/Pilot manifest.  IDs are compared as strings:
# "episode_001" is never interpreted as Habitat ID 1.  No resampling, no ordinal
# fallback.  The four selected Episodes are run one at a time through the
# existing Habitat server single-Episode selection; no 88-Episode run starts.
#
# ABLATION_GROUP selects one of the three pre-registered arms:
#   D = eligible selection, landmark queries OFF, geometry fallback OFF
#   E = eligible selection, landmark queries ON,  geometry fallback OFF
#   F = eligible selection, landmark queries ON,  geometry fallback ON
# F is the default and reproduces the launcher's previous flags exactly, so an
# invocation without ABLATION_GROUP keeps the old experiment's behaviour.
#
# Every arm writes under OUTPUT_ROOT/group_<G>, so D, E and F never overwrite
# each other.  resolved_config.json records the effective group, the three
# grounder switches and the complete client argument string.  Top-level
# resolved_config.json / four_episode_selection.json are compatibility copies
# of the arm's files; the files inside group_<G> are authoritative.
#
# --self-test (or SELF_TEST=1) runs the CPU-only D/E/F dry-run matrix.  It uses
# temporary dummy assets, never starts a server/GPU/Habitat process, and checks
# the resolved flags, exact Episode keys and group-specific output roots.
# DRY_RUN=1 performs CPU-only validation and prints the resolved commands; it
# does not start a server, client or GPU process.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

parse_ablation_group() {
    ABLATION_GROUP=${ABLATION_GROUP:-F}
    ABLATION_GROUP=${ABLATION_GROUP^^}
    case "$ABLATION_GROUP" in
        D)
            GROUNDER_SELECTION=eligible
            GROUNDER_LANDMARK_QUERIES=0
            GROUNDER_GEOMETRY_FALLBACK=0
            ;;
        E)
            GROUNDER_SELECTION=eligible
            GROUNDER_LANDMARK_QUERIES=1
            GROUNDER_GEOMETRY_FALLBACK=0
            ;;
        F)
            GROUNDER_SELECTION=eligible
            GROUNDER_LANDMARK_QUERIES=1
            GROUNDER_GEOMETRY_FALLBACK=1
            ;;
        *)
            echo "[instance_disambiguation] ERROR: ABLATION_GROUP must be one of D, E, F (got '$ABLATION_GROUP')" >&2
            exit 2
            ;;
    esac
}

find_self_test_python() {
    local candidate
    for candidate in "${CLIENT_PYTHON:-}" python3 "$REPO_ROOT/.venv/bin/python"; do
        [ -n "$candidate" ] || continue
        if CUDA_VISIBLE_DEVICES= \
            PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
            "$candidate" -c "import lightnav.cli.eval_habitat" >/dev/null 2>&1; then
            command -v "$candidate" 2>/dev/null || printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

run_self_test() {
    local client_python
    if ! client_python="$(find_self_test_python)"; then
        echo "[instance_disambiguation] SELF_TEST ERROR: no Python interpreter can import lightnav.cli.eval_habitat; set CLIENT_PYTHON" >&2
        return 1
    fi

    SELFTEST_TMP="$(mktemp -d "${TMPDIR:-/tmp}/lightnav_instance_disambiguation_selftest.XXXXXX")"
    trap 'rm -rf "$SELFTEST_TMP"' EXIT
    local tmp="$SELFTEST_TMP"

    mkdir -p "$tmp/data" "$tmp/scenes"
    : > "$tmp/model"
    : > "$tmp/yolo.pt"
    : > "$tmp/seg.pt"
    : > "$tmp/grounder.pt"
    : > "$tmp/vocab.txt"
    printf 'model: {}\n' > "$tmp/grounder.yaml"

    python3 - "$tmp/manifest.json" <<'PY'
import json
import sys

episode_ids = ["487", "168", "1080", "1012"]
rows = [
    {
        "scene_id": (
            f"data/scene_datasets/mp3d/scene{episode_id}/"
            f"scene{episode_id}.glb"
        ),
        "habitat_episode_id": episode_id,
        "rollout_idx": 0,
    }
    for episode_id in episode_ids
]
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({"episodes": rows}, handle, indent=2)
PY

    local group out_base group_root log landmark_label fallback_label
    for group in D E F; do
        out_base="$tmp/out_$group"
        log="$tmp/group_$group.log"
        if ! (
            export MODEL_PATH="$tmp/model"
            export YOLO_MODEL_PATH="$tmp/yolo.pt"
            export SEG_MODEL_PATH="$tmp/seg.pt"
            export GROUNDER_CHECKPOINT="$tmp/grounder.pt"
            export GROUNDER_CONFIG="$tmp/grounder.yaml"
            export GROUNDER_BERT_VOCAB="$tmp/vocab.txt"
            export DATA_PATH="$tmp/data"
            export SCENES_DIR="$tmp/scenes"
            export EPISODE_MANIFEST_PATH="$tmp/manifest.json"
            export EPISODE_SELECTION=""
            export EPISODE_IDS="487 168 1080 1012"
            export OUTPUT_ROOT="$out_base"
            export CLIENT_PYTHON="$client_python"
            export DRY_RUN=1
            export SELF_TEST=0
            export CUDA_VISIBLE_DEVICES=
            export ABLATION_GROUP="$group"
            unset SEED MAX_STEPS BACKEND TASK SPLIT GPU_IDS GPU_MEM_UTIL
            bash "$0"
        ) >"$log" 2>&1; then
            cat "$log" >&2
            echo "[instance_disambiguation] SELF_TEST FAIL: group $group dry-run exited non-zero" >&2
            return 1
        fi

        group_root="$out_base/group_$group"
        if [ ! -f "$group_root/resolved_config.json" ]; then
            echo "[instance_disambiguation] SELF_TEST FAIL: group $group did not write resolved_config.json to $group_root" >&2
            return 1
        fi
        if ! grep -q "DRY_RUN: no server or GPU started" "$log"; then
            echo "[instance_disambiguation] SELF_TEST FAIL: group $group did not take the DRY_RUN path" >&2
            return 1
        fi
        if ! "$client_python" - "$group_root/resolved_config.json" "$group" \
            "$group_root" "$out_base" <<'PY'
import json
import shlex
import sys
from pathlib import Path

resolved_path, group, group_root, base_root = sys.argv[1:5]
resolved = json.loads(Path(resolved_path).read_text(encoding="utf-8"))
landmark, fallback = {
    "D": (False, False),
    "E": (True, False),
    "F": (True, True),
}[group]
episode_ids = ["487", "168", "1080", "1012"]
expected_keys = [
    (f"scene{episode_id}.glb", episode_id, 0)
    for episode_id in episode_ids
]

assert resolved["dry_run"] is True
assert resolved["ablation_group"] == group
assert resolved["ablation"]["id"] == group
assert resolved["ablation"]["grounder_selection_mode"] == "eligible"
assert resolved["ablation"]["grounder_landmark_queries"] is landmark
assert resolved["ablation"]["grounder_geometry_fallback"] is fallback
assert resolved["flags"] == {
    "spatial_l3_grounder_selection": "eligible",
    "spatial_l3_grounder_landmark_queries": landmark,
    "spatial_l3_grounder_geometry_fallback": fallback,
}
assert resolved["episode_key"] == [
    "scene_basename", "habitat_episode_id", "rollout_idx",
]
assert resolved["episode_count"] == 4
keys = [
    (row["scene_basename"], row["habitat_episode_id"], row["rollout_idx"])
    for row in resolved["episodes"]
]
assert keys == expected_keys, keys
assert len(set(keys)) == 4
assert [
    row["habitat_episode_id"] for row in resolved["episodes"]
] == episode_ids
assert all(row["rollout_idx"] == 0 for row in resolved["episodes"])
assert resolved["frozen"]["seed"] == 42
assert resolved["frozen"]["max_steps"] == 500
assert resolved["output_root"] == group_root
assert resolved["base_output_root"] == base_root
assert Path(group_root).name == f"group_{group}"

tokens = shlex.split(resolved["effective_client_args"])
assert "--spatial_l3_grounder_selection" in tokens
assert tokens[tokens.index("--spatial_l3_grounder_selection") + 1] == "eligible"
for flag, enabled in (
    ("--spatial_l3_grounder_landmark_queries", landmark),
    ("--spatial_l3_grounder_geometry_fallback", fallback),
):
    assert (flag in tokens) is enabled, (group, flag, tokens)
    assert (f"--no-{flag[2:]}" in tokens) is (not enabled), (group, flag, tokens)

assert not any(
    path.is_dir() and path.name.startswith("episode_")
    for path in Path(group_root).iterdir()
)
base_resolved = Path(base_root) / "resolved_config.json"
assert base_resolved.is_file()
assert json.loads(base_resolved.read_text(encoding="utf-8")) == resolved
base_selection = Path(base_root) / "four_episode_selection.json"
assert base_selection.is_file()
PY
        then
            echo "[instance_disambiguation] SELF_TEST FAIL: group $group resolved config/flags mismatch" >&2
            return 1
        fi

        case "$group" in
            D) landmark_label=off; fallback_label=off ;;
            E) landmark_label=on; fallback_label=off ;;
            F) landmark_label=on; fallback_label=on ;;
        esac
        echo "[instance_disambiguation] SELF_TEST PASS $group: selection=eligible landmark_queries=$landmark_label geometry_fallback=$fallback_label output=$group_root"
    done

    local invalid_log="$tmp/invalid_group.log"
    if (
        export SELF_TEST=0
        export DRY_RUN=1
        export ABLATION_GROUP=C
        bash "$0"
    ) >"$invalid_log" 2>&1; then
        echo "[instance_disambiguation] SELF_TEST FAIL: ABLATION_GROUP=C should be rejected" >&2
        return 1
    fi
    if ! grep -q "ABLATION_GROUP must be one of D, E, F" "$invalid_log"; then
        cat "$invalid_log" >&2
        echo "[instance_disambiguation] SELF_TEST FAIL: invalid-group diagnostic missing" >&2
        return 1
    fi

    python3 - "$tmp/duplicate_manifest.json" <<'PY'
import json
import sys

rows = [
    {
        "scene_id": "data/scene_datasets/mp3d/sceneA/sceneA.glb",
        "habitat_episode_id": "487",
        "rollout_idx": 0,
    },
    {
        "scene_id": "data/scene_datasets/mp3d/sceneB/sceneB.glb",
        "habitat_episode_id": "487",
        "rollout_idx": 0,
    },
    {
        "scene_id": "data/scene_datasets/mp3d/scene168/scene168.glb",
        "habitat_episode_id": "168",
        "rollout_idx": 0,
    },
    {
        "scene_id": "data/scene_datasets/mp3d/scene1080/scene1080.glb",
        "habitat_episode_id": "1080",
        "rollout_idx": 0,
    },
    {
        "scene_id": "data/scene_datasets/mp3d/scene1012/scene1012.glb",
        "habitat_episode_id": "1012",
        "rollout_idx": 0,
    },
]
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({"episodes": rows}, handle, indent=2)
PY

    local duplicate_log="$tmp/duplicate_triple.log"
    if (
        export MODEL_PATH="$tmp/model"
        export YOLO_MODEL_PATH="$tmp/yolo.pt"
        export SEG_MODEL_PATH="$tmp/seg.pt"
        export GROUNDER_CHECKPOINT="$tmp/grounder.pt"
        export GROUNDER_CONFIG="$tmp/grounder.yaml"
        export GROUNDER_BERT_VOCAB="$tmp/vocab.txt"
        export DATA_PATH="$tmp/data"
        export SCENES_DIR="$tmp/scenes"
        export EPISODE_MANIFEST_PATH="$tmp/duplicate_manifest.json"
        export EPISODE_SELECTION=""
        export EPISODE_IDS="487 168 1080 1012"
        export OUTPUT_ROOT="$tmp/out_duplicate"
        export CLIENT_PYTHON="$client_python"
        export DRY_RUN=1
        export SELF_TEST=0
        export ABLATION_GROUP=F
        bash "$0"
    ) >"$duplicate_log" 2>&1; then
        echo "[instance_disambiguation] SELF_TEST FAIL: ambiguous Habitat episode ID was accepted" >&2
        return 1
    fi
    if ! grep -q "matches=2" "$duplicate_log"; then
        cat "$duplicate_log" >&2
        echo "[instance_disambiguation] SELF_TEST FAIL: ambiguous episode ID diagnostic missing" >&2
        return 1
    fi

    echo "[instance_disambiguation] SELF_TEST PASS: D/E/F CPU dry-runs (no GPU, no Habitat)."
    echo "[instance_disambiguation] SELF_TEST PASS: invalid group rejected, ambiguous episode ID rejected."
    rm -rf "$SELFTEST_TMP"
    trap - EXIT
}

if [ "${1:-}" = "--self-test" ] || [ "${SELF_TEST:-0}" = "1" ]; then
    run_self_test
    exit 0
fi

parse_ablation_group

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
OUTPUT_BASE=${OUTPUT_ROOT:-/root/autodl-tmp/output/instance_disambiguation_4_$(date +%Y%m%d_%H%M%S)}
while [[ "$OUTPUT_BASE" == */ && "$OUTPUT_BASE" != / ]]; do
    OUTPUT_BASE="${OUTPUT_BASE%/}"
done
GROUP_OUTPUT_ROOT="$OUTPUT_BASE/group_${ABLATION_GROUP}"

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
if [ -e "$OUTPUT_BASE" ] && [ ! -d "$OUTPUT_BASE" ]; then
    echo "[instance_disambiguation] ERROR: output base is not a directory: $OUTPUT_BASE" >&2
    exit 1
fi
if [ -L "$GROUP_OUTPUT_ROOT" ] \
    || { [ -e "$GROUP_OUTPUT_ROOT" ] && [ ! -d "$GROUP_OUTPUT_ROOT" ]; }; then
    echo "[instance_disambiguation] ERROR: group output must be a directory: $GROUP_OUTPUT_ROOT" >&2
    exit 1
fi
if [ -d "$GROUP_OUTPUT_ROOT" ] \
    && [ -n "$(find "$GROUP_OUTPUT_ROOT" -mindepth 1 -print -quit)" ]; then
    echo "[instance_disambiguation] ERROR: group output must be new or empty: $GROUP_OUTPUT_ROOT" >&2
    exit 1
fi
mkdir -p "$OUTPUT_BASE" "$GROUP_OUTPUT_ROOT"

SELECTION_PATH="$GROUP_OUTPUT_ROOT/four_episode_selection.json"
RESOLVED_PATH="$GROUP_OUTPUT_ROOT/resolved_config.json"
BASE_SELECTION_PATH="$OUTPUT_BASE/four_episode_selection.json"
BASE_RESOLVED_PATH="$OUTPUT_BASE/resolved_config.json"

GROUNDER_FLAG_ARGS="--spatial_l3_grounder_selection $GROUNDER_SELECTION"
if [ "$GROUNDER_LANDMARK_QUERIES" = "1" ]; then
    GROUNDER_FLAG_ARGS+=" --spatial_l3_grounder_landmark_queries"
else
    GROUNDER_FLAG_ARGS+=" --no-spatial_l3_grounder_landmark_queries"
fi
if [ "$GROUNDER_GEOMETRY_FALLBACK" = "1" ]; then
    GROUNDER_FLAG_ARGS+=" --spatial_l3_grounder_geometry_fallback"
else
    GROUNDER_FLAG_ARGS+=" --no-spatial_l3_grounder_geometry_fallback"
fi

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
  $GROUNDER_FLAG_ARGS \
  --prompt_measurement_enabled"

python3 - "$EPISODE_SELECTION" "$EPISODE_MANIFEST_PATH" "$EPISODE_IDS" \
    "$SELECTION_PATH" "$RESOLVED_PATH" "$MODEL_PATH" "$YOLO_MODEL_PATH" \
    "$SEG_MODEL_PATH" "$GROUNDER_CHECKPOINT" "$GROUNDER_CONFIG" \
    "$GROUNDER_BERT_VOCAB" "$DATA_PATH" "$SCENES_DIR" "$CLIENT_PYTHON" \
    "$GROUP_OUTPUT_ROOT" "$DRY_RUN" "$ABLATION_GROUP" \
    "$GROUNDER_LANDMARK_QUERIES" "$GROUNDER_GEOMETRY_FALLBACK" \
    "$BASE_CLIENT_ARGS" "$OUTPUT_BASE" "$BASE_SELECTION_PATH" \
    "$BASE_RESOLVED_PATH" "$GROUNDER_SELECTION" <<'PY'
import json
import os
import sys
from pathlib import Path

if len(sys.argv) != 25:
    raise SystemExit(
        "internal launcher error: expected 24 config arguments, "
        f"got {len(sys.argv) - 1}"
    )

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
    ablation_group,
    landmark_queries_raw,
    geometry_fallback_raw,
    client_args,
    base_output_root,
    base_selection_path,
    base_resolved_path,
    selection_mode,
) = sys.argv[1:25]

landmark_queries = str(landmark_queries_raw) == "1"
geometry_fallback = str(geometry_fallback_raw) == "1"
expected_group_flags = {
    "D": (False, False),
    "E": (True, False),
    "F": (True, True),
}
if ablation_group not in expected_group_flags:
    raise SystemExit(
        f"unsupported ABLATION_GROUP in resolved config: {ablation_group!r}"
    )
if (landmark_queries, geometry_fallback) != expected_group_flags[ablation_group]:
    raise SystemExit(
        "ABLATION_GROUP and effective grounder flags disagree: "
        f"group={ablation_group!r} "
        f"landmark_queries={landmark_queries} "
        f"geometry_fallback={geometry_fallback}"
    )
if selection_mode != "eligible":
    raise SystemExit(
        "D/E/F all require --spatial_l3_grounder_selection eligible, "
        f"got {selection_mode!r}"
    )
semantics_by_group = {
    "D": "joint eligible selection",
    "E": "joint eligible selection + multi-landmark queries",
    "F": (
        "joint eligible selection + multi-landmark queries + "
        "same-category geometry fallback"
    ),
}


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
    scene_basename = _scene_basename(scene_id)
    if not scene_basename:
        return None
    return scene_basename, str(episode_id), rollout


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

selection_text = (
    json.dumps({"episodes": selected}, indent=2, ensure_ascii=False) + "\n"
)
resolved = {
    "dry_run": str(dry_run) == "1",
    "ablation_group": ablation_group,
    "episode_key": ["scene_basename", "habitat_episode_id", "rollout_idx"],
    "episode_count": len(selected),
    "selection_source": selection_source,
    "episodes": selected,
    "ablation": {
        "id": ablation_group,
        "semantics": semantics_by_group[ablation_group],
        "grounder_selection_mode": selection_mode,
        "grounder_landmark_queries": landmark_queries,
        "grounder_geometry_fallback": geometry_fallback,
    },
    "flags": {
        "spatial_l3_grounder_selection": selection_mode,
        "spatial_l3_grounder_landmark_queries": landmark_queries,
        "spatial_l3_grounder_geometry_fallback": geometry_fallback,
    },
    "effective_client_args": client_args,
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
    "base_output_root": base_output_root,
    "resolved_config_path": out_resolved,
    "selection_path": out_selection,
    "base_resolved_config_path": base_resolved_path,
    "base_selection_path": base_selection_path,
}
resolved_text = json.dumps(resolved, indent=2, ensure_ascii=False) + "\n"
Path(out_selection).write_text(selection_text, encoding="utf-8")
Path(out_resolved).write_text(resolved_text, encoding="utf-8")
# Compatibility copies: the previous launcher wrote these two files directly at
# OUTPUT_ROOT.  The group_<G> copies remain authoritative.
Path(base_selection_path).write_text(selection_text, encoding="utf-8")
Path(base_resolved_path).write_text(resolved_text, encoding="utf-8")
print(resolved_text, end="")
PY

# CPU-only preflight: the inference venv can import the client module, and the
# launcher shell parses.  This does not create a CUDA context or start a server.
bash -n scripts/eval_habitat.sh
if [ "$DRY_RUN" = "1" ]; then
    # Hiding CUDA devices keeps DRY_RUN unambiguously CPU-only.
    CUDA_VISIBLE_DEVICES= \
        PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$CLIENT_PYTHON" -c "import lightnav.cli.eval_habitat" >/dev/null
else
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$CLIENT_PYTHON" -c "import lightnav.cli.eval_habitat" >/dev/null
fi

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

case "$ABLATION_GROUP" in
    D) ABLATION_LABEL="eligible selection, landmark queries OFF, geometry fallback OFF" ;;
    E) ABLATION_LABEL="eligible selection, landmark queries ON, geometry fallback OFF" ;;
    F) ABLATION_LABEL="eligible selection, landmark queries ON, geometry fallback ON" ;;
esac

if [ "$DRY_RUN" = "1" ]; then
    echo "[instance_disambiguation] DRY_RUN: no server or GPU started."
    echo "[instance_disambiguation] ablation group: $ABLATION_GROUP ($ABLATION_LABEL)"
    echo "[instance_disambiguation] grounder flags: $GROUNDER_FLAG_ARGS"
    echo "[instance_disambiguation] resolved config: $RESOLVED_PATH"
    echo "[instance_disambiguation] selection: $SELECTION_PATH"
    echo "[instance_disambiguation] base output: $OUTPUT_BASE"
    echo "[instance_disambiguation] group output: $GROUP_OUTPUT_ROOT"
    while IFS=$'\t' read -r scene_id episode_id rollout_idx; do
        echo "[instance_disambiguation] episode $episode_id rollout=$rollout_idx scene=$scene_id"
        echo "  SERVER_ARGS='--single-scene-id $scene_id --single-episode-id $episode_id' \\"
        echo "  EPISODES=1 OUTPUT_ROOT='$GROUP_OUTPUT_ROOT/episode_${episode_id}_r${rollout_idx}' \\"
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
    episode_out="$GROUP_OUTPUT_ROOT/episode_${episode_id}_r${rollout_idx}"
    echo "[instance_disambiguation] Episode $episode_id rollout=$rollout_idx ($scene_id)"
    echo "[instance_disambiguation] ablation group: $ABLATION_GROUP ($ABLATION_LABEL)"
    echo "[instance_disambiguation] output: $episode_out"
    SERVER_ARGS="--single-scene-id $scene_id --single-episode-id $episode_id" \
    EPISODES=1 \
    OUTPUT_ROOT="$episode_out" \
    CLIENT_ARGS="$BASE_CLIENT_ARGS" \
        bash scripts/eval_habitat.sh
done

echo "[instance_disambiguation] four-Episode run complete: $GROUP_OUTPUT_ROOT"
