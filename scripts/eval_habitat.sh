#!/usr/bin/env bash
# Parallel Habitat evaluation on the local GPUs, one shard per GPU:
#   1. start one Habitat env server per GPU (habitat conda env), each serving a
#      disjoint shard of the split (--split-id i --split-num N)
#   2. wait until every server has written its .ready file
#   3. start one eval client per GPU (the lightnav env; model on the same GPU)
#   4. merge the shards' results.jsonl into OUTPUT_ROOT/summary.json
# Servers and clients are killed on exit (trap).
#
# Knobs (env vars):
#   # ---- model ----
#   MODEL_PATH           checkpoint dir (required)
#   BACKEND              vllm_local | hf (default vllm_local)
#   GPU_MEM_UTIL         vLLM GPU memory fraction for the client (default 0.65 — the
#                        simulator renders on the same GPU)
#   ACTION_TOKENIZER_BUNDLE  RVQ bundle override (default: resolved from eval_config.json)
#   CLIENT_ARGS          extra lightnav-eval-habitat flags, e.g. "--save_video --verbose"
#   # ---- benchmark ----
#   TASK                 vlnce | objectnav (default vlnce)
#   HABITAT_CONFIG       yaml (default habitat_server/configs/vlnce_r2r.yaml)
#   SPLIT                dataset split (default: vlnce -> val_unseen; objectnav -> yaml)
#   SUCCESS_DISTANCE     e.g. 0.25 for HM3D-OVON (default: env default)
#   NAVMESH_CELL_HEIGHT  re-bake navmeshes at this cell_height; REQUIRED as 0.05 for
#                        HM3D ObjectNav v2, leave unset otherwise (objectnav only)
#   DATA_PATH, SCENES_DIR  optional dataset path overrides (see docs/HABITAT_SERVER.md)
#   LANGUAGES            RxR only, e.g. "en-US en-IN"
#   EPISODES             per shard; -1 = whole shard (default -1)
#   MAX_STEPS            default 500
#   SERVER_ARGS          extra lightnav_habitat.serve flags
#   STRICT_FULL_EVAL     1 = require complete val_unseen, standard 500 steps, exact IDs
#   NO_FORCE_STOP        1 = let Habitat truncate at the standard step limit
#   RESUME               1 = skip already completed episode IDs in the same output
#   SEED                 shared Habitat/client random seed for a controlled run
#   EPISODE_MANIFEST_PATH shared manifest path to create/compare across experiment arms
#   EXPECTED_EPISODES    standard R2R val_unseen count (default 1839 in formal mode)
#   PILOT_MANIFEST_PATH  fixed 220-Episode legacy or 88-Episode RGM selection
#   Client progress is streamed to the terminal and saved in client shard logs.
#   # ---- topology ----
#   GPU_IDS              GPUs to use, e.g. "0 1 3" (default: all visible GPUs)
#   NUM_GPUS             alternative to GPU_IDS: use GPUs 0..NUM_GPUS-1
#   BASE_PORT            first ZMQ port (default 5555)
#   READY_TIMEOUT_S      max wait for the servers (default 900)
#   # ---- environments ----
#   HABITAT_CONDA_ENV    conda env with habitat-sim + lightnav_habitat (default habitat), or
#   HABITAT_PYTHON       explicit interpreter for the env server (overrides HABITAT_CONDA_ENV)
#   INFER_VENV           lightnav virtualenv (default <repo>/.venv), or
#   CLIENT_PYTHON        explicit interpreter for the eval client
#   # ---- output ----
#   OUTPUT_ROOT          default output/habitat_<task>_<timestamp>; shards in shard_<i>/
#   LOG_DIR              default $OUTPUT_ROOT/logs
#
# Examples:
#   MODEL_PATH=/path/to/checkpoint bash scripts/eval_habitat.sh                 # R2R on every GPU
#   MODEL_PATH=... HABITAT_CONFIG=habitat_server/configs/vlnce_rxr.yaml LANGUAGES="en-US en-IN" \
#       GPU_IDS="0 1" bash scripts/eval_habitat.sh                              # RxR on 2 GPUs
#   MODEL_PATH=... TASK=objectnav HABITAT_CONFIG=habitat_server/configs/objectnav_ovon.yaml \
#       SPLIT=val_unseen SUCCESS_DISTANCE=0.25 bash scripts/eval_habitat.sh      # HM3D-OVON
#   MODEL_PATH=... TASK=objectnav HABITAT_CONFIG=habitat_server/configs/objectnav_mp3d.yaml \
#       SPLIT=val bash scripts/eval_habitat.sh                                    # MP3D v1
set -euo pipefail

TAG="[eval_habitat]"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── model ────────────────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the checkpoint dir}
[ -d "$MODEL_PATH" ] || { echo "$TAG ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 1; }
BACKEND=${BACKEND:-vllm_local}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.65}
ACTION_TOKENIZER_BUNDLE=${ACTION_TOKENIZER_BUNDLE:-}
CLIENT_ARGS=${CLIENT_ARGS:-}

# ── benchmark ────────────────────────────────────────────────────────────────
TASK=${TASK:-vlnce}
case "$TASK" in vlnce|objectnav) ;; *) echo "$TAG ERROR: TASK must be vlnce or objectnav" >&2; exit 1;; esac
HABITAT_CONFIG=${HABITAT_CONFIG:-$REPO_ROOT/habitat_server/configs/vlnce_r2r.yaml}
[ -f "$HABITAT_CONFIG" ] || { echo "$TAG ERROR: HABITAT_CONFIG not found: $HABITAT_CONFIG" >&2; exit 1; }
HABITAT_CONFIG="$(cd "$(dirname "$HABITAT_CONFIG")" && pwd)/$(basename "$HABITAT_CONFIG")"
if [ -z "${SPLIT+x}" ]; then
    [ "$TASK" = "vlnce" ] && SPLIT=val_unseen || SPLIT=""
fi
SUCCESS_DISTANCE=${SUCCESS_DISTANCE:-}
NAVMESH_CELL_HEIGHT=${NAVMESH_CELL_HEIGHT:-}
DATA_PATH=${DATA_PATH:-}
SCENES_DIR=${SCENES_DIR:-}
LANGUAGES=${LANGUAGES:-}
EPISODES=${EPISODES:--1}
MAX_STEPS=${MAX_STEPS:-500}
SERVER_ARGS=${SERVER_ARGS:-}
STRICT_FULL_EVAL=${STRICT_FULL_EVAL:-0}
NO_FORCE_STOP=${NO_FORCE_STOP:-0}
RESUME=${RESUME:-0}
SEED=${SEED:-}
EPISODE_MANIFEST_PATH=${EPISODE_MANIFEST_PATH:-}
EXPECTED_EPISODES=${EXPECTED_EPISODES:-1839}
PILOT_MANIFEST_PATH=${PILOT_MANIFEST_PATH:-}
if [ -n "$PILOT_MANIFEST_PATH" ]; then
    [ "$STRICT_FULL_EVAL" = 1 ] || {
        echo "$TAG ERROR: Pilot selection requires STRICT_FULL_EVAL=1" >&2; exit 1;
    }
    [ -f "$PILOT_MANIFEST_PATH" ] || {
        echo "$TAG ERROR: Pilot manifest not found: $PILOT_MANIFEST_PATH" >&2; exit 1;
    }
    PILOT_EXPECTED_EPISODES="$(python3 - "$PILOT_MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(88 if manifest.get("protocol_version") == "rgm_pilot_88_v1" else 220)
PY
)"
    [ "$EXPECTED_EPISODES" = "$PILOT_EXPECTED_EPISODES" ] || {
        echo "$TAG ERROR: Pilot selection requires EXPECTED_EPISODES=$PILOT_EXPECTED_EPISODES" >&2
        exit 1
    }
    PILOT_MANIFEST_PATH="$(cd "$(dirname "$PILOT_MANIFEST_PATH")" && pwd -P)/$(basename "$PILOT_MANIFEST_PATH")"
fi
if [ "$STRICT_FULL_EVAL" = 1 ]; then
    if [ -z "$PILOT_MANIFEST_PATH" ] && [ "$EXPECTED_EPISODES" != 1839 ]; then
        echo "$TAG ERROR: formal R2R val_unseen requires EXPECTED_EPISODES=1839" >&2
        exit 1
    fi
    if [ "$TASK" != vlnce ] || [ "$SPLIT" != val_unseen ] || [ "$EPISODES" != -1 ] \
        || [ "$MAX_STEPS" != 500 ] || [ "$NO_FORCE_STOP" != 1 ] \
        || [ -z "$SEED" ] || [ -n "$LANGUAGES" ] || [ -n "$SERVER_ARGS" ]; then
        echo "$TAG ERROR: formal R2R requires vlnce val_unseen, EPISODES=-1, MAX_STEPS=500, NO_FORCE_STOP=1, SEED, no languages/server overrides" >&2
        exit 1
    fi
fi

# ── topology ─────────────────────────────────────────────────────────────────
if [ -n "${GPU_IDS:-}" ]; then
    read -ra GPUS <<< "$GPU_IDS"
elif [ -n "${NUM_GPUS:-}" ]; then
    GPUS=(); for ((i = 0; i < NUM_GPUS; i++)); do GPUS+=("$i"); done
else
    DETECTED=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
    [ "${DETECTED:-0}" -ge 1 ] || { echo "$TAG ERROR: no GPU detected; set GPU_IDS or NUM_GPUS" >&2; exit 1; }
    GPUS=(); for ((i = 0; i < DETECTED; i++)); do GPUS+=("$i"); done
fi
N=${#GPUS[@]}
BASE_PORT=${BASE_PORT:-5555}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-900}

# ── interpreters ─────────────────────────────────────────────────────────────
# Habitat side: `conda activate` only prepends PATH, so use the absolute
# $CONDA_PREFIX/bin/python afterwards and put the env's own libs first for habitat-sim.
HABITAT_CONDA_ENV=${HABITAT_CONDA_ENV:-habitat}
HABITAT_PYTHON=${HABITAT_PYTHON:-}
if [ -z "$HABITAT_PYTHON" ]; then
    CONDA_SH="${CONDA_SH:-}"
    if [ -z "$CONDA_SH" ]; then
        for candidate in \
            "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh}" \
            "$HOME/miniconda3/etc/profile.d/conda.sh" \
            "$HOME/anaconda3/etc/profile.d/conda.sh" \
            "$HOME/miniforge3/etc/profile.d/conda.sh" \
            "/opt/conda/etc/profile.d/conda.sh"; do
            if [ -n "$candidate" ] && [ -f "$candidate" ]; then CONDA_SH="$candidate"; break; fi
        done
    fi
    [ -n "$CONDA_SH" ] || {
        echo "$TAG ERROR: cannot locate conda.sh; set CONDA_SH=/path/to/etc/profile.d/conda.sh or HABITAT_PYTHON=/path/to/python" >&2
        exit 1; }
    set +u   # conda's activate scripts reference unset variables
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$HABITAT_CONDA_ENV" || {
        echo "$TAG ERROR: conda activate $HABITAT_CONDA_ENV failed; available envs:" >&2
        conda env list >&2
        exit 1; }
    set -u
    HABITAT_PYTHON="$CONDA_PREFIX/bin/python"
    HABITAT_LD="$CONDA_PREFIX/lib"
    conda deactivate || true
else
    HABITAT_LD=""
fi
[ -x "$HABITAT_PYTHON" ] || { echo "$TAG ERROR: HABITAT_PYTHON is not executable: $HABITAT_PYTHON" >&2; exit 1; }
PYTHONPATH="$REPO_ROOT/habitat_server${PYTHONPATH:+:$PYTHONPATH}" "$HABITAT_PYTHON" \
    -c "import habitat, habitat_sim, lightnav_habitat.serve" 2>/dev/null || {
    echo "$TAG ERROR: $HABITAT_PYTHON cannot import habitat / habitat_sim / lightnav_habitat (see docs/HABITAT_SERVER.md)" >&2; exit 1; }

# Model side: the lightnav virtualenv.
CLIENT_PYTHON=${CLIENT_PYTHON:-}
INFER_VENV="${INFER_VENV:-$REPO_ROOT/.venv}"
CLIENT_LD=""
if [ -z "$CLIENT_PYTHON" ]; then
    if [ -x "$INFER_VENV/bin/python" ]; then
        CLIENT_PYTHON="$INFER_VENV/bin/python"
        # torch wheels need their own bundled nvJitLink first on LD_LIBRARY_PATH.
        CLIENT_LD=$(echo "$INFER_VENV"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr ' ' ':')
    else
        echo "$TAG WARN: $INFER_VENV not found; using PATH python for the client."
        CLIENT_PYTHON="python"
    fi
fi
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$CLIENT_PYTHON" -c "import lightnav.cli.eval_habitat, zmq" 2>/dev/null || {
    echo "$TAG ERROR: $CLIENT_PYTHON cannot import lightnav (+ pyzmq): pip install -e '.[habitat]'" >&2; exit 1; }

# ── output ───────────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
ATTEMPT_ID="${TS}_$$"
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/output/habitat_${TASK}_${TS}}
# Use the same shell word splitting as the client invocation below. Checking a
# space-delimited substring misses flags separated by tabs or newlines.
# shellcheck disable=SC2206
client_extra_args=($CLIENT_ARGS)
v3_client_enabled=0
v4_client_enabled=0
local_graph_client_enabled=0
spatial_local_graph_client_enabled=0
rgm_client_enabled=0
memory_client_enabled=0
for token in "${client_extra_args[@]}"; do
    [ "$token" = --memory_enabled ] && memory_client_enabled=1
    [ "$token" = --rgm_enabled ] && rgm_client_enabled=1
    [ "$token" = --rgm_v3_enabled ] && v3_client_enabled=1
    [ "$token" = --rgm_v4_enabled ] && v4_client_enabled=1
    [ "$token" = --local_graph_enabled ] && local_graph_client_enabled=1
    [ "$token" = --local_graph_fusion_mode=spatial ] && spatial_local_graph_client_enabled=1
done
for ((arg_index = 0; arg_index + 1 < ${#client_extra_args[@]}; arg_index++)); do
    if [ "${client_extra_args[$arg_index]}" = --local_graph_fusion_mode ] \
        && [ "${client_extra_args[$((arg_index + 1))]}" = spatial ]; then
        spatial_local_graph_client_enabled=1
    fi
done
if [ "$local_graph_client_enabled" -eq 1 ]; then
    while [[ "$OUTPUT_ROOT" == */ && "$OUTPUT_ROOT" != / ]]; do
        OUTPUT_ROOT="${OUTPUT_ROOT%/}"
    done
    if [ "$memory_client_enabled" -ne 1 ]; then
        echo "$TAG ERROR: Local Graph requires --memory_enabled" >&2
        exit 1
    fi
    if [ "$rgm_client_enabled" -eq 1 ] || [ "$v3_client_enabled" -eq 1 ] \
        || [ "$v4_client_enabled" -eq 1 ]; then
        echo "$TAG ERROR: Local Graph cannot share a run with RGM, V3, or V4" >&2
        exit 1
    fi
    if [ "$RESUME" = 1 ]; then
        echo "$TAG ERROR: Local Graph requires a fresh output and cannot resume" >&2
        exit 1
    fi
    if [ "$spatial_local_graph_client_enabled" -eq 1 ]; then
        case "$OUTPUT_ROOT" in
            */local_graph_fusion_88_run2|*/local_graph_fusion_88_run2/*)
                echo "$TAG ERROR: Spatial Local Graph cannot use the legacy 88-Episode result path" >&2
                exit 1
                ;;
        esac
    fi
    case "$OUTPUT_ROOT" in
        */rgm_pilot_88_v1|*/rgm_pilot_88_v1/*|*/rgm_pilot_88_v2|*/rgm_pilot_88_v2/*|*/rgm_v3_pilot_88|*/rgm_v3_pilot_88/*|*/rgm_v4_pilot_88|*/rgm_v4_pilot_88/*)
            echo "$TAG ERROR: Local Graph requires an independent output directory" >&2
            exit 1
            ;;
    esac
    ancestor_dir="$(dirname "$OUTPUT_ROOT")"
    while [ "$ancestor_dir" != . ] && [ "$ancestor_dir" != / ]; do
        if [ -f "$ancestor_dir/config_snapshot.yaml" ] \
            || [ -f "$ancestor_dir/summary.json" ] \
            || [ -f "$ancestor_dir/pilot_complete.json" ] \
            || [ -f "$ancestor_dir/results.jsonl" ] \
            || [ -f "$ancestor_dir/rgm_v3_protocol.json" ] \
            || [ -f "$ancestor_dir/rgm_v4_protocol.json" ] \
            || [ -f "$ancestor_dir/local_graph_protocol.json" ]; then
            echo "$TAG ERROR: Local Graph output cannot be nested inside existing results" >&2
            exit 1
        fi
        ancestor_dir="$(dirname "$ancestor_dir")"
    done
    if [ -L "$OUTPUT_ROOT" ] || { [ -e "$OUTPUT_ROOT" ] && [ ! -d "$OUTPUT_ROOT" ]; }; then
        echo "$TAG ERROR: Local Graph output root must be a new directory" >&2
        exit 1
    fi
    if [ -d "$OUTPUT_ROOT" ] && [ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit)" ]; then
        echo "$TAG ERROR: Local Graph output root must be empty; existing results are preserved" >&2
        exit 1
    fi
fi
if [ "$v3_client_enabled" -eq 1 ] && [ "$v4_client_enabled" -eq 1 ]; then
    echo "$TAG ERROR: RGM V3 and V4 cannot share one run" >&2
    exit 1
fi
if [ "$v3_client_enabled" -eq 1 ]; then
    case "$OUTPUT_ROOT" in
        */rgm_pilot_88_v1|*/rgm_pilot_88_v1/*|*/rgm_pilot_88_v2|*/rgm_pilot_88_v2/*|*/rgm_v4_pilot_88|*/rgm_v4_pilot_88/*)
            echo "$TAG ERROR: RGM V3 requires an independent output directory" >&2
            exit 1
            ;;
    esac
    if [ -f "$OUTPUT_ROOT/rgm_v4_protocol.json" ]; then
        echo "$TAG ERROR: RGM V3 cannot use a V4 result directory" >&2
        exit 1
    fi
    if [ -n "$PILOT_MANIFEST_PATH" ]; then
        echo "$TAG ERROR: RGM V3 cannot use a legacy Pilot Manifest" >&2
        exit 1
    fi
fi
if [ "$v4_client_enabled" -eq 1 ]; then
    case "$OUTPUT_ROOT" in
        */rgm_pilot_88_v1|*/rgm_pilot_88_v1/*|*/rgm_pilot_88_v2|*/rgm_pilot_88_v2/*|*/rgm_v3_pilot_88|*/rgm_v3_pilot_88/*)
            echo "$TAG ERROR: RGM V4 requires an independent output directory" >&2
            exit 1
            ;;
    esac
    if [ -f "$OUTPUT_ROOT/rgm_v3_protocol.json" ]; then
        echo "$TAG ERROR: RGM V4 cannot use a V3 result directory" >&2
        exit 1
    fi
    if [ -n "$PILOT_MANIFEST_PATH" ]; then
        echo "$TAG ERROR: RGM V4 cannot use a legacy Pilot Manifest" >&2
        exit 1
    fi
fi
if [ "$v3_client_enabled" -eq 0 ] && [ "$v4_client_enabled" -eq 0 ]; then
    case "$OUTPUT_ROOT" in
        */rgm_v3_pilot_88|*/rgm_v3_pilot_88/*|*/rgm_v4_pilot_88|*/rgm_v4_pilot_88/*)
            echo "$TAG ERROR: reserved V3/V4 result directory requires its backend" >&2
            exit 1
            ;;
    esac
    if [ -f "$OUTPUT_ROOT/rgm_v3_protocol.json" ] || [ -f "$OUTPUT_ROOT/rgm_v4_protocol.json" ]; then
        echo "$TAG ERROR: V3/V4 result directory cannot be reused by another backend" >&2
        exit 1
    fi
fi
LOG_DIR=${LOG_DIR:-$OUTPUT_ROOT/logs/$ATTEMPT_ID}
READY_DIR="$OUTPUT_ROOT/.ready/$ATTEMPT_ID"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$READY_DIR"

echo "$TAG task=$TASK config=$HABITAT_CONFIG split=${SPLIT:-<yaml>} model=$MODEL_PATH backend=$BACKEND"
echo "$TAG gpus=(${GPUS[*]}) shards=$N base_port=$BASE_PORT output=$OUTPUT_ROOT"

# A JSON document is valid YAML. Keep the exact run settings in each group's
# output and refuse a resumed run whose checkpoint or benchmark settings differ.
SNAPSHOT_OUTPUT_ROOT="$OUTPUT_ROOT" SNAPSHOT_MODEL_PATH="$MODEL_PATH" \
SNAPSHOT_BACKEND="$BACKEND" SNAPSHOT_TASK="$TASK" SNAPSHOT_SPLIT="$SPLIT" \
SNAPSHOT_HABITAT_CONFIG="$HABITAT_CONFIG" SNAPSHOT_SEED="$SEED" \
SNAPSHOT_EPISODES="$EPISODES" SNAPSHOT_MAX_STEPS="$MAX_STEPS" \
SNAPSHOT_NO_FORCE_STOP="$NO_FORCE_STOP" SNAPSHOT_GPU_IDS="${GPUS[*]}" \
SNAPSHOT_STRICT_FULL_EVAL="$STRICT_FULL_EVAL" \
SNAPSHOT_ACTION_BUNDLE="$ACTION_TOKENIZER_BUNDLE" \
SNAPSHOT_GPU_MEM_UTIL="$GPU_MEM_UTIL" \
SNAPSHOT_SERVER_ARGS="$SERVER_ARGS" SNAPSHOT_EXPECTED_EPISODES="$EXPECTED_EPISODES" \
SNAPSHOT_DATA_PATH="$DATA_PATH" SNAPSHOT_SCENES_DIR="$SCENES_DIR" \
SNAPSHOT_PILOT_MANIFEST="$PILOT_MANIFEST_PATH" \
SNAPSHOT_REPO_ROOT="$REPO_ROOT" \
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
"$CLIENT_PYTHON" - "${client_extra_args[@]}" <<'PY'
import hashlib
import json
import os
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

from lightnav.cli.eval_habitat import build_parser, config_from_args
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.v3_config import RGMV3Config
from lightnav.memory.v4_config import RGMV4Config

env = os.environ
habitat_config = Path(env["SNAPSHOT_HABITAT_CONFIG"]).resolve()
model_path = Path(env["SNAPSHOT_MODEL_PATH"]).resolve()
tokens = sys.argv[1:]
common_tokens, memory_tokens = [], []
index = 0
while index < len(tokens):
    token = tokens[index]
    group = memory_tokens if token.startswith((
        "--memory_", "--rgm_", "--no-rgm_", "--local_graph_",
    )) else common_tokens
    group.append(token)
    if token.startswith("--") and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
        index += 1
        group.append(tokens[index])
    index += 1

def digest_if_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

def digest_tree(root, suffixes=None):
    if not root.is_dir():
        return None
    files = sorted(path for path in root.rglob("*") if path.is_file()
                   and (suffixes is None or path.suffix in suffixes))
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()

repo = Path(env["SNAPSHOT_REPO_ROOT"])
code_digest = hashlib.sha256()
for directory in (repo / "src" / "lightnav", repo / "habitat_server" / "lightnav_habitat"):
    for path in sorted(directory.rglob("*.py")):
        code_digest.update(str(path.relative_to(repo)).encode())
        code_digest.update(path.read_bytes())
bundle = Path(env["SNAPSHOT_ACTION_BUNDLE"]) if env["SNAPSHOT_ACTION_BUNDLE"] else model_path / "action_tokenizer"
rgm_values = asdict(RGMConfig())
for index, token in enumerate(tokens):
    if token.startswith(("--rgm_v3_", "--rgm_v4_", "--no-rgm_v4_")):
        continue
    negative = token.startswith("--no-rgm_")
    if not (token.startswith("--rgm_") or negative):
        continue
    name = token.removeprefix("--no-rgm_" if negative else "--rgm_")
    if name not in rgm_values:
        raise SystemExit(f"unknown RGM setting in snapshot: {token}")
    default = rgm_values[name]
    if isinstance(default, bool):
        rgm_values[name] = not negative
    elif negative or index + 1 >= len(tokens):
        raise SystemExit(f"missing value for RGM setting: {token}")
    else:
        rgm_values[name] = type(default)(tokens[index + 1])
rgm_effective = asdict(RGMConfig(**rgm_values))
v3_values = asdict(RGMV3Config())
for index, token in enumerate(tokens):
    if not token.startswith("--rgm_v3_"):
        continue
    name = token.removeprefix("--rgm_v3_")
    if name not in v3_values:
        raise SystemExit(f"unknown RGM V3 setting in snapshot: {token}")
    default = v3_values[name]
    if isinstance(default, bool):
        v3_values[name] = True
    elif index + 1 >= len(tokens):
        raise SystemExit(f"missing value for RGM V3 setting: {token}")
    else:
        v3_values[name] = type(default)(tokens[index + 1])
v3_effective = asdict(RGMV3Config(**v3_values))
v4_values = asdict(RGMV4Config())
for index, token in enumerate(tokens):
    negative = token.startswith("--no-rgm_v4_")
    if not (token.startswith("--rgm_v4_") or negative):
        continue
    name = token.removeprefix("--no-rgm_v4_" if negative else "--rgm_v4_")
    if name not in v4_values:
        raise SystemExit(f"unknown RGM V4 setting in snapshot: {token}")
    default = v4_values[name]
    if isinstance(default, bool):
        v4_values[name] = not negative
    elif negative or index + 1 >= len(tokens):
        raise SystemExit(f"missing value for RGM V4 setting: {token}")
    else:
        v4_values[name] = type(default)(tokens[index + 1])
v4_effective = asdict(RGMV4Config(**v4_values))
if sum((rgm_effective["enabled"], v3_effective["enabled"], v4_effective["enabled"])) > 1:
    raise SystemExit("legacy RGM, RGM V3 and RGM V4 cannot share one run")
local_graph_effective = None
if "--local_graph_enabled" in tokens:
    client_argv = [
        "--model_path", env["SNAPSHOT_MODEL_PATH"],
        "--backend", env["SNAPSHOT_BACKEND"],
        "--episodes", env["SNAPSHOT_EPISODES"],
        "--max_steps", env["SNAPSHOT_MAX_STEPS"],
        "--gpu_memory_utilization", env["SNAPSHOT_GPU_MEM_UTIL"],
    ]
    if env["SNAPSHOT_SEED"]:
        client_argv.extend(("--seed", env["SNAPSHOT_SEED"]))
    if env["SNAPSHOT_NO_FORCE_STOP"] == "1":
        client_argv.append("--no_force_stop")
    if env["SNAPSHOT_ACTION_BUNDLE"]:
        client_argv.extend(("--action_tokenizer_bundle", env["SNAPSHOT_ACTION_BUNDLE"]))
    if env["SNAPSHOT_STRICT_FULL_EVAL"] == "1":
        client_argv.append("--strict_full_eval")
    client_argv.extend(tokens)
    client_argv.extend(("--output_dir", str(Path(env["SNAPSHOT_OUTPUT_ROOT"]) / "shard_0")))
    try:
        parsed = build_parser().parse_args(client_argv)
        effective = config_from_args(parsed)
    except ValueError as exc:
        raise SystemExit(f"invalid Local Graph client configuration: {exc}") from exc
    local_config = effective.memory.local_graph
    if not isinstance(local_config, LocalGraphConfig) or not local_config.enabled:
        raise SystemExit("Local Graph flag did not select LocalGraphConfig")
    if not effective.memory.enabled:
        raise SystemExit("Local Graph requires memory_enabled")
    if (effective.memory.strategy != "hierarchical"
            or set(effective.memory.layers) != {"short", "mid", "long"}):
        raise SystemExit("Local Graph requires hierarchical short,mid,long memory")
    if any((rgm_effective["enabled"], v3_effective["enabled"], v4_effective["enabled"])):
        raise SystemExit("Local Graph cannot share a run with RGM, V3, or V4")
    local_graph_effective = asdict(local_config)

snapshot = {
    "common": {
        "model_path": str(model_path),
        "model_config_sha256": digest_if_file(model_path / "config.json"),
        "eval_config_sha256": digest_if_file(model_path / "eval_config.json"),
        "source_python_sha256": code_digest.hexdigest(),
        "model_weights_sha256": digest_tree(model_path, {".safetensors", ".bin"}),
        "action_bundle_sha256": digest_tree(bundle),
        "backend": env["SNAPSHOT_BACKEND"],
        "task": env["SNAPSHOT_TASK"],
        "split": env["SNAPSHOT_SPLIT"],
        "habitat_config": str(habitat_config),
        "habitat_config_sha256": digest_if_file(habitat_config),
        "data_path": str(Path(env["SNAPSHOT_DATA_PATH"]).resolve()) if env["SNAPSHOT_DATA_PATH"] else "",
        "scenes_dir": str(Path(env["SNAPSHOT_SCENES_DIR"]).resolve()) if env["SNAPSHOT_SCENES_DIR"] else "",
        "seed": int(env["SNAPSHOT_SEED"]) if env["SNAPSHOT_SEED"] else None,
        "episodes": int(env["SNAPSHOT_EPISODES"]),
        "expected_episodes": int(env["SNAPSHOT_EXPECTED_EPISODES"]),
        "max_steps": int(env["SNAPSHOT_MAX_STEPS"]),
        "force_stop_at_max_steps": env["SNAPSHOT_NO_FORCE_STOP"] != "1",
        "gpu_ids": env["SNAPSHOT_GPU_IDS"].split(),
        "action_tokenizer_bundle": env["SNAPSHOT_ACTION_BUNDLE"],
        "gpu_memory_utilization": float(env["SNAPSHOT_GPU_MEM_UTIL"]),
        "client_non_memory_args": common_tokens,
        "inference_env": {
            key: env.get(key, "") for key in (
                "VLN_EVAL_TEMPERATURE", "VLN_EVAL_TOP_P", "VLN_EVAL_TOP_K",
                "VLN_EVAL_TRAJ_TOP1",
            )
        },
    },
    "memory_args": memory_tokens,
    "memory_effective": {
        "enabled": "--memory_enabled" in memory_tokens,
        "rgm": rgm_effective,
    },
    "client_args": tokens,
    "server_args": shlex.split(env["SNAPSHOT_SERVER_ARGS"]),
}
if v3_effective["enabled"]:
    snapshot["protocol_version"] = v3_effective["protocol_version"]
    snapshot["memory_effective"]["rgm_v3"] = v3_effective
if v4_effective["enabled"]:
    snapshot["protocol_version"] = v4_effective["protocol_version"]
    snapshot["memory_effective"]["rgm_v4"] = v4_effective
if local_graph_effective is not None:
    snapshot["protocol_version"] = local_graph_effective["protocol_version"]
    snapshot["local_graph_effective"] = local_graph_effective
    snapshot["memory_effective"]["local_graph_effective"] = local_graph_effective
if env["SNAPSHOT_PILOT_MANIFEST"]:
    pilot_path = Path(env["SNAPSHOT_PILOT_MANIFEST"])
    snapshot["pilot_manifest_path"] = str(pilot_path)
    snapshot["pilot_manifest_sha256"] = digest_if_file(pilot_path)
target = Path(env["SNAPSHOT_OUTPUT_ROOT"]) / "config_snapshot.yaml"
if target.exists():
    if json.loads(target.read_text(encoding="utf-8")) != snapshot:
        raise SystemExit(f"existing config snapshot differs from this run: {target}")
else:
    with target.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
PY

# ── cleanup ──────────────────────────────────────────────────────────────────
SERVER_PIDS=()
CLIENT_PIDS=()
cleanup() {
    echo "$TAG cleaning up..."
    for pid in "${CLIENT_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${CLIENT_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# ── 1. env servers ───────────────────────────────────────────────────────────
server_args=(--task "$TASK" --config "$HABITAT_CONFIG" --max-steps "$MAX_STEPS")
[ -n "$SEED" ] && server_args+=(--seed "$SEED")
[ "$STRICT_FULL_EVAL" = 1 ] && server_args+=(--strict-full-eval)
[ -n "$SPLIT" ] && server_args+=(--split "$SPLIT")
[ -n "$SUCCESS_DISTANCE" ] && server_args+=(--success-distance "$SUCCESS_DISTANCE")
[ -n "$NAVMESH_CELL_HEIGHT" ] && server_args+=(--navmesh-cell-height "$NAVMESH_CELL_HEIGHT")
[ -n "$DATA_PATH" ] && server_args+=(--data-path "$DATA_PATH")
[ -n "$SCENES_DIR" ] && server_args+=(--scenes-dir "$SCENES_DIR")
[ -n "$PILOT_MANIFEST_PATH" ] && server_args+=(--pilot-manifest "$PILOT_MANIFEST_PATH")
# shellcheck disable=SC2206
[ -n "$SERVER_ARGS" ] && server_args+=($SERVER_ARGS)

for ((i = 0; i < N; i++)); do
    g=${GPUS[$i]}
    port=$((BASE_PORT + i))
    log="$LOG_DIR/server_gpu${g}_shard${i}.log"
    echo "$TAG server shard $i/$N on GPU $g -> tcp://localhost:$port  (log: $log)"
    HABITAT_SIM_GPU_ID=$g PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT/habitat_server${PYTHONPATH:+:$PYTHONPATH}" \
    LD_LIBRARY_PATH="${HABITAT_LD}${HABITAT_LD:+:}${LD_LIBRARY_PATH:-}" \
        "$HABITAT_PYTHON" -m lightnav_habitat.serve "${server_args[@]}" \
            --port "$port" --split-id "$i" --split-num "$N" \
            --ready-file "$READY_DIR/shard${i}.ready" >"$log" 2>&1 &
    SERVER_PIDS+=("$!")
done

# ── 2. wait for ready ────────────────────────────────────────────────────────
DEADLINE=$((SECONDS + READY_TIMEOUT_S))
LAST=-1
while :; do
    # -mindepth 1 -type f: without it `find` also matches the $READY_DIR directory itself
    # (it is named ".ready"), so the count was always files+1. Combined with the old strict
    # `-eq $N` test that let the clients start one server early, and hung until
    # READY_TIMEOUT_S whenever a poll never observed exactly N-1 files -- which is the
    # normal case on a multi-GPU box, where the last servers become ready inside one tick.
    READY=$(find "$READY_DIR" -mindepth 1 -maxdepth 1 -type f -name '*.ready' 2>/dev/null | wc -l)
    [ "$READY" -ne "$LAST" ] && { echo "$TAG   $READY/$N servers ready"; LAST=$READY; }
    [ "$READY" -ge "$N" ] && break
    for pid in "${SERVER_PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$TAG ERROR: an env server exited before becoming ready; last log lines:" >&2
            for f in "$LOG_DIR"/server_*.log; do echo "--- $f ---" >&2; tail -15 "$f" >&2; done
            exit 1
        fi
    done
    if [ "$SECONDS" -ge "$DEADLINE" ]; then
        echo "$TAG ERROR: only $READY/$N servers ready after ${READY_TIMEOUT_S}s" >&2
        for f in "$LOG_DIR"/server_*.log; do echo "--- $f ---" >&2; tail -15 "$f" >&2; done
        exit 1
    fi
    sleep 3
done

# The ready files contain the exact IDs loaded by each server after dataset
# filtering and sharding. A formal run checks this before loading model weights.
if [ "$STRICT_FULL_EVAL" = 1 ]; then
    ready_files=()
    for ((i = 0; i < N; i++)); do ready_files+=("$READY_DIR/shard${i}.ready"); done
    manifest_args=(
        collect --output "$OUTPUT_ROOT/episode_manifest.json"
        --expected-count "$EXPECTED_EPISODES"
    )
    [ -n "$EPISODE_MANIFEST_PATH" ] && manifest_args+=(--shared "$EPISODE_MANIFEST_PATH")
    [ -n "$PILOT_MANIFEST_PATH" ] && manifest_args+=(--pilot-selection "$PILOT_MANIFEST_PATH")
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$CLIENT_PYTHON" -m lightnav.habitat.experiment_integrity \
        "${manifest_args[@]}" "${ready_files[@]}"
fi

# ── 3. eval clients ──────────────────────────────────────────────────────────
client_args=(--model_path "$MODEL_PATH" --backend "$BACKEND" --episodes "$EPISODES" \
             --max_steps "$MAX_STEPS" --gpu_memory_utilization "$GPU_MEM_UTIL")
[ -n "$SEED" ] && client_args+=(--seed "$SEED")
[ "$NO_FORCE_STOP" = 1 ] && client_args+=(--no_force_stop)
[ "$STRICT_FULL_EVAL" = 1 ] && client_args+=(--strict_full_eval)
[ "$RESUME" = 1 ] && client_args+=(--resume)
[ -n "$ACTION_TOKENIZER_BUNDLE" ] && client_args+=(--action_tokenizer_bundle "$ACTION_TOKENIZER_BUNDLE")
# shellcheck disable=SC2206
[ -n "$LANGUAGES" ] && client_args+=(--languages $LANGUAGES)
client_args+=("${client_extra_args[@]}")

for ((i = 0; i < N; i++)); do
    g=${GPUS[$i]}
    port=$((BASE_PORT + i))
    shard_dir="$OUTPUT_ROOT/shard_$i"
    log="$LOG_DIR/client_gpu${g}_shard${i}.log"
    progress_log="$LOG_DIR/client_progress_gpu${g}_shard${i}.log"
    echo "$TAG client shard $i/$N on GPU $g -> $shard_dir  (log: $log)"
    # Both descriptors append to the same client log: stdout stays on disk,
    # while progress (stderr) is also streamed live through tee. Each shard
    # labels its own counts; no per-shard count is presented as global.
    : > "$log"
    progress_tty_args=()
    if [ "$N" -eq 1 ] && [ -t 2 ]; then
        progress_tty_args=(--progress_tty)
    fi
    CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    LD_LIBRARY_PATH="${CLIENT_LD}${CLIENT_LD:+:}${LD_LIBRARY_PATH:-}" \
        "$CLIENT_PYTHON" -m lightnav.cli.eval_habitat "${client_args[@]}" \
            --progress_shard_index "$i" --progress_shard_count "$N" \
            "${progress_tty_args[@]}" \
            --server "tcp://localhost:$port" --output_dir "$shard_dir" \
            >>"$log" 2> >(tee -a "$log" "$progress_log" >&2) &
    CLIENT_PIDS+=("$!")
done

FAILED=0
for ((i = 0; i < N; i++)); do
    if wait "${CLIENT_PIDS[$i]}"; then
        echo "$TAG shard $i finished"
    else
        echo "$TAG WARN: shard $i client exited with an error (see $LOG_DIR/client_*_shard${i}.log)" >&2
        FAILED=$((FAILED + 1))
    fi
done

# ── 4. merge ─────────────────────────────────────────────────────────────────
if [ "$local_graph_client_enabled" -eq 1 ]; then
    if [ "$FAILED" -ne 0 ]; then
        echo "$TAG ERROR: Local Graph shard failed; refusing to merge partial results" >&2
        exit 1
    fi
    "$CLIENT_PYTHON" "$REPO_ROOT/scripts/verify_local_graph_shards.py" \
        "$OUTPUT_ROOT" "${GPUS[@]}"
fi
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$CLIENT_PYTHON" -m lightnav.cli.eval_merge "$OUTPUT_ROOT"/shard_* --output "$OUTPUT_ROOT" \
    | tee "$OUTPUT_ROOT/merge.log"
# Rebuild the derived diagnostics file from all shard logs on every attempt.
diagnostics_tmp="$OUTPUT_ROOT/memory_diagnostics.jsonl.$ATTEMPT_ID.tmp"
: > "$diagnostics_tmp"
for ((i = 0; i < N; i++)); do
    [ -f "$OUTPUT_ROOT/shard_$i/memory_diagnostics.jsonl" ] \
        && cat "$OUTPUT_ROOT/shard_$i/memory_diagnostics.jsonl" >> "$diagnostics_tmp"
done
mv "$diagnostics_tmp" "$OUTPUT_ROOT/memory_diagnostics.jsonl"
if [ "$STRICT_FULL_EVAL" = 1 ] && [ "$FAILED" -eq 0 ]; then
    verify_args=(verify "$OUTPUT_ROOT")
    [ -n "$EPISODE_MANIFEST_PATH" ] && verify_args+=(--shared "$EPISODE_MANIFEST_PATH")
    [ -n "$PILOT_MANIFEST_PATH" ] && verify_args+=(--pilot-selection "$PILOT_MANIFEST_PATH")
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$CLIENT_PYTHON" -m lightnav.habitat.experiment_integrity "${verify_args[@]}"
fi
echo "$TAG done: $OUTPUT_ROOT/summary.json  (failed shards: $FAILED)"
[ "$FAILED" -eq 0 ]
