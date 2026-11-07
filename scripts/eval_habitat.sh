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
if [ "$STRICT_FULL_EVAL" = 1 ]; then
    if [ "$EXPECTED_EPISODES" != 1839 ]; then
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
SNAPSHOT_ACTION_BUNDLE="$ACTION_TOKENIZER_BUNDLE" \
SNAPSHOT_GPU_MEM_UTIL="$GPU_MEM_UTIL" SNAPSHOT_CLIENT_ARGS="$CLIENT_ARGS" \
SNAPSHOT_SERVER_ARGS="$SERVER_ARGS" SNAPSHOT_EXPECTED_EPISODES="$EXPECTED_EPISODES" \
SNAPSHOT_DATA_PATH="$DATA_PATH" SNAPSHOT_SCENES_DIR="$SCENES_DIR" \
"$CLIENT_PYTHON" - <<'PY'
import hashlib
import json
import os
import shlex
from pathlib import Path

env = os.environ
habitat_config = Path(env["SNAPSHOT_HABITAT_CONFIG"]).resolve()
model_path = Path(env["SNAPSHOT_MODEL_PATH"]).resolve()
tokens = shlex.split(env["SNAPSHOT_CLIENT_ARGS"])
common_tokens, memory_tokens = [], []
index = 0
while index < len(tokens):
    token = tokens[index]
    group = memory_tokens if token.startswith("--memory_") else common_tokens
    group.append(token)
    if token.startswith("--") and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
        index += 1
        group.append(tokens[index])
    index += 1

def digest_if_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

snapshot = {
    "common": {
        "model_path": str(model_path),
        "model_config_sha256": digest_if_file(model_path / "config.json"),
        "eval_config_sha256": digest_if_file(model_path / "eval_config.json"),
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
    "client_args": tokens,
    "server_args": shlex.split(env["SNAPSHOT_SERVER_ARGS"]),
}
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
# shellcheck disable=SC2206
[ -n "$CLIENT_ARGS" ] && client_args+=($CLIENT_ARGS)

for ((i = 0; i < N; i++)); do
    g=${GPUS[$i]}
    port=$((BASE_PORT + i))
    shard_dir="$OUTPUT_ROOT/shard_$i"
    log="$LOG_DIR/client_gpu${g}_shard${i}.log"
    echo "$TAG client shard $i/$N on GPU $g -> $shard_dir  (log: $log)"
    CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    LD_LIBRARY_PATH="${CLIENT_LD}${CLIENT_LD:+:}${LD_LIBRARY_PATH:-}" \
        "$CLIENT_PYTHON" -m lightnav.cli.eval_habitat "${client_args[@]}" \
            --server "tcp://localhost:$port" --output_dir "$shard_dir" >"$log" 2>&1 &
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
    PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$CLIENT_PYTHON" -m lightnav.habitat.experiment_integrity "${verify_args[@]}"
fi
echo "$TAG done: $OUTPUT_ROOT/summary.json  (failed shards: $FAILED)"
[ "$FAILED" -eq 0 ]
