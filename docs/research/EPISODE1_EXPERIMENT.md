# Local Graph Fusion：88 Episode 单组正式评测

本文件沿用历史文件名；**当前目标是固定的 R2R `val_unseen` 88 Episode Pilot，只运行 Local Graph Fusion 一组**。此前 Episode 1 的 A/B/C 单条对照协议不适用于本轮，也不运行 `--episodes 1`、`--trace_steps` 或 `scripts/compare_episode1_traces.py`。评测启动器为 [eval_habitat.sh](../../scripts/eval_habitat.sh)，Local Graph 的算法与参数见 [MEMORY_DESIGN.md](MEMORY_DESIGN.md)。

## 当前状态

截至 2026-11-14，本工作区没有 AutoDL 上的 R2R 数据、88 条 Pilot manifest、LightNav 权重或 GPU。本次只修复启动器并准备命令，**没有启动评测**。本轮 Local Graph 的 88 条覆盖、SR、SPL、注入率与分片合并结果均待 AutoDL 实测，不能从旧 FIX3 结果推断。

## 固定输入与输出

| 项目 | 本轮设置 |
| --- | --- |
| Episode 集合 | 复用已有 `rgm_pilot_88_v1` 的 88 条 `pilot_manifest.json` 和共享 `episode_manifest.json`，核对精确 ID |
| 数据与环境 | R2R `val_unseen`、原 Habitat YAML/场景、每条最多 500 步、强制 STOP 关闭、随机种子 42 |
| 模型与动作 | 同一个冻结 LightNav checkpoint、原 Pointing/RVQ 接口与动作解码参数；不训练、不改权重 |
| 记忆组 | 仅 `--local_graph_enabled`，配同一 YOLO/DINO 资产；`hierarchical` 和 `short,mid,long` |
| 输出 | 全新、独立的 `OUTPUT_ROOT`，不位于 FIX3 或旧 Pilot 目录内；不复用或删除旧结果 |

88 条 Pilot 的身份由现有 manifest 的 Scene/Episode **完整 pair** 决定，不用 JSON 的 `episode_id=1` 猜测“Episode 1”。本轮是单组正式评测，因此无需启动 baseline 或 FIX3 评测；以后若做横向比较，应先核对权重、Pilot manifest、环境与动作解码完全一致。

## AutoDL 运行命令（本轮不执行）

先把本次启动器修复及当前 Local Graph 代码同步到 AutoDL，使用原推理环境与独立 Habitat Python 3.9 环境。把示例路径替换为真实路径；`CLIENT_ARGS` 的模型路径不能含空格。Pilot manifest 放在新输出目录**之外**。

```bash
export MODEL_PATH=/path/to/LightNav-0
export ACTION_TOKENIZER_BUNDLE=/path/to/LightNav-0/action_tokenizer
export HABITAT_CONFIG=habitat_server/configs/vlnce_r2r.yaml
export DATA_PATH=/path/to/R2R_VLNCE_v1-3_preprocessed
export SCENES_DIR=/path/to/scene_datasets
export YOLO_MODEL_PATH=/path/to/yolo11n.pt
export DINO_MODEL_PATH=/path/to/dinov2-small
export PILOT_MANIFEST_PATH=/path/to/rgm_pilot_88_v1/pilot_manifest.json
export EPISODE_MANIFEST_PATH=/path/to/rgm_pilot_88_v1/episode_manifest.json
export OUTPUT_ROOT=/path/to/output/local_graph_fusion_88_run1
export GPU_IDS="0 1"
export GPU_MEM_UTIL=0.65

export TASK=vlnce SPLIT=val_unseen BACKEND=vllm_local
export STRICT_FULL_EVAL=1 EXPECTED_EPISODES=88 EPISODES=-1
export MAX_STEPS=500 NO_FORCE_STOP=1 SEED=42 RESUME=0
export SERVER_ARGS="" LANGUAGES=""
unset VLN_EVAL_TEMPERATURE VLN_EVAL_TOP_P VLN_EVAL_TOP_K VLN_EVAL_TRAJ_TOP1
export CLIENT_ARGS="--memory_enabled --memory_strategy hierarchical --memory_layers short,mid,long --memory_extractor yolo11 --memory_extractor_model_path $YOLO_MODEL_PATH --memory_extractor_device cpu --memory_extractor_confidence 0.25 --memory_extract_every_n_steps 5 --memory_place_enabled --memory_place_model_path $DINO_MODEL_PATH --memory_place_device cpu --memory_place_extract_every_n_steps 5 --memory_place_similarity_threshold 0.85 --local_graph_enabled --local_graph_scope episode_local --local_graph_prompt_max_chars 320 --local_graph_max_landmarks 2 --local_graph_reactivation_steps 8"

if [ -e "$OUTPUT_ROOT" ]; then
  echo "Choose a new, empty OUTPUT_ROOT" >&2
else
  bash scripts/eval_habitat.sh
fi
```

`GPU_IDS` 按 AutoDL 实际 GPU 调整；脚本每 GPU 起一个 Habitat server 和一个评测 client，以不重叠的 shard 处理同一 88 条集合。`PILOT_MANIFEST_PATH` 固定集合，`EPISODE_MANIFEST_PATH` 与旧运行的共享清单核对；运行前确认两份 manifest 都存在，不能用旧 `results.jsonl` 代替。`--trace_steps` 只支持单 Episode，因此此正式运行使用每步 `memory_diagnostics.jsonl` 与各 shard 的结果文件。Local Graph 输出目录要求全新；中断后保留已有结果，另开新目录重跑，不对已有目录设 `RESUME=1`。

## 运行后核查

启动器应先写根目录 `config_snapshot.yaml`，其中 `memory_args` 包含 `--local_graph_*`，根级 `local_graph_effective`（同时镜像到 `memory_effective.local_graph_effective`）记录 `enabled`、`scope`、`prompt_max_chars`、`max_landmarks`、`reactivation_steps`、`protocol_version`。检查其模型权重、动作 bundle、Habitat YAML、代码与 Pilot manifest 校验信息，然后检查每个 `shard_i/local_graph_protocol.json` 和 `shard_i/episode_manifest.json`。各 shard 的 Episode pair 必须不重叠，合并后的 `results.jsonl` 必须恰有 88 条。

启动器负责合并 `shard_*/results.jsonl` 到根目录，并写 `summary.json`、`merge.log`、合并的 `memory_diagnostics.jsonl`；正式模式再调用 `experiment_integrity verify` 校验覆盖并生成 `pilot_complete.json`。**完成判据是 `pilot_complete.json` 存在、`summary.num_episodes == 88`，且 88 个 Scene/Episode pair 唯一并与 Pilot manifest 一致**；`summary.json` 单独存在可能只是部分结果，不能据此报告正式 SR/SPL。可以在 AutoDL 对产物做一次只读复核：

```bash
python -m lightnav.habitat.experiment_integrity verify "$OUTPUT_ROOT" \
  --shared "$EPISODE_MANIFEST_PATH" \
  --pilot-selection "$PILOT_MANIFEST_PATH"
python - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
snapshot = json.loads((root / "config_snapshot.yaml").read_text())
local = snapshot["local_graph_effective"]
assert local["enabled"] is True and local["protocol_version"] == "local_graph_v1"
assert local == snapshot["memory_effective"]["local_graph_effective"]
rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines() if line]
keys = {(row["scene_id"], row["habitat_episode_id"]) for row in rows}
assert len(rows) == len(keys) == 88
summary = json.loads((root / "summary.json").read_text())
assert summary["num_episodes"] == 88
assert (root / "pilot_complete.json").is_file()
print("88 unique Episodes; SR/SPL:", summary["metrics"]["SR_success_rate_pct"], summary["metrics"]["SPL_pct"])
PY
```

若某 shard 失败或合并结果缺失，保留其日志和目录，不把部分汇总写成正式 88 条结果。后续可按每步诊断统计模型输入中的实际记忆注入，并与已有 FIX3 汇总做同协议比较；这不需要重跑 FIX3。
