# Episode 1：单 Episode 对照协议与当前结果

## 当前实测状态（2026-11-14）

本地检查未找到 R2R `val_unseen.json.gz`、既有 `episode_manifest.json`、LightNav checkpoint 或 FIX3 的逐步结果；用户确认这些资产在 AutoDL。当前环境的 `nvidia-smi` 也无法访问 GPU。因此**本轮没有运行真实 Episode 1**，不能填入其准确 `episode_id`、`scene_id`、指令、起点、SR、SPL、动作分歧或首次偏离正确路线的 Step。旧文档的 88 Episode 汇总不是 Episode 1 的逐步证据，也不替代本实验。

| 核对项 | 本地结果 |
| --- | --- |
| Episode 1 的准确身份、指令、起点 | 待在 AutoDL 用 manifest + 数据集核对；不把 JSON `episode_id=1` 当作 Episode 1 |
| A `baseline_off` | 无可证实的完全匹配单 Episode 结果 |
| B `rgm_v4_fix3` | 无可证实的完全匹配单 Episode 逐步结果 |
| C `local_graph` | 尚未真实运行 |
| 首次动作分歧、无效探索、错误转向、路线偏离 | 均未确定 |

## 1. 先核对身份

在装有本分支代码和 R2R 资产的 AutoDL 环境，设置实际路径；`SCENES_DIR` 必须和 Habitat server 使用的一致：

```bash
export R2R_ROOT=/path/to/R2R_VLNCE_v1-3_preprocessed
export SCENES_DIR=/path/to/scene_datasets
export MANIFEST=/path/to/existing/episode_manifest.json
export MODEL_PATH=/path/to/LightNav-0
export YOLO_MODEL_PATH=/path/to/yolo11n.pt
export DINO_MODEL_PATH=/path/to/dinov2-small
export RVQ_BUNDLE=/path/to/LightNav-0/action_tokenizer
export RUN_ROOT=output/episode1_local_graph_20261114_run1
mkdir -p "$RUN_ROOT"
python scripts/resolve_single_episode.py \
  --manifest "$MANIFEST" \
  --dataset "$R2R_ROOT/val_unseen/val_unseen.json.gz" \
  --scenes-dir "$SCENES_DIR" --episode-number 1 \
  > "$RUN_ROOT/identity.json"
cat "$RUN_ROOT/identity.json"
```

`--episode-number 1` 指 **manifest 第一行**，不是数据集 JSON 的 `episode_id=1`。若过去的“Episode 1”实际上是结果标签 `episode_001`，它对应的可能是 manifest 第二行；应先用旧结果的 `scene_id + habitat_episode_id` 查清，再改为 `--episode-number 2` 等正确序号。脚本会要求数据集恰好有一条匹配，并输出原指令、`start_position`、`start_rotation`、`trajectory_id`、可用的 `reference_path` / `goal_position` 和给服务端的完整筛选参数。参考路线与目标只写入离线身份文件，不送给策略。没有这些真实输出前，本文件不代填身份。

## 2. 只启动该 Episode 的 Habitat 服务

服务端仍使用原 R2R YAML、场景、500 步限制和种子；新 `--single-scene-id` / `--single-episode-id` 在 Habitat 加载后、迭代前精确筛选一个 Episode。服务端在独立的 Habitat Python 3.9 环境运行。

```bash
SCENE_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["single_scene_id"])' "$RUN_ROOT/identity.json")"
EP_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["single_episode_id"])' "$RUN_ROOT/identity.json")"
python -m lightnav_habitat.serve \
  --task vlnce --config habitat_server/configs/vlnce_r2r.yaml \
  --data-path "$R2R_ROOT" --scenes-dir "$SCENES_DIR" \
  --single-scene-id "$SCENE_ID" --single-episode-id "$EP_ID" \
  --max-steps 500 --seed 42 --port 5555 \
  --ready-file "$RUN_ROOT/server_A.ready"
```

在单独终端启动服务，用端口或本组的 ready file 确认就绪。记录服务端 manifest 应为 `expected_count=1`，且 pair 与 `identity.json` 一致。此筛选不能与旧 88/220 Pilot、严格全量评测或分片开关合用。

**每一组评测都要重启一次服务端**：评测客户端结束时发送 `close`，服务端随即退出。运行 A 前使用 `server_A.ready`；运行 B/C 前分别用完全相同的服务命令重启，只把 ready file 换成 `server_B.ready` / `server_C.ready`。确认三次使用相同配置、同一 Scene/Episode、同一种子，并核对 Step 0 的 RGB 像素校验和与位姿。不要把已经退出的服务当作可再次 reset 的实例。

## 3. 三组使用同一冻结模型和动作解码器

在 inference Python 3.11 环境执行。三个输出目录必须事先不存在或为空；`--trace_steps` 会拒绝覆盖已有 RGB/逐步结果。若旧 A/B 结果具有**完全相同**的 Episode pair、权重/源码/环境配置、指令、RVQ 解码、种子和逐步 Prompt/RGB/Token/动作记录，可只复用它们并运行 C；否则仅补跑 A/B 的这一个 Episode。每组结束后让同一单 Episode 服务端 reset；确认 Step 0 RGB/位姿一致。

```bash
common=(--model_path "$MODEL_PATH" --backend vllm_local \
  --server tcp://127.0.0.1:5555 --episodes 1 --max_steps 500 \
  --no_force_stop --seed 42 --trace_steps \
  --action_tokenizer_bundle "$RVQ_BUNDLE" --horizon 10)
memory=(--memory_enabled --memory_strategy hierarchical \
  --memory_layers short,mid,long \
  --memory_extractor yolo11 --memory_extractor_model_path "$YOLO_MODEL_PATH" \
  --memory_extractor_device cpu --memory_extract_every_n_steps 5 \
  --memory_place_enabled --memory_place_model_path "$DINO_MODEL_PATH" \
  --memory_place_device cpu --memory_place_extract_every_n_steps 5)

lightnav-eval-habitat "${common[@]}" \
  --output_dir "$RUN_ROOT/A_baseline_off"
# A 完成后按第 2 节重启单 Episode 服务端，ready file 改为 server_B.ready
lightnav-eval-habitat "${common[@]}" "${memory[@]}" \
  --rgm_v4_enabled --rgm_v4_scope episode_local \
  --output_dir "$RUN_ROOT/B_rgm_v4_fix3"
# B 完成后按第 2 节重启单 Episode 服务端，ready file 改为 server_C.ready
lightnav-eval-habitat "${common[@]}" "${memory[@]}" \
  --local_graph_enabled --local_graph_scope episode_local \
  --output_dir "$RUN_ROOT/C_local_graph"
```

先核对 AutoDL 的 FIX3 实际参数；若其 YOLO/DINO 周期、阈值、RVQ bundle、`max_new_tokens`、图像大小或 V4 scope 不同，应在 B/C 中显式保持其共同参数，不把不同协议叫作 FIX3 对照。不能为迁就环境改用别的 checkpoint、修改权重、训练 LightNav 或改变 RVQ/Pointing 接口。只运行 `--episodes 1`，不要启动 88 条脚本。

## 4. 逐步记录与判读

每组输出 `episode_manifest.json`、`results.jsonl`、`memory_diagnostics.jsonl`、`step_trace.jsonl`、`rgb/step_NNNN.png`。逐步 trace 包含实际 Prompt、Prompt 内记忆、经最终 Token 审计确认的记忆、历史帧 ID、原始 RGB 像素校验和、Pointing Token/像素、RVQ 原始 Token/码、完整 waypoint、实际速度动作、动作前后位姿及目标距离。位姿和目标距离只用于离线评价，**不送入在线记忆或模型**。若某字段不可用，写 `null`，不得补造。

```bash
python scripts/compare_episode1_traces.py \
  --baseline "$RUN_ROOT/A_baseline_off" \
  --fix3 "$RUN_ROOT/B_rgm_v4_fix3" \
  --local-graph "$RUN_ROOT/C_local_graph" \
  --output "$RUN_ROOT/paired_comparison.json"
```

比较器要求三组身份、指令、初始 RGB、初始位姿以及记录到的模型路径、解码器方法/时域、推理上限和历史帧配置相同；报告每步 `action_divergence_vs_baseline`（实际发给 Habitat 的速度命令）、`rvq_token_divergence_vs_baseline`、`pointing_token_divergence_vs_baseline`、`same_rgb`，以及第一处**同 RGB 的动作分歧**和第一处 RGB 分叉。不同 RVQ Token 若解码成相同命令，只记 Token 分歧。分叉后的不同动作受不同观测影响，不能作为相同输入下的模型因果对照。`invalid_exploration`、`wrong_turn` 与 `first_correct_route_deviation_step` 保持未判定：审阅者须把数据集 `reference_path`、每步位姿、RGB、目标距离与指令子句一起检查，标出最早离开正确路线的 Step 和证据；一次目标距离增大或一次动作不同并不足以定性为错误转向。

最终填写三组各自的成功、SPL、最终/最小目标距离、步数、实际注入步数及第一次路线偏离；若 C 失败，也需定位是无证据、证据未进 Token、提示误导还是模型忽略提示。随后可在相同配置下从同场景挑 3–5 条不同长度 Episode 做小规模、预先固定 ID 的配对验证，再决定是否扩大评测。
