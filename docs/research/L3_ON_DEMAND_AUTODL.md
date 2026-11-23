# L1 Original + On-Demand L3：AutoDL 运行、预检与显存预算

本文只覆盖将 **`--memory_prompt_mode l1_original_only`** 与
**`--spatial_l3_mode on_demand`** 组合起来的单臂运行。它不是六组 Prompt 消融
中的任何一组；历史六组结果和启动器不变。L1 仍由 `LocalGraphFusion` 原文构建，
L3 只在 L1 返回 `no_candidates` 或 `budget_excluded` 时查询。

> 路径说明：本仓库没有保存 AutoDL 的绝对路径（历史文档明确记录该信息未知）。
> 下文的 `/root/autodl-tmp/...` 是 AutoDL 常见数据盘约定，必须替换为实际
> `MODEL_PATH`、`DATA_PATH`、`SCENES_DIR`、两份 Manifest 和权重路径。不要用
> 旧 `results.jsonl` 代替 Manifest。

## 1. Bash 变量展开修复

错误写法（`$YOLO` 在赋值生效前展开为空）：

```bash
# WRONG
YOLO=/path/yolo11n.pt CLIENT_ARGS="--memory_extractor_model_path $YOLO" \
  bash scripts/eval_habitat.sh
```

正确写法是先把路径 `export` 成独立语句，再构造 `CLIENT_ARGS`：

```bash
export MODEL_PATH=/root/autodl-tmp/LightNav-0
export YOLO_MODEL_PATH=/root/autodl-tmp/weights/yolo11n.pt
export SEG_MODEL_PATH=/root/autodl-tmp/weights/yolo11n-seg.pt
export CLIP_MODEL_PATH=/root/autodl-tmp/weights/open_clip_ViT-H-14_laion2b_s32b_b79k.bin
export EPISODE_MANIFEST_PATH=/root/autodl-tmp/manifests/88_episode_manifest.json
export PILOT_MANIFEST_PATH=/root/autodl-tmp/manifests/88_episode_pilot.json

export CLIENT_ARGS="--no_force_stop --verbose \
  --memory_enabled --memory_strategy hierarchical \
  --memory_layers short,mid,long --memory_top_k 5 --memory_max_context_chars 600 \
  --memory_extractor yolo11 --memory_extractor_model_path $YOLO_MODEL_PATH \
  --memory_extractor_device cpu --memory_extractor_confidence 0.25 \
  --memory_extract_every_n_steps 5 \
  --local_graph_enabled --local_graph_scope episode_local \
  --local_graph_prompt_max_chars 320 --local_graph_max_landmarks 2 \
  --local_graph_reactivation_steps 8 \
  --memory_prompt_mode l1_original_only \
  --spatial_l3_mode on_demand \
  --spatial_l3_seg_model_path $SEG_MODEL_PATH --spatial_l3_seg_device cpu \
  --spatial_l3_seg_confidence 0.25 --spatial_l3_seg_every_n_steps 5 \
  --spatial_l3_evidence_min_iou 0.5 --spatial_l3_evidence_ambiguity_margin 0.1 \
  --spatial_l3_min_injection_confidence 0.5 \
  --spatial_l3_clip_model_path $CLIP_MODEL_PATH \
  --spatial_l3_clip_model_name ViT-H-14 --spatial_l3_clip_device cuda:0 \
  --spatial_l3_clip_precision fp16 --spatial_l3_clip_batch_size 8 \
  --spatial_l3_clip_top_k 3 --spatial_l3_clip_min_similarity 0.0 \
  --prompt_measurement_enabled"
```

`scripts/preflight_l3_on_demand.py` 会输出 `client_args_shell`（`shlex.join`），
避免手工拼接。测试 `tests/test_preflight_l3_on_demand.py` 已验证
`shlex.split(client_args_shell) == client_args`。

## 2. 本地 OpenCLIP 权重准备（不下载）

ConceptGraphs 官方使用 OpenCLIP `ViT-H-14` / `laion2b_s32b_b79k`。在有网络的
人工机器上下载一次，然后把文件拷到 AutoDL；LightNav 只接受本地文件：

```bash
# 人工在有网络的机器上准备，示例；本仓库不执行下载
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-H-14','laion2b_s32b_b79k')"
# 将缓存的 open_clip_pytorch_model.bin 拷贝为：
# /root/autodl-tmp/weights/open_clip_ViT-H-14_laion2b_s32b_b79k.bin
```

安装可选依赖：

```bash
cd /root/autodl-tmp/LightNav-0-codex-conceptgraphs
export UV_CACHE_DIR=$PWD/.uv-cache
uv pip install --python .venv311/bin/python -e ".[clip]"
```

`LocalClipEncoder` 先检查 `Path.is_file()`，再走 OpenCLIP 的本地 checkpoint
分支；不会访问模型 Hub。`--spatial_l3_clip_precision fp16` 只影响推理精度，
不改官方余弦检索。若本地文件不是 OpenCLIP 可识别的 checkpoint，启动会直接报错。

## 3. AutoDL 不启动 GPU 的环境预检

### 3.1 只检查路径并打印 shell-safe 命令（不导入模型包）

```bash
cd /root/autodl-tmp/LightNav-0-codex-conceptgraphs
PYTHONPATH=src .venv311/bin/python scripts/preflight_l3_on_demand.py \
  --model-path /root/autodl-tmp/LightNav-0 \
  --yolo-model /root/autodl-tmp/weights/yolo11n.pt \
  --seg-model /root/autodl-tmp/weights/yolo11n-seg.pt \
  --clip-model /root/autodl-tmp/weights/open_clip_ViT-H-14_laion2b_s32b_b79k.bin \
  --episode-manifest /root/autodl-tmp/manifests/88_episode_manifest.json \
  --pilot-manifest /root/autodl-tmp/manifests/88_episode_pilot.json \
  --habitat-config habitat_server/configs/vlnce_r2r.yaml \
  --data-path /root/autodl-tmp/data --scenes-dir /root/autodl-tmp/scenes \
  --habitat-conda-env habitat --client-python .venv311/bin/python \
  --output-root /root/autodl-tmp/output/l3_on_demand_l1_original_88 \
  --print-only
```

输出 JSON 中的 `client_args_shell` 可直接放入 `CLIENT_ARGS`。`--print-only`
不会导入 `torch/open_clip/ultralytics`，也不查询 CUDA。单 Episode smoke 用同一
命令加 `--smoke`（自动设 `EPISODES=1`、`MAX_STEPS=40`、`STRICT_FULL_EVAL=0`，
不要求两份 manifest）。

### 3.2 完整导入与 Habitat 预检（仍不加载权重、不推理）

去掉 `--print-only`，脚本会运行：

```bash
.venv311/bin/python -c \
  "import lightnav.cli.eval_habitat, open_clip, ultralytics, zmq, torch, transformers"
conda run -n habitat python -c "import habitat, lightnav_habitat"
```

它只导入包，不加载 checkpoint、不分配 GPU 张量、不启动 Habitat server。
预检失败时脚本以非零退出并输出 `errors`。

### 3.3 只读 GPU/环境信息（不推理）

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
.venv311/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
conda run -n habitat python -c "import habitat; print('habitat ok')"
```

`torch.cuda.is_available()` 只查询驱动，不执行模型推理；如果连这个都失败，
不要继续运行实验。

## 4. 单 Episode GPU Smoke Test

先跑 1 个 Episode、最多 40 步，不加 88 Episode manifest，不要求完整覆盖：

```bash
cd /root/autodl-tmp/LightNav-0-codex-conceptgraphs
export MODEL_PATH=/root/autodl-tmp/LightNav-0
export YOLO_MODEL_PATH=/root/autodl-tmp/weights/yolo11n.pt
export SEG_MODEL_PATH=/root/autodl-tmp/weights/yolo11n-seg.pt
export CLIP_MODEL_PATH=/root/autodl-tmp/weights/open_clip_ViT-H-14_laion2b_s32b_b79k.bin
export CLIENT_PYTHON=$PWD/.venv311/bin/python
export HABITAT_CONDA_ENV=habitat
export HABITAT_CONFIG=$PWD/habitat_server/configs/vlnce_r2r.yaml
export DATA_PATH=/root/autodl-tmp/data
export SCENES_DIR=/root/autodl-tmp/scenes
export GPU_IDS=0
export GPU_MEM_UTIL=0.65
export EPISODES=1
export MAX_STEPS=40
export STRICT_FULL_EVAL=0
export NO_FORCE_STOP=1
export OUTPUT_ROOT=/root/autodl-tmp/output/l3_on_demand_smoke_$(date +%Y%m%d_%H%M%S)
export CLIENT_ARGS="--no_force_stop --verbose \
  --memory_enabled --memory_strategy hierarchical \
  --memory_layers short,mid,long --memory_top_k 5 --memory_max_context_chars 600 \
  --memory_extractor yolo11 --memory_extractor_model_path $YOLO_MODEL_PATH \
  --memory_extractor_device cpu --memory_extractor_confidence 0.25 \
  --memory_extract_every_n_steps 5 \
  --local_graph_enabled --local_graph_scope episode_local \
  --local_graph_prompt_max_chars 320 --local_graph_max_landmarks 2 \
  --local_graph_reactivation_steps 8 \
  --memory_prompt_mode l1_original_only \
  --spatial_l3_mode on_demand \
  --spatial_l3_seg_model_path $SEG_MODEL_PATH --spatial_l3_seg_device cpu \
  --spatial_l3_seg_confidence 0.25 --spatial_l3_seg_every_n_steps 5 \
  --spatial_l3_evidence_min_iou 0.5 --spatial_l3_evidence_ambiguity_margin 0.1 \
  --spatial_l3_min_injection_confidence 0.5 \
  --spatial_l3_clip_model_path $CLIP_MODEL_PATH \
  --spatial_l3_clip_model_name ViT-H-14 --spatial_l3_clip_device cuda:0 \
  --spatial_l3_clip_precision fp16 --spatial_l3_clip_batch_size 8 \
  --spatial_l3_clip_top_k 3 --spatial_l3_clip_min_similarity 0.0 \
  --prompt_measurement_enabled"
bash scripts/eval_habitat.sh
```

Smoke 通过判据：`summary.json`/results 中有 1 个 Episode；日志出现
`L3 CLIP cos` 或至少 `l3_query_stats`；`memory_diagnostics.jsonl` 每步有
`prompt_context.budget`、`spatial_l3.l3_query`；无 `clip_feature_error`/OOM。
如果 Episode 没有触发查询，检查 `l3_query_stats.skipped_reasons` 和
`l3_query.last_query_step`。

## 5. 唯一新增 88 Episode 实验命令

保持 L1 Original 参数、88 Episode 集合、Seed 42、500 步与完整评测口径：

```bash
cd /root/autodl-tmp/LightNav-0-codex-conceptgraphs
export MODEL_PATH=/root/autodl-tmp/LightNav-0
export YOLO_MODEL_PATH=/root/autodl-tmp/weights/yolo11n.pt
export SEG_MODEL_PATH=/root/autodl-tmp/weights/yolo11n-seg.pt
export CLIP_MODEL_PATH=/root/autodl-tmp/weights/open_clip_ViT-H-14_laion2b_s32b_b79k.bin
export EPISODE_MANIFEST_PATH=/root/autodl-tmp/manifests/88_episode_manifest.json
export PILOT_MANIFEST_PATH=/root/autodl-tmp/manifests/88_episode_pilot.json
export CLIENT_PYTHON=$PWD/.venv311/bin/python
export HABITAT_CONDA_ENV=habitat
export HABITAT_CONFIG=$PWD/habitat_server/configs/vlnce_r2r.yaml
export DATA_PATH=/root/autodl-tmp/data
export SCENES_DIR=/root/autodl-tmp/scenes
export GPU_IDS=0
export GPU_MEM_UTIL=0.65
export EPISODES=-1
export MAX_STEPS=500
export STRICT_FULL_EVAL=1
export NO_FORCE_STOP=1
export SEED=42
export EXPECTED_EPISODES=88
export OUTPUT_ROOT=/root/autodl-tmp/output/l3_on_demand_l1_original_88_$(date +%Y%m%d_%H%M%S)
export CLIENT_ARGS="--no_force_stop --verbose \
  --memory_enabled --memory_strategy hierarchical \
  --memory_layers short,mid,long --memory_top_k 5 --memory_max_context_chars 600 \
  --memory_extractor yolo11 --memory_extractor_model_path $YOLO_MODEL_PATH \
  --memory_extractor_device cpu --memory_extractor_confidence 0.25 \
  --memory_extract_every_n_steps 5 \
  --local_graph_enabled --local_graph_scope episode_local \
  --local_graph_prompt_max_chars 320 --local_graph_max_landmarks 2 \
  --local_graph_reactivation_steps 8 \
  --memory_prompt_mode l1_original_only \
  --spatial_l3_mode on_demand \
  --spatial_l3_seg_model_path $SEG_MODEL_PATH --spatial_l3_seg_device cpu \
  --spatial_l3_seg_confidence 0.25 --spatial_l3_seg_every_n_steps 5 \
  --spatial_l3_evidence_min_iou 0.5 --spatial_l3_evidence_ambiguity_margin 0.1 \
  --spatial_l3_min_injection_confidence 0.5 \
  --spatial_l3_clip_model_path $CLIP_MODEL_PATH \
  --spatial_l3_clip_model_name ViT-H-14 --spatial_l3_clip_device cuda:0 \
  --spatial_l3_clip_precision fp16 --spatial_l3_clip_batch_size 8 \
  --spatial_l3_clip_top_k 3 --spatial_l3_clip_min_similarity 0.0 \
  --prompt_measurement_enabled"
bash scripts/eval_habitat.sh
```

注意：`--memory_extractor_device cpu`、`--spatial_l3_seg_device cpu` 与
`GPU_MEM_UTIL=0.65` 和六组 L1/L3 发射器一致，是严格复现的默认值；不要为了
4090 D 显存提前改成其他值。只有在 smoke 实测 OOM 时，才把
`GPU_MEM_UTIL` 降到 0.55、显式设置 `VLN_VLLM_ENFORCE_EAGER=1`，并在
`provenance/config_snapshot` 中记录这次偏离。Prompt 格式与 L1 参数不变。

## 6. RTX 4090 D 24 GB 显存预算

估算（bf16/fp16，单 client、单 Habitat server）：

| 组件 | 估算 | 说明 |
| --- | ---: | --- |
| vLLM 4B 权重 | ~10 GB | bf16 |
| vLLM KV cache | 2 GB | 引擎在 `max_num_seqs=1` 时自动取 2 GiB，不额外覆盖 |
| vLLM CUDA graphs/运行时 | 2–3 GB | eager 可减少 |
| Habitat 渲染/CUDA context | 1.5–2.5 GB | 与 client 同卡 |
| YOLO11n-seg | 0.5–1 GB | `seg_device=cuda:0` |
| OpenCLIP ViT-H-14 fp16 | 1.3 GB 权重 + 1–2 GB 激活 | `batch_size=8` |
| L1 YOLO11n | 0（CPU）或 0.5–1 GB（GPU） | 默认建议 CPU |
| **合计** | **~19–23 GB** | 余量 1–5 GB，取决于峰值 |

主要 OOM 风险：

1. vLLM 启动/图捕获峰值：正式基线先用 `GPU_MEM_UTIL=0.65`（与六组一致）。
   若 smoke OOM，再设 `VLN_VLLM_ENFORCE_EAGER=1`、`GPU_MEM_UTIL=0.55`、
   仍失败则显式 `VLN_KV_CACHE_GIB=2`、降到 0.50，并记录偏离。
2. Habitat 渲染与 vLLM 同卡：长 Episode 或高分辨率渲染会增加峰值；不要在同一
   4090 D 上并行多个 client。
3. CLIP 激活峰值：Mask 很多时 `batch_size=8` 可能偏高；降到 4 或 2。
4. OpenCLIP fp16 与 CPU 设备不兼容：CPU 上必须用 `fp32`；GPU 上用 `fp16`。
5. YOLO L1/L3 同时驻留 GPU：本次默认 L1 在 CPU；若改回 GPU，先减少 CLIP batch。
6. 显存碎片：在同一进程反复加载/卸载不可行；一次启动完成一个 run，失败后重启进程。

建议启动顺序：单 Episode smoke → 检查 `nvidia-smi` 峰值与 `l3_query_stats`
→ 再跑 88 Episode。若 smoke 在 vLLM 启动就 OOM，先调 `GPU_MEM_UTIL`，
不要先动 Prompt/模型参数。

## 7. Query 触发率与整句/短语差异

每步 `prompt_diagnostics.jsonl` 和 `memory_diagnostics.jsonl` 的
`spatial_l3.l3_query`/`l3_query_stats` 记录：

- `trigger`: `built` / `no_candidates` / `budget_excluded` /
  `reactivation_suppressed`；
- `query_executed`、`query_source`（`instruction` 或 `explicit`）、
  `last_query_step`、`steps_since_last_query`；
- `candidates[]`: 原始余弦 `similarity`、`rank`、`object_id`、`evidence_ids`、
  `decision`（`matched` / `ranked_out_by_top_k` / `below_positive_cosine` /
  `below_min_similarity`）；
- `query_trigger_rate = queries_attempted / l1_empty_steps`、
  `match_rate_of_queries`、`queries_injected`、`injection_rate_of_l1_empty`；
- `skipped_reasons` 计数，包括 `empty_map`、`no_object_features`、
  `map_unchanged_since_last_query`、`no_current_pose`、`l1_satisfied`。

自动 Controller 默认把 **完整 R2R 指令** 作为 CLIP 文本查询（官方查询接口接受
调用者文本，不做解析）。完整指令和短对象短语的差异必须用真实 CLIP 在 GPU 上
检查；CPU Mock 只验证文本原样传递和接口行为，不证明真实语义排序。需要短查询时
使用 `MemoryManager.query_l3_objects("the window", K)` 或
`L3QueryController.maybe_query(..., query_text="the window")`；没有新增自动
短语解析器。

如果长期没有查询，查看 `l3_query_stats`：`l1_built` 高说明 L1 正常；
`map_unchanged_since_last_query` 高说明没有新视觉证据；`no_object_features`
说明 CLIP 特征提取失败/未运行；`no_current_pose` 说明 RGB-D 缺失。不要通过
放宽正余弦阈值来“制造”查询；阈值保持 `0.0` 符号边界。
