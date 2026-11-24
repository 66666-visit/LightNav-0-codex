# Geometry-only L3 + Original On-demand Query（`geometry_on_demand`）

> 状态：本地实现、CPU 测试、Git Diff/Hash 审查完成。未启动 GPU、未运行 88 Episode、
> 未下载权重、未 commit / push。默认关闭；原 L1 Prompt、六组消融与历史结果不变。
> Adaptive L3 代码保留但未启用、未继续开发。

## 1. 源码来源（只读）

| 环节 | 版本 | 文件/函数 | 本轮用法 |
| --- | --- | --- | --- |
| 原始几何建图 | `ed5b9b3` | `conceptgraphs_lite.py::ObjectMap.observe()`、`backproject_world`、`_n_ratio`、Greedy Assignment | 原样调用，未修改 |
| 几何候选检索 | `59d09ba` | `spatial_graph_adapter.py::SpatialGraphAdapter.build()` | 原样调用，未重写 |
| CLIP 检索 | `660320f` | `l3_clip_query.py::L3ClipQuery` | **本实验不实例化、不调用** |
| On-demand 触发 | `660320f` | `l3_on_demand.py::L3QueryController.maybe_query()` 的 L1-empty 门控、输入校验、`(map signature, query)` 去重 | 提取为共享 `prepare_query()`；CLIP 路径默认行为不变 |
| Prompt 注入 | `660320f` | `prompt_assembly.py` 的 `l3_on_demand` 分支、`inject_memory_context`、`audit_prompt_injection` | 注入位置/预算/回退保持一致 |

历史几何检索的源码位置：`59d09ba:src/lightnav/memory/spatial_graph_adapter.py`
第 168–315 行（`SpatialGraphAdapter.build`）。当前仓库该文件与 `HEAD:660320f`
及 `59d09ba` 的哈希一致，未修改。

`SpatialGraphAdapter.build()` 的选择规则（原样）：

1. 只保留 `geometry_status == "observed"` 且全部 Evidence `observed`、
   `confidence >= min_injection_confidence` 的融合节点。
2. 对象标签与指令必须有词项交集（`_terms(label) & _terms(instruction)`）；
   没有 CLIP/语义向量。
3. 必须有严格早于当前 step 的真实 `observed_from` 历史 Evidence，且
   viewpoint 的 Scene/Episode/rollout/step 与 Evidence 一致。
4. 按 `(distance, -observation_count, object_id)` 排序，最多
   `max_landmarks` 条，逐条按 `prompt_max_chars` 整行纳入。
5. 行文本只含 label、世界坐标、直线距离/左正方位角、观测次数、source、
   `passage unknown`；没有相似度、没有可通行断言。

## 2. 数据流

```text
原始 L1 LocalGraphFusion（不变）
        │ built / no_candidates / budget_excluded / reactivation_suppressed
        ▼
L3QueryController.prepare_query()          ← 660320f 原始 on-demand 门控
  - 仅 no_candidates / budget_excluded 通过
  - 输入校验、空 query、空 map
  - 几何可用性：至少一个 node 有历史 Evidence
  - 同一 map signature + 同一 instruction → 去重
        │ 不通过 ───────────────► 原 L1 Prompt 原样
        ▼
SpatialGraphAdapter.build()                ← 59d09ba 原始几何候选检索
        │ 无候选 / 预算排除 ─────► 原 L1 Prompt 原样
        ▼
MemoryContext（既有 [Navigation Memory] + Spatial history 行）
        │
        ▼
既有 inject_memory_context / audit_prompt_injection / 回退
        │
        ▼
LightNav 原动作推理（每步一次，未增加前向）
```

## 3. 与 660320f CLIP On-demand 的配置差异

| 项 | 660320f CLIP On-demand | `geometry_on_demand` |
| --- | --- | --- |
| `--spatial_l3_mode` | `on_demand` | `geometry_on_demand` |
| 候选来源 | `L3ClipQuery` 余弦 Top-K | `SpatialGraphAdapter.build` 几何/历史过滤 |
| CLIP 权重 | 必需 | **不需要** |
| OpenCLIP import | 需要 `.[clip]` | **不需要** |
| `--spatial_l3_clip_*` | 必需 | **不传** |
| CLIP 编码 | 每查询一次 `encode_text` + 预存图像特征 | **零次** |
| L1 触发 | `no_candidates` / `budget_excluded` | 相同 |
| 去重 | `(CLIP map signature, query)` | `(geometry map signature, instruction)` |
| 注入位置 | `</navigation_task>` 后 | 相同 |
| 预算 | `min(600,320)` 或 token-aware | 相同 |
| 回退 | 原 L1 Prompt | 相同 |
| Manifest/Seed/Steps/YOLO/Prompt 预算/模型 | 见 88 启动器 | 相同 |

**唯一必要的兼容点**：

- 原 On-demand Controller 的 CLIP 前置条件是“至少一个节点有 CLIP 特征”；几何
  路径没有 CLIP 特征，因此替换为“至少一个节点有严格早于当前 step 的 observed
  Evidence”。检查位置与原 CLIP 前置条件相同（在空 map 之后、去重之前）。
- 原 `map_signature()` 只统计 CLIP 特征观测；几何路径使用
  `geometry_map_signature()`（object_id、observation_count、Evidence IDs、
  geometry_status、点数），否则空 CLIP 签名会让第二次查询永远被误判为重复。
- 原 On-demand Formatter 输出 `L3 CLIP cos ...`；几何路径使用
  `SpatialGraphAdapter` 既有格式 `Spatial history: ...`，不包含相似度字段。
  `l3_query` 诊断的 `retrieval` 为 `geometry_spatial_graph_adapter`，不写
  `top_k` / `min_similarity` / `similarity_semantics`。

以上三处只替换“检索来源/可用性/可用签名”，不改变 L1-empty 触发条件、去重语义、
注入位置、预算和回退路径。

## 4. 统计与诊断

每步 `prompt_diagnostics.jsonl` 的 `adaptive`/`prompt_context` 中：

- `spatial_l3.l3_query.retrieval`：固定 `geometry_spatial_graph_adapter`
- `query_executed`、`query`、`trigger`、`skipped_reason`
- `candidate_count`：实际格式化的几何事实行数
- `injected_count`：进入 Prompt 的行数
- `rejected_object_count`：`objects_total - injected_count`（粗粒度，含标签/
  历史/置信度/预算淘汰；adapter 不暴露逐对象拒绝原因，未伪造）
- `object_ids`、`evidence_ids`：真实 canonical ID
- `objects_total`、`geometry_context_chars`、`query_ms`
- 累计 `l3_query_stats`：`queries_attempted`、`queries_with_match`、
  `queries_without_match`、`prompt_budget_excluded`、`queries_injected`、
  `query_trigger_rate`、`match_rate_of_queries`

## 5. CPU 测试

```bash
cd /home/lilixing/projects/LightNav-0-codex-conceptgraphs
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/memory/test_geometry_on_demand.py \
  tests/habitat/test_geometry_on_demand_policy.py \
  tests/habitat/test_geometry_on_demand_cli.py \
  tests/memory/test_l3_on_demand.py \
  tests/habitat/test_l3_on_demand_policy.py
```

本轮结果：`49 passed`（geometry 20 + CLIP on-demand 29）。全仓 CPU 回归
（`pytest -q -m "not gpu"`）：`1232 passed, 1 failed, 13 skipped, 99 subtests passed`。
唯一失败 `tests/memory/test_rgm_place.py::test_initial_reference_and_global_ambiguity`
是既有字符串断言，已在干净 `660320f` worktree 复现，与本轮无关。

## 6. 唯一正式 88 Episode 启动器

脚本：[`scripts/run_geometry_on_demand_88.sh`](../../scripts/run_geometry_on_demand_88.sh)
（真实 AutoDL 路径来自 `docs/research/L3_ON_DEMAND_AUTODL.md`）。

```bash
cd /root/autodl-tmp/LightNav-0-codex-conceptgraphs
bash scripts/run_geometry_on_demand_88.sh
```

等价手写命令：

```bash
cd /root/autodl-tmp/LightNav-0-codex-conceptgraphs
export MODEL_PATH=/root/autodl-tmp/LightNav-0
export YOLO_MODEL_PATH=/root/autodl-tmp/weights/yolo11n.pt
export SEG_MODEL_PATH=/root/autodl-tmp/weights/yolo11n-seg.pt
export EPISODE_MANIFEST_PATH=/root/autodl-tmp/manifests/88_episode_manifest.json
export PILOT_MANIFEST_PATH=/root/autodl-tmp/manifests/88_episode_pilot.json
export CLIENT_PYTHON=$PWD/.venv311/bin/python
export HABITAT_CONDA_ENV=habitat
export HABITAT_CONFIG=$PWD/habitat_server/configs/vlnce_r2r.yaml
export DATA_PATH=/root/autodl-tmp/data
export SCENES_DIR=/root/autodl-tmp/scenes
export GPU_IDS=0 GPU_MEM_UTIL=0.65 EPISODES=-1 MAX_STEPS=500
export STRICT_FULL_EVAL=1 EXPECTED_EPISODES=88 NO_FORCE_STOP=1 SEED=42
export OUTPUT_ROOT=/root/autodl-tmp/output/geometry_on_demand_l1_original_88_$(date +%Y%m%d_%H%M%S)
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
  --spatial_l3_mode geometry_on_demand \
  --spatial_l3_seg_model_path $SEG_MODEL_PATH --spatial_l3_seg_device cpu \
  --spatial_l3_seg_confidence 0.25 --spatial_l3_seg_every_n_steps 5 \
  --spatial_l3_evidence_min_iou 0.5 --spatial_l3_evidence_ambiguity_margin 0.1 \
  --spatial_l3_min_injection_confidence 0.5 \
  --prompt_measurement_enabled"
bash scripts/eval_habitat.sh
```

与 660320f CLIP 命令逐项相同：模型权重、两份 Manifest、`SEED=42`、
`MAX_STEPS=500`、`STRICT_FULL_EVAL=1`、`EXPECTED_EPISODES=88`、
`GPU_MEM_UTIL=0.65`、YOLO 检测频率 5、L3 分割频率 5、Prompt 预算 320/600、
L1 landmark 2、reactivation 8。唯一运行差异是 `--spatial_l3_mode` 和删除
全部 `--spatial_l3_clip_*`。

## 7. 限制

- 几何标签匹配是词项交集，不是语义检索；同义类名可能查不到。这是 59d09ba
  既有行为，未增加同义词表。
- `SpatialGraphAdapter` 不返回逐对象拒绝原因；`rejected_object_count` 是
  总对象数减实际注入数，不声称具体原因。
- 与 CLIP 路径一样，当前 step 被重新观测的节点不是“历史”，会被 adapter 排除；
  分割频率 5 时，非分割步才有历史候选。
- 没有房间层级、属性、状态、可通行证明；行内只有直线距离/方位与
  `passage unknown`。
- 本轮未在 GPU/88 Episode 上验证导航效果。
