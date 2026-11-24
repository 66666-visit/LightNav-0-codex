# Adaptive L3 Query（实验条件 `adaptive_l3_query`）

> 状态：本轮只做本地实现、CPU 测试和独立审查。未启动 GPU、未运行 88 Episode、
> 未下载权重、未 commit / push。默认关闭；`legacy` 与
> `l1_original_only` 的原始 L1 Prompt 在不查询时逐字节不变。

## 1. 唯一上游与可核对源码

Choi et al., *Memory–Environment Dual-Grounding for Embodied Agents:
Active Information Seeking Beyond Passive Observation*, Findings of EMNLP 2026,
DOI `10.18653/2026.findings-emnlp.16`。
本地参考 checkout：`references/SceneQueryPlanner`（commit `57ea57f`）。

| 上游文件 | 行 | 机制 | LightNav 对应 |
| --- | --- | --- | --- |
| `core/planner/procthor/reactstrq.py` | 50–77 | `Think` / `Act` / `Query` 三分支 | 只保留 Query 分支；Act 仍由原 policy 负责 |
| 同上 | 76–100 | `_parse_interface_call_string` → `_execute_interface_call_from_parsed`；成功写 `Info:`，失败写 `Error:` | `AdaptiveL3QueryRunner` 的 Query 阶段；失败一律弃权 |
| `core/retriever/procthor_retriever.py` | 63–90 | 正则 + `ast.literal_eval` 解析结构化调用 | 不解析自由文本；Controller 只输出 `NO_QUERY` / `QUERY_L3` / `USE` / `ABSTAIN` |
| 同上 | 92–160 | 白名单方法分发；异常写入 `error` | `validate_l3_fact` 只读 ObjectMap；异常 fail-closed |
| `core/retriever/interface_procthor.py` | 43–105 | `find_objects` 按 label/attrs 过滤节点 | `L3ClipQuery` 的语义检索（不改余弦算法） |
| 同上 | 107–182 | `read_node` 返回既有节点字段 | 只读 `ObjectNode` 中心、bbox、观测数、Evidence |
| 同上 | 287–304 | `get_edges_for_node` 返回真实图边 | 只校验 `observed_from` 证据边；不声称 `traversed_to` 或可通行 |
| `core/llm_agent/reactstrq_agent.py` | 64–126 | LLM 输出 `Think:` / `Act:` / `Query:` | 本地模型输出两阶段单行决策；不执行动作 |
| `core/llm_agent/procthor_reactstrq_agent.py` | 23–36 | 系统提示 + 示例 + 任务观测 | 指令 + L1 卡 + 当前视频；无示例库 |
| `resource/sys_prompt/procthor/reactstrq.txt` | 1–11 | Query 白名单与可用属性 | 系统提示只描述只读查询与证据规则 |

## 2. LightNav 适配

```text
当前指令 + 当前视频 + L1 Original 卡（只读）
        │
        ▼
LocalLightNavQueryController.decide()        ← 已有本地 LightNav engine
        │  NO_QUERY ───────────────► 原 L1 Prompt 原样返回
        │  QUERY_L3: <phrase>
        ▼
L3ClipQuery（既有实现，余弦 Top-K 不改）
        │  无有效候选 ─────────────► ABSTAIN
        ▼
LocalLightNavQueryController.select() over C1..Cn
        │  ABSTAIN ────────────────► 原 L1 Prompt 原样返回
        │  USE C1 ...
        ▼
validate_l3_fact()：ObjectNode + 全部 Evidence + observed_from
        │  任一失败 ───────────────► 该对象弃权
        ▼
format_verified_facts()：确定性模板，只渲染真实字段
        │
        ▼
L1 卡 + verified L3 行（L1 前缀逐字节保留）
        │
        ▼
既有 inject_memory_context / audit_prompt_injection / 动作解码（不改）
```

### 与论文方法的区别

1. **没有 `Act` 分支**：机器人动作完全由冻结的 LightNav policy 与动作解码拥有；
   上游的 `Think/Act/Query` 循环只保留 Query 半部。
2. **没有外部 LLM/API**：Controller 复用已加载的本地 `VLNInferenceEngine` 的
   `generate(sample)`，不新建权重、不重复加载、不访问网络。上游用
   guidance/Transformers 或 OpenAI。
3. **查询接口是 CLIP 语义 Top-K，不是 ProcTHOR 场景图 DSL**：LightNav 只有
   ConceptGraphs-Lite 的 ObjectMap，没有 room/asset 层级、属性或 states；
   因此不实现 `find_objects(attrs)`、`get_child_node_names` 的子图语义。
4. **候选选择用别名 `C1..Cn`**：真实 Object ID 是 canonical JSON 数组，
   让动作微调模型逐字复现不现实。别名只用于模型输出；runner 在本地映射回
   真实 Object ID，metadata 和 diagnostics 只记录真实 ID。
5. **两阶段决策而不是多轮 ReAct**：`decide` 一次、`select` 一次；没有示例库、
   token 预算规划器或冷却步数。
6. **验证器比上游更严格**：融合节点的中心/bbox 使用全部观测，因此只要有一个
   Evidence 缺失、类别不一致、`geometry_status != observed`、置信度不足、
   属于当前 step，或没有唯一的 `observed_from` 边，整个对象弃权。

### 不变量

- YOLO11n-seg、RGB-D、K/Pose、点云、`n_ratio`、Greedy Assignment、
  `ObjectMap.observe()`、ObjectNode 几何写入、Evidence/图边生成、CLIP 图像特征
  融合算法、原始 L1 LocalGraphFusion、SlowFast、动作解码：均未修改。
- 唯一触碰的 CLIP 相关代码是 `_attach_clip_features` 的运行门控：原来只在
  `spatial_l3.mode == "on_demand"` 时提取特征；现在 `adaptive_l3_query.enabled`
  时也走同一函数、同一 `encode_image`、同一 `attach_clip_features`，算法和参数未变。
- 不查询时 L1 Prompt 与基线逐字节一致（policy tests 断言完整字符串相等）。
- 没有手工相似度权重、综合评分、固定冷却步数或每步强制 CLIP 查询；Controller
  决定是否查询，runner 只对“同一 query + 同一 map 签名”的完全重复调用去重。

## 3. 开关、配置与诊断

```bash
--adaptive_l3_query
--adaptive_l3_query_max_new_tokens 24
--adaptive_l3_query_max_query_chars 256
--adaptive_l3_query_min_evidence_confidence 0.5
```

约束：

- `--memory_enabled --memory_strategy hierarchical --memory_layers short,mid,long`
- `--local_graph_enabled --memory_extractor yolo11`
- `--spatial_l3_mode shadow`（只建图，不注入空间候选）
- `--spatial_l3_seg_model_path` 与 `--spatial_l3_clip_model_path` 均指向本地文件
- `--memory_prompt_mode legacy` 或 `l1_original_only`
- `VLN_EVAL_TRAJ_TOP1=1` 被拒绝（该限制把解码 mask 到轨迹 token）

诊断：

- legacy：`last_memory_diagnostic["adaptive_l3"]`
- prompt-mode：`last_memory_diagnostic["prompt_context"]["adaptive_l3"]`
- 字段：`decisions`（raw 输出、latency）、`query`、`query_executed`、
  `candidate_count`、`selected_object_ids`、`validated_object_ids`、
  `validated_evidence_ids`、`rejected`、`injected`、`prompt_verified`、
  `skipped_reason`。

## 4. CPU 测试命令与结果

```bash
cd LightNav-0-codex-conceptgraphs
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/memory/test_adaptive_l3_query.py \
  tests/habitat/test_adaptive_l3_query_policy.py \
  tests/habitat/test_adaptive_l3_query_cli.py
```

本轮实测：`36 passed in 1.03s`。全仓 CPU 回归（`pytest -q -m "not gpu"`）：
`1212 passed, 1 failed, 13 skipped, 99 subtests passed in 44.21s`。唯一失败
`tests/memory/test_rgm_place.py::test_initial_reference_and_global_ambiguity`
是既有字符串断言（`ambiguous_global_candidates` vs
`unresolved_competing_candidate`），已在干净 `660320f` 的临时 worktree 中复现，
与本轮改动无关。覆盖十个必须契约：

1. L1 充分 → NO_QUERY，CLIP text encoder 调用 0 次。
2. L1 有候选 → Controller 仍可 QUERY_L3，注入经过验证的事实。
3. CLIP 无有效候选 → ABSTAIN，Prompt 与基线一致。
4. 多候选 → 只保留证据通过验证的对象。
5. 同类别不同 Object ID 不合并。
6. 缺 Evidence 的对象被拒绝，不伪装成历史事实。
7. CLIP/Controller 失败 → L1 Prompt 逐字节不变。
8. 不查询 → 与 L1 Original 基线 Prompt 完整字符串相等（legacy 与
   `l1_original_only` 各一条）。
9. 注入时 metadata/`evidence_refs` 保留真实 Object ID 与 Evidence ID。
10. Episode reset 清空 Controller 调用、`last_diagnostic` 和地图状态。

## 5. AutoDL 下一步命令草案（本轮未执行）

前置：AutoDL 上已有本地 LightNav checkpoint、`yolo11n.pt`、
`yolo11n-seg.pt`、OpenCLIP `ViT-H-14` 本地权重；已装 `.[clip]`；Habitat 仍用
独立 Python 3.9 环境。先跑 1 Episode 的 smoke，不跑 88 Episode。

```bash
cd /path/to/LightNav-0-codex-conceptgraphs

# 1) 先确认 Controller 的本地决策可解析（GPU 上真实模型）
export VLN_EVAL_TRAJ_TOP1=0
export MODEL_PATH=/path/to/LightNav-0
OUTPUT_ROOT=/path/to/output/adaptive_l3_smoke_ep1 \
EPISODES=1 STRICT_FULL_EVAL=0 SEED=42 \
GPU_IDS=0 \
CLIENT_ARGS="--memory_enabled \
  --memory_extractor yolo11 --memory_extractor_model_path /path/to/yolo11n.pt \
  --memory_strategy hierarchical --memory_layers short,mid,long \
  --local_graph_enabled --local_graph_prompt_max_chars 320 \
  --memory_max_context_chars 600 \
  --spatial_l3_mode shadow \
  --spatial_l3_seg_model_path /path/to/yolo11n-seg.pt \
  --spatial_l3_seg_every_n_steps 2 \
  --spatial_l3_clip_model_path /path/to/open_clip_ViT-H-14.bin \
  --spatial_l3_clip_model_name ViT-H-14 \
  --spatial_l3_clip_device cuda --spatial_l3_clip_precision fp16 \
  --spatial_l3_clip_top_k 3 \
  --memory_prompt_mode l1_original_only \
  --adaptive_l3_query \
  --trace_steps" \
  bash scripts/eval_habitat.sh
```

严格对照：复制同一命令，仅把 `--adaptive_l3_query` 去掉，输出到另一个全新目录；
两者其余参数完全一致。

`--spatial_l3_seg_every_n_steps 2` 是刻意的：validator 要求该对象的**全部**
Evidence 都严格早于当前 step；若每步都重新分割，节点在本步被更新，融合中心/bbox
包含当前观测，整节点会被拒绝（“当前观测不是历史”）。每 2 步分割可让奇数步查询到
上一步的历史节点。这不是冷却步数，也不改变几何。

检查顺序：

1. `shard_0/scene_query_protocol.json` 不存在是预期；本轮没有新增该文件。
   `prompt_protocol.json` 的 `adaptive_l3_query` 块（仅 adaptive 运行）记录
   controller/query/validator/formatter 契约。
2. `prompt_diagnostics.jsonl` 每步 `adaptive_l3`：先看
   `decisions[0].decision` 是否为 `NO_QUERY` / `QUERY_L3`；
   若长期是 `unparseable_decision`，说明动作微调模型不遵循文本决策契约，
   应停止扩大实验并报告，而不是加规则强行查询。
3. `query_executed=true` 的步骤看 `candidate_count`、`selected_object_ids`、
   `validated_object_ids`、`rejected`、`injected`。
4. `actual_injection` / `prompt_verified` 必须与 `injected` 一致；
   `prompt_hash_before` 与对照完全一致才说明 L1 未被改动。
5. 先看 1 Episode 的 `success`、`spl` 与步数，不做统计结论。

## 6. 已知限制与阻塞风险

- **Controller 是动作微调模型**：LightNav-0 的 checkpoint 以轨迹 token 为训练
  输出契约。本实现通过 `engine.generate(sample)` 走同一条本地推理路径，并在
  解析失败时 NO_QUERY/ABSTAIN；它没有新增文本 LLM，也没有修改解码。
  模型是否能稳定输出 `NO_QUERY` / `QUERY_L3` 尚未在 GPU 上验证。
  如果 smoke 中 parse 率接近 0，这是真实阻塞点：不能靠规则或外部 API 绕过，
  需要用户决定是否允许一个独立的本地小模型服务。
- L3 地图是 Episode-local ConceptGraphs-Lite，没有房间层级、物体状态、affordance
  或可通行证明；查询只返回直线距离/方位与 `passage unknown`。
- `--spatial_l3_mode shadow` 仍会运行 YOLO11n-seg + CLIP 图像特征提取；它们只读，
  但会增加每步开销。
- Controller 每步都会被问一次（这是“L1 有候选也可查询”的前提）；真正的 CLIP
  查询只在 `QUERY_L3` 且 map/query 未完全重复时发生。
- Controller 继承现有全局解码参数。为保证可复现，保持
  `VLN_EVAL_TEMPERATURE=0`（默认）；`VLN_EVAL_TRAJ_TOP1=1` 会被拒绝。
- Adaptive 模式需要既有 L3 CLIP 编码器（OpenCLIP `ViT-H-14`）一次性加载；
  这是 `L3ClipQuery` 的既有依赖，不是第二份 LightNav 模型。Controller 本身
  不加载任何新模型。
- 与受保护代码的唯一相邻改动：`MemoryManager._attach_clip_features` 的运行门控
  从 `mode == "on_demand"` 扩展为 `mode == "on_demand" or adaptive.enabled`；
  `clip_features.py` 的提取/融合算法和 `l3_clip_query.py` 的余弦 Top-K 均未改。
