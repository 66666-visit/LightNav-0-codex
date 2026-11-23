# L1 + L3 双层记忆 Prompt 组织与融合消融

本文件记录本次只读审计、代码改动、七种 `--memory_prompt_mode`、七组 88 Episode 消融的
运行方式与统计口径，并严格区分**代码验收**与**真实导航效果**。正式 GPU 实验尚未运行，
本文件中的测试数字只证明代码路径与 Prompt 契约，不证明 SR/SPL 变化。

- 分支：`experiment/l1-l3-prompt-ablation`
- 基准提交：`ed5b9b3`
- 历史 75% 基线：`2fb184e`
- 唯一参考论文/源码：HiMemVLN，只读检入于 `.references/HiMemVLN`
- 未修改模型权重、动作输出接口、YOLO11n / YOLO11n-seg、RGB-D 点云构建、
  ObjectMap 关联与融合规则、L1/L3 数据写入机制；未引入 L2、DART 或外部 LLM API。

---

## 1. 参考论文机制与实际借鉴

### 1.1 本地检入的 HiMemVLN 可核对范围

检入 checkout 的 `README.md` 只有两行标题；`base_il_trainer_vlm.py` 与
`base_il_trainer_llm.py` 导入的 `vlnce_baselines.common.navigator.spatialNavigator_vlm`
和 `spatialNavigator` **不在仓库中**（`git ls-tree` 已确认）。因此 Short-Term Localer、
Long-Term Globaler、VGM 和 UARF 的内部实现无法从本地源码核对；论文中关于
512 维视觉节点、余弦重访阈值、动量更新、`{PrimaryDir, FinalTarget, NavPattern}`、
CameFrom 和最近 5 步的细节只能作为**论文陈述**，不能冒充已读源码。

可以核对的只有外层调度（`base_il_trainer_vlm.py` 概略行号）：

| 位置 | 可核对行为 |
| --- | --- |
| 168–264 | 多视角 RGB-D 图像 → 方向桶；候选 waypoint 的角度/距离字典 |
| 399–405 | `Open_Nav_VLM(..., use_global_reflector=True)`；日志记录 VGM/Global Reflector |
| 426–440 | 从指令抽取 `actions` 与 `landmarks`，按 instruction 缓存 |
| 466–476 | `observe_environment_vlm` → 当前观测、`observe_dict`、图像、`spatial_descriptions`；`review_history(nav_history)` → 历史摘要 |
| 478–512 | `estimate_completion`（完成度文本）→ `move_to_next_vp_vlm`（当前观测 + 历史 + 完成度 + 候选）→ `thought_fusion` → `test_decisions_vlm` |
| 530–534 | `save_history(step, next_vp, thought, curr_observe, nav_history, spatial_desc=...)`，每 Episode 结束清空 |

### 1.2 实际借鉴的四个原则

1. **短期与长期分层**：当前视觉状态与历史空间记忆分开描述，不把历史当成当前可见。
2. **当前状态决定历史何时出现**：历史只在当前检测/视觉地点与指令相关时被召回，
   而不是无条件倾倒历史。
3. **历史压缩为少量文本事实**：用有限的观测次数、位置关系和不确定性表达历史，
   不把原始帧再次塞进文本。
4. **指令先抽取地标，再检索**：`landmarks` 先由指令得到，后续观测和历史都围绕它筛选。

### 1.3 没有照搬的部分

没有复制 Localer/Globaler 代码（本地缺失），没有引入 VLM/LLM 调用、UARF、VGM、
第二个规划器、DINO 视觉重识别（作为本实验变量）、CameFrom 图或新的学习式匹配网络。
L1/L3 的写入、ObjectMap 几何关联和动作输出接口均保持原样；本实验只改**读侧文本组织**。

---

## 2. 第一阶段：真实调用链审计

### 2.1 L1 审计结论

1. **L1 实际保存哪些字段**：每个观测步写入 `MemoryEvent`（`object_observation`、
   `landmark`、`instruction_match`、`action`/`turn`、`visual_change`、
   `possible_transition`、`place_observation` 等，`memory_writer.py:97-264`），再持久化为
   `EvidenceRecord`（`v3_schemas.py:123-203`）。字段包括 `evidence_id`、Scene/Episode/
   rollout、`step`、`event_type`、`semantic_label`、`payload`、`visual_descriptor`
   （YOLO 检测为 `None`）、`detector_name`、`raw_detector_confidence`、
   `instruction_terms`、`position`/`heading`（L1 写入器不设置，保持 `None`）、
   `source_evidence_ids`、`observation_status`、`spatial_status`、
   `traversability_status`、`support_count`、`first_seen`/`last_seen`、
   `temporal_support`、`episode_support`、`spatial_consistency`（未设置）、
   `provenance_validity`。当前步事件还带 `bbox_xyxy` 与 SceneGraph `node_id`。
2. **L1 Prompt 在哪里生成**：`memory/local_graph_fusion.py:122-208`；
   `_Candidate.line()` 在 67–74 行生成两种句子。随后
   `memory_context_builder.build_context_text`（`memory_context_builder.py:24-54`）
   做 HTML 转义和整行预算裁剪，`inject_memory_context`（57–96）插入
   `</navigation_task>` 之后。
3. **当前 L1 如何筛选历史**：当前候选只取本步事件，要求
   `_terms(label) ∩ _terms(instruction) ≠ ∅`、`raw_detector_confidence ≥ 0.5`、
   `observation_status == observed`、`provenance_validity`，同标签保留最高置信度。
   历史候选要求当前 DINO 视觉地点存在，并且该地点与 landmark 之间已有
   `inferred_relation`/`spatial_relation` 一跳边；只取当前步之前，`episode_local`
   不跨 Episode；每个标签保留最新一条。
4. **当前 L1 Prompt 如何进入模型输入**：`policy.py:455-659` 构建原始
   `build_vln_traj_sample`，只替换 human 文本中 `</navigation_task>` 后的位置；
   `audit_prompt_injection` 验证原文可逐字恢复、非 human 消息和 `video_segments`
   未被改动；`engine.generate(sample)` 后通过 `_rgm_trace` 和
   `engine.last_input_diagnostic` 检查 memory 文本是否真的进入最终 token。
5. **是否重复描述当前 RGB**：**部分重复**。当前行 `Current RGB: instruction landmark X
   in the center of the current view (detected; passage unknown).` 重复了模型已经直接
   看到的“X 在画面中心”。它额外提供的只有检测标签、指令匹配关系和“通行未知”；
   置信度和时效性没有进入旧 Prompt。历史行不重复当前画面，但只有在 DINO 地点匹配时出现。
6. **信息预算与截断**：Local Graph 路径预算为
   `min(local_graph.prompt_max_chars=320, memory.max_context_chars=600) = 320` 字符
   （`policy.py:497`）。这是**旧记忆模块的信息选择预算**，不是 Qwen3-VL 的上下文
   硬限制：Processor 的实际上限来自 `max_seq_len`（默认 8192），vLLM 后端另有
   `max_model_len`。旧路径 header 计入字符预算，只整行追加，放不下的候选整条丢弃；
   Processor 长度拒绝时丢弃最长 item 重试，全部失败回退原始 sample。

> 说明：模型真正“看见”的短期记忆是 RGB 视频段（`build_vln_traj_sample` +
> `prompts.py` 的 `<video>` 模板）。L1 文本卡是**检测器摘要**，不是短期视觉记忆本身；
> 本次没有改动视频段、时间戳或 Prompt 模板。

### 2.2 L3 审计结论

1. **L3 实际有哪些对象、几何、关系和 Evidence**：
   - `ObjectNode`：`object_id`、`class_name`、`center_world`、`bbox_3d`、
     `observation_count`、`evidence_ids`、`geometry_status`、`points_world`
     （`conceptgraphs_lite.py:328-337`）；
   - `ViewpointNode`：Scene/Episode/rollout/step、Agent Pose、Camera Pose（340–348）；
   - `ObservedFromEdge`：`object_id ↔ viewpoint_id + evidence_id`（351–356）；
   - `TraversedToEdge`：前后真实 Viewpoint、`transition_step`（359–364）；
   - `SpatialEvidence`：类别、分割置信度、来源、Mask 像素数、有效深度点数、
     `geometry_status`、可选 `linked_l1_evidence_id`（367–379）；
   - `InstanceMask`：类别、置信度、Mask、Evidence、可选 L1 关联（239–265）。
   L3 **没有** Place 节点、没有物体间推断关系、没有物体通行性字段。
2. **当前 L3 Prompt 在哪里生成**：`spatial_graph_adapter.py:168-315`；句子模板在
   154–162 行，header 为 `[Navigation Memory]`，与旧 L1 共用同一个注入适配器。
3. **如何从 ObjectMap 中选出候选**：所有对象先要求 `geometry_status == observed`、
   所有 `evidence` 的几何状态为 observed 且分割置信度 ≥
   `min_injection_confidence`；标签必须与指令有词项交集；必须存在当前步之前的
   有效历史 Evidence（同 Episode 或 `scene_persistent` 的旧 Episode）；Evidence 的
   Scene/rollout/step 必须与 Viewpoint 一致；可选 L1 关联必须仍能在 V3 store 中
   核验；每个对象取最新一条历史 Evidence。
4. **当前候选排序依据**：`(distance_m, -observation_count, object_id)`
   （`spatial_graph_adapter.py:279`）——**距离优先**。这正是本实验要改的点。
5. **哪些信息可以被证明，哪些仍然未知**：
   - 可证明：检测类别与原始分割分数、Mask 有效深度点、相机/Agent 位姿换算出的
     世界中心与包围盒、`observed_from`、真实 step 转移且位移超过阈值的
     `traversed_to`、Evidence 来源与 Step；
   - 未知：物体间关系、门/通道是否可通行、未来路径是否存在、直线距离是否可达、
     跨视角身份概率、Place 身份、当前是否仍可见（adapter 只用当前步之前的历史）。
6. **最终如何进入模型输入**：与 L1 共用 `inject_memory_context` 和
   `audit_prompt_injection`；旧路径使用同一个 320 字符模块预算。每个对象一行；
   `inject` 模式在无可信候选时回退 Local Graph（旧行为），本实验的 `l3_*_only`
   模式禁止该回退。

### 2.3 联合路径审计结论

1. **L1 与 L3 是否存在相同物体的重复描述**：会。L1 当前检测与 L3 历史物体可以是
   同一物理对象（例如当前画面中的 chair 与地图里前几步看到的 chair）。旧
   `inject` 路径用 L3 **替换** L1，不合并，因此不显示重复，但也丢失了两层各自的
   证据；`l1_l3_original_concat` 会把两块原文都放进来，重复更明显。
2. **是否存在当前观察和历史地图冲突**：存在潜在冲突，例如 L1 当前检测到某物体，
   而 L3 同一物体几何未验证；或 L3 只保留历史位置而当前已不可见。旧代码没有冲突
   处理：有可信 L3 候选就替换 L1，否则回退 L1，都是“整块二选一”，不会同时保留
   两个来源和不确定性。
3. **是否存在 L1 和 L3 分别描述同一导航目标**：会。两者都用指令词项过滤，同一个
   指令地标可能在 L1 当前行和 L3 历史行各出现一次；旧路径靠替换避免文本重复，
   但没有显式实体对齐。
4. **当前两份 Prompt 的实际拼接顺序**：**旧路径不拼接**。只注入一个
   `MemoryContext`（L3 或 L1）到 human 文本的 `</navigation_task>` 后面（若后面是
   `". "` 则保留该句号），位于任务句与原 Prompt 其余部分之间。
   `l1_l3_original_concat` 才按 L1 原文 → L3 原顺序拼接。
5. **是否存在不同预算导致的信息截断**：L1 与 L3 各自按同一个 320 字符预算独立裁剪；
   旧路径不会同时注入，所以没有“双预算”。`concat` 若简单叠加会超出 320，必须丢弃
   尾部整行；`joint` 在 320 内统一选条目。

---

## 3. L1 原始数据结构与 Prompt

旧 L1 卡（`LocalGraphFusion`）只产生两类句子：

```text
[Navigation Memory]
Current RGB: instruction landmark <label> in the <left|center|right> of the current view (detected; passage unknown).
Local history: <label> was seen near a visually similar area (possible revisit; current direction and passage unknown).
```

- 最多 `max_landmarks=2` 条；当前候选优先，保留一条历史锚点；同一签名在
  `reactivation_steps=8` 内不重复注入；
- 只写标签、画面扇区、可能重访；不写位置、置信度、观测次数、时效；
- 历史依赖 DINO 地点匹配，没有“当前步之前的直接检测历史”这种更简单的证据。

## 4. L3 原始数据结构与 Prompt

旧 L3 卡（`SpatialGraphAdapter`）：

```text
[Navigation Memory]
Spatial history: <label> observed at world (<x>,<y>,<z>) m; now <d> m straight-line, bearing <b> deg (left +); <n> observations; source YOLO11n-seg; passage unknown.
```

- 排序按直线距离优先，然后观测次数、对象 ID；
- 写世界坐标、直线距离、方位、观测次数和“通行未知”；
- 历史 Evidence 只取当前步之前；
- 没有区分“观察关系 / 空间关系 / 通行关系”，也没有显式统计哪些对象几何未验证。

## 5. 两层联合融合设计

新增四个只读模块，均不写记忆：

### 5.1 实体对齐（`memory/joint_prompt.py`）

只接受两类可信身份：

1. L3 `SpatialEvidence.linked_l1_evidence_id` 与 L1 Evidence 精确相等，且该链接在
   读取时仍满足：同 Scene/Episode/rollout/Step、L3 类别与 L1 标签一致、
   L1 `observation_status == observed`；
2. 没有链接时**不合并**。同标签只标记 `uncertain_same_label`，两行都保留。

链接冲突（同一 Evidence 但标签不同）标记 `identity_conflict`，两来源都保留、不合并。
对齐统计输出 `attempted / confirmed / uncertain / unmatched / success_rate`。

### 5.2 信息去重

- L1 内部：同一标签当前多次检测合并为一条，历史重复观测折叠为 `sightings`；
- L3 内部：每个 ObjectNode 一条；
- 联合：已确认对齐的同一对象合并为一条，metadata 同时保留
  `l1_evidence_ids`、`l3_evidence_ids` 与两者的 provenance；
- 只有“同一行文本 + 同一组 Evidence”才计为重复；语义相近但 Evidence 不同的两个
  对象不会被删除。

### 5.3 冲突处理

保留冲突类型与两个来源，不静默覆盖：

| 类型 | 条件 | Prompt 表达 |
| --- | --- | --- |
| `identity_mismatch` | L3 关联的 L1 Evidence 标签与 L3 类别不同 | `sources kept, not merged` |
| `geometry_unverified` | L1 当前检测与 L3 同标签对象几何未验证 | `position unknown, both sources kept` |

L1 当前事实永远作为“当前视觉状态”的最高优先来源；L3 只提供历史空间背景；
历史存在不等于当前可见，所以 L3 行显式写 `map ... history, not current visibility`
和 `passage unknown`。

### 5.4 统一输出

```text
[Navigation Context - L1+L3]
Step <n>; RGB is the current view; map lines are history.
<label>: now visible (L1 0.82, 2x); map 2 views, last step 7, 2.4m line +15deg (not a path); passage unknown.
<label>: map only ...
Conflict: ...
Uncertain: ...
Unknown: object passability; the map proves no path.
```

条目按“已确认联合 → 当前 L1 → 已验证 L3 → L1 历史 → 未验证 L3”排序；
header/state/冲突/不确定说明在 320 字符内按整行择优；被预算丢弃的条目计入
`omitted_items`，不会截断半行。

### 5.5 L1 组织（`memory/l1_prompt_organizer.py`）

- 当前事实：本步、指令相关、置信度达标、provenance 有效；按标签去重；写
  画面扇区、置信度、累计观测次数、首末 Step；
- 历史事实：同 Episode 当前步之前（`scene_persistent` 才允许旧 Episode），
  聚合观测次数并标注“not confirmed visible now”；
- 不再描述 RGB 像素内容；即使当前步没有指令相关检测，也输出状态卡明确
  “没有指令相关地标”（旧 L1 在没有候选时不注入文本，这一点是本模式的明确差异）；
- 指令相关性分数 = 词项覆盖率 + 精确短语加成；当前优先、分数高优先、置信度高优先。

### 5.6 L3 组织（`memory/l3_prompt_organizer.py`）

- 只读 ObjectMap 与 L3 自己的 Evidence；不新增 Place/Edge/Evidence；
- 相关性优先于距离：排序为 `验证状态 → 指令相关度 → 最近观测 → 观测次数 → 距离 →
  标签/对象 ID`，距离只作最后的并列打破；
- 关系显式分层：`observed`（viewpoints）、`spatial`（当前直线距离/方位，标注
  `not a path`）、`traversal`（真实 `traversed_to`）与 `unknown`（物体通行性）；
- 未验证几何/置信度对象单独出现，写 `map evidence unverified; position and passage unknown`；
- 每个条目的 Evidence、类别、观测次数、来源和几何状态都写入 `LayerMemoryItem.metadata`。

---

## 6. 七种模式与默认行为

> 2026-11 更新：本轮新增的 `--spatial_l3_mode on_demand` 是**独立单臂**，不属于
> 下表六组消融；它可以与 `l1_original_only` 组合，保留该组 L1 Original 原文与
> token 预算，只在 L1 返回空候选时追加 L3 CLIP 证据。原有六组代码/启动器/结果
> 不变。运行方式见 [L3_ON_DEMAND_AUTODL.md](L3_ON_DEMAND_AUTODL.md)。

| 模式 | 实际注入 | L3 是否运行 | L1 Prompt 构建器 |
| --- | --- | --- | --- |
| `legacy`（默认） | 旧路径：L3 可信候选，否则旧 Local Graph；不拼接、不改预算 | 按配置 | `LocalGraphFusion` |
| `l1_original_only` | 旧 Local Graph 原文 | **否**（runner 把 L3 sidecar 设为 off） | `LocalGraphFusion` |
| `l1_organized_only` | 组织后的 L1 卡 | **否** | `L1PromptOrganizer` |
| `l3_original_only` | 旧 `Spatial history` 原文；无候选则空 | 是 | 不调用 |
| `l3_organized_only` | 组织后的 L3 卡；无候选则空，不回退 L1 | 是 | 不调用 |
| `l1_l3_original_concat` | 旧 L1 原文 + 旧 L3 原文，按模型派生 token 预算整行截断 | 是 | `LocalGraphFusion` |
| `l1_l3_joint` | L1 + L3 组织、对齐、去重、冲突保留后的联合卡 | 是 | `L1PromptOrganizer` + Joint |
| `legacy`（第七组） | 原 Legacy 路径；只增加测量产物，不改 Prompt | 按原配置 | `LocalGraphFusion` / 原 L3 |

约束：

- 默认 `legacy`；`legacy` 的 Prompt 文本、注入位置、原 320 字符模块预算、确认/撤回
  逻辑和执行路径都不改；`prompt_context` 只出现在实验模式。
- 所有非 legacy 模式要求 `--memory_enabled --memory_strategy hierarchical
  --memory_layers short,mid,long --local_graph_enabled` 和显式 `--output_dir`；
  L3 模式要求 `--spatial_l3_mode inject` 与独立分割权重。
- **预算策略**：`legacy` 保持原 320 字符模块预算；六个实验组统一使用逐 Step
  模型派生 token 预算（上下文上限 − 基座 Prompt token − 生成长度 − 1 token 边界
  余量），整行选择。无法测量时回退原 320 字符并记录具体原因，不猜测更大预算。
- 七组实验只改 `--memory_prompt_mode`（legacy 另加测量开关）；runner 通过把
  L1-only 组的 L3 sidecar 设为 `off` 保证不读取 L3，通过 plan 保证 L3-only 组不调用
  Local Graph 构建器。

### 6.1 320 字符检查结论与模型派生预算

`320` 来自 `min(memory.max_context_chars=600, local_graph.prompt_max_chars=320)`，
是 Local Graph/L3 旧 Prompt 的**字符信息选择预算**，不是模型上下文硬限制：

- Processor 的 `model_max_length` 来自 checkpoint `eval_config.json` 的
  `common.max_seq_len`（默认 8192，`data_processor.py:237,496`）；
- vLLM 后端另有 `max_model_len=max(2048, num_history_frames*24+1024)`
  （`vllm_utils.py:390`），例如 64 帧为 2560，可能小于 8192；
- HF 后端还有模型自身的 `max_position_embeddings`。

因此六个实验组不再把 320 当作固定上限。每步使用：

```text
context_limit = min(bundle.max_seq_len,
                    processor.model_max_length,
                    backend max_model_len / hf max_position_embeddings)
base_tokens   = data_processor.process_sample(base_sample,
                    add_generation_prompt=True) 的 prompt token 数
reserve       = engine.max_new_tokens
available     = context_limit - base_tokens - reserve - 1
```

`+1` 是插入换行的单 token 边界余量，记录在
`safety_margin_tokens`，不是字符预算。选择时由 Processor 的 tokenizer 对
“整行候选块”计数，只保留能完整放入 `available` 的连续前缀；随后再用
`tokenizer.encode(注入后 human) − tokenizer.encode(原始 human)` 做一次精确
token 差核验，超出则丢弃最低优先级条目重试。

`prompt_protocol.json` 记录策略和来源；`prompt_diagnostics.jsonl` 每步记录
`context_limit_tokens`、`base_prompt_tokens`、`generation_reserve_tokens`、
`available_memory_tokens`、`memory_block_tokens`、`memory_tokens_injected`、
`token_overflow_drops`、`budget_source`。无法获得 Processor/Tokenizer、上下文
上限或生成长度时，回退原 320 字符并记录
`token_budget_unavailable_reason`；不会假设一个更大的预算。测量本身需要每步
一次额外的 Processor pass（不跑 ViT/生成），这是本实验明确接受的 CPU 开销。

### 6.2 Legacy 对照核对与第七组

先检查了本仓库及工作区是否存在与本次实验完全匹配的 88 Episode 历史结果：

```bash
find . -path ./.git -prune -o -path ./.references -prune \
  -type f \( -name episode_manifest.json -o -name results.jsonl \) -print
ls -ld output
```

结果：`output/` 不存在，工作区没有任何 `episode_manifest.json` /
`results.jsonl` / `per_episode.csv` 历史 88 结果，因此**没有可复用的 Legacy
结果**。核对清单（checkpoint SHA256、manifest SHA256、seed=42、有效配置、
`per_episode.csv` 88 行）无法在任何候选上完成。

结论：**必须运行第七组 legacy**。launcher 已在 preflight 中加入历史候选扫描，
把候选路径、兼容/不兼容原因写入 `provenance.json` 的
`legacy_history_candidates`；本次扫描为空，`legacy_group_required=true`。
launcher 不自动复用历史结果（避免把非完全匹配的对照带入配对比较）。第七组用
原 `legacy` 代码路径运行，仅附加 `--prompt_measurement_enabled` 以生成
`prompt_protocol.json`、`prompt_diagnostics.jsonl` 和 `per_episode.csv`，不改变
Prompt 文本、注入位置、预算或执行路径。

`l1_l3_original_concat` 不是 legacy：前者按 token 预算拼接两层原文，legacy 仍走
原 L3/L1 二选一注入路径。二者在统计器中是独立组。

### 6.3 Local Graph / Prompt 协议字段冲突与第一组恢复

第一组 88/88 推理完成后，`verify_local_graph_shards.py` 报：

```text
ValueError: root snapshot lacks one enabled Local Graph protocol
```

根因是 `eval_habitat.sh` 的 snapshot 写入了**同一个** `protocol_version` 字段：

1. 先写 `local_graph_effective.protocol_version = local_graph_v1`；
2. 随后为了记录 Prompt 实验协议，把根级 `protocol_version` 覆盖为
   `l1_l3_prompt_ablation_v1`；
3. verifier 要求根级 `protocol_version` 与 `local_graph_effective.protocol_version`
   完全相等，因此判定失败。shard 的 `local_graph_protocol.json` 本身是正确的。

修复：

- 新 snapshot 中根级 `protocol_version` **只表示 Local Graph 后端协议**；
- Prompt/实验协议写入独立字段 `prompt_protocol_version`；
- `prompt_effective.protocol_version` 仍记录 Prompt 协议；
- verifier 不降低 Local Graph 标准：仍要求
  `local_graph_effective` 完整且 enabled、`memory_effective.local_graph_effective`
  一致、每个 shard 的 `local_graph_protocol.json` 逐字段一致；
- 为恢复旧运行，verifier 额外识别**唯一的旧冲突形态**：根级 protocol 等于
  `prompt_effective.protocol_version`，且 `prompt_effective.prompt_mode` 属于已知
  七种模式。缺 prompt_effective、协议不一致或任意 Local Graph 字段缺失仍然失败。

新增 [finalize_l1_l3_group.py](scripts/finalize_l1_l3_group.py)：

- `--check-only` 只检查 shard 是否完整；
- 默认验证 shard 协议、shard 结果唯一且恰好 88、shard manifest 与根 manifest 一致，
  然后合并 `results.jsonl`/`summary.json`、`memory_diagnostics.jsonl`、prompt 产物，
  运行 `experiment_integrity verify`，最后写 completion marker；
- 任何检查失败都不写完成标记；不删除/不重写 shard 结果；
- 支持旧冲突 snapshot 的恢复。

launcher 新增 `--resume`：

- `pilot_complete.json` 已存在的组：验证后跳过；
- 推理完成但后处理失败的组：调用 finalizer 安全恢复，不重新推理；
- 只有部分 shard 的组：停止并保留现场，不覆盖；
- 尚未运行的组：正常启动；
- 恢复前核对 provenance 的 manifest SHA256、checkpoint 权重/config/eval_config
  SHA256、L1/L3 权重 SHA256、Habitat 配置 SHA256、pilot SHA256、seed 与 GPU 顺序；
- 已完成组也必须通过同一 provenance 身份检查，不能只凭 `pilot_complete.json` 被跳过；
- finalizer 要求每个 shard 的 prompt protocol 完全一致，并校验 version/mode/
  layer_activation/strategy/original char budget，拒绝混合来源；
- comparison 对标记完成的组仍检查 prompt 诊断的逐 Step 覆盖，避免“完成标记 + 不完整
  prompt 诊断”永久跳过。

第一组恢复时使用原 `GPU_IDS`（必须与第一次运行的 shard 顺序一致）。本次本地
没有 AutoDL 的第一组目录，因此无法实际执行恢复；恢复逻辑由 finalizer 的合成
shard 测试（新/旧 snapshot、完整、缺失、跨 shard 重复）覆盖。

## 7. 实际修改文件

新增：

- `src/lightnav/memory/prompt_modes.py`：模式计划、预算、后端校验、effective 层激活；
- `src/lightnav/memory/l1_prompt_organizer.py`：L1 组织卡；
- `src/lightnav/memory/l3_prompt_organizer.py`：L3 组织卡；
- `src/lightnav/memory/joint_prompt.py`：实体对齐、去重、冲突、联合卡；
- `src/lightnav/memory/prompt_assembly.py`：按模式调用原/新构建器，整行 token 选择；
- `src/lightnav/memory/token_budget.py`：用现有 Processor/Tokenizer 测量上下文上限、
  基座 Prompt token、生成长度余量，并做整行 token 选择；
- `src/lightnav/habitat/prompt_artifacts.py`：`prompt_protocol.json`、
  `prompt_diagnostics.jsonl`、`per_episode.csv`、legacy 测量行与 shard 合并 CLI；
- `scripts/run_l1_l3_prompt_ablation.sh`：六组实验 + 第七组 legacy、preflight、
  历史 88 结果扫描、`--resume` 安全续跑；
- `scripts/finalize_l1_l3_group.py`：从完整 shard 恢复/合并/验证，不重跑推理；
- `scripts/verify_local_graph_shards.py`：Local Graph/Prompt 协议字段分离与旧冲突
  形态恢复；
- `scripts/compare_l1_l3_prompt_ablation.py`：七组聚合、配对差异、McNemar、
  Bootstrap、`final_report.md`；
- 测试：`tests/memory/test_prompt_modes.py`、
  `tests/memory/test_prompt_organization.py`、
  `tests/habitat/test_prompt_mode_policy.py`、
  `tests/habitat/test_prompt_mode_cli.py`、
  `tests/habitat/test_prompt_artifacts.py`、
  `tests/habitat/test_prompt_token_budget.py`、
  `tests/habitat/test_l1_l3_prompt_ablation_launcher.py`、
  `tests/habitat/test_finalize_l1_l3_group.py`、
  `tests/test_verify_local_graph_shards.py`。

修改：

- `src/lightnav/memory/manager.py`：新增 `prompt_mode`、预算、组织器、上下文构建与
  confirm/discard；不改任何写入方法；
- `src/lightnav/habitat/policy.py`：新增参数与 `_generate_with_prompt_mode` 分支；
  每步测量 token 预算、按 token 整行选择、精确 token 差核验；legacy 分支原样保留；
- `src/lightnav/habitat/runner.py`：`MemoryConfig.prompt_mode/prompt_measurement`、
  有效层激活、prompt 产物写入、legacy 测量行、条件性记录服务器 `path_length`
  （普通 legacy 结果字段不变）；
- `src/lightnav/data_processor.py`：新 `l1_l3_prompt:*` 记忆块走 strict prompt，
  超长时明确拒绝而不是静默截断；
- `src/lightnav/cli/eval_habitat.py`：`--memory_prompt_mode`、
  `--prompt_measurement_enabled`；
- `scripts/eval_habitat.sh`：模式/测量解析、token 预算 snapshot、协议字段分离、
  shard 产物合并；
- `docs/CONFIGURATION.md`：新增模式表与约束。

未修改 `.references/`、模型权重、动作解码、YOLO/RGB-D/ObjectMap 源码。

## 8. 测试结果

本轮本地测试（CPU，`.venv311`，`PYTHONPATH=src`）：

- 新增/扩展定向测试：**90 passed, 17 subtests passed**（模式/组织/联合/隔离/token 预算/
  7 组配置/CLI/产物/launcher/跨 Episode/并发确认/原始 L3 块一致性/构建失败回退/
  legacy 测量/协议字段冲突恢复/shard 恢复/导航效果判定）；
- legacy 一致性：`legacy` 与 `l1_original_only` 在 L3 关闭时 Prompt 与 sample
  conversation 逐字节相同；
- L1-only：L3 分割器调用次数 0，spatial map 为空，Prompt 无 L3 文本；
- L3-only：Local Graph 构建器被替换为抛错函数仍正常运行，Prompt 无 L1 文本；
- 预算：legacy 仍 ≤320 字符；六个实验组在 fake Processor 下按 token 预算选择，
  `budget_source=token_aware_context`，`available=39` 时注入 token ≤39；无
  Processor 时回退 `char_fallback_320` 并记录原因；
- legacy 测量：`--prompt_measurement_enabled` 写出 legacy protocol/diagnostics/
  per_episode，模式/预算策略为 `original_char_budget`，Prompt 路径未变；
- 联合：已确认链接合并、同标签无链接保持 uncertain、不合并、冲突保留、去重；
- 产物：`prompt_protocol.json` 不可变、`prompt_diagnostics.jsonl` 按步
  （含 token 预算字段）、`per_episode.csv` 与两 shard 合并、`final_report.md`
  自动生成；七组配置一致性与 21 个配对；
- 跨 Episode：joint 模式 reset 后上一轮 L1 事实与 L3 ObjectMap 均不进入新 Episode；
- launcher：88 唯一 Episode、SHA256、历史 manifest 冲突停止、非 88 停止、
  `--dry-run` 不写文件、六组实验 + legacy 七组配置一致性正/负例、历史 legacy
  候选扫描为空时 `legacy_group_required=true`、`--resume` provenance 校验；
- 协议：六种非 Legacy 模式的新 root snapshot 均通过 verifier；旧 `protocol_version`
  覆盖形态仅在 prompt_effective 完整且模式已知时通过；缺失块或标记不一致失败；
- 恢复：完整 shard 不重跑推理即可 finalize；缺失 shard、跨 shard 重复 Episode
  被拒绝且不写完成标记；新/旧 snapshot 均可恢复；
- 报告：导航效果判定以 legacy 为唯一对照，McNemar p<0.05 才标记显著；latency
  进入 group/paired 统计，不允许用 Prompt 长度代替导航效果。

14 项验收要求与测试的对应关系：

| 要求 | 测试 |
| --- | --- |
| Legacy Prompt 完全一致 | `test_legacy_and_l1_original_only_are_byte_identical_when_l3_is_off`、`test_l3_original_only_matches_the_legacy_spatial_block_when_available`、`test_legacy_keeps_the_original_prompt_text_and_diagnostic_shape` |
| L1-only 隔离 | `test_l1_only_modes_never_run_the_l3_sidecar_or_inject_map_facts` |
| L3-only 隔离 | `test_l3_only_mode_never_calls_the_local_graph_builder_or_injects_l1_text` |
| L1 组织正确性 | `test_l1_separates_now_from_history_and_folds_duplicate_sightings`、`test_l1_omits_low_confidence_and_irrelevant_detections`、`test_l1_budget_drops_whole_lines_without_exceeding_the_original_budget`、`test_l1_previous_episode_history_requires_persistent_scope` |
| L3 组织正确性 | `test_l3_retrieves_by_instruction_and_recency_not_distance_alone`、`test_l3_separates_observed_spatial_and_traversal_relations`、`test_l3_marks_unverified_geometry_without_a_position_or_path`、`test_l3_current_step_observation_alone_is_not_history` |
| 实体对齐 | `test_joint_confirmed_link_merges_one_fact_but_keeps_both_sources`、`test_joint_same_label_without_link_stays_uncertain_and_unmerged` |
| 重复信息去除 | `test_joint_removes_duplicate_facts_but_keeps_distinct_objects` |
| 冲突处理 | `test_joint_conflicting_link_is_preserved_not_silently_merged`、`test_joint_unverified_geometry_conflict_keeps_position_unknown` |
| 未知关系保护 | `test_joint_budget_and_unknown_relation_protection`、`test_l3_marks_unverified_geometry_without_a_position_or_path` |
| Episode 状态隔离 | `test_episode_state_isolation_resets_context_between_episodes`、`test_joint_episode_reset_removes_previous_l3_objects_and_l1_facts` |
| Prompt 预算一致性 | `test_every_mode_respects_the_original_character_budget`、`test_protocol_is_immutable_and_shared_budget` |
| CLI / Launcher 参数映射 | `test_every_mode_is_accepted_and_mapped`、`test_eval_launcher_snapshot_records_prompt_mode_and_shared_budget`、launcher preflight tests |
| 七组配置一致性 | `test_seven_group_artifacts_pass_verification_and_compare`、`test_a_changed_non_prompt_memory_argument_breaks_consistency` |
| 默认行为兼容性 | `test_default_mode_is_legacy_and_maps_to_runner_config`、legacy 三项测试、`test_prompt_builder_failure_falls_back_to_the_original_sample` |

静态检查：

```bash
.venv311/bin/ruff check <changed files>          # 通过
.venv311/bin/python -m compileall -q src scripts tests
git diff --check                                  # 通过
bash -n scripts/eval_habitat.sh scripts/run_l1_l3_prompt_ablation.sh  # 通过
```

全仓 CPU 套件（`pytest -q -m "not gpu"`）本次结果：**1093 passed, 14 skipped,
1 failed, 99 subtests passed**。唯一失败是改动前已存在的
`tests/memory/test_rgm_place.py::RGMPlaceTests::test_initial_reference_and_global_ambiguity`
（期望 `ambiguous.reason == "unresolved_competing_candidate"`，实际
`ambiguous_global_candidates`），与 L1/L3 Prompt 无关，未修改 place verification 代码。
全仓 `ruff check src scripts tests` 另有 27 项 import-order 报错，全部位于本次未修改的
既有文件；本次新增/修改文件的 ruff 检查通过。

> 上述数字只证明代码与 Prompt 契约，不代表导航效果。

## 9. 独立审查结论

第二轮审查要求“重新阅读所有修改文件 + 对照原始调用链”。主代理完成了逐文件复审；
另启动的两次宽范围独立子代理未在时限内返回可采纳结论，一次窄范围独立子代理返回了
**ACCEPT WITH FIXES**。下表列出该独立审查与主代理复审共同发现并已修复的问题；审查后
重新运行了受影响的定向测试和全仓 CPU 套件。

| # | 发现 | 修复 |
| --- | --- | --- |
| 1 | 非 legacy 的 `path_length` 被无条件写入结果，改变 legacy 结果字段 | 只在 `prompt_mode != legacy` 时记录；legacy 字段集不变 |
| 2 | legacy 的 summary 被额外写入 `memory_prompt_mode` | 只在实验模式写入 |
| 3 | joint 的 identity conflict 分支先 `consumed` 了 L1 fact，导致“保留双方”实际丢了 L1 | 冲突时同时生成 L1 与 L3 两个独立条目；测试断言两个 label 都在 item 中 |
| 4 | concat 的 Local Graph 确认收到 L3 evidence id，可能触发 `confirm_injected` 错误 | manager 记录 pending L1 ids，只确认 L1 子集；新增混合 evidence 测试 |
| 5 | joint 合并条目被标成 `long` 而不是当前的 `short` | `_Candidate` 增加 layer 字段，按 L1 current/history 标注 |
| 6 | Evidence refs 统一用 `last_step`/第一个来源，混合 L1/L3 时 provenance 不准确 | 按每条 Evidence 的 provenance step/source 生成 ref |
| 7 | 去重统计把 joint 顶层计数与 L1 层计数重复相加 | 只统计顶层 `duplicates_removed` |
| 8 | manifest 比较依赖 scene 路径字符串拼写，可能误判不同 spelling | comparison 用 normalize 后的 Episode identity；launcher 去掉 `cmp` |
| 9 | concat 添加了额外 header 并占用预算，不是“原文直接拼接” | 保留两块原文及其原 header，只按同一预算丢弃尾部整行 |
| 10 | concat 在超出预算后 `continue`，可能留下没有事实的裸 L3 header | 改为前缀截断（首次放不下即丢弃剩余行），并移除裸 header |
| 11 | 冲突/uncertain 注释可能在对应条目已被预算丢弃时仍声称“both kept” | 只有双方条目都真正入选时才打印注释；冲突仍保留在 diagnostics |
| 12 | L1 cross-Episode 的 `first_step` 会混入上一 Episode 的 Step 编号 | current 事实的 “since step” 只使用当前 Episode 的观测 |
| 13 | L3 未验证对象的 evidence id 与 fallback provenance 不一致 | 使用同一条 fallback evidence/provenance |
| 14 | L3 链接未检查 L1 的 `provenance_validity` | 增加有效性检查 |
| 15 | L3 文本没有可信度信息 | 增加 `seg conf`，来源保留在 metadata/provenance |
| 16 | L1-only 仍写 `conceptgraphs_l3_protocol.json`，与实际未运行 L3 不符 | 只在实际启用的 layer 上写该协议 |
| 17 | launcher 未把解析出的 Python 导出给 `eval_habitat.sh` | 导出 `CLIENT_PYTHON` |
| 18 | snapshot 不支持 `--flag=value` 形式的预算参数 | 增加 `=` 解析 |
| 19 | Local Graph 的 pending selection 在 concat 丢弃 L1 块后残留 | 未入选时 `discard_selection` |
| 20 | `build_prompt_context` 的异常会直接中断 `act`，而不是回退原 sample | 捕获实验格式器异常并走原始 sample；Processor 拒绝时对空 item 列表做保护 |
| 21 | 六组 pilot manifest digest 未逐组比较 | comparison 比较 `pilot_manifest_path/sha256` 和 protocol version |
| 22 | 测试缺口：eval_habitat snapshot、concat 混合确认、joint reset、原始 L3 块一致性 | 已补对应测试 |
| 23 | 320 字符被当成六组固定上限，但它是旧记忆模块的字符预算，不是 Processor/模型上下文硬限制 | 六组改为逐 Step 模型派生 token 预算；legacy 保持原 320 字符路径 |
| 24 | 新 `l1_l3_prompt:*` 记忆块不在 Processor 的 strict prompt 集合，超长会被静默截断 | 加入 strict 集合，超长明确拒绝，policy 回退/重试 |
| 25 | 六组没有真实 legacy 对照，`l1_l3_original_concat` 不能替代 | 增加第七组 legacy；launcher 扫描历史结果并记录不可复用原因 |
| 26 | legacy 组没有 prompt 产物，统计器无法报告其 Prompt token/截断 | 增加 `--prompt_measurement_enabled`，只测量不改路径 |
| 27 | token 预算可能每步测量失败；若直接假设 320 会掩盖问题 | 回退原 320 并记录 `char_fallback_320` 与原因；统计器汇总 fallback 步数 |
| 28 | token 预算超限时可能只按近似字符裁剪 | 加入精确 human prompt token 差核验与最低优先级条目重试 |
| 29 | `eval_habitat.sh` 用同一个根 `protocol_version` 记录 Local Graph 和 Prompt 协议，覆盖导致 verifier 失败 | 根字段只保留 `local_graph_v1`，Prompt 协议独立为 `prompt_protocol_version` |
| 30 | 旧第一组 snapshot 已带冲突形态，直接重跑会丢结果 | verifier 仅对“prompt_effective 完整、模式已知、协议一致”的旧形态放行；其余仍拒绝 |
| 31 | 推理完成但后处理失败的组无法续跑，`eval_habitat.sh` 又拒绝非空目录 | 新增 finalize 脚本和 launcher `--resume`：完整 shard 只做合并/验证，不重跑推理；partial 停止并保留 |
| 32 | 报告只比较文本/诊断，可能被误读成导航改善 | 新增 legacy 配对导航判定：只有 McNemar p<0.05 的 SR 改善才标记显著；latency 纳入统计 |
| 33 | `--resume` 只凭 `pilot_complete.json` 跳过组，不核对组 config 与 root provenance | comparison 对所有组强制核对 checkpoint/config/eval/Habitat/seed/GPU/pilot SHA256 |
| 34 | finalizer 的 provenance 检查可选且缺少 model config | 改为必需 provenance，并核对权重、model config、eval config、Habitat、seed、GPU、manifest canonical digest、L1/L3 权重 |
| 35 | finalizer 只检查 shard prompt mode/strategy，可能混合不同 budget shard | 要求所有 shard prompt protocol 完全一致，并核对 version/mode/layer/strategy/char budget |
| 36 | resume preflight 未核对 config/eval_config SHA | 加入 checkpoint config/eval_config SHA 检查 |
| 37 | 有完成标记但 prompt 诊断不完整的组可在 verify-only 中漏过 | 标记完成的组也必须通过逐 Step prompt 覆盖检查 |
| 38 | 不完整报告仍可输出“显著改善”且未提多重比较 | 不完整报告只输出“暂定”；6 组对照注明未做多重比较校正 |
| 39 | verifier 对 marker 只查字段存在，null/非法值可通过 | 增加 protocol/scope/正整数校验；旧 prompt 形态要求显式 prompt 块与已知模式 |

复审确认仍然成立的点：

- `legacy` 不会进入新分支；默认值、Prompt 文本、注入位置、320 字符预算与结果字段保持；
- L1-only 的分割器调用次数为 0；L3-only 的 Local Graph 构建器不被调用；
- 所有模式注入块 ≤ 320；只整行截断；
- joint 只在 `linked_l1_evidence_id` 与 L1 Evidence 精确匹配时合并；同标签无链接保持
  `uncertain` 且不合并；identity/geometry 冲突均保留两个来源；
- L3 不生成 Place/Edge/Evidence，不把直线距离写成路径，不把观测写成可通行；
- launcher 不创建新 manifest，两个不一致历史 manifest 时停止；
- `prompt_protocol.json`、`prompt_diagnostics.jsonl`、`per_episode.csv` 按组生成，
  shard 合并有明确定义。

本轮（协议冲突修复/七组续跑/最终报告）的复审重点与结论：

- 根 `protocol_version` 不再被 Prompt 协议覆盖；
- verifier 对 Local Graph 字段和 shard marker 的严格比较未降低；
- 第一组恢复只读 shard，并先验证 88 条唯一 Episode、manifest、协议和配置；
- partial/缺 shard 不写完成标记；
- 报告不会仅凭 Prompt 长度或诊断指标声称导航改善，且必须与 legacy 做配对检验。

上一轮（320 字符/legacy 对照补充）的复审重点与结论：

- `320` 只在 legacy 路径保留；六个实验组使用同一 token 预算策略；
- token 预算来自现有 Processor/Tokenizer、`bundle.max_seq_len`、后端
  `max_model_len`/HF `max_position_embeddings` 和 `engine.max_new_tokens`；
- 测量不可用时只回退原 320 并记录具体原因，不扩大预算；
- `l1_l3_original_concat` 与 legacy 在 launcher/统计器中是独立组；
- legacy 测量产物不改变 Prompt 文本、注入位置、预算或执行路径；
- 历史 88 结果扫描为空，因此第七组必须运行。

第二轮窄范围独立审查（协议恢复/续跑/报告）返回 **ACCEPT WITH FIXES**。其 HIGH
问题是“resume 仅凭 pilot_complete 跳过组、未核对组与 root provenance 的身份”，已通过
comparison 的 group-provenance 绑定修复；MEDIUM 的 finalizer 可选 provenance、
shard prompt protocol 混合、resume 缺 config/eval SHA、标记完成组 prompt 覆盖也在
上表 33–37 修复。独立审查同时确认 finalizer 的 88 唯一 Episode、重复检测、manifest
并集和“未完成不写标记”逻辑是可靠的，未发现 partial 结果被标记完成或 shard 被删除/
重跑。

审查限制：宽范围子代理没有完成对全部文件的独立阅读；manager/组织器/比较脚本/新
token 预算模块的部分结论来自主代理复审与测试。该限制不影响已列出的修复，但审阅
记录必须如实保留。

## 10. 一键运行命令

```bash
MODEL_PATH=/path/to/LightNav-0 \
YOLO_MODEL_PATH=/path/to/yolo11n.pt \
SEG_MODEL_PATH=/path/to/yolo11n-seg.pt \
EPISODE_MANIFEST_PATH=/path/to/existing/episode_manifest.json \
PILOT_MANIFEST_PATH=/path/to/existing/pilot_manifest.json \
bash scripts/run_l1_l3_prompt_ablation.sh \
  --output-root output/l1_l3_prompt_ablation
```

AutoDL 第一组已完成推理但后处理失败时，用**相同 root、相同 GPU_IDS、相同环境**续跑：

```bash
ROOT=/path/to/existing/root
MODEL_PATH=... YOLO_MODEL_PATH=... SEG_MODEL_PATH=... \
EPISODE_MANIFEST_PATH=... PILOT_MANIFEST_PATH=... \
GPU_IDS="$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["gpu_ids"]))' "$ROOT/provenance.json")" \
bash scripts/run_l1_l3_prompt_ablation.sh --output-root "$ROOT" --resume
```

先只检查第一组 shard 是否可恢复（不写文件、不启动 Habitat）：

```bash
GPU_IDS="$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["gpu_ids"]))' "$ROOT/provenance.json")"
PYTHONPATH=src:. python3 scripts/finalize_l1_l3_group.py \
  "$ROOT/l1_original_only" --check-only --gpu-ids $GPU_IDS \
  --episode-manifest "$ROOT/episode_manifest.json" \
  --pilot-manifest "$PILOT_MANIFEST_PATH" \
  --yolo-model "$YOLO_MODEL_PATH" --seg-model "$SEG_MODEL_PATH"
```

成功输出 `"status": "postprocess_ready"` 后，再执行上面的 `--resume`。若检查失败，
按错误逐项报告（shard、Episode、manifest、协议、prompt 诊断），不能写完成标记。

launcher 会：

- 对 `l1_original_only`：发现 `pilot_complete.json` 不存在但 `shard_*` 完整，
  调用 finalizer 只做协议验证、合并、完整性验证，不重新推理；
- 对其余已完整组：验证后跳过；
- 对未运行组：正常启动；
- 对 partial/缺 shard 组：停止并保留现场，不覆盖。

建议先做只读检查（新 root）：

```bash
# 只校验并打印七组命令，不写文件
... --dry-run --output-root output/l1_l3_prompt_ablation_preflight

# 只写 provenance/SHA256，不启动 Habitat
... --preflight-only --output-root output/l1_l3_prompt_ablation_preflight
```

preflight 会：

1. 用 `validate_manifest` 要求恰好 88 条唯一 Episode、`val_unseen`；
2. 用 `check_pilot_selection` 要求 88 Episode server 选择与 manifest 完全一致；
3. 扫描 `output/` 下历史 88 Episode manifest（可用 `MANIFEST_SEARCH_ROOT` 指定），
   若存在两个不一致历史则**停止并报告**；
4. 保存 `manifest.sha256`（原文件字节）与 `checkpoint.sha256`（checkpoint 权重树）；
5. 保存 `provenance.json`（含 manifest canonical SHA256、L1/L3 权重 SHA256、
   Habitat 配置 SHA256、seed、GPU、预算来源）；
6. 扫描历史 legacy 结果并写入 `legacy_history_candidates`；没有完全匹配的 88 结果时
   `legacy_group_required=true`，第七组照常运行；
7. `--resume` 时核对 provenance 的 manifest/checkpoint/L1/L3/Habitat/pilot SHA256、
   seed 和 GPU 顺序，不重写 provenance；
8. 不存在 manifest 时拒绝继续，绝不重新采样或创建新 manifest。

## 11. 七组输出目录与统计方式

```text
<root>/
  episode_manifest.json / manifest.sha256 / checkpoint.sha256 / provenance.json
  l1_original_only/
  l1_organized_only/
  l3_original_only/
  l3_organized_only/
  l1_l3_original_concat/
  l1_l3_joint/
  legacy/
  comparison.csv / comparison.json / paired_episode_deltas.csv / final_report.md
```

每个 group 目录（`OUTPUT_ROOT=<root>/<mode>`）保存：

- `summary.json`：runner 的 SR/OS/SPL/NDTW/NE 汇总；
- `per_episode.csv`：逐 Episode 的 success/SPL/nDTW/NE/TL/steps 等；
- `config_snapshot.yaml`：effective 配置，含 `prompt_effective`；
- `prompt_protocol.json`：模式、层激活、预算策略、注入位置、构建器；
- `prompt_diagnostics.jsonl`：逐推理步的层贡献、证据、对齐、冲突、去重、截断、
  token 预算字段（legacy 为原字符预算测量行）；
- 另有 `results.jsonl`、`memory_diagnostics.jsonl`、`episode_manifest.json`、
  `pilot_complete.json`、`postprocess_provenance.json`（恢复时）、shard/日志。

汇总口径：

| 指标 | 来源 |
| --- | --- |
| SR / OSR / SPL / nDTW / NE | `results.jsonl` 的 `success` / `oracle_success` / `spl` / `ndtw` / `final_distance` |
| TL | 服务器 `info.path_length` 的均值；缺失为 `null`，不用步数替代 |
| 平均 Prompt Token | `prompt_diagnostics` 的 `final_token_count`，缺失为 `null` |
| 可用记忆 token / 实际注入 token | `available_memory_tokens` / `memory_tokens_injected`；legacy 为 `null` |
| Token 预算回退步数 | `budget_source != token_aware_context` 的步数；legacy 不计入 |
| Prompt 截断次数 | `processor_length_rejections` + 预算整行丢弃 `omitted_items` + `token_overflow_drops` |
| L1 信息数量 | 每步 `l1_fact_count`（无 fact_count 的旧 L1 用 `l1_item_count`） |
| L3 检索节点数量 | 每步 `l3_candidate_count`（指令相关的有效对象候选数） |
| 实体匹配成功率 | joint 组 `confirmed / attempted`，只用已记录链接 |
| 冲突数量 | joint 组逐 Step `conflict_count` 之和 |
| 重复信息数量 | L1 内部去重 + Joint 内部完全相同行/证据去重 |
| 模型推理延迟 | `results.jsonl` 的 `inference_latency_ms_mean`（仅模型调用，不含 Prompt/Memory 测量），按组/配对统计 |

统计：逐 Episode 七组结果；所有 21 个两两配对给出均值差与按 Episode 配对的
95% Bootstrap 区间（2000 次，seed 0）；成功/失败翻转用双侧精确 McNemar 检验。
`comparison.json` 保存全量，`comparison.csv` 是 group/paired 的均值和区间，
`paired_episode_deltas.csv` 是逐 Episode 宽表，`final_report.md` 是自动生成的中文报告。
`final_report.md` 第 4 节以 legacy 为唯一对照输出**导航效果判定**：只有配对 SR 提升且
双侧 McNemar p<0.05 才标记统计显著；同时报告 ΔSPL/ΔNE/ΔLatency。Prompt 长度、
Token 数、诊断计数只用于解释机制，不作为导航改善证据；若没有显著 SR 改善，报告明确写
“本数据不能证明任一组织方式真正改善导航效果”。6 个实验组分别与 legacy 比较，
p<0.05 未做多重比较校正，McNemar 结果是探索性的；不完整报告只输出中间结果，不输出
最终导航结论。

## 12. 已知限制

1. **L1 没有对象身份与度量位置**：`EvidenceRecord.position`/`heading` 未被 L1 写入器
   设置，YOLO 也只有一个类别标签；L1 去重只能到“同标签”。不能把同标签不同实例
   强行合并，也不能声称 L1 目标距离。
2. **L3 没有物体间关系/通行性**：只有 `observed_from` 与真实 `traversed_to`（连接
   Viewpoint）；物体通行永远是 `unknown`。直线距离和方位只由当前位姿与可信世界中心
   计算，**不是**路径长度，不能推断可通行。
3. **实体对齐依赖 L3 写入时链接**：只有同类、高 IoU、非歧义、Mask 覆盖率 ≥ 0.8 的
   L1 检测才可能链接；没有链接的同标签对象保持两个来源和 `uncertain`，因此
   joint Prompt 可能仍出现语义相近的两行（这是有意的保守行为）。
4. **Token 预算是逐 Step 测量的，不是固定字符数**：它依赖 Processor/Tokenizer、
   `bundle.max_seq_len`、后端 `max_model_len` 和 `engine.max_new_tokens`。测量需要
   每步一次额外 Processor pass；若这些信息缺失，只回退原 320 字符并记录原因。
5. **Legacy 是第七组而不是 concat**：legacy 走原 L3/L1 二选一注入路径，其 Prompt
   统计来自 `--prompt_measurement_enabled` 的测量行（无层/对齐统计），因此
   legacy 的 L1/L3 信息数量与实体匹配率在统计器中为 `null`，不能与实验组逐项比较。
6. **88 Episode / 11 场景统计功效有限**：McNemar 只检验成功翻转；Bootstrap 是
   episode-level 条件区间，不覆盖场景抽样不确定性；不能把结果外推为完整 val_unseen。
7. **TL 依赖服务器字段**：Habitat 服务器报告 `path_length` 时才有 TL；测试替身或
   某些配置缺失时报告 `null`。
8. **L3-only 仍会核验自己写入的 L1 链接**：`SpatialEvidence.linked_l1_evidence_id`
   是 L3 自己的 provenance；L3-only 只读取该字段做有效性核验，不调用 L1 Prompt
   构建器，也不把 L1 文本写入 Prompt。
9. **本地没有运行真实 Habitat/YOLO/seg/GPU**：CPU 测试使用 fake engine/extractor；
   `ultralytics` 未安装，YOLO 真实推理路径未在本机复跑；本报告不声称任何真实导航
   指标。
10. **HiMemVLN 本地不完整**：Localer/Globaler 源码缺失，只能核对 trainer 外层调度；
   论文细节不能当作已核对实现。
11. **第一组恢复要求源码身份一致**：comparison 的 `common` 配置仍要求
    `source_python_sha256` 一致。本轮的修复只改脚本/测试/文档，`src/lightnav` 未变；
    如果 AutoDL 第一组是在另一次 `src/lightnav` 修改之后运行的，finalizer 仍可合并
    shard，但最终 comparison 会拒绝跨源码版本混用，此时需要重跑第一组。
12. **恢复只适用于完整 shard**：缺 shard、结果不足 88、跨 shard 重复或 prompt 诊断
    不完整时 finalizer 拒绝写完成标记；必须保留现场并人工判断，不能静默续跑。
13. **shard 结果没有独立的 checkpoint 身份文件**：finalizer 用 root provenance、
    root config snapshot、shard manifest 和协议 marker 绑定一次运行；如果人工把另一
    checkpoint 的 shard 结果与相同协议 marker 混入同一目录，仅凭现有文件无法识别。
    正常 launcher 生成的目录不存在这种情况；不要手工混用 shard。
13. **既有测试失败**：`tests/memory/test_rgm_place.py` 的
   `test_initial_reference_and_global_ambiguity` 为改动前既有失败，未修改相关代码。

## 13. 代码验收 vs 真实导航效果

**代码验收（本地已完成）**：七种模式可解析；legacy 逐字节一致且只增加测量产物；
L1/L3 隔离；六个实验组使用同一模型派生 token 预算，无法测量时明确回退并记录原因；
L1 当前/历史区分与去重；L3 指令检索、关系分层、未知关系保护；联合实体对齐、去重、
冲突保留；逐 Episode 状态隔离；历史 88 结果扫描为空；七组真实配置一致性；
`ruff`、`compileall`、`git diff --check` 通过。

**未完成、必须另行在 AutoDL 验证的导航效果**：同一 88 Episode、同一 checkpoint、
同一 seed 下**七组**（六个实验组 + legacy 对照）的 SR/SPL/NE/nDTW/TL/OSR 与配对
检验。代码验收不能替代导航实验；即使七组代码全部通过，也不能预先声称
organized/joint 更好。正式运行请使用第 10 节命令；launcher 必须先运行、且默认会
运行第七组 legacy。实验结束后以 `final_report.md` 与 `comparison.json` 为准。
