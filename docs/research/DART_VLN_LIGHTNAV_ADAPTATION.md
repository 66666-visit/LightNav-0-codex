# DART-VLN Test-Time Memory Decay → LightNav Geometry-only 适配

## 状态与范围

| 项目 | 值 |
| --- | --- |
| 项目目录 | `/home/lilixing/projects/LightNav-0-codex` |
| 分支 | `experiment/geometry-only-73-9` |
| 基准提交 | `ed5b9b3` |
| 基准 SR | `73.86%`（作为对照记录，本机未复测） |
| 官方依据 | `.references/DART-VLN`（论文 README 与 `map_nav_src/`） |
| 实现模式 | 仅 `off` / `decay_only`；默认 `off` |
| 明确未实现 | Anti-Loop、`update_only`、`full`、任何外部 LLM API |
| 明确未修改 | YOLO11n / YOLO11n-seg、RGB-D 点云构建、L1/L3 对象关联与融合规则、模型权重、动作解码接口 |
| Git | 未 commit、未 push；`.references` 保留 |

**结论先行：** 本轮完成的是代码实现、CPU 代码级验证和独立审查。所有列出的 pytest、ruff（修改文件）、compileall、`git diff --check` 已通过；**任何 SR/SPL/NE/TL 导航收益都尚未验证**，必须在 AutoDL 上用真实 Habitat、权重和固定 Episode 集对照测量。CPU Fake 测试不是导航评测，不能替代 AutoDL。

---

## 1. 论文机制

DART-VLN 的 Test-Time Memory Decay 是**读取侧**的槽位重加权，不改写已存储的记忆内容。对每个显式记忆槽 `m_i` 维护：

- 槽龄：`age_i = t - t_last_i`
- 访问次数：`count_i`
- 新息（novelty）：`nu_i = clip(1 - cos(f_old, f_new), 0, 1)`
- 新息 EMA：`n_i = rho * n_i_prev + (1 - rho) * nu_i`

读取时：

```text
w_i = clip(
    exp(-lambda * age_i)
    * (1 - alpha * count_i / (count_i + 1))
    * (0.5 + 0.5 * n_i),
    w_min,
    w_max
)
```

论文默认参数：

| 参数 | 值 |
| --- | ---: |
| `rho` | 0.5 |
| `lambda` | 0.12 |
| `alpha` | 0.15 |
| `w_min` | 0.35 |
| `w_max` | 1.0 |

官方实现的位置与行为：

| 机制 | 官方代码位置 | 说明 |
| --- | --- | --- |
| 公式与默认参数 | `.references/DART-VLN/README.md:39-69`、`:218` | 论文公式；配置默认值 |
| 命令行开关 | `map_nav_src/r2r/parser.py:91-106` | `--dynamic_memory_mode off/update_only/decay_only/full` 等 |
| 模式选择 | `map_nav_src/r2r/env.py:155-163` | `decay_only` 只打开读取侧 decay，不打开写侧 update |
| 槽位元数据初始化 | `map_nav_src/r2r/env.py:199-230` | 新槽 `last_update=step`、`visit_count=1`、`novelty_ema=1` |
| 新息计算 | `map_nav_src/r2r/env.py:237-245` | 余弦距离并裁剪到 `[0,1]` |
| 写侧更新（decay-only 不走） | `map_nav_src/r2r/env.py:246-311` | 匹配、融合、`visit_count += 1`、EMA |
| 元数据读取 | `map_nav_src/r2r/env.py:312-318` | `age = global_step - last_update_step` |
| 元数据进入模型 | `map_nav_src/r2r/env.py:544-546`、`:736-767` | `grid_age`、`grid_visit_count`、`grid_novelty_ema` |
| 读取侧权重 | `map_nav_src/models/vilmodel.py:740-758` | `recency * repeat * novelty`，再 clamp |
| 权重注入 readout | `map_nav_src/models/vilmodel.py:817-848` | 对每个 grid cell 的 patch logits 加 `log(w)`，softmax 聚合 |
| 配置传入模型 | `map_nav_src/models/vilmodel.py:704-713` | `dynamic_memory_decay`、λ、上下界 |
| anti-loop（未实现） | `map_nav_src/r2r/agent.py:452-500` | 与本轮 decay-only 无关 |

需要特别指出：官方 `decay_only` 关闭了写侧 `_dynamic_memory_update`，因此官方实现中的 `visit_count` 和 `novelty_ema` 在 decay-only 下基本停留在初始化值，实际变化的项主要是 recency。本项目按用户要求保留公式中的三项，并把**控制元数据**维护在独立的 Episode-local registry 中；这属于读取侧控制状态，不是对已存 Evidence 的写回，详见第 4 节。

---

## 2. 官方实现到 LightNav 的调用链映射

### 2.1 DART 槽位

DART 的槽位是 GridMM 的 **14×14 grid cell**：每个 cell 聚合多个 patch 特征，`grid_age/grid_visit_count/grid_novelty_ema` 是与 cell 对齐的数组；权重以 `log(w)` 加到该 cell 内各 patch 的聚合 logits 上，再由 softmax 归一化。它是“同一 cell 内多个 patch 的加权聚合”，不是改写槽位特征本身。

### 2.2 LightNav 的 L1、L3、ObjectMap、DINO place

LightNav 没有可微的 grid-cell readout，而是把记忆序列化为短文本，经 Prompt 注入冻结模型。真实调用链如下。

| 层 | 产生 | 存储 | 检索/读取 | 注入 |
| --- | --- | --- | --- | --- |
| L1（YOLO11n 2D 检测） | `src/lightnav/memory/yolo_scene_extractor.py:25,85` 产生 `SceneExtraction`，`MemoryWriter.observe` 转成 `MemoryEvent` | `src/lightnav/memory/manager.py:888` 的 `_persist_v3_events` 写入 `V3MemoryStore.add`（`v3_store.py:161`），得到 `EvidenceRecord` | `LocalGraphFusion._current_candidates` / `_historical_candidates`（`local_graph_fusion.py:225/261`）读取与指令相关的当前/历史 landmark | `MemoryContext` → `memory_context_builder.inject_memory_context`（`:57`），由 `policy.py:451` 的 `_generate_with_local_graph_memory` 注入 |
| L3（YOLO11n-seg 3D 实例） | `src/lightnav/memory/yolo_seg_extractor.py:46,97`；`manager.observe_spatial`（`:420`）调用 `make_spatial_evidence` | `ObjectMap.observe`（`conceptgraphs_lite.py:425`）增量合并 `ObjectNode`（点云、中心、bbox、`observation_count`、Evidence） | `SpatialGraphAdapter.build`（`spatial_graph_adapter.py:201`）选择历史对象、计算距离/方位 | 同一个 `MemoryContext` / Prompt 审计链 |
| ObjectMap | 即 L3 的 `ObjectMap`：`objects`、`viewpoints`、`edges`、`evidence`（`conceptgraphs_lite.py:382-404`） | 每 Episode reset；`SpatialEvidence` 无学习特征 | 同上 | 同上 |
| DINO place（旧视觉地点） | `place_recognition.py:68` 的 DINOv2 提取器；`manager._observe_place_v3` + `_persist_v3_events` | `V3MemoryStore.add_place`（`v3_store.py:261`）更新 `SpatialNodeV3.visual_descriptor`（EMA） | V3 路径的 `V3Retriever`；Local Graph 用 `current_place_id` 限定历史一跳 | 同一个 Prompt 链 |

**LightNav 的读取接口：**最适合做 read-side weighting 的位置不是 YOLO/点云，而是“候选事实 → Prompt 上下文”的边界：

1. `LocalGraphFusion.build` 的历史候选排序（`local_graph_fusion.py:128-160`）。
2. `SpatialGraphAdapter.build` 的历史对象候选排序（`spatial_graph_adapter.py:201-325`）。
3. `MemoryManager._read_memory_context` 对已生成的 `MemoryContext.layer_items` 做最终顺序与诊断（`manager.py:388`）。

这三处只读取元数据和文本，不触碰 Evidence、点云、对象几何、语义标签或验证规则。

---

## 3. 实现设计

### 3.1 独立配置

新增 `src/lightnav/memory/dart_decay_config.py`：

```python
DartDecayConfig(
    mode="off",          # off | decay_only
    rho=0.5,
    decay_lambda=0.12,
    alpha=0.15,
    w_min=0.35,
    w_max=1.0,
)
```

- 默认 `off`；`off` 下不构造 `DartMemoryDecay`，原 Prompt/sample/token/action 路径保持不变。
- `update_only`、`full` 在配置层直接拒绝。
- CLI 新增 `--dart_decay_mode`、`--dart_decay_rho`、`--dart_decay_lambda`、`--dart_decay_alpha`、`--dart_decay_w_min`、`--dart_decay_w_max`。
- `MemoryConfig` 要求 `decay_only` 必须同时 `--memory_enabled`。

### 3.2 槽位定义

`src/lightnav/memory/dart_memory_decay.py` 维护 Episode-local registry：

| 槽类型 | Slot ID | 来源 | age | visit count | novelty |
| --- | --- | --- | --- | --- | --- |
| Evidence 槽 | `evidence:<canonical Evidence key>` 或 `evidence:<event_id>` | L1/L3/V3 的每条 Evidence | `step - record.step` | 1（Evidence 是一次观测的不可变记录） | 无特征，保持 DART 初始化先验 1.0 |
| L3 对象槽 | `object:<object_id>` | `ObjectNode` | `step - 最后观测 step` | `ObjectNode.observation_count` | 无学习特征，保持先验 1.0 |
| DINO place 槽 | `place:<node_id>` | `SpatialNodeV3` / `EvidenceRecord.visual_descriptor` | `step - last_seen_step` | Episode-local 刷新次数（1,2,...） | 真实 DINO 描述子的 `1-cos`，再 EMA |

`DartMemoryDecay.observe_place` 的旧特征优先使用**刷新前的持久节点描述子**（`V3MemoryStore` 中该 place node 的 EMA 描述子），而不是上一次原始观测；这使 novelty 与 DART 的“槽位已存特征 vs 新特征”一致。新 Episode 的第一次 place 观测忽略上一 Episode 的节点旧特征，novelty 从 DART 初始化值 1.0 开始，避免跨 Episode 的 EMA 污染。

### 3.3 公式

`compute_decay_weight` 严格实现：

```text
recency = exp(-lambda * age)
repeat  = 1 - alpha * count / (count + 1)
novelty_factor = 0.5 + 0.5 * novelty
weight  = clip(recency * repeat * novelty_factor, w_min, w_max)
```

- `age` 为非负整数；`count >= 1`；`novelty` clamp 到 `[0,1]`。
- 官方实现对 repeat factor 额外有 `.clamp(min=0.1)`；论文公式没有。默认 `alpha=0.15` 时 repeat factor 至少 0.85，额外 clamp 不会生效，因此本项目按论文公式实现，并在审查记录中注明这一差异。
- `w_max=1.0` 时乘积本身最大约 0.925（count=1），上界只在自定义 `w_max<0.925` 时可见；下界 `w_min` 在槽龄约 10 步后开始生效。

### 3.4 读取侧应用

1. **候选层（candidate admission）**
   - `LocalGraphFusion`：历史候选按 decay weight 降序排序；只有当该候选的所有历史 Evidence 都有槽位权重时才排序，否则保持原顺序（避免 scene-persistent 跨 Episode 候选被部分加权）。
   - `SpatialGraphAdapter`：以 `distance_m / weight` 作为读取分数排序；距离仍是导航先验，但足够旧的近物体会被较新的远物体超过。权重相同（全部新鲜）时与原距离排序完全一致。
2. **最终文本层（final context order）**
   - `MemoryManager._read_memory_context` 对 `MemoryContext.layer_items` 按 weight 稳定排序，并按同一 permutation 重排 `context.text` 的正文行。
   - `build_context_text` 仍按既有字符预算逐行保留；因此排序会把低权重事实放到末尾，在预算紧张时优先丢弃。V3/V4/legacy 至少在最终顺序上使用权重；几何 L3 路径的候选选择也会改变。
3. **不做的事**
   - 不把权重乘进 Evidence 内容、点云、bbox、中心、语义 label。
   - 不改变 `SpatialGraphAdapter` 的置信度门槛、L1 置信度门槛、Evidence 验证规则。
   - 不改变 Prompt 的插入位置、`<navigation_task>` 契约、视频 token、动作解码。

### 3.5 诊断

`MemoryManager.last_decay_diagnostic` 进入 `memory_diagnostics.jsonl`，不进入 Prompt。字段包括：

- `memory_age`
- `visit_count`
- `novelty`
- `decay_weight`
- `decay_applied_count`
- `decay_skipped_count`
- `novelty_source`（`initial_prior` / `cosine_ema`）
- `novelty_measured_count` / `novelty_prior_count`
- `order_changed` / `text_reordered`
- `prompt_order_before` / `prompt_order_after`
- `candidate_selection`（空间候选的 object_id、distance、weight、rank、selected_object_ids；Local Graph 历史候选的 evidence、weight、rank、selected_evidence_ids）
- 每个 item 的 `slot_id`、`slot_type`、`applied`、`skip_reason`

`off` 下 `dart_decay` 为 `null`，原有诊断字段不变。

### 3.6 Episode 隔离

- `MemoryManager.reset_episode` / `close_episode` 调用 `DartMemoryDecay.reset_episode` / `close_episode`，清空 registry。
- scene-persistent 的旧 Evidence 在下一 Episode 读取时没有槽位 → `decay_skipped_count`，保持原顺序，不伪造权重。
- 新 Episode 的 place 首次观测不读取上一 Episode 的旧特征；之后同一 Episode 内的刷新才使用持久节点 EMA。
- `ObjectMap` 本身仍按 V1.1 每 Episode 清空，未被 decay 改变。

---

## 4. 算法迁移差异

| 维度 | DART-VLN | 本实现 | 原因/影响 |
| --- | --- | --- | --- |
| 槽位 | GridMM 14×14 cell | Evidence/Object/Place 三类槽 | LightNav 无 grid-cell readout |
| 权重作用 | 对 cell 内 patch logits 加 `log(w)` 后 softmax | 候选排序 + 最终文本行排序 | LightNav 是冻结模型的文本 Prompt，不可微 |
| age | step 差 | step 差 | 一致 |
| visit count | 写侧匹配更新 | L3 object 用真实 `observation_count`；place 用 Episode-local 刷新次数；L1 evidence 固定 1 | L1 是一次观测记录，没有持久实例身份；不伪造类别频率；place 不让旧 Episode 计数污染新 Episode |
| novelty | 旧/新槽特征余弦 + EMA | 仅 DINO place 槽有真实特征；L1/L3 保持初始化先验 1.0 | L1/L3 没有可比较的学习特征；不用类别、IoU、几何距离冒充 |
| 元数据写入 | decay-only 下写侧关闭，元数据基本静态 | 独立读取侧 registry 更新 age/count/novelty | 用户明确允许控制元数据更新；不写回 Evidence |
| repeat factor | 官方代码额外 clamp 0.1 | 按论文公式，无额外 clamp | 默认 alpha 下无差异；文档明示 |
| Anti-Loop | 可选 | 未实现 | 本轮只做 decay-only |
| 跨 Episode | Episode 内 | registry 每 Episode 清空；scene-persistent 旧项跳过 | 防污染 |
| 外部 API | 无 | 无 | 约束 |

### 4.1 Novelty 的数据接口缺口（必须明确报告）

真实可比较的 LightNav 特征只有 DINO place 描述子：

- `EvidenceRecord.visual_descriptor` 只在 `place_observation` 事件上被赋值（`manager.py:888-900` 一带的 `_v3_pending_descriptors`）。
- L1 `object_observation` / `landmark` 的 `EvidenceRecord.visual_descriptor` 为 `None`；YOLO11n 检测器只输出类别、框、分数。
- L3 `SpatialEvidence`（`conceptgraphs_lite.py:367-379`）和 `ObjectNode`（`:328-336`）没有学习式视觉嵌入；只有 Mask 点云、中心、bbox、观测次数。
- 因此 L1/L3 没有 DART 所需的 `f_old, f_new`。本实现：
  - **不**用类别相同、IoU、中心距离、点云相似度等冒充 novelty；
  - 把 novelty 保持在 DART 初始化先验 `1.0`，诊断明确标为 `novelty_source=initial_prior`；
  - 仅对真实 DINO place 槽计算 `1-cos` 并做 EMA；
  - 在文档中把“缺少 per-slot 学习特征”列为后续若要真正启用 L1/L3 novelty 的待补接口，而不是在本轮偷偷加模型。

这意味着：在只启用 L1、未启用 DINO place 的配置下，novelty 项恒为 1.0，repeat 项对 L1 evidence 恒为 0.925；L1 的 decay 主要体现为 recency，且与既有“越新越优先”的排序高度重合。真正的 count/novelty 选择效应出现在 L3 对象槽和 DINO place 槽。该限制已在诊断和本文中明确，不应在 AutoDL 报告中把它写成 L1 的 novelty 已生效。

---

## 5. 修改文件

### 新增

| 文件 | 作用 |
| --- | --- |
| `src/lightnav/memory/dart_decay_config.py` | `DartDecayConfig`、默认参数、模式与协议版本校验 |
| `src/lightnav/memory/dart_memory_decay.py` | 槽位 registry、余弦 novelty、EMA、权重公式、上下文重排、诊断 |
| `tests/memory/test_dart_memory_decay.py` | 公式、age、count、novelty、上下界、隔离、不可变、排序、候选诊断、Local Graph 选择 |
| `tests/habitat/test_dart_decay_policy.py` | off/decay 对照、L3 选事实、Evidence/几何不可变、Episode 隔离、DINO novelty 接线 |
| `docs/research/DART_VLN_LIGHTNAV_ADAPTATION.md` | 本文 |

### 修改

| 文件 | 作用 |
| --- | --- |
| `src/lightnav/memory/manager.py` | 构造/重置 decay、槽位观测、place 特征快照、读取侧应用、候选诊断 |
| `src/lightnav/memory/local_graph_fusion.py` | 可选 `weight_lookup`；历史候选 read-side 排序；候选诊断 |
| `src/lightnav/memory/spatial_graph_adapter.py` | 可选 `weight_lookup`；`distance/weight` 候选排序；候选诊断 |
| `src/lightnav/habitat/policy.py` | 在 V4/local graph/legacy 诊断中加入 `dart_decay`；不改 sample |
| `src/lightnav/habitat/runner.py` | `MemoryConfig.dart_decay`、构造传递、`dart_decay_protocol.json`、目录 guard |
| `src/lightnav/cli/eval_habitat.py` | 六个 CLI 参数与配置映射 |
| `scripts/eval_habitat.sh` | `memory_args` 分组、`dart_decay_effective` 快照 |
| `tests/habitat/test_runner.py` | CLI 默认 off、参数映射、`requires memory_enabled` |
| `tests/habitat/test_local_graph_eval_launcher.py` | launcher 快照记录 decay 配置 |
| `docs/CONFIGURATION.md` | 新参数、行为、novelty 限制、示例命令 |

### 明确未修改

- `src/lightnav/memory/yolo_scene_extractor.py`
- `src/lightnav/memory/yolo_seg_extractor.py`
- `src/lightnav/memory/conceptgraphs_lite.py`（点云、`ObjectMap` 关联/融合）
- `src/lightnav/memory/v3_store.py`（除被调用外无改动）
- `src/lightnav/memory/v3_schemas.py`
- 模型权重、动作 tokenizer、解码参数、`TrackingAgent`
- `.references/`

---

## 6. 测试结果

### 6.1 已通过的代码测试（本机 CPU）

运行环境：Python 3.11.17、CPU torch 2.15.0、pytest 9.2.0、numpy 2.4.6。未加载真实 LightNav 权重，未启动 Habitat。

| 命令 | 结果 |
| --- | --- |
| `pytest -q tests/memory/test_dart_memory_decay.py tests/habitat/test_dart_decay_policy.py tests/habitat/test_local_graph_eval_launcher.py::...::test_snapshot_records_dart_decay_settings tests/habitat/test_runner.py::test_dart_decay_cli_defaults_off_and_maps_paper_parameters` | **39 passed**（其中 37 个在最后一次 place 计数隔离修改后复跑通过） |
| `pytest -m 'not gpu' -q`（全仓 CPU，最后一次 place 计数隔离修改前） | **1044 passed、1 failed、14 skipped、82 subtests passed，30.67s** |
| `ruff check <所有修改/新增 Python 文件>` | **All checks passed** |
| `python -m compileall -q src tests` | **通过** |
| `git diff --check` | **通过** |

全仓唯一失败是既有 `tests/memory/test_rgm_place.py::RGMPlaceTests::test_initial_reference_and_global_ambiguity`：期望 `unresolved_competing_candidate`，实际 `ambiguous_global_candidates`。该文件与 `place_verification.py` 在本轮均未修改，属于已知既有失败；本轮没有用修改测试期望的方式掩盖它。

全仓 `ruff check .` 仍报告 1222 项，位于未修改的 `.references/`、`reference_repos/` 与既有脚本中；本轮修改文件单独检查为 0 项。

### 6.2 需求对应

| 需求 | 覆盖 |
| --- | --- |
| 1. 权重公式单元测试 | `test_paper_weight_formula_is_exact` |
| 2. age 增加权重变化 | `test_age_monotonically_decreases_weight_until_floor`、`test_slot_metadata_age_and_visit_count_change_weight` |
| 3. visit count 增加权重变化 | `test_visit_count_monotonically_decreases_weight`、`test_slot_metadata_age_and_visit_count_change_weight` |
| 4. novelty 对权重影响 | `test_novelty_monotonically_increases_weight`、`test_place_novelty_ema...`、`test_place_novelty_uses_persistent_node_feature_when_supplied` |
| 5. 权重上下界 | `test_weight_lower_and_upper_bounds_are_clamped` |
| 6. 原始 Evidence 不可变 | `test_decay_does_not_mutate_stored_evidence_or_object_geometry`、`test_diagnostics_are_json_safe_and_do_not_touch_item_metadata` |
| 7. Episode 状态隔离 | `test_episode_reset_isolates_slots`、`test_episode_reset_clears_decay_slots_and_old_objects`、`test_first_episode_observation_ignores_persistent_node_old_feature` |
| 8. decay_only 与 off 对照 | `test_decay_changes_which_complete_spatial_fact_enters_prompt`、`test_off_matches_default_policy_prompts_and_tokens` |
| 9. L1/L3 与 Memory 回归 | 既有 `tests/memory`、`test_local_graph_fusion.py`、`test_spatial_l3_policy.py`、`test_policy.py`；全量 1044 passed |
| 10. 原有模式兼容 | 默认 `off`、CLI 映射测试、off/default Prompt 与 token 相等测试、全量回归 |

### 6.3 关键代码级证据

`tests/habitat/test_dart_decay_policy.py::test_decay_changes_which_complete_spatial_fact_enters_prompt` 用真实 `ObjectMap`/`SpatialGraphAdapter`/Prompt 链构造两个历史对象：

- `chair` 更近但更旧（age=3，weight≈0.645）
- `table` 稍远但更新（age=1，weight≈0.820）
- `off`：距离排序 → Prompt 选 `chair`
- `decay_only`：`distance/weight` → Prompt 选 `table`
- 诊断 `candidate_selection` 记录两者 rank/weight，`selected_object_ids` 只有 `table`。

这是 CPU Fake 分割下的**代码路径验证**，不是真实 Habitat 导航结果。

### 6.4 尚未验证的导航效果

以下全部未测：

- 真实 Habitat + LightNav checkpoint 的 SR、SPL、NE、TL、OSR
- 真实 YOLO11n / YOLO11n-seg / DINO 输出下的权重分布
- decay_only 对动作序列、停止时机、碰撞、路径长度的影响
- 不同 Scene/Episode 的方差、失败案例
- 计算开销（只做了代码级复杂度审查；未测真实 step latency）

---

## 7. 独立审查发现的问题与修复

实现完成后切换到审查角色，重新阅读修改文件，不依赖实现时的结论。发现并修复：

1. **只在最终文本排序会“只记录 weight 不真正使用”**
   - 初始设计只对已选中的 1–2 条文本排序；单条事实时 Prompt 完全不变。
   - 修复：在 `LocalGraphFusion` 历史候选和 `SpatialGraphAdapter` 对象候选的**准入阶段**加入 weight；L3 集成测试证明会改变最终进入 Prompt 的完整事实。
2. **Place novelty 的旧特征基准不对**
   - 初版用“上一次原始观测”作 old feature；DART 用槽位已存特征。
   - 修复：Manager 在 `_persist_v3_events` 前快照 place node 的持久 EMA 描述子，`observe_place` 优先用它作 old feature；新增 `test_place_novelty_uses_persistent_node_feature_when_supplied`。
3. **跨 Episode 的 place old feature 可能污染新 Episode EMA**
   - 修复：新 Episode 的第一次 place 观测忽略上一 Episode 的节点旧特征，novelty 从初始化先验开始；新增 `test_first_episode_observation_ignores_persistent_node_old_feature`。
4. **Prompt 行数与 items 非一一对应时的诊断会虚报 applied**
   - 修复：该布局下所有 item 标为 skipped，`decay_applied_count=0`，不声称已生效。
5. **混合上下文里 skipped item 的 `skip_reason` 为 None**
   - 修复：默认 `no_slot_metadata`。
6. **Place 特征校验在元数据更新之后**
   - 修复：先校验特征，再更新 slot，避免异常时留下半更新状态。
7. **Launcher 对 `--dart_decay_lambda` 的快照解析错误**
   - 根因：CLI 名称 `lambda` 与 dataclass 字段 `decay_lambda` 不同；已映射并加测试，同时支持 `--flag=value`。
8. **输出目录 guard 不认识 decay 协议**
   - 修复：`dart_decay_protocol.json` 纳入非 V3/V4 目录复用检查。
9. **缺少候选级审计**
   - 修复：新增 `candidate_selection` 诊断（候选 rank/weight/selected IDs），便于 AutoDL 解释“为什么某条历史事实被换掉”。
10. **L1 的 count/novelty 实际不可用**
    - 审查确认这不是代码 bug，而是数据接口缺失：L1 Evidence 无持久实例身份、无学习特征。没有用类别频率、IoU 或几何距离伪造；已在第 4.1 节和诊断中明确。
11. **重复衰减审计**
    - 检查所有路径：`retrieve` 的每个 return 只调用一次 `_read_memory_context`；`get_local_graph_context` / `get_spatial_context` 各自只调用一次；V4 的 `reformat_v4_context` 针对新 formatter 结果再读一次，不是同一 context 累乘；候选准入与最终行排序共享同一权重，但没有对内容做第二次乘法。结论：**不存在同一 context 的重复乘法衰减**。
12. **原始输入审计**
    - `off` 下 `_read_memory_context` 原样返回同一 context；集成测试比较 off/default 的 Prompt、token 诊断完全相等；全量既有测试通过。Prompt 插入点、视频 segment、`<navigation_task>` 契约未改。
13. **几何/融合审计**
    - `conceptgraphs_lite.py`、`v3_store.py`、YOLO extractor、RGB-D 反投影均未修改；集成测试在 4 步 L3 运行后比较 Evidence 字典和 ObjectNode 的中心/bbox/点云/计数/证据列表，未变化。
14. **外部 API / Anti-Loop / 写侧消融**
    - 未引入任何 API；配置层拒绝 `update_only`/`full`；未实现 anti-loop。
15. **隐藏推理开销**
    - 没有新增模型 forward；novelty 只用已经存在的 DINO place 描述子；place 快照只在有 `place_observation` 的 step 做；候选排序是 O(n log n) 的小列表。真实 latency 仍需 AutoDL 测量。

---

## 8. AutoDL 运行命令

以下命令**没有在本机执行**；路径、权重、manifest 必须替换为 AutoDL 实际路径。目标是同一 checkpoint、同一 Habitat 配置、同一 88 Episode Pilot，仅切换 `--dart_decay_mode`。

### 8.1 公共环境

```bash
export REPO_ROOT=/path/to/LightNav-0-codex
export MODEL_PATH=/path/to/LightNav-0
export ACTION_TOKENIZER_BUNDLE=/path/to/LightNav-0/action_tokenizer
export HABITAT_CONFIG=$REPO_ROOT/habitat_server/configs/vlnce_r2r.yaml
export DATA_PATH=/path/to/R2R_VLNCE_v1-3_preprocessed
export SCENES_DIR=/path/to/scene_datasets
export YOLO_MODEL_PATH=/path/to/yolo11n.pt
export YOLO_SEG_MODEL_PATH=/path/to/yolo11n-seg.pt
export DINO_MODEL_PATH=/path/to/dinov2-small          # 仅 novelty 需要
export PILOT_MANIFEST_PATH=/path/to/rgm_pilot_88_v1/pilot_manifest.json
export EPISODE_MANIFEST_PATH=/path/to/rgm_pilot_88_v1/episode_manifest.json

export TASK=vlnce SPLIT=val_unseen BACKEND=vllm_local
export STRICT_FULL_EVAL=1 EXPECTED_EPISODES=88 EPISODES=-1
export MAX_STEPS=500 NO_FORCE_STOP=1 SEED=42 RESUME=0
export SERVER_ARGS="" LANGUAGES="" GPU_MEM_UTIL=0.65
unset VLN_EVAL_TEMPERATURE VLN_EVAL_TOP_P VLN_EVAL_TOP_K VLN_EVAL_TRAJ_TOP1
```

### 8.2 off 对照

```bash
export GPU_IDS="0 1"                       # 与 decay 组保持完全一致
export OUTPUT_ROOT=/path/to/output/dart_decay_88_off
export CLIENT_ARGS="--memory_enabled --memory_strategy hierarchical --memory_layers short,mid,long \
  --memory_extractor yolo11 --memory_extractor_model_path $YOLO_MODEL_PATH \
  --memory_extractor_device cpu --memory_extractor_confidence 0.25 --memory_extract_every_n_steps 5 \
  --memory_place_enabled --memory_place_model_path $DINO_MODEL_PATH \
  --memory_place_device cpu --memory_place_extract_every_n_steps 5 \
  --memory_place_similarity_threshold 0.85 \
  --local_graph_enabled --local_graph_scope episode_local \
  --local_graph_prompt_max_chars 320 --local_graph_max_landmarks 2 \
  --local_graph_reactivation_steps 8 \
  --spatial_l3_mode inject --spatial_l3_seg_model_path $YOLO_SEG_MODEL_PATH \
  --spatial_l3_seg_device cpu --spatial_l3_seg_every_n_steps 5 \
  --dart_decay_mode off"

[ -e "$OUTPUT_ROOT" ] && echo "choose a fresh OUTPUT_ROOT" || bash scripts/eval_habitat.sh
```

### 8.3 decay_only 组

```bash
export GPU_IDS="0 1"                       # 与 off 组保持完全一致
export OUTPUT_ROOT=/path/to/output/dart_decay_88_decay_only
export CLIENT_ARGS="--memory_enabled --memory_strategy hierarchical --memory_layers short,mid,long \
  --memory_extractor yolo11 --memory_extractor_model_path $YOLO_MODEL_PATH \
  --memory_extractor_device cpu --memory_extractor_confidence 0.25 --memory_extract_every_n_steps 5 \
  --memory_place_enabled --memory_place_model_path $DINO_MODEL_PATH \
  --memory_place_device cpu --memory_place_extract_every_n_steps 5 \
  --memory_place_similarity_threshold 0.85 \
  --local_graph_enabled --local_graph_scope episode_local \
  --local_graph_prompt_max_chars 320 --local_graph_max_landmarks 2 \
  --local_graph_reactivation_steps 8 \
  --spatial_l3_mode inject --spatial_l3_seg_model_path $YOLO_SEG_MODEL_PATH \
  --spatial_l3_seg_device cpu --spatial_l3_seg_every_n_steps 5 \
  --dart_decay_mode decay_only \
  --dart_decay_rho 0.5 --dart_decay_lambda 0.12 --dart_decay_alpha 0.15 \
  --dart_decay_w_min 0.35 --dart_decay_w_max 1.0"

[ -e "$OUTPUT_ROOT" ] && echo "choose a fresh OUTPUT_ROOT" || bash scripts/eval_habitat.sh
```

### 8.4 运行后核查

```bash
export OUTPUT_ROOT=/path/to/output/dart_decay_88_decay_only

python -m lightnav.habitat.experiment_integrity verify "$OUTPUT_ROOT" \
  --shared "$EPISODE_MANIFEST_PATH" \
  --pilot-selection "$PILOT_MANIFEST_PATH"

python - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines() if line]
keys = {(row["scene_id"], row["habitat_episode_id"]) for row in rows}
summary = json.loads((root / "summary.json").read_text())
assert len(rows) == len(keys) == 88
assert summary["num_episodes"] == 88
snapshot = json.loads((root / "config_snapshot.yaml").read_text())
decay = snapshot["dart_decay_effective"]
assert decay["mode"] == "decay_only"
assert decay["rho"] == 0.5 and decay["decay_lambda"] == 0.12
assert decay["alpha"] == 0.15 and decay["w_min"] == 0.35 and decay["w_max"] == 1.0
print("SR/SPL:", summary["metrics"]["SR_success_rate_pct"], summary["metrics"]["SPL_pct"])
PY
```

每个 shard 还应有 `dart_decay_protocol.json`（decay 组）和 `local_graph_protocol.json`；off 组不应有 `dart_decay_protocol.json`。

---

## 9. 需要统计的实验指标

### 9.1 导航主指标（AutoDL）

- `SR`（Success Rate）
- `SPL`（Success weighted by Path Length）
- `NE`（Navigation Error, m）
- `TL`（Trajectory Length, m）
- `OSR`（Oracle Success Rate）
- 每 Episode 的 paired delta：`decay_only - off`，以及 88 条上的 McNemar（SR）与 bootstrap 95% CI（SPL/NE/TL）
- 分 Scene 的失败/成功案例，尤其“历史事实被 decay 换掉后是否更接近目标”

### 9.2 记忆读取指标（`memory_diagnostics.jsonl`）

- `decay_applied_count` / `decay_skipped_count` 的逐步分布
- `memory_age` 的均值/分位数
- `visit_count` 的均值/分位数
- `novelty` 的均值；`novelty_source=cosine_ema` 与 `initial_prior` 的比例
- `decay_weight` 的均值/分位数；低于 0.5 的比例
- `order_changed` / `text_reordered` 的比例
- `candidate_selection.candidates` 中 selected 的 rank 分布；`decay_only` 与 `off` 的 selected object/evidence 差异
- 实际注入率：`actual_injection`、`actual_historical_injection`、`injected_evidence_ids` 数量
- Prompt 字符/token 增量（`prompt_chars`、`prompt_token_delta`）
- `novelty_prior_count` 高但 `novelty_measured_count=0` 的 Episode，说明该 Scene 的 L1/L3 无特征 novelty，只能解释为 recency/repeat

### 9.3 效率与稳定性

- 每步 `last_model_inference_ms` 的均值/P95（decay 本身不应增加模型推理时间）
- YOLO L1 / YOLO-seg L3 / DINO place 的提取耗时（与 off 组应一致）
- 内存/候选列表大小、`dart_decay` 诊断行大小
- 失败 Episode 的 decay 诊断是否出现 `skip_reason`、负 age、缺失槽位
- 每 Episode reset 后 registry 清空（可在单 Episode smoke 中检查 `DartMemoryDecay.episode`）

### 9.4 建议的最小 smoke 顺序

1. `--episodes 1 --trace_steps` + `decay_only`：检查 `dart_decay` 字段、Prompt 注入、原 Evidence 不变、无异常。
2. 同一 Episode 的 `off` 对照：检查 Prompt/Token 契约；decay 组允许 Prompt 不同，但视频 segment、动作解码接口必须一致。
3. 88 Episode Pilot：先跑 off 与 decay_only 各一遍，合并结果，确认 88 条唯一且 manifest 一致。
4. 再决定是否扩展到 full split；不要用 88 条 Pilot 的 SR 直接宣称 full split 收益。

---

## 10. 交付结论

**已通过（代码级）：**

- DART decay-only 公式、默认参数、age/count/novelty/EMA、上下界。
- 读取侧 candidate admission + final text ordering 的真实使用路径。
- L3 场景中完整事实被替换的 CPU 代码路径验证。
- off/default Prompt 与 token 诊断相等；Episode 隔离；Evidence/几何不可变。
- CLI、runner、launcher 配置与协议快照。
- 39 个新增/修改用例通过（37 个在最后一次 place 计数修改后复跑）；全仓 CPU 套件 1044 passed、1 个既有失败、14 skipped；修改文件 ruff 通过；compileall 通过；`git diff --check` 通过。

**仍待 AutoDL 验证（未通过、未运行）：**

- 真实 SR/SPL/NE/TL/OSR。
- 真实 DINO/YOLO/L3 数据下的 novelty 分布、候选替换频率与导航收益。
- 真实延迟、失败案例、不同 Episode 的方差。
- `decay_only` 是否优于 `off`，以及是否值得进入 full split。

在 AutoDL 结果出来之前，不能把本轮结论写成“SR 提升”或“导航效果已验证”。
