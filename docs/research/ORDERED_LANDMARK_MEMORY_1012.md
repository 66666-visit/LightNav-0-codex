# LightNav-0：Episode 1012 有序地标记忆协作（A/B/C）与根因分析

> 本轮只针对 Habitat Episode 1012 做代码审查、CPU 实现与 CPU 回归。
> 未启动 GPU、未启动 Habitat server、未运行任何 Episode（包括 1012 本身）、
> 未运行 88 Episode、未执行 git add/commit/push。
> D/E/F 的 GPU 端 `prompt_diagnostics.jsonl` / `step_trace.jsonl` 不在本仓库中，
> 本文把"可从代码调用链证明的结论"与"必须由外部日志确认的推断"分开标注。

## 1. 结论摘要

1012 的真实指令（`tests/memory/test_instruction_landmarks.py` 与
`docs/research/INSTANCE_DISAMBIGUATION.md`）：

```text
walk away from sink, turn left, walk towards foot of bed,
exit through doorway on left, turn right and step down two stairs and stop.
```

解析结果（CPU 实际输出）：

| order | landmark_role | phrase | action_phrase | target_object | role |
| --- | --- | --- | --- | --- | --- |
| 1 | source | `sink` | `walk away from` | `sink` | reference/source |
| 2 | intermediate | `foot of bed` | `walk towards` | `bed` | target |
| 3 | transition | `doorway` | `exit through` | `doorway` | target |
| 4 | final | `two stairs` | `step down` | `stairs` | target |

根因不是"Grounder 分数不对"，而是四个事实叠加：

1. **Doorway / Stairs 在 ObjectMap 中的存在性由分割模型词表决定。**
   仓库唯一的分割路径是 `YOLOSegExtractor`（`yolo11n-seg.pt`，COCO 80 类）。
   COCO 没有 `doorway`、`door`、`stairs`、`stair`。因此按仓库默认权重，
   这两个地标**根本不会出现 ObjectNode / Evidence / proposal**。若实际运行
   换用了带 `door`/`stairs` 类的自定义 `.pt`，则 `stairs` 可通过单复数匹配，
   但 `door` 对 `doorway` 会被旧的精确词项匹配判为 `class_mismatch`。
2. **旧渲染器只渲染"有可格式化事实"的地标，且按 target 优先/指令顺序。**
   1012 的 target 顺序是 bed → doorway → stairs；单条
   `SpatialGraphAdapter` 事实行约 161 字符，`[Navigation Memory]` 预算 320
   字符，只能容纳**第一条** bed 事实。于是即使 doorway/stairs 有候选，也会
   被预算丢弃。这与实验报告的"E/F 注入主要集中在 Bed"完全一致。
3. **F 的 geometry fallback 对 sink/doorway/stairs 确实执行了，但没有同类候选。**
   旧代码只记录 `fallback_reason`，不记录"是否尝试"；`injection_provenance`
   只统计真正被选中的 fallback 项，所以看起来像"fallback 没执行 / provenance=0"。
   实际是：bed 的 grounding 成功（不需要 fallback）；sink/doorway/stairs 在
   `full_context.layer_items` 中没有同类别项，`geometry_fallback_item()` 返回
   `None`。本轮新增 `geometry_fallback_attempted` / `geometry_fallback_reason`
   把这一区别显式化。
4. **D/E/F 在关键步的 Prompt 内容可能完全相同。**
   当 ObjectMap 只有 bed 能形成有效事实时，D（全指令 argmax→eligible）、
   E（地标查询）、F（地标查询+fallback）最终都只渲染同一条 bed 适配器事实；
   CPU 复现三者的 `MemoryContext.text` 逐字节相同（180 字符），
   SHA-256 相同。评测使用 greedy（`temperature=0.0`），相同 Prompt + 相同
   视频历史 ⇒ 相同动作。因此"D/E/F 记忆差异没有改变动作"不是模型忽略记忆，
   而是三组可见记忆在关键步没有差异。

**原始 Geometry On-demand 为什么能在 1012 成功？**
代码层面能证明的差异是：B 路径直接渲染
`SpatialGraphAdapter.build()` 的**距离排序候选**（最多 2 条），没有
"逐地标 target 类别匹配"；D/E/F 则按指令 target 顺序逐地标重排。若关键步
B 的最近候选是 stairs（final）而不是 bed，B 的 Prompt 会包含行为关键的
final 事实，而 E/F 的 320 字符被第一条 bed 消耗。

这一差异已用 CPU 复现（`test_original_geometry_can_inject_the_final_stairs_when_it_is_closest`）：
同一组 ObjectNode（stairs 最近 0.5m，bed/sink 10m）下，

- B 的 `[Navigation Memory]` 实际输出 `stairs observed ...`，不含 bed；
- E/F（有 Grounder）实际输出 bed 事实，不含 stairs 事实；
- G 在 320 字符内保留 `"two stairs"` final 语义行，并优先把 stairs 事实
  （以及能放下的 bed 事实）写入事实行。

因此 B 与 E/F 的记忆内容**可以**在关键步不同，且差异正好落在 final
landmark 上。这不是猜测，而是同一份 `SpatialGraphAdapter` 输出在两种排序/
预算策略下的可复现结果。**B 当步实际对象顺序仍需 B 的
`prompt_diagnostics.jsonl` / `step_trace.jsonl` 最终确认**；本仓库没有该
文件，因此本文不把"B 因此成功"写成已由外部日志证明的事实。

## 2. 只读代码审查：七个文件的调用链

### 2.1 `instruction_landmarks.py`

- 真实 1012 指令被解析为 4 个 `InstructionLandmark`，每个 phrase 都有
  原始 `[start, end)` span；`tests/memory/test_instruction_landmarks.py`
  逐项断言 `instruction[start:end] == phrase`。
- 旧字段只有 `role=target|reference` 和 `index`，没有路径角色，也没有
  "第几个地标"的显式字段。`sink` 是 source reference，`foot of bed` /
  `doorway` / `two stairs` 都是 target；旧诊断无法直接看出
  source/intermediate/transition/final。
- 本轮新增 `navigation_role`、`landmark_order`（`to_dict()` 同时输出
  `landmark_role` / `landmark_order`）。角色只由动作短语与顺序推导：
  `away/out` → source；最后一个 target → final；动作含
  `through/enter/exit/pass/cross` → transition；其余 target → intermediate。
  无 Episode ID 分支。

### 2.2 `l3_evidence_selector.py`

- `build_proposal_eligibility()` 把每个 official proposal 映射到
  `persistent_object_id`（即 `score.object_id`）和 `node.evidence_ids`；
  eligibility 四项检查：ObjectNode observed、Evidence 映射 + sidecar、可格式化、
  类别匹配。
- `label_matches_target()` 只做归一化词项包含，旧别名表没有
  `door ↔ doorway`。因此即使检测器输出 `door`，1012 的 `doorway` 查询也会
  `class_mismatch`；`SpatialGraphAdapter` 的 `label_relevant_to_instruction`
  同样不会把 `door` 纳入 `full_context`，导致 fallback 也没有候选。
- 本轮新增结构同义词（`doorway→door`、`staircase/stairway→stair`），并保留
  `dining table` 与 `coffee table` 不互相匹配的边界。
- `geometry_fallback_item()` 旧实现只返回 item/None。本轮新增
  `resolve_geometry_fallback()`，返回 `attempted` + `reason`
  （`fallback_disabled` / `no_landmark_target` /
  `no_formatable_geometry_item` / `no_same_category_geometry_item` /
  `same_category_already_injected`）。旧函数保留，行为不变。

### 2.3 `l3_instance_grounder.py`

- `ViL3DRelGrounder.select()` 把 sidecar 的**全部完整 proposals**送入官方
  Student；`InstanceGroundingResult.proposals` 保留每个 proposal 的
  `proposal_id`、`object_id`、`class_name`、`evidence_ids`、原始
  `logit`/`probability`。没有阈值、没有 top-k、没有重新归一化。
- Argmax 只是 `result.selected_object_id`；真正的注入选择在 manager 的
  eligible 路径。Grounder 本身不能"选中" doorway/stairs：它只能对 sidecar
  提供的 proposals 打分；没有 proposal 就没有分。
- 本轮没有修改 Grounder 的模型、checkpoint、预处理或分数。只把每地标
  proposal/eligible 计数、evidence_ids、fallback 结果带入诊断。

### 2.4 `manager.py`

调用链：

```text
policy.act
  -> prompt_mode = l1_original_only
  -> L1 local graph context is None (no_candidates / budget_excluded)
  -> manager.get_geometry_on_demand_context
     -> L3QueryController.prepare_query
        gate: l1_status in L1_EMPTY_STATUSES
        dedupe: (geometry_map_signature, query) unchanged -> return None
     -> _grounded_geometry_context (eligible)
        SpatialGraphAdapter.build(UNBOUNDED, max_landmarks=len(objects))
          -> full_context: only observed, evidence-valid, instruction-relevant items
        for each landmark:
          grounder.select
          build_proposal_eligibility(class_match=target_object)
          choose_first_eligible
          if none: resolve_geometry_fallback (same-category only)
        old branch: render facts in target-first / instruction order
        new branch: render ordered plan (skeleton + real facts)
     -> MemoryContext
  -> inject_memory_context (after </navigation_task>)
  -> engine.generate (greedy)
```

关键点：

- `prepare_query()` 在 `(map signature, query)` 未变化时直接
  `skipped_reason="map_unchanged_since_last_query"`（`l3_on_demand.py:208-217`）。
  L3 块不是跨步持久的；某一步若被 dedupe 或 L1 非空，该步 Prompt 没有 L3。
  这可以解释 Step 38 前后"记忆时有时无"：必须在 `step_trace.jsonl` 中看
  `query_executed` / `skipped_reason`，不能只看注入总数。
- `policy.py:564-573` 只在 `context is None` 时调用 L3。L1 built 的步不会
  触发 L3。
- `_grounded_geometry_context` 的 `full_context` 用
  `max_landmarks=max(1, len(objects))`，不是 top-k；但它仍受
  `SpatialGraphAdapter` 的 observed/evidence/confidence/标签相关性过滤。
- F 的 fallback 只在 `chosen is None` 时进入；bed 的 grounding 成功时
  fallback 不会执行。对 sink/doorway/stairs，`resolve_geometry_fallback`
  返回 `no_same_category_geometry_item`，不会跨类别取 bed。
- 旧渲染 `render_context(selected_items, max_chars=320)` 对整行做预算：
  bed 行约 161 字符 + header 19 = 180；第二条行再加约 161 就超过 320，
  因此 doorway/stairs 事实被 `continue` 跳过。CPU 复现见 §6。

### 2.5 `spatial_graph_adapter.py`

- `SpatialGraphAdapter.build()` 只输出：
  - `geometry_status == observed`；
  - 全部 Evidence observed 且 `confidence >= min_injection_confidence`；
  - 有合法、早于当前 step 的 `observed_from` 历史；
  - 标签与指令相关（eligible 路径 `alias_aware=True`）。
- 输出按 `distance_m` 升序，最多 `max_landmarks` 条；每条行包含真实
  `center_world`、当前直线距离/方位、观测次数、检测器来源，并明确写
  `passage unknown`。
- 这就是"实际空间事实"的唯一来源。本轮没有修改 adapter 的排序、几何、
  阈值或文本；有序记忆只复用它的 item。

### 2.6 `prompt_diagnostics.py`（实际实现为 `habitat/prompt_artifacts.py`）

- `prompt_diagnostic_row()` 把 policy 的 `spatial_l3.l3_query` 展平为
  `l3_*` 字段。旧字段包含 `grounding_query`、proposal 数、fallback_reason、
  actual_injection/evidence_ids，但没有 `landmark_role/order/phrase`、
  `eligible_candidate_count`、`geometry_fallback_attempted`、
  `ordered_landmarks`。
- 本轮新增上述展平字段；完整的 `ordered_landmarks` 行同时保留在
  `spatial_l3.l3_query.ordered_landmarks` 与 `l3_ordered_landmarks`。

### 2.7 `step_trace.py`（实际实现为 `habitat/runner.py` 的 trace 写入）

- 原有字段：`position`、`rotation`、`action`、`decoded_action`、
  `decoded_action_label`、`distance_to_goal_pre/post`、
  `prompt_hash_before/after`、`actual_injection`、
  `actual_injected_evidence_ids`、`geometry_fallback_reason`、
  `memory_diagnostic`、`model_input_diagnostic`。
- 本轮新增：`landmark_role/order/phrase`、`grounder_proposal_count`、
  `eligible_candidate_count`、`geometry_fallback_attempted`、
  `selection_source`、`landmark_rows`、`decoded_stop_decision`；env.step 后
  新增 `stop_called`、`stop_decision`。
- `stop_decision` 只读取 evaluator 已有的 `is_stop_called`；`decoded_stop_decision`
  只读取 policy 已产生的 `predicted_traj`。二者都不进入动作接口。

## 3. 算法改进 A：有序地标表示

- 保留原始指令，不重写、不截断。
- 每个地标新增：
  - `navigation_role`：source / intermediate / transition / final / reference；
  - `landmark_order`：原始顺序，1-based；
  - 既有 `target_phrase`、`action_phrase`、`target_span`、`action_span`、
    `grounding_query`、`grounding_query_span` 全部保留。
- 角色规则（只读原文，无 Episode ID、无物体词表）：

```text
if role == reference and relation_kind == source: source
elif index == last target index:                     final
elif action_phrase contains through/enter/exit/
     pass/cross/across:                              transition
elif role == target:                                 intermediate
else:                                                reference
```

1012 输出与 §1 表格一致；`tests/memory/test_ordered_landmark_memory.py`
还用交换子句的通用指令证明角色随文本变化，而不是随 Episode ID 变化。

## 4. 算法改进 B：逐地标 Evidence 检索

保持既有 eligible selection（168 已证明有效）不变：

```text
all ViL3DRel proposals (original logits/probabilities, no renormalization)
  -> proposal_id -> persistent_object_id -> evidence_ids
  -> eligibility: ObjectNode + Evidence + formatable + class_match
  -> choose_first_eligible (official score order, no threshold, no top-k)
  -> if none: resolve_geometry_fallback(same target category only)
```

新增/显式化：

- 每个地标独立记录 `grounder_proposal_count`、`eligible_candidate_count`、
  `selected_object_id`、`evidence_ids`、`selection_source`；
- `geometry_fallback_attempted` + `geometry_fallback_reason` 区分：
  - fallback 未启用；
  - 全指令查询无 landmark target；
  - 无任何可格式化 item；
  - 无同类别 item；
  - 同类别 item 已注入；
  - 选中同类别 item。
- 结构同义词只扩展**同一地标类别**：`door↔doorway`、
  `staircase/stairway↔stairs`；不扩展到 window/table 等其他类别。
- 不新增置信度阈值、不硬编码 top-k、不改变 ViL3DRel checkpoint。

## 4.5 算法流程（A → B → C）

```text
original instruction (never rewritten)
        │
        ▼
[A] extractive ordered parser (once per Episode)
    sink(source) -> foot of bed(intermediate)
    -> doorway(transition) -> two stairs(final)
    every phrase keeps its exact [start,end) span
        │
        ▼
[gate] L1 context empty?
    no  -> keep L1 Original, no L3 block
    yes -> L3QueryController.prepare_query
             (l1_empty + historical geometry + map/query dedupe)
        │
        ▼
[B] SpatialGraphAdapter.build (existing, unchanged)
    observed ObjectNodes + observed Evidence + observed_from history
    + confidence floor + instruction relevance
        │
        ├── full_context items_by_object (all formatable facts)
        │
        ▼
    for each landmark in instruction order:
        grounder.select(all complete proposals)
          -> official logits/probabilities (unchanged)
          -> build_proposal_eligibility
             (ObjectNode + Evidence + formatable + class_match)
          -> choose_first_eligible (official score order)
          -> if none: resolve_geometry_fallback
             (same category only; attempted + reason recorded)
        │
        ▼
[C] render_ordered_landmark_memory
    try per-landmark skeleton lines (all roles/phrases/order)
      reserve skeleton bytes
      enrich facts by final > transition > intermediate > source > reference
      if >=1 fact fits: output per-line plan
      else: output one ordered "Plan: ..." line + verified fact lines
    no fact line -> return None (no fake evidence injection)
        │
        ▼
MemoryContext -> inject after </navigation_task>
(original prompt/template/video/action/STOP untouched)
        │
        ▼
greedy engine.generate -> unchanged action decode
```

## 5. 算法改进 C：有序记忆组织

新增 `ordered_landmark_memory.py`，只在既有 `[Navigation Memory]` 注入位置
生成内容，不修改 LightNav 模板：

- 先为**所有**地标保留一行紧凑语义骨架：
  `{order} {role} "{verbatim phrase}": no 3D evidence.`
- 再按 `final > transition > intermediate > source > reference` 的顺序，
  用剩余字符把有真实 item 的地标替换为事实行：
  `{order} {role} "{phrase}": {label} at world (x,y,z)m; now d m line,b deg;
   n obs; {selection_source}.`
- 事实行只从选中 `LayerMemoryItem.description`/metadata 读取；没有 item 的
  地标永远不出现 `world (...)`、距离或方位。
- 输出行顺序始终是原始指令顺序；预算只决定"哪些行从语义骨架升级为事实行"。
- `render_ordered_landmark_memory()` 做整行字符预算，保证
  `len(text) <= max_chars`；policy 原有的 Token Budget/overflow 逻辑不变。
- 至少一条事实行进入 Prompt 才返回 context；否则返回 None，避免把纯语义
  文本伪装成"evidence injection"。
- 当地标很多（例如 168 的 6 个地标）时，逐一骨架行会占满 320 字符；渲染器
  自动退回"单行有序 Plan + 事实行"形式，仍然保留全部 phrase/role/order，
  并把剩余预算优先给 final 事实。168 的 eligible selection 因此在 G 组仍
  完整保留（CPU 测试 `test_168_eligible_selection_is_retained_by_ordered_memory`）。

1012 在 320 字符预算下的实际输出（CPU）：

只有 bed 时（239 字符）：

```text
[Navigation Memory]
1 source "sink": no 3D evidence.
2 intermediate "foot of bed": bed at world (2.0,0.0,1.0)m; now 2.4m line,-117deg; 1 obs; vil3dref_grounding.
3 transition "doorway": no 3D evidence.
4 final "two stairs": no 3D evidence.
```

有 bed/door/stairs 时（306 字符，final 优先）：

```text
[Navigation Memory]
1 source "sink": no 3D evidence.
2 intermediate "foot of bed": no 3D evidence.
3 transition "doorway": door at world (3.0,0.0,1.0)m; now 3.3m line,-108deg; 1 obs; vil3dref_grounding.
4 final "two stairs": stairs at world (4.0,0.0,1.0)m; now 4.2m line,-104deg; 1 obs; vil3dref_grounding.
```

同地图旧渲染器输出（180 字符，只有 bed）：

```text
[Navigation Memory]
Spatial history: bed observed at world (2.0,0.0,1.0) m; now 2.4 m straight-line, bearing -117 deg (left +); 1 observations; source YOLO11n-seg; passage unknown.
```

这正是"E/F 注入主要是 Bed，doorway/stairs 没有有效 Evidence 注入"的
代码级复现。

## 6. CPU 测试结果

命令（本机 3.11 venv；`make test` 等价入口在具备依赖的环境执行）：

```bash
PYTHONPATH=src /home/lilixing/projects/LightNav-0-codex/.venv311/bin/python \
  -m pytest -q tests/memory/test_ordered_landmark_memory.py \
  tests/memory/test_instruction_landmarks.py \
  tests/memory/test_instance_disambiguation.py \
  tests/memory/test_l3_instance_grounding_manager.py \
  tests/habitat/test_geometry_on_demand_cli.py \
  tests/habitat/test_prompt_artifacts.py \
  tests/habitat/test_runner.py \
  tests/habitat/test_instance_disambiguation_1012_launcher.py
```

本轮实际结果：

```text
tests/memory/test_ordered_landmark_memory.py                    17 passed
tests/memory/test_instruction_landmarks.py                      17 passed
tests/memory/test_instance_disambiguation.py                    19 passed
tests/memory/test_l3_instance_grounding_manager.py              10 passed
tests/habitat/test_geometry_on_demand_cli.py                    12 passed
tests/habitat/test_prompt_artifacts.py                          10 passed
tests/habitat/test_runner.py                                    49 passed, 5 skipped
tests/habitat/test_instance_disambiguation_1012_launcher.py      2 passed
总计                                                            136 passed, 5 skipped
```

完整 CPU 门禁（`pytest -q -m "not gpu"`）：`1316 passed, 15 skipped,
99 subtests passed`，唯一失败是仓库既有、与本次无关的
`tests/memory/test_rgm_place.py::test_initial_reference_and_global_ambiguity`
（`ambiguous_global_candidates` vs `unresolved_competing_candidate`，
`docs/research/INSTANCE_DISAMBIGUATION.md` 已记录其在干净 HEAD 同样失败）。

关键回归断言：

- 1012 角色/顺序/phrase/action span 逐字正确；
- 同一地图、stairs 最近时：B 输出 stairs 事实，E/F 输出 bed 事实，G 保留
  final stairs 语义；
- D/E/F 在"只有 bed"场景下 `MemoryContext.text` 逐字节相同、SHA-256 相同；
- G 与 D/E/F 文本不同，且包含 `"two stairs"` 语义行；
- 旧渲染器只出 bed；G 在 320 字符内把 final/transition 事实升级，bed 保留语义行；
- `door` 可用于 `doorway`（结构同义词），但不会跨到 window/table；
- 无事实行的行不含 `world (` / ` line,`；
- 320/300/260 字符预算内 `len(text) <= budget`；
- `trim_context_for_overflow` 连续丢弃 3 条后，final `"two stairs"` 仍保留；
- Prompt 注入 roundtrip 通过，删除注入块恢复原始 Prompt，video segments 不变；
- `prepare_query` dedupe 的未执行步为每个地标记录
  `query_executed=False, query_skip_reason=map_unchanged_since_last_query`。

### 6.1 用既有 D/E/F 日志复核 Step 38

不需要重新跑 GPU；对每个组现有的 `step_trace.jsonl`：

```bash
python3 - <<'PY' path/to/step_trace.jsonl
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    if not 30 <= int(row.get("step", -1)) <= 45:
        continue
    spatial = (row.get("memory_diagnostic") or {}).get("spatial_l3") or {}
    l3 = spatial.get("l3_query") or {}
    rows = l3.get("ordered_landmarks") or l3.get("landmark_rows") or []
    final = next((r for r in rows if r.get("landmark_role") == "final"), {})
    print(
        row["step"],
        "dtg_pre", row.get("distance_to_goal_pre"),
        "dtg_post", row.get("distance_to_goal_post"),
        "action", row.get("decoded_action_label"),
        "stop", row.get("stop_decision"),
        "inject", row.get("actual_injection"),
        "query", l3.get("query_executed"),
        "skip", l3.get("skipped_reason"),
        "final_fact", final.get("spatial_fact_included"),
        "final_ev", final.get("actual_injected_evidence_ids"),
        "hash", row.get("prompt_hash_after"),
    )
PY
```

判读规则：

- Step 38 附近 `distance_to_goal_post <= 3.0` 且 `stop_decision` 一直是
  `continue`：模型确实进入了成功半径但未 STOP。
- `query_executed=False` 或 `actual_injection=False`：该步根本没有 L3 块，
  记忆无法影响该步动作（dedupe / L1 non-empty / 预算排除）。
- `actual_injection=True` 但 `final_fact=False`：L3 块存在，但 final 地标
  没有真实事实进入 Prompt；这是旧渲染器 bed-first 预算的典型结果。
- D/E/F 三组在 Step 38 的 `prompt_hash_after` 相同：三组 Prompt 相同，
  greedy 动作必然相同，可直接证明"记忆差异没有改变动作"。

## 7. Prompt 不变性

- 未修改 `src/lightnav/prompts.py`、`VLN_TRAJ_PROMPT_TEMPLATE`、
  system prompt、原始 instruction、视频历史组织、模型输入格式、动作解码、
  STOP 接口、LightNav/ViL3DRel checkpoint。
- 有序记忆仍只插入 `</navigation_task>` 之后，`inject_memory_context()` 的
  roundtrip 审计通过；删除注入块可逐字节恢复原始 human prompt。
- 既有冻结 hash 测试继续通过：
  - 1080 memory-disabled prompt:
    `ec014d7d9799597312e036345b19561e297f8c2d44d337e237597478e14dacfb`
  - memory-disabled VLN template:
    `d4538cd84b1bdcaf5fca1019c4035ceca88cd2a31ac49c6d4469c80391681b90`
- D/E/F 默认行为不变：`grounder_ordered_landmark_memory=False` 时仍走旧
  `render_context` 分支；现有 D/E/F 回归全部通过。

## 8. 1012 评测命令（本轮未执行）

新增 `scripts/run_instance_disambiguation_1012.sh`，只解析并运行 1012，
不启动 88 Episode。CPU dry-run：

```bash
export MODEL_PATH=/path/to/LightNav-0
export YOLO_MODEL_PATH=/path/to/yolo11n.pt
export SEG_MODEL_PATH=/path/to/yolo11n-seg.pt
export GROUNDER_CHECKPOINT=/path/to/nr3d/ckpts/model_epoch_96.pt
export GROUNDER_BERT_VOCAB=/path/to/bert-base-uncased/vocab.txt
export EPISODE_MANIFEST_PATH=/path/to/manifest_with_1012.json
export DATA_PATH=/path/to/data
export SCENES_DIR=/path/to/scenes
export CLIENT_PYTHON=/path/to/inference/venv/bin/python
export OUTPUT_ROOT=/path/to/output/instance_disambiguation_1012_dry
DRY_RUN=1 bash scripts/run_instance_disambiguation_1012.sh
```

正式 1012（GPU 由人类启动；本轮未执行）：

```bash
export MODEL_PATH=/path/to/LightNav-0
export YOLO_MODEL_PATH=/path/to/yolo11n.pt
export SEG_MODEL_PATH=/path/to/yolo11n-seg.pt
export GROUNDER_CHECKPOINT=/path/to/nr3d/ckpts/model_epoch_96.pt
export GROUNDER_BERT_VOCAB=/path/to/bert-base-uncased/vocab.txt
export EPISODE_MANIFEST_PATH=/path/to/manifest_with_1012.json
export DATA_PATH=/path/to/data
export SCENES_DIR=/path/to/scenes
export CLIENT_PYTHON=/path/to/inference/venv/bin/python
export OUTPUT_ROOT=/path/to/output/instance_disambiguation_1012
bash scripts/run_instance_disambiguation_1012.sh
```

等价底层命令（脚本内部实际执行）：

```bash
SERVER_ARGS='--single-scene-id <1012 scene_id> --single-episode-id 1012' \
EPISODES=1 MAX_STEPS=500 NO_FORCE_STOP=1 SEED=42 \
OUTPUT_ROOT='<...>/ordered_landmarks' \
CLIENT_ARGS='--trace_steps --no_force_stop --verbose \
  --memory_enabled --memory_strategy hierarchical \
  --memory_layers short,mid,long --memory_top_k 5 --memory_max_context_chars 600 \
  --memory_extractor yolo11 --memory_extractor_model_path <yolo11n.pt> \
  --memory_extractor_device cpu --memory_extractor_confidence 0.25 \
  --memory_extract_every_n_steps 5 \
  --local_graph_enabled --local_graph_scope episode_local \
  --local_graph_prompt_max_chars 320 --local_graph_max_landmarks 2 \
  --local_graph_reactivation_steps 8 \
  --memory_prompt_mode l1_original_only \
  --spatial_l3_mode geometry_on_demand \
  --spatial_l3_seg_model_path <yolo11n-seg.pt> --spatial_l3_seg_device cpu \
  --spatial_l3_seg_confidence 0.25 --spatial_l3_seg_every_n_steps 5 \
  --spatial_l3_evidence_min_iou 0.5 --spatial_l3_evidence_ambiguity_margin 0.1 \
  --spatial_l3_min_injection_confidence 0.5 --spatial_l3_rgb_sidecar \
  --spatial_l3_instance_grounder \
  --spatial_l3_grounder_checkpoint <student.pt> \
  --spatial_l3_grounder_config <nr3d_gtlabelpcd_mix_model.yaml> \
  --spatial_l3_grounder_device cuda \
  --spatial_l3_grounder_bert_vocab <vocab.txt> \
  --spatial_l3_grounder_seed 0 \
  --spatial_l3_grounder_selection eligible \
  --spatial_l3_grounder_landmark_queries \
  --spatial_l3_grounder_geometry_fallback \
  --spatial_l3_grounder_ordered_landmarks \
  --prompt_measurement_enabled' \
bash scripts/eval_habitat.sh
```

## 9. 预期注入内容及 Evidence 来源

### 9.1 默认 yolo11n-seg（COCO）实际可能的情况

COCO 只能检测到 1012 中的 `sink` / `bed`（如果画面出现且证据合格）。
预期 G 组：

```text
[Navigation Memory]
1 source "sink": no 3D evidence.                # 或 sink 事实行（若 sink 合格）
2 intermediate "foot of bed": <bed 的真实事实>  # 仅当 bed 被选中且预算允许
3 transition "doorway": no 3D evidence.
4 final "two stairs": no 3D evidence.
```

- `bed` 事实的 Evidence 来源：`vil3dref_grounding`（ViL3DRel eligible 选中），
  或 `geometry_fallback`（Grounding 无有效项但 adapter 有同类别 bed item）。
- `doorway`/`stairs` 不会出现坐标；`selection_source=None`，
  `geometry_fallback_attempted=True`，
  `geometry_fallback_reason="no_same_category_geometry_item"`（如果连
  `door`/`stairs` ObjectNode 都没有）。
- 如果 1012 的关键步没有触发 L3（`query_executed=False`），诊断会明确写
  `skipped_reason`，step trace 中 `actual_injection=False`。

### 9.2 自定义分割模型包含 door/stairs 时

- `door` 现在可通过结构同义词匹配 `doorway`，`stairs` 可通过单复数匹配
  `two stairs`。
- 如果它们满足 observed/Evidence/observed_from/confidence 条件，预期：
  - `doorway`：`selection_source="vil3dref_grounding"`（或
    `geometry_fallback`），`evidence_ids=[...]`；
  - `two stairs`：同上；
  - 320 字符预算下，final stairs 事实优先于 bed 事实进入 Prompt；
    bed 仍保留 `2 intermediate "foot of bed": no 3D evidence.` 语义行。
- 如果它们不满足条件，诊断会逐项说明是 `class_mismatch`、
  `missing_evidence_mapping`、`not_formatable`、`no_geometry_candidate`
  还是预算排除，不再出现"fallback provenance=0 但原因未知"。

## 10. 明确未证明/未执行的事项

- 未启动 GPU、未加载 checkpoint、未运行 Habitat、未跑 1012 或 88 Episode。
- D/E/F 的 GPU 端日志不在仓库，本文没有伪造其 `prompt_hash`、对象列表或
  Step 38 轨迹。Step 38 的最终解释需要在正式 1012 运行后用
  `step_trace.jsonl` 的 `query_executed`、`actual_injection`、
  `landmark_rows`、`stop_decision` 和 `distance_to_goal_*` 验证。
- 结构同义词只能解决"检测器输出 door、指令说 doorway"的类别匹配；
  如果 COCO 根本没有 door/stairs，任何记忆算法都不能凭空生成 3D Evidence。
  真正的 doorway/stairs Evidence 需要换用带这些类别的分割模型（例如
  ScanNet200 类词表），本轮未新增模型或依赖。
- 本轮没有 commit/push，也没有修改 `artifacts/` 或
  `references/vil3dref/`。

## 11. 修改文件清单

新增：

- `src/lightnav/memory/ordered_landmark_memory.py`：有序 Plan/事实行渲染、
  整行字符预算、compact plan fallback、`LayerMemoryItem` 构建。
- `scripts/run_instance_disambiguation_1012.sh`：只跑 1012 的评测入口与
  CPU dry-run。
- `tests/memory/test_ordered_landmark_memory.py`：A/B/C 的 CPU 回归。
- `tests/habitat/test_instance_disambiguation_1012_launcher.py`：1012
  launcher 的 CPU dry-run。
- `docs/research/ORDERED_LANDMARK_MEMORY_1012.md`：本文。

修改：

- `src/lightnav/memory/instruction_landmarks.py`：新增
  `navigation_role` / `landmark_order` 与纯文本角色推导。
- `src/lightnav/memory/l3_evidence_selector.py`：结构同义词
  `door↔doorway`、`staircase/stairway↔stairs`；新增
  `resolve_geometry_fallback()` 的 attempted/reason 诊断。
- `src/lightnav/memory/spatial_l3_config.py`：
  `grounder_ordered_landmark_memory` 开关及契约校验。
- `src/lightnav/memory/manager.py`：逐地标诊断、fallback 原因、有序记忆
  分支、未执行查询的每地标占位行。
- `src/lightnav/cli/eval_habitat.py`：
  `--spatial_l3_grounder_ordered_landmarks`。
- `src/lightnav/habitat/prompt_artifacts.py`：新增字段展平。
- `src/lightnav/habitat/runner.py`：step trace 新字段与 STOP 决策。
- `docs/CONFIGURATION.md`：新 flag 文档。
- `tests/habitat/test_geometry_on_demand_cli.py`、`test_prompt_artifacts.py`、
  `test_runner.py`：新字段/开关回归。

未修改（按约束）：

- `src/lightnav/prompts.py` / LightNav system prompt；
- 原始 Navigation Instruction；
- 视频历史组织、模型输入格式；
- 动作解码/动作接口、STOP 接口；
- LightNav-0 / ViL3DRel checkpoint；
- L1 Original 实现；
- `artifacts/`、`references/vil3dref/`；
- 没有新增 Python 依赖或外部 API。
