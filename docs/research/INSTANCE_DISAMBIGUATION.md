# Instance Disambiguation：真实指令、联合筛选与消融验收

> 分支：`experiment/l3-instance-disambiguation`，基准 HEAD `d659e57`。
> 本轮只做第二轮针对性修正与 CPU 验收：未启动 GPU、未跑 Habitat、未跑 88
> Episode、未 commit/push。`artifacts/` 与 `references/vil3dref/` 未修改。

## 1. 上一轮指令错误：报告错误还是实现/测试错误？

**是测试夹具与报告同时错误，不是报告笔误。**

上一轮 `test_instruction_landmarks.py` / `test_instance_disambiguation.py`
使用了两条自行拼造的指令：

- 1012：`"Go to the sink at the foot of the bed, past the doorway on the left, and up the two stairs."`
- 168：`"Walk past the double doors, by the dining table, ..."`

解析器对这两条拼造文本给出的 `at the foot of` / `by the dining table` /
`up` 是**对拼造输入的正确输出**，但这些文本不是 Episode 的真实指令。上一份
报告直接引用了这些测试夹具，因此报告中的 relation/action 也不真实。

本轮已用四条真实指令重写解析器测试与 Episode 回归：

```text
487  : Turn around and walk out of the room. Wait at the top of the stairs.
1080 : Go forward toward the windows. Go toward the the far couch,
       Stop next to the couch, in front of the windows.
1012 : walk away from sink, turn left, walk towards foot of bed,
       exit through doorway on left, turn right and step down
       two stairs and stop.
168  : Walk through double doors into the house.
       Continue around the dining table and through the entry way
       to the next room. Walk up to the couch and armchairs
       surrounding a coffee table.
```

解析器没有任何 Episode ID 分支；它只按原文 span、词性/关系词和运动方向规则
工作。测试 `test_no_landmark_hardcoding_by_episode_id` 通过交换两条子句验证
输出随原文变化，而不是随 ID 变化。

## 2. 真实指令解析结果（CPU 实际输出）

每个 phrase 都有原始 `[start, end)` span；测试逐项断言
`instruction[start:end] == phrase`。`target_object_span` 也精确指向目标词。

| Episode | role / relation | target_phrase | target_object | action_phrase | reference |
| --- | --- | --- | --- | --- | --- |
| 487 | target | `stairs` | `stairs` | `Wait at the top of` | — |
| 1080 | target | `far couch` | `couch` | `Go toward` | `next to the couch`, `in front of the windows` |
| 1080 | reference / waypoint_reference | `windows` | `windows` | — | `in front of the windows`（host=couch） |
| 1080 | reference / locative | `windows` | `windows` | — | `in front of the windows`（host=couch） |
| 1012 | reference / source | `sink` | `sink` | `walk away from` | `away from sink` |
| 1012 | target / destination_descriptor | `foot of bed` | `bed` | `walk towards` | — |
| 1012 | target | `doorway` | `doorway` | `exit through` | — |
| 1012 | target | `two stairs` | `stairs` | `step down` | — |
| 168 | target | `double doors` | `double doors` | `Walk through` | — |
| 168 | target | `dining table` | `dining table` | `Continue around` | — |
| 168 | target | `entry way` | `entry way` | `through` | — |
| 168 | target | `couch` | `couch` | `Walk up to` | — |
| 168 | target | `armchairs` | `armchairs` | — | `surrounding a coffee table` |
| 168 | reference / locative | `coffee table` | `coffee table` | — | `surrounding a coffee table`（host=armchairs） |

关键验收：

- **487**：`room`、`out` 等泛化地点/方向粒子不成为目标；只保留 `stairs`。
- **1080**：`target_objects == ["couch"]`；`windows` 是 reference，
  不是 couch 查询的 target。couch 的 grounding query 是原文连续 span
  `far couch, Stop next to the couch, in front of the windows`；
  不会把 windows 当成目标。
- **1012**：保留 `walk away from`、`walk towards`、`exit through`、
  `step down`；不出现 `up`，动作方向没有反转；`sink` 是 source reference，
  `bed/doorway/stairs` 是 target。
- **168**：没有 `by the dining table`；真实关系是
  `Continue around the dining table`、`through the entry way`、
  `Walk up to the couch`、`surrounding a coffee table`。
  `armchairs` 与 `coffee table` 同时保留。
- 所有 reference phrase 都是原文精确子串；没有凭空添加关系。

## 3. 联合筛选算法审查

`src/lightnav/memory/l3_evidence_selector.py`：

```text
proposal_id -> persistent_object_id -> evidence_ids -> formatable LayerMemoryItem
```

每次 forward 仍把**全部 proposals**送入官方 Student。官方 logits/softmax
原样保留，不重新归一化。每个 proposal 记录：

- `proposal_id`、`proposal_object_id`、`persistent_object_id`、`evidence_ids`；
- `grounding_logit`、`grounding_probability`；
- `global_grounding_rank`：全部 proposals 中的官方分数排名；
- `eligible_grounding_rank`：eligible 子集中的排名；
- `object_node_valid`、`evidence_mapping_valid`、`formatable`、`class_match`；
- `rejection_reason`（`missing_object_node` / `missing_evidence_mapping` /
  `not_formatable` / `class_mismatch`）。

选择规则：在 `eligible` 集合中按官方分数取第一个未注入候选；
**全局 argmax 不可注入时不会丢弃其他有效候选**。没有新增置信度阈值，也没有
硬编码 top-k。

诊断新增：

- `global_grounding_rank`、`eligible_grounding_rank`；
- `original_top1_probability`、`selected_probability`；
- `selected_is_global_top1`；
- `rejection_reason`；
- `selected_is_global_top1_flags` / `selected_global_grounding_ranks` /
  `selected_eligible_grounding_ranks`（逐条注入）。

测试 `test_diagnostics_never_renormalize_probabilities` 使用总和为 0.9 的
测试概率，断言 `original_top1_probability == 0.6`、
`selected_probability == 0.2`，没有被重新归一化。

### 类别别名

`label_matches_target` 使用别名 + 单复数归一化 + 双向词项包含：

- `couch ↔ sofa`；
- `chair ↔ armchair`；
- 泛化 `table` 可匹配 `dining table` / `coffee table`；
- `dining table` 与 `coffee table` **不互相匹配**。

在 `grounder_selection_mode=eligible` 下，`SpatialGraphAdapter.build` 的标签
相关性判断使用同一别名函数（`alias_aware=True`），避免 detector 输出
`chair`/`table` 时被原始字符串交集丢弃。A/B/C 路径仍使用原始字符串交集，
旧实验不被静默改变。

## 4. Geometry fallback 验证

### 4.1 完全不可用时与原 Geometry 路径一致

`test_geometry_fallback_equals_original_geometry_path`：

- control：`spatial_l3_mode=geometry_on_demand`，无 sidecar、无 Grounder；
- fallback：eligible + 全指令 + Grounder 全部拒绝；
- 断言两者 `context.text`、`object_id`、`description` 完全一致；
- fallback 使用原 `SpatialGraphAdapter.build(max_chars, max_landmarks)`，
  不重写检索、排序或格式器。

### 4.2 unbounded full_context 没有绕过过滤器

`unbounded` 只解除内部字符上限并允许全部对象，不绕过：

- `geometry_status == observed`；
- 全部 Evidence observed 且 `confidence >= min_injection_confidence`；
- 合法 `observed_from` 历史；
- 标签相关性；
- 最终 Prompt 字符/Token Budget。

测试 `test_unbounded_full_context_does_not_bypass_history_or_budget`：

- 最高分对象没有 `observed_from` 历史 → 仍不可格式化，不被注入；
- `max_chars=40` → 即使内部候选存在，最终 `prompt_budget_excluded`，
  `actual_injection=False`。

### 4.3 多目标时不会重新引入错误类别

- 1012 只有 bed 时，sink source 查询不会 fallback 到 bed；
  `test_geometry_fallback_never_reintroduces_another_target_class`。
- 多目标 fallback 只允许该地标自己的 `label_matches_target` 类别；
  有 landmark 查询时**不做**全局 geometry fallback，避免把其他地标对象
  注入到错误目标。全局 bounded fallback 只用于 full-instruction 查询。
- 每条注入事实记录 `matches[].selection_source` 与
  `injection_provenance[]`：`vil3dref_grounding` 或 `geometry_fallback`；
  L1 Original 由 policy 的 `context_source`/`injected_evidence_ids` 标识。
  没有新增 Prompt 角色或标记。

## 5. 消融开关（A–F）

| 组 | 语义 | flags |
| --- | --- | --- |
| A | L1 Original | `--memory_prompt_mode l1_original_only`，`--spatial_l3_mode off` |
| B | 原 Geometry On-demand | `--spatial_l3_mode geometry_on_demand`，不加 sidecar/grounder |
| C | 原 ViL3DRel 实例消歧（d659e57） | sidecar + grounder，`--spatial_l3_grounder_selection argmax`（默认），landmark off（默认），fallback off（默认） |
| D | 联合候选筛选 | C + `--spatial_l3_grounder_selection eligible` |
| E | 联合筛选 + 多地标查询 | D + `--spatial_l3_grounder_landmark_queries` |
| F | 联合筛选 + 多地标 + Geometry fallback | E + `--spatial_l3_grounder_geometry_fallback` |

默认值故意是 C，保证不显式加 flag 时不会把旧 ViL3DRel 实验静默改成新算法。
`landmark_queries` / `geometry_fallback` 在 config 层强制要求
`selection=eligible`，避免“设置了但被静默忽略”。

`tests/memory/test_instance_disambiguation.py` 与
`tests/habitat/test_geometry_on_demand_cli.py` 覆盖 C/D/E/F 的独立行为；
`build_prompt_protocol` 记录实际 selection/landmark/fallback 语义。

## 6. 逐步导航诊断

runner 已有 `--trace_steps` 的 `<output_dir>/step_trace.jsonl`，本轮扩展它，
没有另造第二套轨迹文件。新增/明确的顶层字段：

```text
scene_id, habitat_episode_id, rollout_idx, step,
position, rotation, action, decoded_action, decoded_action_label,
collision,
memory_text, actual_injection, actual_injected_evidence_ids,
prompt_hash_before, prompt_hash_after,
grounding_query, selected_object_ids, geometry_fallback_reason,
distance_to_goal_pre, distance_to_goal_post
```

- `position`/`rotation` 只来自 server 已提供的 `agent_pose`；没有就写 `null`，
  不估计。
- `collision` 只读取 server `info` 的 `collisions`/`collision`，没有就 `null`。
- `action` 是实际发给环境的动作；`decoded_action` 沿用原字段，
  `decoded_action_label` 是人类可读形式。
- `distance_to_goal_*` 仍是 evaluator-only 离线字段，绝不进入
  policy、Grounder 或记忆选择。
- 诊断在 `policy.act` 之后构建；开关不改变导航策略。

`tests/habitat/test_runner.py::test_single_episode_trace_records_model_input_and_evaluator_state`
覆盖这些字段。

## 7. 四 Episode launcher 与 dry-run

`scripts/run_instance_disambiguation_4.sh`：

- 按 `(scene basename, exact Habitat episode ID, rollout_idx)` 对齐 manifest；
- 四个 ID 字符串精确匹配；`episode_001` 不会被解释成 Habitat ID `1`；
- 不做重采样，不猜测 AutoDL 路径（关键路径用 `${VAR:?}` 强制显式提供）；
- 逐条使用 `--single-scene-id/--single-episode-id`，`EPISODES=1`，
  `MAX_STEPS=500`，`NO_FORCE_STOP=1`，`SEED=42`；
- 输出 `resolved_config.json` 和 `four_episode_selection.json`；
- `DRY_RUN=1` 只做资产校验、manifest 对齐、`bash -n` 和 client import，
  不启动 server/GPU/client；
- 正式 `eval_habitat.sh` 只在 `SERVER_ARGS` 含 `--single-scene-id` 时省略
  `--split-id/--split-num`（Habitat server 不允许二者同时出现）。

CPU dry-run 测试：

- `tests/habitat/test_instance_disambiguation_launcher.py`：
  - 成功解析 487/168/1080/1012，写出 F 的实际配置；
  - `001` manifest 行在请求 `1` 时必须失败（不按序号解释）。

```bash
export MODEL_PATH=... YOLO_MODEL_PATH=... SEG_MODEL_PATH=...
export GROUNDER_CHECKPOINT=... GROUNDER_BERT_VOCAB=...
export EPISODE_MANIFEST_PATH=... DATA_PATH=... SCENES_DIR=...
export CLIENT_PYTHON=...
export OUTPUT_ROOT=/root/autodl-tmp/output/instance_disambiguation_4_dry
DRY_RUN=1 bash scripts/run_instance_disambiguation_4.sh
```

正式运行把 `DRY_RUN=1` 去掉即可；本轮未执行。

## 8. Prompt 不变性

固定真实 1080 instruction、5 帧 pooled sample 的冻结 SHA-256：

```text
ec014d7d9799597312e036345b19561e297f8c2d44d337e237597478e14dacfb
```

固定 instruction 的 memory-disabled policy Prompt（`VLN_TRAJ_PROMPT_TEMPLATE`）
冻结 SHA-256：

```text
d4538cd84b1bdcaf5fca1019c4035ceca88cd2a31ac49c6d4469c80391681b90
```

- Memory disabled：逐字节等于 template / 冻结 hash。
- Memory enabled 但无注入：与 memory-disabled baseline 完全一致，
  `prompt_hash_before == prompt_hash_after`，无 `[Navigation Memory]`。
- 有注入：只在 `</navigation_task>` 后插入既有
  `[Navigation Memory]` 块；删除该块可恢复 baseline；角色、video segment、
  原始 instruction 数量和动作接口不变。
- 未改 System Prompt、原始 instruction、视频历史组织、模型输入格式、
  动作解码/接口、checkpoint。

## 9. 跨帧实例身份（概念审查，未改代码）

对照 `references/ConceptGraphs`：

| 官方 | 位置 | LightNav |
| --- | --- | --- |
| spatial similarity | `conceptgraph/slam/mapping.py:17` | `ObjectMap._candidate_scores` |
| visual similarity | `conceptgraph/slam/mapping.py:46` | Lite 在 `observe()` 后 `attach_clip_features`（`conceptgraphs_lite.py:545`） |
| aggregate | `conceptgraph/slam/mapping.py:66` | 纯几何 Lite |
| greedy association | `conceptgraph/slam/mapping.py:83` | `ObjectMap.observe` 一对一 greedy |
| `merge_obj2_into_obj1` | `conceptgraph/slam/utils.py:178` | 点云拼接/downsample/bbox 重算 |

未发现具体 bug；未修改 `conceptgraphs_lite.py` 的关联公式或阈值。既有测试
`test_conceptgraphs_lite.py:186/202/279` 覆盖同物体跨帧更新、相邻同类不合并、
greedy 一对一。

## 10. 测试结果

```text
CPU 全量：1 failed, 1303 passed, 14 skipped, 99 subtests passed
```

唯一失败 `tests/memory/test_rgm_place.py::RGMPlaceTests::test_initial_reference_and_global_ambiguity`
在干净 HEAD `d659e57` 上同样失败，是既有断言不一致
（`ambiguous_global_candidates` vs `unresolved_competing_candidate`），
与本次改动无关。
