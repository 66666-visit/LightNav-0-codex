# LightNav-0：Navigation Memory 渲染消融（H1/H2/H3）

> 本轮只修改 G 最终 Navigation Memory 的组织与渲染方式。
> 未启动 GPU、未启动 Habitat server、未运行任何 Episode（包括 487/168/1080/1012）、
> 未运行 88 Episode，未执行 `git add` / `commit` / `push`。
> 所有结论来自 CPU 合成 ObjectMap、真实指令文本和现有 `SpatialGraphAdapter`
> 输出；没有加载 YOLO、ViL3DRel 或 LightNav checkpoint。

## 1. 结论

G 相对 F 同时改变了三件事：

1. 无 Evidence 地标的占位文本（`no 3D evidence` / `Plan:`）；
2. 人工 source/intermediate/transition/final 角色与固定顺序；
3. 空间事实句的格式（G 的 `at world (...)m; now ... line,...deg`）。

本轮把 1 和 3 做成两个**独立布尔开关**，角色与顺序在所有组保持不变：

| 组 | 占位文本 | 事实句 | 说明 |
| --- | --- | --- | --- |
| F | 无 | F 原始 `render_context` | 原有 F 行为，未改 |
| G | 保留 | G 紧凑事实句 | 默认配置，逐字节复现原 G |
| H1 | 关闭 | G 紧凑事实句 | 只关闭无 Evidence 占位；释放的预算可容纳更多真实事实 |
| H2 | 保留 | F 原始 `render_context` | 保留占位、角色、顺序，只替换事实句 |
| H3 | 关闭 | F 原始 `render_context` | H1 + H2 |

默认配置（`include_missing_landmark_placeholders=True`、
`preserve_original_spatial_fact=False`）与修改前 G 的 Prompt 输出逐字节一致；
F 路径未改变。渲染切换可能改变“哪些整行事实最终进入字符预算”，该差异由
`spatial_l3.l3_query.ordered_memory_rendering` 显式记录，不静默发生。

## 2. 开关定义与调用路径

`src/lightnav/memory/spatial_l3_config.py`：

```python
grounder_ordered_landmark_memory: bool = False
include_missing_landmark_placeholders: bool = True
preserve_original_spatial_fact: bool = False
```

CLI：

```text
--spatial_l3_grounder_ordered_landmarks
--spatial_l3_grounder_include_missing_landmark_placeholders
--no-spatial_l3_grounder_include_missing_landmark_placeholders
--spatial_l3_grounder_preserve_original_spatial_fact
--no-spatial_l3_grounder_preserve_original_spatial_fact
```

后两个开关只在 `--spatial_l3_grounder_ordered_landmarks` 开启时合法；
否则 `SpatialL3Config` 直接拒绝。CLI 默认值分别是 `True` 和 `False`，
因此不传新参数时 D/E/F/G 的原有行为不变。

调用路径：

```text
policy.act
  -> MemoryManager.get_geometry_on_demand_context
     -> _grounded_geometry_context
        -> SpatialGraphAdapter.build           # 未改
        -> ViL3DRel eligible selection         # 未改
        -> build rows (role/order/phrase)      # 未改
        -> render_ordered_landmark_memory(
               include_missing_landmark_placeholders=...,
               preserve_original_spatial_fact=...,
           )
        -> build_ordered_memory_context         # 未改
  -> inject_memory_context                      # 未改
```

H2/H3 的事实句不是复制 F 的模板：manager 在构建每行时调用现有
`l3_evidence_selector.render_context([item], ...)`，把返回的 F 事实句原样存入
`LandmarkMemoryRow.original_spatial_fact`；有序渲染器只加回 G 的
`{order} {role} "{phrase}": ` 前缀。方位角符号、observed、straight-line、
source、passage unknown 全部来自 F 的原始字符串，没有被重新计算或改写。

## 3. 各分支的渲染语义

### 3.1 G（默认）

- 先为所有保留地标保留 skeleton 行：`{order} {role} "{phrase}": no 3D evidence.`；
- 再按 `final > transition > intermediate > source > reference` 用剩余字符把
  有事实的地标升级为 G 紧凑事实行；
- 地标多到 skeleton 行放不下事实时，退回一行 `Plan: ...` + 事实行；
- 至少一条事实行进入 Prompt 才返回 context。

### 3.2 H1

- 不保留 skeleton/`Plan:` 行，只渲染有 `selected_object_id` + `evidence_ids` +
  `spatial_fact` 的真实事实行；
- 仍使用与 G 相同的整行字符预算和相同的 `final > transition > ...` 升级顺序；
- 输出顺序仍是原始指令顺序；
- 没有事实时仍返回 `None`，不会用纯语义文本伪装成 evidence injection；
- 因为不再为占位行预留字符，H1 可能比 G 多容纳真实事实（见 §4.3）。

### 3.3 H2

- 占位、角色、顺序、候选筛选、Evidence ID、事实升级顺序全部与 G 相同；
- 每行事实体替换为 F 的原始 `render_context` 句子；
- F 句子明显更长，因此在 320 字符预算下可能比 G 少容纳事实，甚至在某些
  6 地标 compact 场景下没有一条 F 事实行能放下而返回 `None`。

### 3.4 H3

- 同时使用 H1 的“仅事实行”和 H2 的 F 原始事实句；
- 与 H2 相比没有 `Plan:`/skeleton 预留，因此通常会多容纳 F 事实行；
- 仍然不注入任何无 Evidence 占位。

### 3.5 预算差异记录

`ordered_memory_rendering` 的字段：

```json
{
  "include_missing_landmark_placeholders": true,
  "preserve_original_spatial_fact": false,
  "line_count": 4,
  "fact_line_count": 2,
  "placeholder_line_count": 2
}
```

`line_count` 是 header 之外实际写入的整行数；`fact_line_count` 是有真实事实的
行数；`placeholder_line_count` 是 skeleton/`Plan:` 行数。因此“H1 多注入了
一条 bed 事实”或“H2 因 F 句子过长只注入了 stairs”都能从日志直接复核。

## 4. 同输入状态的完整渲染对照

### 4.1 四个地标，320 字符，bed/door/stairs 均有可格式化事实

输入状态：

```text
instruction = walk away from sink, turn left, walk towards foot of bed,
              exit through doorway on left, turn right and step down
              two stairs and stop.
selected rows = sink(no fact), foot of bed(bed), doorway(door), two stairs(stairs)
```

F（180 字符，SHA-256 `6e6da018…df70aba`）：

```text
[Navigation Memory]
Spatial history: bed observed at world (2.0,0.0,1.0) m; now 2.4 m straight-line, bearing -117 deg (left +); 1 observations; source YOLO11n-seg; passage unknown.
```

G（306 字符，SHA-256 `0cc8270e…58cc18ce`）：

```text
[Navigation Memory]
1 source "sink": no 3D evidence.
2 intermediate "foot of bed": no 3D evidence.
3 transition "doorway": door at world (3.0,0.0,1.0)m; now 3.3m line,-108deg; 1 obs; vil3dref_grounding.
4 final "two stairs": stairs at world (4.0,0.0,1.0)m; now 4.2m line,-104deg; 1 obs; vil3dref_grounding.
```

H1（227 字符，SHA-256 `2094b528…795fd6da4`）：

```text
[Navigation Memory]
3 transition "doorway": door at world (3.0,0.0,1.0)m; now 3.3m line,-108deg; 1 obs; vil3dref_grounding.
4 final "two stairs": stairs at world (4.0,0.0,1.0)m; now 4.2m line,-104deg; 1 obs; vil3dref_grounding.
```

H2（304 字符，SHA-256 `2c2a5e23…e733747e3`）：

```text
[Navigation Memory]
Plan: 1 source "sink"; 2 intermediate "foot of bed"; 3 transition "doorway"; 4 final "two stairs".
4 final "two stairs": Spatial history: stairs observed at world (4.0,0.0,1.0) m; now 4.2 m straight-line, bearing -104 deg (left +); 1 observations; source YOLO11n-seg; passage unknown.
```

H3（205 字符，SHA-256 `7dff1672…96b69c5`）：

```text
[Navigation Memory]
4 final "two stairs": Spatial history: stairs observed at world (4.0,0.0,1.0) m; now 4.2 m straight-line, bearing -104 deg (left +); 1 observations; source YOLO11n-seg; passage unknown.
```

同一状态下五组逐地标选择结果完全相同（bed/door/stairs 的 persistent Object ID
与 Evidence ID 一致）；差异只在最终注入行和预算选择：

| 组 | 注入 Object ID | 注入 Evidence ID | 事实行 | 占位行 |
| --- | --- | --- | --- | --- |
| F | bed | `l3_seg_1_0` | 1 | 0 |
| G | door, stairs | `l3_seg_1_1`, `l3_seg_1_2` | 2 | 2 |
| H1 | door, stairs | `l3_seg_1_1`, `l3_seg_1_2` | 2 | 0 |
| H2 | stairs | `l3_seg_1_2` | 1 | 1 |
| H3 | stairs | `l3_seg_1_2` | 1 | 0 |

F 的 180 字符是旧排序/预算结果；G/H1/H2/H3 的差异均来自渲染格式与整行预算，
不是候选筛选或 Evidence 映射变化。

### 4.2 六个地标，320 字符，compact 模式

输入为 168 的指令和 6 个地标（double doors / dining table / entry way /
couch / armchairs / coffee table），所有地标都有真实事实：

- G（286 字符）：`Plan:` 一行 + final `armchairs` 事实，`fact_line_count=1`；
- H1（238 字符）：`double doors` + `armchairs` 两条事实行，`fact_line_count=2`；
- H2：返回 `None`。`Plan:` 行保留了全部角色/顺序，但剩余预算放不下任何一条
  F 事实行；这是 H2 在 320 字符下的真实失败关闭，不是静默丢事实；
- H3（203 字符）：只放 final `armchairs` 的 F 事实行，`fact_line_count=1`。

### 4.3 预算选择的显式差异

同一 4 地标状态、360 字符：

- G 仍只放 door + stairs 两条事实（skeleton 行占用预算）；
- H1 额外放入 bed 事实（3 条事实），且 G 已渲染的 door/stairs 行逐字节不变。

这正是 `include_missing_landmark_placeholders=False` 释放预算的可复现结果；
测试 `test_h1_reallocates_freed_budget_to_real_facts_and_records_it` 断言
`fact_line_count 2 -> 3` 和 `placeholder_line_count >0 -> 0`。

## 5. 不变性

- 未修改候选筛选、`build_proposal_eligibility`、`choose_first_eligible`、
  `resolve_geometry_fallback`、Evidence ID 生成、事实升级顺序、地图更新、
  查询频率、Grounding、Token Budget 或 SEED。
- 未修改 YOLO11n、YOLO11n-seg、L1 Original、ConceptGraphs Lite、ObjectMap、
  三维关联、ViL3DRel 权重、Eligible Selection、Multi-landmark queries、
  Geometry fallback、原始指令、官方 Prompt 模板、视频历史、动作解码、STOP
  接口或生成配置。
- 无 Evidence 地标仍完整记录在 `ordered_landmarks` / `landmark_rows`，只是
  H1/H3 不把占位文本写入 Prompt。
- 默认 G/F Prompt 逐字节不变：CPU 冻结 hash 见 §6。
- D/E/F launcher 的 `effective_client_args` 与 HEAD 逐字节一致（用
  `git archive HEAD` 的临时 checkout 复核）；`resolved_config.json` 的
  `flags` 只新增了默认值字段，不改变实际 CLI 命令。
- Prompt roundtrip：H1/H2/H3 均通过 `audit_prompt_injection`，删除注入块可
  恢复原始 human prompt；video segments 不被修改。

## 6. CPU 测试

执行环境：本机 Python 3.11 venv，`PYTHONPATH=src`，CPU only。

```bash
PYTHONPATH=src /path/to/python -m pytest -q \
  tests/memory/test_ordered_landmark_memory.py \
  tests/memory/test_instruction_landmarks.py \
  tests/memory/test_instance_disambiguation.py \
  tests/memory/test_l3_instance_grounding_manager.py \
  tests/habitat/test_geometry_on_demand_cli.py \
  tests/habitat/test_prompt_artifacts.py \
  tests/habitat/test_runner.py \
  tests/habitat/test_instance_disambiguation_launcher.py \
  tests/habitat/test_instance_disambiguation_1012_launcher.py \
  tests/habitat/test_geometry_on_demand_policy.py
```

新增/更新的关键断言：

1. 默认 G 四地标输出 SHA-256 `0cc8270e…58cc18ce`，默认 F 输出
   `6e6da018…df70aba`；两者都来自修改前代码的冻结基线。
2. 所测 1012 合成状态在 320 字符下，H1 逐行等于 G 的事实行（只删除
   placeholder）；在 360 字符下额外容纳 bed 事实，并记录在
   `ordered_memory_rendering`。
3. H2 的每条事实行都以 F 的 `original_spatial_fact` / `spatial_fact` 原文结尾；
   `original_spatial_fact == spatial_fact`，没有重新模板化。
4. H3 同时满足“无 `no 3D evidence` / `Plan:`”和“事实句是 F 原文”。
5. 无有效 Evidence 时五组都返回 `None`，`actual_injection=False`，
   `skipped_reason=ordered_memory_budget_excluded`，不注入纯占位文本。
6. 同一输入状态下 F/G/H1/H2/H3 的逐地标 `selected_object_id`、`evidence_ids`、
   `selection_source`、`geometry_fallback_reason` 完全一致；Grounder 调用序列一致。
7. Prompt roundtrip、冻结 hash、Token Budget 的 `select_context_lines` 和
   `trim_context_for_overflow` 通过。
8. 新增非 1012 指令 `walk towards the red chair next to the window and stop.`
   的通用测试，证明实现不依赖 Episode ID 或四条案例字符串。

实际结果（本轮）：

```text
tests/memory/test_ordered_landmark_memory.py                 27 passed
tests/habitat/test_geometry_on_demand_cli.py                 16 passed
tests/habitat/test_prompt_artifacts.py                       10 passed
tests/habitat/test_instance_disambiguation_launcher.py        5 passed
完整 CPU 门禁（pytest -q -m "not gpu"）：
  1333 passed, 15 skipped, 99 subtests passed, 1 failed
```

唯一失败是仓库既有的、与本次无关的
`tests/memory/test_rgm_place.py::test_initial_reference_and_global_ambiguity`
（`ambiguous_global_candidates` vs `unresolved_competing_candidate`）；
`docs/research/ORDERED_LANDMARK_MEMORY_1012.md` 已记录它在干净 HEAD 同样失败；
本轮用 `git archive HEAD` 的临时目录复核，失败相同。本次没有修改
`place_verification.py`。

Launcher CPU self-test：

```text
SELF_TEST PASS D/E/F/G/H1/H2/H3
SELF_TEST PASS: invalid group rejected, ambiguous episode ID rejected.
```

## 7. H1/H2/H3 评测命令（本轮未执行）

所有路径替换为真实本机路径。三组使用不同 `OUTPUT_ROOT`，互不覆盖。
每组严格解析并运行 487、168、1080、1012 四条 Episode，不启动 88 条。

先准备一次公共环境：

```bash
cd /path/to/LightNav-0-codex-instance-disambiguation

export MODEL_PATH=/path/to/LightNav-0
export YOLO_MODEL_PATH=/path/to/yolo11n.pt
export SEG_MODEL_PATH=/path/to/yolo11n-seg.pt
export GROUNDER_CHECKPOINT=/path/to/nr3d/ckpts/model_epoch_96.pt
export GROUNDER_BERT_VOCAB=/path/to/bert-base-uncased/vocab.txt
export GROUNDER_CONFIG=/path/to/nr3d_gtlabelpcd_mix_model.yaml
export EPISODE_MANIFEST_PATH=/path/to/manifest_with_487_168_1080_1012.json
export DATA_PATH=/path/to/data
export SCENES_DIR=/path/to/scenes
export CLIENT_PYTHON=/path/to/inference/venv/bin/python
```

H1（关闭无 Evidence 占位，G 事实句）：

```bash
ABLATION_GROUP=H1 \
OUTPUT_ROOT=/path/to/output/navmem_h1 \
EPISODE_IDS="487 168 1080 1012" \
bash scripts/run_instance_disambiguation_4.sh
```

H2（保留占位，F 原始事实句）：

```bash
ABLATION_GROUP=H2 \
OUTPUT_ROOT=/path/to/output/navmem_h2 \
EPISODE_IDS="487 168 1080 1012" \
bash scripts/run_instance_disambiguation_4.sh
```

H3（关闭占位 + F 原始事实句）：

```bash
ABLATION_GROUP=H3 \
OUTPUT_ROOT=/path/to/output/navmem_h3 \
EPISODE_IDS="487 168 1080 1012" \
bash scripts/run_instance_disambiguation_4.sh
```

CPU-only 预检（不启动 GPU/Habitat，只解析 manifest 和打印命令）：

```bash
DRY_RUN=1 ABLATION_GROUP=H1 OUTPUT_ROOT=/tmp/navmem_h1_dry \
  bash scripts/run_instance_disambiguation_4.sh
```

每组内部仍使用 `EPISODES=1 MAX_STEPS=500 NO_FORCE_STOP=1 SEED=42`，
每个 Episode 单独启动一次 server/client 对，输出在
`OUTPUT_ROOT/group_<Hx>/episode_<id>_r0/`。

## 8. H4 最小实现可行性（只读分析，本轮不实现）

### 8.1 已经存在的内容

`instruction_landmarks.py` 已经提供：

- `action_phrase` + 精确 `action_span`；
- `target_phrase` + `target_span` / `phrase_span`；
- `reference_phrase(s)` + `reference_span(s)` + `reference_objects`；
- `host_index`、`relation_kind`、`grounding_query_span`；
- source/transition 的文本推导与 `navigation_role` / `landmark_order`。

所有 phrase 都是 `instruction[start:end]` 的精确切片，且 parser 对 provider
输出做 span 校验，失败则回退 deterministic extractor。因此“直接截取原始
文本”在单地标、单动作片段上是可行的，不需要复制 F 模板。

### 8.2 不能直接可靠完成的部分

1. **同一动作支配多个地标时没有 action group。**
   `action_phrase` 是逐地标从后向前搜索得到的，不是动作 ID。实际 CPU 输出：

   ```text
   walk towards the table and chairs and stop.
   table  -> action='walk towards', action_span=(0,12)
   chairs -> action=None,        action_span=None
   ```

   第二个并列物体拿不到 action；H4 若直接用 `action_phrase` 会丢失“同一动作”
   关系，若从第一个地标继承又需要判断中间是否出现新动词/新从句。

2. **跨从句引用可能被错误归属。**
   `walk towards the table, then turn left and stop at the chairs.` 中
   `chairs` 被解析为 `host_index=table` 的 reference，但语义上它是
   `turn left ... at the chairs` 的地点。`_host_for_reference` 只看句子终止符，
   逗号后的新动作不会切断 host。H4 若直接按 host 组织关系会把两个动作合并。

3. **并列/参照关系只部分保留。**
   `reference_phrases/spans` 和 `host_index` 能保留
   “armchairs surrounding a coffee table”这类关系；但
   “table and chairs”这类并列没有“并列组”字段，只有两个独立 target。
   角色规则仍会给 table `intermediate`、chairs `final`，这是顺序推导，不是
   动作关系。

4. **没有统一的“动作支配区间”。**
   可以从 `action_span` 和下一个 `action_span` 之间截取原文，但需要新增
   边界判定（标点、连词、关系词、是否出现新动词）。这正是 H4 的新解析逻辑，
   超出本轮“不实现新子指令解析器或关系图”的边界。

### 8.3 可行的最小 H4 形态（如果后续批准）

建议只在**可证明无歧义**时启用：

```text
for each landmark:
    if landmark.action_span is None: fallback F
    if another target/action span overlaps or is non-contiguous: fallback F
    if sentence punctuation lies inside [action_start, target_end]: fallback F
    if host_index points across a comma+verb boundary: fallback F
    else: use instruction[action_start:target_end] verbatim
```

即使这样，也需要新增“动作分组/边界”函数才能把并列地标归到同一动作；
仅靠现有字段只能在单动作单地标时直接截取。因此本轮不实现 H4，等 H1/H2/H3
评测结果决定是否值得引入这层解析。

## 9. 修改文件与代码行数

本轮未 commit。`git diff --numstat`（已跟踪文件）：

| 文件 | 新增 | 删除 |
| --- | ---: | ---: |
| `src/lightnav/memory/ordered_landmark_memory.py` | 108 | 10 |
| `src/lightnav/memory/manager.py` | 46 | 0 |
| `src/lightnav/memory/spatial_l3_config.py` | 21 | 0 |
| `src/lightnav/cli/eval_habitat.py` | 22 | 0 |
| `src/lightnav/habitat/prompt_artifacts.py` | 11 | 0 |
| `scripts/run_instance_disambiguation_4.sh` | 176 | 32 |
| `tests/memory/test_ordered_landmark_memory.py` | 428 | 0 |
| `tests/habitat/test_geometry_on_demand_cli.py` | 77 | 0 |
| `tests/habitat/test_instance_disambiguation_launcher.py` | 35 | 0 |
| `tests/habitat/test_prompt_artifacts.py` | 14 | 0 |
| `docs/CONFIGURATION.md` | 17 | 0 |
| `docs/research/NAVIGATION_MEMORY_RENDERING_ABLATION.md`（新文件） | 463 | 0 |
| **合计** | **1418** | **42** |

源码（`src/`）净变化 +198 行，测试净变化 +554 行，launcher 净变化 +144 行，
文档净变化 +480 行。

源码修改集中在：

- `src/lightnav/memory/ordered_landmark_memory.py`：两个渲染开关、fact-only
  分支、F 原始事实字段；
- `src/lightnav/memory/manager.py`：复用 F formatter、开关透传、
  `ordered_memory_rendering` 诊断；
- `src/lightnav/memory/spatial_l3_config.py`：开关默认值与契约校验；
- `src/lightnav/cli/eval_habitat.py`：CLI 开关；
- `src/lightnav/habitat/prompt_artifacts.py`：诊断展平；
- `scripts/run_instance_disambiguation_4.sh`：G/H1/H2/H3 独立 arm 与 self-test；
- 测试与 `docs/CONFIGURATION.md`。

## 10. 明确未执行

- 未启动 GPU、未加载 YOLO/ViL3DRel/LightNav 权重；
- 未启动 Habitat server，未运行 487/168/1080/1012 或 88 Episode；
- 未执行 `git add` / `commit` / `push`；
- 未修改 `artifacts/` 或 `references/vil3dref/`。
