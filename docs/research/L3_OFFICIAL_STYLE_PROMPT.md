# L3 官方 Prompt Guide 风格重构（实验条件 `l3_official_style_only`）

> 状态：代码实现、定向测试、全仓 CPU 测试、新旧 Prompt 实际输出对照、独立审查均已完成。
> 尚未正式运行 88 Episode，未 commit / push，未操作 AutoDL。
>
> 本轮只研究一个变量：**L3 已检索到的三维地图事实如何被序列化成 Prompt**。
> 不改 L1、不改 L3 建图、不改候选检索、不改动作解码、不训练、不引入外部 LLM API。

---

## 0. 结论（先看这一段）

`l3_official_style_only` 已经实现并接入现有 `--memory_prompt_mode`：

- **候选检索完全复用** `L3PromptOrganizer.collect()`。新模式和 `l3_organized_only`
  调用的候选代码、证据门槛、排序键、`max_items` 完全相同；
- **只重写表达**：每条被选中的已验证 L3 事实被写成“机器人视角方向 + 明确物体 +
  直线距离 + 记录 Step/Views”，并按官方指南的证据优先级，在同一行最多附加一种
  消歧策略（关系锚点 / 极值 / 序数）；
- **L1 Prompt 构建完全关闭**：不调用 `L1PromptOrganizer.build`、
  `LocalGraphFusion.build`、`JointPromptBuilder.build`；L1 检测器仍按原样写入
  Evidence（这是 L3 关联证据所必需的），但 L1 文本不进入 Prompt；
- **历史/当前、直线/路径、通行性**：原子头部一次性声明“history, not current
  view”“straight-line ranges, not paths”“passage unverified”；每条事实带
  每行以 `Map record (step N, V views)` 开头；
- **没有编造几何**：`next to` / `on` 只由两个已观测 3D bounding box 计算，
  且锚点本身也要通过同样的 geometry/history 证据门槛；不满足就退回方向+距离，
  或者不注入；
- **旧模式不受影响**：`l3_original_only`、`l3_organized_only` 的 Prompt 文本和
  诊断结构保持不变；共享代码只做了一次行为等价的 `collect()` 抽取。

---

## 1. 官方依据与适配边界

### 1.1 README 的 Prompt Guide（`README.md:267-329`）

官方模板：

```text
[action verb] + [direction (optional)] + [disambiguated object phrase] + [and stop (optional)]
```

四种表达策略，按可靠性排序：

| 排序 | 策略 | 官方示例 | 关键要求 |
| --- | --- | --- | --- |
| ① | Direction + Object | `Go to the front-left TV` / `the desk on your right` | 方向相对于**机器人**，不是房间；室内优先 `front-left` / `front-right` / `in front` |
| ② | Relational Anchor | `the trash can next to the green lawn` / `the vase on the dining table ahead` / `behind you` | 锚点必须是**大型、显著**的空间物体，不能是另一个小物体 |
| ③ | Extremes | `the leftmost TV in front` / `the rightmost curtain` | `leftmost/rightmost` 优于需要深度估计的 `nearest/farthest` |
| ④ | Ordinals | `the second stone bench from the left` | 必须带**明确计数方向**；不要超过 `third`；能不用就不用 |

官方同时要求：一条指令只选一种消歧策略，不要堆叠；物体描述必须无歧义。

### 1.2 指南针对“导航指令”，L3 是“历史地图记忆”

这是本轮最重要的适配判断。官方 Prompt Guide 的目标是让用户写出一条好的
**command**；L3 的输入是机器人过去观测形成的 ObjectMap。因此本实现刻意做了以下
取舍：

| 官方语境 | L3 适配 | 原因 |
| --- | --- | --- |
| `Go to X` 命令句 | `The X is ...` 陈述句 | L3 是记忆，不是当前任务指令；不能把历史地标伪装成“现在去这里” |
| 当前视野中的物体 | 每行明确写 `Map record (step N)`，头部写 `history, not current view` | 防止冻结模型把历史地标当成当前 RGB 观察 |
| 方向相对机器人 | **保留**，用当前 `agent_pose` 计算 | 官方明确要求相对机器人；L3 有真实位姿和真实 3D 中心 |
| `nearest/farthest` | 不使用，改用 `leftmost/rightmost` 或方向+直线距离 | 官方明确 `nearest/farthest` 需要深度估计；L3 的直线距离不是路径距离 |
| 关系 `next to/on` | 只从两个已观测 3D box 计算，并要求证据 | ObjectMap 没有 object-object 边；不能凭空补关系 |
| 一条指令一种策略 | 每条事实：方向（模板方向槽）+ **最多一种消歧策略** | 官方示例本身也同时出现方向与关系（`the vase on the dining table ahead`）；这里禁止的是关系+极值+序数堆叠 |
| 序数 `first/second/third` | 仅在 ≥3 个同类、稳定左右序、前方半平面时使用 | 计数方向必须明确；默认 `max_landmarks=2` 时该路径处于休眠状态 |

---

## 2. 现有 L3 实现（源码事实）

### 2.1 L3 原始地图字段

`src/lightnav/memory/conceptgraphs_lite.py`：

| 结构 | 字段 | 用途 |
| --- | --- | --- |
| `ObjectNode`（`:328`） | `object_id`, `class_name`, `center_world`, `bbox_3d`, `observation_count`, `evidence_ids`, `geometry_status`, `points_world` | 物体节点；中心点是融合点云均值，bbox 是观测点云的 min/max |
| `ViewpointNode`（`:340`） | `viewpoint_id`, `scene_id`, `episode_id`, `rollout_idx`, `step_id`, `agent_pose`, `camera_pose` | 真实观测视角 |
| `ObservedFromEdge`（`:351`） | `object_id`, `viewpoint_id`, `evidence_id` | 物体 ← 哪个视角的哪条证据 |
| `TraversedToEdge`（`:359`） | `source_viewpoint_id`, `target_viewpoint_id`, `transition_step` | 真实发生过的 viewpoint→viewpoint 位移 |
| `SpatialEvidence`（`:367`） | `evidence_id`, identity, `step_id`, `class_name`, `confidence`, `source`, `mask_pixel_count`, `depth_point_count`, `geometry_status`, `linked_l1_evidence_id` | 每张 mask 的几何证据 |

**没有 object-object relation 字段。** 因此本实现的 `next to` / `on` 不是读取
某个已存的“关系边”，而是从两个 `ObjectNode.bbox_3d` 的已观测几何计算出来，并且
两个节点都必须通过下面的证据门槛。

### 2.2 坐标系

- Habitat world，单位米，`+Y` 向上（模块 docstring，`conceptgraphs_lite.py:1-6`）；
- 相机/agent 局部坐标：`+X` 向右、`-Z` 向前；
- 四元数显式 `[w, x, y, z]`，不猜顺序；
- `_relative_position`（`spatial_graph_adapter.py:109-136`）：
  `local = R^T (center_world - agent_position)`，
  `bearing = atan2(-local.x, -local.z)`，**左为正、右为负**；
  `distance = ||center_world - agent_position||`，是直线距离。

### 2.3 机器人位姿从哪里来

- `RGBDFrame.from_habitat` 读取 Habitat server 的
  `info["spatial_rgbd"]["agent_pose"]`（`conceptgraphs_lite.py:189-235`）；
- 每个推理 Step，`policy._generate_with_prompt_mode` 先调用
  `manager.observe_spatial(...)`，再把 `manager.last_spatial_frame` 作为 `frame`
  传给 `build_prompt_context`（`policy.py:695-723`）；
- L3 Organizer 用 `frame.agent_pose.position` / `rotation_wxyz` 计算当前
  robot-relative bearing（`l3_prompt_organizer.py:310-313`）。

### 2.4 哪些空间关系有真实证据

| 关系 | 证据 | 本实现是否使用 |
| --- | --- | --- |
| `observed_from`（物体被哪个视角看到） | `ObservedFromEdge` + `SpatialEvidence` + `ViewpointNode`，并校验 identity/step/scope/class/confidence | 作为“历史事实”的门槛 |
| `traversed_to`（两个视角之间真实移动过） | `TraversedToEdge` | 只保留在 metadata；不转成“可通行”或“路径” |
| object↔object `next to/on` | **没有存边**；只能从两个已观测 `bbox_3d` 计算 | 是的，严格几何规则 + 双节点证据门槛 |
| `behind you` | 当前 robot-relative bearing | 是，属于方向分区 |
| 通行性/可走路径 | 无 | 从不生成 |

### 2.5 现有 L3 Organizer 的选择、排序、截断

`l3_prompt_organizer.py`：

1. **相关性**：标签词项与 instruction 词项的交集；`score = overlap/len(label_terms)`，
   完整短语再加 0.5；`score == 0` 不是候选（`:52-62`, `:226-230`）；
2. **证据门槛**：物体的全部 Evidence 必须 `geometry_status == "observed"` 且
   `confidence >= min_confidence`；历史必须是同 Scene、同 Episode 的更早 Step，
   或 `scene_persistent` 下的旧 Episode；记录的 class 必须与节点 class 一致；
   linked L1 Evidence 必须 identity/step/provenance/label 全部有效（`:211-357`）；
3. **排序键**：`verified → -match_score → -last_step → -observation_count →
   distance → label → object_id`（`:305-315`）；
4. **截断**：`max_items = local_graph.max_landmarks`（默认 2，`manager.py:140-143`）；
   未验证几何最多再放 1 条；每条整行按 escaped 长度择优；放不下就整行丢弃并计入
   `omitted_items`（`:414-455`）。

### 2.6 Prompt 最终在哪里注入

- `policy._generate_with_prompt_mode`（`policy.py:666-998`）：
  `manager.build_prompt_context(...)` → 得到 `MemoryContext` →
  `inject_memory_context(base_sample, context, max_chars=budget)`；
- `memory_context_builder.inject_memory_context`（`:57-96`）把文本块插到 human
  message 的 `</navigation_task>` 之后，保留原句、video 和 action 接口；
- `audit_prompt_injection`（`:99-146`）校验原 Prompt 可恢复、video 对象未变；
- Token Budget：`measure_memory_token_budget`（`token_budget.py:155-221`）用
  checkpoint 的 Processor/Tokenizer 测 `context_limit - base_prompt - reserve - 1`；
  `select_context_lines`（`:239-288`）只保留能整行放入的连续前缀；
  `policy.py:773-808` 再用 `tokenizer.encode(augmented) - encode(base)` 精确复核，
  超了丢最低优先级条目重试。

---

## 3. 新模式设计

### 3.1 开关与隔离

`prompt_modes.py:46-49,156-158`：

```python
OFFICIAL_STYLE_PROMPT_MODE = "l3_official_style_only"
MEMORY_PROMPT_MODES = ("legacy", *ABLATION_PROMPT_MODES, OFFICIAL_STYLE_PROMPT_MODE)
_PLANS[OFFICIAL_STYLE_PROMPT_MODE] = PromptModePlan(
    OFFICIAL_STYLE_PROMPT_MODE, l1_source=None, l3_source="official",
    combination="single",
)
```

在 `prompt_assembly.build_prompt_context`（`:163-176`）中，`l1_source is None`
且 `l3_source == "official"`：

- 不进入 L1 original / organized 分支；
- 不进入 joint / concat 分支；
- 只调用 `official_l3_organizer.build(...)`。

`manager.py:140-150` 构造同一个 `L3PromptOrganizer` 作为 `collector`，保证候选
检索实例与 `l3_organized_only` 完全一致。

**L1 检测仍运行**：`MemoryManager.observe` / `observe_spatial` 仍然写 L1 Evidence，
因为 L3 的 `SpatialEvidence.linked_l1_evidence_id` 和 provenance 需要它。被关闭的是
**L1 Prompt 构建与注入**，不是 L1 观测写入。测试用 monkeypatch 把
`local_graph_fusion.build`、`l1_prompt_organizer.build`、
`joint_prompt_builder.build` 全部替换为抛错函数，运行 2 个 Step 不触发，证明零调用。

### 3.2 候选集合同源（实验控制的核心）

```python
# l3_prompt_organizer.py
def build(...):
    organization = self.collect(...)      # 唯一检索路径
    self._serialize(organization, ...)    # organized 的旧序列化

# l3_official_prompt.py
organization = self.collector.collect(...)  # 同一个 collect
self._serialize(organization, ...)          # 只换写法
```

`collect()` 的代码体是原 `build()` 主体原样搬移，未改评分、门槛、证据校验或排序；
`max_chars` 校验留在 `build()`。因此：

- `candidate_count` / `verified_count` / `unverified_count` /
  `current_only_skipped` 一致；
- `organization.facts` 的 `to_dict()` 逐字段一致（单测直接比较）；
- `selected_facts` 的已验证子集一致（下面 3.3 解释）。

**关于未验证事实**：`collect()` 的排序键第一位是 `0 if geometry_verified else 1`，
所有已验证事实排在未验证事实之前。因此取 `max_items` 个已验证事实，等价于
“旧模式选择集去掉最多 1 条未验证行”，不会把旧模式排在未验证行之后的事实提前。
旧模式会注入最多 1 条 `map evidence unverified; position and passage unknown`；
官方风格无法为它给出真实方向/关系，因此**不注入**，只计入
`expression.unverified_omitted`。这是“舍弃”而非“编造”。

### 3.3 Direction + Object（主策略）

`direction_phrase`（`l3_official_prompt.py:88-107`）把左正 bearing 分成 8 个扇区
（边界 22.5°）：

| bearing（左正） | 输出 |
| --- | --- |
| `[-22.5, 22.5]` | `in front of you` |
| `(22.5, 67.5]` | `to your front-left` |
| `(67.5, 112.5]` | `to your left` |
| `(112.5, 157.5]` | `behind you on the left` |
| `(157.5, 180]` / `[-180, -157.5)` | `behind you` |
| 右侧镜像 | `front-right` / `right` / `behind you on the right` |

事实句（`_line`, `:566-579`）：

```text
Map record (step {N}, {V} view(s)): the {object phrase} is {direction}, {d:.1f} m straight-line.
```

- 方向来自当前 `agent_pose` + 地图中心，**不是**房间坐标；
- 距离是 `||center - agent_position||`，明写 `straight-line`，不写成路径；
- 每行以 `Map record (step N, V views)` 开头，把来源写在句首；旧 Episode 再加 `, earlier episode`；
- Step 用 `latest_step`（最新一条 History 记录的 Step），不是 `max(steps)`；
  `scene_persistent` 下旧 Episode 的 Step 可能更大，不能用它冒充最新记录；
- `views` 是支持该事实的**不同历史 viewpoint 总数**（跨 Episode 合并），不是
  “最后一个 Step 的视角数”；当前 L3 ObjectMap 在 Episode reset 时清空，混合
  Episode 历史实际上不可达，这里是防御性处理；
- 没有位姿时（`frame is None`）不猜方向：

```text
Map record (step N, V views): the {object phrase} was recorded, but its
direction from you is unavailable; passage unverified.
```

### 3.4 Relational Anchor（第二种策略）

**证据规则**（`support_relation`, `:202-247`）：

- `on`：
  - 目标与锚点的水平 footprint 重叠率 `overlap / min(area)` ≥ `0.35`；
  - 目标观测底面 − 锚点观测顶面 ∈ `[-0.05, +0.10]` m（可见悬空 >10 cm
    不再声称支撑）；
  - 目标顶面高于锚点顶面；
  - 目标 footprint ≤ 锚点 footprint（锚点必须是更大的支撑面）；
  - 目标 footprint 至少 80% 落在锚点 footprint 内（悬挑过多不算 `on`）。
- `next to`：
  - 两个水平 footprint **完全不重叠**（重叠/部分穿插/嵌套一律不叫“旁边”）；
  - 水平 AABB 间隙 ≤ `0.5` m；
  - 垂直间隙 ≤ `0.75` m（墙挂物体不会因为水平接近就被说成“在桌子旁边”）。
- 其他一律 `None`，不猜。

**锚点池**（`_map_objects`, `:341-415`）：

- 遍历整个 ObjectMap，但每个锚点必须通过与 `collect()` 相同的证据门槛：
  全部 Evidence `observed` 且 `confidence >= min_confidence`；
  `valid_history` 非空（同 Episode 更早 Step，或 persistent scope 的旧 Episode）；
- 因此**当前 Step 才第一次看到的物体不能当锚点**，和历史事实一样被排除；
- 锚点可以不与 instruction 相关（官方示例的 `green lawn` 也不需要是目标）；
- 它只作为“指代参照”，不会进入 `selected_facts`，也不单独描述它的位置。

**选择与唯一性**（`_relational_anchor`, `:441-483`；`_relation_disambiguates`, `:418-439`）：

- 只在**同名标签在已记录地图中出现 ≥2 次**时使用（否则方向已经能唯一指代，
  不需要额外消歧）；
- 锚点必须 footprint ≥ 目标的 `0.75`（大型、显著）；
- 按 `on` > `next to` → 锚点 footprint → volume → observation_count → 是否候选 →
  label → object_id 排序；
- **唯一性检查**：如果同标签的另一个已记录物体也满足同一个锚点的同一关系，
  这个关系不能区分二者，丢弃，退回极值/序数或方向；如果另一个同类对象没有
  bbox，无法证明唯一，也丢弃。避免 “两个椅子都在桌子旁边” 时说出无消歧力的关系；
- **锚点标签唯一**：如果地图记录了两个 “dining table”，`the chair next to the
  dining table` 仍然指代不明，因此该关系也不使用；
- `anchors_used` 和 `LayerMemoryItem.metadata` 记录锚点的 `evidence_ids` 与
  `bbox_3d`，关系声明可以从 artifact 审计；
- 极值/序数只在**该标签所有已验证历史成员都有有限 bearing** 时启用；否则
  “of the N recorded in front” 的 N 会漏掉无方向的成员。

示例输出：

```text
Map record (step 7, 1 view): the chair next to the dining table is to your front-left, 2.3 m straight-line.
```

### 3.5 Extremes / Ordinals（第三、第四策略）

只在“同名标签出现 ≥2 次”时考虑，并且必须满足（`_group_order`, `:485-512`）：

- 所有成员都在**前方半平面** `|bearing| ≤ 90°`（后方物体不参与左右计数）；
- 目标本身在该前方组内；
- 按 bearing 从大到小（左→右）排序后，相邻间隔都 ≥ `15°`；
- 计数域是“已记录且在前方的同名物体”，不是全场/当前视野全部物体。

表达：

| 位置 | 文本 |
| --- | --- |
| 最左 | `the leftmost {label} of the {n} recorded in front` |
| 最右 | `the rightmost {label} of the {n} recorded in front` |
| 中间第 2/3 个 | `the second/third {label} from the left among the {n} recorded in front` |

- 序数最多到 `third`（`_MAX_ORDINAL = 3`）；
- 计数方向在文本中显式写出 `from the left`；
- 默认 `max_landmarks=2` 时最多两个事实，序数路径处于休眠；只有把
  `--local_graph_max_landmarks` 调到 ≥3 才会出现。本轮仍实现且用单测覆盖，
  因为它是官方策略之一，且未来可能需要。

### 3.6 头部与事实句

头部是一行**原子**文本（`:51-53`）：

```text
[Navigation Map Memory - L3; history, not current view; directions relative to you;
straight-line ranges, not paths; passage unverified]
```

设计原因：

- Token/字符选择只保留“连续整行前缀”。头部放在第一行，任何事实行被保留时头部
  一定在；不会出现“有事实但没有历史/路径/通行性声明”的卡片；
- 头部+所有事实行都放不下时，整个 context 为 `None`（不注入半个卡片）；
- 事实行不写命令句、不写 `reachable/walkable/path is`，不声称通行性；
- 旧 Organized 卡的 `seg conf` 被移出文本，保留在 metadata（`confidence`）；
- 旧卡的 `Step {step}` 状态行被头部替代；当前 Step 由对话/诊断知道，事实行保留
  各自的 `Map record (step ...)`，更符合“历史事实”的表达。

### 3.7 预算与整行截断

- 复用现有模型派生 Token Budget：`builder_chars = UNBOUNDED_CHAR_LIMIT`（有 token
  budget 时）或原 320 字符 fallback（无 token budget 时）；
- 序列化器内部按 escape 后的整行长度检查 `max_chars`，不做半行截断；
- 之后 `select_context_lines` 再做 token 整行前缀选择；
- `policy` 的精确 token delta 复核和 `trim_context_for_overflow` 继续生效；
- 头部越长，320 字符 fallback 下能放的事实行越少（官方头部 136 字符）。
  这是 fallback 的已知取舍；正式实验使用模型派生 token 预算，不受此限。

### 3.8 诊断

`L3Organization.diagnostics()` 新增（仅在官方模式非空时）：

```json
{
  "expression_version": "l3_official_style_v1",
  "strategy_counts": {"direction": 0, "relational_anchor": 1, "extreme": 1,
                      "ordinal": 0, "direction_without_pose": 0},
  "unverified_omitted": 0,
  "duplicates_removed": 0,
  "ambiguous_labels": {"chair": 2},
  "anchors_used": [{"object_id": "...", "label": "dining table", "relation": "next to"}],
  "candidate_object_ids": ["..."],
  "selected_object_ids": ["..."]
}
```

`prompt_artifacts.prompt_diagnostic_row` 额外写入 `l3_expression`，逐 Step 可审计。
旧的 `l3_organized_only` 不写 `expression`，诊断结构保持不变。

---

## 4. 新旧 Prompt 实际输出对照

以下输出由同一份合成 ObjectMap、同一个 `frame`、同一个 Step/instruction 生成；
可用 `PYTHONPATH=src python scripts/show_l3_prompt_comparison.py` 在 CPU 上复现
（不需要 Habitat/模型）。`l3_original_only` 使用它自己的 320 字符预算和
`max_landmarks=2`，所以它只展示 1 条；Organized/Official 用 2000 字符展示完整
候选集，便于看清“同一批事实的不同写法”。

### A. 两个已记录椅子；一个紧挨着大餐桌

Original（`l3_original_only`）：

```text
[Navigation Memory]
Spatial history: chair observed at world (-1.0,0.5,-2.0) m; now 2.3 m straight-line, bearing +27 deg (left +); 2 observations; source YOLO11n-seg; passage unknown.
```

Organized（`l3_organized_only`）：

```text
[Navigation Context - L3]
Step 10; map facts are history, not current visibility.
chair: last step 7, 1 view; 2.3m line +27deg (not a path); seg conf 0.90; passage unknown.
chair: last step 7, 1 view; 4.6m line -49deg (not a path); seg conf 0.90; passage unknown.
```

Official（`l3_official_style_only`，同一批候选事实）：

```text
[Navigation Map Memory - L3; history, not current view; directions relative to you; straight-line ranges, not paths; passage unverified]
Map record (step 7, 1 view): the chair next to the dining table is to your front-left, 2.3 m straight-line.
Map record (step 7, 1 view): the rightmost chair of the 2 recorded in front is to your front-right, 4.6 m straight-line.
```

Bookkeeping：

```json
{"strategy_counts": {"direction": 0, "relational_anchor": 1, "extreme": 1,
                     "ordinal": 0, "direction_without_pose": 0},
 "ambiguous_labels": {"chair": 2},
 "anchors_used": [{"object_id": "object_2", "label": "dining table", "relation": "next to",
                    "evidence_ids": ["[\"scene\",\"episode\",0,\"l3_seg_7_2\"]"]}]}
```

### B. 三个已记录椅子；稳定左右序

Original：

```text
[Navigation Memory]
Spatial history: chair observed at world (-1.0,0.5,-2.0) m; now 2.3 m straight-line, bearing +27 deg (left +); 2 observations; source YOLO11n-seg; passage unknown.
```

Organized：

```text
[Navigation Context - L3]
Step 10; map facts are history, not current visibility.
chair: last step 7, 1 view; 2.3m line +27deg (not a path); seg conf 0.90; passage unknown.
chair: last step 7, 1 view; 4.6m line -49deg (not a path); seg conf 0.90; passage unknown.
chair: last step 7, 1 view; 6.0m line 0deg (not a path); seg conf 0.90; passage unknown.
```

Official：

```text
[Navigation Map Memory - L3; history, not current view; directions relative to you; straight-line ranges, not paths; passage unverified]
Map record (step 7, 1 view): the leftmost chair of the 3 recorded in front is to your front-left, 2.3 m straight-line.
Map record (step 7, 1 view): the rightmost chair of the 3 recorded in front is to your front-right, 4.6 m straight-line.
Map record (step 7, 1 view): the second chair from the left among the 3 recorded in front is in front of you, 6.0 m straight-line.
```

Bookkeeping：`extreme=2, ordinal=1, ambiguous_labels={"chair": 3}`。

### C. 有地图事实，但该 Step 没有机器人位姿

Original：`<empty>`（原 adapter 需要 frame，无 frame 时不注入）。

Organized：

```text
[Navigation Context - L3]
Step 10; map facts are history, not current visibility.
chair: last step 7, 1 view; position unknown; seg conf 0.90; passage unknown.
```

Official：

```text
[Navigation Map Memory - L3; history, not current view; directions relative to you; straight-line ranges, not paths; passage unverified]
Map record (step 7, 1 view): the chair was recorded, but its direction from you is unavailable; passage unverified.
```

### D. 实际注入到 frozen policy 的 human message（l3_official_style_only）

用测试里的真实 policy / Observer / L3 sidecar 路径跑 2 个 Step，最后一条 Prompt：

```text
Imagine you are a robot programmed for navigation tasks. You have been given a sequence of observations <video>. Your assigned task is: <navigation_task>find the chair</navigation_task>. [Navigation Map Memory - L3; history, not current view; directions relative to you; straight-line ranges, not paths; passage unverified]
Map record (step 0, 1 view): the chair is in front of you, 2.2 m straight-line.
You may be in the beginning, middle or end of the task. Predict your future trajectory as a single trajectory id token.
```

诊断：`prompt_context.source == "l3_official"`，`layers` 只有 `l3`，没有 `l1`；
`actual_injection == True`；`evidence_ids` 来自 L3 `SpatialEvidence`。

---

## 5. 保留 / 舍弃（相对原始 L3 与旧 Organized L3）

### 5.1 保留

| 信息 | 原始 L3 | 旧 Organized | 新 Official |
| --- | --- | --- | --- |
| 指令相关的 L3 候选物体 | 有（按距离取 top-2） | 有（相关性/新鲜度排序） | **完全相同**（共享 `collect`） |
| 物体标签 | 有 | 有 | 有 |
| 当前 robot-relative 直线距离 | 有 | 有 | 有（1 位小数，`straight-line`） |
| 当前 robot-relative 方位 | 有（带符号度数） | 有（带符号度数） | 有（8 扇区短语；精确度数保留在 metadata） |
| 记录 Step / views | observation 数 | last step / views | Map record (step, views) |
| 通行性未知 | 有 | 有 | 有（头部 `passage unverified`） |
| 直线≠路径 | 有 | 有 | 有（头部 + 每行 `straight-line`） |
| 历史≠当前 | 部分（world 坐标） | 有（`history, not current visibility`） | 有（头部 + 每行 `Map record (step ...)`） |
| 关系/极值/序数 | 无 | 无 | **有**（证据支持时） |
| Evidence / provenance / confidence | 有（metadata） | 有（metadata） | 有（metadata） |

### 5.2 只保留在 metadata，不进入文本

| 信息 | 为什么不进文本 |
| --- | --- |
| 精确 bearing 度数 | 已被 robot-relative 短语替代；模型不需要同时读两套方向表示；metadata 保留 `current_relative_bearing_deg` |
| 精确 `center_world` / `bbox_3d` | 房间世界坐标不是官方导航语言；L3 地图保留，metadata 可审计 |
| `seg conf` | 对冻结模型的导航表达没有直接价值；保留在 `LayerMemoryItem.confidence` |
| `observation_count` / `traversal_links` / `traversal_steps` | 保留在 metadata；只写 views（视角数）作为历史证据强度，避免把 `traversed_to` 说成可通行 |
| `match_score`、provenance 细节 | 保留在 metadata / diagnostics |

### 5.3 舍弃（不注入）

| 信息 | 原因 |
| --- | --- |
| 未验证几何的 L3 事实 | 没有可信位置，无法给出真实方向/关系；旧 Organized 会注入 1 条 `map evidence unverified`，新模式选择不注入，只计数 |
| `traversal recorded between observed viewpoints` 文本 | 它只证明视角间移动过，不证明物体可通行；官方语言里没有对应表达，保留在 metadata 即可 |
| 命令句 / “去这里” | L3 是历史记忆，不能伪装成当前指令 |
| nearest/farthest | 官方明确需要深度估计、弱于 leftmost/rightmost；本实现不使用 |
| 任何 `reachable/walkable/path is` | 地图没有证明通行性 |

### 5.4 与哪个历史组是“只改序列化”对照（重要）

本实现固定的“L3 候选检索”是**当前 L3 Organizer 的 `collect()`**，因此：

- 与 `l3_organized_only`（73.86%）相比：**候选集合、证据、排序、`max_items` 完全相同**，
  唯一差异是 `L3OfficialPromptOrganizer` 的序列化（以及它无法表达时舍弃的未验证
  warning 行）。这是本实验的**受控对照**；
- 与 `l3_original_only`（71.59%）相比：后者使用更旧的 `SpatialGraphAdapter`
  （按距离排序、没有相关性评分）。新模式**没有**沿用这个更旧的排序，所以 71.59%
  的对照同时包含“检索排序变化 + 序列化变化”，不是纯粹的表达式对照。

这样选择的理由：任务要求“L3 候选检索算法保持不变”且“不使用旧 L3 Organizer 的
组织策略”。这里把 `collect()` 视为候选检索、把 `_serialize` 视为组织策略；新模式
替换后者、保留前者。若实验负责人实际想要的是**原始 adapter 的距离 top-N** 作为
检索基线，需要把 `collector` 换成 adapter-backed collector；那是另一个实验条件，
本轮没有实现，也不应把两个检索混在同一个 71.59% 对照里解释。

### 5.5 一个明确的取舍

官方头部 136 字符。若模型派生 Token Budget 不可用而回退到原 320 字符，
Official 卡通常只能放 1 条事实（Organized 可放 2 条）。正式实验使用 vLLM +
checkpoint Processor/Tokenizer，会走 token-aware 路径；fallback 只作为安全网，
并在 `prompt_diagnostics.jsonl` 记录 `budget_source=char_fallback_320` 与原因。
如果实际运行出现大量 fallback，应视为实验无效信号，而不是结果差异。

---

## 6. 测试与验证

### 6.1 新增/更新的定向测试

| 文件 | 覆盖 |
| --- | --- |
| `tests/memory/test_l3_official_prompt.py`（38 passed） | 候选集合逐字段一致；旧 Organized 文本不变；8 扇区方向；旋转后的 robot-relative 方向；无位姿 fallback；`next to`/`on` 证据规则（部分穿插、垂直分离、悬空、悬挑）；共享关系/重复锚点拒绝；当前 Step-only 事实与锚点拒绝；leftmost/rightmost；同 bearing 不稳定序拒绝；无 bearing 成员拒绝；后方物体不计数；序数计数方向；未验证省略；头部原子性/预算；重复事实去重；token 整行选择；已验证选择集 = 旧选择集去掉未验证行；latest_step 跨 Episode；标签净化 |
| `tests/habitat/test_prompt_mode_policy.py`（+3） | 官方模式 2 个 Step 不调用 L1/Joint 构建器且注入 L3 历史；旧 `l3_original_only`/`l3_organized_only` 文本不变；Episode reset 后旧地图事实不泄漏 |
| `tests/memory/test_prompt_modes.py` | 新模式 plan、mode 列表、backend 校验、manager 预算 |
| `tests/habitat/test_l1_l3_prompt_ablation_launcher.py`（+2） | `--official-style` 需要 historical root；缺任一 L3 历史组报错；dry-run 只跑新模式 |
| `tests/test_verify_local_graph_shards.py` | shard verifier 识别新模式 |
| 既有 `tests/memory/test_prompt_organization.py` | 共享 `collect()` 抽取后旧 L3/Joint 测试全部通过 |

### 6.2 11 项验收

| # | 要求 | 证据 |
| --- | --- | --- |
| 1 | L1 记忆构建器调用次数为零 | policy 测试把 3 个 build 方法替换为抛错函数，官方模式 2 Step 不触发；`layers` 无 `l1` |
| 2 | 原始 L3 候选集合与新模式完全一致 | 单测逐字段比较 `official.facts == organized.facts`，并比较 4 个计数。这里“原始”指现有 L3 Organizer 的 `collect()`；与 `l3_original_only` 的更旧距离排序之间的差异见 5.4 |
| 3 | 坐标转换正确 | 复用已验证的 `_relative_position`；旋转 90° 后世界前方物体变 `to your right` |
| 4 | 机器人视角方向正确 | 8 扇区参数化测试 + 旋转测试 + 无位姿测试 |
| 5 | 空间关系有证据支持 | `support_relation` 正/负例（部分穿插、垂直分离、悬空、悬挑）；锚点 geometry/history 门槛；当前 Step-only 锚点拒绝；共享关系/重复锚点标签拒绝 |
| 6 | 历史与当前观察明确区分 | 头部 `history, not current view`；每行以 `Map record (step ...)` 开头；无位姿 fallback 不猜方向 |
| 7 | 不凭空生成通行性 | 头部 `passage unverified`；无命令句、无 reachable/walkable/path is |
| 8 | 不重复注入相同事实 | 归一化行去重；单测 `duplicates_removed == 1` |
| 9 | Token Budget 不超限 | 序列化 max_chars + `select_context_lines` 整行前缀；单测 capacity 检查；policy token 路径复用旧实现 |
| 10 | 原始 L3 模式不受影响 | 旧 Organized 文本/诊断字段不变；旧 Original 仍 `Spatial history:`；`collect` 抽取行为等价 |
| 11 | 不同 Episode 无记忆污染 | ObjectMap 在 `reset_episode` 清空；policy 测试 reset 后旧 chair 不再出现 |

### 6.3 全仓 CPU 测试

命令：

```bash
PYTHONPATH=src .tooling/venv/bin/python -m pytest -q -m "not gpu"
```

结果：

```text
1 failed, 1160 passed, 14 skipped, 105 subtests passed in 80.68s
```

唯一失败是 **既有、未修改文件** 的断言：

```text
tests/memory/test_rgm_place.py::RGMPlaceTests::test_initial_reference_and_global_ambiguity
AssertionError: 'ambiguous_global_candidates' != 'unresolved_competing_candidate'
```

`src/lightnav/memory/place_verification.py` 与 `tests/memory/test_rgm_place.py`
均未在本轮改动（`git diff` 为空），失败与本模式无关，属于当前基线的既有失败；
本轮没有为了让全仓变绿而修改它。

`ruff check`：

- 改动文件：全部通过；
- `ruff check src tests scripts habitat_server evt_bench`：27 个既有 `I001`
  import-order 问题，全部在未改动文件；`.references/` 下另有一批未跟踪的参考仓库
  告警。两者都不是本轮引入。

---

## 7. 独立审查

独立子代理只做只读审查（不改文件），逐项检查了坐标、证据、候选一致性、L1 隔离、
预算、注入安全、旧模式回归和测试缺口。共提出 1 个 blocker、6 个 major、6 个 minor、
1 个 nit；本轮全部处理或明确记录取舍，处理后又跑了一遍定向测试与全仓 CPU 测试。

### 7.1 已修复

| # | 严重度 | 问题 | 修复 |
| --- | --- | --- | --- |
| 1 | **blocker** | 比较器只允许 historical-core 与 supplemental 的源码哈希不同；official 组不在任何阶段集合里，`--allow-source-difference` 会在真正的 88 Episode 比较时直接报错 | 新增 `OFFICIAL_GROUPS`，源码差异检查改为“每个阶段内部一致、至少两个阶段不同”；新增跨阶段比较测试（historical source=2，official source=3） |
| 2 | major | footprint 部分重叠（0 < overlap < 0.35）会落入 `next to`，把塞进桌子 footprint 的椅子说成“在桌子旁边” | 有 footprint 重叠时只允许 `on` 或 `None`，不再落入 `next to`；新增部分穿插测试 |
| 3 | major | `next to` 完全不看垂直距离，墙挂 2 m 高的物体也会“在桌子旁边” | 增加垂直间隙 ≤ 0.75 m；新增同 XZ 间隙、不同 Y 的测试（地面物体仍为 next to，墙挂为 None） |
| 4 | major | `on` 允许 35 cm 悬空、允许目标比锚点大 25%、允许大量悬挑 | `on` 接触容差收紧为 `[-0.05, +0.10]` m；目标 footprint ≤ 锚点；目标至少 80% 落在锚点 footprint 内；新增悬空/悬挑测试 |
| 5 | major | `The chair is ...` 现在时句子只靠括号标记历史，仍可能被读成当前观察 | 每行改为 `Map record (step N, V views): ...`，把来源放在句首；同步更新测试与输出对照 |
| 6 | major | `last_step = max(steps)`；`scene_persistent` 下旧 Episode 的 step 可能大于当前 Episode，导致行内 Step 错 | `L3Fact` 增加 `latest_step`（最新一条 History 记录的 step），**不加入 `to_dict`** 以保持旧诊断不变；新增“旧 Episode step 400 + 当前 step 10 → 输出 step 10”测试 |
| 7 | major | 官方模式丢弃未验证事实，旧 Organized 模式会注入 1 条未验证 warning，所选事实集不完全相同 | 确认 `collect` 排序保证“已验证事实全部排在未验证之前”，因此已验证子集完全一致；仅旧模式多一条无法表达位置的 warning。文档明确写出该差异，并新增选择集测试 |
| 8 | minor | token overflow 重试时，`trim_context_for_overflow` 若丢掉最后一条事实，会返回只剩 caveat 头部的 context 并被注入 | policy 的 overflow 分支增加“原来有 item、trim 后没有 item”则放弃注入的 guard；正常路径不变 |
| 9 | minor | 极值/序数分组会跳过没有 bearing 的同类成员，但文本仍说“of the N recorded in front”，N 可能少算 | 只要该标签任一已验证历史成员没有有限 bearing，就不使用极值/序数；新增 NaN center 测试 |
| 10 | minor | 两个同名锚点（两张 dining table）时，“the dining table”仍然指代不明；关系声明也缺少锚点证据 | 锚点标签在已记录池中必须唯一；`anchors_used`/metadata 增加锚点 `evidence_ids` 与 `bbox_3d`；新增重复锚点测试 |
| 11 | minor | 把 `l3_official` 加入共享 `builders` 字典会改变所有历史模式的 `prompt_protocol.json`，resume 时旧文件不匹配 | 只有 `plan.l3_official` 时才写 `builders["l3_official"]`；历史协议保持逐字节不变；新增 protocol 测试 |
| 12 | minor | anchor 排序用的 volume 从原始 bbox 角点计算，倒置角点会得到 0 | volume 改用 `_bbox_corners` 排序后的角点 |
| 13 | minor | official stage 在历史根缺 legacy 时加 `--allow-incomplete`，会把顶层 status 也变成 incomplete；preflight 错误信息仍写 supplemental | 不再加 `--allow-incomplete`（顶层 complete 只要求所有请求组完整）；错误信息改为通用 |
| 14 | nit | 比较器硬编码 mode 字符串；official 组仍用基础 `evaluation_mode`；测试名仍写 nine | 使用 `OFFICIAL_STYLE_PROMPT_MODE`/`OFFICIAL_GROUPS`；新增 `..._official_style` evaluation mode；测试名改为 ten |

### 7.2 审查确认无问题的区域

- **L1 隔离**：official plan 的 `l1_source=None`；assembly 不会进入
  `LocalGraphFusion` / `L1PromptOrganizer` / `JointPromptBuilder` 分支；policy 测试
  把三个 build 都替换为抛错函数仍通过。
- **坐标**：`center_world`/`agent_pose` 经 `_relative_position`，左正 bearing、
  robot-relative 扇区、直线距离均正确；旋转测试有效。
- **候选集合**：序列化器不重新评分、不重算 Evidence；`facts` 逐字段一致。
- **转义**：标签经 `_plain_label`（只保留字母数字/连字符，截断 48 字符），序列化行
  再做 `escape(quote=False)`；未发现换行或尖括号注入路径。
- **旧模式文本**：`l3_original_only` 仍是 `Spatial history:`；`l3_organized_only`
  仍是 `[Navigation Context - L3]` + `last step` + `not a path` + `passage unknown`。

### 7.3 仍未消除的限制

- 关系/极值阈值是几何先验，未在 88 Episode 上做敏感性分析（本轮不做调参）。
- 锚点可以不是 instruction 候选（官方“大型显著锚点”的要求）；如果审稿坚持
  “只用候选事实”，关系策略在 `max_landmarks=2` 下几乎不会触发。
- 未验证 warning 行被完全舍弃；这是有意选择，不是遗漏（见 5.3 与 7.1 #7）。

---

## 8. AutoDL 一键实验命令

### 8.1 固定变量

以下必须与历史 88 Episode 完全一致：

- `MODEL_PATH`：LightNav-0 checkpoint；
- `YOLO_MODEL_PATH`：L1 的 `yolo11n.pt`；
- `SEG_MODEL_PATH`：L3 独立分割的 `yolo11n-seg.pt`（必须与 L1 文件不同）；
- `EPISODE_MANIFEST_PATH`：固定 88 Episode manifest；
- `PILOT_MANIFEST_PATH`：固定 88 Episode server 选择 manifest；
- `HABITAT_CONFIG`：`habitat_server/configs/vlnce_r2r.yaml`（默认）；
- `GPU_IDS`、`GPU_MEM_UTIL`、`BASE_PORT`；**`GPU_IDS` 必须与历史组完全一致**，
  比较器把 `gpu_ids` 列入 provenance identity；
- `--historical-root DIR`：包含已完成的 `l3_original_only/` 与
  `l3_organized_only/`（以及可选的 `legacy/`）的根目录；不会重跑这些组；
- `MANIFEST_SEARCH_ROOT` 指向只包含本次历史结果（或干净空目录）的父目录；
  launcher 的 preflight 会拒绝同时出现两个不同的 88 Episode manifest。

### 8.2 命令

```bash
cd /path/to/LightNav-0-codex

MODEL_PATH=/path/to/checkpoints/LightNav-0 \
YOLO_MODEL_PATH=/path/to/yolo11n.pt \
SEG_MODEL_PATH=/path/to/yolo11n-seg.pt \
EPISODE_MANIFEST_PATH=/path/to/episode_manifest.json \
PILOT_MANIFEST_PATH=/path/to/pilot_manifest.json \
HABITAT_CONFIG=habitat_server/configs/vlnce_r2r.yaml \
GPU_IDS="0 1" GPU_MEM_UTIL=0.65 BASE_PORT=5555 \
MANIFEST_SEARCH_ROOT=/path/to/historical_results_parent \
  bash scripts/run_l3_official_style_ablation.sh \
    --historical-root /path/to/historical_88ep_root \
    --output-root output/l3_official_style_88ep
```

先干跑检查（不启动 Habitat、不写结果；把同一组环境变量再带一次）：

```bash
MODEL_PATH=/path/to/checkpoints/LightNav-0 \
YOLO_MODEL_PATH=/path/to/yolo11n.pt \
SEG_MODEL_PATH=/path/to/yolo11n-seg.pt \
EPISODE_MANIFEST_PATH=/path/to/episode_manifest.json \
PILOT_MANIFEST_PATH=/path/to/pilot_manifest.json \
HABITAT_CONFIG=habitat_server/configs/vlnce_r2r.yaml \
GPU_IDS="0 1" GPU_MEM_UTIL=0.65 BASE_PORT=5555 \
MANIFEST_SEARCH_ROOT=/path/to/historical_results_parent \
  bash scripts/run_l3_official_style_ablation.sh \
    --historical-root /path/to/historical_88ep_root \
    --output-root /tmp/l3_official_preflight --dry-run
```

### 8.3 运行后产物

```text
output/l3_official_style_88ep/
  provenance.json                 # manifest/checkpoint/seed/GPU/源哈希；historical_root
  l3_official_style_only/
    prompt_protocol.json          # mode, layer_activation(l3=official), builders(l3_official)
    prompt_diagnostics.jsonl      # 每 Step：budget, tokens, l3_expression, strategy_counts
    per_episode.csv               # 88 行 success/SPL/NE/nDTW...
    results.jsonl / summary.json
  comparison.json                 # 与历史 l3_original_only / l3_organized_only 的配对
  comparison.csv
  final_report.md
```

`comparison.json` 中重点看：

- `groups.l3_official_style_only.metrics.SR_pct.mean`；
- `paired_differences["l3_official_style_only_minus_l3_original_only"]`；
- `paired_differences["l3_official_style_only_minus_l3_organized_only"]`；
- `prompt.strategy_counts` 的分布（关系/极值/序数实际用了多少）；
- `budget_source` 是否全为 `token_aware_context`（若大量 `char_fallback_320`，
  实验解释力下降）。

### 8.4 本轮不执行

按任务要求，**不正式运行 88 Episode、不 commit、不 push、不操作 AutoDL**。
上面的命令只是交付物；需要人工在 AutoDL 上执行。

---

## 9. 关键代码路径

| 位置 | 作用 |
| --- | --- |
| `src/lightnav/memory/prompt_modes.py:46-49,156-158` | `l3_official_style_only` 的 plan（L1=None, L3=official, single） |
| `src/lightnav/memory/l3_prompt_organizer.py:208-360` | `collect()`：唯一候选检索/证据门槛/排序 |
| `src/lightnav/memory/l3_prompt_organizer.py:362-419` | `valid_history()`：历史证据校验（锚点复用） |
| `src/lightnav/memory/l3_official_prompt.py:88-107` | robot-relative 8 扇区方向 |
| `src/lightnav/memory/l3_official_prompt.py:202-247` | `support_relation`：`on` / `next to` 的几何证据规则 |
| `src/lightnav/memory/l3_official_prompt.py:341-415` | 锚点池：全图验证 + 历史门槛 |
| `src/lightnav/memory/l3_official_prompt.py:418-563` | 关系唯一性、锚点标签唯一、极值/序数稳定序 |
| `src/lightnav/memory/l3_official_prompt.py:514-563` | 每条事实的策略选择 |
| `src/lightnav/memory/l3_official_prompt.py:588-757` | 头部原子预算、整行选择、去重、诊断、LayerMemoryItem |
| `src/lightnav/memory/prompt_assembly.py:163-176` | 官方分支；不进入 L1/joint/concat |
| `src/lightnav/memory/manager.py:140-150,496-543` | 共享 collector、构造并传入官方 organizer |
| `src/lightnav/habitat/policy.py:666-998` | 通用 prompt-mode 推理路径、注入、token 复核 |
| `src/lightnav/memory/memory_context_builder.py:57-146` | 在 `</navigation_task>` 后注入并审计 |
| `src/lightnav/memory/token_budget.py:155-309` | 模型派生 token 预算、整行选择、overflow 重试 |
| `src/lightnav/habitat/prompt_artifacts.py:97-104,236-239` | 记录 `l3_official` builder 与 `l3_expression` |
| `scripts/show_l3_prompt_comparison.py` | 第 4 节对照输出的 CPU 复现脚本 |
| `scripts/run_l3_official_style_ablation.sh` | 单组 follow-up 入口 |
| `scripts/run_l1_l3_prompt_ablation.sh:24-110,600-700` | `--official-style` stage、历史根校验、最终配对 |
| `scripts/compare_l1_l3_prompt_ablation.py:64-66` | 比较器允许新模式 |

复现第 4 节输出：

```bash
PYTHONPATH=src python scripts/show_l3_prompt_comparison.py
```

脚本用同一 `ObjectMap` 手工构造 3 个 `ObjectNode` + `SpatialEvidence` +
`ObservedFromEdge` + `ViewpointNode`（Step 7），当前 Step 10，
`agent_pose = identity`；分别调用 `SpatialGraphAdapter`（320 chars, max_landmarks=2）、
`L3PromptOrganizer`（2000 chars, max_items）和
`L3OfficialPromptOrganizer`（2000 chars, 同一 max_items）。

---

## 10. 限制与后续

1. **88 Episode 导航效果未知**：本轮只做代码、CPU 测试和 Prompt 对照；71.59% /
   73.86% 是历史结果，新模式必须实跑才能比较。
2. **阈值是几何先验，不是调参结果**：0.5 m 水平邻接、0.75 m 垂直邻接带、
   `on` 的 `[-0.05, +0.10]` m 接触容差、0.35 footprint overlap、80% 目标包容度、
   0.75 锚点 footprint 比、15° 稳定序、22.5° 扇区，全部是固定的、可审计的常量；
   没有在 88 Episode 上调过。若实跑发现关系/极值表达有系统性问题，应做敏感性分析，
   而不是针对评测集手调。
3. **锚点可以不是 instruction 候选**：这是官方“large salient anchor”的要求，也与
   “候选集合不变”并不矛盾（候选事实集合不变；锚点只作为参照，不单独注入位置）。
   如果审稿要求更严格的“只用候选事实”，可以把锚点池限制在 `selected_facts`，
   但那样关系策略在 `max_landmarks=2` 下几乎不会触发。
4. **序数默认休眠**：默认 `max_landmarks=2` 时只可能出现 `leftmost/rightmost`，
   `second/third` 需要 ≥3 个同类候选。本轮保留实现和单测，不把它当作主要贡献。
5. **320 字符 fallback 下事实更少**：正式实验应确认 `budget_source` 为
   `token_aware_context`；否则新模式会被头部长度拖累。
6. **L1 检测仍运行**：这是 L3 Evidence 关联所需，不是 L1 注入。若审稿把
   “L1 关闭”理解为连检测器都不跑，需要先确认那会改变 L3 的 linked Evidence，
   从而违反“L3 候选证据不变”。
7. **L3 ObjectMap 是 Episode-local**：`manager.reset_episode`/`close_episode`
   会清空它，所以 `past_episode`/混合 Episode 历史在当前生命周期下不会出现；
   `latest_step` 和旧 Episode 测试是防御性的，不表示本次实验真的会用到跨
   Episode 地图。
