# RGM-LightNav Agent 项目交接

> 审计基准：2026-11-10 本地工作树，分支 `codex-adaptive-memory`，HEAD `0b960a81ce734e37aab4550c2c79b5076b1957f2`。RGM V1 多数代码目前是 **未提交的工作树内容**；HEAD 不是完整 RGM V1 版本。本文不声称已访问 AutoDL 或取得真实评测结果。配套图见 [Agent 架构](AGENT_ARCHITECTURE.md)，对话速览见 [ChatGPT 上下文](CHATGPT_CONTEXT.md)。

## 1. 项目身份与代码来源

**项目主体是一个依托 LightNav 的具身导航 Agent。** LightNav 提供预训练视觉语言导航模型、视频帧历史/SlowFast 采样、原始 VLN Prompt、`<traj_k>` 或 RVQ 动作 token、`(H,3)` 相对路点解码、速度控制接口、推理后端与 Habitat 客户端。这些都是基础能力，不是 RGM 的算法贡献。代码入口分别见 `src/lightnav/inference/policies.py:16,106-147`、`src/lightnav/inference/samples.py:134-150`、`src/lightnav/tracking.py:60,102-148`、`src/lightnav/inference/engine.py:236-325`、`src/lightnav/velocity.py:19-56`。

**本项目新增的决策层** 在每次动作前从当前 RGB 和已有命令建立 Episode 内证据，执行 YOLO 检测、可选 DINOv2 地点识别，维护工作/情节/语义与图记忆，经 Write、Retrieval、Prompt 三道 Gate 判定哪些视觉事实能进入 LightNav 的 Prompt。LightNav 仍负责最终路点和动作生成。它是带显式、Episode 局部记忆与证据筛选的闭环视觉语言导航 Agent；闭环仅指“动作后获得下一帧再更新”，**没有真实位移/碰撞反馈闭环，也没有已实现的自适应感知调度**。这使它区别于只依据视频历史和指令逐步解码的普通 VLN Policy，但目前不能把它描述成完成了物理地图验证的自主 SLAM Agent。

**可核实的归属边界。** 本地 `origin/main` 到 HEAD 的 Git 差异显示 `src/lightnav/memory/` 在上游为零个文件，本分支提交了最初的 MemoryManager、Writer、YOLO、DINO、SceneGraph、三层记忆和实验工具；`habitat/policy.py`、`runner.py`、`inference/engine.py` 是在原 LightNav 文件上修改。当前工作树又修改了 `manager.py`、`place_recognition.py`、`policy.py`、`runner.py` 等，并新增未跟踪的 `rgm_config.py`、三个 Gate、`object_association.py`、`place_verification.py`、88 Pilot 脚本与测试。这个结论是 **仓库版本来源**，不等于能从 Git 证明个人作者身份；实际作者归属仍需项目方确认。对比命令：`git diff --name-status origin/main...HEAD`、`git status --short`。

**算法贡献的准确表述。** V1 的可审计贡献是：基于视觉证据出处和指令相关性的分层记忆准入、检索与 Prompt 门控；D 组增加类别时序共识、DINO 全局候选加 Patch 几何复核，以及没有物理反馈时的拓扑弃权规则；无有效记忆时精确回到 LightNav 基础 sample。这些是已编码的机制，**不是已证明带来成功率或算力收益的研究结论**；相对于相关论文的科学新颖性也尚未做文献对照。`AdaptiveController` 和 `RGMObservationEvidence` 类在代码中不存在；现有证据类名为 `RGMEvidence`（`src/lightnav/memory/schemas.py:250-260`），`adaptive_extraction_enabled=True` 明确报错（`rgm_config.py:38-40`）。

## 2. 输入、状态、输出和证据边界

| 层 | 实际可见字段 | 当前用途与限制 |
|---|---|---|
| Habitat 环境 | `rgb: H×W×3 uint8`、`depth: H×W float32`、`instruction`；还可能有 `goal_distance/progress`，`info` 中有成功率/距离/速度范围（`habitat_server/lightnav_habitat/vlnce.py:102-110,562-563`） | Runner 拿观测和评价信息；Evaluator 衍生的距离/进度不能作为 Agent 证据 |
| Policy → `ObservationPacket` | `step_id`、瞬时 RGB、指令、上次发出的 `velocity_control` 命令、上次模型文本（`policy.py:229-239`；`schemas.py:187-205`） | 没有 Depth、Agent Pose、里程计、碰撞、实际执行位移；`info` 被有意排除 |
| Agent 状态 | `TrackingAgent` 视频历史；`WorkingMemory` 最近 8 步；`EpisodicMemory` 原始事件与采纳索引；`SemanticIndex` 类别汇总；`TopologicalMemory` 视觉地点；Gate 的 Episode 状态（`manager.py:88-116,122-146`） | 状态在 Episode 关闭时清空；DINO 模型可重用，但地点关键帧和记忆不跨 Episode |
| 输出 | 可选 Prompt 记忆文本与诊断；LightNav 文本 token → `H×3` `[forward_m,lateral_m,yaw_rad]` → Habitat `velocity_control`（`policy.py:406-473`） | RGM 不直接选择路点，也不改 RVQ/动作解码 |

`ObservationPacket.to_dict()` 不保存 RGB（`schemas.py:195-205`）；工作记忆也只保存结构化观察（`working_memory.py:54-57`）。视觉样本当前由 `TrackingAgent` 的帧缓存供 LightNav 使用，DINO 关键帧保存的是 Patch/描述子而非原 RGB。不能把环境可提供 Depth 等同于 Agent 已使用 Depth。`runner.py:969-1024` 才读取 `info` 中的 `distance_to_goal`、`success` 等评价量；Policy `act(obs, info)` 虽接收 `info`，却没有把它送入记忆。速度范围来自环境重置 `info` 用于动作标定，不是目标 oracle。

88 Pilot 的 `strict_full_eval` 在服务端还要求关闭以 `distance_to_goal` 判断停滞的早停规则（`habitat_server/lightnav_habitat/vlnce.py:329-339,583-595`）。这避免该 evaluator 信号通过早停改变 Agent 的动作序列；仍须检查实际启动快照确实启用了严格模式。

## 3. 从 Habitat 到下一轮动作的真实调用链

下表中省略目录的 `memory/*.py` 均位于 `src/lightnav/memory/`，`samples.py` 和 `frame_preprocessing.py` 位于 `src/lightnav/inference/`。

| 顺序 | 代码与函数 | 输入 → 输出；关键维度 | 异常或回退 |
|---|---|---|---|
| 1 | `habitat_server/lightnav_habitat/vlnce.py:102-110` 的 `VLNCEEnv.reset/step`；`src/lightnav/habitat/remote_env.py:115-129` 的 `RemoteEnvClient` | ZMQ 返回 `obs`/`info`；RGB `H×W×3 uint8`、Depth `H×W float32`；执行 `velocity_control` | 远程调用失败向上层抛出；不构造 RGM 证据 |
| 2 | `src/lightnav/habitat/runner.py:687,737-750,883-921,946` 的 `run_habitat_eval` | 构造 engine/env/policy，验证 Manifest，逐 Episode `reset`、逐步 `policy.act(obs,info)`，随后 `env.step(action)` | 严格模式校验 Episode 终止和指标 `:989-999`；恢复时修复末尾 JSONL `:755-770` |
| 3 | `src/lightnav/habitat/policy.py:170-185,216-239` 的 `TrajVocabVLNCEPolicy.reset/act` | 提取指令；`TrackingAgent.observe(obs['rgb'])`；建立 `ObservationPacket`（不含 `info`） | RGB 不存在时 Runner 中断 `:906-908`；写入器拒绝步号倒退/指令变化 |
| 4 | `src/lightnav/inference/policies.py:106-147`、`frame_preprocessing.py:57`、`samples.py:29-150` | 当前 RGB 转模型张量，缓存帧并按 checkpoint 的 SlowFast/窗口规则形成 `[T,C,H,W]` 视频；`build_vln_traj_sample` 得 `video_segments`、`conversations` | 视频尺寸由模型配置决定；不由 RGM 改写 |
| 5 | `src/lightnav/memory/manager.py:148-190`、`memory_writer.py:96-127,202-259` | Packet → 直方图、动作事件、YOLO 的 `DetectedObject(label,confidence,xyxy)` → `MemoryEvent`；固定步频提取 | YOLO 异常记录 `last_error` 并返回空提取（`yolo_scene_extractor.py:85-122`）；非法证据在提交事件前报错 |
| 6 | `memory_write_gate.py:46-89`；`manager.py:179-187,283-415` | 原始事件 → `RGMEvidence`/`GateDecision`；DINO 在固定步频给 C 全局 384 维描述子、D 全局+`N×384` Patch；D 产生 MATCH/NEW/UNKNOWN；更新视觉拓扑 | DINO/地点更新异常记录错误，D 路径弃权 UNKNOWN；无物理位移不生成 VERIFIED_TRAVERSAL |
| 7 | `manager.py:200-245`；`working_memory.py:190`、`episodic_memory.py:180`、`topological_memory.py:351` | 查询 `MemoryQuery(instruction,current_step,place_id,top_k,max_chars)` → L1/L2/L3 `LayerMemoryItem` 列表；L3 拓扑启用时取 TopologicalMemory，否则 SemanticIndex | 未采纳原始事件仍保留审计，但不会作为 RGM 检索证据 |
| 8 | `memory_retrieval_gate.py:21-81`、`memory_fusion.py:39-135`、`memory_prompt_gate.py:21-79` | 按出处/指令/空间可信性过滤，按事件 ID 去重，限量合并，形成 `MemoryContext.text`；最多 3 条/400 字符 | 缺来源、未采纳、不相关、未验证空间推断被拒；可返回空文本 |
| 9 | `policy.py:241-272`；`memory_context_builder.py:19-140` | 基础 sample → 只在 `</navigation_task>` 后插入安全文本；`audit_prompt_injection` 检查原模板可逆、视频不变 | 无有效记忆直接把 `base_sample` 传 Fake/真实 Engine；审计 `ValueError` 回退基础 sample |
| 10 | `inference/engine.py:236-325`；`policy.py:406-473`；`tracking.py:102-148`；`velocity.py:19-56` | Engine 生成 traj/RVQ token；解为 `H×3` 路点；选第一可驱动路点，`v=forward/dt`、`ω=degrees(yaw)/dt`，输出归一化动作字典 | token 解码 `ValueError` 返回零速度；RGM 不改这一路径 |
| 11 | `runner.py:946-1024`；`manager.py:192-198` | Habitat 执行动作并给下一轮 `obs`；命令先暂存，下一个 RGB 到来才记为历史事件；Runner 单独写诊断与指标 | 收到下一帧只证明 step 返回，不证明实际发生位移；每个 Episode `close_episode` 清空 `:981-987` |

注意：第 4 步的模型视频 sample 与第 5 步的 RGM 证据包都是由 Policy 对同一帧构建；具体执行顺序是 `observe(RGB)` → MemoryManager `observe(Packet)` → `build_vln_traj_sample` → 检索注入（`policy.py:222-247`）。RGM 以 `ObservationPacket`/`MemoryContext` 接在 Policy 外围；Engine、权重和动作解码继续使用 LightNav 的现有接口。

## 4. 感知、关联和地点决策

**图像质量与变化。** `MemoryWriter._rgb_histogram/_appearance_delta` 计算各 RGB 通道 8 桶直方图变化；`0.45` 记视觉变化、`0.65` 记更强变化（`memory_writer.py:128-145,312-329`）。DINO 的图像质量是灰度标准差除以 64、截断到 1；小于 `0.20` 则地点判定 UNKNOWN（`place_recognition.py:169`；`place_verification.py:97-100`）。这两者都是廉价代理；光照、旋转、遮挡可以改变值，未经过真实场景校准。

**YOLO 与物体关联。** `YOLOSceneExtractor.should_extract` 在 step 0 和每 N 步运行，88 Pilot 中 N=5；把 RGB 转 BGR 供 Ultralytics，读取类别、检测分数和 `xyxy`（`yolo_scene_extractor.py:81-112,129-165`）。`MemoryWriter` 对每个框创建 `SceneGraph` 物体节点和原始事件（`memory_writer.py:202-242`），Write Gate 再标注是否采纳。默认检测阈值 `0.25`；与指令词完整匹配且分数 ≥`0.60` 的观察可以直接采纳，**C 即使关闭时序共识也会采纳**；D 还允许最近 3 次成功检测中至少 2 个不同步长的同类证据（`memory_write_gate.py:72-86`；`object_association.py:21-57`）。同类不等于同一物体：静态 `instance_match` 函数未接入，当前没有跨帧物理实例轨迹或 ROI 对齐。

**DINO 两阶段。** `DINOv2PlaceDescriptorExtractor._extract_features` 一次 forward 读取 `last_hidden_state`，移除 CLS/可选 register tokens，对 Patch 均值池化并归一化为 384 维全局向量；D 额外输出按处理后 `pixel_values` 网格取样、归一化的 `N×384 float16` Patch，记录索引、网格和原/处理后尺寸（`place_recognition.py:156-223`）。网格匹配的是 **模型预处理后的图像**；没有 crop/resize 逆映射，因此不能拿 Patch 坐标直接定位原始 YOLO 框。C 使用只传全局向量的 `extract()`（同文件 `:229-231`）。D 的 `TwoStagePlaceVerifier.decide` 先取全局余弦 Top 3（候选 ≥`0.75`；新地点边界 ≤`0.65`），再以 Patch 互为最近邻、余弦 ≥`0.70`、至少 8 对、OpenCV RANSAC 内点比 ≥`0.50` 复核；歧义、质量低、OpenCV 缺失或证据不够时返回 UNKNOWN；新地点通常要求第二次独立观察（`place_verification.py:29-64,74-162`）。阈值是 V1 规则，不是校准概率。C 只用 `TopologicalMemory` 的全局阈值 `0.85`，回环证据弱于 D。

**两套图。** `SceneGraph` 保存 YOLO 框对应的原始物体节点，若提取器真的给 `place_label` 才可造语义地点及 `possible_transition`（`memory_writer.py:137-171`）；当前 YOLO 返回物体，不返回房间名，故不能声称 YOLO 建了房间图。`TopologicalMemory.observe` 用描述子创建/合并视觉地点，关联同帧已采纳类别；只有前后地点 ID 不同且存在命令事件 ID 才造 `observed_transition`（`topological_memory.py:188-335`）。D 额外用 RGB 直方图差阈值 `0.35` 限制边，UNKNOWN 不入图（`manager.py:339-405`）。这些是视觉与命令的观察关系；没有 Agent Pose、碰撞、测程，不能证明实际跨越或边可通行，`verified_traversal=False` 写在事件元数据。更细的工程风险：`MemoryWriter._has_positive_forward_command` 检查的是归一化 `linear_velocity > 0`（`memory_writer.py:341-350`），而 `velocity.py:44-55` 将物理速度映射到 `[-1,1]`；该条件并非准确的“正向物理位移”判据，甚至不等价于所有正向速度命令。

| V1 配置/默认值（`rgm_config.py:8-36`） | 作用 |
|---|---|
| `min_candidate_confidence=0.25`；`min_instruction_landmark_confidence=0.60` | 检测候选与指令词直接采纳门槛 |
| `temporal_window=3`；`temporal_min_support=2` | D 的成功提取窗口与不同观察步数；C 关闭此准入支路 |
| `place_top_k=3`；`place_global_candidate_min=0.75`；`place_global_new_max=0.65` | DINO 全局候选、边界弃权和新地点判断 |
| `place_local_cosine_min=0.70`；`place_local_min_matches=8`；`place_local_inlier_min=0.50`；`place_ransac_reprojection_threshold=0.08` | Patch 互匹配与局部几何复核 |
| `place_min_image_quality=0.20`；`place_global_margin_min=0.03`；`place_new_temporal_similarity_min=0.85`；`place_keyframes_max=3` | 低质/歧义弃权、新地点复看和关键帧上限 |
| `topology_visual_delta_min=0.35`；`prompt_max_records=3`；`prompt_max_chars=400` | D 的观察边附加限制、最终记忆注入预算 |

## 5. 记忆生命周期与 Prompt 决策

| 层/模块 | 写入、更新、淘汰与检索 | 当前可信性边界 |
|---|---|---|
| L1 `WorkingMemoryStore` | 当前指令、step、上次动作/预测、最近 8 个结构化观察（`working_memory.py:46-64,114-140,190-282`）；队列自然淘汰 | 观察事件可溯源；无原 RGB，也无位姿 |
| L2 `EpisodicMemory` | 保存所有原始 `MemoryEvent`，只把 `rgm_adopted` 的关键事件建索引/片段；相邻同类片段合并；Episode 结束清空（`episodic_memory.py:83-128,180-252`） | 拒绝事件仍保留审计，不得误用为导航事实 |
| L3 `SemanticIndex` | 对采纳类别作汇总（`semantic_index.py:84-150`）；拓扑关闭时供 L3 检索 | 不是跨 Episode 知识库或向量数据库 |
| L3 `TopologicalMemory` | DINO 视觉地点描述子均值更新、同地点类别证据、观察边；每 Episode 清空（`topological_memory.py:188-335,351-507`） | 地点 ID 是视觉假设；观察边不具物理通行认证 |
| `MemoryManager` | `reset_episode` 先关闭旧状态，`close_episode` 清空全部存储/关键帧/Gate（`manager.py:122-146,250-281`） | 测试能验证代码清空；真实长跑跨 Episode 泄漏仍需日志审核 |

检索按 `MemoryQuery` 的指令、当前 step、地点、条数/字符预算组织（`schemas.py:97-112`）。Write Gate 先按出处、指令与类别时序决定是否可索引；Retrieval Gate 再拒绝未采纳、无出处、指令无关和未经物理验证的空间断言，先排序再按事件 ID 去重；`MemoryFusion` 合并层级、保留独立证据并受预算限制；Prompt Gate 只把指令相关的视觉类别写成保守句子，Step 0 仅允许当前高分观察，之后当前观察需至少两个独立观察步支持（`memory_prompt_gate.py:21-79`）。它没有学习式置信度模型，也没有完整的互斥命题冲突图；主要以拒绝不确定空间断言和丢弃无关类别来控制污染。诊断字段见 `policy.py:284-405`：候选/关联/三 Gate 理由、DINO TopK/局部内点、拓扑状态、Prompt Hash、字符数和模型延迟。

基础 Prompt 由 `build_vln_traj_sample` 按 checkpoint 的 SlowFast/窗口规则生成（`samples.py:29-150`）；LightNav 原来的 `generate_from_frames` 内部也调用同一 builder 然后 `generate`（`inference/engine.py:315-317`）。RGM 只在 `</navigation_task>` 后添加经 HTML 转义的记忆块；`audit_prompt_injection` 检查删除记忆块后原 Prompt 逐字恢复，角色、非用户消息和视频对象不变（`memory_context_builder.py:19-140`）。没有有效记忆时 Policy **直接**传 `base_sample`（`policy.py:253-257`），异常回退同样传基础 sample（`:267-272`），不注入 `place_000` 等内部编号。模型输出仍按原 Flat/RVQ 解码并走原速度映射。

## 6. 正式实验协议与能回答的研究问题

当前协议由 `scripts/run_rgm_pilot_88.sh:1-197` 和 `scripts/compare_memory_ablation.py:939-1125` 定义。`scripts/run_memory_ablation.sh:8-10` 的 `--rgm-88` **直接 `exec` 到前者**；旧脚本其余入口仍保留 220 Pilot 与正式 1839 Episode 路径。历史 C=`hierarchical_topology`、D=`rgm_adaptive` 命名已过时；本轮 **B 取消**。

| 组 | 运行时开关 | 比较目的 |
|---|---|---|
| A `baseline_off` | 原始 LightNav，记忆关闭；尝试严格兼容复用旧 220 中选中的 88 条，否则重跑 88 | 基线导航结果 |
| C `rgm_gates` | L1/L2/L3+YOLO+DINO；Write/Retrieval/Prompt 三 Gate 开；YOLO 时序、DINO 两阶段、拓扑运动验证关 | 基础证据门控效应 |
| D `rgm_full` | C + 上述三项验证开 | 验证策略的附加效应 |

88 条由旧 220 Manifest 派生：11 场景各 8 条，seed 42，长度分层 3/3/2，并检查 ID 唯一、父 Manifest 和数据来源（`sample_r2r_pilot.py:297-380`）。A/C/D 必须是同一 `(scene_id, episode_id)` 集合，公共模型路径/配置指纹、后端 `vllm_local`、种子、R2R `val_unseen`、500 步、历史帧、动作解码、Prompt 预算、YOLO/DINO 每 5 步调用等受运行快照和比较器核验（`run_rgm_pilot_88.sh:40-129`；`compare_memory_ablation.py:959-1069`）。D 与 C 的可变内存标志仅三项；实际是否完全同权重仍以运行快照与权重 hash 验证。`reuse_legacy_a.inspect_legacy_a` 要求旧 A 覆盖完整、所选 ID 匹配、公共配置与源码/模型/动作 bundle 指纹兼容；缺指纹返回 `rerun_a_required`，不会静默复用（`scripts/reuse_legacy_a.py:32-137`）。旧 A 跨硬件复用时不比较延迟（`compare_memory_ablation.py:1087-1094`）。中断恢复检查固定快照、Manifest 和 JSONL 覆盖，不允许悄悄补成另一套 Episode（`runner.py:742-770`；`run_rgm_pilot_88.sh:59-115,131-164`）。

**指标解释。** SR 是终点达到成功条件的 Episode 比例；OS 是轨迹曾经过目标成功半径的比例，可能高于 SR，属于 evaluator 指标；SPL 对每条成功轨迹按最短路径长与实际路径长之比加权；NE 是终点到目标的导航误差（米，低为好）；nDTW 是预测轨迹与参考轨迹的一致性分数（高为好）。实现以 `success/oracle_success/spl/ndtw/final_distance` 写结果，比较器生成 SR、OS、SPL、NDTW、NE、平均步数与模型推理延迟（`runner.py:1008-1024`；`reuse_legacy_a.py:165-170`；`compare_memory_ablation.py:28-35,939-1125`）。`prompt_chars` 是字符数，不是 tokenizer token 数；当前没有充分的实际输入/输出 Token 使用量证明。YOLO/DINO 运行时间有诊断字段，但固定步频意味着 D 没有减少视觉调用次数的设计基础。要主张算力节省，须统计成功检测/识别次数、各自耗时、模型 token、端到端延迟与质量，并在同硬件条件比较。

**目前不能得出的结论。** 本地没有 `output/`、权重或 R2R 结果，因此不能断言 RGM 提高 SR、降低推理成本、减少污染的实际幅度，也不能称 88 Pilot 已完成。88 条是探索性配对 Pilot，比较器计算 Episode 配对差和 bootstrap 区间；显著性/泛化仍需正式独立评测。A/C/D 相同初始 Episode 与公共配置提高公平性，但策略改动使后续轨迹和图像不同；C 对 D 的组合验证消融也不能单独识别三个验证分量的贡献。C/D 同步固定 YOLO/DINO 调度，不能将任何延迟差归因于“自适应感知”。若要支持研究结论，需报告分场景/路线长度的 SR/SPL/NE/nDTW、拒绝原因与人工核验的错误记忆率、模型与视觉成本，以及 A/C/D 每项配置/权重/数据哈希。

## 7. 实现与验收状态

| 能力 | 当前状态 | 本地可见证据与缺口 |
|---|---|---|
| LightNav 原始视频→路点→动作 | 已有基础路径 | `policy.py:406-473`；当前无权重/GPU，未本机运行真实推理 |
| TrackingAgent、Policy、Runner、ZMQ Habitat 接口 | 已编码并有 Fake Engine/合成环境测试 | `tests/habitat/test_policy.py`、`test_runner.py`；本机缺 NumPy/PyTorch/pytest，相关集成测试未重跑 |
| YOLO 检测、Write Gate、类别级时序 | 已编码；类别级，不是实例追踪 | `tests/memory/test_rgm_gates.py` 10/10 通过；真实 YOLO 权重未本机验证 |
| DINO 全局/Patch 一次 forward、局部几何复核 | 已编码，依赖 torch/Pillow/OpenCV；C 不启用两阶段 | `tests/memory/test_place_recognition.py:152-198` 有 Fake 模型测试，但本机依赖缺失，未验证实际 DINO 权重 |
| 视觉拓扑、三层记忆、Fusion、三 Gate | 已编码；物理通行证据不可用 | `manager.py:148-245,339-415`；RGM 单测可运行，真实 Habitat 未验证 |
| Prompt 精确空记忆回退、Step 0 条件化 | 已编码；Policy 集成测试存在 | `policy.py:253-272`；`tests/habitat/test_policy.py:602-655` 因本机依赖缺失未重跑，纯逻辑测试通过 |
| Adaptive Controller、跨帧实例跟踪、真实里程计/碰撞闭环 | 仅目标或未接线 | `rgm_config.py:38-40` 拒绝自适应；`object_association.py:59-91` 仅有未接线静态匹配器；Packet 无运动传感字段 |
| 88 Pilot 工具和旧 A 严格复用 | 已编码，合成单测通过 | 本次 `test_rgm_compare.py` 3/3、`test_rgm_pilot_88.py` 2/2 通过；真实旧 Manifest/AutoDL 结果未访问 |
| 真实 R2R Habitat 评测、导航收益 | 尚未验证 | 本地无 `output/`、`checkpoints/`、`.venv/`；不能出具 SR/SPL 提升结论 |

本次只读验证：`PYTHONPATH=src:. python3 -m unittest discover` 分别运行 `test_rgm_gates.py`、`test_rgm_compare.py`、`test_rgm_pilot_88.py`，合计 **15/15 通过**。另外重跑 `test_rgm_manager.py`、`test_rgm_place.py`、`test_rgm_config.py`，共 **10 项全部跳过**：`test_rotation_and_rgb_shift_do_not_create_rgm_topology_edge`、`test_rotation_and_stop_are_not_rgm_forward_motion`、`test_unknown_place_keeps_one_reference_and_no_transition`、`test_global_match_needs_local_consistency`、`test_initial_reference_and_global_ambiguity`、`test_keyframes_remain_bounded_and_keep_initial_reference`、`test_local_mutual_matches_have_geometric_inliers`、`test_top_k_local_check_can_rescue_second_candidate`、`test_two_locally_supported_candidates_are_ambiguous`（均缺 NumPy）；`test_c_d_differ_only_in_the_three_extensions`（缺 NumPy 和推理测试环境）。`bash -n` 对 88 launcher、旧 launcher、Habitat launcher 通过；`git diff --check` 通过。本机 Python `3.10.12`，`numpy`、`PIL`、`torch`、`pytest`、`cv2`、`vllm`、`habitat_sim` 均不可导入；因此这 **不是** `make test`、GPU/权重集成或真实 Habitat 验收。此前会话中曾有更大测试记录，但这里不以无法复现的数量代替本次证据。

## 8. GitHub / AutoDL 工程状态与交接步骤

本地 Git：分支 `codex-adaptive-memory`；HEAD `0b960a8`（`2026-11-08T11:08:30+08:00`，`fix: rebalance R2R pilot scene quotas`）；跟踪 `personal/codex-adaptive-memory`。`origin=https://github.com/lightorigins/LightNav-0.git` 是上游，`personal=https://github.com/66666-visit/LightNav-0-codex.git` 是本地配置的个人远端；远端最新实际状态未联网核验。本次开始前工作树约 **23 个已跟踪文件修改、22 个未跟踪路径**，含先前审查产物与 RGM 源码/脚本/测试；这些均未清理。此交接新增的三份文档不在该基线计数中。先前开发、验收和修复不能仅凭未提交工作树视作云端已经部署。

工程链条应是：本地 WSL/Codex 审查和测试 → 经项目方审核后才 Git commit/push → AutoDL 人工 `git pull` 并核对 commit/环境/资产 → GPU 推理与 R2R 实验 → 结果及 Manifest/日志回传 → 本地比较。**本次只写文档，不执行链条后续操作。** 本地 `pyproject.toml:23-26,61` 规定 torch 下限、`transformers==5.8.0`、`vllm==0.19.1` 和 `nvidia-cutlass-dsl==4.5.2`；Habitat-Sim 在独立环境（`AGENTS.md`），这些是依赖声明，**不是 AutoDL 实测版本**。本地也没有权重、数据、结果目录。AutoDL 项目目录、当前 commit、Python/CUDA/PyTorch/vLLM/Habitat-Sim 实际版本、`MODEL_PATH`/`YOLO_MODEL_PATH`/`DINO_MODEL_PATH`/`DATA_PATH`/`SCENES_DIR`、既有 220 结果目录，均 **未知**。

请项目方日后在 AutoDL **只读**执行以下核验并回传输出（不要回传凭据）：

```bash
pwd
git status --short --branch
git rev-parse HEAD
git remote -v
python3 --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 - <<'PY'
import importlib.metadata as m
for package in ('torch', 'torchvision', 'transformers', 'vllm', 'habitat-sim', 'habitat-lab', 'ultralytics'):
    try: print(package, m.version(package))
    except m.PackageNotFoundError: print(package, 'not installed in this Python')
try:
    import torch
    print('torch_cuda_runtime', torch.version.cuda, 'cuda_available', torch.cuda.is_available())
except ImportError:
    pass
PY
for name in MODEL_PATH YOLO_MODEL_PATH DINO_MODEL_PATH DATA_PATH SCENES_DIR; do
    value="${!name:-}"
    printf '%s=%s\n' "$name" "${value:-<unset>}"
    [ -z "$value" ] || ls -ld -- "$value"
done
ls -ld output/r2r_pilot_220_v2 output/rgm_pilot_88_v1 2>&1
```

Habitat 独立 conda Python 还需在 **对应环境** 重跑版本查询；`nvidia-smi` 的 CUDA 显示驱动能力，不代替 PyTorch runtime 版本。旧 220 A 结果只有经 `reuse_legacy_a.py` 的文件/Manifest/指纹检查才可复用。

**未来经过批准后的唯一 88 Pilot 启动入口**（此处仅供交接，未执行）：

```bash
MODEL_PATH=<checkpoint-dir> YOLO_MODEL_PATH=<yolo-pt> DINO_MODEL_PATH=<dinov2-dir> DATA_PATH=<r2r-root> \
  bash scripts/run_memory_ablation.sh --rgm-88 --parent-manifest <old-220-pilot-manifest.json>
```

脚本默认输出 `output/rgm_pilot_88_v1`，旧 A 默认查 `output/r2r_pilot_220_v2`；若 AutoDL 实际路径不同，使用 `--legacy-root`/`--output-root`。`--resume` 只用于中断恢复，同一 run snapshot 必须完全相符（`run_rgm_pilot_88.sh:59-115`）。未授权前不得运行真实评测。

## 9. 接手工程师的判断与下一步

**真正解决的问题：** 让只带帧历史的 VLN Policy 能利用当前 Episode 的视觉证据，同时控制“检测一次就写成事实”“外观相似就宣称地点相同”“内部地图状态误入 Prompt”等污染路径。可靠之处是证据来源、Episode 清空、Gate 理由、Prompt 可逆审计和严格实验键值校验；这些由代码及合成单测支持。仍属假设的是检测分数/手工阈值可代表真实可信度、类别时序代表物体稳定、Patch 几何足以识别地点、命令加颜色变化足以描述拓扑运动，以及记忆注入能提高真实导航指标。

**当前最薄弱处** 是没有真实自运动与碰撞证据，却希望使用拓扑关系辅助导航；其次是缺真实权重/场景测试与标注错误记忆率；第三是“自适应”仍只是目标命名。当前 88 Pilot 工具本身可支持配对探索，但没有结果，无法支撑成功率、成本或因果贡献的论文结论。

继续开发优先做三件事：

1. 在保持 evaluator 隔离的前提下定义可部署的位移/碰撞反馈，校准归一化动作与物理速度，明确 `OBSERVED_TRANSITION` 升级为 `VERIFIED_TRAVERSAL` 的证据契约；做无位移/碰撞/转头反例测试。
2. 在可用 GPU/权重环境完成 C/D 的 DINO/YOLO 真实推理和 Fake Engine Policy 集成验收，运行获批的 88 配对 Pilot；同步人工核查错误注入、地点误匹配和诊断成本，再决定阈值。
3. 只有在固定步频 V1 的导航与成本基线建立后，设计并单独消融真正的 Adaptive Controller（何时执行 YOLO/DINO、拒绝和预算策略），用实际调用次数、token、延迟和 SR/SPL 证明净收益。
