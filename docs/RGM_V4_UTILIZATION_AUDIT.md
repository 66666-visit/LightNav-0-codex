# RGM V4 Memory Utilization：根因审计与最小修复

审查基线为 `codex-adaptive-memory` 的 `efb82b0d818e174d1eea29605639628f219e92c4`。本报告依据本地源码和确定性 CPU 测试。用户提供的 88 Episode 汇总为：6557 次决策，4508 步有历史 Top-K，4203 步有历史候选却选择 `NONE`，305 步实际注入，70/88 个 Episode 从未注入，3 个 Episode 贡献约 77.4% 的注入，SR 为 72.7%。这些数值是**旧实验的汇总**，本轮没有重新运行 Habitat，也没有在工作区找到可逐候选核对的 V4 JSONL 日志或实验产物。

## 按严重程度排序的结论

| 等级 | 结论 | 证据与处置 | 旧实验影响的可证程度 |
| --- | --- | --- | --- |
| 高：已证实的实现缺陷 | 抑制重复注入的签名取自格式化**之前**的整组候选。同一已写入 Prompt 的证据 A，可在 Top-K 从 `(A,B)` 变成 `(A,C)` 时再次注入；即使格式化预算只允许 A，整组签名仍变化。A→B→A 也可能重注入 A。 | `v4_utilization.select` 的候选签名、`v4_formatter.format` 的预算顺序与 Fake Engine 复现。现按最终输入中实际验证的 Evidence ID 记录已注入源；后续过滤证据集合已全部注入的事实，记 `repeated_without_new_evidence`。新增支持证据的复合关系仍可附带旧来源。Policy 只在 Processor／vLLM 最终 Token 检查通过后确认。 | 能证明机制存在与修复生效；不能证明旧实验中 3 个高注入 Episode 的集中现象由此造成。 |
| 中：已证实的 Schema 缺口 | `MemoryWriter` 正式产生、V3 Store 保存、V3 Retriever 可召回的 `place_hypothesis`，旧 V4 Context 未列入支持的视觉事件，因此落入 `unsupported_event_type`。 | `memory_writer.py` 的固定描述 `RGB extractor described …; place identity unresolved`、`manager._persist_v3_events`、`v4_context.build_navigation_memory_context`。现只识别该固定描述并抽取标签，以 `INFERRED`、地点身份／访问／通行 `UNKNOWN` 输出；真实来源、Episode、Step 仍保留。 | 可以证明有此事件时会被错误丢弃；现有 YOLO SceneExtractor 不产出 `place_label`，所以不能把旧实验的 3691 次 `unsupported_event_type` 归因于此类型。 |
| 高影响可能、尚非缺陷 | `unrelated_history` 的严格字面相关性：V3 可按描述、事件类型或指令词的**部分**重合召回，V4 只接受语义标签所有规范化词均在指令锚点内。 | 例如标签 `dining table` 对 `find the table` 可由 V3 召回，V4 判为不相关。它也能挡住真无关记忆；旧日志缺候选内容，无法量化误拒。未整体放宽规则。 | 14155 是拒绝事件次数，不是 14155 条被证实有用的记忆。 |
| 保守的证据约束 | `unrelated_relation` 要求拓扑边端点的已有语义标签与指令相交；纯 DINO 地点边没有可验证的语义关联。`visible_in_native_rgb` 仅在同 Scene／Episode／rollout 的精确 Step→frame ID 出现在当前视频样本中成立。 | 保留关系的通行状态 `UNKNOWN`；不把视觉相似当地点身份，也不把时间间隔当 RGB 重复的证明。 | 不能仅由 4237 或 3740 的计数推断两规则过严。 |
| 观测能力限制 | `PROGRESS_AUDIT` 的已验证通行分支需要独立测得的位移；`REVISIT_AUDIT` 需要当前和至少两个独立历史 DINO 观察、同 Scene 且有关联地标。 | 当前 Habitat 在线观测没有独立里程计证据；若没有启用或成功提取 DINO 描述子，重访模式无输入。状态 `COMPLETION_UNKNOWN` 与“仅观察、通行未知”的进度审计可以共存。 | 仅凭汇总无法判断 70 个从未注入的 Episode 分别缺少哪类证据，也无法解释 3 个 Episode 的全部注入。 |

此处“缺陷”均由源码路径和复现测试确定；“可能影响”不等于对 88 Episode 的逐条归因。旧实验的成功、失败、SR 或 SPL 标签没有进入在线规则。

## 四种模式的真实触发条件

`MemoryManager.retrieve` 使用原有 V3 四通道检索与 RRF，Context 展开原始 `EvidenceRecord`，再由 `MemoryUtilizationManager.select` 按以下顺序尝试。最终 Formatter 的整项字符／条数预算及 Processor 的最终 Token 预算仍可令选择回退为 `NONE`。

| 优先级 | 模式 | 真实条件 |
| --- | --- | --- |
| 1 | `REVISIT_AUDIT` | Navigation State 有当前 `place_observation` 的 DINO 描述子；同 Scene 历史中至少两个独立观测时间有相似描述子；其中当前或历史同时间观测关联指令地标；Context 重新验证每个证据的时间、Scene、状态；地点身份仍为 `unknown`。 |
| 2 | `PROGRESS_AUDIT` | 有与指令相连且以独立位移测量支持的 `verified_traversal`，**或**原始指令含 `through/pass/past/enter/exit/cross/across` 之一且有相关历史视觉事实。后者只说“曾观察、通行未知”，不称已完成。 |
| 3 | `HISTORY_REFERENCE` | 有与指令相关、非当前帧、未被精确原生 RGB 覆盖的历史视觉事实，或来源完整且端点标签相关的不确定关系。普通已发出动作不会单独触发。 |
| 4 | `NONE` | 无上述可用事实，候选已经确认注入过、文本不合模式／预算、或 Processor 最终输入无法安全容纳。Policy 使用原始 sample 对象。 |

所以 `HISTORY_REFERENCE` 没有因“候选非空”自动触发：历史 Top-K 还可能由动作、视觉变化、无关物体、当前仍可见的 RGB 帧、重复 Evidence、无标签地点边组成。`PROGRESS_AUDIT` 也不会把多数普通历史动作或物体计数当作成功导航进度。`REVISIT_AUDIT` 确实依赖当前原生 RGB 本身不能给出的**历史比较证据**；当前图像的 DINO 向量只是重访候选输入，并不建立地点身份。

## 拒绝原因的逐项核对

| 原因 | 实际判定 | 本轮判断 |
| --- | --- | --- |
| `unrelated_history` | 对历史视觉原子事实，`_related` 从语义标签提词，并要求标签词集合是指令锚点的子集；无标签 `place_observation` 不转成描述性 Prompt。 | 存在部分词重合被拒的确定性例子，但“部分重合即有用”不成立；需真实候选标签及指令才能区分误拒。 |
| `unsupported_event_type` | Context 展开 V3 的单条 Evidence 或 Episodic Segment 后逐项分类。Segment 本身并未被整体排除；其中观察、ISSUED 动作分别存放。 | `place_hypothesis` 是确认的映射缺口，已修复。`visual_change`、`possible_transition` 和派生 `instruction_match` 不足以证明地点、通行或完成度，保留排除。 |
| `unrelated_relation` | 仅对精确匹配 Store 边 Evidence 的关系建立事实；至少一个端点有与指令相交的语义标签，否则丢弃。 | 无标签 DINO 地点连接不能充当指令相关地标，更不能被提升为验证可通行；暂不放宽。此诊断也覆盖边引用不一致，未来有逐条日志时可拆分原因。 |
| `visible_in_native_rgb` | 历史事实的每个来源必须同 Scene／Episode／rollout，且 Policy 记录的准确 Step→frame ID 在本次 `video_segments[*].frame_indices` 中。 | 精确身份才判重复；不使用 DINO 相似度、Step 距离或时间差代替。跨 Episode 来源没有对应当前帧映射。 |
| `duplicate_evidence` | 同一次 Context 展开时，相同 Evidence 可经原子记录、节点或 Segment 重复进入 Top-K；Context 和 Formatter 不重复序列化相同 ID。 | 属于同一步的去重，不等于跨步重复注入。 |
| `repeated_without_new_evidence` | 修复前比较未格式化的候选整组签名；修复后额外检查**实际通过最终 Token 验证**的 Evidence ID。全由旧 Evidence 组成的事实不再因候选集合变化而重注入；有新增支持的复合事实仍可携带旧来源。 | 可确定性验证；旧日志缺逐项内容，不能回算会减少多少注入。 |

`NavigationState` 的地标状态仅把已观察、词面相关的物体或地标标为 `OBSERVED`；`place_hypothesis` 仍不会更新为子目标已完成。`PROGRESS_AUDIT` 可以审计“曾观察但通行未知”，因此在总体状态为 `COMPLETION_UNKNOWN` 时触发并不矛盾。

## 开源函数到当前实现

本轮本地 `.references/3-step-Nav/` **不存在**，因此不能声称重新读取其四个指定本地文件。另通过公开源码重新读取了固定 commit 的 [spatialNavigator.py](https://github.com/ZoeyZheng0/ThreeStepNav/blob/5cdbdcff356cda301e23496c5d850c15b2433d87/vlnce_baselines/common/navigator/spatialNavigator.py) 与 [MIT LICENSE](https://github.com/ZoeyZheng0/ThreeStepNav/blob/5cdbdcff356cda301e23496c5d850c15b2433d87/LICENSE)；[prompts.py](https://github.com/ZoeyZheng0/ThreeStepNav/blob/main/vlnce_baselines/common/navigator/prompts.py)、[decision_agent.py](https://github.com/ZoeyZheng0/ThreeStepNav/blob/main/vlnce_baselines/common/navigator/decision_agent.py)、[navigation_capabilities.py](https://github.com/ZoeyZheng0/ThreeStepNav/blob/main/vlnce_baselines/common/navigator/navigation_capabilities.py) 本轮只读到公开 `main` 版本，固定 commit 的这三份文件无法独立核对。仓库既有的 [RGM V4 来源报告](RGM_V4.md) 是先前审查记录，不能替代本轮缺失的本地源码。本轮没有从任一参考项目复制代码。

| 原始函数 | 当前 V4 对应 | 仍缺的机制／迁移边界 | 冻结 LightNav 下 |
| --- | --- | --- | --- |
| Three-Step Nav `ACTION_DETECTION`、`LANDMARK_DETECTION` prompt（本轮仅公开 `main` 可读） | `NavigationStateUpdater` 的词项锚点 | 源码依赖额外 LLM 拆解指令与抽取地标；词项锚点不能冒充同等解析能力。 | 原指令和简单词项可保留；LLM 语义分解不可直接迁移。 |
| Three-Step Nav `save_history`、`review_history`（固定 commit 公开源码已读；本地缺失） | `MemoryWriter.observe`、V3 Store、`build_navigation_memory_context` | `save_history` 用 LLM 总结观察和 thought，再保存 Step、目标 viewpoint 等；其驱动路径可在 `env.step` 前保存，故目标动作不能视作已执行结果。`review_history` 串联这些条目。 | Episode／Step 历史组织可以；GPT 总结不引入。 |
| Three-Step Nav `estimate_completion`（固定 commit 公开源码已读；本地缺失） | `NavigationStateUpdater.update` 的进度与不确定性 | 该函数需要额外 LLM；固定版本的活动路径另从多模态响应解析 `Completion Estimation:`。 | 证据状态可实现；LLM 完成判断不可直接迁移。 |
| Three-Step Nav `NavigationDecision`、四类 navigation capabilities（本轮仅公开 `main` 可读；本地缺失） | `MemoryUtilizationManager.select` 的四种**记忆使用**模式 | `Continue`／`Stay`／`Backtrack`／`Look Around` 属于原项目的状态和动作接口；固定 commit 的三个相关文件本轮未独立核对。 | 只保留状态管理思想；不由记忆模块输出动作。 |
| MapGPT `make_history` | V3 事件及 `issued_commands` | 该函数按 Step 记录选择的动作标签；选择不证明实际位移。 | 可以保留 ISSUED 轨迹，不能提升为 VERIFIED 通行。 |
| MapGPT `make_map_prompt` | V3 节点／边、V4 `history_evidence` 与 `uncertain_relations` | 源码将 `trajectory` 中的地点与其他已观察、未在轨迹中的 ghost 节点分开。LightNav 缺其离散地点 ID 与已访问证明。 | 可以保留观察／访问状态区分；不能复制图节点动作。 |
| MapGPT `make_r2r_prompts` | 原始 Instruction 加选择性 V4 Context | 源码分区展示 Instruction、History、Trajectory、Map、Supplementary Info、Previous Planning；实现每步都写 Supplementary Info，虽说明文字说仅在出错回退时考虑。 | 可迁移信息职责分区；不应无条件加历史文本。 |
| MapGPT `parse_planning` | 无对应策略解析；LightNav 保持原动作解码 | 源码解析 GPT-4v `New Planning`／`Action` 文本，依赖另一个生成协议。 | 不添加额外 LM 或改变 LightNav 动作接口时不可忠实迁移。 |

本地 MapGPT revision 为 `7c642f4507a8dde6703e2c61c8fd8ff3bc4bd322`；该树未见 `LICENSE`／`COPYING`。这不是许可证授权结论，故只读源码、不复制其代码。Three-Step Nav 的 MIT 信息本轮经公开固定 commit 的许可证文件核对；本地目录仍然缺失。

## 验证与边界

修复前，既有 V4 Context／Utilization／State 和 V3 Store／Retriever 针对测试 **50 passed**；两个新增复现测试 **2 failed**：正式存储的 `place_hypothesis` 被丢弃，且旧利用管理器没有按实际注入证据确认的机制。另直接加载 `HEAD` 版本的利用管理器做确定性重放：候选从 `(A,B)` 变为 `(A,C)`、格式化限制一项时，两次 `HISTORY_REFERENCE` 都输出**完全相同的 A**。修复后，V4 Context／Utilization／Fake Processor／Fake vLLM 聚焦测试 **33 passed**；扩大到全部 V4 CPU 测试为 **69 passed、1 skipped、8 subtests passed**。其中包含真实 `MemoryWriter → V3 Store/Retriever → V4 Context/Formatter` 链路、旧 Evidence 在新增候选后不重注入、格式化只输出首项时不因未输出候选变化而重注入，以及最终输入中的来源 Ref。V3／Legacy 针对回归为 **78 passed、2 skipped、1 deselected**；被取消选择的是既有 Legacy 用例，其期望 `unresolved_competing_candidate`，未修改的实现返回 `ambiguous_global_candidates`。完整可运行的 CPU 回归为 **894 passed、14 skipped、1 deselected、76 subtests passed**，命令排除了已知阻塞的 `tests/test_batcher.py` 和 `tests/test_tracking_service.py`，并取消选择上述 Legacy 用例。未经排除的完整 CPU 命令在 90 秒截止前出现该 Legacy 失败并停在约 72% 进度；两个异步测试模块各自单独运行 15 秒仍超时，因此不能宣称全套无排除通过。修改文件的 Ruff 检查通过。仓库范围 Ruff 检查报告 51 项既有文件／未跟踪参考项目中的问题，本环境没有 `make` 命令。

测试验证的是代码契约：无有效记忆时保留 Baseline sample；有效记忆可进入最终 `input_ids`，假 vLLM 收到同一 `prompt_token_ids`；Legacy 与 V3 代码未修改。真实 checkpoint Processor、真实 vLLM／Habitat 效果、修复后 88 Episode 注入分布和 SR **均未在本轮验证**。旧工件若只含候选 ID 与拒绝理由而无每条 Evidence 的原始事件、标签、来源与当前视频帧清单，不能据此伪造离线重放。要逐 Episode 归因，需要旧运行对应的只读逐步诊断和可复核的候选内容；本轮未发现它们。
