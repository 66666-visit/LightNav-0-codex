# 给下一位 ChatGPT 的 RGM-LightNav 项目上下文

请先读 [完整交接](AGENT_PROJECT_HANDOFF.md) 和 [架构图](AGENT_ARCHITECTURE.md)。本文以 2026-11-10 的 **本地未提交工作树** 为准：分支 `codex-adaptive-memory`，HEAD `0b960a8` 并不包含全部 RGM V1。用户的研究项目是 **RGM-LightNav Agent**；LightNav 是冻结的基础导航模型和动作生成路径，不要把其视频编码、SlowFast、Qwen3-VL、RVQ/轨迹 token、路点/速度解码算作 RGM 创新。

## 一句话机制

每个 Habitat 步，Policy 从 RGB、指令、上一轮命令/模型文本建立 `ObservationPacket`；MemoryManager 在固定步频运行 YOLO、DINO，写入 Episode 局部三层记忆；Write → Retrieval → Prompt Gate 只允许可溯源且指令相关的视觉类别进入原 LightNav Prompt；若无有效记忆，Engine 收到原基础 sample；LightNav 仍预测路点并映射成 Habitat 动作。动作后下一帧触发记忆更新。**没有 Pose、碰撞、实际位移输入；拓扑边未验证可通行。**

## 代码入口与真实边界

- `src/lightnav/habitat/runner.py:687,883-946`：Manifest、Episode 循环、调用 Policy、执行 Habitat 动作；`info` 中 evaluator 指标仅用于 `:969-1024` 报告。
- `src/lightnav/habitat/policy.py:170-185,216-272,406-473`：重置；RGB → `TrackingAgent`/`ObservationPacket` → Manager → 基础 VLN sample/条件化注入 → Engine → 原动作解码。Packet 构造明确不含 `info`、Depth、`goal_distance`。
- `src/lightnav/memory/manager.py:122-245,339-415`：Episode 隔离、固定步频提取、三层检索与 Gate；D 的地点不确定时返回 UNKNOWN，不更新拓扑。
- `src/lightnav/memory/schemas.py:187-296`：`ObservationPacket`、`RGMEvidence`、`VisualPlaceFeatures`、`PlaceDecision`、`MemorySelection`。不存在 `RGMObservationEvidence` 类。
- `src/lightnav/memory/yolo_scene_extractor.py:81-165`：YOLO 类别/分数/框，每 5 步示例；`object_association.py:21-57` 是类别跨步共识，**没有实例级跟踪**。
- `src/lightnav/memory/place_recognition.py:156-231` 和 `place_verification.py:74-162`：DINO 一次 forward 得 384 维全局描述子和处理后网格的 Patch；D 用全局 TopK + 局部互匹配/RANSAC 作 MATCH/NEW/UNKNOWN。Patch 坐标未映射回原 RGB/YOLO ROI。
- `src/lightnav/memory/memory_write_gate.py`、`memory_retrieval_gate.py`、`memory_prompt_gate.py`：三道证据准入。C 即使关闭 YOLO 时序共识，当前指令词匹配且高分的物体仍能写入/注入。检测分数不是校准概率。
- `src/lightnav/memory/topological_memory.py:188-335`：视觉地点与 `observed_transition`；命令和视觉变化不是实际运动。`manager.py:401` 明示 `verified_traversal=False`。
- `src/lightnav/memory/memory_context_builder.py:52-140`：在 `</navigation_task>` 后条件注入，审计原模板可逐字恢复和视频对象不变。空记忆 Policy 直接传基础 sample。
- `src/lightnav/memory/rgm_config.py:38-40`：`adaptive_extraction_enabled=True` 被拒绝；**Adaptive Controller 仍是目标架构**，当前 YOLO/DINO 都固定步频。

## 当前实验协议，切勿沿用旧组名

最新正式 Pilot 为 **88 Episode、A/C/D、B 取消**。A=`baseline_off`；C=`rgm_gates`，三个 Gate 开，YOLO 时序/DINO 两阶段/拓扑运动验证关；D=`rgm_full`，在 C 上打开这三项。旧文档里 C=`hierarchical_topology`、D=`rgm_adaptive` 已过时。`scripts/run_memory_ablation.sh --rgm-88` 会直接转给 `scripts/run_rgm_pilot_88.sh`。`scripts/sample_r2r_pilot.py:297-380` 从旧 220 Manifest 选 11 场景×8 条；`scripts/reuse_legacy_a.py:32-137` 只在旧 A 的 ID、配置和源码/权重/动作指纹严格兼容时复用；`scripts/compare_memory_ablation.py:939-1125` 检查配对覆盖和 C/D 配置，只比较可比的延迟。

目前 **没有本地真实 R2R 结果或权重**，AutoDL 的目录、commit、依赖版本和实验状态未知。不能声称 SR/SPL 提升、降低视觉调用成本或通过真实 Habitat。当前本地 Python 3.10.12 缺 NumPy、Pillow、torch、pytest、OpenCV、vLLM、Habitat-Sim。本次只读单测 15/15 通过（RGM Gate 10、88 采样 2、比较器 3），另有 10 项因缺依赖跳过；Fake Engine Policy、实际 YOLO/DINO/GPU 与 Habitat 运行未在本机重跑。Git 远端配置：上游 `lightorigins/LightNav-0`，个人 `66666-visit/LightNav-0-codex`；工作树有大量已有未提交改动。不要把本地状态推断为 GitHub/AutoDL 已部署。

## 研究判断与下一步

V1 的真实贡献是 **可审计的视觉证据门控、分层 Episode 记忆、两阶段地点弃权、条件化 Prompt 与精确回退**；其收益仍是待验证假设。最薄弱处是缺实际位移/碰撞与可标定运动反馈，归一化速度命令 `>0` 不等价于真正前移；其次是没有实例追踪或真实环境错误记忆率。优先：①建立无 oracle 的物理反馈和通行证据；②在获批后跑相同 88 Episode 的真实 A/C/D 并核查错误记忆/成本；③在固定步频基线后单独实现和消融真正自适应调度。不要改原模型、SlowFast、RVQ 或动作解码来制造混杂。

**本轮用户约束：** 只生成这三份交接文档；不改源代码/配置/测试/权重/结果，不训练、不启动真实评测，不 commit/push/pull，不操作 AutoDL。后续任何开发先确认用户新的授权与具体目标。
