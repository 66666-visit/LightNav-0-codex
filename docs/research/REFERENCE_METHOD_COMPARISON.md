# 三项 VLN 记忆工作的源码对照

本轮只读取 `.references/` 中的源码；三个仓库不进入主项目依赖，也没有复制其实现。检查前主项目为 `codex-adaptive-memory`、HEAD `54b2d9bd1c1b6deb2462789e451748a7a82a5a22`，`.references/` 是原有的未跟踪目录。

## 论文机制与检入源码

| 项目 | 源码中可确认的做法 | 对冻结 LightNav 的可迁移部分 | 不能直接迁移的部分 |
| --- | --- | --- | --- |
| HSGM | [mapper.py](../../.references/HSGM_public/src/mapper.py) 的逐层 scene、可通行、障碍、轨迹和物体实体来自 RGB-D/位姿；[YOLOE 分割](../../.references/HSGM_public/src/segmentation/instance_segmentation.py) 的 mask 与深度转成 3D 物体，跨帧关联后绘制到智能体中心俯视图。[agent.py](../../.references/HSGM_public/src/agent/agent.py) 将地图图像和多视角 RGB 发给 VLM。 | 几何、语义和历史轨迹分层；只向当前决策展示局部证据。 | 需要 RGB-D、准确位姿、额外 GPT 地图阅读和 waypoint/A* 控制。其源码没有可直接复制的显式“地点—物体—关系”语义图；`get_objnav_affordance_map` 并非主决策路径的调用点。 |
| OVER-NAV | [论文](https://arxiv.org/html/2403.17334v1) 描述跨轮地点、关键词与路径边的 Omnigraph。检入的连续环境配置启用 [KeywordsMappingModule6](../../.references/OVER-NAV-IVLN-CE/ivlnce_baselines/common/obs_transforms.py)：已访问 viewpoint 内存、7 m 局部筛选、每点 Top-3 标签、跨点同名去重，输出关键词、相对方位和距离（[keywords_mapper6.py](../../.references/OVER-NAV-IVLN-CE/ivlnce_baselines/common/mapping_module/keywords_mapper6.py)）。 | **主方案**：根据当前位置提取局部 ego 图，只把与当前指令有关的少量地标及其可确认的视角关系组织成导航上下文。 | [map_cma_policy.py](../../.references/OVER-NAV-IVLN-CE/ivlnce_baselines/models/map_cma_policy.py) 用新增编码器、注意力和融合层学习使用图特征；冻结 LightNav 没有这些层或候选 viewpoint 动作。 |
| HiMemVLN | [论文](https://arxiv.org/html/2603.14807) 描述 Localer 的短期视觉图、Globaler 的全局导航 schema、CameFrom 和最近 5 步。检入的 [base_il_trainer_vlm.py](../../.references/HiMemVLN/vlnce_baselines/common/base_il_trainer_vlm.py) 可确认多视角 RGB-D、历史审阅、完成度估计、候选 waypoint 选择与执行的调用次序。 | 可借“当前状态决定历史何时再现”的思想，保留短期/长期职责区别。 | 本地 checkout 缺少被 trainer 导入的 `spatialNavigator_vlm`/`spatialNavigator`，因此 Localer、Globaler、VGM、CameFrom 的**内部代码无法核对**；其 Qwen2-VL 多视角候选动作接口也不同。 |

HSGM 的物体来自 [instance_segmentation.py:10–52](../../.references/HSGM_public/src/segmentation/instance_segmentation.py) 与 [geometry.py:20–38](../../.references/HSGM_public/src/mapping_utils/geometry.py)；`mapper.py:73–102,1582–1733,1796–1923` 建楼层/几何/物体层，`agent.py:91–140,190–270,1531–1553,2439–2469` 画并提交地图。地点更多是候选 waypoint 或子任务坐标，物体间关系留在地图像素和坐标中。仓库含 Apache-2.0 `LICENSE`。

OVER-NAV 的源码细节不能简化为“代码按指令关键词过滤”：`keywords_mapper6.py:168–184` 的函数接收 `keywords` 但没有使用，调用处 `obs_transforms.py:262–264` 传 `None`；指令关联由 `map_cma_policy.py:367–425` 的学习式注意力完成。检入评测 YAML 启用 `KEYWORDS_MAP`，未启用 `DISTANCE_FEAT`；这是该配置的观察，不能推断论文全部实验忽略距离。`keywords_mapper3.py:130–239` 另有动态 viewpoint/按指令检测路径，不能与评测实际选择的 mapper6 混为一谈。仓库含 MIT `LICENSE`。

OVER-NAV 的论文把多轮导航留下的地点、关键词、连接关系称为 Omnigraph；可运行 mapper6 的具体存储则是 `scan/viewpoint → 已见指令 ID、检测标签/分数、36 视角索引`，并从检测数据库加载过往轮次的记录（`keywords_mapper6.py:17–57,85–161`）。其局部子图不是整个 Scene：仅保留可见且与当前位姿水平距离小于 7 m 的 viewpoint，每点最多 3 个标签，再按标签取最高分去重（`168–195,237–332`）。当前最近 viewpoint 的 `viewIndex` 按每 30° 映射为相对朝向，距离设为 0；邻居 viewpoint 则用其已知世界坐标算相对方位和距离（`217–330`）。这些度量值依赖该工作预置的 viewpoint 坐标与机器人位姿，不能直接转给当前仅有 RGB/DINO 假设的 LightNav；本方案只使用当前 YOLO 框的图像左右区域，历史方位和距离明确标未知。Omnigraph 融合在 `map_cma_policy.py:367–425` 把关键词 embedding 与方位/可选距离特征经压缩、卷积和指令注意力接入策略状态，因此没有现成的冻结 LightNav 兼容特征接口。

HiMemVLN 检入仓库仅可核对 `base_il_trainer_vlm.py:168–264,403–405,426–535,573–576` 的外层调度。论文给出的 512 维视觉节点、余弦重访阈值、动量更新、`{PrimaryDir, FinalTarget, NavPattern}`、CameFrom 和最近行动文本均属于**论文陈述**，本地缺失的导航器源码不能独立证实。该 checkout 没有 `LICENSE`，所以未复制代码。

## LightNav 输入和动作边界

1. [prompts.py](../../src/lightnav/prompts.py) 保留原训练用 `navigation_task`、`<video>` 和输出句式；RVQ checkpoint 只由 `to_rvq_prompt` 替换输出句。不可改写成参考项目的候选 viewpoint Prompt。
2. [inference/policies.py](../../src/lightnav/inference/policies.py) 的 `NavigationPolicy.observe` 给 RGB 分配绝对 frame ID；普通模式保留窗口，SlowFast 保留整个 Episode。[samples.py](../../src/lightnav/inference/samples.py) 将帧排成单段、历史/当前双段或 SlowFast 多段。后者的稀疏帧由 [slowfast.py](../../src/lightnav/slowfast.py) 选择。
3. [data_processor.py](../../src/lightnav/data_processor.py) 对 SlowFast 使用绝对帧序号产生真实时间戳；普通窗口仍使用训练时的窗口相对时间戳。新记忆不能把旧视频帧当作任意文本替换。
4. [engine.py](../../src/lightnav/inference/engine.py) 的 `generate_from_frames(..., task_type="vlnce_traj")` 生成原 VLN sample。[habitat/policy.py](../../src/lightnav/habitat/policy.py) 调用模型后，[tracking.py](../../src/lightnav/tracking.py) 解析 `<act_l*>`、RVQ 解码成 `[forward, lateral, yaw]`，再映射速度。[serving/protocol.py](../../src/lightnav/serving/protocol.py) 另从原输出解析 Pointing Token；新模块不碰它们。
5. FIX3 的 [manager.py](../../src/lightnav/memory/manager.py)、[v4_context.py](../../src/lightnav/memory/v4_context.py)、[v4_utilization.py](../../src/lightnav/memory/v4_utilization.py)、[v4_formatter.py](../../src/lightnav/memory/v4_formatter.py) 经过 V3 四通道检索、事实重构、模式筛选和逐条来源文本，再插入原 Prompt。`visible_in_native_rgb`、严格语义标签包含和已注入证据抑制会减少注入；旧 88 Episode 汇总不能确定哪条规则导致 Episode 1 失败。

冻结策略只接受原 RGB 视频和文本 Prompt，输出原 Pointing/RVQ Token。局部图因此只能在 Prompt 的既有安全插入点转换成**观察上下文**，不能假装有额外训练过的图特征入口、真实位移测量或离散图节点动作。
