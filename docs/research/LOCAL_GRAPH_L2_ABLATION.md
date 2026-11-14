# Local Graph Fusion 的 L2 消融：基线依赖审查

基线：`2fb184e`（`baseline-local-graph-fusion-88-sr75`）；实验分支：`experiment/local-graph-no-l2`。本节先记录修改前的真实依赖。AutoDL 上正常组已有 `output/local_graph_fusion_88_run2`；本机没有该目录，因此运行前须以 AutoDL 的 `config_snapshot.yaml` 核对所有非 L2 参数。本轮只准备新消融组，不重跑或覆盖正常组。

| 审查问题 | 源码依据与结论 |
| --- | --- |
| LocalGraphFusion 是否读取 EpisodeMemory？ | 否。`src/lightnav/memory/local_graph_fusion.py:122–208,244–302` 的 `build()` 只使用传入的 V3 Store Evidence、Place/Landmark/Edge 与当前 events；不调用 `episodes_for_scene()`、`active_segment()`、`EpisodicMemory.retrieve()`。 |
| MemoryWriter 是否依赖 EpisodicMemory.events？ | 是。`src/lightnav/memory/memory_writer.py:286–310` 将每个原始 `MemoryEvent` 写入 `EpisodicMemory.add()`，再写 L1 Working Memory 并返回；DINO 地点事件也在 `manager.py:623–636` 这样流转。`episodic_memory.py:104–118` 的 `_events` 同时是原始事件日志，不能删除。其 `_key_events`/`_segments` 才是可独立禁用的 L2 索引与片段。 |
| V3 Store `_journal` 的两种职责？ | `v3_store.py:158,193` 每条 Evidence 都把 key 写入 `_journal`，维持原始写入日志；`active_segment()`（`:220–253`）及 `close_episode()`（`:467–495`）才从它构造 L2 `EpisodeMemory`。消融应停用构造结果，但继续写 `_journal`、Evidence、Node/Edge，并保留关闭清理。 |
| `_episodes`、`active_segment()` 是否在 Local Graph 中使用？ | 否。它们进入 `v3_retrieval.py:313–334` 的 episodic 通道；`policy.py:423–448,593–594` 的 Local Graph 分支只调用 `manager.get_local_graph_context()`，绕过 V3Retriever 和 `retrieve_layers()`。 |
| L2 写入/关闭是否影响 L1/L3？ | 原始事件传递是共享依赖；L2 索引自身不是。`manager.py:281–303` 将 MemoryWriter 返回的同一 events 写入 L1 与 V3 Store；`manager.py:696–716` 从 Evidence 建立 Place/Landmark/Edge，不读取 L2 片段。V3 Store 的 Scene/Episode 身份检查与关闭清理须维持；`_episodes` 目前还承担已关闭 Episode ID 检查（`v3_store.py:153–155`），禁用后需独立身份集合。 |

**依赖结论：**正常组可以有 L2 索引和 EpisodeMemory，但 Local Graph 的候选、Prompt、最终 Token 原本就不读取它们。合成 A/B 比较证实模型输入一致；因此当前架构中 L2 对该路径的模型输入为零影响，不能为制造性能差异改动 L1、L3 或融合算法。

## 实施边界与诊断定义

新增 `--local_graph_mid_enabled true|false`，默认 `true`。正常模式保持原 `local_graph_v1` 协议和 LocalGraphFusion 行为。`false` 只做两件事：

1. `EpisodicMemory(index_enabled=False)` 保留经 `MemoryWriter` 验证、去重的 `_events` 原始日志，但不建立 `_key_events`、时间片段，也不返回 mid 检索结果。
2. `V3MemoryStore(mid_enabled=False)` 继续写 Evidence、`_journal`、短期缓存、Place/Landmark/Edge；不构造活动 `EpisodeMemory` 片段，也不在关闭时生成 `_episodes`。独立 `_closed_episode_ids` 维持同 Scene 重复 Episode 身份检查。关闭时照常清理 journal/short，并按原 scope 决定是否保留 Scene 数据。

未改动 `memory_writer.py`、`episodic_memory.py`、`local_graph_fusion.py`、L1/L3 算法、视觉检测、Prompt 融合、模型或动作接口。消融协议为 `local_graph_no_l2_v1`，每个 shard 标记显式记录 `mid_enabled=false`；旧 `local_graph_v1` 标记格式仍可只读校验。

每步 Local Graph 诊断增加：

| 字段 | 含义 |
| --- | --- |
| `l2_enabled` | 真正的 L2 开关；即使当前没有候选也会如实显示 `true` |
| `mid_memory_records` | V3 Store 中已组织的 EpisodeMemory 数量加当前活动片段；禁用时由真实空组织得到 0 |
| `mid_candidates` | 只用于诊断的 EpisodicMemory 索引查询数量；**不传入** LocalGraphFusion |
| `mid_prompt_records` | 最终选中卡片中 layer=`mid` 的条数；当前 Local Graph 两组都应为 0 |
| `short_memory_records`、`long_memory_nodes`、`long_memory_edges` | 原 V3 Store 的短期 Evidence 缓存和拓扑规模，两组应一致 |
| `actual_injection`、`actual_historical_injection` | 原有最终 Token 审计结果，两组应一致 |

## 合成 A/B 实测结果

本地可运行的 V3 Store / LocalGraphFusion 纯 CPU 对照，给两组写入**同一** couch 观测、地点、动作和图边：A 建立活动片段与关闭后的 `EpisodeMemory`，B 的这两项和 episodic 检索均为空；两组 `_journal`、Evidence、Node、Edge、Local Graph 候选卡片和注入后的 Prompt 相同。原始 `EpisodicMemory` 日志在 B 中仍保留。该结果符合预期的模型输入零影响机制。

同一三步合成 Observation/Action 序列通过 `MemoryManager → LocalGraphFusion → Prompt → Policy` 的 A/B 对照。A 的 L2 索引、片段、检索候选和 V3 `EpisodeMemory` 实际存在；B 的这些组织和检索实际为空，两组原始事件日志、L1 Working Memory、V3 Evidence/Node/Edge、Local Graph 候选和 Prompt 逐步相同。fake vLLM 路径进一步验证最终 Token ID/hash、视频输入、解码动作和注入诊断逐步相同。定向测试结果：`20 passed, 8 subtests passed`；启动器修订后复测 `11 passed, 10 subtests passed`。这证明该合成路径中的预期零影响，不等于已完成 GPU 88 Episode 复现。

其余 CPU 套件在排除基线已有的 3 项断言失败、因 `test_respects_max_batch_size` 超时而暂排除整个 `test_batcher.py`（11 项），并排除 `test_tracking_service.py` 的空缓冲等待测试后，结果为 `957 passed, 14 skipped, 4 deselected, 86 subtests passed`。排除项均不在本轮修改的 L2、Local Graph 或启动器路径：旧 `MemoryStub` 缺少 `local_graph_config`；step trace 错误文本和 RGM 地点验证 reason 的测试期望与 `2fb184e` 实现不一致；两组异步服务测试在本机等待不结束。未调整这些无关实现或测试。

## 独立 88 Episode 命令（本轮不运行）

沿用已完成正常组的 checkpoint、R2R `val_unseen` 数据、88 条 Pilot/共享 Episode manifest、Habitat YAML、YOLO/DINO 资产、种子、每步上限、GPU 和推理参数。`PILOT_MANIFEST_PATH`、`EPISODE_MANIFEST_PATH` 必须指向正常组实际使用的文件；以下路径是占位符。此命令不重跑正常组：

```bash
export MODEL_PATH=/path/to/LightNav-0
export ACTION_TOKENIZER_BUNDLE=/path/to/LightNav-0/action_tokenizer
export HABITAT_CONFIG=habitat_server/configs/vlnce_r2r.yaml
export DATA_PATH=/path/to/R2R_VLNCE_v1-3_preprocessed
export SCENES_DIR=/path/to/scene_datasets
export YOLO_MODEL_PATH=/path/to/yolo11n.pt
export DINO_MODEL_PATH=/path/to/dinov2-small
export PILOT_MANIFEST_PATH=/path/to/rgm_pilot_88_v1/pilot_manifest.json
export EPISODE_MANIFEST_PATH=/path/to/rgm_pilot_88_v1/episode_manifest.json
export OUTPUT_ROOT=output/local_graph_no_l2_88
export GPU_IDS="0 1" GPU_MEM_UTIL=0.65

export TASK=vlnce SPLIT=val_unseen BACKEND=vllm_local
export STRICT_FULL_EVAL=1 EXPECTED_EPISODES=88 EPISODES=-1
export MAX_STEPS=500 NO_FORCE_STOP=1 SEED=42 RESUME=0
export SERVER_ARGS="" LANGUAGES=""
unset VLN_EVAL_TEMPERATURE VLN_EVAL_TOP_P VLN_EVAL_TOP_K VLN_EVAL_TRAJ_TOP1
export CLIENT_ARGS="--memory_enabled --memory_strategy hierarchical --memory_layers short,mid,long --memory_extractor yolo11 --memory_extractor_model_path $YOLO_MODEL_PATH --memory_extractor_device cpu --memory_extractor_confidence 0.25 --memory_extract_every_n_steps 5 --memory_place_enabled --memory_place_model_path $DINO_MODEL_PATH --memory_place_device cpu --memory_place_extract_every_n_steps 5 --memory_place_similarity_threshold 0.85 --local_graph_enabled --local_graph_scope episode_local --local_graph_prompt_max_chars 320 --local_graph_max_landmarks 2 --local_graph_reactivation_steps 8 --local_graph_mid_enabled false"

test ! -e "$OUTPUT_ROOT" && bash scripts/eval_habitat.sh
```

运行前先在该机器执行 CPU 集成测试。运行后核对根目录 `config_snapshot.yaml` 的 `local_graph_effective.mid_enabled=false`、`protocol_version=local_graph_no_l2_v1` 与每个 shard 的 `local_graph_protocol.json`；用 `scripts/verify_local_graph_shards.py` 校验配置，再按原 88 Pilot 完整性检查验证 88 个 Scene/Episode pair。比较正常组与消融组时，模型、manifest、环境、YOLO/DINO、L1/L3、Prompt 预算和推理参数必须一致；代码哈希、输出路径、L2 开关和消融协议版本会不同。
