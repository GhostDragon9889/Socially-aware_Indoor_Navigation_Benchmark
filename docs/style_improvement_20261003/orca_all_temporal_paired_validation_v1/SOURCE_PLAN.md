# Formal EMA40k 与旧63e：固定validation配对 open-loop 评估方案

准备阶段只读源码、JSON、NumPy validation states/window indices；不读权重，不运行NN/SDK/GPU，不读sealed34，不写未来输出或终态结论。新40k仍训练中；只有真正closed owner与77025终态reader通过后才执行模型评估。

## 固定来源与模型身份

- 实际formal plan `../orca_official_all_temporal_formal_preparation_v1/commands_and_audit_plan.json` SHA `2beed1bd10777658eb1c6ec8e1a2aa3bbc7c8571646ddf8f5fa9f3fbec9ab945`。
- 原NN/conditioning来自 `orca_interact_usable_staging_v1/paper_interact.py`，固定loader来自 `orca_all_temporal_view_staging_v3/all_temporal_corpus.py`（cb289037），metric来自 `orca_official_all_temporal_formal_preparation_v1/metrics_all_temporal.py`（340ce40c）。不用重写loss、DDPM、坐标/几何或空peer数学。
- OLD：原sealed `output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt`，279052920B，SHA `63e7cad07a479b2a3cf2f776a4f2e7d141a90a1e3cc568c8446d5e967a85adac`；使用其`model`。
- NEW：已登记 `output/diffusion_amp/official_ORCA_all_temporal_EMA40000_micro200_accum2_v1/step_40000.ckpt`；当前bytes/SHA=null。未来由 `actual_terminal40000_audit.json` 的真实readonlycheckpoint record提供，使用其`model=EMA`。不比较raw而冒称deployment。
- 未来准入复用原77025 auditor的schema/status、exact reader/plan/owner/checkpoint records、full raw/EMA/Adam strictload与production EMA逐tensor flags；不重新审计大完整状态或建新launch/ledger框架。

## 冻结选择（不按模型效果挑选）

使用实际新两domain manifest `orca_{maps,interact}_all_temporal_1000_v2/manifest.json`，**只validation**，不train/test。

1. 主population各域100个validation episodes，在manifest/episode既有顺序中每episode统一随机选1个已存canonical合法window。固定PCG64 seed Maps=`20261301`、Interact=`20261302`；这些选择seed在模型读取前登记。主指标各域N=100，另可给equal-domain N=200描述；不能声称full-window分布或作者benchmark。
2. 独立诊断population为全部 **504个当前15m内零条件邻居** validation windows（Maps355、Interact149），保持canonical原顺序，不重新采样。包含单人和多人但当前远离其他人的窗口。metric仍看全部记录actor的未来，不把零condition误当无其他actor。
3. 每个row保存domain/local corpus index、scene/episode/agent/frame、source artifact identity、current条件邻居数量、实际其他actor数量、是否与主population重合。诊断和主population分开报告，**不相加到704个独立主样本**；重合关系明确列出。

准备只生成独立 `selection.json`（及小registration），不生成predictions。canonical anchor边界从已核history31/future52/frames100合同推导；不优化选窗或排除静止、collision、single、empty-map场景。

## 原数学与公平随机性

薄 `evaluate_paired.py` 调原 `PaperDiffusionTrainBatch.from_world`、`training_loss(training=False,drop_conditioning=False, explicit timesteps/noise)`、`PaperDiffusionPlanner.plan(num_samples=1, seed=...)` 和metric340。两模型严格相同完整config/normalization buffers（实际加载后检查），52帧clean loss、100-step DDPM；open-loop XY collision/ADE/FDE原50未来点。没有waypoint/goal/collision guidance、候选筛选、动态额外输入或改defaults。

- 固定batch8，固定domain/stratum顺序及尾batch；同batch两模型共享原numpy inputs、同CPU生成的loss timestep/noise。
- loss seed base=`20261101`，plan seed base=`20261401`；domain/stratum/batch start的确定offset在registration中明确。两个模型同case/start用相同seed与shape，原plan局部generator产生相同初始/每步噪声；不声称不同硬件bitwise。
- map encoder策略精确复用 `encoding_map_mask(...,'trace_unknown_raster')`：Interact数据presence仍False/全.5 raster，encoder policy mask另为True；Maps真实map保持。实际metadata须同时保存两种mask，避免把缺图冒充实图。
- constructor/evaluation在原fork_rng设备域内或分别显式局部generator，保存原corpus私有RNG。explicit numpy window indices不消费union RNG；模型之间不靠global随机序列公平。

## 执行后保存与聚合

每model/domain/stratum/batch独立保存indices、输入array hashes、loss timesteps/noise、prediction `xy/yaw/velocity/sample_states/selected_sample`、原metrics与真正peer observation counts。任何非finite/异常保留首日志/已存outputs，不能删failed/collision/single/idles来继续。

主/诊断分别以**各自全部windows**作ADE/FDE/loss/collision加权分母，保留单人collision0/known-minNone和InteractmapN/A。min对已存在finite值取min、空scope None及原knownreason；实际peer observation counts求和。保留perwindow原metric calls的loss-free标量对照，使配对差异能审阅；不声称open-loop碰撞等于closed-loop物理。

最少实现：一个薄entry（prepare-selection和actual-evaluate两模式）+selection/registration两个JSON，直接复用上述source函数。Root未来直接运行original Python argv及实际toolreceipt；不加新owner/importframework，不修改任何原源/数据/模型。实际futurecheckpoint/report缺失时evaluate失败关闭，prepare-selection仍不读取它们。
