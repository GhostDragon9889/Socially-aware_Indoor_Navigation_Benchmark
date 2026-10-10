# 官方 ORCA 全时间锚点视图：独立正式训练准备

2026-10-03。**实际 CPU 数据构建/QC 与必要空邻居指标验证已经通过；新 GPU pilot、resume 和正式训练尚未执行。** 最终来源与命令登记见 [source_registration.json](source_registration.json) 和 [commands_and_audit_plan.json](commands_and_audit_plan.json)。未来 checkpoint 身份仍为 null，由 root 实际运行后的独立记录绑定，封存计划不再回填。

## 数据选择与真实计数

固定 TRACE commit `594b2af0824c5853e1a1884e167cd5ed097dd81c` 的 agent-centric 索引按 ego 类型及历史/未来时长建立；15 m 决定条件邻居，零邻居仍可取样。[索引](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/dataset.py#L392)、[条件邻居](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/trajdata/src/trajdata/data_structures/batch_element.py#L164)、[padding](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/utils/trajdata_utils.py#L135) 和 [空条件池化](https://github.com/nv-tlabs/trace/blob/594b2af0824c5853e1a1884e167cd5ed097dd81c/tbsim/models/trace_helpers.py#L160) 是直接源码依据；8 个固定 Git blob 的独立核查见 [来源报告](independent_TRACE_source_review.json)。

新视图复用原封存 100 点 states、地图、split、episode 顺序；每个 agent 使用包含当前点的 31 帧历史、随后 52 帧未来，锚点 30…47。独立 int32 NPY 存窗口索引，允许 0…19 条真实邻居；不按几何或速度过滤，不强制 Maps/Interact 50:50。

| split | Maps | Interact | union | 相对原邻居子集新增 |
|---|---:|---:|---:|---:|
| train | 81,396 | 151,650 | 233,046 | 4,458 |
| validation | 10,566 | 19,386 | 29,952 | 504 |
| test | 9,558 | 19,908 | 29,466 | 645 |

训练新增 4,458 / 233,046 = **1.912927%**。此前 21.84% 混用了全分割人数与仅训练窗口数，已纠正。自然零邻居窗口与已有窗口的邻居 dropout 是不同的数据选择。

实际数据来源：[v3 源码封存](../orca_all_temporal_view_staging_v3/source_freeze_v3.json)、[全构建结果](../orca_all_temporal_view_staging_v3/actual_build_QC.json)、[实际 QC/三 split identity](../orca_all_temporal_view_staging_v3/actual_QC.json)、[真实 CPU tool 闭合](../orca_all_temporal_view_staging_v3/actual_CPU_closures.json)。实际 build `61706→5f10ba`、QC `64635→ee6c13` 均 exit0；2000 个新 NPY 与两个新 manifest 已封存。正式入口只使用 `orca_all_temporal_view_staging_v3` 和两域 `*_all_temporal_1000_v2/manifest.json`。

两次失败保留：v1 导入 class 名多一个 C，predata exit1；v2 错把外层 `validation` 与原 manifest 的 `val` 直接比较，800 个 Maps train 索引完成后 exit1。v3 明确 canonical split 映射并使用独立新输出。原静态 compile peer 不代表运行成功，原数据、失败源码及部分输出均未覆盖。

## 薄源码与指标补齐

[新 main](train_formal_all_temporal.py) 从 c344 正式主程序派生；[8 处声明 diff](formal_source_derivation.diff) 和 [完整逆向 bytes/AST](formal_source_derivation.json) 记录来源。变化限定 helpers/view/data/output 接线、指标入口和 final sampling 元数据。**训练 update、五状态 NN、DDPM/loss、Adam、累积/clip、whole-state EMA 和 RNG 保存恢复未改。** 原空邻居 decoder 已支持全 false mask；新完整 view identity 使原 exact resume 拒绝旧 filtered 数据模型。

[metrics_all_temporal.py](metrics_all_temporal.py) 处理真实单 actor 的空 peer 集合：min 为 null 并保存明确 reason、实际 peer 观测数。它按同 episode 的全部 recorded peers 计算，0 条当前 15 m conditioning 邻居不意味着没有其他 actor；多 actor 原公式及全部 B×50 ego 点碰撞分母保留，single/失败案例不丢弃。Maps 已知空地图碰撞率为 0；Interact 障碍物指标为 N/A。

[独立源码 peer](../orca_all_temporal_metrics_independent_review_v1/peer_review.json) Required0。必要 [实际纯 NumPy 验证](actual_empty_peer_metrics_CPU.json) / [closed receipt](actual_empty_peer_metrics_CPU_closedreceipt.json) 在 tool `84139a` 实际 exit0，6 个 Maps/Interact single/multi/mixed **合成数组**案例通过：多 actor 原 existing keys 完全一致、single 空 reduce 边界、0 conditioning 但真实 peer 碰撞、mixed 全 B×50 分母、`allow_nan=False` 均覆盖。没有读取训练数组、执行 NN/Torch/SDK/GPU；不是完整语料行为或模型效果证明。不再追加 CPU 验证。

[终态 reader](audit_terminal.py) 仅从 167c reader 适配四处 metadata/I/O，见 [diff](terminal_reader_derivation.diff)。实际生产 `CrowdSim.humanoid.diffusion.PaperDiffusionPlanner` 将 strictload `model=EMA` 并逐 tensor 对照，随后 raw/Adam strictload 和保存 RNG 结构检查。CPU reader 不执行 forward/optimizerstep、不移除 CUDA RNG、不调用 `restore_rng`，不称完整 CUDA 恢复。

## 已登记的 GPU 顺序

仅 root 串行执行，三个独立目录已登记：

1. 新 view **4 optimizer updates**，micro200 × accumulation2，effective400，eval/save interval4，eval_batches1。启用原 raw+EMA final sampling，每域16窗，实际覆盖新指标/summary 分支；不代表完整100场景质量评估。
2. 从新 pilot 封存 step4 **加 1 update**，验证 full raw/EMA/Adam/union/global/CUDA RNG restore；此短 resume 保留 skip-sampling，不做第二次 GPU 下一步 bitwise 比较，也不重复 batch400 OOM。
3. 正式 **fresh 原 4b6 step1000** 初始化，重建 Adam/RNG，40,000 updates / 80,000 microsteps / 16 M windows，LR2e-4、no clipping、EMA `.995/every1/start2000`，eval/save interval1000、eval_batches4，最终每域16窗 raw+EMA sampling。

正式训练不继承 pilot 权重。BN 实际 microbatch200，区别于作者 batch400；effective400 不消除该差异。本地五状态/52 帧实现保留，不能宣称整个 NN、语料或结果等同作者。

原 [filtered 硬件决定](../orca_official_formal_actual_runs_v1/formal40000_actual_root_decision.json) 904c、旧 pilot/resume、旧计划/源/数据保持冻结。既有 GPU 峰值仅作为当前形状的历史硬件依据；root 将实际新4/新1关闭后另写 owner/decision 绑定本 immutable plan，不构成计划自循环。正式40k及物理效果尚无实测结论。
