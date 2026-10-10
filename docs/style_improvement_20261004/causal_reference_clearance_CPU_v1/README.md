# raw → causal reference：三组已封计划的真实 CPU 净空诊断

2026-10-04。实际工具 session **99952**，launch chunk `423bb3`，completion `57f84a`，观察 exit0；脚本内 CPU 耗时 **1.15343 s**。只读取三个已封存 `reference_plans.npz` 和小 JSON，没有读取 physical/ground arrays、权重、模型、SDK、CUDA 或 sealed34，没有改动原数据或生产源码。[完整结果](actual_diagnostic.json)、[真实工具记录](actual_CPU_tool_receipt.json)、[脚本](compare_clearance.py)。

## 结论

**raw-safe → reference-unsafe 确实存在，但没有证据支持“适配器整体恶化净空”或“适配器造成70/279次物理接触”。** 全未来预测与实际消费前缀的结果不同，不能把完整5秒预测全部当成执行过的轨迹。

下表是每次 replan 到下一次 replan 之间真正消费的 **saved target reference 前缀**，仍是目标轨迹代理，不是身体位置或碰撞测量。每组完整6000 controls × 6无序行人对 = **36,000 pair-phases**，完整6000 × 4人 = **24,000 actor-phases**。

| 组别 | peer unsafe phases：raw → ref | 同相位 raw-safe → ref-unsafe | 整个前缀 raw-safe → ref-unsafe pair-items | static unsafe phases：raw → ref |
| --- | ---: | ---: | ---: | ---: |
| OLD63e S1 | 239 → 128 | 31 | 3 | 0 → 0 |
| NEWb563 S1 | 676 → 625 | 147 | 11 | 10 → 5 |
| NEWb563 S10 | 864 → 576 | 30 | 0 | 61 → 76 |

适配在某些时刻新增违例，也在另一些时刻消除违例；三组 prefix peer 违例总数均减少。尤其 S10 的 ref prefix unsafe phases **576 < S1 的625**，实际物理接触却 **279 > 70**，说明该 root-disk 代理不能直接替代 physical contact 或解释其差异。OLD 的 prefix ref 128也同时具有实际peer0，进一步体现判定面与真实碰撞体不同。

## 完整 0..5 秒预测，包括 t0

全部联合计划保留，不删除失败、停滞、reset附近计划或不安全初始点。下表的一条 item 是一次联合 replan 的一个无序行人对，或一个行人。

| 组别 | 联合 replans／actor plans | peer unsafe phases：raw → ref | 全151点 raw-safe → ref-unsafe pair-items | static 全151点 raw-safe → ref-unsafe actor-items |
| --- | ---: | ---: | ---: | ---: |
| OLD63e S1 | 420／1680 | 1287 → 1302 | 12 | 16 |
| NEWb563 S1 | 409／1636 | 2722 → 3057 | 18 | 18 |
| NEWb563 S10 | 421／1684 | 3184 → 3422 | 26 | 32 |

完整5秒的 peer 违例阶段数增加，但适配后“存在任何 peer 违例”的 pair-item 总数为60／126／161，raw则为78／127／178；不能只挑一个计数声称单向恶化。完整预测中许多点在重规划后被替换，不能归入已消费时长。

| 起点 raw/ref 偏移 | OLD S1 | NEW S1 | NEW S10 |
| --- | ---: | ---: | ---: |
| 均值 m | .111562 | .140253 | .141304 |
| P95 m | .266948 | .287524 | .313377 |
| 最大 m | 1.278060 | 1.480924 | 1.203926 |

raw 起点是当前物理 root XY；普通重规划的 reference 起点是上次执行的虚拟参考状态。上述偏移符合 producer 的明确合同，不是文件错位。只有首次或真实 reset 才从物理位置／速度初始化 reference。

## 方法、时间与来源边界

- 三组 NPZ headers 实际为 raw `[N,51,2]`、reference `[N,151,2]`。raw 含 t0..5 s、步长.1；按有理数 `phase/3` 线性采样到151点、步长1/30，float64计算净空。与producer保存 `deviation_m` 的最大数值差分别为3.031／2.645／2.634微米，反映原float32与诊断float64算术差异。
- peer clearance=`||p_i−p_j||−2×.6`，所以中心距离阈值是 **1.2 m**。static clearance取五个真实 cuboid signed AABB distance−.8及域内边界余量−.8的最小值；与原 guidance 使用的世界米值和半径一致。`clearance<0` 为代理违例，0为几何边界。没有把 .6错当 pair separation，也没有重建身体碰撞。
- 原 NN guidance 成本作用52个未来点 .1..5.2 s；这个 archive只保存选中前50未来点加t0。报告分别保留全部151点、future150点及实际消费前缀；**这些是 dense proxy counts，不能称原NN指导碰撞率**，也不能重建未保存的5.1/5.2或所有未选候选。
- 每次记录完整 ordered actor_ids0..3、相同control_step、相同joint selected_sample；共 **7500 pair CSV rows** 和 **5000 actor CSV rows**。[全部行人对](all_joint_plan_pair_rows.csv)、[全部行人静态净空](all_actor_plan_static_rows.csv)。prefix长度来自下一次保存的 replan control，最后截止6000，三组均覆盖完整6000 controls，没有重新运行政策。
- archive没有episode/reset字段。本报告的episode **明确是推导值**：依据既有已封 fall summaries 的 actor/completed_control、三组真实time-limit reset0，以及producer在该完成步后reset、下一plan前初始化的合同。保留所有0／1／3跌倒和原idle/contact摘要，不声称从plan数组直接观测episode或重建接触力。
- 模型身份只读取已有source-bound结果：AMP均62c4；DP旧63e、新b563，S1/S10分别1/10候选；场景均同一bounded fixture `[2,2,22,22]` 和五个 cuboid。NPZ/result/GPU owner/CPU owner/raw audit均逐字节绑定既有摘要记录及只读单链接文件，原 GPU/CPU owner真实closed0。

本次支持的结论仅是：**现有后处理未保持原始路径的几何净空合同，局部改善与退化并存；不能据此将它确定为物理失败来源，也不能据此立即删除适配器或更换默认。** 后续可由 root 调度，将首次 peer 接触与同一 saved target prefix 的实际身体位置／支撑另作事件关系诊断；当前没有执行。不会为此新增门禁或并行仿真。
