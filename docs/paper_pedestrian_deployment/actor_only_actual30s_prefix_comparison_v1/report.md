# Actor-only 实际30秒：双基线逐字节对比

实际 GPU 运行完整结束，CPU 对比状态 **measured_exact_prefix**。候选目录为 `output/diffusion_amp/pure_actor_only_baseline_30s_v1`；执行900控制步、30秒、4人，wall elapsed 122.115863 s。以下结论来自真实保存数据，未用合成轨迹、allclose 或删除失败帧替代。

## 两组完整原始字段比较

| 捕获 | 每个基线字段数 | 每个基线元素数 |
|---|---:|---:|
| physical_states | 26 | 1,209,600 |
| ground_support | 42 | 9,197,128 |
| reference_plans | 9 | 250,976 |

新actor-only运行分别与旧 `pure_inference_baseline_30s_v1` 和原selected200秒运行前900步比较，共 **154个字段全部形状、dtype、原始bytes一致**。实际differences为空；每字段两侧SHA、shape、dtype与位一致标志见 [raw_field_equality.json](raw_field_equality.json)。动态physical/foot-force/COM/episode/reset/规划数据均包含，不只比较root或aggregate。

## 原生测量门禁

复用原冻结prefix reader与native auditor，不改数值/接触公式，203项检查全部通过：900controls、3600actor-frames、3600shared physics substeps、14,400actor-substeps、57,600foot substep observations，实际counter0→3600连续、每control4有效子步。原生测量记录已验证；native engine自身时间戳并未采集，submitted counter不能改称native engine clock。

严格1N足接触、foot/body名映射、force有限/shape、COM/mass组合、static/peer通道、episode/reset同帧合同完整保留。独立COM重构最大误差 9.937908949e-07 m，运行记录最大误差 1.907348633e-06 m；mass合计 63.31394896 kg。参考AMASS trueCOM仍未知，未伪造GT；脚body-origin速度仍是proxy，非真实接触点滑移。

## 本段实际表现与源码

0 falls / 0 time-limits / 0 postinitial resets；路线vertices `[1,0,1,0]` 共2。最低pelvis 0.728029072 m。static/peer接触字段与旧前缀均为0，但本报告不推广为全200秒无碰撞。原200秒仍是完整时长的质量证据。

运行声明并且新源码支持只保留actor/task_encoder；两normalizer与mask保留，critic/D/log_std完整验证后弃用，没有Normal或critic forward。前置source168 manifest/archive全部成员SHA核验；它是实际prelaunch源码身份，未虚构完整运行import ledger。旧metadata的critic_evaluated=false并不能抹去旧实现曾计算弃用critic的事实。

权重/profile/scene/data资源、控制/物理协议、seed与legacy CPU RNG preparation和旧pure30完全相同。实际GPU数据在这次900步前缀中逐字节相同，仍不构成所有GPU RNG状态、任意场景/时长或任意运行绝对确定性的证明。新state-action34输出未读取。

完整artifact/source SHA、两组字段比较、203项原生检查见 [comparison.json](comparison.json)，工具与执行证据见 [source_freeze.json](source_freeze.json)。GPU原日志和CPU执行日志分别保存为GPU_run.log、CPU_comparison.log。
