# 新旧 Diffusion：同源码的物理对照

固定 current185 源码、默认 AMP `62c4…`、配置、seed7、路线和物理／引导参数，仅将 Diffusion `63e7…` 换成 `b563…`。[事前登记](../default_newDP_physical_preparation_v1/README.md)、[独立源码检查](../default_newDP_physical_independent_peer_v1/peer_review.json)。历史168／174源码的结果单列，不代替本轮配对。

## 实际状态

| 运行 | 实际完成状态 | 原始物理结果 |
| --- | --- | --- |
| OLD63e，四人200秒 | GPU工具进程72264、CPU工具进程78941均已退出0 | 6,000控制帧；0跌倒／0静态接触actor-controls／0行人接触pair-controls；路线顶点[2,8,6,5]，合计21 |
| NEWb563，四人200秒 | GPU工具进程37121、修正后CPU工具进程66455均已退出0 | 6,000控制帧；1跌倒／0静态接触actor-controls／70行人接触pair-controls；路线顶点[6,1,2,0]合计9 |

OLD：[GPU实际退出记录](fixture_old_actual_launch.json) SHA `5eecb374…`，[CPU实际退出记录](fixture_old_CPU_actual_launch.json) SHA `e9717fdb…`，[独立原始通道报告](fixture_old_audit.json)。检查完整24,000物理子步、96,000 actor-substeps及384,000 foot observations；所有停滞、失败和重置通道保持原样。GPU训练／评价进程与CPU读取分别记录。

当前OLD与历史默认fixture的全部physical_states／reference_plans数组，其dtype、shape和C-order字节完全相等：[实际比较](../default_DP_pair_actual_comparison_v1/OLD_current185_vs_historical.json) SHA `faac9271…`，CPU工具进程73503已退出0。这个观察不保证未来PhysX逐位确定性。此比较未读取运行中的NEW采集。

NEW：[GPU实际退出记录](fixture_new_actual_launch.json) SHA `e31f690e…`。首次CPU来源准入在工具进程54730退出1，尚未创建raw auditor子进程：[失败记录](fixture_new_CPU_first_failure.json)。原因是读取器将保留的AMP历史训练provenance误当当前部署DP，而真正的新部署身份在 `evaluation_planner_change`；冻结v1保留，修正仅涉及元数据准入，物理结果未重跑。

修正后[CPU真实退出记录](fixture_new_CPU_actual_launch.json) SHA `ebdbf67c…`，[原始通道报告](fixture_new_audit.json) SHA `efe7f896…`：203项检查通过，覆盖完整24,000物理子步、96,000 actor-substeps和384,000 foot observations。70个接触pair-controls对应140个actor-controls，峰值1288.423N；原始peer力矩阵没有归档，报告不声称逐值重建该矩阵。

原比较数学保留的[实际OLD／NEW配对](../default_DP_pair_actual_comparison_v1/current185_OLD_NEW.json) SHA `ca1361e1…`及[CPU关闭记录](../default_DP_pair_actual_comparison_v1/OLD_NEW_CPU_actual_launch.json)：工具进程1421退出0，只改变DP权重与输出身份，保留全部跌倒、接触、停滞和重置。检查通过表示记录有效，不表示新模型物理质量通过。

NEW已不满足预登记的零跌倒条件，后续native4／15不执行。风格分数不筛选物理运行。默认权重保持。

[轻量失败事件诊断](../default_newDP_failed_event_diagnostics_v1/README.md)：141.833秒行人3跌倒，当步四个物理子步均无足部支撑且无静态／行人接触标记；完整接触与停滞保留，这些同帧关系不构成因果证明。
