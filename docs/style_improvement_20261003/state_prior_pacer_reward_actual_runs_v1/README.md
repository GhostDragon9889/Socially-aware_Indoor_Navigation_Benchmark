# PACER 奖励比例：实际运行记录

本轮使用2,848条真实AMASS，保持Diffusion＋AMP接口、数据划分和课程，单独检验AMP外部权重2→0.5。目标是改进路线进展，并同时观察风格、上肢与动作覆盖的取舍。

## 已完成

- [固定实验协议](../state_prior_pacer_reward_experiment_v1/registration.json)与[说明](../state_prior_pacer_reward_experiment_v1/README.md)。
- [最终935v2真实CPU冷加载](../state_prior_pacer_reward_training_preparation_v1/final_actual_CPU_readiness.json)：初始模型、RMS、构造RNG及动作与原权重2条件完全相同，只有外部AMP贡献为四分之一；不代表运动效果通过。
- [独立来源／启动器审查](../pacer_reward_launch_peer_v1/final_source_binding_peer.json)；[恢复奖励合同检查](../resume_reward_guard_peer_review_v1/review_v2.json)。
- [真实pilot4启动与结束](pilot4_actual_launch.json)：正常退出，实际观察到子进程结束，完成4更新／32,768物理转换。
- [退出后封存](pilot4_seal.json)：step4是关闭后的实际last.pt独立副本，原字节一致；结果、完整4行指标与已观察的import ledger封存。SDK最终清理标记仍为false，不补造final事件。

## 后续实际状态

完整2,000更新必须在pilot4的完整状态CPU审计通过后重新初始化，不能从旧权重续训。本页后续更新仅依据真实进程退出及工件；目前没有完整训练或评估成效结论。完整跟踪、fixture、原生Warehouse及第二场景的顺序见固定协议。默认模型保留原step1000。

- pilot4完整状态审计[2,081项通过](../state_prior_pacer_reward_training_audit_v1/actual_pilot4_v1/audit.json)，随后另起[fresh2000真实训练](fullfresh2000_actual_launch.json)；已真实退出0，完成16,384,000次转换。
- step500独立快照[已只读封存](step500_root_seal.json)，4,096,000转换；[完整恢复审计2,068项通过](../state_prior_pacer_reward_training_audit_v1/actual_milestone500_v1/audit.json)。
- step1000独立快照[已只读封存](step1000_root_seal.json)，8,192,000转换；[完整恢复审计2,068项通过](../state_prior_pacer_reward_training_audit_v1/actual_milestone1000_v1/audit.json)。

上述500／1,000快照均是milestone。随后[2,000轮终态已封存](terminal2000_root_seal.json)，[完整状态审计2,073项通过](../state_prior_pacer_reward_training_audit_v1/terminal2000/audit.json)。完整22跟踪15/22、四人200秒3次跌倒；没有替换默认模型，没有执行未通过准入的native4／15。[完整质量结果和范围](../state_prior_pacer_reward_complete_review_v1/README.md)。

另行完成[1,000种ORCA地图数据扩展](../orca_maps_expansion_v1/README.md)并启动独立Diffusion训练；本轮AMP评估使用原5000轮Diffusion权重，来源与预算保持。

## 保留的工具失败

第一935 prepared副本文案仍称初始化为唯一新变量及AMP权重2，独立审查拒绝该来源说明；原冻结保留，另建最终v2正确元数据。第一CPU准备发布报告受launcher.chdir影响，失败保留并改绝对输出路径。第一次root封存假设trainer已生成step_4.pt，实际只有last.pt；[错误记录](pilot4_first_seal_failure.json)与随后正确独立复制均保存。以上不被冒充物理失败或运动质量验收。
