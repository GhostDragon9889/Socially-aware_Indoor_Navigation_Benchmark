# AMP 外层奖励权重对照：已完成，保留原默认模型

本轮固定2,848条AMASS、公开耦合判别器先验、地形统计保护、混合课程、seed7和16,384,000次物理转换，只把外层AMP权重2.0改为0.5。内部scale仍2.0。该数值参考公共PACER配置；NavIsaacLab未公开内部奖励转换，不能把此次尝试称为作者原始配置。

## 实际结果

训练、完整22条跟踪和四人200秒导航均真实运行结束，子进程退出码0。过程成功运行不表示动作质量通过。失败、跌倒、reset和idle数据全部保留。

| 同协议指标 | 原权重2，19f334… | 新权重0.5，09e802… |
| --- | ---: | ---: |
| 完整跟踪成功数／22 | 17 | 15 |
| 跟踪跌倒数 | 0 | 0 |
| 平均位置误差 m | 0.209859 | 0.251094 |
| 平均旋转误差 rad | 0.810943 | 0.891657 |
| 跟踪90D MMD²，固定相同5层，越低越好 | 0.042092 | 0.110080 |
| 导航跌倒／200秒 | 0 | 3 |
| 导航静态接触 actor-frames | 0 | 0 |
| 导航行人接触 unordered-pair-frames | 0 | 8 |
| 实际路线顶点到达 | 40 | 29 |
| 导航90D MMD²，固定共同3层 | 0.068117 | 0.063999 |
| 导航45D MMD²，固定原基底 | 0.046967 | 0.030504 |
| 导航45D覆盖 K50／K100 | 0.58／0.52 | 0.82／0.67 |

新模型导航90D支持5层，总体MMD²为0.090405；原权重2只支持3层，不直接比较这两个总体分数。当前默认62c4模型同样支持5层，原总体为0.072381，共同3层为0.040983。姿态45D和部分共同分层有所改善，但完整跟踪、物理稳定性和路线进展退步，因此没有替换默认模型。

导航完整6000步／200秒，3次跌倒reset、0次时限reset、1708个Diffusion计划；四人分别到达[7,8,8,6]个顶点。行人接触共16个actor-frames、8个unordered-pair-frames，最大记录力658.04 N。按预先登记的零跌倒、零静态／行人接触要求，该模型不能进入后续native4→native15分支。未绕过这一要求，未执行或补造这些未来结果。

## 同帧诊断

[参考、相位和初始状态审查](../state_prior_pacer_reward_tracking_reference_peer_v1/README.md)检查2,552项配对，均一致；保存的训练配置数学差异只有上述权重。腕部自身局部角误差下降，但腕部世界位置及骨盆旋转误差增加。原近地脚速代理下降伴随支撑样本减少；固定相同参考支撑帧后，新脚部速度反而增加。该代理没有接触力，不能作为真实脚滑改善证据。

## 可复查入口

- [预先登记协议](../state_prior_pacer_reward_experiment_v1/registration.json)：历史准备文件保持原状。
- [真实训练关闭记录](../state_prior_pacer_reward_actual_runs_v1/fullfresh2000_actual_launch.json)、[2,000轮终态封存](../state_prior_pacer_reward_actual_runs_v1/terminal2000_root_seal.json)、[完整训练状态审计](../state_prior_pacer_reward_training_audit_v1/terminal2000/audit.json)：2,073项通过；训练287次跌倒reset不等于论文场景跌倒率。
- [完整跟踪关闭记录](../state_prior_pacer_reward_evaluation_actual_runs_v1/tracking_actual_launch.json)、[原始跟踪封存](../state_prior_pacer_reward_evaluation_actual_runs_v1/tracking_root_seal.json)、[90D风格](../pacer_reward_metrics_v1/tracking90/evaluation.json)。
- [导航关闭记录](../state_prior_pacer_reward_evaluation_actual_runs_v1/navigation_actual_launch.json)、[完整原始导航封存](../state_prior_pacer_reward_evaluation_actual_runs_v1/navigation_root_seal.json)、[90D风格](../pacer_reward_metrics_v1/navigation90/evaluation.json)、[45D姿态目录](../pacer_reward_metrics_v1/navigation45/)。
- [实际CPU指标完成记录v2](../state_prior_pacer_reward_CPU_metrics_actual_runs_v1/navigation_CPU_completion_v2.json)与[导航对照说明v2](../state_prior_pacer_reward_CPU_metrics_actual_runs_v1/navigation_CPU_report_v2.md)：显式选择历史`backward_ground200`基线；初版说明的基线选择错误保留在历史文件中，原始指标未改变。

两次评估的已观察源码事件保留，`runtime_finalization_observed=false`如实保留，未声称完整运行时导入清单或PhysX逐位恢复。未读取封存34条测试输出作本轮调参。作者2,848条动作ID、八场景清单及风格度量参数仍不匹配；当前结果不能直接与论文数值排名。

## 后续独立实验

新[1,000场景ORCA数据](../orca_maps_expansion_v1/README.md)已真实生成、独立检查、只读封存；[新的Diffusion训练登记](../orca_maps_diffusion_experiment_v1/registration.json)沿用模型结构和学习规则，更换训练地图规模。该模型需要独立旧／新验证和固定AMP物理测试。另一条AMP改进将保留权重2，只调整现有专家采样映射；两条机制分别评价，不合并归因。
