# S1：完整 Pop-Art 对照结果（2026-10-05）

## 结论

S1 改善了共享 CNN 行人控制器的数值稳定性和部分独立指标，但仍不合格。
完整 validation22 的跟踪成功率仍是 0/22；四人 200 秒出现 5 次跌倒和
37 个同行人接触 pair-controls，只到达 3 个路线点。动作簇覆盖率下降。
这些结果尚不能证明达到或超过论文；部署默认权重保持。

## 固定条件与实际执行

S0/S1 均从 seed7 全新初始化，使用相同 v7 真实 AMASS 2,848 条片段，
256 人／64 组、32 控制步／更新、2,000 更新，实际各 16,384,000 条物理转换、
8,192,000 个专家窗口和 256,000 个 PhysX 子步。没有续训短检查权重。
S1 唯一数值改动是在 value RMS 更新时补偿 scalar critic 的末层仿射输出；
未改动 actor、共享 encoder、奖励、GAE、专家采样、Adam 状态及物理实现。

- [S1 训练与 CPU 审计退出记录](style_improvement_20261005/shared_social_evaluation_v1/popart_pilot2000_execution.json)：训练实际退出0（`0b3746`），审计退出0（`e481da`），封存退出0（`9a5d5c`）。
- [S1 完整后续流程退出记录](style_improvement_20261005/shared_social_evaluation_v1/popart_post_training_execution.json)：实际终态 `c57118`／退出0；独立加载、全部22动作／风格、30步导航、6000步导航及 CPU 原始数组审计五阶段均实际退出0。
- 终态权重 `shared_social_popart_pilot2000_v1/step_002000.pt`：651,035,694 bytes，SHA256 `29b5903553e800007a5e8c004432f0ba9ec77b6d1573b2c6feab5df6765aa079`。
- AMASS 清单 SHA256 `68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca`。这仍是本地重建清单，作者私有动作 ID 没有恢复。

## 独立动作与风格

全部22条 validation、11,982个 post-physics 控制帧；不在动作中因跌倒重置，
不做 pelvis／yaw／时间最优对齐。所有失败帧保留；风格使用11,960个合法同片段间隔。
风格参考 `c274…`、45D 中心 `b25d…` 固定；不读取 Test34 数组，不重新拟合中心。

| 指标 | S0 | S1 |
| --- | ---: | ---: |
| 跟踪成功 | 0/22 | 0/22 |
| 有跌倒的动作 | 22/22 | 7/22 |
| 平均位置误差，m | 1.865725 | 1.008892 |
| 平均旋转误差，rad | 2.138779 | 1.332191 |
| 45D inclusive MMD² | 0.863276 | 0.035621 |
| K50／K100 覆盖率 | 0.84／0.66 | 0.64／0.49 |
| K50／K100 归一化熵 | 0.573853／0.499683 | 0.602864／0.573645 |
| 速度分布 W1，m/s | 0.811324 | 0.606784 |
| 90D 支持速度区间数 | 5 | 4 |
| **共同4区间** 90D MMD² 均值 | **0.509862** | **0.068384** |
| 保留的跌倒帧 | 11,298 | 1,260 |

共同区间是 `[0,.1)`、`[.1,.5)`、`[.5,1)`、`[1,1.5)` m/s；
本次每侧各区间实际平衡抽样数量都为148／256／256／256。
S1 的 `[1.5,2)` 只有20个生成间隔，达不到30样本下限，不能填0或纳入共同均值。
姿态距离降低与覆盖率下降同时发生，不能仅凭 MMD² 宣称风格全面改善。

## 独立路线与物理

相同 seed7、障碍物 fixture、formal Diffusion EMA40000（SHA256 `4581f29d…`）、
4人、6000个30Hz控制步／200s、24,000 actor-controls、120Hz物理。
物理接触为原生法向力，严格大于1N；peer 采用无序对计数。

| 指标 | S0 | S1 |
| --- | ---: | ---: |
| 跌倒终止／重置 | 956 | 5 |
| 静态接触 actor-controls | 0 | 0 |
| 同行人接触 pair-controls | 0 | 37 |
| 路线点，总计 | 0 | 3 |
| 路线点，逐人 | 0／0／0／0 | 1／1／1／0 |
| 同 episode 根节点平均速度，m/s | 0.905002 | 0.178477 |
| 双脚支撑子步比例 | 0.086740 | 0.904354 |

S1 的低跌倒伴随低速度和长时间双脚支撑，仍需改善运动与路线执行。
进一步的[原始CPU运动诊断](style_improvement_20261005/shared_social_locomotion_CPU_v1/README.md)
确认有效位置间隔80.46%低速，连续双支撑最长149–176s；全部失败帧保留。
native速度中位数.005760m/s，执行参考速度中位数.102451m/s。
较小的即时root→参考距离也受到低速参考影响，不能与论文全局Path Error直接比较。
支撑期间脚刚体原点速度只是代理量，不能称为接触点滑移。
这是单场景／单 seed 的本地对照；不能把5次跌倒换算成论文 Table II 的1.12，
也不能用本地37个 pair-controls 对比论文 Agent Col.7.81。

## 数值诊断与下一轮

[独立终态 CPU 稳定性诊断](style_improvement_20261005/shared_social_evaluation_v1/popart_training_stability_CPU.json)
重算原始 GAE，没有改变公式。S1 终态 rollout 的更新前 critic 均值42.0592、标准差12.9639；
value RMS 方差283.2055（S0约177亿）。最后100次更新训练跌倒标志421（S0为6768）。
这些是训练诊断，独立质量以上表为准。

最后100次 S1 更新全部触发 KL 提前停止，每更新平均3.34个 PPO minibatch，
配置上限为12个。[PACER 原始循环](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/learning/amp_continuous.py)
执行完整 mini-epochs；其公开 SMPL CNN 配置使用6轮和 constant LR。
据此实施 S2：保持 Pop-Art、相同数据／seed／预算，仅将 `training.target_kl` 从0.02改为null。
真实8人及256人短检查、CPU终态审计和冻结加载已经通过，全新2000轮训练正在运行。
关闭 KL 停止可能扩大策略变化；短检查的有限性和正常退出不证明最终动作质量。

该改动不使本地实现等价于 PACER 或作者私有 AMP。原始
[NavIsaacLab 论文](https://arxiv.org/html/2606.26265v1)的2,848动作 ID、
八场景任务列表／碰撞分母及风格数值参数仍未匹配。
后续必须分别改善完整动作、风格覆盖、速度分布、路线和物理，并完成多 seed／多场景评价。

## 可复现对照报告

[S0→S1 JSON 报告](style_improvement_20261005/shared_social_evaluation_v1/S0_S1_independent_comparison.json)
实际 CPU 退出0（`e35d78`）。其历史 reader 原字节保存在
[comparison_reader_S0_S1.py](style_improvement_20261005/shared_social_evaluation_v1/comparison_reader_S0_S1.py)。
规范入口 [compare_shared_social_evidence.py](../scripts/compare_shared_social_evidence.py)
现在同时接受三份旧式退出记录或单份完整 post 退出记录，方便后续 S1→S2 对照。
它核对实际退出、结果哈希、全部动作窗、固定参考／中心、核参数、失败帧与物理预算，
共同支持速度区间另报，不输出整体晋升结论。
2026-10-05另加[严格补充准入](style_improvement_20261005/shared_social_admission_v2/S0_S1_supplemental_admission_v2.json)：
核对实际child输出、完整physics／参数／velocity合同、底层风格实现与本方captures；
90D桶只有相同平衡样本数与参考索引才可给出方向。当前四桶均通过，原S0/S1数值保持。
S2自动owner保留既定冻结reader，完成后的比较还必须经此补充层校验。

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/compare_shared_social_evidence.py \
  --baseline-tracking-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/tracking_style_execution.json \
  --baseline-navigation-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/navigation200s_execution.json \
  --baseline-audit-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/navigation200s_CPU_audit_execution.json \
  --candidate-post-receipt docs/style_improvement_20261005/shared_social_evaluation_v1/popart_post_training_execution.json \
  --output /tmp/shared_social_comparison_new.json
```

输出需为新路径，保留既有结果；CPU比较不启动SDK或CUDA。
