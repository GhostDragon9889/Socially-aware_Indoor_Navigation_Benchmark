# 下一项小步：匹配的 state-only 判别器消融

本文件是研究建议，未修改生产 185 个源文件、未建立新实验框架、未启动训练。先完成正在进行的 q-expert 完整评价，再决定是否执行。

## 需要解释的现象

旧 `19f…` 完整 22 条 validation 的同帧身体旋转误差为 **.810943 rad**，论文报告目标为 .343 rad；作者动作 ID／指标实现不完整，两者只能作为差距线索，不能据此作严格排名。[现有 decomposition](../state_prior_metrics_v1/tracking_decomposition/decomposition.json)保留原指标及失败帧。

去掉父关节整体朝向后，左右腕的旋转误差仍为 **.698／.683 rad**，左右肘约 **.750／.683 rad**。这是根方向以外的姿态差异；尚未证明其训练原因。训练的 195 维 AMP 帧特征已经包含腕部旋转，不能用 90 维评价不含手部旋转来推断判别器看不到腕姿态。[同帧诊断](../state_prior_pacer_reward_pose_diagnostics_v1/README.md)

## 唯一机制变量与最低切口

建议测试：**去掉判别器的 69 个 action 输入通道，保留完整 10×195 维状态历史。** 当前 expert action 来自下一帧 mocap 关节位置的归一化 PD 目标，不是 AMASS 实测控制动作；它与物理策略动作的语义差异可能为 D 提供区分捷径。这是待检验假设，不是已经定位的 bug。

现成配置接口足够，只在同一基线复制出的 fresh 隔离配方中调整：

```yaml
training:
  discriminator_input: state
  warm_start_discriminator: true
  warm_start_action_discriminator_from_state_prior: false
```

这三个字段共同实现一个机制变量：把原先公共 1950 维 D 扩展的 69 个零初始 action 列移除，并直接复制同一公共状态 D 和同一历史归一化器。初始化声明应准确改为完整 state-only D；不需要重写网络、AMP 观测或奖励。

保留所选基线的 q、2848 条动作、任务／AMP 权重、地形、镜像系数、57 个有效动作、公共 actor／critic、统计保护、seed7 和 fresh2000 预算。不能直接使用旧 `amp_pacer_adaptation.yaml`：它还改变奖励、镜像和其他配方。不同 D 架构会改变 RNG 消耗；相同 seed 也不能冒充逐步随机样本相同。不能 resume 原 2019 维 D checkpoint。

具体既有代码切口：`CrowdSim/humanoid/training/trainer.py:83` 的 D 输入维度、`:214` 的公共 D 转移和 `:314` 的分类输入；expert 代理来源在 `CrowdSim/humanoid/amp_world.py:268`。`CrowdSim/humanoid/amp/observations.py:175` 的 195 维姿态观测保持原样。

## 论文依据与方法边界

[公开 PACER 作者代码，固定 commit `baeb6241…`](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/learning/amp_continuous.py#L666)实际以 `amp_obs` 评估 D；本地下载字节及 commit 匹配证据见[来源记录](../paper_fidelity_research_v1/pacer_commit_binding.json)。[NavIsaacLab 式(2)](https://arxiv.org/html/2606.26265v1#S4.SS2)写的是历史加 action，所以这项实验属于**公开 PACER 路线的机制消融**，不能标成严格 Nav 原方法。保留原 state-action 主分支。

[TRACE/PACER 补充 B.2/B.3，印刷页16–17](https://nv-tlabs.github.io/trace-pace/docs/trace_and_pace.pdf)以状态历史学习风格，讨论多自由度控制与局部窗口的困难；式(22)约束策略的镜像等变性，不要求同帧左右姿态相等。此轮不同时改变镜像。原二维路线任务也没有唯一的参考上身姿态或运动相位，因此状态分布改善不能保证同帧旋转误差下降。

只沿用现有完整 22 条／11,982 帧的 tracking、90D、覆盖及 parent-relative 诊断和完整 200s fixture 物理统计，保留所有失败与最坏动作；不增加质量门或筛掉已完成结果。若 D 分布改善而腕／肘误差不变，应保留这个负结论。

**上身姿态条件／style latent 属于后续方法扩展。** 它需要额外条件、训练目标和部署可用的姿态生成，能处理同一 XY 路线对应多种上身动作的问题，但不能把参考未来姿态直接喂给 actor 后称为原论文复现。本轮建议只检查现成 D 的 action 代理是否干扰姿态先验；不承诺达到或超过作者效果。
