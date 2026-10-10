# 新奖励权重试验：完整同帧检查与动作诊断

2026-10-03，CPU 实际检查；未启动 GPU、加载模型权重或读取 sealed34 输出。

## 输入对齐

新权重 `09e802…` 的完整 22 条 validation 跟踪输出，在真实 GPU 进程退出并由 root 封存后消费。CPU 指标执行器也已实际闭合，来源准入绑定训练 source `1fd9…`、terminal `c0e93…` 和控制器完整 SHA。

直接复用原 [字节比较实现](../../../docs/state_prior_tracking_reference_peer_v1/compare_actual_references.py)；对新模型与 `19f`、`62c4`、state-action2000、protected2000 四组历史输出进行比较：

- 88 对完整动作片段，共 1,584 项完整参考动作／时间／相位／导航命令数组比较。
- 968 项初始 reset 诊断数组比较，包括 root、DOF、24 个身体及 AMP history。
- 全部 2,552 项 shape、dtype 和 C-order bytes 一致；初始状态也全部一致。
- 新旧 `19f` 的保存 TrainingConfig 只有 `amp_reward_weight: 2.0 → 0.5` 不同。
- 生成动作在 reset 后的轨迹、误差和失败状态保留，没有要求它们相等。

实际全量证据：[actual_reference_pairing.json](actual_reference_pairing.json)。原 GPU import ledger 没有观察到 runtime finalization，报告保留此限制。

## 已完成指标

全部 22 条、11,982 个 metric frames、11,960 个合法片段内 style intervals 保留，包括新模型的 7 条跟踪失败片段。

| 指标 | 旧 19f，AMP 外权重 2 | 新 09e802，AMP 外权重 0.5 |
|---|---:|---:|
| 跟踪成功片段 | 17 / 22 | 15 / 22 |
| 全身体位置误差，m | 0.209859 | 0.251094 |
| 全身体旋转误差，rad | 0.810943 | 0.891657 |
| 固定 5 个支持速度区间的 style MMD² | 0.042092 | 0.110080 |
| 原近地 toe 代理速度，m/s | 0.791035 | 0.290713 |
| 原代理选中的足部间隔数 | 15,572 | 5,996 |

这轮试验没有改善整体跟踪和当前风格指标。论文 PACER 的奖励系数比例不保证适合本地模型、判别器和训练数据。

## 脚部代理的选择效应

原近地代理只统计生成脚部同时满足高度和垂直速度条件的间隔，新模型符合这一条件的样本明显减少。为解释这一变化，另使用两轮完全相同的 **reference** 几何支撑掩码，对同帧生成 toe 速度统计，沿用原 `b09…` summary 数学：

| 固定参考支撑帧统计 | 旧 19f | 新 09e802 |
|---|---:|---:|
| 左 toe XY 速度均值，6,424 个间隔，m/s | 0.835103 | 1.302997 |
| 右 toe XY 速度均值，6,378 个间隔，m/s | 0.900754 | 1.260769 |
| 左足生成几何支撑覆盖率 | 0.683375 | 0.331258 |
| 右足生成几何支撑覆盖率 | 0.682346 | 0.191910 |

因此原代理数值下降不能解释为真实脚滑改善。toe 身体原点的几何支撑与 XY 运动没有测量接触点或鞋底滑移。

## 上肢和腿部

既有 SO(3) 同帧分解显示，左／右腕的 parent-relative 旋转误差均值从 `0.698 / 0.683` 降到 `0.335 / 0.311 rad`，但手腕世界位置误差从 `0.279 / 0.288` 增到 `0.377 / 0.349 m`。最大世界旋转误差增加出现在左踝 `+0.274 rad`、左 toe `+0.269 rad`，随后是膝、肘和 pelvis。这里的下降或增加描述样本均值，没有进行显著性检验。

诊断与全部间隔保留在 [new935_actual](../state_prior_pacer_reward_pose_diagnostics_v1/new935_actual/diagnostics.json)，固定参考支撑统计及逐身体差值在 [fixed_reference_stance_comparison.json](../state_prior_pacer_reward_pose_diagnostics_v1/new935_actual/fixed_reference_stance_comparison.json)。所有这些是诊断，没有新增质量门槛或替换默认模型。
