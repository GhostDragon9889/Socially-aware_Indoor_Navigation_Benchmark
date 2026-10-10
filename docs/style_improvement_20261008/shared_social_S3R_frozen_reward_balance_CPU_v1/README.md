# S3-R：冻结判别器奖励与有界任务项比较

CPU实际退出0、Ruff通过；读取已经保存的评分，没有加载Torch、模型或SDK。
使用完整8192个策略窗口及4096个实际专家窗口，保存的失败窗口保持。
这是旧S3-R的局部奖励诊断，不是当前episode2000候选的评价结果。

## 结果

当前外层权重为AMP1／任务0.5，判别器奖励scale2。
轨迹项为`exp(-2 delta)`、`delta>=0`，任意两个状态的加权轨迹奖励差最大不超过0.5。
显式`backwards_penalty`方向项范围为[-1,0]；轨迹与方向两项合计的加权增益上界为1.0。
这里没有为带符号力矩功率的能量项推导上界，因此不对总奖励作同样判断。

| 同一8192个策略窗口的对照 | 平均加权AMP增幅 | 增幅>0.5 | 增幅>1.0 |
| --- | ---: | ---: | ---: |
| 重复当前帧，速度保持 | 1.794786 | 97.3022% | 95.5322% |
| 静止姿势，速度置零 | 2.665086 | 98.9380% | 98.2910% |
| 刚体平移，关节／角速度置零 | 2.580787 | 98.9380% | 98.1567% |

4096个专家窗口中，静止对照AMP增幅>0.5的比例为23.5840%；这不是全2848数据的站立比例。
源评分使用post-update12500的冻结判别器和同一更新捕获的PRE输入；没有重现原训练奖励trace。
十帧对照保持当前动作，但运动学构造不证明接触、支撑或实际PD力矩可实现。
没有计算真实反事实POST位置、能量、未来回报或PPO更新；这些上界比较不证明总奖励支配或停滞因果。

## 方法对齐与下一步

[NavIsaacLab IV-B](https://arxiv.org/html/2606.26265v1#S4.SS2)给出AMP＋0.5×任务的外层配比；当前保持该式。
[PACER示例](../../../third_party/pacer/pacer/data/cfg/train/rlg/amp_humanoid_smpl_cnn_task.yaml)的0.5／0.5及scale2属于另一配置，不能直接认定是Nav作者配置。
当前softplus与scale2仍是复现假设。此诊断提示后续要独立比较奖励变换／尺度与时间动态约束，
同时保留AMASS的合理低运动窗口；任何下一候选都需自己的训练及完整跟踪、风格、路线／物理评价。
本轮episode训练继续，待终态质量结果后决定具体改动；没有切换默认模型。

## 证据与复现

- [完整统计、分位数及来源](result.json)
- [实际执行记录](actual_CPU_execution.json)
- [源实现快照](source_snapshot_index.json)
- [宿主训练观察](training_host_observation.json)：05:34:23，662/2000、5423104物理转换，实际owner／SDK身份保持。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/diagnose_shared_social_frozen_reward_balance.py --output /tmp/S3R_reward_balance_new
```

输出目录必须不存在。完整源评分保持只读，没有重新计算判别器或修改任何训练来源。
