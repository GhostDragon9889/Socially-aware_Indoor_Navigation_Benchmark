# S3-R 时间历史与动作条件：实际 CPU 诊断

本批读取已通过恢复训练终态审计的 S3-R 第 12,500 次更新权重及该次完整 rollout。
原判别器、共享 195 维逐帧 RMS、69 维实际施加动作和 AMP 奖励函数均来自原实现。
判别器将 10 帧历史与动作组成 2,019 维输入；逐帧共享归一化不意味着判别器忽略时间顺序。

实际 CPU 执行退出 0，未初始化 CUDA 或启动 Isaac SDK。
8,192 个物理策略窗口和 4,096 个实际专家窗口分别接受六种条件，共评分 73,728 对历史／动作。
专家窗口包含固定 2,848 条真实 AMASS 训练片段中的 2,184 个不同动作 ID，未重新采样或替换训练清单。
RMS、判别器参数和原始输入保持；全部保存分数的统计量通过独立 CPU 逐值核对。

## 结果

两组均使用原 AMP 奖励函数。表中为全体窗口均值，未剔除失败窗口。

| 条件 | 物理策略 AMP 奖励 | AMASS 专家 AMP 奖励 |
| --- | ---: | ---: |
| 原历史与动作 | 0.409293 | 3.613396 |
| 当前帧保持，过去帧逆序 | 0.594555 | 3.501223 |
| 当前帧保持，过去帧固定置换 | 0.470855 | 3.322005 |
| 将当前帧重复为十帧 | 2.204078 | 3.817751 |
| 原历史，动作在全体窗口间循环错配 | 0.409342 | 3.612815 |
| 原历史，动作置零 | 0.409524 | 3.611999 |

物理策略的逆序／置换平均绝对 logit 差为 0.510943／0.240852，
动作循环错配为 0.000442。重复当前帧的平均 logit 从 -1.748603 升至 0.687081。
本批能确认判别器对时间顺序有响应，并观察到这些窗口上较弱的动作条件响应。

**重复当前帧保留了该帧速度及原动作，可能违反运动学一致性，不能等同真实站立。**
奖励升高提示应继续检查时间一致性，但不能证明策略实际利用了该现象，或认定它造成路线停滞。
评分使用更新后的第 12,500 次权重及该次更新前捕获的输入，不能重现原训练日志的奖励 trace。
本批是训练判别器诊断，不是独立动作风格评价，也不构成论文效果达标证据。

当前 episode2000 的已登记来源与配方保持。
待其独立跟踪、风格与路线／物理评价完成后，再决定是否登记时间不一致负样本候选；
真实 AMASS 正样本、独立参考及全部失败后帧应继续保留。

## 复现

选择尚不存在的输出目录；脚本会拒绝覆盖已有证据。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  scripts/diagnose_shared_social_amp_temporal.py \
  --output /tmp/new_amp_temporal_diagnostic
```

实现使用 CPU `weights_only=True`／`mmap=True` 读取权重，
调用原判别器及原奖励方法，并以 `no_grad()` 评分。
6 项 CPU 检查／5 个 subtest 与 Ruff 实际退出 0；测试中的合成网络仅用于实现验证。

证据：[实际执行](actual_CPU_execution.json)、[完整统计](result.json)、
[逐窗口分数](scores.npz)、[来源副本索引](source_snapshot_index.json)。

## 方法依据与边界

[AMP 原论文](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf)第 5.1／5.3 节使用状态转换学习风格，
演示动作不需要配套控制输入；本文本地配方的 state-action 条件与奖励形式已另行声明，未在本轮中更换。
[官方 ProtoMotions steering 配置](https://github.com/NVlabs/ProtoMotions/blob/main/examples/experiments/steering/mlp.py)
使用历史身体观测配置判别器，不能据此补出 NavIsaacLab 未公开的私有超参数。
CPU 读取与无梯度前向依据 [PyTorch 2.11 torch.load](https://docs.pytorch.org/docs/2.11/generated/torch.load.html)
和 [no_grad](https://docs.pytorch.org/docs/2.11/generated/torch.no_grad.html) 文档。
