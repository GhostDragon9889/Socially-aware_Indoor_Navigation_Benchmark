# Diffusion + AMP 训练指标快照

生成时间：2026-10-02T14:50:30.063693+00:00。这是读取日志时的快照，脚本可在训练期间及结束后重复运行。

## Diffusion

数据类型：`amass_individual`。图标题：AMASS root-trajectory diffusion: raw training metrics。标签仅依据输入 JSON 的数据声明；精确原文和字段路径见 summary.json。

已记录步数：1000；初始验证 loss：1.5976815223693848；最终验证 loss：0.17749340273439884；独立测试 loss：0.38898174464702606。

![Diffusion 训练与验证](diffusion_training.png)

## AMP

判别器输入：`state_action`。图标题：Action-conditioned AMP discriminator。记录的训练配置、profile 路径与数据 provenance 见 summary.json；不会根据专家动作字段推断判别器结构。

共读取 1000 条完整日志，最新 update 为 1000，累计物理 transitions 字段为 2048000；检测到 1 个进程片段。

最新 reward mean：0.6738139390945435；该进程累计 fall resets：87；time-limit resets：6745；该进程运行最低 pelvis height：0.3537866771221161 m。

![AMP 优化目标](amp_optimization.png)

![AMP 奖励与物理 rollout](amp_rollouts.png)

## 解释边界

- 图中均为原始记录，不平滑，不依据 loss/reward 自动判断稳定或收敛。
- 恢复训练后当前 runner 的累计 reset、最低高度、完成 episode 平均时长和 elapsed 会从新进程重新开始；曲线在检测到的边界断开。
- fall 与 time-limit 分别采用明确字段；旧日志没有 fall 字段时保持未知。reset 差值仅比较同一进程的相邻记录。
- AMP result.json 可能是上一轮已完成进程的结果；summary.json 单独记录其 update 是否与最新日志一致。
- Recorded diffusion losses do not establish closed-loop crowd performance or author-dataset equivalence.
- Recorded assumptions.neighbors: Empty: AMASS supplies no synchronized pedestrian interaction observations
- Recorded assumptions.raster: Uniform walkable=1, obstacles=0: AMASS supplies no scene occupancy maps
- Recorded assumptions.limitations: This run trains five-state locomotion prediction, not validated social/map-conditioned avoidance
- 输入缺失或矛盾时标签保持 unknown；这些图不能证明论文八场景导航评估、作者 2,848 动作清单或作者权重的一致性。
- 精确输入快照字节数、SHA-256、进程边界与最后完整记录见 [summary.json](summary.json)。
