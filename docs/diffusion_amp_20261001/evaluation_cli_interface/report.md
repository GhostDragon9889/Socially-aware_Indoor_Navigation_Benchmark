# 行人姿态与物理记录评估

**本报告仅验证合成输入接口；不是实际物理仿真结果。**
输入类型：`synthetic_interface`；生成时间：2026-10-01T09:29:03.815195+00:00。

CPU 读取 16 帧 × 4 人，选取 64 个生成姿态；AMASS train/test 各 2048 个姿态，seed=7。

## 固定 train cluster 与 held-out

| K | train coverage | test coverage | generated coverage | generated entropy |
| --- | --- | --- | --- | --- |
| 50 | 1.000000 | 0.920000 | 0.080000 | 0.354368 |
| 100 | 1.000000 | 0.810000 | 0.040000 | 0.301030 |

Generated ↔ test Gaussian MMD²=0.032631485，MMD=0.180641869，固定 σ=1.0，biased V-statistic。
Train ↔ test 参考 MMD²=0.035753752，MMD=0.189086626。

## 完整物理序列统计

XY 路程（排除 episode/reset 跳跃）：[0.1400000000000074, 0.1499999999999999, 0.15, 0.15000000000000002] m。
跌倒 flag 事件起点：0；受影响已观测 episode 比例：0.000000。
同刻 XY 跟踪平均误差：0.100000 m；RMSE：0.100000 m。
Episode ID 与 reset mask 边界不一致数：0。
地图碰撞：未知（无记录 mask）；行人碰撞：未知（无记录 mask）。

## 解释边界

- 不是作者参数/作者权重/八场景 benchmark 的严格复现；不判断稳定、收敛或论文数字一致性。
- MMD 使用显式局部 45D 单帧姿态和固定核；节点、尺度、抽样量会影响数值。
- 两份 cluster artifact 只来自训练数据。完整源 hash、采样 motion/frame/agent 索引和未知项见 [evaluation.json](evaluation.json)。
- reset mask 使用 before_frame：每个新 episode 的首个 post-step 状态。物理路径仅计算同 episode 的相邻记录。
