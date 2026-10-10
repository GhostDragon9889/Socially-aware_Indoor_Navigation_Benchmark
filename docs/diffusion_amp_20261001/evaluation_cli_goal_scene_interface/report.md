# 行人姿态与物理记录评估

**本报告仅验证合成输入接口；不是实际物理仿真结果。**
输入类型：`synthetic_interface`；生成时间：2026-10-01T09:46:41.201905+00:00。

CPU 读取 16 帧 × 4 人，选取 64 个生成姿态；AMASS train/test 各 2048 个姿态，seed=7。

## 固定 train cluster 与 held-out

| K | train coverage | test coverage | generated coverage | generated entropy |
| --- | --- | --- | --- | --- |
| 50 | 1.000000 | 0.920000 | 0.080000 | 0.354368 |
| 100 | 1.000000 | 0.810000 | 0.040000 | 0.301030 |

Generated ↔ test Gaussian MMD²=0.032631485，MMD=0.180641869，固定 σ=1.0，biased V-statistic。
Train ↔ test 参考 MMD²=0.035753752，MMD=0.189086626。

## 完整物理序列统计

XY 路程（排除 episode/reset 跳跃）：[0.14000000000000723, 0.1499999999999999, 8.67204877951757, 0.15000000000000036] m。
跌倒 flag 事件起点：0；受影响已观测 episode 比例：0.000000。
同刻 XY 跟踪平均误差：0.100000 m；RMSE：0.100000 m。
Episode ID 与 reset mask 边界不一致数：0。
地图碰撞：未知（无记录 mask）；行人碰撞：未知（无记录 mask）。

## 记录目标、计划终点与路线推进

记录目标语义：Sparse diffusion-conditioning goal; arrival at a full route/global destination is not established by tracking error. In this fixture the recorded global_goal_xy is an A* arclength-lookahead point, which can change at replanning.
已记录 route vertices reached：[2, 0, 1, 0]；A* query failures：5。
全局/完整路线到达率保持未知；近时刻轨迹跟踪误差不代表到达。
Physical root 到记录的 planner condition XY 目标距离：平均 5.027709 m，末帧各人 [4.9583162464691535, 4.911466176204414, 4.911466176204413, 4.911466176204413] m。
Diffusion 计划末端到记录目标 residual：平均 2.000000 m，p95 2.000000 m；按物理记录帧计，计划可能重复。

## 显式几何代理

静态 cuboid XY root-point/domain proxy：{'observed_flagged_frames': 9, 'observed_frames': 64, 'flagged_frame_fraction': 0.140625, 'observed_event_starts': 2, 'observed_episodes': 5, 'affected_observed_episodes': 2, 'affected_episode_fraction': 0.4, 'definition': 'explicit XY root-point cuboid/outside-domain proxy', 'method': 'Shared fixture ReconstructionScene.occupied XY root-point cuboid footprint OR outside declared bounds; no PhysX contact data', 'scene_name': 'synthetic_goal_scene_interface', 'bounds_xy': [0, 0, 10, 10], 'cuboids': [{'center': [5, 5, 1], 'size': [2, 2, 2]}], 'outside_domain_is_overlap': True, 'inside_cuboid_footprint_frame_agents': 1, 'outside_domain_frame_agents': 8, 'z_geometry': 'XY projection: no full-body footprint, vertical contact, or shape intersection test'}。
共享场景 XY circle proxy：{'observed_flagged_frames': 46, 'observed_frames': 64, 'flagged_frame_fraction': 0.71875, 'observed_event_starts': 6, 'observed_episodes': 5, 'affected_observed_episodes': 4, 'affected_episode_fraction': 0.8, 'definition': 'explicit shared-world XY disc overlap proxy', 'radius_m': 0.3, 'shared_world_declared': True, 'self_pairs_excluded': True}。
这些代理单独报告；实际 PhysX contact 缺失仍为未知。

## 解释边界

- 不是作者参数/作者权重/八场景 benchmark 的严格复现；不判断稳定、收敛或论文数字一致性。
- MMD 使用显式局部 45D 单帧姿态和固定核；节点、尺度、抽样量会影响数值。
- 两份 cluster artifact 只来自训练数据。完整源 hash、采样 motion/frame/agent 索引和未知项见 [evaluation.json](evaluation.json)。
- reset mask 使用 before_frame：每个新 episode 的首个 post-step 状态。物理路径仅计算同 episode 的相邻记录。
