# 定时全局路点：CPU 实现与回归

状态：**51 项测试通过（43 项新增，8 项原场景回归），Ruff 通过。** 没有启动 Kit／GPU，也没有修改训练 runner、AMP world、正在运行的训练或原有两个场景配置。JUnit：[route_waypoint_tests.xml](route_waypoint_tests.xml)；文件 hash 与检查清单：[route_waypoint_review.json](route_waypoint_review.json)。

## 接口与端点兼容

`ReconstructionScene.route_waypoints(positions[B,2], route_indices[B], actor_ids=None, times=(.5,1,…,5))` 返回 `points[B,K,2], updated_indices[B], reached[B]`。支持显式 actor 子集；所有输入必须有限，索引为范围内整数，时间为严格递增的 `(0,5]` 秒一维数组。空 batch 返回形状正确的空结果。

每个正常查询只调用一次仓库 `Path_Planner.get_astar_path`，沿该路径的累计弧长采样 `goal_lookahead_m * time / 5`；默认 lookahead 为 5m，对应标称 1m/s。短路径到达最终网格点后保持末点。5 秒点与原 `route_goal` 的端点逐位相同，包含非整数网格起点、不同 lookahead、到达路由顶点后的索引推进。这里的时间点是规划提示，不是物理行人已经行走的轨迹或全局到达保证。

原 `route_goal` 不启用起点恢复，其浮点计算顺序与失败计数保持原实现。测试固定了改动前 5.0m／3.7m 两组端点 float32 位值，而非用新辅助函数生成期望。

## 显式起点恢复

新配置 [shared_obstacle_timed_validation.yaml](/project/EXP/SangforAndSaut/ProtoMotions/configs/diffusion_amp/shared_obstacle_timed_validation.yaml) 复制原几何、路由、名称与 `author_benchmark_scene: false`，仅添加 `route_guidance: timed_path` 和 `start_recovery_radius_m: 0.5`。runner 是否启用定时提示仍由 root 的显式评估选项决定；本模块不会更改默认评估模式。

恢复半径默认 0，允许有限的 `[0,.75]` m，仅用于定时 API。实际 root 必须在场景域内且不在真实 cuboid 内；只有其四舍五入网格起点落在 planner 的 clearance 占据区时才搜索候选。候选必须在半径内、planner 网格空闲、实际 root→候选线段不穿过真实 cuboid，按距离及网格坐标确定顺序，选择最近的可达起点。A* 仍使用仓库既有 0.25m 网格、0.35m clearance、无平滑设置。

规划路径保留 `actual root → recovered grid start` 的首段，此首段消耗弧长，不修改、重置或 teleport 物理 root。该首段只证明不穿真实 box，**不认证 planner clearance**；原起点已经不符合该 clearance。

`successful_start_recoveries` 单独累计成功规划事件，非唯一 actor 数。`last_start_recovery_segments` 每次定时调用重置，逐项记录 `query_row`、`actor_id`、实际位置、网格起点、连接段长度和 `initial_segment_clearance_certified: false`。旧 API 调用不产生恢复事件。

## 失败与实证边界

域外、真实障碍内部、恢复半径不足、候选不可达或无 A* 路径，均保持所有 K 个路点等于实际 root；每 actor 查询只增加一次 `global_path_failures`。有效 route 顶点的索引推进仍按原到达半径执行，不以规划成功伪造行人已经到达。

CPU 测试实际执行仓库 A*，验证中央 box 绕行（也检查相邻采样点之间的线段）、单次路径调用、短路径、actor 子集与索引 wrap、非法形状／时间／半径、无路径原地保持、恢复首段的弧长消耗与不修改输入、墙内／域外不可恢复、恢复计数和元数据。另验证梯度输入被显式视为不可微 CPU 规划，避免 NumPy 转换错误。

本结果不证明真实物理行走、planner 条件与 diffusion 预测一致或最终全局导航成功；需 root 单独使用明确 variant 运行评估并保留 provenance。场景为本项目新建验证 fixture，不是论文未公开的八个 benchmark 场景。

改动前源码备份：[reconstruction_scene.py](before/CrowdSim/humanoid/reconstruction_scene.py)。
