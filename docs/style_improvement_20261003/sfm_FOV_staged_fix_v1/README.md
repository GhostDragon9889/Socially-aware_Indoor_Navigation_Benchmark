# SFM FOV 行对齐：隔离修复与未来同 AMP 对照

## 已完成的最小隔离修复

生产 `CrowdSim/world/sfm.py` 保持 10,471 B / `88f3274e…`，当前 935 训练、production V12 和 V5 评估协议未改。完整原字节在 [before](before/sfm.py)，拟议修复在 [after](after/sfm.py)，[精确 diff](sfm_fov_rows.patch) 仅三处语句增量：

1. 相对速度先应用与 ID、距离和相对位置相同的 `in_fov` mask。
2. `>5` 分支再应用相同的距离排序索引。
3. `<=5` 分支返回已经过滤的相对速度。

原实现可以返回零个邻居 ID，却返回两个邻居的速度；部分过滤时 TTC 数组也不再表示同一个邻居。`>5` 分支原来还将过滤后距离的排序索引用于未过滤速度，可能产生错行，即使输出形状看起来正确。

[一个实际行为回归](test_fov_rows.py) 覆盖全部过滤、部分过滤、过滤后 `>5` 最近邻排序，并在相同 fixture 上比较 `FOV=False` 的 `get_action` 输出与四类力项。实际原版 **1 failed**（`(2,2)` 相对速度对 `(0,2)` 过滤行）；staged 同一回归 **1 passed**，三个集合全部正确对齐且 `FOV=False` 力输出 bitwise 相同。[closed CPU receipt](actual_CPU_regression.json)

修复后源为 10,525 B / `4b1fe12f…`。逆转三处语句后全文等于原文件，生产文件仍与 before 逐字节相同。代码尚未合入任何生产/训练 epoch。

## 已保留的独立限制

此次只修复对应行，没有重定义原 FOV 条件 `dot(-relpos, current_velocity) >= 0`。`nav_manager._neighbor_state` 声明 `relpos=neighbor-minus-ego`；这个符号与通常的前半平面约定需要另行澄清。默认 `self.fov=False`，上述分支不参与当前直接 AMP 行人训练/评估。未来最小 SFM 对照保持 `FOV=False`，不把未验证的 FOV 语义掺入同一实验。

现有 SFM 包含目标速度回归、EDT 静态指数排斥、邻居指数排斥、TTC 预测、确定性 pass-right 和速度 clamp。它已是本地扩展；NavIsaacLab 作者的实际 SFM 参数和适配方式未公开，本地结果不能称为作者同配置复现。

## 当前接口缺口

- `application/configuration.py` 的 trajectory source 列表只有 diffusion、procedural、AMASS root 和 mixed，没有 SFM。
- 直接 `train_paper_amp.py` 在 `curriculum=None` 时构造 `PaperDiffusionPlanner`。当前不能用 CLI 单独换 SFM 并保持同一个 AMP。
- 现有 `nav_manager` 的 SFM 消费当前 world XY / velocity、world occupancy EDT、邻居相对状态和当前路由 waypoint。它生成的是 MaskedMimic target，外加速度平滑、yaw / acceleration 有界未来 target；不能把该 controller 的结果充当同 AMP 的 SFM 对照。
- 直接 AMP world 当前状态为 `[x,y,vx,vy,yaw]`；共享场景提供全局邻居当前状态和 mask。Diffusion raster 是 actor-aligned，正 local y 为 increasing row；SFM 原 world EDT 采用 top-left image / world-y 翻转。这两者不能直接共用索引。

## 下一独立 epoch 的最小计划

1. 新建隔离的 SFM planner adapter，复用已明确来源的 `Social_Force.get_action` 和现有有界 future-target 方法。只改变高层计划源，固定同一个 AMP checkpoint、rig、PD、scene、routes、actor 身份、重规划时钟、physics steps、collision/fall 定义和样本保留。
2. 由相同实际 scene occupancy 构造 world EDT；像素距离乘地图 resolution 变为米。明确 world-to-row/col、y 翻转、map origin、障碍物、actor radius 和边界。以纯 CPU 已知墙面 fixture 验证梯度方向与坐标转换；不能拿 actor raster 当 world map。
3. 每次重规划对所有 actor 使用同一个 physical pre-state 快照。邻居输入固定为 neighbor-minus-ego 的 XY / XY velocity，排除自身和不属于场景的 filler；披露邻居半径和距离筛选。`FOV=False`，保留 TTC / pass-right 等本地参数声明。
4. 将 SFM 的 desired velocity 经已有有界运动预测转换为 `t=0,.1,..5 s` 的 world XY `[B,51,2]`，保持 actor 最终接收相同的十个 `+.5..5 s` target。SFM Euler `dt` 必须等于其实际更新周期（`replan_steps/30`），不能默认为物理步长却只在另一频率调用。不得用预测位置改写物理 root、真实历史或 AMP actions。
5. 增加独立 evaluate-only 入口/来源声明与对照 registration；现有 V5 和正在运行的 935 完全保持原字节。没有 Diffusion 时明确不加载/执行其权重，记录 SFM 参数、adapter SHA、原 map SHA、actual planner calls 和 host/device 开销。
6. 先用必要 CPU 解析 fixture 和一次短物理启动确认 shape、mask、clocks 和来源，再以相同完整导航预算比较 Diffusion+同AMP 与 SFM+同AMP；保留所有停滞、跌倒、碰撞以及各 scene 结果。SFM 的参数未知、相同 seed 不意味着 paired stochastic trajectories，以及高层结果不能单独证明上肢风格均需写入报告。

本目录仅完成 staged FOV 行修复及 CPU 证据；未来 adapter、物理 SFM ablation 和质量评估尚未执行。
