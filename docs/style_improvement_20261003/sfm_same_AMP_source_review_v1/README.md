# 同一 AMP 的 SFM 高层对照：源码接口审查

状态：仅源码与论文审查。没有启动 SDK/GPU、构造神经网络、加载权重、读取封存 test34 或修改生产/训练/历史冻结源；尚未实现或运行 SFM+AMP 对照。

## 论文要求和公开边界

[NavIsaacLab v1 §IV-C / Table II](https://arxiv.org/html/2606.26265v1#S4.SS3) 明确保留同一个 AMP 关节身体控制器，仅将 Diffusion 高层轨迹生成替换为 SFM target generator；8 个场景、每场景 3–15 人、200 s。论文没有给出 SFM 增益、FOV、积分步长、未来目标转换或路径跟随细节。这里可补齐同 AMP 的本地方法对照，不能宣称作者 SFM 参数或八场景完全复现。

## 实际文件与已存在部分

请求中的 `/project/EXP/NavIsaaclab2.0/world/sfm.py` 不存在。实际源是 `/project/EXP/NavIsaaclab2.0/CrowdSim/world/sfm.py`，与当前 `CrowdSim/world/sfm.py` **逐字节相同**：10,471 B / `88f3274e1bfa2361dd92bb827b7b924d0d7c90d88f4a868ba76913bdfeccfd9c`。

- [已隔离 FOV 修正](../sfm_FOV_staged_fix_v1/after/sfm.py)：10,525 B / `4b1fe12f980dde21b4105e0757d00f4d02bd7fa86bb437030a6c46b46ad67be3`。仅相对速度与邻居 ID/距离/相对位置应用同一 mask/排序；现有一次真实 CPU 回归已保留，不重复运行。默认 `fov=False`；`dot(-relpos, velocity)` 的视野符号没有在该修正中重新定义，应继续保持关闭。
- `CrowdSim/world/sfm.py:8–198`：`Social_Force(edt, cfg).get_action(ego_state, env_obs)` 返回 **下一速度**及四类力项，不能把返回值当位置或未积分的力。目标速度回归、静态/邻居指数排斥、TTC/pass-right 和速度 clamp 已存在；这些本地参数不由 Nav 论文证实。
- `CrowdSim/nav_manager.py:799 / 1160 / 1538 / 2584`：已有实际位置/速度反馈、邻居构造和像素 EDT→米。该 manager 的后续目标属于 **MaskedMimic**，另加 low-pass、yaw-rate、acceleration 等适配；不能直接把它作为同 AMP 对照。
- `CrowdSim/humanoid/amp_world.py:221 / 346`：实际 planar 状态 `[x,y,vx,vy,yaw]`；同场景邻居历史及 mask 已存在。`planner_inputs` 的 raster 是 actor-aligned 224² 裁剪，不能当 world SFM EDT。
- `CrowdSim/humanoid/training/reference.py:105 / 176`：已有共享 `CausalReferenceAdapter`，输入 51 个 10 Hz XY，输出 151 个 30 Hz reference XY/velocity。普通 replan 从上一段已执行 reference 初始化；仅 episode 初始化/reset 使用实际身体 XY/velocity。两种高层源都应保持此规则。

## 当前真实代码缺口

1. `scripts/train_paper_amp.py:134` 的 `main` 只支持 `evaluation_actor_factory` / `native_scene_factory`，没有 `target_generator`。`application/configuration.py:36` 的 source choices 没有 SFM；`application/deployment.py:111` 固定 Diffusion。`train_paper_amp.py:460` 在 `curriculum=None` 时必加载 `PaperDiffusionPlanner`。
2. `application/resources.py:249–263` 必检查 Diffusion checkpoint；runner `312–320 / 393–397 / 463–484` 还混合实际 planner 与历史 AMP 训练 Diffusion 来源。仅 monkeypatch planner 类会产生“实际运行 Diffusion”的虚假 runtime 声明。隔离 SFM 入口必须明确实际 `trajectory_source=sfm`、零 Diffusion forward/采样，并原样保留 AMP 的历史完整 training provenance。
3. `train_paper_amp.py:552–585` 在 chunk 上调用 `.plan`；51 XY 和 `selected_sample` 是共享 reference capture 的消费边界。同一身体快照的全场景 SFM 目标应先计算再切 actor chunk，不能逐 actor 改写身体或 history，也不能用预计邻居位置当未来真值。若保留 capture 中 `selected_sample=0`，必须明确只是单个确定性目标方案索引，不是 Diffusion 抽样。
4. 现有物理 CPU reader 的 source/runtime 准入硬要求 Diffusion；几何/fall/contacts/progress 的原数学可复用，但需一处薄 SFM metadata 准入替换。不能直接把既有 Diffusion audit 的 PASS 当作 SFM admission。

## 最小外部 target generator 契约

建议纯 NumPy adapter 接收：

```text
map: immutable free/blocked grid + origin_xy_m + resolution_m (+ yaw=0 initially)
actual_states: [N,5] world x,y [m], vx,vy [m/s], yaw [rad]
actor_ids: [B] original scene IDs; exclude self/filler/other scenes
route_waypoints: [N,K,2] same existing global-route targets, world metres
waypoint_times_s: existing .5,1,...,5 seconds
replan_dt_s: actual external clock, 15/30 = .5 seconds
 -> raw_xy[B,51,2], uniform t=0,.1,...,5 seconds
```

`raw_xy[:,0]` 必须是该次真实身体 XY；51 XY 再进入**相同**现有 causal reference adapter。AMP checkpoint、alpha、rig/PD、observations/actions、30 Hz 控制、120 Hz 物理、scene/routes/arrival/reset、replan 时钟、原 contacts/fall 和完整预算保持固定。不再叠加 MaskedMimic 的 yaw/low-pass 适配。

邻居行固定 `relpos=p_neighbor-p_ego`、`relvel=v_neighbor-v_ego`，距离是中心距 [m]；`<4 m` 筛选可直接沿现有 manager，FOV 关闭。所有 actor 的第一步使用同一个实际 pre-state。没有力或 predicted state 可写回 simulator。

### 米制地图坐标

- 原 `CrowdSim/world/map.py:252–275` 支持 ROS lower-left origin 和 image top-left row：`cell=floor((world-origin) @ R / resolution)`，`row=H-1-cell_y`，`col=cell_x`；cell center 逆变换用 `(col+.5, H-.5-row)*resolution @ R.T+origin`。
- EDT 为 `distance_transform(free_mask) * resolution` **米**。SFM 梯度为 `(+grad_col, -grad_row)`，`np.gradient(..., resolution, resolution)`。现有 SFM 没有将梯度按地图 yaw 旋转；首个小 slice 应显式拒绝非零 yaw，或在另一个明确变更中旋转梯度，不能悄悄忽略。
- native 现成来源 `runtime/native_grid_scene.py:212–249 / 279–308` 为 .1 m、yaw0、row0=maxY。`provider._map==254` 才是 known free；unknown/exterior 必须 blocked。`_walkable` 已额外做 radius/两 cell half-diagonal 膨胀；给原 SFM 的 EDT 不能误用这张膨胀图再声称原 EDT，必须记录选用 raw known-free 或 conservative expanded 的差异。
- 局部 cuboid fixture `ReconstructionScene._get_global_planner` 的 A* 数组是 increasing-y rows，与 SFM top-left 方向相反；应从同一真实 cuboids/bounds 单独生成 top-left world grid，不能直接给原数组换名。域外必须显式失败/停止并保留诊断，不能负索引环绕或夹到边缘后冒称有效地图。

### 必须先决定的积分问题

既有 SFM `gain_k=5` 且 `v_next=v+dt*5*(v_desired-v)+...`。直接把 `env.dt` 改为 `.5 s` 会在近目标 `v_desired=0` 时给出 `v_next=-1.5*v`（clamp 前），发生反向放大；原导航 manager 通常在约 .04/.033 s 更新。**这不是可安全忽略的接口单位变化。**

建议首个独立版本使用原 SFM 方程的 `.1 s` 同步预测子步，50 步形成 5 s / 51 XY；每次外部 `.5 s` 或 forced replan 都从真实全场景身体快照重新初始化。内部预测不得推进真实 route-arrival 计数。SFM 在预测中如何推进既有 A* waypoint，应单独明确；建议使用同一条已生成路径的下一个未到达 waypoint，而非穿墙指向五秒终点。这是本地未来-target 补齐，论文未披露，不称作者精确实现。

## 建议的下一授权 slice

可做一个小外部 `sfm_target_generator.py`，加一个隔离 evaluate-only runner 接缝，保留原 controller/control/physics/reference 数学；实际 DP loader/runtime/source 标签是必要替换节点。先用一张已知墙面和两行人纯 CPU fixture 验证 Y flip/EDT 米/邻居符号/51 含 t0/同快照/子步时钟，再由 root 单独决定真实物理启动。此报告没有创建 adapter、commands、owner 或新测试框架；等待 root 对这个具体实现 slice 的授权。
