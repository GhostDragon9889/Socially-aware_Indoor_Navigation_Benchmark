# SMPL link-origin／COM 速度只读复核（2026-10-02）

## 当前结论

**当前源码与 AMASS link-origin 参考速度一致，此风险关闭；不需要添加另一层 COM 补偿。** `simulator.py` 实际读取 `body_link_lin_vel_w`／`body_link_ang_vel_w`，root／object root 同样读取显式 `root_link_lin_vel_w`。这些接口已经在本次长训练前存在，本复核未修改任何业务代码。

当前 simulator SHA 为 `2c29945dbd5c36ad3a16f5dffa897fa08e85f6ccf5c56a3f077a0dd81da1fb7f`，与真实32步 tracking pilot 的 `selected_source_sha256` 完全一致。不能把该 pilot 猜成使用 COM alias 的旧版本。其他历史运行须按各自 source hash 判断。

## 1. 已安装 SDK 的精确契约

| 接口 | 已安装源码 | 语义 |
|---|---|---|
| `body_pos_w`／`body_quat_w` | [base_articulation_data.py](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py:1370) | link actor origin 的世界 pose |
| `body_lin_vel_w` alias | [同文件](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py:1388) | COM linear velocity；不能仅凭命名认为它对应 body_pos_w 的导数 |
| `root_pos_w`／`root_quat_w` | [同文件](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py:1322) | root link actor pose |
| `root_lin_vel_w` alias | [同文件](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py:1340) | root COM linear velocity |
| `body_com_vel_w` | [PhysX articulation_data.py](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py:900) | Tensor API `get_link_velocities()`；linear component at COM，world coordinates |
| `body_link_vel_w` | [同文件](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py:843) | SDK 根据 COM pose、angular velocity、link pose 做点位转换 |
| `root_link_vel_w` | [同文件](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py:725) | 对 root 做相同点位转换 |
| `body_com_pose_b` | [同文件](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py:935) | `get_coms()`；COM/principal-axis pose relative to link actor frame |

设 `c_B` 为 actor origin 到 COM 的局部偏移，`R_WB` 为 actor rotation：

\[
p_{COM,W}=p_{link,W}+R_{WB}c_B,
\quad v_{COM,W}=v_{link,W}+\omega_W\times(R_{WB}c_B),
\quad v_{link,W}=v_{COM,W}-\omega_W\times(R_{WB}c_B).
\]

SDK 的 [实际 kernel](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:16) 已执行该修正。COM principal axes 与 actor orientation 可以不同；当前 code 的 `body_quat_w`／`root_quat_w` 明确仍是 actor orientation。世界角速度不因选择刚体的另一个固定点而改变。

PhysX 公共接口也明确指出 `getLinearVelocity()` 的参照点是 COM：[官方 PxRigidDynamic.h](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/include/PxRigidDynamic.h)。这支持底层 COM 语义，但不自动证明闭源 Isaac Gym wrapper 没有另做转换。

## 2. 当前读取链与影响范围

- [simulator body getter](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/simulator/isaaclab/simulator.py:747)：link pose 与 **显式 link** linear/angular velocity 配对。
- [root getter](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/simulator/isaaclab/simulator.py:931)：root link pose 与 **显式 root link** linear velocity 配对；object root 同样使用此接口。
- [AMASS conversion](/project/EXP/SangforAndSaut/ProtoMotions/data/scripts/convert_amass_to_proto.py:202)／[pose kinematics](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/components/pose_lib.py:1468)：gvs 来自 FK body origins 的位置差分；旧转换使用 multi-horizon minimum3，新 curation 同样从重建的 FK positions 计算、明确使用 forward horizon1。两者都没有把参考点改到 COM。

若一个版本错误地把 COM alias 当作 link origin velocity，具体 feature 变化为：

| 当前 PACER-compatible feature | 会变化的通道（零起点，半开） | 不由点位改变的部分 |
|---|---|---|
| self357 |24×3 linear velocity，`[213,285)` | pose／rotation／angular velocity |
| AMP195 | root linear velocity3D，`[6,9)` | root rotation／angular velocity、DOF velocity57D、key body positions |
|5D planner current/history | vx／vy（如果输入直接取 root velocity） | root XY与yaw |

这些是 source-defined 受影响通道，不是当前代码仍存在的错误；当前显式 link API 已处理点位。此问题与另一份审计的 **parent／child joint DOF velocity basis** 不同，不能把它们合成同一个速度 adapter，也不能对当前 state 再做一次 COM 转换。

## 3. 公共 PACER 的证据边界

[PACER humanoid.py](/project/EXP/SangforAndSaut/ProtoMotions/third_party/pacer/pacer/env/tasks/humanoid.py:183) 直接取 Isaac Gym rigid-body tensor 的 pos `[0:3]`、rot `[3:7]`、linear velocity `[7:10]`、angular velocity `[10:13]`；[self observation](/project/EXP/SangforAndSaut/ProtoMotions/third_party/pacer/pacer/env/tasks/humanoid.py:1623) 没有 COM offset 修正。

公共 PACER 源码能证明这个读取行为；没有找到可验证 Isaac Gym 闭源 tensor linear velocity 参照点的官方实现或明确回答。官方 [Isaac Gym 页面](https://developer.nvidia.com/isaac-gym) 不给该细节；已查 [NVIDIA 官方论坛上的同题](https://forums.developer.nvidia.com/t/which-point-does-acquire-rigid-body-state-tensor-measure-from/297479) 只有提问、无权威回答。因此公共权重 transfer 的该项绝对一致性 **未证实**，此界限不构成对当前源码再加破坏性补偿的依据。

## 4. 数值边界与下一次物理采样

本次32步 capture 没有 `get_coms()`／true COM pose／COM velocity arrays。CPU 打开实际缓存 USD：24个 rigid body 均未 author `physics:centerOfMass`，实际 COM 由 collider mass-properties 在物理初始化时计算。因此不能从这些旧数组或空 USD 属性给出真实 COM offset，不能报告已测得某个 COM correction 数量，也不采用 MuJoCo质量几何猜测来替代 PhysX 数据。

若下一阶段需要量化，应在同一120Hz已完成子步后同时捕获 `body_link_pose_w`、`body_com_pose_b`、`body_com_vel_w`、`body_link_vel_w`／root 对应字段，并验证上述刚体点位公式及 API timestamp。再与 FK-origin reference 比较时，须明确30Hz前向差分和原生同瞬时速度的时间估计区别。本审计不启动该采样、不阻塞当前训练、不静默更改历史 provenance。

机器证据与文件 SHA：[body_link_com_velocity_audit.json](body_link_com_velocity_audit.json)。
