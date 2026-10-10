# 真实质心与脚部地面接触：固定 SDK 接口复核

日期：2026-10-02。仅阅读固定源代码、解析 Python AST 和已有物理报告；没有导入 Kit、调用原生视图或进行仿真。本报告提供接口方案，**不证明新采集通道已通过物理验证**。

## 版本与证据

- IsaacLab checkout：`4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8`。
- 安装的 tensor API：`omni.physics.tensors-110.1.11+110.1.1.lx64.r.cp312.u7f4/omni/physics/tensors/api.py`；SHA256 `b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903`。
- [固定 PhysX ArticulationData 源码](https://github.com/isaac-sim/IsaacLab/blob/4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py)、[固定 COM 计算 kernel](https://github.com/isaac-sim/IsaacLab/blob/4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8/source/isaaclab_physx/isaaclab_physx/assets/kernels.py)。在线 [IsaacLab assets 文档](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html) 也区分 post-step update；实际四元数和 API 以本地固定源为准。

## 可用通道及其含义

| 通道 | 固定版本接口 / Torch 形状 | 含义 |
|---|---|---|
| link origin | `robot.data.body_link_pos_w.torch` / `[B,J,3]` | 刚体 actor/link 原点；`body_pos_w` 是该通道的 alias |
| link rotation | `robot.data.body_link_quat_w.torch` / `[B,J,4]` | 刚体原点朝向，**XYZW** |
| world COM | `robot.data.body_com_pos_w.torch` / `[B,J,3]` | 基于 native link pose 与真实 local COM offset 得到的质心位置 |
| local COM | `robot.data.body_com_pose_b.torch` / `[B,J,7]` | link-local COM translation + inertia principal-axis XYZW rotation |
| masses | `root_view.get_masses()` / `[B,J]` | 当前原生刚体质量；可能在 CPU parameter device，转换后显式移动到计算设备 |
| native local COM | `root_view.get_coms()` / `[B,J,7]` | **不是 world COM**，是 rigid-body-prim frame 内的 principal-axis pose |
| filtered force | `RigidContactView.get_contact_force_matrix(dt)` / `[sensors,filters,3]` | 指定身体与指定地面间，所有 contact patches 的法向力向量和；单位 N，无摩擦通道 |
| detailed contact | `get_contact_data(dt)` | 六个真实返回数组：force、point、normal、separation、count、start-index。文档类型注释误写“五个”，实现实际返回六个 |

实际 `RobotState.rigid_body_pos` 当前来自 link origin；DOF 位置是关节广义坐标，既不是 link world position，也不是 COM。不能把质量加权 body origins 标成真实 COM。

质心计算是 `p_com_world = p_link_world + R(q_link_xyzw) @ p_com_local`，而非直接相加 local offset。COM principal-axis rotation 与 link rotation 也不能混用。固定 kernel 对两个 transform 相乘。`.torch.detach().clone()` 或 `wp.to_torch(...).detach().clone()` 才能保存快照；ProxyArray 的 Warp 形状不包含 vec3/quat 的分量，不能先按 `.shape[-1]` 判定 Torch 形状。

## SMPL 顺序合同

建议以名称建立 common SMPL 到原生的双射，并记录完整名字与索引。当前 PhysX `Articulation.body_names` 直接返回 `root_view.shared_metatype.link_names`；现有 `data_conversion.body_convert_to_common` 按该名字列表构建。对于新增通道，仍在初始化时核对 `robot.data.body_names`、`robot.body_names`、`root_view.shared_metatype.link_names` 和 common 名字集合；拒绝重复、缺失和未知的 24-body rig。不要假设未来后端的 data public order 与 root_view native order相同。

- 读取 `data.body_com_pos_w` 时按 data 的名字映射；读取 `root_view.get_masses/get_coms/get_link_transforms` 时按 root view 的 native 名字映射。
- 完整 COM 与质量必须采用同一 common 顺序。质量严格有限且大于零，形状 `[B,24]`；COM `[B,24,3]`。不同 actor 的随机质量不能只保存 actor0。
- 脚部实际身体顺序固定声明为 `L_Ankle,L_Toe,R_Ankle,R_Toe`；使用现有 `resolve_body_prim_paths` 从实际 USD 得到 rigid-body prim，不能从 body index 拼路径。

## 最小可选采集接口

可新增独立 observer，保留现有 map/peer 碰撞 monitor 的定义：

```text
create_optional_style_observer(world, enabled=False)
  initialize: resolve exact foot rigid-body paths and enabled ground CollisionAPI descendants
  begin_control_step(): clear temporary four-substep buffers
  sample_completed_substep(physics_dt): clone filtered normal-force matrix after physics+scene.update
  measurements(): return four substep forces, mean/max/any-contact reductions and final world body COM
```

现有场景地面根是 `/World/ground`，默认 terrain mesh 是 `/World/ground/terrain/mesh`。应从 live stage 确認 enabled CollisionAPI 的实际后代，保存 exact paths；不能把 Xform 根、所有静态障碍或净接触力作为地面接触来源。新独立 filter view 可以将 `4*B` 个 exact foot source paths 与每行相同的 exact ground collider 列表匹配。必须验证 `sensor_paths`、`filter_names`、`sensor_count` 和 `filter_count`，然后依实际元数据重排，不能只依构造列表的顺序猜测。

初始化需确认 contact-report activation（当前 `contact_reporting=True` 设置 contact_bodies，spawn 激活传感器），`SimulationManager.get_physics_sim_view()` 非空，所需 getter callable。若用户要求采集且不满足条件，明确失败；若未启用，保持原行为。不要用全零代替接口缺失。已有各 body 的 ContactSensor 也有 ground-filtered `force_matrix_w`，但其 filters 可包含 scene objects；必须检查列名并正确取地面列，不能使用 `net_forces_w` 来隔离地面。

当前控制循环有四个 120Hz substeps，每个 `_sim.step(render=False)` 后 `_scene.update(physics_dt)`；observer 应在此时读取真实 `dt=1/120` 的刚完成步。使用 `dt=1/30` 会把冲量换算力缩小四倍；只采最后 substep 会遗漏瞬态支撑。建议存 `foot_ground_normal_force_w[T,B,4,4,G,3]`（control、actor、substep、foot、ground collider、XYZ），并标明：

- 四 substep arithmetic mean：控制区间平均法向力向量；不是 peak。
- 四 substep peak magnitude / any `>threshold_n`：瞬态接触是否出现。
- 地面法向向量和的 magnitude 不等于斜面多个 contact patch scalar force 的和。若要精确每 patch 法向载荷和或真实接触点，需要 `get_contact_data`，按 count/start 范围还原；缓冲满、索引越界和非有限数据应失败，不丢弃。
- 两脚 mask 需声明 `left=any(L_Ankle,L_Toe)`、`right=any(R_Ankle,R_Toe)` 及采用何种 substep reduction、阈值。脚原点在接触时的 XY 速度仍是 body-origin slip proxy；真实 contact-patch slip 需结合接触点、刚体线/角速度及地面速度另算。

所有 COM/力应 clone，不允许保存 native pointer view 后被下一步覆盖。capture 建议保存 `body_com_world[T,B,24,3]`、`body_masses_kg[B,24]`、foot force 原始数组及 contract；reset/episode 对齐使用已有 frame flags，不在 reset teleport 跨帧计算速度。

## 验证边界

本次 AST 确认安装 API 存在：`create_rigid_contact_view`（393）、`get_link_transforms`（1438）、`get_masses`（2253）、`get_coms`（2305）、`get_contact_force_matrix`（4950）、`get_contact_data`（4986）。固定 PhysX data 的 world COM getter 在1383，COM transform组合在872，local COM getter在935。未调用真实 view，因此新通道仍需一个 actual physics probe：静止脚对地面有载荷、腾空脚无地面载荷、墙接触不进入地面 filter、质量加权 COM 与手工 transform 组成一致、四 substep 数量一致。现有历史 artifact 未包含这些 raw channels，不能据此补写其“真实 COM/脚支撑”指标。
