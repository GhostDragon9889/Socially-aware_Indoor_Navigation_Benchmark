# Sim 6.1 / Lab 3EA D6 断点：真实根因及修复

核对目标源：`/project/EXP/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py`，`base_articulation.py`、`PhysxActuatorControl` 与旧 SDK。

| 合约 | 旧 Lab12 / Sim6.0 | 目标 Lab17.0.2 / Sim6.1 | 项目处理 |
|---|---|---|---|
| `num_joints` | PhysX `shared_metatype.dof_count` | 同样是物理 DOF 数 | 保留检查，SMPL 必须 69 |
| `joint_names` | 每 DOF 名，D6 `host:0/:1/:2` | public names，默认来自 `backend_joint_names`；仍每 DOF | 映射全部 69 轴，不补虚假标签 |
| backend names | PhysX `shared_metatype.dof_names` | 同一原始属性 | 未引入不存在的 `find_dofs` API |
| actuator selection | `joint_names_expr` | `ActuatorCollection`→`find_joints(...as_proxy=True)` | 仍选择 69 DOF |
| joint state / torque | `[N, D]` | Warp / ProxyArray `[N, D]` | `as_torch`；要求 D=69 |
| write / targets | `_index` 官方接口 | `_index` 官方接口 | 保留现有官方调用 |
| importer root Physics variant | 默认 `physx` | 实际新生成默认 `physics` | 显式 `UsdFileCfg.variants={'Physics':'physx'}` |
| PhysX D6 repair | 存于 `physx.usda` | 同位置已正确修复 | 保留修复，不改权重、不改 SDK |

## 实证

失败日志 `physics/SecondFloor_N1_60_03.log`：closed articulation / y,z skipped 并非名字重新定义，而是加载了并列 hinge 通用图层；其剩下的 23 个名字对应 23 个实际保留的 hinge DOF。

读取该次转换产物：`~/.cache/protomotions/isaaclab_mjcf_usd/smpl_humanoid_6eddf048006da702/smpl_humanoid/smpl_humanoid.usda`。纯 CPU USD schema 检查：

- 原始 selection=`physics`：69 RevoluteJoint，0 D6，24 bodies。
- session 选择 `physx`：23 D6，69 未锁死旋转轴，24 bodies。
- 原始 USD 字节未改变；不修改生成缓存。
- `resolve_robot_prim_paths` 和运行时 spawn 使用同一 payload selection。

加入 `tests/test_isaaclab61_physx_variants.py`：6 个通过，真实 USD variant/session schema 测试 + 实际 SMPL 69 轴映射、丢失物理 DOF 与重复名字拒绝。未启动 Kit，未运行 GPU物理；该事实仅证明资产图层和维度，真实性/稳定性等待父代理真实场景复验。
