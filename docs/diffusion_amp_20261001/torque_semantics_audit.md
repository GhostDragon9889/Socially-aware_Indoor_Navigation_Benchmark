# 当前 ImplicitActuator 扭矩与功率语义审查

审查方式：只读当前固定 SDK／项目源码与 NVIDIA 官方文档。未启动 Kit／GPU，未改业务、环境或已有 checkpoint。

## 已确认的问题

当前 SMPL 使用 BUILT_IN_PD → ImplicitActuatorCfg。`data.applied_torque` 是裁剪后的近似 PD 估算：`clip(Kp*(q_target-q_pre)+Kd*(qd_target-qd_pre)+tau_ff)`；它由 actuator.compute 的 telemetry 写入，并非 solver 测得的实际 implicit drive输出。具体见 [actuator_pd.py](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab/isaaclab/actuators/actuator_pd.py:136) 和 [articulation.py](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:4329)。

[当前物理循环](/project/EXP/SangforAndSaut/ProtoMotions/protomotions/simulator/isaaclab/simulator.py:646) 在每个120Hz子步先计算／提交控制，再 step、update。最终 `_get_simulator_dof_forces()` 读到第4子步前的估算（控制步内3/120s），但奖励配的是第4子步后的速度（4/120s），相差 **1/120s≈8.33ms**。同一 controlstep 并不等于同一物理时间。PD刚度较高、制动、限位、碰撞时，功率大小甚至符号可能偏差。

物理引擎的隐式 drive 使用步末状态求解；从步初状态计算的显式 PD 估算不能据此视为等价实测。[NVIDIA PhysX 官方驱动说明](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html#articulation-joint-drives)

## 当前版本确实存在的公开接口

| 接口 | 实际含义 | 能否直接替代当前隐式扭矩 |
|---|---|---|
| `robot.root_view.get_dof_actuation_forces()` | effort/feedforward指令缓冲；implicit位置／速度drive另行提交 | 不能；纯位置PD时指令可全零 |
| `robot.root_view.get_dof_projected_joint_forces()` | solver将incoming joint force沿自由DOF方向投影的active joint effort，shape `(B,max_dofs)` | 可用于明确声明的 measured joint-effort方案；不是隔离纯motor torque |
| `robot.root_view.get_link_incoming_joint_force()` | 每link的6维joint wrench，child-side joint frame，shape `(B,L,6)` | 不能直接截成69DOF扭矩 |

Getter 的确切来源是当前安装的 `omni.physics.tensors110.1.11` [api.py](/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.1.11+110.1.1.lx64.r.cp312.u7f4/omni/physics/tensors/api.py:1991)。当前 Lab 的公开 `root_view` property 可直接访问，不需要创建旧deprecated Articulation wrapper。旧 wrapper 的 `get_measured_joint_efforts()` 自身也调用这个 projected getter。

读数是最近物理步内的平均关节传递力；自由轴分量包含 drive、用户effort、摩擦、速度夹紧和位置限位。正常自由轴可反映drive，但不能保证每帧都为单独电机输出。[NVIDIA PhysX 官方关节力说明](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html#link-incoming-joint-force)

## 建议新 reward stage 的可执行实现

保持同一 implicitPD、120Hz物理、decimation4，不换控制器。在**每个** `_sim.step()` 与 `_scene.update()` 后读取：

```python
# Same native DOF order. Clone before the next getter refreshes its buffer.
tau = wp.to_torch(robot.root_view.get_dof_projected_joint_forces()).clone()
qd = wp.to_torch(robot.root_view.get_dof_velocities()).clone()
signed_power_substep = (tau * qd).sum(-1)
# After all four physical substeps:
signed_power_control = torch.stack(substep_powers).mean(0)
energy = -0.0005 * signed_power_control
```

当前SMPL全为旋转DOF：Nm × rad/s → W。两数组必须同一nativeDOF序，或共同应用相同native→common排列；不能将body wrench直接拼作DOF扭矩。平均保留功率尺度；四项直接sum会把奖励放大4倍。`dt*sum(P)` 是J，若需要维持论文系数对应的W应再除控制dt1/30s。

此实现使用每子步平均力 × 该子步步末速度，是明确的右端点离散选择，不是精确连续功积分，也不是作者未公开实现的精确复刻。真实getter返回、非零值、符号、有限值、69DOF顺序和吞吐仍需root的有界PhysX probe确认；接口失败不能静默fallback到原proxy。

## 若只能采用估算

每次 `write_data_to_sim()` 后、`sim.step()` 前同时复制 `applied_torque` 与pre-step `qd`，四个同时间点的signedproduct取平均。另一种显式选择是在post-step用相同target／gain／feedforward和post-state重新计算PDproxy，再配post-state速度。两者都修正错时刻，但都仍不是solver实际drive测量；不能通过调用compute覆盖目标或推进一次仿真来伪造测量。

## 已训练结果的保留

保留首轮1000-update checkpoint／metrics／原sourcehash与原provenance。旧checkpoint的信号切换需显式 `--new-power-stage`，记录sourcecheckpoint SHA、sourceupdate、getter、signal-kind、substep timestamp、aggregation、单位、系数。已有NPZ／粗粒度metrics没有完整120Hz力与速度轨迹，不能事后重算精确新奖励。

论文Eq5保留signed `−0.0005 Σ torque_i*omega_i`，但没有给出具体测力getter／子步聚合；重建中的该选择必须说明。精确文件hash、代码行号与未验证边界见 [torque_semantics_audit.json](torque_semantics_audit.json)。

## 修正后状态

上述“当前”问题指首次审查的修正前源码快照，原hash保留。root已新增专用 `PaperPowerSimulator` 与显式功率stage；独立CPU复核 **33 passed，Ruff PASS**。详见 [joint_power_patch_review.md](joint_power_patch_review.md)。真实PhysXgetterprobe尚不属于本审查结果。
