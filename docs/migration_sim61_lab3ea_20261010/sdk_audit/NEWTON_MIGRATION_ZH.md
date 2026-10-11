# Newton 1.5.2 源码迁移与 CPU 验证

目标为 `/project/EXP/IsaacLab` 实际安装的 Newton 1.5.2 / Warp 1.16.0。目标行人仿真的主后端仍为 Isaac Lab / PhysX；以下覆盖可选的 ProtoMotions Newton 后端。未提交或推送。

## 已修复断点

| API / 行为 | 修正 | 验证 |
|---|---|---|
| `SolverNotifyFlags` 从 `newton.solvers` 移除 | 使用根模块 `ModelFlags`，仅旧版回退原接口 | 两个实际 SDK 的导入 / 枚举 |
| `Control.joint_target_pos` 删除 | 选择 `joint_target_q`，旧版回退旧属性 | 活体模型的 q 布局 / DOF 布局 |
| `SolverMuJoCo(ls_parallel=...)` 删除 | 根据真实签名选择参数，无法执行旧串行选项时明确警告 | 实际 CPU SolverMuJoCo 初始化 |
| 新 ContactSensor 参数 / 输出 | `sensing_bodies`, `sensing_indices`, `total_force`，旧接口有显式回退 | Warp CPU 联系核，包含摩擦 / 环境顺序 |
| `model.contacts()` 旧分配路径 | 保存 `CollisionPipeline`，通过 `pipeline.contacts()` 分配 | 实际模型传感器缓冲 |
| fixed base 没有根速度 | 输出形状 `[N,6]` 的零根速度 | G1 fixed / floating 初始化与 FK |
| 奇数 decimation 状态复制方向颠倒 | 保留最新 `state_0`，刷新 `state_1` 的下一帧输入 | decimation=1,2,3,4，各连续三帧 |
| 不支持的场景物体 / 非平坦地形静默丢失 | 初始化时抛出明确 NotImplementedError | 实际构造路径的 CPU 回归 |
| viewer 不关闭、close 不幂等 | close 后清 viewer / graph、停止运行；异常也清理 | 部分初始化 / 两次 close / close 异常 |
| 不匹配的球关节 / 额外自由根布局 | 启动时校验 q / DOF stride，阻止错误 PD 索引 | scalar hinge 布局与根偏移核 |
| headless 截图错误 | 明确 RuntimeError | 静态代码与关闭回归 |

兼容层位于 `src/protomotions/simulator/newton/api_compat.py`；调用修正在 `simulator.py`。保留原有 exp_map 非等价独立 hinge 的明确拒绝，未将 SMPL 轴角关节近似替换成不等价模型。

## 实际证据

- `newton_all_cpu.xml`: **35 passed, 2 skipped**；2 项 CUDA 专用测试在本轮被明确跳过。
- `newton_legacy_cpu.xml`: **13 passed**，使用实际 Newton 1.4.0.dev0 / Warp 1.15 SDK，属于早一轮同一兼容层测试。
- `newton_wrapper_cpu.json`: 真实 `NewtonSimulator` + G1 的 floating / fixed 两种模型均初始化、前向运动学和有限状态读取成功。floating q=36/DOF=35，fixed q=29/DOF=29，33 bodies。
- 实际旧 Newton 1.0 不在机器上。旧签名和旧 contact 输出的回退仅通过接口边界测试覆盖，不宣称已对 1.0 动态验证。

## 明确边界 / 未验证项

1. Newton 1.5.2 自带 native CPU `SolverMuJoCo.update_contacts()` 仍抛出 `NotImplementedError`。接触聚合测试执行了真实 Warp SensorContact 核并输入确定的 contact buffers，不能视为 MuJoCo CPU 接触导出端到端测试。
2. CUDA graph 捕获、CUDA MuJoCo 接触生成、GUI 渲染 / 视频、长时 PPO / AMP 训练未由本子任务运行。
3. Newton 后端不实现动态场景物体、任意室内 USD 碰撞场景、非平坦 terrain；现已阻止无声地忽略这些输入。
4. `_setup_markers` 原实现为空、SMPL/SOMA exp_map 控制不受该后端支持。原版标量 hinge 模型可使用；室内真实行人场景应使用目标 Isaac Lab PhysX 后端。
5. 上述证据证明选定的 API 与状态交换回归，不证明所有物理参数 / 动作真实性都已评估。


---

打包说明：本页保留审查当时的阶段结论。最新证据与复制范围见 [审计索引](README.md)；未随包复制的大清单、源码快照与完整日志见 [来源索引](SOURCE_INDEX.md)。项目源码位置以仓库相对路径表述；SDK 绝对路径仅记录被审查机器。
