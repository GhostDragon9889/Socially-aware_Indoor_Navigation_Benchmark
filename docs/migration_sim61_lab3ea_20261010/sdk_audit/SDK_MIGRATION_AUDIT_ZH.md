# Isaac Lab 3.0 EA / Isaac Sim 6.1 SDK 比较

审查方式：源文件比较、AST、已安装包元数据、官方文档。SDK 与项目均未由此扫描器修改，未启动 Kit 或 GPU。随后 root 单独执行修复与运行验证，修复结果以 root 的运行收据为准。

## 已确认版本

| 项 | 现有 IsaacLab-Protomotions | 目标 /project/EXP/IsaacLab |
| --- | --- | --- |
| 产品 VERSION | 3.0.0 | 3.0.0 |
| Git HEAD | 5951d48dde083986489957a7b108ad48a46822a7 | ae37b028ea415c91ea2bc32609efcd759ed2b974 |
| isaaclab Python 包 | 12.0.0 | 17.0.2 |
| isaaclab-physx | 2.9.1 | 6.0.0 |
| Isaac Sim | 6.0.0.1 | 6.1.0.0 |
| Python | 3.12.13 | 3.12.13 |
| Torch | 2.11.0+cu128 | 2.11.0+cu128 |
| Warp | 1.15.0.dev20260626 | 1.16.0 |
| USD exchange | 3.0.0 | 2.3.0 |
| Newton | 1.4.0.dev0 | 1.5.2 |
| MuJoCo / MuJoCo Warp | 3.8.0 / 3.8.0.3 | 3.11.0 / 3.11.0 |
| Simulation Manager extension | 1.15.4 | 1.17.1 |

目标 uv.lock 已存在用户修改；目标 pyproject 与源码保持原状。不要用产品 VERSION=3.0.0 推断两个开发快照 API 一致。

## 实际迁移断点

1. **USDOnlyInteractiveScene 的旧 queue workaround 不兼容目标。** 目标 `_collect_asset_cfgs()` 返回 `(cfgs, global_paths)`，`REPLICATION_QUEUE` 从 `(cfg, backend)` 改成 cfg。目标官方 `replicate_physics=False` 已实现 USD-only 克隆。旧 subclass 会错误迭代或错误 tuple 解包，必须改为目标公开路径并检查完整 USD 克隆、原型引用与物理数量。
2. **6.0 生命周期与 native overlay 不能移用。** TRACE lifecycle 和 sealed helper 校验旧 SDK 文件 SHA/大小。目标这些源码已经改变，旧 tensor API 缓存路径 `110.1.11+110.1.1` 也不存在。旧 .so 是对旧 Kit / USD / extension 编译；目标 Kit 110.3.0、simulation manager 1.17.1，应使用目标官方生命周期或另行构建、验证目标 ABI，而不是更新 SHA 后强行解锁旧代码。
3. **两项物理默认值静默改变。** `SimulationCfg.use_newton_actuators: False -> True`（在 PhysX 上也可改变 explicit actuator execution）；`PhysxCfg.enable_external_forces_every_iteration: False -> True`（TGS 外力）。应显式锁定已训练权重对应行为，或记录切换后的扭矩、速度、步态与稳定性评估。
4. **MJCF 缓存缺 SDK 指纹。** 旧缓存只纳入 MJCF/资源内容、options、固定 workaround 版本，可跨 6.0/6.1 重用旧 USD。目标需独立缓存或将 SDK/importer/schema 版本纳入缓存键，并实际重建 SMPL，检验 24 body / 69 DOF 名称、FK、限位、驱动、碰撞与脚部路径。
5. **初始化失败可能错误 exit 0。** `train_paper_amp.py` 在 world 未完成时使用默认 `app.close()`；6.1 fast shutdown 默认终止进程且默认 exit_code=0。失败报告不应被操作系统成功状态掩盖。验证初始化、运行、捕获、终止各条失败路径。
6. **Newton alternate backend 已有确定断点。** `SolverNotifyFlags` 已被 `ModelFlags` 替代；`SolverMuJoCo(ls_parallel=...)` 参数删除；`joint_target_pos/vel` 为 RemovedAttribute，必须迁移为 q/qd 并检查 target layout。Newton 后端修复与验证另行记录，不能视为 PhysX 成功的附带证明。

## 保留但应清理的兼容桥

- 目标 `wp.to_torch(ProxyArray)` 暂有兼容 shim，显式 `.torch` 是当前接口。
- `net_forces_w` / `force_matrix_w` 仍有 aliases，但 PhysX 返回的是法向力并警告；用 `net_normal_forces_w` / `normal_force_matrix_w` 准确表达语义。
- `write_root_link_state_to_sim`、`write_joint_state_to_sim`、无 suffix 目标写方法仍有 deprecated 桥；新 index / mask 接口能明确稀疏数据形状。
- 旧 schema Cfg 名称仍有 deprecated aliases，应使用 PhysX 后端拥有的配置类，保持参数值一致。
- XYZW 已在项目 IsaacLab adapter 明确采用，目标继续 XYZW，不应再次对已迁移数据重复换序。
- joint/body ordering 默认 None 与 backend 原始顺序一致；项目 raw root_view 操作默认可维持。任何自定义 ordering 必须先把 public IDs 映射到 backend IDs。

## 测试建议

| 范围 | 必须验证的可观察行为 |
| --- | --- |
| 启动 | 目标版本/源码/原始 native plugin 与依赖一致；physics-only 和 RTX 两种启动正常 |
| 资源 | 独立 6.1 MJCF conversion；场景 USD closure、mesh、材质、地板 collider、机器人配置 |
| articulation | 精确 joint/body 名称与顺序；root-link vs COM；reset 后 FK；69 DOF限位/gain/effort |
| 控制 | torque / explicit PD / implicit PD 的目标、限幅、实际 applied torque、每步 dt |
| 接触 | 24传感器；N1/N4 feet-floor、peer、obstacle 过滤；完整四个120Hz substeps、无reset混样 |
| 克隆 | N1/N4 普通 USD copy 和更大 training replication；原型无悬空关系、physics数量一致 |
| 行走 | 既有 checkpoint 的动作/速度/摔倒/足滑/左右对称性；独立 heldout 动作风格 |
| 场景 | 四场景真实物理路线完成、碰撞、安全距离；短测试通过后原60秒协议 |
| 录制 | RGB非空、warmup、准确帧数与时间、物理不多步、不插值替代姿态、相机姿态 |
| 生命周期 | 正常退出、构造失败、运行失败、录像失败、SIGINT/SIGTERM；连续多进程无native异常 |
| alternate backend | Newton1.5 CPU导入/模型/control/solver、随机化与目标读写；GPU只由root统一安排 |

## 扫描范围与证据

- project src/scripts 759 Python 文件 AST 解析成功，存在一个 rotations.py docstring invalid escape SyntaxWarning。
- 项目 SDK 静态 import 107条，目标对应 IsaacLab module 文件都存在。此检查不等于运行时 symbol、extension、GPU 或训练成功。
- 两SDK Python文件目标2273 /现有1910；1935相对模块文件新增/删除/内容变化。详细 SHA/行数在 sdk_changed_modules.json，不代表1935处功能退化。
- machine_inventory.json 保存全部已安装分发版本与 requirements；project_sdk_imports.json / project_sdk_constructor_calls.json / project_sdk_data_accesses.json 保存实际项目调用位置。

## 官方资料

[Isaac Lab 3.0 EA](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html)、[迁移指南](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/migration/migrating_to_isaaclab_3-0.html)说明 backend 分包、Warp数据、XYZW、显式写接口。

[joint/body ordering](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/concepts/joint_and_body_ordering.html)明确 raw view 与 public indices 的边界。[PhysX tuning](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/concepts/solver-tuning/tune_physx.html)、[actuators](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/concepts/actuators.html)用于核对动力学配置。

[Isaac Sim 6.1 release notes](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/release_notes.html)确认 Kit 110.3.0，以及 simulation manager 启动 callback 控制、Newton升级。[6.1 migration](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/migration_guides/isaac_sim_6_1/index.html)、[系统要求](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html)和[Known Issues](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/known_issues.html)是版本特定资料。

[Lab Known Issues](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/refs/issues.html)仍记载传感器reset读数和初始空白帧等限制。因此完整测试收据只能证明已执行路径，不能保证任意输入、平台、未来SDK均无bug。


---

打包说明：本页保留审查当时的阶段结论。最新证据与复制范围见 [审计索引](README.md)；未随包复制的大清单、源码快照与完整日志见 [来源索引](SOURCE_INDEX.md)。项目源码位置以仓库相对路径表述；SDK 绝对路径仅记录被审查机器。
