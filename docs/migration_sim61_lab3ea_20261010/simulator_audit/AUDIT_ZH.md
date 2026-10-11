# Isaac Lab 3 EA / Isaac Sim 6.1 仿真适配审阅

已登记 22 个文件、7728 行和全部函数位置，AST解析/编译通过。详细清单见 `coverage.json`。

19项 CPU 边界回归通过。它们验证状态布局、API调用、选择/空集合以及缓存失效，没有启动Kit/GPU，不能代表真实物理结果。真实SDK/物理/录像验证由主任务统一执行。

修正包括：

- Official cfg-only nonreplicating clone path selected by public replicate signature; old tuple owner filter preserved
- Explicit use_newton_actuators=False and enable_external_forces_every_iteration=False preserve old training dynamics
- Explicit ProxyArray.torch/raw Warp/Torch access
- Supported root link pose/velocity and joint state/control index APIs
- Object contact subsets index concatenated tensor rather than Python list
- Semantic MJCF COM body IDs mapped to live PhysX body names
- Terrain vertex copy prevents cumulative height offsets
- MJCF USD cache includes installed SDK/importer versions and cache format v3
- Empty scenes return zero-object root states
- UI scope reserved action names rejected
- TYPE_CHECKING terrain importer path corrected

尚需集成验证：

- Full target Kit/PhysX articulation and D6 conversion test
- N1/N4 official USD-only cloning with actual contacts and clean shutdown
- Real motion and independent fall/style/route metrics
- GPU domain randomization mass/inertia/COM/friction branch tests
- Interactive viewport keyboard/camera/recording and asynchronous capture completion
- Scene collision cache source changes and mesh material fidelity

项目修改保留工作区，未提交、未推送。没有任何测试能证明项目永远没有bug。


## 全仓 CPU 回归后的补充

- 缓存 key 增加 SDK 版本时保留原有 tuple 路径索引，版本信息追加到尾部；避免既有调用方位置契约被破坏。
- 真实方法 AST harness 已使用实际 `as_torch`／`write_root_link_state` helper，并验证 pose、velocity、impulse 的 public index 写入参数与数值。
- SDK 安装元数据测试覆盖默认 17.0.2／6.1.0.0 与 legacy extra 12.0.0／6.0.0.1。上述 3 项与新增 19 项合计 22 CPU 测试通过。
- 独立 overlay 的 dm-control 原 1.0.40 在 SDK MuJoCo 3.11 构建真实模型时失效，已按官方发行说明更新为 1.0.44；真实 CPU 1,000 步重力、碰撞、复制、reset 通过。见 `../runtime/dm_control_cpu_regression.json`。
- 封存上游 MaskedMimic 旧 pickle／YAML 不兼容属于单独资源版本问题，没有重写模型或给旧类添加失真 alias。
