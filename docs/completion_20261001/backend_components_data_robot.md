# 后端、组件、数据与机器人契约复核（2026-10-01）

## 范围与证据

本轮涵盖 base simulator（不含 root 负责的 record.py）、IsaacLab/Newton/MuJoCo、MotionLib/PoseLib/SceneLib、地形、机器人配置及转换/prepare 脚本。以 `/project/EXP/NavIsaaclab2.0/protomotions` 为旧接口对照，并直接核查当前 pin 的 Lab3、Newton、MuJoCo 源码与真实运行。静态逐文件清单和函数接口差异见 [覆盖 JSON](backend_component_coverage.json)；静态扫描不表示每个分支都运行过。

最终限定回归 **616 passed / 0 skipped / 0 failed**，共 42 测试文件。包含真实 Newton CUDA、MuJoCo FK 导数、当前 Lab3 原生 ProxyArray、真实 MuJoCo MJCF 对照、缓存文件 IO 与 ONNX FK。见 [日志](backend_component_tests.log)、[JUnit](backend_component_tests.xml)、[测试清单](backend_component_test_files.json)。本 agent 未启动 Kit；实际 PhysX 探针由 root 单实例执行。

## 已确认并处理的问题

| ID | 问题 | 处理结果 |
|---|---|---|
| B01 | G1 cached skeleton fixture mismatch | diagnosed; cache remains mismatched |
| B02 | Body origin vs COM velocity mismatch | fixed |
| B03 | MuJoCo angular velocity frame and cvel reference point | fixed |
| B04 | Exp-map SMPL/SOMA state passed to Euler three-axis backends | guarded unsupported contract |
| B05 | MJCF FK ignored body rotation forms, hinge pivots and ref angles | fixed |
| B06 | G1 CSV sampler silently mislabeled FPS and broke single-frame shape | fixed |
| B07 | Contact smoothing accepted any integer as binary | fixed |
| B08 | Scene motion frame mismatch and invalid FPS | fixed |
| B09 | Generated terrain cache failed with current PyTorch restricted loading | fixed |
| B10 | Out-of-grid terrain interpolation extrapolated arbitrary heights | fixed |
| B11 | Motion subset tool discarded goal-state labels | fixed |
| B12 | Test oracles masked actual contract issues | fixed tests |

详细源码证据、边界与测试在 [报告 JSON](backend_components_data_robot.json)。本轮 18 份既有源码/测试修改、7 份新增脚本/测试均有最终 SHA256；所有既有文件修改前已保存快照，见 [本轮变更清单](backend_component_changes.json) 与 `before/`。

## G1 的资源不一致仍然存在

原 `test_newton_simulator_fk_matches_motionlib_default_frame` 平均误差约 3.097mm，根因是示例缓存使用旧骨架尺寸：waist_roll_link 局部 z 为 .035 而当前 .044；torso_link 缓存 .019 而当前 0；肩部缓存 .23778 而当前 .24778。独立当前资产 PoseLib、MuJoCo 和 Newton FK 结果对齐。回归保留原 **1mm** 门槛，并分别报告后端 FK 与资源一致性。

[首/中/末资源检测](g1_cached_motion_asset_consistency.json) 最大偏差 **10.051mm**，状态仍为 **mismatch**。原 `.motion`、MJCF 与模型权重未修改。需要从原始动作源按当前 G1 资产重新转换，后端测试通过不能当作缓存资源已经修复。

独立工具可安全检查已有 `.motion/.pt`（CPU，`weights_only=True`，`mmap=True`，仅读抽样）：

```bash
PYTHONPATH=. ../IsaacLab-Protomotions/.venv/bin/python scripts/check_motion_asset_consistency.py   examples/data/g1_crouch_to_run.motion --robot g1 --output /tmp/g1_asset_check.json
```

mismatch 返回退出码 2；工具不重写用户缓存。

## 目标 SMPL/IsaacLab3 表示已验证

[AMASS validation 抽样](smpl_motion_asset_consistency.json) 第 0、153523、307045 帧与当前 SMPL MJCF 对齐，最大位置误差 **5.264e-7m**，旋转误差 **6.66e-5°**。仅读 mmap 数值，没有装载或哈希巨大 train 包。

[root 的真实 Kit FK 探针](isaaclab_smpl_fk.json) 没有推进物理时间：AMASS 首/中批次 exp_map 最大位置误差 **5.840e-7m**；非小角三轴合成姿势误差 **3.424e-7m**。错误的 Euler/hinge oracle 分别偏差 **83.744mm / 208.467mm**。这明确验证 PhysX D6 的当前 joint 数据与 exp_map 约定相符。

## Newton/MuJoCo 三轴支持限制

[真实 Newton SMPL 诊断](newton_smpl_representation_probe.json) 首/中/末姿势相对正确 exp_map FK 偏差 **12.365 / 107.422 / 58.686mm**，其动作缓存与当前资产一致。这是后端独立 hinge/Euler 坐标与 common exp_map 的表示不兼容。

两种后端现默认对 exp_map 三自由度机器人 **初始化时明确报错**。配置 `common_three_dof_representation="hinge"` 只允许具有一致 hinge 数据、观测、动作与策略的自定义任务；此参数不转换预训练权重。位置、速度、控制目标与广义力都需要完整非线性适配，当前没有假定仅转换 reset 角度就可以兼容。G1 单轴 Newton 真实回归继续通过。目标行人模拟使用已验证的 IsaacLab3 路径。

## 资源与剩余范围

[机器人资源契约](robot_asset_contracts.json) 检查 g1/h1_2/smpl/smplx/soma23/amp 六种配置，MJCF 全部存在；G1 35 个、H1_2 30 个声明网格路径存在且不是 LFS 指针。其余模型使用当前 MJCF 解析出的基本几何；本轮没有替换机器人或模型资源。

仍需处理/单独验证：G1 旧缓存重转；Newton/MuJoCo 三轴完整适配；逐个机器人/后端的物理执行；完整 AMASS 全库逐片段验证。严格论文复现的 gait diffusion、SMPL AMP 权重与精确资源清单缺口沿用此前资源审计，本轮没有生成替代权重。
